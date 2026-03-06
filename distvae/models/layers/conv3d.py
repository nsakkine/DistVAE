import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair, _triple

from torch.nn.common_types import _size_3_t
from typing import Optional, List, Tuple, Union
from distvae.utils import DistributedEnv

class PatchConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        block_size: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:

        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchConv3d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchConv3d"
        self.block_size = block_size
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype)
        
    def _get_world_size_and_rank(self):
        group_world_size = DistributedEnv.get_group_world_size()
        global_rank = DistributedEnv.get_global_rank()
        rank_in_group = DistributedEnv.get_rank_in_vae_group()
        local_rank = DistributedEnv.get_local_rank()
        return group_world_size, global_rank, rank_in_group, local_rank

    def _calc_patch_index(self, patch_height_list: List[Tensor]):
        height_index = []
        cur = 0
        for t in patch_height_list:
            height_index.append(cur)
            cur += t.item()
        height_index.append(cur)
        return height_index

    def _calc_bottom_halo_width(self, rank, height_index, kernel_size, padding = 0, stride = 1):
        assert rank >= 0, "rank should not be smaller than 0"
        assert rank < len(height_index) - 1, "rank should be smaller than the length of height_index - 1"
        assert padding >= 0, "padding should not be smaller than 0"
        assert stride > 0, "stride should be larger than 0"

        if rank == DistributedEnv.get_group_world_size() - 1:
            return 0
        nstep_before_bottom = (height_index[rank + 1] + padding - (kernel_size - 1) // 2 + stride - 1) // stride
        if nstep_before_bottom <= 0:
            # Patch too small for at least one conv step; use nstep=0 so halo is minimal
            bottom_halo_width = -stride + kernel_size - padding - height_index[rank + 1]
            return max(0, bottom_halo_width)
        bottom_halo_width =  (nstep_before_bottom - 1) * stride + kernel_size - padding - height_index[rank + 1]
        return max(0, bottom_halo_width)

    def _calc_top_halo_width(self, rank, height_index, kernel_size, padding = 0, stride = 1):
        assert rank >= 0, "rank should not be smaller than 0"
        assert rank < len(height_index) - 1, "rank should be smaller than the length of height_index - 1"
        assert padding >= 0, "padding should not be smaller than 0"
        assert stride > 0, "stride should be larger than 0"

        if rank == 0:
            return 0
        nstep_before_top = (height_index[rank] + padding - (kernel_size - 1) // 2 + stride - 1) // stride
        if nstep_before_top <= 0:
            # Patch boundary or tiny patch; no top halo
            return 0
        top_halo_width = height_index[rank] - (nstep_before_top * stride - padding)
        return max(0, top_halo_width)


    def _calc_halo_width(self, rank, height_index, kernel_size, padding = 0, stride = 1):
        ''' 
            Calculate the width of halo region in height dimension. 
            The halo region is the region that is used for convolution but not included in the output.
            return value: (top_halo_width, bottom_halo_width)
        '''
        halo_width = [
            self._calc_top_halo_width(rank, height_index, kernel_size, padding, stride),
            self._calc_bottom_halo_width(rank, height_index, kernel_size, padding, stride)
        ]
        if rank == 0:
            halo_width[0] = 0
        elif rank == DistributedEnv.get_group_world_size() - 1:
            halo_width[1] = 0
        return tuple(halo_width)
        

    # in 3d case, padding is a tuple of 6 integers: (W_l, W_r, H_l, H_r, F_l, F_r)
    def _adjust_padding_for_patch(self, padding, rank, world_size, patch_dim: int = 2):
        if isinstance(padding, tuple):
            padding = list(padding)
        elif isinstance(padding, int):
            padding = [padding] * 6
        # indices for (left, right) of each spatial dim: W=0,1; H=2,3; F=4,5
        left_idx, right_idx = {2: (4, 5), 3: (2, 3), 4: (0, 1)}[patch_dim]
        if rank == 0:
            padding[right_idx] = 0
        elif rank == world_size - 1:
            padding[left_idx] = 0
        else:
            padding[left_idx] = 0
            padding[right_idx] = 0
        return tuple(padding)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, f, h, w = input.shape

        group_world_size, global_rank, rank_in_group, local_rank = self._get_world_size_and_rank()

        if (group_world_size == 1):
            if self.padding_mode != 'zeros':
                return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _triple(0), self.dilation, self.groups)
            return F.conv3d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
            
        else:
            patch_dim = DistributedEnv.get_patch_dim()
            patch_dim = patch_dim if patch_dim >= 0 else input.ndim + patch_dim
            spatial_idx = patch_dim - 2
            k = self.kernel_size[spatial_idx] if isinstance(self.kernel_size, tuple) else self.kernel_size
            p = self.padding[spatial_idx] if isinstance(self.padding, tuple) else self.padding
            s = self.stride[spatial_idx] if isinstance(self.stride, tuple) else self.stride
            patch_size = input.shape[patch_dim]
            patch_list = [torch.zeros(1, dtype=torch.int64, device=DistributedEnv.get_device()) for _ in range(group_world_size)]
            dist.all_gather(patch_list, torch.tensor([patch_size], dtype=torch.int64, device=DistributedEnv.get_device()), group=DistributedEnv.get_vae_group())
            patch_index = self._calc_patch_index(patch_list)
            halo_width = self._calc_halo_width(rank_in_group, patch_index, k, p, s)
            prev_bottom = self._calc_bottom_halo_width(rank_in_group - 1, patch_index, k, p, s) if rank_in_group != 0 else 0
            next_top = self._calc_top_halo_width(rank_in_group + 1, patch_index, k, p, s) if rank_in_group != group_world_size - 1 else 0
            next_top = max(0, next_top)
            assert halo_width[0] <= patch_size and halo_width[1] <= patch_size, "halo width larger than patch size"

            to_next, to_prev = None, None
            top_halo_recv, bottom_halo_recv = None, None
            global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1) if rank_in_group + 1 < group_world_size else None
            global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1) if rank_in_group > 0 else None
            if next_top > 0:
                slice_end = (slice(None),) * patch_dim + (slice(-next_top, None),) + (slice(None),) * (4 - patch_dim)
                to_next = dist.isend(input[slice_end].contiguous(), global_rank_of_next, group=DistributedEnv.get_vae_group())
            if halo_width[0] > 0:
                assert patch_index[rank_in_group] - halo_width[0] >= patch_index[rank_in_group - 1], \
                    "width of top halo region is larger than the input tensor of prev rank"
                recv_shape = list(input.shape)
                recv_shape[patch_dim] = halo_width[0]
                top_halo_recv = torch.empty(recv_shape, dtype=input.dtype, device=DistributedEnv.get_device())
                dist.recv(top_halo_recv, global_rank_of_prev, group=DistributedEnv.get_vae_group())
            if prev_bottom > 0:
                slice_start = (slice(None),) * patch_dim + (slice(0, prev_bottom),) + (slice(None),) * (4 - patch_dim)
                if global_rank_of_prev is None:
                    global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
                to_prev = dist.isend(input[slice_start].contiguous(), global_rank_of_prev, group=DistributedEnv.get_vae_group())
            if halo_width[1] > 0:
                assert patch_index[rank_in_group + 1] + halo_width[1] <= patch_index[rank_in_group + 2], \
                    "width of bottom halo region is larger than the input tensor of next rank"
                recv_shape = list(input.shape)
                recv_shape[patch_dim] = halo_width[1]
                bottom_halo_recv = torch.empty(recv_shape, dtype=input.dtype, device=DistributedEnv.get_device())
                if global_rank_of_next is None:
                    global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
                dist.recv(bottom_halo_recv, global_rank_of_next, group=DistributedEnv.get_vae_group())
            if halo_width[0] < 0:
                trim_slice = [slice(None)] * 5
                trim_slice[patch_dim] = slice(-halo_width[0], None)
                input = input[tuple(trim_slice)]
            if top_halo_recv is not None:
                input = torch.cat([top_halo_recv, input], dim=patch_dim)
            if bottom_halo_recv is not None:
                input = torch.cat([input, bottom_halo_recv], dim=patch_dim)
            if to_next is not None:
                to_next.wait()
            if to_prev is not None:
                to_prev.wait()

            padding = self._adjust_padding_for_patch(
                self._reversed_padding_repeated_twice, rank=rank_in_group, world_size=group_world_size, patch_dim=patch_dim
            )
            input = F.pad(input, padding, mode="constant", value=0.0)
            bs, channels, f, h, w = input.shape

            block_ok_f = (f <= self.block_size) if isinstance(self.block_size, int) else (f <= self.block_size[0])
            block_ok_h = (h <= self.block_size) if isinstance(self.block_size, int) else (h <= self.block_size[1])
            block_ok_w = (w <= self.block_size) if isinstance(self.block_size, int) else (w <= self.block_size[2])

            if self.block_size == 0 or (block_ok_f and block_ok_h and block_ok_w):
                conv_res: Tensor
                if self.padding_mode != 'zeros':
                    conv_res = F.conv3d(F.pad(input, padding, mode=self.padding_mode),
                                        weight, bias, self.stride,
                                        _triple(0), self.dilation, self.groups)
                else:
                    if (
                        self.stride[spatial_idx] == 1 and
                        self.padding[spatial_idx] == 1 and
                        self.kernel_size[spatial_idx] == 3
                    ):
                        conv_res = F.conv3d(input, weight, bias, self.stride,
                                            self.padding, self.dilation, self.groups)
                        crop_slice = [slice(None)] * 5
                        if halo_width[1] == 0:
                            crop_slice[patch_dim] = slice(halo_width[0], None)
                        else:
                            crop_slice[patch_dim] = slice(halo_width[0], -halo_width[1])
                        return conv_res[tuple(crop_slice)].contiguous()

                    else:
                        conv_res = F.conv3d(F.pad(input, padding, "constant", 0.0),
                                            weight, bias, self.stride,
                                            _triple(0), self.dilation, self.groups)
                        return conv_res


            else:
                if self.padding_mode != "zeros":
                    input = F.pad(input, padding, mode=self.padding_mode)
                elif self.padding != 0:
                    input = F.pad(input, padding, mode="constant")

                _, _, f, h, w = input.shape
                if isinstance(self.block_size, int):
                    num_chunks_in_f = (f + self.block_size - 1) // self.block_size
                    num_chunks_in_h = (h + self.block_size - 1) // self.block_size
                    num_chunks_in_w = (w + self.block_size - 1) // self.block_size
                else:
                    num_chunks_in_f = (f + self.block_size[0] - 1) // self.block_size[0]
                    num_chunks_in_h = (h + self.block_size[1] - 1) // self.block_size[1]
                    num_chunks_in_w = (w + self.block_size[2] - 1) // self.block_size[2]
                unit_chunk_size_f = f // num_chunks_in_f
                unit_chunk_size_h = h // num_chunks_in_h
                unit_chunk_size_w = w // num_chunks_in_w
                if isinstance(self.kernel_size, int):
                    kernel_size_f, kernel_size_h, kernel_size_w = self.kernel_size, self.kernel_size, self.kernel_size
                else:
                    kernel_size_f, kernel_size_h, kernel_size_w = self.kernel_size
                if isinstance(self.stride, int):
                    stride_f, stride_h, stride_w = self.stride, self.stride, self.stride
                else:
                    stride_f, stride_h, stride_w = self.stride

                def correct_end(end, kernel_size, stride):
                    return ((end + stride - 1) // stride - 1) * stride + kernel_size

                def correct_start(start, stride):
                    return ((start + stride - 1) // stride) * stride

                outputs = []
                for idx_f in range(num_chunks_in_f):
                    outer_output = []
                    for idx_h in range(num_chunks_in_h):
                        inner_output = []
                        for idx_w in range(num_chunks_in_w):
                            start_f = idx_f * unit_chunk_size_f
                            start_w = idx_w * unit_chunk_size_w
                            start_h = idx_h * unit_chunk_size_h
                            end_f = (idx_f + 1) * unit_chunk_size_f
                            end_w = (idx_w + 1) * unit_chunk_size_w
                            end_h = (idx_h + 1) * unit_chunk_size_h
                            if idx_f + 1 < num_chunks_in_f:
                                end_f = correct_end(end_f, kernel_size_f, stride_f)
                            else:
                                end_f = f
                            if idx_w + 1 < num_chunks_in_w:
                                end_w = correct_end(end_w, kernel_size_w, stride_w)
                            else:
                                end_w = w
                            if idx_h + 1 < num_chunks_in_h:
                                end_h = correct_end(end_h, kernel_size_h, stride_h)
                            else:
                                end_h = h
                            if idx_f > 0:
                                start_f = correct_start(start_f, stride_f)
                            if idx_w > 0:
                                start_w = correct_start(start_w, stride_w)
                            if idx_h > 0:
                                start_h = correct_start(start_h, stride_h)

                            inner_output.append(
                                F.conv3d(
                                    input[:, :, start_f:end_f, start_h:end_h, start_w:end_w],
                                    weight,
                                    bias,
                                    self.stride,
                                    0,
                                    self.dilation,
                                    self.groups,
                                )
                            )
                        outer_output.append(torch.cat(inner_output, dim=-1))
                    outputs.append(torch.cat(outer_output, dim=-2))
                out = torch.cat(outputs, dim=-3)
                out_len = out.shape[patch_dim]
                if out_len == patch_size:
                    slice_out = (slice(None),) * patch_dim + (slice(0, patch_size),) + (slice(None),) * (4 - patch_dim)
                else:
                    slice_out = (slice(None),) * patch_dim + (slice(halo_width[0], halo_width[0] + patch_size),) + (slice(None),) * (4 - patch_dim)
                return out[slice_out].contiguous()