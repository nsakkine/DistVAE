import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair, _triple

from torch.nn.common_types import _size_2_t
from typing import Optional, List, Tuple, Union
from distvae.utils import DistributedEnv

class PatchConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        block_size: Union[int, Tuple[int, int, int]] = 0,
        pre_conv_padding: Optional[Tuple[int, int, int, int, int, int]] = None,
    ) -> None:

        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchConv3d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchConv3d"
        self.block_size = block_size
        self.pre_conv_padding = pre_conv_padding
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
        

    # in 3d case, padding is a tuple of 6 integers: [left_pad, right_pad, top_pad, bottom_pad, front_pad, back_pad]
    def _adjust_padding_for_patch(self, padding, rank, world_size, causal_f: bool = False):
        if isinstance(padding, tuple):
            padding = list(padding)
        elif isinstance(padding, int):
            padding = [padding] * 6

        if rank == 0:
            padding[-1] = 0  # F_right
        elif rank == world_size - 1:
            if not causal_f:
                padding[-2] = 0  # F_left; for causal, last rank keeps F_left so conv has enough context
        else:
            padding[-2:] = [0, 0]
        return tuple(padding)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, f, h, w = input.shape

        group_world_size, global_rank, rank_in_group, local_rank = self._get_world_size_and_rank()

        if (group_world_size == 1):
            if self.pre_conv_padding is not None:
                pad = self._adjust_padding_for_patch(self.pre_conv_padding, rank=0, world_size=1, causal_f=True)
                input = F.pad(input, pad, mode="constant", value=0.0)
                return F.conv3d(input, weight, bias, self.stride,
                                _triple(0), self.dilation, self.groups)
            if self.padding_mode != 'zeros':
                return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _triple(0), self.dilation, self.groups)
            return F.conv3d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
            
        else:
        # 1. get the meta data of input tensor and conv operation
            patch_f = f  # patch F size before halo; used for crop when pre_conv_padding is set
            effective_padding_f = 0 if self.pre_conv_padding is not None else (
                self.padding[0] if isinstance(self.padding, tuple) else self.padding
            )
            patch_height_list = [torch.zeros(1, dtype=torch.int64, device=DistributedEnv.get_device()) for _ in range(group_world_size)]
            dist.all_gather(patch_height_list, torch.tensor([f], dtype=torch.int64, device=DistributedEnv.get_device()), group=DistributedEnv.get_vae_group())
            patch_height_index = self._calc_patch_index(patch_height_list)
            halo_width = self._calc_halo_width(rank_in_group,  patch_height_index, self.kernel_size[0], effective_padding_f, self.stride[0])
            prev_bottom_halo_width: int = 0
            next_top_halo_width: int = 0
            if rank_in_group != 0:
                prev_bottom_halo_width = self._calc_bottom_halo_width(rank_in_group - 1, patch_height_index, self.kernel_size[0], effective_padding_f, self.stride[0])
            if rank_in_group != group_world_size - 1:
                next_top_halo_width = self._calc_top_halo_width(rank_in_group + 1, patch_height_index, self.kernel_size[0], effective_padding_f, self.stride[0])
                next_top_halo_width = max(0, next_top_halo_width)
            

            assert halo_width[0] <= f and halo_width[1] <= f, "halo width is larger than the height of input tensor"


        # 2. get the halo region from other ranks
        # up to down
            to_next = None
            to_prev = None
            top_halo_recv = None
            bottom_halo_recv = None
            global_rank_of_next, global_rank_of_prev  = None, None
            if next_top_halo_width > 0:
                # isend to next
                bottom_halo_send = input[:, :, -next_top_halo_width:, :, :].contiguous()
                global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
                to_next = dist.isend(bottom_halo_send, global_rank_of_next, group=DistributedEnv.get_vae_group())
                
            if halo_width[0] > 0:
                # recv from prev
                assert patch_height_index[rank_in_group] - halo_width[0] >= patch_height_index[rank_in_group-1], \
                    "width of top halo region is larger than the height of input tensor of last rank"
                top_halo_recv = torch.empty([bs, channels, halo_width[0], h, w], dtype=input.dtype, device=DistributedEnv.get_device())
                global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
                dist.recv(top_halo_recv, global_rank_of_prev, group=DistributedEnv.get_vae_group())

        # down to up
            if prev_bottom_halo_width > 0:
                # isend to prev
                top_halo_send = input[:, :, :prev_bottom_halo_width, :, :].contiguous()
                if global_rank_of_prev is None:
                    global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
                to_prev = dist.isend(top_halo_send, global_rank_of_prev, group=DistributedEnv.get_vae_group())
            
            if halo_width[1] > 0:
                # recv from next
                assert patch_height_index[rank_in_group+1] + halo_width[1] <= patch_height_index[rank_in_group+2], \
                    "width of bottom halo region is larger than the height of input tensor of next rank"
                bottom_halo_recv = torch.empty([bs, channels, halo_width[1], h, w], dtype=input.dtype, device=DistributedEnv.get_device())
                if global_rank_of_next is None:
                    global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
                dist.recv(bottom_halo_recv, global_rank_of_next, group=DistributedEnv.get_vae_group())
        
        # Remove redundancy at the top of the input
            if halo_width[0] < 0:
                input = input[:, :, -halo_width[0]:, :, :]
        # concat the halo region to the input tensor            
            if top_halo_recv is not None:
                input = torch.cat([top_halo_recv, input], dim=-3)
            if bottom_halo_recv is not None:
                input = torch.cat([input, bottom_halo_recv], dim=-3)
            
        # wait for the communication to finish
            if to_next is not None:
                to_next.wait()
            if to_prev is not None:
                to_prev.wait()

        # 3. do convolution and postprocess
            conv_res: Tensor
            if self.pre_conv_padding is not None:
                padding = self._adjust_padding_for_patch(self.pre_conv_padding, rank=rank_in_group, world_size=group_world_size, causal_f=True)
                input = F.pad(input, padding, mode="constant", value=0.0)
            else:
                padding = self._adjust_padding_for_patch(self._reversed_padding_repeated_twice, rank=rank_in_group, world_size=group_world_size)
            bs, channels, f, h, w = input.shape
            if self.block_size == 0 or (f <= self.block_size and h <= self.block_size and w <= self.block_size):
                if self.pre_conv_padding is not None:
                    conv_res = F.conv3d(input, weight, bias, self.stride,
                                    _triple(0), self.dilation, self.groups)
                    conv_res = conv_res[:, :, halo_width[0]:halo_width[0] + patch_f, :, :].contiguous()
                    return conv_res
                if self.padding_mode != 'zeros':
                    conv_res = F.conv3d(F.pad(input, padding, mode=self.padding_mode),
                                    weight, bias, self.stride,
                                    _triple(0), self.dilation, self.groups)
                else:
                    if self.stride[0] == 1 and self.stride[1] == 1 and self.stride[2] == 1 and self.padding[0] == 1 and self.padding[1] == 1 and self.padding[2] == 1 and self.kernel_size[0] == 3 and self.kernel_size[1] == 3 and self.kernel_size[2] == 3:
                        conv_res = F.conv3d(input, weight, bias, self.stride,
                                    self.padding, self.dilation, self.groups)
                        if halo_width[1] == 0:
                            conv_res = conv_res[:, :, halo_width[0]:, :, :].contiguous()
                        else:
                            conv_res = conv_res[:, :, halo_width[0]:-halo_width[1], :, :]
                    else:
                        conv_res = F.conv3d(F.pad(input, padding, "constant", 0.0),
                                        weight, bias, self.stride,
                                        _triple(0), self.dilation, self.groups)
                return conv_res

        # 3.1. if block_size is not 0, split patch to block and do convolution to 
                # reduce memory spike
            else:
                if self.pre_conv_padding is None:
                    if self.padding_mode != "zeros":
                        input = F.pad(input, padding, mode=self.padding_mode)
                    elif self.padding != 0:
                        input = F.pad(input, padding, mode="constant")

                _, _, f, h, w = input.shape
                num_chunks_in_f = 0
                num_chunks_in_h = 0
                num_chunks_in_w = 0
                if isinstance(self.block_size, int):
                    num_chunks_in_f = (f + self.block_size - 1) // self.block_size
                    num_chunks_in_h = (h + self.block_size - 1) // self.block_size
                    num_chunks_in_w = (w + self.block_size - 1) // self.block_size
                elif isinstance(self.block_size, tuple):
                    num_chunks_in_f = (f + self.block_size[0] - 1) // self.block_size[0]
                    num_chunks_in_h = (h + self.block_size[1] - 1) // self.block_size[1]
                    num_chunks_in_w = (w + self.block_size[2] - 1) // self.block_size[2]
                unit_chunk_size_f = f // num_chunks_in_f
                unit_chunk_size_h = h // num_chunks_in_h
                unit_chunk_size_w = w // num_chunks_in_w
                if isinstance(self.kernel_size, int):
                    kernel_size_f, kernel_size_h, kernel_size_w = self.kernel_size, self.kernel_size, self.kernel_size
                elif isinstance(self.kernel_size, tuple):
                    kernel_size_f, kernel_size_h, kernel_size_w = self.kernel_size
                else:
                    raise ValueError(
                        f"kernel_size should be int or tuple, type:{type(self.kernel_size)}"
                    )

                if isinstance(self.stride, int):
                    stride_f, stride_h, stride_w = self.stride, self.stride, self.stride
                elif isinstance(self.stride, tuple):
                    stride_f, stride_h, stride_w = self.stride
                else:
                    raise ValueError(
                        f"stride should be int or tuple, type: {type(self.stride)}"
                    )

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
                if self.pre_conv_padding is not None:
                    out = out[:, :, halo_width[0]:halo_width[0] + patch_f, :, :].contiguous()
                return out