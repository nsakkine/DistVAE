from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t

from distvae.utils import DistributedEnv
from distvae.models.layers.conv import (
    get_world_size_and_rank,
    calc_patch_index,
    calc_halo_width,
    calc_bottom_halo_width,
    calc_top_halo_width
)


class PatchConv2d(nn.Conv2d):
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
        block_size: Union[int, Tuple[int, int]] = 0,
        patch_dim: int = -2,
    ) -> None:

        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchConv2d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchConv2d"
        assert patch_dim in (-2, -1, 2, 3), (
            "PatchConv2d patch_dim must be H (-2 or 2) or W (-1 or 3)"
        )
        self.block_size = block_size
        self.patch_dim = patch_dim
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)

        # in 2d case, padding is a tuple of 4 integers: (W_l, W_r, H_l, H_r) for F.pad
    def _adjust_padding_for_patch(self,padding, rank, world_size, patch_dim: int = 2):
        if isinstance(padding, tuple):
            padding = list(padding)
        elif isinstance(padding, int):
            padding = [padding] * 4
        # patch_dim_d 2 = H (indices 2,3), 3 = W (indices 0,1)
        right_idx = (3, 1)[patch_dim - 2]
        left_idx = (2, 0)[patch_dim - 2]
        if rank == 0:
            padding[right_idx] = 0
        elif rank == world_size - 1:
            padding[left_idx] = 0
        else:
            padding[left_idx] = 0
            padding[right_idx] = 0
        return tuple(padding)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, h, w = input.shape

        group_world_size, global_rank, rank_in_group, local_rank = get_world_size_and_rank()

        if (group_world_size == 1):
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)

        else:
            patch_dim = self.patch_dim if self.patch_dim >= 0 else input.ndim + self.patch_dim
            patch_size = input.shape[patch_dim]
            spatial_idx = patch_dim - 2
            kernel_size_patch_dim = (
                self.kernel_size[spatial_idx]
                if isinstance(self.kernel_size, tuple) else self.kernel_size
            )
            padding_patch_dim = (
                self.padding[spatial_idx]
                if isinstance(self.padding, tuple) else self.padding
            )
            stride_patch_dim = (
                self.stride[spatial_idx]
                if isinstance(self.stride, tuple) else self.stride
            )

            # 1. get the meta data of input tensor and conv operation
            patch_height_list = [
                torch.zeros(1, dtype=torch.int64, device=DistributedEnv.get_device())
                for _ in range(group_world_size)
            ]
            dist.all_gather(
                patch_height_list,
                torch.tensor([patch_size], dtype=torch.int64, device=DistributedEnv.get_device()),
                group=DistributedEnv.get_vae_group()
            )
            patch_height_index = calc_patch_index(patch_height_list)
            halo_width = calc_halo_width(
                rank_in_group,
                patch_height_index,
                kernel_size_patch_dim,
                padding_patch_dim,
                stride_patch_dim
            )
            prev_bottom_halo_width: int = 0
            next_top_halo_width: int = 0
            if rank_in_group != 0:
                prev_bottom_halo_width = calc_bottom_halo_width(
                    rank_in_group - 1,
                    patch_height_index,
                    kernel_size_patch_dim,
                    padding_patch_dim,
                    stride_patch_dim
                )
            if rank_in_group != group_world_size - 1:
                next_top_halo_width = calc_top_halo_width(
                    rank_in_group + 1,
                    patch_height_index,
                    kernel_size_patch_dim,
                    padding_patch_dim,
                    stride_patch_dim
                )
                next_top_halo_width = max(0, next_top_halo_width)
            assert halo_width[0] <= patch_size and halo_width[1] <= patch_size, (
                "halo width is larger than the patch dimension of input tensor"
            )

            # 2. get the halo region from other ranks
            to_next = None
            to_prev = None
            top_halo_recv = None
            bottom_halo_recv = None
            global_rank_of_next, global_rank_of_prev = None, None
            indices_end = [slice(None)] * 4
            indices_end[patch_dim] = slice(-next_top_halo_width, None)
            indices_start = [slice(None)] * 4
            indices_start[patch_dim] = slice(0, prev_bottom_halo_width)
            # up to down
            if next_top_halo_width > 0:
                bottom_halo_send = input[tuple(indices_end)].contiguous()
                global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
                to_next = dist.isend(
                    bottom_halo_send,
                    global_rank_of_next,
                    group=DistributedEnv.get_vae_group()
                )
            if halo_width[0] > 0:
                assert patch_height_index[rank_in_group] - halo_width[0] >= patch_height_index[rank_in_group - 1], (
                    "width of top halo region is larger than the input tensor of prev rank"
                )
                recv_shape = list(input.shape)
                recv_shape[patch_dim] = halo_width[0]
                top_halo_recv = torch.empty(recv_shape, dtype=input.dtype, device=DistributedEnv.get_device())
                global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
                dist.recv(
                    top_halo_recv,
                    global_rank_of_prev,
                    group=DistributedEnv.get_vae_group()
                )
            # down to up
            if prev_bottom_halo_width > 0:
                top_halo_send = input[tuple(indices_start)].contiguous()
                if global_rank_of_prev is None:
                    global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
                to_prev = dist.isend(
                    top_halo_send,
                    global_rank_of_prev,
                    group=DistributedEnv.get_vae_group()
                )
            if halo_width[1] > 0:
                assert patch_height_index[rank_in_group + 1] + halo_width[1] <= patch_height_index[rank_in_group + 2], (
                    "width of bottom halo region is larger than the input tensor of next rank"
                )
                recv_shape = list(input.shape)
                recv_shape[patch_dim] = halo_width[1]
                bottom_halo_recv = torch.empty(recv_shape, dtype=input.dtype, device=DistributedEnv.get_device())
                if global_rank_of_next is None:
                    global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
                dist.recv(
                    bottom_halo_recv,
                    global_rank_of_next,
                    group=DistributedEnv.get_vae_group()
                )
            # Remove redundancy at the top of the input
            if halo_width[0] < 0:
                trim_slice = [slice(None)] * 4
                trim_slice[patch_dim] = slice(-halo_width[0], None)
                input = input[tuple(trim_slice)]
            # concat the halo region to the input tensor
            if top_halo_recv is not None:
                input = torch.cat([top_halo_recv, input], dim=patch_dim)
            if bottom_halo_recv is not None:
                input = torch.cat([input, bottom_halo_recv], dim=patch_dim)
            # wait for the communication to finish
            if to_next is not None:
                to_next.wait()
            if to_prev is not None:
                to_prev.wait()

            # 3. do convolution and postprocess
            conv_res: Tensor
            padding = self._adjust_padding_for_patch(
                self._reversed_padding_repeated_twice,
                rank=rank_in_group,
                world_size=group_world_size,
                patch_dim=patch_dim
            )
            bs, channels, h, w = input.shape
            if self.block_size == 0 or (
                    (h <= self.block_size) if isinstance(self.block_size, int) else (h <= self.block_size[0]) and
                    (w <= self.block_size) if isinstance(self.block_size, int) else (w <= self.block_size[1])
            ):
                if self.padding_mode != 'zeros':
                    conv_res = F.conv2d(F.pad(input, padding, mode=self.padding_mode),
                                    weight, bias, self.stride,
                                    _pair(0), self.dilation, self.groups)
                else:
                    if (
                        stride_patch_dim == 1 and
                        padding_patch_dim == 1 and
                        kernel_size_patch_dim == 3
                    ):
                        conv_res = F.conv2d(input, weight, bias, self.stride,
                                    self.padding, self.dilation, self.groups)
                        crop_slice = 4 * [slice(None),]
                        if halo_width[1] == 0:
                            crop_slice[patch_dim] = slice(halo_width[0], None)
                        else:
                            crop_slice[patch_dim] = slice(halo_width[0], -halo_width[1])
                        conv_res = conv_res[tuple(crop_slice)].contiguous()
                    else:
                        conv_res = F.conv2d(F.pad(input, padding, "constant", 0.0),
                                        weight, bias, self.stride,
                                        _pair(0), self.dilation, self.groups)
                return conv_res
            else:
                if self.padding_mode != "zeros":
                    input = F.pad(input, padding, mode=self.padding_mode)
                elif self.padding != 0:
                    input = F.pad(input, padding, mode="constant")

                _, _, h, w = input.shape
                num_chunks_in_h = 0
                num_chunks_in_w = 0
                if isinstance(self.block_size, int):
                    num_chunks_in_h = (h + self.block_size - 1) // self.block_size
                    num_chunks_in_w = (w + self.block_size - 1) // self.block_size
                elif isinstance(self.block_size, tuple):
                    num_chunks_in_h = (h + self.block_size[0] - 1) // self.block_size[0]
                    num_chunks_in_w = (w + self.block_size[1] - 1) // self.block_size[1]
                unit_chunk_size_h = h // num_chunks_in_h
                unit_chunk_size_w = w // num_chunks_in_w
                if isinstance(self.kernel_size, int):
                    kernel_size_h, kernel_size_w = self.kernel_size, self.kernel_size
                elif isinstance(self.kernel_size, tuple):
                    kernel_size_h, kernel_size_w = self.kernel_size
                else:
                    raise ValueError(
                        f"kernel_size should be int or tuple, type:{type(self.kernel_size)}"
                    )

                if isinstance(self.stride, int):
                    stride_h, stride_w = self.stride, self.stride
                elif isinstance(self.stride, tuple):
                    stride_h, stride_w = self.stride
                else:
                    raise ValueError(
                        f"stride should be int or tuple, type: {type(self.stride)}"
                    )

                def correct_end(end, kernel_size, stride):
                    return ((end + stride - 1) // stride - 1) * stride + kernel_size

                def correct_start(start, stride):
                    return ((start + stride - 1) // stride) * stride

                outputs = []
                for idx_h in range(num_chunks_in_h):
                    inner_output = []
                    for idx_w in range(num_chunks_in_w):
                        start_w = idx_w * unit_chunk_size_w
                        start_h = idx_h * unit_chunk_size_h
                        end_w = (idx_w + 1) * unit_chunk_size_w
                        end_h = (idx_h + 1) * unit_chunk_size_h
                        if idx_w + 1 < num_chunks_in_w:
                            end_w = correct_end(end_w, kernel_size_w, stride_w)
                        else:
                            end_w = w
                        if idx_h + 1 < num_chunks_in_h:
                            end_h = correct_end(end_h, kernel_size_h, stride_h)
                        else:
                            end_h = h

                        if idx_w > 0:
                            start_w = correct_start(start_w, stride_w)
                        if idx_h > 0:
                            start_h = correct_start(start_h, stride_h)

                        inner_output.append(
                            F.conv2d(
                                input[:, :, start_h:end_h, start_w:end_w],
                                weight,
                                bias,
                                self.stride,
                                0,
                                self.dilation,
                                self.groups,
                            )
                        )
                    outputs.append(torch.cat(inner_output, dim=-1))
                outputs = torch.cat(outputs, dim=-2)
                if outputs.shape[patch_dim] == patch_size:
                    crop_slice = (
                        (slice(None),) * patch_dim +
                        (slice(0, patch_size),) +
                        (slice(None),) * (3 - patch_dim)
                    )
                else:
                    crop_slice = (
                        (slice(None),) * patch_dim +
                        (slice(halo_width[0], halo_width[0] + patch_size),) +
                        (slice(None),) * (3 - patch_dim)
                    )
                return outputs[tuple(crop_slice)].contiguous()
