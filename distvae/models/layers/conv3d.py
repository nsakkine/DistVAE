from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _triple
from torch.nn.common_types import _size_3_t

from distvae.utils import DistributedEnv
from distvae.models.layers.conv import (
    get_world_size_and_rank,
    calc_patch_index,
    calc_halo_width,
    calc_bottom_halo_width,
    calc_top_halo_width,
    correct_end,
    correct_start,
    build_crop_slice,
    adjust_padding_for_patch,
    exchange_halo,
)

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
        patch_dim: int = -2,
    ) -> None:

        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchConv3d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchConv3d"
        assert patch_dim in (-3, -2, -1, 2, 3, 4), (
            "PatchConv3d patch_dim must be F (-3 or 3) or H (-2 or 2) or W (-1 or 4)"
        )
        self.block_size = block_size
        self.patch_dim = patch_dim
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)

    def _adjust_padding_for_patch(self, padding, rank, world_size, patch_dim: int = 2):
        return adjust_padding_for_patch(padding, rank, world_size, patch_dim, ndim=5)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, f, h, w = input.shape

        group_world_size, global_rank, rank_in_group, local_rank = get_world_size_and_rank()

        if (group_world_size == 1):
            if self.padding_mode != 'zeros':
                return F.conv3d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _triple(0), self.dilation, self.groups)
            return F.conv3d(input, weight, bias, self.stride,
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
            patch_list = [
                torch.zeros(1, dtype=torch.int64, device=DistributedEnv.get_device())
                for _ in range(group_world_size)
            ]
            dist.all_gather(
                patch_list,
                torch.tensor([input.shape[patch_dim]], dtype=torch.int64, device=DistributedEnv.get_device()),
                group=DistributedEnv.get_vae_group()
            )
            patch_index = calc_patch_index(patch_list)
            halo_width = calc_halo_width(
                rank_in_group,
                patch_index,
                kernel_size_patch_dim,
                padding_patch_dim,
                stride_patch_dim
            )
            prev_bottom_halo_width: int = 0
            next_top_halo_width: int = 0
            if rank_in_group != 0:
                prev_bottom_halo_width = calc_bottom_halo_width(
                    rank_in_group - 1,
                    patch_index,
                    kernel_size_patch_dim,
                    padding_patch_dim,
                    stride_patch_dim
                )
            if rank_in_group != group_world_size - 1:
                next_top_halo_width = calc_top_halo_width(
                    rank_in_group + 1,
                    patch_index,
                    kernel_size_patch_dim,
                    padding_patch_dim,
                    stride_patch_dim
                )
                next_top_halo_width = max(0, next_top_halo_width)

            input = exchange_halo(
                input,
                patch_dim,
                patch_index,
                halo_width,
                prev_bottom_halo_width,
                next_top_halo_width,
                group_world_size,
                rank_in_group,
            )

            # 3. do convolution and postprocess
            conv_res: Tensor
            padding = self._adjust_padding_for_patch(
                self._reversed_padding_repeated_twice,
                rank=rank_in_group,
                world_size=group_world_size,
                patch_dim=patch_dim
            )
            bs, channels, f, h, w = input.shape
            if (
                self.block_size == 0 or
                (
                    (f <= self.block_size) if isinstance(self.block_size, int) else (f <= self.block_size[0]) and
                    (h <= self.block_size) if isinstance(self.block_size, int) else (h <= self.block_size[1]) and
                    (w <= self.block_size) if isinstance(self.block_size, int) else (w <= self.block_size[2])
                )
            ):
                if self.padding_mode != 'zeros':
                    conv_res = F.conv3d(F.pad(input, padding, mode=self.padding_mode),
                                        weight, bias, self.stride,
                                        _triple(0), self.dilation, self.groups)
                else:
                    if (
                        stride_patch_dim == 1 and
                        padding_patch_dim == 1 and
                        kernel_size_patch_dim == 3
                    ):
                        conv_res = F.conv3d(input, weight, bias, self.stride,
                                            self.padding, self.dilation, self.groups)
                        crop_slice = [slice(None)] * 5
                        if halo_width[1] == 0:
                            crop_slice[patch_dim] = slice(halo_width[0], None)
                        else:
                            crop_slice[patch_dim] = slice(halo_width[0], -halo_width[1])
                        conv_res = conv_res[tuple(crop_slice)].contiguous()
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
                outputs = torch.cat(outputs, dim=-3)
                crop_slice = build_crop_slice(
                    patch_dim, patch_size, halo_width, outputs.shape[patch_dim], ndim=5
                )
                return outputs[tuple(crop_slice)].contiguous()
