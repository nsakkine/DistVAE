from typing import List, Tuple, Union

import torch
import torch.distributed as dist
from torch import Tensor

from distvae.utils import DistributedEnv


def get_world_size_and_rank():
    group_world_size = DistributedEnv.get_group_world_size()
    global_rank = DistributedEnv.get_global_rank()
    rank_in_group = DistributedEnv.get_rank_in_vae_group()
    local_rank = DistributedEnv.get_local_rank()
    return group_world_size, global_rank, rank_in_group, local_rank


def calc_patch_index(patch_list: List[Tensor]):
    height_index = []
    cur = 0
    for t in patch_list:
        height_index.append(cur)
        cur += t.item()
    height_index.append(cur)
    return height_index


def calc_bottom_halo_width(rank, height_index, kernel_size, padding = 0, stride = 1):
    assert rank >= 0, "rank should not be smaller than 0"
    assert rank < len(height_index) - 1, "rank should be smaller than the length of height_index - 1"
    assert padding >= 0, "padding should not smaller than 0"
    assert stride > 0, "stride should be larger than 0"

    if rank == DistributedEnv.get_group_world_size() - 1:
        return 0
    nstep_before_bottom = (height_index[rank + 1] + padding - (kernel_size - 1) // 2 + stride - 1) // stride
    assert nstep_before_bottom > 0, "nstep_before_bottom should be larger than 0"
    bottom_halo_width =  (nstep_before_bottom - 1) * stride + kernel_size - padding - height_index[rank + 1]
    return max(0, bottom_halo_width)


def calc_top_halo_width(rank, height_index, kernel_size, padding = 0, stride = 1):
    assert rank >= 0, "rank should not be smaller than 0"
    assert rank < len(height_index) - 1, "rank should be smaller than the length of height_index - 1"
    assert padding >= 0, "padding should not smaller than 0"
    assert stride > 0, "stride should be larger than 0"

    if rank == 0:
        return 0
    nstep_before_top = (height_index[rank] + padding - (kernel_size - 1) // 2 + stride - 1) // stride
    top_halo_width = height_index[rank] - (nstep_before_top * stride - padding)
    return top_halo_width


def calc_halo_width(rank, height_index, kernel_size, padding = 0, stride = 1):
    '''
        Calculate the width of halo region in height dimension.
        The halo region is the region that is used for convolution but not included in the output.
        return value: (top_halo_width, bottom_halo_width)
    '''
    halo_width = [
        calc_top_halo_width(rank, height_index, kernel_size, padding, stride),
        calc_bottom_halo_width(rank, height_index, kernel_size, padding, stride)
    ]
    if rank == 0:
        halo_width[0] = 0
    elif rank == DistributedEnv.get_group_world_size() - 1:
        halo_width[1] = 0
    return tuple(halo_width)


def correct_end(end, kernel_size, stride):
    return ((end + stride - 1) // stride - 1) * stride + kernel_size


def correct_start(start, stride):
    return ((start + stride - 1) // stride) * stride


def build_crop_slice(
    patch_dim: int,
    patch_size: int,
    halo_width: tuple,
    out_len: int,
    ndim: int,
) -> tuple:
    if out_len == patch_size:
        patch_slice = slice(0, patch_size)
    else:
        patch_slice = slice(halo_width[0], halo_width[0] + patch_size)
    return (
        (slice(None),) * patch_dim
        + (patch_slice,)
        + (slice(None),) * (ndim - 1 - patch_dim)
    )


def adjust_padding_for_patch(
    padding: Union[int, tuple],
    rank: int,
    world_size: int,
    patch_dim: int,
    ndim: int,
) -> tuple:
    if ndim == 4:
        if isinstance(padding, tuple):
            padding = list(padding)
        else:
            padding = [padding] * 4
        right_idx = (3, 1)[patch_dim - 2]
        left_idx = (2, 0)[patch_dim - 2]
    else:
        assert ndim == 5
        if isinstance(padding, tuple):
            padding = list(padding)
        else:
            padding = [padding] * 6
        left_idx, right_idx = {2: (4, 5), 3: (2, 3), 4: (0, 1)}[patch_dim]
    if rank == 0:
        padding[right_idx] = 0
    elif rank == world_size - 1:
        padding[left_idx] = 0
    else:
        padding[left_idx] = 0
        padding[right_idx] = 0
    return tuple(padding)


def exchange_halo(
    input: Tensor,
    patch_dim: int,
    patch_index: list,
    halo_width: tuple,
    prev_bottom_halo_width: int,
    next_top_halo_width: int,
    group_world_size: int,
    rank_in_group: int,
) -> Tensor:
    ndim = input.ndim
    indices_end = [slice(None)] * ndim
    indices_end[patch_dim] = slice(-next_top_halo_width, None)
    indices_start = [slice(None)] * ndim
    indices_start[patch_dim] = slice(0, prev_bottom_halo_width)

    to_next = None
    to_prev = None
    top_halo_recv = None
    bottom_halo_recv = None
    global_rank_of_next = None
    global_rank_of_prev = None

    if next_top_halo_width > 0:
        global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
        bottom_halo_send = input[tuple(indices_end)].contiguous()
        to_next = dist.isend(
            bottom_halo_send,
            global_rank_of_next,
            group=DistributedEnv.get_vae_group(),
        )
    if halo_width[0] > 0:
        assert patch_index[rank_in_group] - halo_width[0] >= patch_index[rank_in_group - 1], (
            "width of top halo region is larger than the input tensor of prev rank"
        )
        recv_shape = list(input.shape)
        recv_shape[patch_dim] = halo_width[0]
        top_halo_recv = torch.empty(
            recv_shape, dtype=input.dtype, device=DistributedEnv.get_device()
        )
        global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
        dist.recv(top_halo_recv, global_rank_of_prev, group=DistributedEnv.get_vae_group())
    if prev_bottom_halo_width > 0:
        top_halo_send = input[tuple(indices_start)].contiguous()
        if global_rank_of_prev is None:
            global_rank_of_prev = DistributedEnv.get_global_rank_from_group_rank(rank_in_group - 1)
        to_prev = dist.isend(
            top_halo_send,
            global_rank_of_prev,
            group=DistributedEnv.get_vae_group(),
        )
    if halo_width[1] > 0:
        assert patch_index[rank_in_group + 1] + halo_width[1] <= patch_index[rank_in_group + 2], (
            "width of bottom halo region is larger than the input tensor of next rank"
        )
        recv_shape = list(input.shape)
        recv_shape[patch_dim] = halo_width[1]
        bottom_halo_recv = torch.empty(
            recv_shape, dtype=input.dtype, device=DistributedEnv.get_device()
        )
        if global_rank_of_next is None:
            global_rank_of_next = DistributedEnv.get_global_rank_from_group_rank(rank_in_group + 1)
        dist.recv(
            bottom_halo_recv,
            global_rank_of_next,
            group=DistributedEnv.get_vae_group(),
        )
    if halo_width[0] < 0:
        trim_slice = [slice(None)] * ndim
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
    return input

