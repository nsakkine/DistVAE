from typing import List

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
        _calc_top_halo_width(rank, height_index, kernel_size, padding, stride),
        _calc_bottom_halo_width(rank, height_index, kernel_size, padding, stride)
    ]
    if rank == 0:
        halo_width[0] = 0
    elif rank == DistributedEnv.get_group_world_size() - 1:
        halo_width[1] = 0
    return tuple(halo_width)

