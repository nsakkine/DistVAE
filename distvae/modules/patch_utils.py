import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from distvae.utils import DistributedEnv


def _pad_tuple_for_dim(ndim: int, patch_dim: int, pad_amount: int):
    """Build F.pad tuple to append pad_amount at end of patch_dim. Last dims order: W,H for 4D; W,H,F for 5D."""
    if ndim == 4:
        # (N,C,H,W) -> (W_l, W_r, H_l, H_r)
        if patch_dim == 2:
            return (0, 0, 0, pad_amount)
        if patch_dim == 3:
            return (0, pad_amount, 0, 0)
    else:
        # 5D (N,C,F,H,W) -> (W_l, W_r, H_l, H_r, F_l, F_r)
        if patch_dim == 2:
            return (0, 0, 0, 0, 0, pad_amount)
        if patch_dim == 3:
            return (0, 0, 0, pad_amount, 0, 0)
        if patch_dim == 4:
            return (0, pad_amount, 0, 0, 0, 0)
    raise ValueError(f"patch_dim {patch_dim} not supported for ndim {ndim}")


class Patchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_world_size = DistributedEnv.get_group_world_size()
        self.rank_in_vae_group = DistributedEnv.get_rank_in_vae_group()
        self.patch_dim = DistributedEnv.get_patch_dim()

    def forward(self, hidden_state):
        d = self.patch_dim
        length = hidden_state.shape[d]
        padded_length = ((length + self.group_world_size - 1) // self.group_world_size) * self.group_world_size
        if padded_length > length:
            pad_amount = padded_length - length
            pad_tuple = _pad_tuple_for_dim(hidden_state.ndim, d, pad_amount)
            hidden_state = F.pad(hidden_state, pad_tuple, mode="constant", value=0.0)
            length = padded_length
        chunk_size = (length + self.group_world_size - 1) // self.group_world_size
        start_idx = chunk_size * self.rank_in_vae_group
        end_idx = min(chunk_size * (self.rank_in_vae_group + 1), length)
        indices = [slice(None)] * hidden_state.ndim
        indices[d] = slice(start_idx, end_idx)
        return hidden_state[tuple(indices)].clone()


class DePatchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_world_size = DistributedEnv.get_group_world_size()
        self.rank_in_vae_group = DistributedEnv.get_rank_in_vae_group()
        self.local_rank = DistributedEnv.get_local_rank()
        self.patch_dim = DistributedEnv.get_patch_dim()

    def forward(self, patch_hidden_state, original_length=None):
        d = self.patch_dim
        patch_size_list = [torch.empty([1], dtype=torch.int64, device=DistributedEnv.get_device()) for _ in range(self.group_world_size)]
        dist.all_gather(
            patch_size_list,
            torch.tensor([patch_hidden_state.shape[d]], dtype=torch.int64, device=DistributedEnv.get_device()),
            group=DistributedEnv.get_vae_group()
        )
        hidden_state_shape = list(patch_hidden_state.shape)
        patch_hidden_state_list = []
        for i in range(self.group_world_size):
            hidden_state_shape[d] = patch_size_list[i].item()
            patch_hidden_state_list.append(
                torch.empty(hidden_state_shape, dtype=patch_hidden_state.dtype, device=DistributedEnv.get_device())
            )
        dist.all_gather(
            patch_hidden_state_list,
            patch_hidden_state.contiguous(),
            group=DistributedEnv.get_vae_group()
        )
        result = torch.cat(patch_hidden_state_list, dim=d)
        if original_length is not None:
            indices = [slice(None)] * result.ndim
            indices[d] = slice(0, original_length)
            result = result[tuple(indices)].contiguous()
        return result
