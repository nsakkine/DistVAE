import torch
import torch.nn as nn
import torch.distributed as dist

from distvae.utils import DistributedEnv


class Patchify(nn.Module):
    def __init__(self, patch_dim: int = -2):
        super().__init__()
        self.group_world_size = DistributedEnv.get_group_world_size()
        self.rank_in_vae_group = DistributedEnv.get_rank_in_vae_group()
        self.patch_dim = patch_dim

    def forward(self, hidden_state):
        patch_dim = self.patch_dim if self.patch_dim >= 0 else hidden_state.ndim + self.patch_dim
        chunks = torch.chunk(hidden_state, self.group_world_size, dim=patch_dim)
        return chunks[self.rank_in_vae_group].clone()


class DePatchify(nn.Module):
    def __init__(self, patch_dim: int = -2):
        super().__init__()
        self.group_world_size = DistributedEnv.get_group_world_size()
        self.rank_in_vae_group = DistributedEnv.get_rank_in_vae_group()
        self.local_rank = DistributedEnv.get_local_rank()
        self.patch_dim = patch_dim

    def forward(self, patch_hidden_state):
        patch_dim = self.patch_dim if self.patch_dim >= 0 else patch_hidden_state.ndim + self.patch_dim
        patch_size_list = [
            torch.empty([1], dtype=torch.int64, device=patch_hidden_state.device)
            for _ in range(self.group_world_size)
        ]
        dist.all_gather(
            patch_size_list,
            torch.tensor([patch_hidden_state.shape[patch_dim]], dtype=torch.int64, device=patch_hidden_state.device),
            group=DistributedEnv.get_vae_group()
        )
        hidden_state_shape = list(patch_hidden_state.shape)
        patch_hidden_state_list = []
        for i in range(self.group_world_size):
            hidden_state_shape[patch_dim] = patch_size_list[i].item()
            patch_hidden_state_list.append(
                torch.empty(hidden_state_shape, dtype=patch_hidden_state.dtype, device=patch_hidden_state.device)
            )
        dist.all_gather(
            patch_hidden_state_list,
            patch_hidden_state.contiguous(),
            group=DistributedEnv.get_vae_group()
        )
        return torch.cat(patch_hidden_state_list, dim=patch_dim)
