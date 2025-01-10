import torch
import torch.nn as nn
import torch.distributed as dist
import os
from distvae.utils import DistributedEnv

class Patchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_world_size = DistributedEnv.get_group_world_size()
        self.rank_in_vae_group = DistributedEnv.get_rank_in_vae_group()
    def forward(self, hidden_state):
        height = hidden_state.shape[2]
        start_idx = (height + self.group_world_size - 1) // self.group_world_size * self.rank_in_vae_group
        end_idx = min((height + self.group_world_size - 1) // self.group_world_size * (self.rank_in_vae_group + 1), height)

        return hidden_state[:, :, start_idx: end_idx, :].clone()


class DePatchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_world_size = DistributedEnv.get_group_world_size()
        self.rank_in_vae_group = DistributedEnv.get_rank_in_vae_group()
        self.local_rank = DistributedEnv.get_local_rank()
    
    def forward(self, patch_hidden_state):
        patch_height_list = [torch.empty([1], dtype=torch.int64, device=f"cuda:{self.local_rank}") for _ in range(self.group_world_size)]
        dist.all_gather(
            patch_height_list, 
            torch.tensor(
                [patch_hidden_state.shape[2]], 
                dtype=torch.int64, 
                device=f"cuda:{self.local_rank}"
            ),
            group=DistributedEnv.get_vae_group()
        )
        patch_hidden_state_list = [
            torch.empty(
                [patch_hidden_state.shape[0], patch_hidden_state.shape[1], patch_height_list[i].item(), patch_hidden_state.shape[-1]], 
                dtype=patch_hidden_state.dtype,
                device=f"cuda:{self.local_rank}"
            ) for i in range(self.group_world_size)
        ]
        dist.all_gather(
            patch_hidden_state_list, 
            patch_hidden_state.contiguous(),
            group=DistributedEnv.get_vae_group()
        )
        return torch.cat(patch_hidden_state_list, dim=2)

        