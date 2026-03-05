import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from distvae.utils import DistributedEnv


class Patchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_world_size = DistributedEnv.get_group_world_size()
        self.rank_in_vae_group = DistributedEnv.get_rank_in_vae_group()

    def forward(self, hidden_state):
        length = hidden_state.shape[2]
        padded_length = ((length + self.group_world_size - 1) // self.group_world_size) * self.group_world_size
        if padded_length > length:
            pad_amount = padded_length - length
            if hidden_state.ndim == 4:
                # (N, C, H, W): F.pad order (W_left, W_right, H_left, H_right)
                hidden_state = F.pad(hidden_state, (0, 0, 0, pad_amount), mode="constant", value=0.0)
            else:
                # 5D (N, C, F, H, W): F.pad order (W_l, W_r, H_l, H_r, F_l, F_r)
                hidden_state = F.pad(hidden_state, (0, 0, 0, 0, 0, pad_amount), mode="constant", value=0.0)
            length = padded_length
        start_idx = (length + self.group_world_size - 1) // self.group_world_size * self.rank_in_vae_group
        end_idx = min((length + self.group_world_size - 1) // self.group_world_size * (self.rank_in_vae_group + 1), length)
        return hidden_state[:, :, start_idx:end_idx, ...].clone()


class DePatchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_world_size = DistributedEnv.get_group_world_size()
        self.rank_in_vae_group = DistributedEnv.get_rank_in_vae_group()
        self.local_rank = DistributedEnv.get_local_rank()
    
    def forward(self, patch_hidden_state, original_length=None):
        patch_height_list = [torch.empty([1], dtype=torch.int64, device=DistributedEnv.get_device()) for _ in range(self.group_world_size)]
        dist.all_gather(
            patch_height_list,
            torch.tensor(
                [patch_hidden_state.shape[2]],
                dtype=torch.int64,
                device=DistributedEnv.get_device()
            ),
            group=DistributedEnv.get_vae_group()
        )
        hidden_state_shape = list(patch_hidden_state.shape)
        patch_hidden_state_list = []
        for i in range(self.group_world_size):
            hidden_state_shape[2] = patch_height_list[i].item()
            patch_hidden_state_list.append(
                torch.empty(
                    hidden_state_shape,
                    dtype=patch_hidden_state.dtype,
                    device=DistributedEnv.get_device()
                )
            )
        dist.all_gather(
            patch_hidden_state_list,
            patch_hidden_state.contiguous(),
            group=DistributedEnv.get_vae_group()
        )
        result = torch.cat(patch_hidden_state_list, dim=2)
        if original_length is not None:
            result = result[:, :, :original_length, ...].contiguous()
        return result
