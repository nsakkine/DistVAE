from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from distvae.utils import DistributedEnv


class WanAttentionBlockAdapter(torch.nn.Module):
    """Runs attention on the full sequence by gathering along the patch dim, then narrows back to the local patch.

    Supports unequal patch sizes across ranks (e.g. after Patchify without padding).
    Uses DistributedEnv.get_patch_dim() when chunk_dim is not provided.
    """

    def __init__(
        self,
        module: nn.Module,
        chunk_dim: int | None = None,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        self.module = module
        self.chunk_dim = chunk_dim if chunk_dim is not None else DistributedEnv.get_patch_dim()
        self.rank = rank if rank is not None else DistributedEnv.get_rank_in_vae_group()
        self.world_size = world_size if world_size is not None else DistributedEnv.get_group_world_size()

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        chunk_dim = self.chunk_dim
        d = chunk_dim if chunk_dim >= 0 else hidden_states.ndim + chunk_dim
        rank = DistributedEnv.get_rank_in_vae_group()
        world_size = DistributedEnv.get_group_world_size()
        device = hidden_states.device

        size_list = [torch.empty(1, dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(
            size_list,
            torch.tensor([hidden_states.shape[d]], dtype=torch.int64, device=device),
            group=DistributedEnv.get_vae_group(),
        )
        chunk_sizes = [size_list[i].item() for i in range(world_size)]

        base_shape = list(hidden_states.shape)
        gathered_tensors = []
        for i in range(world_size):
            shape = base_shape.copy()
            shape[d] = chunk_sizes[i]
            gathered_tensors.append(torch.empty(shape, dtype=hidden_states.dtype, device=device))
        dist.all_gather(gathered_tensors, hidden_states.contiguous(), group=DistributedEnv.get_vae_group())

        combined_tensor = torch.cat(gathered_tensors, dim=d)
        forward_output = self.module(combined_tensor, *args, **kwargs)
        start_idx = sum(chunk_sizes[:rank])
        local_output = torch.narrow(
            forward_output, d, start_idx, chunk_sizes[rank]
        )
        return local_output