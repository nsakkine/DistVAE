from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from distvae.utils import DistributedEnv


class WanAttentionBlockAdapter(torch.nn.Module):
    """Runs attention on the full sequence by gathering along the patch dim, then narrows back to the local patch.

    Requires all ranks to have the same hidden_states.shape (e.g. after Patchify with even split).
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
        # Use patch dim from DistributedEnv so this block matches Patchify/DePatchify and PatchConv*
        self.chunk_dim = chunk_dim if chunk_dim is not None else DistributedEnv.get_patch_dim()
        self.rank = rank if rank is not None else DistributedEnv.get_rank_in_vae_group()
        self.world_size = world_size if world_size is not None else DistributedEnv.get_group_world_size()

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        gathered_tensors = [torch.empty_like(hidden_states) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, hidden_states.contiguous(), group=DistributedEnv.get_vae_group())

        combined_tensor = torch.cat(gathered_tensors, dim=self.chunk_dim)

        forward_output = self.module(combined_tensor, *args, **kwargs)

        chunk_sizes = [t.size(self.chunk_dim) for t in gathered_tensors]
        start_idx = sum(chunk_sizes[: self.rank])
        local_output = torch.narrow(
            forward_output, self.chunk_dim, start_idx, chunk_sizes[self.rank]
        )
        return local_output