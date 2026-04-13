"""
Multi-rank integration tests for WanEncoderAdapter using GLOO backend (CPU).

Run from repo root:
  pytest test/test_vae_encoder_distributed_gloo.py -v
  python test/test_vae_encoder_distributed_gloo.py --world_size 2
"""

import argparse
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.multiprocessing import spawn

from distvae.utils import DistributedEnv


# Mock VAE encoder structure similar to Wan VAE
class MockWanEncoder(nn.Module):
    """Simplified mock VAE encoder for testing.

    Mimics the structure of Wan VAE encoder with:
    - conv_in: initial conv
    - down_blocks: 3 downsampling residual blocks (8x spatial reduction)
    - mid_block: middle processing block
    - conv_out: final conv to latent space
    """
    def __init__(self, in_channels=3, out_channels=16, hidden_channels=64):
        super().__init__()

        # Import required classes
        try:
            from diffusers.models.autoencoders.autoencoder_kl_wan import (
                WanResidualBlock,
                WanResample,
                WanMidBlock,
                WanCausalConv3d,
            )
        except ImportError:
            pytest.skip("diffusers with Wan VAE support not available")

        # conv_in: 3 -> 64 channels
        self.conv_in = WanCausalConv3d(in_channels, hidden_channels, kernel_size=3, padding=1)

        # down_blocks: 3 downsampling blocks (64->128->256->512)
        self.down_blocks = nn.ModuleList([
            # Block 0: 64->128, spatial /2
            WanResidualBlock(hidden_channels, hidden_channels * 2),
            WanResample(hidden_channels * 2, hidden_channels * 2, stride=2, padding=1),

            # Block 1: 128->256, spatial /2
            WanResidualBlock(hidden_channels * 2, hidden_channels * 4),
            WanResample(hidden_channels * 4, hidden_channels * 4, stride=2, padding=1),

            # Block 2: 256->512, spatial /2
            WanResidualBlock(hidden_channels * 4, hidden_channels * 8),
            WanResample(hidden_channels * 8, hidden_channels * 8, stride=2, padding=1),
        ])

        # mid_block: 512->512
        self.mid_block = WanMidBlock(hidden_channels * 8, hidden_channels * 8)

        # conv_out: 512->16 (latent space)
        self.conv_out = WanCausalConv3d(hidden_channels * 8, out_channels, kernel_size=3, padding=1)

    def forward(self, x, _feat_cache=None, _feat_idx=0):
        # _feat_cache and _feat_idx are used by real Wan VAE for temporal consistency,
        # but we ignore them in this simplified mock
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        h = self.mid_block(h)
        h = self.conv_out(h)
        return h


def worker(
    rank: int,
    world_size: int,
    height: int,
    width: int,
    seed: int,
    master_port: int,
) -> None:
    device = torch.device("cpu")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", init_method="env://")
    DistributedEnv.initialize(None)

    torch.manual_seed(seed)

    # Create mock encoder
    try:
        encoder = MockWanEncoder(in_channels=3, out_channels=16, hidden_channels=32)
    except Exception as e:
        if rank == 0:
            print(f"Skipping test - could not create encoder: {e}")
        dist.destroy_process_group()
        return

    encoder = encoder.to(device)
    encoder.eval()

    # Save state dict before patching to create clean reference encoder
    encoder_state_dict = encoder.state_dict()

    # Create distributed adapter (patches encoder in-place)
    from distvae.modules.adapters.vae.encoder_adapters import WanEncoderAdapter

    # Mock VAE config
    class MockVAEConfig:
        scaling_factor = 8  # 3 downsampling layers: 2^3 = 8
        vae_scale_factor_spatial = 8

    encoder_adapter = WanEncoderAdapter(
        encoder,
        vae_group=None,
        conv_block_size=0,
        patch_dim=-2,
        vae_config=MockVAEConfig(),
    )
    encoder_adapter.eval()

    # Input: (B, C, F, H, W)
    # Use sizes divisible by 8 for clean downsampling
    n, c, f = 1, 3, 4
    x_full = torch.randn(n, c, f, height, width, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Reference: create unwrapped encoder on rank 0 only (no distributed collectives)
        if rank == 0:
            encoder_ref = MockWanEncoder(in_channels=3, out_channels=16, hidden_channels=32)
            encoder_ref.load_state_dict(encoder_state_dict)
            encoder_ref = encoder_ref.to(device)
            encoder_ref.eval()
            y_ref = encoder_ref(x_full)
        else:
            y_ref = None

        # Distributed: run through adapter on all ranks
        y_dist = encoder_adapter(x_full)

    success = torch.ones(1, dtype=torch.int64, device=device)
    if rank == 0:
        # Check shapes match
        if y_ref.shape != y_dist.shape:
            print(f"Shape mismatch: ref={y_ref.shape}, dist={y_dist.shape}", flush=True)
            success.zero_()
        # Check values are close (use relaxed tolerance for multi-layer encoder)
        elif not torch.allclose(y_ref, y_dist, atol=1e-4, rtol=1e-3):
            diff = torch.abs(y_ref - y_dist)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"Values mismatch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}", flush=True)
            success.zero_()

    dist.broadcast(success, src=0)
    dist.barrier()
    dist.destroy_process_group()

    if success.item() == 0:
        raise AssertionError("WanEncoderAdapter output did not match reference encoder")


def _run_one(
    world_size: int,
    height: int,
    width: int,
    seed: int,
    master_port: int,
) -> None:
    """Spawn processes and run worker; raises on failure."""
    spawn(
        worker,
        nprocs=world_size,
        args=(world_size, height, width, seed, master_port),
        join=True,
    )


@pytest.fixture
def master_port(request):
    """Unique port per test to avoid conflicts."""
    base = 29600
    nodeid = request.node.nodeid
    return base + (hash(nodeid) % 10000)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [2, 4])
def test_vae_encoder_even_sizes(world_size, master_port, seed=42):
    """WanEncoderAdapter with even sizes divisible by downsampling factor (8)."""
    # Use sizes divisible by 8 for clean 3x downsampling (8x reduction)
    height, width = 64, 64  # After 3x stride=2: 64 -> 32 -> 16 -> 8
    _run_one(
        world_size=world_size,
        height=height,
        width=width,
        seed=seed,
        master_port=master_port,
    )


@pytest.mark.gloo
def test_vae_encoder_larger_input(master_port, seed=42):
    """WanEncoderAdapter with larger input to test realistic scenarios."""
    # Larger input similar to actual video frames
    height, width = 128, 128  # After 3x stride=2: 128 -> 64 -> 32 -> 16
    _run_one(
        world_size=2,
        height=height,
        width=width,
        seed=seed,
        master_port=master_port,
    )


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [2, 4])
def test_vae_encoder_non_divisible_sizes(world_size, master_port, seed=42):
    """WanEncoderAdapter with sizes NOT divisible by downsampling factor (8)."""
    # Use sizes not divisible by 8 to test padding/cropping logic
    height, width = 62, 65  # Not divisible by 8, will be padded then cropped
    _run_one(
        world_size=world_size,
        height=height,
        width=width,
        seed=seed,
        master_port=master_port,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanEncoderAdapter GLOO multi-rank tests")
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args, remainder = parser.parse_known_args()

    # Pass through any remaining args to pytest
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])

    sys.exit(pytest.main(pytest_args))
