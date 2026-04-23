"""
Test WanEncoderAdapter with real WanEncoder3d from diffusers.

This test uses actual Wan encoder classes instead of mocks to demonstrate
the exact output matching between distributed and single-rank encoders.
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

    # Create real WanEncoder3d
    try:
        from diffusers.models.autoencoders.autoencoder_kl_wan import WanEncoder3d
    except ImportError:
        if rank == 0:
            print("Skipping test - diffusers with Wan VAE not available")
        dist.destroy_process_group()
        return

    # Create a small encoder: 3 spatial downsamples (8x reduction)
    encoder = WanEncoder3d(
        in_channels=3,
        dim=32,  # Small for testing
        z_dim=16,
        dim_mult=[1, 2, 4, 8],  # 32 -> 32 -> 64 -> 128 -> 256
        num_res_blocks=1,  # Minimal for speed
        attn_scales=[],  # No attention
        temperal_downsample=[False, True, True, False],  # 3 spatial downsamples, no temporal
        dropout=0.0,
        is_residual=False,
    )
    encoder = encoder.to(device)
    encoder.eval()

    # Save state dict before patching
    encoder_state_dict = encoder.state_dict()

    # Create distributed adapter
    from distvae.modules.adapters.vae.encoder_adapters import WanEncoderAdapter

    encoder_adapter = WanEncoderAdapter(
        encoder,
        vae_group=None,
        conv_block_size=0,
        patch_dim=-2,
        vae_scale_factor=8,
        use_uniform_patch=True,
    )
    encoder_adapter.eval()

    # Input: (B, C, F, H, W)
    n, c, f = 1, 3, 4
    x_full = torch.randn(n, c, f, height, width, device=device, dtype=torch.float32)

    with torch.no_grad():
        # Reference: create unwrapped encoder on rank 0 only
        if rank == 0:
            encoder_ref = WanEncoder3d(
                in_channels=3,
                dim=32,
                z_dim=16,
                dim_mult=[1, 2, 4, 8],
                num_res_blocks=1,
                attn_scales=[],
                temperal_downsample=[False, True, True, False],
                dropout=0.0,
                is_residual=False,
            )
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
        # Always write debug info
        with open("/tmp/real_wan_encoder_test_debug.txt", "a") as f:
            f.write(f"\nTest executed: world_size={world_size}, height={height}, width={width}\n")
            f.write(f"y_ref shape: {y_ref.shape}, y_dist shape: {y_dist.shape}\n")

        # Check shapes match
        if y_ref.shape != y_dist.shape:
            print(f"Shape mismatch: ref={y_ref.shape}, dist={y_dist.shape}", flush=True)
            success.zero_()
        # Check values are close (use realistic tolerance for real encoder with padding/cropping)
        elif not torch.allclose(y_ref, y_dist, atol=1e-4, rtol=1e-3):
            diff = torch.abs(y_ref - y_dist)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            rel_diff = (diff / (torch.abs(y_ref) + 1e-8)).mean().item()
            print(f"Values mismatch: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}", flush=True)
            with open("/tmp/real_wan_encoder_test_debug.txt", "a") as f:
                f.write(f"FAILED tolerance check: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}\n")
                # Sample some values for debugging
                f.write(f"y_ref sample: {y_ref.flatten()[:10].tolist()}\n")
                f.write(f"y_dist sample: {y_dist.flatten()[:10].tolist()}\n")
            success.zero_()
        else:
            max_diff = torch.abs(y_ref - y_dist).max().item()
            mean_diff = torch.abs(y_ref - y_dist).mean().item()
            msg = f"\n{'='*60}\n"
            msg += f"SUCCESS: WanEncoderAdapter output matches reference encoder\n"
            msg += f"Input shape: {x_full.shape}\n"
            msg += f"Output shape: {y_dist.shape}\n"
            msg += f"Max absolute difference: {max_diff:.2e}\n"
            msg += f"Mean absolute difference: {mean_diff:.2e}\n"
            msg += f"{'='*60}\n"
            print(msg, flush=True)
            # Write to file
            with open("/tmp/real_wan_encoder_test_results.txt", "a") as f:
                f.write(f"\nTest: world_size={world_size}, height={height}, width={width}\n")
                f.write(msg)

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
    base = 29700
    nodeid = request.node.nodeid
    return base + (hash(nodeid) % 10000)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_real_wan_encoder_even_sizes(world_size, master_port, seed=42):
    """Real WanEncoder3d with even sizes divisible by downsampling factor (8)."""
    height, width = 64, 64  # After 3x stride=2: 64 -> 32 -> 16 -> 8
    _run_one(
        world_size=world_size,
        height=height,
        width=width,
        seed=seed,
        master_port=master_port,
    )


@pytest.mark.gloo
def test_real_wan_encoder_larger_input(master_port, seed=42):
    """Real WanEncoder3d with larger input."""
    height, width = 128, 128  # After 3x stride=2: 128 -> 64 -> 32 -> 16
    _run_one(
        world_size=2,
        height=height,
        width=width,
        seed=seed,
        master_port=master_port,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real WanEncoder3d GLOO multi-rank tests")
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args, remainder = parser.parse_known_args()

    # Pass through any remaining args to pytest
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None:
        pytest_args.extend(["-k", f"[{args.world_size}]"])

    sys.exit(pytest.main(pytest_args))
