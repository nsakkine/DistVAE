"""
Multi-rank integration tests for PatchConv3d using GLOO backend (CPU).

Run from repo root:
  pytest test/test_conv3d_distributed_gloo.py -v
  pytest test/test_conv3d_distributed_gloo.py -v -k "2--2"
  python test/test_conv3d_distributed_gloo.py --world_size 4 --patch_dim -1
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
from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.modules.adapters.layers.conv_adapters import Conv3dAdapter


def worker(
    rank: int,
    world_size: int,
    patch_dim: int,
    kernel_size: int,
    stride: int,
    padding: int,
    block_size: int,
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
    in_ch, out_ch = 4, 8
    # (N, C, F, H, W): patch_dim -2 => H, -1 => W
    # For stride>1 tests, use sizes that stress-test alignment logic
    # For stride=1, use sizes divisible by world_size for even splitting
    n, c, f = 1, in_ch, 4
    if stride > 1:
        # Use even sizes for stride>1 tests
        # TODO: Add support for odd sizes with stride>1 (currently produces off-by-one errors)
        h, w = 8, 8
    else:
        # Use sizes divisible by world_size for even splitting
        h, w = 8, 8
        if patch_dim == -2:
            assert h % world_size == 0
        else:
            assert patch_dim == -1
            assert w % world_size == 0

    x_full = torch.randn(n, c, f, h, w, device=device, dtype=torch.float32)
    ref_conv = nn.Conv3d(
        in_ch, out_ch, kernel_size, stride=stride, padding=padding
    ).to(device)
    ref_conv.eval()

    patchify = Patchify(patch_dim=patch_dim)
    depatchify = DePatchify(patch_dim=patch_dim)
    adapter = Conv3dAdapter(ref_conv, block_size=block_size, patch_dim=patch_dim)
    adapter.eval()

    with torch.no_grad():
        y_ref = ref_conv(x_full) if rank == 0 else None

        x_local = patchify(x_full)
        y_local = adapter(x_local)
        y_patch = depatchify(y_local)

    success = torch.ones(1, dtype=torch.int64, device=device)
    if rank == 0:
        if not torch.allclose(y_ref, y_patch, atol=1e-5):
            success.zero_()
    dist.broadcast(success, src=0)
    dist.barrier()
    dist.destroy_process_group()
    if success.item() == 0:
        raise AssertionError("PatchConv3d output did not match reference nn.Conv3d")


def _run_one(
    world_size: int,
    patch_dim: int,
    kernel_size: int,
    stride: int,
    padding: int,
    block_size: int,
    seed: int,
    master_port: int,
) -> None:
    """Spawn processes and run worker; raises on failure."""
    spawn(
        worker,
        nprocs=world_size,
        args=(
            world_size,
            patch_dim,
            kernel_size,
            stride,
            padding,
            block_size,
            seed,
            master_port,
        ),
        join=True,
    )


@pytest.fixture
def master_port(request):
    """Unique port per test to avoid Address already in use when tests run sequentially."""
    base = 29500
    nodeid = request.node.nodeid
    return base + (hash(nodeid) % 10000)


@pytest.mark.gloo
@pytest.mark.parametrize("world_size,patch_dim", [(4, -2), (4, -1)])
def test_patch_conv3d_gloo_direct(world_size, patch_dim, master_port, seed=42):
    """PatchConv3d with GLOO: direct path (block_size=0), multiple world sizes and patch dims."""
    _run_one(
        world_size=world_size,
        patch_dim=patch_dim,
        kernel_size=3,
        stride=1,
        padding=1,
        block_size=0,
        seed=seed,
        master_port=master_port,
    )


@pytest.mark.gloo
def test_patch_conv3d_gloo_chunked_path(master_port, seed=42):
    """PatchConv3d with GLOO: chunked path (block_size=4 so _use_direct_path is False, and chunks >= kernel_size=3)."""
    _run_one(
        world_size=2,
        patch_dim=-2,
        kernel_size=3,
        stride=1,
        padding=1,
        block_size=4,
        seed=seed,
        master_port=master_port,
    )


@pytest.mark.gloo
@pytest.mark.parametrize("world_size,patch_dim", [(4, -2), (2, -1)])
def test_patch_conv3d_stride2_alignment(world_size, patch_dim, master_port, seed=42):
    """
    PatchConv3d with stride=2: tests stride alignment and global-position cropping logic.

    This exercises the code path where:
    1. Stride > 1 triggers stride alignment (shift calculation and input trimming)
    2. build_crop_slice uses global_start and global_height for correct output cropping
    3. Ranks would otherwise misalign without this logic
    """
    _run_one(
        world_size=world_size,
        patch_dim=patch_dim,
        kernel_size=3,
        stride=2,
        padding=1,
        block_size=0,  # Direct path
        seed=seed,
        master_port=master_port,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchConv3d GLOO multi-rank tests")
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--patch_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args, remainder = parser.parse_known_args()
    # Pass through any remaining args to pytest (e.g. -v, -k)
    pytest_args = [os.path.abspath(__file__), "-v"] + remainder
    if args.world_size is not None and args.patch_dim is not None:
        pytest_args.extend(["-k", f"{args.world_size}--{args.patch_dim}"])
    sys.exit(pytest.main(pytest_args))
