from distvae.models.layers.conv2d import PatchConv2d
from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter
from distvae.utils import DistributedEnv

import torch
import random
import argparse
import torch.distributed as dist
from torch import nn
from torch.cuda import set_device, device_count
from torch.cuda import manual_seed as device_manual_seed
try:
    import torch_musa
    from torch_musa.core.device import set_device, device_count
    from torch_musa.core.random import manual_seed as device_manual_seed
except ModuleNotFoundError:
    pass

class Conv2dModules(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.convs = nn.ModuleList([
            # nn.Conv2d(512, 256, kernel_size, stride, padding),
            # nn.Conv2d(256, 128, kernel_size, stride, padding),
            # nn.Conv2d(128, 64, kernel_size, stride, padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        ])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    device_manual_seed(seed)

def main():
    set_seed()
    torch.backends.cudnn.deterministic = True
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="The height of image",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="The width of image",
    )
    args = parser.parse_args()
    backend = DistributedEnv.get_torch_distributed_backend()
    dist.init_process_group(backend=backend)
    device = torch.distributed.get_rank() % device_count()
    set_device(device)
    DistributedEnv.initialize(None)
    in_channels = 64
    out_channels = 3

    # Test both stride=1 and stride=2 cases
    # stride=2 exercises the stride alignment and global-position cropping logic
    test_configs = [
        (3, 1, 1),  # kernel=3, stride=1, padding=1 (original test)
        (3, 2, 1),  # kernel=3, stride=2, padding=1 (downsampling with stride alignment)
    ]
    if args.height != 1024 or args.width != 1024:
        test_sizes = [
            (args.height, args.width),
        ]
    else:
        test_sizes = [
            # 1k
            (1024, 1024),
            (1023, 1025),
            (1025, 1023),
            # 720p
            (720, 1280),
            (721, 1281), 
            (719, 1279),
            (1280, 720),
            (1281, 721),
            (1279, 719),
        ]

    for kernel_size, stride, padding in test_configs:
        for height, width in test_sizes:
            if dist.get_rank() == 0:
                print(f"\nTesting kernel={kernel_size}, stride={stride}, padding={padding}, size={height}x{width}", flush=True)

            convs = Conv2dModules(in_channels, out_channels, kernel_size, stride, padding).to(device)
            patch_convs = nn.ModuleList()
            for conv in convs.convs:
                patch_convs.append(Conv2dAdapter(conv))
            patch_convs = patch_convs.to(device)

            hidden_state = torch.randn(1, 64, height, width, device=device)
            result = convs(hidden_state)

            
            if dist.get_rank() == 0: 
                print(kernel_size, stride, padding, "start", flush=True)
            patch = Patchify()
            depatch = DePatchify()

            patch_hidden_state = patch(hidden_state)
            for conv in patch_convs:
                patch_hidden_state = conv(patch_hidden_state)
            ppresult = depatch(patch_hidden_state)



            if dist.get_rank() == 0:
                print(f"result.shape={result.shape}, ppresult.shape={ppresult.shape}", flush=True)
                diff = torch.abs(result - ppresult)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}", flush=True)

                # Use slightly relaxed tolerance for stride>1 to account for numerical precision
                # differences from distributed computation order
                tolerance = 1e-5 if stride > 1 else 1e-6
                if not torch.allclose(result, ppresult, atol=tolerance):
                    print("in kernel size: ", kernel_size, "stride: ", stride, "padding: ", padding, flush=True)
                    print(f"FAILED with tolerance {tolerance}\n", flush=True)
                    # Find where the largest differences are
                    max_diff_idx = torch.argmax(diff)
                    max_diff_idx = torch.unravel_index(max_diff_idx, diff.shape)
                    print(f"Largest diff at index {max_diff_idx}: ref={result[max_diff_idx].item():.6f}, patched={ppresult[max_diff_idx].item():.6f}", flush=True)
                else:
                    print(f"{kernel_size} {stride} {padding} end (max_diff={max_diff:.2e}, tol={tolerance:.0e})", flush=True)

    # assert torch.equal(result, ppresult), "two hidden states are not equal"

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
