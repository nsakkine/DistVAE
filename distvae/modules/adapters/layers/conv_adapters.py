import torch
import torch.nn as nn

from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from distvae.models.layers.conv2d import PatchConv2d
from distvae.models.layers.conv3d import PatchConv3d


class Conv2dAdapter(nn.Module):
    def __init__(
        self, 
        conv2d: nn.Conv2d,
        *,
        block_size = 0,
    ):
        super().__init__()
        for i in conv2d.dilation:
            assert i == 1, "dilation is not supported in Conv2dAdapter"
        self.conv2d = PatchConv2d(
            in_channels=conv2d.in_channels,
            out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size,
            stride=conv2d.stride,
            padding=conv2d.padding,
            dilation=conv2d.dilation,
            groups=conv2d.groups,
            bias=conv2d.bias is not None,
            padding_mode=conv2d.padding_mode,
            device=conv2d.weight.device,
            dtype=conv2d.weight.dtype,
            block_size=block_size,
        )
        self.conv2d.weight.data = conv2d.weight.data
        self.conv2d.bias.data = conv2d.bias.data

    def forward(self, x):
        return self.conv2d(x)


class Conv3dAdapter(nn.Module):
    def __init__(
        self, 
        conv3d: nn.Conv3d,
        *,
        block_size = 0,
    ):
        super().__init__()
        for i in conv3d.dilation:
            assert i == 1, "dilation is not supported in Conv3dAdapter"
        self.conv3d = PatchConv3d(
            in_channels=conv3d.in_channels,
            out_channels=conv3d.out_channels,
            kernel_size=conv3d.kernel_size,
            stride=conv3d.stride,
            padding=conv3d.padding,
            dilation=conv3d.dilation,
            groups=conv3d.groups,
            bias=conv3d.bias is not None,
            padding_mode=conv3d.padding_mode,
            device=conv3d.weight.device,
            dtype=conv3d.weight.dtype,
            block_size=block_size,
        )
        self.conv3d.weight.data = conv3d.weight.data
        self.conv3d.bias.data = conv3d.bias.data

    def forward(self, x):
        return self.conv3d(x)


class WanCausalConv3dAdapter(nn.Module):
    def __init__(
        self, 
        causal_conv3d: WanCausalConv3d,
        *,
        block_size = 0,
    ):
        super().__init__()
        self.conv3d = PatchConv3d(
            in_channels=causal_conv3d.in_channels,
            out_channels=causal_conv3d.out_channels,
            kernel_size=causal_conv3d.kernel_size,
            stride=causal_conv3d.stride,
            padding=causal_conv3d.padding,
            block_size=block_size,
        )
        self.conv3d.weight.data = causal_conv3d.weight.data
        self.conv3d.bias.data = causal_conv3d.bias.data
        self._padding = causal_conv3d._padding
        self.padding = causal_conv3d.padding

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = nn.functional.pad(x, padding)
        return self.conv3d(x)