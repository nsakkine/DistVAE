from typing import Optional

import torch
import torch.nn as nn

from distvae.models.upsampling import PatchUpsample2D
from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter, WanCausalConv3dAdapter
from distvae.modules.adapters.resnet_adapters import WanResidualBlockAdapter
from diffusers.models.upsampling import Upsample2D
from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample, WanUpBlock


class Upsample2DAdapter(nn.Module):
    def __init__(
        self, 
        upsample2d: Upsample2D,
        *,
        conv_block_size = 0,
    ):
        super().__init__()
        assert upsample2d.norm is None, "upsample2dBlock2DAdapter does not support normalization"
        if upsample2d.name == "conv":
            assert not isinstance(upsample2d.conv, nn.ConvTranspose2d), "upsample2dBlock2DAdapter does not support transpose conv"
        else:
            assert not isinstance(upsample2d.Conv2d_0, nn.ConvTranspose2d), "upsample2dBlock2DAdapter does not support transpose conv"
        self.upsample2d = PatchUpsample2D(
            channels=upsample2d.channels,
            use_conv=upsample2d.use_conv,
            use_conv_transpose=upsample2d.use_conv_transpose,
            out_channels=upsample2d.out_channels,
            name=upsample2d.name,
            kernel_size=None,
            padding=1,
            interpolate=upsample2d.interpolate
        )
        if upsample2d.name == "conv":
            self.upsample2d.conv = Conv2dAdapter(upsample2d.conv, block_size=conv_block_size)
        else:
            self.upsample2d.Conv2d_0 = Conv2dAdapter(upsample2d.Conv2d_0, block_size=conv_block_size)
        

    def forward(
        self, hidden_states: torch.FloatTensor, output_size: Optional[int] = None, *args, **kwargs
    ):
        return self.upsample2d(hidden_states, output_size, *args, **kwargs)


class WanResampleAdapter(nn.Module):
    def __init__(self, wan_resample: WanResample, conv_block_size = 0):
        super().__init__()
        assert isinstance(wan_resample, WanResample), "WanResampleAdapter does not support resample except WanResample"
        self.resample = wan_resample
        if hasattr(wan_resample, "time_conv"):
            wan_resample.time_conv = WanCausalConv3dAdapter(wan_resample.time_conv, block_size=conv_block_size)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        return self.resample(x, feat_cache=feat_cache, feat_idx=feat_idx)


class WanUpBlockAdapter(nn.Module):
    def __init__(self, wan_up_block: WanUpBlock, conv_block_size = 0):
        super().__init__()
        assert isinstance(wan_up_block, WanUpBlock), "WanUpBlockAdapter does not support up block except WanUpBlock"
        self.up_block = wan_up_block
        self.up_block.resnets = nn.ModuleList([
            WanResidualBlockAdapter(resnet, conv_block_size=conv_block_size) for resnet in wan_up_block.resnets
        ])
        if wan_up_block.upsamplers is not None:
            self.up_block.upsamplers = nn.ModuleList([
                WanResampleAdapter(upsampler, conv_block_size=conv_block_size) for upsampler in wan_up_block.upsamplers
            ])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        return self.up_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
