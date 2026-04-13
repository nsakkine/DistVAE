import torch.nn as nn
from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter, WanCausalConv3dAdapter
from distvae.modules.adapters.resnet_adapters import WanResidualBlockAdapter
from diffusers.models.autoencoders.autoencoder_kl_wan import WanResample, WanResidualDownBlock


class WanResampleDownAdapter(nn.Module):
    """
    Adapter for WanResample used in downsampling operations.
    Handles temporal convolution and spatial downsampling with distributed patching.
    """
    def __init__(
        self,
        wan_resample: WanResample,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        assert isinstance(wan_resample, WanResample), (
            "WanResampleDownAdapter does not support resample except WanResample"
        )
        self.resample = wan_resample
        if patch_dim == -3:
            raise ValueError("WanResampleDownAdapter does not support patch_dim F (-3); use H (-2) or W (-1).")

        # Adapt time_conv if present
        if hasattr(wan_resample, "time_conv") and wan_resample.time_conv is not None:
            wan_resample.time_conv = WanCausalConv3dAdapter(
                wan_resample.time_conv, block_size=conv_block_size, patch_dim=patch_dim
            )

        # Adapt the resample layers
        if isinstance(wan_resample.resample, nn.Sequential):
            resample = []
            for layer in wan_resample.resample:
                if isinstance(layer, nn.ZeroPad2d):
                    # Skip ZeroPad2d layers - we'll use symmetric Conv2d padding instead
                    # Original model: ZeroPad2d(0,1,0,1) + Conv2d(padding=0, stride=2)
                    # Distributed: Conv2d(padding=1, stride=2)
                    #
                    # This changes from asymmetric (0,1,0,1) to symmetric (1,1,1,1) padding,
                    # which adds 1 extra pixel on left/top edges. However, this is necessary
                    # for distributed correctness with PatchConv2d's halo exchange mechanism.
                    # The extra left/top padding is compensated by:
                    # 1. Adding edge_pad before encoding (in encoder_adapters.py)
                    # 2. Cropping with offset after encoding (in encoder_adapters.py)
                    continue
                elif isinstance(layer, nn.Conv2d):
                    # Set symmetric padding for distributed halo exchange compatibility
                    layer.padding = (1, 1)
                    resample.append(
                        Conv2dAdapter(layer, block_size=conv_block_size, patch_dim=patch_dim)
                    )
                else:
                    resample.append(layer)
            self.resample.resample = nn.Sequential(*resample)
        else:
            # Single conv layer
            if isinstance(wan_resample.resample, nn.Conv2d):
                self.resample.resample = Conv2dAdapter(
                    wan_resample.resample, block_size=conv_block_size, patch_dim=patch_dim
                )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        return self.resample(x, feat_cache=feat_cache, feat_idx=feat_idx)


class WanResidualDownBlockAdapter(nn.Module):
    """
    Adapter for WanResidualDownBlock used in the encoder (Wan2.2).
    Patches residual blocks and downsampler with distributed processing support.
    """
    def __init__(
        self,
        wan_residual_down_block: WanResidualDownBlock,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        assert isinstance(wan_residual_down_block, WanResidualDownBlock), (
            "WanResidualDownBlockAdapter only supports WanResidualDownBlock"
        )
        if patch_dim == -3:
            raise ValueError("WanResidualDownBlockAdapter does not support patch_dim F (-3); use H (-2) or W (-1).")

        self.down_block = wan_residual_down_block

        # Adapt residual blocks
        if hasattr(wan_residual_down_block, "resnets") and wan_residual_down_block.resnets is not None:
            adapted_resnets = []
            for resnet in wan_residual_down_block.resnets:
                adapted_resnets.append(
                    WanResidualBlockAdapter(resnet, conv_block_size=conv_block_size, patch_dim=patch_dim)
                )
            self.down_block.resnets = nn.ModuleList(adapted_resnets)

        # Adapt downsampler if present (check both singular and plural forms)
        if hasattr(wan_residual_down_block, "downsampler") and wan_residual_down_block.downsampler is not None:
            # Singular form (5B model, others)
            if isinstance(wan_residual_down_block.downsampler, WanResample):
                self.down_block.downsampler = WanResampleDownAdapter(
                    wan_residual_down_block.downsampler, conv_block_size=conv_block_size, patch_dim=patch_dim
                )
        
    def forward(self, hidden_states, feat_cache=None, feat_idx=[0]):
        return self.down_block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
