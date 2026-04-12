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
            # Check if there's a ZeroPad2d before Conv2d (common pattern for stride-2 downsampling)
            pending_pad = None
            for layer in wan_resample.resample:
                if isinstance(layer, nn.ZeroPad2d):
                    # Store padding info to apply to next Conv2d
                    pending_pad = layer.padding
                    # Skip the padding layer - PatchConv will handle padding across ranks
                    continue
                elif isinstance(layer, nn.Conv2d):
                    # If there was a pending ZeroPad2d, we need to adjust the conv
                    if pending_pad is not None:
                        # IMPORTANT: Asymmetric padding preservation vs distributed correctness
                        #
                        # The original model uses ZeroPad2d with asymmetric padding (e.g., (0,1,0,1))
                        # for stride-2 downsampling. While padding=(1,1) is NOT semantically equivalent
                        # to ZeroPad2d(0,1,0,1) in single-GPU inference, we use symmetric padding here
                        # because it's required for distributed correctness:
                        #
                        # 1. PatchConv2d's halo exchange mechanism is designed for symmetric padding
                        # 2. Applying asymmetric padding locally on each rank interferes with boundary
                        #    communication between ranks, causing misalignment at boundaries
                        # 3. Symmetric padding produces correct output dimensions and no visual artifacts
                        # 4. Preserving exact asymmetric padding causes visible horizontal line artifacts
                        #
                        # This is a necessary trade-off: we sacrifice strict single-GPU semantic
                        # equivalence (one extra column/row of padding on left/top) to achieve
                        # distributed correctness. In practice, the slight padding difference does
                        # not cause visual issues, whereas exact asymmetric padding breaks distributed
                        # boundary handling.
                        #
                        # ZeroPad2d.padding is (left, right, top, bottom), typically (0, 1, 0, 1)
                        layer.padding = (1, 1)
                        pending_pad = None
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
        elif hasattr(wan_residual_down_block, "downsamplers") and wan_residual_down_block.downsamplers is not None:
            # Plural form (some other models)
            adapted_downsamplers = []
            for downsampler in wan_residual_down_block.downsamplers:
                if isinstance(downsampler, WanResample):
                    adapted_downsamplers.append(
                        WanResampleDownAdapter(downsampler, conv_block_size=conv_block_size, patch_dim=patch_dim)
                    )
                else:
                    # Keep other types as-is
                    adapted_downsamplers.append(downsampler)
            self.down_block.downsamplers = nn.ModuleList(adapted_downsamplers)

    def forward(self, hidden_states, feat_cache=None, feat_idx=[0]):
        return self.down_block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
