import torch.nn as nn
from distvae.models.layers.wan.zeropadconv2d import WanZeroPadConv2d
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
        use_uniform_patch: bool = True,
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
                wan_resample.time_conv,
                block_size=conv_block_size,
                patch_dim=patch_dim,
                use_uniform_patch=use_uniform_patch,
            )

        # Adapt the resample layers
        if isinstance(wan_resample.resample, nn.Sequential):
            count = 0
            for layer in wan_resample.resample:
                count += 1
                if isinstance(layer, nn.ZeroPad2d):
                    continue
                elif isinstance(layer, nn.Conv2d):
                    in_channels = layer.in_channels
                    out_channels = layer.out_channels
                    kernel_size = layer.kernel_size
                    if (
                        isinstance(layer.stride, int) and layer.stride != 2 or
                        isinstance(layer.stride, tuple) and (layer.stride[0] != 2 or layer.stride[1] != 2)
                    ):
                        raise ValueError(f"Unsupported stride: {layer.stride}")
                    if (
                        isinstance(layer.padding, int) and layer.padding != 0 or
                        isinstance(layer.padding, tuple) and (sum(layer.padding) != 0)
                    ):
                        raise ValueError(f"Unsupported padding: {layer.padding}")
                    dilation = layer.dilation
                    groups = layer.groups
                    bias = layer.bias is not None
                    device = layer.weight.device
                    dtype = layer.weight.dtype
                    _weight = layer.weight
                    _bias = layer.bias
                else:
                    raise ValueError(f"Unsupported layer type: {type(layer)}")
            if count != 2:
                raise ValueError(f"WanResampleDownAdapter expects 2 layers, got {count}")

            self.resample.resample = WanZeroPadConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(2, 2),
                dilation=dilation,
                groups=groups,
                bias=bias,
                device=device,
                dtype=dtype,
                reversed_zero_padding=(0, 1, 0, 1),
                block_size=conv_block_size,
                patch_dim=patch_dim,
                use_uniform_patch=use_uniform_patch,                
            )
            self.resample.resample.weight.data = _weight.data
            if _bias is not None:
                self.resample.resample.bias.data = _bias.data
        else:
            # Single conv layer
            if isinstance(wan_resample.resample, nn.Conv2d):
                self.resample.resample = Conv2dAdapter(
                    wan_resample.resample,
                    block_size=conv_block_size,
                    patch_dim=patch_dim,
                    use_uniform_patch=use_uniform_patch
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
        use_uniform_patch: bool = True,
    ):
        super().__init__()
        assert isinstance(wan_residual_down_block, WanResidualDownBlock), (
            "WanResidualDownBlockAdapter only supports WanResidualDownBlock"
        )
        if patch_dim == -3:
            raise ValueError("WanResidualDownBlockAdapter does not support patch_dim F (-3); use H (-2) or W (-1).")

        self.down_block = wan_residual_down_block
        if hasattr(wan_residual_down_block, "resnets"):
            adapted_resnets = []
            for resnet in wan_residual_down_block.resnets:
                adapted_resnets.append(
                    WanResidualBlockAdapter(
                        resnet,
                        conv_block_size=conv_block_size,
                        patch_dim=patch_dim,
                        use_uniform_patch=use_uniform_patch
                    )
                )
            self.down_block.resnets = nn.ModuleList(adapted_resnets)
        if hasattr(wan_residual_down_block, "downsampler") and wan_residual_down_block.downsampler is not None:
            if isinstance(wan_residual_down_block.downsampler, WanResample):
                self.down_block.downsampler = WanResampleDownAdapter(
                    wan_residual_down_block.downsampler,
                    conv_block_size=conv_block_size,
                    patch_dim=patch_dim,
                    use_uniform_patch=use_uniform_patch
                )
        
    def forward(self, hidden_states, feat_cache=None, feat_idx=[0]):
        return self.down_block(hidden_states, feat_cache=feat_cache, feat_idx=feat_idx)
