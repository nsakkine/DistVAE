from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

from distvae.modules.adapters.layers.conv_adapters import WanCausalConv3dAdapter
from distvae.modules.adapters.midblock_adapters import WanMidBlockAdapter
from distvae.modules.adapters.downsampling_adapters import (
    WanResidualDownBlockAdapter,
    WanResampleDownAdapter
)
from distvae.modules.adapters.resnet_adapters import WanResidualBlockAdapter
from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.utils import DistributedEnv

class WanEncoderAdapter(nn.Module):
    """
    Parallel adapter for Wan VAE encoder using distributed parallelism with overlap.

    This adapter parallelizes the encoder across multiple GPUs by:
    1. Patching conv layers with WanCausalConv3dAdapter
    2. Patching mid_block with WanMidBlockAdapter
    3. Patching down_blocks with appropriate adapters
    4. Using Patchify/DePatchify for distributed computation with overlap

    The encoder can have two different structures:
    - is_residual=True (Wan2.2): Uses WanResidualDownBlock
    - is_residual=False (Wan2.1): Uses individual WanResidualBlock and WanResample

    Args:
        encoder: The original Wan encoder to parallelize
        vae_group: ProcessGroup for distributed VAE computation
        conv_block_size: Block size for convolution adapters (default: 0)
        patch_dim: Dimension to patch along (-2 for H, -1 for W, -3 not supported)
        vae_config: VAE config for computing expected output dimensions
    """
    def __init__(
        self,
        encoder,
        vae_group: ProcessGroup = None,
        *,
        conv_block_size = 0,
        patch_dim: int = -2,
        vae_config = None,
    ):
        super().__init__()
        if patch_dim == -3:
            raise ValueError("WanEncoderAdapter does not support patch_dim F (-3); use H (-2) or W (-1).")

        DistributedEnv.initialize(vae_group)
        self.patch_dim = patch_dim
        DistributedEnv.set_patch_dim(patch_dim)
        self.encoder = encoder
        self.vae_config = vae_config

        # Patch the conv_in layer
        self.encoder.conv_in = WanCausalConv3dAdapter(
            encoder.conv_in, block_size=conv_block_size, patch_dim=patch_dim
        )

        # Patch the mid_block
        self.encoder.mid_block = WanMidBlockAdapter(
            encoder.mid_block, conv_block_size=conv_block_size, patch_dim=patch_dim
        )

        # Patch the down_blocks
        # Import all possible block types
        from diffusers.models.autoencoders.autoencoder_kl_wan import (
            WanResidualDownBlock,
            WanResidualBlock,
            WanResample,
            WanAttentionBlock,
        )

        down_blocks = []
        for i, down_block in enumerate(encoder.down_blocks):
            if isinstance(down_block, WanResidualDownBlock):
                # Wan2.2 style: wrapped in WanResidualDownBlock
                down_blocks.append(
                    WanResidualDownBlockAdapter(down_block, conv_block_size=conv_block_size, patch_dim=patch_dim)
                )
            elif isinstance(down_block, WanResidualBlock):
                # Wan2.1 style: individual residual block
                down_blocks.append(
                    WanResidualBlockAdapter(down_block, conv_block_size=conv_block_size, patch_dim=patch_dim)
                )
            elif isinstance(down_block, WanResample):
                # Wan2.1 style: individual downsample block
                down_blocks.append(
                    WanResampleDownAdapter(down_block, conv_block_size=conv_block_size, patch_dim=patch_dim)
                )
            elif isinstance(down_block, WanAttentionBlock):
                # Attention blocks are kept as-is (no spatial splitting needed)
                down_blocks.append(down_block)
            else:
                # Unknown block type - keep as-is and log warning
                import warnings
                warnings.warn(
                    f"Unsupported down_block type {type(down_block).__name__} at index {i} in encoder, "
                    f"keeping original. This may cause issues with parallel VAE."
                )
                down_blocks.append(down_block)

        self.encoder.down_blocks = nn.ModuleList(down_blocks)

        # Patch the conv_out layer
        self.encoder.conv_out = WanCausalConv3dAdapter(
            encoder.conv_out, block_size=conv_block_size, patch_dim=patch_dim
        )

        # Setup patchify/depatchify for overlap handling
        self.patchify = Patchify(patch_dim=patch_dim)
        self.depatchify = DePatchify(patch_dim=patch_dim)

    def _forward(
        self,
        sample: torch.FloatTensor,
        feat_cache: Optional[torch.FloatTensor] = None,
        feat_idx: Optional[int] = 0,
        patchify: bool = True,
    ):
        """Internal forward with optional patchify."""
        # Store original spatial dimensions
        original_shape = sample.shape

        if patchify:
            # Add padding to ensure dimensions are divisible by world_size * 2^downsample_count
            # This follows SGLang's split_for_parallel_encode approach
            group_world_size = DistributedEnv.get_group_world_size()

            # Count number of downsampling operations (stride=2 layers)
            # For Wan VAE, typically 3 downsample layers (8x spatial reduction)
            downsample_count = 3
            if self.vae_config is not None:
                vae_spatial_scale = getattr(self.vae_config, 'scaling_factor', 8)
                if hasattr(self.vae_config, 'vae_scale_factor_spatial'):
                    vae_spatial_scale = self.vae_config.vae_scale_factor_spatial
                # Infer downsample count from scale factor (2^downsample_count = scale)
                downsample_count = int(math.log2(vae_spatial_scale))

            # Calculate required padding to make the split dimension divisible
            factor = group_world_size * (2 ** downsample_count)

            # Pad along the configured spatial split dimension so patchify/chunking
            # preserves even patch sizes and stride alignment guarantees.
            patch_dim = self.patch_dim
            if patch_dim not in (-2, -1):
                raise ValueError(f"Unsupported patch_dim for spatial padding: {patch_dim}")

            orig_spatial = original_shape[patch_dim]
            pad_spatial = (factor - orig_spatial % factor) % factor

            # F.pad expects (W_left, W_right, H_left, H_right, F_left, F_right) for 5D
            if pad_spatial > 0:
                if patch_dim == -2:
                    sample = F.pad(sample, (0, 0, 0, pad_spatial, 0, 0), mode='constant', value=0)
                else:
                    sample = F.pad(sample, (0, pad_spatial, 0, 0, 0, 0), mode='constant', value=0)

            # Now patchify the padded tensor
            sample = self.patchify(sample)

        # Call encoder without return_dict (WanEncoder3d doesn't support it)
        sample = self.encoder(sample, feat_cache=feat_cache, feat_idx=feat_idx)

        sample = self.depatchify(sample)

        # Crop to expected dimensions to match what diffusers expects
        # This removes the padding we added above
        if patchify and self.vae_config is not None:
            # Calculate expected output dimensions based on VAE scaling
            vae_spatial_scale = getattr(self.vae_config, 'scaling_factor', 8)
            if hasattr(self.vae_config, 'vae_scale_factor_spatial'):
                vae_spatial_scale = self.vae_config.vae_scale_factor_spatial

            expected_h = original_shape[-2] // vae_spatial_scale
            expected_w = original_shape[-1] // vae_spatial_scale

            # Crop if dimensions don't match (due to parallel VAE padding)
            if sample.shape[-2] != expected_h or sample.shape[-1] != expected_w:
                sample = sample[..., :expected_h, :expected_w]

        return sample

    def forward(
        self,
        sample: torch.FloatTensor,
        feat_cache: Optional[torch.FloatTensor] = None,
        feat_idx: Optional[int] = 0,
        patchify: bool = True,
    ):
        """
        Forward pass through the encoder.

        Args:
            sample: Input tensor to encode
            feat_cache: Optional feature cache for temporal consistency
            feat_idx: Feature index for caching
            patchify: Whether to apply patchify/depatchify (default: True)

        Returns:
            Encoded latent tensor
        """
        return self._forward(sample, feat_cache, feat_idx, patchify)
