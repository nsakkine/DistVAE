import time
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.profiler import profile, ProfilerActivity

from distvae.models.vae import PatchDecoder
from distvae.modules.adapters.layers.attn_adapters import WanAttentionBlockAdapter
from distvae.modules.adapters.layers.conv_adapters import Conv2dAdapter, WanCausalConv3dAdapter
from distvae.modules.adapters.layers.norm_adapters import GroupNormAdapter
from distvae.modules.adapters.resnet_adapters import WanResidualBlockAdapter
from distvae.modules.adapters.unets.unet_2d_blocks_adapters import UpDecoderBlock2DAdapter
from distvae.modules.adapters.upsampling_adapters import WanUpBlockAdapter
from distvae.modules.patch_utils import Patchify, DePatchify
from distvae.utils import DistributedEnv
from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.unets.unet_2d_blocks import UpDecoderBlock2D
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanMidBlock,
)

try:
    import torch_musa
except ModuleNotFoundError:
    pass

class DecoderAdapter(nn.Module):
    def __init__(
        self, 
        decoder: Decoder, 
        vae_group: ProcessGroup = None,
        *,
        use_profiler: bool = False,
        conv_block_size = 0,
    ):
        super().__init__()
        assert isinstance(decoder.conv_norm_out, nn.GroupNorm), "DecoderAdapter does not support normalization method except GroupNorm"
        for up_block in decoder.up_blocks:
            assert isinstance(up_block, UpDecoderBlock2D), "DecoderAdapter does not support up block except UpDecoderBlock2D"
        DistributedEnv.initialize(vae_group)
        self.decoder = PatchDecoder()
        self.decoder.layers_per_block = decoder.layers_per_block
        self.decoder.conv_in = decoder.conv_in
        self.decoder.mid_block = decoder.mid_block
        self.decoder.up_blocks = nn.ModuleList([
            UpDecoderBlock2DAdapter(up_block, conv_block_size=conv_block_size) for up_block in decoder.up_blocks
        ])
        self.decoder.conv_norm_out = GroupNormAdapter(decoder.conv_norm_out)
        self.decoder.conv_act = decoder.conv_act
        self.decoder.conv_out = Conv2dAdapter(decoder.conv_out, block_size=conv_block_size)
        self.use_profiler = use_profiler
        self.vae_group = vae_group

    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ):
        rank = DistributedEnv.get_global_rank()
        device_type = DistributedEnv.get_device_type()
        start_time = time.time()
        elapsed_time = 0
        if self.use_profiler:
            if device_type == "musa":
                torch.musa.memory._record_memory_history(enabled=None)
                activities=[ProfilerActivity.CPU,ProfilerActivity.MUSA]
            else:
                torch.cuda.memory._record_memory_history(enabled=None)
                activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]

            with profile(
                activities=activities,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"./profile/patch_vae_{rank}"
                ),
                profile_memory=True,
                with_stack=True,
                record_shapes=True,
            ) as prof:
                output = self.decoder(sample, latent_embeds)
            prof.export_memory_timeline(f"patch_vae_profiler_mem_{rank}.html")
        else:
            output =  self.decoder(sample, latent_embeds)

        end_time = time.time()
        elapsed_time = end_time - start_time
        peak_memory = DistributedEnv.get_peak_memory(device_type)

        if rank == 0:
            print(f"Patch vae: [elapsed_time: {elapsed_time:.2f} sec, peak_memory: {peak_memory/1e9} GB]")
        return output


class WanMidBlockAdapter(nn.Module):
    def __init__(self, wan_mid_block: WanMidBlock, conv_block_size = 0, patch_dim: int = -2):
        super().__init__()

        assert isinstance(wan_mid_block, WanMidBlock), "WanMidBlockAdapter does not support mid block except WanMidBlock"
        self.mid_block = wan_mid_block
        self.mid_block.resnets = nn.ModuleList([
            WanResidualBlockAdapter(
                resnet, conv_block_size=conv_block_size, patch_dim=patch_dim
            ) for resnet in wan_mid_block.resnets
        ])
        self.mid_block.attentions = nn.ModuleList([
            WanAttentionBlockAdapter(attn, patch_dim=patch_dim)
            for attn in wan_mid_block.attentions
        ])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        return self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)  


class WanDecoderAdapter(nn.Module):
    def __init__(
        self, 
        decoder: Decoder, 
        vae_group: ProcessGroup = None,
        *,
        use_profiler: bool = False,
        conv_block_size = 0,
        patch_dim: int = -2,
    ):
        super().__init__()
        if patch_dim == -3:
            raise ValueError("WanDecoderAdapter does not support patch_dim F (-3); use H (-2) or W (-1).")
        DistributedEnv.initialize(vae_group)
        self.patch_dim = patch_dim
        DistributedEnv.set_patch_dim(patch_dim)
        self.decoder = decoder
        self.decoder.conv_in = WanCausalConv3dAdapter(
            decoder.conv_in, block_size=conv_block_size, patch_dim=patch_dim
        )
        self.decoder.mid_block = WanMidBlockAdapter(
            decoder.mid_block, conv_block_size=conv_block_size, patch_dim=patch_dim
        )
        self.decoder.up_blocks = nn.ModuleList([
            WanUpBlockAdapter(
                up_block,
                conv_block_size=conv_block_size,
                patch_dim=patch_dim
            ) for up_block in decoder.up_blocks
        ])
        self.decoder.conv_out = WanCausalConv3dAdapter(
            decoder.conv_out, block_size=conv_block_size, patch_dim=patch_dim
        )
        self.patchify = Patchify()
        self.depatchify = DePatchify()
        self.use_profiler = use_profiler
        self.vae_group = vae_group

    def _forward(
        self,
        sample: torch.FloatTensor,
        feat_cache: Optional[torch.FloatTensor] = None,
        feat_idx: Optional[int] = 0,
        first_chunk: bool = False,
        patchify: bool = True
    ):
        if patchify:
            sample = self.patchify(sample)
        sample = self.decoder(sample, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)
        sample = self.depatchify(sample)
        return sample

    def forward(
        self,
        sample: torch.FloatTensor,
        feat_cache: Optional[torch.FloatTensor] = None,
        feat_idx: Optional[int] = 0,
        first_chunk: bool = False,
        patchify: bool = True,
    ):
        rank = DistributedEnv.get_global_rank()
        device_type = DistributedEnv.get_device_type()
        start_time = time.time()
        elapsed_time = 0
        if self.use_profiler:
            if device_type == "musa":
                torch.musa.memory._record_memory_history(enabled=None)
                activities=[ProfilerActivity.CPU,ProfilerActivity.MUSA]
            else:
                torch.cuda.memory._record_memory_history(enabled=None)
                activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]

            with profile(
                activities=activities,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    f"./profile/patch_vae_{rank}"
                ),
                profile_memory=True,
                with_stack=True,
                record_shapes=True,
            ) as prof:
                output = self._forward(sample, feat_cache, feat_idx, first_chunk, patchify)
            prof.export_memory_timeline(f"patch_vae_profiler_mem_{rank}.html")
        else:
            output =  self._forward(sample, feat_cache, feat_idx, first_chunk, patchify)

        end_time = time.time()
        elapsed_time = end_time - start_time
        peak_memory = DistributedEnv.get_peak_memory(device_type)

        if rank == 0:
            print(f"Patch vae: [elapsed_time: {elapsed_time:.2f} sec, peak_memory: {peak_memory/1e9} GB]")
        return output