import torch
from diffusers.models import UNet2DModel

class UNet2DSimpleConditionModel(UNet2DModel):
    def __init__(
        self,
        sample_channels,
        condition_channels,
        sample_size=None,
        in_channels=None,
        out_channels=None,
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
        block_out_channels=(64,),
        layers_per_block=1,
        attention_head_dim=8,
        norm_num_groups=32,
        norm_eps=1e-5,
        **kwargs
    ):
        # Set in_channels to sample_channels + condition_channels
        in_channels = sample_channels + condition_channels
        super().__init__(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            **kwargs
        )

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        if encoder_hidden_states is not None:
            assert sample.shape[2:] == encoder_hidden_states.shape[2:], "Spatial dimensions must match"
            sample = torch.cat([sample, encoder_hidden_states], dim=1)
        return super().forward(sample, timestep, **kwargs)