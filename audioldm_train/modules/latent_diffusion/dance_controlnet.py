from typing import Any
import torch.nn as nn
from audioldm_train.utilities.model_util import instantiate_from_config


class DanceControlNetWrapper(nn.Module):
    def __init__(self, feature_encoder_config, controlnet_model_config):
        super().__init__()
        self.feature_encoder: Any = instantiate_from_config(feature_encoder_config)
        self.controlnet: Any = instantiate_from_config(controlnet_model_config)

    def forward(
        self,
        x,
        hint,
        timesteps=None,
        y=None,
        context_list=None,
        context_attn_mask_list=None,
        **kwargs,
    ):
        """
        x: latent (N, C, T, F)
        hint: keypoints (N, T, J*F) or (N, J*F, T)
        return: (N, C, T, F)
        """
        hint_features = self.feature_encoder(hint)  # (N, C_model, T, F)
        out = self.controlnet(
            x,
            hint_features,
            timesteps=timesteps,
            y=y,
            context_list=context_list,
            context_attn_mask_list=context_attn_mask_list,
        )  # (N, C, T, F)
        return out