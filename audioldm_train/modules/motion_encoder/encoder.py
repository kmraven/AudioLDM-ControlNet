from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from audioldm_train.utilities.model_util import instantiate_from_config

class TemporalConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # x: [B, T, D]
        residual = x
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)  # [B, T, D]
        x = self.norm(x + residual)
        return x


class MotionBERT2AudioLDMEncoder(nn.Module):
    def __init__(
        self,
        kpt_num_joints: int,
        kpt_feat_dim: int,
        audio_channels: int,
        audio_freq: int,
        d_model: int = 512,
        num_temporal_blocks: int = 3,
    ):
        super().__init__()
        self.kpt_num_joints = kpt_num_joints
        self.kpt_feat_dim = kpt_feat_dim
        self.audio_channels = audio_channels
        self.audio_freq = audio_freq

        d_in = kpt_num_joints * kpt_feat_dim
        self.input_proj = nn.Linear(d_in, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.temporal_blocks = nn.ModuleList(
            [TemporalConvBlock(d_model) for _ in range(num_temporal_blocks)]
        )

        self.output_proj = nn.Linear(d_model, audio_channels * audio_freq)

    def forward(self, x):
        """
        x: [B, T, J, F]  (MotionBERT get_representation() output)
        return: [B, C, T, Freq]  (AudioLDM/ControlNet input)
        """
        B, T, J, F = x.shape
        assert J == self.kpt_num_joints and F == self.kpt_feat_dim

        x = x.view(B, T, J * F)  # [B, T, D_in]

        h = self.input_proj(x)          # [B, T, D_model]
        h = self.input_norm(h)

        for block in self.temporal_blocks:
            h = block(h)                # [B, T, D_model]

        h = self.output_proj(h)         # [B, T, C*Freq]
        h = h.view(B, T, self.audio_channels, self.audio_freq)
        h = h.permute(0, 2, 1, 3)       # [B, C, T, Freq]

        return h


class MotionEncoderWrapper(nn.Module):
    def __init__(self, motion_bert_config, motion_bert_pretrained_weights_path, feature_mapper_config):
        super().__init__()
        self.motion_bert: Any = instantiate_from_config(motion_bert_config)
        if not torch.cuda.is_available():
            raise RuntimeError("MotionBERT (DSTformer) requires GPU.")
        self.motion_bert = nn.DataParallel(self.motion_bert)
        self.motion_bert = self.motion_bert.cuda()
        if motion_bert_pretrained_weights_path is not None:
            print("Reload MotionBERT (DSTformer) from %s" % motion_bert_pretrained_weights_path)
            checkpoint = torch.load(motion_bert_pretrained_weights_path, map_location=lambda storage, loc: storage)
            self.motion_bert.load_state_dict(checkpoint['model_pos'], strict=True)
        self.motion_bert.eval()
        for param in self.motion_bert.parameters():
            param.requires_grad = False
        self.motion_encoder: Any = instantiate_from_config(feature_mapper_config)

    def forward(self, keypoints):
        """
        keypoints: [B, T, J, F]
        return: [B, C, T, Freq]
        """
        with torch.no_grad():
            motion_features = self.motion_bert.module.get_representation(keypoints)
        encoded_motion = self.motion_encoder(motion_features)
        return encoded_motion
