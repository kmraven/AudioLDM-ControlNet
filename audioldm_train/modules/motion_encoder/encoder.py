from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from audioldm_train.utilities.model_util import instantiate_from_config

class ZeroConv1dProject(nn.Module):
    def __init__(self, d_model: int, C: int, Freq: int, kernel_size: int = 1, padding: int = 0):
        super().__init__()
        out_channels = C * Freq
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.C = C
        self.Freq = Freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D_model]
        return: [B, C, T, Freq]
        """
        B, T, D = x.shape
        assert D == self.conv.in_channels

        x = x.transpose(1, 2)             # [B, D_model, T]
        y = self.conv(x)                  # [B, C*Freq, T]

        B, CF, T_out = y.shape
        assert CF == self.C * self.Freq
        y = y.view(B, self.C, self.Freq, T_out)  # [B, C, Freq, T]
        y = y.permute(0, 1, 3, 2)         # [B, C, T, Freq]

        return y


class MotionBERT2AudioLDMEncoder(nn.Module):
    def __init__(
        self,
        kpt_num_joints: int,
        kpt_feat_dim: int,
        audio_channels: int,
        audio_freq: int,
        d_model: int = 512,
    ):
        super().__init__()
        self.kpt_num_joints = kpt_num_joints
        self.kpt_feat_dim = kpt_feat_dim

        d_in = kpt_num_joints * kpt_feat_dim
        self.input_proj = nn.Linear(d_in, d_model)
        self.output_proj = ZeroConv1dProject(d_model, audio_channels, audio_freq)

    def forward(self, x):
        """
        x: [B, T, J, F]  (MotionBERT get_representation() output)
        return: [B, C, T, Freq]  (AudioLDM/ControlNet input)
        """
        B, T, J, F = x.shape
        assert J == self.kpt_num_joints and F == self.kpt_feat_dim
        x = x.view(B, T, J * F)  # [B, T, D_in]
        h = self.input_proj(x)          # [B, T, D_model]
        h = self.output_proj(h)         # [B, C, T, Freq]
        return h


class MotionBERT2BeatDance2AudioLDMEncoder(nn.Module):
    def __init__(
        self,
        kpt_num_joints: int,
        kpt_feat_dim: int,
        audio_channels: int,
        audio_freq: int,
        beatdance_output_dim: int,
        beatdance_config: Any,
    ):
        super().__init__()
        self.kpt_num_joints = kpt_num_joints
        self.kpt_feat_dim = kpt_feat_dim

        self.beatdance: Any = instantiate_from_config(beatdance_config)
        self.output_proj = ZeroConv1dProject(beatdance_output_dim, audio_channels, audio_freq)

    def forward(self, x):
        """
        motion_feature: [B, T, J, F]  (MotionBERT get_representation() output)
        motion_beat_feature: [B, T, beat_dim]  (extracted motion beat feature)
        return: [B, C, T, Freq]  (AudioLDM/ControlNet input)
        """
        motion_feature = x['motion_feature']
        motion_beat_feature = x['motion_beat_feature']
        B, T, J, F = motion_feature.shape
        assert J == self.kpt_num_joints and F == self.kpt_feat_dim
        h = self.beatdance(motion_feature, motion_beat_feature)  # [B, T, beatdance_output_dim]
        h = self.output_proj(h)         # [B, C, T, Freq]
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
