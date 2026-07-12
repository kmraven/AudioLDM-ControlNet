import json
from types import SimpleNamespace

import torch
import torch.nn as nn

try:
    from BeatDance.config.base_config import Config
    from BeatDance.modules.transformer import PositionalEncoding
except ModuleNotFoundError:  # Support running scripts from inside BeatDance/.
    from config.base_config import Config
    from modules.transformer import PositionalEncoding


class CLIPTransformer(nn.Module):
    """BeatDance music-motion encoder used during contrastive pretraining."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        config.pooling_type = "avg"

        video_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_mha_heads,
            dropout=config.dropout1,
        )
        music_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_mha_heads,
            dropout=config.dropout1,
        )
        self.music_transformer = nn.TransformerEncoder(
            video_encoder_layer,
            num_layers=config.num_layers,
        )
        self.video_transformer = nn.TransformerEncoder(
            music_encoder_layer,
            num_layers=config.num_layers,
        )

        self.music_linear = nn.Linear(768, config.embed_dim)
        self.video_linear = nn.Linear(17 * 512, config.embed_dim)
        self.clip_logit_scale = torch.tensor([4.6052], device="cuda")

        video_beat_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_mha_heads,
            dropout=config.dropout1,
        )
        music_beat_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_mha_heads,
            dropout=config.dropout1,
        )
        self.music_beat_transformer = nn.TransformerEncoder(
            video_beat_encoder_layer,
            num_layers=config.num_layers,
        )
        self.video_beat_transformer = nn.TransformerEncoder(
            music_beat_encoder_layer,
            num_layers=config.num_layers,
        )
        self.music_beat_linear = nn.Sequential(
            nn.Linear(3, 64),
            nn.Linear(64, config.embed_dim),
        )
        self.video_beat_linear = nn.Sequential(
            nn.Linear(2, 64),
            nn.Linear(64, config.embed_dim),
        )

        self.l1 = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.l2 = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.music_transformer_fuse = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_mha_heads,
            batch_first=True,
            dropout=config.dropout2,
        )
        self.video_transformer_fuse = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_mha_heads,
            batch_first=True,
            dropout=config.dropout2,
        )

        self.music_pos_encoding = PositionalEncoding(
            config.embed_dim,
            max_len=128,
            dropout=0.1,
        )
        self.music_beat_pos_encoding = PositionalEncoding(
            config.embed_dim,
            max_len=128,
            dropout=0.1,
        )
        self.video_pos_encoding = PositionalEncoding(
            config.embed_dim,
            max_len=128,
            dropout=0.1,
        )
        self.video_beat_pos_encoding = PositionalEncoding(
            config.embed_dim,
            max_len=128,
            dropout=0.1,
        )

    def forward(self, data, phase="test"):
        del phase
        music_data = self.music_linear(data["music"])
        music_beat = data["music_beat"].float().permute(0, 2, 1)
        music_beat = self.music_pos_encoding(self.music_beat_linear(music_beat))
        music_features = self.music_multimodel_fuse(music_data, music_beat)

        video_data = self.video_pos_encoding(self.video_linear(data["video"]))
        video_beat = self.video_beat_pos_encoding(
            self.video_beat_linear(data["video_beat"].float())
        )
        video_features = self.video_multimodel_fuse(video_data, video_beat)
        return music_features, video_features

    def music_multimodel_fuse(self, music_data, music_beat):
        music_data = self.music_transformer(music_data)
        music_beat = self.music_beat_transformer(music_beat)
        fused = self.l1(
            torch.cat([music_data + music_beat, music_data * music_beat], dim=-1)
        )
        attended_beat, _ = self.music_transformer_fuse(
            music_beat,
            music_data,
            music_data,
        )
        return {
            "music_data": music_data,
            "music_beat": attended_beat,
            "music_fuse": fused,
        }

    def video_multimodel_fuse(self, video_data, video_beat):
        video_data = self.video_transformer(video_data)
        video_beat = self.video_beat_transformer(video_beat)
        fused = self.l2(
            torch.cat([video_data + video_beat, video_data * video_beat], dim=-1)
        )
        attended_beat, _ = self.video_transformer_fuse(
            video_beat,
            video_data,
            video_data,
        )
        return {
            "video_data": video_data,
            "video_beat": attended_beat,
            "video_fuse": fused,
        }


class DanceOnlyBeatDanceEncoder(nn.Module):
    """Dance branch of BeatDance used to condition AudioLDM ControlNet."""

    def __init__(
        self,
        beatdance_config_path,
        kpt_num_joints: int,
        kpt_feat_dim: int,
        beat_dim: int,
        beatdance_output_dim: int,
        num_frames: int,
    ):
        super().__init__()
        with open(beatdance_config_path) as config_file:
            config = SimpleNamespace(**json.load(config_file))

        self.config = config
        self.config.embed_dim = beatdance_output_dim
        self.config.num_frames = num_frames
        self.kpt_num_joints = kpt_num_joints
        self.kpt_feat_dim = kpt_feat_dim

        motion_layer = nn.TransformerEncoderLayer(
            d_model=beatdance_output_dim,
            nhead=config.num_mha_heads,
            dropout=config.dropout1,
        )
        beat_layer = nn.TransformerEncoderLayer(
            d_model=beatdance_output_dim,
            nhead=config.num_mha_heads,
            dropout=config.dropout1,
        )
        self.video_transformer = nn.TransformerEncoder(
            motion_layer,
            num_layers=config.num_layers,
        )
        self.video_beat_transformer = nn.TransformerEncoder(
            beat_layer,
            num_layers=config.num_layers,
        )
        self.video_linear = nn.Linear(
            kpt_num_joints * kpt_feat_dim,
            beatdance_output_dim,
        )
        self.video_beat_linear = nn.Linear(beat_dim, beatdance_output_dim)
        self.l2 = nn.Linear(beatdance_output_dim * 2, beatdance_output_dim)

    def forward(self, motion_feature, motion_beat_feature):
        batch_size, num_frames, num_joints, feature_dim = motion_feature.shape
        if num_joints != self.kpt_num_joints or feature_dim != self.kpt_feat_dim:
            raise ValueError(
                "motion_feature shape does not match the configured joint and "
                "feature dimensions"
            )

        motion_feature = motion_feature.reshape(
            batch_size,
            num_frames,
            num_joints * feature_dim,
        )
        motion_feature = self.video_transformer(self.video_linear(motion_feature))
        motion_beat_feature = self.video_beat_transformer(
            self.video_beat_linear(motion_beat_feature)
        )
        return self.l2(
            torch.cat(
                [
                    motion_feature + motion_beat_feature,
                    motion_feature * motion_beat_feature,
                ],
                dim=-1,
            )
        )


# Backward-compatible name used by existing configs and checkpoints.
DanceOnlyBeatDanceWrapper = DanceOnlyBeatDanceEncoder
