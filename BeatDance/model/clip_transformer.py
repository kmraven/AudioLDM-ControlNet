import torch
import torch.nn as nn
from config.base_config import Config
from config.all_config import CusConfig
from modules.transformer import PoseTransformer, PositionalEncoding
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id, beat_similarity, qb_norm
from torch import nn
import torch
from einops import rearrange
import typing as tp

class CLIPTransformer(nn.Module):
    def __init__(self, config: Config):
        super(CLIPTransformer, self).__init__()
        self.config = config

        config.pooling_type = 'avg'
        self.dropout1 = config.dropout1
        self.dropout2 = config.dropout2

        video_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads, dropout=self.dropout1) # self-atten + feedforward
        music_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads, dropout=self.dropout1)
        self.music_transformer = nn.TransformerEncoder(video_encoder_layer, num_layers=config.num_layers) # stack of transformerencoder layers
        self.video_transformer = nn.TransformerEncoder(music_encoder_layer, num_layers=config.num_layers)

        self.music_linear = nn.Linear(768, config.embed_dim)
        self.video_linear = nn.Linear(17* 512, config.embed_dim)

        self.clip_logit_scale = torch.FloatTensor([4.6052]).cuda()

        video_beat_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads, dropout=self.dropout1)
        music_beat_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads, dropout=self.dropout1)
        self.music_beat_transformer = nn.TransformerEncoder(video_beat_encoder_layer, num_layers=config.num_layers)
        self.video_beat_transformer = nn.TransformerEncoder(music_beat_encoder_layer, num_layers=config.num_layers)

        self.music_beat_linear = nn.Sequential(
            nn.Linear(3, 64), # this part was supposed to be num_frames instead of 4
            nn.Linear(64, config.embed_dim),
        )

        self.video_beat_linear = nn.Sequential(
            nn.Linear(2, 64), # this part was supposed to be num_frames instead of 3
            nn.Linear(64, config.embed_dim),
        )

        self.l1, self.l2= nn.Linear(512, 256), nn.Linear(512, 256)

        self.music_transformer_fuse = nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=config.num_mha_heads, batch_first=True, dropout=self.dropout2)
        self.video_transformer_fuse = nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=config.num_mha_heads, batch_first=True, dropout=self.dropout2)

        self.music_pos_encoding = PositionalEncoding(config.embed_dim, max_len=128, dropout=0.1)
        self.music_beat_pos_encoding = PositionalEncoding(config.embed_dim, max_len=128, dropout=0.1)
        self.video_pos_encoding = PositionalEncoding(config.embed_dim, max_len=128, dropout=0.1)
        self.video_beat_pos_encoding = PositionalEncoding(config.embed_dim, max_len=128, dropout=0.1)

    def forward(self, data, phase='test'):
        music_data = data['music'] # ([B, L, 768])
        music_data = (self.music_linear(music_data)) # MLP (B,L,768) -> (B,L,256)

        music_beat = data['music_beat'].to(torch.float) # ([B, 4, L])
        music_beat = music_beat.permute(0,2,1) #(B,L,4)

        music_beat = self.music_beat_linear(music_beat) # (B,L,4) -> (B,L,256)
        music_beat = self.music_pos_encoding(music_beat)

        music_features_trans = self.music_multimodel_fuse(music_data, music_beat) # cross attn on attended music feats by attended mbfeats

        video_data = data['video']
        video_data = (self.video_linear(video_data))
        video_data = self.video_pos_encoding(video_data)

        video_beat = data['video_beat'].to(torch.float) # (B, L, 2)
        # video_beat = video_beat.permute(0,2,1)
        video_beat = self.video_beat_linear(video_beat) #(B,L,3)->(B,L,256)
        video_beat = self.video_beat_pos_encoding(video_beat)

        video_features_trans = self.video_multimodel_fuse(video_data, video_beat) # cross attn on attended dance feats by attended dbfeats

        # framewise approach do not aggregate the embedding and frame dimensions
        #music_features_trans['music_fuse'] = music_features_trans['music_fuse'].reshape(-1, int(self.config.embed_dim) * (self.config.num_frames))
        #music_features_trans['music_beat'] = music_features_trans['music_beat'].reshape(-1, int(self.config.embed_dim) * (self.config.num_frames))
        #video_features_trans['video_fuse'] = video_features_trans['video_fuse'].reshape(-1, int(self.config.embed_dim) * (self.config.num_frames))
        #video_features_trans['video_beat'] = video_features_trans['video_beat'].reshape(-1, int(self.config.embed_dim) * (self.config.num_frames))

        return music_features_trans, video_features_trans

    def music_multimodel_fuse(self, music_data, music_beat):
        music_data = self.music_transformer(music_data) # attention along time?
        music_beat = self.music_beat_transformer(music_beat)
        add_data = music_data + music_beat
        mul_data = music_data * music_beat
        fuse_data = self.l1(torch.cat([add_data, mul_data], dim=-1)) # (B, L, 2D)-> (B,L,D)
        music_beat_attended, _ = self.music_transformer_fuse(music_beat, music_data, music_data)
        return {'music_data': music_data, 'music_beat': music_beat_attended, 'music_fuse': fuse_data}


    def video_multimodel_fuse(self, video_data, video_beat):
        video_data = self.video_transformer(video_data) # transformer along time? (B, L, D)
        video_beat = self.video_beat_transformer(video_beat)
        add_data = video_data + video_beat
        mul_data = video_data * video_beat
        fuse_data = self.l2(torch.cat([add_data, mul_data], dim=-1)) #(B, L, 2D)-> (B,L,D)

        # update video beat feature with multihead attention (B,L,D)
        video_beat_attended, _ = self.video_transformer_fuse(video_beat, video_data, video_data)
        return {'video_data': video_data, 'video_beat': video_beat_attended, 'video_fuse': fuse_data}

class DanceOnlyBeatDanceWrapper(CLIPTransformer):
    def __init__(
            self,
            beatdance_config_path,
            kpt_num_joints: int,
            kpt_feat_dim: int,
            beat_dim: int,
            beatdance_output_dim: int,
            num_frames: int,
        ):
        config = CusConfig(beatdance_config_path, exp_name="")
        config.embed_dim = beatdance_output_dim
        config.num_frames = num_frames
        super(DanceOnlyBeatDanceWrapper, self).__init__(config)
        del self.dropout2
        del self.music_transformer
        del self.music_linear
        del self.clip_logit_scale
        del self.music_beat_transformer
        del self.music_beat_linear
        del self.l1
        del self.music_transformer_fuse
        self.video_linear = nn.Linear(kpt_feat_dim * kpt_num_joints, config.embed_dim)
        self.video_beat_linear = nn.Linear(beat_dim, config.embed_dim)

    def forward(self, motion_feature, motion_beat_feature):
        """
        motion_feature: [B, T, J, F]  (MotionBERT get_representation() output)
        motion_beat_feature: [B, T, beat_dim]  (extracted motion beat feature)
        return: [B, C, T, Freq]  (AudioLDM/ControlNet input)
        """
        B, T, J, F = motion_feature.shape
        motion_feature = motion_feature.reshape(B, T, J * F)
        motion_feature = self.video_linear(motion_feature)
        motion_beat_feature = self.video_beat_linear(motion_beat_feature)
        video_features_trans = self.video_multimodel_fuse(motion_feature, motion_beat_feature)
        return video_features_trans['video_fuse']