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

class MotionEncoder(nn.Module): #todo: replace with my own encoder
    """
    A conditioner that takes motion 2D keypoints as input and outputs a embedding. temporarily adopted Yang's code

    Args:
        seq_length: 2D keypoints time steps
        output_length: feature time steps, should be the same as seq_length?
        input_dim: 2 for 2D keypoints
        output_dim: the dimension of the output embeddings
    """
    def __init__(
            self,
            seq_length: int, # input sequence length
            output_length: int, # output sequence length to match SAO
            input_dim: int, # channels corresponding to keypoints
            output_dim: int, # feature dimension 64
    ):
        super().__init__()
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.conv1 = nn.Conv1d(in_channels=34, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=self.output_dim, kernel_size=3, stride=1, padding=1)

        self.self_attn1 = nn.MultiheadAttention(self.output_dim, num_heads=4, batch_first=True)
        self.self_attn2 = nn.MultiheadAttention(self.output_dim, num_heads=4, batch_first=True)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=output_length)
        self.output_length = output_length


    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (b, 50, 17, 2) for 2 seconds @ fps 25

        if isinstance(x, list):
            x = torch.stack(x)
        b, seq_len, num_joints, input_dim = x.shape
        # reshape tensor, combine joint and coordinate dimension
        x = rearrange(x, 'b seq_len num_joints input_dim -> b (num_joints input_dim) seq_len')
        x = self.conv1(x)  # (b, 256, 50)
        x = nn.ReLU()(x)
        x = self.conv2(x)  # (b, 128, 50)
        x = nn.ReLU()(x)
        x = self.conv3(x)  # (b, 64, 50) (b, feat_dim, seq_len)
        x = nn.ReLU()(x)

        x = x.permute(0, 2, 1) # (b, T=50, d=64)
        # self attention along time with embed_dim=64
        x = self.self_attn1(x)  # (b, 50, 64)
        x = self.self_attn2(x)  # (b, 50, 64)
        # resample temporal dimension to L
        x = x.permute(0, 1, 2) #(b, 64, 50)
        x = self.adaptive_pool(x) #(b, 64, 43)
        # make sure the output dimension same as latent space dimension
        return x, torch.ones(x.shape[0], x.shape[2])


class PoseEncoder(nn.Module): #todo: replace with my own encoder
    """
    A encoder that takes precomputed pose features (T, 138) as input and outputs a embedding.

    conv1d along feature dimension to reshape to 256
    adaptive average pooling along time dimension to reshape to L

    Args:
        input_length: input sequence length T
        output_length: output sequence length L
        output_dim: the dimension of the output embeddings
    """
    def __init__(
            self,
            input_dim: int, # input feature dimension, should be 138
            output_length: int, # output sequence length to match SAO
            output_dim: int, # feature dimension 256, maybe it doesn't matter
    ):

        super().__init__()
        self.output_dim = output_dim
        output_length = int(output_length)
        self.batchnorm1d = nn.BatchNorm1d(num_features=input_dim) # apply batchnorm per feature dimension
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=self.output_dim, kernel_size=4, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=output_length)


    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        # pose shape: (b, T=50, 138) for 2 seconds @ fps 25
        x = x.permute(0, 2, 1) #(B, input_dim, T)
        # apply batch normalization along each feature channel, input should be (B,C,T)
        x = self.batchnorm1d(x)
        # apply three blocks of (conv1d, relu, maxpool)
        x = self.conv1(x)  # (b, 512, 50)
        x = nn.ReLU()(x)
        x = self.conv2(x)  # (b, 256, 50)
        x = nn.ReLU()(x)
        x = self.conv3(x)  # (b, 256, 50)
        x = nn.ReLU()(x)
        x = self.adaptive_pool(x) #(b, 256, L=43)
        return x.permute(0, 2, 1) #(b, L, 256)


class CLIPTransformer(nn.Module): # the model being adopted
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
        self.video_linear = nn.Linear(512, config.embed_dim)
        self.pose_encoder = PoseEncoder(input_dim=138, output_length=110, output_dim=256) # L = 110

        self.clip_logit_scale = torch.FloatTensor([4.6052]).cuda()

        video_beat_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads, dropout=self.dropout1)
        music_beat_encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_mha_heads, dropout=self.dropout1)
        self.music_beat_transformer = nn.TransformerEncoder(video_beat_encoder_layer, num_layers=config.num_layers)
        self.video_beat_transformer = nn.TransformerEncoder(music_beat_encoder_layer, num_layers=config.num_layers)

        self.music_beat_linear = nn.Sequential(
            nn.Linear(4, 64), # this part was supposed to be num_frames instead of 4
            nn.Linear(64, config.embed_dim),
        )

        self.video_beat_linear = nn.Sequential(
            nn.Linear(3, 64), # this part was supposed to be num_frames instead of 3
            nn.Linear(64, config.embed_dim),
        )

        self.l1, self.l2= nn.Linear(512, 256), nn.Linear(512, 256)

        self.music_transformer_fuse = nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=config.num_mha_heads, batch_first=True, dropout=self.dropout2)
        self.video_transformer_fuse = nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=config.num_mha_heads, batch_first=True, dropout=self.dropout2)

        self.music_pos_encoding = PositionalEncoding(config.embed_dim, max_len=110, dropout=0.1)
        self.music_beat_pos_encoding = PositionalEncoding(config.embed_dim, max_len=110, dropout=0.1)
        self.video_pos_encoding = PositionalEncoding(config.embed_dim, max_len=110, dropout=0.1)
        self.video_beat_pos_encoding = PositionalEncoding(config.embed_dim, max_len=110, dropout=0.1)

    def forward(self, data, phase='test'):
        music_data = data['music'] # ([B, L, 768])
        music_data = (self.music_linear(music_data)) # MLP (B,L,768) -> (B,L,256)

        music_beat = data['music_beat'].to(torch.float) # ([B, 4, L])
        music_beat = music_beat.permute(0,2,1) #(B,L,4)
        music_beat = self.music_beat_linear(music_beat) # (B,L,4) -> (B,L,256)
        music_beat = self.music_pos_encoding(music_beat)

        music_features_trans = self.music_multimodel_fuse(music_data, music_beat) # cross attn on attended music feats by attended mbfeats

        video_beat = data['video_beat'].to(torch.float) # (B, 3, L)
        video_beat = video_beat.permute(0,2,1)
        video_beat = self.video_beat_linear(video_beat) #(B,L,3)->(B,L,256)
        video_beat = self.video_beat_pos_encoding(video_beat)

        pose_data = data['pose'] # pose feature (B,T,138)
        pose_data = (self.pose_encoder(pose_data))  #(B,T,138 )-> (B,L,256)
        pose_data = self.video_pos_encoding(pose_data)

        video_features_trans = self.video_multimodel_fuse(pose_data, video_beat) # cross attn on attended dance feats by attended dbfeats

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