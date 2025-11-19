import torch
import torch.nn as nn
import torch.nn.functional as F
from audioldm_train.utilities.diffusion_util import zero_module

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
        num_joints: int,
        feat_dim: int,
        audio_channels: int,   # e.g. 4
        audio_freq: int,       # e.g. 64 (AudioLDM latent Freq size)
        d_model: int = 512,
        num_temporal_blocks: int = 3,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        self.audio_channels = audio_channels
        self.audio_freq = audio_freq

        d_in = num_joints * feat_dim
        self.input_proj = nn.Linear(d_in, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.temporal_blocks = nn.ModuleList(
            [TemporalConvBlock(d_model) for _ in range(num_temporal_blocks)]
        )

        self.output_proj = nn.Linear(d_model, audio_channels * audio_freq)

    def forward(self, x):
        """
        x: [B, T, J, F]  (MotionBERT get_representation() 出力)
        return: [B, C, T, Freq]  (AudioLDM/ControlNet 用)
        """
        B, T, J, F = x.shape
        assert J == self.num_joints and F == self.feat_dim

        # (1) flatten joints & features
        x = x.view(B, T, J * F)  # [B, T, D_in]

        # (2) project to model dim
        h = self.input_proj(x)          # [B, T, D_model]
        h = self.input_norm(h)

        # (3) temporal modeling
        for block in self.temporal_blocks:
            h = block(h)                # [B, T, D_model]

        # (4) project to AudioLDM latent shape
        h = self.output_proj(h)         # [B, T, C*Freq]
        h = h.view(B, T, self.audio_channels, self.audio_freq)
        h = h.permute(0, 2, 1, 3)       # [B, C, T, Freq]

        return h


class MotionEncoderWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_bert = lambda x: x  # Placeholder for MotionBERT model
        self.motion_encoder = MotionBERT2AudioLDMEncoder(
            num_joints=25,
            feat_dim=6,
            audio_channels=4,
            audio_freq=64,
            d_model=512,
            num_temporal_blocks=3,
        )

    def forward(self, keypoints):
        """
        keypoints: [B, T, J, F]
        return: [B, C, T, Freq]
        """
        motion_features = self.motion_bert(keypoints)
        encoded_motion = self.motion_encoder(motion_features)
        return encoded_motion


class InputHintBlockKeypoints1D(nn.Module):
    """
    2D keypoints(COCO) を時間Conv1dで処理 → 学習された周波数ゲートで (N,C,T,F) へ拡張 →
    最後に zero-init 1x1 Conv で model_channels へ射影する block。
    - hint は (N, T, C_kp) または (N, C_kp, T) を受け付ける
    - 出力は (N, model_channels, T_out, F_target)
    """
    def __init__(
        self,
        hint_channels: int,      # 例: 138 (= 17 keypoints * (x,y,conf) * ？フレーム毎整形数)
        model_channels: int,
        F_target: int,           # UNet latent の周波数サイズに合わせる
        mid_channels: int = 64,
        T_target: int | None = None,   # UNet latent の時間サイズに事前に合わせたい場合(任意)
    ):
        super().__init__()
        self.hint_channels = hint_channels
        self.F_target = F_target
        self.T_target = T_target

        # 1) 時間方向のConv1dスタック（keypointsの「列」を時系列特徴へ）
        self.time_stack = nn.Sequential(
            nn.Conv1d(hint_channels, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # 2) 周波数ゲート（学習パラメータ）。rank-1 の外積で (N, mid, T, F) を生成
        #    初期ゼロに近い値で制御が徐々に効くようにする（ControlNet的ふるまい）
        self.freq_gate = nn.Parameter(torch.zeros(1, 1, 1, F_target))

        # 3) 最後に zero-init 1x1 Conv（周波数方向は広がった後、チャネルだけ合わせる）
        self.proj_out = zero_module(nn.Conv2d(mid_channels, model_channels, kernel_size=1))

    def forward(self, hint, emb=None, context_list=None, context_attn_mask_list=None):
        """
        hint: (N, T, Ck) or (N, Ck, T)
        return: (N, model_channels, T_out, F_target)
        """
        # Conv1d 入力の (N, C, T) に合わせる
        if hint.dim() != 3:
            raise ValueError("hint must be 3D (N,T,Ck) or (N,Ck,T)")
        if hint.size(1) == self.hint_channels:
            # (N, Ck, T)
            h1d = hint
        elif hint.size(2) == self.hint_channels:
            # (N, T, Ck) -> (N, Ck, T)
            h1d = hint.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"hint last dim must be {self.hint_channels}, "
                             f"got shape {tuple(hint.shape)}")

        # 1) 時間Conv1d
        feat = self.time_stack(h1d)  # (N, mid, T')

        # 任意: UNet の時間長に事前整合（未指定なら later で足し込み前に合わせてもOK）
        if (self.T_target is not None) and (feat.size(-1) != self.T_target):
            # 2D化して両軸同時に補間（幅1→F_targetへも同時に行うので後段不要）
            feat2d = feat.unsqueeze(-1)                                 # (N, mid, T', 1)
            feat2d = F.interpolate(feat2d, size=(self.T_target, self.F_target),
                                   mode="bilinear", align_corners=False)  # (N, mid, T_target, F_target)
        else:
            # 2) 周波数ゲートで 2D 展開（(N, mid, T', 1) × (1,1,1,F) → (N, mid, T', F)）
            feat2d = feat.unsqueeze(-1) * self.freq_gate                # (N, mid, T', F_target)

        # 3) zero-init 1x1 Conv で model_channels に整形
        out = self.proj_out(feat2d)                                     # (N, model_channels, T_out, F_target)
        return out