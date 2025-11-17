"""
adopted from
https://github.com/lllyasviel/ControlNet/blob/ed85cd1e25a5ed592f7d8178495b4483de0331bf/cldm/cldm.py
"""
import os
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from audioldm_train.modules.diffusionmodules.attention import SpatialTransformer
from audioldm_train.modules.diffusionmodules.openaimodel import (
    AttentionBlock,
    Downsample,
    ResBlock,
    TimestepEmbedSequential,
    UNetModel,
)
from audioldm_train.modules.latent_diffusion.ddim import DDIMSampler
from audioldm_train.modules.latent_diffusion.ddpm import (
    LatentDiffusion,
)
from audioldm_train.utilities.diffusion_util import (
    conv_nd,
    linear,
    timestep_embedding,
    zero_module,
)
from audioldm_train.utilities.model_util import instantiate_from_config


class ControlledUnetModel(UNetModel):
    """
    modified based on differences between
        - 'UNetModel' in StableDiffusion and 'UNetModel' in AudioLDM
    """

    def forward(
        self,
        x,
        timesteps=None,
        y=None,
        context_list=None,
        context_attn_mask_list=None,
        controlnet_hint_list=None,
        only_mid_control=False,
        **kwargs,
    ):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(
                timesteps, self.model_channels, repeat_only=False
            )
            emb = self.time_embed(t_emb)
            if getattr(self, "use_extra_film_by_concat", False):
                assert y is not None
                emb = th.cat([emb, self.film_emb(y)], dim=-1)

            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context_list, context_attn_mask_list)
                hs.append(h)
            h = self.middle_block(h, emb, context_list, context_attn_mask_list)

        if len(controlnet_hint_list) > 0:
            for controlnet_hint in controlnet_hint_list:
                h = h + controlnet_hint.pop()

        for module in self.output_blocks:
            skip = hs.pop()
            if only_mid_control or len(controlnet_hint_list) == 0:
                h = th.cat([h, skip], dim=1)
            else:
                for controlnet_hint in controlnet_hint_list:
                    h = th.cat([h, skip + controlnet_hint.pop()], dim=1)
            h = module(h, emb, context_list, context_attn_mask_list)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class PoseEncoder(nn.Module):
    """
    Simple Pose Encoder for 2D COCO keypoints to match a target latent dimension.
    - Input: keypoints of shape (N, T, J, 2) or (N, T, J, 3) where the 3rd channel can be visibility/confidence (v).
    - Output: (N, T, out_dim) by default. You can set out_layout='NCT' to get (N, out_dim, T),
              or return_2d_hint=True to get (N, out_dim, T, 1) for 2D-conv backbones.

    Design (lightweight, MotionBERT downstream-friendly):
      1) Root-centering using pelvis (midpoint of left/right hip).
      2) Scale normalization using torso length (distance between mid-shoulders and mid-hips).
      3) Optional per-joint learnable embeddings (joint id embedding).
      4) Per-frame MLP to project (J * feat_per_joint) -> hidden -> out_dim.
      5) Optional shallow temporal depthwise conv for smoothing (residual).
      6) Confidence/visibility masking: if provided, downweight or drop unreliable joints.

    Note:
      - Time dimension T is preserved as-is (already aligned with music latent).
      - Keep it simple so that you can later replace this with MotionBERT features.
    """

    # COCO 17-joint order (common variant):
    # 0:nose,1:left_eye,2:right_eye,3:left_ear,4:right_ear,
    # 5:left_shoulder,6:right_shoulder,7:left_elbow,8:right_elbow,9:left_wrist,10:right_wrist,
    # 11:left_hip,12:right_hip,13:left_knee,14:right_knee,15:left_ankle,16:right_ankle
    COCO_LEFT_SHOULDER = 5
    COCO_RIGHT_SHOULDER = 6
    COCO_LEFT_HIP = 11
    COCO_RIGHT_HIP = 12

    def __init__(
        self,
        num_joints: int = 17,
        in_channels: int = 2,          # 2 (x,y) or 3 (x,y,v)
        out_dim: int = 256,            # target latent dim to match music latent
        hidden_dim: int = 512,         # MLP hidden
        use_joint_embed: bool = True,  # learnable joint id embeddings
        joint_embed_dim: int = 8,      # small embedding for joint identity
        use_temporal_smoothing: bool = True,
        temporal_kernel: int = 3,      # temporal depthwise kernel
        out_layout: str = "NTC",       # "NTC" -> (N,T,C), "NCT" -> (N,C,T)
        return_2d_hint: bool = False,  # if True, returns (N, C, T, 1) for 2D conv consumers
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.J = num_joints
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.eps = epsilon
        self.use_joint_embed = use_joint_embed
        self.joint_embed_dim = joint_embed_dim if use_joint_embed else 0
        self.use_temporal_smoothing = use_temporal_smoothing
        self.temporal_kernel = temporal_kernel
        self.out_layout = out_layout
        self.return_2d_hint = return_2d_hint

        feat_per_joint = 2  # we only encode (x,y) after pre-processing; confidence handled as weights
        in_per_frame = self.J * (feat_per_joint + self.joint_embed_dim)

        # per-joint id embedding (optional)
        if self.use_joint_embed:
            self.joint_embed = nn.Embedding(self.J, self.joint_embed_dim)
            nn.init.normal_(self.joint_embed.weight, std=0.02)

        # simple per-frame MLP
        self.proj = nn.Sequential(
            nn.Linear(in_per_frame, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

        # shallow temporal smoothing with depthwise separable conv (residual)
        if self.use_temporal_smoothing:
            # conv expects (N, C, T). We'll permute around it.
            self.dw_conv = nn.Conv1d(out_dim, out_dim, kernel_size=self.temporal_kernel,
                                     padding=self.temporal_kernel // 2, groups=out_dim, bias=True)
            self.pw_conv = nn.Conv1d(out_dim, out_dim, kernel_size=1, bias=True)
            nn.init.zeros_(self.dw_conv.bias)
            nn.init.zeros_(self.pw_conv.bias)

    @staticmethod
    def _midpoint(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a, b: (..., 2)
        return 0.5 * (a + b)

    def _root_and_scale(self, kpt_xy: torch.Tensor, vis: torch.Tensor = None):
        """
        Root-center (pelvis) and scale-normalize by torso length.
        kpt_xy: (N, T, J, 2)
        vis: (N, T, J, 1) or None (0/1 mask or soft weights). If None, use ones.
        Returns:
          norm_xy: (N, T, J, 2), scale-invariant, root-centered
          weights: (N, T, J, 1) in [0,1]
        """
        N, T, J, _ = kpt_xy.shape
        device = kpt_xy.device

        if vis is None:
            weights = torch.ones((N, T, J, 1), device=device, dtype=kpt_xy.dtype)
        else:
            # clamp to [0,1]
            weights = torch.clamp(vis, 0.0, 1.0)

        # compute pelvis = midpoint of left/right hip
        lhip = kpt_xy[..., self.COCO_LEFT_HIP, :]   # (N,T,2)
        rhip = kpt_xy[..., self.COCO_RIGHT_HIP, :]  # (N,T,2)
        pelvis = self._midpoint(lhip, rhip)         # (N,T,2)

        # root-center
        centered = kpt_xy - pelvis.unsqueeze(-2)    # (N,T,J,2)

        # torso length = distance between mid-shoulders and mid-hips
        lsho = kpt_xy[..., self.COCO_LEFT_SHOULDER, :]
        rsho = kpt_xy[..., self.COCO_RIGHT_SHOULDER, :]
        mid_sho = self._midpoint(lsho, rsho)        # (N,T,2)
        mid_hip = self._midpoint(lhip, rhip)        # (N,T,2)
        torso = torch.linalg.norm(mid_sho - mid_hip, dim=-1, keepdim=True)  # (N,T,1)
        torso = torch.clamp(torso, min=self.eps)

        norm_xy = centered / torso.unsqueeze(-2)    # (N,T,J,2)

        # if some frames are fully invalid, keep them small (already handled by weights in aggregation)
        return norm_xy, weights

    def forward(self, keypoints: torch.Tensor):
        """
        keypoints: (N, T, J, 2) or (N, T, J, 3) with (x,y[,v])
        Returns:
          (N, T, out_dim) by default, or (N, out_dim, T) if out_layout='NCT',
          or (N, out_dim, T, 1) if return_2d_hint=True.
        """
        assert keypoints.dim() == 4, "keypoints must be (N, T, J, C)"
        N, T, J, C = keypoints.shape
        assert J == self.J, f"num_joints mismatch: expected {self.J}, got {J}"
        assert C in (2, 3), "last channel must be 2:(x,y) or 3:(x,y,v)"

        if C == 3:
            xy = keypoints[..., :2]
            v = keypoints[..., 2:].unsqueeze(-1) if keypoints.size(-1) == 3 else None  # (N,T,J,1)
            # if keypoints[...,2] is already (N,T,J,1) you can adapt above line
            v = keypoints[..., 2].unsqueeze(-1)  # ensure (N,T,J,1)
        else:
            xy = keypoints
            v = None

        # 1) root-centering & scale-normalization
        xy, w = self._root_and_scale(xy, v)  # xy: (N,T,J,2), w: (N,T,J,1)

        # 2) (optional) joint id embeddings
        if self.use_joint_embed:
            # joint ids: 0..J-1
            joint_ids = torch.arange(self.J, device=xy.device).long()  # (J,)
            je = self.joint_embed(joint_ids)  # (J, E)
            je = je.view(1, 1, self.J, self.joint_embed_dim).expand(N, T, self.J, self.joint_embed_dim)
            feat = torch.cat([xy, je], dim=-1)  # (N,T,J,2+E)
        else:
            feat = xy  # (N,T,J,2)

        # 3) apply visibility/confidence weights (elementwise, pass-through)
        if w is not None:
            # multiply only the coordinate part; for joint embeddings we can keep them unchanged
            if self.use_joint_embed:
                coords = feat[..., :2]
                rest = feat[..., 2:]
                coords = coords * w
                feat = torch.cat([coords, rest], dim=-1)
            else:
                feat = feat * w

        # 4) per-frame MLP projection
        feat = feat.reshape(N, T, -1)        # (N,T,J*(2+E))
        proj = self.proj(feat)               # (N,T,out_dim)

        # 5) optional shallow temporal smoothing (depthwise separable conv)
        if self.use_temporal_smoothing:
            # to (N,C,T)
            x = proj.transpose(1, 2)         # (N,out_dim,T)
            res = self.pw_conv(self.dw_conv(x))
            x = x + res                      # residual
            proj = x.transpose(1, 2)         # (N,T,out_dim)

        # 6) layout / hint shape
        if self.return_2d_hint:
            # (N, out_dim, T, 1) for 2D-conv consumers (e.g., ControlNet hint path)
            out = proj.transpose(1, 2).unsqueeze(-1)
            return out  # (N, C, T, 1)

        if self.out_layout == "NCT":
            return proj.transpose(1, 2)      # (N,out_dim,T)
        # default: "NTC"
        return proj                          # (N,T,out_dim)


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


class ControlNet(nn.Module):
    """
    modified based on differences between
        - 'ControlNet' in StableDiffusion and 'UNetModel' in StableDiffusion
        - 'UNetModel' in StableDiffusion and 'UNetModel' in AudioLDM
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        extra_sa_layer=True,
        num_classes=None,
        extra_film_condition_dim=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        latent_t_size=256,
        latent_f_size=16,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1, (
                "Either num_heads or num_head_channels has to be set"
            )
        if num_head_channels == -1:
            assert num_heads != -1, (
                "Either num_heads or num_head_channels has to be set"
            )

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.extra_sa_layer = extra_sa_layer
        self.num_classes = num_classes
        self.extra_film_condition_dim = extra_film_condition_dim
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.use_spatial_transformer = use_spatial_transformer
        self.legacy = legacy

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.use_extra_film_by_concat = self.extra_film_condition_dim is not None
        if self.extra_film_condition_dim is not None:
            self.film_emb = nn.Linear(self.extra_film_condition_dim, time_embed_dim)

        if context_dim is not None and not isinstance(context_dim, list):
            context_dim = [context_dim]
        elif context_dim is None:
            context_dim = [None]  # At least use one spatial transformer

        self.input_hint_block = InputHintBlockKeypoints1D(
            hint_channels=138,
            model_channels=model_channels,
            F_target=latent_f_size,
            mid_channels=64,
            T_target=latent_t_size,
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim
                        if (not self.use_extra_film_by_concat)
                        else time_embed_dim * 2,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if extra_sa_layer:
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=None,
                            )
                        )
                    for ctx_dim in context_dim:
                        if not use_spatial_transformer:
                            layers.append(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads,
                                    num_head_channels=dim_head,
                                    use_new_attention_order=use_new_attention_order,
                                )
                            )
                        else:
                            layers.append(
                                SpatialTransformer(
                                    ch,
                                    num_heads,
                                    dim_head,
                                    depth=transformer_depth,
                                    context_dim=ctx_dim,
                                )
                            )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim
                            if (not self.use_extra_film_by_concat)
                            else time_embed_dim * 2,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        middle_layers = [
            ResBlock(
                ch,
                time_embed_dim
                if (not self.use_extra_film_by_concat)
                else time_embed_dim * 2,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        ]
        if extra_sa_layer:
            middle_layers.append(
                SpatialTransformer(
                    ch, num_heads, dim_head, depth=transformer_depth, context_dim=None
                )
            )
        for ctx_dim in context_dim:
            if not use_spatial_transformer:
                middle_layers.append(
                    AttentionBlock(
                        ch,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads,
                        num_head_channels=dim_head,
                        use_new_attention_order=use_new_attention_order,
                    )
                )
            else:
                middle_layers.append(
                    SpatialTransformer(
                        ch,
                        num_heads,
                        dim_head,
                        depth=transformer_depth,
                        context_dim=ctx_dim,
                    )
                )
        middle_layers.append(
            ResBlock(
                ch,
                time_embed_dim
                if (not self.use_extra_film_by_concat)
                else time_embed_dim * 2,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            )
        )
        self.middle_block = TimestepEmbedSequential(*middle_layers)
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels: int):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        )

    def forward(
        self,
        x,  # latent (N, C, T, F)
        hint,
        timesteps=None,
        y=None,
        context_list=None,
        context_attn_mask_list=None,
        **kwargs,
    ):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.use_extra_film_by_concat:
            assert y is not None, (
                "must specify y if and only if the model is class-conditional or film embedding conditional"
            )
            emb = th.cat([emb, self.film_emb(y)], dim=-1)

        guided_hint = self.input_hint_block(
            hint, emb, context_list, context_attn_mask_list
        )

        outs = []
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context_list, context_attn_mask_list)
            if guided_hint is not None:
                h = h + guided_hint
                guided_hint = None
            outs.append(zero_conv(h, emb, context_list, context_attn_mask_list))

        h = self.middle_block(h, emb, context_list, context_attn_mask_list)
        outs.append(self.middle_block_out(h, emb, context_list, context_attn_mask_list))
        return outs


class ControlLDM(LatentDiffusion):
    """
    modified based on differences between
        - 'ControlLDM' in StableDiffusion and 'LatentDiffusion' in StableDiffusion
        - 'LatentDiffusion' in StableDiffusion and 'LatentDiffusion' in AudioLDM
    """

    def __init__(
        self,
        controlnet_stage_config,
        evaluator_config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.controlnet_stage_models = nn.ModuleList([])
        self.instantiate_controlnet_stage(controlnet_stage_config)
        self.controlnet_scales = [1.0] * 13
        self.metrics_buffer = {
            "val/frechet_audio_distance": 32.0,
            "val/beat_coverage_score": 0.0,
            "val/beat_hit_score": 0.0,
            "val/tempo_difference": 100.0,
            "val/clap_score": 0.0,
        }
        if self.global_rank == 0:
            self.evaluator = instantiate_from_config(evaluator_config)
            print("ControlLDM: evaluator instantiated.")
        else:
            self.evaluator = None

    def instantiate_controlnet_stage(self, config):
        self.controlnet_stage_model_metadata = {}
        for i, controlnet_model_key in enumerate(config.keys()):
            model = instantiate_from_config(config[controlnet_model_key])
            self.controlnet_stage_models.append(model)
            self.controlnet_stage_model_metadata[controlnet_model_key] = {
                "model_idx": i,
                "controlnet_stage_key": config[controlnet_model_key]["controlnet_stage_key"],
            }

    @torch.no_grad()
    def get_input(self, batch, k, *args, **kwargs):
        z, cond_dict = super().get_input(batch, k, *args, **kwargs)
        for controlnet_model_key in self.controlnet_stage_model_metadata.keys():
            controlnet_stage_key = self.controlnet_stage_model_metadata[
                controlnet_model_key
            ]["controlnet_stage_key"]
            assert controlnet_stage_key in batch, f"'{controlnet_stage_key}' not found in batch."
            control = batch[controlnet_stage_key].to(self.device)
            control = control.to(memory_format=torch.contiguous_format).float()
            cond_dict[controlnet_model_key] = control
        return z, cond_dict

    def filter_useful_cond_dict(self, cond_dict):
        new_cond_dict = {}
        for model_key in cond_dict.keys():
            if model_key in self.cond_stage_model_metadata.keys():
                new_cond_dict[model_key] = cond_dict[model_key]
            elif model_key in self.controlnet_stage_model_metadata.keys():
                new_cond_dict[model_key] = cond_dict[model_key]

        # All the conditional model_key in the metadata should be used
        for model_key in self.cond_stage_model_metadata.keys():
            assert model_key in new_cond_dict.keys(), "%s, %s" % (
                model_key,
                str(new_cond_dict.keys()),
            )
        for model_key in self.controlnet_stage_model_metadata.keys():
            assert model_key in new_cond_dict.keys(), "%s, %s" % (
                model_key,
                str(new_cond_dict.keys()),
            )

        return new_cond_dict

    def _flatten_sorted(self, keys, prefix):
        return sorted([k for k in keys if k.startswith(prefix)])

    def _get_unet_input(self, x, t, cond_dict):
        """
        This is just a partial copy of 'DiffusionWrapper.forward' method
        """
        x = x.contiguous()
        t = t.contiguous()

        # x with condition (or maybe not)
        xc = x

        y = None
        context_list, attn_mask_list, controlnet_dict = [], [], {}

        conditional_keys = cond_dict.keys()

        for key in conditional_keys:
            if "concat" in key:
                xc = torch.cat([x, cond_dict[key].unsqueeze(1)], dim=1)
            elif "film" in key:
                if y is None:
                    y = cond_dict[key].squeeze(1)
                else:
                    y = torch.cat([y, cond_dict[key].squeeze(1)], dim=-1)
            elif "crossattn" in key:
                # assert context is None, "You can only have one context matrix, got %s" % (cond_dict.keys())
                if isinstance(cond_dict[key], dict):
                    for k in cond_dict[key].keys():
                        if "crossattn" in k:
                            context, attn_mask = cond_dict[key][
                                k
                            ]  # crossattn_audiomae_pooled: torch.Size([12, 128, 768])
                else:
                    assert len(cond_dict[key]) == 2, (
                        "The context condition for %s you returned should have two element, one context one mask"
                        % (key)
                    )
                    context, attn_mask = cond_dict[key]

                # The input to the UNet model is a list of context matrix
                context_list.append(context)
                attn_mask_list.append(attn_mask)
            elif "controlnet" in key:
                controlnet_dict[key] = cond_dict[key]
            elif (
                "noncond" in key
            ):  # If you use loss function in the conditional module, include the keyword "noncond" in the return dictionary
                continue
            else:
                raise NotImplementedError()
        return xc, t, y, context_list, attn_mask_list, controlnet_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        The 'control' argument in the 'ControlledUnetModel.forward' method is added based on the 'UNetModel.forward' method.
        The 'DiffusionWrapper' class also needs to be modified to pass additional arguments.
        However, from a maintainability standpoint, it is undesirable to modify the original 'DiffusionWrapper' directly.
        Since the 'LatentDiffusion' class already inherits from the original 'DiffusionWrapper', creating and using a new 'DiffusionWrapper' would be cumbersome.
        Because the role of 'DiffusionWrapper' is limited, the modification will be handled here instead.
        """
        xc, t, y, context_list, context_attn_mask_list, controlnet_dict = self._get_unet_input(
            x_noisy, t, cond
        )

        controlnet_hint_list = []
        for controlnet_model_key, controlnet_hint in controlnet_dict.items():
            controlnet_model_idx = self.controlnet_stage_model_metadata[
                controlnet_model_key
            ]["model_idx"]
            controlnet_hint = self.controlnet_stage_models[controlnet_model_idx](
                x=x_noisy,
                hint=controlnet_hint,
                timesteps=t,
                y=y,
                context_list=context_list,
                context_attn_mask_list=context_attn_mask_list,
            )
            controlnet_hint = [c * scale for c, scale in zip(controlnet_hint, self.controlnet_scales)]
            controlnet_hint_list.append(controlnet_hint)

        # call the unet model directly, without through DiffusionWrapper
        diffusion_model = self.model.diffusion_model
        out = diffusion_model(
            xc,
            timesteps=t,
            y=y,
            context_list=context_list,
            context_attn_mask_list=context_attn_mask_list,
            controlnet_hint_list=controlnet_hint_list,
        )
        return out

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        try:
            return super().get_unconditional_conditioning(N)
        except Exception:
            return self.get_learned_conditioning([""] * N)

    def configure_optimizers(self):
        """
        make only ControlNet parameters to be trainable
        """
        lr = self.learning_rate
        params = None
        for each in self.controlnet_stage_models:
            if params is None:
                params = list(each.parameters())
            else:
                params = params + list(each.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
