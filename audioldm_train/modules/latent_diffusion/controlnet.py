"""
adopted from
https://github.com/lllyasviel/ControlNet/blob/ed85cd1e25a5ed592f7d8178495b4483de0331bf/cldm/cldm.py
"""

import torch
import torch as th
import torch.nn as nn

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
        control=None,
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

        if control is not None:
            h = h + control.pop()

        for module in self.output_blocks:
            skip = hs.pop()
            if only_mid_control or control is None:
                h = th.cat([h, skip], dim=1)
            else:
                h = th.cat([h, skip + control.pop()], dim=1)
            h = module(h, emb, context_list, context_attn_mask_list)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


# TODO: check and modify the shape of zero_module conv layers if necessary
class MotionEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, dims):
        super().__init__()
        self.model = TimestepEmbedSequential(
            conv_nd(dims, in_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, out_channels, 3, padding=1)),
        )

    def forward(self, x):
        return self.model(x)


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
        hint_channels,
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

        self.input_hint_block = MotionEncoder(hint_channels, model_channels, dims)

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
        control_stage_config,
        control_key,
        only_mid_control=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def get_input(self, batch, k, *args, **kwargs):
        z, cond_dict = super().get_input(batch, k, *args, **kwargs)
        assert self.control_key in batch, f"'{self.control_key}' not found in batch."
        control = batch[self.control_key].to(self.device)
        # TODO: check a shape of control hint
        if control.dim() == 3:
            control = control.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected control hint shape: {control.shape}")

        control = control.to(memory_format=torch.contiguous_format).float()
        cond = dict(cond_dict=cond_dict, control_hint=control)
        return z, cond

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
        context_list, attn_mask_list = [], []

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

            elif (
                "noncond" in key
            ):  # If you use loss function in the conditional module, include the keyword "noncond" in the return dictionary
                continue
            else:
                raise NotImplementedError()
        return xc, t, y, context_list, attn_mask_list

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        """
        The 'control' argument in the 'ControlledUnetModel.forward' method is added based on the 'UNetModel.forward' method.
        The 'DiffusionWrapper' class also needs to be modified to pass additional arguments.
        However, from a maintainability standpoint, it is undesirable to modify the original 'DiffusionWrapper' directly.
        Since the 'LatentDiffusion' class already inherits from the original 'DiffusionWrapper', creating and using a new 'DiffusionWrapper' would be cumbersome.
        Because the role of 'DiffusionWrapper' is limited, the modification will be handled here instead.
        """
        assert isinstance(cond, dict) and "cond_dict" in cond, (
            "cond must be {'cond_dict':..., 'control_hint':...}"
        )
        cond_dict = cond["cond_dict"]
        control_hint = cond.get("control_hint", None)

        xc, t, y, context_list, context_attn_mask_list = self._get_unet_input(
            x_noisy, t, cond_dict
        )

        if control_hint is None:
            control = None
        else:
            control = self.control_model(
                x=x_noisy,
                hint=control_hint,
                timesteps=t,
                y=y,
                context_list=context_list,
                context_attn_mask_list=context_attn_mask_list,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]

        # call the unet model directly, without through DiffusionWrapper
        diffusion_model = self.model.diffusion_model
        out = diffusion_model(
            xc,
            timesteps=t,
            y=y,
            context_list=context_list,
            context_attn_mask_list=context_attn_mask_list,
            control=control,
            only_mid_control=self.only_mid_control,
        )
        return out

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        try:
            return super().get_unconditional_conditioning(N)
        except Exception:
            return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim=True, ddim_steps=50, eta=0.0, **kwargs):
        """
        TODO: check this implementation
        """
        ddim_sampler = DDIMSampler(self)
        control_hint = cond.get("control_hint", None)
        if control_hint is None:
            raise ValueError("sample_log requires 'control_hint' to infer shape (T,F).")
        _, _, t_size, f_size = control_hint.shape
        factor = getattr(self, "downsampling_factor", 8)
        shape = (self.channels, t_size // factor, f_size // factor)
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            batch_size,
            shape,
            cond,
            verbose=False,
            eta=eta,
            ddim=ddim,
            **kwargs,
        )
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not getattr(self, "sd_locked", True):
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
