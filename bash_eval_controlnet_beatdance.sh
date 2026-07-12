#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-audioldm_train/config/2025_11_23_dance_controlnet_beatdance/audioldm_original_medium_stretch_pretrained_frozen.yaml}"
CHECKPOINT="${CHECKPOINT:-log/latent_diffusion_controlnet_beatdance/2025_11_23_dance_controlnet_beatdance/audioldm_original_medium_stretch_pretrained_frozen/checkpoints/checkpoint-global_step=299999.ckpt}"

python3 audioldm_train/train/latent_diffusion_controlnet_beatdance_test.py \
  --config_yaml "$CONFIG" \
  --reload_from_ckpt "$CHECKPOINT"
