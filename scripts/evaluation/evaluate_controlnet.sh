#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-audioldm_train/config/2025_11_08_dance_controlnet/audioldm_original_medium_stretch.yaml}"
: "${CHECKPOINT:?Set CHECKPOINT to the AudioLDM ControlNet checkpoint to evaluate}"

python3 audioldm_train/train/latent_diffusion_controlnet_test.py \
  --config_yaml "$CONFIG" \
  --reload_from_ckpt "$CHECKPOINT"
