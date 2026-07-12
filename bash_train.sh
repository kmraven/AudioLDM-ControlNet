#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_crossattn_flant5.yaml}"
COMMAND=(python3 audioldm_train/train/latent_diffusion.py --config_yaml "$CONFIG")
if [[ -n "${CHECKPOINT:-}" ]]; then
  COMMAND+=(--reload_from_ckpt "$CHECKPOINT")
fi
"${COMMAND[@]}"
