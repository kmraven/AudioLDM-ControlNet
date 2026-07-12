#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

: "${KEYPOINTS:?Set KEYPOINTS to a COCO keypoint pickle}"
CHECKPOINT="${CHECKPOINT:-data/checkpoints/latest_epoch.bin}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m BeatDance.preprocess.test_motionbert_extraction \
  --checkpoint "$CHECKPOINT" \
  --keypoints "$KEYPOINTS"
