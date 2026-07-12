# MotionBERT feature extraction quick start

From the repository root, activate the environment and run a one-file smoke test:

```bash
conda activate audioldm_train
KEYPOINTS=path/to/keypoints.pkl \
CHECKPOINT=data/checkpoints/latest_epoch.bin \
bash run_motionbert_test.sh
```

The keypoint pickle must have shape `[frames, 17, 3]` in COCO order. A successful
smoke test returns a finite MotionBERT feature tensor with shape `[128, 8704]`.

Extract the complete AIST++ feature set with:

```bash
python -m BeatDance.preprocess.extract_features
```

Use `--audio-dir`, `--keypoints-dir`, `--output-dir`, and
`--motionbert-checkpoint` when files are stored outside the default relative
paths. The output tree is:

```text
data/dataset/aist/beatdance_features/
├── music_feature/  # [128, 768]
├── music_beat/     # [3, 128]
├── video_feature/  # [128, 8704]
└── video_beat/     # [128, 2]
```

Continue with the Stage 1 and Stage 2 commands in [README.md](README.md).
