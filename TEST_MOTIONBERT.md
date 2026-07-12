# MotionBERT extraction verification

The smoke test loads one COCO keypoint pickle, applies the same normalization as
the dataset pipeline, loads the pretrained MotionBERT checkpoint, and verifies
that the resulting `[128, 8704]` tensor is finite.

```bash
conda activate audioldm_train
KEYPOINTS=data/dataset/aist/keypoints_clips/test/example.pkl \
CHECKPOINT=data/checkpoints/latest_epoch.bin \
CUDA_VISIBLE_DEVICES=0 \
bash run_motionbert_test.sh
```

The equivalent Python entry point is:

```bash
python -m BeatDance.preprocess.test_motionbert_extraction \
  --checkpoint data/checkpoints/latest_epoch.bin \
  --keypoints path/to/keypoints.pkl
```

Requirements:

- the checkpoint contains `model_pos`, matching MotionBERT's DataParallel state;
- keypoints have shape `[frames, 17, 3]` in COCO order;
- CUDA and the project environment are available.

Sequences longer than MotionBERT's 243-frame positional limit are first
resampled to 243 frames. The representation is then resampled to the fixed
128-frame BeatDance input used by the paper configs. See
[MOTIONBERT_MAXLEN_HANDLING.md](MOTIONBERT_MAXLEN_HANDLING.md) for details.
