# MotionBERT Verification Checklist

This file is retained as the verification checklist for the preprocessing
pipeline. Run the current smoke test rather than relying on historical output:

```bash
KEYPOINTS=path/to/keypoints.pkl \
CHECKPOINT=data/checkpoints/latest_epoch.bin \
bash scripts/testing/run_motionbert_test.sh
```

Expected invariants:

- input shape is `[frames, 17, 3]`;
- sequences longer than 243 frames are resampled before MotionBERT;
- output shape is `[128, 8704]`;
- output contains no NaN or infinity;
- all four BeatDance feature trees use matching relative clip names.

Dataset-wide feature extraction is run with:

```bash
python -m BeatDance.preprocess.extract_features
```
