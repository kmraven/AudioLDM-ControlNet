# Preprocessing implementation notes

The current preprocessing path uses the pretrained MotionBERT representation
instead of handcrafted pose features. Shared keypoint operations live in
`audioldm_train/utilities/data/keypoints.py`, and the runnable extraction entry
point is `python -m BeatDance.preprocess.extract_features`.

For every AIST++ clip, the pipeline writes:

| Feature | Shape |
| --- | --- |
| MERT music representation | `[128, 768]` |
| Music beat representation | `[3, 128]` |
| MotionBERT representation | `[128, 8704]` |
| Motion beat representation | `[128, 2]` |

The MotionBERT checkpoint defaults to `data/checkpoints/latest_epoch.bin` and
must contain the `model_pos` state dictionary. See [README.md](README.md) for the
complete two-stage reproduction workflow.
