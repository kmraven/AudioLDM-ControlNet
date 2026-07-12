# MotionBERT sequence-length handling

The pretrained MotionBERT configuration has a maximum positional length of 243
frames, while AIST++ clips can be longer. Feature extraction therefore uses the
following deterministic sequence:

1. interpolate missing COCO keypoint values;
2. normalize camera coordinates and convert COCO joints to H36M order;
3. resample only inputs longer than 243 frames down to 243;
4. obtain MotionBERT representations with shape `[T, 17, 512]`;
5. linearly resample the representations to the fixed BeatDance length of 128;
6. flatten joints and channels to `[128, 8704]`.

Shorter inputs are not expanded before MotionBERT. All clips are converted to
128 frames only after representation extraction, matching the Stage 1 feature
files and Stage 2 configuration.
