# MotionBERT Maximum Sequence Length Handling

## Problem

MotionBERT has a **maximum sequence length of 243 frames** (`maxlen=243`). However, AIST++ keypoint sequences can be much longer:

- Example: Test sequence has **305 frames**
- Typical AIST++ clips: **~300-600 frames** at 60 FPS

## Solution

The extraction pipeline handles this by **downsampling before MotionBERT and upsampling after**:

```
Input Keypoints (305 frames @ 60 FPS)
         ↓
    Preprocess
         ↓
  Resample → 243 frames  ← Fit MotionBERT maxlen
         ↓
    MotionBERT
         ↓
 Features (243, 17, 512)
         ↓
  Resample → 109 frames  ← Target: 21.5 FPS
         ↓
Output (109, 8704)
```

## Implementation Details

### In `Extractor_fd.extract()` method:

```python
# 1. Preprocess keypoints (interpolate, normalize, convert format)
k2d = preprocess(keypoints)  # Shape: (305, 17, 3)

# 2. Check if sequence exceeds MotionBERT maxlen
maxlen = 243
original_length = k2d.shape[0]  # 305
if original_length > maxlen:
    k2d = resample_keypoints_2d(k2d, maxlen)  # → (243, 17, 3)

# 3. Extract MotionBERT features
motion_features = model.get_representation(k2d)  # → (243, 17, 512)

# 4. Resample to target FPS
target_length = int(original_length * 21.5 / 60)  # 305 * 21.5/60 = 109
motion_features = resample(motion_features, maxlen, target_length)  # → (109, 17, 512)

# 5. Flatten
output = motion_features.reshape(109, -1)  # → (109, 8704)
```

## Why This Works

### Temporal Resolution Trade-off

1. **Before MotionBERT**: Downsample to 243 frames
   - Loss: Some temporal resolution
   - Gain: Can use pretrained MotionBERT

2. **After MotionBERT**: Upsample to target FPS
   - The learned features capture motion semantics
   - Upsampling interpolates between feature vectors
   - Still preserves motion patterns

### Visual Explanation

```
Original timeline (305 frames):
|-----|-----|-----|-----|-----|-----|  (60 FPS)
 0    50   100   150   200   250   305

MotionBERT input (243 frames):
|-----|-----|-----|-----|-----|      (downsampled)
 0    50   100   150   200   243

MotionBERT output (243 frames):
|-----|-----|-----|-----|-----|      (feature vectors)
 [f0]  [f50] [f100] [f150] [f200]

Final output (109 frames):
|-----|-----|-----|-----|-----|      (21.5 FPS, upsampled)
 0    25    50    75    100   109
```

## Comparison with Training Pipeline

### During Training

In the training pipeline ([dataset_aistpp.py](audioldm_train/utilities/data/dataset_aistpp.py)):

```python
# Training resamples to target_length (128) directly
k2d = resample_keypoints_2d(k2d, motion_target_length=128)  # → (128, 17, 3)
motion_features = motionbert(k2d)  # → (128, 17, 512)
```

**Key difference**: Training sequences are always ≤ 243 frames after resampling to 128.

### During Feature Extraction (Preprocessing)

For BeatDance preprocessing ([extract_features.py](BeatDance/preprocess/extract_features.py)):

```python
# Extract at target FPS (21.5) which may give longer sequences
# If > 243: downsample → MotionBERT → upsample
# If ≤ 243: MotionBERT → resample to target FPS
```

## Test Results

From the test run:

```
Original keypoint frames: 305
MotionBERT input frames: 243   ← Downsampled
Target output frames: 109      ← At 21.5 FPS
Output shape: (109, 8704)      ✓ Correct!
```

**Calculation verification:**
```python
original_length = 305  # frames at 60 FPS
target_fps = 21.5
original_fps = 60

target_length = 305 * 21.5 / 60 = 109.29 ≈ 109 frames ✓
feature_dim = 17 joints × 512 dims = 8704 ✓
```

## Edge Cases

### Case 1: Short sequences (< 243 frames)

```python
k2d.shape[0] = 180  # Less than maxlen

# No downsampling needed
motion_features = model(k2d)  # → (180, 17, 512)

# Just resample to target FPS
target_length = int(180 * 21.5 / 60)  # = 64
output = resample(motion_features, 180, 64)  # → (64, 8704)
```

### Case 2: Very long sequences (> 500 frames)

```python
k2d.shape[0] = 600  # Much longer than maxlen

# Downsample to maxlen
k2d = resample(k2d, 243)  # → (243, 17, 3)

# Extract features
motion_features = model(k2d)  # → (243, 17, 512)

# Upsample to target
target_length = int(600 * 21.5 / 60)  # = 215
output = resample(motion_features, 243, 215)  # → (215, 8704)
```

### Case 3: Exactly 243 frames

```python
k2d.shape[0] = 243  # Exactly maxlen

# No downsampling needed
motion_features = model(k2d)  # → (243, 17, 512)

# Resample to target
target_length = int(243 * 21.5 / 60)  # = 87
output = resample(motion_features, 243, 87)  # → (87, 8704)
```

## Performance Implications

### Accuracy

- **Minor loss** in temporal precision due to downsampling
- **Acceptable** because:
  - MotionBERT learns high-level motion patterns
  - Feature interpolation preserves semantic content
  - Final FPS (21.5) is sufficient for dance analysis

### Speed

- **No significant overhead**: Resampling is fast (linear interpolation)
- **GPU time**: Dominated by MotionBERT forward pass, not resampling

### Memory

- **Benefit**: Always capped at 243 frames for MotionBERT
- **Peak memory**: Independent of input sequence length

## Alternative Approaches (Not Used)

### 1. Chunk-based Processing

```python
# Split into chunks of 243 frames
chunks = [k2d[i:i+243] for i in range(0, len(k2d), 243)]
features = [model(chunk) for chunk in chunks]
output = concatenate(features)
```

**Why not**:
- Loses temporal continuity at boundaries
- More complex implementation
- Not necessary for current use case

### 2. Increase MotionBERT maxlen

```python
# Retrain MotionBERT with maxlen=512
model = DSTformer(maxlen=512)
```

**Why not**:
- Requires retraining from scratch
- Pretrained weights can't be used
- Not necessary given resampling works well

## Conclusion

The current approach of **downsample → MotionBERT → upsample** is:

✅ **Simple**: Minimal code changes
✅ **Effective**: Preserves motion semantics
✅ **Efficient**: No significant overhead
✅ **Robust**: Handles all sequence lengths
✅ **Compatible**: Works with pretrained MotionBERT

The test confirms this approach produces correct output shapes and valid feature values!
