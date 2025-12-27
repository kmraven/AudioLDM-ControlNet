# MotionBERT Feature Extraction Test

This guide helps you test the new MotionBERT-based feature extraction before running the full preprocessing pipeline.

## What Changed

The `Extractor_fd` class in `BeatDance/preprocess/extract_features.py` has been replaced:

**Old (Pose-based):**
- Extracted geometric features (joint positions, velocities, bone angles)
- Output: `(T, 138)` dimensions

**New (MotionBERT-based):**
- Extracts learned motion representations using pretrained MotionBERT
- Output: `(T, 8704)` dimensions (17 joints × 512 feature dims)
- Consistent with training pipeline preprocessing

## Quick Test

### Option 1: Run with script (easiest)

```bash
cd /home/sangheon/Desktop/AudioLDM-ControlNet
bash run_motionbert_test.sh
```

### Option 2: Run manually

```bash
cd /home/sangheon/Desktop/AudioLDM-ControlNet
conda activate audioldm_train
export CUDA_VISIBLE_DEVICES=1
python test_motionbert_simple.py
```

## What the Test Does

The test script will:

1. ✓ Check CUDA availability
2. ✓ Verify checkpoint and keypoint files exist
3. ✓ Load and inspect raw keypoints
4. ✓ Initialize MotionBERT model with pretrained weights
5. ✓ Preprocess keypoints (interpolate, normalize, convert format)
6. ✓ Extract MotionBERT features
7. ✓ Resample from 60 FPS to 21.5 FPS
8. ✓ Flatten output to `(T, 8704)` shape
9. ✓ Verify no NaN/Inf values
10. ✓ Test save/load functionality

## Expected Output

If successful, you'll see:

```
================================================================================
TEST SUMMARY
================================================================================
✅ All tests passed successfully!
✓ MotionBERT feature extraction is working correctly
✓ Output shape: torch.Size([109, 8704])
✓ Output feature dimension: 8704 (17 joints × 512 dims)
✓ Ready for full feature extraction pipeline
================================================================================
```

## After Successful Test

Once the test passes, you can run the full feature extraction:

```bash
cd /home/sangheon/Desktop/AudioLDM-ControlNet/BeatDance/preprocess
conda activate audioldm_train
export CUDA_VISIBLE_DEVICES=1
python extract_features.py
```

This will extract:
- **Music features** (fm): MERT-based audio features → `beatdance_features/music_feature/`
- **Music beat features** (fbm): Librosa beat tracking → `beatdance_features/music_beat/`
- **Video features** (fd): **MotionBERT representations** → `beatdance_features/video_feature/`
- **Video beat features** (fbd): Motion velocity beats → `beatdance_features/video_beat/`

## Troubleshooting

### CUDA not available
```
❌ ERROR: CUDA not available. MotionBERT requires GPU.
```
**Solution:** Ensure you're running on a machine with a GPU and CUDA is properly installed.

### Checkpoint not found
```
❌ ERROR: MotionBERT checkpoint not found
```
**Solution:** Download MotionBERT checkpoint and place at:
```
data/checkpoints/latest_epoch.bin
```

### Out of memory
If you get CUDA OOM errors during full extraction:
- Reduce batch processing in `extract_features.py`
- Use a GPU with more memory
- Process files in smaller batches

### Shape mismatch
If output shape doesn't match expected:
- Check preprocessing parameters match training config
- Verify keypoint files have correct format `(T, 17, 3)`

## Files Created

Test files:
- `test_motionbert_simple.py` - Simple standalone test script
- `run_motionbert_test.sh` - Convenience script to run test
- `BeatDance/preprocess/test_motionbert_extraction.py` - Detailed test (requires pandas)
- `BeatDance/preprocess/run_test.sh` - Alternative test runner

Modified files:
- `BeatDance/preprocess/extract_features.py` - Updated `Extractor_fd` class

## Technical Details

### MotionBERT Model Configuration

```python
DSTformerWrapper(
    dim_in=3,           # Input: (x, y, confidence)
    dim_out=3,          # Output pose estimation (not used)
    dim_feat=512,       # Intermediate feature dimension
    dim_rep=512,        # Representation dimension
    depth=5,            # Transformer depth
    num_heads=8,        # Attention heads
    mlp_ratio=2,        # MLP expansion ratio
    num_joints=17,      # H36M skeleton format
    maxlen=243          # Max sequence length
)
```

### Preprocessing Pipeline

1. Load COCO keypoints `(T, 17, 3)`
2. Interpolate NaN values
3. Normalize to camera coordinates
4. Convert COCO → H36M format
5. Crop and scale
6. Feed to MotionBERT → `(1, T, 17, 512)`
7. Resample 60 FPS → 21.5 FPS
8. Flatten → `(T_resampled, 8704)`

### Output Format

The extracted features are saved as PyTorch tensors (`.pt` files):

```python
features: torch.Tensor  # Shape: (T, 8704)
# where T ≈ original_frames * 21.5 / 60
# and 8704 = 17 joints × 512 dims
```

## Next Steps

After extracting features, you can:
1. Train the BeatDance model using these features
2. Use features for dance-to-music generation
3. Integrate with AudioLDM-ControlNet training pipeline
