# Quick Start Guide - MotionBERT Feature Extraction

✅ **Status**: All tests passed! Ready to use.

## 1. Test MotionBERT Feature Extraction (5 minutes)

```bash
cd /home/sangheon/Desktop/AudioLDM-ControlNet
bash run_motionbert_test.sh
```

**Expected output:** `✅ All tests passed successfully!`

## 2. Extract Features for Full Dataset (may take hours)

```bash
cd /home/sangheon/Desktop/AudioLDM-ControlNet/BeatDance/preprocess
conda activate audioldm_train
export CUDA_VISIBLE_DEVICES=1
python extract_features.py
```

**Output location:** `data/dataset/AIST/beatdance_features/`

## 3. Train AudioLDM-ControlNet

```bash
cd /home/sangheon/Desktop/AudioLDM-ControlNet
bash bash_train_controlnet_beatdance.sh
```

## Key Changes

### What's Different in Feature Extraction

| Aspect | Old (Pose-based) | New (MotionBERT-based) |
|--------|------------------|------------------------|
| Method | Handcrafted geometric features | Learned representations |
| Input | COCO keypoints | COCO keypoints |
| Processing | Direct calculation | Pretrained neural network |
| Output dim | 138 | 8704 (17×512) |
| Captures | Low-level geometry | High-level motion patterns |

### Files Modified

- ✏️ `BeatDance/preprocess/extract_features.py` - `Extractor_fd` class replaced

### Files Added

- ✨ `test_motionbert_simple.py` - Test script
- ✨ `run_motionbert_test.sh` - Test runner
- ✨ `TEST_MOTIONBERT.md` - Detailed documentation
- ✨ `QUICK_START.md` - This file

## Troubleshooting Quick Fixes

**CUDA not available?**
```bash
# Check GPU
nvidia-smi
# Set correct GPU
export CUDA_VISIBLE_DEVICES=1
```

**Module import errors?**
```bash
# Ensure conda environment is activated
conda activate audioldm_train
# Check python version
python --version  # Should be 3.10.x
```

**Out of memory?**
- Use a GPU with more VRAM (currently using GPU 1)
- Process dataset in batches
- Reduce batch size in the script

**Checkpoint not found?**
```bash
ls -lh data/checkpoints/latest_epoch.bin
# Should show ~162MB file
# If missing, download from MotionBERT releases
```

## Dataset Structure

```
data/dataset/AIST/
├── audio_clips/          # Original audio files
├── keypoints_clips/      # Raw 2D keypoints (input)
│   ├── train/
│   └── test/
└── beatdance_features/   # Extracted features (output)
    ├── music_feature/    # MERT audio features
    ├── music_beat/       # Music beat tracking
    ├── video_feature/    # MotionBERT motion features ✨ NEW
    └── video_beat/       # Motion beat tracking
```

## Feature Dimensions

| Feature Type | Shape | Description |
|--------------|-------|-------------|
| Music feature | (128, 768) | MERT representations |
| Music beat | (4, 110) | Beat presence vectors |
| **Video feature** | **(~109, 8704)** | **MotionBERT motion representations** ✨ |
| Video beat | (3, 110) | Motion beat + velocity |

**Note:** Video feature length varies by clip duration: `T ≈ original_frames * 21.5 / 60`

## Verification Commands

**Check extracted features exist:**
```bash
ls -lh data/dataset/AIST/beatdance_features/video_feature/test/mJS3/*.pt | head -5
```

**Inspect a feature file:**
```python
import torch
feat = torch.load('path/to/feature.pt')
print(feat.shape)  # Should be (T, 8704)
```

**Compare old vs new feature dimensions:**
```python
# Old: (T, 138) - geometric features
# New: (T, 8704) - learned features
# Ratio: 8704/138 ≈ 63x more feature dimensions
```

## Need More Help?

- 📖 See [TEST_MOTIONBERT.md](TEST_MOTIONBERT.md) for detailed documentation
- 📖 See [README.md](README.md) for full training setup
- 🐛 Check error logs in the terminal output
- 💡 Each test script includes detailed error messages
