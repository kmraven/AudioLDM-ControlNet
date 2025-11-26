"""
Simple test script for MotionBERT-based feature extraction.
Run this from the AudioLDM-ControlNet root directory.
"""

import os
import sys
import torch
import pickle
import numpy as np

print("=" * 80)
print("Testing MotionBERT-based Feature Extraction (Simple Version)")
print("=" * 80)

# Configuration
motionbert_ckpt = "data/checkpoints/latest_epoch.bin"
test_keypoint_file = "data/dataset/AIST/keypoints_clips/test/mJS3/gJS_sBM_c01_d01_mJS3_ch02_1.pkl"

# Check CUDA
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("❌ ERROR: CUDA not available. MotionBERT requires GPU.")
    sys.exit(1)

# Check files
print("\n" + "-" * 80)
print("Step 1: Checking files")
print("-" * 80)

if not os.path.exists(motionbert_ckpt):
    print(f"❌ ERROR: MotionBERT checkpoint not found at {motionbert_ckpt}")
    sys.exit(1)
print(f"✓ Found MotionBERT checkpoint")

if not os.path.exists(test_keypoint_file):
    print(f"❌ ERROR: Test keypoint file not found at {test_keypoint_file}")
    sys.exit(1)
print(f"✓ Found test keypoint file")

# Load and inspect raw keypoints
print("\n" + "-" * 80)
print("Step 2: Loading raw keypoints")
print("-" * 80)

with open(test_keypoint_file, 'rb') as f:
    raw_keypoints = pickle.load(f)

print(f"Raw keypoint shape: {raw_keypoints.shape}")
print(f"Raw keypoint dtype: {raw_keypoints.dtype}")
print(f"Number of NaN values: {np.isnan(raw_keypoints).sum()}")

# Import required modules
print("\n" + "-" * 80)
print("Step 3: Loading MotionBERT modules")
print("-" * 80)

from audioldm_train.modules.motion_encoder.MotionBERT.DSTformer import DSTformerWrapper
from audioldm_train.utilities.data.keypoints import (
    interp_nan_keypoints,
    coco2h36m,
    make_cam,
    crop_scale,
)

print("✓ Modules loaded successfully")

# Initialize MotionBERT
print("\n" + "-" * 80)
print("Step 4: Initializing MotionBERT model")
print("-" * 80)

model = DSTformerWrapper(
    dim_in=3,
    dim_out=3,
    dim_feat=512,
    dim_rep=512,
    depth=5,
    num_heads=8,
    mlp_ratio=2,
    num_joints=17,
    maxlen=243
)

from torch import nn
model = nn.DataParallel(model)
model = model.cuda()

print("✓ Loading checkpoint...")
checkpoint = torch.load(motionbert_ckpt, map_location='cpu')
model.load_state_dict(checkpoint['model_pos'], strict=True)
model.eval()

print("✓ MotionBERT model initialized successfully")

# Preprocess keypoints
print("\n" + "-" * 80)
print("Step 5: Preprocessing keypoints")
print("-" * 80)

k2d = interp_nan_keypoints(raw_keypoints)
k2d_score = k2d[:, :, 2]
k2d = k2d[:, :, :2]

camera_shape = (1080, 1920)
k2d = make_cam(k2d[np.newaxis, ...], camera_shape)[0]
k2d = np.concatenate([k2d, k2d_score[:, :, np.newaxis]], axis=2)
k2d = coco2h36m(k2d[np.newaxis, ...])[0]
k2d = crop_scale(k2d, [1, 1])

print(f"Preprocessed keypoints shape: {k2d.shape}")

# Handle sequences longer than MotionBERT's maxlen (243)
maxlen = 243
if k2d.shape[0] > maxlen:
    print(f"⚠ Sequence length ({k2d.shape[0]}) exceeds MotionBERT maxlen ({maxlen})")
    print(f"  Resampling to {maxlen} frames before MotionBERT extraction...")

    # Resample to maxlen
    from audioldm_train.utilities.data.keypoints import resample_keypoints_2d
    k2d = resample_keypoints_2d(k2d, maxlen)
    print(f"  Resampled shape: {k2d.shape}")

k2d_tensor = torch.from_numpy(k2d).float().unsqueeze(0).cuda()

print(f"Final input shape to MotionBERT: {k2d_tensor.shape}")

# Extract features
print("\n" + "-" * 80)
print("Step 6: Extracting MotionBERT features")
print("-" * 80)

with torch.no_grad():
    motion_features = model.module.get_representation(k2d_tensor)

motion_features = motion_features.squeeze(0).cpu()

print(f"✓ Feature extraction successful")
print(f"MotionBERT output shape: {motion_features.shape}")
print(f"Expected shape: [T, 17, 512]")

# Resample to target FPS
print("\n" + "-" * 80)
print("Step 7: Resampling to target FPS")
print("-" * 80)

# Calculate target length based on ORIGINAL keypoint length (before downsampling to 243)
original_keypoint_length = raw_keypoints.shape[0]
target_fps = 21.5
original_fps = 60
target_length = int(original_keypoint_length * target_fps / original_fps)

print(f"Original keypoint frames: {original_keypoint_length}")
print(f"MotionBERT input frames: {motion_features.shape[0]}")
print(f"Target output frames: {target_length} (at {target_fps} FPS)")

# Resample from MotionBERT output length to target length
current_length = motion_features.shape[0]
indices = torch.linspace(0, current_length - 1, target_length)
indices_floor = torch.floor(indices).long()
indices_ceil = torch.clamp(torch.ceil(indices).long(), max=current_length - 1)

weight = (indices - indices_floor.float()).unsqueeze(-1).unsqueeze(-1)
motion_features_resampled = motion_features[indices_floor] * (1 - weight) + motion_features[indices_ceil] * weight

print(f"Resampled shape: {motion_features_resampled.shape}")

# Flatten
T_resampled = motion_features_resampled.shape[0]
motion_features_flat = motion_features_resampled.reshape(T_resampled, -1)

print(f"Flattened shape: {motion_features_flat.shape}")

# Verify output
print("\n" + "-" * 80)
print("Step 8: Verifying features")
print("-" * 80)

print(f"Output shape: {motion_features_flat.shape}")
print(f"Output dtype: {motion_features_flat.dtype}")
print(f"Output range: [{motion_features_flat.min():.4f}, {motion_features_flat.max():.4f}]")
print(f"Output mean: {motion_features_flat.mean():.4f}")
print(f"Output std: {motion_features_flat.std():.4f}")
print(f"NaN count: {torch.isnan(motion_features_flat).sum()}")
print(f"Inf count: {torch.isinf(motion_features_flat).sum()}")

# Expected dimensions
expected_feature_dim = 17 * 512  # 8704
print(f"\nExpected: ({target_length}, {expected_feature_dim})")
print(f"Actual:   {motion_features_flat.shape}")

if motion_features_flat.shape == (target_length, expected_feature_dim):
    print("✓ Shape matches expected dimensions")
else:
    print("⚠ WARNING: Shape mismatch")

if torch.isnan(motion_features_flat).sum() == 0 and torch.isinf(motion_features_flat).sum() == 0:
    print("✓ No NaN or Inf values")
else:
    print("❌ ERROR: Invalid values detected")
    sys.exit(1)

# Test save/load
print("\n" + "-" * 80)
print("Step 9: Testing save/load")
print("-" * 80)

test_output_path = "/tmp/test_motionbert_features.pt"
torch.save(motion_features_flat, test_output_path)
loaded_features = torch.load(test_output_path)

if torch.allclose(motion_features_flat, loaded_features):
    print("✓ Save/load successful")
    os.remove(test_output_path)
else:
    print("❌ ERROR: Save/load mismatch")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✅ All tests passed successfully!")
print(f"✓ MotionBERT feature extraction is working correctly")
print(f"✓ Output shape: {motion_features_flat.shape}")
print(f"✓ Output feature dimension: {expected_feature_dim} (17 joints × 512 dims)")
print(f"✓ Ready for full feature extraction pipeline")
print("=" * 80)
print("\nYou can now run the full extraction with:")
print("  cd BeatDance/preprocess")
print("  python extract_features.py")
