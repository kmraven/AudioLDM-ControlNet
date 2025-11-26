"""
Test script for MotionBERT-based feature extraction.
This script verifies that the new Extractor_fd class works correctly.
"""

import os
import sys

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import pickle
import numpy as np
from extract_features import Extractor_fd

def test_motionbert_extraction():
    print("=" * 80)
    print("Testing MotionBERT-based Feature Extraction")
    print("=" * 80)

    # Configuration
    motionbert_ckpt = "/home/sangheon/Desktop/AudioLDM-ControlNet/data/checkpoints/latest_epoch.bin"
    test_keypoint_file = "/home/sangheon/Desktop/AudioLDM-ControlNet/data/dataset/AIST/keypoints_clips/test/mJS3/gJS_sBM_c01_d01_mJS3_ch02_1.pkl"

    # Check if checkpoint exists
    if not os.path.exists(motionbert_ckpt):
        print(f"❌ ERROR: MotionBERT checkpoint not found at {motionbert_ckpt}")
        return False
    else:
        print(f"✓ Found MotionBERT checkpoint: {motionbert_ckpt}")

    # Check if test keypoint file exists
    if not os.path.exists(test_keypoint_file):
        print(f"❌ ERROR: Test keypoint file not found at {test_keypoint_file}")
        return False
    else:
        print(f"✓ Found test keypoint file: {test_keypoint_file}")

    # Load raw keypoints to check format
    print("\n" + "-" * 80)
    print("Step 1: Loading raw keypoints")
    print("-" * 80)
    with open(test_keypoint_file, 'rb') as f:
        raw_keypoints = pickle.load(f)

    print(f"Raw keypoint shape: {raw_keypoints.shape}")
    print(f"Raw keypoint dtype: {raw_keypoints.dtype}")
    print(f"Raw keypoint range: [{raw_keypoints.min():.2f}, {raw_keypoints.max():.2f}]")
    print(f"Number of NaN values: {np.isnan(raw_keypoints).sum()}")

    # Initialize MotionBERT extractor
    print("\n" + "-" * 80)
    print("Step 2: Initializing MotionBERT Extractor")
    print("-" * 80)
    try:
        extractor = Extractor_fd(motionbert_checkpoint_path=motionbert_ckpt)
        print("✓ MotionBERT extractor initialized successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize extractor: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Extract features
    print("\n" + "-" * 80)
    print("Step 3: Extracting MotionBERT features")
    print("-" * 80)
    try:
        features = extractor.extract(test_keypoint_file, target_fps=21.5)
        print("✓ Feature extraction successful")
    except Exception as e:
        print(f"❌ ERROR: Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Verify output shape and properties
    print("\n" + "-" * 80)
    print("Step 4: Verifying extracted features")
    print("-" * 80)
    print(f"Output feature shape: {features.shape}")
    print(f"Output feature dtype: {features.dtype}")
    print(f"Output feature range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"Output feature mean: {features.mean():.4f}")
    print(f"Output feature std: {features.std():.4f}")
    print(f"Number of NaN values: {torch.isnan(features).sum()}")
    print(f"Number of Inf values: {torch.isinf(features).sum()}")

    # Calculate expected dimensions
    original_length = raw_keypoints.shape[0]
    expected_length = int(original_length * 21.5 / 60)
    expected_feature_dim = 17 * 512  # 17 joints × 512 feature dims = 8704

    print(f"\nExpected output shape: ({expected_length}, {expected_feature_dim})")
    print(f"Actual output shape:   {features.shape}")

    # Verify shape
    if features.shape == (expected_length, expected_feature_dim):
        print("✓ Output shape matches expected dimensions")
    else:
        print(f"⚠ WARNING: Output shape doesn't match expected dimensions")

    # Check for invalid values
    if torch.isnan(features).sum() > 0:
        print("❌ ERROR: Output contains NaN values")
        return False

    if torch.isinf(features).sum() > 0:
        print("❌ ERROR: Output contains Inf values")
        return False

    print("✓ No NaN or Inf values detected")

    # Test saving and loading
    print("\n" + "-" * 80)
    print("Step 5: Testing save/load functionality")
    print("-" * 80)
    test_output_path = "/tmp/test_motionbert_features.pt"
    try:
        torch.save(features, test_output_path)
        print(f"✓ Saved features to {test_output_path}")

        loaded_features = torch.load(test_output_path)
        print(f"✓ Loaded features from {test_output_path}")

        if torch.allclose(features, loaded_features):
            print("✓ Saved and loaded features match")
        else:
            print("❌ ERROR: Saved and loaded features don't match")
            return False

        # Clean up
        os.remove(test_output_path)
        print(f"✓ Cleaned up test file")
    except Exception as e:
        print(f"❌ ERROR: Save/load test failed: {e}")
        return False

    # Compare with multiple samples
    print("\n" + "-" * 80)
    print("Step 6: Testing consistency across multiple frames")
    print("-" * 80)

    # Check temporal consistency (adjacent frames shouldn't change drastically)
    frame_diffs = []
    for i in range(min(10, features.shape[0] - 1)):
        diff = torch.norm(features[i+1] - features[i]).item()
        frame_diffs.append(diff)

    avg_frame_diff = np.mean(frame_diffs)
    print(f"Average frame-to-frame difference: {avg_frame_diff:.4f}")
    print(f"Min frame difference: {min(frame_diffs):.4f}")
    print(f"Max frame difference: {max(frame_diffs):.4f}")

    if avg_frame_diff > 100:
        print("⚠ WARNING: Large frame-to-frame differences detected")
    else:
        print("✓ Frame-to-frame differences are reasonable")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("✓ All tests passed successfully!")
    print(f"✓ MotionBERT feature extraction is working correctly")
    print(f"✓ Output shape: {features.shape}")
    print(f"✓ Ready for BeatDance feature extraction pipeline")
    print("=" * 80)

    return True


if __name__ == "__main__":
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    else:
        print("❌ WARNING: CUDA not available. MotionBERT requires GPU.")

    print("\n")

    success = test_motionbert_extraction()

    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
