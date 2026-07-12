"""Run a MotionBERT feature-extraction smoke test on one keypoint file."""

import argparse

import torch

from BeatDance.preprocess.extract_features import Extractor_fd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="MotionBERT checkpoint")
    parser.add_argument("--keypoints", required=True, help="COCO keypoint pickle")
    parser.add_argument("--device", default=None)
    parser.add_argument("--target-length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    extractor = Extractor_fd(
        motionbert_ckpt=args.checkpoint,
        device=args.device,
        target_length=args.target_length,
    )
    features = extractor.extract(args.keypoints)
    expected_shape = (args.target_length, 17 * 512)
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected feature shape {tuple(features.shape)}; expected {expected_shape}"
        )
    if not torch.isfinite(features).all():
        raise RuntimeError("MotionBERT features contain NaN or infinity")
    print(f"MotionBERT smoke test passed: shape={tuple(features.shape)}")


if __name__ == "__main__":
    main()
