#!/usr/bin/env python3
from audioldm_train.metrics.eval import ControlNetEvaluationHelper
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, help="Path to generated audio folder")
    parser.add_argument("--groundtruth", required=True, help="Path to ground truth audio folder")
    parser.add_argument("--sr", type=int, default=16000, help="Sampling rate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files")
    parser.add_argument("--recalculate", action="store_true", help="Recalculate FAD embeddings")

    args = parser.parse_args()

    evaluator = ControlNetEvaluationHelper(sampling_rate=args.sr)

    metrics = evaluator.calculate_metrics(
        generate_files_path=args.generated,
        groundtruth_path=args.groundtruth,
        same_name=True,
        limit_num=args.limit,
        recalculate=args.recalculate
    )

    print("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.7f}")