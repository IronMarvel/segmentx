#!/usr/bin/env python3
"""Command-line entry point for SegmentX."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from src.evaluation import load_binary_mask
from src.pipeline import SUPPORTED_METHODS, apply_ground_truth, run_method, save_artifacts, save_comparison
from src.preprocessing import load_image


def parse_seed(value: str) -> tuple[int, int]:
    try:
        x, y = value.split(",")
        return int(x), int(y)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seed must use the form x,y, e.g. 120,180") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SegmentX: multi-method image segmentation and object extraction toolkit.",
    )
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument(
        "--method",
        choices=[*SUPPORTED_METHODS, "all", "compare"],
        default="compare",
        help="Segmentation method or all/compare",
    )
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--ground-truth", help="Optional binary ground-truth mask for metrics")
    parser.add_argument("--clusters", type=int, default=4, help="K-Means cluster count")
    parser.add_argument("--seed", type=parse_seed, help="Region-growing seed as x,y")
    parser.add_argument("--region-threshold", type=int, default=18)
    parser.add_argument("--canny-low", type=int, default=70)
    parser.add_argument("--canny-high", type=int, default=160)
    parser.add_argument("--mean-shift-spatial", type=int, default=16)
    parser.add_argument("--mean-shift-color", type=int, default=24)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.clusters < 2:
        parser.error("--clusters must be at least 2")

    image = load_image(args.input)
    gt = load_binary_mask(args.ground_truth) if args.ground_truth else None
    methods = list(SUPPORTED_METHODS) if args.method in {"all", "compare"} else [args.method]

    stem = Path(args.input).stem
    results = []

    print("SegmentX")
    print(f"Input : {args.input}")
    print(f"Shape : {image.shape[1]} x {image.shape[0]}")
    print(f"Mode  : {args.method}")
    print()

    for method in methods:
        artifact = run_method(
            image,
            method,
            clusters=args.clusters,
            seed_xy=args.seed,
            region_threshold=args.region_threshold,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            mean_shift_spatial=args.mean_shift_spatial,
            mean_shift_color=args.mean_shift_color,
        )
        if gt is not None:
            artifact = apply_ground_truth(artifact, gt)
        save_artifacts(artifact, args.output, stem)
        results.append(artifact.result)
        r = artifact.result
        metrics = f"IoU={r.iou:.4f}, Dice={r.dice:.4f}, Acc={r.pixel_accuracy:.4f}" if r.iou is not None else "No ground truth"
        print(f"[{method:15}] time={r.processing_time_sec:.4f}s | objects={r.num_objects:3d} | foreground={r.foreground_ratio:.3f} | {metrics}")

    if len(results) > 1:
        save_comparison(results, args.output, stem)
        print(f"\nComparison saved under: {args.output}/comparisons")

    print(f"Results saved under : {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, IOError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
