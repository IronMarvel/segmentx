"""Quantitative evaluation helpers for SegmentX."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MethodResult:
    method: str
    processing_time_sec: float
    num_objects: int
    foreground_ratio: float
    iou: float | None = None
    dice: float | None = None
    pixel_accuracy: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def binary_metrics(pred_mask: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Compute foreground IoU, Dice, and pixel accuracy for binary masks."""
    pred = pred_mask > 0
    gt = ground_truth > 0
    if pred.shape != gt.shape:
        gt = cv2.resize(gt.astype(np.uint8), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(intersection / union) if union else 1.0
    denom = pred.sum() + gt.sum()
    dice = float(2 * intersection / denom) if denom else 1.0
    accuracy = float((pred == gt).mean())
    return {"iou": iou, "dice": dice, "pixel_accuracy": accuracy}


def load_binary_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth mask not found: {path}")
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not decode ground-truth mask: {path}")
    return mask
