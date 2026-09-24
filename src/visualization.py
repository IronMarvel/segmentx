"""Visualization and artifact-saving functions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image.copy()
    color = np.zeros_like(image)
    color[:, :, 1] = 255
    m = mask > 0
    out[m] = cv2.addWeighted(image[m], 1 - alpha, color[m], alpha, 0)
    return out


def labels_to_color(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    labels = labels.astype(np.int32)
    n = int(labels.max()) + 1 if labels.size else 1
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 255, size=(n, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette[labels]
