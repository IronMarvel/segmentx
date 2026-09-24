"""Image loading and preprocessing utilities for SegmentX."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings for reproducible image preprocessing."""

    max_width: int = 1200
    max_height: int = 800
    gaussian_kernel: int = 5
    median_kernel: int = 0
    clahe: bool = True
    clip_limit: float = 2.0
    tile_grid_size: int = 8


def load_image(path: str | Path) -> np.ndarray:
    """Load a color image as BGR uint8, raising a useful error on failure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode the image: {path}")
    return image


def _resize_keep_aspect(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_width / float(w), max_height / float(h))
    if scale == 1.0:
        return image.copy()
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def preprocess(image: np.ndarray, config: Optional[PreprocessConfig] = None) -> dict[str, np.ndarray]:
    """Run the standard preprocessing pipeline and return useful representations."""
    cfg = config or PreprocessConfig()
    if image is None or image.size == 0:
        raise ValueError("Image is empty.")
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    resized = _resize_keep_aspect(image, cfg.max_width, cfg.max_height)
    blurred = cv2.GaussianBlur(resized, (cfg.gaussian_kernel, cfg.gaussian_kernel), 0)

    denoised = blurred
    if cfg.median_kernel and cfg.median_kernel >= 3:
        k = cfg.median_kernel if cfg.median_kernel % 2 == 1 else cfg.median_kernel + 1
        denoised = cv2.medianBlur(denoised, k)

    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    enhanced_gray = gray.copy()
    if cfg.clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clip_limit,
            tileGridSize=(cfg.tile_grid_size, cfg.tile_grid_size),
        )
        enhanced_gray = clahe.apply(gray)

    return {
        "original": image,
        "resized": resized,
        "denoised": denoised,
        "gray": enhanced_gray,
    }
