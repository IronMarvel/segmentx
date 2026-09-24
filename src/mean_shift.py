"""Mean-shift based color segmentation."""
from __future__ import annotations

import cv2
import numpy as np


def mean_shift_segment(
    image: np.ndarray,
    spatial_radius: int = 16,
    color_radius: int = 24,
    max_level: int = 1,
    clusters: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply mean-shift filtering followed by compact color-region labeling.

    The mean-shift stage smooths and shifts pixels toward local modes in combined spatial/color
    space. A small deterministic K-Means labeling step on the mean-shifted colors converts the
    filtered image into explicit region labels for downstream object extraction. This hybrid
    keeps runtime predictable while making the mean-shift contribution directly observable.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Mean-shift segmentation expects a BGR color image")
    if spatial_radius <= 0 or color_radius <= 0 or clusters < 2:
        raise ValueError("spatial_radius, color_radius must be positive and clusters >= 2")

    shifted = cv2.pyrMeanShiftFiltering(
        image,
        sp=int(spatial_radius),
        sr=int(color_radius),
        maxLevel=int(max_level),
    )

    lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.25)
    cv2.setRNGSeed(42)
    _, labels, _ = cv2.kmeans(
        pixels,
        int(clusters),
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    return shifted, labels.reshape(h, w).astype(np.int32)
