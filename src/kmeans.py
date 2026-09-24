"""K-Means image segmentation."""
from __future__ import annotations

import cv2
import numpy as np


def segment_kmeans(image: np.ndarray, clusters: int = 4, attempts: int = 10, seed: int = 42) -> np.ndarray:
    """Segment an image into ``clusters`` color regions and return integer labels."""
    if clusters < 2:
        raise ValueError("clusters must be at least 2")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("K-Means segmentation expects a BGR color image")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(
        pixels,
        clusters,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )
    _ = centers
    return labels.reshape(h, w).astype(np.int32)


def labels_to_color(labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Visualize integer labels with a deterministic color palette."""
    labels = np.asarray(labels)
    n = int(labels.max()) + 1 if labels.size else 1
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    return palette[labels]
