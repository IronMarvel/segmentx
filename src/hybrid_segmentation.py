"""Boundary-aware hybrid segmentation for SegmentX.

This module is a project-specific extension: it combines color clustering
with boundary evidence instead of treating the segmentation algorithms as
independent outputs.
"""
from __future__ import annotations

import cv2
import numpy as np

from .edge_segmentation import canny_edges, edge_to_regions
from .kmeans import labels_to_color, segment_kmeans
from .object_extraction import labels_to_foreground


def segment_hybrid(
    image: np.ndarray,
    clusters: int = 4,
    canny_low: int = 70,
    canny_high: int = 160,
    edge_dilate: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce a boundary-aware foreground mask using K-Means + Canny.

    K-Means proposes color-based regions. Canny supplies boundary evidence.
    The edge regions are dilated slightly and used to split/trim the color
    mask, followed by a light morphological cleanup. The function returns
    the refined binary mask and a visualization image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Hybrid segmentation expects a BGR color image")
    if clusters < 2:
        raise ValueError("clusters must be at least 2")
    if edge_dilate < 1 or edge_dilate % 2 == 0:
        raise ValueError("edge_dilate must be a positive odd integer")

    labels = segment_kmeans(image, clusters=clusters)
    color_mask = labels_to_foreground(labels)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny_edges(gray, canny_low, canny_high)
    boundary_regions = edge_to_regions(edges, close_kernel=5, min_contour_area=40.0)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (edge_dilate, edge_dilate)
    )
    expanded_boundaries = cv2.dilate(edges, kernel, iterations=1)

    # Remove pixels that sit directly on strong boundaries, then recover
    # enclosed contour regions. This makes the hybrid result less dependent
    # on the border-label assumption of plain K-Means extraction.
    trimmed = color_mask.copy()
    trimmed[expanded_boundaries > 0] = 0
    hybrid = cv2.bitwise_or(trimmed, boundary_regions)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hybrid = cv2.morphologyEx(hybrid, cv2.MORPH_CLOSE, clean_kernel, iterations=1)
    hybrid = cv2.morphologyEx(hybrid, cv2.MORPH_OPEN, clean_kernel, iterations=1)

    # Use the K-Means labels as the visual background and overlay the hybrid
    # foreground in white for an easy side-by-side inspection.
    visual = labels_to_color(labels)
    visual[hybrid == 0] = (0, 0, 0)
    return hybrid.astype(np.uint8), visual
