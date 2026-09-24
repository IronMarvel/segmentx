"""Edge-based segmentation using Canny and contour filling."""
from __future__ import annotations

import cv2
import numpy as np


def canny_edges(gray: np.ndarray, low: int = 70, high: int = 160) -> np.ndarray:
    """Return a binary Canny edge image."""
    if gray.ndim != 2:
        raise ValueError("Canny expects a grayscale image")
    return cv2.Canny(gray, low, high)


def edge_to_regions(edges: np.ndarray, close_kernel: int = 5, min_contour_area: float = 50.0) -> np.ndarray:
    """Close small gaps and fill closed contours to obtain a foreground mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    valid = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
    if valid:
        cv2.drawContours(mask, valid, -1, 255, thickness=cv2.FILLED)
    return mask


def segment_edges(gray: np.ndarray, low: int = 70, high: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Return edge image and a contour-based foreground segmentation mask."""
    edges = canny_edges(gray, low, high)
    mask = edge_to_regions(edges)
    return edges, mask
