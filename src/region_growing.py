"""Seeded region-growing segmentation."""
from __future__ import annotations

from collections import deque

import numpy as np


def grow_region(gray: np.ndarray, seed_xy: tuple[int, int], threshold: int = 18, connectivity: int = 8) -> np.ndarray:
    """Grow a region from (x, y) using intensity similarity to the current region mean."""
    if gray.ndim != 2:
        raise ValueError("Region growing expects a grayscale image")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    x0, y0 = map(int, seed_xy)
    h, w = gray.shape
    if not (0 <= x0 < w and 0 <= y0 < h):
        raise ValueError(f"Seed {seed_xy} is outside image bounds {(w, h)}")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx, dy) != (0, 0)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    q: deque[tuple[int, int]] = deque([(x0, y0)])
    visited[y0, x0] = True
    mask[y0, x0] = 255

    mean_value = float(gray[y0, x0])
    count = 1

    while q:
        x, y = q.popleft()
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= w or ny < 0 or ny >= h or visited[ny, nx]:
                continue
            visited[ny, nx] = True
            if abs(float(gray[ny, nx]) - mean_value) <= threshold:
                mask[ny, nx] = 255
                q.append((nx, ny))
                count += 1
                mean_value += (float(gray[ny, nx]) - mean_value) / count

    return mask
