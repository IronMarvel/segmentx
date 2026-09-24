"""Convert segmentation results into isolated objects and measurements."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ExtractedObject:
    object_id: int
    area: int
    x: int
    y: int
    width: int
    height: int
    centroid_x: float
    centroid_y: float

    def to_dict(self) -> dict:
        return asdict(self)


def labels_to_foreground(labels: np.ndarray, border_fraction: float = 0.02) -> np.ndarray:
    """Choose the dominant border label as background and mark other labels as foreground."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D integer array")
    h, w = labels.shape
    b = max(1, int(round(min(h, w) * border_fraction)))
    border = np.concatenate([
        labels[:b, :].ravel(),
        labels[-b:, :].ravel(),
        labels[:, :b].ravel(),
        labels[:, -b:].ravel(),
    ])
    values, counts = np.unique(border, return_counts=True)
    background = values[int(np.argmax(counts))]
    return np.where(labels == background, 0, 255).astype(np.uint8)


def clean_mask(mask: np.ndarray, open_kernel: int = 3, close_kernel: int = 7) -> np.ndarray:
    """Perform light morphological cleanup."""
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if open_kernel >= 3:
        ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko, iterations=1)
    if close_kernel >= 3:
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=1)
    return mask


def extract_objects(
    image: np.ndarray,
    mask: np.ndarray,
    min_area_ratio: float = 0.002,
    max_objects: int = 30,
) -> tuple[list[ExtractedObject], list[np.ndarray], np.ndarray]:
    """Find connected components and return metadata, cropped objects, and ID mask."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR color image")
    mask = clean_mask(mask)
    h, w = mask.shape
    min_area = max(10, int(h * w * min_area_ratio))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates: list[tuple[int, ExtractedObject]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        cx, cy = map(float, centroids[label_id])
        candidates.append((label_id, ExtractedObject(
            object_id=0,
            area=area,
            x=x,
            y=y,
            width=width,
            height=height,
            centroid_x=cx,
            centroid_y=cy,
        )))

    candidates.sort(key=lambda item: item[1].area, reverse=True)
    candidates = candidates[:max_objects]

    object_crops: list[np.ndarray] = []
    id_mask = np.zeros_like(mask, dtype=np.uint16)
    returned_objects: list[ExtractedObject] = []
    for object_id, (label_id, obj) in enumerate(candidates, start=1):
        obj.object_id = object_id
        comp = labels == label_id
        id_mask[comp] = object_id
        x, y, bw, bh = obj.x, obj.y, obj.width, obj.height
        crop = image[y:y + bh, x:x + bw].copy()
        crop[~comp[y:y + bh, x:x + bw]] = 0
        object_crops.append(crop)
        returned_objects.append(obj)

    return returned_objects, object_crops, id_mask


def draw_bounding_boxes(image: np.ndarray, objects: list[ExtractedObject]) -> np.ndarray:
    """Annotate an image with object bounding boxes and IDs."""
    out = image.copy()
    for obj in objects:
        cv2.rectangle(out, (obj.x, obj.y), (obj.x + obj.width - 1, obj.y + obj.height - 1), (0, 255, 0), 2)
        cv2.putText(out, f"Object {obj.object_id}", (obj.x, max(18, obj.y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return out
