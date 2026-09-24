"""Core orchestration for the SegmentX toolkit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .edge_segmentation import segment_edges
from .evaluation import MethodResult, binary_metrics, time_call
from .kmeans import labels_to_color as kmeans_labels_to_color
from .kmeans import segment_kmeans
from .hybrid_segmentation import segment_hybrid
from .mean_shift import mean_shift_segment
from .object_extraction import draw_bounding_boxes, extract_objects, labels_to_foreground
from .preprocessing import PreprocessConfig, load_image, preprocess
from .region_growing import grow_region
from .visualization import labels_to_color, save_csv, save_image, save_json


SUPPORTED_METHODS = ("kmeans", "region-growing", "edge", "mean-shift", "hybrid")


@dataclass
class RunArtifacts:
    method: str
    label_image: np.ndarray | None
    segmentation_mask: np.ndarray
    visualization: np.ndarray
    annotated: np.ndarray
    objects: list
    object_crops: list[np.ndarray]
    result: MethodResult


def _region_seed(gray: np.ndarray, provided: tuple[int, int] | None) -> tuple[int, int]:
    if provided is not None:
        return provided
    # Safe default: image center. The CLI exposes --seed for images where center is not a useful seed.
    h, w = gray.shape
    return w // 2, h // 2


def run_method(
    image: np.ndarray,
    method: str,
    *,
    clusters: int = 4,
    seed_xy: tuple[int, int] | None = None,
    region_threshold: int = 18,
    canny_low: int = 70,
    canny_high: int = 160,
    mean_shift_spatial: int = 16,
    mean_shift_color: int = 24,
) -> RunArtifacts:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported method: {method}. Choose from {', '.join(SUPPORTED_METHODS)}")

    pre = preprocess(image)
    base = pre["denoised"]
    gray = pre["gray"]

    if method == "kmeans":
        (labels, elapsed) = time_call(segment_kmeans, base, clusters)
        mask = labels_to_foreground(labels)
        visual = kmeans_labels_to_color(labels)
    elif method == "region-growing":
        seed = _region_seed(gray, seed_xy)
        (mask, elapsed) = time_call(grow_region, gray, seed, region_threshold)
        labels = None
        visual = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif method == "edge":
        (edge_pair, elapsed) = time_call(segment_edges, gray, canny_low, canny_high)
        edges, mask = edge_pair
        labels = None
        visual = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif method == "mean-shift":
        (shifted_labels, elapsed) = time_call(
            mean_shift_segment,
            base,
            mean_shift_spatial,
            mean_shift_color,
        )
        shifted, labels = shifted_labels
        mask = labels_to_foreground(labels)
        visual = cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)
        visual = cv2.cvtColor(visual, cv2.COLOR_RGB2BGR)
    else:
        (hybrid_pair, elapsed) = time_call(
            segment_hybrid,
            base,
            clusters,
            canny_low,
            canny_high,
        )
        mask, visual = hybrid_pair
        labels = None

    objects, crops, id_mask = extract_objects(base, mask)
    annotated = draw_bounding_boxes(base, objects)
    result = MethodResult(
        method=method,
        processing_time_sec=elapsed,
        num_objects=len(objects),
        foreground_ratio=float((mask > 0).mean()),
    )
    return RunArtifacts(
        method=method,
        label_image=labels,
        segmentation_mask=mask,
        visualization=visual,
        annotated=annotated,
        objects=objects,
        object_crops=crops,
        result=result,
    )


def apply_ground_truth(artifact: RunArtifacts, gt_mask: np.ndarray) -> RunArtifacts:
    metrics = binary_metrics(artifact.segmentation_mask, gt_mask)
    artifact.result.iou = metrics["iou"]
    artifact.result.dice = metrics["dice"]
    artifact.result.pixel_accuracy = metrics["pixel_accuracy"]
    return artifact


def save_artifacts(artifact: RunArtifacts, out_dir: str | Path, stem: str) -> None:
    out_dir = Path(out_dir)
    masks = out_dir / "masks"
    objects_dir = out_dir / "objects" / artifact.method
    comparisons = out_dir / "comparisons"

    save_image(masks / f"{stem}_{artifact.method}_mask.png", artifact.segmentation_mask)
    save_image(comparisons / f"{stem}_{artifact.method}_segmentation.png", artifact.visualization)
    save_image(comparisons / f"{stem}_{artifact.method}_annotated.png", artifact.annotated)

    metadata = [obj.to_dict() for obj in artifact.objects]
    save_json(objects_dir / "objects.json", {"method": artifact.method, "objects": metadata})
    for idx, crop in enumerate(artifact.object_crops, start=1):
        save_image(objects_dir / f"object_{idx:02d}.png", crop)


def save_comparison(results: list[MethodResult], out_dir: str | Path, stem: str) -> None:
    rows = [r.to_dict() for r in results]
    save_csv(Path(out_dir) / "comparisons" / f"{stem}_comparison.csv", rows)
    save_json(Path(out_dir) / "reports" / f"{stem}_comparison.json", {"results": rows})
