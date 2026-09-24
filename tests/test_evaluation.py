import numpy as np

from src.evaluation import binary_metrics


def test_binary_metrics_perfect_match():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    metrics = binary_metrics(mask, mask)
    assert metrics["iou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0
