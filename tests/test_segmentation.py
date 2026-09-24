import numpy as np
import cv2

from src.edge_segmentation import canny_edges
from src.kmeans import segment_kmeans
from src.mean_shift import mean_shift_segment
from src.region_growing import grow_region


def synthetic_image():
    image = np.full((80, 100, 3), 235, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (75, 60), (50, 180, 70), -1)
    return image


def test_kmeans_labels_are_integer_regions():
    labels = segment_kmeans(synthetic_image(), clusters=3)
    assert labels.shape == (80, 100)
    assert labels.dtype == np.int32
    assert len(np.unique(labels)) >= 2


def test_region_growing_finds_seed_region():
    gray = cv2.cvtColor(synthetic_image(), cv2.COLOR_BGR2GRAY)
    mask = grow_region(gray, (40, 40), threshold=25)
    assert mask[40, 40] == 255
    assert mask.sum() > 0


def test_canny_finds_edges():
    gray = cv2.cvtColor(synthetic_image(), cv2.COLOR_BGR2GRAY)
    edges = canny_edges(gray)
    assert edges.shape == gray.shape
    assert int((edges > 0).sum()) > 0


def test_mean_shift_preserves_image_dimensions():
    shifted, labels = mean_shift_segment(synthetic_image(), spatial_radius=8, color_radius=16)
    assert shifted.shape == (80, 100, 3)
    assert labels.shape == (80, 100)


def test_hybrid_segmentation_returns_binary_mask():
    from src.hybrid_segmentation import segment_hybrid

    image = np.zeros((80, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (75, 60), (180, 40, 40), -1)
    mask, visual = segment_hybrid(image, clusters=3)
    assert mask.shape == image.shape[:2]
    assert visual.shape == image.shape
    assert set(np.unique(mask)).issubset({0, 255})
