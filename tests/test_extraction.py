import numpy as np
import cv2

from src.object_extraction import extract_objects


def test_extracts_connected_object():
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    mask = np.zeros((100, 120), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (50, 55), 255, -1)
    objects, crops, id_mask = extract_objects(image, mask, min_area_ratio=0.001)
    assert len(objects) == 1
    assert len(crops) == 1
    assert objects[0].area > 500
    assert id_mask.max() == 1
