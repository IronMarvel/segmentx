import numpy as np

from src.preprocessing import preprocess


def test_preprocess_returns_expected_keys_and_shapes():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result = preprocess(image)
    assert set(result) == {"original", "resized", "denoised", "gray"}
    assert result["gray"].shape == result["denoised"].shape[:2]
    assert result["gray"].dtype == np.uint8
