#!/usr/bin/env python3
"""Generate a deterministic synthetic image and binary foreground mask for demos/tests."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "input" / "demo_objects.png"
GT = ROOT / "data" / "ground_truth" / "demo_objects_mask.png"


def main() -> None:
    h, w = 480, 720
    rng = np.random.default_rng(7)
    image = np.full((h, w, 3), (235, 235, 235), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Soft background gradient.
    for y in range(h):
        value = 220 + int(25 * y / (h - 1))
        image[y, :, :] = (value, value, value)

    # Colored objects.
    cv2.circle(image, (155, 150), 75, (55, 75, 220), -1)
    cv2.rectangle(image, (315, 90), (500, 235), (60, 180, 70), -1)
    pts = np.array([[520, 340], [630, 250], [675, 390], [555, 425]], dtype=np.int32)
    cv2.fillPoly(image, [pts], (215, 85, 55))
    cv2.ellipse(image, (255, 350), (95, 55), -15, 0, 360, (70, 150, 200), -1)

    # Ground-truth foreground masks matching those objects.
    cv2.circle(mask, (155, 150), 75, 255, -1)
    cv2.rectangle(mask, (315, 90), (500, 235), 255, -1)
    cv2.fillPoly(mask, [pts], 255)
    cv2.ellipse(mask, (255, 350), (95, 55), -15, 0, 360, 255, -1)

    # Add very light deterministic noise so preprocessing has something meaningful to do.
    noise = rng.normal(0, 2.5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    INPUT.parent.mkdir(parents=True, exist_ok=True)
    GT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(INPUT), image)
    cv2.imwrite(str(GT), mask)
    print(INPUT)
    print(GT)


if __name__ == "__main__":
    main()
