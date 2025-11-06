from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _ensure_3ch(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def segment(image: np.ndarray, weights_path: str | None = None) -> Tuple[np.ndarray, dict]:
    """
    Model-agnostic segmentation API.
    - If weights are unavailable, returns an empty mask with metadata.
    - Later, wire this to Detectron2 Mask R-CNN using the trained weights.
    """
    image = _ensure_3ch(image)
    h, w = image.shape[:2]

    weights_path = str(weights_path) if weights_path else None
    if not weights_path or not Path(weights_path).exists():
        empty = np.zeros((h, w), dtype=np.uint8)
        meta = {
            "ok": False,
            "reason": "weights_not_found",
            "weights_path": weights_path,
            "mask_source": "empty",
        }
        return empty, meta

    # Placeholder: real model loading/inference will replace this block
    # For now, return empty to keep the API stable pre-training
    empty = np.zeros((h, w), dtype=np.uint8)
    meta = {
        "ok": True,
        "reason": "stub_inference",
        "weights_path": weights_path,
        "mask_source": "stub",
    }
    return empty, meta



