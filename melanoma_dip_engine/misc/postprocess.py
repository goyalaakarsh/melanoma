from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def remove_small_objects(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    if mask.dtype != np.uint8:
        work = (mask > 0).astype(np.uint8) * 255
    else:
        work = (mask > 0).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    out = np.zeros_like(work)
    for i in range(1, num):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    work = (mask > 0).astype(np.uint8)
    h, w = work.shape
    flood = work.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    inv = 1 - flood
    filled = np.logical_or(work == 1, inv == 1)
    return (filled.astype(np.uint8) * 255)


def smooth_contours(mask: np.ndarray, kernel_size: int = 5, iters: int = 1) -> np.ndarray:
    work = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel, iterations=iters)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iters)
    return closed


def threshold_probability(prob: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return (prob >= thr).astype(np.uint8) * 255


def full_postprocess(mask: np.ndarray, min_area: int = 100, kernel_size: int = 5, iters: int = 1) -> np.ndarray:
    m = remove_small_objects(mask, min_area=min_area)
    m = fill_holes(m)
    m = smooth_contours(m, kernel_size=kernel_size, iters=iters)
    return m



