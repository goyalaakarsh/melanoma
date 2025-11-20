from __future__ import annotations

import numpy as np

from melanoma_dip_engine.metrics import compute_binary_metrics
from melanoma_dip_engine.postprocess import remove_small_objects, fill_holes, smooth_contours


def test_metrics_identity():
    m = np.zeros((32, 32), dtype=np.uint8)
    m[8:24, 8:24] = 255
    mm = m.copy()
    a = compute_binary_metrics(m > 0, mm > 0)
    assert a.dice == 1.0
    assert a.iou == 1.0
    assert a.precision == 1.0
    assert a.recall == 1.0


def test_remove_small_objects():
    m = np.zeros((32, 32), dtype=np.uint8)
    m[0:2, 0:2] = 255  # tiny
    m[8:24, 8:24] = 255  # big
    out = remove_small_objects(m, min_area=50)
    assert out.sum() > 0
    assert out[1, 1] == 0  # removed tiny object


def test_fill_holes():
    m = np.zeros((32, 32), dtype=np.uint8)
    m[4:28, 4:28] = 255
    m[12:20, 12:20] = 0  # hole
    out = fill_holes(m)
    assert out[16, 16] == 255


def test_smooth_contours():
    m = np.zeros((64, 64), dtype=np.uint8)
    m[10:54, 10:54] = 255
    out = smooth_contours(m, kernel_size=5, iters=1)
    # Basic sanity: still non-empty and same shape
    assert out.shape == m.shape
    assert out.sum() > 0







