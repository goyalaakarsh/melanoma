from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import csv
import json


@dataclass
class BinaryMetrics:
    dice: float
    iou: float
    precision: float
    recall: float


def _to_bool_mask(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.bool_:
        return arr > 0
    return arr


def compute_binary_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> BinaryMetrics:
    gt = _to_bool_mask(gt_mask)
    pr = _to_bool_mask(pred_mask)

    tp = float(np.logical_and(gt, pr).sum())
    fp = float(np.logical_and(np.logical_not(gt), pr).sum())
    fn = float(np.logical_and(gt, np.logical_not(pr)).sum())

    denom_dice = (2.0 * tp + fp + fn)
    denom_iou = (tp + fp + fn)

    dice = (2.0 * tp) / denom_dice if denom_dice > 0 else 1.0
    iou = tp / denom_iou if denom_iou > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    return BinaryMetrics(dice=dice, iou=iou, precision=precision, recall=recall)


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    _, m = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return m > 0


def evaluate_directory(gt_dir: Path, pred_dir: Path) -> Tuple[List[Dict], Dict[str, float]]:
    rows: List[Dict] = []
    metrics_accum: List[BinaryMetrics] = []

    for gt_path in sorted(gt_dir.glob("*")):
        if gt_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue

        base = gt_path.stem
        # Accept both "_mask" suffix or exact name matches for predictions
        candidates = [
            pred_dir / f"{base}.png",
            pred_dir / f"{base}.jpg",
            pred_dir / f"{base}.jpeg",
            pred_dir / f"{base}_mask.png",
            pred_dir / f"{base}_mask.jpg",
            pred_dir / f"{base}_mask.jpeg",
        ]
        pred_path = next((p for p in candidates if p.exists()), None)
        if pred_path is None:
            continue

        gt = read_mask(gt_path)
        pr = read_mask(pred_path)

        # Resize pred to gt if needed
        if gt.shape != pr.shape:
            pr = cv2.resize((pr * 255).astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

        m = compute_binary_metrics(gt, pr)
        metrics_accum.append(m)
        rows.append({"id": base, **asdict(m)})

    if metrics_accum:
        avg = {
            "dice": float(np.mean([m.dice for m in metrics_accum])),
            "iou": float(np.mean([m.iou for m in metrics_accum])),
            "precision": float(np.mean([m.precision for m in metrics_accum])),
            "recall": float(np.mean([m.recall for m in metrics_accum])),
        }
    else:
        avg = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    return rows, avg


def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = ["id", "dice", "iou", "precision", "recall"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def save_json(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    gt = Path("melanoma_dip_engine/data/val/masks")
    pr = Path("outputs/preds")
    pr.mkdir(parents=True, exist_ok=True)
    rows, avg = evaluate_directory(gt, pr)
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    save_csv(rows, out_dir / "eval_report.csv")
    save_json({"average": avg, "count": len(rows)}, out_dir / "eval_report.json")
    print("Saved metrics to models/eval_report.{csv,json}")



