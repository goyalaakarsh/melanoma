from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class ImageMaskPairReport:
    image_path: str
    mask_path: str | None
    image_shape: Tuple[int, int]
    mask_shape: Tuple[int, int] | None
    has_mask: bool
    dims_match: bool
    mask_area_px: int | None
    mask_coverage_pct: float | None


@dataclass
class SplitSummary:
    split_name: str
    num_images: int
    num_masks: int
    paired: int
    missing_masks: int
    dim_mismatch: int
    mask_area_px_min: int | None
    mask_area_px_max: int | None
    mask_coverage_pct_min: float | None
    mask_coverage_pct_max: float | None


def _pair_paths(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path | None]]:
    pairs: List[Tuple[Path, Path | None]] = []
    for img_path in sorted(images_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        base = img_path.stem
        candidate_masks = [
            masks_dir / f"{base}_mask.png",
            masks_dir / f"{base}_mask.jpg",
            masks_dir / f"{base}_mask.jpeg",
            masks_dir / f"{base}.png",
            masks_dir / f"{base}.jpg",
            masks_dir / f"{base}.jpeg",
        ]
        mask_path = next((p for p in candidate_masks if p.exists()), None)
        pairs.append((img_path, mask_path))
    return pairs


def _analyze_pair(img_path: Path, mask_path: Path | None) -> ImageMaskPairReport:
    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    h, w = image.shape[:2]

    if mask_path is None:
        return ImageMaskPairReport(
            image_path=str(img_path),
            mask_path=None,
            image_shape=(h, w),
            mask_shape=None,
            has_mask=False,
            dims_match=False,
            mask_area_px=None,
            mask_coverage_pct=None,
        )

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ImageMaskPairReport(
            image_path=str(img_path),
            mask_path=str(mask_path),
            image_shape=(h, w),
            mask_shape=None,
            has_mask=False,
            dims_match=False,
            mask_area_px=None,
            mask_coverage_pct=None,
        )

    mh, mw = mask.shape[:2]
    dims_match = (mh == h and mw == w)

    # Binarize and compute coverage
    _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask_area = int(np.count_nonzero(bin_mask))
    mask_coverage = float(mask_area / float(h * w) * 100.0)

    return ImageMaskPairReport(
        image_path=str(img_path),
        mask_path=str(mask_path),
        image_shape=(h, w),
        mask_shape=(mh, mw),
        has_mask=True,
        dims_match=dims_match,
        mask_area_px=mask_area,
        mask_coverage_pct=mask_coverage,
    )


def validate_split(split_name: str, images_dir: Path, masks_dir: Path) -> Tuple[SplitSummary, List[ImageMaskPairReport]]:
    pairs = _pair_paths(images_dir, masks_dir)
    reports: List[ImageMaskPairReport] = []
    for img_path, mask_path in pairs:
        reports.append(_analyze_pair(img_path, mask_path))

    num_images = len(pairs)
    num_masks = sum(1 for r in reports if r.has_mask)
    paired = sum(1 for r in reports if r.has_mask and r.dims_match)
    missing_masks = sum(1 for r in reports if not r.has_mask)
    dim_mismatch = sum(1 for r in reports if r.has_mask and not r.dims_match)

    coverages = [r.mask_coverage_pct for r in reports if r.mask_coverage_pct is not None]
    areas = [r.mask_area_px for r in reports if r.mask_area_px is not None]

    summary = SplitSummary(
        split_name=split_name,
        num_images=num_images,
        num_masks=num_masks,
        paired=paired,
        missing_masks=missing_masks,
        dim_mismatch=dim_mismatch,
        mask_area_px_min=min(areas) if areas else None,
        mask_area_px_max=max(areas) if areas else None,
        mask_coverage_pct_min=min(coverages) if coverages else None,
        mask_coverage_pct_max=max(coverages) if coverages else None,
    )
    return summary, reports


def run_validator(data_root: Path = Path("melanoma_dip_engine") / "data") -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}

    splits = {
        "train": (data_root / "train" / "images", data_root / "train" / "masks"),
        "val": (data_root / "val" / "images", data_root / "val" / "masks"),
    }

    for split_name, (img_dir, msk_dir) in splits.items():
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)
        summary, reports = validate_split(split_name, img_dir, msk_dir)
        out[split_name] = {
            "summary": asdict(summary),
            "samples": [asdict(r) for r in reports[:50]],
        }

    out_dir = data_root
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


if __name__ == "__main__":
    report = run_validator()
    print("Dataset validation completed. Report saved to data/dataset_report.json")



