from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .inference_api import segment


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    image = image.copy()
    overlay = image.copy()
    color_arr = np.zeros_like(image)
    color_arr[:, :] = color
    overlay = np.where(mask[..., None] > 0, (1 - alpha) * overlay + alpha * color_arr, overlay).astype(np.uint8)
    return overlay


def run_smoke(val_images_dir: Path, out_dir: Path, weights_path: str | None = None, limit: int = 10) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted([p for p in val_images_dir.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])[:limit]

    for img_path in imgs:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        t0 = perf_counter()
        mask, meta = segment(image, weights_path)
        dt = (perf_counter() - t0) * 1000.0

        overlay = overlay_mask(image, mask)
        out_path = out_dir / f"{img_path.stem}_overlay.jpg"
        cv2.imwrite(str(out_path), overlay)

        print(f"{img_path.name}: ok={meta.get('ok')} reason={meta.get('reason')} time_ms={dt:.1f} saved={out_path.name}")


if __name__ == "__main__":
    val_dir = Path("melanoma_dip_engine/data/val/images")
    out_dir = Path("outputs/smoke")
    run_smoke(val_dir, out_dir, weights_path=None, limit=10)



