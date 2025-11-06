from __future__ import annotations

from pathlib import Path


def export_onnx(weights_path: str, out_path: str, input_size=(3, 512, 512)) -> None:
    wp = Path(weights_path)
    if not wp.exists():
        print(f"[export] Weights not found: {wp}. Train first, then retry.")
        return
    print("[export] ONNX export stub. Integrate Detectron2 ONNX export post-training.")


def export_torchscript(weights_path: str, out_path: str) -> None:
    wp = Path(weights_path)
    if not wp.exists():
        print(f"[export] Weights not found: {wp}. Train first, then retry.")
        return
    print("[export] TorchScript export stub. Integrate post-training serialization here.")


if __name__ == "__main__":
    export_onnx("models/final_lesion_segmenter.pth", "models/final_lesion_segmenter.onnx")
    export_torchscript("models/final_lesion_segmenter.pth", "models/final_lesion_segmenter.ts")




