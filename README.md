# dip

Pretraining‑independent utilities to validate data and evaluate predictions without running training yet.

## Data layout

Place PH2-organized data under:

```
melanoma_dip_engine/data/
  train/
    images/*.jpg
    masks/*.png|jpg ("<id>_mask.*" also supported)
  val/
    images/*.jpg
    masks/*.png|jpg
```

## Quick start

1) Environment setup (Windows PowerShell):

```
scripts/setup.ps1
```

2) Validate dataset pairs and coverage:

```
python -m melanoma_dip_engine.dataset_validator
# writes melanoma_dip_engine/data/dataset_report.json
```

3) Smoke test inference (no weights yet; saves overlays as placeholders):

```
python -m melanoma_dip_engine.smoke_test_inference
# outputs to outputs/smoke/
```

4) Evaluate predictions folder when available:

```
python -m melanoma_dip_engine.run_eval --gt melanoma_dip_engine/data/val/masks --pred outputs/preds --out models
```

5) Write run metadata:

```
python -m melanoma_dip_engine.run_metadata
# writes models/run_metadata.json
```

## Notes

- `inference_api.segment` returns an empty mask until weights are trained; API is stable for later integration.
- `postprocess.py` provides morphology utilities usable on any binary mask.
- Once training finishes, wire `predict.py` to call `inference_api.segment` and update it to load the trained Detectron2 model.

