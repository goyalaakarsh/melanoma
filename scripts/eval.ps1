param(
  [string]$Gt = "melanoma_dip_engine/data/val/masks",
  [string]$Pred = "outputs/preds",
  [string]$Out = "models"
)

& .\venv\Scripts\python -m melanoma_dip_engine.run_eval --gt $Gt --pred $Pred --out $Out

Write-Host "[eval] Reports written to $Out"




