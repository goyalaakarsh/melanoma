param(
  [string]$Root = "melanoma_dip_engine",
  [string]$Ph2 = "",
  [int]$ValSplit = 20
)

if ($Ph2 -eq "") {
  Write-Host "Usage: scripts/organize_ph2.ps1 -Ph2 C:\\path\\to\\PH2Dataset"
  exit 1
}

Set-Location $Root
Write-Host "[PH2] Organizing from: $Ph2"

& ..\venv\Scripts\python organize_ph2_dataset.py --source "$Ph2" --val_split $ValSplit

Write-Host "[PH2] Done"







