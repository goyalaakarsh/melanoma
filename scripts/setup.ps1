param(
  [string]$PythonExe = "python",
  [string]$Req = "requirements.txt"
)

Write-Host "[setup] Creating venv if missing..."
if (-not (Test-Path -Path "venv")) {
  & $PythonExe -m venv venv
}

Write-Host "[setup] Upgrading pip..."
& .\venv\Scripts\python -m pip install --upgrade pip

Write-Host "[setup] Installing requirements..."
& .\venv\Scripts\python -m pip install -r $Req

Write-Host "[setup] Done. Activate with: .\\venv\\Scripts\\Activate.ps1"




