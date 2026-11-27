# Wait for the dense run to exit, then evaluate, export, and publish to Pages.
# Note: the DSH shell is Windows PowerShell 5.1, so child scripts are launched
# through $PSHOME\powershell.exe (pwsh.exe is not on PATH on this machine).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"

Write-Host "waiting for dense training to finish..." -ForegroundColor Cyan
while ($true) {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*dense_25m*" }
    if (-not $procs) { break }
    Start-Sleep -Seconds 120
}

Write-Host "training exited; evaluating best checkpoint" -ForegroundColor Cyan
python eval/evaluate.py --ckpt checkpoints/dense_25m/best.pt --tag dense_25m --device cuda
if ($LASTEXITCODE -ne 0) { Write-Host "eval failed, continuing to export anyway" -ForegroundColor Yellow }

Write-Host "exporting + publishing" -ForegroundColor Cyan
& "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File scripts\deploy_pages.ps1

Write-Host "DENSE_FINISH_DONE" -ForegroundColor Green
