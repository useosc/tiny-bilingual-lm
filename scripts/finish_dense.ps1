# Wait for the dense run to exit, then evaluate, export, and publish to Pages.
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

Write-Host "exporting + publishing" -ForegroundColor Cyan
pwsh scripts/deploy_pages.ps1

Write-Host "DENSE_FINISH_DONE" -ForegroundColor Green
