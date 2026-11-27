# Overnight chain: wait for dense -> eval -> MoE -> eval -> looped -> eval -> export -> deploy.
# Every step logs and continues on failure; a final marker file says DONE.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$log = "logs/overnight.log"
function Say($msg) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}
New-Item -ItemType Directory -Force -Path logs | Out-Null
Remove-Item logs/OVERNIGHT_DONE -ErrorAction SilentlyContinue

Say "waiting for dense training"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -like "*dense_25m*" }) { Start-Sleep -Seconds 120 }
Say "dense finished"

Say "evaluate dense"
python eval/evaluate.py --ckpt checkpoints/dense_25m/best.pt --tag dense_25m --device cuda *>> $log

Say "train MoE (120 min budget)"
python train/train.py --config configs/moe_4e.toml --max-minutes 120 *>> $log
Say "evaluate MoE"
python eval/evaluate.py --ckpt checkpoints/moe_4e/best.pt --tag moe_4e --device cuda *>> $log

Say "train looped (80 min budget)"
python train/train.py --config configs/looped_16m.toml --max-minutes 80 *>> $log
Say "evaluate looped"
python eval/evaluate.py --ckpt checkpoints/looped_16m/best.pt --tag looped_16m --device cuda *>> $log

Say "export + publish dense model"
& "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File scripts\deploy_pages.ps1 *>> $log

Say "plot curves"
python scripts/plot_metrics.py *>> $log

Set-Content -Path logs/OVERNIGHT_DONE -Value "done $(Get-Date -Format s)" -Encoding UTF8
Say "OVERNIGHT_DONE"
