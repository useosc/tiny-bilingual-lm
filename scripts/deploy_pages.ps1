# Re-export the final checkpoint and publish it to GitHub Pages (gh-pages branch).
# Usage: pwsh scripts/deploy_pages.ps1 [-Ckpt checkpoints/dense_25m/best.pt]
param(
    [string]$Ckpt = "checkpoints/dense_25m/best.pt",
    [string]$PagesDir = "D:\deepseek\tiny-bilingual-lm-pages"
)
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== export ONNX ==" -ForegroundColor Cyan
python export/onnx_export.py --ckpt $Ckpt --out web/model/model.onnx

Write-Host "== quantise int8 ==" -ForegroundColor Cyan
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('web/model/model.onnx','web/model/model_int8.onnx',weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul','Gemm'])"
Remove-Item web/model/model_test*.onnx -ErrorAction SilentlyContinue

Write-Host "== copy to pages worktree ==" -ForegroundColor Cyan
Copy-Item web/index.html, web/app.js, web/bpe.js -Destination $PagesDir -Force
New-Item -ItemType Directory -Force -Path (Join-Path $PagesDir "model") | Out-Null
Copy-Item web/model/model_int8.onnx, web/model/tokenizer.json, web/model/config.json -Destination (Join-Path $PagesDir "model") -Force

Write-Host "== commit + push gh-pages ==" -ForegroundColor Cyan
Push-Location $PagesDir
$env:GIT_SSH_COMMAND = "ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
git add -A
git -c user.name=useosc -c user.email=992383472@qq.com commit -m "pages: update model to $(Split-Path $Ckpt -Leaf)"
git push origin gh-pages
Pop-Location

Write-Host "published: https://useosc.github.io/tiny-bilingual-lm/" -ForegroundColor Green
