# End-to-end pipeline (Windows PowerShell). Run from the repo root.
# Every step is idempotent: existing artifacts are reused.
$ErrorActionPreference = "Stop"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "== 1. download ==" -ForegroundColor Cyan
if (-not (Test-Path "data/raw/skypile/zh_0000.jsonl")) {
    python data/scripts/download.py --zh-shards 1 --en-shards 1
}

Write-Host "== 2. clean ==" -ForegroundColor Cyan
if (-not (Test-Path "data/processed/zh.jsonl.gz")) {
    python data/scripts/clean.py --only zh
}
if (-not (Test-Path "data/processed/en.jsonl.gz")) {
    python data/scripts/clean_en_parallel.py
}

Write-Host "== 3. tokenizer ==" -ForegroundColor Cyan
if (-not (Test-Path "tokenizer/artifacts/tokenizer.json")) {
    python tokenizer/train_tokenizer.py
}

Write-Host "== 4. pack + mix ==" -ForegroundColor Cyan
if (-not (Test-Path "data/tokenized/train.bin")) {
    python data/scripts/pack_tokens.py
    python data/scripts/mix.py --total-tokens 500000000 --zh-ratio 0.5
}

Write-Host "== 5. train variants ==" -ForegroundColor Cyan
if (-not (Test-Path "checkpoints/dense_25m/best.pt")) {
    python train/train.py --config configs/dense_25m.toml --max-minutes 600
}
if (-not (Test-Path "checkpoints/moe_4e/best.pt")) {
    python train/train.py --config configs/moe_4e.toml --max-minutes 150
}
if (-not (Test-Path "checkpoints/looped_16m/best.pt")) {
    python train/train.py --config configs/looped_16m.toml --max-minutes 100
}

Write-Host "== 6. eval + export ==" -ForegroundColor Cyan
python eval/evaluate.py --ckpt checkpoints/dense_25m/best.pt --tag dense_25m
python eval/evaluate.py --ckpt checkpoints/moe_4e/best.pt --tag moe_4e
python eval/evaluate.py --ckpt checkpoints/looped_16m/best.pt --tag looped_16m
python export/onnx_export.py --ckpt checkpoints/dense_25m/best.pt --out web/model/model.onnx

Write-Host "== 7. quantise ==" -ForegroundColor Cyan
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('web/model/model.onnx','web/model/model_int8.onnx',weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul','Gemm'])"

Write-Host "done. serve: python -m http.server 8080 --directory web" -ForegroundColor Green
