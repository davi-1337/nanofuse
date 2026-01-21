Low-memory strategy
- Use --lora-only to avoid full model writes
- Use --safe-mode to force CPU and reduce rank
- Use --max-vram-gb and --max-ram-gb to cap resources
- Use --mosaic-size and --hadamard-keep to reduce compute

Example for 4GB VRAM / 8GB RAM
```
conda run --no-capture-output -n nanofuse nanofuse merge \
  --base Qwen/Qwen2.5-0.5B \
  --models Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-0.5B \
  --rank 64 \
  --adaptive-rank \
  --energy-threshold 0.95 \
  --min-rank 8 \
  --max-rank 64 \
  --mosaic-size 8192 \
  --hadamard-keep 0.25 \
  --io-workers 2 \
  --moefrac 0.6 \
  --output outputs/nanofuse-mini-0.5b \
  --lora-output outputs/nanofuse-mini-0.5b-lora \
  --lora-only \
  --dare-drop 0.3 \
  --report \
  --safe-mode \
  --max-vram-gb 4 \
  --max-ram-gb 8 \
  --dtype bf16 \
  -v
```
