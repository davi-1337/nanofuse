Recipes

Reasoning-leaning merge (two small models)
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
  --io-workers 4 \
  --moefrac 0.6 \
  --output outputs/nanofuse-mini-0.5b \
  --lora-output outputs/nanofuse-mini-0.5b-lora \
  --lora-only \
  --dare-drop 0.3 \
  --report \
  -v
```

Preview merge to validate quality
```
conda run -n nanofuse nanofuse merge \
  --base Qwen/Qwen2.5-0.5B \
  --models Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-0.5B \
  --rank 64 \
  --preview \
  --output outputs/nanofuse-preview-0.5b \
  --quant none
```

Benchmark merged model
```
conda run -n nanofuse nanofuse benchmark \
  --model outputs/nanofuse-mini-0.5b \
  --tasks hellaswag,arc_easy,piqa,winogrande
```
