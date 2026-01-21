Merge flow
1) Resolve model dirs and open safetensors lazily
2) Read one tensor at a time
3) Compute deltas against base
4) Apply DARE, low-rank merge, consensus, and MoE
5) Save full merge shards or LoRA adapters

Full merge example
```
conda run -n nanofuse nanofuse merge \
  --base Qwen/Qwen2.5-0.5B \
  --models Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-0.5B \
  --rank 64 \
  --moefrac 0.6 \
  --output outputs/nanofuse-mini-0.5b \
  --quant q4 \
  --dare-drop 0.3 \
  --report \
  --shard-size-mb 256
```

LoRA-only example
```
conda run -n nanofuse nanofuse merge \
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
  --report
```
