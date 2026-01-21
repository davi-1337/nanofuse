NanoFuse is a fast low-rank LLM merger focused on lightweight, low-memory workflows.

Highlights
- Streaming safetensors merge, no full model in RAM
- LoRA-only mode for tiny outputs and fast runs
- Adaptive rank per layer for quality and size control
- Hadamard low-pass and mosaic blocks for low VRAM usage
- GPTQ/AWQ quantization with real-text calibration

Quick start
1) Create env and install deps
- Create: conda create -n nanofuse python=3.11
- Activate: conda activate nanofuse
- Install: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
- Pip: pip install -r requirements.txt
- CLI: pip install -e .

2) Merge two tiny models (LoRA-only, low memory)
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
  --safe-mode \
  --max-vram-gb 4 \
  --max-ram-gb 8 \
  --dtype bf16 \
  -v
```

3) Benchmark a merged model
```
conda run -n nanofuse nanofuse benchmark --model outputs/nanofuse-mini-0.5b --tasks hellaswag,arc_easy,piqa
```

Docs
- docs/overview.md
- docs/cli.md
- docs/merge.md
- docs/low-memory.md
- docs/recipes.md
