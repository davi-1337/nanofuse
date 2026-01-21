Command
- nanofuse merge: merge models or create LoRA adapters
- nanofuse benchmark: run lm-eval-harness subset

Merge options
- --base: base model id or local path
- --model/-m: repeat per model id
- --models: comma-separated model ids
- --rank: base rank
- --adaptive-rank: auto rank per layer
- --energy-threshold: spectral energy target
- --min-rank, --max-rank: adaptive rank bounds
- --moefrac: blend between MoE and consensus
- --dare-drop: DARE drop rate
- --align: Procrustes alignment
- --mosaic-size: block size for mosaic processing
- --hadamard-keep: low-pass keep ratio
- --io-workers: I/O worker threads
- --preview: merge only mid layers
- --preview-start, --preview-end: explicit layer range
- --lora-output: output directory for LoRA
- --lora-rank: LoRA rank
- --lora-only: skip full merge output
- --quant: q4, none, fp16, fp32
- --safe-mode: CPU-only, lower rank, no quant
- --max-vram-gb, --max-ram-gb: memory budgets
- --dtype: bf16, fp16, fp32
- --report: conflict report
- --verbose/-v: progress output

Benchmark options
- --model: model directory
- --tasks: lm-eval-harness task list
