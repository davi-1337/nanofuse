NanoFuse merges multiple LLMs using a low-rank fusion pipeline with optional LoRA-only output.

Key ideas
- Task deltas: per-tensor deltas are computed against a base model
- Randomized SVD: fast low-rank approximation for merges
- Consensus + MoE: conflict pruning and top-k expert mixing
- Adaptive rank: per-layer rank based on spectral energy
- LoRA-only: output only adapters for minimal disk and memory use

Project layout
- src/cli.py: Typer CLI
- src/merger.py: streaming merge pipeline
- src/quant.py: GPTQ/AWQ wrappers with dataset calibration
- src/utils.py: safetensors discovery and helpers
- src/errors.py: custom error types

Outputs
- Full merge: model shards + config + tokenizer
- LoRA-only: adapter_model.safetensors + adapter_config.json
- Quantization: optional GPTQ/AWQ outputs
