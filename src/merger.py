from __future__ import annotations

import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Sequence, TypeAlias

import torch

import utils
from errors import ModelDimensionMismatchError, ModelFileNotFoundError, TensorShapeMismatchError

LayerScores: TypeAlias = dict[str, list[float]]
LayerMap: TypeAlias = dict[int, int]


class ModelAccess:
    def __init__(self, model_dir: Path, stack: ExitStack) -> None:
        self.model_dir = model_dir
        files, weight_map = utils.find_weight_files(model_dir)
        self.index = utils.build_tensor_index(model_dir, files, weight_map)
        self.handles = {}
        from safetensors import safe_open

        for file_path in sorted({self.index[k] for k in self.index}):
            self.handles[file_path] = stack.enter_context(
                safe_open(str(file_path), framework="pt", device="cpu")
            )

    def keys(self) -> list[str]:
        return sorted(self.index.keys())

    def get_tensor(self, key: str, device: str) -> torch.Tensor:
        file_path = self.index[key]
        tensor = self.handles[file_path].get_tensor(key)
        if device != "cpu":
            return tensor.to(device)
        return tensor


class ShardWriter:
    def __init__(self, output_dir: Path, max_shard_size_bytes: int) -> None:
        self.output_dir = output_dir
        self.max_shard_size_bytes = max_shard_size_bytes
        self.buffer: dict[str, torch.Tensor] = {}
        self.buffer_size = 0
        self.shard_index = 0
        self.weight_map: dict[str, str] = {}
        self.total_size = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        tensor_cpu = tensor.detach().cpu()
        size = tensor_cpu.numel() * tensor_cpu.element_size()
        if size > self.max_shard_size_bytes and self.buffer:
            self.flush()
        if size > self.max_shard_size_bytes:
            self._write_shard({name: tensor_cpu})
            return
        self.buffer[name] = tensor_cpu
        self.buffer_size += size
        if self.buffer_size >= self.max_shard_size_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self._write_shard(self.buffer)
        self.buffer = {}
        self.buffer_size = 0

    def _write_shard(self, tensors: dict[str, torch.Tensor]) -> None:
        from safetensors.torch import save_file

        self.shard_index += 1
        filename = f"model-{self.shard_index:05d}.safetensors"
        path = self.output_dir / filename
        save_file(tensors, str(path))
        for key, value in tensors.items():
            self.weight_map[key] = filename
            self.total_size += value.numel() * value.element_size()

    def finalize(self) -> None:
        self.flush()
        index_path = self.output_dir / "model.safetensors.index.json"
        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": self.weight_map,
        }
        index_path.write_text(json.dumps(index, indent=2))


def svd_low_rank(stack: torch.Tensor, rank: int, niter: int = 2) -> torch.Tensor:
    if stack.dim() != 2:
        raise ModelDimensionMismatchError("SVD expects a 2D tensor")
    m, n = stack.shape
    limit = min(m, n)
    if limit <= 1:
        return stack
    q = min(rank, limit - 1)
    if q <= 0:
        return stack
    u, s, v = torch.svd_lowrank(stack, q=q, niter=niter)
    return (u * s) @ v.t()


def apply_dare(delta: torch.Tensor, drop_rate: float) -> torch.Tensor:
    if drop_rate <= 0:
        return delta
    if drop_rate >= 1:
        return torch.zeros_like(delta)
    flat = delta.abs().flatten()
    threshold = torch.quantile(flat, drop_rate)
    mask = (delta.abs() >= threshold).to(delta.dtype)
    scale = 1.0 / (1.0 - drop_rate)
    return delta * mask * scale


def align_vectors(stack: torch.Tensor) -> torch.Tensor:
    if stack.shape[0] < 2:
        return stack
    ref = stack.mean(dim=0, keepdim=True)
    aligned = [utils.procrustes_align(stack[i], ref).squeeze(0) for i in range(stack.shape[0])]
    return torch.stack(aligned)


def similarity_weights(stack: torch.Tensor) -> torch.Tensor:
    norms = stack / (stack.norm(dim=1, keepdim=True) + 1e-8)
    sim = norms @ norms.t()
    weights = torch.relu(sim.mean(dim=1))
    total = weights.sum()
    if total <= 1e-8:
        return torch.ones_like(weights) / weights.numel()
    return weights / total


def consensus_mask(stack: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weighted = (weights[:, None] * stack).sum(dim=0)
    sign_ref = torch.sign(weighted)
    agree = (torch.sign(stack) == sign_ref).float()
    agree_score = (weights[:, None] * agree).sum(dim=0) / (weights.sum() + 1e-8)
    mask = (agree_score >= 0.5).float()
    return weighted * mask


def moe_fuse(stack: torch.Tensor, weights: torch.Tensor, top_k: int = 4) -> torch.Tensor:
    router = utils.entropy_router(weights, top_k=top_k)
    return (router[:, None] * stack).sum(dim=0)


def conflict_score(stack: torch.Tensor) -> float:
    if stack.shape[0] < 2:
        return 0.0
    mean = stack.mean(dim=0)
    mean_norm = mean / (mean.norm() + 1e-8)
    norms = stack / (stack.norm(dim=1, keepdim=True) + 1e-8)
    sims = (norms @ mean_norm).clamp(-1.0, 1.0)
    return float(1.0 - sims.mean().item())


def is_embedding_key(key: str) -> bool:
    return key.endswith("embed_tokens.weight") or key.endswith("tok_embeddings.weight")


def is_lm_head_key(key: str) -> bool:
    return key.endswith("lm_head.weight") or key.endswith("output.weight")


def resize_vocab(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.shape[0] == new_size:
        return tensor
    if tensor.shape[0] > new_size:
        return tensor[:new_size]
    pad_rows = new_size - tensor.shape[0]
    pad = tensor.mean(dim=0, keepdim=True).repeat(pad_rows, 1)
    return torch.cat([tensor, pad], dim=0)


def merge_tokenizers(base_dir: Path, model_dirs: Sequence[Path]) -> tuple[object, int]:
    from transformers import AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(str(base_dir), use_fast=True)
    base_vocab = set(base_tokenizer.get_vocab().keys())
    special_tokens: dict[str, object] = {}
    extra_tokens: set[str] = set()
    for model_dir in model_dirs:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        extra_tokens.update(set(tokenizer.get_added_vocab().keys()) - base_vocab)
        for key, value in tokenizer.special_tokens_map.items():
            if key not in base_tokenizer.special_tokens_map:
                special_tokens[key] = value
                continue
            base_value = base_tokenizer.special_tokens_map.get(key)
            if isinstance(value, list) and isinstance(base_value, list):
                merged = list(dict.fromkeys(base_value + value))
                if merged != base_value:
                    special_tokens[key] = merged
    if special_tokens:
        base_tokenizer.add_special_tokens(special_tokens)
    if extra_tokens:
        base_tokenizer.add_tokens(sorted(extra_tokens))
    return base_tokenizer, len(base_tokenizer)


def parse_layer_map(path: str | None, model_ids: Sequence[str]) -> LayerMap:
    if not path:
        return {}
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    if not data:
        return {}
    items = data.get("layers", [])
    layer_map: LayerMap = {}
    for item in items:
        if isinstance(item, str):
            model_key, range_part = item.split(":", 1)
            start_str, end_str = range_part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
        elif isinstance(item, dict):
            model_key = item.get("model")
            range_value = item.get("range")
            if range_value:
                start, end = range_value
            else:
                start = item.get("start")
                end = item.get("end")
        else:
            continue
        if model_key not in model_ids:
            raise ModelDimensionMismatchError(f"Model not in merge list: {model_key}")
        if start is None or end is None:
            raise ModelDimensionMismatchError("Layer map ranges must include start and end")
        model_index = model_ids.index(model_key)
        for layer_idx in range(min(start, end), max(start, end) + 1):
            layer_map[layer_idx] = model_index
    return layer_map


def preflight_similarity(
    base: ModelAccess,
    models: Sequence[ModelAccess],
    keys: Sequence[str],
    device: str,
    threshold: float,
) -> None:
    if len(models) < 2:
        return
    sample_keys = [keys[0], keys[len(keys) // 2], keys[-1]] if keys else []
    scores: list[float] = []
    for key in sample_keys:
        base_tensor = base.get_tensor(key, device)
        model_tensors = [m.get_tensor(key, device) for m in models]
        if not torch.is_floating_point(base_tensor):
            continue
        stack = torch.stack([(t.float() - base_tensor.float()).reshape(-1) for t in model_tensors])
        scores.append(1.0 - conflict_score(stack))
    if scores and sum(scores) / len(scores) < threshold:
        print("Warning: Models appear orthogonal. Merging may degrade quality.")


def compute_fused_delta(
    base_tensor: torch.Tensor,
    model_tensors: Sequence[torch.Tensor],
    rank: int,
    moefrac: float,
    top_k: int,
    align: bool,
    dare_drop: float,
) -> tuple[torch.Tensor, float]:
    deltas = []
    for tensor in model_tensors:
        if tensor.shape != base_tensor.shape:
            raise TensorShapeMismatchError("Tensor shapes do not match")
        delta = tensor.float() - base_tensor.float()
        delta = apply_dare(delta, dare_drop)
        deltas.append(delta.reshape(-1))
    stack = torch.stack(deltas)
    low_rank = svd_low_rank(stack, rank)
    aligned = align_vectors(low_rank) if align else low_rank
    weights = similarity_weights(aligned)
    masked = consensus_mask(low_rank, weights)
    moe = moe_fuse(low_rank, weights, top_k=min(top_k, low_rank.shape[0]))
    fused = moefrac * moe + (1.0 - moefrac) * masked
    return fused.reshape(base_tensor.shape).to(base_tensor.dtype), conflict_score(stack)


def lora_keys(base_key: str) -> tuple[str, str]:
    base = base_key[:-7] if base_key.endswith(".weight") else base_key
    if base.startswith("model."):
        prefix = f"base_model.{base}"
    else:
        prefix = f"base_model.model.{base}"
    return f"{prefix}.lora_A.weight", f"{prefix}.lora_B.weight"


def lora_decompose(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if delta.dim() != 2:
        raise ModelDimensionMismatchError("LoRA requires 2D weights")
    out_dim, in_dim = delta.shape
    q = min(rank, out_dim - 1, in_dim - 1)
    if q <= 0:
        return torch.zeros((1, in_dim), device=delta.device), torch.zeros(
            (out_dim, 1), device=delta.device
        )
    u, s, v = torch.svd_lowrank(delta, q=q, niter=2)
    b = u * s
    a = v.t()
    return a, b


def save_lora_adapter(
    output_dir: Path,
    adapter_tensors: dict[str, torch.Tensor],
    base_model_id: str,
    target_modules: Sequence[str],
    rank: int,
) -> None:
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = output_dir / "adapter_model.safetensors"
    save_file(adapter_tensors, str(adapter_path))
    config = {
        "base_model_name_or_path": base_model_id,
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "target_modules": list(sorted(set(target_modules))),
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
    }
    (output_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))


def run_benchmarks(model_dir: str, tasks: str | None = None) -> str | None:
    tasks = tasks or "hellaswag,arc_easy,piqa"
    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_dir}",
        "--tasks",
        tasks,
        "--batch_size",
        "auto",
    ]
    if torch.cuda.is_available():
        cmd += ["--device", "cuda"]
    subprocess.run(cmd, check=False)
    return tasks


def merge(
    base_id: str,
    model_ids: Sequence[str],
    rank: int,
    output: str,
    quant: str,
    moefrac: float,
    device: str,
    report: bool,
    preview: bool,
    preview_start: int | None,
    preview_end: int | None,
    dare_drop: float,
    align: bool,
    lora_output: str | None,
    lora_rank: int,
    layer_map_path: str | None,
    preflight_threshold: float,
    shard_size_mb: int,
    verbose: bool,
) -> str:
    if rank <= 0:
        raise ModelDimensionMismatchError("Rank must be positive")
    if not model_ids:
        raise ModelDimensionMismatchError("At least one model is required")
    if device not in {"cpu", "cuda"}:
        raise ModelDimensionMismatchError("Device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ModelDimensionMismatchError("CUDA is not available")
    if not 0.0 <= moefrac <= 1.0:
        raise ModelDimensionMismatchError("MoE fraction must be between 0 and 1")
    if not 0.0 <= dare_drop <= 1.0:
        raise ModelDimensionMismatchError("DARE drop rate must be between 0 and 1")
    if preflight_threshold < 0.0 or preflight_threshold > 1.0:
        raise ModelDimensionMismatchError("Preflight threshold must be between 0 and 1")
    if lora_rank <= 0:
        raise ModelDimensionMismatchError("LoRA rank must be positive")
    if shard_size_mb <= 0:
        raise ModelDimensionMismatchError("Shard size must be positive")
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print("Resolving base model...", flush=True)
    base_dir = utils.resolve_model_dir(base_id, verbose=verbose)
    model_dirs = []
    for mid in model_ids:
        if verbose:
            print(f"Resolving merge model: {mid}", flush=True)
        model_dirs.append(utils.resolve_model_dir(mid, verbose=verbose))
    layer_map = parse_layer_map(layer_map_path, model_ids)
    with ExitStack() as stack:
        if verbose:
            print("Opening safetensors handles...", flush=True)
        base = ModelAccess(base_dir, stack)
        models = [ModelAccess(model_dir, stack) for model_dir in model_dirs]
        base_keys = base.keys()
        if not base_keys:
            raise ModelFileNotFoundError("No tensors found in base model")
        for model in models:
            if set(model.keys()) != set(base_keys):
                raise ModelDimensionMismatchError("Model tensors do not match base")
        from transformers import AutoConfig

        if verbose:
            print("Loading config and tokenizer...", flush=True)
        config = AutoConfig.from_pretrained(str(base_dir))
        num_layers = int(getattr(config, "num_hidden_layers", 0))
        preview_range = utils.select_preview_layers(num_layers, preview, preview_start, preview_end)
        tokenizer, new_vocab_size = merge_tokenizers(base_dir, model_dirs)
        config.vocab_size = new_vocab_size
        config.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        preflight_similarity(base, models, base_keys, device, preflight_threshold)
        writer = ShardWriter(output_dir, max(1, shard_size_mb) * 1024 * 1024)
        lora_tensors: dict[str, torch.Tensor] = {}
        lora_targets: set[str] = set()
        conflict_stats: LayerScores = {}
        total_keys = len(base_keys)
        if verbose:
            print(f"Merging {total_keys} tensors...", flush=True)
        last_layer = None
        for idx, key in enumerate(base_keys, start=1):
            base_tensor = base.get_tensor(key, device)
            if base_tensor.dim() >= 2 and (is_embedding_key(key) or is_lm_head_key(key)):
                base_tensor = resize_vocab(base_tensor, new_vocab_size)
            if not torch.is_floating_point(base_tensor):
                writer.add(key, base_tensor)
                continue
            layer_id = utils.extract_layer_id(key)
            if verbose and layer_id != last_layer:
                print(f"Processing {layer_id}", flush=True)
                last_layer = layer_id
            layer_idx = utils.layer_number(layer_id)
            if preview_range and layer_idx is not None:
                if layer_idx < preview_range[0] or layer_idx > preview_range[1]:
                    writer.add(key, base_tensor)
                    continue
            if layer_idx is not None and layer_idx in layer_map:
                selected = [models[layer_map[layer_idx]]]
            else:
                selected = models
            model_tensors = [m.get_tensor(key, device) for m in selected]
            if base_tensor.dim() >= 2 and (is_embedding_key(key) or is_lm_head_key(key)):
                model_tensors = [resize_vocab(t, new_vocab_size) for t in model_tensors]
            fused_delta = None
            merged = None
            conflict = 0.0
            try:
                fused_delta, conflict = compute_fused_delta(
                    base_tensor,
                    model_tensors,
                    rank,
                    moefrac,
                    4,
                    align,
                    dare_drop,
                )
                merged = base_tensor + fused_delta
            except RuntimeError as exc:
                if device == "cuda" and "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    device = "cpu"
                    base_tensor = base.get_tensor(key, device)
                    if base_tensor.dim() >= 2 and (
                        is_embedding_key(key) or is_lm_head_key(key)
                    ):
                        base_tensor = resize_vocab(base_tensor, new_vocab_size)
                    model_tensors = [m.get_tensor(key, device) for m in selected]
                    if base_tensor.dim() >= 2 and (
                        is_embedding_key(key) or is_lm_head_key(key)
                    ):
                        model_tensors = [resize_vocab(t, new_vocab_size) for t in model_tensors]
                    fused_delta, conflict = compute_fused_delta(
                        base_tensor,
                        model_tensors,
                        rank,
                        moefrac,
                        4,
                        align,
                        dare_drop,
                    )
                    merged = base_tensor + fused_delta
                else:
                    raise
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if merged is None or fused_delta is None:
                raise ModelDimensionMismatchError("Fusion failed for tensor")
            writer.add(key, merged)
            if report:
                scores = conflict_stats.setdefault(layer_id, [])
                scores.append(conflict)
            if lora_output and merged.dim() == 2:
                lora_a, lora_b = lora_decompose(fused_delta.float(), lora_rank)
                a_key, b_key = lora_keys(key)
                lora_tensors[a_key] = lora_a.cpu()
                lora_tensors[b_key] = lora_b.cpu()
                module_name = key.split(".")[-2]
                lora_targets.add(module_name)
            del base_tensor, model_tensors, fused_delta, merged
            if verbose and idx % 50 == 0:
                print(f"Merged {idx}/{total_keys} tensors", flush=True)
        writer.finalize()
        if verbose:
            print("Weights saved.", flush=True)
        if lora_output:
            save_lora_adapter(Path(lora_output), lora_tensors, base_id, lora_targets, lora_rank)
            if verbose:
                print("LoRA adapter saved.", flush=True)
        if report:
            print_conflict_report(conflict_stats)
        if quant and quant.lower() not in {"none", "fp16", "fp32"}:
            import quant as quant_mod

            quant_mod.quantize_model(output, tokenizer, quant)
            if verbose:
                print("Quantization complete.", flush=True)
    return output


def print_conflict_report(conflict_stats: LayerScores) -> None:
    if not conflict_stats:
        print("No conflict data collected.")
        return
    items = []
    for layer_id, scores in conflict_stats.items():
        items.append((layer_id, float(sum(scores) / len(scores))))
    items.sort(key=lambda x: utils.layer_number(x[0]) or 0)
    try:
        from rich.console import Console
        from rich.progress import BarColumn, Progress, TextColumn

        console = Console()
        with Progress(
            TextColumn("{task.fields[layer]}", justify="right"),
            BarColumn(),
            TextColumn("{task.fields[value]:.3f}"),
            console=console,
            transient=True,
        ) as progress:
            for layer_id, value in items:
                progress.add_task("", total=1.0, completed=value, layer=layer_id, value=value)
    except Exception:
        for layer_id, value in items:
            print(f"{layer_id}: {value:.3f}")
