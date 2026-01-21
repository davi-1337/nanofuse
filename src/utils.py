from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, TypeAlias

import torch

from errors import ModelFileNotFoundError

TensorIndex: TypeAlias = dict[str, Path]


def resolve_model_dir(
    model_id: str, cache_dir: str | None = None, verbose: bool = False
) -> Path:
    path = Path(model_id)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    tqdm_class = None
    if verbose:
        from tqdm import tqdm

        tqdm_class = tqdm
        print(f"Downloading model: {model_id}", flush=True)
    local_dir = snapshot_download(model_id, cache_dir=cache_dir, tqdm_class=tqdm_class)
    return Path(local_dir)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def find_weight_files(model_dir: Path) -> tuple[list[Path], dict[str, str] | None]:
    single = model_dir / "model.safetensors"
    index_path = model_dir / "model.safetensors.index.json"
    if single.exists():
        return [single], None
    if index_path.exists():
        index_data = load_json(index_path)
        weight_map = index_data.get("weight_map", {})
        files = sorted({model_dir / v for v in weight_map.values()})
        return files, weight_map
    raise ModelFileNotFoundError(f"No safetensors weights found in {model_dir}")


def build_tensor_index(
    model_dir: Path, files: Iterable[Path], weight_map: dict[str, str] | None
) -> TensorIndex:
    if weight_map is not None:
        return {k: model_dir / v for k, v in weight_map.items()}
    from safetensors import safe_open

    index: TensorIndex = {}
    for file_path in files:
        with safe_open(str(file_path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                index[key] = file_path
    return index


def extract_layer_id(key: str) -> str:
    patterns = [
        r"(?:^|\.)layers\.(\d+)",
        r"(?:^|\.)layer\.(\d+)",
        r"(?:^|\.)h\.(\d+)",
        r"(?:^|\.)blocks\.(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, key)
        if match:
            return f"layer_{match.group(1)}"
    return "global"


def layer_number(layer_id: str) -> int | None:
    match = re.search(r"layer_(\d+)", layer_id)
    if not match:
        return None
    return int(match.group(1))


def select_preview_layers(
    num_layers: int, preview: bool, start: int | None, end: int | None
) -> tuple[int, int] | None:
    if not preview:
        return None
    if start is not None and end is not None:
        return min(start, end), max(start, end)
    if num_layers <= 0:
        return None
    span = max(1, num_layers // 3)
    mid = num_layers // 2
    return max(0, mid - span // 2), min(num_layers - 1, mid + span // 2)


def procrustes_align(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a0 = a - a.mean(dim=0, keepdim=True)
    b0 = b - b.mean(dim=0, keepdim=True)
    u, _, v = torch.linalg.svd(a0.t() @ b0, full_matrices=False)
    r = u @ v
    return a0 @ r + b.mean(dim=0, keepdim=True)


def entropy_router(
    logits: torch.Tensor, top_k: int = 4, temperature: float = 1.0
) -> torch.Tensor:
    if logits.dim() != 1:
        logits = logits.flatten()
    p = torch.softmax(logits / temperature, dim=0)
    ent = -torch.sum(p * torch.log(p + 1e-8))
    scores = p * (1.0 + ent)
    k = min(top_k, scores.numel())
    top = torch.topk(scores, k)
    mask = torch.zeros_like(scores)
    mask[top.indices] = torch.softmax(top.values, dim=0)
    return mask
