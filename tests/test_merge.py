import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import merger


def test_svd_low_rank_shape():
    x = torch.randn(4, 16)
    y = merger.svd_low_rank(x, rank=2)
    assert y.shape == x.shape


def test_apply_dare_drops_values():
    x = torch.randn(100)
    y = merger.apply_dare(x, 0.5)
    assert y.shape == x.shape
    assert (y == 0).sum().item() > 0


def test_compute_fused_delta_shape():
    torch.manual_seed(0)
    base = torch.randn(4, 4)
    models = [base + 0.1 * torch.randn_like(base) for _ in range(3)]
    delta, conflict = merger.compute_fused_delta(
        base, models, rank=2, moefrac=0.6, top_k=2, align=False, dare_drop=0.0
    )
    assert delta.shape == base.shape
    assert isinstance(conflict, float)


def test_lora_only_requires_output():
    try:
        merger.merge(
            base_id="dummy",
            model_ids=["dummy"],
            rank=8,
            output="out",
            quant="none",
            moefrac=0.5,
            device="cpu",
            report=False,
            preview=False,
            preview_start=None,
            preview_end=None,
            dare_drop=0.0,
            align=False,
            lora_output=None,
            lora_rank=8,
            lora_only=True,
            layer_map_path=None,
            preflight_threshold=0.1,
            shard_size_mb=1,
            verbose=False,
            safe_mode=True,
            max_vram_gb=None,
            max_ram_gb=None,
            dtype="fp32",
        )
    except Exception as exc:
        assert "LoRA output" in str(exc)
