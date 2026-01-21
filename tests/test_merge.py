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
