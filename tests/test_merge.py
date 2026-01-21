import torch
import merger
import utils


def test_merge_state_dicts_shapes():
    torch.manual_seed(0)
    base = {
        "layer1.weight": torch.randn(4, 4),
        "layer1.bias": torch.randn(4),
        "layer2.weight": torch.randn(4, 4),
    }
    models = []
    for _ in range(3):
        state = {k: v + 0.01 * torch.randn_like(v) for k, v in base.items()}
        models.append(state)
    merged = merger.merge_state_dicts(base, models, rank=2, moefrac=0.6, top_k=2)
    assert set(merged.keys()) == set(base.keys())
    for k in base:
        assert merged[k].shape == base[k].shape
    assert not torch.allclose(merged["layer1.weight"], base["layer1.weight"])


def test_entropy_router_topk():
    logits = torch.tensor([0.1, 0.2, 0.3, 0.4])
    mask = utils.entropy_router(logits, top_k=2)
    assert torch.isclose(mask.sum(), torch.tensor(1.0))
    assert (mask > 0).sum().item() == 2
