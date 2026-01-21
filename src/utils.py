import torch


def load_hf_safe(model_id, dtype, device_map, mmap=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = dict(
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
    )
    if mmap:
        kwargs["mmap"] = True
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return model, tokenizer


def procrustes_align(a, b):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a0 = a - a.mean(dim=0, keepdim=True)
    b0 = b - b.mean(dim=0, keepdim=True)
    u, _, v = torch.linalg.svd(a0.t() @ b0, full_matrices=False)
    r = u @ v
    return a0 @ r + b.mean(dim=0, keepdim=True)


def entropy_router(logits, top_k=4, temperature=1.0):
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
