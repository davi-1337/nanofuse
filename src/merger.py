import sys
import subprocess
import torch
import utils


def to_cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def compute_task_vectors(base_state, model_states):
    deltas = []
    for state in model_states:
        delta = {}
        for k, v in base_state.items():
            if k in state and state[k].shape == v.shape:
                delta[k] = state[k].to(v.dtype) - v
        deltas.append(delta)
    return deltas


def svd_low_rank(stack, rank):
    u, s, vh = torch.linalg.svd(stack, full_matrices=False)
    r = min(rank, s.numel())
    u = u[:, :r]
    s = s[:r]
    vh = vh[:r, :]
    return (u * s) @ vh


def umap_embed(x, n_components):
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components, metric="cosine")
        emb = reducer.fit_transform(x.cpu().numpy())
        return torch.from_numpy(emb).to(x.device).float()
    except Exception:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        emb = pca.fit_transform(x.cpu().numpy())
        return torch.from_numpy(emb).to(x.device).float()


def svd_umap_align(stack, rank):
    low_rank = svd_low_rank(stack, rank)
    if low_rank.shape[0] < 2:
        return low_rank, torch.zeros((low_rank.shape[0], 1), device=low_rank.device, dtype=low_rank.dtype)
    dims = min(2, low_rank.shape[0] - 1)
    emb = umap_embed(low_rank, dims)
    ref = emb.mean(dim=0, keepdim=True)
    aligned = []
    for i in range(emb.shape[0]):
        aligned_i = utils.procrustes_align(emb[i], ref).squeeze(0)
        aligned.append(aligned_i)
    return low_rank, torch.stack(aligned)


def similarity_weights(stack):
    norms = stack / (stack.norm(dim=1, keepdim=True) + 1e-8)
    sim = norms @ norms.t()
    weights = sim.mean(dim=1)
    weights = torch.relu(weights)
    denom = weights.sum()
    if denom < 1e-8:
        return torch.ones_like(weights) / weights.numel()
    weights = weights / denom
    return weights


def consensus_mask(stack, weights):
    weighted = (weights[:, None] * stack).sum(dim=0)
    sign_ref = torch.sign(weighted)
    agree = (torch.sign(stack) == sign_ref).float()
    agree_score = (weights[:, None] * agree).sum(dim=0) / (weights.sum() + 1e-8)
    mask = (agree_score >= 0.5).float()
    return weighted * mask


def moe_fuse(stack, weights, top_k=4):
    router = utils.entropy_router(weights, top_k=top_k)
    return (router[:, None] * stack).sum(dim=0)


def merge_state_dicts(base_state, model_states, rank=128, moefrac=0.6, top_k=4):
    deltas = compute_task_vectors(base_state, model_states)
    merged = {}
    for k, base_w in base_state.items():
        if not torch.is_floating_point(base_w):
            merged[k] = base_w
            continue
        tensors = []
        for d in deltas:
            if k in d:
                tensors.append(d[k].reshape(-1).float())
        if len(tensors) == 0:
            merged[k] = base_w
            continue
        stack = torch.stack(tensors)
        low_rank, aligned = svd_umap_align(stack, rank)
        weights = similarity_weights(aligned if aligned.numel() > 0 else low_rank)
        masked = consensus_mask(low_rank, weights)
        moe = moe_fuse(low_rank, weights, top_k=min(top_k, low_rank.shape[0]))
        fused = moefrac * moe + (1.0 - moefrac) * masked
        merged[k] = base_w + fused.reshape(base_w.shape).to(base_w.dtype)
    return merged


def run_benchmarks(model_dir, tasks=None):
    tasks = tasks or "hellaswag,arc_easy,piqa"
    try:
        import lm_eval
    except Exception:
        return None
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


def merge(base_id, model_ids, rank, output, quant, moefrac, device_map="auto", mmap=True):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    base_model, tokenizer = utils.load_hf_safe(base_id, dtype, device_map, mmap=mmap)
    base_state = to_cpu_state_dict(base_model)
    model_states = []
    for mid in model_ids:
        model, _ = utils.load_hf_safe(mid, dtype, device_map, mmap=mmap)
        model_states.append(to_cpu_state_dict(model))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    merged_state = merge_state_dicts(base_state, model_states, rank=rank, moefrac=moefrac, top_k=4)
    base_model.load_state_dict(merged_state, strict=False)
    base_model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    if quant and quant.lower() not in ["none", "fp16", "fp32"]:
        import quant as quant_mod
        quant_mod.quantize_model(output, tokenizer, quant)
    run_benchmarks(output)
    return output
