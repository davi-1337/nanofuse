from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from errors import QuantizationError


def _load_texts(
    dataset_name: str, dataset_config: str, split: str, max_samples: int
) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split)
    texts = [t for t in dataset["text"] if isinstance(t, str) and t.strip()]
    if not texts:
        raise QuantizationError("Calibration dataset is empty")
    return texts[:max_samples]


def _calib_samples(
    tokenizer: Any,
    n_samples: int = 128,
    seq_len: int = 2048,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train[:1%]",
) -> list[dict[str, torch.Tensor]]:
    texts = _load_texts(dataset_name, dataset_config, split, max(n_samples * 2, 8))
    if tokenizer.pad_token_id is None:
        pad = tokenizer.eos_token or tokenizer.unk_token
        if pad is None:
            raise QuantizationError("Tokenizer has no pad, eos, or unk token")
        tokenizer.pad_token = pad
    samples: list[dict[str, torch.Tensor]] = []
    for text in texts:
        if len(samples) >= n_samples:
            break
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        samples.append(
            {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
        )
    if not samples:
        raise QuantizationError("Failed to build calibration samples")
    return samples


def quantize_gptq(
    model_dir: str,
    tokenizer: Any,
    output_dir: str,
    bits: int = 4,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
) -> str:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    quant_config = BaseQuantizeConfig(bits=bits, group_size=128, damp_percent=0.1, desc_act=False)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_dir,
        quantize_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    samples = _calib_samples(
        tokenizer, dataset_name=dataset_name, dataset_config=dataset_config
    )
    model.quantize(samples, use_triton=False)
    model.save_quantized(str(out), use_safetensors=True)
    tokenizer.save_pretrained(str(out))
    return str(out)


def quantize_awq(model_dir: str, tokenizer: Any, output_dir: str) -> str:
    from awq import AutoAWQForCausalLM

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}
    model = AutoAWQForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))
    return str(out)


def quantize_model(
    model_dir: str,
    tokenizer: Any,
    quant: str,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
) -> dict[str, str | None]:
    q = quant.lower()
    gptq_dir: str | None = None
    awq_dir: str | None = None
    if q.startswith("q4"):
        try:
            gptq_dir = quantize_gptq(
                model_dir,
                tokenizer,
                model_dir + "-gptq",
                bits=4,
                dataset_name=dataset_name,
                dataset_config=dataset_config,
            )
        except Exception as exc:
            raise QuantizationError("GPTQ quantization failed") from exc
        try:
            awq_dir = quantize_awq(model_dir, tokenizer, model_dir + "-awq")
        except Exception:
            awq_dir = None
    return {"gptq": gptq_dir, "awq": awq_dir}
