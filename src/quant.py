import os
import torch


def _calib_samples(tokenizer, n_samples=8, seq_len=128):
    vocab = tokenizer.vocab_size
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    samples = []
    for _ in range(n_samples):
        ids = torch.randint(low=0, high=vocab, size=(1, seq_len))
        ids[0, -1] = eos
        attn = torch.ones_like(ids)
        samples.append({"input_ids": ids, "attention_mask": attn})
    return samples


def quantize_gptq(model_dir, tokenizer, output_dir, bits=4):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    os.makedirs(output_dir, exist_ok=True)
    quant_config = BaseQuantizeConfig(bits=bits, group_size=128, damp_percent=0.1, desc_act=False)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_dir,
        quantize_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    samples = _calib_samples(tokenizer)
    model.quantize(samples, use_triton=False)
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def quantize_awq(model_dir, tokenizer, output_dir):
    from awq import AutoAWQForCausalLM
    os.makedirs(output_dir, exist_ok=True)
    quant_config = {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"}
    model = AutoAWQForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def quantize_model(model_dir, tokenizer, quant):
    q = quant.lower()
    gptq_dir = None
    awq_dir = None
    if q.startswith("q4"):
        try:
            gptq_dir = quantize_gptq(model_dir, tokenizer, model_dir + "-gptq", bits=4)
        except Exception:
            gptq_dir = None
        try:
            awq_dir = quantize_awq(model_dir, tokenizer, model_dir + "-awq")
        except Exception:
            awq_dir = None
    return {"gptq": gptq_dir, "awq": awq_dir}
