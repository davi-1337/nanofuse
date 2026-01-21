from __future__ import annotations

import typer
import torch

import merger

app = typer.Typer(no_args_is_help=True)


@app.command()
def merge(
    base: str = typer.Option(..., "--base", help="Base model id or path"),
    models: list[str] | None = typer.Option(
        None, "--model", "-m", help="Model id (repeat per model)"
    ),
    models_csv: str | None = typer.Option(
        None, "--models", help="Comma-separated model ids"
    ),
    rank: int = typer.Option(128, "--rank", help="Low-rank dimension"),
    output: str = typer.Option(..., "--output", help="Output directory"),
    quant: str = typer.Option("q4", "--quant", help="Quantization mode"),
    moefrac: float = typer.Option(0.6, "--moefrac", help="MoE blend fraction"),
    device: str = typer.Option("auto", "--device", help="Device: auto, cpu, cuda"),
    report: bool = typer.Option(False, "--report", help="Show conflict report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    safe_mode: bool = typer.Option(False, "--safe-mode", help="Low memory mode"),
    max_vram_gb: float | None = typer.Option(
        None, "--max-vram-gb", help="VRAM budget in GB"
    ),
    max_ram_gb: float | None = typer.Option(
        None, "--max-ram-gb", help="RAM budget in GB"
    ),
    dtype: str = typer.Option("bf16", "--dtype", help="bf16, fp16, fp32"),
    preview: bool = typer.Option(False, "--preview", help="Merge preview layers only"),
    preview_start: int | None = typer.Option(
        None, "--preview-start", help="Preview start layer"
    ),
    preview_end: int | None = typer.Option(None, "--preview-end", help="Preview end layer"),
    dare_drop: float = typer.Option(0.0, "--dare-drop", help="DARE drop rate"),
    align: bool = typer.Option(False, "--align", help="Apply alignment"),
    lora_output: str | None = typer.Option(None, "--lora-output", help="LoRA output dir"),
    lora_rank: int = typer.Option(64, "--lora-rank", help="LoRA rank"),
    layer_map: str | None = typer.Option(None, "--layer-map", help="Layer map YAML"),
    preflight_threshold: float = typer.Option(
        0.1, "--preflight-threshold", help="Preflight similarity threshold"
    ),
    shard_size_mb: int = typer.Option(512, "--shard-size-mb", help="Shard size MB"),
) -> None:
    device = device.lower()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = list(models or [])
    if models_csv:
        selected.extend([m.strip() for m in models_csv.split(",") if m.strip()])
    if not selected:
        raise typer.BadParameter("At least one model id is required")
    merger.merge(
        base_id=base,
        model_ids=selected,
        rank=rank,
        output=output,
        quant=quant,
        moefrac=moefrac,
        device=device,
        report=report or verbose,
        preview=preview,
        preview_start=preview_start,
        preview_end=preview_end,
        dare_drop=dare_drop,
        align=align,
        lora_output=lora_output,
        lora_rank=lora_rank,
        layer_map_path=layer_map,
        preflight_threshold=preflight_threshold,
        shard_size_mb=shard_size_mb,
        verbose=verbose,
        safe_mode=safe_mode,
        max_vram_gb=max_vram_gb,
        max_ram_gb=max_ram_gb,
        dtype=dtype,
    )


@app.command()
def benchmark(
    model: str = typer.Option(..., "--model", help="Merged model directory"),
    tasks: str = typer.Option("hellaswag,arc_easy,piqa", "--tasks", help="Tasks"),
) -> None:
    merger.run_benchmarks(model, tasks)


if __name__ == "__main__":
    app()
