from typing import List
import typer
import torch
import merger

app = typer.Typer()


@app.command()
def merge(
    base: str = typer.Option(..., "--base"),
    models: List[str] = typer.Option(..., "--models", nargs=-1),
    rank: int = typer.Option(128, "--rank"),
    output: str = typer.Option(..., "--output"),
    quant: str = typer.Option("q4", "--quant"),
    moefrac: float = typer.Option(0.6, "--moefrac"),
):
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    merger.merge(
        base_id=base,
        model_ids=list(models),
        rank=rank,
        output=output,
        quant=quant,
        moefrac=moefrac,
        device_map=device_map,
        mmap=True,
    )


if __name__ == "__main__":
    app()
