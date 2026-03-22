import asyncio
import json
import os
import re
from pathlib import Path

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

app = typer.Typer()


@app.command()
def generate(
    n: int = typer.Option(500, "--n", "-n", help="Total number of images to generate."),
    out_dir: Path = typer.Option(Path("data/generated"), "--out-dir", "-o", help="Output directory."),
    model: str = typer.Option("gemini-3.1-flash-image-preview", "--model", "-m", help="Model name."),
    prefix: str = typer.Option("gen", "--prefix", "-p", help="Filename prefix."),
    workers: int = typer.Option(20, "--workers", "-w", help="Number of parallel requests."),
) -> None:
    """Generate goban images with AI. Resumable — skips already generated files."""
    asyncio.run(_generate_async(n, out_dir, model, prefix, workers))


async def _generate_async(
    n: int,
    out_dir: Path,
    model: str,
    prefix: str,
    workers: int,
) -> None:
    from dotenv import load_dotenv
    from google import genai

    from moku.generate import generate_image_async, make_prompt

    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        typer.echo("ERROR: GEMINI_API_KEY not set in environment or .env", err=True)
        raise typer.Exit(1)

    client = genai.Client(api_key=api_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find already generated files to resume
    existing = set()
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.png$")
    for p in out_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            existing.add(int(m.group(1)))

    if existing:
        typer.echo(f"Resuming — {len(existing)} images already exist, skipping them.")

    # Build list of indices to generate
    indices: list[int] = []
    idx = 0
    while len(indices) < n - len(existing):
        if idx not in existing:
            indices.append(idx)
        idx += 1

    if not indices:
        typer.echo(f"All {n} images already exist. Nothing to do.")
        raise typer.Exit(0)

    semaphore = asyncio.Semaphore(workers)
    generated = len(existing)
    total_failures = 0

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    async def _worker(target_idx: int) -> bool:
        nonlocal generated, total_failures

        prompt, board_size = make_prompt()
        max_retries = 3

        for attempt in range(max_retries):
            async with semaphore:
                try:
                    img = await generate_image_async(client, prompt, model=model)
                except Exception as e:
                    progress.console.print(f"  [red]Error idx={target_idx:04d}: {e}[/red]")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    total_failures += 1
                    return False

            if img is not None:
                path = out_dir / f"{prefix}_{target_idx:04d}.png"
                img.save(path)
                generated += 1
                progress.update(task, completed=generated)
                return True

            # No image returned — retry
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

        progress.console.print(f"  [yellow]No image for idx={target_idx:04d} after {max_retries} attempts[/yellow]")
        total_failures += 1
        return False

    with progress:
        task = progress.add_task(f"Generating (×{workers})", total=n, completed=generated)
        tasks = [asyncio.create_task(_worker(i)) for i in indices]
        await asyncio.gather(*tasks)

    typer.echo(f"Done — {generated}/{n} images in {out_dir} ({total_failures} failures)")


@app.command()
def predict(
    images_dir: Path = typer.Option(
        Path("data/generated"), "--images-dir", "-i", help="Directory of images to predict on."
    ),
    out_dir: Path = typer.Option(
        Path("data/annotate_generated"), "--out-dir", "-o", help="Output directory for annotator workspace."
    ),
    model_name: str = typer.Option("kaya-go/moku-v2", "--model", "-m", help="HF model name or local path."),
    threshold: float = typer.Option(0.3, "--threshold", "-t", help="Confidence score threshold."),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Inference batch size."),
) -> None:
    """Run model predictions on images and prepare annotator workspace.

    Produces an ``images.json`` file compatible with the annotation server,
    plus copies/symlinks images into the annotator workspace.
    """
    import shutil

    import torch
    from PIL import Image
    from transformers import RTDetrForObjectDetection

    from moku.model import load_image_processor

    if not images_dir.is_dir():
        typer.echo(f"ERROR: {images_dir} is not a directory", err=True)
        raise typer.Exit(1)

    all_image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not all_image_paths:
        typer.echo(f"No images found in {images_dir}", err=True)
        raise typer.Exit(1)

    # Prepare output
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load existing images.json if present
    images_json_path = out_dir / "images.json"
    images_meta: list[dict] = []
    annotations: dict[str, dict] = {}
    already_done: set[str] = set()

    if images_json_path.exists():
        with open(images_json_path) as f:
            existing_data = json.load(f)
        images_meta = existing_data.get("images", [])
        annotations = existing_data.get("annotations", {})
        already_done = {img["filename"] for img in images_meta}

    # Filter to only new images
    image_paths = [p for p in all_image_paths if p.name not in already_done]

    typer.echo(
        f"Found {len(all_image_paths)} images total, {len(already_done)} already predicted, {len(image_paths)} new"
    )

    if not image_paths:
        typer.echo("Nothing to do — all images already predicted.")
        raise typer.Exit(0)

    # Load model
    typer.echo(f"Loading model {model_name} ...")
    image_processor = load_image_processor(model_name)
    model = RTDetrForObjectDetection.from_pretrained(model_name)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress, torch.no_grad():
        task = progress.add_task("Predicting", total=len(image_paths))

        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            pil_images = [Image.open(p).convert("RGB") for p in batch_paths]

            inputs = image_processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            target_sizes = torch.tensor([[img.height, img.width] for img in pil_images]).to(device)
            results = image_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=threshold
            )

            for path, img, result in zip(batch_paths, pil_images, results):
                filename = path.name

                # Copy image to annotator workspace
                dst = out_images_dir / filename
                if not dst.exists():
                    shutil.copy2(path, dst)

                images_meta.append(
                    {
                        "id": filename,
                        "filename": filename,
                        "width": img.width,
                        "height": img.height,
                        "source": "generated",
                    }
                )

                boxes_list = []
                for ann_id, (box, score, label) in enumerate(
                    zip(
                        result["boxes"].cpu().tolist(),
                        result["scores"].cpu().tolist(),
                        result["labels"].cpu().tolist(),
                    )
                ):
                    x1, y1, x2, y2 = box
                    boxes_list.append(
                        {
                            "id": ann_id,
                            "x": round(x1, 2),
                            "y": round(y1, 2),
                            "w": round(x2 - x1, 2),
                            "h": round(y2 - y1, 2),
                            "category": int(label),
                            "score": round(score, 4),
                        }
                    )

                annotations[filename] = {"boxes": boxes_list}

            progress.update(task, advance=len(batch_paths))

    # Write images.json (merged: existing + new)
    output_data = {"images": images_meta, "annotations": annotations}
    with open(images_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    n_total_boxes = sum(len(a["boxes"]) for a in annotations.values())
    typer.echo(f"Done — {len(images_meta)} images total, {n_total_boxes} predictions")
    typer.echo(f"Annotator workspace: {out_dir}")
    typer.echo(f"Launch annotator:  python tools/annotator/server.py --data-dir {out_dir}")
