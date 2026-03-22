import json
import os
import re
import time
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
    delay: float = typer.Option(2.0, "--delay", "-d", help="Seconds between requests."),
) -> None:
    """Generate goban images with AI. Resumable — skips already generated files."""
    from dotenv import load_dotenv
    from google import genai

    from moku.generate import generate_image, make_prompt

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

    generated = len(existing)
    failures = 0
    idx = 0

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task("Generating", total=n, completed=generated)

        while generated < n:
            # Skip existing indices
            while idx in existing:
                idx += 1

            prompt, board_size = make_prompt()
            progress.update(task, description=f"[{board_size}] idx={idx:04d}")

            try:
                img = generate_image(client, prompt, model=model)
            except Exception as e:
                failures += 1
                progress.console.print(f"  [red]Error: {e}[/red]")
                if failures >= 10:
                    progress.console.print("[red]Too many consecutive failures, stopping.[/red]")
                    raise typer.Exit(1)
                time.sleep(delay * 2)
                continue

            if img is not None:
                path = out_dir / f"{prefix}_{idx:04d}.png"
                img.save(path)
                progress.console.print(f"  Saved {path.name} ({img.size})")
                generated += 1
                failures = 0
                idx += 1
                progress.update(task, completed=generated)
            else:
                progress.console.print("  [yellow]No image returned, retrying...[/yellow]")
                failures += 1
                if failures >= 10:
                    progress.console.print("[red]Too many consecutive failures, stopping.[/red]")
                    raise typer.Exit(1)

            if generated < n:
                time.sleep(delay)

    typer.echo(f"Done — {generated} images in {out_dir}")


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
