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
    out_dir: Path = typer.Option(Path("data/annotate_generated"), "--out-dir", "-o", help="Output directory."),
    model: str = typer.Option("gemini-3.1-flash-image-preview", "--model", "-m", help="Model name."),
    prefix: str = typer.Option("gen", "--prefix", "-p", help="Filename prefix."),
    workers: int = typer.Option(10, "--workers", "-w", help="Number of parallel requests."),
    image_size: int = typer.Option(640, "--image-size", help="Synthetic image size."),
) -> None:
    """Generate photorealistic goban images conditioned on synthetic inputs.

    For each sample: generates a synthetic goban with perfect annotations,
    sends it to Gemini for style transfer, and saves both the photorealistic
    image and the COCO annotations. Resumable — skips already generated files.
    """
    asyncio.run(_generate_conditioned_async(n, out_dir, model, prefix, workers, image_size))


async def _generate_conditioned_async(
    n: int,
    out_dir: Path,
    model: str,
    prefix: str,
    workers: int,
    image_size: int,
) -> None:
    from dotenv import load_dotenv
    from google import genai

    from moku.generate import make_style_transfer_prompt, synthetic_to_real_async
    from moku.synthetic import generate_synthetic_sample

    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        typer.echo("ERROR: GEMINI_API_KEY not set in environment or .env", err=True)
        raise typer.Exit(1)

    client = genai.Client(api_key=api_key)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Find already generated files to resume (need both .png and .json)
    existing = set()
    png_pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.png$")
    for p in images_dir.iterdir():
        m = png_pattern.match(p.name)
        if m:
            idx_val = int(m.group(1))
            json_path = images_dir / f"{prefix}_{idx_val:04d}.json"
            if json_path.exists():
                existing.add(idx_val)

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

    board_sizes = [9, 13, 19]
    board_weights = [0.15, 0.15, 0.7]

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    async def _worker(target_idx: int) -> bool:
        nonlocal generated, total_failures

        import random

        board_size = random.choices(board_sizes, weights=board_weights, k=1)[0]
        max_retries = 3

        for attempt in range(max_retries):
            # Generate synthetic image + annotations
            # ~40% of the time use a larger margin to give Gemini room
            # for context objects around the board
            margin_frac: float | None = None
            if random.random() < 0.4:
                margin_frac = random.uniform(0.05, 0.15)
            synth_image, annotation = generate_synthetic_sample(
                board_size=board_size,
                image_size=image_size,
                margin_frac=margin_frac,
            )

            prompt = make_style_transfer_prompt(board_size)

            async with semaphore:
                try:
                    real_image = await synthetic_to_real_async(
                        client,
                        synth_image,
                        prompt,
                        model=model,
                    )
                except Exception as e:
                    progress.console.print(f"  [red]Error idx={target_idx:04d}: {e}[/red]")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    total_failures += 1
                    return False

            if real_image is not None:
                filename = f"{prefix}_{target_idx:04d}.png"
                path = images_dir / filename

                # Resize Gemini output to match image_size
                real_image = real_image.resize((image_size, image_size))
                real_image.save(path)

                # Save per-image annotation JSON (atomic per worker)
                objects = annotation["objects"]
                boxes_list = []
                for ann_id, obj in enumerate(
                    zip(
                        objects["id"],
                        objects["bbox"],
                        objects["category"],
                        objects["area"],
                    )
                ):
                    obj_id, bbox, category, area = obj
                    boxes_list.append(
                        {
                            "id": ann_id,
                            "x": round(bbox[0], 2),
                            "y": round(bbox[1], 2),
                            "w": round(bbox[2], 2),
                            "h": round(bbox[3], 2),
                            "category": int(category),
                        }
                    )

                ann_data = {
                    "image": {
                        "id": filename,
                        "filename": filename,
                        "width": image_size,
                        "height": image_size,
                        "source": "synthetic_conditioned",
                        "board_size": board_size,
                    },
                    "boxes": boxes_list,
                }
                json_path = images_dir / f"{prefix}_{target_idx:04d}.json"
                with open(json_path, "w") as jf:
                    json.dump(ann_data, jf, indent=2)

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
        task = progress.add_task(f"Generating conditioned (×{workers})", total=n, completed=generated)
        tasks = [asyncio.create_task(_worker(i)) for i in indices]
        await asyncio.gather(*tasks)

    # Rebuild images.json from all per-image JSON files
    images_json_path = out_dir / "images.json"
    images_meta: list[dict] = []
    annotations: dict[str, dict] = {}

    for jf in sorted(images_dir.glob(f"{prefix}_*.json")):
        with open(jf) as f:
            ann_data = json.load(f)
        img_meta = ann_data["image"]
        images_meta.append(img_meta)
        annotations[img_meta["filename"]] = {"boxes": ann_data["boxes"]}

    output_data = {"images": images_meta, "annotations": annotations}
    with open(images_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    n_total_boxes = sum(len(a["boxes"]) for a in annotations.values())
    typer.echo(f"Done — {generated}/{n} images in {out_dir} ({total_failures} failures)")
    typer.echo(f"Annotations: {n_total_boxes} boxes across {len(annotations)} images")
    typer.echo(f"Launch annotator:  python tools/annotator/server.py --data-dir {out_dir}")
