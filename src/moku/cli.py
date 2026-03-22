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
