"""Command-line entry point for the batch reliability pipeline.

Usage::

    python run_pipeline.py --dataset imagenette --limit 100
    python run_pipeline.py --dataset imagewoof --dataset-type id
    python run_pipeline.py --dataset imagenette --model-id 2 --limit 500

    # Phase 7 E8 — corrupted imagenette (4 types x 5 severities x 250 images)
    python run_pipeline.py --dataset imagenette-c \\
        --corruptions gaussian_noise,defocus_blur,brightness,contrast \\
        --severities 1,2,3,4,5 --per-cell 250

The script loads the configuration, queries the database for the requested
(or latest) model, runs the appropriate batch function with a tqdm progress
bar, then prints a concise summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the project root importable when invoked as a script
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from src.batch_evaluation import run_batch, run_corrupted_batch
from src.corruption_generator import SUPPORTED_CORRUPTIONS
from src.repository import Repository
from src.utils import load_config


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the reliability pipeline over a dataset split."
    )
    p.add_argument(
        "--dataset",
        default="imagenette",
        choices=["imagenette", "imagewoof", "dtd", "imagenette-c"],
        help="Dataset to evaluate (default: imagenette).",
    )
    p.add_argument(
        "--dataset-type",
        default="id",
        choices=["id", "near_ood", "far_ood", "corrupted", "uploaded"],
        help="Value stored in images.dataset_type (default: id). "
             "Ignored for imagenette-c (always 'corrupted').",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N images (default: entire dataset). "
             "Ignored for imagenette-c (use --per-cell).",
    )
    p.add_argument(
        "--model-id",
        type=int,
        default=None,
        metavar="ID",
        help="PK in the models table.  Defaults to the most recently created model.",
    )
    p.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml).",
    )
    # -- imagenette-c specific arguments ------------------------------------
    p.add_argument(
        "--corruptions",
        default=",".join(SUPPORTED_CORRUPTIONS),
        help=(
            "Comma-separated corruption types for imagenette-c. "
            f"Default: all supported ({', '.join(SUPPORTED_CORRUPTIONS)})."
        ),
    )
    p.add_argument(
        "--severities",
        default="1,2,3,4,5",
        help="Comma-separated severity levels (1-5) for imagenette-c. Default: 1,2,3,4,5.",
    )
    p.add_argument(
        "--per-cell",
        type=int,
        default=250,
        metavar="N",
        help="Images per corruption x severity cell for imagenette-c. Default: 250.",
    )
    return p


def _resolve_model_id(repo: Repository, requested: int | None) -> int:
    """Return *requested* if given, else the most recently created model_id."""
    if requested is not None:
        return requested
    row = repo.conn.execute(
        "SELECT model_id FROM models ORDER BY model_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        print(
            "ERROR: no models found in the database.  "
            "Run src/train_model.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return row["model_id"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    cfg  = load_config(args.config)

    repo     = Repository(cfg["database"]["path"])
    model_id = _resolve_model_id(repo, args.model_id)

    # ------------------------------------------------------------------
    # imagenette-c branch — corruption experiment (Phase 7 E8)
    # ------------------------------------------------------------------
    if args.dataset == "imagenette-c":
        corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
        severities  = [int(s.strip()) for s in args.severities.split(",") if s.strip()]
        per_cell    = args.per_cell
        total       = len(corruptions) * len(severities) * per_cell

        print("=" * 60)
        print("Reliability pipeline  [corrupted imagenette]")
        print(f"  corruptions : {', '.join(corruptions)}")
        print(f"  severities  : {severities}")
        print(f"  per cell    : {per_cell}")
        print(f"  total images: {total}")
        print(f"  model_id    : {model_id}")
        print(f"  n_runs      : {cfg['mc_dropout']['n_runs']}")
        print("=" * 60)

        bar   = tqdm(total=total, unit="img", desc="Corrupted")
        n_done = 0

        def _cb_c(completed: int, _total: int) -> None:
            nonlocal n_done
            bar.update(completed - n_done)
            n_done = completed

        t0 = time.perf_counter()

        run_id = run_corrupted_batch(
            cfg=cfg,
            model_id=model_id,
            corruptions=corruptions,
            severities=severities,
            per_cell=per_cell,
            progress_cb=_cb_c,
        )

        elapsed = time.perf_counter() - t0
        bar.close()

        n_images = repo.conn.execute(
            "SELECT COUNT(*) FROM images WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        per_img  = elapsed / n_images if n_images else 0.0

        print()
        print("=" * 60)
        print("Run complete  [corrupted]")
        print(f"  run_id          : {run_id}")
        print(f"  images processed: {n_images}")
        print(f"  elapsed         : {elapsed:.1f} s")
        print(f"  per-image mean  : {per_img * 1000:.1f} ms")
        print("=" * 60)
        return

    # ------------------------------------------------------------------
    # Standard branch — imagenette / imagewoof / dtd
    # ------------------------------------------------------------------
    from src.dataset_loader import get_dataset

    dataset, _ = get_dataset(args.dataset, cfg)
    total = min(args.limit, len(dataset)) if args.limit is not None else len(dataset)

    print("=" * 60)
    print("Reliability pipeline")
    print(f"  dataset    : {args.dataset} ({args.dataset_type})")
    print(f"  images     : {total}")
    print(f"  model_id   : {model_id}")
    print(f"  n_runs     : {cfg['mc_dropout']['n_runs']}")
    print("=" * 60)

    bar    = tqdm(total=total, unit="img", desc="Processing")
    n_done = 0

    def _cb(completed: int, _total: int) -> None:
        nonlocal n_done
        delta = completed - n_done
        bar.update(delta)
        n_done = completed

    t0 = time.perf_counter()

    run_id = run_batch(
        cfg=cfg,
        model_id=model_id,
        dataset_name=args.dataset,
        dataset_type=args.dataset_type,
        limit=args.limit,
        progress_cb=_cb,
    )

    elapsed = time.perf_counter() - t0
    bar.close()

    n_images = repo.conn.execute(
        "SELECT COUNT(*) FROM images WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    per_image_mean = elapsed / n_images if n_images else 0.0

    print()
    print("=" * 60)
    print("Run complete")
    print(f"  run_id          : {run_id}")
    print(f"  images processed: {n_images}")
    print(f"  elapsed         : {elapsed:.1f} s")
    print(f"  per-image mean  : {per_image_mean * 1000:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
