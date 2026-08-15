"""Command-line entry point for the batch reliability pipeline.

Usage::

    python run_pipeline.py --dataset imagenette --limit 100
    python run_pipeline.py --dataset imagewoof --dataset-type id
    python run_pipeline.py --dataset imagenette --model-id 2 --limit 500

The script loads the configuration, queries the database for the requested
(or latest) model, runs ``src.batch_evaluation.run_batch`` with a tqdm
progress bar, then prints a concise summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the project root importable when invoked as a script
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from src.batch_evaluation import run_batch
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
        choices=["imagenette", "imagewoof", "dtd"],
        help="Dataset to evaluate (default: imagenette).",
    )
    p.add_argument(
        "--dataset-type",
        default="id",
        choices=["id", "near_ood", "far_ood", "corrupted", "uploaded"],
        help="Value stored in images.dataset_type (default: id).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N images (default: entire dataset).",
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
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()
    cfg  = load_config(args.config)

    repo = Repository(cfg["database"]["path"])

    # Resolve model_id — use provided value or fall back to the latest model
    if args.model_id is not None:
        model_id = args.model_id
    else:
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
        model_id = row["model_id"]

    # Determine total so we can size the progress bar
    from src.dataset_loader import get_dataset

    dataset, _ = get_dataset(args.dataset, cfg)
    total = min(args.limit, len(dataset)) if args.limit is not None else len(dataset)

    print("=" * 60)
    print(f"Reliability pipeline")
    print(f"  dataset    : {args.dataset} ({args.dataset_type})")
    print(f"  images     : {total}")
    print(f"  model_id   : {model_id}")
    print(f"  n_runs     : {cfg['mc_dropout']['n_runs']}")
    print("=" * 60)

    # Progress bar — driven by the progress_cb
    bar = tqdm(total=total, unit="img", desc="Processing")
    n_done = 0
    n_failures = [0]

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

    # Query summary stats
    conn = repo.conn
    n_images = conn.execute(
        "SELECT COUNT(*) FROM images WHERE run_id = ?", (run_id,)
    ).fetchone()[0]

    per_image_mean = elapsed / n_images if n_images else 0.0

    print()
    print("=" * 60)
    print(f"Run complete")
    print(f"  run_id          : {run_id}")
    print(f"  images processed: {n_images}")
    print(f"  elapsed         : {elapsed:.1f} s")
    print(f"  per-image mean  : {per_image_mean * 1000:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
