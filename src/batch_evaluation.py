"""Full per-image pipeline orchestration.

``run_batch`` is the single entry point for batch evaluation.  It:

1. Creates a run record in the database.
2. Loads the model checkpoint.
3. Runs the per-image loop:
   MC-Dropout forward pass → prediction metrics → fixed target-class
   Grad-CAM → explanation metrics → artefact storage → DB insert.
4. Assigns risk groups via a median quadrant split.
5. Computes reference baselines (B1 / B2 / B3).
6. Marks the run ``completed`` (or ``failed`` on any unhandled error).

Progress is reported through an optional callback so this module never
imports Streamlit (CLAUDE.md §9).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

from models.resnet_dropout import build_mc_dropout_resnet18, enable_mc_dropout
from src import baselines as _baselines
from src import metrics_explanation as _mex
from src import metrics_prediction as _mpred
from src.corruption_generator import SUPPORTED_CORRUPTIONS
from src.dataset_loader import get_dataset
from src.gradcam_generator import generate_repeated_cams
from src.inference_mc_dropout import mc_dropout_forward
from src.repository import Repository
from src.utils import ensure_dirs, get_device, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _DeviceDataset:
    """Thin dataset wrapper that moves image tensors to a target device.

    ``src.baselines.compute_all`` draws images directly from the dataset and
    passes them to model forward calls.  When the model is on CUDA the images
    must also be on CUDA.  This wrapper intercepts ``__getitem__`` and calls
    ``.to(device)`` on the image so baselines.py does not need to know the
    device.

    All other attributes (``classes``, ``imgs``, etc.) are forwarded to the
    wrapped dataset transparently.
    """

    def __init__(self, dataset, device: torch.device) -> None:
        self._ds = dataset
        self._device = device

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        img, label = self._ds[idx]
        return img.to(self._device), label

    def __getattr__(self, name: str):
        # Forward attribute lookup to the wrapped dataset
        return getattr(self._ds, name)


def _save_heatmap(arr: np.ndarray, path: Path) -> None:
    """Normalise *arr* to [0, 255] and write as a JET-colourmap PNG."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        norm = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(arr, dtype=np.uint8)
    coloured = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imwrite(str(path), coloured)


def _assign_risk_groups(metrics_for_risk: list[dict]) -> list[dict]:
    """Assign each image to a risk quadrant via median split.

    Axes:
    - Prediction uncertainty  : entropy        (high ≥ median → uncertain)
    - Explanation instability : 1 - cam_iou_mean (high ≥ median → unstable)

    Median split avoids hard-coded thresholds and balances the four quadrants
    across the actual run distribution.

    | pred uncertain | expl unstable | risk_group           |
    |----------------|---------------|----------------------|
    | no             | no            | stable               |
    | yes            | yes           | unstable_both        |
    | yes            | no            | pred_unstable_only   |
    | no             | yes           | hidden_risk          |

    Args:
        metrics_for_risk: List of dicts, each with keys
            ``image_id``, ``entropy``, ``cam_iou_mean``.

    Returns:
        List of dicts with keys ``image_id``, ``risk_group``, ``risk_score``
        (always ``None`` — risk_score is reserved for future use).
    """
    if not metrics_for_risk:
        return []

    entropies     = np.array([m["entropy"]       for m in metrics_for_risk])
    instabilities = np.array([1.0 - m["cam_iou_mean"] for m in metrics_for_risk])

    med_ent  = float(np.median(entropies))
    med_inst = float(np.median(instabilities))

    flags: list[dict] = []
    for m, ent, inst in zip(metrics_for_risk, entropies, instabilities):
        high_pred = bool(ent  >= med_ent)
        high_expl = bool(inst >= med_inst)

        if   not high_pred and not high_expl:
            group = "stable"
        elif     high_pred and     high_expl:
            group = "unstable_both"
        elif     high_pred and not high_expl:
            group = "pred_unstable_only"
        else:
            group = "hidden_risk"

        flags.append({
            "image_id":   m["image_id"],
            "risk_group": group,
            "risk_score": None,
        })

    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_batch(
    cfg: dict,
    model_id: int,
    dataset_name: str,
    dataset_type: str = "id",
    limit: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> int:
    """Run the full reliability pipeline over a dataset split.

    Sets the seed from ``cfg``, creates a run record, loads the checkpoint
    registered under *model_id*, processes each image (skipping failures),
    writes artefacts, assigns risk groups, computes B1/B2/B3 baselines, then
    marks the run as ``completed``.  If any unhandled exception escapes the
    loop or post-loop steps the run is marked ``failed`` before re-raising.

    Per-image failures are caught, logged, and counted — they do not abort the
    run.  This matches the requirement in CLAUDE.md §5: "The pipeline logs and
    skips a failed image rather than aborting a 4,000-image run."

    Args:
        cfg: Full configuration dict from ``load_config``.
        model_id: Primary key in the ``models`` table.  The checkpoint path
                  is read from there.
        dataset_name: One of ``"imagenette"``, ``"imagewoof"``, ``"dtd"``.
        dataset_type: Value for the ``images.dataset_type`` column.
                      Default ``"id"`` (in-distribution).
        limit: Cap on images processed.  ``None`` means the full dataset.
        progress_cb: Optional ``(completed, total) -> None`` callback fired
                     after every image (including skipped ones).  Keeps this
                     module free of any Streamlit dependency.

    Returns:
        The newly created ``run_id``.

    Raises:
        ValueError: If *model_id* is not in the database.
        Re-raises any other exception after marking the run as ``failed``.
    """
    set_seed(cfg["seed"])
    ensure_dirs()
    device = get_device()

    repo   = Repository(cfg["database"]["path"])
    run_id = repo.create_run(model_id, dataset_name, cfg)
    logger.info(
        "Run %d started  dataset=%s  device=%s", run_id, dataset_name, device
    )

    try:
        # ----------------------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------------------
        row = repo.conn.execute(
            "SELECT checkpoint_path FROM models WHERE model_id = ?", (model_id,)
        ).fetchone()
        if row is None:
            raise ValueError(
                f"model_id={model_id} not found in the models table."
            )

        ckpt_path = row["checkpoint_path"]
        model = build_mc_dropout_resnet18(
            num_classes=cfg["model"]["num_classes"],
            p=cfg["model"]["dropout_p"],
        )
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()
        logger.info("Checkpoint loaded: %s", ckpt_path)

        # ----------------------------------------------------------------
        # Dataset
        # ----------------------------------------------------------------
        dataset, class_names = get_dataset(dataset_name, cfg)
        total = min(limit, len(dataset)) if limit is not None else len(dataset)
        logger.info("Images to process: %d", total)

        n_runs  = cfg["mc_dropout"]["n_runs"]
        raw_dir = Path(cfg["paths"]["outputs"]) / "gradcam" / "raw"
        png_dir = Path(cfg["paths"]["outputs"]) / "gradcam" / "png"
        raw_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        metrics_for_risk: list[dict] = []
        n_ok       = 0
        n_failures = 0

        # ----------------------------------------------------------------
        # Per-image loop
        # ----------------------------------------------------------------
        for idx in range(total):
            try:
                img_tensor, label_idx = dataset[idx]
                true_label = class_names[label_idx]

                # Resolve the image path — ImageFolder exposes .imgs; DTD does not
                if hasattr(dataset, "imgs"):
                    image_path = dataset.imgs[idx][0]
                elif hasattr(dataset, "_image_files"):
                    image_path = str(dataset._image_files[idx])
                else:
                    image_path = f"{dataset_name}/{idx}"

                image = img_tensor.to(device)

                # ---- Batched MC-Dropout forward pass ----
                probs_t = mc_dropout_forward(model, image, n_runs)   # (n_runs, C)
                probs   = probs_t.cpu().numpy()

                # ---- Prediction metrics ----
                pred_met = _mpred.compute_all(probs, true_label, class_names)

                # ---- Fixed target class (argmax of mean — CLAUDE.md §1.3) ----
                target_class = int(probs.mean(axis=0).argmax())

                # ---- Grad-CAM maps ----
                enable_mc_dropout(model)
                cams_raw = generate_repeated_cams(
                    model, image, target_class, n_runs
                )  # (n_runs, 7, 7) float32

                # ---- Explanation metrics (upsampled) ----
                expl_met = _mex.compute_all(cams_raw, cfg)

                # ---- Artefacts ----
                npz_path = raw_dir / f"{run_id}_{idx}.npz"
                mean_png = png_dir / f"{run_id}_{idx}_mean.png"
                var_png  = png_dir / f"{run_id}_{idx}_var.png"

                # Raw CAMs: float16, ~3 KB each — all N maps in one file
                np.savez_compressed(
                    str(npz_path), cams=cams_raw.astype(np.float16)
                )

                # Mean and variability maps at upsample resolution
                upsampled = _mex.upsample_cams(
                    cams_raw, size=cfg["gradcam"]["upsample_size"]
                )
                _save_heatmap(upsampled.mean(axis=0), mean_png)
                _save_heatmap(_mex.variability_map(upsampled), var_png)

                # 6 representative frames evenly spaced
                rep_indices = np.linspace(0, n_runs - 1, 6, dtype=int)
                for k, ri in enumerate(rep_indices):
                    _save_heatmap(
                        upsampled[ri],
                        png_dir / f"{run_id}_{idx}_rep{k}.png",
                    )

                # ---- Atomic three-table insert ----
                image_id = repo.insert_image_result(
                    run_id=run_id,
                    image_meta={
                        "image_path":  image_path,
                        "dataset_type": dataset_type,
                        "true_label":  true_label,
                    },
                    prediction={
                        "pred_label":         pred_met["pred_label"],
                        "correct":            pred_met["correct"],
                        "confidence":         pred_met["confidence"],
                        "entropy":            pred_met["entropy"],
                        "pred_variance":      pred_met["pred_variance"],
                        "pred_agreement":     pred_met["pred_agreement"],
                        "mutual_information": pred_met["mutual_information"],
                    },
                    explanation={
                        "cam_corr_mean":     expl_met["cam_corr_mean"],
                        "cam_corr_std":      expl_met["cam_corr_std"],
                        "cam_iou_mean":      expl_met["cam_iou_mean"],
                        "topk_overlap":      expl_met["topk_overlap"],
                        "cam_npz_path":      str(npz_path.resolve()),
                        "mean_png_path":     str(mean_png.resolve()),
                        "variance_png_path": str(var_png.resolve()),
                    },
                )

                metrics_for_risk.append({
                    "image_id":    image_id,
                    "entropy":     pred_met["entropy"],
                    "cam_iou_mean": expl_met["cam_iou_mean"],
                })
                n_ok += 1

            except Exception as exc:
                logger.warning("Image %d skipped: %s", idx, exc)
                n_failures += 1

            if progress_cb is not None:
                progress_cb(idx + 1, total)

        logger.info(
            "Loop complete  processed=%d  skipped=%d", n_ok, n_failures
        )

        # ----------------------------------------------------------------
        # Risk group assignment (median quadrant split over the full run)
        # ----------------------------------------------------------------
        flags = _assign_risk_groups(metrics_for_risk)
        if flags:
            repo.upsert_risk_flags(run_id, flags)

        # ----------------------------------------------------------------
        # Reference baselines (B1 / B2 / B3)
        # Device-aware wrapper ensures images reach the model's device.
        # ----------------------------------------------------------------
        device_dataset = _DeviceDataset(dataset, device)
        _baselines.compute_all(model, device_dataset, cfg, repo, run_id)

        if n_ok == 0:
            logger.error(
                "Run %d marked failed: loop completed but 0 images were "
                "processed successfully (n_failures=%d).",
                run_id, n_failures,
            )
            repo.update_run_status(run_id, "failed")
        else:
            repo.update_run_status(run_id, "completed")
            logger.info("Run %d completed.", run_id)

    except Exception as exc:
        logger.error("Run %d FAILED: %s", run_id, exc)
        repo.update_run_status(run_id, "failed")
        raise

    return run_id


def run_corrupted_batch(
    cfg: dict,
    model_id: int,
    corruptions: list[str],
    severities: list[int],
    per_cell: int = 250,
    progress_cb: Callable[[int, int], None] | None = None,
) -> int:
    """Run the reliability pipeline over all corruption × severity cells.

    A **single** run_id is created for the entire corrupted experiment so that
    E8 analysis can query all cells with one run ID.  Each image row records
    its ``corruption_type`` and ``corruption_severity`` columns (already in the
    schema).

    Baselines are **not** recomputed for the corrupted run — they are inherited
    from the clean imagenette run (run_id 11).  B1/B2/B3 values are already in
    the database and are not meaningful on corrupted inputs.

    The same *per_cell* images (fixed seed=42) are evaluated for every cell, so
    direct cross-cell comparison is valid.

    Args:
        cfg: Full configuration dict from ``load_config``.
        model_id: Primary key in the ``models`` table.
        corruptions: List of corruption type strings (subset of
            ``SUPPORTED_CORRUPTIONS``).
        severities: List of severity integers (1–5).
        per_cell: Number of images per corruption × severity cell.  Default 250.
        progress_cb: Optional ``(completed, total) -> None`` callback.

    Returns:
        The newly created ``run_id``.
    """
    for c in corruptions:
        if c not in SUPPORTED_CORRUPTIONS:
            raise ValueError(
                f"Unknown corruption {c!r}. Supported: {SUPPORTED_CORRUPTIONS}"
            )
    for s in severities:
        if not (1 <= s <= 5):
            raise ValueError(f"severity must be in [1, 5], got {s}.")

    set_seed(cfg["seed"])
    ensure_dirs()
    device = get_device()

    repo   = Repository(cfg["database"]["path"])
    # Use "imagenette-c" as the dataset name in the runs table
    run_id = repo.create_run(model_id, "imagenette-c", cfg)
    logger.info(
        "Corrupted run %d started  corruptions=%s  severities=%s",
        run_id, corruptions, severities,
    )

    total_cells  = len(corruptions) * len(severities)
    total_images = total_cells * per_cell
    n_done_global = 0

    try:
        # ----------------------------------------------------------------
        # Load checkpoint once
        # ----------------------------------------------------------------
        row = repo.conn.execute(
            "SELECT checkpoint_path FROM models WHERE model_id = ?", (model_id,)
        ).fetchone()
        if row is None:
            raise ValueError(
                f"model_id={model_id} not found in the models table."
            )

        ckpt_path = row["checkpoint_path"]
        model = build_mc_dropout_resnet18(
            num_classes=cfg["model"]["num_classes"],
            p=cfg["model"]["dropout_p"],
        )
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()
        logger.info("Checkpoint loaded: %s", ckpt_path)

        n_runs  = cfg["mc_dropout"]["n_runs"]
        raw_dir = Path(cfg["paths"]["outputs"]) / "gradcam" / "raw"
        png_dir = Path(cfg["paths"]["outputs"]) / "gradcam" / "png"
        raw_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        metrics_for_risk: list[dict] = []
        n_ok       = 0
        n_failures = 0

        # ----------------------------------------------------------------
        # Outer loop: corruption type × severity
        # ----------------------------------------------------------------
        for corruption_type in corruptions:
            for severity in severities:
                dataset, class_names = get_dataset(
                    "imagenette-c",
                    cfg,
                    corruption_type=corruption_type,
                    severity=severity,
                    n=per_cell,
                    seed=42,
                )
                cell_total = len(dataset)
                logger.info(
                    "Cell  corruption=%s  severity=%d  n=%d",
                    corruption_type, severity, cell_total,
                )

                for idx in range(cell_total):
                    try:
                        img_tensor, label_idx = dataset[idx]
                        true_label = class_names[label_idx]

                        # Resolve source image path via .imgs attribute
                        image_path = dataset.imgs[idx][0]

                        image = img_tensor.to(device)

                        # MC-Dropout forward pass
                        probs_t = mc_dropout_forward(model, image, n_runs)
                        probs   = probs_t.cpu().numpy()

                        # Prediction metrics
                        pred_met = _mpred.compute_all(probs, true_label, class_names)

                        # Fixed target class (mean argmax — CLAUDE.md §1.3)
                        target_class = int(probs.mean(axis=0).argmax())

                        # Grad-CAM maps
                        enable_mc_dropout(model)
                        cams_raw = generate_repeated_cams(
                            model, image, target_class, n_runs
                        )

                        # Explanation metrics
                        expl_met = _mex.compute_all(cams_raw, cfg)

                        # Artefacts — tag with cell info to avoid collisions
                        cell_tag = f"{corruption_type}_s{severity}"
                        npz_path = raw_dir / f"{run_id}_{cell_tag}_{idx}.npz"
                        mean_png = png_dir / f"{run_id}_{cell_tag}_{idx}_mean.png"
                        var_png  = png_dir / f"{run_id}_{cell_tag}_{idx}_var.png"

                        np.savez_compressed(
                            str(npz_path), cams=cams_raw.astype(np.float16)
                        )

                        upsampled = _mex.upsample_cams(
                            cams_raw, size=cfg["gradcam"]["upsample_size"]
                        )
                        _save_heatmap(upsampled.mean(axis=0), mean_png)
                        _save_heatmap(_mex.variability_map(upsampled), var_png)

                        # Atomic three-table insert — passes corruption metadata
                        image_id = repo.insert_image_result(
                            run_id=run_id,
                            image_meta={
                                "image_path":          image_path,
                                "dataset_type":        "corrupted",
                                "true_label":          true_label,
                                "corruption_type":     corruption_type,
                                "corruption_severity": severity,
                            },
                            prediction={
                                "pred_label":         pred_met["pred_label"],
                                "correct":            pred_met["correct"],
                                "confidence":         pred_met["confidence"],
                                "entropy":            pred_met["entropy"],
                                "pred_variance":      pred_met["pred_variance"],
                                "pred_agreement":     pred_met["pred_agreement"],
                                "mutual_information": pred_met["mutual_information"],
                            },
                            explanation={
                                "cam_corr_mean":     expl_met["cam_corr_mean"],
                                "cam_corr_std":      expl_met["cam_corr_std"],
                                "cam_iou_mean":      expl_met["cam_iou_mean"],
                                "topk_overlap":      expl_met["topk_overlap"],
                                "cam_npz_path":      str(npz_path.resolve()),
                                "mean_png_path":     str(mean_png.resolve()),
                                "variance_png_path": str(var_png.resolve()),
                            },
                        )

                        metrics_for_risk.append({
                            "image_id":    image_id,
                            "entropy":     pred_met["entropy"],
                            "cam_iou_mean": expl_met["cam_iou_mean"],
                        })
                        n_ok += 1

                    except Exception as exc:
                        logger.warning(
                            "Cell %s/s%d image %d skipped: %s",
                            corruption_type, severity, idx, exc,
                        )
                        n_failures += 1

                    n_done_global += 1
                    if progress_cb is not None:
                        progress_cb(n_done_global, total_images)

        logger.info(
            "Corrupted loop complete  processed=%d  skipped=%d",
            n_ok, n_failures,
        )

        # Risk group assignment over the full corrupted run
        flags = _assign_risk_groups(metrics_for_risk)
        if flags:
            repo.upsert_risk_flags(run_id, flags)

        # Baselines not computed for corrupted runs — inherit from clean run
        if n_ok == 0:
            logger.error(
                "Corrupted run %d marked failed: loop completed but 0 images "
                "were processed successfully (n_failures=%d).",
                run_id, n_failures,
            )
            repo.update_run_status(run_id, "failed")
        else:
            repo.update_run_status(run_id, "completed")
            logger.info("Corrupted run %d completed.", run_id)

    except Exception as exc:
        logger.error("Corrupted run %d FAILED: %s", run_id, exc)
        repo.update_run_status(run_id, "failed")
        raise

    return run_id
