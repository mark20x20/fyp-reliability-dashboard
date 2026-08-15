"""Integration tests — TC28 through TC31.

These tests exercise the full pipeline end-to-end against a temporary
in-memory database and a synthetic dataset of structured images.  No
Imagenette download is required.

Marks
-----
``slow`` — each test exercises model inference (MC-Dropout + Grad-CAM)
which is GPU-heavy.  Run with ``pytest -m "not slow"`` to skip on CI.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.batch_evaluation import run_batch
from src.repository import Repository
from src.utils import load_config


# ---------------------------------------------------------------------------
# Helpers shared across TC28-TC31
# ---------------------------------------------------------------------------

class _SyntheticDataset:
    """Minimal Dataset that yields structured 224x224 images.

    Each image is a per-channel gradient plus a Gaussian blob (same
    construction as the ``sample_image`` fixture), scaled to vary slightly
    across indices so that the model does not see the exact same tensor
    every time.  The dataset exposes ``classes`` and ``imgs`` so that
    ``run_batch`` can resolve image paths and class names.

    Args:
        n: Number of images.
        num_classes: Number of class names.
    """

    def __init__(self, n: int = 10, num_classes: int = 10) -> None:
        self.n = n
        self.classes = [f"class_{i}" for i in range(num_classes)]
        # Mimic ImageFolder's .imgs attribute
        self.imgs = [(f"synthetic/{i}.jpg", i % num_classes) for i in range(n)]
        self._num_classes = num_classes

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        # Structured image: gradient + blob -- same construction as conftest.py
        x = np.linspace(0, 1, 224, dtype=np.float32)
        y = np.linspace(0, 1, 224, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        # Vary slightly per index so we get different images
        shift = float(idx) / max(self.n, 1) * 0.1
        r = np.clip(0.30 + 0.40 * xx + shift, 0.0, 1.0)
        g = np.clip(0.30 + 0.40 * yy + shift, 0.0, 1.0)
        b = np.clip(0.30 + 0.20 * (xx + yy) + shift, 0.0, 1.0)

        cx, cy = 0.25, 0.25
        sigma = 0.12
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        r    = np.clip(r + 0.50 * blob, 0.0, 1.0)
        g    = np.clip(g + 0.40 * blob, 0.0, 1.0)
        b_ch = np.clip(b + 0.30 * blob, 0.0, 1.0)

        img = np.stack([r, g, b_ch], axis=0).astype(np.float32)

        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img  = (img - mean) / std

        label = idx % self._num_classes
        return torch.from_numpy(img), label


def _make_cfg(temp_db: str, tmp_path: Path, n_runs: int = 5) -> dict:
    """Build a config dict wired to the temporary database and output dir."""
    cfg = load_config("configs/config.yaml")
    cfg["database"]["path"]     = temp_db
    cfg["paths"]["outputs"]     = str(tmp_path / "outputs")
    cfg["mc_dropout"]["n_runs"] = n_runs   # fewer passes for speed
    return cfg


def _register_model(temp_db: str, ckpt_path: str) -> int:
    """Insert a model record and return model_id."""
    repo = Repository(temp_db)
    return repo.create_model(
        name="test_model",
        arch="resnet18",
        dropout_p=0.2,
        dropout_layers="layer2,layer3",
        checkpoint_path=ckpt_path,
    )


# ---------------------------------------------------------------------------
# TC28 -- 10 images -> 10 rows, no NULLs in required columns
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_tc28_pipeline_10_images_no_nulls(
    temp_db: str, model, tmp_path: Path
) -> None:
    """TC28: pipeline over 10 synthetic images produces 10 complete rows.

    Checks:
    - Exactly 10 rows in each of ``images``, ``predictions``, ``explanations``.
    - No NULL in the nine numeric columns that are never nullable:
      ``confidence``, ``entropy``, ``pred_variance``, ``pred_agreement``,
      ``mutual_information``, ``cam_corr_mean``, ``cam_corr_std``,
      ``cam_iou_mean``, ``topk_overlap``.
    """
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)

    cfg      = _make_cfg(temp_db, tmp_path)
    model_id = _register_model(temp_db, str(ckpt))
    dataset  = _SyntheticDataset(n=10, num_classes=10)

    with patch("src.batch_evaluation.get_dataset", return_value=(dataset, dataset.classes)):
        # Skip baselines to keep the test fast
        with patch("src.batch_evaluation._baselines.compute_all"):
            run_id = run_batch(cfg, model_id, "imagenette", limit=10)

    conn = sqlite3.connect(temp_db)

    n_images = conn.execute(
        "SELECT COUNT(*) FROM images WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    n_preds = conn.execute(
        "SELECT COUNT(*) FROM predictions p "
        "JOIN images i ON i.image_id = p.image_id WHERE i.run_id = ?",
        (run_id,),
    ).fetchone()[0]
    n_expls = conn.execute(
        "SELECT COUNT(*) FROM explanations e "
        "JOIN images i ON i.image_id = e.image_id WHERE i.run_id = ?",
        (run_id,),
    ).fetchone()[0]

    assert n_images == 10, f"Expected 10 image rows, got {n_images}"
    assert n_preds  == 10, f"Expected 10 prediction rows, got {n_preds}"
    assert n_expls  == 10, f"Expected 10 explanation rows, got {n_expls}"

    rows = conn.execute(
        """
        SELECT p.confidence, p.entropy, p.pred_variance, p.pred_agreement,
               p.mutual_information,
               e.cam_corr_mean, e.cam_corr_std, e.cam_iou_mean, e.topk_overlap
        FROM images i
        JOIN predictions  p ON p.image_id = i.image_id
        JOIN explanations e ON e.image_id = i.image_id
        WHERE i.run_id = ?
        """,
        (run_id,),
    ).fetchall()

    assert len(rows) == 10
    for row in rows:
        assert all(v is not None for v in row), (
            f"NULL found in a required column: {row}"
        )
    conn.close()


# ---------------------------------------------------------------------------
# TC29 -- baselines computed; B1 = 1.000; B2 and B3 within expected ranges
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_tc29_baselines_within_expected_ranges(
    temp_db: str, model, tmp_path: Path
) -> None:
    """TC29: B1 correlation = 1.000; B2 correlation is a low finite value;
    B3 correlation is finite and between B2 and 1.0.

    B1 must be exactly 1.000 -- any deviation indicates a Grad-CAM or seed bug.
    B2 uses Gaussian-smoothed random maps and should be near zero.
    B3 uses cross-image pairs and is expected to be >= B2.
    """
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)

    # Use 20 images so cross_image_reference has a better chance of finding
    # same-class pairs from the model's own predictions.
    cfg      = _make_cfg(temp_db, tmp_path, n_runs=3)
    model_id = _register_model(temp_db, str(ckpt))
    dataset  = _SyntheticDataset(n=20, num_classes=10)

    with patch("src.batch_evaluation.get_dataset", return_value=(dataset, dataset.classes)):
        run_id = run_batch(cfg, model_id, "imagenette", limit=20)

    repo      = Repository(temp_db)
    baselines = repo.fetch_baselines(run_id)

    # ---- B1 upper bound ----
    assert "upper" in baselines, "B1 (upper) baseline missing from database"
    b1_corr = baselines["upper"]["correlation"]
    assert b1_corr == pytest.approx(1.0, abs=1e-5), (
        f"B1 correlation should be 1.000, got {b1_corr:.8f}"
    )

    # ---- B2 lower bound ----
    assert "lower" in baselines, "B2 (lower) baseline missing from database"
    b2_corr = baselines["lower"]["correlation"]
    assert math.isfinite(b2_corr), f"B2 correlation is not finite: {b2_corr}"
    assert b2_corr < 0.5, (
        f"B2 (random lower bound) correlation should be < 0.5, got {b2_corr:.4f}"
    )

    # ---- B3 cross-image ----
    assert "cross_image" in baselines, "B3 (cross_image) baseline missing from database"
    b3_corr = baselines["cross_image"]["correlation"]
    assert math.isfinite(b3_corr), f"B3 correlation is not finite: {b3_corr}"
    # B3 should be >= B2 (cross-image same-class CAMs are more similar than random)
    assert b3_corr >= b2_corr - 1e-6, (
        f"B3 ({b3_corr:.4f}) should be >= B2 ({b2_corr:.4f})"
    )


# ---------------------------------------------------------------------------
# TC30 -- every image has exactly one risk_group
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_tc30_every_image_has_one_risk_group(
    temp_db: str, model, tmp_path: Path
) -> None:
    """TC30: after run_batch, every image row has exactly one risk_group.

    Checks that the risk-flag assignment step ran and that no image was
    left without a group.
    """
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)

    cfg      = _make_cfg(temp_db, tmp_path)
    model_id = _register_model(temp_db, str(ckpt))
    dataset  = _SyntheticDataset(n=10, num_classes=10)

    with patch("src.batch_evaluation.get_dataset", return_value=(dataset, dataset.classes)):
        with patch("src.batch_evaluation._baselines.compute_all"):
            run_id = run_batch(cfg, model_id, "imagenette", limit=10)

    conn = sqlite3.connect(temp_db)

    n_images = conn.execute(
        "SELECT COUNT(*) FROM images WHERE run_id = ?", (run_id,)
    ).fetchone()[0]

    n_flagged = conn.execute(
        """
        SELECT COUNT(*) FROM risk_flags rf
        JOIN images i ON i.image_id = rf.image_id
        WHERE i.run_id = ?
        """,
        (run_id,),
    ).fetchone()[0]

    assert n_images > 0, "No images were processed"
    assert n_flagged == n_images, (
        f"Expected {n_images} risk_flag rows, got {n_flagged}"
    )

    # No image should have more than one flag row
    n_duplicates = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT rf.image_id, COUNT(*) AS cnt
            FROM risk_flags rf
            JOIN images i ON i.image_id = rf.image_id
            WHERE i.run_id = ?
            GROUP BY rf.image_id
            HAVING cnt > 1
        )
        """,
        (run_id,),
    ).fetchone()[0]
    assert n_duplicates == 0, (
        f"{n_duplicates} image(s) have multiple risk_flag rows"
    )

    # All assigned groups must be among the four valid names
    valid_groups = {"stable", "unstable_both", "pred_unstable_only", "hidden_risk"}
    groups = {
        row[0]
        for row in conn.execute(
            """
            SELECT rf.risk_group FROM risk_flags rf
            JOIN images i ON i.image_id = rf.image_id
            WHERE i.run_id = ?
            """,
            (run_id,),
        ).fetchall()
    }
    assert groups.issubset(valid_groups), (
        f"Unexpected risk groups: {groups - valid_groups}"
    )

    conn.close()


# ---------------------------------------------------------------------------
# TC31 -- interrupted run -> run marked 'failed', no orphan rows
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_tc31_interrupted_run_marked_failed_no_orphans(
    temp_db: str, model, tmp_path: Path
) -> None:
    """TC31: if the pipeline fails after processing some images, the run is
    marked 'failed' and every completed image has a full set of rows.

    Simulates a failure in ``baselines.compute_all`` (which runs after the
    per-image loop) so some images are already written when the exception
    fires.  Verifies:

    - ``runs.status`` = 'failed'.
    - No orphan rows: every image row has exactly one prediction row and one
      explanation row (guaranteed by the atomic three-table insert).
    """
    ckpt = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt)

    cfg      = _make_cfg(temp_db, tmp_path)
    model_id = _register_model(temp_db, str(ckpt))
    dataset  = _SyntheticDataset(n=5, num_classes=10)

    def _raise_in_baselines(*args, **kwargs):
        raise RuntimeError("Simulated baselines failure")

    with patch("src.batch_evaluation.get_dataset", return_value=(dataset, dataset.classes)):
        with patch(
            "src.batch_evaluation._baselines.compute_all",
            side_effect=_raise_in_baselines,
        ):
            with pytest.raises(RuntimeError, match="Simulated baselines failure"):
                run_batch(cfg, model_id, "imagenette", limit=5)

    # run_batch raises, so run_id is not returned -- recover from the DB
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT run_id, status FROM runs ORDER BY run_id DESC LIMIT 1"
    ).fetchone()
    assert row is not None, "No run record found in the database"
    run_id = row[0]
    status = row[1]
    assert status == "failed", f"Expected run status 'failed', got {status!r}"

    # No orphan images (image row without a matching prediction row)
    n_orphan_preds = conn.execute(
        """
        SELECT COUNT(*) FROM images i
        LEFT JOIN predictions p ON p.image_id = i.image_id
        WHERE i.run_id = ? AND p.image_id IS NULL
        """,
        (run_id,),
    ).fetchone()[0]

    # No orphan images (image row without a matching explanation row)
    n_orphan_expls = conn.execute(
        """
        SELECT COUNT(*) FROM images i
        LEFT JOIN explanations e ON e.image_id = i.image_id
        WHERE i.run_id = ? AND e.image_id IS NULL
        """,
        (run_id,),
    ).fetchone()[0]

    assert n_orphan_preds == 0, (
        f"{n_orphan_preds} image(s) are missing a prediction row"
    )
    assert n_orphan_expls == 0, (
        f"{n_orphan_expls} image(s) are missing an explanation row"
    )

    # Whatever images were successfully processed must have complete data
    n_images = conn.execute(
        "SELECT COUNT(*) FROM images WHERE run_id = ?", (run_id,)
    ).fetchone()[0]

    if n_images > 0:
        n_complete = conn.execute(
            """
            SELECT COUNT(*) FROM images i
            JOIN predictions  p ON p.image_id = i.image_id
            JOIN explanations e ON e.image_id = i.image_id
            WHERE i.run_id = ?
            """,
            (run_id,),
        ).fetchone()[0]
        assert n_complete == n_images, (
            f"Expected {n_images} fully-complete image rows, got {n_complete}"
        )

    conn.close()
