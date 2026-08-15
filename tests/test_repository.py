"""Repository tests — TC24 through TC27.

All tests use the ``temp_db`` fixture from conftest.py, which creates an
isolated SQLite database from db/schema.sql.  The production database at
db/reliability.db is never touched.
"""

import sqlite3

import pytest

from src.repository import Repository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CFG = {
    "mc_dropout": {"n_runs": 30},
    "seed": 42,
    "gradcam": {"iou_percentile": 80, "topk_k": 0.10},
}

_PREDICTION = {
    "pred_label": "tench",
    "correct": 1,
    "confidence": 0.95,
    "entropy": 0.10,
    "pred_variance": 0.02,
    "pred_agreement": 1.0,
    "mutual_information": 0.01,
}

_EXPLANATION = {
    "cam_corr_mean": 0.80,
    "cam_corr_std": 0.05,
    "cam_iou_mean": 0.60,
    "topk_overlap": 0.70,
}


@pytest.fixture
def repo(temp_db: str) -> Repository:
    """Open a Repository over the temporary test database."""
    r = Repository(temp_db)
    yield r
    r.conn.close()


@pytest.fixture
def model_id(repo: Repository) -> int:
    """Insert a minimal model row and return its model_id."""
    return repo.create_model(
        name="test_resnet",
        arch="resnet18",
        dropout_p=0.2,
        dropout_layers="layer2,layer3",
        checkpoint_path="models/checkpoints/test.pth",
    )


@pytest.fixture
def run_id(repo: Repository, model_id: int) -> int:
    """Insert a run row for the test model and return its run_id."""
    return repo.create_run(
        model_id=model_id,
        dataset_name="imagenette",
        cfg=_CFG,
    )


# ---------------------------------------------------------------------------
# TC24 — insert a run
# ---------------------------------------------------------------------------

def test_tc24_insert_run(repo: Repository, model_id: int) -> None:
    """TC24: create_run returns a positive run_id and the row is present.

    Input : valid model_id, dataset_name, config dict.
    Expect: integer run_id returned; row present in runs table with correct
            dataset_name and status = 'running'.
    """
    rid = repo.create_run(
        model_id=model_id,
        dataset_name="imagenette",
        cfg=_CFG,
    )

    assert isinstance(rid, int), f"Expected int, got {type(rid)}"
    assert rid > 0, f"Expected positive run_id, got {rid}"

    row = repo.conn.execute(
        "SELECT * FROM runs WHERE run_id = ?", (rid,)
    ).fetchone()
    assert row is not None, "Run row not found after insert"
    assert row["dataset_name"] == "imagenette"
    assert row["status"] == "running"
    assert row["n_runs"] == 30
    assert row["seed"] == 42


# ---------------------------------------------------------------------------
# TC25 — insert image result
# ---------------------------------------------------------------------------

def test_tc25_insert_image_result(repo: Repository, run_id: int) -> None:
    """TC25: insert_image_result writes rows to images, predictions, explanations.

    Input : valid run_id; image_meta, prediction, explanation dicts.
    Expect: image_id returned; rows present in all three tables.
    """
    image_meta = {
        "image_path": "data/imagenette2-320/val/n01440764/ILSVRC2012_val_00000293.JPEG",
        "dataset_type": "id",
        "true_label": "tench",
    }

    image_id = repo.insert_image_result(
        run_id=run_id,
        image_meta=image_meta,
        prediction=_PREDICTION,
        explanation=_EXPLANATION,
    )

    assert isinstance(image_id, int), f"Expected int image_id, got {type(image_id)}"
    assert image_id > 0

    assert repo.conn.execute(
        "SELECT 1 FROM images WHERE image_id = ?", (image_id,)
    ).fetchone() is not None, "Row missing from images"

    assert repo.conn.execute(
        "SELECT 1 FROM predictions WHERE image_id = ?", (image_id,)
    ).fetchone() is not None, "Row missing from predictions"

    assert repo.conn.execute(
        "SELECT 1 FROM explanations WHERE image_id = ?", (image_id,)
    ).fetchone() is not None, "Row missing from explanations"

    # Spot-check stored values
    pred_row = repo.conn.execute(
        "SELECT confidence, pred_agreement FROM predictions WHERE image_id = ?",
        (image_id,),
    ).fetchone()
    assert pred_row["confidence"] == pytest.approx(0.95)
    assert pred_row["pred_agreement"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TC26 — invalid dataset_type raises IntegrityError
# ---------------------------------------------------------------------------

def test_tc26_invalid_dataset_type_raises(repo: Repository, run_id: int) -> None:
    """TC26: insert_image_result raises IntegrityError for an invalid dataset_type.

    Input : dataset_type='garbage' (not in the CHECK constraint allowlist).
    Expect: sqlite3.IntegrityError raised; no rows committed to any table.

    This verifies both the CHECK constraint enforcement and the atomicity of
    the three-table insert — no orphan rows should remain after the failure.
    """
    image_meta = {
        "image_path": "data/test.jpg",
        "dataset_type": "garbage",   # not in ('id','near_ood','far_ood','corrupted','uploaded')
        "true_label": None,
    }
    bad_pred = dict(_PREDICTION, correct=None, mutual_information=None)

    with pytest.raises(sqlite3.IntegrityError):
        repo.insert_image_result(
            run_id=run_id,
            image_meta=image_meta,
            prediction=bad_pred,
            explanation=_EXPLANATION,
        )

    # No orphan rows in images
    count = repo.conn.execute(
        "SELECT count(*) FROM images WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    assert count == 0, f"Orphan image rows after failed insert: {count}"


# ---------------------------------------------------------------------------
# TC27 — fetch_metrics confidence filter
# ---------------------------------------------------------------------------

def test_tc27_fetch_metrics_confidence_filter(repo: Repository, run_id: int) -> None:
    """TC27: fetch_metrics with a confidence filter returns only matching rows.

    Input : two rows with confidence 0.95 and 0.60; filter confidence__gte=0.90.
    Expect: exactly one row returned (the 0.95 row).
    """
    high_pred = dict(_PREDICTION, confidence=0.95)
    low_pred  = dict(_PREDICTION, confidence=0.60)

    for idx, pred in enumerate([high_pred, low_pred]):
        repo.insert_image_result(
            run_id=run_id,
            image_meta={
                "image_path": f"data/img_{idx}.jpg",
                "dataset_type": "id",
                "true_label": "tench",
            },
            prediction=pred,
            explanation=_EXPLANATION,
        )

    df = repo.fetch_metrics(run_id, filters={"confidence__gte": 0.90})

    assert len(df) == 1, f"Expected 1 row, got {len(df)}"
    assert df.iloc[0]["confidence"] >= 0.90

    # Unfiltered should return both rows
    df_all = repo.fetch_metrics(run_id)
    assert len(df_all) == 2
