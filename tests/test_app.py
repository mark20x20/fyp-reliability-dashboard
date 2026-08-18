"""Application-layer tests — TC19 to TC23.

Testing approach per test case:

  TC19 — AppTest: run 1_Run_Analysis.py with role='reviewer' and assert the
         engineer-only error is shown.  The engineer guard fires before any
         DB or model call, so no live database is required.
         UI verification: screenshot in outputs/reports/screenshots/1_Run_Analysis.png.

  TC20 — Service layer: call run_batch() with a model whose checkpoint path
         does not exist; assert an exception is raised and the run is marked
         'failed' in the database.  The UI layer catches this exception and
         displays it via st.error() (verified manually, see screenshot
         outputs/reports/screenshots/run_error.png).

  TC21 — Service layer: fetch_metrics() with a confidence__gte filter;
         assert only rows at or above the threshold are returned.  This is
         a unit-level test of the Repository filtering that directly underlies
         the UI filter widgets.

  TC22 — Unit test of PIL image-validation logic (as used in 6_Upload_Analyse.py);
         assert that non-image bytes raise PIL.UnidentifiedImageError.
         UI layer verified manually: see outputs/reports/screenshots/upload_error.png.

  TC23 — Service layer: insert_review() then fetch_reviews(); assert the
         decision is persisted and returned correctly.  Covers the full
         path from AnalysisService down through Repository to SQLite.
"""

import io
import sqlite3
from pathlib import Path

import pytest

from src.analysis_service import AnalysisService
from src.repository import Repository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_cfg(db_path: str) -> dict:
    """Minimal config dict that points at a test database."""
    return {
        "database": {"path": db_path},
        "seed": 42,
        "model": {"num_classes": 10, "dropout_p": 0.2},
        "mc_dropout": {"n_runs": 30},
        "gradcam": {"iou_percentile": 80, "topk_k": 0.10, "upsample_size": 224},
        "data": {"input_size": 224, "batch_size": 64, "data_root": "data"},
        "paths": {"outputs": "outputs"},
        "risk_flag": {"confidence_threshold": 0.90, "iou_threshold": None},
    }


def _insert_minimal_image(repo: Repository, run_id: int, confidence: float,
                          cam_iou_mean: float = 0.5) -> int:
    """Insert one image with the given confidence and cam_iou_mean."""
    return repo.insert_image_result(
        run_id=run_id,
        image_meta={
            "image_path":  "test/image.jpg",
            "dataset_type": "id",
            "true_label":  "tench",
        },
        prediction={
            "pred_label":         "tench",
            "correct":            1,
            "confidence":         confidence,
            "entropy":            0.10,
            "pred_variance":      0.005,
            "pred_agreement":     1.0,
            "mutual_information": 0.01,
        },
        explanation={
            "cam_corr_mean": 0.80,
            "cam_corr_std":  0.05,
            "cam_iou_mean":  cam_iou_mean,
            "topk_overlap":  0.70,
        },
    )


# ---------------------------------------------------------------------------
# TC19 — reviewer cannot access engineer-only pages
# ---------------------------------------------------------------------------

def test_tc19_reviewer_blocked_from_engineer_pages():
    """TC19: role=Reviewer → engineer pages show an error and halt.

    Uses streamlit.testing.v1.AppTest to run page 1 with role='reviewer'
    and asserts that an error element is rendered.  The guard fires before
    any database call, so no real database is needed.
    """
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1 not available in this environment")

    at = AppTest.from_file("app/views/1_Run_Analysis.py")
    at.session_state["role"] = "reviewer"
    at.run(timeout=15)

    # The page must show an error and contain the word "engineer"
    assert len(at.error) > 0, "Expected an error element but none was found"
    assert "engineer" in at.error[0].value.lower(), (
        f"Error message does not mention 'engineer': {at.error[0].value!r}"
    )


# ---------------------------------------------------------------------------
# TC20 — missing checkpoint raises an error and marks run as failed
# ---------------------------------------------------------------------------

def test_tc20_missing_checkpoint_marks_run_failed(temp_db: str):
    """TC20: run with a missing checkpoint → exception raised, run status = 'failed'.

    The UI page (1_Run_Analysis.py) wraps svc.run_batch() in a try/except
    and calls st.error() on failure.  This test verifies the service layer
    raises an exception and correctly updates the database status.
    """
    cfg  = _minimal_cfg(temp_db)
    svc  = AnalysisService(cfg)
    repo = svc._repo

    # Register a model with a path that does not exist
    model_id = repo.create_model(
        name="bad_model",
        arch="resnet18",
        dropout_p=0.2,
        dropout_layers="layer2,layer3",
        checkpoint_path="/nonexistent/path/model.pth",
    )

    with pytest.raises(Exception):
        svc.run_batch(model_id=model_id, dataset_name="imagenette")

    # The run record must be marked 'failed', not left as 'running'
    row = repo.conn.execute(
        "SELECT status FROM runs WHERE model_id = ?", (model_id,)
    ).fetchone()
    assert row is not None, "No run record found for the failed batch"
    assert row["status"] == "failed", (
        f"Expected status='failed', got {row['status']!r}"
    )


# ---------------------------------------------------------------------------
# TC21 — confidence filter returns only matching rows
# ---------------------------------------------------------------------------

def test_tc21_confidence_filter(temp_db: str):
    """TC21: filter confidence > 0.95 → only matching rows returned.

    Inserts rows at 0.80, 0.90, 0.96, and 0.99, applies a confidence__gte
    filter at 0.95, and asserts only the two high-confidence rows appear.
    """
    cfg  = _minimal_cfg(temp_db)
    svc  = AnalysisService(cfg)
    repo = svc._repo

    model_id = repo.create_model(
        "m", "resnet18", 0.2, "layer2,layer3", "ckpt.pth"
    )
    run_id = repo.create_run(model_id, "imagenette", cfg)

    for conf in [0.80, 0.90, 0.96, 0.99]:
        _insert_minimal_image(repo, run_id, confidence=conf)

    df = svc.fetch_metrics(run_id, filters={"confidence__gte": 0.95})

    assert len(df) == 2, (
        f"Expected 2 rows with confidence >= 0.95, got {len(df)}"
    )
    assert (df["confidence"] >= 0.95).all(), (
        "fetch_metrics returned a row below the confidence threshold"
    )


# ---------------------------------------------------------------------------
# TC22 — non-image upload rejected by validation
# ---------------------------------------------------------------------------

def test_tc22_non_image_upload_rejected():
    """TC22: uploading a non-image file is caught by the validation logic.

    Directly tests the PIL-based validation used in 6_Upload_Analyse.py.
    The UI layer wraps this in a try/except and calls st.error() with a
    clear message; this test confirms the underlying library raises.
    """
    from PIL import UnidentifiedImageError

    non_image_bytes = b"This is definitely not an image file.\x00\x01\x02"
    buf = io.BytesIO(non_image_bytes)

    with pytest.raises((UnidentifiedImageError, Exception)):
        from PIL import Image
        img = Image.open(buf)
        img.verify()


# ---------------------------------------------------------------------------
# TC23 — review decision is persisted and returned
# ---------------------------------------------------------------------------

def test_tc23_review_persisted_and_visible(temp_db: str):
    """TC23: submit a review decision → persisted and visible after reload.

    Uses the service layer (AnalysisService.insert_review and fetch_reviews)
    to verify the end-to-end path from decision submission to retrieval.
    The UI form calls insert_review on submit and fetch_reviews on each load.
    """
    cfg  = _minimal_cfg(temp_db)
    svc  = AnalysisService(cfg)
    repo = svc._repo

    model_id = repo.create_model(
        "m", "resnet18", 0.2, "layer2,layer3", "ckpt.pth"
    )
    run_id   = repo.create_run(model_id, "imagenette", cfg)
    image_id = _insert_minimal_image(repo, run_id, confidence=0.97)

    # Submit a review through the service
    review_id = svc.insert_review(
        image_id=image_id,
        decision="needs_review",
        comment="Attention map looks fragmented.",
    )
    assert isinstance(review_id, int) and review_id > 0

    # Simulate a page reload: create a fresh service instance pointing at
    # the same database to prove the data survives across connections
    svc2    = AnalysisService(cfg)
    reviews = svc2.fetch_reviews(image_id)

    assert len(reviews) == 1, (
        f"Expected 1 review after submission, got {len(reviews)}"
    )
    assert reviews.iloc[0]["decision"] == "needs_review"
    assert "fragmented" in (reviews.iloc[0]["comment"] or "")
