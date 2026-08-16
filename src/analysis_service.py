"""Facade for the presentation layer.

Pages call AnalysisService methods only -- never batch_evaluation, Repository,
or inference code directly.  This keeps the app/ layer decoupled from the
domain and persistence layers (ARCHITECTURE.md).

Streamlit is never imported here.  Progress during long operations is exposed
through a progress_cb callback so this module remains UI-agnostic (CLAUDE.md 9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.repository import Repository
from src.utils import load_config


class AnalysisService:
    """Single object the presentation layer uses for all data access.

    Args:
        cfg: Full configuration dict from load_config().
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg  = cfg
        self._repo = Repository(cfg["database"]["path"])

    # ------------------------------------------------------------------
    # Run / model listings
    # ------------------------------------------------------------------

    def list_runs(self) -> pd.DataFrame:
        """Return all runs with model name and image count, newest first."""
        return pd.read_sql_query(
            """
            SELECT r.run_id,
                   r.dataset_name,
                   r.status,
                   r.n_runs,
                   r.started_at,
                   r.finished_at,
                   m.name      AS model_name,
                   m.dropout_p,
                   COUNT(i.image_id) AS n_images
            FROM runs r
            JOIN models m ON m.model_id = r.model_id
            LEFT JOIN images i ON i.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.run_id DESC
            """,
            self._repo.conn,
        )

    def list_models(self) -> pd.DataFrame:
        """Return all model records, newest first."""
        return pd.read_sql_query(
            "SELECT * FROM models ORDER BY model_id DESC",
            self._repo.conn,
        )

    # ------------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------------

    def fetch_run_summary(self, run_id: int) -> dict:
        """Return run metadata plus aggregate metrics for one run.

        Returns:
            Dict with keys: n_images, accuracy, mean_confidence, mean_entropy,
            mean_cam_iou, mean_cam_corr, plus all columns from the runs row.
            Empty dict if the run_id is not found.
        """
        run_row = self._repo.conn.execute(
            """
            SELECT r.run_id, r.dataset_name, r.status, r.n_runs, r.seed,
                   r.iou_threshold, r.topk_k, r.started_at, r.finished_at,
                   m.name AS model_name, m.arch, m.dropout_p
            FROM runs r
            JOIN models m ON m.model_id = r.model_id
            WHERE r.run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if run_row is None:
            return {}

        agg_row = self._repo.conn.execute(
            """
            SELECT COUNT(*)             AS n_images,
                   AVG(p.correct)       AS accuracy,
                   AVG(p.confidence)    AS mean_confidence,
                   AVG(p.entropy)       AS mean_entropy,
                   AVG(e.cam_iou_mean)  AS mean_cam_iou,
                   AVG(e.cam_corr_mean) AS mean_cam_corr
            FROM images i
            JOIN predictions  p ON p.image_id = i.image_id
            JOIN explanations e ON e.image_id = i.image_id
            WHERE i.run_id = ?
            """,
            (run_id,),
        ).fetchone()

        result = dict(run_row)
        if agg_row:
            result.update(dict(agg_row))
        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def fetch_metrics(
        self, run_id: int, filters: dict | None = None
    ) -> pd.DataFrame:
        """Return the full analysis frame for a run, optionally filtered.

        Filters use column__op keys (gte, lte, eq).
        """
        return self._repo.fetch_metrics(run_id, filters)

    # ------------------------------------------------------------------
    # Image detail
    # ------------------------------------------------------------------

    def fetch_image_detail(self, image_id: int) -> dict:
        """Return everything the image-detail page needs in one call.

        Augments the DB row with:
        - baselines: {baseline_type: {metric_name: value}}
        - cams_array: float16 (n_runs, H, W) numpy array from .npz, or None.

        Returns:
            Dict of all columns plus baselines and cams_array.
            Empty dict if image_id is not found.
        """
        detail = self._repo.fetch_image_detail(image_id)
        if not detail:
            return {}

        detail["baselines"] = self._repo.fetch_baselines(detail["run_id"])

        npz_path = detail.get("cam_npz_path")
        if npz_path and Path(npz_path).exists():
            try:
                detail["cams_array"] = np.load(npz_path)["cams"]
            except Exception:
                detail["cams_array"] = None
        else:
            detail["cams_array"] = None

        return detail

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def fetch_baselines(self, run_id: int) -> dict:
        """Return {baseline_type: {metric_name: value}} for a run."""
        return self._repo.fetch_baselines(run_id)

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def run_batch(
        self,
        model_id: int,
        dataset_name: str,
        dataset_type: str = "id",
        limit: int | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> int:
        """Execute the reliability pipeline.  Returns the new run_id."""
        from src.batch_evaluation import run_batch as _run
        return _run(
            cfg=self._cfg,
            model_id=model_id,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            limit=limit,
            progress_cb=progress_cb,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_metrics_csv(self, run_id: int) -> bytes:
        """Return the metrics table as UTF-8 CSV bytes."""
        return self._repo.fetch_metrics(run_id).to_csv(index=False).encode("utf-8")

    def export_reviews_csv(self, run_id: int | None = None) -> bytes:
        """Return review decisions as UTF-8 CSV bytes."""
        if run_id is not None:
            sql = """
                SELECT rv.review_id, rv.image_id, i.image_path,
                       rv.decision, rv.comment, rv.created_at
                FROM reviews rv
                JOIN images i ON i.image_id = rv.image_id
                WHERE i.run_id = ?
                ORDER BY rv.created_at DESC
            """
            df = pd.read_sql_query(sql, self._repo.conn, params=(run_id,))
        else:
            df = pd.read_sql_query(
                """
                SELECT rv.review_id, rv.image_id, i.image_path,
                       rv.decision, rv.comment, rv.created_at
                FROM reviews rv
                JOIN images i ON i.image_id = rv.image_id
                ORDER BY rv.created_at DESC
                """,
                self._repo.conn,
            )
        return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Process-level singleton — one DB connection shared across all pages
# ---------------------------------------------------------------------------

_instance: AnalysisService | None = None


def get_service(cfg_path: str = "configs/config.yaml") -> AnalysisService:
    """Return the process-level AnalysisService singleton.

    Thread safety note: creating two instances in a race is harmless —
    both point to the same SQLite file with check_same_thread=False.
    """
    global _instance
    if _instance is None:
        _instance = AnalysisService(load_config(cfg_path))
    return _instance
