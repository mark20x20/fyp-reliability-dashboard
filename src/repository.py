"""Database access layer.  All SQL lives here and nowhere else.

Every other module calls Repository methods; no raw SQL appears outside this file.
See docs/DATABASE.md for the schema and common queries.
"""

import sqlite3
from datetime import datetime
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# SQL constants
# ---------------------------------------------------------------------------

_INSERT_MODEL = """
    INSERT INTO models (name, arch, dropout_p, dropout_layers, checkpoint_path, val_accuracy)
    VALUES (?, ?, ?, ?, ?, ?)
"""

_INSERT_RUN = """
    INSERT INTO runs (model_id, user_id, dataset_name, n_runs, seed,
                      iou_threshold, topk_k, status, started_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, 'running', CURRENT_TIMESTAMP)
"""

_UPDATE_RUN_STATUS = """
    UPDATE runs SET status = ?, finished_at = CURRENT_TIMESTAMP WHERE run_id = ?
"""

_INSERT_IMAGE = """
    INSERT INTO images (run_id, image_path, dataset_type,
                        corruption_type, corruption_severity, true_label)
    VALUES (?, ?, ?, ?, ?, ?)
"""

_INSERT_PREDICTION = """
    INSERT INTO predictions (image_id, pred_label, correct, confidence,
                             entropy, pred_variance, pred_agreement, mutual_information)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_EXPLANATION = """
    INSERT INTO explanations (image_id, cam_corr_mean, cam_corr_std,
                              cam_iou_mean, topk_overlap,
                              cam_npz_path, mean_png_path, variance_png_path)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_BASELINE = """
    INSERT INTO baselines (run_id, baseline_type, metric_name, value)
    VALUES (?, ?, ?, ?)
"""

_UPSERT_RISK_FLAG = """
    INSERT OR REPLACE INTO risk_flags (image_id, risk_group, risk_score)
    VALUES (?, ?, ?)
"""

_INSERT_REVIEW = """
    INSERT INTO reviews (image_id, user_id, decision, comment)
    VALUES (?, ?, ?, ?)
"""

# Full-frame query for analysis — see DATABASE.md Common queries
_FULL_FRAME = """
    SELECT i.image_id, i.dataset_type, i.corruption_type, i.corruption_severity,
           i.true_label, p.pred_label, p.correct,
           p.confidence, p.entropy, p.pred_variance, p.pred_agreement,
           p.mutual_information,
           e.cam_corr_mean, e.cam_corr_std, e.cam_iou_mean, e.topk_overlap,
           r.risk_group
    FROM images i
    JOIN predictions  p ON p.image_id = i.image_id
    JOIN explanations e ON e.image_id = i.image_id
    LEFT JOIN risk_flags r ON r.image_id = i.image_id
    WHERE i.run_id = ?
"""

_FETCH_DETAIL = """
    SELECT i.image_id, i.run_id, i.image_path, i.dataset_type,
           i.corruption_type, i.corruption_severity, i.true_label,
           p.pred_label, p.correct, p.confidence, p.entropy,
           p.pred_variance, p.pred_agreement, p.mutual_information,
           e.cam_corr_mean, e.cam_corr_std, e.cam_iou_mean, e.topk_overlap,
           e.cam_npz_path, e.mean_png_path, e.variance_png_path,
           r.risk_group, r.risk_score
    FROM images i
    JOIN predictions  p ON p.image_id = i.image_id
    JOIN explanations e ON e.image_id = i.image_id
    LEFT JOIN risk_flags r ON r.image_id = i.image_id
    WHERE i.image_id = ?
"""

_FETCH_BASELINES = """
    SELECT baseline_type, metric_name, value
    FROM baselines WHERE run_id = ?
"""

_BATCH_SIZE = 100


class Repository:
    """Single-connection repository over the SQLite reliability database.

    All callers share one connection per process.  Streamlit serves pages from
    a worker thread, so ``check_same_thread=False`` is required.  Foreign-key
    enforcement is enabled per connection (SQLite default is off).
    """

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def create_model(
        self,
        name: str,
        arch: str,
        dropout_p: float,
        dropout_layers: str,
        checkpoint_path: str,
        val_accuracy: float | None = None,
    ) -> int:
        """Insert a model record and return its model_id.

        Args:
            name: Human-readable model name.
            arch: Architecture string (e.g. "resnet18").
            dropout_p: Dropout probability used during MC-Dropout inference.
            dropout_layers: Comma-separated layer names (e.g. "layer2,layer3").
            checkpoint_path: Path to the saved .pth checkpoint.
            val_accuracy: Validation accuracy from training.  Nullable.

        Returns:
            Newly created model_id (integer primary key).
        """
        with self.conn:
            cur = self.conn.execute(
                _INSERT_MODEL,
                (name, arch, dropout_p, dropout_layers, checkpoint_path, val_accuracy),
            )
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(
        self,
        model_id: int,
        dataset_name: str,
        cfg: dict,
        user_id: int | None = None,
    ) -> int:
        """Create a run record and return its run_id.

        Reads reproducibility parameters from *cfg*:
        ``mc_dropout.n_runs``, ``seed``, ``gradcam.iou_percentile``,
        ``gradcam.topk_k``.  These are stored in the run row so every
        reported number is traceable — see CLAUDE.md §1.5.

        Args:
            model_id: FK to the models table.
            dataset_name: Name of the dataset evaluated in this run.
            cfg: Full configuration dict as returned by ``load_config``.
            user_id: Optional FK to users table (nullable).

        Returns:
            Newly created run_id.
        """
        with self.conn:
            cur = self.conn.execute(
                _INSERT_RUN,
                (
                    model_id,
                    user_id,
                    dataset_name,
                    cfg["mc_dropout"]["n_runs"],
                    cfg["seed"],
                    cfg["gradcam"]["iou_percentile"],
                    cfg["gradcam"]["topk_k"],
                ),
            )
        return cur.lastrowid

    def update_run_status(self, run_id: int, status: str) -> None:
        """Set ``status`` and stamp ``finished_at`` for a run.

        Args:
            run_id: The run to update.
            status: One of ``'pending'``, ``'running'``, ``'completed'``,
                    ``'failed'``.
        """
        with self.conn:
            self.conn.execute(_UPDATE_RUN_STATUS, (status, run_id))

    # ------------------------------------------------------------------
    # Image results (images + predictions + explanations — atomic)
    # ------------------------------------------------------------------

    def insert_image_result(
        self,
        run_id: int,
        image_meta: dict,
        prediction: dict,
        explanation: dict,
    ) -> int:
        """Insert one image's analysis result atomically.

        Writes one row each to ``images``, ``predictions``, and
        ``explanations`` inside a single transaction.  If any insert fails
        (e.g. a CHECK constraint violation) the whole transaction rolls back.

        Args:
            run_id: FK to the runs table.
            image_meta: Keys: ``image_path``, ``dataset_type``,
                ``corruption_type`` (optional), ``corruption_severity``
                (optional), ``true_label`` (optional).
            prediction: Keys: ``pred_label``, ``correct``, ``confidence``,
                ``entropy``, ``pred_variance``, ``pred_agreement``,
                ``mutual_information``.
            explanation: Keys: ``cam_corr_mean``, ``cam_corr_std``,
                ``cam_iou_mean``, ``topk_overlap``, ``cam_npz_path``
                (optional), ``mean_png_path`` (optional),
                ``variance_png_path`` (optional).

        Returns:
            The newly created image_id.

        Raises:
            sqlite3.IntegrityError: On CHECK or FK constraint violation
                (e.g. invalid ``dataset_type``).
        """
        with self.conn:
            cur = self.conn.execute(
                _INSERT_IMAGE,
                (
                    run_id,
                    image_meta["image_path"],
                    image_meta["dataset_type"],
                    image_meta.get("corruption_type"),
                    image_meta.get("corruption_severity"),
                    image_meta.get("true_label"),
                ),
            )
            image_id = cur.lastrowid

            self.conn.execute(
                _INSERT_PREDICTION,
                (
                    image_id,
                    prediction["pred_label"],
                    prediction.get("correct"),
                    prediction["confidence"],
                    prediction["entropy"],
                    prediction["pred_variance"],
                    prediction["pred_agreement"],
                    prediction.get("mutual_information"),
                ),
            )

            self.conn.execute(
                _INSERT_EXPLANATION,
                (
                    image_id,
                    explanation["cam_corr_mean"],
                    explanation["cam_corr_std"],
                    explanation["cam_iou_mean"],
                    explanation["topk_overlap"],
                    explanation.get("cam_npz_path"),
                    explanation.get("mean_png_path"),
                    explanation.get("variance_png_path"),
                ),
            )

        return image_id

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def insert_baselines(self, run_id: int, baselines: list[dict]) -> None:
        """Insert baseline records for a run.

        Args:
            run_id: FK to the runs table.
            baselines: List of dicts with keys ``baseline_type``,
                ``metric_name``, ``value``.
        """
        rows = [(run_id, b["baseline_type"], b["metric_name"], b["value"])
                for b in baselines]
        for i in range(0, len(rows), _BATCH_SIZE):
            with self.conn:
                self.conn.executemany(_INSERT_BASELINE, rows[i:i + _BATCH_SIZE])

    # ------------------------------------------------------------------
    # Risk flags
    # ------------------------------------------------------------------

    def upsert_risk_flags(self, run_id: int, flags: list[dict]) -> None:
        """Upsert risk-flag rows for all images in a run.

        Uses INSERT OR REPLACE so it is safe to call again after a re-run.
        Writes in batches of ``_BATCH_SIZE`` rows per transaction.

        Args:
            run_id: Informational — used by callers to scope the flag set;
                not written to the risk_flags table (which stores image_id).
            flags: List of dicts with keys ``image_id``, ``risk_group``,
                and optionally ``risk_score``.
        """
        rows = [(f["image_id"], f["risk_group"], f.get("risk_score"))
                for f in flags]
        for i in range(0, len(rows), _BATCH_SIZE):
            with self.conn:
                self.conn.executemany(_UPSERT_RISK_FLAG, rows[i:i + _BATCH_SIZE])

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------

    def insert_review(
        self,
        image_id: int,
        user_id: int,
        decision: str,
        comment: str | None = None,
    ) -> int:
        """Insert a reviewer decision and return the new review_id.

        Args:
            image_id: FK to the images table.
            user_id: FK to the users table.
            decision: One of ``'accept'``, ``'needs_review'``, ``'reject'``.
            comment: Optional free-text comment.

        Returns:
            Newly created review_id.
        """
        with self.conn:
            cur = self.conn.execute(
                _INSERT_REVIEW, (image_id, user_id, decision, comment)
            )
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def fetch_metrics(
        self, run_id: int, filters: dict | None = None
    ) -> pd.DataFrame:
        """Return the full analysis frame for a run as a pandas DataFrame.

        Uses the canonical full-frame query from DATABASE.md.  Filters
        are expressed as a dict with ``column__op`` keys where ``op`` is
        one of ``gte``, ``lte``, ``eq`` (default when no op is given):

        .. code-block:: python

            repo.fetch_metrics(run_id, {"confidence__gte": 0.9})
            repo.fetch_metrics(run_id, {"pred_agreement": 1.0})
            repo.fetch_metrics(run_id, {"dataset_type": "id"})

        Column names without a ``__op`` suffix use ``=``.

        Args:
            run_id: Restricts results to one run.
            filters: Optional dict of additional WHERE conditions.

        Returns:
            DataFrame with one row per image.  17,000 rows load in well
            under one second with the schema indexes in place.
        """
        # Map filter keys to their SQL column expressions
        _col_map = {
            "confidence":    "p.confidence",
            "entropy":       "p.entropy",
            "pred_variance": "p.pred_variance",
            "pred_agreement":"p.pred_agreement",
            "dataset_type":  "i.dataset_type",
            "correct":       "p.correct",
            "risk_group":    "r.risk_group",
        }
        _op_map = {"gte": ">=", "lte": "<=", "eq": "="}

        sql = _FULL_FRAME
        params: list[Any] = [run_id]

        if filters:
            for key, value in filters.items():
                parts = key.split("__", 1)
                col_key = parts[0]
                op_str = parts[1] if len(parts) == 2 else "eq"
                col = _col_map.get(col_key, col_key)   # fall back to bare name
                op = _op_map.get(op_str, "=")
                sql += f" AND {col} {op} ?"
                params.append(value)

        return pd.read_sql_query(sql, self.conn, params=params)

    def fetch_image_detail(self, image_id: int) -> dict:
        """Return all stored data for one image as a plain dict.

        Joins images, predictions, explanations, and risk_flags (LEFT JOIN).
        Used by the Reviewer image-detail page and the review service.

        Args:
            image_id: PK of the images table.

        Returns:
            Dict of all columns; keys are column names.  Returns an empty
            dict if no row is found.
        """
        row = self.conn.execute(_FETCH_DETAIL, (image_id,)).fetchone()
        if row is None:
            return {}
        return dict(row)

    def fetch_baselines(self, run_id: int) -> dict:
        """Return baseline values for a run, keyed by type and metric.

        Args:
            run_id: FK to the runs table.

        Returns:
            Nested dict ``{baseline_type: {metric_name: value}}``.
            Example: ``{"upper": {"cam_corr_mean": 1.0}, ...}``
        """
        rows = self.conn.execute(_FETCH_BASELINES, (run_id,)).fetchall()
        result: dict[str, dict[str, float]] = {}
        for row in rows:
            result.setdefault(row["baseline_type"], {})[row["metric_name"]] = row["value"]
        return result
