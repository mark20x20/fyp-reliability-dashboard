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

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------

    def _get_or_create_reviewer(self) -> int:
        """Return user_id of the default reviewer account, creating it if absent."""
        row = self._repo.conn.execute(
            "SELECT user_id FROM users WHERE username = 'reviewer'"
        ).fetchone()
        if row:
            return row["user_id"]
        with self._repo.conn:
            cur = self._repo.conn.execute(
                "INSERT INTO users (username, role) VALUES (?, ?)",
                ("reviewer", "reviewer"),
            )
        return cur.lastrowid

    def insert_review(
        self,
        image_id: int,
        decision: str,
        comment: str | None = None,
    ) -> int:
        """Persist a reviewer decision.  Returns the new review_id.

        Creates the default reviewer account if the users table is empty
        (first-run convenience so the UI does not need a sign-in flow).

        Args:
            image_id: FK to the images table.
            decision: One of ``'accept'``, ``'needs_review'``, ``'reject'``.
            comment: Optional free-text annotation.

        Returns:
            Newly created review_id.
        """
        user_id = self._get_or_create_reviewer()
        return self._repo.insert_review(image_id, user_id, decision, comment)

    def fetch_reviews(self, image_id: int) -> pd.DataFrame:
        """Return all reviews for one image, newest first.

        Args:
            image_id: PK of the images table.

        Returns:
            DataFrame with columns review_id, decision, comment, created_at,
            username.  Empty DataFrame if no reviews exist.
        """
        return pd.read_sql_query(
            """
            SELECT rv.review_id, rv.decision, rv.comment,
                   rv.created_at, u.username
            FROM reviews rv
            JOIN users u ON u.user_id = rv.user_id
            WHERE rv.image_id = ?
            ORDER BY rv.created_at DESC
            """,
            self._repo.conn,
            params=(image_id,),
        )

    # ------------------------------------------------------------------
    # Single-image upload pipeline (FR-11)
    # ------------------------------------------------------------------

    def run_uploaded_image(
        self,
        pil_image,
        model_id: int | None = None,
    ) -> tuple[int, int]:
        """Run the full pipeline on one uploaded PIL image.

        Loads the requested model checkpoint, preprocesses the image,
        runs N MC-Dropout passes, computes all metrics, saves artefacts,
        writes the result to the database under ``dataset_type='uploaded'``,
        and returns identifiers so the caller can display Image Detail.

        Args:
            pil_image: ``PIL.Image.Image`` in any mode (converted to RGB).
            model_id: PK of the model to use.  Defaults to the first model
                      registered if not specified.

        Returns:
            ``(run_id, image_id)`` tuple.

        Raises:
            RuntimeError: If no models are in the database.
            ValueError: If *model_id* is not found.
        """
        import numpy as np
        import torch
        import torchvision.transforms as T

        from models.resnet_dropout import build_mc_dropout_resnet18, enable_mc_dropout
        from src import metrics_explanation as _mex
        from src import metrics_prediction as _mpred
        from src.gradcam_generator import generate_repeated_cams
        from src.inference_mc_dropout import mc_dropout_forward
        from src.utils import ensure_dirs, get_device, set_seed

        def _save_heatmap(arr: np.ndarray, path: "Path") -> None:
            import cv2
            lo, hi = float(arr.min()), float(arr.max())
            if hi > lo:
                norm = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                norm = np.zeros_like(arr, dtype=np.uint8)
            cv2.imwrite(str(path), cv2.applyColorMap(norm, cv2.COLORMAP_JET))

        from pathlib import Path

        set_seed(self._cfg["seed"])
        ensure_dirs()
        device = get_device()

        # Resolve model
        if model_id is None:
            models_df = self.list_models()
            if models_df.empty:
                raise RuntimeError("No models registered in the database.")
            model_id = int(models_df.iloc[0]["model_id"])

        row = self._repo.conn.execute(
            "SELECT checkpoint_path FROM models WHERE model_id = ?", (model_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"model_id={model_id} not found in the models table.")

        cfg = self._cfg
        model = build_mc_dropout_resnet18(
            num_classes=cfg["model"]["num_classes"],
            p=cfg["model"]["dropout_p"],
        )
        model.load_state_dict(
            torch.load(row["checkpoint_path"], map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()

        # Create an 'uploaded' run
        run_id = self._repo.create_run(model_id, "uploaded", cfg)

        # Preprocess
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(pil_image.convert("RGB")).to(device)

        # MC-Dropout inference
        n_runs = cfg["mc_dropout"]["n_runs"]
        probs_t = mc_dropout_forward(model, img_tensor, n_runs)
        probs = probs_t.cpu().numpy()

        # Prediction metrics — no ground-truth label for uploaded images
        num_classes = cfg["model"]["num_classes"]
        class_names = [str(i) for i in range(num_classes)]
        pred_met = _mpred.compute_all(probs, None, class_names)

        # Fixed target class
        target_class = int(probs.mean(axis=0).argmax())

        # Grad-CAM
        enable_mc_dropout(model)
        cams_raw = generate_repeated_cams(model, img_tensor, target_class, n_runs)

        # Explanation metrics
        expl_met = _mex.compute_all(cams_raw, cfg)

        # Artefacts
        raw_dir = Path(cfg["paths"]["outputs"]) / "gradcam" / "raw"
        png_dir = Path(cfg["paths"]["outputs"]) / "gradcam" / "png"
        uploads_dir = Path(cfg["paths"]["outputs"]) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        tag = f"upload_{run_id}_0"
        npz_path = raw_dir / f"{tag}.npz"
        mean_png = png_dir / f"{tag}_mean.png"
        var_png  = png_dir / f"{tag}_var.png"
        img_save = uploads_dir / f"{tag}.png"

        np.savez_compressed(str(npz_path), cams=cams_raw.astype(np.float16))
        pil_image.convert("RGB").save(str(img_save))

        upsampled = _mex.upsample_cams(cams_raw, size=cfg["gradcam"]["upsample_size"])
        _save_heatmap(upsampled.mean(axis=0), mean_png)
        _save_heatmap(_mex.variability_map(upsampled), var_png)

        rep_indices = np.linspace(0, n_runs - 1, 6, dtype=int)
        for k, ri in enumerate(rep_indices):
            _save_heatmap(upsampled[ri], png_dir / f"{tag}_rep{k}.png")

        # Atomic insert
        image_id = self._repo.insert_image_result(
            run_id=run_id,
            image_meta={
                "image_path":  str(img_save.resolve()),
                "dataset_type": "uploaded",
                "true_label":  None,
            },
            prediction={
                "pred_label":         pred_met["pred_label"],
                "correct":            None,
                "confidence":         pred_met["confidence"],
                "entropy":            pred_met["entropy"],
                "pred_variance":      pred_met["pred_variance"],
                "pred_agreement":     pred_met["pred_agreement"],
                "mutual_information": pred_met["mutual_information"],
            },
            explanation={
                "cam_corr_mean": expl_met["cam_corr_mean"],
                "cam_corr_std":  expl_met["cam_corr_std"],
                "cam_iou_mean":  expl_met["cam_iou_mean"],
                "topk_overlap":  expl_met["topk_overlap"],
                "cam_npz_path":      str(npz_path.resolve()),
                "mean_png_path":     str(mean_png.resolve()),
                "variance_png_path": str(var_png.resolve()),
            },
        )

        # Simple risk flag for a single image (no median split possible)
        th_conf = float(cfg.get("risk_flag", {}).get("confidence_threshold") or 0.90)
        th_iou  = float(cfg.get("risk_flag", {}).get("iou_threshold")  or 0.30)
        conf = pred_met["confidence"]
        iou  = expl_met["cam_iou_mean"]
        if conf >= th_conf and iou <= th_iou:
            risk_group = "hidden_risk"
        elif conf < th_conf and iou <= th_iou:
            risk_group = "unstable_both"
        else:
            risk_group = "stable"
        self._repo.upsert_risk_flags(run_id, [
            {"image_id": image_id, "risk_group": risk_group, "risk_score": None}
        ])

        self._repo.update_run_status(run_id, "completed")
        return run_id, image_id

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
