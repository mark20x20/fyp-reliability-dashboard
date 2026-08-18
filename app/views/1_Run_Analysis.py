"""Run Analysis page -- Engineer role only (FR-01, FR-02, FR-03).

Allows an engineer to select a model, choose a dataset, set parameters, and
execute a batch reliability analysis from within the application.  Progress is
displayed via a live progress bar driven by the batch_evaluation progress_cb.
"""

import sys
import time
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis_service import get_service

# ---------------------------------------------------------------------------
# Guard: engineer only
# ---------------------------------------------------------------------------
if st.session_state.get("role") != "engineer":
    st.error("This page is only available to engineers.")
    st.stop()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Run Analysis")
st.caption("Execute a batch reliability pipeline on a dataset split.")

svc = get_service()

# ---------------------------------------------------------------------------
# Configuration form
# ---------------------------------------------------------------------------
models_df = svc.list_models()

if models_df.empty:
    st.error(
        "No models registered in the database.  "
        "Train a model first (src/train_model.py) and register it."
    )
    st.stop()

col_form, col_info = st.columns([2, 1])

with col_form:
    st.subheader("Analysis parameters")

    # Model selector (FR-01)
    model_options = models_df["model_id"].tolist()
    model_labels  = {
        int(r["model_id"]): (
            f"{r['name']}  (p={r['dropout_p']}, "
            f"acc={r['val_accuracy']:.1%})"
            if r["val_accuracy"] is not None
            else f"{r['name']}  (p={r['dropout_p']})"
        )
        for _, r in models_df.iterrows()
    }
    model_id = st.selectbox(
        "Model checkpoint",
        options=model_options,
        format_func=lambda m: model_labels.get(m, str(m)),
        help="Select the registered model to use for MC-Dropout inference.",
    )

    # Dataset selector
    dataset_name = st.selectbox(
        "Dataset",
        options=["imagenette", "imagewoof", "dtd"],
        help="Dataset to evaluate.  imagenette-c (corrupted) is run via CLI.",
    )

    # Dataset type
    dataset_type = st.selectbox(
        "Dataset type",
        options=["id", "near_ood", "far_ood"],
        help=(
            "Label written to images.dataset_type.  "
            "Use id for in-distribution, near_ood / far_ood for OOD datasets."
        ),
    )

    # Limit
    use_limit = st.checkbox("Limit number of images", value=False)
    limit: int | None = None
    if use_limit:
        limit = st.number_input(
            "Max images",
            min_value=1,
            max_value=10_000,
            value=100,
            step=50,
        )

with col_info:
    # FR-02: display current parameters (read-only from config)
    st.subheader("Current config")
    cfg = svc._cfg
    st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| n_runs (MC passes) | `{cfg['mc_dropout']['n_runs']}` |
| dropout_p | `{cfg['model']['dropout_p']}` |
| dropout_layers | `{', '.join(cfg['model']['dropout_layers'])}` |
| IoU percentile | `{cfg['gradcam']['iou_percentile']}` |
| Top-k | `{cfg['gradcam']['topk_k']}` |
| Seed | `{cfg['seed']}` |
""")
    st.caption(
        "Parameters come from configs/config.yaml.  "
        "Edit that file and restart the app to change them."
    )

st.divider()

# ---------------------------------------------------------------------------
# Run button + live progress (FR-03)
# ---------------------------------------------------------------------------
if "run_active" not in st.session_state:
    st.session_state["run_active"] = False
if "run_log"    not in st.session_state:
    st.session_state["run_log"]    = []

run_btn = st.button(
    "Run analysis",
    type="primary",
    disabled=st.session_state["run_active"],
)

if run_btn:
    st.session_state["run_active"] = True
    st.session_state["run_log"]    = []

    n_total_ref = [0]   # mutable container so the closure can write to it
    n_done_ref  = [0]

    progress_bar  = st.progress(0.0, text="Initialising...")
    status_text   = st.empty()
    log_container = st.empty()

    # Log lines visible during the run
    log_lines: list[str] = []

    def _progress_cb(completed: int, total: int) -> None:
        n_done_ref[0]  = completed
        n_total_ref[0] = total
        pct = completed / total if total > 0 else 0.0
        progress_bar.progress(
            pct,
            text=f"Processing {completed} / {total} images ({pct:.0%})"
        )
        # Surface a log line every 50 images to keep I/O light
        if completed % 50 == 0 or completed == total:
            log_lines.append(
                f"[{time.strftime('%H:%M:%S')}]  {completed}/{total} done"
            )
            log_container.code("\n".join(log_lines[-20:]), language=None)

    try:
        t0     = time.perf_counter()
        run_id = svc.run_batch(
            model_id=int(model_id),
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            limit=int(limit) if limit else None,
            progress_cb=_progress_cb,
        )
        elapsed = time.perf_counter() - t0

        progress_bar.progress(1.0, text="Complete!")
        n_img = n_done_ref[0]
        status_text.success(
            f"Run {run_id} completed in {elapsed:.0f} s "
            f"({n_img} images, {elapsed / n_img * 1000:.0f} ms/img)."
        )
        log_lines.append(
            f"[{time.strftime('%H:%M:%S')}]  Run {run_id} completed."
        )
        log_container.code("\n".join(log_lines), language=None)

        # Make the new run active in the sidebar
        st.session_state["run_id"] = run_id

    except Exception as exc:
        progress_bar.progress(0.0, text="Failed")
        status_text.error(f"Run failed: {exc}")
        log_lines.append(f"[{time.strftime('%H:%M:%S')}]  ERROR: {exc}")
        log_container.code("\n".join(log_lines), language=None)

    finally:
        st.session_state["run_active"] = False

# ---------------------------------------------------------------------------
# Recent runs table for reference
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Run history")
runs_df = svc.list_runs()
if not runs_df.empty:
    st.dataframe(
        runs_df[[
            "run_id", "dataset_name", "model_name",
            "n_images", "status", "started_at",
        ]].rename(columns={
            "run_id":      "Run",
            "dataset_name":"Dataset",
            "model_name":  "Model",
            "n_images":    "Images",
            "status":      "Status",
            "started_at":  "Started",
        }),
        use_container_width=True,
        hide_index=True,
    )
