"""Risk Queue page -- Reviewer role (FR-08).

Shows all images that meet the hidden-risk rule:
  confidence >= TH_CONF  AND  cam_iou_mean <= TH_IOU

TH_IOU defaults to the run's B3 cross-image baseline so the threshold is
grounded in the actual distribution, not an arbitrary constant (CLAUDE.md 1.8).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis_service import get_service

# ---------------------------------------------------------------------------
# Guard: reviewer only
# ---------------------------------------------------------------------------
if st.session_state.get("role") != "reviewer":
    st.error("This page is only available to reviewers.")
    st.stop()

run_id = st.session_state.get("run_id")
if run_id is None:
    st.info("Select a completed run from the sidebar.")
    st.stop()

st.title("Risk Queue")
st.caption(f"Active run: {run_id}")

svc = get_service()


@st.cache_data(ttl=300, show_spinner="Loading metrics...")
def _load_df(run_id: int) -> pd.DataFrame:
    return get_service().fetch_metrics(run_id)


@st.cache_data(ttl=300, show_spinner=False)
def _load_baselines(run_id: int) -> dict:
    return get_service().fetch_baselines(run_id)


df        = _load_df(int(run_id))
baselines = _load_baselines(int(run_id))

if df.empty:
    st.warning("No data for this run.")
    st.stop()

# ---------------------------------------------------------------------------
# Default TH_IOU from B3 — CLAUDE.md §1.8
# ---------------------------------------------------------------------------
b3      = baselines.get("cross_image", {})
b3_iou  = b3.get("iou") or b3.get("cam_iou_mean")
default_th_iou = round(float(b3_iou), 4) if b3_iou is not None else 0.30

# ---------------------------------------------------------------------------
# Threshold sliders
# ---------------------------------------------------------------------------
st.subheader("Thresholds")
col1, col2 = st.columns(2)
with col1:
    th_conf = st.slider(
        "TH_CONF  (minimum confidence)",
        min_value=0.50, max_value=1.00,
        value=0.90, step=0.01,
        help="Images with confidence >= this value are considered high-confidence.",
    )
with col2:
    th_iou = st.slider(
        "TH_IOU  (maximum cam_iou_mean)",
        min_value=0.00, max_value=1.00,
        value=default_th_iou, step=0.01,
        help=(
            "Images with cam_iou_mean <= this value are considered unstable.  "
            "Default = B3 cross-image baseline."
        ),
    )

if b3_iou is not None:
    st.caption(
        f"B3 cross-image baseline = {b3_iou:.4f}.  "
        "TH_IOU defaults to this value so the threshold is grounded in the run distribution."
    )

# ---------------------------------------------------------------------------
# Apply thresholds and rank
# ---------------------------------------------------------------------------
mask    = (df["confidence"] >= th_conf) & (df["cam_iou_mean"] <= th_iou)
risk_df = df[mask].copy()
# Rank: highest confidence first (most deceptive — confident but unstable)
risk_df = risk_df.sort_values("confidence", ascending=False).reset_index(drop=True)

st.divider()
st.subheader(
    f"Hidden-risk candidates: {len(risk_df):,} of {len(df):,} images"
    f"  ({len(risk_df)/len(df):.1%})"
)
st.info(
    f"Rule: confidence >= {th_conf:.2f}  AND  cam_iou_mean <= {th_iou:.4f}"
    + (f"  (B3 = {b3_iou:.4f})" if b3_iou is not None else "")
)

if risk_df.empty:
    st.warning("No images match the current thresholds.  Try lowering TH_CONF or raising TH_IOU.")
    st.stop()

# ---------------------------------------------------------------------------
# Ranked table with row-selection
# ---------------------------------------------------------------------------
display_cols = [c for c in [
    "image_id", "true_label", "pred_label", "correct",
    "confidence", "cam_iou_mean", "cam_corr_mean", "entropy",
    "pred_agreement", "risk_group",
] if c in risk_df.columns]

event = st.dataframe(
    risk_df[display_cols],
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "image_id":      st.column_config.NumberColumn("ID",           format="%d"),
        "confidence":    st.column_config.NumberColumn("Confidence",   format="%.4f"),
        "cam_iou_mean":  st.column_config.NumberColumn("cam_iou_mean", format="%.4f"),
        "cam_corr_mean": st.column_config.NumberColumn("cam_corr_mean",format="%.4f"),
        "entropy":       st.column_config.NumberColumn("Entropy",      format="%.4f"),
        "pred_agreement":st.column_config.NumberColumn("pred_agree",   format="%.2f"),
        "correct":       st.column_config.CheckboxColumn("Correct"),
    },
)

# ---------------------------------------------------------------------------
# Row selection → navigate to Image Detail
# ---------------------------------------------------------------------------
sel_rows = (
    event.selection.get("rows", [])
    if hasattr(event, "selection") and event.selection
    else []
)

if sel_rows:
    row_idx = sel_rows[0]
    selected_image_id = int(risk_df.iloc[row_idx]["image_id"])
    st.session_state["image_id"] = selected_image_id

    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Open Image Detail", type="primary"):
            # Absolute path ensures Streamlit matches the registered page
            # regardless of CWD vs main-script dir resolution.
            _target = str(Path(__file__).with_name("5_Image_Detail.py"))
            st.switch_page(_target)
    with col_b:
        st.caption(
            f"Selected image_id = {selected_image_id}  |  "
            f"confidence = {float(risk_df.iloc[row_idx]['confidence']):.4f}  |  "
            f"cam_iou_mean = {float(risk_df.iloc[row_idx]['cam_iou_mean']):.4f}"
        )
