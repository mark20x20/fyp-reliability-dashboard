"""Metrics Table page -- Engineer role (FR-05, FR-07).

Sortable, filterable table of all image-level metrics for the active run.
Row selection opens an inline detail panel.  CSV export is available.

Performance target: 10,000 rows render in under 3 s (st.dataframe uses
client-side virtualisation and handles this natively).
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
# Guard: engineer only
# ---------------------------------------------------------------------------
if st.session_state.get("role") != "engineer":
    st.error("This page is only available to engineers.")
    st.stop()

# ---------------------------------------------------------------------------
# Require an active run
# ---------------------------------------------------------------------------
run_id = st.session_state.get("run_id")
if run_id is None:
    st.info("Select a completed run from the sidebar.")
    st.stop()

st.title("Metrics Table")
st.caption(f"Active run: {run_id}")

svc = get_service()

# ---------------------------------------------------------------------------
# Load data (cached per run_id; filters applied in pandas for UI speed)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Loading metrics...")
def _load_df(run_id: int) -> pd.DataFrame:
    return get_service().fetch_metrics(run_id)


df_full = _load_df(int(run_id))

if df_full.empty:
    st.warning("No images found for this run.")
    st.stop()

# ---------------------------------------------------------------------------
# Filter sidebar (FR-05)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Filters")

    dtype_opts = ["(all)"] + sorted(df_full["dataset_type"].dropna().unique().tolist())
    sel_dtype  = st.selectbox("Dataset type", dtype_opts, key="mt_dtype")

    correct_map = {"(all)": None, "Correct only": 1, "Incorrect only": 0}
    sel_correct = st.selectbox("Prediction", list(correct_map.keys()), key="mt_correct")

    rg_vals = sorted(df_full["risk_group"].dropna().unique().tolist())
    rg_opts = ["(all)"] + rg_vals
    sel_rg  = st.selectbox("Risk group", rg_opts, key="mt_rg")

    conf_lo = float(df_full["confidence"].min())
    conf_hi = float(df_full["confidence"].max())
    sel_conf = st.slider(
        "Confidence",
        min_value=round(conf_lo, 2),
        max_value=round(conf_hi, 2),
        value=(round(conf_lo, 2), round(conf_hi, 2)),
        step=0.01,
        key="mt_conf",
    )

    iou_lo = float(df_full["cam_iou_mean"].min())
    iou_hi = float(df_full["cam_iou_mean"].max())
    sel_iou = st.slider(
        "cam_iou_mean",
        min_value=round(iou_lo, 2),
        max_value=round(iou_hi, 2),
        value=(round(iou_lo, 2), round(iou_hi, 2)),
        step=0.01,
        key="mt_iou",
    )

    corrupt_types = sorted(df_full["corruption_type"].dropna().unique().tolist())
    if corrupt_types:
        ct_opts = ["(all)"] + corrupt_types
        sel_ct  = st.selectbox("Corruption type", ct_opts, key="mt_ct")
        sev_vals = sorted(df_full["corruption_severity"].dropna().astype(int).unique().tolist())
        sev_opts = ["(all)"] + [str(s) for s in sev_vals]
        sel_sev  = st.selectbox("Corruption severity", sev_opts, key="mt_sev")
    else:
        sel_ct  = "(all)"
        sel_sev = "(all)"

    st.divider()
    if st.button("Reset filters", key="mt_reset"):
        for k in ["mt_dtype","mt_correct","mt_rg","mt_conf","mt_iou","mt_ct","mt_sev"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
df = df_full.copy()

if sel_dtype != "(all)":
    df = df[df["dataset_type"] == sel_dtype]

cv = correct_map[sel_correct]
if cv is not None:
    df = df[df["correct"] == cv]

if sel_rg != "(all)":
    df = df[df["risk_group"] == sel_rg]

df = df[df["confidence"].between(sel_conf[0], sel_conf[1])]
df = df[df["cam_iou_mean"].between(sel_iou[0], sel_iou[1])]

if sel_ct != "(all)":
    df = df[df["corruption_type"] == sel_ct]

if sel_sev not in ("(all)", None):
    df = df[df["corruption_severity"] == int(sel_sev)]

# ---------------------------------------------------------------------------
# Summary bar
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows (filtered)", f"{len(df):,}")
c2.metric("Rows (total)",    f"{len(df_full):,}")

acc_val = df["correct"].mean() if df["correct"].notna().any() and len(df) > 0 else None
c3.metric("Accuracy", f"{acc_val:.1%}" if acc_val is not None else "N/A")

iou_val = df["cam_iou_mean"].mean() if len(df) > 0 else None
c4.metric("Mean cam_iou", f"{iou_val:.3f}" if iou_val is not None else "N/A")

# ---------------------------------------------------------------------------
# Table (FR-05) with single-row selection for drill-down
# ---------------------------------------------------------------------------
DISPLAY_COLS = [
    "image_id", "dataset_type", "true_label", "pred_label", "correct",
    "confidence", "entropy", "pred_agreement",
    "cam_corr_mean", "cam_iou_mean", "topk_overlap",
    "risk_group", "corruption_type", "corruption_severity",
]
show_cols = [c for c in DISPLAY_COLS if c in df.columns]

st.subheader(f"Results ({len(df):,} rows)")

event = st.dataframe(
    df[show_cols].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
    height=480,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "image_id":    st.column_config.NumberColumn("ID",         format="%d"),
        "correct":     st.column_config.NumberColumn("Correct",    format="%d"),
        "confidence":  st.column_config.NumberColumn("Confidence", format="%.3f"),
        "entropy":     st.column_config.NumberColumn("Entropy",    format="%.4f"),
        "pred_agreement":  st.column_config.NumberColumn("Pred agr", format="%.3f"),
        "cam_corr_mean":   st.column_config.NumberColumn("CAM corr", format="%.4f"),
        "cam_iou_mean":    st.column_config.NumberColumn("CAM IoU",  format="%.4f"),
        "topk_overlap":    st.column_config.NumberColumn("Top-k",    format="%.4f"),
        "corruption_severity": st.column_config.NumberColumn("Sev", format="%d"),
    },
)

# ---------------------------------------------------------------------------
# Inline detail panel when a row is selected
# ---------------------------------------------------------------------------
sel_rows = event.selection.rows if event.selection else []
if sel_rows:
    row      = df.iloc[sel_rows[0]]
    image_id = int(row["image_id"])
    st.session_state["image_id"] = image_id

    st.divider()
    with st.expander(f"Detail — Image {image_id}", expanded=True):
        detail = svc.fetch_image_detail(image_id)
        if not detail:
            st.warning("Detail not found.")
        else:
            meta_col, metric_col = st.columns(2)
            with meta_col:
                st.markdown("**Classification**")
                st.markdown(f"""
| Field | Value |
|-------|-------|
| True label | `{detail.get('true_label','N/A')}` |
| Pred label | `{detail.get('pred_label','N/A')}` |
| Correct | `{detail.get('correct','N/A')}` |
| Risk group | `{detail.get('risk_group','N/A')}` |
| Dataset type | `{detail.get('dataset_type','N/A')}` |
""")

            with metric_col:
                st.markdown("**Metrics**")
                baselines = detail.get("baselines", {})
                b3_iou = (baselines.get("cross_image") or {}).get("iou")
                b3_corr = (baselines.get("cross_image") or {}).get("correlation")

                iou_val  = detail.get("cam_iou_mean",  0)
                corr_val = detail.get("cam_corr_mean", 0)
                iou_delta  = f"{iou_val  - b3_iou :.3f} vs B3" if b3_iou  else None
                corr_delta = f"{corr_val - b3_corr:.3f} vs B3" if b3_corr else None

                st.markdown(f"""
| Metric | Value | vs B3 |
|--------|-------|-------|
| Confidence | `{detail.get('confidence',0):.4f}` | |
| Entropy | `{detail.get('entropy',0):.4f}` | |
| Pred agreement | `{detail.get('pred_agreement',0):.4f}` | |
| cam_corr_mean | `{corr_val:.4f}` | {corr_delta or ''} |
| cam_iou_mean | `{iou_val:.4f}` | {iou_delta or ''} |
| topk_overlap | `{detail.get('topk_overlap',0):.4f}` | |
""")

            # Grad-CAM images
            mean_png = detail.get("mean_png_path")
            var_png  = detail.get("variance_png_path")
            if mean_png and Path(mean_png).exists():
                img_c1, img_c2 = st.columns(2)
                img_c1.image(mean_png,  caption="Mean Grad-CAM",   use_column_width=True)
                if var_png and Path(var_png).exists():
                    img_c2.image(var_png, caption="Variability map", use_column_width=True)

# ---------------------------------------------------------------------------
# CSV export (FR-07)
# ---------------------------------------------------------------------------
st.divider()
csv_bytes = svc.export_metrics_csv(int(run_id))
st.download_button(
    label=f"Export run {run_id} metrics as CSV",
    data=csv_bytes,
    file_name=f"reliability_run{run_id}.csv",
    mime="text/csv",
)
