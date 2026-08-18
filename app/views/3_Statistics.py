"""Statistics page -- Engineer role (FR-04, FR-06).

Shows:
  - Baseline reference panel (B1, B2, B3) so metric values are interpretable.
  - Correlation heatmap (E2).
  - Confidence vs cam_iou_mean scatter.
  - Stratified box plot by confidence bin (E3 central result).
  - Quadrant scatter with median split lines (E4).

Figures are generated on first access per run_id and cached to
outputs/figures/run_{run_id}/.  For run_id 11 the figures already exist at
outputs/figures/; they are regenerated into the per-run directory on first
request.
"""

import sys
import traceback
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

run_id = st.session_state.get("run_id")
if run_id is None:
    st.info("Select a completed run from the sidebar.")
    st.stop()

st.title("Statistics")
st.caption(f"Active run: {run_id}")

svc = get_service()

# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
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
# FR-06: Baseline reference panel — shown prominently so values are readable
# ---------------------------------------------------------------------------
st.subheader("Baseline reference")
st.caption(
    "B1 = deterministic upper bound (dropout off).  "
    "B2 = random-map lower bound.  "
    "B3 = cross-image same-class reference."
)

# Structured display: rows = baselines, cols = metrics
_BL_LABELS = {"upper": "B1 (upper)", "lower": "B2 (lower)", "cross_image": "B3 (cross-image)"}
_METRIC_LABELS = {
    "correlation": "cam_corr_mean",
    "cam_corr_mean": "cam_corr_mean",
    "iou": "cam_iou_mean",
    "cam_iou_mean": "cam_iou_mean",
}

bl_rows = []
for btype, label in _BL_LABELS.items():
    m = baselines.get(btype, {})
    corr_key = next((k for k in ["correlation","cam_corr_mean"] if k in m), None)
    iou_key  = next((k for k in ["iou","cam_iou_mean"] if k in m), None)
    bl_rows.append({
        "Baseline":     label,
        "cam_corr_mean": f"{m[corr_key]:.4f}" if corr_key else "—",
        "cam_iou_mean":  f"{m[iou_key]:.4f}"  if iou_key  else "—",
        "Interpretation": {
            "upper":       "Perfect stability (B1=1.000 is the implementation check)",
            "lower":       "Random noise floor — values below this are worse than chance",
            "cross_image": "Cross-image same-class reference — the meaningful floor",
        }.get(btype, ""),
    })

if bl_rows:
    bl_df = pd.DataFrame(bl_rows)
    st.dataframe(bl_df, use_container_width=True, hide_index=True)
else:
    st.info("No baselines stored for this run.  Baselines are computed at the end of run_batch.")

# B3 for quick reference in text
b3_iou  = (baselines.get("cross_image") or {})
b3_iou_val = b3_iou.get("iou") or b3_iou.get("cam_iou_mean")
if b3_iou_val is not None:
    st.info(
        f"B3 cam_iou_mean = {b3_iou_val:.4f}.  "
        "An image in the bottom decile of IoU is below this reference."
    )

st.divider()

# ---------------------------------------------------------------------------
# Figure generation helper — lazy, per-run, cached on disk
# ---------------------------------------------------------------------------
def _figures_dir(run_id: int) -> Path:
    d = _ROOT / f"outputs/figures/run_{run_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@st.cache_data(ttl=600, show_spinner="Generating figures...")
def _ensure_figures(run_id: int) -> dict[str, Path | None]:
    """Generate analysis figures for run_id if not yet on disk.

    Returns a dict mapping figure name -> Path (or None if generation failed).
    """
    from src.analysis import correlation_analysis, quadrant_analysis
    from src.stratified_analysis import stratified_analysis

    out_dir = _figures_dir(run_id)
    df_local = get_service().fetch_metrics(run_id)
    bl_local  = get_service().fetch_baselines(run_id)

    paths: dict[str, Path | None] = {}

    # --- Correlation heatmap + confidence vs IoU scatter ---
    corr_png  = out_dir / "correlation_heatmap.png"
    scatter_png = out_dir / "confidence_vs_cam_iou.png"
    if not corr_png.exists() or not scatter_png.exists():
        try:
            correlation_analysis(df_local, out_dir)
        except Exception:
            pass
    paths["correlation_heatmap"]   = corr_png   if corr_png.exists()   else None
    paths["confidence_vs_cam_iou"] = scatter_png if scatter_png.exists() else None

    # --- Quadrant scatter ---
    quad_png = out_dir / "quadrant_scatter.png"
    if not quad_png.exists():
        try:
            quadrant_analysis(df_local, out_dir)
        except Exception:
            pass
    paths["quadrant_scatter"] = quad_png if quad_png.exists() else None

    # --- Stratified box plot (E3) ---
    strat_png = out_dir / "stratified_by_confidence_bin.png"
    if not strat_png.exists():
        try:
            stratified_analysis(df_local, out_dir, bl_local)
        except Exception:
            pass
    paths["stratified"] = strat_png if strat_png.exists() else None

    return paths


# Run figure generation (cached after first call)
fig_status = st.empty()
fig_status.info("Checking/generating figures...")
try:
    fig_paths = _ensure_figures(int(run_id))
    fig_status.empty()
except Exception as exc:
    fig_status.warning(f"Figure generation error: {exc}")
    fig_paths = {}

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_corr, tab_scatter, tab_e3, tab_quad = st.tabs([
    "Correlation heatmap",
    "Confidence vs IoU",
    "E3 — Confidence bins",
    "E4 — Quadrant",
])

with tab_corr:
    st.subheader("E2 — Metric correlation heatmap")
    st.caption(
        "Pearson correlations between all metric pairs.  "
        "Entropy and cam_iou_mean should be negatively correlated."
    )
    p = fig_paths.get("correlation_heatmap")
    if p and p.exists():
        st.image(str(p), use_column_width=True)
    else:
        # Inline fallback using plotly
        try:
            import plotly.express as px
            num_cols = [
                "confidence","entropy","pred_variance","pred_agreement",
                "cam_corr_mean","cam_iou_mean","topk_overlap",
            ]
            num_cols = [c for c in num_cols if c in df.columns]
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Metric correlation matrix",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not render correlation heatmap: {exc}")

with tab_scatter:
    st.subheader("Confidence vs cam_iou_mean")
    st.caption(
        "Each dot is one image.  Colour = risk group.  "
        "High confidence and low IoU is the hidden-risk region."
    )
    p = fig_paths.get("confidence_vs_cam_iou")
    if p and p.exists():
        st.image(str(p), use_column_width=True)
    else:
        try:
            import plotly.express as px
            plot_df = df[["confidence","cam_iou_mean","risk_group","correct"]].dropna(subset=["confidence","cam_iou_mean"])
            fig = px.scatter(
                plot_df,
                x="confidence",
                y="cam_iou_mean",
                color="risk_group",
                opacity=0.4,
                labels={
                    "confidence":  "Confidence",
                    "cam_iou_mean":"cam_iou_mean",
                    "risk_group":  "Risk group",
                },
                title="Confidence vs cam_iou_mean",
            )
            # Add B3 horizontal reference line
            if b3_iou_val is not None:
                fig.add_hline(
                    y=b3_iou_val,
                    line_dash="dash",
                    line_color="grey",
                    annotation_text=f"B3={b3_iou_val:.3f}",
                )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not render scatter: {exc}")

with tab_e3:
    st.subheader("E3 — Explanation stability stratified by confidence bin")
    st.caption(
        "Subset: pred_agreement == 1.0.  "
        "Within each confidence bin, cam_iou_mean distribution is shown.  "
        "If explanation instability is an independent signal, it should be "
        "present even in the highest-confidence bin."
    )
    p = fig_paths.get("stratified")
    if p and p.exists():
        st.image(str(p), use_column_width=True)
    else:
        # Inline fallback
        try:
            import plotly.express as px
            stable = df[df["pred_agreement"] == 1.0].copy()
            if len(stable) > 0:
                bins   = [0.90, 0.95, 0.99, 1.001]
                labels = ["0.90-0.95", "0.95-0.99", "0.99-1.00"]
                stable["conf_bin"] = pd.cut(
                    stable["confidence"], bins=bins, labels=labels, right=False
                )
                plot_s = stable.dropna(subset=["conf_bin"])
                if len(plot_s) > 0:
                    fig = px.box(
                        plot_s,
                        x="conf_bin",
                        y="cam_iou_mean",
                        title="cam_iou_mean by confidence bin (pred_agreement=1.0)",
                        labels={"conf_bin":"Confidence bin","cam_iou_mean":"cam_iou_mean"},
                    )
                    if b3_iou_val is not None:
                        fig.add_hline(y=b3_iou_val, line_dash="dash", line_color="grey",
                                      annotation_text=f"B3={b3_iou_val:.3f}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data in confidence bins [0.90, 1.00] for pred_agreement=1.0 subset.")
            else:
                st.info("No images with pred_agreement == 1.0 in this run.")
        except Exception as exc:
            st.error(f"Could not render E3 plot: {exc}")

with tab_quad:
    st.subheader("E4 — Quadrant scatter (median split)")
    st.caption(
        "Axes: entropy (prediction uncertainty) and 1-cam_iou_mean "
        "(explanation instability).  Dashed lines = medians.  "
        "Hidden risk = low entropy AND high instability (Q4)."
    )
    p = fig_paths.get("quadrant_scatter")
    if p and p.exists():
        st.image(str(p), use_column_width=True)
    else:
        try:
            import plotly.express as px
            plot_df = df[["entropy","cam_iou_mean","risk_group","correct"]].dropna()
            plot_df = plot_df.copy()
            plot_df["instability"] = 1.0 - plot_df["cam_iou_mean"]
            med_ent  = float(plot_df["entropy"].median())
            med_inst = float(plot_df["instability"].median())

            fig = px.scatter(
                plot_df.sample(min(len(plot_df), 3000), random_state=42),
                x="entropy",
                y="instability",
                color="risk_group",
                opacity=0.4,
                title="Quadrant scatter (entropy vs explanation instability)",
                labels={
                    "entropy":     "Entropy (prediction uncertainty)",
                    "instability": "1 - cam_iou_mean (explanation instability)",
                    "risk_group":  "Risk group",
                },
            )
            fig.add_vline(x=med_ent,  line_dash="dash", line_color="grey",
                          annotation_text=f"med={med_ent:.3f}")
            fig.add_hline(y=med_inst, line_dash="dash", line_color="grey",
                          annotation_text=f"med={med_inst:.3f}")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not render quadrant scatter: {exc}")

# ---------------------------------------------------------------------------
# Run-level summary stats for context
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Descriptive statistics", expanded=False):
    num_cols = [
        "confidence","entropy","pred_variance","pred_agreement",
        "cam_corr_mean","cam_iou_mean","topk_overlap",
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    st.dataframe(
        df[num_cols].describe().T.round(4),
        use_container_width=True,
    )
