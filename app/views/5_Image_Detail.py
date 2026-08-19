"""Image Detail page -- Reviewer role (FR-09, FR-10).

Shows the full analysis for one image:
  - Input image + label comparison
  - Metric panel with B1/B2/B3 baseline deltas
  - Six representative Grad-CAM maps (loaded from disk PNGs or rendered from .npz)
  - Mean map and variability map
  - Rule-based interpretation referencing the baselines
  - Review decision form (accept / needs_review / reject + comment)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

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

svc = get_service()


# ---------------------------------------------------------------------------
# Image-id selector
# ---------------------------------------------------------------------------
st.title("Image Detail")

# Allow manual entry if not coming from Risk Queue
if "image_id" not in st.session_state or st.session_state["image_id"] is None:
    entered = st.number_input("Enter image_id", min_value=1, step=1, value=1)
    if st.button("Load"):
        st.session_state["image_id"] = int(entered)
        st.rerun()
    st.stop()

image_id = int(st.session_state["image_id"])

# Clear button in sidebar area
col_hdr, col_clr = st.columns([4, 1])
with col_hdr:
    st.caption(f"image_id = {image_id}")
with col_clr:
    if st.button("Clear", help="Go back to Risk Queue"):
        st.session_state["image_id"] = None
        st.rerun()


# ---------------------------------------------------------------------------
# Cached data load
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Loading image data...")
def _load_detail(image_id: int) -> dict:
    return get_service().fetch_image_detail(image_id)


detail = _load_detail(image_id)
if not detail:
    st.error(f"image_id {image_id} not found in the database.")
    st.stop()

baselines  = detail.get("baselines", {})
b1         = baselines.get("upper",       {})
b3         = baselines.get("cross_image", {})
b3_iou     = b3.get("iou") or b3.get("cam_iou_mean")
b3_corr    = b3.get("correlation") or b3.get("cam_corr_mean")
b1_iou     = b1.get("iou") or b1.get("cam_iou_mean")
b1_corr    = b1.get("correlation") or b1.get("cam_corr_mean")

conf_val   = float(detail.get("confidence", 0) or 0)
iou_val    = float(detail.get("cam_iou_mean", 0) or 0)
corr_val   = float(detail.get("cam_corr_mean", 0) or 0)
risk_group = detail.get("risk_group", "")

# ---------------------------------------------------------------------------
# Top row: image + labels + risk
# ---------------------------------------------------------------------------
col_img, col_lbl, col_risk = st.columns([1, 2, 1])

with col_img:
    img_path_str = detail.get("image_path", "")
    if img_path_str and Path(img_path_str).exists():
        try:
            pil = Image.open(img_path_str).convert("RGB")
            st.image(pil, caption="Input image", use_column_width=True)
        except Exception as exc:
            st.warning(f"Cannot load image file: {exc}")
    else:
        st.info("Image file not available on disk.")

with col_lbl:
    st.subheader("Labels")
    st.metric("True label",      detail.get("true_label", "—") or "—")
    st.metric("Predicted label", detail.get("pred_label", "—") or "—")
    correct = detail.get("correct")
    if correct is None:
        st.caption("Correctness: N/A (no ground truth)")
    elif int(correct):
        st.success("Correct prediction")
    else:
        st.error("Incorrect prediction")

    if detail.get("dataset_type"):
        st.caption(f"Dataset type: {detail['dataset_type']}")
    if detail.get("corruption_type"):
        st.caption(
            f"Corruption: {detail['corruption_type']}, "
            f"severity {detail.get('corruption_severity', '?')}"
        )

with col_risk:
    st.subheader("Risk")
    _risk_colours = {
        "hidden_risk":        ":red[hidden_risk]",
        "unstable_both":      ":orange[unstable_both]",
        "pred_unstable_only": ":blue[pred_unstable_only]",
        "stable":             ":green[stable]",
    }
    st.markdown(_risk_colours.get(risk_group, risk_group or "—"))
    if detail.get("risk_score") is not None:
        st.metric("Risk score", f"{detail['risk_score']:.4f}")

st.divider()

# ---------------------------------------------------------------------------
# Metric panel — each metric shown against its B1/B3 baseline (FR-09)
# ---------------------------------------------------------------------------
st.subheader("Metrics")
st.caption("Deltas are relative to B3 (cross-image same-class reference).")

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Confidence",    f"{conf_val:.4f}")
mc2.metric("Entropy",       f"{detail.get('entropy', 0):.4f}")
mc3.metric("pred_variance", f"{detail.get('pred_variance', 0):.4f}")
mc4.metric("pred_agreement",f"{detail.get('pred_agreement', 0):.4f}")

ec1, ec2, ec3 = st.columns(3)
ec1.metric(
    "cam_iou_mean",
    f"{iou_val:.4f}",
    delta=f"{iou_val - b3_iou:+.4f} vs B3" if b3_iou is not None else None,
)
ec2.metric(
    "cam_corr_mean",
    f"{corr_val:.4f}",
    delta=f"{corr_val - b3_corr:+.4f} vs B3" if b3_corr is not None else None,
)
ec3.metric("topk_overlap",  f"{detail.get('topk_overlap', 0):.4f}")

with st.expander("Baseline reference (B1 / B2 / B3)", expanded=False):
    bl_rows = []
    for btype, label, interp in [
        ("upper",       "B1 (upper)",       "Deterministic bound — implementation check, should equal 1.0"),
        ("lower",       "B2 (lower)",       "Random map floor — values below this are worse than chance"),
        ("cross_image", "B3 (cross-image)", "Same-class cross-image reference — the meaningful floor"),
    ]:
        m = baselines.get(btype, {})
        ck = next((k for k in ["correlation", "cam_corr_mean"] if k in m), None)
        ik = next((k for k in ["iou", "cam_iou_mean"] if k in m), None)
        bl_rows.append({
            "Baseline":      label,
            "cam_corr_mean": f"{m[ck]:.4f}" if ck else "—",
            "cam_iou_mean":  f"{m[ik]:.4f}" if ik else "—",
            "Interpretation": interp,
        })
    if bl_rows:
        st.dataframe(pd.DataFrame(bl_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No baselines stored for this run.")

st.divider()

# ---------------------------------------------------------------------------
# Rule-based interpretation text (FR-09)
# ---------------------------------------------------------------------------
st.subheader("Interpretation")

if b3_iou is not None:
    th_conf = float(svc._cfg.get("risk_flag", {}).get("confidence_threshold") or 0.90)

    if conf_val >= th_conf and iou_val <= b3_iou:
        st.warning(
            f"REVIEW PRIORITY: confidence = {conf_val:.4f} (>= {th_conf:.2f}) "
            f"and cam_iou_mean = {iou_val:.4f} (<= B3 = {b3_iou:.4f}).  "
            "The model is highly confident yet its attention region shifts between "
            "MC-Dropout passes, suggesting the prediction may not rest on stable "
            "visual evidence."
        )
    elif risk_group == "hidden_risk":
        st.warning(
            "Flagged as hidden_risk: low prediction entropy (confident) alongside "
            "high explanation instability.  The gradient-weighted attention map "
            "is inconsistent across stochastic passes even though the predicted "
            "class does not change."
        )
    elif risk_group == "stable":
        st.success(
            "Classified as stable: prediction uncertainty is low and explanation "
            "maps are consistent across MC-Dropout passes.  cam_iou_mean "
            f"({iou_val:.4f}) is above B3 ({b3_iou:.4f})."
        )
    elif risk_group == "unstable_both":
        st.info(
            "Both prediction and explanation are unstable: the model is uncertain "
            "and the Grad-CAM regions change between passes.  This is expected "
            "behaviour for hard or ambiguous images."
        )
    else:
        st.info(
            "Prediction uncertainty is elevated but explanation maps are "
            f"relatively consistent (cam_iou_mean = {iou_val:.4f}).  "
            "The model is uncertain but attends to the same region consistently."
        )
else:
    st.info(
        "No baselines available for this run; qualitative interpretation "
        f"only.  Risk group: {risk_group or 'N/A'}."
    )

st.divider()

# ---------------------------------------------------------------------------
# Grad-CAM visualisations (FR-09)
# ---------------------------------------------------------------------------
st.subheader("Grad-CAM maps")

tab_rep, tab_mv = st.tabs(["6 representative maps", "Mean & Variability"])

with tab_rep:
    st.caption(
        "Evenly spaced across the 30 MC-Dropout passes.  Variability across "
        "the six maps indicates explanation instability."
    )

    npz_path_str = detail.get("cam_npz_path")
    shown = False

    if npz_path_str:
        npz_path = Path(npz_path_str)
        base     = npz_path.stem
        png_dir  = npz_path.parent.parent / "png"
        rep_pngs = [png_dir / f"{base}_rep{k}.png" for k in range(6)]
        existing = [p for p in rep_pngs if p.exists()]

        if existing:
            cols = st.columns(3)
            for k, p in enumerate(existing):
                with cols[k % 3]:
                    st.image(str(p), caption=f"Map {k + 1}",
                             use_column_width=True)
            shown = True

    if not shown:
        # Fallback: render heatmaps inline from the .npz array
        cams_arr = detail.get("cams_array")
        if cams_arr is not None and len(cams_arr) > 0:
            import cv2
            n = cams_arr.shape[0]
            indices = np.linspace(0, n - 1, 6, dtype=int)
            cols = st.columns(3)
            for k, idx in enumerate(indices):
                cam = cams_arr[idx].astype(np.float32)
                lo, hi = float(cam.min()), float(cam.max())
                if hi > lo:
                    norm = ((cam - lo) / (hi - lo) * 255).astype(np.uint8)
                else:
                    norm = np.zeros_like(cam, dtype=np.uint8)
                coloured = cv2.applyColorMap(
                    cv2.resize(norm, (224, 224), interpolation=cv2.INTER_LINEAR),
                    cv2.COLORMAP_JET,
                )
                with cols[k % 3]:
                    st.image(
                        cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB),
                        caption=f"Map {k + 1} (rendered)",
                        use_column_width=True,
                    )
        else:
            st.info("Grad-CAM maps not available for this image.")

with tab_mv:
    m1, m2 = st.columns(2)
    with m1:
        mean_path = detail.get("mean_png_path")
        if mean_path and Path(mean_path).exists():
            st.image(mean_path, caption="Mean Grad-CAM (across all passes)",
                     use_column_width=True)
        else:
            st.info("Mean map file not available.")
    with m2:
        var_path = detail.get("variance_png_path")
        if var_path and Path(var_path).exists():
            st.image(var_path, caption="Variability map (pixel-wise std)",
                     use_column_width=True)
        else:
            st.info("Variability map file not available.")

st.divider()

# ---------------------------------------------------------------------------
# Review decision form (FR-10)
# ---------------------------------------------------------------------------
st.subheader("Review Decision")

# Show any previously submitted reviews (no caching — always fresh)
existing_reviews = svc.fetch_reviews(image_id)
if not existing_reviews.empty:
    st.caption(f"Previous reviews for this image ({len(existing_reviews)}):")
    st.dataframe(
        existing_reviews[["decision", "comment", "created_at"]],
        hide_index=True,
        use_container_width=True,
    )

with st.form("review_form"):
    decision = st.radio(
        "Decision",
        options=["accept", "needs_review", "reject"],
        horizontal=True,
        help=(
            "accept: model behaviour is acceptable for this image.  "
            "needs_review: flag for further expert inspection.  "
            "reject: the prediction should not be trusted."
        ),
    )
    comment = st.text_area("Comment (optional)", max_chars=500, height=80)
    submitted = st.form_submit_button("Submit", type="primary")

if submitted:
    try:
        review_id = svc.insert_review(image_id, decision=decision,
                                      comment=comment.strip() or None)
        st.success(f"Review saved (review_id = {review_id}).")
        # Invalidate the detail cache so fresh reviews appear on next load
        _load_detail.clear()
        st.rerun()
    except Exception as exc:
        st.error(f"Failed to save review: {exc}")
