"""Upload & Analyse page -- Reviewer role (FR-11).

Accepts an arbitrary image file, runs the full MC-Dropout + Grad-CAM pipeline
on it, stores the result with dataset_type='uploaded', and renders the same
Image Detail view inline.
"""

import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

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

st.title("Upload & Analyse")
st.caption(
    "Upload any image to run MC-Dropout inference and Grad-CAM stability analysis "
    "on it in real time."
)

svc = get_service()

# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------
models_df = svc.list_models()
if models_df.empty:
    st.error(
        "No models registered in the database.  "
        "An engineer must train and register a model before this page can be used."
    )
    st.stop()

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
    "Model to use",
    options=model_options,
    format_func=lambda m: model_labels.get(m, str(m)),
)

# ---------------------------------------------------------------------------
# File uploader with validation (TC22)
# ---------------------------------------------------------------------------
st.subheader("Upload image")
uploaded = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"],
    help="Non-image files are rejected with a clear message.",
)

if uploaded is None:
    st.info("Upload an image file to begin analysis.")
    st.stop()

# Validate that the file is actually a readable image
try:
    raw_bytes = uploaded.read()
    buf = io.BytesIO(raw_bytes)
    pil_image = Image.open(buf)
    pil_image.verify()          # raises UnidentifiedImageError for non-images
    # Re-open: verify() closes the internal file pointer
    pil_image = Image.open(io.BytesIO(raw_bytes))
except (UnidentifiedImageError, Exception) as exc:
    st.error(
        f"The uploaded file is not a valid image and cannot be analysed.  "
        f"Please upload a JPEG, PNG, or other supported image format.  "
        f"({type(exc).__name__}: {exc})"
    )
    st.stop()

# Show preview
col_prev, col_info = st.columns([1, 2])
with col_prev:
    st.image(pil_image.convert("RGB"), caption=f"{uploaded.name}", use_container_width=True)
with col_info:
    w, h = pil_image.size
    st.markdown(f"""
| Property | Value |
|----------|-------|
| Filename | `{uploaded.name}` |
| Format   | `{pil_image.format or 'unknown'}` |
| Size     | {w} × {h} px |
| Mode     | `{pil_image.mode}` |
""")
    n_runs = svc._cfg.get("mc_dropout", {}).get("n_runs", 30)
    st.caption(
        f"Pipeline: {n_runs} MC-Dropout passes, Grad-CAM at layer4, "
        "all metrics stored to DB."
    )

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
st.divider()

if "upload_result" not in st.session_state:
    st.session_state["upload_result"] = None

run_btn = st.button("Analyse image", type="primary")

if run_btn:
    st.session_state["upload_result"] = None
    with st.spinner("Running MC-Dropout inference and Grad-CAM analysis..."):
        try:
            run_id, image_id = svc.run_uploaded_image(
                pil_image, model_id=int(model_id)
            )
            st.session_state["upload_result"] = (run_id, image_id)
            st.session_state["image_id"] = image_id
            st.success(
                f"Analysis complete.  run_id = {run_id},  image_id = {image_id}"
            )
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Inline results — mirrors Image Detail after a successful run
# ---------------------------------------------------------------------------
result = st.session_state.get("upload_result")
if result is None:
    st.stop()

run_id, image_id = result

st.divider()
st.subheader("Analysis results")

detail = svc.fetch_image_detail(image_id)
if not detail:
    st.warning("Results not found in the database.  This should not happen.")
    st.stop()

baselines = detail.get("baselines", {})
b3        = baselines.get("cross_image", {})
b3_iou    = b3.get("iou") or b3.get("cam_iou_mean")
b3_corr   = b3.get("correlation") or b3.get("cam_corr_mean")

conf_val  = float(detail.get("confidence",    0) or 0)
iou_val   = float(detail.get("cam_iou_mean",  0) or 0)
corr_val  = float(detail.get("cam_corr_mean", 0) or 0)

# Summary metrics
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Predicted class", detail.get("pred_label", "—"))
mc2.metric("Confidence",       f"{conf_val:.4f}")
mc3.metric("Entropy",          f"{detail.get('entropy', 0):.4f}")
mc4.metric("pred_agreement",   f"{detail.get('pred_agreement', 0):.4f}")

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
ec3.metric("topk_overlap", f"{detail.get('topk_overlap', 0):.4f}")

st.caption(
    f"Risk group: **{detail.get('risk_group', 'N/A')}**  |  "
    f"run_id = {run_id},  image_id = {image_id}"
)

# Risk interpretation
if b3_iou is not None:
    th_conf = float(svc._cfg.get("risk_flag", {}).get("confidence_threshold") or 0.90)
    if conf_val >= th_conf and iou_val <= b3_iou:
        st.warning(
            f"REVIEW PRIORITY: confidence = {conf_val:.4f} >= {th_conf:.2f} "
            f"and cam_iou_mean = {iou_val:.4f} <= B3 = {b3_iou:.4f}.  "
            "The model is confident but its attention region is unstable."
        )
    elif detail.get("risk_group") == "stable":
        st.success("Stable: consistent prediction and explanation maps.")

# Grad-CAM maps
st.subheader("Grad-CAM maps")
npz_str = detail.get("cam_npz_path")
import numpy as np

if npz_str:
    npz_path = Path(npz_str)
    base     = npz_path.stem
    png_dir  = npz_path.parent.parent / "png"
    rep_pngs = [png_dir / f"{base}_rep{k}.png" for k in range(6)]
    existing = [p for p in rep_pngs if p.exists()]
    if existing:
        cols = st.columns(3)
        for k, p in enumerate(existing):
            with cols[k % 3]:
                st.image(str(p), caption=f"Map {k + 1}", use_container_width=True)

mean_path = detail.get("mean_png_path")
var_path  = detail.get("variance_png_path")
if mean_path or var_path:
    m1, m2 = st.columns(2)
    with m1:
        if mean_path and Path(mean_path).exists():
            st.image(mean_path, caption="Mean Grad-CAM", use_container_width=True)
    with m2:
        if var_path and Path(var_path).exists():
            st.image(var_path, caption="Variability map", use_container_width=True)

st.divider()
if st.button("Open in Image Detail"):
    st.switch_page("pages/5_Image_Detail.py")
