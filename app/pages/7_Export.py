"""Export page -- Both roles (FR-07, FR-12).

Downloads available:
  - Metrics CSV   : all per-image metrics for a run
  - Reviews CSV   : all reviewer decisions (optionally filtered by run)
  - Figure bundle : ZIP of all analysis PNG files for the active run
"""

import io
import sys
import zipfile
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis_service import get_service

st.title("Export")

run_id = st.session_state.get("run_id")
svc    = get_service()

# ---------------------------------------------------------------------------
# Metrics CSV (FR-07)
# ---------------------------------------------------------------------------
st.subheader("Metrics CSV")
if run_id is not None:
    csv_bytes = svc.export_metrics_csv(int(run_id))
    st.download_button(
        label=f"Download run {run_id} metrics ({len(csv_bytes):,} bytes)",
        data=csv_bytes,
        file_name=f"reliability_run{run_id}.csv",
        mime="text/csv",
    )
else:
    st.info("Select a run from the sidebar first.")

# ---------------------------------------------------------------------------
# Reviews CSV (FR-12)
# ---------------------------------------------------------------------------
st.subheader("Reviews CSV")
reviews_bytes = svc.export_reviews_csv(int(run_id) if run_id else None)
st.download_button(
    label=(
        f"Download run {run_id} reviews"
        if run_id is not None
        else "Download all reviews"
    ),
    data=reviews_bytes,
    file_name=f"reviews_run{run_id}.csv" if run_id else "reviews_all.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Figure bundle (ZIP of PNGs for active run)
# ---------------------------------------------------------------------------
st.subheader("Figure bundle")
if run_id is not None:
    fig_dir = _ROOT / f"outputs/figures/run_{run_id}"
    if fig_dir.exists():
        fig_files = sorted(fig_dir.glob("*.png"))
        if fig_files:
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in fig_files:
                    zf.write(p, p.name)
            zip_buf.seek(0)
            st.download_button(
                label=(
                    f"Download run {run_id} figures "
                    f"({len(fig_files)} PNG files)"
                ),
                data=zip_buf,
                file_name=f"figures_run{run_id}.zip",
                mime="application/zip",
            )
        else:
            st.info(
                "No figures generated yet for this run.  "
                "Visit the Statistics page to generate them."
            )
    else:
        st.info(
            "No figures directory found for this run.  "
            "Visit the Statistics page to generate them."
        )
else:
    st.info("Select a run from the sidebar to download its figures.")
