"""Export page -- Both roles (FR-07, FR-12)."""

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis_service import get_service

st.title("Export")

run_id = st.session_state.get("run_id")
svc    = get_service()

st.subheader("Metrics CSV")
if run_id is not None:
    csv_bytes = svc.export_metrics_csv(int(run_id))
    st.download_button(
        label=f"Download run {run_id} metrics",
        data=csv_bytes,
        file_name=f"reliability_run{run_id}.csv",
        mime="text/csv",
    )
else:
    st.info("Select a run from the sidebar first.")

st.subheader("Reviews CSV")
reviews_bytes = svc.export_reviews_csv(int(run_id) if run_id else None)
st.download_button(
    label="Download review decisions",
    data=reviews_bytes,
    file_name="reviews.csv",
    mime="text/csv",
)
