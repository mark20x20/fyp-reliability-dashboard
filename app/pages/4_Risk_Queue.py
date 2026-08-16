"""Risk Queue page -- Reviewer role (FR-08).  Phase 5B."""

import streamlit as st

if st.session_state.get("role") != "reviewer":
    st.error("This page is only available to reviewers.")
    st.stop()

st.title("Risk Queue")
st.info("Phase 5B — Reviewer pages are implemented in the next phase.")
