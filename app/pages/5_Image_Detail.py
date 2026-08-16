"""Image Detail page -- Reviewer role (FR-09, FR-10).  Phase 5B."""

import streamlit as st

if st.session_state.get("role") != "reviewer":
    st.error("This page is only available to reviewers.")
    st.stop()

st.title("Image Detail")
st.info("Phase 5B — Reviewer pages are implemented in the next phase.")
