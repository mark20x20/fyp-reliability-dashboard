"""Main Streamlit entry point — role-based navigation and home page.

Run from the project root:
    streamlit run app/streamlit_app.py

Role is held in st.session_state["role"] and controls which pages appear in
the sidebar navigation.  Sidebar elements (role selector, run selector) are
rendered here so they are visible on every page.

Engineer pages : Run Analysis, Metrics Table, Statistics
Reviewer pages : Risk Queue, Image Detail, Upload & Analyse
Both           : Home, Export
"""

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Project root on sys.path so src.* imports work from any page
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis_service import get_service  # noqa: E402  (after sys.path)

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Reliability Analysis",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
if "role"     not in st.session_state:
    st.session_state["role"]     = "engineer"
if "run_id"   not in st.session_state:
    st.session_state["run_id"]   = None
if "model_id" not in st.session_state:
    st.session_state["model_id"] = None
if "image_id" not in st.session_state:
    st.session_state["image_id"] = None


# ---------------------------------------------------------------------------
# Home page (callable — keeps the home content in this file)
# ---------------------------------------------------------------------------
def _home() -> None:
    st.title("Reliability Analysis Dashboard")
    st.caption(
        "MC-Dropout prediction uncertainty and Grad-CAM explanation stability "
        "for image classification."
    )

    svc    = get_service()
    run_id = st.session_state.get("run_id")

    if run_id is not None:
        summary = svc.fetch_run_summary(run_id)
        if summary:
            st.subheader(
                f"Run {run_id} — {summary.get('dataset_name', '')}  "
                f"({summary.get('status', '')})"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total images",    f"{summary.get('n_images', 0):,}")

            acc = summary.get("accuracy")
            c2.metric("Accuracy",
                      f"{acc:.1%}" if acc is not None else "N/A")

            conf = summary.get("mean_confidence")
            c3.metric("Mean confidence",
                      f"{conf:.3f}" if conf is not None else "N/A")

            iou = summary.get("mean_cam_iou")
            c4.metric("Mean cam_iou",
                      f"{iou:.3f}" if iou is not None else "N/A")
        else:
            st.warning(f"Run {run_id} not found.")
    else:
        st.info("Select a completed run from the sidebar to see summary tiles.")

    st.divider()
    st.subheader("All runs")

    runs_df = svc.list_runs()
    if not runs_df.empty:
        display_cols = [
            "run_id", "dataset_name", "model_name",
            "n_images", "status", "started_at", "finished_at",
        ]
        st.dataframe(
            runs_df[display_cols].rename(columns={
                "run_id":      "Run",
                "dataset_name":"Dataset",
                "model_name":  "Model",
                "n_images":    "Images",
                "status":      "Status",
                "started_at":  "Started",
                "finished_at": "Finished",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No runs found.  Use Run Analysis (engineer role) to start one.")


# ---------------------------------------------------------------------------
# Build navigation based on current role
# ---------------------------------------------------------------------------
def _make_navigation():
    role = st.session_state.get("role", "engineer")

    home_page = st.Page(_home, title="Home", icon=":material/home:", default=True)

    engineer_pages = [
        st.Page("pages/1_Run_Analysis.py",  title="Run Analysis",  icon=":material/play_circle:"),
        st.Page("pages/2_Metrics_Table.py", title="Metrics Table", icon=":material/table_chart:"),
        st.Page("pages/3_Statistics.py",    title="Statistics",    icon=":material/bar_chart:"),
    ]
    reviewer_pages = [
        st.Page("pages/4_Risk_Queue.py",     title="Risk Queue",      icon=":material/warning:"),
        st.Page("pages/5_Image_Detail.py",   title="Image Detail",    icon=":material/image:"),
        st.Page("pages/6_Upload_Analyse.py", title="Upload & Analyse",icon=":material/upload:"),
    ]
    shared_pages = [
        st.Page("pages/7_Export.py", title="Export", icon=":material/download:"),
    ]

    if role == "engineer":
        return st.navigation({
            "":         [home_page],
            "Engineer": engineer_pages,
            "Shared":   shared_pages,
        })
    else:
        return st.navigation({
            "":         [home_page],
            "Reviewer": reviewer_pages,
            "Shared":   shared_pages,
        })


pg = _make_navigation()

# ---------------------------------------------------------------------------
# Sidebar — rendered on every page
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Reliability Analysis")

    # Role selector
    st.selectbox(
        "Role",
        options=["engineer", "reviewer"],
        key="role",
        help="Engineer: run analyses and view metrics.  Reviewer: inspect and annotate flagged images.",
    )

    st.divider()

    # Run selector — completed runs only
    svc      = get_service()
    runs_df  = svc.list_runs()
    completed = runs_df[runs_df["status"] == "completed"] if not runs_df.empty else runs_df

    if not completed.empty:
        run_options = completed["run_id"].tolist()
        run_labels  = {
            int(r["run_id"]): (
                f"Run {r['run_id']} | {r['dataset_name']} "
                f"({int(r['n_images'])} imgs)"
            )
            for _, r in completed.iterrows()
        }

        current = st.session_state.get("run_id")
        default_idx = (
            run_options.index(current)
            if current in run_options
            else 0
        )

        selected_run = st.selectbox(
            "Active run",
            options=run_options,
            index=default_idx,
            format_func=lambda r: run_labels.get(r, str(r)),
        )
        st.session_state["run_id"] = int(selected_run)

        # Show run detail inline
        sel = completed[completed["run_id"] == selected_run].iloc[0]
        st.caption(
            f"Model: {sel['model_name']} | "
            f"N={int(sel['n_runs'])} | "
            f"p={sel['dropout_p']}"
        )
    else:
        st.caption("No completed runs yet.")

# ---------------------------------------------------------------------------
# Run the selected page
# ---------------------------------------------------------------------------
pg.run()
