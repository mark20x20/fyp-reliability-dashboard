"""Headless check that every Streamlit page loads without TypeError in 1.36.

Uses streamlit.testing.v1.AppTest for each page.  Pages that require a
non-engineer role or a specific session_state key are initialised before the
run() call.

For each page we assert:
  1. No exception attribute set on the AppTest result.
  2. No element in at.error that contains 'unexpected keyword argument'
     (which is how Streamlit surfaces API version errors at render time).

Note: pages that call st.stop() early (e.g. because run_id is None or
role mismatch) are still exercised for the import + pre-guard code path,
which is where version errors would appear.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print("streamlit.testing.v1 not available — skip headless check")
    sys.exit(0)

PASS = []
FAIL = []


def _check(label, at):
    # Check for uncaught exception
    if at.exception:
        FAIL.append(f"{label}: exception={at.exception}")
        print(f"  [FAIL] {label}: {at.exception}")
        return
    # Check for version-mismatch TypeError surfaced as st.error()
    for e in at.error:
        if "unexpected keyword argument" in e.value:
            FAIL.append(f"{label}: TypeError in error element: {e.value[:120]}")
            print(f"  [FAIL] {label}: {e.value[:120]}")
            return
    PASS.append(label)
    print(f"  [OK]   {label}")


print("\n=== Headless page checks (Streamlit 1.36 API) ===\n")

# ── 1. Home page (via streamlit_app.py acting as the entry point) ────────────
# We test individual pages directly to isolate errors.

# ── Page 1: Run Analysis (engineer) ─────────────────────────────────────────
at = AppTest.from_file("app/views/1_Run_Analysis.py", default_timeout=15)
at.session_state["role"] = "engineer"
at.run()
_check("1_Run_Analysis (engineer)", at)

# ── Page 1: Run Analysis (reviewer guard) ────────────────────────────────────
at2 = AppTest.from_file("app/views/1_Run_Analysis.py", default_timeout=15)
at2.session_state["role"] = "reviewer"
at2.run()
# Reviewer sees an error but NOT a TypeError
has_type_err = any("unexpected keyword argument" in e.value for e in at2.error)
if at2.exception or has_type_err:
    FAIL.append("1_Run_Analysis (reviewer guard)")
    print(f"  [FAIL] 1_Run_Analysis (reviewer guard)")
else:
    PASS.append("1_Run_Analysis (reviewer guard)")
    print(f"  [OK]   1_Run_Analysis (reviewer guard) - guard fires correctly")

# ── Page 2: Metrics Table ────────────────────────────────────────────────────
at = AppTest.from_file("app/views/2_Metrics_Table.py", default_timeout=15)
at.session_state["role"] = "engineer"
at.session_state["run_id"] = 11
at.run()
_check("2_Metrics_Table (run_id=11)", at)

# ── Page 3: Statistics ───────────────────────────────────────────────────────
at = AppTest.from_file("app/views/3_Statistics.py", default_timeout=30)
at.session_state["role"] = "engineer"
at.session_state["run_id"] = 11
at.run()
_check("3_Statistics (run_id=11)", at)

# ── Page 4: Risk Queue ───────────────────────────────────────────────────────
at = AppTest.from_file("app/views/4_Risk_Queue.py", default_timeout=15)
at.session_state["role"] = "reviewer"
at.session_state["run_id"] = 11
at.run()
_check("4_Risk_Queue (run_id=11)", at)

# ── Page 5: Image Detail (no image_id → shows input form) ───────────────────
at = AppTest.from_file("app/views/5_Image_Detail.py", default_timeout=15)
at.session_state["role"] = "reviewer"
# No image_id set → should show number_input + Load button
at.run()
_check("5_Image_Detail (no image_id)", at)

# ── Page 5: Image Detail (with real image_id) ────────────────────────────────
at = AppTest.from_file("app/views/5_Image_Detail.py", default_timeout=20)
at.session_state["role"] = "reviewer"
at.session_state["image_id"] = 4253   # confirmed real image_id in run_id=11
at.run()
_check("5_Image_Detail (image_id=4253)", at)

# ── Page 6: Upload & Analyse (no file) ───────────────────────────────────────
at = AppTest.from_file("app/views/6_Upload_Analyse.py", default_timeout=15)
at.session_state["role"] = "reviewer"
at.run()
_check("6_Upload_Analyse (no file)", at)

# ── Page 7: Export ───────────────────────────────────────────────────────────
at = AppTest.from_file("app/views/7_Export.py", default_timeout=15)
at.session_state["run_id"] = 11
at.run()
_check("7_Export (run_id=11)", at)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
if FAIL:
    print("\nFailed:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All pages passed headless checks.")
