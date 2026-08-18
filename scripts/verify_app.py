"""自動検証スクリプト -- アプリケーション動作確認."""
import ast
import io
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

PASS = []
FAIL = []


def ok(msg):
    PASS.append(msg)
    print(f"  [OK]  {msg}")


def err(msg, exc=None):
    detail = f": {exc}" if exc else ""
    FAIL.append(msg + detail)
    print(f"  [ERR] {msg}{detail}")


# ── 1. 構文チェック ────────────────────────────────────────────────────────────
print("\n=== 1. Page syntax check ===")
pages = [
    "app/streamlit_app.py",
    "app/views/1_Run_Analysis.py",
    "app/views/2_Metrics_Table.py",
    "app/views/3_Statistics.py",
    "app/views/4_Risk_Queue.py",
    "app/views/5_Image_Detail.py",
    "app/views/6_Upload_Analyse.py",
    "app/views/7_Export.py",
]
for p in pages:
    fpath = ROOT / p
    try:
        ast.parse(fpath.read_text(encoding="utf-8"))
        ok(p)
    except SyntaxError as e:
        err(p, e)

# ── 2. src モジュールの import ────────────────────────────────────────────────
print("\n=== 2. src module imports ===")
src_mods = [
    "src.utils",
    "src.repository",
    "src.metrics_prediction",
    "src.metrics_explanation",
    "src.analysis_service",
]
for mod in src_mods:
    try:
        import importlib
        importlib.import_module(mod)
        ok(mod)
    except Exception as e:
        err(mod, e)

# ── 3. analysis_service 全メソッド (実データ) ────────────────────────────────
print("\n=== 3. AnalysisService methods (live DB) ===")
from src.analysis_service import AnalysisService
from src.utils import load_config

try:
    cfg = load_config(str(ROOT / "configs/config.yaml"))
    svc = AnalysisService(cfg)
    ok("AnalysisService instantiated")
except Exception as e:
    err("AnalysisService instantiated", e)
    sys.exit(1)

# list_runs
try:
    runs = svc.list_runs()
    ok(f"list_runs() -> {len(runs)} rows")
except Exception as e:
    err("list_runs()", e)

# list_models
try:
    models = svc.list_models()
    ok(f"list_models() -> {len(models)} rows")
except Exception as e:
    err("list_models()", e)

# fetch_run_summary for each run
for rid in [11, 12, 13, 14]:
    try:
        s = svc.fetch_run_summary(rid)
        if s:
            ok(f"fetch_run_summary({rid}) -> n_images={s.get('n_images')}, status={s.get('status')}")
        else:
            err(f"fetch_run_summary({rid}) -> empty dict (run not found?)")
    except Exception as e:
        err(f"fetch_run_summary({rid})", e)

# fetch_metrics
for rid in [11, 14]:
    try:
        df = svc.fetch_metrics(rid)
        ok(f"fetch_metrics({rid}) -> {len(df)} rows, cols={list(df.columns[:5])}")
    except Exception as e:
        err(f"fetch_metrics({rid})", e)

# fetch_metrics with filter
try:
    df11 = svc.fetch_metrics(11, {"confidence__gte": 0.95})
    ok(f"fetch_metrics(11, confidence>=0.95) -> {len(df11)} rows")
except Exception as e:
    err("fetch_metrics with filter", e)

# fetch_baselines
for rid in [11, 14]:
    try:
        bl = svc.fetch_baselines(rid)
        keys = list(bl.keys())
        ok(f"fetch_baselines({rid}) -> types={keys}")
    except Exception as e:
        err(f"fetch_baselines({rid})", e)

# export_metrics_csv
try:
    b = svc.export_metrics_csv(11)
    ok(f"export_metrics_csv(11) -> {len(b):,} bytes")
except Exception as e:
    err("export_metrics_csv(11)", e)

# export_reviews_csv
try:
    b = svc.export_reviews_csv()
    ok(f"export_reviews_csv() -> {len(b):,} bytes")
except Exception as e:
    err("export_reviews_csv()", e)

# ── 4. fetch_image_detail (run_id 14, corrupted run) ──────────────────────────
print("\n=== 4. fetch_image_detail (run_id=14 corrupted) ===")
try:
    df14 = svc.fetch_metrics(14)
    if df14.empty:
        err("No rows for run_id=14")
    else:
        # 最初の image_id を取得
        first_id = int(df14.iloc[0]["image_id"])
        detail = svc.fetch_image_detail(first_id)
        if not detail:
            err(f"fetch_image_detail({first_id}) -> empty")
        else:
            npz = detail.get("cam_npz_path")
            cams = detail.get("cams_array")
            npz_exists = Path(npz).exists() if npz else False
            ok(f"fetch_image_detail({first_id}) -> keys={len(detail)}, "
               f"npz_exists={npz_exists}, cams_array={'array' if cams is not None else 'None'}")

            # 代表PNGが存在するか (corrupted では存在しないはず)
            if npz:
                base = Path(npz).stem
                png_dir = Path(npz).parent.parent / "png"
                rep0 = png_dir / f"{base}_rep0.png"
                ok(f"  rep0.png exists={rep0.exists()} -> "
                   f"{'disk PNGs available' if rep0.exists() else 'fallback to cams_array needed'}")
                if cams is not None:
                    ok(f"  cams_array shape={cams.shape}, dtype={cams.dtype} (fallback will work)")
                else:
                    err("  cams_array is None — fallback to cams_array will NOT work")
except Exception as e:
    err("fetch_image_detail(run_id=14)", e)

# ── 5. hidden_risk の件数 (run_id=14) ─────────────────────────────────────────
print("\n=== 5. hidden_risk count (run_id=14) ===")
try:
    df14 = svc.fetch_metrics(14)
    hr = df14[df14["risk_group"] == "hidden_risk"]
    all_groups = df14["risk_group"].value_counts().to_dict()
    ok(f"run_id=14: hidden_risk={len(hr)}, all groups={all_groups}")
except Exception as e:
    err("hidden_risk count", e)

# ── 6. reviews テーブル 書き込み・読み出し ────────────────────────────────────
print("\n=== 6. Reviews write/read ===")
try:
    df11 = svc.fetch_metrics(11)
    test_image_id = int(df11.iloc[0]["image_id"])
    review_id = svc.insert_review(test_image_id, "needs_review", "自動検証テスト")
    ok(f"insert_review(image_id={test_image_id}) -> review_id={review_id}")

    reviews = svc.fetch_reviews(test_image_id)
    if len(reviews) >= 1:
        latest = reviews.iloc[0]
        ok(f"fetch_reviews({test_image_id}) -> {len(reviews)} rows, "
           f"latest decision={latest['decision']!r}")
    else:
        err("fetch_reviews returned empty after insert")
except Exception as e:
    err("reviews write/read", e)

# ── 7. figures ディレクトリ ────────────────────────────────────────────────────
print("\n=== 7. Figure files ===")
for rid in [11, 14]:
    fig_dir = ROOT / f"outputs/figures/run_{rid}"
    if fig_dir.exists():
        pngs = sorted(fig_dir.glob("*.png"))
        ok(f"outputs/figures/run_{rid}/ -> {len(pngs)} PNG files: "
           f"{[p.name for p in pngs]}")
    else:
        err(f"outputs/figures/run_{rid}/ does not exist")

# ── 結果サマリ ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PASS: {len(PASS)}   FAIL: {len(FAIL)}")
if FAIL:
    print("\nFailed items:")
    for f in FAIL:
        print(f"  - {f}")
else:
    print("All checks passed.")
