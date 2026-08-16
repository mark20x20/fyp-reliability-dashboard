"""E7 — Complementarity: does adding explanation metrics improve detection?

Compares two logistic regression classifiers:
  M1: confidence, entropy, pred_variance
  M2: M1 + cam_corr_mean, cam_iou_mean, topk_overlap

Uses stratified 5-fold CV; out-of-fold scores are pooled for the overall
AUROC.  Bootstrap CI for delta AUROC is computed by resampling the pooled
out-of-fold (prediction, truth) pairs.

A delta of approximately zero is a valid finding: it would mean explanation-
side metrics largely duplicate prediction-side information.  The number is
reported without adjustment regardless of direction.

Rows where ``cam_corr_mean == 0.0`` (degenerate maps) are excluded from
both M1 and M2 so the evaluation set is identical for both models.  The
exclusion count is stated.

CLI:
    python src/complementarity.py --run-ids 11,12,13 --task misclassification
    python src/complementarity.py --run-ids 11,12,13 --task ood
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.repository import Repository
from src.utils import load_config

_M1_FEATURES = ["confidence", "entropy", "pred_variance"]
_M2_FEATURES = ["confidence", "entropy", "pred_variance",
                 "cam_corr_mean", "cam_iou_mean", "topk_overlap"]


def _bootstrap_delta_ci(
    y_true: np.ndarray,
    scores_m1: np.ndarray,
    scores_m2: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI for delta AUROC using pooled out-of-fold scores.

    Resamples (score_m1, score_m2, label) triples with replacement and
    computes delta = AUROC_M2 - AUROC_M1 for each resample.  CI from
    the 2.5th and 97.5th percentiles.

    Args:
        y_true: True binary labels (0/1).
        scores_m1: Out-of-fold predicted probabilities for M1.
        scores_m2: Out-of-fold predicted probabilities for M2.
        n_boot: Number of bootstrap iterations.
        seed: RNG seed.

    Returns:
        (ci_lo, ci_hi) — 95% confidence interval for delta AUROC.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        a1 = roc_auc_score(y_b, scores_m1[idx])
        a2 = roc_auc_score(y_b, scores_m2[idx])
        deltas.append(float(a2 - a1))
    if not deltas:
        return float("nan"), float("nan")
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return float(ci_lo), float(ci_hi)


def complementarity(
    df: pd.DataFrame,
    task: str,
    out_dir: str | Path,
) -> dict:
    """E7 — AUROC comparison: M1 (prediction only) vs M2 (prediction + explanation).

    Args:
        df: Combined analysis frame across all loaded runs.
        task: ``"misclassification"`` or ``"ood"``.
        out_dir: Output directory for figures.

    Returns:
        Dict with AUROC M1, AUROC M2, delta AUROC, bootstrap CI, per-fold
        AUROCs, and dataset size.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if task not in ("misclassification", "ood"):
        raise ValueError(f"task must be 'misclassification' or 'ood', got {task!r}")

    # -- Subset selection and label construction --------------------------------
    if task == "misclassification":
        # In-distribution only; rows with ground-truth label
        sub = df[(df["dataset_type"] == "id") & df["correct"].notna()].copy()
        sub["label"] = (sub["correct"] == 0).astype(int)  # positive = misclassified
        task_desc = "Misclassification (positive = wrong, in-distribution only)"
    else:  # ood
        sub = df.copy()
        sub["label"] = (sub["dataset_type"] != "id").astype(int)  # positive = OOD
        task_desc = "OOD detection (positive = dataset_type != 'id')"

    # Exclude degenerate rows (cam_corr_mean == 0.0) — same set for M1 and M2
    n_before = len(sub)
    sub = sub[sub["cam_corr_mean"] != 0.0].copy()
    n_excluded = n_before - len(sub)

    sub = sub[_M2_FEATURES + ["label"]].dropna()
    n = len(sub)
    n_pos = int(sub["label"].sum())
    n_neg = n - n_pos

    print("=" * 60)
    print(f"E7: COMPLEMENTARITY  --  task = {task}")
    print("=" * 60)
    print(f"  {task_desc}")
    print(f"  Total after degenerate exclusion: {n}  (excluded {n_excluded})")
    print(f"  Positive (n={n_pos}),  Negative (n={n_neg})")
    print()

    if n_pos < 10 or n_neg < 10:
        print("  WARNING: too few samples in one class -- results may be unreliable.")
        print()

    X_m1 = sub[_M1_FEATURES].values
    X_m2 = sub[_M2_FEATURES].values
    y = sub["label"].values

    # -- Stratified 5-fold CV --------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_scores_m1 = np.zeros(n)
    oof_scores_m2 = np.zeros(n)
    fold_aurocs_m1: list[float] = []
    fold_aurocs_m2: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_m2, y), 1):
        scaler = StandardScaler()
        X_tr_m1 = scaler.fit_transform(X_m1[train_idx])
        X_va_m1 = scaler.transform(X_m1[val_idx])

        scaler2 = StandardScaler()
        X_tr_m2 = scaler2.fit_transform(X_m2[train_idx])
        X_va_m2 = scaler2.transform(X_m2[val_idx])

        clf1 = LogisticRegression(max_iter=1000, random_state=42)
        clf1.fit(X_tr_m1, y[train_idx])
        oof_scores_m1[val_idx] = clf1.predict_proba(X_va_m1)[:, 1]

        clf2 = LogisticRegression(max_iter=1000, random_state=42)
        clf2.fit(X_tr_m2, y[train_idx])
        oof_scores_m2[val_idx] = clf2.predict_proba(X_va_m2)[:, 1]

        a1 = roc_auc_score(y[val_idx], oof_scores_m1[val_idx])
        a2 = roc_auc_score(y[val_idx], oof_scores_m2[val_idx])
        fold_aurocs_m1.append(float(a1))
        fold_aurocs_m2.append(float(a2))

    # Overall AUROC from pooled out-of-fold scores
    auroc_m1 = float(roc_auc_score(y, oof_scores_m1))
    auroc_m2 = float(roc_auc_score(y, oof_scores_m2))
    delta = auroc_m2 - auroc_m1

    # Per-fold mean ± std
    m1_fold_mean = float(np.mean(fold_aurocs_m1))
    m1_fold_std = float(np.std(fold_aurocs_m1, ddof=1))
    m2_fold_mean = float(np.mean(fold_aurocs_m2))
    m2_fold_std = float(np.std(fold_aurocs_m2, ddof=1))

    # Bootstrap CI for delta AUROC
    ci_lo, ci_hi = _bootstrap_delta_ci(y, oof_scores_m1, oof_scores_m2)

    print(f"  {'Model':<8} {'Features':<45} {'AUROC (pooled)':>15} {'CV mean+/-std':>16}")
    print("  " + "-" * 88)
    m1_feat_str = ", ".join(_M1_FEATURES)
    m2_feat_str = "M1 + cam_corr_mean, cam_iou_mean, topk_overlap"
    print(
        f"  {'M1':<8} {m1_feat_str:<45} {auroc_m1:>15.4f}"
        f"  {m1_fold_mean:.4f} +/- {m1_fold_std:.4f}"
    )
    print(
        f"  {'M2':<8} {m2_feat_str:<45} {auroc_m2:>15.4f}"
        f"  {m2_fold_mean:.4f} +/- {m2_fold_std:.4f}"
    )
    print()
    print(f"  delta AUROC (M2 - M1):  {delta:+.4f}   95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # Interpret delta
    if ci_lo <= 0 <= ci_hi:
        interpretation = (
            "The CI crosses zero -- explanation metrics do not reliably improve "
            "detection over prediction-side metrics alone for this task."
        )
    elif delta > 0:
        interpretation = (
            f"Positive delta (delta AUROC = {delta:+.4f}) -- explanation metrics add "
            "information beyond prediction-side metrics."
        )
    else:
        interpretation = (
            f"Negative delta (delta AUROC = {delta:+.4f}) -- explanation metrics reduce "
            "detection performance; they may introduce noise."
        )
    print(f"\n  Interpretation: {interpretation}")
    print()

    # Paste-ready table
    print("  [paste into EXPERIMENTS.md E7 table]")
    print("  | Task | AUROC M1 | AUROC M2 | delta AUROC | 95% CI |")
    task_label = "Misclassification" if task == "misclassification" else "OOD"
    print(
        f"  | {task_label} | {auroc_m1:.4f} | {auroc_m2:.4f} | "
        f"{delta:+.4f} | [{ci_lo:+.4f}, {ci_hi:+.4f}] |"
    )
    print()

    results = {
        "task": task,
        "n": n,
        "n_pos": n_pos,
        "n_excluded_degen": n_excluded,
        "auroc_m1": auroc_m1,
        "auroc_m2": auroc_m2,
        "delta_auroc": delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "fold_aurocs_m1": fold_aurocs_m1,
        "fold_aurocs_m2": fold_aurocs_m2,
        "m1_fold_mean": m1_fold_mean,
        "m1_fold_std": m1_fold_std,
        "m2_fold_mean": m2_fold_mean,
        "m2_fold_std": m2_fold_std,
        "interpretation": interpretation,
    }

    # -- Figure: delta AUROC bar chart -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: AUROC M1 vs M2 per fold + overall
    folds = list(range(1, 6))
    x = np.arange(len(folds))
    width = 0.35
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, fold_aurocs_m1, width, label="M1", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, fold_aurocs_m2, width, label="M2", color="darkorange", alpha=0.8)
    ax.axhline(auroc_m1, color="steelblue", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(auroc_m2, color="darkorange", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"AUROC per fold — {task}")
    ax.legend()

    # Right: delta AUROC with CI
    ax2 = axes[1]
    fold_deltas = [a2 - a1 for a1, a2 in zip(fold_aurocs_m1, fold_aurocs_m2)]
    ax2.bar(x, fold_deltas, color=["steelblue" if d >= 0 else "firebrick" for d in fold_deltas],
            alpha=0.7)
    ax2.axhline(delta, color="black", linestyle="-", linewidth=1.5, label=f"Overall δ = {delta:+.4f}")
    ax2.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    # Bootstrap CI as shaded region
    ax2.fill_between([-0.5, len(folds) - 0.5], ci_lo, ci_hi,
                     alpha=0.15, color="black", label=f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Fold {f}" for f in folds])
    ax2.set_ylabel("ΔAUROC (M2 − M1)")
    ax2.set_title(f"ΔAUROC per fold — {task}")
    ax2.legend(fontsize=8)

    fig.suptitle(f"E7: Complementarity analysis — {task}", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "delta_auroc_comparison.png", dpi=300)
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# E7 OOD split: ID vs far-OOD and ID vs near-OOD separately
# ---------------------------------------------------------------------------


def complementarity_ood_split(df: pd.DataFrame, out_dir: str | Path) -> dict:
    """E7 OOD split — ΔAUROC for each OOD distance band separately.

    The combined OOD ΔAUROC can be dominated by the far-OOD band (E6 showed
    a large Cliff's delta there, only small for near-OOD).  Running the
    task separately reveals whether the lift is distance-specific.

    Tasks:
      - ``id_vs_far_ood``:  positive = far_ood rows
      - ``id_vs_near_ood``: positive = near_ood rows

    Uses the same 5-fold CV and bootstrap CI as ``complementarity``.
    Shares the degenerate exclusion rule (cam_corr_mean == 0.0 excluded from
    both M1 and M2 to keep evaluation sets identical).

    Saves ``delta_auroc_ood_split.png``.

    Args:
        df: Combined analysis frame (all run_ids).
        out_dir: Output directory for figures.

    Returns:
        Dict keyed by task name with AUROC M1, M2, delta, CI.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("id_vs_far_ood", "far_ood", "ID vs far-OOD"),
        ("id_vs_near_ood", "near_ood", "ID vs near-OOD"),
    ]

    print("=" * 60)
    print("E7 OOD SPLIT -- per-distance AUROC")
    print("=" * 60)
    print(
        "  Note: E6 showed far-OOD Cliff's delta = large, near-OOD = small."
    )
    print(
        "  If the overall delta AUROC is driven by far-OOD, near-OOD is a limitation."
    )
    print()

    all_results: dict = {}
    split_labels: list[str] = []
    split_auroc_m1: list[float] = []
    split_auroc_m2: list[float] = []
    split_deltas: list[float] = []
    split_ci_lo: list[float] = []
    split_ci_hi: list[float] = []

    for task_key, ood_type, task_desc in splits:
        sub = df[df["dataset_type"].isin(["id", ood_type])].copy()
        sub["label"] = (sub["dataset_type"] == ood_type).astype(int)

        # Degenerate exclusion
        n_before = len(sub)
        sub = sub[sub["cam_corr_mean"] != 0.0]
        n_excluded = n_before - len(sub)

        sub = sub[_M2_FEATURES + ["label"]].dropna()
        n = len(sub)
        n_pos = int(sub["label"].sum())
        n_neg = n - n_pos

        X_m1 = sub[_M1_FEATURES].values
        X_m2 = sub[_M2_FEATURES].values
        y = sub["label"].values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_m1 = np.zeros(n)
        oof_m2 = np.zeros(n)

        for train_idx, val_idx in skf.split(X_m2, y):
            sc1 = StandardScaler()
            clf1 = LogisticRegression(max_iter=1000, random_state=42)
            clf1.fit(sc1.fit_transform(X_m1[train_idx]), y[train_idx])
            oof_m1[val_idx] = clf1.predict_proba(sc1.transform(X_m1[val_idx]))[:, 1]

            sc2 = StandardScaler()
            clf2 = LogisticRegression(max_iter=1000, random_state=42)
            clf2.fit(sc2.fit_transform(X_m2[train_idx]), y[train_idx])
            oof_m2[val_idx] = clf2.predict_proba(sc2.transform(X_m2[val_idx]))[:, 1]

        auroc_m1 = float(roc_auc_score(y, oof_m1))
        auroc_m2 = float(roc_auc_score(y, oof_m2))
        delta = auroc_m2 - auroc_m1

        ci_lo, ci_hi = _bootstrap_delta_ci(y, oof_m1, oof_m2)

        if ci_lo <= 0 <= ci_hi:
            verdict = "CI crosses zero -- inconclusive at this sample size."
        elif delta > 0:
            verdict = (
                f"Positive delta ({delta:+.4f}) -- explanation metrics add "
                "information for this OOD band."
            )
        else:
            verdict = (
                f"Negative delta ({delta:+.4f}) -- explanation metrics hurt "
                "performance for this OOD band."
            )

        print(
            f"  {task_desc:<22} n={n}  pos={n_pos}  excl={n_excluded}"
        )
        print(
            f"    AUROC M1={auroc_m1:.4f}  M2={auroc_m2:.4f}"
            f"  delta={delta:+.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]"
        )
        print(f"    {verdict}")
        print()

        all_results[task_key] = {
            "task_desc": task_desc,
            "n": n, "n_pos": n_pos, "n_excluded_degen": n_excluded,
            "auroc_m1": auroc_m1, "auroc_m2": auroc_m2,
            "delta_auroc": delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "verdict": verdict,
        }
        split_labels.append(task_desc)
        split_auroc_m1.append(auroc_m1)
        split_auroc_m2.append(auroc_m2)
        split_deltas.append(delta)
        split_ci_lo.append(ci_lo)
        split_ci_hi.append(ci_hi)

    # Paste-ready table
    print("  [paste into EXPERIMENTS.md E7 OOD split table]")
    print(
        "  | Task | AUROC M1 | AUROC M2 | delta AUROC | 95% CI |"
    )
    for r in all_results.values():
        print(
            f"  | {r['task_desc']} | {r['auroc_m1']:.4f} | {r['auroc_m2']:.4f} |"
            f" {r['delta_auroc']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] |"
        )
    print()

    # Figure
    x = np.arange(len(split_labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: AUROC M1 vs M2
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, split_auroc_m1, width, label="M1", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, split_auroc_m2, width, label="M2", color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("AUROC M1 vs M2 by OOD band")
    ax.legend()

    # Right: delta AUROC with CI
    ax2 = axes[1]
    bar_colors = ["steelblue" if d >= 0 else "firebrick" for d in split_deltas]
    ax2.bar(x, split_deltas, color=bar_colors, alpha=0.8)
    for xi, (lo, hi) in enumerate(zip(split_ci_lo, split_ci_hi)):
        ax2.plot([xi, xi], [lo, hi], color="black", linewidth=2)
        ax2.plot([xi - 0.1, xi + 0.1], [lo, lo], color="black", linewidth=1.5)
        ax2.plot([xi - 0.1, xi + 0.1], [hi, hi], color="black", linewidth=1.5)
    ax2.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(split_labels, fontsize=9)
    ax2.set_ylabel("delta AUROC (M2 - M1)")
    ax2.set_title("delta AUROC by OOD band with 95% CI")

    fig.suptitle("E7 OOD split: ID vs far-OOD / near-OOD", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "delta_auroc_ood_split.png", dpi=300)
    plt.close(fig)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 E7: complementarity — ΔAUROC for explanation metrics"
    )
    parser.add_argument(
        "--run-ids",
        required=True,
        help="Comma-separated run IDs, e.g. 11,12,13",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=["misclassification", "ood"],
        help="Detection task",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    run_ids = [int(x.strip()) for x in args.run_ids.split(",")]
    cfg = load_config(args.config)
    repo = Repository(cfg["database"]["path"])

    dfs: list[pd.DataFrame] = []
    for rid in run_ids:
        df = repo.fetch_metrics(rid)
        df["run_id"] = rid
        dfs.append(df)
        dtypes = df["dataset_type"].value_counts().to_dict()
        print(f"Loaded run_id={rid}: {len(df)} rows  dataset_types={dtypes}")
    print()

    combined = pd.concat(dfs, ignore_index=True)
    complementarity(combined, args.task, args.out_dir)

    if args.task == "ood":
        complementarity_ood_split(combined, args.out_dir)


if __name__ == "__main__":
    _main()
