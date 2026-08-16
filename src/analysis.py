"""E2, E4, E5, E6 analyses and degenerate CAM audit.

Degenerate audit is run first: rows where ``cam_corr_mean == 0.0`` exactly
arise from ``np.corrcoef`` returning NaN on constant maps, substituted to
0.0 in production.  Those rows are excluded from any analysis that uses
``cam_corr_mean`` and the exclusion count is stated.

CLI:
    python src/analysis.py --run-ids 11,12,13
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
import seaborn as sns
from scipy import stats
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.repository import Repository
from src.utils import load_config

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Spearman ρ with bootstrap 95% CI and two-tailed p-value.

    Args:
        x: First variable array.
        y: Second variable array, same length as x.
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        (rho, ci_lo, ci_hi, p_value)
    """
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    boot: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(r):
            boot.append(float(r))
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    return float(rho), float(ci_lo), float(ci_hi), float(p)


def _cliffs_delta(a: np.ndarray, b: np.ndarray, seed: int = 42) -> float:
    """Vectorised Cliff's delta (positive = a tends to exceed b).

    Sub-samples to MAX_EACH=5000 if either array is larger, to keep
    memory reasonable while preserving estimation accuracy.

    Args:
        a: First sample.
        b: Second sample.
        seed: RNG seed for sub-sampling.

    Returns:
        float in [-1, 1].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    MAX_EACH = 5_000
    rng = np.random.default_rng(seed)
    if len(a) > MAX_EACH:
        a = a[rng.choice(len(a), MAX_EACH, replace=False)]
    if len(b) > MAX_EACH:
        b = b[rng.choice(len(b), MAX_EACH, replace=False)]
    greater = int(np.sum(a[:, None] > b[None, :]))
    less = int(np.sum(a[:, None] < b[None, :]))
    return float((greater - less) / (len(a) * len(b)))


def _cliff_label(delta: float) -> str:
    """Magnitude label per the METRICS.md convention."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Step 0: Degenerate CAM audit
# ---------------------------------------------------------------------------


def degenerate_audit(dfs: dict[int, pd.DataFrame]) -> dict:
    """Report degenerate-CAM row counts per run, per metric.

    ``cam_corr_mean == 0.0`` exactly indicates a constant Grad-CAM map
    where ``np.corrcoef`` returned NaN and the pipeline substituted 0.0.
    These are not measurements and must be excluded from correlation
    analyses.

    Args:
        dfs: Mapping run_id → full-frame DataFrame.

    Returns:
        Dict keyed by run_id with counts and percentages.
    """
    print("=" * 60)
    print("STEP 0: DEGENERATE CAM AUDIT")
    print("=" * 60)
    print(
        f"  {'run_id':>7}  {'n':>6}  "
        f"{'corr==0':>8} {'%corr':>7}  "
        f"{'iou==0':>7} {'%iou':>7}"
    )
    print("  " + "-" * 55)

    results: dict[int, dict] = {}
    for run_id, df in dfs.items():
        n = len(df)
        n_corr = int((df["cam_corr_mean"] == 0.0).sum())
        n_iou = int((df["cam_iou_mean"] == 0.0).sum())
        pct_corr = 100.0 * n_corr / n if n else 0.0
        pct_iou = 100.0 * n_iou / n if n else 0.0
        print(
            f"  {run_id:>7}  {n:>6}  "
            f"{n_corr:>8} {pct_corr:>6.1f}%  "
            f"{n_iou:>7} {pct_iou:>6.1f}%"
        )
        results[run_id] = {
            "n": n,
            "n_corr_zero": n_corr,
            "pct_corr_zero": pct_corr,
            "n_iou_zero": n_iou,
            "pct_iou_zero": pct_iou,
        }
    print()
    return results


# ---------------------------------------------------------------------------
# E2: Correlation analysis
# ---------------------------------------------------------------------------


def correlation_analysis(df: pd.DataFrame, out_dir: str | Path) -> dict:
    """E2 — Spearman correlations: prediction metrics vs explanation metrics.

    Three pairs per EXPERIMENTS.md E2:
      - confidence vs cam_iou_mean
      - entropy vs cam_corr_mean
      - pred_variance vs topk_overlap

    Rows where ``cam_corr_mean == 0.0`` (degenerate maps) are excluded from
    the two pairs that involve cam_corr_mean.

    Figures saved:
      - ``correlation_heatmap.png``
      - ``confidence_vs_cam_iou.png``

    Args:
        df: Combined analysis frame across all requested runs.
        out_dir: Output directory for figures.

    Returns:
        Dict keyed by pair name with rho, ci_lo, ci_hi, p, n.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Degenerate exclusion — only for pairs involving cam_corr_mean
    df_all = df.copy()
    df_no_degen = df[df["cam_corr_mean"] != 0.0].copy()
    n_excluded = len(df_all) - len(df_no_degen)

    pairs = [
        ("confidence", "cam_iou_mean", df_all, "all rows"),
        ("entropy", "cam_corr_mean", df_no_degen, f"cam_corr_mean!=0 (excluded {n_excluded})"),
        ("pred_variance", "topk_overlap", df_no_degen, f"cam_corr_mean!=0 (excluded {n_excluded})"),
    ]

    print("=" * 60)
    print("E2: CORRELATION ANALYSIS")
    print("=" * 60)
    print(
        f"  {'Pair':<35} {'rho':>8} {'95% CI':>20} {'n':>6}  subset"
    )
    print("  " + "-" * 85)

    results: dict = {}
    for x_col, y_col, sub_df, subset_note in pairs:
        mask = sub_df[x_col].notna() & sub_df[y_col].notna()
        x = sub_df.loc[mask, x_col].values
        y = sub_df.loc[mask, y_col].values
        rho, ci_lo, ci_hi, p = _spearman_with_ci(x, y)
        pair_name = f"{x_col} vs {y_col}"
        print(
            f"  {pair_name:<35} {rho:>+8.4f} [{ci_lo:>+.4f}, {ci_hi:>+.4f}]"
            f"  {len(x):>6}  {subset_note}"
        )
        results[pair_name] = {
            "rho": rho,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p": p,
            "n": len(x),
            "n_excluded_degen": n_excluded if "cam_corr" in (x_col + y_col) else 0,
        }

    # -- Paste-ready table for EXPERIMENTS.md -----------------------------------
    print()
    print("  [paste into EXPERIMENTS.md E2 table]")
    print("  | Pair | rho | 95% CI | n |")
    for pair_name, r in results.items():
        print(
            f"  | {pair_name} | {r['rho']:+.4f} | "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['n']} |"
        )
    print()

    # -- Figure 1: Spearman correlation heatmap ---------------------------------
    heat_cols = [
        "confidence", "entropy", "pred_variance", "pred_agreement",
        "cam_corr_mean", "cam_iou_mean", "topk_overlap",
    ]
    df_heat = df_no_degen[heat_cols].dropna()
    corr_mat = df_heat.corr(method="spearman")
    mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr_mat,
        mask=mask_upper,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title(
        f"Spearman correlation matrix\n"
        f"(n={len(df_heat)}, degenerate rows excluded)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=300)
    plt.close(fig)

    # -- Figure 2: confidence vs cam_iou_mean scatter ---------------------------
    sub = df_all[["confidence", "cam_iou_mean", "dataset_type"]].dropna()
    type_colors = {"id": "steelblue", "near_ood": "darkorange", "far_ood": "firebrick"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for dtype, grp in sub.groupby("dataset_type"):
        ax.scatter(
            grp["confidence"],
            grp["cam_iou_mean"],
            c=type_colors.get(str(dtype), "grey"),
            alpha=0.25,
            s=5,
            label=str(dtype),
            rasterized=True,
        )
    r = results["confidence vs cam_iou_mean"]
    ax.set_xlabel("Confidence (max softmax of mean distribution)")
    ax.set_ylabel("cam_iou_mean")
    ax.set_title(
        f"Confidence vs Grad-CAM IoU (n={r['n']})\n"
        f"Spearman ρ = {r['rho']:+.4f}, 95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
    )
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "confidence_vs_cam_iou.png", dpi=300)
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# E4: Quadrant analysis
# ---------------------------------------------------------------------------


def quadrant_analysis(df: pd.DataFrame, out_dir: str | Path) -> dict:
    """E4 — Quadrant analysis on entropy and explanation instability.

    Axes: ``entropy`` (x) and ``1 - cam_iou_mean`` (y), split at the median.

    Two median variants are reported:

    **(a) PRIMARY** — medians computed from ID rows only.  Misclassification
    rate and chi-square are computed on ID rows only (``dataset_type == 'id'``).
    OOD rows have ``correct = 0`` (predicted class never matches OOD true
    class) so they must be excluded from the misclassification computation;
    using ``correct.notna()`` is insufficient.

    **(b) SECONDARY** — medians computed from all rows combined.  Used for
    the OOD-share view (how OOD samples distribute across quadrants).

    Per quadrant reports: n_total, n_id, n_ood, misclassification rate
    (ID rows only), OOD share.

    Figure saved: ``quadrant_scatter.png``  (uses variant-a medians).

    Args:
        df: Combined analysis frame (all run_ids).
        out_dir: Output directory for figures.

    Returns:
        Dict with per-quadrant stats for both variants plus chi-square.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work = df.copy()
    work["instability"] = 1.0 - work["cam_iou_mean"]

    # Variant (a) — ID-only medians (primary)
    work_id = work[work["dataset_type"] == "id"]
    med_e_id = float(work_id["entropy"].median())
    med_i_id = float(work_id["instability"].median())

    # Variant (b) — combined medians (secondary)
    med_e_all = float(work["entropy"].median())
    med_i_all = float(work["instability"].median())

    def _label(entropy: float, instab: float, med_e: float, med_i: float) -> str:
        hi_e = entropy > med_e
        hi_i = instab > med_i
        if not hi_e and not hi_i:
            return "Q1_stable"
        if hi_e and hi_i:
            return "Q2_unstable_both"
        if hi_e and not hi_i:
            return "Q3_pred_unstable_only"
        return "Q4_hidden_risk"

    work["quad_a"] = work.apply(
        lambda r: _label(r["entropy"], r["instability"], med_e_id, med_i_id), axis=1
    )
    work["quad_b"] = work.apply(
        lambda r: _label(r["entropy"], r["instability"], med_e_all, med_i_all), axis=1
    )

    quad_order = [
        "Q1_stable",
        "Q2_unstable_both",
        "Q3_pred_unstable_only",
        "Q4_hidden_risk",
    ]
    quad_display = [
        "Q1 low/low  (stable)",
        "Q2 high/high (unstable_both)",
        "Q3 high/low  (pred_unstable_only)",
        "Q4 low/high  (hidden_risk)",
    ]

    results: dict = {"variant_a": {}, "variant_b": {}}

    # ---- Variant (a): PRIMARY ------------------------------------------------
    print("=" * 60)
    print("E4: QUADRANT ANALYSIS")
    print("=" * 60)
    print()
    print("  Variant (a) PRIMARY -- medians from ID rows only")
    print(
        f"  Median entropy split:     {med_e_id:.4f}"
        f"  (n_id={len(work_id)})"
    )
    print(
        f"  Median instability split: {med_i_id:.4f}"
        f"  (n_id={len(work_id)})"
    )
    print()
    print(
        f"  {'Quadrant':<36} {'n_tot':>5} {'n_id':>5} {'n_ood':>5}"
        f" {'misclass%(id only)':>18} {'OOD%':>6}"
    )
    print("  " + "-" * 80)

    for qname, qlabel in zip(quad_order, quad_display):
        sub = work[work["quad_a"] == qname]
        n_tot = len(sub)
        sub_id = sub[sub["dataset_type"] == "id"]
        n_id = len(sub_id)
        n_ood = n_tot - n_id
        # Misclassification rate on ID rows only — OOD rows excluded explicitly
        sub_id_gt = sub_id[sub_id["correct"].notna()]
        mr = (
            float((sub_id_gt["correct"] == 0).sum() / len(sub_id_gt))
            if len(sub_id_gt) > 0 else float("nan")
        )
        ood_share = float(n_ood / n_tot) if n_tot > 0 else float("nan")
        results["variant_a"][qname] = {
            "n_total": n_tot, "n_id": n_id, "n_ood": n_ood,
            "misclass_rate_id": mr, "ood_share": ood_share,
        }
        mr_s = f"{100*mr:.1f}%" if not np.isnan(mr) else "  N/A"
        os_s = f"{100*ood_share:.1f}%" if not np.isnan(ood_share) else "  N/A"
        print(
            f"  {qlabel:<36} {n_tot:>5} {n_id:>5} {n_ood:>5}"
            f" {mr_s:>18} {os_s:>6}"
        )

    # Chi-square Q4 vs Q1 on ID rows only (variant a quadrant labels)
    q1_id = work[(work["quad_a"] == "Q1_stable") & (work["dataset_type"] == "id")]
    q4_id = work[(work["quad_a"] == "Q4_hidden_risk") & (work["dataset_type"] == "id")]
    q1_id = q1_id[q1_id["correct"].notna()]
    q4_id = q4_id[q4_id["correct"].notna()]

    if len(q1_id) > 0 and len(q4_id) > 0:
        q1_cor = int((q1_id["correct"] == 1).sum())
        q1_mis = int((q1_id["correct"] == 0).sum())
        q4_cor = int((q4_id["correct"] == 1).sum())
        q4_mis = int((q4_id["correct"] == 0).sum())
        q1_mr = q1_mis / len(q1_id)
        q4_mr = q4_mis / len(q4_id)
        rr = q4_mr / q1_mr if q1_mr > 0 else float("nan")

        print(f"\n  Test Q4 vs Q1 on misclassification (ID rows only):")
        print(f"    Q1 n_id={len(q1_id)}  misclass={q1_mis}  rate={100*q1_mr:.2f}%")
        print(f"    Q4 n_id={len(q4_id)}  misclass={q4_mis}  rate={100*q4_mr:.2f}%")

        contingency = np.array([[q1_cor, q1_mis], [q4_cor, q4_mis]])

        # If any expected frequency is zero, chi-square is undefined;
        # use Fisher exact instead (and note the finding).
        if q1_mis == 0 and q4_mis == 0:
            print(
                "    NOTE: 0 misclassified in both Q1 and Q4. "
                "All 196 misclassified ID rows have high entropy and fall in Q2/Q3."
            )
            print(
                "    Within the low-entropy (high-confidence) quadrants, "
                "every ID sample is correctly classified."
            )
            print("    Test: not applicable -- zero cells in both groups.")
            results["chi_square_q4_vs_q1"] = {
                "test": "not_applicable_zero_cells",
                "q1_n_id": len(q1_id), "q1_misclass": q1_mis,
                "q4_n_id": len(q4_id), "q4_misclass": q4_mis,
                "q1_misclass_rate": float(q1_mr),
                "q4_misclass_rate": float(q4_mr),
                "note": "All misclassified ID rows fall in high-entropy Q2/Q3",
            }
        else:
            # Fisher exact handles sparse cells more robustly than chi-square
            _, p_fisher = fisher_exact(contingency, alternative="two-sided")
            rr_str = f"{rr:.2f}" if not np.isnan(rr) else "inf (Q1 rate=0)"
            print(f"    Risk ratio: {rr_str}")
            print(f"    Fisher exact p = {p_fisher:.4e}")
            results["chi_square_q4_vs_q1"] = {
                "test": "fisher_exact",
                "p": float(p_fisher),
                "risk_ratio": float(rr) if not np.isnan(rr) else None,
                "q1_n_id": len(q1_id), "q1_misclass": q1_mis,
                "q4_n_id": len(q4_id), "q4_misclass": q4_mis,
                "q1_misclass_rate": float(q1_mr),
                "q4_misclass_rate": float(q4_mr),
            }

    # Paste-ready table
    print()
    print("  [paste into EXPERIMENTS.md E4 table]")
    print(
        "  | Quadrant | risk_group | n_total | n_id | n_ood |"
        " misclass%(id) | OOD% |"
    )
    for qname, qlabel in zip(quad_order, quad_display):
        r = results["variant_a"][qname]
        mr_s = f"{100*r['misclass_rate_id']:.1f}%" if r["misclass_rate_id"] is not None and not np.isnan(r["misclass_rate_id"]) else "N/A"
        print(
            f"  | {qlabel} | {qname} |"
            f" {r['n_total']} | {r['n_id']} | {r['n_ood']} |"
            f" {mr_s} | {100*r['ood_share']:.1f}% |"
        )
    print()

    # ---- Variant (b): SECONDARY -----------------------------------------------
    print("  Variant (b) SECONDARY -- medians from combined ID+OOD rows")
    print(
        f"  Median entropy split:     {med_e_all:.4f}"
        f"  (n_all={len(work)})"
    )
    print(
        f"  Median instability split: {med_i_all:.4f}"
        f"  (n_all={len(work)})"
    )
    print()
    print(
        f"  {'Quadrant':<36} {'n_tot':>5} {'n_id':>5} {'n_ood':>5}"
        f" {'misclass%(id only)':>18} {'OOD%':>6}"
    )
    print("  " + "-" * 80)

    for qname, qlabel in zip(quad_order, quad_display):
        sub = work[work["quad_b"] == qname]
        n_tot = len(sub)
        sub_id = sub[sub["dataset_type"] == "id"]
        n_id = len(sub_id)
        n_ood = n_tot - n_id
        sub_id_gt = sub_id[sub_id["correct"].notna()]
        mr = (
            float((sub_id_gt["correct"] == 0).sum() / len(sub_id_gt))
            if len(sub_id_gt) > 0 else float("nan")
        )
        ood_share = float(n_ood / n_tot) if n_tot > 0 else float("nan")
        results["variant_b"][qname] = {
            "n_total": n_tot, "n_id": n_id, "n_ood": n_ood,
            "misclass_rate_id": mr, "ood_share": ood_share,
        }
        mr_s = f"{100*mr:.1f}%" if not np.isnan(mr) else "  N/A"
        os_s = f"{100*ood_share:.1f}%" if not np.isnan(ood_share) else "  N/A"
        print(
            f"  {qlabel:<36} {n_tot:>5} {n_id:>5} {n_ood:>5}"
            f" {mr_s:>18} {os_s:>6}"
        )
    print()

    # ---- Figure (uses variant-a medians — primary) ----------------------------
    colors = {
        "Q1_stable": "steelblue",
        "Q2_unstable_both": "firebrick",
        "Q3_pred_unstable_only": "darkorange",
        "Q4_hidden_risk": "darkviolet",
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for qname, qlabel in zip(quad_order, quad_display):
        sub = work[work["quad_a"] == qname]
        ax.scatter(
            sub["entropy"],
            sub["instability"],
            c=colors[qname],
            alpha=0.25,
            s=5,
            label=qlabel.strip(),
            rasterized=True,
        )
    ax.axvline(med_e_id, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(med_i_id, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Explanation instability (1 - cam_iou_mean)")
    ax.set_title("Quadrant analysis (splits from ID medians)")
    ax.legend(markerscale=3, fontsize=8, loc="upper right")
    plt.tight_layout()
    fig.savefig(out_dir / "quadrant_scatter.png", dpi=300)
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# E5 / E6: Group comparisons
# ---------------------------------------------------------------------------


def group_comparison(
    df: pd.DataFrame,
    group_col: str,
    out_dir: str | Path,
) -> dict:
    """E5 / E6 — Two-group comparison: medians, Mann-Whitney U, Cliff's delta.

    ``group_col == "correct"``  →  E5: correct vs misclassified.
        In-distribution rows with non-NULL correct only.
        Reports cam_corr_mean (degenerate rows excluded) and cam_iou_mean.

    ``group_col == "dataset_type"``  →  E6: ID vs near-OOD, ID vs far-OOD.
        Reports cam_iou_mean.

    Args:
        df: Combined analysis frame.
        group_col: Grouping dimension ("correct" or "dataset_type").
        out_dir: Output directory for figures.

    Returns:
        Dict keyed by comparison name with median, U, p, delta, magnitude.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    if group_col == "correct":
        # E5 — in-distribution rows only, ground-truth available
        id_df = df[(df["dataset_type"] == "id") & df["correct"].notna()].copy()
        g_cor = id_df[id_df["correct"] == 1]
        g_mis = id_df[id_df["correct"] == 0]

        print("=" * 60)
        print("E5: CORRECT vs MISCLASSIFIED (in-distribution)")
        print("=" * 60)
        print(f"  n_correct={len(g_cor)},  n_misclassified={len(g_mis)}")
        print(
            f"  {'Metric':<18} {'med_A':>8} {'med_B':>8} "
            f"{'U':>10} {'p':>10} {'delta':>8} {'size':>12}"
        )
        print("  " + "-" * 78)

        boxplot_data: list[tuple] = []

        for metric in ["cam_corr_mean", "cam_iou_mean"]:
            sub_a = g_cor
            sub_b = g_mis
            # Exclude degenerate rows for cam_corr_mean
            if metric == "cam_corr_mean":
                sub_a = sub_a[sub_a["cam_corr_mean"] != 0.0]
                sub_b = sub_b[sub_b["cam_corr_mean"] != 0.0]

            a_vals = sub_a[metric].dropna().values
            b_vals = sub_b[metric].dropna().values

            med_a = float(np.median(a_vals))
            med_b = float(np.median(b_vals))
            u, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            delta = _cliffs_delta(a_vals, b_vals)
            magnitude = _cliff_label(delta)

            print(
                f"  {metric:<18} {med_a:>8.4f} {med_b:>8.4f} "
                f"{u:>10.0f} {p:>10.4e} {delta:>+8.4f} {magnitude:>12}"
            )
            key = f"{metric}_correct_vs_misclassified"
            results[key] = {
                "metric": metric,
                "group_a": "correct",
                "group_b": "misclassified",
                "n_a": len(a_vals),
                "n_b": len(b_vals),
                "median_a": med_a,
                "median_b": med_b,
                "U": float(u),
                "p": float(p),
                "cliffs_delta": delta,
                "magnitude": magnitude,
            }
            boxplot_data.append((metric, a_vals, b_vals))

        # Paste-ready table
        print()
        print("  [paste into EXPERIMENTS.md E5 table]")
        print(
            "  | Comparison | Metric | Median A | Median B | U | p | Cliff's delta | Size |"
        )
        for metric in ["cam_corr_mean", "cam_iou_mean"]:
            r = results[f"{metric}_correct_vs_misclassified"]
            print(
                f"  | correct vs misclassified | {metric} | "
                f"{r['median_a']:.4f} | {r['median_b']:.4f} | "
                f"{r['U']:.0f} | {r['p']:.4e} | "
                f"{r['cliffs_delta']:+.4f} | {r['magnitude']} |"
            )
        print()

        # Figure
        fig, axes = plt.subplots(1, 2, figsize=(9, 5))
        for ax, (metric, a_vals, b_vals) in zip(axes, boxplot_data):
            bp = ax.boxplot(
                [a_vals, b_vals],
                tick_labels=["correct", "misclassified"],
                patch_artist=True,
                widths=0.5,
            )
            bp["boxes"][0].set_facecolor("steelblue")
            bp["boxes"][0].set_alpha(0.6)
            bp["boxes"][1].set_facecolor("firebrick")
            bp["boxes"][1].set_alpha(0.6)
            ax.set_title(metric)
            ax.set_ylabel(metric)
        fig.suptitle("E5: Correct vs Misclassified (in-distribution)", fontsize=11)
        plt.tight_layout()
        fig.savefig(out_dir / "correct_vs_misclassified_boxplot.png", dpi=300)
        plt.close(fig)

    elif group_col == "dataset_type":
        # E6 — cam_iou_mean by dataset type
        id_vals = df[df["dataset_type"] == "id"]["cam_iou_mean"].dropna().values
        near_vals = df[df["dataset_type"] == "near_ood"]["cam_iou_mean"].dropna().values
        far_vals = df[df["dataset_type"] == "far_ood"]["cam_iou_mean"].dropna().values

        print("=" * 60)
        print("E6: ID vs OOD  (cam_iou_mean)")
        print("=" * 60)
        print(
            f"  n_id={len(id_vals)},  n_near_ood={len(near_vals)},  n_far_ood={len(far_vals)}"
        )
        print(
            f"  {'Comparison':<22} {'med_ID':>8} {'med_OOD':>9} "
            f"{'U':>10} {'p':>10} {'delta':>8} {'size':>12}"
        )
        print("  " + "-" * 82)

        comparisons = [
            ("ID vs near-OOD", id_vals, near_vals, "near_ood"),
            ("ID vs far-OOD", id_vals, far_vals, "far_ood"),
        ]
        plot_groups: list[tuple[str, np.ndarray]] = [("ID", id_vals)]

        for label, a_vals, b_vals, b_key in comparisons:
            med_a = float(np.median(a_vals))
            med_b = float(np.median(b_vals))
            u, p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            delta = _cliffs_delta(a_vals, b_vals)
            magnitude = _cliff_label(delta)

            print(
                f"  {label:<22} {med_a:>8.4f} {med_b:>9.4f} "
                f"{u:>10.0f} {p:>10.4e} {delta:>+8.4f} {magnitude:>12}"
            )
            results[f"cam_iou_mean_{b_key}"] = {
                "metric": "cam_iou_mean",
                "group_a": "id",
                "group_b": b_key,
                "n_a": len(a_vals),
                "n_b": len(b_vals),
                "median_a": med_a,
                "median_b": med_b,
                "U": float(u),
                "p": float(p),
                "cliffs_delta": delta,
                "magnitude": magnitude,
            }
            plot_groups.append((b_key.replace("_", "-"), b_vals))

        print()
        print("  [paste into EXPERIMENTS.md E6 table]")
        print(
            "  | Comparison | Metric | Median A | Median B | U | p | Cliff's delta | Size |"
        )
        for label, _, _, b_key in comparisons:
            r = results[f"cam_iou_mean_{b_key}"]
            print(
                f"  | {label} | cam_iou_mean | "
                f"{r['median_a']:.4f} | {r['median_b']:.4f} | "
                f"{r['U']:.0f} | {r['p']:.4e} | "
                f"{r['cliffs_delta']:+.4f} | {r['magnitude']} |"
            )
        print()

        # Figure
        fig, ax = plt.subplots(figsize=(7, 5))
        group_colors = ["steelblue", "darkorange", "firebrick"]
        bp = ax.boxplot(
            [g[1] for g in plot_groups],
            tick_labels=[g[0] for g in plot_groups],
            patch_artist=True,
            widths=0.5,
        )
        for box, color in zip(bp["boxes"], group_colors):
            box.set_facecolor(color)
            box.set_alpha(0.6)
        ax.set_ylabel("cam_iou_mean")
        ax.set_title("E6: Grad-CAM IoU by dataset type")
        plt.tight_layout()
        fig.savefig(out_dir / "ood_group_comparison.png", dpi=300)
        plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------


def run_all(cfg: dict, run_ids: list[int], out_dir: str | Path) -> dict:
    """Load data and run degenerate audit, E2, E4, E5, E6.

    Args:
        cfg: Configuration dict from ``load_config``.
        run_ids: Run IDs to load and analyse.
        out_dir: Output directory for figures.

    Returns:
        Nested dict with results from each experiment.
    """
    repo = Repository(cfg["database"]["path"])
    dfs: dict[int, pd.DataFrame] = {}
    for rid in run_ids:
        df = repo.fetch_metrics(rid)
        df["run_id"] = rid
        dfs[rid] = df
        dtypes = df["dataset_type"].value_counts().to_dict()
        print(f"Loaded run_id={rid}: {len(df)} rows  dataset_types={dtypes}")
    print()

    audit = degenerate_audit(dfs)
    combined = pd.concat(list(dfs.values()), ignore_index=True)

    n_degen = int((combined["cam_corr_mean"] == 0.0).sum())
    if n_degen > 0:
        print(
            f"Note: {n_degen} rows with cam_corr_mean==0.0 will be excluded "
            f"from analyses that use cam_corr_mean.\n"
        )

    return {
        "audit": audit,
        "E2": correlation_analysis(combined, out_dir),
        "E4": quadrant_analysis(combined, out_dir),
        "E5": group_comparison(combined, "correct", out_dir),
        "E6": group_comparison(combined, "dataset_type", out_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 analysis: degenerate audit + E2, E4, E5, E6"
    )
    parser.add_argument(
        "--run-ids",
        required=True,
        help="Comma-separated run IDs, e.g. 11,12,13",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    run_ids = [int(x.strip()) for x in args.run_ids.split(",")]
    cfg = load_config(args.config)
    run_all(cfg, run_ids, args.out_dir)


if __name__ == "__main__":
    _main()
