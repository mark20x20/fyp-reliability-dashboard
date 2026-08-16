"""E3 — Stratified analysis on the prediction-stable subset (central result).

Selects ``pred_agreement == 1.0`` rows, bins by confidence, and reports
the distribution of ``cam_corr_mean`` and ``cam_iou_mean`` per bin.
Overlays B2 (random floor) and B3 (same-class reference) from the baselines
table.

The claim: within the top confidence bin every sample is indistinguishable
on the prediction side.  If ``cam_corr_mean`` still spans a wide range
within that bin, prediction-side metrics do not determine explanation
stability.

CLI:
    python src/stratified_analysis.py --run-id 11
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
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.repository import Repository
from src.utils import load_config

# Confidence bins per EXPERIMENTS.md E3
_BINS = [(0.90, 0.95), (0.95, 0.99), (0.99, 1.001)]
_BIN_LABELS = ["0.90-0.95", "0.95-0.99", "0.99-1.00"]


def _top_bin_tail_analysis(
    top_df: pd.DataFrame,
    b3_corr: float | None,
    b3_iou: float | None,
) -> dict:
    """Tail quantification and misclassification test for the top confidence bin.

    Reports for both ``cam_corr_mean`` and ``cam_iou_mean``:
      - Percentile table: P1, P5, P10, P25, P50.
      - Count and percentage of samples falling below three thresholds:
          * B3 (cross-image reference)
          * B3 + 0.05
          * midpoint between B3 and the bin median
      - Fisher exact test: misclassification rate in the bottom 10th-percentile
        decile vs the rest, with risk ratio and 95% CI.

    Misclassification uses ``correct`` column; if correct is NULL for any
    rows they are excluded (should not happen for an ID-only top bin).

    Args:
        top_df: Rows in the top confidence bin (pred_agreement == 1.0,
                confidence in [0.99, 1.00]).
        b3_corr: B3 cross-image reference for cam_corr_mean.
        b3_iou: B3 cross-image reference for cam_iou_mean.

    Returns:
        Dict with per-metric tail statistics and Fisher test results.
    """
    results: dict = {}

    print("=" * 60)
    print("E3 TAIL ANALYSIS: TOP BIN (0.99-1.00, pred_agreement=1.0)")
    print(f"  n = {len(top_df)}")
    print("=" * 60)

    for metric, b3_val in [("cam_corr_mean", b3_corr), ("cam_iou_mean", b3_iou)]:
        vals = top_df[metric].dropna().values
        n = len(vals)
        med = float(np.median(vals))

        print()
        print(f"  {metric}  (n={n})")

        # -- Percentile table --------------------------------------------------
        pctiles = [1, 5, 10, 25, 50]
        pct_vals = {p: float(np.percentile(vals, p)) for p in pctiles}
        print(
            f"    {'Percentile':<12} "
            + "  ".join(f"P{p:>2}" for p in pctiles)
        )
        print(
            f"    {'Value':<12} "
            + "  ".join(f"{pct_vals[p]:>6.4f}" for p in pctiles)
        )

        # -- Counts below thresholds -------------------------------------------
        if b3_val is not None:
            thresholds = [
                (b3_val, f"B3 ({b3_val:.4f})"),
                (b3_val + 0.05, f"B3+0.05 ({b3_val+0.05:.4f})"),
                ((b3_val + med) / 2, f"midpt(B3,med) ({(b3_val+med)/2:.4f})"),
            ]
            print(f"    {'Threshold':<28} {'n_below':>8} {'%below':>8}")
            print("    " + "-" * 46)
            thr_results = {}
            for thr, thr_label in thresholds:
                n_below = int((vals < thr).sum())
                pct_below = 100.0 * n_below / n
                print(f"    {thr_label:<28} {n_below:>8} {pct_below:>7.1f}%")
                thr_results[thr_label] = {"n_below": n_below, "pct_below": pct_below}
            results.setdefault(metric, {})["threshold_counts"] = thr_results

        # -- Fisher exact: bottom-decile misclassification ---------------------
        p10_val = float(np.percentile(vals, 10))
        bottom_mask = top_df[metric] < p10_val
        bottom = top_df[bottom_mask]
        rest = top_df[~bottom_mask]

        bottom_id = bottom[bottom["correct"].notna()]
        rest_id = rest[rest["correct"].notna()]

        n1 = len(bottom_id)
        n2 = len(rest_id)
        n1_mis = int((bottom_id["correct"] == 0).sum()) if n1 > 0 else 0
        n2_mis = int((rest_id["correct"] == 0).sum()) if n2 > 0 else 0
        n1_cor = n1 - n1_mis
        n2_cor = n2 - n2_mis

        r1 = n1_mis / n1 if n1 > 0 else float("nan")
        r2 = n2_mis / n2 if n2 > 0 else float("nan")
        rr = r1 / r2 if (r2 > 0 and not np.isnan(r2)) else float("nan")

        # Risk ratio 95% CI (log method; requires n1_mis > 0, n2_mis > 0)
        if n1_mis > 0 and n2_mis > 0 and n1 > n1_mis and n2 > n2_mis:
            se_log = np.sqrt(1/n1_mis - 1/n1 + 1/n2_mis - 1/n2)
            rr_lo = float(np.exp(np.log(rr) - 1.96 * se_log))
            rr_hi = float(np.exp(np.log(rr) + 1.96 * se_log))
        else:
            rr_lo = rr_hi = float("nan")

        table = np.array([[n1_cor, n1_mis], [n2_cor, n2_mis]])
        _, p_fisher = fisher_exact(table, alternative="two-sided")

        print()
        print(
            f"    Fisher exact: bottom 10th pctile (< {p10_val:.4f}) vs rest"
        )
        print(
            f"      Bottom decile: n={n1}  misclass={n1_mis}"
            f"  rate={100*r1:.2f}%"
        )
        print(
            f"      Rest:          n={n2}  misclass={n2_mis}"
            f"  rate={100*r2:.2f}%"
        )
        rr_ci_str = (
            f"[{rr_lo:.2f}, {rr_hi:.2f}]"
            if not np.isnan(rr_lo) else "N/A"
        )
        print(
            f"      Risk ratio: {rr:.2f}  95% CI {rr_ci_str}"
            f"  p = {p_fisher:.4e}"
        )

        # Interpretation
        if np.isnan(rr):
            interp = "Insufficient data for risk ratio."
        elif p_fisher >= 0.05:
            interp = (
                "Not significant (p >= 0.05): the bottom-decile misclassification "
                "rate does not reliably differ from the rest within this controlled "
                "subset. The range statistic alone supports the distributional claim."
            )
        else:
            interp = (
                f"Significant (p = {p_fisher:.2e}): the bottom {metric} decile "
                f"misclassifies at {100*r1:.1f}% vs {100*r2:.1f}% in the rest "
                f"(RR = {rr:.2f}). Direct evidence for O2 within the controlled subset."
            )
        print(f"      Interpretation: {interp}")

        results.setdefault(metric, {}).update({
            "percentiles": pct_vals,
            "p10_threshold": p10_val,
            "bottom_n": n1, "bottom_misclass": n1_mis, "bottom_rate": r1,
            "rest_n": n2, "rest_misclass": n2_mis, "rest_rate": r2,
            "risk_ratio": rr, "rr_ci_lo": rr_lo, "rr_ci_hi": rr_hi,
            "p_fisher": float(p_fisher),
        })

    # Headline conclusion
    print()
    print("  Summary for the report:")
    corr_r = results.get("cam_corr_mean", {})
    iou_r = results.get("cam_iou_mean", {})
    if b3_corr is not None:
        n_below_b3_corr = corr_r.get("threshold_counts", {})
        corr_below_key = next((k for k in n_below_b3_corr if "B3 (" in k), None)
        if corr_below_key:
            cb = n_below_b3_corr[corr_below_key]
            print(
                f"    cam_corr_mean: {cb['n_below']} / {len(top_df)}"
                f" ({cb['pct_below']:.1f}%) samples fall below B3={b3_corr:.4f}"
            )
    if b3_iou is not None:
        iou_below_key = next(
            (k for k in iou_r.get("threshold_counts", {}) if "B3 (" in k), None
        )
        if iou_below_key:
            ib = iou_r["threshold_counts"][iou_below_key]
            print(
                f"    cam_iou_mean:  {ib['n_below']} / {len(top_df)}"
                f" ({ib['pct_below']:.1f}%) samples fall below B3={b3_iou:.4f}"
            )
    print()

    return results


def stratified_analysis(
    df: pd.DataFrame,
    out_dir: str | Path,
    baselines: dict | None = None,
) -> dict:
    """E3 — Stratified analysis on the prediction-stable subset.

    Steps:
      1. Select ``pred_agreement == 1.0``.
      2. Bin by confidence: [0.90, 0.95), [0.95, 0.99), [0.99, 1.00].
      3. Report n, min, Q1, median, Q3, max for ``cam_corr_mean`` and
         ``cam_iou_mean`` per bin.
      4. Plot against B2 and B3 baseline values.

    Rows where ``cam_corr_mean == 0.0`` (degenerate maps) are excluded
    and the exclusion count is stated.

    For the top bin, also reports the interquartile range (IQR) and the
    ratio of max to min of ``cam_corr_mean``.

    Figures saved:
      - ``stratified_by_confidence_bin.png``
      - ``pred_stable_subset_distribution.png``

    Args:
        df: Full analysis frame for one run (in-distribution).
        out_dir: Output directory for figures.
        baselines: Nested dict ``{baseline_type: {metric_name: value}}``.
                   If None, baseline lines are omitted from plots.

    Returns:
        Dict with per-bin summary statistics and the target sentence.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Exclude degenerate rows
    n_total = len(df)
    df_clean = df[df["cam_corr_mean"] != 0.0].copy()
    n_excluded = n_total - len(df_clean)

    # Select prediction-stable rows
    df_stable = df_clean[df_clean["pred_agreement"] == 1.0].copy()
    n_stable = len(df_stable)

    print("=" * 60)
    print("E3: STRATIFIED ANALYSIS -- PREDICTION-STABLE SUBSET")
    print("=" * 60)
    print(f"  Total rows in run:          {n_total}")
    print(f"  Degenerate (cam_corr==0.0): {n_excluded} excluded")
    print(f"  pred_agreement == 1.0:      {n_stable}")
    print()

    # Baseline values (for plot annotation)
    b2_corr = b3_corr = b2_iou = b3_iou = None
    if baselines:
        # Metric names in the baselines table are "correlation" and "iou"
        b2_corr = baselines.get("lower", {}).get("correlation") or baselines.get("lower", {}).get("cam_corr_mean")
        b3_corr = baselines.get("cross_image", {}).get("correlation") or baselines.get("cross_image", {}).get("cam_corr_mean")
        b2_iou = baselines.get("lower", {}).get("iou") or baselines.get("lower", {}).get("cam_iou_mean")
        b3_iou = baselines.get("cross_image", {}).get("iou") or baselines.get("cross_image", {}).get("cam_iou_mean")

    results: dict = {
        "n_total": n_total,
        "n_excluded_degen": n_excluded,
        "n_stable": n_stable,
        "bins": {},
    }

    # ---- Per-bin statistics ---------------------------------------------------
    for metric in ["cam_corr_mean", "cam_iou_mean"]:
        print(f"  Metric: {metric}")
        print(
            f"  {'Bin':<12} {'n':>5} {'min':>7} {'Q1':>7} "
            f"{'median':>8} {'Q3':>7} {'max':>7}"
        )
        print("  " + "-" * 58)

        for (lo, hi), label in zip(_BINS, _BIN_LABELS):
            bin_df = df_stable[
                (df_stable["confidence"] >= lo) & (df_stable["confidence"] < hi)
            ]
            n = len(bin_df)
            vals = bin_df[metric].dropna().values

            if n == 0:
                print(f"  {label:<12} {0:>5}  (no data)")
                results["bins"].setdefault(label, {})[metric] = None
                continue

            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            summary = {
                "n": n,
                "min": float(vals.min()),
                "Q1": float(q1),
                "median": float(med),
                "Q3": float(q3),
                "max": float(vals.max()),
            }
            results["bins"].setdefault(label, {})[metric] = summary

            print(
                f"  {label:<12} {n:>5} {vals.min():>7.4f} {q1:>7.4f} "
                f"{med:>8.4f} {q3:>7.4f} {vals.max():>7.4f}"
            )
        print()

    # ---- Top-bin detail for cam_corr_mean ------------------------------------
    top_lo, top_hi = _BINS[-1]
    top_label = _BIN_LABELS[-1]
    top_df = df_stable[
        (df_stable["confidence"] >= top_lo) & (df_stable["confidence"] < top_hi)
    ]
    top_corr = top_df["cam_corr_mean"].dropna().values

    if len(top_corr) > 0:
        top_q1, top_q3 = np.percentile(top_corr, [25, 75])
        top_iqr = float(top_q3 - top_q1)
        top_ratio = float(top_corr.max() / top_corr.min()) if top_corr.min() > 0 else float("nan")
        results["top_bin_iqr"] = top_iqr
        results["top_bin_max_to_min_ratio"] = top_ratio
        results["top_bin_n"] = len(top_df)
        results["top_bin_corr_min"] = float(top_corr.min())
        results["top_bin_corr_max"] = float(top_corr.max())

        print(f"  Top-bin detail  ({top_label},  n={len(top_df)})")
        print(f"    cam_corr_mean IQR:       {top_iqr:.4f}")
        print(f"    cam_corr_mean max/min:   {top_ratio:.2f}")
        print()

        # Build the target sentence for the report
        b2_str = f"{b2_corr:.3f}" if b2_corr is not None else "?"
        b3_str = f"{b3_corr:.3f}" if b3_corr is not None else "?"
        target_sentence = (
            f"Restricted to samples with confidence >= 0.99 whose predicted "
            f"class was identical across all 30 passes (n = {len(top_df)}), "
            f"mean pairwise Grad-CAM correlation ranged from "
            f"{top_corr.min():.3f} to {top_corr.max():.3f}, "
            f"against a random-map floor of {b2_str} and a same-class reference of {b3_str}."
        )
        results["target_sentence"] = target_sentence
        print("  [target sentence for report]")
        print(f"  {target_sentence}")
        print()

    # ---- Paste-ready table ---------------------------------------------------
    print("  [paste into EXPERIMENTS.md E3 table - cam_corr_mean]")
    print(f"  | Bin | n | corr min | Q1 | median | Q3 | max |")
    for label in _BIN_LABELS:
        s = results["bins"].get(label, {}).get("cam_corr_mean")
        if s is None:
            print(f"  | {label} | 0 | - | - | - | - | - |")
        else:
            print(
                f"  | {label} | {s['n']} | {s['min']:.4f} | "
                f"{s['Q1']:.4f} | {s['median']:.4f} | "
                f"{s['Q3']:.4f} | {s['max']:.4f} |"
            )
    print()
    print("  [paste into EXPERIMENTS.md E3 table - cam_iou_mean]")
    print(f"  | Bin | n | iou min | Q1 | median | Q3 | max |")
    for label in _BIN_LABELS:
        s = results["bins"].get(label, {}).get("cam_iou_mean")
        if s is None:
            print(f"  | {label} | 0 | - | - | - | - | - |")
        else:
            print(
                f"  | {label} | {s['n']} | {s['min']:.4f} | "
                f"{s['Q1']:.4f} | {s['median']:.4f} | "
                f"{s['Q3']:.4f} | {s['max']:.4f} |"
            )
    print()

    # ---- Tail analysis for the top bin ---------------------------------------
    if len(top_df) > 0:
        tail_results = _top_bin_tail_analysis(top_df, b3_corr, b3_iou)
        results["top_bin_tail"] = tail_results

    # ---- Figure 1: per-bin box plot ------------------------------------------
    corr_bin_data = []
    iou_bin_data = []
    bin_ns = []

    for (lo, hi), label in zip(_BINS, _BIN_LABELS):
        bin_df = df_stable[
            (df_stable["confidence"] >= lo) & (df_stable["confidence"] < hi)
        ]
        corr_bin_data.append(bin_df["cam_corr_mean"].dropna().values)
        iou_bin_data.append(bin_df["cam_iou_mean"].dropna().values)
        bin_ns.append(len(bin_df))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    tick_labels = [f"{l}\n(n={n})" for l, n in zip(_BIN_LABELS, bin_ns)]

    for ax, data, metric, b2_val, b3_val in [
        (axes[0], corr_bin_data, "cam_corr_mean", b2_corr, b3_corr),
        (axes[1], iou_bin_data, "cam_iou_mean", b2_iou, b3_iou),
    ]:
        non_empty = [d for d in data if len(d) > 0]
        if non_empty:
            bp = ax.boxplot(
                data if all(len(d) > 0 for d in data) else non_empty,
                tick_labels=tick_labels if all(len(d) > 0 for d in data) else tick_labels[:len(non_empty)],
                patch_artist=True,
                widths=0.5,
            )
            for box in bp["boxes"]:
                box.set_facecolor("steelblue")
                box.set_alpha(0.6)

        if b2_val is not None:
            ax.axhline(
                b2_val, color="firebrick", linestyle="--",
                linewidth=1.2, label=f"B2 random ({b2_val:.3f})",
            )
        if b3_val is not None:
            ax.axhline(
                b3_val, color="darkorange", linestyle="--",
                linewidth=1.2, label=f"B3 same-class ({b3_val:.3f})",
            )
        ax.set_ylabel(metric)
        ax.set_xlabel("Confidence bin")
        ax.set_title(f"{metric} by confidence bin\n(pred_agreement == 1.0)")
        if b2_val is not None or b3_val is not None:
            ax.legend(fontsize=8)

    fig.suptitle("E3: Stratified analysis — prediction-stable subset", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "stratified_by_confidence_bin.png", dpi=300)
    plt.close(fig)

    # ---- Figure 2: distribution of cam_corr_mean in stable subset ------------
    fig, ax = plt.subplots(figsize=(7, 5))
    # KDE per bin using histogram approach
    colors_bin = ["steelblue", "darkorange", "firebrick"]
    for data, label, color in zip(corr_bin_data, _BIN_LABELS, colors_bin):
        if len(data) > 1:
            ax.hist(
                data,
                bins=30,
                density=True,
                alpha=0.4,
                color=color,
                label=f"conf {label} (n={len(data)})",
            )
    if b2_corr is not None:
        ax.axvline(b2_corr, color="black", linestyle="--", linewidth=1.2,
                   label=f"B2 random ({b2_corr:.3f})")
    if b3_corr is not None:
        ax.axvline(b3_corr, color="grey", linestyle="-.", linewidth=1.2,
                   label=f"B3 same-class ({b3_corr:.3f})")
    ax.set_xlabel("cam_corr_mean")
    ax.set_ylabel("Density")
    ax.set_title(
        "Distribution of Grad-CAM correlation\n"
        "Prediction-stable subset (pred_agreement == 1.0)"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "pred_stable_subset_distribution.png", dpi=300)
    plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6 E3: stratified analysis on prediction-stable subset"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        type=int,
        help="Run ID for the in-distribution dataset (e.g. 11)",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo = Repository(cfg["database"]["path"])

    df = repo.fetch_metrics(args.run_id)
    print(f"Loaded run_id={args.run_id}: {len(df)} rows  "
          f"dataset_types={df['dataset_type'].value_counts().to_dict()}")
    print()

    baselines = repo.fetch_baselines(args.run_id)
    if baselines:
        print("Baselines:")
        for btype, metrics in baselines.items():
            for mname, val in metrics.items():
                print(f"  {btype:<15} {mname:<20} {val:.6f}")
        print()
    else:
        print("Warning: no baselines found for this run_id.  Baseline lines omitted from plots.\n")

    stratified_analysis(df, args.out_dir, baselines=baselines)


if __name__ == "__main__":
    _main()
