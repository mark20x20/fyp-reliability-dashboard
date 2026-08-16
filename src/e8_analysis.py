"""E8 analysis — corruption robustness experiment.

Analyses the corrupted-imagenette run (Phase 7) to quantify how prediction
uncertainty and explanation stability degrade with corruption type and severity.

Key outputs
-----------
* Severity table: n, accuracy, confidence, entropy, pred_agreement,
  cam_corr_mean, cam_iou_mean, high-confidence errors per severity level
  (pooled across types and per type).
* Spearman rho (severity vs cam_iou_mean) with bootstrap 95 % CI.
* Kruskal-Wallis across severities with Bonferroni-corrected post-hoc
  Mann-Whitney U tests.
* Re-run E4 quadrant analysis on corrupted rows only.
* Corrupted E3: pred_agreement==1.0 AND confidence>=0.90 subset, Fisher
  exact on bottom cam_iou_mean decile vs rest.
* Figure: corruption_severity_trend.png.

All print output uses ASCII-only characters to avoid cp932 encoding errors on
Japanese Windows.

CLI::

    python src/e8_analysis.py --run-id 14
    python src/e8_analysis.py --run-id 14 --out-dir outputs/figures
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
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.repository import Repository
from src.utils import load_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Confidence threshold for "high-confidence error" classification
_HCE_THRESHOLD = 0.90

# Display order for severities
_SEVERITIES = [1, 2, 3, 4, 5]

# Bonferroni-corrected alpha for 10 pairwise severity comparisons
_ALPHA_BONF = 0.05 / 10


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Spearman rho with 1000-resample bootstrap 95% CI.

    Returns:
        (rho, ci_lo, ci_hi, p_value)
    """
    rho, p = spearmanr(x, y)
    rng    = np.random.default_rng(seed)
    n      = len(x)
    boot   = np.empty(n_boot)
    for i in range(n_boot):
        idx      = rng.integers(0, n, size=n)
        boot[i], _ = spearmanr(x[idx], y[idx])
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    return float(rho), ci_lo, ci_hi, float(p)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_corrupted_df(run_id: int, cfg: dict) -> pd.DataFrame:
    """Return a DataFrame of all corrupted-run rows with metadata columns."""
    repo = Repository(cfg["database"]["path"])
    conn = repo.conn

    df = pd.read_sql_query(
        """
        SELECT
            i.image_id,
            i.corruption_type,
            i.corruption_severity  AS severity,
            i.true_label,
            p.pred_label,
            p.correct,
            p.confidence,
            p.entropy,
            p.pred_variance,
            p.pred_agreement,
            p.mutual_information,
            e.cam_corr_mean,
            e.cam_corr_std,
            e.cam_iou_mean,
            e.topk_overlap
        FROM images i
        JOIN predictions p ON p.image_id = i.image_id
        JOIN explanations e ON e.image_id = i.image_id
        WHERE i.run_id = ?
        """,
        conn,
        params=(run_id,),
    )
    return df


# ---------------------------------------------------------------------------
# Severity table
# ---------------------------------------------------------------------------

def _severity_table(df: pd.DataFrame, label: str = "pooled") -> pd.DataFrame:
    """Compute per-severity summary statistics.

    Args:
        df: Subset of the corrupted DataFrame (may be filtered by type).
        label: Used only for display purposes.

    Returns:
        DataFrame with one row per severity level.
    """
    rows = []
    for sev in _SEVERITIES:
        sub = df[df["severity"] == sev]
        if len(sub) == 0:
            continue
        hce = int(((sub["correct"] == 0) & (sub["confidence"] >= _HCE_THRESHOLD)).sum())
        rows.append({
            "severity":         sev,
            "n":                len(sub),
            "accuracy":         float(sub["correct"].mean()),
            "confidence_mean":  float(sub["confidence"].mean()),
            "entropy_mean":     float(sub["entropy"].mean()),
            "pred_agreement":   float(sub["pred_agreement"].mean()),
            "cam_corr_mean":    float(sub["cam_corr_mean"].mean()),
            "cam_iou_mean":     float(sub["cam_iou_mean"].mean()),
            "hce_n":            hce,
            "hce_rate":         hce / len(sub),
        })
    return pd.DataFrame(rows)


def _print_severity_table(tbl: pd.DataFrame, label: str = "Pooled") -> None:
    """Print a severity table to stdout (ASCII-safe)."""
    print(f"\n--- Severity table: {label} ---")
    header = (
        f"{'sev':>3}  {'n':>5}  {'acc':>6}  {'conf':>6}  "
        f"{'ent':>7}  {'p_agr':>6}  {'corr':>6}  {'iou':>6}  "
        f"{'hce_n':>5}  {'hce%':>5}"
    )
    print(header)
    print("-" * len(header))
    for _, r in tbl.iterrows():
        print(
            f"{int(r['severity']):>3}  {int(r['n']):>5}  "
            f"{r['accuracy']:.4f}  {r['confidence_mean']:.4f}  "
            f"{r['entropy_mean']:.5f}  {r['pred_agreement']:.4f}  "
            f"{r['cam_corr_mean']:.4f}  {r['cam_iou_mean']:.4f}  "
            f"{int(r['hce_n']):>5}  {r['hce_rate'] * 100:.1f}%"
        )


# ---------------------------------------------------------------------------
# Spearman + Kruskal-Wallis
# ---------------------------------------------------------------------------

def _severity_spearman(df: pd.DataFrame, metric: str = "cam_iou_mean") -> dict:
    """Spearman rho between integer severity and *metric*.

    Uses per-image rows (not group means) for maximum statistical power.
    """
    valid = df[["severity", metric]].dropna()
    x = valid["severity"].to_numpy(dtype=float)
    y = valid[metric].to_numpy(dtype=float)
    rho, ci_lo, ci_hi, p = _spearman_with_ci(x, y)
    return {"rho": rho, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p, "n": len(x)}


def _kruskal_wallis(df: pd.DataFrame, metric: str = "cam_iou_mean") -> dict:
    """Kruskal-Wallis test across severity levels with Bonferroni post-hoc."""
    groups = [
        df.loc[df["severity"] == s, metric].dropna().to_numpy()
        for s in _SEVERITIES
        if (df["severity"] == s).any()
    ]
    sev_labels = [s for s in _SEVERITIES if (df["severity"] == s).any()]

    stat, p = kruskal(*groups)

    # Bonferroni-corrected pairwise Mann-Whitney U
    pairs: list[dict] = []
    for i in range(len(sev_labels)):
        for j in range(i + 1, len(sev_labels)):
            u, p_mw = mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            pairs.append({
                "sev_a": sev_labels[i],
                "sev_b": sev_labels[j],
                "u":     float(u),
                "p_raw": float(p_mw),
                "sig":   p_mw < _ALPHA_BONF,
            })
    return {"stat": float(stat), "p": float(p), "pairs": pairs}


# ---------------------------------------------------------------------------
# Corrupted E4 (quadrant re-analysis on corrupted rows)
# ---------------------------------------------------------------------------

def _corrupted_e4(df: pd.DataFrame) -> dict:
    """Re-run quadrant analysis using medians computed from corrupted rows.

    Reports n per quadrant and misclassification rate within each quadrant.
    Uses entropy and cam_iou_mean (inverted for instability), same axes as
    the original E4.
    """
    sub = df[["entropy", "cam_iou_mean", "correct"]].dropna()
    if len(sub) == 0:
        return {}

    med_ent  = float(sub["entropy"].median())
    med_inst = float((1.0 - sub["cam_iou_mean"]).median())

    sub = sub.copy()
    sub["instability"]  = 1.0 - sub["cam_iou_mean"]
    sub["high_ent"]     = sub["entropy"] >= med_ent
    sub["high_inst"]    = sub["instability"] >= med_inst

    def _quad(row) -> str:
        if   not row["high_ent"] and not row["high_inst"]: return "stable"
        elif     row["high_ent"] and     row["high_inst"]: return "unstable_both"
        elif     row["high_ent"] and not row["high_inst"]: return "pred_unstable_only"
        else:                                               return "hidden_risk"

    sub["quadrant"] = sub.apply(_quad, axis=1)

    quadrant_order = ["stable", "pred_unstable_only", "unstable_both", "hidden_risk"]
    result: dict[str, dict] = {}
    for q in quadrant_order:
        qsub = sub[sub["quadrant"] == q]
        result[q] = {
            "n":        int(len(qsub)),
            "miss_rate": float(1.0 - qsub["correct"].mean()) if len(qsub) > 0 else float("nan"),
        }
    result["medians"] = {"entropy": med_ent, "instability": med_inst}
    return result


# ---------------------------------------------------------------------------
# Corrupted E3 (pred_agreement==1 and confidence>=0.90 subset)
# ---------------------------------------------------------------------------

def _corrupted_e3(df: pd.DataFrame) -> dict:
    """E3 variant on corrupted data: stable predictions, high confidence.

    Subset: pred_agreement==1.0 AND confidence>=0.90.
    Fisher exact: bottom cam_iou_mean decile vs rest, correct vs incorrect.
    """
    sub = df[(df["pred_agreement"] == 1.0) & (df["confidence"] >= 0.90)].copy()
    n_total = len(df)
    n_sub   = len(sub)

    if n_sub < 10:
        return {
            "n_total": n_total,
            "n_subset": n_sub,
            "note": "Subset too small for analysis (n < 10).",
        }

    thresh = float(np.percentile(sub["cam_iou_mean"], 10))
    bottom = sub[sub["cam_iou_mean"] <= thresh]
    rest   = sub[sub["cam_iou_mean"] >  thresh]

    # 2x2: bottom/rest vs correct/incorrect
    a = int((bottom["correct"] == 0).sum())  # bottom, incorrect
    b = int((bottom["correct"] == 1).sum())  # bottom, correct
    c = int((rest["correct"]   == 0).sum())  # rest, incorrect
    d = int((rest["correct"]   == 1).sum())  # rest, correct

    table = np.array([[a, b], [c, d]])
    or_, p = fisher_exact(table, alternative="greater")

    # Risk ratio with log-method 95% CI
    p1 = a / (a + b) if (a + b) > 0 else float("nan")
    p2 = c / (c + d) if (c + d) > 0 else float("nan")
    rr = p1 / p2 if (p2 > 0) else float("nan")

    if not np.isnan(rr) and rr > 0 and p1 > 0 and p2 > 0 and (a + b) > 0 and (c + d) > 0:
        se_log_rr = np.sqrt(b / (a * (a + b)) + d / (c * (c + d)))
        rr_lo = float(np.exp(np.log(rr) - 1.96 * se_log_rr))
        rr_hi = float(np.exp(np.log(rr) + 1.96 * se_log_rr))
    else:
        rr_lo = rr_hi = float("nan")

    return {
        "n_total":          n_total,
        "n_subset":         n_sub,
        "iou_decile_10":    thresh,
        "n_bottom":         int(len(bottom)),
        "n_rest":           int(len(rest)),
        "miss_rate_bottom": p1,
        "miss_rate_rest":   p2,
        "risk_ratio":       rr,
        "rr_ci_lo":         rr_lo,
        "rr_ci_hi":         rr_hi,
        "fisher_p":         float(p),
    }


# ---------------------------------------------------------------------------
# Severity-stratified E3 (confound control)
# ---------------------------------------------------------------------------

def _rr_with_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Risk ratio with log-method 95 % CI.

    Table layout::

        bottom  |  a (incorrect)  b (correct)
        rest    |  c (incorrect)  d (correct)

    Returns:
        (rr, ci_lo, ci_hi) — all NaN when undefined.
    """
    p1 = a / (a + b) if (a + b) > 0 else float("nan")
    p2 = c / (c + d) if (c + d) > 0 else float("nan")
    rr = p1 / p2 if (p2 and p2 > 0) else float("nan")
    if np.isnan(rr) or rr <= 0 or np.isnan(p1) or p1 <= 0 or np.isnan(p2) or p2 <= 0:
        return rr, float("nan"), float("nan")
    # Need a > 0 and c > 0 for the log-method SE
    if a == 0 or c == 0:
        return rr, float("nan"), float("nan")
    se = np.sqrt(b / (a * (a + b)) + d / (c * (c + d)))
    lo = float(np.exp(np.log(rr) - 1.96 * se))
    hi = float(np.exp(np.log(rr) + 1.96 * se))
    return rr, lo, hi


def _one_stratum_table(
    sub: pd.DataFrame,
) -> dict:
    """Within-stratum bottom-decile analysis for the E3 stable-prediction subset.

    Args:
        sub: Rows already filtered to pred_agreement==1.0 & confidence>=0.90
             for one stratum value.

    Returns:
        Dict with keys: n, iou_thresh, a, b, c, d, miss_bottom, miss_rest,
        rr, rr_lo, rr_hi, fisher_p, note (only present when too small).
    """
    n = len(sub)
    if n < 20:
        return {"n": n, "note": f"too small (n={n} < 20)"}

    thresh  = float(np.percentile(sub["cam_iou_mean"], 10))
    bottom  = sub[sub["cam_iou_mean"] <= thresh]
    rest    = sub[sub["cam_iou_mean"] >  thresh]

    a = int((bottom["correct"] == 0).sum())
    b = int((bottom["correct"] == 1).sum())
    c = int((rest["correct"]   == 0).sum())
    d = int((rest["correct"]   == 1).sum())

    _, fp = fisher_exact(np.array([[a, b], [c, d]]), alternative="greater")
    rr, rr_lo, rr_hi = _rr_with_ci(a, b, c, d)

    return {
        "n":           n,
        "n_bottom":    int(len(bottom)),
        "n_rest":      int(len(rest)),
        "iou_thresh":  thresh,
        "a": a, "b": b, "c": c, "d": d,
        "miss_bottom": a / (a + b) if (a + b) > 0 else float("nan"),
        "miss_rest":   c / (c + d) if (c + d) > 0 else float("nan"),
        "rr":          rr,
        "rr_lo":       rr_lo,
        "rr_hi":       rr_hi,
        "fisher_p":    float(fp),
    }


def _cmh_test(strata: list[dict]) -> dict:
    """Cochran-Mantel-Haenszel test pooling 2x2 tables across strata.

    Uses the Mantel-Haenszel common odds ratio estimator with
    Robins-Breslow-Greenland (1986) variance for the 95 % CI.

    Table convention per stratum::

        bottom  |  a  b   (row 1 = bottom decile)
        rest    |  c  d   (row 2 = rest)
                   ^  ^ incorrect, correct

    Args:
        strata: List of per-stratum dicts from ``_one_stratum_table``.
                Dicts without key ``"a"`` (too-small strata) are skipped.

    Returns:
        Dict with keys: or_mh, or_lo, or_hi, chi2, p, n_strata_used,
        n_strata_skipped.
    """
    from scipy.stats import chi2 as _chi2_dist

    valid = [s for s in strata if "a" in s]
    if len(valid) < 2:
        return {
            "note": f"Only {len(valid)} usable strata — CMH requires >= 2.",
            "n_strata_used": len(valid),
            "n_strata_skipped": len(strata) - len(valid),
        }

    # Collect per-stratum quantities
    As, Bs, Cs, Ds, Ts = [], [], [], [], []
    for s in valid:
        a, b, c, d = s["a"], s["b"], s["c"], s["d"]
        T = a + b + c + d
        As.append(a); Bs.append(b); Cs.append(c); Ds.append(d); Ts.append(T)

    As = np.array(As, dtype=float)
    Bs = np.array(Bs, dtype=float)
    Cs = np.array(Cs, dtype=float)
    Ds = np.array(Ds, dtype=float)
    Ts = np.array(Ts, dtype=float)

    m1 = As + Bs  # row 1 totals (bottom)
    m2 = Cs + Ds  # row 2 totals (rest)
    n1 = As + Cs  # col 1 totals (incorrect)
    n2 = Bs + Ds  # col 2 totals (correct)

    # CMH chi^2 (no continuity correction)
    E_a   = m1 * n1 / Ts
    Var_a = m1 * m2 * n1 * n2 / (Ts ** 2 * (Ts - 1))
    # Guard against zero variance (stratum with no variation)
    Var_a = np.where(Var_a > 0, Var_a, np.nan)
    valid_mask = ~np.isnan(Var_a)
    if valid_mask.sum() < 2:
        return {
            "note": "Insufficient variance across strata for CMH.",
            "n_strata_used": int(valid_mask.sum()),
            "n_strata_skipped": len(strata) - int(valid_mask.sum()),
        }
    num   = (As[valid_mask] - E_a[valid_mask]).sum()
    denom = Var_a[valid_mask].sum()
    chi2_stat = float(num ** 2 / denom)
    p_val     = float(1.0 - _chi2_dist.cdf(chi2_stat, df=1))

    # MH common OR
    R = float((As * Ds / Ts).sum())
    S = float((Bs * Cs / Ts).sum())
    or_mh = R / S if S > 0 else float("nan")

    # Robins-Breslow-Greenland variance of log(OR_MH)
    if np.isnan(or_mh) or or_mh <= 0 or R == 0 or S == 0:
        or_lo = or_hi = float("nan")
    else:
        Pk   = (As + Ds) / Ts
        Qk   = (Bs + Cs) / Ts
        term1 = float((Pk * As * Ds / Ts).sum()) / (2 * R ** 2)
        term2 = float(((Pk * Bs * Cs + Qk * As * Ds) / Ts).sum()) / (2 * R * S)
        term3 = float((Qk * Bs * Cs / Ts).sum()) / (2 * S ** 2)
        var_log_or = term1 + term2 + term3
        se = np.sqrt(var_log_or)
        or_lo = float(np.exp(np.log(or_mh) - 1.96 * se))
        or_hi = float(np.exp(np.log(or_mh) + 1.96 * se))

    return {
        "or_mh":           or_mh,
        "or_lo":           or_lo,
        "or_hi":           or_hi,
        "chi2":            chi2_stat,
        "p":               p_val,
        "n_strata_used":   len(valid),
        "n_strata_skipped": len(strata) - len(valid),
    }


def _e3_stratified(
    df: pd.DataFrame,
    strat_col: str,
    strat_label: str,
) -> dict:
    """Repeat corrupted E3 bottom-decile test stratified by *strat_col*.

    Within each value of *strat_col*:
    - Filters to pred_agreement==1.0 AND confidence>=0.90.
    - Computes bottom-decile threshold WITHIN that stratum.
    - Reports n, threshold, miss_rate bottom vs rest, RR [95% CI], Fisher p.

    Then pools across strata with a CMH test.

    Args:
        df: Full corrupted-run DataFrame.
        strat_col: Column to stratify on ("severity" or "corruption_type").
        strat_label: Human-readable name for display.

    Returns:
        Dict with per-stratum results and CMH summary.
    """
    stable_sub = df[(df["pred_agreement"] == 1.0) & (df["confidence"] >= 0.90)].copy()

    strat_values = sorted(df[strat_col].dropna().unique().tolist())
    per_stratum: dict = {}
    tables_for_cmh: list[dict] = []

    for val in strat_values:
        sub = stable_sub[stable_sub[strat_col] == val]
        result = _one_stratum_table(sub)
        per_stratum[val] = result
        tables_for_cmh.append(result)

    cmh = _cmh_test(tables_for_cmh)

    return {
        "strat_col":   strat_col,
        "strat_label": strat_label,
        "per_stratum": per_stratum,
        "cmh":         cmh,
    }


def _print_e3_stratified(res: dict) -> None:
    """Print stratified E3 results (ASCII-safe)."""
    label = res["strat_label"]
    print(f"\n--- Corrupted E3 stratified by {label} ---")
    print(
        f"{'stratum':<20}  {'n_sub':>5}  {'iou_thr':>7}  "
        f"{'miss_bot':>8}  {'miss_rst':>8}  "
        f"{'RR':>5}  {'CI_lo':>5}  {'CI_hi':>5}  {'fisher_p':>10}"
    )
    print("-" * 90)
    for val, s in res["per_stratum"].items():
        if "note" in s:
            print(f"  {str(val):<18}  {s.get('n', '?'):>5}  -- {s['note']}")
            continue
        rr   = s["rr"]
        rr_s = f"{rr:.2f}" if not np.isnan(rr) else "undef"
        lo_s = f"{s['rr_lo']:.2f}" if not np.isnan(s['rr_lo']) else "undef"
        hi_s = f"{s['rr_hi']:.2f}" if not np.isnan(s['rr_hi']) else "undef"
        print(
            f"  {str(val):<18}  {s['n']:>5}  {s['iou_thresh']:>7.4f}  "
            f"{s['miss_bottom']:>8.4f}  {s['miss_rest']:>8.4f}  "
            f"{rr_s:>5}  {lo_s:>5}  {hi_s:>5}  {s['fisher_p']:>10.4g}"
        )

    cmh = res["cmh"]
    print(f"\n  CMH pooled ({cmh['n_strata_used']} strata used, {cmh['n_strata_skipped']} skipped):")
    if "note" in cmh:
        print(f"    {cmh['note']}")
    else:
        or_s  = f"{cmh['or_mh']:.2f}" if not np.isnan(cmh['or_mh']) else "undef"
        lo_s  = f"{cmh['or_lo']:.2f}" if not np.isnan(cmh['or_lo']) else "undef"
        hi_s  = f"{cmh['or_hi']:.2f}" if not np.isnan(cmh['or_hi']) else "undef"
        print(
            f"    MH common OR = {or_s} [{lo_s}, {hi_s}]  "
            f"chi2 = {cmh['chi2']:.2f}  p = {cmh['p']:.4g}"
        )


def _interpret_stratified(res_sev: dict, res_type: dict) -> None:
    """Print a plain-language interpretation of the stratified E3 result."""
    print("\n--- Interpretation ---")

    cmh_sev  = res_sev["cmh"]
    cmh_type = res_type["cmh"]

    # Count strata where the direction holds (RR > 1) and Fisher p < 0.05
    sev_strata  = [(v, s) for v, s in res_sev["per_stratum"].items()  if "a" in s]
    type_strata = [(v, s) for v, s in res_type["per_stratum"].items() if "a" in s]

    sev_dir  = sum(1 for _, s in sev_strata  if not np.isnan(s["rr"]) and s["rr"] > 1)
    sev_sig  = sum(1 for _, s in sev_strata  if s["fisher_p"] < 0.05)
    type_dir = sum(1 for _, s in type_strata if not np.isnan(s["rr"]) and s["rr"] > 1)
    type_sig = sum(1 for _, s in type_strata if s["fisher_p"] < 0.05)

    # Severity conclusion
    if "note" not in cmh_sev:
        cmh_p = cmh_sev["p"]
        cmh_or = cmh_sev["or_mh"]
        if cmh_p < 0.05:
            print(
                f"SEVERITY: Effect holds after controlling for severity. "
                f"CMH OR = {cmh_or:.2f}, p = {cmh_p:.4g}. "
                f"Direction consistent in {sev_dir}/{len(sev_strata)} severity strata "
                f"({sev_sig} individually significant at p<0.05). "
                f"Explanation instability predicts misclassification independently "
                f"of how degraded the input is."
            )
        else:
            print(
                f"SEVERITY: Effect does NOT survive severity stratification. "
                f"CMH OR = {cmh_or:.2f}, p = {cmh_p:.4g}. "
                f"The RR = 4.84 in the pooled E3 is likely driven by severity "
                f"confounding: higher severity lowers both IoU and accuracy. "
                f"The claim must be stated as 'severity lowers both stability "
                f"and accuracy', not as 'explanation instability independently "
                f"predicts misclassification'."
            )
    else:
        print(f"SEVERITY: {cmh_sev['note']}")

    # Type conclusion
    if "note" not in cmh_type:
        cmh_p_t  = cmh_type["p"]
        cmh_or_t = cmh_type["or_mh"]
        if cmh_p_t < 0.05:
            print(
                f"TYPE: Effect is not specific to gaussian_noise. "
                f"CMH OR = {cmh_or_t:.2f}, p = {cmh_p_t:.4g}. "
                f"Direction consistent in {type_dir}/{len(type_strata)} corruption types "
                f"({type_sig} individually significant at p<0.05)."
            )
        else:
            print(
                f"TYPE: Effect does not generalise across corruption types. "
                f"CMH OR = {cmh_or_t:.2f}, p = {cmh_p_t:.4g}. "
                f"Direction in {type_dir}/{len(type_strata)} types. "
                f"Treat as type-specific."
            )
    else:
        print(f"TYPE: {cmh_type['note']}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_severity_trend(
    pooled_tbl: pd.DataFrame,
    per_type: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Line chart: severity vs cam_iou_mean (pooled and per type)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel — pooled metrics
    ax = axes[0]
    ax.plot(
        pooled_tbl["severity"],
        pooled_tbl["cam_iou_mean"],
        marker="o", linewidth=2, label="cam_iou_mean",
    )
    ax.plot(
        pooled_tbl["severity"],
        pooled_tbl["cam_corr_mean"],
        marker="s", linewidth=2, linestyle="--", label="cam_corr_mean",
    )
    ax2 = ax.twinx()
    ax2.plot(
        pooled_tbl["severity"],
        pooled_tbl["accuracy"],
        marker="^", color="C2", linewidth=2, label="accuracy",
    )
    ax2.set_ylabel("Accuracy", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Explanation stability")
    ax.set_title("Pooled: severity vs stability / accuracy")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # Right panel — per type (cam_iou_mean only)
    ax = axes[1]
    for ctype, tbl in per_type.items():
        if tbl.empty:
            continue
        ax.plot(tbl["severity"], tbl["cam_iou_mean"], marker="o", label=ctype)
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("cam_iou_mean")
    ax.set_title("Per type: cam_iou_mean vs severity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "corruption_severity_trend.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {out_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def e8_analysis(run_id: int, cfg: dict, out_dir: Path) -> dict:
    """Run E8 analysis on a corrupted imagenette batch run.

    Args:
        run_id: The run_id of the imagenette-c batch (created by run_corrupted_batch).
        cfg: Full config dict.
        out_dir: Directory for figure output.

    Returns:
        Dict containing all computed results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"E8 Corruption robustness  run_id={run_id}")
    print(f"{'=' * 60}")

    df = _load_corrupted_df(run_id, cfg)
    print(f"\nRows loaded: {len(df)}")

    corruption_types = sorted(df["corruption_type"].dropna().unique().tolist())
    severities_found = sorted(df["severity"].dropna().unique().tolist())
    print(f"Corruption types : {corruption_types}")
    print(f"Severities found : {severities_found}")

    # -- Severity tables --
    pooled_tbl = _severity_table(df, label="pooled")
    _print_severity_table(pooled_tbl, label="Pooled across all types")

    per_type: dict[str, pd.DataFrame] = {}
    for ctype in corruption_types:
        sub = df[df["corruption_type"] == ctype]
        tbl = _severity_table(sub, label=ctype)
        per_type[ctype] = tbl
        _print_severity_table(tbl, label=ctype)

    # -- Spearman rho --
    print(f"\n--- Spearman rho: severity vs cam_iou_mean ---")
    sp = _severity_spearman(df, "cam_iou_mean")
    print(
        f"rho = {sp['rho']:.3f}  "
        f"[{sp['ci_lo']:.3f}, {sp['ci_hi']:.3f}]  "
        f"p = {sp['p']:.4g}  n = {sp['n']}"
    )

    sp_corr = _severity_spearman(df, "cam_corr_mean")
    print(
        f"  (cam_corr_mean)  rho = {sp_corr['rho']:.3f}  "
        f"[{sp_corr['ci_lo']:.3f}, {sp_corr['ci_hi']:.3f}]  "
        f"p = {sp_corr['p']:.4g}"
    )

    # -- Kruskal-Wallis --
    print(f"\n--- Kruskal-Wallis: cam_iou_mean across severities ---")
    kw = _kruskal_wallis(df, "cam_iou_mean")
    print(f"H = {kw['stat']:.3f}  p = {kw['p']:.4g}")
    print(
        f"Bonferroni alpha = {_ALPHA_BONF:.4f}  "
        f"({sum(1 for pp in kw['pairs'] if pp['sig'])} / {len(kw['pairs'])} pairs significant)"
    )
    sig_pairs = [pp for pp in kw["pairs"] if pp["sig"]]
    if sig_pairs:
        print("  Significant pairs (after Bonferroni):")
        for pp in sig_pairs:
            print(f"    s{pp['sev_a']} vs s{pp['sev_b']}  U={pp['u']:.0f}  p={pp['p_raw']:.4g}")

    # -- Corrupted E4 --
    print(f"\n--- Corrupted E4: quadrant analysis (corrupted rows) ---")
    e4 = _corrupted_e4(df)
    if e4:
        meds = e4.pop("medians", {})
        print(
            f"Medians  entropy={meds.get('entropy', float('nan')):.4f}  "
            f"instability={meds.get('instability', float('nan')):.4f}"
        )
        for quad, stats in e4.items():
            print(
                f"  {quad:<22}  n={stats['n']:>5}  "
                f"miss_rate={stats['miss_rate']:.3f}"
            )

    # -- Corrupted E3 --
    print(f"\n--- Corrupted E3: stable-prediction subset ---")
    ce3 = _corrupted_e3(df)
    print(
        f"n_total={ce3['n_total']}  "
        f"n_subset(pa==1 & conf>=0.90)={ce3.get('n_subset', 'N/A')}"
    )
    if "note" in ce3:
        print(f"  Note: {ce3['note']}")
    else:
        print(
            f"  Bottom decile threshold (cam_iou_mean) = {ce3['iou_decile_10']:.4f}"
        )
        print(
            f"  miss_rate  bottom={ce3['miss_rate_bottom']:.4f}  "
            f"rest={ce3['miss_rate_rest']:.4f}"
        )
        rr   = ce3["risk_ratio"]
        rrlo = ce3["rr_ci_lo"]
        rrhi = ce3["rr_ci_hi"]
        print(
            f"  RR = {rr:.2f} [{rrlo:.2f}, {rrhi:.2f}]  "
            f"Fisher p = {ce3['fisher_p']:.4g}"
        )

    # -- Corrupted E3 stratified by severity (confound control) --
    print(f"\n{'=' * 60}")
    print("Corrupted E3 -- severity confound control")
    print(f"{'=' * 60}")
    e3_by_sev = _e3_stratified(df, "severity", "severity")
    _print_e3_stratified(e3_by_sev)

    # -- Corrupted E3 stratified by corruption type --
    print(f"\n{'=' * 60}")
    print("Corrupted E3 -- corruption type confound control")
    print(f"{'=' * 60}")
    e3_by_type = _e3_stratified(df, "corruption_type", "corruption_type")
    _print_e3_stratified(e3_by_type)

    # -- Plain-language interpretation --
    _interpret_stratified(e3_by_sev, e3_by_type)

    # -- Figure --
    _plot_severity_trend(pooled_tbl, per_type, out_dir)

    results = {
        "pooled_severity_table": pooled_tbl,
        "per_type_tables":       per_type,
        "spearman_iou":          sp,
        "spearman_corr":         sp_corr,
        "kruskal_wallis":        kw,
        "corrupted_e4":          e4,
        "corrupted_e3":          ce3,
        "e3_by_severity":        e3_by_sev,
        "e3_by_type":            e3_by_type,
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="E8 corruption robustness analysis."
    )
    p.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="run_id of the imagenette-c batch in the database.",
    )
    p.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml).",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/figures",
        help="Directory for figures (default: outputs/figures).",
    )
    return p


def _main() -> None:
    args    = _build_parser().parse_args()
    cfg     = load_config(args.config)
    out_dir = Path(args.out_dir)
    e8_analysis(args.run_id, cfg, out_dir)


if __name__ == "__main__":
    _main()
