# Experiments

Protocol, run recipes, and the results log. Every number reported in Chapter 5 is recorded here first, with the run that produced it.

A result whose configuration and seed are not recorded cannot be reported.

---

## Experiment list

| # | Experiment | Answers | Report | Tier |
|---|---|---|---|---|
| E1 | Model training and reference baselines | Is the measurement apparatus sound? | 5.4.1 | Must |
| E2 | Correlation, prediction vs explanation | Preliminary relationship | 5.4.2 | Must |
| E3 | **Stratified analysis, prediction-stable subset** | **O2 — the central result** | 5.4.3 | Must |
| E4 | Quadrant analysis | O2 | 5.4.4 | Must |
| E5 | Correct vs misclassified | O2 | 5.4.5 | Must |
| E6 | In-distribution vs OOD | O2 | 5.4.5 | Should |
| E7 | **Complementarity, ΔAUROC** | **O4** | 5.4.6 | Must |
| E8 | Corruption severity trend | Input degradation | 5.4.5 | Should |
| E9 | Ablation: dropout placement | Validity | 5.4.7 | Should |
| E10 | Ablation: number of passes | Justifies N = 30 | 5.4.7 | Should |
| E11 | Ablation: dropout rate, binarisation threshold | Robustness | 5.4.7 | Could |

E1, E3 and E7 carry the project. If time runs out, those three plus a working application still evidence all four objectives.

---

## Protocol

Fixed across every experiment unless an ablation varies it deliberately.

| Parameter | Value |
|---|---|
| Model | ResNet-18, ImageNet-pretrained, fine-tuned |
| Dropout | `Dropout2d(p=0.2)` after `layer2` and `layer3` |
| Stochastic passes | N = 30 |
| Grad-CAM target layer | `layer4` |
| Grad-CAM target class | argmax of the **mean** predictive distribution, fixed across passes |
| IoU binarisation | 80th percentile, per map |
| Top-k | 10% of pixels |
| Seed | 42 |
| Input | 224 × 224, ImageNet normalisation |

Record any deviation in the run log and state it in the report.

---

## E1 — Model and baselines

```bash
python src/train_model.py --config configs/config.yaml
python src/baselines.py --run-id <id>
```

Acceptance: validation accuracy ≥ 95%; **B1 = 1.000**.

B1 below 1.000 means the Grad-CAM path or seed handling is broken. Stop and fix before collecting anything else.

| Quantity | Value | Date |
|---|---|---|
| Validation accuracy | — | — |
| B1 correlation / IoU | — | — |
| B2 correlation / IoU | — | — |
| B3 correlation / IoU | — | — |

B3 becomes the default `TH_IOU` for the risk flag.

---

## E2 — Correlation

```bash
python src/analysis.py --run-id <id> --experiment correlation
```

Spearman ρ with bootstrap 95% CI for: confidence vs `cam_iou_mean`, entropy vs `cam_corr_mean`, `pred_variance` vs `topk_overlap`.

**Preliminary result, not a conclusion.** Across the full set the relationship is dominated by the trivial mechanism that when the prediction moves, the explanation moves. E3 removes that.

| Pair | ρ | 95% CI | n |
|---|---|---|---|
| confidence vs cam_iou_mean | — | — | — |
| entropy vs cam_corr_mean | — | — | — |
| pred_variance vs topk_overlap | — | — | — |

---

## E3 — Stratified analysis (central result)

```bash
python src/stratified_analysis.py --run-id <id>
```

1. Grad-CAM target class already fixed at generation.
2. Select `pred_agreement == 1.0` — every pass agreed on the class.
3. Bin by confidence: [0.90, 0.95), [0.95, 0.99), [0.99, 1.00].
4. Report the distribution of `cam_corr_mean` and `cam_iou_mean` per bin, against B2 and B3.

Every sample in the top bin is effectively identical on the prediction side. If explanation stability still spans a wide range within that bin, prediction-side metrics do not determine explanation stability — which is the claim.

| Bin | n | corr min | Q1 | median | Q3 | max |
|---|---|---|---|---|---|---|
| 0.90–0.95 | — | — | — | — | — | — |
| 0.95–0.99 | — | — | — | — | — | — |
| **0.99–1.00** | — | — | — | — | — | — |

Target sentence for the report:

> Restricted to samples with confidence ≥ 0.99 whose predicted class was identical across all 30 passes (n = ___), mean pairwise Grad-CAM correlation ranged from ___ to ___, against a random-map floor of ___ and a same-class reference of ___.

---

## E4 — Quadrant analysis

Axes: entropy and (1 − `cam_iou_mean`), split at the run median.

| Quadrant | `risk_group` | n | misclass. rate | OOD share |
|---|---|---|---|---|
| Q1 low/low | stable | — | — | — |
| Q2 high/high | unstable_both | — | — | — |
| Q3 high/low | pred_unstable_only | — | — | — |
| **Q4 low/high** | **hidden_risk** | — | — | — |

Test whether Q4 exceeds Q1 on misclassification rate: chi-square, plus the risk ratio.

---

## E5, E6 — Group comparisons

Report medians, Mann–Whitney U, and **Cliff's delta**. At this sample size a p-value alone says nothing.

| Comparison | Metric | Median A | Median B | U | p | Cliff's δ | Size |
|---|---|---|---|---|---|---|---|
| correct vs misclassified | cam_corr_mean | — | — | — | — | — | — |
| correct vs misclassified | cam_iou_mean | — | — | — | — | — | — |
| ID vs near-OOD | cam_iou_mean | — | — | — | — | — | — |
| ID vs far-OOD | cam_iou_mean | — | — | — | — | — | — |

δ: < 0.147 negligible, < 0.33 small, < 0.474 medium, otherwise large.

---

## E7 — Complementarity (ΔAUROC)

```bash
python src/complementarity.py --run-id <id> --task misclassification
python src/complementarity.py --run-id <id> --task ood
```

| Model | Features |
|---|---|
| M1 | confidence, entropy, pred_variance |
| M2 | M1 + cam_corr_mean, cam_iou_mean, topk_overlap |

Logistic regression, standardised features, stratified 5-fold cross-validation.

| Task | AUROC M1 | AUROC M2 | ΔAUROC | 95% CI |
|---|---|---|---|---|
| Misclassification | — | — | — | — |
| OOD | — | — | — | — |

**A ΔAUROC of approximately zero is a valid finding.** It would mean explanation-side metrics largely duplicate prediction-side information — a clear, reportable answer to O4. Write that interpretation before seeing the number, so the result is not rationalised after the fact.

---

## E8 — Corruption severity

```bash
python run_pipeline.py --dataset imagenette-c \
  --corruptions gaussian_noise,defocus_blur,brightness,contrast \
  --severities 1,2,3,4,5 --per-cell 500
```

| Severity | n | confidence | entropy | cam_corr_mean | cam_iou_mean |
|---|---|---|---|---|---|
| clean | — | — | — | — | — |
| 1 … 5 | — | — | — | — | — |

Test monotonicity: Spearman ρ between severity and `cam_iou_mean`, plus Kruskal–Wallis with the Bonferroni correction stated.

---

## E9 — Ablation: dropout placement

The one that matters most for validity.

| Condition | Placement | Expected |
|---|---|---|
| A | after `layer4` / before FC | `cam_corr_mean` ≈ 1.0 for every image — measurement impossible |
| B | after `layer2` and `layer3` | wide distribution |

| Condition | mean corr | std | min | max |
|---|---|---|---|---|
| A (FC) | — | — | — | — |
| B (conv) | — | — | — | — |

Condition A is not a failed run — it is the evidence for a genuine contribution: **dropout placement relative to the Grad-CAM target layer determines whether explanation stability is measurable at all.** Report it in Chapter 6 as a contribution.

---

## E10, E11 — Ablations

**Number of passes** — N ∈ {5, 10, 20, 30, 50} on a fixed 500-image subset. Plot mean and standard deviation of each metric against N; identify where it converges; justify N = 30.

**Dropout rate** — p ∈ {0.1, 0.2, 0.3}. Report validation accuracy and the spread of `cam_corr_mean`. Too small and every map is stable; too large and accuracy collapses.

**Binarisation threshold** — percentile ∈ {70, 80, 90}. Confirm the conclusions do not depend on the choice.

---

## Run log

Newest first. One row per pipeline execution.

| run_id | Date | Dataset | n | N | Seed | Purpose | Notes |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

---

## Figures

| File | Experiment | Report section |
|---|---|---|
| `baseline_reference_panel.png` | E1 | 5.4.1 |
| `correlation_heatmap.png` | E2 | 5.4.2 |
| `confidence_vs_cam_iou.png` | E2 | 5.4.2 |
| `stratified_by_confidence_bin.png` | E3 | **5.4.3** |
| `pred_stable_subset_distribution.png` | E3 | **5.4.3** |
| `quadrant_scatter.png` | E4 | 5.4.4 |
| `correct_vs_misclassified_boxplot.png` | E5 | 5.4.5 |
| `ood_group_comparison.png` | E6 | 5.4.5 |
| `corruption_severity_trend.png` | E8 | 5.4.5 |
| `delta_auroc_comparison.png` | E7 | **5.4.6** |
| `ablation_dropout_placement.png` | E9 | 5.4.7 |
| `ablation_N_convergence.png` | E10 | 5.4.7 |

All at 300 dpi into `outputs/figures/`. Every figure needs a caption and a List of Figures entry.

---

## Rules

**Write the interpretation before running.** For each experiment, decide in advance what a positive result, a null result, and a negative result would each mean. This prevents post-hoc rationalisation, and it means a null result is still reportable rather than a crisis.

**Never report a number that is not in this file** with its run_id and date.

**Notebooks are for exploration only.** Anything reported comes from a script in `src/`, executed against a recorded configuration.

**Re-run after any pipeline change.** Mixing results from before and after a code change produces a table that cannot be defended.