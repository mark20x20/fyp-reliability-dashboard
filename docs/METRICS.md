# Metrics

Implementer reference. Every formula, its edge cases, its expected range, and its unit test.

Notation: $N$ stochastic passes; class set $C$, $|C| = 10$; softmax output of pass $t$ is $p_t \in \mathbb{R}^{|C|}$; mean predictive distribution $\bar{p} = \frac{1}{N}\sum_t p_t$; predicted class $\hat{c} = \arg\max_c \bar{p}_c$.

All prediction-side functions take an array of shape `(N, num_classes)`. All explanation-side functions take `(N, H, W)`. Both are pure — no I/O, no model, no database.

---

## Prediction side

### Confidence (maximum softmax probability)

$$\text{conf} = \max_c \bar{p}_c$$

```python
def confidence(probs: np.ndarray) -> float:
    return float(probs.mean(axis=0).max())
```

Range [1/|C|, 1] = [0.1, 1.0]. Column `confidence`.

> Confidence and maximum softmax probability are the same quantity. Do not create both a `confidence` and an `msp` column.

### Predictive entropy

$$H[\bar{p}] = -\sum_c \bar{p}_c \log \bar{p}_c$$

```python
def predictive_entropy(probs, eps=1e-12) -> float:
    p = probs.mean(axis=0)
    return float(-(p * np.log(p + eps)).sum())
```

Range [0, ln 10] = [0, 2.3026]. Natural log throughout — do not mix in log2. The `eps` guard prevents `0 · log 0` producing NaN. Column `entropy`.

### Expected entropy

$$\mathbb{E}_t H[p_t] = \frac{1}{N}\sum_t H[p_t]$$

Intermediate value only, used by mutual information. Not stored.

### Mutual information (BALD)

$$I = H[\bar{p}] - \frac{1}{N}\sum_t H[p_t]$$

```python
def mutual_information(probs, eps=1e-12) -> float:
    mean_p = probs.mean(axis=0)
    total = -(mean_p * np.log(mean_p + eps)).sum()
    expected = -(probs * np.log(probs + eps)).sum(axis=1).mean()
    return float(total - expected)
```

Range [0, ln 10], and always ≤ predictive entropy. A small negative value from floating-point error should be clamped to 0. Separates epistemic uncertainty (disagreement between passes) from total uncertainty. Column `mutual_information`.

### Predictive variance

$$\sigma^2 = \frac{1}{N}\sum_t \left(p_{t,\hat{c}} - \bar{p}_{\hat{c}}\right)^2$$

```python
def predictive_variance(probs) -> float:
    c_hat = int(probs.mean(axis=0).argmax())
    return float(probs[:, c_hat].var())
```

Variance is taken over the **predicted class channel only**, not over all classes. Range [0, 0.25]. Column `pred_variance`.

### Prediction agreement

$$a = \frac{1}{N}\max_c \sum_t \mathbb{1}\!\left[\arg\max p_t = c\right]$$

```python
def prediction_agreement(probs) -> float:
    per_pass = probs.argmax(axis=1)
    return float(np.bincount(per_pass, minlength=probs.shape[1]).max() / len(per_pass))
```

Range [1/N, 1.0]. A value of 1.0 means every pass agreed on the same class.

**This is the control variable for the whole analysis.** The central claim is tested on the subset where `pred_agreement == 1.0`, because within that subset any variation in explanation stability cannot be attributed to the prediction moving. Column `pred_agreement`.

### Correctness

```python
correct = int(pred_label == true_label)
```

`NULL` for OOD and uploaded images, where no ground-truth label exists in the label space. Column `correct`.

---

## Grad-CAM

For target class $c$ with target-layer activations $A^k$:

$$\alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}, \qquad L^c = \mathrm{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$

ResNet-18 at 224×224 gives a 7×7 map from `layer4`.

### Target class — fixed

```python
mean_probs = probs.mean(axis=0)
target_class = int(mean_probs.argmax())
# all N maps use this same target_class
```

Never the per-pass argmax. If the predicted class flips between passes, the map necessarily changes, and the measured variation would be prediction instability rather than explanation instability. Removing that confound is the entire reason this analysis is designed the way it is.

### Normalisation and upsampling

```python
def normalise(cam, eps=1e-8):
    lo, hi = cam.min(), cam.max()
    return np.zeros_like(cam) if hi - lo < eps else (cam - lo) / (hi - lo)
```

Min-max to [0, 1] per map, then bilinear upsample to 224×224 for metric computation. A constant map — which happens when ReLU zeroes everything — returns all zeros rather than dividing by zero.

Raw 7×7 maps are stored as float16 `.npz` (~3 KB per image). Metrics are computed at 224×224; the 7×7 form is what gets archived.

---

## Explanation side

With $N$ maps there are $P = \binom{N}{2}$ unordered pairs — 435 at $N = 30$. Every metric averages over all pairs.

### Pairwise correlation

$$\bar{r} = \frac{1}{P}\sum_{a<b} \rho_{\text{Pearson}}(L_a, L_b)$$

```python
def cam_correlation(cams):
    flat = cams.reshape(len(cams), -1)
    vals = [np.corrcoef(flat[a], flat[b])[0, 1]
            for a, b in itertools.combinations(range(len(cams)), 2)]
    vals = np.nan_to_num(vals, nan=0.0)
    return float(np.mean(vals)), float(np.std(vals))
```

Range [-1, 1], in practice [0, 1]. A constant map gives zero variance and `np.corrcoef` returns NaN — substitute 0.0. Columns `cam_corr_mean`, `cam_corr_std`.

### Pairwise IoU

Binarise each map at **its own 80th percentile**, then

$$\overline{\text{IoU}} = \frac{1}{P}\sum_{a<b} \frac{|M_a \cap M_b|}{|M_a \cup M_b|}$$

```python
def cam_iou(cams, percentile=80):
    masks = [c >= np.percentile(c, percentile) for c in cams]
    vals = []
    for a, b in itertools.combinations(range(len(masks)), 2):
        inter = np.logical_and(masks[a], masks[b]).sum()
        union = np.logical_or(masks[a], masks[b]).sum()
        vals.append(inter / union if union else 0.0)
    return float(np.mean(vals))
```

Range [0, 1]. Percentile-based, never a fixed absolute threshold — Grad-CAM magnitudes differ from image to image, so a fixed cut is not comparable across samples. Empty union returns 0.0. Column `cam_iou_mean`.

### Top-k overlap

$$\overline{O} = \frac{1}{P}\sum_{a<b}\frac{|T_a \cap T_b|}{k}$$

where $T_t$ is the index set of the top $k$ pixels, $k = 0.10 \cdot HW$.

```python
def topk_overlap(cams, k_frac=0.10):
    k = int(k_frac * cams[0].size)
    tops = [set(np.argpartition(c.ravel(), -k)[-k:]) for c in cams]
    return float(np.mean([len(tops[a] & tops[b]) / k
                          for a, b in itertools.combinations(range(len(tops)), 2)]))
```

Range [0, 1]. More permissive than IoU, since it ignores how the regions are shaped. Column `topk_overlap`.

### Variability map

$$V_{ij} = \mathrm{std}_t\left(L_{t,ij}\right)$$

Saved as a heatmap PNG, not stored as a scalar. Shows *where* the explanation is unstable, which is what makes an unstable case legible to a reviewer.

---

## Reference baselines

Without these, "IoU 0.42" is uninterpretable.

| ID | Baseline | Method | Expected |
|---|---|---|---|
| **B1** | Upper bound | Dropout disabled, same image, two maps | correlation 1.000, IoU 1.000 |
| **B2** | Lower bound | 1,000 pairs of Gaussian-smoothed random maps | correlation ≈ 0, IoU ≈ 0.20 |
| **B3** | Cross-image | 1,000 pairs from different images of the same class | between B2 and B1 |

```python
def random_baseline(n_pairs=1000, size=224, sigma=8):
    vals = []
    for _ in range(n_pairs):
        a = gaussian_filter(np.random.rand(size, size), sigma)
        b = gaussian_filter(np.random.rand(size, size), sigma)
        vals.append(iou_pair(normalise(a), normalise(b)))
    return float(np.mean(vals))
```

Random maps are smoothed because raw white noise has no spatial structure at all and would give an unrealistically low floor. The smoothing scale should roughly match the spatial scale of a real Grad-CAM.

**B1 is also the implementation check.** If it is not 1.000, there is a bug in the Grad-CAM path or in seed handling — stop and fix it before collecting any results.

B3 supplies the default `TH_IOU` for the risk flag, so the threshold is grounded in a measured value rather than chosen arbitrarily.

Stored in the `baselines` table, one row per run.

---

## Derived quantities

### Risk group

Axes: prediction uncertainty (`entropy`) and explanation instability ($1 -$ `cam_iou_mean`), split at the median of the run.

| Quadrant | Condition | `risk_group` |
|---|---|---|
| Q1 | low / low | `stable` |
| Q2 | high / high | `unstable_both` |
| Q3 | high / low | `pred_unstable_only` |
| Q4 | **low / high** | `hidden_risk` |

Q4 is the case the project exists to expose: the prediction looks trustworthy and the reasoning is not.

### Risk flag

```python
if confidence > TH_CONF and cam_iou_mean < TH_IOU:
    flag = "REVIEW PRIORITY"
```

Defaults `TH_CONF = 0.95`, `TH_IOU` = B3. A threshold rule, not a proposed metric — no construct validity argument is required for it, and none should be claimed.

---

## Statistical reporting

At n ≈ 17,000 a p-value below 0.001 is uninformative on its own.

| Situation | Report |
|---|---|
| Correlation | Spearman ρ with bootstrap 95% CI; p in parentheses only |
| Two groups | Medians, Mann–Whitney U, **Cliff's delta** |
| Multi-group | Kruskal–Wallis with the Bonferroni correction stated |
| Detection | AUROC, AUPR, FPR@95TPR with bootstrap CIs |

Cliff's delta: |δ| < 0.147 negligible, < 0.33 small, < 0.474 medium, otherwise large.

Do not write: "a significant correlation was found (p < 0.001)".
Write: "ρ = 0.31, 95% CI [0.28, 0.34] — a moderate negative association".

---

## Expected ranges

Use as sanity checks. A value outside these bounds is a bug, not a finding.

| Metric | Range | Typical (correct in-distribution) |
|---|---|---|
| `confidence` | [0.1, 1.0] | 0.90–1.00 |
| `entropy` | [0, 2.30] | 0.0–0.4 |
| `mutual_information` | [0, entropy] | 0.0–0.2 |
| `pred_variance` | [0, 0.25] | 0.000–0.010 |
| `pred_agreement` | [1/N, 1.0] | 1.0 |
| `cam_corr_mean` | [-1, 1] | 0.6–0.95 |
| `cam_iou_mean` | [0, 1] | 0.5–0.9 |
| `topk_overlap` | [0, 1] | 0.6–0.95 |

If `cam_corr_mean` sits above 0.99 for **every** image, the dropout layers are in the wrong place — see [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

---

## Unit tests

Each is a required test case in [`TESTING.md`](TESTING.md).

| TC | Input | Expected |
|---|---|---|
| TC1 | one-hot distribution | entropy = 0.0 |
| TC2 | uniform over 10 classes | entropy = ln 10 = 2.3026 |
| TC3 | identical passes | pred_variance = 0.0 |
| TC4 | identical passes | mutual_information = 0.0 |
| TC5 | all passes predict class 3 | pred_agreement = 1.0 |
| TC6 | 15/30 class 3, 15/30 class 7 | pred_agreement = 0.5 |
| TC7 | two identical maps | correlation = 1.0 |
| TC8 | two identical maps | IoU = 1.0 |
| TC9 | map and its negation | IoU = 0.0 |
| TC10 | constant map | no division-by-zero |
| TC11 | N = 30 maps | 435 pairwise values |
