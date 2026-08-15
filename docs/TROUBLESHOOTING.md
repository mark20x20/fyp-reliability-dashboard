# Troubleshooting

Known failure modes, in the order they are likely to be hit.

---

## 1. Grad-CAM maps are identical on every pass

**Symptom** — TC16 fails. `cam_corr_mean` is 0.999 or 1.000 for every image. The stability metric has no variance and nothing correlates with anything.

**Cause** — the dropout layers sit after the Grad-CAM target layer. Grad-CAM reads activations at `layer4`; anything downstream of that has no effect on the map.

```python
# WRONG — the standard tutorial placement
model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 10))

# WRONG — still after layer4
model.layer4 = nn.Sequential(model.layer4, nn.Dropout2d(0.2))

# CORRECT
model.layer2 = nn.Sequential(model.layer2, nn.Dropout2d(0.2))
model.layer3 = nn.Sequential(model.layer3, nn.Dropout2d(0.2))
```

**Diagnose**

```python
for name, m in model.named_modules():
    if isinstance(m, (nn.Dropout, nn.Dropout2d)):
        print(name, m.training)
```

Every dropout module must appear at a path under `layer2` or `layer3`, and must report `training=True` after `enable_mc_dropout()`.

This is the single most expensive mistake available in this project. Everything downstream is invalid.

---

## 2. Maps vary but predictions are nonsense

**Symptom** — accuracy collapses to around 10% during MC inference, though the checkpoint evaluates correctly.

**Cause** — `model.train()` was called. That switches BatchNorm to training mode, so batch statistics replace the running statistics.

**Fix** — switch only the dropout modules.

```python
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout2d):
        m.train()
```

**Verify** — every `BatchNorm2d` should report `training=False`.

---

## 3. Maps barely vary even with correct placement

**Symptom** — TC16 passes, but `cam_corr_mean` is 0.97–0.99 across the whole dataset, so there is no spread to analyse.

**Causes and fixes**

| Cause | Fix |
|---|---|
| `Dropout` instead of `Dropout2d` | Convolutional activations are spatially correlated; element-wise dropout barely perturbs the attribution. Switch to channel-wise |
| `p` too small | Try 0.2, then 0.3. Check validation accuracy does not collapse |
| Dropout too early in the network | `layer1` is too far from the target layer for the perturbation to survive. Keep `layer2` and `layer3` |

Run ablation E9 to record the effect of `p`; the result belongs in report section 5.4.7.

---

## 4. B1 is not exactly 1.000

**Symptom** — the deterministic upper-bound baseline returns 0.998 rather than 1.000.

**Causes**

- A dropout module is still in training mode. Assert all are `training=False` before computing B1.
- cuDNN non-determinism.

```python
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

- Float32 accumulation noise. Anything above 0.9999 is acceptable; below that, investigate.

Do not proceed to data collection until B1 is effectively 1.000. It is the check that the measurement apparatus works.

---

## 5. IoU takes only a few discrete values

**Symptom** — `cam_iou_mean` clusters on 0, 0.33, 0.5, 1.0. The scatter plot looks like horizontal lines.

**Cause** — metrics are being computed on the raw 7×7 map. Forty-nine cells with a 20% cut gives ten pixels, so the intersection can only take a handful of values.

**Fix** — upsample to 224×224 before computing metrics. Store the raw 7×7; measure on the upsampled version.

If this appears with CIFAR-10 at 32×32, the map is 4×4 and no amount of upsampling recovers the lost detail. That is the reason for the move to Imagenette.

---

## 6. `_ARRAY_API not found` on import

**Symptom**

```
A module compiled using NumPy 1.x cannot be run in NumPy 2.x
AttributeError: _ARRAY_API not found
```

**Cause** — something pulled in NumPy 2.x. OpenCV 4.10 and pytorch-grad-cam 1.5 are built against the 1.x ABI.

**Fix**

```bash
pip install "numpy==1.26.4" --force-reinstall
python -c "import numpy, cv2; print(numpy.__version__)"
```

Check `pip list | grep numpy` after installing anything new. Some packages upgrade it silently.

---

## 7. CUDA out of memory

**Symptom** — OOM during batch-replicated inference.

**Cause** — Grad-CAM retains the computation graph for the backward pass. At N = 30 and 224×224, activations plus gradients approach the 12 GB limit if anything else is resident.

**Fixes, in order**

1. Free the graph between images: `del cams, logits; torch.cuda.empty_cache()`
2. Split the replication into chunks of 10 and concatenate
3. Reduce `data.batch_size` — it does not affect N
4. Close other GPU consumers (browsers with hardware acceleration, other notebooks)

Do not reduce N as a first response; N is an experimental parameter, not a memory knob.

---

## 8. `imagecorruptions` raises on tensor input

**Symptom** — `corrupt()` fails with a dtype or shape error.

**Cause** — it expects a uint8 HWC NumPy array, not a normalised CHW tensor.

```python
img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
corrupted = corrupt(img_np, corruption_name="gaussian_noise", severity=3)
tensor_out = torch.from_numpy(corrupted).permute(2, 0, 1).float() / 255
```

Apply corruption **before** ImageNet normalisation. Corrupting normalised values produces artefacts that do not correspond to any real degradation.

---

## 9. Batch run is far slower than expected

**Symptom** — 4,000 images takes over an hour instead of about ten minutes.

**Causes**

| Cause | Fix |
|---|---|
| Looping N forward passes | Replicate the image N times into one batch. Dropout masks are sampled per-sample, so one call yields N distinct passes |
| One database commit per image | Batch roughly 100 rows per transaction |
| Writing all N maps as PNG | Write six representative maps plus the mean and variability maps. All N produces 500,000+ files |
| CPU-only PyTorch | Check `torch.cuda.is_available()`. Reinstall from the CUDA index if `False` |

---

## 10. Streamlit shows stale data after a write

**Symptom** — a recorded review does not appear until the app is restarted.

**Cause** — `@st.cache_data` is serving a cached query result.

```python
repo.insert_review(...)
fetch_metrics.clear()      # clear the specific cached function
st.rerun()
```

Use `@st.cache_resource` for the model — never `cache_data`, which tries to hash it.

---

## 11. `SQLite objects created in a thread...`

**Symptom** — `ProgrammingError` when the app queries the database.

**Cause** — Streamlit serves pages from a worker thread.

```python
sqlite3.connect(db_path, check_same_thread=False)
```

Keep writes on one connection. SQLite tolerates concurrent readers, not concurrent writers.

---

## 12. Foreign key violations are not raised

**Symptom** — a row with an invalid `run_id` inserts successfully.

**Cause** — SQLite disables foreign key enforcement by default.

```python
conn.execute("PRAGMA foreign_keys = ON")   # per connection, every time
```

Test case TC26 depends on this.

---

## 13. `pred_agreement` is 1.0 for everything

**Symptom** — the stratified analysis subset is the entire dataset, so nothing is being controlled for.

**Diagnose** — that may be correct. A well-trained model on clean in-distribution data with `p = 0.2` genuinely produces stable class predictions on most images. Check the OOD and corrupted subsets; if agreement is 1.0 there too, dropout is not perturbing the network enough (see item 3).

The stratified analysis is still valid when the subset is large. The point is that prediction-side metrics are constant within it, not that the subset is small.

---

## 14. `np.corrcoef` returns NaN

**Symptom** — `cam_corr_mean` is NaN for some images.

**Cause** — a constant map. ReLU in Grad-CAM can zero the entire output when no positive contribution exists for the target class, leaving zero variance.

**Production fix** — `mean_pairwise_correlation` in `src/metrics_explanation.py` applies `nan_to_num(nan=0.0)` before averaging.  Zero is the correct substitute: a flat map carries no spatial information, and treating its correlation with anything as zero is right.  Count how often it occurs; if frequent, check the target class or layer choice.

**Test behaviour differs** — TC16 must assert that zero constant maps are present *before* computing correlation.  If degenerate maps were silently absorbed by `nan_to_num`, a run where every CAM is blank would score `mean_r = 0.0` and trivially satisfy `r < 0.99`, hiding the defect rather than catching it.  The guard is:

```python
stds = cams.reshape(len(cams), -1).std(axis=1)
n_constant = int((stds < 1e-8).sum())
assert n_constant == 0, f"{n_constant}/{len(cams)} constant maps — degenerate input"
```

Only after this assertion passes does calling `mean_pairwise_correlation` inside the test make sense.

---

## 15. Figures are unreadable in the report

**Symptom** — text is illegible once the figure is placed in Word.

**Fixes**

```python
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.rcParams.update({"font.size": 11})
```

Size for a single A4 column, roughly 6×4 inches. Test by printing one page before generating all of them.

---

## When none of this applies

1. Re-run `pytest tests/test_gradcam.py -v` — most problems trace back to the gate.
2. Check the expected ranges table in `docs/METRICS.md`. A value outside its bound is a bug, not a finding.
3. Compare against the recorded configuration in the `runs` row for the last known-good run.
4. Log the issue in `PROGRESS.md` section 7 with the symptom, the cause, and the fix. Several of these entries belong in report section 5.3 and Chapter 6 Limitations.