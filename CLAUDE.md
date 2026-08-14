# CLAUDE.md

Project instructions for AI-assisted development. Read this before writing any code in this repository.

**Project**: FYP — reliability analysis for image classification using prediction uncertainty and explanation stability
**Student**: TP073279, BSc (Hons) Computer Science with Artificial Intelligence, APU
**Full specification**: `docs/FYP_Master_Working_Document.md`

---

## 1. HARD CONSTRAINTS — violating any of these invalidates the research

These are not style preferences. Each one, if broken, silently destroys the validity of every downstream result. If a change would touch any of these, stop and ask before proceeding.

### 1.1 Dropout placement

`Dropout2d(p)` goes **after `layer2` and after `layer3`** — before the Grad-CAM target layer (`layer4`).

```python
model.layer2 = nn.Sequential(model.layer2, nn.Dropout2d(p=p))
model.layer3 = nn.Sequential(model.layer3, nn.Dropout2d(p=p))
```

Never after `layer4`, after global average pooling, or before the fully-connected layer. Grad-CAM reads activations and gradients at `layer4`; dropout placed after it leaves those activations unchanged and every explanation map comes out identical, making the entire measurement impossible.

Placing dropout immediately before the FC layer is the conventional default in most tutorials. It is wrong here.

Use `Dropout2d`, not `Dropout`. Convolutional activations are spatially correlated, so element-wise dropout barely perturbs the attribution; channel-wise dropout disables whole feature detectors, which is what produces meaningful variation.

### 1.2 Inference mode switching

Only the dropout modules switch to training mode.

```python
def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()
    return model
```

Calling `model.train()` also puts BatchNorm into training mode, which replaces the running statistics with batch statistics and destroys the predictions.

### 1.3 Grad-CAM target class is fixed

```python
mean_probs = probs.mean(dim=0)
target_class = int(mean_probs.argmax())
# every one of the N CAMs uses this same target_class
```

Never the per-pass argmax. If the predicted class flips between passes, the map necessarily changes, and the measured variation would be prediction instability rather than explanation instability. Removing that confound is the entire reason this analysis is designed the way it is.

### 1.4 Storage is SQLite, not CSV

All metrics go to `db/reliability.db` (schema in `db/schema.sql`). CSV is an **export feature**, never the storage mechanism. Do not add code paths that write metrics only to CSV. Flat-file storage is not acceptable for a CSAI application deliverable.

All SQL lives in `src/repository.py`. No raw queries scattered across other modules.

### 1.5 Seeds and configuration are recorded

Every run writes `n_runs`, `seed`, `dropout_p`, `dropout_layers`, `iou_threshold`, `topk_k` into the `runs` table. Set the seed from config at the start of every entry point. A result that cannot be reproduced cannot be reported.

### 1.6 Two user roles

The application defines an engineer role and a reviewer role, each with its own pages and features. This is an assessment requirement for CSAI, not a design flourish. Do not collapse them into a single interface.

### 1.7 IoU binarisation is percentile-based

Threshold each map at its own 80th percentile. Never a fixed absolute value — Grad-CAM magnitudes differ from image to image, so a fixed cut is not comparable across samples.

### 1.8 No new composite metrics

Report established metrics individually. Do not invent, aggregate, or helpfully combine them into a single reliability score. The one exception is the risk flag, which is an explicit threshold rule and is described as such:

```python
if confidence > TH_CONF and cam_iou_mean < TH_IOU:
    risk_flag = "REVIEW PRIORITY"
```

`TH_IOU` defaults to the measured cross-image baseline (B3), not an arbitrary number.

---

## 2. Academic integrity — the most important rule in this file

**The report prose must be written by the student.** AI-generated content is capped at 20% by the university and students have failed on this.

| Claude does | Claude does not |
|---|---|
| Write code | Write Chapters 1–6 prose |
| Explain a concept so the student can write it | Draft paragraphs for the report |
| Review the student's own draft and point out gaps | Produce report text to be pasted in |
| Generate tables, figures, schemas, test cases | Generate the discussion around them |

When asked for report content, explain the substance and let the student write it. Structure, tables, and specifications are fine; flowing academic prose for submission is not.

---

## 3. Stack

| Component | Version | Note |
|---|---|---|
| Python | 3.11 | Not 3.12 — pinned versions lack wheels |
| PyTorch | 2.3.1 + cu121 | Install first, from the CUDA index |
| torchvision | 0.18.1 | |
| pytorch-grad-cam | 1.5.0 | |
| NumPy | **1.26.4** | Pinned below 2.0 for OpenCV and pytorch-grad-cam ABI compatibility |
| pandas | 2.2.2 | |
| scikit-learn | 1.5.0 | |
| opencv-python | 4.10.0.84 | |
| imagecorruptions | 1.1.2 | |
| Streamlit | 1.36.0 | |
| pytest | 8.2.2 | |

If anything upgrades NumPy to 2.x, pin it back. The symptom is `_ARRAY_API not found` at import.

Hardware: Windows 11, NVIDIA RTX 4070 SUPER (12 GB).

---

## 4. Layout

```
app/            Streamlit pages, one file per page
src/            pipeline, metrics, analysis, repository
models/         resnet_dropout.py + checkpoints/
db/             schema.sql, reliability.db
configs/        config.yaml — all parameters live here
tests/          pytest
scripts/        download_data.py, init_db.py
outputs/        gradcam/{raw,png}/, figures/, runs/
docs/           documentation
notebooks/      exploration only, never the source of reported results
```

Dependencies point downward: `app/` → `src/` → database. `src/` never imports from `app/`.

---

## 5. Conventions

- Parameters come from `configs/config.yaml`, read once at the entry point and passed down. No magic numbers in function bodies, no module reading config on its own.
- Metric functions are pure: arrays in, floats out. No I/O, no model, no database. This is what makes them testable against analytical values.
- Type hints on public functions; docstrings stating units and expected ranges.
- The pipeline logs and skips a failed image rather than aborting a 4,000-image run. Count failures and surface them in the run summary.
- Batch database writes, roughly 100 rows per transaction. One commit per image turns a ten-minute run into an hour.
- Replicate the image N times into one batch rather than looping N forward passes. Dropout masks are sampled per-sample, so one call yields N distinct passes.

---

## 6. Metric definitions

Full reference with edge cases in `docs/METRICS.md`. Summary:

**Prediction side** — `confidence` (max softmax of the mean distribution), `entropy` (natural log), `pred_variance` (over the predicted-class channel only), `pred_agreement` (fraction of passes agreeing on the modal class), `mutual_information` (BALD).

`confidence` and maximum softmax probability are the same quantity. Do not create both columns.

`pred_agreement` is the control variable for the central analysis — the stratified result is computed on the subset where it equals 1.0.

**Explanation side** — `cam_corr_mean` and `cam_corr_std` (mean and standard deviation of all pairwise Pearson correlations), `cam_iou_mean` (percentile-binarised), `topk_overlap` (top 10% of pixels), plus a pixel-wise variability map saved as PNG.

At N = 30 there are 435 pairs. Average over all of them.

**Baselines** — B1 upper bound (dropout off, expect 1.000, doubles as an implementation check), B2 lower bound (smoothed random maps), B3 cross-image reference (different images of the same class). Without these, a value like "IoU 0.42" cannot be interpreted.

---

## 7. Verification gates

```bash
pytest tests/test_gradcam.py -v
```

| Test | Setup | Requirement | Failure means |
|---|---|---|---|
| **TC16** | Dropout on, one image, 10 maps | mean pairwise correlation **< 0.99** | Dropout is in the wrong place; nothing downstream is valid |
| **TC17** | Dropout off, one image, 2 maps | correlation **= 1.000** | Bug in the Grad-CAM path or seed handling |

**These must pass before any other development proceeds.** Re-run after any change to the model, inference, or Grad-CAM code.

Sanity check: if `cam_corr_mean` sits above 0.99 for *every* image in a run, the dropout layers are misplaced regardless of what the tests said.

---

## 8. Performance

| Operation | Target |
|---|---|
| Per image, N = 30 | < 100 ms |
| Full batch, 4,000 images | < 10 min |
| Table render, 10,000 rows | < 3 s |

Raw Grad-CAM maps are stored as 7×7 float16 `.npz`, about 3 KB each — all 17,000 fit in roughly 50 MB. Only six representative maps plus the mean and variability maps are written as PNG. Writing all N per image would create over 500,000 files.

---

## 9. Streamlit notes

- `@st.cache_data` for read-only queries; clear after a write.
- `@st.cache_resource` for the model — never `cache_data`.
- `sqlite3.connect(..., check_same_thread=False)`; Streamlit serves from a worker thread.
- Role is held in `st.session_state` and controls which pages appear.
- Progress is exposed from `batch_evaluation` through a callback, so that module never imports Streamlit.
- No HTML `<form>` elements; use ordinary widgets and buttons.

---

## 10. Working style

- Read the relevant `docs/` file before implementing a module.
- Small commits with descriptive messages.
- Update `PROGRESS.md` at the end of each session.
- When a constraint in section 1 conflicts with a requested change, say so rather than silently complying.
- Prefer explicit over clever. This code is read by a second marker.

---

## 11. Out of scope

Do not add, and do not suggest adding:

- New composite reliability metrics (ERS, DURS) — future work only
- Shortcut learning detection — stability metrics cannot detect it by construction, since shortcuts produce *stable* explanations of the wrong region
- Modalities other than images
- Cloud deployment or a server-based database
- Additional architectures or explanation methods beyond the Tier 3 extension list
- CIFAR-10 at 32×32 — the resulting 4×4 Grad-CAM has 16 cells and IoU takes only four discrete values