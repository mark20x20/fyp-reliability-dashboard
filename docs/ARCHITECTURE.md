# Architecture

Three layers. Dependencies point downward only — the application never touches inference code directly, and `src/` never imports from `app/`.

```
┌─────────────────────────────────────────────────────┐
│  Presentation        app/streamlit_app.py + views/  │
│                      role-based navigation          │
└───────────────────────────┬─────────────────────────┘
                            │ AnalysisService, ReviewService
┌───────────────────────────▼─────────────────────────┐
│  Domain              src/  pipeline · metrics ·     │
│                            baselines · analysis     │
└───────────────────────────┬─────────────────────────┘
                            │ Repository
┌───────────────────────────▼─────────────────────────┐
│  Persistence         SQLite (metrics) +             │
│                      filesystem (.npz, .png)        │
└─────────────────────────────────────────────────────┘
```

Metrics live in the database. Grad-CAM arrays and images live on disk, with paths recorded in the database. Nothing reads an artefact without going through a row that points at it.

---

## Modules

### `models/resnet_dropout.py`

Builds ResNet-18 with `Dropout2d` inserted after `layer2` and `layer3`, and switches only the dropout modules into training mode for stochastic inference.

```python
def build_mc_dropout_resnet18(num_classes: int = 10, p: float = 0.2) -> nn.Module
def enable_mc_dropout(model: nn.Module) -> nn.Module
def get_target_layer(model: nn.Module) -> nn.Module    # for Grad-CAM
```

Dropout must sit **before** the Grad-CAM target layer. Placed after `layer4` it leaves the target activations unchanged and every explanation comes out identical. `enable_mc_dropout` must leave BatchNorm in eval mode. See `CLAUDE.md`.

### `src/dataset_loader.py`

Returns `(dataset, class_names)` for `imagenette`, `imagewoof`, `dtd`, `imagenette-c`. Applies the 224×224 resize, centre crop, and ImageNet normalisation. Every loader yields the same tensor contract regardless of source.

### `src/corruption_generator.py`

Wraps `imagecorruptions.corrupt`. Applied on the fly during loading, so corrupted images are never written to disk. Converts tensor → uint8 HWC NumPy → corrupt → back.

### `src/inference_mc_dropout.py`

```python
def mc_dropout_forward(model, image: Tensor, n_runs: int) -> Tensor   # (n_runs, num_classes)
```

Replicates the image `n_runs` times into one batch. Dropout masks are sampled independently per sample, so a single forward call produces `n_runs` distinct stochastic passes — roughly ten times faster than looping.

### `src/gradcam_generator.py`

```python
def generate_repeated_cams(model, image, target_class: int, n_runs: int) -> np.ndarray  # (n_runs, 7, 7)
```

`target_class` is supplied by the caller and is the argmax of the **mean** predictive distribution. It is never recomputed per pass. Using the per-pass argmax would mean a class flip forces a map change, and the measurement would capture prediction instability rather than explanation instability.

Also writes the `.npz`, the six representative PNGs, the mean map, and the variability map.

### `src/metrics_prediction.py`

Pure functions over the `(n_runs, num_classes)` probability array. No I/O, no model. Returns confidence, entropy, predictive variance, prediction agreement, mutual information. See [`METRICS.md`](METRICS.md).

### `src/metrics_explanation.py`

Pure functions over the `(n_runs, H, W)` map array. Upsamples to 224, normalises, then computes all pairwise correlation, IoU, and top-k overlap values. Returns means and standard deviations.

### `src/baselines.py`

Computes B1 (deterministic upper bound), B2 (random lower bound), B3 (cross-image reference) and writes them to the `baselines` table. B1 doubles as an implementation check: a value below 1.000 means the Grad-CAM path or the seed handling is broken.

### `src/repository.py`

The only module holding SQL. Everything else goes through it.

```python
class Repository:
    def create_run(...) -> int
    def insert_image_result(run_id, image_meta, prediction, explanation) -> int
    def fetch_metrics(run_id, filters=None) -> pd.DataFrame
    def fetch_image_detail(image_id) -> dict
    def upsert_risk_flags(run_id, flags) -> None
    def insert_review(...) -> int
    def fetch_baselines(run_id) -> dict
```

### `src/batch_evaluation.py`

Orchestrates one run: create the run row, iterate images, compute both metric families, persist, then compute baselines and assign risk groups. Progress is exposed through a callback so Streamlit can show a bar without this module importing Streamlit.

### `src/analysis_service.py`

Facade for the presentation layer. Wraps `batch_evaluation` and `repository` so pages call one object rather than assembling the pipeline themselves.

### Analysis modules

| Module | Produces |
|---|---|
| `stratified_analysis.py` | Prediction-stable subset, confidence-binned distributions — the central result |
| `complementarity.py` | ΔAUROC for misclassification and OOD detection |
| `analysis.py` | Correlations, quadrant assignment, group comparisons, figures |
| `ablation.py` | Dropout placement, dropout rate, number of passes, binarisation threshold |

---

## Data flow — batch analysis

```
config.yaml ─┐
             ▼
        AnalysisService.run_batch()
             │
             ├─► Repository.create_run()          → runs row (params + seed)
             │
             └─► for each image:
                   dataset_loader        → tensor
                   mc_dropout_forward    → (N, C) probabilities
                   metrics_prediction    → confidence, entropy, variance, agreement
                   argmax(mean probs)    → target_class          ◄── fixed here
                   generate_repeated_cams→ (N, 7, 7) + artefacts
                   metrics_explanation   → correlation, IoU, top-k
                   Repository.insert_image_result()
             │
             ├─► baselines.compute_all()          → baselines rows
             └─► RiskClassifier.assign()          → risk_flags rows
```

The target class is determined **once**, from the mean of the stochastic predictions, and then held constant for all N explanation maps. That single decision is what makes the rest of the analysis interpretable.

---

## Data flow — single image detail

```
Reviewer → ImageDetailPage
             → Repository.fetch_image_detail(image_id)
                  ├── predictions row
                  ├── explanations row + artefact paths
                  ├── risk_flags row
                  └── baselines for the run
             → load .npz and PNGs from disk
             → render metrics against baseline reference values
             → ReviewService.record(decision, comment)
```

Metric values are always rendered next to their baselines. "IoU 0.42" alone tells the reviewer nothing; "0.42, against a random-map floor of 0.20 and a same-class reference of 0.35" does.

---

## Application pages

| Page | Role | Service calls |
|---|---|---|
| Home | Both | `fetch_run_summary` |
| Run Analysis | Engineer | `run_batch` with progress callback |
| Metrics Table | Engineer | `fetch_metrics` with filters |
| Statistics | Engineer | `fetch_metrics`, `fetch_baselines`, plotting |
| Risk Queue | Reviewer | `fetch_metrics(risk_group='hidden_risk')` |
| Image Detail | Reviewer | `fetch_image_detail`, `record_review` |
| Upload & Analyse | Reviewer | `analyse_single_image` |
| Export | Both | `fetch_metrics`, `fetch_reviews` |

Role is held in `st.session_state` and controls which pages appear. This is the two-target-user requirement, so it must be visible in the interface and in the use case diagram, not just in code.

---

## Conventions

**Configuration** — read once at entry point, passed down explicitly. No module reads `config.yaml` on its own.

**Seeding** — set at entry point from `config.seed` and written into the `runs` row. A result that cannot be reproduced cannot be reported.

**Pure metric functions** — `metrics_*` modules take arrays and return floats. No file access, no model, no database. This is what makes them unit-testable against analytical values.

**Errors** — the pipeline logs and skips a failed image rather than aborting a 4,000-image run. Failures are counted and surfaced in the run summary.

**Caching** — `@st.cache_data` on read-only queries, cleared after a write. Never cache the model object with `cache_data`; use `cache_resource`.

---

## Extension points

| To add | Change |
|---|---|
| Another architecture | `models/`, keep the `get_target_layer` contract |
| Another explanation method | `gradcam_generator.py`, keep the `(N, H, W)` return shape |
| Another metric | the relevant `metrics_*` module, plus a column and a migration |
| Another dataset | `dataset_loader.py`, plus a `dataset_type` value |

The `(N, num_classes)` and `(N, H, W)` array contracts are the seams. Keeping them stable means the metric and analysis layers never need to change.
