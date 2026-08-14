# Reliability Analysis for Image Classification

A prototype system that measures whether a confident image classification prediction is supported by *stable reasoning*, by comparing prediction-side uncertainty against the stability of Grad-CAM explanations across repeated stochastic inference passes.

Final Year Project, BSc (Hons) Computer Science with Artificial Intelligence, Asia Pacific University.
Student: TP073279.

---

## What it does

A model can return the same class with high confidence on every stochastic pass while attending to a *different region of the image* each time. Confidence and entropy cannot detect this. This system can.

For each image it computes:

- **Prediction side** — confidence, predictive entropy, predictive variance, prediction agreement, mutual information
- **Explanation side** — pairwise Grad-CAM correlation, IoU, top-k overlap, and a pixel-wise variability map

Both families are stored per image in SQLite, analysed together under controlled conditions, and presented through an application with two user roles: an engineer who runs and configures analyses, and a reviewer who inspects flagged samples.

---

## Quickstart

```bash
git clone <repo-url> && cd fyp-reliability-dashboard
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

python scripts/download_data.py                      # datasets
python scripts/init_db.py                            # create db/reliability.db

pytest tests/test_gradcam.py -v                      # VERIFICATION GATE — must pass
python src/train_model.py --config configs/config.yaml
python run_pipeline.py --dataset imagenette --n-runs 30
streamlit run app/streamlit_app.py
```

Full instructions: [`docs/SETUP.md`](docs/SETUP.md)

> **`pytest tests/test_gradcam.py` must pass before anything else is worth running.** TC16 confirms explanations actually vary under Monte Carlo Dropout; TC17 confirms they are exactly reproducible without it. If TC16 fails, the dropout layers are in the wrong place and every downstream number is meaningless. See [`docs/TESTING.md`](docs/TESTING.md).

---

## Documentation

| File | Contents |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Hard constraints for anyone (or anything) writing code here. Read first |
| [`PROGRESS.md`](PROGRESS.md) | Current status, phase tracking, decisions, session log |
| [`docs/SETUP.md`](docs/SETUP.md) | Environment, dependencies, dataset acquisition |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layers, module responsibilities, data flow |
| [`docs/METRICS.md`](docs/METRICS.md) | Every formula, with edge cases and expected ranges |
| [`docs/DATABASE.md`](docs/DATABASE.md) | Schema, data dictionary, common queries |
| [`docs/TESTING.md`](docs/TESTING.md) | Test cases, verification gates, how to run |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Experiment protocol, run recipes, results log |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | Known failure modes and fixes |

---

## Repository layout

```
app/            Streamlit application, one file per page
src/            pipeline, metrics, analysis, repository
models/         model definition and checkpoints
db/             schema.sql and reliability.db
configs/        config.yaml — all parameters
tests/          pytest suite
scripts/        one-off utilities (download, init, export)
outputs/        gradcam artefacts, figures, run logs
docs/           documentation
notebooks/      exploration only, never the source of reported results
```

---

## Stack

Python 3.11 · PyTorch 2.3 · pytorch-grad-cam · scikit-learn · pandas · SQLite · Streamlit

Developed on Windows 11 with an NVIDIA RTX 4070 SUPER. CPU execution works but repeated inference is slow.

---

## Datasets

| Dataset | Role | Images |
|---|---|---|
| Imagenette | in-distribution | 3,925 validation |
| Imagewoof | near out-of-distribution | 1,500 |
| DTD | far out-of-distribution | 1,500 |
| Imagenette-C (generated) | corrupted | 10,000 |

Imagenette is used rather than CIFAR-10 because at 32×32 input the final feature map of ResNet-18 is 4×4, giving a Grad-CAM of 16 cells — IoU between two such maps can take only four discrete values, which cannot support the analysis.

---

## Status

See [`PROGRESS.md`](PROGRESS.md). This is active FYP work, not a finished product.

## Licence

Academic coursework. Not licensed for reuse.
