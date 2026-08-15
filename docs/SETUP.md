# Setup

Target environment: Windows 11, Python 3.11, NVIDIA RTX 4070 SUPER (12 GB).

---

## 1. Python

Use **3.11** or **3.10**. Not 3.12 — several pinned versions have no wheels for it.

```bash
python --version        # expect 3.11.x
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate   # Linux/macOS
```

Always work inside the virtual environment. A globally-installed NumPy 2.x will silently break OpenCV.

---

## 2. PyTorch with CUDA

Install PyTorch **first**, from the CUDA index, before anything else. Installing it from the default index gives a CPU-only build and repeated inference becomes unusably slow.

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
# 2.3.1+cu121 True NVIDIA GeForce RTX 4070 SUPER
```

If `cuda.is_available()` is `False`, check the NVIDIA driver version with `nvidia-smi` and reinstall from the matching CUDA index.

---

## 3. Remaining dependencies

```bash
pip install -r requirements.txt
```

```
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
scikit-learn==1.5.0
pytorch-grad-cam==1.5.0
Pillow==10.3.0
opencv-python==4.10.0.84
imagecorruptions==1.1.2
matplotlib==3.9.0
plotly==5.22.0
seaborn==0.13.2
streamlit==1.36.0
tqdm==4.66.4
pyyaml==6.0.1
joblib==1.4.2
pytest==8.2.2
```

**NumPy is pinned below 2.0 deliberately.** OpenCV 4.10 and pytorch-grad-cam 1.5 were built against the 1.x ABI. With NumPy 2.x you get `_ARRAY_API not found` at import. If anything upgrades NumPy, pin it back:

```bash
pip install "numpy==1.26.4" --force-reinstall
```

Verify the whole stack imports:

```bash
python -c "import torch, cv2, numpy, sklearn, streamlit; from pytorch_grad_cam import GradCAM; print('ok')"
```

---

## 4. Datasets

All go under `data/`. Roughly 3 GB total.

### Imagenette — in-distribution

```bash
curl -L -o data/imagenette2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf data/imagenette2-320.tgz -C data/
```

Ten classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute. Structure is `train/n01440764/...` and `val/...`, loadable with `torchvision.datasets.ImageFolder`.

### Imagewoof — near out-of-distribution

```bash
curl -L -o data/imagewoof2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz
tar -xzf data/imagewoof2-320.tgz -C data/
```

Ten dog breeds. Same source domain and resolution as Imagenette, but the classes are unseen — this is what makes it *near* OOD.

### DTD — far out-of-distribution

```python
from torchvision.datasets import DTD
DTD(root="data", split="test", download=True)
```

Textures, natively high resolution. Chosen over SVHN because SVHN is 32×32 and upscaling it to 224 confounds "out of distribution" with "blurred" — a detection result could not then be attributed to distribution shift.

### Imagenette-C — corrupted

Generated on the fly, not downloaded. `src/corruption_generator.py` applies `imagecorruptions` to the Imagenette validation set:

```python
from imagecorruptions import corrupt
corrupted = corrupt(img_np, corruption_name="gaussian_noise", severity=3)
```

Four types (`gaussian_noise`, `defocus_blur`, `brightness`, `contrast`) × five severities × 500 images = 10,000 samples. `corrupt` expects a uint8 HWC NumPy array, not a tensor.

`python scripts/download_data.py` performs all of the above.

Expected result:

```
data/
├── imagenette2-320/{train,val}/
├── imagewoof2-320/{train,val}/
└── dtd/
```

---

## 5. Database

```bash
python scripts/init_db.py
```

Executes `db/schema.sql` and creates `db/reliability.db` with nine tables. Safe to re-run; it is idempotent. To reset, delete the file and re-run. Idempotency is achieved by counting existing tables at startup and skipping schema execution when the database is already populated — `db/schema.sql` does not use `CREATE TABLE IF NOT EXISTS`, so re-executing the script against an initialised database would otherwise raise an error.

Inspect with the DB Browser for SQLite, or:

```bash
python -c "import sqlite3;print([r[0] for r in sqlite3.connect('db/reliability.db').execute(\"SELECT name FROM sqlite_master WHERE type='table'\")])"
```

See [`DATABASE.md`](DATABASE.md).

---

## 6. Configuration

Everything parameterised lives in `configs/config.yaml`. No magic numbers in function bodies.

```yaml
seed: 42
model:
  arch: resnet18
  pretrained: true
  num_classes: 10
  dropout_p: 0.2
  dropout_layers: [layer2, layer3]
mc_dropout:
  n_runs: 30
gradcam:
  target_layer: layer4
  upsample_size: 224
  iou_percentile: 80
  topk_k: 0.10
data:
  input_size: 224
  batch_size: 64
  data_root: data
risk_flag:
  confidence_threshold: 0.95
  iou_threshold: null    # populated from baseline B3
database:
  path: db/reliability.db
paths:
  checkpoints: models/checkpoints
  outputs: outputs
```

`risk_flag.iou_threshold` stays `null` until baselines have been computed. The default is the measured cross-image reference value, not an arbitrary number.

---

## 7. Verification gate

Before writing any further code:

```bash
pytest tests/test_gradcam.py -v
```

| Test | Requirement |
|---|---|
| TC16 | Dropout enabled → 10 maps of one image, mean pairwise correlation **< 0.99** |
| TC17 | Dropout disabled → 2 maps of one image, correlation **= 1.000** |

TC16 failing means the dropout layers sit after the Grad-CAM target layer and the explanations never vary. TC17 failing means a bug in the Grad-CAM path or in seed handling. See [`TESTING.md`](TESTING.md) and [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).

---

## 8. First full run

```bash
python src/train_model.py --config configs/config.yaml     # ~20-30 min, expect val acc >= 95%
python run_pipeline.py --dataset imagenette --limit 100    # smoke test
python run_pipeline.py --dataset imagenette                # full, ~10 min
python run_pipeline.py --dataset dtd --dataset-type far_ood
python src/baselines.py                                    # B1, B2, B3
streamlit run app/streamlit_app.py
```

Always run with `--limit 100` first. A schema mismatch surfacing after 4,000 images wastes an hour.

---

## 9. Storage

| Item | Size |
|---|---|
| Datasets | ~3 GB |
| Checkpoints | ~45 MB each |
| Raw Grad-CAM `.npz` (7×7 float16, ~3 KB/image) | ~50 MB for the full set |
| Representative PNGs | ~200 MB |
| Database | ~20 MB |

Only six representative maps plus the mean and variability maps are written as PNG. Writing all N per image would create over 500,000 files.

---

## 10. Git

```
.venv/
data/
outputs/
models/checkpoints/*.pth
db/*.db
__pycache__/
*.pyc
.ipynb_checkpoints/
```

Commit `db/schema.sql`, never `db/reliability.db`. Keep the repository private.
