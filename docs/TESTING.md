# Testing

Two audiences. `pytest` proves the code is correct; the test case tables below go into report sections 5.2 and 5.3 in the format the FYP guidelines require.

Semester 2 is assessed on *Design, Coding and **Testing***. Testing is a third of the title.

---

## Running

```bash
pytest                                   # everything
pytest tests/test_gradcam.py -v          # the verification gate
pytest -m "not slow"                     # skip GPU-heavy tests
pytest --cov=src --cov-report=term       # coverage
```

```
tests/
├── conftest.py                    fixtures: model, sample image, temp db
├── test_metrics_prediction.py     TC1–TC6
├── test_metrics_explanation.py    TC7–TC11
├── test_model_dropout.py          TC12–TC15
├── test_gradcam.py                TC16–TC18   ◄── gate
├── test_repository.py             TC24–TC27
└── test_integration.py            TC28–TC31
```

---

## Verification gate

**TC16 and TC17 must pass before any other development proceeds.**

| TC | Setup | Requirement | Failure means |
|---|---|---|---|
| **TC16** | Dropout enabled, one image, 10 maps | mean pairwise correlation **< 0.99** | Dropout sits after the Grad-CAM target layer; explanations never vary; every downstream number is meaningless |
| **TC17** | Dropout disabled, one image, 2 maps | correlation **= 1.000** | Bug in the Grad-CAM path or in seed handling |

```python
def test_tc16_cams_vary_under_dropout(model, sample_image):
    enable_mc_dropout(model)
    cams = generate_repeated_cams(model, sample_image, target_class=0, n_runs=10)
    r = mean_pairwise_correlation(cams)
    assert r < 0.99, f"CAMs are not varying (r={r:.4f}) — check Dropout2d placement"

def test_tc17_cams_deterministic_without_dropout(model, sample_image):
    model.eval()
    a = generate_single_cam(model, sample_image, target_class=0)
    b = generate_single_cam(model, sample_image, target_class=0)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] == pytest.approx(1.0, abs=1e-6)
```

Re-run after any change to the model, inference, or Grad-CAM code. TC17 is also baseline B1, so the result is reportable.

---

## Unit test cases

Report format: TC-No, Input, Expected output, Actual output, Status. Blank in 5.2, filled in 5.3.

### `metrics_prediction.py`

| TC | Input | Expected output |
|---|---|---|
| TC1 | one-hot [1,0,…,0] | entropy = 0.0 |
| TC2 | uniform over 10 classes | entropy = 2.3026 (ln 10) |
| TC3 | 30 identical distributions | pred_variance = 0.0 |
| TC4 | 30 identical distributions | mutual_information = 0.0 |
| TC5 | all 30 passes predict class 3 | pred_agreement = 1.0 |
| TC6 | 15 predict class 3, 15 predict class 7 | pred_agreement = 0.5 |

### `metrics_explanation.py`

| TC | Input | Expected output |
|---|---|---|
| TC7 | two identical maps | correlation = 1.0 |
| TC8 | two identical maps | IoU = 1.0 |
| TC9 | a map and its negation | IoU = 0.0 |
| TC10 | constant map | returns a value, no ZeroDivisionError or NaN |
| TC11 | 30 maps | 435 pairwise values computed |

### `resnet_dropout.py`

| TC | Input | Expected output |
|---|---|---|
| TC12 | built model | exactly two `Dropout2d` modules present |
| TC13 | after `enable_mc_dropout()` | all `Dropout2d` in training mode |
| TC14 | after `enable_mc_dropout()` | all `BatchNorm2d` in evaluation mode |
| TC15 | forward pass, batch of 30 | output shape (30, 10) |

TC14 matters: calling `model.train()` would put BatchNorm into training mode, replacing running statistics with batch statistics and destroying the predictions.

### `gradcam_generator.py`

| TC | Input | Expected output |
|---|---|---|
| TC16 | dropout on, 10 maps | mean pairwise correlation < 0.99 |
| TC17 | dropout off, 2 maps | correlation = 1.000 |
| TC18 | N maps generated | all generated against the same target class |

### `repository.py`

| TC | Input | Expected output |
|---|---|---|
| TC24 | insert a run | run_id returned, row present |
| TC25 | insert image result | rows in `images`, `predictions`, `explanations` |
| TC26 | insert with invalid `dataset_type` | IntegrityError raised |
| TC27 | fetch_metrics with a confidence filter | only matching rows returned |

### Integration

| TC | Input | Expected output |
|---|---|---|
| TC28 | pipeline over 10 images | 10 rows, no nulls in required columns |
| TC29 | baselines computed | B1 = 1.000; B2 and B3 within expected ranges |
| TC30 | risk groups assigned | every image has exactly one `risk_group` |
| TC31 | run interrupted mid-batch | run marked `failed`, no orphan rows |

### Application

| TC | Input | Expected output |
|---|---|---|
| TC19 | role set to Reviewer | engineer-only pages hidden |
| TC20 | run with a missing checkpoint | error message shown, no crash |
| TC21 | filter confidence > 0.95 | only matching rows displayed |
| TC22 | upload a non-image file | rejected with a message |
| TC23 | submit a review decision | persisted and visible after reload |

Streamlit tests can use `streamlit.testing.v1.AppTest`, or be run manually and evidenced with screenshots. Either is acceptable — state which was used.

---

## Fixtures

```python
# conftest.py
@pytest.fixture(scope="session")
def model():
    return build_mc_dropout_resnet18(num_classes=10, p=0.2)

@pytest.fixture
def sample_image():
    torch.manual_seed(42)
    return torch.randn(3, 224, 224)

@pytest.fixture
def temp_db(tmp_path):
    path = tmp_path / "test.db"
    conn = sqlite3.connect(path)
    conn.executescript(Path("db/schema.sql").read_text())
    conn.close()
    return str(path)

@pytest.fixture
def identical_cams():
    base = np.random.RandomState(0).rand(7, 7)
    return np.stack([base] * 10)
```

Use a temporary database for every repository test. Never point tests at `db/reliability.db`.

---

## Performance testing

Reported in report section 5.3.

| Measure | Target | How |
|---|---|---|
| Analysis per image, N = 30 | < 100 ms | time 100 images, divide |
| Full batch, 4,000 images | < 10 min | wall clock |
| Table render, 10,000 rows | < 3 s | browser timing |
| Database query, full frame | < 1 s | `time.perf_counter` around `read_sql_query` |

```python
@pytest.mark.slow
def test_throughput(model, dataset):
    t0 = time.perf_counter()
    for img in itertools.islice(dataset, 100):
        analyse_single(model, img, n_runs=30)
    per_image = (time.perf_counter() - t0) / 100
    assert per_image < 0.10
```

---

## User acceptance testing

Minimum three testers, each an actual target user, each form signed. Full protocol, recruitment messages, and the form itself are in [`../UAT_PACK.md`](../UAT_PACK.md).

---

## Reporting into the report

**Section 5.2 — Test Plan.** Tables above with Expected output filled, Actual output and Status blank. Plus the blank UAT form.

**Section 5.3 — Testing Results.** Same tables with Actual output and Status completed, then 2–5 lines of discussion after each. Completed UAT forms with signatures.

**Show the failures.** The guideline sample includes a failed case in red. A test table where everything passed on the first attempt is not credible, and Chapter 6 needs material for the limitations section. Record what failed, what the cause was, and what changed.

---

## Before submission

- [ ] All unit tests pass
- [ ] TC16 and TC17 pass and the results are recorded
- [ ] B1 = 1.000
- [ ] Integration tests pass on a fresh database
- [ ] Performance figures measured on the target hardware
- [ ] Three signed UAT forms collected
- [ ] Section 5.2 tables blank, 5.3 tables filled
- [ ] Failures documented with their resolution
