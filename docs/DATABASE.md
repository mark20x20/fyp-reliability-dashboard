# Database

SQLite, nine tables, at `db/reliability.db`. Schema in `db/schema.sql`.

**The database is the system of record.** CSV is an export feature (FR-07), never the storage mechanism. Flat-file storage is not acceptable for a CSAI application deliverable.

This schema is the source for the Entity Relationship Diagram in report section 4.3.

---

## Entity relationships

```
users ──┬──< runs >── models
        │      │
        │      ├──< images ──┬── predictions   (1:1)
        │      │             ├── explanations  (1:1)
        │      │             ├── risk_flags    (1:1)
        │      │             └──< reviews
        │      │
        │      └──< baselines
        └──< reviews
```

| Relationship | Cardinality |
|---|---|
| models → runs | one to many |
| users → runs | one to many (nullable) |
| runs → images | one to many |
| images → predictions | one to one |
| images → explanations | one to one |
| images → risk_flags | one to one |
| images → reviews | one to many |
| runs → baselines | one to many |

---

## Schema

```sql
CREATE TABLE users (
    user_id   INTEGER PRIMARY KEY,
    username  TEXT NOT NULL UNIQUE,
    role      TEXT NOT NULL CHECK(role IN ('engineer','reviewer'))
);

CREATE TABLE models (
    model_id        INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    arch            TEXT NOT NULL,
    dropout_p       REAL NOT NULL,
    dropout_layers  TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    val_accuracy    REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE runs (
    run_id        INTEGER PRIMARY KEY,
    model_id      INTEGER NOT NULL REFERENCES models(model_id),
    user_id       INTEGER REFERENCES users(user_id),
    dataset_name  TEXT NOT NULL,
    n_runs        INTEGER NOT NULL,
    seed          INTEGER NOT NULL,
    iou_threshold REAL NOT NULL,
    topk_k        REAL NOT NULL,
    status        TEXT NOT NULL CHECK(status IN ('pending','running','completed','failed')),
    started_at    TIMESTAMP,
    finished_at   TIMESTAMP
);

CREATE TABLE images (
    image_id            INTEGER PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES runs(run_id),
    image_path          TEXT NOT NULL,
    dataset_type        TEXT NOT NULL
        CHECK(dataset_type IN ('id','near_ood','far_ood','corrupted','uploaded')),
    corruption_type     TEXT,
    corruption_severity INTEGER CHECK(corruption_severity BETWEEN 1 AND 5),
    true_label          TEXT
);

CREATE TABLE predictions (
    prediction_id      INTEGER PRIMARY KEY,
    image_id           INTEGER NOT NULL UNIQUE REFERENCES images(image_id),
    pred_label         TEXT NOT NULL,
    correct            INTEGER CHECK(correct IN (0,1)),
    confidence         REAL NOT NULL,
    entropy            REAL NOT NULL,
    pred_variance      REAL NOT NULL,
    pred_agreement     REAL NOT NULL,
    mutual_information REAL
);

CREATE TABLE explanations (
    explanation_id    INTEGER PRIMARY KEY,
    image_id          INTEGER NOT NULL UNIQUE REFERENCES images(image_id),
    cam_corr_mean     REAL NOT NULL,
    cam_corr_std      REAL NOT NULL,
    cam_iou_mean      REAL NOT NULL,
    topk_overlap      REAL NOT NULL,
    cam_npz_path      TEXT,
    mean_png_path     TEXT,
    variance_png_path TEXT
);

CREATE TABLE risk_flags (
    image_id   INTEGER PRIMARY KEY REFERENCES images(image_id),
    risk_group TEXT NOT NULL
        CHECK(risk_group IN ('stable','unstable_both','pred_unstable_only','hidden_risk')),
    risk_score REAL
);

CREATE TABLE reviews (
    review_id  INTEGER PRIMARY KEY,
    image_id   INTEGER NOT NULL REFERENCES images(image_id),
    user_id    INTEGER NOT NULL REFERENCES users(user_id),
    decision   TEXT NOT NULL CHECK(decision IN ('accept','needs_review','reject')),
    comment    TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE baselines (
    baseline_id   INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES runs(run_id),
    baseline_type TEXT NOT NULL
        CHECK(baseline_type IN ('upper','lower','cross_image')),
    metric_name   TEXT NOT NULL,
    value         REAL NOT NULL
);

CREATE INDEX idx_images_run   ON images(run_id);
CREATE INDEX idx_images_type  ON images(dataset_type);
CREATE INDEX idx_pred_conf    ON predictions(confidence);
CREATE INDEX idx_pred_agree   ON predictions(pred_agreement);
CREATE INDEX idx_expl_iou     ON explanations(cam_iou_mean);
CREATE INDEX idx_risk_group   ON risk_flags(risk_group);
CREATE INDEX idx_reviews_img  ON reviews(image_id);
```

SQLite does not enforce foreign keys by default. Enable per connection:

```python
conn.execute("PRAGMA foreign_keys = ON")
```

---

## Data dictionary

### `runs` — the reproducibility record

| Column | Type | Notes |
|---|---|---|
| `n_runs` | INTEGER | Stochastic passes, default 30 |
| `seed` | INTEGER | From config; without it results cannot be reproduced |
| `iou_threshold` | REAL | Binarisation percentile, default 80 |
| `topk_k` | REAL | Top-k fraction, default 0.10 |
| `status` | TEXT | `pending` / `running` / `completed` / `failed` |

Storing the parameters here rather than only in a config file means every reported number is traceable to the exact configuration that produced it. This is what makes Chapter 5 defensible.

### `images`

| Column | Notes |
|---|---|
| `dataset_type` | Drives group comparisons; `uploaded` covers ad-hoc reviewer analyses |
| `corruption_type` / `corruption_severity` | NULL unless `dataset_type = 'corrupted'` |
| `true_label` | NULL for OOD and uploaded images |

### `predictions`

| Column | Range | Notes |
|---|---|---|
| `correct` | 0/1 | NULL where no ground truth exists |
| `confidence` | [0.1, 1.0] | Maximum softmax probability of the mean distribution |
| `entropy` | [0, 2.30] | Natural log |
| `pred_variance` | [0, 0.25] | Over the predicted-class channel only |
| `pred_agreement` | [1/N, 1.0] | **The control variable for the stratified analysis** |
| `mutual_information` | [0, entropy] | Nullable — optional metric |

### `explanations`

| Column | Range | Notes |
|---|---|---|
| `cam_corr_mean` | [-1, 1] | Mean over all 435 pairs at N = 30 |
| `cam_iou_mean` | [0, 1] | Percentile-binarised |
| `topk_overlap` | [0, 1] | Top 10% of pixels |
| `*_path` | — | Filesystem paths; arrays are not stored as BLOBs |

Grad-CAM arrays stay on disk. A 7×7 float16 `.npz` is ~3 KB, so all 17,000 fit in about 50 MB. Storing them as BLOBs would bloat the database and make it hard to inspect.

---

## Common queries

**Prediction-stable, high-confidence subset — the central analysis**

```sql
SELECT i.image_id, p.confidence, e.cam_corr_mean, e.cam_iou_mean
FROM images i
JOIN predictions  p ON p.image_id = i.image_id
JOIN explanations e ON e.image_id = i.image_id
WHERE i.run_id = ?
  AND p.pred_agreement = 1.0
  AND p.confidence >= 0.99;
```

Every row here is indistinguishable on the prediction side. If `cam_corr_mean` still spans a wide range, the central claim holds.

**Hidden-risk queue**

```sql
SELECT i.image_id, i.image_path, p.confidence, e.cam_iou_mean, r.risk_score
FROM risk_flags r
JOIN images       i ON i.image_id = r.image_id
JOIN predictions  p ON p.image_id = r.image_id
JOIN explanations e ON e.image_id = r.image_id
WHERE r.risk_group = 'hidden_risk' AND i.run_id = ?
ORDER BY r.risk_score DESC
LIMIT 50;
```

**Correct vs misclassified**

```sql
SELECT p.correct, COUNT(*) AS n,
       AVG(e.cam_corr_mean) AS corr, AVG(e.cam_iou_mean) AS iou
FROM predictions p
JOIN explanations e ON e.image_id = p.image_id
JOIN images       i ON i.image_id = p.image_id
WHERE i.run_id = ? AND p.correct IS NOT NULL
GROUP BY p.correct;
```

**Corruption severity trend**

```sql
SELECT i.corruption_severity AS sev, COUNT(*) AS n,
       AVG(p.confidence) AS conf, AVG(e.cam_iou_mean) AS iou
FROM images i
JOIN predictions  p ON p.image_id = i.image_id
JOIN explanations e ON e.image_id = i.image_id
WHERE i.dataset_type = 'corrupted' AND i.run_id = ?
GROUP BY i.corruption_severity
ORDER BY sev;
```

**Full frame for the analysis modules**

```sql
SELECT i.image_id, i.dataset_type, i.corruption_type, i.corruption_severity,
       i.true_label, p.pred_label, p.correct,
       p.confidence, p.entropy, p.pred_variance, p.pred_agreement,
       p.mutual_information,
       e.cam_corr_mean, e.cam_corr_std, e.cam_iou_mean, e.topk_overlap,
       r.risk_group
FROM images i
JOIN predictions  p ON p.image_id = i.image_id
JOIN explanations e ON e.image_id = i.image_id
LEFT JOIN risk_flags r ON r.image_id = i.image_id
WHERE i.run_id = ?;
```

`LEFT JOIN` on `risk_flags` because risk groups are assigned after the per-image loop finishes.

**Baselines for a run**

```sql
SELECT baseline_type, metric_name, value
FROM baselines WHERE run_id = ?;
```

---

## Access pattern

All SQL lives in `src/repository.py`. No other module writes queries.

```python
class Repository:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row

    def create_run(self, model_id, dataset_name, cfg, user_id=None) -> int: ...
    def insert_image_result(self, run_id, image_meta, prediction, explanation) -> int: ...
    def fetch_metrics(self, run_id, filters=None) -> pd.DataFrame: ...
    def fetch_image_detail(self, image_id) -> dict: ...
    def upsert_risk_flags(self, run_id, flags) -> None: ...
    def insert_review(self, image_id, user_id, decision, comment) -> int: ...
    def fetch_baselines(self, run_id) -> dict: ...
```

`check_same_thread=False` is required for Streamlit, which serves pages from a worker thread.

---

## Performance

Batch the writes. One commit per image turns a ten-minute run into an hour.

```python
BATCH = 100
def insert_batch(self, rows):
    with self.conn:                       # single transaction
        self.conn.executemany(INSERT_SQL, rows)
```

Use `pandas.read_sql_query` for analysis reads — 17,000 rows load in well under a second with the indexes above.

---

## Migrations

There is no migration framework. To add a column:

1. Write `db/migrations/00X_description.sql`
2. `ALTER TABLE ... ADD COLUMN ...`
3. Update `db/schema.sql` so a fresh install matches
4. Update `Repository`
5. Note it in `PROGRESS.md`

Re-running the pipeline is often faster than backfilling. A full run takes about 15 minutes.

---

## Backup

`db/reliability.db` is gitignored. Copy it before any schema change and before submission.

```bash
copy db\reliability.db db\reliability_backup_YYYYMMDD.db
```

The database holds every reported number. Losing it means re-running everything.
