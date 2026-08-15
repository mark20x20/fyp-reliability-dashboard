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
