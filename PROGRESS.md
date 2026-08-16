# PROGRESS.md

Working state of the project. Update at the end of each session.

**Student**: TP073279 | **Programme**: CSAI
**Repository**: fyp-reliability-dashboard
**Last updated**: [date]

---

## 1. Where things stand

| | |
|---|---|
| Current phase | Phase 5 — Streamlit application |
| Blocking item | None — Phases 0-4 complete |
| Next supervisor meeting | [date] — Meeting 4 of 6 |
| Log sheets outstanding | 3 |
| UAT participants secured | 0 of 3 |

---

## 2. Phase status

Exit conditions from `docs/FYP_Master_Working_Document.md`, Part II Section F. Do not start a phase until the previous one has met its exit condition.

| Phase | Work | Exit condition | Status |
|---|---|---|---|
| **0** | `resnet_dropout.py`, `enable_mc_dropout()`, gate tests | **TC16 and TC17 pass** | ✅ TC16 r=0.1594, TC17 r=1.000 |
| 1 | Fine-tune ResNet-18 on Imagenette | Validation accuracy ≥ 95%, checkpoint registered | ✅ val_acc=95.13%, model_id=1 |
| 2 | Metrics modules and baselines | TC1–TC11 pass; B1 = 1.000 | ✅ TC1-TC11 pass, B1=1.000 |
| 3 | Dataset download, inference, Grad-CAM | datasets present, batched inference working | ✅ 3 datasets downloaded |
| 4 | Batch pipeline, repository, integration tests | 100 images end to end, no nulls, TC28-TC31 pass | ✅ run_id=1, 100 images, 22 tests pass |
| 4 | Stratified analysis, ΔAUROC, figures | ΔAUROC computed; stratified figure produced | ⬜ |
| 5A | Application, engineer role | TC19–TC23 pass | ✅ 27 tests pass |
| 5B | Reviewer pages, upload, export | TC19–TC23 pass | ✅ 27 tests pass |
| 6 | OOD, corruption, ablations | Figures produced | ⬜ |
| 7 | UAT with three or more target users | Three signed forms collected | ⬜ |
| 8 | Chapters 4–6, poster, appendices | Turnitin under 20% for similarity and AI content | ⬜ |

Do not start Phase 6 until Phases 0–5 are complete. Half-finished extensions are worth less than a complete core.

---

## 3. Verification gates

| Gate | Requirement | Result | Date |
|---|---|---|---|
| **TC16** | Dropout on → 10 maps, mean pairwise correlation **< 0.99** | r = 0.1594 | 2026-08-15 |
| **TC17** | Dropout off → 2 maps, correlation **= 1.000** | r = 1.00000000 | 2026-08-15 |
| B1 | Deterministic reproduction = 1.000 | corr = 1.000, IoU = 1.000 | 2026-08-15 |
| Val accuracy | ≥ 95% on Imagenette | 95.13% | 2026-08-15 |

If TC16 fails, the dropout layers sit after the Grad-CAM target layer and nothing downstream is valid. Fix before doing anything else.

---

## 4. Decisions made

| Date | Decision | Reason | Approval |
|---|---|---|---|
| — | ERS and DURS moved to future work | Construct validity for a new composite metric is beyond an undergraduate FYP timeframe | Pending supervisor |
| — | Main dataset CIFAR-10 → Imagenette | At 32×32 the final feature map is 4×4, so Grad-CAM has 16 cells and IoU takes only four discrete values | Pending supervisor |
| — | OOD dataset SVHN → DTD | Upscaling 32×32 SVHN to 224 confounds "out of distribution" with "blurred" | Pending supervisor |
| — | Shortcut learning declared out of scope | Shortcuts produce *stable* explanations of the wrong region, so stability metrics cannot detect them by construction | Pending supervisor |
| — | Storage moved from CSV to SQLite | Flat-file storage is not acceptable for a CSAI application deliverable | Pending supervisor |
| — | `Dropout2d` after `layer2` and `layer3` | Must sit before the Grad-CAM target layer or explanations do not vary at all | Technical, no approval needed |
| — | Grad-CAM target class fixed to the mean-prediction argmax | Per-pass argmax would measure prediction instability rather than explanation instability | Technical, no approval needed |

---

## 5. Open items

| # | Item | Owner | Needed by | Status |
|---|---|---|---|---|
| 1 | Supervisor approval on the nine questions in the master document, Section H | Student | Next meeting | Waiting |
| 2 | **Three outstanding supervision log sheets** — Appendix C requires six | Student | Before submission | ⚠ Outstanding |
| 3 | Confirm SDG mapping against the registered PPF | Student | Before the abstract is final | Open |
| 4 | Confirm which design diagram set Chapter 4 requires | Supervisor | Before Phase 5 | Open |
| 5 | Secure three UAT participants matching the two roles | Student | Before Phase 7 | Open |
| 6 | Title hyphen and missing action verb — guidelines discourage both, but titles are locked at IR | Supervisor | Next meeting | Open |

---

## 6. Results recorded so far

Fill in as they are produced. These are the numbers Chapter 5 is built from.

| Quantity | Value | Date |
|---|---|---|
| Validation accuracy (Imagenette) | 95.13% | 2026-08-15 |
| B1 upper bound (deterministic reproduction) | corr = 1.000, IoU = 1.000 | 2026-08-15 |
| B2 lower bound — correlation / IoU | -0.0008 / 0.1124 | 2026-08-15 |
| B3 cross-image reference — correlation / IoU | 0.5544 / 0.3612 | 2026-08-15 |
| Mean cam_corr_mean (100-image run_id=1) | 0.9775 | 2026-08-15 |
| Mean cam_iou_mean (100-image run_id=1) | 0.8216 | 2026-08-15 |
| Prediction-stable subset size (`pred_agreement == 1.0`) | — | — |
| `cam_corr_mean` range within confidence ≥ 0.99 | — | — |
| ΔAUROC, misclassification detection | — | — |
| ΔAUROC, OOD detection | — | — |
| Hidden-risk quadrant: size and misclassification rate | — | — |

Detailed protocol and the full run log are in `docs/EXPERIMENTS.md`.

---

## 7. Known issues

| # | Issue | Impact | Workaround | Status |
|---|---|---|---|---|
| — | — | — | — | — |

---

## 8. Session log

Newest first. Two or three lines each.

### [date] — Session 1
- Repository initialised, `CLAUDE.md` and this file added.
- Next: install dependencies, download Imagenette, build `resnet_dropout.py`, run TC16 and TC17.

---

## 9. Reminders

- **Chapters 1–6 prose must be written by the student.** AI-generated content is capped at 20% and students have failed on this. Claude explains; the student writes.
- Every result in the report needs a stored configuration and seed behind it. If a number cannot be reproduced, it cannot be used.
- Phase 7 (UAT) needs participants arranged weeks in advance. Do not leave it until the application is finished.
- Do not start Phase 6 until Phases 0–5 are complete.
- Log sheet items for discussion are noted **before** each meeting — see `LOG_SHEETS.md`.


### 2026-08-15 — Session 1
- Repository initialised and pushed to GitHub. 65 files tracked.
- Environment verified: torch 2.3.1+cu121, CUDA available, numpy 1.26.4.
- Resolved: PyPI package name is `grad-cam` not `pytorch-grad-cam`; `setuptools<81` required for `imagecorruptions`.
- Next: Phase 0 — resnet_dropout.py, conftest.py, test_gradcam.py. TC16/TC17.

### 2026-08-15 — Session 2
- Phase 0 complete. TC16/TC17/TC18 pass, 5 consecutive runs identical.
- TC16 r=0.1594 (untrained fc, p=0.2). TC17 r=1.00000000 → B1.
- Finding 1: pytorch-grad-cam BaseCAM.__init__ calls model.eval() internally,
  silently disabling MC Dropout. Fixed by snapshotting Dropout2d training flags
  and restoring them inside the GradCAM context.
- Finding 2: Grad-CAM ReLU produces constant maps when all channel contributions
  to the target class are negative → np.corrcoef returns nan. Caused by noise
  input + randomly initialised fc. Fixed with a structured fixture image and by
  using the model's own argmax as target class, matching CLAUDE.md §1.3.
- Next: Phase 1.

### 2026-08-16 — Session 4
- Bug fix: raw Grad-CAM .npz files were 1.57 MB each (224×224 float32) instead of ~3 KB (7×7 float16).
- Root cause: pytorch-grad-cam's GradCAM.__call__ returns upsampled 224×224 maps. generate_repeated_cams
  was returning those directly. Fixed by extracting raw 7×7 activations and gradients from
  cam.activations_and_grads hooks after each pass and applying the GradCAM formula manually.
- Verification: shape=(30, 7, 7) float16, file_size≈2.8 KB/image, total raw/ = 270.9 KB for 100 images.
- Metrics unchanged: avg cam_corr_mean=0.9775, avg cam_iou_mean=0.8216 (no regression).
- All 22 tests pass. Log: logs/20260816_013220.log.
- Next: Phase 5 — Streamlit application.

### 2026-08-15 — Session 3
- Phases 1-4 complete.
- Phase 1: ResNet-18 fine-tuned on Imagenette, val_acc=95.13% (target >=95%), model_id=1.
  Checkpoint: models/checkpoints/resnet18_dp0.2_acc0.9513.pth.
- Phase 2: metrics_prediction.py (6 functions, TC1-TC6), metrics_explanation.py (5 additions,
  TC7-TC11), baselines.py (B1/B2/B3). 18 existing tests pass.
- Phase 3: download_data.py (Imagenette/Imagewoof/DTD), inference_mc_dropout.py (batched),
  train_model.py (AdamW + cosine LR, 15 epochs, mixed precision).
- Phase 4: batch_evaluation.py (run_batch with median risk quadrant, device-aware dataset
  wrapper for baselines, per-image loop with skip-on-failure), run_pipeline.py (CLI),
  tests/test_integration.py (TC28-TC31). Added pytest.ini to register 'slow' marker.
- Acceptance: python run_pipeline.py --dataset imagenette --limit 100 → run_id=1,
  100 images, 45.0s elapsed (449.5ms/image on RTX 4070 SUPER), 0 NULLs.
  B1=1.000, B2_corr=-0.001, B3_corr=0.554. Risk groups: stable=26, unstable_both=26,
  pred_unstable_only=24, hidden_risk=24. gradcam/ 182MB.
  Log: logs/20260815_234829.log.
- All 22 tests pass (TC1-TC11, TC16-TC18, TC24-TC31).
- Note: per-image time 449ms exceeds 100ms target; sequential Grad-CAM (N=30 passes)
  is the bottleneck. Batching Grad-CAM is a Phase 6 optimisation if needed.
- Next: Phase 5 — Streamlit application (two roles, eight pages, TC19-TC23).

### 2026-08-17 — Phase 5A + 5B complete
- Phase 5A: src/analysis_service.py facade, app/streamlit_app.py with role-gated navigation,
  pages 1_Run_Analysis, 2_Metrics_Table, 3_Statistics. fetch_metrics 59ms / 3925 rows.
- Phase 5B: Reviewer pages 4_Risk_Queue (FR-08), 5_Image_Detail (FR-09/10),
  6_Upload_Analyse (FR-11), updated 7_Export with figure bundle ZIP (FR-12).
- analysis_service.py extended: insert_review, fetch_reviews, run_uploaded_image (single-image
  MC-Dropout + Grad-CAM pipeline, stored as dataset_type='uploaded').
- TC19: AppTest verifies engineer pages blocked for reviewer role.
- TC20: Service layer — bad checkpoint path → run marked 'failed', exception raised.
- TC21: fetch_metrics with confidence__gte=0.95 returns only matching rows.
- TC22: PIL.Image.open + verify() rejects non-image bytes.
- TC23: insert_review → fetch_reviews across fresh DB connection.
- All 27 tests pass (TC1-TC11, TC16-TC18, TC19-TC23, TC24-TC31).
- Screenshots pending: run app and capture one per page into outputs/reports/screenshots/.
- Next: Phase 6 — OOD analysis, corruption figures, or Phase 7 UAT preparation.