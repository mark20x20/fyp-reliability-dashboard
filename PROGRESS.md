# PROGRESS.md

Working state of the project. Update at the end of each session.

**Student**: TP073279 | **Programme**: CSAI
**Repository**: fyp-reliability-dashboard
**Last updated**: [date]

---

## 1. Where things stand

| | |
|---|---|
| Current phase | Phase 0 — verification |
| Blocking item | TC16 / TC17 not yet run |
| Next supervisor meeting | [date] — Meeting 4 of 6 |
| Log sheets outstanding | 3 |
| UAT participants secured | 0 of 3 |

---

## 2. Phase status

Exit conditions from `docs/FYP_Master_Working_Document.md`, Part II Section F. Do not start a phase until the previous one has met its exit condition.

| Phase | Work | Exit condition | Status |
|---|---|---|---|
| **0** | `resnet_dropout.py`, `enable_mc_dropout()`, gate tests | **TC16 and TC17 pass** | ⬜ Not started |
| 1 | Fine-tune ResNet-18 on Imagenette | Validation accuracy ≥ 95%, checkpoint registered | ⬜ |
| 2 | Metrics modules and baselines | TC1–TC11 pass; B1 = 1.000 | ⬜ |
| 3 | Inference, Grad-CAM, batch pipeline, repository | 100 images end to end, no nulls in required columns | ⬜ |
| 4 | Stratified analysis, ΔAUROC, figures | ΔAUROC computed; stratified figure produced | ⬜ |
| 5 | Application, eight pages, two roles | TC19–TC23 pass | ⬜ |
| 6 | OOD, corruption, ablations | Figures produced | ⬜ |
| 7 | UAT with three or more target users | Three signed forms collected | ⬜ |
| 8 | Chapters 4–6, poster, appendices | Turnitin under 20% for similarity and AI content | ⬜ |

Do not start Phase 6 until Phases 0–5 are complete. Half-finished extensions are worth less than a complete core.

---

## 3. Verification gates

| Gate | Requirement | Result | Date |
|---|---|---|---|
| **TC16** | Dropout on → 10 maps, mean pairwise correlation **< 0.99** | — | — |
| **TC17** | Dropout off → 2 maps, correlation **= 1.000** | — | — |
| B1 | Deterministic reproduction = 1.000 | — | — |
| Val accuracy | ≥ 95% on Imagenette | — | — |

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
| Validation accuracy (Imagenette) | — | — |
| B1 upper bound (deterministic reproduction) | — | — |
| B2 lower bound — correlation / IoU | — | — |
| B3 cross-image reference — correlation / IoU | — | — |
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