# FYP Master Working Document

**Student**: [Your Full Name] | **TP**: TP073279 | **Programme**: BSc (Hons) Computer Science with Artificial Intelligence
**Supervisor**: [Supervisor's Name] | **Second Marker**: [Second Marker's Name]
**Version**: 0.1 (working draft) | **Date**: [Date]

---

## How to use this document

This document has three parts and serves two purposes at once.

| Part | Purpose |
|---|---|
| **Part I — Report Skeleton** | Mirrors the official *FYP Report Writing Guidelines (SOC & SOT), V1-AUGUST2023* section by section. Content decided so far is filled in; everything else is marked `[TODO]`. Write the final report directly into this structure. |
| **Part II — Implementation Specification** | The engineering backbone. Module-by-module specification, algorithms, formulas, database schema, and acceptance criteria. Build the codebase from this. |
| **Part III — Compliance and Open Decisions** | Checklist against the guidelines, plus items requiring the supervisor's decision. |

`[TODO]` marks work not yet done. `[CONFIRM]` marks an item requiring supervisor approval.

---
---

# PART I — REPORT SKELETON

---

## Front matter

| Item | Requirement | Status |
|---|---|---|
| Cover page | Use the template from the FYP manager's folder. APU logo only, on the right, unless dual degree with DMU. **No header or footer on the cover page.** | `[TODO]` |
| Declaration of Thesis Confidentiality | Form from the FYP manager's folder | `[TODO]` |
| Library Form | Form from the FYP manager's folder | `[TODO]` |
| Acknowledgement | One page. Supervisor, FYP manager, interview and UAT participants, family | `[TODO]` |
| Abstract | **Maximum 200 words**, one paragraph. Must state purpose, research problem, methods, conclusion or preliminary result, significance, and **SDG mapping**. **Maximum 6 keywords** below the abstract | Draft below |
| Table of Contents | Generated from this structure | `[TODO]` |
| List of Figures | Diagrams, screenshots, graphs, charts, code snippets | `[TODO]` |
| List of Tables | All tables | `[TODO]` |

**Each of these must start on a new page.**

### Project title

> A Prototype System for Reliability-Oriented Image Classification Analysis Using Prediction and Explanation Uncertainty

Word count: 13 (limit 15). ✔

`[CONFIRM]` **Two possible conflicts with the guidelines:**

1. The guidelines state the title must not contain special characters, listing `; -? !, . /`. The hyphen in "Reliability-Oriented" may fall under this.
2. The guidelines state the title should contain an action verb (Investigating, Exploring, Developing, Designing, and so on). The current title is a noun phrase.

However, the guidelines also state titles are **locked at the IR stage and must not change**. The supervisor may make minor grammatical revisions. Raise both points at the meeting and follow the supervisor's instruction. If a revision is permitted, a compliant alternative would be:

> Developing a Prototype System to Analyse Prediction Uncertainty and Explanation Stability in Image Classification

(17 words — would need trimming to 15.) **Do not change the title without explicit approval.**

### Abstract draft (~190 words)

> Deep learning models for image classification often produce confident predictions that cannot be relied upon, because existing reliability measures describe only the output distribution and not the stability of the reasoning behind it. This project investigates whether a model can produce stable, high-confidence predictions while its explanation attends to different regions on each stochastic inference pass. A prototype application was developed that computes prediction-side uncertainty using Monte Carlo Dropout and explanation-side stability from repeated Grad-CAM maps, storing both families of metrics per image in a relational database. The analysis controls for prediction variation by fixing the explanation target class and restricting comparison to samples whose predicted class was identical across all passes. Reference baselines establish the interpretable range of each stability metric. Complementarity is assessed by the change in AUROC when explanation-side metrics are added to a prediction-side baseline for misclassification and out-of-distribution detection. `[TODO: state the headline finding]` The application supports two user roles and connects batch-level statistics to individual-image explanation detail, supporting SDG 9 through more accountable artificial intelligence infrastructure.
>
> **Keywords** — Explainable Artificial Intelligence, Uncertainty Quantification, Image Classification, Model Reliability, Gradient-weighted Class Activation Mapping, Monte Carlo Dropout

---

## CHAPTER 1: INTRODUCTION

### 1.1 Introduction
Chapter overview: what the reader will find in Chapter 1. `[TODO — write last]`

### 1.2 Problem Background

Carry forward from the IR, with one change. Three problems, each supported by citations.

**Problem 1 — Reliability evaluation is concentrated on the prediction side.**
Evaluation of image classification reliability relies on accuracy, confidence, calibration, and entropy. These describe how stable or well-calibrated the output is, but do not assess whether the reasoning behind that output is stable. `[Citations: 2-3 required]`

**Problem 2 — A model can be confident and consistent in output while being inconsistent in reasoning.**
Under stochastic inference, a model may return the same class with high confidence on every pass while its explanation map attends to different regions each time. Output-level measures cannot reveal this. `[Citations required]`

**Problem 3 — Uncertainty-only measures may miss hidden failure cases.**
Confidence and entropy can appear safe for misclassified samples and for inputs drawn from a different distribution, leaving certain failure modes undetected. `[Citations required]`

> **Change from IR**: Problem 3 previously also referred to shortcut reliance. Removed — see Section 1.5 and Chapter 6 Limitations for the reasoning.

### 1.3 Project Aim

A single sentence. Unchanged from the IR.

> To develop a prototype system for reliability-oriented image classification analysis by integrating prediction uncertainty and explanation uncertainty within a single analysis workflow.

### 1.4 Objectives

Four objectives (guideline: minimum three, maximum four). Each begins with "To", uses a measurable verb, and has a defined evidence of achievement.

| | Objective | Evidence of achievement | Where demonstrated |
|---|---|---|---|
| **O1** | To implement a batch evaluation pipeline that computes prediction-side uncertainty and explanation-side stability metrics for image classification outputs. | A pipeline that processes a dataset and populates the database with all required metrics, with unit tests passing. | 4.5, 5.3.1 |
| **O2** | To analyse the relationship between prediction uncertainty and explanation stability, including cases where stable predictions are accompanied by unstable explanations. | Correlation analysis, stratified analysis controlling for prediction stability, and quadrant analysis, with reported effect sizes. | 5.4 |
| **O3** | To develop a reliability analysis application that presents batch-level metrics and single-image explanation details for two target user roles. | A functional application with role-specific features, evidenced by screenshots and unit test results. | 4.4, 4.5, 5.3.1 |
| **O4** | To evaluate whether explanation-side metrics provide complementary information for identifying potentially unreliable predictions, and to assess the usability of the application based on user feedback. | ΔAUROC from the complementarity experiment, and UAT results from a minimum of three target users. | 5.3.2, 5.4 |

> **Change from IR**: the previous objectives included developing and validating two new composite metrics (Explanation Reliability Score and Dual-Uncertainty Risk Score). These are removed as objectives. O4 now carries the usability evaluation, which connects the IR interview data to the final deliverable. `[CONFIRM]`

### 1.5 Scope

**Tasks to be executed**

- Fine-tune ResNet-18 on Imagenette and verify the Monte Carlo Dropout configuration
- Implement repeated stochastic inference and repeated Grad-CAM generation
- Compute prediction-side metrics: confidence, predictive entropy, predictive variance, prediction agreement
- Compute explanation-side metrics: pairwise Grad-CAM correlation, IoU, top-k overlap, variability map
- Compute three reference baselines to make metric values interpretable
- Store all results in a relational database
- Analyse the relationship between the two metric families under controlled conditions
- Develop an application with two user roles
- Conduct unit testing, integration testing, and user acceptance testing

**Constraints**

- Single architecture (ResNet-18); single explanation method (Grad-CAM); single uncertainty estimation method (Monte Carlo Dropout)
- Local execution on a single workstation with one consumer GPU
- Local SQLite database, not a distributed or cloud database

**What will not be done**

- Development or validation of new composite reliability metrics (ERS, DURS) — future work only
- Detection of shortcut learning — excluded by construction, see Chapter 6
- Modalities other than images: text, audio, video
- Generative artificial intelligence tasks
- Exhaustive benchmarking across explanation methods, architectures, or uncertainty estimation methods
- Production or clinical deployment

### 1.6 Potential Benefit

**Tangible benefits**
- A working application that identifies images where a confident prediction rests on unstable reasoning
- A reusable batch evaluation pipeline and a metrics database supporting reproducible experiments
- A quantified answer to whether explanation-side metrics add detection power over prediction-side metrics

**Intangible benefits**
- Supports more cautious interpretation of confident model outputs
- Provides a concrete way to inspect model reasoning rather than accepting a single explanation map
- Contributes to accountable artificial intelligence infrastructure (SDG 9)

**Target users**

| Role | Who they are | What they need |
|---|---|---|
| **Machine learning engineer / researcher** | Practitioners who train and evaluate image classification models | To run batch analyses, compare configurations, and see distribution-level statistics |
| **Reviewer** | Users who inspect model outputs before acting on them | To find and examine samples whose predictions may not be trustworthy, and record a decision |

### 1.7 Overview of the FYP Documentation
One paragraph per chapter. `[TODO — write last]`

### 1.8 Project Plan
Reference to the combined Gantt chart in Appendix E, plus a short narrative. `[TODO]`

---

## CHAPTER 2: LITERATURE REVIEW

### 2.1 Introduction
`[TODO]`

### 2.2 Domain Research

General to specific, five domains. Every sub-topic requires a comprehensive, critical, and detailed review. Target minimum five citations per domain, 95% within five years.

| # | Domain | Content | Status |
|---|---|---|---|
| 2.2.1 | Image classification reliability | Accuracy does not imply trustworthiness; what "reliability" means for a classifier | Carry from IR, expand |
| 2.2.2 | Uncertainty quantification | Aleatoric vs epistemic uncertainty; Monte Carlo Dropout; Deep Ensembles; calibration | Carry from IR |
| 2.2.3 | Explainable artificial intelligence | Saliency methods; Class Activation Mapping and Gradient-weighted Class Activation Mapping; alternatives (Integrated Gradients, LIME, SHAP) and why Grad-CAM was chosen | Carry from IR |
| 2.2.4 | Evaluation of explanations | Faithfulness, localisation, robustness, and stability; why stability under stochastic inference is comparatively under-examined | **Expand — this is the research gap** |
| 2.2.5 | Reliability-related failure modes | Misclassification without confidence drop; distribution shift; input corruption | Carry from IR |

### 2.3 Similar Works

Characteristics, strengths, weaknesses, then a conclusion identifying the gap.

| Work | Approach | Strength | Weakness relative to this project |
|---|---|---|---|
| `[Work 1 from IR Table 2]` | | | |
| `[Work 2]` | | | |
| `[Work 3]` | | | |
| `[Work 4]` | | | |

**Conclusion — research gap**: prediction-side uncertainty and explanation-side behaviour are generally studied separately and are rarely computed for the same samples within a single inspectable analysis workflow. This project computes both families per image, stores them together, analyses their relationship under controlled conditions, and presents the result through an application.

### 2.4 Technical Research

| Category | Selection | Justification |
|---|---|---|
| Programming language | Python 3.11 | Ecosystem for deep learning and statistical analysis; required libraries are Python-native |
| Deep learning framework | PyTorch 2.3 | Hook-based access to intermediate activations and gradients, required for Grad-CAM; standard in current literature |
| Model architecture | ResNet-18 (ImageNet-pretrained) | Standard baseline; residual blocks give a clean final convolutional layer for Grad-CAM; low computational cost; reproducible |
| Explanation library | pytorch-grad-cam 1.5 | Maintained implementation of Grad-CAM; avoids reimplementation error |
| Statistical library | scikit-learn 1.5, SciPy 1.13 | AUROC, AUPR, logistic regression, cross-validation, correlation and non-parametric tests |
| Data handling | pandas 2.2, NumPy 1.26 | Tabular metric handling; NumPy pinned below 2.0 for OpenCV and pytorch-grad-cam compatibility |
| Corruption generation | imagecorruptions 1.1 | Reproduces the exact procedure used to construct CIFAR-10-C |
| Interface framework | Streamlit 1.36 | Rapid construction of a multi-page data application in Python; keeps the analysis code and interface in one language |
| Database management system | SQLite 3 | Serverless relational database; sufficient for single-workstation deployment; supports the required schema and constraints |
| Operating system | Windows 11 | Development machine; CUDA-supported |
| Integrated development environment | Visual Studio Code | Python and Jupyter support; integrated Git |
| Version control | Git and GitHub (private) | Change history and backup |
| Hardware | NVIDIA RTX 4070 SUPER (12 GB), 32 GB system RAM | Sufficient for ResNet-18 fine-tuning and repeated inference at 224×224 |

### 2.5 Summary
`[TODO]`

---

## CHAPTER 3: METHODOLOGY

*(CS and CSAI structure)*

### 3.1 Introduction
`[TODO]`

### 3.2 System Development Methodology

**Introduction** — the project produces a functional application while also involving model training, batch experimentation, and analysis of the resulting data.

**Choice and justification** — a **hybrid methodology** combining an iterative system development methodology with a data analytics workflow. Neither alone covers both aspects: a pure software methodology has no phase for data understanding and model evaluation, and a pure analytics methodology has no phase for interface design and user testing. `[CONFIRM the specific pairing with supervisor, e.g. Rapid Application Development + CRISP-DM]`

**Phases**

| Phase | Activities | Output |
|---|---|---|
| 1. Problem definition and literature review | Domain and similar works review; gap identification | Chapters 1–2 |
| 2. Requirements gathering | Semi-structured interviews with target users; thematic analysis | Section 3.3–3.4 |
| 3. Data understanding and preparation | Dataset selection; corruption generation; verification of the Monte Carlo Dropout configuration | Section 4.5.1 |
| 4. Design | Use case, activity, sequence, class diagrams; database design; interface design | Sections 4.2–4.4 |
| 5. Implementation | Pipeline, metrics, database, application; iterative build | Sections 4.5–4.6 |
| 6. Evaluation and testing | Experiments; unit, integration, and user acceptance testing | Chapter 5 |

### 3.3 Data Gathering Design

Carried from the IR without change.

| Item | Content |
|---|---|
| Technique | Semi-structured interviews (qualitative) |
| Instrument | Ten questions, each mapped to an objective (IR Table 5) |
| Participants | Minimum three (guideline requirement); users who interpret image classification outputs |
| Ethics | Ethics form submitted and approved in Semester 1 — **Appendix B** |
| Demographic profile | **Required in Appendix G** |

> The guidelines state explicitly that data gathering must be conducted with the identified target audience and **not with friends**. Ensure the demographic profile in Appendix G makes the participants' roles evident.

### 3.4 Analysis

Thematic analysis of the interview data (IR Table 6), extended with the mapping from theme to implemented feature. **This mapping is the link that carries the IR data collection through to the final deliverable.**

| Interview theme | Derived requirement | Implemented feature | Section |
|---|---|---|---|
| Preference for several separate indicators over a single combined score (IR Table 5, Q10) | Metrics must be presented individually, not aggregated | Metrics table with independent columns for confidence, entropy, correlation, IoU | 4.4, 4.5 |
| Desire to see the basis for a reliability judgement | The interface must show the evidence, not only the score | Single-image detail view: repeated Grad-CAMs, mean map, variability map | 4.4, 4.5 |
| Large result sets cannot be inspected exhaustively | The system must prioritise which samples to inspect | Risk queue with adjustable thresholds | 4.4, 4.5 |
| `[TODO: remaining themes from IR Table 6]` | | | |

**Final list of user requirements** — see the functional requirements table in Part II, Section B.1.

### 3.5 Summary
`[TODO]`

---

## CHAPTER 4: DESIGN AND IMPLEMENTATION

### 4.1 Introduction
`[TODO]`

### 4.2 Design

`[CONFIRM]` The guidelines say the diagram set depends on the development model chosen. The application uses a relational database but is implemented in object-oriented Python. Ask the supervisor which set to present.

| Diagram | Purpose | Status |
|---|---|---|
| System architecture diagram | Three layers: pipeline, database, application | `[TODO]` |
| Use case diagram | Two actors and their use cases — **directly evidences the two-target-user requirement** | `[TODO]` |
| Use case specification | Step-by-step flow for the main use cases (run batch analysis; review flagged sample; analyse uploaded image) | `[TODO]` |
| Activity diagram | Batch analysis execution flow | `[TODO]` |
| Sequence diagram | Single-image analysis interaction | `[TODO]` |
| Class diagram | Application structure | `[TODO]` |
| *(Alternative if structured design is required)* Context diagram, DFD level 0, DFD level 1 | | `[TODO]` |

Content specification for each diagram is in Part II, Section B.2.

### 4.3 Database Design

Entity Relationship Diagram plus a data dictionary for nine tables. Full schema in Part II, Section C.

### 4.4 Interface Design

Wireframes for eight pages, navigation map, and a statement of the design guideline followed (consistency, feedback, error prevention, recognition over recall). Page specifications in Part II, Section B.3.

### 4.5 Implementation

The guidelines state that for projects involving machine learning analysis, the analysis steps may be included in this section: data collection, data pre-processing, data understanding, model building, and model evaluation. This project therefore uses the following sub-structure.

| Section | Content |
|---|---|
| 4.5.1 | Data collection — datasets, sources, and links |
| 4.5.2 | Data pre-processing — transforms, normalisation, corruption generation |
| 4.5.3 | Data understanding — class distribution, sample images, dataset statistics |
| 4.5.4 | Model building — ResNet-18 fine-tuning; Dropout insertion and its verification |
| 4.5.5 | Uncertainty and explanation generation — Monte Carlo Dropout and repeated Grad-CAM |
| 4.5.6 | Metric computation — prediction-side and explanation-side |
| 4.5.7 | Reference baselines |
| 4.5.8 | Application screenshots, one per page, each with a discussion |

### 4.6 Sample Codes

Screenshots of key programs with a 3–5 line description each. Minimum set:

- `resnet_dropout.py` — Dropout insertion and inference-mode switching
- `inference_mc_dropout.py` — batch-replication stochastic inference
- `gradcam_generator.py` — repeated Grad-CAM with fixed target class
- `metrics_explanation.py` — pairwise stability metrics
- `complementarity.py` — ΔAUROC experiment

### 4.7 Summary
`[TODO]`

---

## CHAPTER 5: RESULTS AND DISCUSSIONS

### 5.1 Introduction
`[TODO]`

### 5.2 Test Plan

The guidelines require the **blank templates** here, and the **filled results** in 5.3.

#### 5.2.1 Unit Testing

Test case tables per program, with Expected output filled and Actual output and Status left blank. Full test case list in Part II, Section D.1.

#### 5.2.2 User Acceptance Testing

Blank UAT form. Required elements per the guidelines:

- Tester demographic profile: Name, Age, **Role in the system**
- Rating scale 1–5 (Strongly disagree to Strongly agree)
- **User interface criteria** table
- **Functionality criteria** table (Yes / No)
- Tester comment box
- Tester's signature

Draft criteria in Part II, Section D.2. **Minimum three testers**, who must be actual target users.

### 5.3 Testing Results and Discussion

#### 5.3.1 Unit Testing
Same tables with Actual output and Status filled. Failures must be shown, not hidden — the guideline sample includes a failed case. Follow each table with a 2–5 line discussion.

#### 5.3.2 User Acceptance Testing
Completed forms from a minimum of three testers, followed by a discussion paragraph.

### 5.4 Experimental Results and Discussion

`[CONFIRM]` The Chapter 5 introduction in the guidelines states the chapter explains results achieved through "project coding, **experiments**, and testing", so an experimental results section belongs here. Confirm the numbering with the supervisor.

| Section | Content | Answers |
|---|---|---|
| 5.4.1 | Model performance and reference baselines | Is the measurement apparatus sound? |
| 5.4.2 | Correlation between prediction-side and explanation-side metrics | Preliminary relationship |
| 5.4.3 | Stratified analysis on the prediction-stable subset | **O2 — the central result** |
| 5.4.4 | Quadrant analysis and the hidden-risk group | O2 |
| 5.4.5 | Group comparisons: correct vs misclassified; in-distribution vs out-of-distribution | O2 |
| 5.4.6 | Complementarity experiment (ΔAUROC) | **O4** |
| 5.4.7 | Ablation: Dropout placement and number of passes | Reproducibility and validity |

### 5.5 Summary
`[TODO]`

---

## CHAPTER 6: CONCLUSION

### 6.1 Critical Evaluation

**Achievement, objective by objective**

| Objective | Achieved by | Evidence |
|---|---|---|
| O1 | Batch evaluation pipeline populating the database | 4.5, 5.3.1 |
| O2 | Stratified and quadrant analysis | 5.4.3, 5.4.4 |
| O3 | Two-role application | 4.4, 4.5.8, 5.3.1 |
| O4 | ΔAUROC experiment and UAT | 5.3.2, 5.4.6 |

**Contribution towards community and industry**
- A method for exposing confidently-predicted samples whose reasoning is unstable
- Evidence on whether explanation-stability metrics add detection power beyond prediction-side metrics
- A practical finding that Dropout placement relative to the Grad-CAM target layer determines whether explanation stability is measurable at all

**Strengths**
- Both metric families computed on the same samples in one workflow
- Confound between prediction variation and explanation variation explicitly controlled
- Metric values anchored to measured reference baselines rather than reported as bare numbers
- Reproducible: every result traceable to a stored configuration and random seed

### 6.2 Limitation

1. **Shortcut learning cannot be detected by this approach.** Shortcut learning typically produces *stable* explanations that consistently attend to the wrong region. Explanation-stability metrics therefore cannot detect it by construction. Detection would require localisation metrics and segmentation masks.

   | | Attended region correct | Attended region incorrect |
   |---|---|---|
   | **Explanation stable** | Sound | Shortcut — not detectable by this method |
   | **Explanation unstable** | Reasoning fluctuating | Risky — detectable by this method |

2. Single architecture, single explanation method, single uncertainty estimation method — generalisation not established.
3. Monte Carlo Dropout approximates epistemic uncertainty; results may differ under Deep Ensembles.
4. Grad-CAM resolution is 7×7 before upsampling, limiting fine spatial discrimination.
5. Ten-class datasets; behaviour on fine-grained or large-label-space tasks not examined.
6. UAT with the minimum number of participants; not a statistically powered usability study.

### 6.3 Recommendation

1. Add localisation metrics with segmentation masks to address the shortcut case
2. Compare Monte Carlo Dropout against Deep Ensembles and test-time augmentation
3. Extend to transformer architectures with attention-based explanation methods
4. Revisit ERS and DURS with a proper construct validity study
5. Evaluate whether the risk queue reduces reviewer error rate in a controlled task

---

## REFERENCES

APA style. Journals, articles, and books preferred; website citations discouraged. Every figure, table, and quotation cited. `[TODO — consolidate from IR and add Chapter 4–6 sources]`

---

## APPENDICES

| Appendix | Content | Status |
|---|---|---|
| **A** | PPF — Title Registration Proposal (screenshot, **all pages**) | `[TODO]` |
| **B** | Ethics Form (Fast Track / Full Track) as submitted in Moodle | `[TODO]` |
| **C** | **Log Sheets — 6 total** (3 from IR + 3 from this semester), each signed by the supervisor | ⚠ **3 outstanding** |
| **D** | Poster — A3, full colour. Header: APU logo, title, name, TP number, programme, supervisor and second marker full names with titles. Content: introduction, objectives, problem statements, methodology, application screenshots, conclusion | `[TODO]` |
| **E** | Gantt Chart — full timeline | `[TODO]` |
| **F** | Sample Code Implementation | `[TODO]` |
| **G** | Respondent Demographic Profile — interview participants **and** system testers, as two tables. Pseudonyms permitted | `[TODO]` |
| **H** | Turnitin Similarity Report — first two pages. **Maximum 20% similarity; maximum 20% AI-generated content** | `[TODO]` |

> **Appendix C is the immediate priority.** The guidelines require six signed log sheets and state the student must hold a minimum of three supervisory meetings per semester.

---
---

# PART II — IMPLEMENTATION SPECIFICATION

---

## A. Algorithms and formulas

Notation: input image $x$; class set $C$ with $|C| = 10$; number of stochastic passes $N$; softmax output of pass $t$ is $p_t \in \mathbb{R}^{|C|}$.

### A.1 Monte Carlo Dropout

Dropout remains active at inference. Each pass samples a different sub-network, giving $N$ predictive distributions. The mean predictive distribution is

$$\bar{p} = \frac{1}{N}\sum_{t=1}^{N} p_t$$

**Implementation note.** ResNet-18 contains no dropout layers by default. `Dropout2d(p)` is inserted **after `layer2` and after `layer3`**, i.e. **before** the Grad-CAM target layer. Dropout placed after `layer4` leaves the Grad-CAM input unchanged and the explanation maps become identical on every pass, making the measurement impossible.

`Dropout2d` rather than `Dropout`: convolutional activations are spatially correlated, so element-wise dropout has little effect. Channel-wise dropout disables whole feature detectors, which produces meaningful variation in the resulting attribution.

At inference, only the dropout modules are switched to training mode; batch normalisation must remain in evaluation mode, otherwise batch statistics replace the running statistics and predictions degrade.

**Efficiency.** Dropout masks are sampled independently per sample in a batch, so replicating the image $N$ times into a single batch yields $N$ distinct passes from one forward and backward call.

### A.2 Prediction-side metrics

| Metric | Formula | Column |
|---|---|---|
| Confidence (maximum softmax probability) | $\max_{c} \bar{p}_c$ | `confidence` |
| Predictive entropy | $H[\bar{p}] = -\sum_{c} \bar{p}_c \log \bar{p}_c$ | `entropy` |
| Expected entropy | $\frac{1}{N}\sum_t H[p_t]$ | *(intermediate)* |
| Mutual information (BALD) | $H[\bar{p}] - \frac{1}{N}\sum_t H[p_t]$ | `mutual_information` |
| Predictive variance | $\frac{1}{N}\sum_t \left(p_{t,\hat{c}} - \bar{p}_{\hat{c}}\right)^2$, where $\hat{c} = \arg\max_c \bar{p}_c$ | `pred_variance` |
| **Prediction agreement** | $\frac{1}{N}\max_c \sum_t \mathbb{1}\!\left[\arg\max p_t = c\right]$ | `pred_agreement` |
| Correctness | $\mathbb{1}[\hat{c} = y]$ | `correct` |

`pred_agreement` is the control variable for the stratified analysis. A value of 1.0 means every pass predicted the same class.

> `confidence` and maximum softmax probability are the same quantity. Do not create both columns.

### A.3 Grad-CAM

For target class $c$ and target layer activations $A^k$:

$$\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j} \frac{\partial y^c}{\partial A^k_{ij}}, \qquad L^c = \mathrm{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$

Each map is min-max normalised to $[0,1]$ and bilinearly upsampled to 224×224 for metric computation. Raw 7×7 maps are stored.

**Target class fixing (critical).** The target class is $\hat{c} = \arg\max_c \bar{p}_c$ — the class predicted by the *mean* distribution — and is held constant across all $N$ passes. Using the per-pass argmax would mean that when the predicted class changes, the map necessarily changes, and the resulting variation would measure prediction instability rather than explanation instability. This is the confound the whole analysis design exists to avoid.

### A.4 Explanation-side metrics

Let $\{L_1,\dots,L_N\}$ be the normalised, upsampled maps and $P = \binom{N}{2}$ the number of unordered pairs.

| Metric | Definition | Column |
|---|---|---|
| Mean pairwise correlation | $\frac{1}{P}\sum_{a<b} \rho_{\text{Pearson}}(L_a, L_b)$ over flattened maps | `cam_corr_mean` |
| Correlation standard deviation | standard deviation of the same pairwise values | `cam_corr_std` |
| Mean pairwise IoU | binarise each map at its own 80th percentile giving $M_t$; then $\frac{1}{P}\sum_{a<b}\frac{\lvert M_a \cap M_b\rvert}{\lvert M_a \cup M_b\rvert}$ | `cam_iou_mean` |
| Top-k overlap | $T_t$ = indices of the top 10% of pixels; $\frac{1}{P}\sum_{a<b}\frac{\lvert T_a \cap T_b\rvert}{k}$ | `topk_overlap` |
| Variability map | pixel-wise standard deviation across the $N$ maps | PNG artefact |

Percentile-based binarisation rather than a fixed threshold, because the absolute scale of Grad-CAM values differs from image to image.

### A.5 Reference baselines

Without these, a value such as "mean IoU 0.42" cannot be interpreted.

| ID | Baseline | Method | Expected | Role |
|---|---|---|---|---|
| **B1** | Upper bound | Dropout disabled; two Grad-CAMs of the same image | 1.000 | Implementation check — a value below 1.000 indicates a bug in the Grad-CAM path or seed handling |
| **B2** | Lower bound | 1,000 pairs of Gaussian-smoothed random heatmaps | — | The value corresponding to no relationship |
| **B3** | Cross-image reference | 1,000 pairs of Grad-CAMs from *different* images of the same class | — | The value corresponding to unrelated inputs |

B3 also supplies the default `TH_IOU` for the risk flag rule, so the threshold is grounded in a measured quantity rather than chosen arbitrarily.

### A.6 Analysis procedures

**A.6.1 Stratified analysis (the central result)**

1. Fix the Grad-CAM target class as in A.3.
2. Select the subset where `pred_agreement == 1.0`.
3. Within that subset, bin by `confidence`: [0.90, 0.95), [0.95, 0.99), [0.99, 1.00].
4. Report the distribution of `cam_corr_mean` and `cam_iou_mean` within each bin, against baselines B2 and B3.

The claim is supported if explanation stability spans a wide range within a bin — that is, prediction-side metrics being effectively identical does not imply explanation stability is identical.

**A.6.2 Quadrant analysis**

Axes: prediction uncertainty (`entropy`) and explanation instability ($1 - $ `cam_iou_mean`), split at the median.

| Quadrant | Condition | `risk_group` |
|---|---|---|
| Q1 | low / low | `stable` |
| Q2 | high / high | `unstable_both` |
| Q3 | high / low | `pred_unstable_only` |
| Q4 | **low / high** | `hidden_risk` |

Report misclassification rate and out-of-distribution proportion per quadrant, and test whether Q4 exceeds Q1.

**A.6.3 Complementarity experiment (ΔAUROC)**

Tasks: misclassification detection (positive = `correct == 0`); out-of-distribution detection (positive = `dataset_type != 'id'`).

| Model | Features |
|---|---|
| M1 | `confidence`, `entropy`, `pred_variance` |
| M2 | M1 features + `cam_corr_mean`, `cam_iou_mean`, `topk_overlap` |

Logistic regression with standardised features, stratified 5-fold cross-validation. Report AUROC mean ± standard deviation for each, and **ΔAUROC = AUROC(M2) − AUROC(M1)** with a bootstrap confidence interval.

> A result of ΔAUROC ≈ 0 is a valid finding and must be reported as such: explanation-side metrics would then largely duplicate prediction-side information. Prepare this interpretation in advance.

**A.6.4 Statistical reporting rules**

At these sample sizes, p-values are uninformative on their own.

| Situation | Report |
|---|---|
| Correlation | Spearman $\rho$ with a bootstrap 95% confidence interval; p-value in parentheses only |
| Two-group comparison | Medians, Mann–Whitney U, and **Cliff's delta** |
| Multi-group (severity) | Kruskal–Wallis with **Bonferroni correction stated explicitly** |
| Detection performance | AUROC, AUPR, FPR@95TPR, each with bootstrap confidence intervals |

Cliff's delta interpretation: $\lvert\delta\rvert < 0.147$ negligible; $< 0.33$ small; $< 0.474$ medium; otherwise large.

Avoid: "a significant correlation was found (p < 0.001)".
Use: "$\rho = 0.31$, 95% CI [0.28, 0.34] — a moderate negative association".

---

## B. System specification

### B.1 Functional requirements

| ID | Role | Requirement | Priority |
|---|---|---|---|
| FR-01 | Engineer | Register and select trained model checkpoints | Must |
| FR-02 | Engineer | Configure analysis parameters (passes, dropout rate, thresholds) | Must |
| FR-03 | Engineer | Execute a batch analysis from within the application, with progress feedback | Must |
| FR-04 | Engineer | View summary statistics and distribution plots | Must |
| FR-05 | Engineer | View, sort, and filter the metrics table | Must |
| FR-06 | Engineer | View reference baseline values alongside metric values | Must |
| FR-07 | Engineer | Export the metrics table as CSV | Must |
| FR-08 | Reviewer | View a prioritised queue of flagged samples | Must |
| FR-09 | Reviewer | View full detail for one image, including repeated Grad-CAMs and the variability map | Must |
| FR-10 | Reviewer | Record a decision and comment against an image | Must |
| FR-11 | Reviewer | Upload an arbitrary image and receive an analysis | Must |
| FR-12 | Reviewer | Export review decisions | Should |
| FR-13 | Both | Switch role, changing the available pages | Must |
| FR-14 | Engineer | Compare two analysis runs side by side | Could |

**Non-functional**: analysis under 100 ms per image at N = 30; table render under 3 s for 10,000 rows; every result traceable to a stored configuration and seed.

### B.2 Design diagram content

**Use case diagram** — Actors: ML Engineer, Reviewer. Engineer use cases: Manage Models, Configure Analysis, Run Batch Analysis, View Statistics, Browse Metrics, Export Metrics. Reviewer use cases: View Risk Queue, Inspect Image, Record Review, Analyse Uploaded Image, Export Reviews. Shared: Select Role. `Run Batch Analysis` «includes» `Compute Metrics`; `Inspect Image` «includes» `Load Explanation Artefacts`.

**Activity diagram — Run Batch Analysis**: Select model → Select dataset → Set parameters → Validate configuration → *(loop per image)* Load image → Replicate N times → Stochastic forward pass → Compute prediction metrics → Determine target class → Generate N Grad-CAMs → Compute stability metrics → Persist to database → *(end loop)* → Compute baselines → Assign risk groups → Display summary.

**Sequence diagram — Analyse Uploaded Image**: Reviewer → UploadPage → AnalysisService → ModelRegistry → MCDropoutInference → GradCAMGenerator → MetricCalculator → Repository → UploadPage → Reviewer.

**Class diagram — main classes**: `ModelRegistry`, `MCDropoutResNet`, `InferenceEngine`, `GradCAMGenerator`, `PredictionMetrics`, `ExplanationMetrics`, `BaselineCalculator`, `RiskClassifier`, `AnalysisService`, `Repository`, `ReviewService`, and the entity classes mirroring the tables in Section C.

### B.3 Interface specification

| Page | Role | Key elements |
|---|---|---|
| Home | Both | Role selector, active model, recent runs, summary tiles |
| Run Analysis | Engineer | Model dropdown, dataset selector, parameter form, Run button, progress bar, log |
| Metrics Table | Engineer | Filter sidebar, sortable table, row selection, Export button |
| Statistics | Engineer | Correlation heatmap, confidence vs IoU scatter, stratified box plot, quadrant scatter, baseline panel |
| Risk Queue | Reviewer | Threshold sliders, ranked list with thumbnails, jump to detail |
| Image Detail | Reviewer | Input image, labels, metric panel with baseline comparison, grid of six Grad-CAMs, mean map, variability map, interpretation text, decision form |
| Upload and Analyse | Reviewer | File uploader, model selector, Analyse button, result panel identical to Image Detail |
| Export | Both | Metrics CSV, reviews CSV, figure bundle |

**Risk flag rule** (a threshold rule, not a proposed metric):

```
if confidence > TH_CONF and cam_iou_mean < TH_IOU:
    risk_flag = "REVIEW PRIORITY"
```

Defaults: `TH_CONF = 0.95`; `TH_IOU` = baseline B3 value. Both adjustable in the interface.

---

## C. Database design

SQLite. Nine tables. This schema is the Entity Relationship Diagram for Section 4.3.

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
    dataset_type        TEXT NOT NULL CHECK(dataset_type IN ('id','near_ood','far_ood','corrupted','uploaded')),
    corruption_type     TEXT,
    corruption_severity INTEGER,
    true_label          TEXT
);

CREATE TABLE predictions (
    prediction_id      INTEGER PRIMARY KEY,
    image_id           INTEGER NOT NULL UNIQUE REFERENCES images(image_id),
    pred_label         TEXT NOT NULL,
    correct            INTEGER,
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
    risk_group TEXT NOT NULL CHECK(risk_group IN ('stable','unstable_both','pred_unstable_only','hidden_risk')),
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
    baseline_type TEXT NOT NULL CHECK(baseline_type IN ('upper','lower','cross_image')),
    metric_name   TEXT NOT NULL,
    value         REAL NOT NULL
);

CREATE INDEX idx_images_run  ON images(run_id);
CREATE INDEX idx_pred_conf   ON predictions(confidence);
CREATE INDEX idx_expl_iou    ON explanations(cam_iou_mean);
```

**CSV export** remains a feature (FR-07) so that the "metrics CSV" deliverable is preserved, but the database is the system of record.

---

## D. Testing specification

### D.1 Unit test cases

Follow the guideline table format: TC-No, Input, Expected output, Actual output, Status.

**Program: `metrics_prediction.py`**

| TC | Input | Expected output |
|---|---|---|
| TC1 | One-hot distribution [1,0,…,0] | entropy = 0.0 |
| TC2 | Uniform distribution over 10 classes | entropy = ln(10) ≈ 2.3026 |
| TC3 | Identical distributions across all N passes | pred_variance = 0.0 |
| TC4 | Identical distributions across all N passes | mutual_information = 0.0 |
| TC5 | All N passes predict class 3 | pred_agreement = 1.0 |
| TC6 | 15 of 30 passes predict class 3, 15 predict class 7 | pred_agreement = 0.5 |

**Program: `metrics_explanation.py`**

| TC | Input | Expected output |
|---|---|---|
| TC7 | Two identical maps | correlation = 1.0 |
| TC8 | Two identical maps | IoU = 1.0 |
| TC9 | Map and its negation | IoU = 0.0 |
| TC10 | Constant map | Handled without division-by-zero error |
| TC11 | N = 30 maps | 435 pairwise values computed |

**Program: `resnet_dropout.py`**

| TC | Input | Expected output |
|---|---|---|
| TC12 | Built model | Exactly two `Dropout2d` modules present |
| TC13 | After `enable_mc_dropout()` | All `Dropout2d` in training mode |
| TC14 | After `enable_mc_dropout()` | All `BatchNorm2d` in evaluation mode |
| TC15 | Forward pass, batch 30 | Output shape (30, 10) |

**Program: `gradcam_generator.py`** — the two verification tests

| TC | Input | Expected output |
|---|---|---|
| **TC16** | Dropout **enabled**, same image, 10 maps | Mean pairwise correlation **< 0.99** (maps do vary) |
| **TC17** | Dropout **disabled**, same image, 2 maps | Correlation **= 1.000** (fully reproducible) |
| TC18 | N maps generated | All generated against the same target class |

> **TC16 and TC17 must pass before any other development proceeds.** If TC16 fails, the Dropout placement is wrong and no downstream result is meaningful. If TC17 fails, there is a bug in the Grad-CAM path or seed handling.

**Program: `streamlit_app.py`**

| TC | Input | Expected output |
|---|---|---|
| TC19 | Role set to Reviewer | Engineer-only pages hidden |
| TC20 | Run with a non-existent checkpoint | Error message shown, no crash |
| TC21 | Filter confidence > 0.95 | Only matching rows displayed |
| TC22 | Upload a non-image file | Rejected with a message |
| TC23 | Submit a review decision | Row persisted in `reviews` and visible on reload |

### D.2 User Acceptance Testing form

**Tester demographic profile**: Name, Age, **Role in the system** (ML Engineer / Reviewer).

**User interface criteria** (1 = Strongly disagree … 5 = Strongly agree)

| | Criterion |
|---|---|
| I | The layout of the metrics table is clear and easy to read. |
| II | The Grad-CAM visualisations are clear enough to interpret. |
| III | The navigation between pages is straightforward. |
| IV | The interface presents the reliability information without feeling cluttered. |

**Functionality criteria** (Yes / No)

| | Criterion |
|---|---|
| I | The system runs an analysis without error. |
| II | The risk queue helps identify samples that need attention. |
| III | Showing the metrics separately is more useful than a single combined score. |
| IV | The reference baselines make the metric values easier to interpret. |
| V | A review decision can be recorded and retrieved successfully. |

Plus tester comment box and signature.

> Criterion III deliberately tests the interview theme from IR Table 5 Q10, closing the loop from data gathering to evaluation.

---

## E. Project structure

```
fyp-reliability-dashboard/
├── app/
│   ├── streamlit_app.py
│   └── pages/
│       ├── 1_Run_Analysis.py      5_Image_Detail.py
│       ├── 2_Metrics_Table.py     6_Upload_Analyse.py
│       ├── 3_Statistics.py        7_Export.py
│       └── 4_Risk_Queue.py
├── configs/config.yaml
├── data/{imagenette2-320, imagewoof2-320, dtd}/
├── db/{schema.sql, reliability.db}
├── models/{resnet_dropout.py, checkpoints/}
├── src/
│   ├── dataset_loader.py          metrics_prediction.py
│   ├── corruption_generator.py    metrics_explanation.py
│   ├── train_model.py             baselines.py
│   ├── inference_mc_dropout.py    batch_evaluation.py
│   ├── gradcam_generator.py       analysis.py
│   ├── repository.py              stratified_analysis.py
│   ├── analysis_service.py        complementarity.py
│   └── utils.py                   ablation.py
├── tests/{test_metrics_prediction.py, test_metrics_explanation.py,
│          test_model_dropout.py, test_gradcam.py, test_integration.py}
├── outputs/{runs/, gradcam/{raw,png}/, figures/, ablation/, reports/}
├── notebooks/
├── requirements.txt
├── README.md
└── run_pipeline.py
```

### config.yaml

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
risk_flag:
  confidence_threshold: 0.95
  iou_threshold: null   # set from baseline B3
database:
  path: db/reliability.db
```

### Experimental scale

| Dataset | Type | Images |
|---|---|---|
| Imagenette validation | id | 3,925 |
| Imagewoof | near_ood | 1,500 |
| DTD test | far_ood | 1,500 |
| Imagenette-C (4 types × 5 severities × 500) | corrupted | 10,000 |
| **Total** | | **≈16,925** |

Estimated wall-clock time at N = 30 with batch replication on an RTX 4070 SUPER: fine-tuning 20–30 min; full analysis 15–25 min; ablations 1–2 h.

### Storage

Raw Grad-CAM maps are 7×7 float16 — approximately 3 KB per image, so all can be stored (~50 MB total). Only six representative maps plus the mean and variability maps are written as PNG; writing all N per image would produce over 500,000 files.

### requirements.txt

```
torch==2.3.1              matplotlib==3.9.0
torchvision==0.18.1       plotly==5.22.0
pytorch-grad-cam==1.5.0   seaborn==0.13.2
numpy==1.26.4             streamlit==1.36.0
pandas==2.2.2             tqdm==4.66.4
scipy==1.13.1             pyyaml==6.0.1
scikit-learn==1.5.0       joblib==1.4.2
Pillow==10.3.0            pytest==8.2.2
opencv-python==4.10.0.84
imagecorruptions==1.1.2
```

NumPy is pinned below 2.0 for OpenCV and pytorch-grad-cam compatibility.

---

## F. Build order

Each phase has an exit condition. Do not proceed until it is met.

| Phase | Work | Exit condition |
|---|---|---|
| **0. Verification** | Build `resnet_dropout.py`; implement `enable_mc_dropout()`; generate 10 maps with dropout on and 2 with dropout off | **TC16 and TC17 pass.** Without this, nothing downstream is valid |
| **1. Model** | Fine-tune ResNet-18 on Imagenette | Validation accuracy ≥ 95%; checkpoint registered in `models` |
| **2. Metrics** | `metrics_prediction.py`, `metrics_explanation.py`, `baselines.py` | TC1–TC11 pass; B1 returns 1.000 |
| **3. Pipeline** | `inference_mc_dropout.py`, `gradcam_generator.py`, `batch_evaluation.py`, `repository.py` | 100 images processed end to end; database populated with no nulls in required columns |
| **4. Analysis** | `stratified_analysis.py`, `complementarity.py`, `analysis.py` | ΔAUROC computed; stratified figure produced |
| **5. Application** | All eight pages, both roles | TC19–TC23 pass |
| **6. Extension** | Out-of-distribution, corruption, ablations | Figures produced |
| **7. Evaluation** | UAT with three or more target users | Three signed forms collected |
| **8. Documentation** | Chapters 4–6, poster, appendices | Turnitin under 20% for both similarity and AI-generated content |

---
---

# PART III — COMPLIANCE AND OPEN DECISIONS

---

## G. Compliance checklist

| # | Requirement | Source | Status |
|---|---|---|---|
| 1 | Title ≤ 15 words | Guidelines p.2 | ✔ 13 words |
| 2 | Title contains no special characters | Guidelines p.2 | ⚠ hyphen — see front matter |
| 3 | Title contains an action verb | Guidelines p.2 | ⚠ see front matter |
| 4 | Title unchanged since IR | Guidelines p.2 | ✔ |
| 5 | Abstract ≤ 200 words with SDG mapping and ≤ 6 keywords | Guidelines p.3 | Draft prepared |
| 6 | Cover, acknowledgement, abstract, TOC, list of figures, list of tables each on a new page | Guidelines p.4 | `[TODO]` |
| 7 | Objectives: 3–4, measurable, begin with "To" | Guidelines p.5 | ✔ four |
| 8 | Scope states what will and will not be done | Guidelines p.5 | ✔ |
| 9 | Potential benefit covers tangible, intangible, and target user | Guidelines p.5 | ✔ |
| 10 | Chapter 2 covers domain, similar works, technical research | Guidelines p.6 | Structure ready |
| 11 | Chapter 3 covers methodology, data gathering, analysis | Guidelines p.7–8 | Structure ready |
| 12 | Interview participants ≥ 3, actual target audience, not friends | Guidelines p.7 | Carried from IR |
| 13 | Respondent demographic profile in appendix | Guidelines p.8, p.27 | `[TODO]` |
| 14 | Chapter 4 includes design, database design, interface design, implementation, sample codes | Guidelines p.9–11 | Structure ready |
| 15 | Database design included (project has a database) | Guidelines p.9 | ✔ schema ready |
| 16 | Chapter 5 has both blank test plan and filled results | Guidelines p.13–18 | Structure ready |
| 17 | UAT testers ≥ 3, with demographic profile, criteria tables, comment, signature | Guidelines p.15, p.18 | Form drafted |
| 18 | Chapter 6 covers critical evaluation, limitation, recommendation | Guidelines p.20 | Structure ready |
| 19 | Each chapter starts on a new page | Guidelines p.20 | `[TODO]` |
| 20 | References in APA style, journals and books preferred | Guidelines p.21 | `[TODO]` |
| 21 | Appendix A: PPF, all pages | Guidelines p.22 | `[TODO]` |
| 22 | Appendix B: ethics form | Guidelines p.22 | `[TODO]` |
| 23 | **Appendix C: 6 signed log sheets, min 3 meetings per semester** | Guidelines p.23 | ⚠ **3 outstanding** |
| 24 | Appendix D: A3 full-colour poster with required header fields | Guidelines p.24 | `[TODO]` |
| 25 | Appendix E: Gantt chart | Guidelines p.25 | `[TODO]` |
| 26 | Appendix F: sample code | Guidelines p.26 | `[TODO]` |
| 27 | Appendix G: respondent demographic profile and system testers | Guidelines p.27 | `[TODO]` |
| 28 | Appendix H: Turnitin first two pages, similarity ≤ 20% | Guidelines p.28 | `[TODO]` |
| 29 | AI-generated content ≤ 20% | Semester 2 briefing | ⚠ **write Chapters 4–6 in own words** |

## H. Open decisions for the supervisor

| # | Question | Why it matters |
|---|---|---|
| 1 | Are the four revised objectives acceptable, in particular removing ERS and DURS as objectives? | Determines Chapters 1, 5, and 6 |
| 2 | Is the dataset change from CIFAR-10 to Imagenette acceptable? | At 32×32 the Grad-CAM map is 4×4, giving only four possible IoU values — unusable for the analysis |
| 3 | Is declaring shortcut learning out of scope acceptable, with the reasoning in Limitations? | It cannot be detected by stability metrics by construction |
| 4 | Structured design (context diagram, DFD, ERD) or object-oriented design (use case, class) for Chapter 4? | The project has a relational database but object-oriented code |
| 5 | Is a local SQLite database sufficient for the CSAI application requirement? | Determines whether a server-based database is needed |
| 6 | Is 5.4 Experimental Results an acceptable addition to Chapter 5? | The research findings need a home; the Chapter 5 introduction refers to experiments |
| 7 | Are the two roles an acceptable basis for UAT participant selection, and are there recommended participants? | UAT needs verified target users and takes time to arrange |
| 8 | The title contains a hyphen and no action verb, both of which the guidelines discourage — but titles are locked at IR. Leave as is? | Only the supervisor can authorise a revision |
| 9 | How should the three outstanding log sheets be handled? | Appendix C requires six |

## I. Version history

| Version | Date | Change |
|---|---|---|
| 0.1 | [Date] | Initial consolidation: report skeleton aligned to guidelines, implementation specification, compliance checklist |
