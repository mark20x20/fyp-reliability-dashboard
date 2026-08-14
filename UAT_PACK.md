# UAT Pack

Everything needed to run User Acceptance Testing, from recruitment to the completed forms that go into Chapter 5.3.2 and Appendix G.

**Start recruiting now.** The application does not need to be finished. Securing participants takes weeks; running the sessions takes an afternoon.

---

## 1. Requirements

From the FYP Report Writing Guidelines (p.15, p.18) and the Semester 2 briefing:

- **Minimum three testers**
- Each must be an **actual target user** — the guidelines state explicitly that data gathering must not be conducted with friends
- Identity and role must be verifiable; the supervisor may ask how you know a participant holds the role they claim
- Each completed form needs a **tester signature**
- Demographic profile of all testers goes in **Appendix G**, in a table separate from the interview respondents
- Pseudonyms are permitted in the report to protect confidentiality

---

## 2. Who qualifies

Two roles are defined in the application, so participants should map onto them.

### Role A — Machine learning engineer / researcher

Someone who trains or evaluates image classification models.

| Source | Notes |
|---|---|
| APU lecturers in machine learning, computer vision, or data science | Strongest option — role is self-evidently verifiable |
| Postgraduate students working on vision or deep learning | Good fit; usually willing |
| Research assistants in the school | Good fit |
| Final-year CSAI students whose own FYP involves model training | Acceptable, but state the basis for their expertise in Appendix G |

### Role B — Reviewer

Someone who interprets model output before acting on it, without necessarily building models.

| Source | Notes |
|---|---|
| **Interview participants from the IR** | **Best option.** Already identified as target users, already consented, and it creates continuity from Chapter 3 to Chapter 5 |
| Practitioners in a domain that uses image classification | Strong if reachable |
| Data annotators or QA staff | Reasonable fit |

> Aim for at least one from each role, and reuse IR interview participants wherever possible. That link is worth stating explicitly in Chapter 5.

---

## 3. Recruitment message

Send now, before the application is complete.

**For a lecturer or researcher (email)**

> Subject: Request to participate in FYP user testing (30 minutes)
>
> Dear [Name],
>
> I am a final-year CSAI student working on a project that analyses the reliability of image classification models by comparing prediction uncertainty against the stability of Grad-CAM explanations across stochastic inference passes.
>
> I am looking for a small number of participants to test the application I have built. The session takes about 30 minutes: I would ask you to work through a few short tasks in the interface and then complete a short evaluation form. It can be in person or online, whichever is easier.
>
> I am aiming to run these sessions in [week/dates]. Would you be willing to take part? I am happy to work around your schedule.
>
> My supervisor for this project is [Supervisor's Name].
>
> Thank you for considering it.
>
> [Your Name], TP073279

**For a returning interview participant (shorter)**

> Hi [Name], thank you again for the interview earlier this year — the points you raised shaped how I built the interface. I have now developed the application and am looking for a few people to test it. It would take about 30 minutes, either in person or online, sometime in [week]. Would you be willing to take part?

**Follow-up if no reply after five days**: one short message, then move to the next candidate. Do not rely on a single person.

---

## 4. Before the session

- [ ] Application runs end to end without crashing on the demo dataset
- [ ] Database pre-populated with a completed analysis run — do not make the participant wait for a batch job
- [ ] At least one clear hidden-risk example ready in the risk queue
- [ ] Blank UAT forms printed, or an editable copy ready
- [ ] Screen recording or notes, if the participant consents
- [ ] A fallback screenshot walkthrough in case something breaks

---

## 5. Session script (~30 minutes)

**Introduction (3 min)**

> Thank you for your time. This project looks at whether a confident prediction from an image classifier can be trusted, by checking whether the model looks at the same region each time it makes that prediction. I will ask you to try a few tasks. There are no wrong answers — I am testing the system, not you. Please think aloud where you can. You can stop at any point, and your name will be replaced with a pseudonym in the report.

**Consent (1 min)** — verbal is sufficient if the ethics form covers it. Confirm: participation is voluntary, may withdraw at any time, results reported anonymously.

**Tasks (20 min)** — give the goal, not the steps. Observing where they hesitate is the point.

| # | Task | Role | Observe |
|---|---|---|---|
| 1 | Find out how many images in this run were misclassified. | Both | Do they find the summary, or start scanning the table? |
| 2 | Find an image where the model was very confident but the explanation was unstable. | Reviewer | Do they discover the risk queue, or filter manually? |
| 3 | Open that image and tell me whether you would trust the prediction, and why. | Reviewer | Do the repeated Grad-CAMs and variability map communicate anything? |
| 4 | Record a decision on that image and add a comment. | Reviewer | Is the decision control discoverable? |
| 5 | Change the confidence threshold for the risk flag and describe what changes. | Engineer | Is the effect of the threshold legible? |
| 6 | Find the relationship between confidence and explanation stability. | Engineer | Do they reach the statistics page? Is the stratified plot readable? |
| 7 | Upload this image and interpret the result. | Reviewer | Does the upload path work on an unseen input? |
| 8 | Export the results. | Engineer | Straightforward? |

**Debrief (5 min)** — three open questions before handing over the form:

1. Was there anything you expected the system to do that it did not?
2. Was showing the metrics separately more useful than a single combined score would have been, or less?
3. Would the baseline values change how you read a number like "IoU 0.42"?

> Question 2 directly tests the IR interview theme (Table 5, Q10). Record the answers — they belong in the Chapter 5.3.2 discussion.

**Form (5 min)** — leave the participant to complete it. Do not read it over their shoulder.

---

## 6. UAT form

Reproduce this exactly; the field layout follows the guideline sample on p.15.

---

### User Acceptance Testing

**Tester demographic profile**

| | |
|---|---|
| Name | |
| Age | |
| Role in the system | |

*Rating scale: 1 = Strongly disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly agree*

**User interface criteria**

| | Criterion | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| I | The layout of the metrics table is clear and easy to read. | | | | | |
| II | The Grad-CAM visualisations are clear enough to interpret. | | | | | |
| III | Navigation between pages is straightforward. | | | | | |
| IV | The interface presents the reliability information without feeling cluttered. | | | | | |
| V | The single-image detail view shows the information I needed in one place. | | | | | |

**Functionality criteria**

| | Criterion | Yes | No |
|---|---|---|---|
| I | The system runs an analysis without error. | | |
| II | The risk queue helps identify samples that need attention. | | |
| III | Showing the metrics separately is more useful than a single combined score. | | |
| IV | The reference baselines make the metric values easier to interpret. | | |
| V | A review decision can be recorded and retrieved successfully. | | |
| VI | The uploaded-image analysis returns a usable result. | | |

**Tester comment**

<br><br><br>

**Tester's signature** ______________________  **Date** ____________

---

## 7. After the sessions

- [ ] Scan or photograph the signed forms — these go into the report as images
- [ ] Transfer ratings into a summary table for Chapter 5.3.2
- [ ] Write the discussion: what scored well, what scored poorly, what you would change
- [ ] Fill in the Appendix G table below
- [ ] Log any defects found into `PROGRESS.md` section 7

**Summary table for Chapter 5.3.2**

| Criterion | T1 | T2 | T3 | Mean |
|---|---|---|---|---|
| UI-I | | | | |
| UI-II | | | | |
| … | | | | |

Report the low scores as well as the high ones. A form where everything is 5 reads as untrustworthy, and Chapter 6 needs material for the limitations section.

**Appendix G — System testers**

| Participant | Gender | Age | Occupation |
|---|---|---|---|
| [Pseudonym] | | | |
| [Pseudonym] | | | |
| [Pseudonym] | | | |

*Note: participant names may be pseudonyms rather than actual names, to protect confidentiality.*

Keep this table separate from the interview respondents' table, and make each person's role in the system evident from the occupation column.

---

## 8. Timeline

| When | Action |
|---|---|
| **Now** | Send recruitment messages to five or six candidates |
| +1 week | Follow up; confirm three, hold one in reserve |
| Phase 5 complete | Fix session dates |
| Session week | Run sessions, collect signed forms |
| +2 days | Write up 5.3.2 and Appendix G while the sessions are fresh |

Five or six approaches for three confirmed participants is the realistic ratio. People decline, and people cancel.
