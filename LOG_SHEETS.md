# Supervision Log Sheets

Appendix C requires **six signed log sheets**: three from the IR semester and three from this semester. Three are outstanding.

This file prepares the content for those three meetings so each session produces a complete, signable sheet.

---

## 1. Requirements

From the guidelines (p.23) and the log sheet sample:

- Minimum **three supervisory meetings per semester**, six across the whole FYP
- Each meeting is designed for **more than 15 minutes**
- **Items for discussion are noted by the student BEFORE the meeting** — this is the part that gets prepared in advance, and the reason this file exists
- Record of discussion is noted **during** the meeting
- Action list is noted for the **next** meeting
- The sheet must be **signed by the supervisor**
- A copy is emailed to the supervisor and to the administrator after each session
- Sheets are submitted with the report and uploaded to Moodle

Fields on each sheet: student name, date, meeting number, project title, intake code, supervisor name and signature, items for discussion, record of discussion, action list.

> The Semester 2 briefing noted that a single meeting may be split across the required log sheets if the supervisor agrees. If the schedule is tight, ask — but ask, do not assume.

---

## 2. Status

| # | Semester | Date | Status |
|---|---|---|---|
| 1 | IR | | ✔ Complete |
| 2 | IR | | ✔ Complete |
| 3 | IR | | ✔ Complete |
| **4** | FYP | [next week] | ⚠ Scheduled |
| **5** | FYP | | ⚠ Not scheduled |
| **6** | FYP | | ⚠ Not scheduled |

Fill in the IR dates from the completed sheets so the numbering is continuous.

---

## 3. Meeting 4 — direction and approvals

**Purpose**: get decisions on the outstanding questions so development can proceed without rework.

### Items for discussion (prepare and send before the meeting)

1. IR feedback from supervisor and second marker
2. Approval of the four revised objectives, in particular removing ERS and DURS as project objectives
3. Approval of the dataset change from CIFAR-10 to Imagenette, on the grounds that a 4×4 final feature map yields only four discrete IoU values
4. Approval of declaring shortcut learning out of scope, with the reasoning stated in Limitations
5. Which design diagram set Chapter 4 requires: structured or object-oriented
6. Whether a local SQLite database satisfies the CSAI application requirement
7. Whether Section 5.4 Experimental Results is an acceptable addition to Chapter 5
8. The project title: it contains a hyphen and no action verb, both discouraged by the guidelines, but titles are locked at IR
9. How to handle the three outstanding log sheets
10. Recommendations for UAT participants matching the two defined roles

### Record of discussion
*(complete during the meeting — one line per item above)*

### Action list for the next meeting
- Apply the approved changes to Chapters 1–3
- Complete Phase 0 verification (TC16, TC17) and Phase 1 model training
- Produce the agreed design diagrams
- Confirm UAT participants

---

## 4. Meeting 5 — implementation review

**Purpose**: demonstrate working software. The Semester 2 briefing was explicit that the supervisor must see the application before the presentation, so that they cannot say at the panel that they never saw it.

### Items for discussion (draft — revise nearer the date)

1. Demonstration of the current application, both user roles
2. Verification results: TC16 and TC17, and the three reference baselines
3. Model training result and the validation accuracy achieved
4. Review of the design diagrams produced since Meeting 4
5. Database schema review against the ERD requirement
6. Preliminary analysis results — stratified analysis and ΔAUROC
7. Whether to attempt the corruption and out-of-distribution extensions, or consolidate the core
8. UAT scheduling and confirmation of participants
9. Review of the drafted Chapters 4 and 5 structure

### Record of discussion
*(complete during the meeting)*

### Action list for the next meeting
- Address feedback on the application
- Complete remaining analysis
- Run UAT
- Draft Chapters 4–6

---

## 5. Meeting 6 — final documentation

**Purpose**: sign-off before submission.

### Items for discussion (draft)

1. Demonstration of the completed application
2. UAT results and the discussion written from them
3. Review of the full draft report, Chapters 1–6
4. Turnitin report: similarity and AI-generated content, both under 20%
5. Poster draft (Appendix D)
6. Appendix completeness: PPF, ethics form, log sheets, poster, Gantt chart, sample code, demographic profiles, Turnitin
7. Presentation arrangements and what the panel will expect
8. Confirmation that all four objectives can be evidenced

### Record of discussion
*(complete during the meeting)*

### Action list
- Final corrections
- Submission
- Presentation preparation

---

## 6. Practical notes

**Send the items for discussion in advance.** The briefing was blunt about students who arrive with nothing prepared. A short list sent the day before also demonstrates that the item was genuinely noted beforehand, as the log sheet format requires.

**Complete the sheet during or immediately after the meeting.** Reconstructing a discussion from memory a month later produces a thin record, and the supervisor may decline to sign it.

**Get the signature at the time.** Chasing signatures at submission is the standard way this goes wrong.

**Email a copy after each session** to the supervisor and the administrator, as the guidelines require, and keep the sent message as evidence.

**Number the sheets consecutively 1–6** and file them in order in Appendix C.

---

## 7. Checklist before submission

- [ ] Six sheets, numbered 1 to 6
- [ ] Every sheet dated
- [ ] Every sheet signed by the supervisor
- [ ] Project title and intake code identical on all sheets
- [ ] Items for discussion, record of discussion, and action list completed on each
- [ ] Scanned and inserted into Appendix C
- [ ] Uploaded to the required Moodle link
