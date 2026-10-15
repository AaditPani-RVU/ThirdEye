# STEPWISE — Requirements Specification

**Project:** STEPWISE — Spatio-Temporal Error Perception With Instruction-Grounded Step Evaluation  
**Version:** 1.0 · **Date:** 2026-09-18 · **Status:** Draft for team review

**Source of truth:** `STEPWISE_Project_Plan.md` (plan of 2026-09-18). Every requirement below traces to a plan section; where this document and the plan disagree, the plan wins and this document is a bug.

---

## 1. Purpose and scope

### 1.1 Purpose

This document turns the STEPWISE project plan into a numbered, testable set of requirements. It exists so that (a) each module owner knows exactly what "done" means for their module, (b) the evaluation in week 11 has pre-agreed pass/fail criteria rather than post-hoc ones, and (c) procurement and environment setup in week 1 is a checklist rather than a scramble.

### 1.2 Product scope

STEPWISE observes a person performing a physical procedure through one or more fixed cameras and raises real-time alerts when a step is missed, performed out of order, or performed incorrectly. The procedure itself is not hard-coded: an LLM compiles a written manual or rulebook into a machine-readable **procedure spec**, and a deterministic checker compares observed world state against that spec.

Two procedures are in scope:

| Procedure | Spec form | Role |
| --- | --- | --- |
| LEGO assembly (Model A, Model B) | Dependency graph (DAG) | Must-have demo and primary training domain |
| Jenga (game rules) | Constraint set, no fixed order | Generality result and live adversarial demo |

### 1.3 Document scope

Covered: functional, non-functional, data, hardware, software, interface, evaluation and milestone-gate requirements, plus the open items that block requirements from being finalised.

Not covered: implementation design (that is plan §2 and §4), work allocation (plan §9), and the report/poster content (plan §11).

### 1.4 Definitions

| Term | Meaning |
| --- | --- |
| **Procedure spec** | JSON document compiled from a manual/rulebook; either DAG form (§4.2 of plan) or constraint form (§4.3) |
| **World state** | Current physical configuration as tracked by the system: LEGO stud grid, or Jenga 18×3 lattice |
| **Event** | A discrete observed change (`ADD`, `REMOVE`, `STEP_START`, `STEP_END`) emitted by the state tracker or the temporal model |
| **Checker** | Deterministic engine that evaluates spec preconditions against events and raises alerts |
| **Hard dependency** | Physically necessary precondition (support). A violation indicates *our* perception failed, never a user error |
| **Soft dependency** | Order stated by the manual. A violation is a genuine out-of-order user error |
| **Occlusion window** | Interval during which a hand covers the workspace and the state tracker is frozen |
| **Tier 1 / 2 / 3** | The generality ladder of plan §1.2: unseen assembly / unseen domain / unseen procedure topology |

### 1.5 Identifier and priority scheme

Requirement IDs are `<CLASS>-<MODULE>-<n>` and are stable — never renumber, mark superseded instead.

| Class | Meaning |
| --- | --- |
| `FR` | Functional requirement |
| `NFR` | Non-functional requirement |
| `DR` | Data requirement |
| `HW` | Hardware requirement |
| `SW` | Software / environment requirement |
| `IF` | Interface / contract requirement |
| `EV` | Evaluation acceptance criterion |
| `MG` | Milestone gate |

Priority uses MoSCoW, aligned to plan §6:

| Priority | Meaning |
| --- | --- |
| **M** | Must have — the minimum viable project. Failure here fails the project |
| **S1**–**S6** | Should have, in the plan's priority order. Delivered only once all **M** items work end to end |
| **C** | Could have / stretch. Attempted only if week 13 slack is genuinely clear |
| **W** | Won't have this semester — recorded so it is not silently re-scoped in |

Verification method per requirement: **T** = automated test, **D** = live demonstration, **A** = analysis / measurement against recorded data, **I** = inspection or review.

---

## 2. Stakeholders

| Stakeholder | Interest | Requirements they sign off |
| --- | --- | --- |
| Course panel / evaluators | Working live demo, defensible results | FR-UI, EV-*, MG-* |
| Perception lead | Calibration, detector, tracking, both state trackers | FR-CAL, FR-PER, FR-STA |
| Temporal model lead | Features, MS-TCN++, benchmark scripts | FR-TMP, EV-SEG |
| Reasoning lead | Schemas, compiler, validator, checker, explainer | FR-CMP, FR-CHK, FR-EXP, IF-* |
| App lead | Streaming backend, UI, integration, latency | FR-UI, NFR-PERF, IF-WS |
| Demo participants (panel member, teammates) | Can operate it without training | NFR-USE |
| Recorded subjects | Consent and release terms for published data | NFR-PRIV, DR-REL |

---

## 3. Functional requirements

### 3.1 FR-CMP — LLM procedure compiler (plan §2.1, §4.2, §4.3)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-CMP-1 | The compiler shall accept a plain-text manual (one instruction per step) and emit a procedure spec in **DAG form** conforming to the §4.2 schema. | M | T |
| FR-CMP-2 | For each step the compiler shall emit: step id, verbatim instruction, part descriptor (colour + size), target pose (x, y, layer, rotation) on the stud grid, `hard_depends_on`, `soft_depends_on`, `optional`, and an expected duration range. | M | T |
| FR-CMP-3 | The compiler shall distinguish **hard** (physical support) from **soft** (manual-stated order) dependencies and record both independently. | M | T |
| FR-CMP-4 | The compiler shall accept a rulebook and emit a procedure spec in **constraint form**: one or more action schemas, each with preconditions (`expr`, human-readable `text`, `violation` code) and effects, plus terminal conditions. | S5 | T |
| FR-CMP-5 | Precondition expressions shall be drawn from a **fixed predicate vocabulary supplied in the prompt** and evaluated by a small safe interpreter. Python `eval` or any equivalent arbitrary-code path shall not be used. | S5 | I, T |
| FR-CMP-6 | The validator shall reject a spec that fails any of: JSON-schema conformance, dependency-cycle check, collision check (two parts occupying the same cells), support check (no floating parts), and — for constraint specs — a satisfiability check that at least one legal action exists from the initial state. | M | T |
| FR-CMP-7 | On a validation failure the system shall report the failing check and the offending step/constraint id, and shall permit **one automated repair round** through the LLM before requiring human intervention. | M | T |
| FR-CMP-8 | The system shall provide a **teach-by-demonstration** path: record one correct performance, capture the world state after each action, and use the LLM only to name the resulting steps. | S6 | D |
| FR-CMP-9 | The UI shall allow a human to edit and re-validate a compiled spec before a session runs. | S6 | D |
| FR-CMP-10 | Compilation shall be an offline, once-per-procedure operation. No LLM call shall occur on the per-frame or per-event path. | M | I |
| FR-CMP-11 | The LLM shall never be given raw video or video-derived imagery. Its inputs are text only. | M | I |

### 3.2 FR-CAL — Calibration and cameras (plan §2.2)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-CAL-1 | The system shall detect four printed ArUco markers at the baseplate corners and compute a homography from image pixels to **stud coordinates**. | M | T, D |
| FR-CAL-2 | Calibration shall succeed from any reasonable top-down camera angle and shall not require the camera to be perfectly perpendicular. | M | D |
| FR-CAL-3 | The homography shall be recomputed every 10 frames, or immediately when marker positions shift beyond a threshold. | M | T |
| FR-CAL-4 | A manual four-corner click fallback shall be available when marker detection fails. | S1 | D |
| FR-CAL-5 | The system shall support a **side camera** calibrated against the same world frame, used as the primary height signal for LEGO. | S1 | D |
| FR-CAL-6 | The system shall support **two side cameras at 90°** for Jenga, calibrated against table markers plus known block dimensions (75 × 25 × 15 mm) for scale. | S4 | D |
| FR-CAL-7 | The system shall detect and report loss of calibration (markers occluded or out of frame) rather than silently emitting wrong coordinates. | S1 | T |

### 3.3 FR-PER — Perception (plan §2.3)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-PER-1 | The system shall detect hand keypoints every frame and derive: hand-over-workspace, grasp, and **number of hands in contact**. | M | T |
| FR-PER-2 | The system shall detect parts every frame with a fine-tuned YOLO detector, classifying LEGO parts by colour and size (`red_2x4`, `blue_2x2`, `yellow_1x4`, …). | M | T |
| FR-PER-3 | A zero-shot open-vocabulary detector shall be available as a documented fallback and as the ablation baseline. | M | A |
| FR-PER-4 | The system shall maintain stable part identities across frames using a multi-object tracker, so that part motion can be attributed. | M | T |
| FR-PER-5 | The Jenga detector shall use a **single part class**, so that a Jenga failure is attributable to state or reasoning logic rather than to detection. | S4 | I |
| FR-PER-6 | While a hand occludes the workspace the world state shall be **frozen**; updates resume only after the hand has left. | M | T |

### 3.4 FR-STA — World-state tracker (plan §2.4)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-STA-1 | The world-state tracker shall be an **interface** exposing `current()` and `diff()`, with LEGO and Jenga as independent implementations. No checker code shall branch on procedure type. | M | I, T |
| FR-STA-2 | The LEGO tracker shall maintain the model as a list of placed bricks `(type, x, y, layer, rotation)` in stud coordinates. | M | T |
| FR-STA-3 | A brick shall be counted as placed only after it has remained still on the plate for ≈0.5 s following a hand exit. | M | T |
| FR-STA-4 | The tracker shall emit a state diff after each placement or removal as an event, e.g. `ADD red_2x4 @ (0,0,L0,rot0)`. | M | T |
| FR-STA-5 | Layer (height) shall be inferred from top-down occupancy **only** where no side camera is available; once integrated, the side camera shall be the primary height signal. | S1 | A |
| FR-STA-6 | The tracker shall support a **forced re-verification** request for named prior steps, resolved once the workspace view is clear (see FR-CHK-7). | M | T |
| FR-STA-7 | The Jenga tracker shall maintain an 18 × 3 occupancy array plus per-layer orientation, derived from two orthogonal side views. | S4 | T |
| FR-STA-8 | The Jenga tracker shall emit `ADD (layer, slot)` and `REMOVE (layer, slot)` events, and `COLLAPSE` as a terminal event detected from a large single-frame drop in tower height. | S4 | T, D |

### 3.5 FR-TMP — Step recognizer (plan §2.5)

The step recognizer is a **benchmark-only track, time-boxed to weeks 5 and 9**. It is not on the live critical path for either demo. Any requirement here that threatens a Must-have requirement is cut, not rescheduled.

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-TMP-1 | The system shall extract clip-level features from a pretrained video encoder plus perception features (hand positions, part-in-hand, part counts), and cache them per video. | M | T |
| FR-TMP-2 | A temporal action segmentation model shall label each moment with a generic action class: `search`, `grasp`, `place`, `adjust`, `remove`, `idle`. | M | T |
| FR-TMP-3 | The model shall predict **generic verbs, not step ids**, so that it transfers to Model B and to Jenga without retraining. Step identity comes from the world-state tracker. | M | I |
| FR-TMP-4 | The temporal model shall run asynchronously at ≈2 Hz and shall never block a video frame. | M | T, A |
| FR-TMP-5 | The temporal model shall supply the **only** action signal during occlusion windows, and its events shall be consumable by the checker on the same interface as state events. | S | T |
| FR-TMP-6 | The system shall run the step recognizer standalone on a public benchmark that has no stud grid. | M | A |

### 3.6 FR-CHK — Checker (plan §2.6, §4.4)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-CHK-1 | The checker shall maintain per-step status in {`pending`, `active`, `done`, `error`} and evaluate preconditions on every received event. | M | T |
| FR-CHK-2 | The checker's native interface shall be the **constraint form**; DAG steps shall be lowered into it so both procedures execute identical engine code. | M | I, T |
| FR-CHK-3 | The checker shall raise **missed step**: a required, non-optional step still `pending` at session end, or implied by a soft-dependency violation. | M | T, D |
| FR-CHK-4 | The checker shall raise **out of order**: a placement matching step *k* while a step it softly depends on is still `pending`. | M | T, D |
| FR-CHK-5 | The checker shall raise **wrong brick**: a placed part whose type matches no pending step at that pose. | M | T, D |
| FR-CHK-6 | The checker shall raise **wrong position / rotation** (type matches a pending step but pose is off by ≥1 stud or 90°) and **extra / removed brick**. | S2 | T, D |
| FR-CHK-7 | An unmet **hard** dependency shall **never** produce a user-facing alert. It shall defer the decision, request state re-verification, and resolve as either a logged **perception miss** (part was present, we missed it) or a genuine **missed step** alert. | M | T |
| FR-CHK-8 | Perception misses recorded by the re-verification path shall be logged separately and counted against detector recall in the evaluation, not against the user. | M | A, I |
| FR-CHK-9 | Events below a configurable confidence threshold shall produce a **soft (yellow) warning** rather than a red alert. | S2 | T, D |
| FR-CHK-10 | For Jenga the checker shall raise: **illegal source layer**, **layer started early**, **two-handed move**, **wrong replacement**, and **collapse** (terminal). | S5 | T, D |
| FR-CHK-11 | Every alert shall be attributable to one explicit rule or precondition id. No alert shall originate from a learned model's output alone. | M | I |
| FR-CHK-12 | The checker shall write a complete, timestamped event and alert log for every session. | M | T |

### 3.7 FR-UI — Interface, streaming and explainer (plan §2.7, §4.5)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-UI-1 | The live view shall show the video with part bounding boxes and a grid (LEGO) or lattice (Jenga) overlay. | M | D |
| FR-UI-2 | The live view shall show a **checklist** that ticks steps off as they complete (LEGO) or a **live legality panel** (Jenga). | M | D |
| FR-UI-3 | Alerts shall appear in the live view distinguishing red (error) from yellow (soft warning), with the human-readable rule text. | M | D |
| FR-UI-4 | The demo process shall read the camera **locally in the Python process** and push only overlays and alerts over the WebSocket. Browser-side frame capture may remain available for remote use but is not the demo path. | M | I, A |
| FR-UI-5 | A **ghost overlay** shall indicate the next expected placement. | S6 | D |
| FR-UI-6 | Session start, stop and procedure selection shall be operable from the UI without editing code or config files. | S | D |
| FR-EXP-1 | At session end (or on `COLLAPSE`) the system shall pass the event log to the LLM and produce a readable session report citing timestamps, the step involved and the consequence. | S6 | D |
| FR-EXP-2 | The session report shall be generated from the event log only — never from video. | S6 | I |

### 3.8 FR-SYS — System-level behaviour

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| FR-SYS-1 | The end-to-end path camera → perception → state → checker → alerts shall run as a single live session with no manual step between stages. | M | D |
| FR-SYS-2 | Procedure, camera and model choices shall be set by YAML config; switching between Model A, Model B and Jenga shall require no code change. | S | I, D |
| FR-SYS-3 | A session shall recover from transient detector failure (dropped frames, momentary marker loss) without terminating. | S | T |
| FR-SYS-4 | The system shall run entirely offline except for the offline compile and post-session explain calls; a local LLM fallback shall be documented. | S | I |

---

## 4. Non-functional requirements

### 4.1 NFR-PERF — Performance and latency (plan §3.1)

Budget is measured on the demo laptop in week 6 and re-measured in week 11.

| ID | Requirement | Target | Pri | Verify |
| --- | --- | --- | --- | --- |
| NFR-PERF-1 | End-to-end live frame rate | ≥ 15 FPS sustained | M | A |
| NFR-PERF-2 | Per-frame critical path total | ≤ 40 ms (66 ms available) | M | A |
| NFR-PERF-3 | Part detector, per frame | 15–25 ms | M | A |
| NFR-PERF-4 | Hand keypoints, per frame | 5–10 ms | M | A |
| NFR-PERF-5 | Tracker, per frame | < 2 ms | M | A |
| NFR-PERF-6 | Homography refresh, amortised | < 1 ms | M | A |
| NFR-PERF-7 | State update + diff, per hand-exit | < 1 ms | M | A |
| NFR-PERF-8 | Checker, per event | < 1 ms | M | A |
| NFR-PERF-9 | Temporal model, async, off critical path | ≤ 200 ms per 0.5 s window | M | A |
| NFR-PERF-10 | WebSocket overlay push | ≤ 10 ms per frame | M | A |
| NFR-PERF-11 | Median delay from error occurrence to alert shown | < 2 s | M | A |

### 4.2 NFR-* — Other quality attributes

| ID | Attribute | Requirement | Pri | Verify |
| --- | --- | --- | --- | --- |
| NFR-REL-1 | Reliability | A full Model A session (12–15 steps) shall complete without a crash or a manual restart in ≥ 9 of 10 consecutive runs. | M | A |
| NFR-REL-2 | Quietness | ≤ 1.0 false alerts per fully correct Model A build. | M | A |
| NFR-USE-1 | Usability | A person who has not used the system before shall complete a guided Model A build using only on-screen information. | S | D |
| NFR-USE-2 | Usability | Alert text shall name the step and the physical consequence in plain language, not a rule id alone. | S | I |
| NFR-MNT-1 | Maintainability | Adding a new procedure of an existing state model shall require only a new manual and a compiled spec — no code change. | M | D |
| NFR-MNT-2 | Maintainability | Adding a new **state model** shall require implementing the `WorldState` interface only; checker and compiler code shall be untouched. | S | I |
| NFR-PORT-1 | Portability | Training shall be runnable on a free-tier cloud GPU (Colab/Kaggle) via precomputed features and small model variants. | M | D |
| NFR-PORT-2 | Portability | Inference shall run on a single consumer GPU; a documented CPU-only degraded mode is a stretch goal. | M / C | A |
| NFR-COST-1 | Cost | Total LLM API spend shall stay negligible (≈20 calls across the project); a local model fallback shall be documented. | M | A |
| NFR-PRIV-1 | Privacy | Recorded subjects shall give written consent covering research use and, separately, public release. Faces need not be captured — frame the workspace, not the person. | M | I |
| NFR-PRIV-2 | Privacy | Secrets (LLM API keys) shall live in environment variables or an untracked `.env`; no key shall be committed. | M | I, T |
| NFR-REP-1 | Reproducibility | Every reported number shall be regenerable from a committed script, a fixed data split and a recorded config. | M | I |
| NFR-REP-2 | Reproducibility | Dataset splits shall be fixed in week 8 and never adjusted after a result is reported from them. | M | I |
| NFR-TRC-1 | Traceability | Every experiment run shall be logged to the experiment tracker with its config hash and git commit. | S | I |

---

## 5. Data requirements (plan §5)

### 5.1 Public datasets

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-PUB-1 | Access, licence and storage size for every candidate public dataset shall be **verified in week 1** — the plan's descriptions are from memory and are unverified. | M | I |
| DR-PUB-2 | One public procedural-error dataset with mistake labels shall be secured as the primary benchmark (first choice: a multi-view assembly dataset with fine-grained action and mistake labels). | M | I |
| DR-PUB-3 | At least one alternative benchmark shall be identified and kept viable in case access is denied or slow. | M | I |
| DR-PUB-4 | Any third benchmark is a stretch item and shall not be scheduled. | C | — |

### 5.2 STEPWISE-LEGO (own recordings)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-LEG-1 | **Model A**: 12–15 steps, designed **nearly flat (2 layers)** so that the must-have demo never depends on height inference. | M | I |
| DR-LEG-2 | **Model B**: 10–12 steps, 3–4 layers, built from the same ~8–10 brick types, used **only for testing** (zero-shot, tier 1). | S3 | I |
| DR-LEG-3 | Brick vocabulary: ~8–10 standard types (2×2, 2×4, 1×4, 1×2) in clearly distinct colours. No tiny pieces (1×1 plates, Technic pins), no hinged or angled parts. | M | I |
| DR-LEG-4 | Model A: **40–60 runs** from 4–6 people, roughly half containing planted errors. | M | I |
| DR-LEG-5 | Model B: **15–20 runs**, test split only, never used for training or tuning. | S3 | I |
| DR-LEG-6 | Recording shall **start in week 4** and accumulate continuously, not begin in week 7. | M | I |
| DR-LEG-7 | Planted error types shall cover all eight: skip a step; swap two steps; wrong colour; wrong size; off-by-one-stud position; 90° rotation; extra brick; remove a placed brick. | M | I |
| DR-LEG-8 | Labels: step start/end times and error type per run; final state per step (from teach-by-demonstration); bounding boxes on 300–600 frames. | M | I |
| DR-LEG-9 | Labelling budget: **30 person-hours**, shared across the team in week 8. | M | A |

### 5.3 STEPWISE-JENGA (own recordings)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-JEN-1 | 15–20 games from 3–4 people, recorded from two orthogonal side views. | S4 | I |
| DR-JEN-2 | Planted violations shall cover: pull from the top complete layer; start a new layer early; two-handed pull; return a block to an illegal slot. Collapses occur naturally and need no planting. | S5 | I |
| DR-JEN-3 | Labels: move start/end, source and destination slot, violation type. Budget **8 person-hours**. | S4 | A |
| DR-JEN-4 | Detector training set: ~150 labelled frames, single class, against a dark backdrop. | S4 | I |

### 5.4 Compiler eval corpus

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-CRP-1 | **15–20 short written manuals** for models that are never physically built, each with a hand-authored ground-truth spec. Authored in **week 2**. | M | I |
| DR-CRP-2 | At least 3–4 corpus entries shall be rulebook-style sources, so that the constraint form is measured and not only the DAG form. | S5 | I |
| DR-CRP-3 | Compiler errors shall be reported under an explicit taxonomy: wrong pose, missed dependency, hallucinated step, wrong constraint. | M | A |
| DR-CRP-4 | This corpus is the one result that survives independently of the vision stack, and shall be completed even if perception slips. | M | I |

### 5.5 Release

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| DR-REL-1 | The compiler eval corpus is text-only and shall be released unconditionally with the code. | M | I |
| DR-REL-2 | Video datasets shall be released only with the consent of everyone recorded (NFR-PRIV-1). | S | I |

---

## 6. Hardware requirements (plan §3.1, §2.2)

Procurement checklist for **week 1**. Items marked M block the must-have demo.

| ID | Item | Spec | Qty | Pri |
| --- | --- | --- | --- | --- |
| HW-1 | Top-down camera | 1080p webcam or phone with a stable USB/IP feed | 1 | M |
| HW-2 | Overhead camera stand / rig | Rigid, repeatable position; must not drift during a session | 1 | M |
| HW-3 | LEGO baseplate | 16×16 or 32×32 studs | 1–2 | M |
| HW-4 | Printed ArUco markers | 4, mounted at baseplate corners, rigid and non-glossy | 1 set | M |
| HW-5 | LEGO brick set | ~8–10 distinct types (2×2, 2×4, 1×4, 1×2) in clearly separable colours; enough duplicates for Model A and Model B | 1 | M |
| HW-6 | Plain mat and even lighting | Matte, non-reflective; consistent illumination across the plate | 1 | M |
| HW-7 | Laptop with GPU | Runs the full live pipeline within the §4.1 budget; remote GPU acceptable for training only | 1 | M |
| HW-8 | Side camera | Same class as HW-1; primary LEGO height signal | 1 | S1 |
| HW-9 | Second side camera | Mounted at 90° to HW-8 for Jenga full observability | 1 | S4 |
| HW-10 | Jenga set | Standard 54 blocks, 75 × 25 × 15 mm | 1 | S4 |
| HW-11 | Dark backdrop | High contrast behind the Jenga tower | 1 | S4 |
| HW-12 | Table markers for Jenga | Around the tower base, for side-view calibration and scale | 1 set | S4 |

| ID | Constraint | Pri |
| --- | --- | --- |
| HW-C1 | A minimum of **2 cameras** is required for Jenga; **3** is ideal for the full system. The achievable mount and sync count is an open item (§9). | S4 |
| HW-C2 | All cameras shall be time-aligned well enough that a single move is attributable to one timestamp within one frame interval. | S4 |
| HW-C3 | Camera positions shall be reproducible between sessions, or recalibration shall be part of the documented session startup. | M |

---

## 7. Software and environment requirements (plan §3)

### 7.1 Module → technology commitments

| ID | Module | Primary | Fallback | Trained? |
| --- | --- | --- | --- | --- |
| SW-1 | Calibration | OpenCV ArUco + homography | Manual 4-corner click | No |
| SW-2 | Hands | MediaPipe Hands | MMPose hand models | No |
| SW-3 | Part detection | YOLO (Ultralytics), fine-tuned | YOLO-World zero-shot | **Yes** — 300–600 LEGO frames, ~150 Jenga |
| SW-4 | Tracking | ByteTrack | SORT | No |
| SW-5 | World state | Own grid / lattice modules (NumPy) | — | No |
| SW-6 | Video features | VideoMAE or I3D, precomputed | DINOv2 per-frame | No |
| SW-7 | Step segmentation | MS-TCN++ | ASFormer | **Yes** — main temporal model |
| SW-8 | LLM (compiler, explainer) | Claude Sonnet 5 via API | Local Qwen or Llama via Ollama | No — prompt + JSON schema |
| SW-9 | Checker | Plain Python state machine | — | No |
| SW-10 | Backend | FastAPI + WebSocket streaming | — | — |
| SW-11 | Frontend | React | Streamlit for early versions | — |
| SW-12 | Labelling | CVAT (boxes, step segments) | Label Studio | — |
| SW-13 | Experiment tracking | Weights & Biases | TensorBoard | — |

### 7.2 Environment requirements

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| SW-E1 | The project shall pin a reproducible Python environment (lockfile committed) in **week 1**; the versions in §7.3 are indicative floors, not the final pins. | M | I |
| SW-E2 | Python 3.11+ with PyTorch and CUDA matched to the available GPU driver. | M | T |
| SW-E3 | `pip install -r requirements.txt` (or the chosen lock tool's equivalent) on a clean machine shall bring up a working inference environment. Verified once per semester on a second machine. | M | T |
| SW-E4 | Training-only dependencies shall live in a separate extras group so the demo machine installs the minimum. | S | I |
| SW-E5 | The repository shall follow the directory layout of plan §4.1. | M | I |
| SW-E6 | Unit tests shall exist for the stud grid, the Jenga lattice, the state diff and every checker rule. | M | T |
| SW-E7 | Data, weights and videos shall not be committed to git; a documented storage location and a fetch script shall be provided instead. | M | I |

### 7.3 Dependency groups (indicative; pin in week 1)

**Runtime — live demo**

```
numpy
opencv-contrib-python        # ArUco lives in contrib
torch                        # CUDA build matched to the local driver
ultralytics                  # YOLO detector + ByteTrack
mediapipe                    # hand keypoints
pydantic                     # procedure-spec schemas
fastapi
uvicorn[standard]            # WebSocket streaming
websockets
pyyaml                       # configs
anthropic                    # LLM compiler + explainer
python-dotenv                # API key, never committed
```

**Training and evaluation (extras)**

```
torchvision
transformers                 # VideoMAE features
scikit-learn
pandas
matplotlib
wandb
tqdm
```

**Development (extras)**

```
pytest
ruff
mypy
```

**Frontend:** Node 20+, React, a WebSocket client. Streamlit is acceptable for pre-week-12 internal views only.

**External services:** an LLM API key (≈20 calls total) with Ollama + a local model as the offline fallback; a labelling server (CVAT) reachable by the whole team.

---

## 8. Interface and contract requirements (plan §4.2–§4.4)

| ID | Requirement | Pri | Verify |
| --- | --- | --- | --- |
| IF-SPEC-1 | Both spec forms shall be defined as Pydantic models and validated on load. A spec that does not validate shall not run. | M | T |
| IF-SPEC-2 | The DAG form shall carry: `procedure`, `kind: "dag"`, `state_model`, baseplate dimensions, and a `steps` array as specified in FR-CMP-2. | M | T |
| IF-SPEC-3 | The constraint form shall carry: `procedure`, `kind: "constraints"`, `state_model`, geometry (e.g. `tower: {layers, slots_per_layer}`), an `actions` array with `preconditions` and `effects`, and a `terminal` array. | S5 | T |
| IF-SPEC-4 | Every precondition shall carry a machine `expr`, a human `text` and a `violation` code; alerts shall surface the `text` and log the `violation`. | S5 | T |
| IF-SPEC-5 | Specs shall be versioned; a spec file shall record the compiler version and model that produced it. | S | I |
| IF-EVT-1 | Events shall conform to a single schema: `t` (seconds since session start), `source` (`state` \| `temporal`), `kind` (`ADD` \| `REMOVE` \| `STEP_START` \| `STEP_END`), optional `part`, optional `pose`, and `confidence`. | M | T |
| IF-EVT-2 | The checker shall accept events from the state tracker and the temporal model through the same entry point, distinguished only by the `source` field. | M | T |
| IF-STA-1 | Every state model shall implement `current()` and `diff()` and nothing procedure-specific shall leak past that boundary. | M | I, T |
| IF-WS-1 | The WebSocket shall carry overlays, checklist/legality status and alerts **outbound only** on the demo path; video frames shall not travel inbound during a demo. | M | I, A |
| IF-WS-2 | The WebSocket message schema shall be documented and versioned so the UI and backend can be developed independently. | S | I |
| IF-LOG-1 | The session log shall be a machine-readable file sufficient to regenerate the alert timeline and the LLM report without the video. | M | T |

---

## 9. Evaluation acceptance criteria (plan §7)

These are the pass/fail gates for week 11. They are fixed **now**, before results exist.

| ID | Question | Metric | Target | Pri |
| --- | --- | --- | --- | --- |
| EV-DET-1 | Are parts detected where they are? | Median centroid error in studs | ≤ 0.4 stud | M |
| EV-DET-2 | " | p95 centroid error in studs | ≤ 1.0 stud | M |
| EV-DET-3 | " | mAP@0.5 | Secondary/reported only — **not** a gate | M |
| EV-STA-1 | Is the world state right? (Model A) | Placement accuracy, type + pose exact, per step | ≥ 90% | M |
| EV-STA-2 | Is the world state right? (Model B) | Same | ≥ 85% | S3 |
| EV-STA-3 | Is the Jenga lattice right? | Slot occupancy accuracy vs hand-labelled state | ≥ 95% | S4 |
| EV-SEG-1 | Are steps recognized? | Frame accuracy, Edit score, F1@{10, 25, 50} | Within reach of published numbers on the chosen benchmark | M |
| EV-ERR-1 | Are errors caught? (Model A) | Precision at fixed recall, **per error type** | Precision ≥ 0.8 at recall ≥ 0.8 | M |
| EV-ERR-2 | Are errors caught? (Model B) | Same | Precision ≥ 0.7 at recall ≥ 0.7 | S3 |
| EV-ERR-3 | Is it quiet when nothing is wrong? | False alerts per fully correct build (Model A) | ≤ 1.0 | M |
| EV-ERR-4 | How early? | Median delay from error to alert | < 2 s | M |
| EV-LAT-1 | Is it usable live? | Per-stage latency vs the §4.1 budget | ≥ 15 FPS, per-frame ≤ 40 ms | M |
| EV-CMP-1 | Is the compiler right? | Step accuracy over the eval corpus (n = 15–20) | ≥ 0.9 | M |
| EV-CMP-2 | " | Dependency accuracy | ≥ 0.8 | M |
| EV-CMP-3 | " | Pose and constraint accuracy | Reported with the §5.4 error taxonomy | M |
| EV-JEN-1 | Do Jenga rules hold? | Violation precision/recall **per rule** | Precision ≥ 0.8 at recall ≥ 0.8 | S5 |
| EV-JEN-2 | " | Collapse detection | Reported | S5 |

### 9.1 Required baselines

| ID | Baseline | Pri |
| --- | --- | --- |
| EV-BL-1 | **Temporal model only** — no world-state tracker; errors decided from recognized actions alone. | M |
| EV-BL-2 | **End-to-end VLM** — sampled frames plus the manual, asked to list errors. Reported **as measured**; no outcome is predicted in advance, and a split result (VLM better on missed step, worse on position and timing) is a legitimate and interesting finding. | M |
| EV-BL-3 | **Published results** from the benchmark dataset's own paper. | M |

### 9.2 Required ablations

| ID | Ablation | Pri |
| --- | --- | --- |
| EV-AB-1 | Without the world-state tracker (actions only). | M |
| EV-AB-2 | Without the temporal model (state only). | M |
| EV-AB-3 | **Occlusion-window slice** — errors that begin and end inside an occlusion window, state-only vs fused. This is the specific, pre-registered hypothesis of plan §2.5 and the only place the temporal model is expected to win. | S |
| EV-AB-4 | LLM-generated spec vs hand-written spec. | M |
| EV-AB-5 | Zero-shot open-vocabulary detector vs fine-tuned detector. | M |
| EV-AB-6 | With vs without soft warnings, measured on false-alarm rate. | S2 |
| EV-AB-7 | Top-down only vs top-down + side camera, on Model B height accuracy. | S1 |
| EV-AB-8 | Tier 1 vs tier 2 vs tier 3 generality (plan §1.2). | S |

### 9.3 Generality ladder — reporting requirement

| ID | Requirement | Pri |
| --- | --- | --- |
| EV-GEN-1 | **Tier 1** — unseen LEGO assembly and unseen manual, same bricks and detector. Supports: compiler and checker generalise across assemblies. | S3 |
| EV-GEN-2 | **Tier 2** — Jenga tower *build*: unseen domain, new part vocabulary, same graph form and checker. Supports: the perception stack is retargetable. | S4 |
| EV-GEN-3 | **Tier 3** — Jenga *game*: unseen procedure topology (constraints, not order), same checker interface. Supports: the spec format itself is general. **This is the claim worth making.** | S5 |
| EV-GEN-4 | Tier 1 shall not be presented alone as the generality result; the report shall state the ladder honestly and shall not overclaim. | M |

---

## 10. Milestone gates (plan §8)

| ID | Week | Gate — the requirement is met when… | Pri |
| --- | --- | --- | --- |
| MG-1 | 1 | Related-work summary written; dataset access requested and licences verified (DR-PUB-1); hardware in hand (§6); LLM API question settled; environment pinned (SW-E1). | M |
| MG-2 | 2 | Valid specs for Model A and Model B; spec schema (both forms) committed; compiler v1 + validator running; **eval corpus committed** (DR-CRP-1). | M |
| MG-3 | 3 | Bricks detected and mapped to studs live; 300+ frames labelled; detector v1 trained. | M |
| MG-4 | 4 | Correct live build state for Model A; teach-by-demo recorder working; **first Model A runs banked** (DR-LEG-6). | M |
| MG-5 | 5 | Live alerts on Model A for missed / out-of-order / wrong brick; benchmark features precomputed; first segmentation numbers. *(Temporal box 1 of 2.)* | M |
| MG-6 | 6 | **End-to-end v0** demo runs; latency budget table filled and measured (NFR-PERF-*); mid-term review passed. | M |
| MG-7 | 7 | Model A recording topped up; Model B test runs recorded; compiler accuracy table complete over the corpus (EV-CMP-*). | M |
| MG-8 | 8 | Labels exported; **splits fixed and frozen** (NFR-REP-2); detector v2; **side camera integrated** and Model B heights correct (FR-CAL-5, FR-STA-5). | S1 |
| MG-9 | 9 | Occlusion-window fusion measured (EV-AB-3). **Jenga go/no-go gate decided against must-have status** — if green, Jenga rig, detector and lattice state stand up; if red, Jenga is formally deferred and recorded as such. *(Temporal box 2 of 2.)* | S4 |
| MG-10 | 10 | Remaining LEGO rules in (position, rotation, extra/removed); soft warnings; LLM report; Jenga constraint compiler and checker firing. | S2 |
| MG-11 | 11 | Full evaluation complete: benchmark, Model A, Model B, Jenga, all baselines, all ablations, latency. | M |
| MG-12 | 12 | UI polished (checklist, alerts, ghost overlay, spec editor, Jenga legality panel); someone outside the team can operate it; demo video recorded. | S6 |
| MG-13 | 13 | **Reserved slack — not allocated to features.** Absorbs overruns from any earlier week. Stretch goals only if genuinely clear. | — |
| MG-14 | 14 | Report, poster, demo video and final presentation submitted. | M |

**Gate rule:** MG-9 is a hard decision point. Jenga work shall not begin before the must-have set (all **M** requirements) is demonstrably green, and Jenga shall never be promoted to must-have.

---

## 11. Deliverable requirements (plan §11)

| ID | Deliverable | Requirement | Pri |
| --- | --- | --- | --- |
| DL-1 | Code | Public repository with setup instructions, trained weights and evaluation scripts. | M |
| DL-2 | Dataset | STEPWISE-LEGO and STEPWISE-JENGA videos, specs and labels, released subject to DR-REL-2. Compiler eval corpus released unconditionally. | S |
| DL-3 | Report | Written as a short paper: problem, related work, method, results, ablations, the tier 1–3 generality table, limitations. | M |
| DL-4 | Demo video | 2–3 min: a correct Model A build with the checklist ticking; each error type caught live; Model B from its manual alone; a Jenga game with violations called in real time. | M |
| DL-5 | Live demo | Baseplate, cameras, bricks and a Jenga tower. A panel member builds Model B and tries to break it by skipping or swapping steps, then plays Jenga and tries to cheat. | M |
| DL-6 | Publication | Submission to a student research track or a workshop on egocentric/procedural video. | C |

---

## 12. Constraints, assumptions and exclusions

### 12.1 Constraints

| ID | Constraint |
| --- | --- |
| CON-1 | 14 calendar weeks, with week 13 reserved as slack and not available for features. |
| CON-2 | Team of 3 or 4 (open item OQ-1). With 3, the Perception and App roles merge. |
| CON-3 | One consumer GPU, or free-tier cloud GPUs. All model choices must respect this. |
| CON-4 | Negligible budget: bricks, a Jenga set, cameras, a stand, printed markers, and ≈20 LLM API calls. |
| CON-5 | Fixed cameras only. No head-mounted or moving camera for our own recordings. |
| CON-6 | All visual judgment comes from CV models; every alert comes from an explicit rule. The LLM is text-in, text-out. |

### 12.2 Assumptions

| ID | Assumption | If false |
| --- | --- | --- |
| ASM-1 | Lighting is even and the mat is matte during recording and demo. | Detector precision drops; more labelled frames and lighting control needed. |
| ASM-2 | One person works in the workspace at a time. | Out of scope (§12.3); state attribution breaks. |
| ASM-3 | Public dataset access is granted within a few weeks. | Fall back to an alternative benchmark (DR-PUB-3); own data alone still carries the demo. |
| ASM-4 | Two orthogonal side views give full observability of all 54 Jenga slots, because each layer is end-on to exactly one camera. | Jenga state becomes partially unobservable; tier 3 result is at risk. |
| ASM-5 | Standard 75 × 25 × 15 mm Jenga blocks provide adequate scale reference. | Add an explicit scale marker to the rig. |
| ASM-6 | ~150 labelled frames suffice for the single-class Jenga detector. | Label more; Jenga's cheapness argument weakens but does not break. |

### 12.3 Explicitly out of scope

| ID | Excluded | Pri |
| --- | --- | --- |
| OOS-1 | Tiny pieces (1×1 plates, Technic pins), hinged or angled parts. | W |
| OOS-2 | Multiple people building simultaneously. | W |
| OOS-3 | Head-mounted (moving) cameras for our own data. | W |
| OOS-4 | Jenga stability or lean estimation — legality is checked, physics is not. | W |
| OOS-5 | VLM reading picture-only official LEGO instruction pages — **demoted**: a separate research problem with its own failure modes, and Jenga now carries the generality claim for less work. Revisit only if weeks 12–13 are clear. | C |

### 12.4 Stretch items

| ID | Item | Pri |
| --- | --- | --- |
| STR-1 | Synthetic training data from LDraw models rendered in Blender (random poses, lighting, backgrounds). | C |
| STR-2 | Laptop-CPU or Jetson edge deployment. | C |
| STR-3 | Voice alerts. | C |

---

## 13. Open items blocking requirement finalisation (plan §12)

Each of these leaves a requirement provisional. All should be closed in week 1 unless noted.

| ID | Open question | Blocks | Due |
| --- | --- | --- | --- |
| OQ-1 | Team size: 3 or 4? | CON-2, §2 role assignment | Week 1 |
| OQ-2 | Which bricks do we already have, and what must we buy? | HW-5, DR-LEG-3 | Week 1 |
| OQ-3 | Final designs for Model A (flat, 2 layers) and Model B (3–4 layers). | DR-LEG-1, DR-LEG-2, MG-2 | Week 2 |
| OQ-4 | What GPU is available: local card, college lab, or Colab/Kaggle only? | NFR-PORT-1, NFR-PORT-2, SW-E2 | Week 1 |
| OQ-5 | Course review dates, to line up the timeline and the week-9 Jenga gate. | §10 (all MG) | Week 1 |
| OQ-6 | How many cameras can we actually mount and synchronise? (2 minimum for Jenga, 3 ideal.) | HW-C1, HW-C2, FR-CAL-6 | Week 1 |
| OQ-7 | ~~Is an LLM API allowed or budgeted?~~ **Closed** — ≈20 calls, cost negligible, Ollama fallback exists; confirm the account in week 1. | NFR-COST-1, SW-8 | Closed |

---

## 14. Traceability

### 14.1 Requirement class → plan section

| Requirement class | Plan section |
| --- | --- |
| FR-CMP | §2.1, §4.2, §4.3 |
| FR-CAL | §2.2 |
| FR-PER | §2.3, §4.6 |
| FR-STA | §2.4 |
| FR-TMP | §2.5, §4.7 |
| FR-CHK | §2.6, §4.4 |
| FR-UI, FR-EXP | §2.7, §4.5 |
| NFR-PERF | §3.1 |
| DR-* | §5 |
| HW-* | §3.1, §2.2 |
| SW-* | §3, §4.1 |
| IF-* | §4.2, §4.3, §4.4 |
| EV-* | §7 |
| MG-* | §8 |
| DL-* | §11 |
| CON, ASM, OOS, STR | §6, §10 |
| OQ-* | §12 |

### 14.2 Risk → mitigating requirement (plan §10)

| Risk | Likelihood | Mitigated by |
| --- | --- | --- |
| Height/layer wrong from top-down view | **High** | DR-LEG-1 (flat Model A), FR-CAL-5, FR-STA-5, HW-8, MG-8, EV-AB-7 |
| Similar bricks confused (red/orange, 2×3/2×4) | High | DR-LEG-3 (distinct vocabulary), DR-LEG-8 (more labels), STR-1 |
| Hands block the view during placement | High | FR-PER-6 (freeze), FR-STA-3 (stability filter), FR-TMP-5, EV-AB-3 |
| Labelling takes longer than planned | High | DR-LEG-9 (30 h budget), DR-LEG-8 (boundaries only), FR-CMP-8 (states from demo) |
| Jenga scope creep sinks the must-have | Medium | MG-9 (go/no-go gate), priority scheme (§1.5): Jenga is never **M** |
| Jenga far-side blocks unobservable | Medium | FR-CAL-6, HW-9, ASM-4 |
| Step recognition too inaccurate | Medium | FR-TMP scope note (benchmark-only), FR-STA-4 (demo relies on state diffs) |
| Too many false alarms | Medium | FR-CHK-9 (soft warnings), FR-STA-3, NFR-REL-2 and EV-ERR-3 (explicit budget) |
| Dataset access slow or denied | Medium | DR-PUB-1, DR-PUB-3 (alternatives kept viable) |
| Not enough GPU | Medium | FR-TMP-1 (precomputed features), SW-3 (small detector variant), NFR-PORT-1 |
| LLM specs wrong | Low | FR-CMP-6 (validator), FR-CMP-7 (one repair round), FR-CMP-8, FR-CMP-9, EV-CMP-* |
| Jenga tower collapses mid-run | Certain | FR-STA-8 (terminal event, auto-saved clip) — a demo highlight, not a failure |

### 14.3 Must-have set (the minimum viable project)

The project passes its minimum bar when every **M** requirement is verified. In plan terms (§6): the compiler produces a validated dependency spec; calibration, detection, tracking and the world-state tracker work; the checker catches **missed step**, **out of order** and **wrong brick**; the live demo runs at 15+ FPS within the latency budget; the step recognizer is trained and benchmarked against at least one baseline; and compiler accuracy is measured on the eval corpus.

---

## 15. Change control

| ID | Requirement |
| --- | --- |
| CC-1 | Requirement IDs are stable. A requirement that is dropped is marked **superseded** with a reason; it is never deleted or renumbered. |
| CC-2 | Any change to an **EV-** target after week 11 results exist must be recorded as a change with a date and a rationale, and the original target reported alongside. |
| CC-3 | Promotion of any **S** or **C** item to **M** requires the must-have set to be green first (MG-9 rule). |
| CC-4 | This document is regenerated from the plan whenever the plan changes materially; the plan is the source of truth. |

---

*STEPWISE Requirements Specification v1.0 — 2026-09-18. Derived from `STEPWISE_Project_Plan.md`.*
