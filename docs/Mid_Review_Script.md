# Third Eye: Mid Review Video Script

**Pre CIE 03 · 5 marks** (Title 1, Literature Survey 2, Problem Statement 2)
**Deadline:** 27 Sept 2026, 11:30 PM, upload to GCR
**Target length:** about 8 minutes (speaking at about 140 words a minute)

| # | Speaker | Section | Time |
| :-: | :-- | :-- | :-: |
| 1 | Aadit Pani | Team introduction, title, motivation | 1:30 |
| 2 | Megh Mavani | Literature survey, part 1: datasets and step recognition (papers 1–10) | 2:30 |
| 3 | Mitul KM | Literature survey, part 2: mistake detection, LLMs, perception (papers 11–18), and research gaps | 2:30 |
| 4 | Aadit Pani | Problem statement, scope, closing | 1:45 |

> Each speaker introduces themselves on camera the
> first time they speak, which covers guideline 01. Keep your face visible in a corner while
> the slides are on screen.

---

## Slide 1 · Title (Aadit)

**On screen:**
> **Third Eye**
> Real-Time Detection of Procedural Errors from Video Using LLM-Compiled Rules
> Aadit Pani (1RVU23CSE002) · Megh Mavani (1RVU23CSE260) · Mitul KM (1RVU23CSE273)
> Mentor: Prof. Manjul K Gupta · CS3235

**Script:**

> Good evening. I'm **Aadit Pani**, USN 1RVU23CSE002, and with me are **Megh Mavani** and
> **Mitul KM**. We're working under **Prof. Manjul K Gupta**.
>
> Our project is called **Third Eye: Real-Time Detection of Procedural Errors from Video Using
> LLM-Compiled Rules.**
>
> The name comes from the idea: a third eye watching over your shoulder while you follow a
> procedure, which tells you the moment you go wrong rather than after you've finished.
>
> The subtitle says how. **Real-time** means the alert comes while the mistake is happening.
> **Procedural errors** means a missed step, a step done out of order, the wrong part, or a
> move that breaks a rule. **LLM-compiled rules** means nobody writes the rules by hand: a
> language model reads the manual or rulebook and turns it into rules a program can check.

## Slide 2 · Why this matters (Aadit)

**On screen:** three icons (lab bench, hospital sink, assembly line). One line: *"Most
procedural errors are omissions or wrong order, and usually nobody is watching."*

**Script:**

> Why does this matter? In lab work, in clinical routines like hand hygiene, and on assembly
> lines, most mistakes aren't clumsy motions. Someone **skips a step** or does steps **in the
> wrong order**, and usually nobody is watching when it happens.
>
> We test this on two everyday procedures. The first is a **jigsaw puzzle**, which has an
> ordered method: corners, then edges, then the interior. The second is **Jenga**, which has
> no order at all, only rules, like "never pull from the top complete layer". If one system
> handles both, it handles more than one kind of procedure.
>
> Megh will now walk you through what already exists.

---

## Slide 3 · Literature survey: datasets (Megh)

**On screen:** papers 1–6 from the table at the end, with one key phrase each.

**Script:**

> Hi, I'm **Megh Mavani**, USN 1RVU23CSE260. We surveyed eighteen papers in four groups. I'll
> cover the first two.
>
> **Group one is datasets for procedural mistakes.**
>
> **Assembly101** (Sener et al., CVPR 2022) is a multi-view dataset of people assembling and
> taking apart toy vehicles, with labels for mistakes. It showed that mistake detection is a
> task in its own right, and a hard one.
>
> **HoloAssist** (Wang et al., ICCV 2023) records people doing tasks with a HoloLens while an
> instructor guides them. It labels mistakes and the moments when the instructor steps in.
>
> **IndustReal** (Schoonbeek et al., WACV 2024) moves this to an industrial-style assembly and
> introduces *procedure step recognition*, meaning whether a step was actually completed,
> including when errors happen.
>
> **EgoPER** (Lee et al., CVPR 2024) and **CaptainCook4D** (Peddi et al., NeurIPS 2024) do
> the same for cooking, with deliberately planted errors in recipes.
>
> **Ego4D** (Grauman et al., CVPR 2022) is the large first-person video corpus many of these
> build on.
>
> **What these datasets have in common:** each covers one domain, and methods trained on them
> need labelled video of that same task.

## Slide 4 · Literature survey: step recognition and task structure (Megh)

**On screen:** papers 7–10.

**Script:**

> **Group two is recognising steps and learning task structure.**
>
> **MS-TCN** (Farha and Gall, CVPR 2019) and **ASFormer** (Yi et al., BMVC 2021) are the
> standard models for *temporal action segmentation*: labelling every frame of a video with
> the step being performed. MS-TCN uses stacked temporal convolutions and ASFormer uses a
> transformer.
>
> **Paprika** (Zhou et al., CVPR 2023) pre-trains video models with a procedural knowledge
> graph built from how-to articles, so the model knows which steps usually follow which.
>
> **Video-Mined Task Graphs** (Ashutosh et al., NeurIPS 2023) learns a graph of step order
> from many instructional videos and uses it to improve step recognition.
>
> **What we took from this group:** step recognition works, but it tells you *which action*
> happened, not whether the object ended up in the right place. And task graphs capture
> *order*, but not rules that have no order.
>
> Mitul will cover mistake detection and the gaps.

---

## Slide 5 · Literature survey: mistake detection and LLM reasoning (Mitul)

**On screen:** papers 11–14.

**Script:**

> Hi, I'm **Mitul KM**, USN 1RVU23CSE273.
>
> **Group three is online mistake detection and reasoning with language models.**
>
> **PREGO** (Flaborea et al., CVPR 2024) is the closest work to ours. It detects mistakes
> *online*: one branch recognises the current step, a language model predicts the next step,
> and a mismatch is flagged as a mistake.
>
> **Differentiable Task Graph Learning** (Seminara et al., NeurIPS 2024) learns the task graph
> directly from example sequences and uses it for online mistake detection on egocentric
> video.
>
> **SayCan** (Ahn et al., CoRL 2022) and **Code as Policies** (Liang et al., ICRA 2023) come
> from robotics. They show that an LLM can turn natural-language instructions into structured
> plans or code a machine can execute. That's the idea we borrow: the LLM **compiles** the
> manual into rules, and it never looks at the video.

## Slide 6 · Literature survey: perception building blocks (Mitul)

**On screen:** papers 15–18.

**Script:**

> **Group four is the perception building blocks we use.**
>
> **ArUco markers** (Garrido-Jurado et al., Pattern Recognition 2014) give reliable reference
> points even with partial occlusion. We use them to map the camera view onto the jigsaw
> board.
>
> **MediaPipe Hands** (Zhang et al., 2020) tracks 21 hand landmarks in real time. It tells us
> when a hand is in the workspace, so we don't judge the scene while it's covered.
>
> **YOLO-World** (Cheng et al., CVPR 2024) is an open-vocabulary detector that finds objects
> from a text prompt without task-specific training.
>
> **Jigsaw solving with unknown orientation** (Gallagher, CVPR 2012) matches pieces by
> appearance and handles rotation. We use a similar idea, matching each piece against the
> box picture, so no detector training is needed.

## Slide 7 · Research gaps (Mitul)

**On screen:** five numbered gaps, one line each.

**Script:**

> Across these eighteen papers we found **five gaps**.
>
> **One: per-task training.** Almost every method learns from labelled video of the same
> task, so a new procedure needs new data and a new model.
>
> **Two: order only.** Task graphs describe *sequences*. A procedure like Jenga, which is
> defined only by rules with no fixed order, can't be expressed at all.
>
> **Three: actions, not state.** Step recognisers can say "a piece was placed", but not
> whether it went into the *right cell* or was *rotated wrongly*.
>
> **Four: no explanation.** End-to-end models and vision-language models give a verdict you
> can't trace back to a rule, and they can hallucinate, which in a live system means false
> alarms.
>
> **Five: offline evaluation.** Most results are measured after the video ends. Few report
> alert delay, latency, or how many false alerts a user sees on a correct run.
>
> Aadit will now state the problem we're addressing.

---

## Slide 8 · Problem statement (Aadit)

**On screen:** the boxed statement below.

> **Given a written procedure (an instruction manual or a rulebook) and a live camera view of
> a person performing it, detect in real time every deviation from the procedure (a missed
> step, an out-of-order step, a wrong part, a wrong position or rotation, or an illegal move)
> without training a new model for each procedure, and make every alert traceable to an
> explicit rule.**

**Script:**

> Here is our problem statement. *(Read the box aloud.)*
>
> We address it by splitting the problem into three parts:
>
> - An **LLM compiler** reads the manual *once*, offline, and produces a *procedure spec*: what
>   must be true before each move, and what each move should change. That covers gaps one
>   and two, because a new procedure needs only its text, and the spec can be either ordered
>   steps or rules with no order.
> - A **vision pipeline** tracks hands and parts every frame and keeps a live *world state*:
>   which jigsaw cells are filled and with which piece, and which Jenga slots are occupied.
>   That covers gap three.
> - A **deterministic checker** compares the two and raises the alert. Every alert names the
>   rule that fired, and when the view is blocked the checker waits instead of guessing. That
>   covers gap four.

## Slide 9 · Significance and scope (Aadit)

**On screen:** two columns, "In scope" and "Out of scope", and a row of measurable targets.

**Script:**

> **Significance:** the same system, given a different manual, checks a different procedure.
> The same approach carries over to lab protocols, clinical checklists and assembly lines.
>
> **Scope.** Our two demonstrations are:
>
> - a **jigsaw puzzle** under one overhead webcam, where pieces are identified by matching
>   against the box picture, and
> - **Jenga**, using an iPhone Pro's **LiDAR depth camera**, because every block looks the same
>   and depth shows which slots are empty.
>
> **Out of scope:** multiple people working at once, moving head-mounted cameras, and Jenga
> tower physics. We check whether a move is *legal*, not whether the tower will fall.
>
> **How we'll measure it** (gap five): precision and recall for each error type, false alerts
> per correct run, delay from the mistake to the alert, frame rate, and how accurately the
> LLM compiles the manuals.
>
> So far we've built the spec format, the validator, the LLM compiler, the checker for both
> procedures, and the jigsaw piece matcher, all covered by automated tests. Next are the live
> camera loop and the LiDAR input.
>
> That's Third Eye. Thank you.

---

## Slide: References (show for 5–10 s at the end, or keep as the last slide)

| # | Paper | Venue |
| :-: | :-- | :-- |
| 1 | F. Sener et al., *Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities* | CVPR 2022 |
| 2 | X. Wang et al., *HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World* | ICCV 2023 |
| 3 | T. Schoonbeek et al., *IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors in Egocentric Videos in an Industrial-Like Setting* | WACV 2024 |
| 4 | S.-P. Lee et al., *Error Detection in Egocentric Procedural Task Videos* (EgoPER) | CVPR 2024 |
| 5 | R. Peddi et al., *CaptainCook4D: A Dataset for Understanding Errors in Procedural Activities* | NeurIPS 2024 (D&B) |
| 6 | K. Grauman et al., *Ego4D: Around the World in 3,000 Hours of Egocentric Video* | CVPR 2022 |
| 7 | Y. Abu Farha, J. Gall, *MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation* | CVPR 2019 |
| 8 | F. Yi, H. Wen, T. Jiang, *ASFormer: Transformer for Action Segmentation* | BMVC 2021 |
| 9 | H. Zhou et al., *Procedure-Aware Pretraining for Instructional Video Understanding* (Paprika) | CVPR 2023 |
| 10 | K. Ashutosh et al., *Video-Mined Task Graphs for Keystep Recognition in Instructional Videos* | NeurIPS 2023 |
| 11 | A. Flaborea et al., *PREGO: Online Mistake Detection in PRocedural EGOcentric Videos* | CVPR 2024 |
| 12 | L. Seminara, G. M. Farinella, A. Furnari, *Differentiable Task Graph Learning: Procedural Activity Representation and Online Mistake Detection from Egocentric Videos* | NeurIPS 2024 |
| 13 | M. Ahn et al., *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances* (SayCan) | CoRL 2022 |
| 14 | J. Liang et al., *Code as Policies: Language Model Programs for Embodied Control* | ICRA 2023 |
| 15 | S. Garrido-Jurado et al., *Automatic Generation and Detection of Highly Reliable Fiducial Markers under Occlusion* (ArUco) | Pattern Recognition 2014 |
| 16 | F. Zhang et al., *MediaPipe Hands: On-device Real-time Hand Tracking* | CVPR Workshops 2020 |
| 17 | T. Cheng et al., *YOLO-World: Real-Time Open-Vocabulary Object Detection* | CVPR 2024 |
| 18 | A. C. Gallagher, *Jigsaw Puzzles with Pieces of Unknown Orientation* | CVPR 2012 |

---

## Recording checklist

- [ ] Each speaker says their name on camera before their section
- [ ] Mic check: record 10 seconds and play it back before the full take
- [ ] Slides readable at 720p; face visible in a corner
- [ ] Total runtime about 8 minutes; cut the "so far we've built" lines if you run long
- [ ] Final slide: references (18 papers, minimum is 15)
