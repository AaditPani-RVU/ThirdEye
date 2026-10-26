<div align="center">

# 👁️ Third Eye

### Catch mistakes in a procedure *as they happen*, from video, with rules an LLM reads from the manual.

![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-410%20passing-2e7d32)
![Status](https://img.shields.io/badge/status-in%20development-b8860b)
![LLM](https://img.shields.io/badge/LLM-Groq%20%C2%B7%20Ollama%20%C2%B7%20Gemini%20%C2%B7%20Claude-D97757)
![Cloud](https://img.shields.io/badge/deploy-Cloud%20Run%20%2B%20Firebase-4285F4)

[How it works](#how-it-works) · [Architecture](#architecture) · [Quick start](#quick-start) · [Status](#status) · [Roadmap](#roadmap)

</div>

---

## ✨ What it does

Someone plays **Jenga** or builds a **jigsaw** in front of a camera. Third Eye follows along and
flags the moment they break the procedure:

| 🚫 Alert | Example |
| :-- | :-- |
| **Illegal move** | Pulling a block from the top complete layer |
| **Out of order** | Filling the middle before the border is done |
| **Missed step** | Finishing with a gap nobody filled |
| **Wrong part** | Putting in a piece that belongs somewhere else |
| **Wrong position / rotation** | Right piece, wrong spot or turned the wrong way |

You don't write any rules by hand. Give it the **manual or rulebook as text**, and an LLM turns
it into the rules the checker enforces.

---

<a id="how-it-works"></a>

## 💡 How it works

<table>
<tr>
<td width="33%" valign="top">

### 📖 Read
**LLM compiler**, offline, once per procedure

Turns the manual into a *procedure spec*: what must be true before each move, and what the move
should change.

</td>
<td width="33%" valign="top">

### 👁️ Watch
**Computer vision**, every frame

Tracks hands and parts and keeps a live *world state*: which Jenga slots are full, which jigsaw
cells are filled.

</td>
<td width="33%" valign="top">

### ⚖️ Judge
**Deterministic checker**

Compares what the camera sees against the spec. Every alert comes from an explicit rule, never
from the LLM.

</td>
</tr>
</table>

> [!NOTE]
> **The LLM never sees video.** It only reads text: it compiles the manual, and later explains
> the session. All visual judgement is made by CV models, and every alert can be traced back to
> the rule that raised it.

---

## 🎯 Procedures

| | Procedure | Status | Why it's a good test |
| :-: | :-- | :-: | :-- |
| 🟫 | **Jenga** | 🟢 Active | **No step order, only rules.** Tests whether the system handles constraints, not just sequences. All blocks look alike, so any failure is a reasoning failure, not a detection one. |
| 🧩 | **Jigsaw** | 🟢 Active | **A grid of pieces.** Pieces are identified by matching against the box picture, so no training is needed. Its instructions are written as ordered steps (corners → edges → interior). |
| 🧱 | **LEGO** | ⏸️ Parked | The original demo. Code and tests are kept; set aside for now. |

### 📷 Hardware

- **Jenga:** an iPhone Pro with **LiDAR** at one corner of the tower (colour + depth), plus a
  second camera at the opposite corner. Depth makes empty and pushed-out slots easy to see even
  though every block is the same colour.
- **Jigsaw:** one overhead **webcam** above a board with printed ArUco markers at the corners.

---

<a id="architecture"></a>

## 🏗️ Architecture

```mermaid
flowchart LR
    subgraph OFF["📖 Offline · once per procedure"]
        direction TB
        M["Manual / rulebook"] --> L["LLM compiler"]
        L --> V["Validator"]
        V --> G[("Procedure spec")]
    end

    subgraph ON["👁️ Online · every frame"]
        direction TB
        CAM["📷 Cameras<br/>iPhone LiDAR · webcam"] --> K["Calibration"]
        K --> HP["Hands + parts"]
        HP --> S["World state<br/>Jenga · Jigsaw · LEGO"]
    end

    G --> C{"⚖️ Checker"}
    S --> C
    C --> A["🚨 Live alerts<br/>+ checklist"]
    C --> R["📝 Session report"]

    classDef done fill:#d4edda,stroke:#2e7d32,color:#1b3d20
    classDef todo fill:#fff3cd,stroke:#b8860b,color:#4d3800
    class L,V,G,K,HP,S,C,A,R,CAM done
```

<div align="center"><sub>🟩 built (real-camera tuning still to do: see Status)</sub></div>

**Deployment.** The cameras, perception and checker run on an **edge laptop**; the UI, specs,
session history and replays run on **Google Cloud** (FastAPI on Cloud Run, Firestore + Cloud
Storage, Firebase sign-in). The edge sends text only: protocol messages and the final log.

<details>
<summary><b>🔍 Design details</b></summary>

<br>

**One checker for every procedure.** The world state is an interface (`state/base.py`) with one
implementation per procedure. The checker never asks which one it has, so adding a new procedure
means writing a state tracker. The checker and compiler stay the same.

**Two kinds of spec:**

| Form | For | Looks like |
| :-- | :-- | :-- |
| `DagSpec` | Ordered procedures (jigsaw steps, LEGO) | Steps with parts, poses and dependencies |
| `ConstraintSpec` | Rule-based procedures (Jenga) | Actions with preconditions on the current state |

Both are lowered to the same list of actions with preconditions, so the checker has a single
code path.

**Safe rules.** Preconditions are small expressions like `src.layer < top_layer`. They're checked
against a whitelist when the spec is compiled and evaluated by an AST walker. There is no
`eval` anywhere.

**No guessing.** When the camera can't see enough to judge a move, the checker waits for a
clearer view instead of raising an alert. An alert is a claim about the user, so it never comes
from a failure of our own.

</details>

---

<a id="quick-start"></a>

## 🚀 Quick start

**1. Install** (Python 3.11+)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # core + tests
```

<details>
<summary>Optional extras</summary>

```bash
uv pip install -e ".[live]"     # camera path: mediapipe, Record3D, the server
uv pip install -e ".[server]"   # just the web server (replays, specs, history)
uv pip install -e ".[cloud]"    # Firestore, Cloud Storage, Firebase token checks
uv pip install -e ".[llm]"      # only for Claude; Groq, Gemini, Ollama need nothing
./scripts/fetch_models.sh       # MediaPipe hand model → models/
```

</details>

**2. Run the tests**, no hardware needed

```bash
pytest
```

**3. Replay a Jenga game** through the real tracker and checker

```bash
python -m thirdeye.replay scripts/scenarios/jenga_fouls.yaml
```
```text
jenga_game: 6 moves judged, 6 alerts, 0 warnings
  ended: game_over
```

or a jigsaw with an interior piece put in before the border is done:

```bash
python -m thirdeye.replay scripts/scenarios/jigsaw_early_interior.yaml
```
```text
[  11.60] ! out_of_order: The instructions finish the corners and the edges before starting on the interior. ...
jigsaw_4x3: 11/12 steps done, 1 alerts, 0 warnings
```

A spec for any rectangular jigsaw: `python -m thirdeye.compiler.jigsaw 6 4 -o specs/jigsaw_6x4.json`.

**4. Watch it in the browser**: a replay, streamed live to the web UI

```bash
(cd web && npm ci && npm run build)    # the React UI, once
python -m thirdeye.server.app          # then open http://127.0.0.1:8000
```

Pick a scenario and press **Replay**. The checklist ticks, alerts appear, and a Jenga game shows
the tower with the legal slots outlined. The report appears at the end and the session lands in
**History**. **Procedures** compiles instructions with the LLM and lets you edit and re-validate
the spec. (`cd web && npm run dev` gives a hot-reloading UI against the same server.)

**5. Compile a manual into a spec**

```bash
echo 'GROQ_API_KEY=...' >> .env        # untracked; or pick another provider with --model
python -m thirdeye.compiler.compile manuals/eval_corpus/jenga_standard.txt --kind constraints -o specs/jenga.json
python -m thirdeye.compiler.jigsaw 6 4 --instructions manuals/eval_corpus/jigsaw_border_first.txt -o specs/j.json
python -m thirdeye.compiler.compile manual.txt --kind dag -o s.json --model ollama:qwen3:8b   # local
```

`--model` takes `groq:<model>` (default `groq:openai/gpt-oss-120b`), `ollama:<model>`,
`gemini:<model>` or `claude:<model>`. For a jigsaw the LLM only reads the *order* (as phases);
positions come from the board size.

**6. Measure it**

```bash
python -m thirdeye.eval.errors --scenarios scripts/scenarios        # P/R per error, false alerts, delay
python -m thirdeye.eval.robustness                                  # same, under simulated detector noise
python -m thirdeye.eval.compiler --compile                          # compiler accuracy (~20 API calls)
python -m thirdeye.eval.faithfulness                                # LLM report vs log, LLM-judged
python -m thirdeye.eval.latency data/logs                           # FPS + per-stage ms over sessions
python -m thirdeye.eval.vlm_baseline run.mp4 --spec S --manual M -o vlm.json   # VLM baseline
python -m thirdeye.explain.report run.json [--llm]                  # session report from a log
```

Once real runs are labelled (format in `thirdeye/eval/labels.py`):
`python -m thirdeye.eval.errors --labels data/labels --logs data/logs`.

> [!TIP]
> Scenario scripts describe the *scene* over time (what's on the table, where the hands are),
> not the events, so the replay exercises the same code a live camera does. Generate one per
> planted error with `scripts/make_scenarios.py`.

<details>
<summary><b>📷 Live camera: jigsaw (and LEGO, parked)</b></summary>

```bash
# jigsaw: start with the board empty; the first clear frame is taken as the empty board
python -m thirdeye.live --spec specs/jigsaw_4x3.json --reference box.jpg
# ...and record the run while it is judged, with the run id its label file will use
python -m thirdeye.live --spec specs/jigsaw_4x3.json --reference box.jpg \
    --record data/raw/jigsaw_R001.mp4 --run jigsaw_R001 --log data/logs/jigsaw_R001.json

# LEGO (parked)
python scripts/make_markers.py markers.png       # print at 100%, tape at the plate corners
python -m thirdeye.live --spec specs/model_a.json --zero-shot
```

Jenga (iPhone LiDAR via Record3D, one or two corners, ArUco mat around the tower):

```bash
python -m thirdeye.perception.record3d game.r3d            # what a recording holds
python -m thirdeye.live_jenga --spec specs/example_constraints.json --stream 0 [--stream 1]
python -m thirdeye.live_jenga --spec specs/example_constraints.json --r3d A.r3d --r3d B.r3d \
    --log data/logs/jenga_R001.json
```

Baselines on the same runs: `live.py --colour-only` (jigsaw matching by colour alone) and
`live_jenga.py --rgb-only` (no depth).

</details>

<details>
<summary><b>☁️ Cloud deployment and the edge laptop</b></summary>

```bash
gcloud auth login
PROJECT=<gcp-project> ALLOWED=you@x.com,teammate@y.com ./deploy/deploy.sh   # Cloud Run, Firestore, GCS, secrets
# on the demo laptop, with THIRDEYE_EDGE_TOKEN in .env (the script says where to get it):
python -m thirdeye.server.edge --cloud https://<service>.run.app --scenario jenga_fouls   # test the link
python -m thirdeye.server.edge --cloud https://<service>.run.app \
    --spec specs/jigsaw_4x3.json --reference box.jpg --camera 0
```

Sign-in needs one manual step in the Firebase console (Google provider, a web app whose config
goes in `web/.env.production`), then `FIREBASE=1 ./deploy/deploy.sh`. `deploy/hosting.sh` puts the
UI on Firebase Hosting as well. Server settings: `THIRDEYE_STORE` (`local` | `firestore`),
`THIRDEYE_AUTH` (`none` | `firebase`), `THIRDEYE_ALLOWED_EMAILS`, `THIRDEYE_EDGE_TOKEN`.

</details>

---

<a id="status"></a>

## 📊 Project status

| Component | | Notes |
| :-- | :-: | :-- |
| Spec schema + validator | ✅ | Cycles, collisions, safe expressions |
| LLM compiler | ✅ | Groq / Ollama / Gemini / Claude; structured output + one repair round; jigsaw instructions → phases |
| Compiler accuracy | ✅ | **Groq gpt-oss-120b: steps 1.00, order P/R 0.98/1.00, rules 1.00**; 15/16 compiled |
| Checker engine | ✅ | One loop for both spec forms; looks again before an out-of-order alert |
| Jenga lattice + collapse detection | ✅ | |
| Jenga depth perception | 🟨 | Two corner depth views → slot occupancy. On real iPhone data, one camera finds 3 of 10 pulls at 75 % precision; two corners not yet recorded ([eval](docs/Jenga_Eval.md)) |
| Jenga live loop | ✅ | Record3D stream or `.r3d` → tower → lattice, hands, collapse; synthetic-tested |
| iPhone LiDAR input | 🟨 | Record3D `.r3d` reader built; format to confirm on a real recording; live USB stream untested |
| Calibration + hand tracking | ✅ | ArUco homography, MediaPipe |
| Session log + scripted replay | ✅ | Replay can simulate detector noise |
| Jigsaw spec, phases + board tracker | ✅ | Corners → edges → interior as phases; scripted replays |
| Jigsaw piece matching | ✅ | Against the box picture; synthetic-tested, not yet on real photos |
| Jigsaw live loop | ✅ | Webcam → matcher → tracker, with `--record`; tested on a synthetic video |
| Session report | ✅ | From the log only; LLM findings must cite a logged alert |
| Server + web UI | ✅ | FastAPI + WebSocket; React UI: live view, procedures (compile/edit/validate), history |
| Cloud deployment | 🟨 | Store, sign-in, edge link, Dockerfile and deploy scripts built; not yet deployed (needs the team's GCP project) |
| Evaluation scripts | ✅ | Errors, noise sweep, compiler accuracy, report faithfulness, latency, VLM + colour-only + RGB-only baselines |
| Compiler eval corpus | ✅ | 16 manuals with ground truth (LEGO, jigsaw, Jenga variants); not yet run on a real LLM |
| Step recognizer | ⬜ | Demoted to Could (plan v2.0 §2.5) |

---

<a id="roadmap"></a>

## 🗺️ Roadmap

<details open>
<summary><b>🟫 Jenga</b></summary>

- [x] Read colour + depth from Record3D exports (`perception/record3d.py`)
- [x] Diagonal-corner views, two cameras merged (`perception/tower.py`)
- [x] Empty and pushed-out slot detection from depth
- [x] Live loop: Record3D stream → tower reading → lattice (`live_jenga.py`)
- [x] Jenga rulebook → compiled spec, on a real LLM (4 rulebooks, all exact)
- [x] Depth margins on real iPhone data (teammate's Oct 15 games, one camera): pose refinement, per-slot baseline, camera following, collapse hold ([eval](docs/Jenga_Eval.md))
- [x] Colour-only layer finder: 73 % exact, 86 % top-2 on 49 labelled pulls (`perception/layerfinder.py`)
- [ ] Record two-corner games with pull taps (`scripts/mark_pulls.py`) and score them (`scripts/eval_framedir.py --r3d`)
- [ ] Check the `.r3d` reader on a real Record3D export

</details>

<details open>
<summary><b>🧩 Jigsaw</b></summary>

- [x] Jigsaw spec: grid, pieces, ordered steps and rules
- [x] Piece-grid state tracker
- [x] Piece position + rotation by matching against the box picture
- [x] Live loop: webcam → matcher → tracker (`live.py`)
- [x] Jigsaw instructions → phases, through the LLM
- [ ] Tune the matcher on real photos (`python -m thirdeye.perception.pieces`)
- [ ] Record and label real runs

</details>

<details>
<summary><b>🔧 Shared</b></summary>

- [x] LLM session report (`explain/`)
- [x] FastAPI + WebSocket server and web UI
- [x] Evaluation: precision/recall per error, false alerts per run, alert delay, noise sweep
- [x] Compiler eval corpus + scorer, run on Groq (and local Ollama)
- [x] React UI, cloud store, sign-in, edge link, deploy scripts
- [x] Report faithfulness, latency pooling, VLM / colour-only / RGB-only baselines
- [x] `docs/` updated to jigsaw + Jenga (plan v2.0, requirements v2.0)
- [ ] Deploy to the team's Google Cloud project
- [ ] Step recognizer (`temporal/`), now a Could

</details>

<details>
<summary><b>🔭 Later</b></summary>

- [ ] Read picture-only instructions (official LEGO booklets) with a vision model: step order
      and parts from the pages, positions from LDraw files or one demonstrated build

</details>

---

## 📁 Repository layout

<details>
<summary>Show tree</summary>

```
thirdeye/
├── compiler/      📖 schema · validator · safe expressions · LLM compiler · jigsaw phases
├── perception/    👁️ calibration · hands · detection · jigsaw pieces · tower depth · Record3D
├── state/
│   ├── base.py       WorldState interface
│   ├── jenga/        tower lattice · collapse detection
│   ├── jigsaw/       piece board
│   └── lego/         stud grid · build tracker   (parked)
├── checker/       ⚖️ engine · expression evaluator · rule sets
├── eval/          📏 labels · errors · noise sweep · compiler · faithfulness · latency · VLM baseline
├── explain/       📝 session report
├── server/        🌐 FastAPI app · protocol · runner · store · auth · edge link
├── session.py     joins a tracker and the checker for one run
├── sessionlog.py  JSON log, the report's only input
├── replay.py      scripted sessions through the real pipeline (optionally noisy)
├── live.py        webcam in, alerts out (jigsaw; LEGO parked)
└── live_jenga.py  corner depth cameras in, alerts out
web/               the React UI (built into web/dist, served by the app)
ui/                the old one-page UI, used when there is no build
deploy/            Cloud Run + Firebase deploy scripts
specs/             procedure specs (JSON)
manuals/           text manuals · eval_corpus/ with ground-truth specs
scripts/           markers · scenario generator · eval corpus generator · model fetch
tests/             unit + scenario tests
docs/              project plan + requirements
```

Still empty: `temporal/`, `configs/`.

</details>

---

## 📚 Documentation

| | |
| :-- | :-- |
| 📋 [Project plan](docs/Third_Eye_Project_Plan.md) | v2.0: jigsaw + Jenga, architecture, deployment, evaluation, timeline, risks |
| ✅ [Requirements](docs/Third_Eye_Requirements.md) | v2.0 change record (§0) over the numbered `FR-*` / `NFR-*` IDs cited in the code |
