"""Baseline: an end-to-end vision-language model, shown frames and the manual (EV-BL-2).

    python -m thirdeye.eval.vlm_baseline data/raw/jigsaw_R012.mp4 --spec specs/jigsaw_4x3.json \
        --manual manuals/eval_corpus/jigsaw_border_first.txt --run jigsaw_R012 \
        -o data/vlm_logs/jigsaw_R012.json
    python -m thirdeye.eval.vlm_baseline data/raw/jenga_R003_A.r3d --spec specs/example_constraints.json \
        --manual manuals/eval_corpus/jenga_standard.txt --run jenga_R003 -o data/vlm_logs/jenga_R003.json
    python -m thirdeye.eval.errors --labels data/labels --logs data/vlm_logs     # scored like ours

The comparison the plan asks for: no calibration, no tracker, no rules -- a
vision model sees evenly sampled frames (each stamped with its time), reads the
same manual or rulebook the compiler read, and lists the mistakes it sees. Its
answer is written as a session log with the checker's own alert codes and
targets, so `eval.errors` scores it against the same labels, by the same
matching, as the system.

We do not predict the outcome (plan 7). It may beat the system on missed
steps, which are visible in the last frame, and lose on timing and position.

The model must support images: e.g. `ollama:gemma4:e4b` (the default, local and free),
`ollama:gemma3:4b`, or `gemini:gemini-2.5-flash`. (Groq lists no vision model on
this account at the time of writing.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from thirdeye.compiler.schema import ConstraintSpec, DagSpec, load_spec
from thirdeye.events import Alert
from thirdeye.sessionlog import SessionRecord

DEFAULT_MODEL = "ollama:gemma4:e4b"
#: Placement errors the checker can raise, in its own words.
DAG_CODES = {
    "missed_step": "a step was never done by the end",
    "out_of_order": "a step was done before one the instructions put first",
    "wrong_part": "a piece or part that belongs elsewhere was put down",
    "wrong_position": "the right piece, in the wrong place",
    "wrong_rotation": "the right piece and place, turned the wrong way",
    "extra_part": "something was put down that is not in the instructions",
    "removed_part": "a part that was correctly placed was taken away",
}


class Seen(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    code: str
    target: str
    explanation: str


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    errors: list[Seen]


def sample(path: Path, n: int = 12, width: int = 640) -> list[tuple[float, bytes]]:
    """`n` evenly spaced frames as (seconds, JPEG), from a video or a `.r3d`."""
    if path.suffix.lower() == ".r3d":
        from thirdeye.perception.record3d import R3DRecording

        rec = R3DRecording(path)
        idx = np.linspace(0, len(rec) - 1, min(n, len(rec))).round().astype(int)
        frames = [(i / rec.fps, rec.frame(int(i)).colour) for i in idx]
    else:
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total <= 0:
            raise ValueError(f"cannot read frames from {path}")
        frames = []
        for i in np.linspace(0, total - 1, min(n, total)).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, img = cap.read()
            if ok:
                frames.append((i / fps, img))
        cap.release()
    out = []
    for t, img in frames:
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        cv2.putText(img, f"t={t:.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            out.append((float(t), buf.tobytes()))
    return out


def vocabulary(spec: DagSpec | ConstraintSpec) -> tuple[dict[str, str], list[str]]:
    """The codes the model may use, and the targets it may name."""
    if isinstance(spec, ConstraintSpec):
        codes = {p.violation: p.text for a in spec.actions for p in a.preconditions
                 if getattr(p, "violation", None)}
        codes["game_over"] = "the tower fell"
        return codes, ["move"]
    return dict(DAG_CODES), [s.id for s in spec.steps]


def prompt(manual: str, spec, frames: list[tuple[float, bytes]]) -> tuple[str, str]:
    codes, targets = vocabulary(spec)
    listed = "\n".join(f"- {c}: {d}" for c, d in codes.items())
    if isinstance(spec, DagSpec) and spec.board is not None:
        where = (f"The board is a {spec.board.cols} x {spec.board.rows} grid; piece rRcC belongs "
                 "in row R, column C, counted from the top-left.")
    else:
        where = ""
    system = (
        "You watch a person carry out a task, through still frames from one camera, and "
        "report every mistake they make against the instructions. Use only these error "
        f"codes:\n{listed}\n\nFor each mistake give the time `t` (seconds, from the frame "
        "where it first shows), the `code`, the `target` (one of: "
        f"{', '.join(targets[:60])}{' ...' if len(targets) > 60 else ''}) and one short "
        "`explanation`. If the task was done correctly, return no errors. Do not report a "
        "mistake you cannot see."
    )
    times = ", ".join(f"{t:.1f}s" for t, _ in frames)
    user = (f"Instructions:\n\n{manual.strip()}\n\n{where}\n\n"
            f"{len(frames)} frames follow, in order, at {times}. Each is stamped with its time.")
    return system, user


def run(video: Path, spec, manual: str, llm, n: int = 12, run_id: str = "") -> SessionRecord:
    from thirdeye.compiler.compile import _strict

    frames = sample(video, n)
    system, user = prompt(manual, spec, frames)
    raw = llm.complete(system, user, _strict(Answer.model_json_schema()),
                       images=[jpeg for _, jpeg in frames])
    answer = Answer.model_validate_json(raw)
    codes, _ = vocabulary(spec)
    alerts = [Alert(t=e.t, code=e.code, rule="vlm", text=e.explanation,
                    target="move" if isinstance(spec, ConstraintSpec) else e.target)
              for e in answer.errors if e.code in codes]
    return SessionRecord(
        procedure=spec.procedure, kind=spec.kind, alerts=alerts, log=[], checklist=[],
        outcome="game_over" if any(a.code == "game_over" for a in alerts) else None,
        meta={"run": run_id or video.stem, "baseline": "vlm", "model": getattr(llm, "name", ""),
              "frames": len(frames), "video": str(video),
              "dropped_codes": sorted({e.code for e in answer.errors} - set(codes))},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", type=Path, help="a recorded run (.mp4/.avi or .r3d)")
    ap.add_argument("--spec", type=Path, required=True, help="for the codes and targets only")
    ap.add_argument("--manual", type=Path, required=True, help="the instructions or rulebook")
    ap.add_argument("--run", default="", help="run id, as in the label file")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args(argv)

    from thirdeye.compiler.llm import make_llm

    record = run(args.video, load_spec(args.spec), args.manual.read_text(),
                 make_llm(args.model, max_tokens=4000), n=args.frames, run_id=args.run)
    for a in record.alerts:
        print(a)
    print(f"log: {record.write(args.out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
