"""The live session: camera in, checklist and alerts out (FR-SYS-1, FR-UI-4, plan 4.5).

    python -m thirdeye.live --spec specs/model_a.json --zero-shot
    python -m thirdeye.live --spec specs/model_a.json --weights runs/detect/best.pt
    python -m thirdeye.live --spec specs/model_a.json --zero-shot --video run07.mp4 --no-show
    python -m thirdeye.live --spec specs/jigsaw_4x3.json --reference box.jpg
    python -m thirdeye.live --spec specs/jigsaw_4x3.json --reference box.jpg \
        --record raw/jigsaw_R012.mp4 --run jigsaw_R012 --log logs/jigsaw_R012.json

The camera is read here, in the Python process (FR-UI-4); nothing travels to a
browser but what is drawn. This is the terminal-and-window version of the demo:
the same session the WebSocket server will drive, with an OpenCV window for the
overlay -- plate outline, detections, hands, a ghost of the next step, the
checklist and the alerts.

A jigsaw spec (one with a `board`) runs the piece matcher instead of a
detector: it needs the box picture (`--reference`), and a photo of the empty
board -- given with `--empty`, or taken from the first clear frame, so start
with the board empty.

`--record` saves the raw camera frames as they are judged, so a live session
is also a recording (plan 5): the saved video replays through `--video` into
the same judgements, and its log carries the run id its label file will use.

On exit (q, Ctrl-C, end of video) the session is finished, the missed-step
sweep runs, the log is written (IF-LOG-1), and the per-stage latency table is
printed so the plan 3.1 budget is measured on the machine that will run the
demo.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from thirdeye.compiler.schema import DagSpec, load_spec
from thirdeye.perception.calibrate import Calibrator, MarkerLayout
from thirdeye.perception.detect import Detector, YoloDetector, ZeroShotDetector
from thirdeye.perception.hands import DEFAULT_MODEL, HandDetector

MODELS = DEFAULT_MODEL.parent
from thirdeye.perception.pieces import PieceMatcher
from thirdeye.perception.pipeline import FrameResult, JigsawPipeline, TopDownPipeline
from thirdeye.session import Session
from thirdeye.sessionlog import SessionRecord
from thirdeye.state.lego.build_state import BuildState
from thirdeye.state.lego.grid import StudGrid, extent

RED, YELLOW, GREEN, CYAN, WHITE = (40, 40, 230), (0, 210, 255), (60, 200, 60), (230, 200, 0), \
    (240, 240, 240)


def _to_px(inv: np.ndarray, pts: list[tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(arr, inv).reshape(-1, 2).astype(np.int32)


def draw(frame: np.ndarray, result: FrameResult, session: Session, recent: list[str],
         grid: StudGrid) -> np.ndarray:
    """The overlay (FR-UI-1..3, FR-UI-5), drawn onto a copy of the frame."""
    out = frame.copy()
    cal = result.calibration
    status_colour = GREEN if cal.status == "ok" else YELLOW if cal.usable else RED
    cv2.putText(out, f"calibration: {cal.status}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                status_colour, 2)
    if result.hands_present:
        cv2.putText(out, "hand over plate - frozen", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    YELLOW, 2)

    if cal.usable and cal.homography is not None:
        inv = np.linalg.inv(cal.homography)
        plate = _to_px(inv, [(0, 0), (grid.studs_x, 0), (grid.studs_x, grid.studs_y),
                             (0, grid.studs_y)])
        cv2.polylines(out, [plate], True, CYAN, 1)
        nxt = session.checker.next_expected()
        if nxt is not None and nxt.part is not None and nxt.pose is not None:
            w, h = extent(nxt.part, nxt.pose.rot)
            p = nxt.pose
            ghost = _to_px(inv, [(p.x, p.y), (p.x + w, p.y), (p.x + w, p.y + h), (p.x, p.y + h)])
            cv2.polylines(out, [ghost], True, WHITE, 2, cv2.LINE_AA)
            cv2.putText(out, f"next: {nxt.id} {nxt.part}", tuple(int(v) for v in ghost[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    for det in result.detections:
        x0, y0, x1, y1 = (int(v) for v in det.box)
        cv2.rectangle(out, (x0, y0), (x1, y1), GREEN, 1)
    if cal.usable and cal.homography is not None and not result.detections:
        # No detector boxes (the jigsaw path): outline each identified cell.
        inv = np.linalg.inv(cal.homography)
        for r in result.readings:
            x, y = r.pose.x, r.pose.y
            cell = _to_px(inv, [(x + 0.1, y + 0.1), (x + 0.9, y + 0.1), (x + 0.9, y + 0.9),
                                (x + 0.1, y + 0.9)])
            colour = GREEN if r.confidence >= session.confidence_threshold else YELLOW
            cv2.polylines(out, [cell], True, colour, 1)
            name = str(r.part).removeprefix("piece ")
            cv2.putText(out, f"{name}/{r.pose.rot}", tuple(int(v) for v in cell[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    for hand in result.hands:
        for x, y in hand.keypoints.astype(int):
            cv2.circle(out, (int(x), int(y)), 2, YELLOW, -1)

    # Checklist and the latest alerts down the right-hand side.
    panel_x = out.shape[1] - 330
    cv2.rectangle(out, (panel_x - 10, 0), (out.shape[1], out.shape[0]), (20, 20, 20), -1)
    y = 22
    marks = {"done": ("[x]", GREEN), "error": ("[!]", RED), "active": ("[~]", YELLOW),
             "pending": ("[ ]", WHITE)}
    for step_id, status, _instruction in session.checklist():
        mark, colour = marks[status]
        cv2.putText(out, f"{mark} {step_id}", (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, 1)
        y += 18
    y += 10
    for line in recent[-6:]:
        colour = RED if " ! " in line else YELLOW
        cv2.putText(out, line[:48], (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        y += 16
    return out


def build_pipeline(
    spec: DagSpec, *, reference: Path | None = None, empty: Path | None = None,
    weights: str | None = None, zero_shot: bool = False,
    zero_shot_weights: str = str(MODELS / "yolov8s-worldv2.pt"), conf: float = 0.25,
    device: str | None = None, marker_size: float | None = None,
    marker_gap: float | None = None, hands: bool = True,
) -> TopDownPipeline | JigsawPipeline:
    """The perception pipeline a placement spec needs. Raises ValueError, with
    a message for the person at the keyboard, when something is missing."""
    hand_detector = HandDetector() if hands else None
    if spec.board is not None:
        if reference is None:
            raise ValueError("a jigsaw needs --reference, the box picture")
        picture = cv2.imread(str(reference))
        if picture is None:
            raise ValueError(f"cannot read {reference}")
        layout = MarkerLayout.around_plate(
            spec.board.cols, spec.board.rows,
            side_studs=1.0 if marker_size is None else marker_size,
            gap_studs=0.25 if marker_gap is None else marker_gap,
        )
        jigsaw = JigsawPipeline(Calibrator(layout), PieceMatcher(picture, spec.board),
                                hands=hand_detector)
        if empty is not None and not _set_empty(jigsaw, empty):
            raise ValueError(f"cannot calibrate on the empty-board photo {empty}")
        return jigsaw
    if not (weights or zero_shot):
        raise ValueError("a LEGO spec needs a detector: --weights or --zero-shot")
    assert spec.baseplate is not None
    grid = StudGrid.from_spec(spec.baseplate)
    layout = MarkerLayout.around_plate(
        grid.studs_x, grid.studs_y,
        side_studs=3.0 if marker_size is None else marker_size,
        gap_studs=0.5 if marker_gap is None else marker_gap,
    )
    detector: Detector = (YoloDetector(weights, conf=conf, device=device) if weights
                          else ZeroShotDetector(zero_shot_weights, conf=conf, device=device))
    return TopDownPipeline(Calibrator(layout), detector, grid, hands=hand_detector)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", type=Path, required=True)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0)
    src.add_argument("--video", type=Path, help="a recorded run instead of a live camera")
    det = ap.add_mutually_exclusive_group()
    det.add_argument("--weights", help="LEGO: fine-tuned YOLO weights (class names = part names)")
    det.add_argument("--zero-shot", action="store_true",
                     help="LEGO: YOLO-World fallback (FR-PER-3)")
    ap.add_argument("--zero-shot-weights", default=str(MODELS / "yolov8s-worldv2.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu; default picks the GPU")
    ap.add_argument("--reference", type=Path, help="jigsaw: the box picture")
    ap.add_argument("--empty", type=Path,
                    help="jigsaw: photo of the empty board (default: the first clear frame)")
    ap.add_argument("--marker-size", type=float, default=None,
                    help="printed marker side in grid units (default 3 studs; 1 cell)")
    ap.add_argument("--marker-gap", type=float, default=None,
                    help="gap between marker and grid, grid units (default 0.5 studs; 0.25 cell)")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    ap.add_argument("--log", type=Path, default=None, help="session log path (IF-LOG-1)")
    ap.add_argument("--run", help="run id for the log, as in its label file (default: log name)")
    ap.add_argument("--record", type=Path, help="also save the raw frames here (.mp4 or .avi)")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    if not isinstance(spec, DagSpec):
        ap.error("the live top-down loop runs placement procedures; Jenga needs the side rig")
    session = Session(spec)
    assert isinstance(session.state, BuildState)
    try:
        pipeline = build_pipeline(
            spec, reference=args.reference, empty=args.empty, weights=args.weights,
            zero_shot=args.zero_shot, zero_shot_weights=args.zero_shot_weights, conf=args.conf,
            device=args.device, marker_size=args.marker_size, marker_gap=args.marker_gap,
            hands=not args.no_hands,
        )
    except ValueError as exc:
        ap.error(str(exc))
    grid = pipeline.grid

    cap = cv2.VideoCapture(str(args.video) if args.video else args.camera)
    if not cap.isOpened():
        print(f"cannot open {'video ' + str(args.video) if args.video else 'camera'}",
              file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, frames, recent = time.monotonic(), 0, []
    t, blind = 0.0, 0
    writer = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.record is not None:
                writer = writer or _writer(args.record, fps, frame)
                writer.write(frame)
            # A file is timed by its frames, a camera by the clock.
            t = frames / fps if args.video else time.monotonic() - start
            frames += 1
            result = pipeline.process(frame, t)
            if getattr(result, "captured_empty", False):
                print(f"[{t:7.2f}] empty board captured; start placing pieces", flush=True)
            blind += not result.usable
            if result.usable:
                session.state.observe(t, result.readings, hands_present=result.hands_present)
                for alert in session.pump():
                    print(alert, flush=True)
                    recent.append(str(alert))
            if not args.no_show:
                cv2.imshow("Third Eye", draw(frame, result, session, recent, grid))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.seconds is not None and t >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    for alert in session.finish(t):
        print(alert)
    elapsed = time.monotonic() - start
    print()
    print(session.summary())
    print(f"\n{frames} frames in {elapsed:.1f} s ({frames / max(elapsed, 1e-9):.1f} FPS)")
    print(pipeline.timings.table())
    if blind:
        print(f"\n{blind} of {frames} frames had no usable calibration and were not judged"
              " -- are all four ArUco markers in view? (FR-CAL-7)")
    if args.log:
        record = SessionRecord.of(session, meta={
            "spec": str(args.spec), "frames": frames,
            "run": args.run or args.log.stem,
            "video": str(args.video or args.record or "") or None,
        })
        print(f"log: {record.write(args.log)}")
    return 0


def _set_empty(pipeline: JigsawPipeline, path: Path) -> bool:
    """Calibrate on the empty-board photo and remember it as the background."""
    photo = cv2.imread(str(path))
    if photo is None:
        return False
    cal = pipeline.calibrator.refresh(photo)
    if not cal.usable or cal.homography is None:
        return False
    pipeline.matcher.set_background(photo, cal.homography)
    return True


def _writer(path: Path, fps: float, frame: np.ndarray) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "MJPG" if path.suffix.lower() == ".avi" else "mp4v"
    h, w = frame.shape[:2]
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (w, h))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
