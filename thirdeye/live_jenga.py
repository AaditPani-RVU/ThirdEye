"""The live Jenga session: corner depth cameras in, rule alerts out (FR-PER-8, FR-STA-7, FR-STA-8).

    python -m thirdeye.live_jenga --spec specs/example_constraints.json --stream 0
    python -m thirdeye.live_jenga --spec specs/example_constraints.json --stream 0 --stream 1
    python -m thirdeye.live_jenga --spec specs/example_constraints.json \
        --r3d game07_cornerA.r3d --r3d game07_cornerB.r3d --log logs/jenga_R007.json

One or two iPhones running Record3D, at opposite corners of the tower (see
`perception/tower.py` for why opposite corners). Each frame:

    colour  -> tower pose from the mat markers (once per camera; the rig is fixed)
            -> hands (MediaPipe): a mask for the depth map, and a count on the tower
    depth   -> end-face readings per camera -> merged slot verdicts
            -> tower height -> collapse detector
            -> a settled, hand-free reading that holds for a few frames
               -> `TowerLattice.observe`, carrying the most hands seen since the last one

The lattice judges settled towers only, so the stability filter lives here: a
reading reaches it when no block is partway out, no hand touches the tower,
and the occupancy has been the same for `stable_frames` frames. With one camera
only, each slot is seen from one end; `merge` then demands a deeper recess
before calling a slot empty.

Two recordings are stepped in lockstep by frame index -- start both apps
together; the per-frame skew of a few tens of milliseconds is far below the
time a pull takes.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from thirdeye.compiler.schema import ConstraintSpec, load_spec
from thirdeye.perception.hands import HandDetector, hands_in_contact
from thirdeye.perception.pipeline import Timings
from thirdeye.perception.record3d import Frame, LiveStream, R3DRecording
from thirdeye.perception.tower import (
    STANDARD,
    DepthView,
    TowerGeometry,
    TowerReading,
    merge,
    outline,
    read_view,
    tower_pose,
)
from thirdeye.perception.tower_rgb import RgbTowerReader
from thirdeye.session import Session
from thirdeye.sessionlog import SessionRecord
from thirdeye.state.jenga.collapse import CollapseDetector
from thirdeye.state.jenga.lattice import TowerLattice

RED, YELLOW, GREEN, WHITE = (40, 40, 230), (0, 210, 255), (60, 200, 60), (240, 240, 240)


@dataclass
class JengaResult:
    t: float
    #: Cameras whose tower pose is known.
    posed: int
    reading: TowerReading | None = None
    hands: int = 0
    #: This frame's reading went to the lattice.
    observed: bool = False
    collapsed: bool = False

    @property
    def usable(self) -> bool:
        return self.reading is not None and self.reading.usable


@dataclass
class JengaPipeline:
    """Frames from one or two corner cameras -> lattice readings for a session."""

    lattice: TowerLattice
    cameras: int = 2
    hands: list[HandDetector] | None = None
    geom: TowerGeometry = STANDARD
    #: Frames an unchanged, settled, hand-free reading must hold.
    stable_frames: int = 5
    #: Hand pixels are blanked out of the depth map, grown by this much.
    hand_margin_px: int = 12
    poses: list[tuple[np.ndarray, np.ndarray] | None] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)
    #: Baseline: read the tower from colour alone (`tower_rgb.py`), ignoring depth.
    #: Each camera's first posed frame is taken as the full tower.
    rgb_only: bool = False

    def __post_init__(self) -> None:
        self.poses = self.poses or [None] * self.cameras
        self._rgb: list[RgbTowerReader | None] = [None] * self.cameras
        self.collapse = CollapseDetector(self.lattice)
        self._candidate: list[list[int]] | None = None
        self._held = 0
        #: Most hands on the tower in any frame since the last observation.
        self._max_hands = 0
        self._outline_layers = len(self.lattice.current()) + 1

    def process(self, frames: list[Frame | None], t: float) -> JengaResult:
        views: list[DepthView] = []
        rgb: list[TowerReading] = []
        hands_now = 0
        for i, frame in enumerate(frames):
            if frame is None:
                continue
            if self.poses[i] is None:
                t0 = time.perf_counter()
                self.poses[i] = tower_pose(frame.colour, frame.K)
                self.timings.add("pose", time.perf_counter() - t0)
            pose = self.poses[i]
            if pose is None:
                continue
            view = frame.depth_view(*pose)
            if self.hands is not None:
                t0 = time.perf_counter()
                seen = self.hands[i].detect(frame.colour, t)
                self.timings.add("hands", time.perf_counter() - t0)
                view.mask = self._hand_mask(seen, frame, view)
                region = self._colour_outline(frame, *pose)
                hands_now = max(hands_now, hands_in_contact(seen, region))
            if self.rgb_only:
                t0 = time.perf_counter()
                reader = self._rgb[i]
                if reader is None:
                    reader = self._rgb[i] = RgbTowerReader(frame.K, *pose, reference=frame.colour,
                                                           geom=self.geom)
                mask = None if view.mask is None else cv2.resize(
                    view.mask.astype(np.uint8), frame.colour.shape[1::-1],
                    interpolation=cv2.INTER_NEAREST)
                rgb.append(reader.read(frame.colour, mask))
                self.timings.add("tower", time.perf_counter() - t0)
            views.append(view)
        result = JengaResult(t=t, posed=sum(p is not None for p in self.poses), hands=hands_now)
        if not views:
            return result

        if self.rgb_only:
            reading = _merge_rgb(rgb)
        else:
            t0 = time.perf_counter()
            readings = [read_view(v, self.geom) for v in views]
            reading = merge(readings[0], readings[1] if len(readings) > 1 else {})
            self.timings.add("tower", time.perf_counter() - t0)
        result.reading = reading
        self._max_hands = max(self._max_hands, hands_now)

        if self.collapse.feed(t, reading.height_mm()):
            result.collapsed = True
            self.lattice.observe(self.lattice.current(), t=t,
                                 hands_in_contact=self._hands_or_none())
            result.observed = True
            return result
        if not reading.usable or not reading.settled or hands_now:
            self._candidate, self._held = None, 0
            return result
        occupancy = reading.occupancy(self.lattice.current(), self.geom.slots)
        if occupancy != self._candidate:
            self._candidate, self._held = occupancy, 1
            return result
        self._held += 1
        current = [list(r) for r in self.lattice.current()]
        if self._held == self.stable_frames and occupancy != current:
            self.lattice.observe(occupancy, t=t, hands_in_contact=self._hands_or_none())
            self._max_hands = 0
            self._outline_layers = len(occupancy) + 1
            result.observed = True
        return result

    def _hands_or_none(self) -> int | None:
        """No hand detector means we did not look, which is not zero hands."""
        return None if self.hands is None else self._max_hands

    def _colour_outline(self, frame: Frame, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        view = DepthView(frame.depth_m, frame.K, R, t)
        return outline(view, self._outline_layers)

    def _hand_mask(self, hands, frame: Frame, view: DepthView) -> np.ndarray | None:
        if not hands:
            return None
        h, w = frame.colour.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        for hand in hands:
            hull = cv2.convexHull(hand.keypoints.astype(np.float32)).astype(np.int32)
            cv2.fillConvexPoly(mask, hull, 255)
        if self.hand_margin_px:
            k = 2 * self.hand_margin_px + 1
            mask = np.asarray(cv2.dilate(mask, np.ones((k, k), np.uint8)), np.uint8)
        dh, dw = view.depth_mm.shape
        return cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST) > 0


def _merge_rgb(readings: list[TowerReading]) -> TowerReading:
    """Two colour verdicts per slot. A hole often shows from one side only (from
    the other it is a shallow recess that barely changes the picture), so a slot
    either camera sees empty is empty."""
    if len(readings) == 1:
        return readings[0]
    a, b = readings
    states = {}
    for slot in set(a.states) | set(b.states):
        sa, sb = a.states.get(slot, "unknown"), b.states.get(slot, "unknown")
        known = {sa, sb} - {"unknown"}
        states[slot] = "empty" if "empty" in known else known.pop() if known else "unknown"
    top = max((s.layer for s, st in states.items() if st == "block"), default=-1)
    return TowerReading(states, settled=True,
                        unknown=[s for s, st in states.items() if st == "unknown" and s.layer <= top])


def draw(frame: Frame, pose, result: JengaResult, session: Session, recent: list[str],
         layers: int) -> np.ndarray:
    """One camera's colour image with the tower outline, the slot verdicts this
    camera faces, the legal-move summary and the latest alerts."""
    out = frame.colour.copy()
    if pose is not None:
        view = DepthView(frame.depth_m, frame.K, *pose)
        cv2.polylines(out, [outline(view, layers).astype(np.int32)], True, WHITE, 1)
        if result.reading is not None:
            colours = {"block": GREEN, "empty": RED, "moving": YELLOW}
            for slot, state in result.reading.states.items():
                if state not in colours:
                    continue
                for centre, normal in STANDARD.ends(slot.layer, slot.slot).values():
                    if view.faces(normal, centre):
                        uv, _ = view.project(centre[None])
                        cv2.circle(out, tuple(int(v) for v in uv[0]), 5, colours[state], -1)
    status = ("collapsed" if getattr(session.state, "collapse_detected", False) else
              "hands on tower" if result.hands else
              "settled" if result.reading and result.reading.settled else "moving")
    cv2.putText(out, f"{status} | posed {result.posed}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, YELLOW if status != "settled" else GREEN, 2)
    y = 50
    for line in recent[-6:]:
        cv2.putText(out, line[:70], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
        y += 18
    return out


class _Recordings:
    def __init__(self, paths: list[Path]):
        self.recs = [R3DRecording(p) for p in paths]
        self.fps = self.recs[0].fps
        self.n = min(len(r) for r in self.recs)
        self.i = 0

    def next(self) -> tuple[list[Frame | None], float] | None:
        if self.i >= self.n:
            return None
        frames: list[Frame | None] = [r.frame(self.i) for r in self.recs]
        t = self.i / self.fps
        self.i += 1
        return frames, t


class _Streams:  # pragma: no cover - needs phones on USB
    def __init__(self, indices: list[int]):
        self.streams = [LiveStream(i) for i in indices]
        self.t0 = time.monotonic()

    def next(self) -> tuple[list[Frame | None], float] | None:
        frames = [s.next_frame() for s in self.streams]
        if all(f is None for f in frames):
            return None
        return frames, time.monotonic() - self.t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", type=Path, required=True, help="a constraint-form (Jenga) spec")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--r3d", type=Path, action="append", help="a Record3D export (one per corner)")
    src.add_argument("--stream", type=int, action="append",
                     help="a Record3D USB device index (one per corner)")
    ap.add_argument("--stable-frames", type=int, default=5)
    ap.add_argument("--rgb-only", action="store_true",
                    help="baseline: ignore depth, read the tower from colour alone")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--seconds", type=float, default=None, help="stop after this long")
    ap.add_argument("--log", type=Path, default=None, help="session log path (IF-LOG-1)")
    ap.add_argument("--run", help="run id for the log, as in its label file")
    args = ap.parse_args(argv)

    spec = load_spec(args.spec)
    if not isinstance(spec, ConstraintSpec):
        ap.error("live_jenga runs constraint-form specs; placement procedures use thirdeye.live")
    sources = args.r3d or args.stream
    if len(sources) > 2:
        ap.error("one or two corner cameras")
    session = Session(spec)
    lattice = session.state
    assert isinstance(lattice, TowerLattice)
    hands = None if args.no_hands else [HandDetector() for _ in sources]
    pipeline = JengaPipeline(lattice, cameras=len(sources), hands=hands,
                             stable_frames=args.stable_frames, rgb_only=args.rgb_only)
    feed = _Recordings(args.r3d) if args.r3d else _Streams(args.stream)

    start, frames, recent, t = time.monotonic(), 0, [], 0.0
    unposed_warned = False
    try:
        while (got := feed.next()) is not None:
            batch, t = got
            frames += 1
            t0 = time.perf_counter()
            result = pipeline.process(batch, t)
            if result.posed < len(sources) and frames > 30 and not unposed_warned:
                print(f"[{t:7.2f}] only {result.posed} of {len(sources)} camera(s) see two mat "
                      "markers; is the mat in view?", flush=True)
                unposed_warned = True
            if result.collapsed:
                print(f"[{t:7.2f}] tower collapsed", flush=True)
            if result.observed:
                for alert in session.pump():
                    print(alert, flush=True)
                    recent.append(str(alert))
            pipeline.timings.add("frame", time.perf_counter() - t0)
            if not args.no_show and batch[0] is not None:
                img = draw(batch[0], pipeline.poses[0], result, session, recent,
                           len(lattice.current()) + 1)
                cv2.imshow("Third Eye - Jenga", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if session.outcome is not None:
                break
            if args.seconds is not None and t >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_show:
            cv2.destroyAllWindows()

    for alert in session.finish(t):
        print(alert)
    elapsed = time.monotonic() - start
    print()
    print(session.summary())
    print(f"\n{frames} frames in {elapsed:.1f} s ({frames / max(elapsed, 1e-9):.1f} FPS)")
    print(pipeline.timings.table())
    if args.log:
        record = SessionRecord.of(session, meta={
            "spec": str(args.spec), "frames": frames, "run": args.run or args.log.stem,
            "video": ",".join(str(p) for p in (args.r3d or [])) or None,
            "latency": pipeline.timings.summary(), "fps": round(frames / max(elapsed, 1e-9), 2),
            "baseline": "rgb_only" if args.rgb_only else None,
        })
        print(f"log: {record.write(args.log)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
