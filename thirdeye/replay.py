"""Replay a scripted session through the real tracker and checker.

    python -m thirdeye.replay scripts/scenarios/model_a_swap.yaml [--log out.json]

A script describes the *scene*, not the events: which bricks are on the plate,
whether a hand is over it, which Jenga block was pulled. The replayer films
that scene at a fixed frame rate and feeds each frame to the same `observe` a
camera would, so the stability filter, the occlusion freeze and the move pairer
all do their real work. Scripting events directly would test the checker and
nothing else; the checker already has unit tests.

This is what makes the planted-error protocol of plan 5.1 runnable before any
hardware exists, and it is the harness the live path is checked against later:
a recorded run, once labelled, is just a script with real timings.

Script format (YAML)::

    spec: ../../specs/example_dag.json   # relative to the script
    fps: 10                              # optional, default 10
    end: 12.0                            # optional, default last keyframe + 2 s
    keyframes:
      - t: 0.0
        bricks: []                       # the whole LEGO scene, persists
      - t: 1.0
        hands: true                      # persists until changed
      - t: 2.0
        hands: false
        bricks: ["red_2x4 @ 0,0", "blue_2x2 @ 0,2,1 conf=0.3"]
      - t: 2.0                           # jigsaw: the same keys, with pieces
        add: ["r0c1 @ 1,0", "r2c3 @ 0,0,90"]
      - t: 3.0
        add: ["yellow_1x4 @ 4,4,0,90"]   # or edit the running scene
        remove: ["red_2x4 @ 0,0"]         # by cell
      - t: 3.0                           # Jenga: edits to the running tower
        pull: [5, 1]
        hands: 1                         # most hands on the tower since last reading
      - t: 6.0
        place: [18, 0]
      - t: 7.0
        collapse: true
      - t: 4.0                           # a one-shot event, e.g. a temporal hint
        event: {kind: STEP_START, source: temporal, part: red_2x4}

A brick is `color_WxH @ x,y[,layer[,rot]]` with an optional `conf=`. A jigsaw
piece is `name @ col,row[,rot]`, also with an optional `conf=`; a piece lies
flat, so it has no layer.

*Noise.* A script films a perfect detector. `--noise drop=0.1,ghost=0.05` films
an imperfect one instead -- parts that vanish for a frame, a part seen where
none is, a hand flag that flickers, frames lost -- so the stability filter and
the occlusion freeze are tested against the failures they exist for. Seeded, so
a failing run can be replayed exactly. Placement scenes only: the lattice takes
a settled reading from perception and has no filter of its own to test.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from thirdeye.compiler.schema import PartRef, PieceRef, Pose, load_spec
from thirdeye.events import Alert, Event, Slot
from thirdeye.session import Session
from thirdeye.sessionlog import SessionRecord
from thirdeye.state.jenga.lattice import TowerLattice, full_tower
from thirdeye.state.lego.build_state import BuildState
from thirdeye.state.lego.grid import Reading

_BRICK = re.compile(
    r"^\s*(?P<color>[a-z]+)_(?P<size>\d+x\d+)\s*@\s*(?P<pose>[\d,\s]+?)"
    r"(?:\s+conf=(?P<conf>[\d.]+))?\s*$"
)


_PIECE = re.compile(
    r"^\s*(?P<piece>[a-z0-9_]+)\s*@\s*(?P<pose>[\d,\s]+?)"
    r"(?:\s+conf=(?P<conf>[\d.]+))?\s*$"
)


class ScriptError(ValueError):
    """The script itself is malformed -- the author's bug, not a finding."""


class Stopped(Exception):
    """Raised from `on_frame` to end a replay early; the session still finishes."""


def part(text: str) -> PartRef | PieceRef:
    """`red_2x4` -> PartRef, the detector's own class naming (FR-PER-2); any
    other name is a jigsaw piece."""
    color, _, size = text.partition("_")
    if re.fullmatch(r"\d+x\d+", size):
        return PartRef(color=color, size=size)
    return PieceRef(piece=text)


def brick(text: str) -> Reading:
    """Parse one brick of a LEGO scene, or one jigsaw piece, into the reading a
    detector would give."""
    m = _BRICK.match(text)
    if m is None:
        return _piece(text)
    nums = [int(n) for n in m["pose"].replace(" ", "").split(",") if n]
    if not 2 <= len(nums) <= 4:
        raise ScriptError(f"brick {text!r} needs x,y and optionally layer and rot")
    x, y, layer, rot = (*nums, 0, 0)[:4]
    return Reading(
        part=PartRef(color=m["color"], size=m["size"]),
        pose=Pose(x=x, y=y, layer=layer, rot=rot),  # type: ignore[arg-type]
        residual=0.0,
        confidence=float(m["conf"]) if m["conf"] else 1.0,
    )


def _piece(text: str) -> Reading:
    m = _PIECE.match(text)
    if m is None:
        raise ScriptError(
            f"cannot read {text!r}; expected a brick 'red_2x4 @ x,y[,layer[,rot]]' "
            f"or a piece 'r0c1 @ col,row[,rot]'"
        )
    nums = [int(n) for n in m["pose"].replace(" ", "").split(",") if n]
    if not 2 <= len(nums) <= 3:
        raise ScriptError(f"piece {text!r} needs col,row and optionally rot")
    x, y, rot = (*nums, 0)[:3]
    return Reading(
        part=PieceRef(piece=m["piece"]),
        pose=Pose(x=x, y=y, rot=rot),  # type: ignore[arg-type]
        residual=0.0,
        confidence=float(m["conf"]) if m["conf"] else 1.0,
    )


def _slots(value: Any) -> list[Slot]:
    """`[5, 1]` or `[[5, 1], [6, 0]]` -> slots."""
    if not value:
        return []
    if isinstance(value[0], int):
        value = [value]
    return [Slot(int(layer), int(slot)) for layer, slot in value]


def _event(data: dict[str, Any], t: float) -> Event:
    fields = dict(data)
    if "part" in fields:
        fields["part"] = part(fields["part"])
    if "pose" in fields:
        fields["pose"] = Pose(**fields["pose"])
    for key in ("src", "dst"):
        if key in fields:
            fields[key] = Slot(*fields[key])
    return Event(t=float(fields.pop("t", t)), **fields)


@dataclass(frozen=True)
class Noise:
    """Per-frame detector failures, each a probability."""

    #: Each part on the plate goes unseen this frame.
    drop: float = 0.0
    #: A part is seen in a random empty cell this frame (a misdetection).
    ghost: float = 0.0
    #: The hand flag reads the opposite of the truth this frame.
    hands: float = 0.0
    #: The whole frame is lost (no reading at all).
    skip: float = 0.0
    seed: int = 0

    @classmethod
    def parse(cls, text: str) -> Noise:
        """`drop=0.1,ghost=0.05,seed=3` -> Noise."""
        kw: dict[str, float | int] = {}
        for item in filter(None, (x.strip() for x in text.split(","))):
            key, _, value = item.partition("=")
            if key not in cls.__dataclass_fields__:
                raise ScriptError(f"unknown noise {key!r}; use drop, ghost, hands, skip, seed")
            kw[key] = int(value) if key == "seed" else float(value)
        return cls(**kw)  # type: ignore[arg-type]

    def __bool__(self) -> bool:
        return any((self.drop, self.ghost, self.hands, self.skip))


@dataclass
class Replay:
    """A session driven from a script instead of a camera."""

    session: Session
    keyframes: list[dict[str, Any]]
    fps: float = 10.0
    end: float | None = None
    noise: Noise = field(default_factory=Noise)
    #: Called with the session time after every judged frame -- the server
    #: paces a replay to the wall clock and streams it from here. Raising
    #: `Stopped` ends the replay early.
    on_frame: Callable[[float], None] | None = None
    #: Alerts in the order they were raised, for the demo printout.
    raised: list[Alert] = field(default_factory=list, init=False)

    @classmethod
    def load(cls, path: str | Path, noise: Noise | None = None) -> Replay:
        path = Path(path)
        script = yaml.safe_load(path.read_text())
        if not isinstance(script, dict) or "spec" not in script:
            raise ScriptError(f"{path}: a script needs a 'spec' and 'keyframes'")
        spec = load_spec((path.parent / script["spec"]).resolve())
        frames = sorted(script.get("keyframes", []), key=lambda k: float(k["t"]))
        return cls(
            session=Session(spec),
            keyframes=frames,
            fps=float(script.get("fps", 10.0)),
            end=script.get("end"),
            noise=noise or Noise(),
        )

    def run(self) -> SessionRecord:
        end = self._end()
        try:
            if isinstance(self.session.state, BuildState):
                self._film_lego()
            elif isinstance(self.session.state, TowerLattice):
                self._film_jenga()
            else:  # pragma: no cover - a new state model needs a filming rule here
                raise ScriptError(f"cannot replay onto {type(self.session.state).__name__}")
        except Stopped:
            end = self.t
        self.t = end
        self.raised += self.session.finish(end)
        return SessionRecord.of(self.session)

    #: Session time of the last frame filmed.
    t: float = field(default=0.0, init=False)

    def _tick(self, t: float) -> None:
        self.t = t
        if self.on_frame is not None:
            self.on_frame(t)

    def _end(self) -> float:
        if self.end is not None:
            return float(self.end)
        return (float(self.keyframes[-1]["t"]) if self.keyframes else 0.0) + 2.0

    def _film_lego(self) -> None:
        """Film the scene at `fps`, holding each keyframe until the next."""
        state = self.session.state
        assert isinstance(state, BuildState)
        bricks: list[Reading] = []
        hands = False
        dt = 1.0 / self.fps
        stops = [float(k["t"]) for k in self.keyframes[1:]] + [self._end()]
        for key, stop in zip(self.keyframes, stops, strict=True):
            t = float(key["t"])
            if "bricks" in key:
                bricks = [brick(b) for b in key["bricks"] or []]
            gone = {brick(b).cell for b in key.get("remove") or []}
            bricks = [b for b in bricks if b.cell not in gone]
            bricks += [brick(b) for b in key.get("add") or []]
            hands = bool(key.get("hands", hands))
            if "event" in key:
                self.raised += self.session.submit(_event(key["event"], t))
            while t < stop - 1e-9:
                if not (self.noise and self._rng.random() < self.noise.skip):
                    seen, hand = self._corrupt(bricks, hands, state) if self.noise else (bricks, hands)
                    state.observe(t, seen, hands_present=hand)
                    self.raised += self.session.pump()
                self._tick(t)
                t += dt

    @property
    def _rng(self) -> random.Random:
        if not hasattr(self, "_rng_"):
            self._rng_ = random.Random(self.noise.seed)
        return self._rng_

    def _corrupt(
        self, bricks: list[Reading], hands: bool, state: BuildState
    ) -> tuple[list[Reading], bool]:
        """One frame as an imperfect detector would report it."""
        rng, n = self._rng, self.noise
        seen = [b for b in bricks if rng.random() >= n.drop]
        if rng.random() < n.ghost:
            parts = [s.part for s in self.session.spec.steps]  # type: ignore[union-attr]
            used = {b.cell for b in bricks}
            free = [
                (x, y) for x in range(state.grid.studs_x) for y in range(state.grid.studs_y)
                if (x, y, 0) not in used
            ]
            if parts and free:
                x, y = rng.choice(free)
                seen.append(Reading(part=rng.choice(parts), pose=Pose(x=x, y=y),
                                    residual=0.0))
        return seen, (not hands) if rng.random() < n.hands else hands

    def _film_jenga(self) -> None:
        """A lattice reading per keyframe that changes the tower.

        The lattice has no stability filter of its own -- perception hands it a
        settled reading after the hands leave -- so there is nothing to film
        between keyframes.
        """
        state = self.session.state
        assert isinstance(state, TowerLattice)
        rows = [list(r) for r in full_tower(slots=state.slots_per_layer)]
        for key in self.keyframes:
            t = float(key["t"])
            if "event" in key:
                self.raised += self.session.submit(_event(key["event"], t))
            touched = False
            for s in _slots(key.get("pull")):
                rows[s.layer][s.slot] = 0
                touched = True
            for s in _slots(key.get("place")):
                while len(rows) <= s.layer:
                    rows.append([0] * state.slots_per_layer)
                rows[s.layer][s.slot] = 1
                touched = True
            if key.get("collapse"):
                state.collapse_detected = True
                touched = True
            if touched:
                hands = key.get("hands")
                state.observe(rows, t=t, hands_in_contact=None if hands is None else int(hands))
                self.raised += self.session.pump()
            self._tick(t)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("script", type=Path)
    ap.add_argument("--log", type=Path, help="write the session log (IF-LOG-1) here")
    ap.add_argument("--run", help="run id for the log (default: the script's name)")
    ap.add_argument("--noise", type=Noise.parse, default=Noise(),
                    help="detector noise, e.g. drop=0.1,ghost=0.05,hands=0.02,skip=0.05,seed=1")
    ap.add_argument("-q", "--quiet", action="store_true", help="summary only")
    args = ap.parse_args(argv)

    replay = Replay.load(args.script, noise=args.noise)
    record = replay.run()
    record.meta["script"] = str(args.script)
    record.meta["run"] = args.run or args.script.stem
    if not args.quiet:
        for alert in replay.raised:
            print(alert)
        print()
    print(replay.session.summary())
    if args.log:
        print(f"log: {record.write(args.log)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

