"""Score the depth pipeline on a recorded game against the player's pull taps.

    # one iPhone frame folder (the teammate's capture app; taps in *_mark.json)
    .venv/bin/python scripts/eval_framedir.py <folder> --refine --relative

    # our rig: one Record3D export per corner, taps in a JSON list of seconds
    .venv/bin/python scripts/eval_framedir.py --r3d a.r3d --r3d b.r3d --marks pulls.json

Runs `JengaPipeline` (no hands) over the recording, logs every settled reading
that removed a block until a collapse ends the game, and matches those against
the taps: a tap is hit by a removal within `--window` seconds *before* it (the
player taps after the block is out) or 2 s after. Prints recall, precision and
the delay. `scripts/mark_pulls.py` writes the taps file while you play.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from thirdeye.live_jenga import JengaPipeline, _Recordings
from thirdeye.perception.framedir import FrameDir
from thirdeye.perception.record3d import Frame
from thirdeye.state.jenga.lattice import TowerLattice

Feed = Iterator[tuple[list[Frame | None], float]]


def folder_feed(rec: FrameDir, every: int = 1) -> Feed:
    for i in range(0, len(rec), every):
        f = rec.frame(i)
        yield [f], f.t


def r3d_feed(paths: list[Path], every: int = 1) -> Feed:
    recs = _Recordings(paths)
    while (got := recs.next()) is not None:
        if (recs.i - 1) % every == 0:
            yield got


def run(feed: Feed, cameras: int, stable: int = 5, refine: bool = False,
        relative: bool = False, alone: float = 20.0, protrude: float | None = None,
        repose: bool = False) -> tuple[list[dict], dict]:
    # Recorded games start mid-height, not at 18 layers: an empty lattice makes
    # the first settled reading the tower, so slots never seen are not blocks.
    lattice = TowerLattice(())
    pipe = JengaPipeline(lattice, cameras=cameras, hands=None, stable_frames=stable,
                         refine=refine, relative=relative, alone_mm=alone,
                         protrude_mm=protrude, repose=repose)
    seeded = False
    removals, stats = [], {"frames": 0, "posed": 0, "usable": 0, "duration": 0.0}
    for frames, t in feed:
        res = pipe.process(frames, t)
        stats["frames"] += 1
        stats["posed"] += res.posed > 0
        stats["usable"] += res.usable
        stats["duration"] = round(t, 2)
        if res.collapsed:  # the session ends here; the heap is not more pulls
            stats["collapse_t"] = round(pipe.collapse.t_collapse, 2)
            break
        if res.observed:
            events = lattice.diff()
            if not seeded:
                seeded = True
                continue
            for ev in events:
                if ev.kind == "REMOVE":
                    removals.append({"t": round(t, 2), "layer": ev.src.layer, "slot": ev.src.slot})
    return removals, stats


def score(removals: list[dict], marks: list[float], window: float) -> dict:
    used, delays, hit = set(), [], 0
    for m in marks:
        cand = [(abs(m - r["t"]), j) for j, r in enumerate(removals)
                if j not in used and -2.0 <= m - r["t"] <= window]
        if cand:
            _, j = min(cand)
            used.add(j)
            hit += 1
            delays.append(round(m - removals[j]["t"], 2))
    return {"marks": len(marks), "detected": len(removals), "hit": hit,
            "recall": round(hit / len(marks), 3) if marks else None,
            "precision": round(hit / len(removals), 3) if removals else None,
            "delays": delays}


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("folder", type=Path, nargs="?", help="an iPhone frame folder")
    src.add_argument("--r3d", type=Path, action="append", help="a Record3D export (one per corner)")
    ap.add_argument("--marks", type=Path, help="JSON list of pull times in seconds (with --r3d)")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--stable", type=int, default=5)
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--refine", action="store_true", help="correct the marker pose against depth")
    ap.add_argument("--relative", action="store_true", help="read slots against the first frame")
    ap.add_argument("--repose", action="store_true", help="follow the camera when it moves")
    ap.add_argument("--alone", type=float, default=20.0)
    ap.add_argument("--protrude", type=float, default=None)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    if a.r3d:
        if len(a.r3d) > 2:
            ap.error("one or two corner cameras")
        feed, cameras, name = r3d_feed(a.r3d, a.every), len(a.r3d), "+".join(p.stem for p in a.r3d)
        marks = json.loads(a.marks.read_text()) if a.marks else []
    else:
        rec = FrameDir(a.folder)
        feed, cameras, name, marks = folder_feed(rec, a.every), 1, a.folder.name, rec.marks()
    removals, stats = run(feed, cameras, a.stable, a.refine, a.relative, a.alone, a.protrude,
                          a.repose)
    res = score(removals, marks, a.window)
    print(json.dumps({"recording": name, **stats, **res, "removals": removals}, indent=1))
    if a.json:
        a.json.write_text(json.dumps({"stats": stats, "marks": marks, "score": res,
                                      "removals": removals}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
