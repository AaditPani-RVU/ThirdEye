"""The live Jenga loop, on ray-cast depth frames from two corner cameras."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from synthetic_tower import CORNER_A, CORNER_B, K, render

from thirdeye.compiler.schema import load_spec
from thirdeye.live_jenga import JengaPipeline
from thirdeye.perception.hands import Hand
from thirdeye.perception.record3d import Frame
from thirdeye.session import Session
from thirdeye.state.jenga.lattice import full_tower

SPECS = Path(__file__).resolve().parents[1] / "specs"


def tower(*holes, extra=None):
    rows = [list(r) for r in full_tower()]
    for layer, slot in holes:
        rows[layer][slot] = 0
    return rows + (extra or [])


_RENDERED: dict[str, np.ndarray] = {}


def frames(occ, pushed=None, cameras=(CORNER_A, CORNER_B)) -> list[Frame]:
    """Ray casting is slow, and the same towers recur, so each is cast once."""
    out = []
    for i, (R, t) in enumerate(cameras):
        key = repr((occ, sorted((pushed or {}).items()), i))
        if key not in _RENDERED:
            _RENDERED[key] = render(occ, R, t, pushed) / 1000.0
        depth = _RENDERED[key]
        colour = np.zeros((*depth.shape, 3), np.uint8)
        out.append(Frame(t=0.0, colour=colour, depth_m=depth, K=K))
    return out


class FakeHands:
    def __init__(self, script):
        self.script = script

    def detect(self, frame, t):
        return self.script(t)


def run(pipeline, session, timeline):
    alerts, t = [], 0.0
    for occ, pushed, n in timeline:
        for _ in range(n):
            res = pipeline.process(frames(occ, pushed, cameras=CAMS[: pipeline.cameras]), t)
            if res.observed:
                alerts += session.pump()
            t += 0.1
    return alerts


CAMS = (CORNER_A, CORNER_B)


def new(cameras=2, hands=None):
    session = Session(load_spec(SPECS / "example_constraints.json"))
    pipe = JengaPipeline(session.state, cameras=cameras, hands=hands,
                         poses=list(CAMS[:cameras]), stable_frames=3)
    return session, pipe


def test_a_pull_from_the_top_complete_layer_is_called_once_settled() -> None:
    session, pipe = new()
    alerts = run(pipe, session, [
        (tower(), None, 4),
        (tower(), {(17, 1): 30.0}, 4),           # sliding out: not settled, not judged
        (tower((17, 1)), None, 4),
        (tower((17, 1), extra=[[1, 0, 0]]), None, 4),
    ])
    assert [a.code for a in alerts] == ["illegal_source_layer"]


def test_a_legal_move_raises_nothing() -> None:
    session, pipe = new()
    alerts = run(pipe, session, [
        (tower(), None, 4),
        (tower((5, 1)), None, 4),
        (tower((5, 1), extra=[[1, 0, 0]]), None, 4),
    ])
    assert alerts == []
    assert session.state.current()[5][1] == 0


def test_a_flicker_shorter_than_the_hold_is_ignored() -> None:
    session, pipe = new()
    run(pipe, session, [(tower(), None, 4), (tower((5, 1)), None, 2), (tower(), None, 4)])
    assert session.state.current()[5][1] == 1


def test_one_camera_still_reads_the_tower() -> None:
    session, pipe = new(cameras=1)
    run(pipe, session, [(tower(), None, 4), (tower((5, 0)), None, 4)])
    assert session.state.current()[5][0] == 0


def test_hands_hold_the_reading_and_their_count_rides_on_the_move() -> None:
    def two_hands_then_none(t):
        if 0.35 <= t < 0.8:   # both hands on the tower during the pull
            pts = np.tile([[128.0, 96.0]], (21, 1)) + np.random.default_rng(0).normal(0, 3, (21, 2))
            return [Hand(pts), Hand(pts + 2)]
        return []

    hands = [FakeHands(two_hands_then_none), FakeHands(lambda t: [])]
    session, pipe = new(hands=hands)
    alerts = run(pipe, session, [
        (tower(), None, 4),
        (tower((5, 1)), None, 9),
        (tower((5, 1), extra=[[1, 0, 0]]), None, 4),
    ])
    assert [a.code for a in alerts] == ["two_handed_move"]


def test_a_collapse_ends_the_game() -> None:
    session, pipe = new()
    run(pipe, session, [(tower(), None, 4), ([[1, 1, 1]] * 6, None, 4)])
    assert session.state.collapse_detected
    assert session.outcome is not None


def test_the_colour_only_baseline_drives_the_same_session() -> None:
    def shaded(occ):
        out = []
        for i, (R, t) in enumerate(CAMS):
            depth = frames(occ)[i].depth_m
            grey = np.clip(255 - (depth * 1000 - 300) / 2.5, 0, 255).astype(np.uint8)
            colour = np.repeat(grey[..., None], 3, axis=2)
            out.append(Frame(t=0.0, colour=colour, depth_m=np.zeros_like(depth), K=K))
        return out

    session = Session(load_spec(SPECS / "example_constraints.json"))
    pipe = JengaPipeline(session.state, poses=list(CAMS), stable_frames=3, rgb_only=True)
    t = 0.0
    for occ in [tower()] * 4 + [tower((5, 0))] * 4:   # an edge slot: a deep hole
        if pipe.process(shaded(occ), t).observed:
            session.pump()
        t += 0.1
    assert session.state.current()[5][0] == 0
