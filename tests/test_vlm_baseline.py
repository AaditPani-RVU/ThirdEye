"""The end-to-end VLM baseline: frames + manual in, a scoreable session log out."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from thirdeye.compiler.schema import load_spec
from thirdeye.eval.vlm_baseline import run, sample, vocabulary

SPECS = Path(__file__).resolve().parents[1] / "specs"


def _video(path: Path, seconds: int = 4, fps: int = 10) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    for i in range(seconds * fps):
        out.write(np.full((240, 320, 3), i * 2 % 255, np.uint8))
    out.release()
    return path


class Vision:
    name = model = "scripted-vision"

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def complete(self, system, user, schema, images=None):
        self.calls.append((system, user, images))
        return json.dumps(self.answer)


def test_frames_are_sampled_evenly_and_stamped(tmp_path) -> None:
    frames = sample(_video(tmp_path / "v.avi"), n=5)
    assert len(frames) == 5
    assert frames[0][0] == 0.0 and abs(frames[-1][0] - 3.9) < 1e-6
    assert all(jpeg[:2] == b"\xff\xd8" for _, jpeg in frames)


def test_the_answer_becomes_a_log_in_the_checkers_codes(tmp_path) -> None:
    spec = load_spec(SPECS / "jigsaw_4x3.json")
    llm = Vision({"errors": [
        {"t": 2.0, "code": "wrong_rotation", "target": "r1c2", "explanation": "turned"},
        {"t": 3.0, "code": "felt_wrong", "target": "r0c0", "explanation": "not a code"},
    ]})
    rec = run(_video(tmp_path / "v.avi"), spec, "corners, then edges", llm, n=4, run_id="R1")
    assert [(a.code, a.target) for a in rec.alerts] == [("wrong_rotation", "r1c2")]
    assert rec.meta["run"] == "R1" and rec.meta["dropped_codes"] == ["felt_wrong"]
    system, user, images = llm.calls[0]
    assert len(images) == 4 and "corners, then edges" in user and "out_of_order" in system
    assert "4 x 3 grid" in user


def test_a_rulebook_uses_its_own_violation_codes(tmp_path) -> None:
    spec = load_spec(SPECS / "example_constraints.json")
    codes, targets = vocabulary(spec)
    assert "illegal_source_layer" in codes and "game_over" in codes and targets == ["move"]
    llm = Vision({"errors": [{"t": 1.0, "code": "game_over", "target": "", "explanation": "fell"}]})
    rec = run(_video(tmp_path / "v.avi"), spec, "rules", llm, n=3)
    assert rec.outcome == "game_over" and rec.alerts[0].target == "move"
