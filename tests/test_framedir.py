"""The iPhone frame-folder reader and the baseline-relative tower readings."""

from __future__ import annotations

import json

import cv2
import numpy as np

from thirdeye.events import Slot
from thirdeye.perception.framedir import FrameDir
from thirdeye.perception.tower import EndReading, baseline, merge, rebase


def write_frame(folder, ms, depth_m=0.4):
    stem = folder / str(ms)
    cv2.imwrite(f"{stem}_rgb.jpg", np.zeros((48, 64, 3), np.uint8))
    np.full((6, 8), depth_m, np.float32).tofile(f"{stem}_depth_8x6.bin")
    np.full((6, 8), 2, np.uint8).tofile(f"{stem}_conf_8x6.bin")
    (folder / f"{ms}_intrinsics.json").write_text(
        json.dumps({"fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0}))


def test_framedir_reads_frames_and_marks(tmp_path):
    write_frame(tmp_path, 1000)
    write_frame(tmp_path, 1500, depth_m=0.5)
    (tmp_path / "2000_mark.json").write_text(json.dumps({"mark": 1, "t_ms": 2000}))
    rec = FrameDir(tmp_path)
    assert len(rec) == 2
    f = rec.frame(1)
    assert f.t == 0.5 and f.depth_m.shape == (6, 8) and np.allclose(f.depth_m, 0.5)
    assert f.K[0, 0] == 50.0 and f.confidence is not None
    assert rec.marks() == [1.0]


def test_rebase_removes_a_per_slot_bias():
    a, b = Slot(0, 0), Slot(0, 1)
    seen = {a: EndReading("seen", 18.0), b: EndReading("seen", 40.0)}
    out = rebase(seen, {a: 17.0, b: 12.0})
    assert out[a].offset_mm == 1.0 and out[b].offset_mm == 28.0
    states = merge(out, {}, alone_mm=15.0).states
    assert states[a] == "block" and states[b] == "empty"


def test_baseline_leaves_out_the_air_above_the_tower():
    block, air = Slot(3, 0), Slot(14, 0)
    base = baseline({block: EndReading("seen", 17.0), air: EndReading("seen", 320.0),
                     Slot(2, 1): EndReading("occluded")})
    assert base == {block: 17.0}
    # Rebased, the empty slot keeps its depth and still reads as a hole.
    assert rebase({air: EndReading("seen", 320.0)}, base)[air].offset_mm == 320.0
