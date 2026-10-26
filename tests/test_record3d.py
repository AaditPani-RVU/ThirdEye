"""Reading Record3D exports. The archives here are built to the documented
layout, so these tests prove the parsing, not the format (see the module doc)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic_tower import CORNER_A, CORNER_B, K, render

from thirdeye.perception.record3d import R3DRecording, main
from thirdeye.perception.tower import merge, read_view
from thirdeye.state.jenga.lattice import full_tower

liblzfse = pytest.importorskip("liblzfse")
DH, DW = 192, 256
RGB_W, RGB_H = 1024, 768
#: Colour intrinsics: the depth K scaled up 4x.
K_RGB = np.diag([4.0, 4.0, 1.0]) @ K


def write_r3d(path: Path, depths_mm: list[np.ndarray], with_conf: bool = True) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("metadata", json.dumps({
            "w": RGB_W, "h": RGB_H, "dw": DW, "dh": DH, "fps": 30,
            "K": K_RGB.T.ravel().tolist(),                     # column-major
        }))
        for i, d in enumerate(depths_mm):
            colour = np.full((RGB_H, RGB_W, 3), 120, np.uint8)
            z.writestr(f"rgbd/{i}.jpg", cv2.imencode(".jpg", colour)[1].tobytes())
            z.writestr(f"rgbd/{i}.depth",
                       liblzfse.compress((d / 1000.0).astype(np.float32).tobytes()))
            if with_conf:
                conf = np.full((DH, DW), 2, np.uint8)
                conf[:10] = 0                                  # a low-confidence band
                z.writestr(f"rgbd/{i}.conf", liblzfse.compress(conf.tobytes()))
    return path


def test_frames_come_back_in_metres_with_the_right_intrinsics(tmp_path: Path) -> None:
    occ = [list(r) for r in full_tower()]
    depth = render(occ, *CORNER_A)
    rec = R3DRecording(write_r3d(tmp_path / "a.r3d", [depth, depth]))
    assert len(rec) == 2 and rec.fps == 30 and rec.depth_size == (DW, DH)
    assert np.allclose(rec.K, K_RGB)
    f = rec.frame(1)
    assert f.t == pytest.approx(1 / 30)
    assert f.colour.shape == (RGB_H, RGB_W, 3)
    assert np.allclose(f.depth_m * 1000, depth, atol=0.01)


def test_a_recording_feeds_the_tower_reader(tmp_path: Path) -> None:
    """Intrinsics scaled from colour to depth resolution, metres to mm, low
    confidence blanked: the tower reads back exactly."""
    occ = [list(r) for r in full_tower()]
    occ[9][1] = 0
    a = R3DRecording(write_r3d(tmp_path / "a.r3d", [render(occ, *CORNER_A)]))
    b = R3DRecording(write_r3d(tmp_path / "b.r3d", [render(occ, *CORNER_B)]))
    va = a.frame(0).depth_view(*CORNER_A)
    vb = b.frame(0).depth_view(*CORNER_B)
    assert np.allclose(va.K, K) and (va.depth_mm[:10] == 0).all()
    reading = merge(read_view(va), read_view(vb))
    assert reading.occupancy() == occ


def test_the_command_line_describes_a_recording(tmp_path: Path, capsys) -> None:
    path = write_r3d(tmp_path / "a.r3d", [render([[1, 1, 1]] * 18, *CORNER_A)], with_conf=False)
    assert main([str(path), "--frame", "0", "-o", str(tmp_path / "f.png")]) == 0
    out = capsys.readouterr().out
    assert "1 frames at 30 fps" in out and "confidence maps: no" in out
    assert (tmp_path / "f.png").exists()


def _eval_script():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_framedir.py"
    spec = importlib.util.spec_from_file_location("eval_framedir", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_eval_feeds_two_corner_recordings_in_step(tmp_path: Path) -> None:
    depth = render([list(r) for r in full_tower()], *CORNER_A)
    a = write_r3d(tmp_path / "a.r3d", [depth] * 7)
    b = write_r3d(tmp_path / "b.r3d", [depth] * 5)
    got = list(_eval_script().r3d_feed([a, b], every=2))
    assert [round(t, 3) for _, t in got] == [0.0, round(2 / 30, 3), round(4 / 30, 3)]
    assert all(len(frames) == 2 and None not in frames for frames, _ in got)


def test_the_eval_matches_each_tap_to_one_removal_just_before_it() -> None:
    score = _eval_script().score
    removals = [{"t": 10.0}, {"t": 11.0}, {"t": 30.0}, {"t": 50.0}]
    res = score(removals, marks=[12.0, 31.0, 40.0], window=6.0)
    # 12 s takes the nearer of 10/11; 31 s takes 30; 40 s has nothing in reach.
    assert res["hit"] == 2 and res["delays"] == [1.0, 1.0]
    assert res["recall"] == round(2 / 3, 3) and res["precision"] == 0.5
