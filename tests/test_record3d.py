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
