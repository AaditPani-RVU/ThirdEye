"""The live jigsaw loop, end to end on a synthetic video (no camera, no hands model).

A board is filmed filling up piece by piece through the same tilted camera the
matcher tests use, with the planted mistake of an interior piece put down
before the border is finished. The video goes through `thirdeye.live` exactly
as a recorded run would.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from test_pieces import BOARD, CELL_FROM_FRAME, LAYOUT, REFERENCE, camera, flat_scene

from thirdeye.compiler.jigsaw import grid_spec
from thirdeye.live import main
from thirdeye.perception.calibrate import Calibrator
from thirdeye.perception.pieces import PieceMatcher
from thirdeye.perception.pipeline import JigsawPipeline
from thirdeye.session import Session

FPS = 10
#: Border first, except r1c1 (interior) goes down fourth.
ORDER = [(0, 0), (0, 3), (2, 0), (1, 1), (2, 3), (0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2),
         (1, 2)]


def frames(order=ORDER, hold_s: float = 1.0) -> list[np.ndarray]:
    """An empty board for a second, then each piece in turn, held `hold_s`."""
    out = [camera(flat_scene([]))] * FPS
    placed = []
    for rc in order:
        placed.append((rc, rc, 0))
        out += [camera(flat_scene(placed))] * int(hold_s * FPS)
    return out + [out[-1]] * FPS


def test_the_pipeline_captures_the_empty_board_then_reads_pieces() -> None:
    pipe = JigsawPipeline(Calibrator(LAYOUT), PieceMatcher(REFERENCE, BOARD))
    first = pipe.process(camera(flat_scene([])), 0.0)
    assert first.usable and first.captured_empty and first.readings == []
    later = pipe.process(camera(flat_scene([((0, 0), (0, 0), 0)])), 0.1)
    assert [str(r.part) for r in later.readings] == ["piece r0c0"]
    assert np.allclose(pipe.calibrator.current.homography / pipe.calibrator.current.homography[2, 2],
                       CELL_FROM_FRAME / CELL_FROM_FRAME[2, 2], atol=0.05)


def test_the_pipeline_drives_a_session_to_the_planted_error() -> None:
    session = Session(grid_spec(4, 3))
    pipe = JigsawPipeline(Calibrator(LAYOUT), PieceMatcher(REFERENCE, BOARD))
    alerts = []
    for i, frame in enumerate(frames()):
        r = pipe.process(frame, i / FPS)
        session.state.observe(i / FPS, r.readings, hands_present=r.hands_present)
        alerts += session.pump()
    alerts += session.finish(len(frames()) / FPS)
    assert [(a.code, a.target) for a in alerts] == [("out_of_order", "r1c1")]


def test_a_video_through_the_command_line_records_and_logs(tmp_path: Path, capsys) -> None:
    video, box = tmp_path / "run.avi", tmp_path / "box.png"
    cv2.imwrite(str(box), REFERENCE)
    seq = frames()
    h, w = seq[0].shape[:2]
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (w, h))
    for f in seq:
        out.write(f)
    out.release()
    spec = tmp_path / "jigsaw.json"
    spec.write_text(json.dumps(grid_spec(4, 3).model_dump(mode="json", exclude_none=True)))

    log, copy = tmp_path / "R001.json", tmp_path / "copy.avi"
    assert main(["--spec", str(spec), "--video", str(video), "--reference", str(box),
                 "--no-hands", "--no-show", "--log", str(log), "--record", str(copy)]) == 0
    text = capsys.readouterr().out
    assert "empty board captured" in text
    assert "out_of_order" in text and "11/12 steps done" in text   # r1c1 counts as an error
    data = json.loads(log.read_text())
    assert data["meta"]["run"] == "R001"
    assert [(a["code"], a["target"]) for a in data["alerts"]] == [("out_of_order", "r1c1")]
    assert cv2.VideoCapture(str(copy)).get(cv2.CAP_PROP_FRAME_COUNT) == len(seq)
