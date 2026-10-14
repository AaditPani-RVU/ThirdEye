"""Piece identification against the box picture, on synthetic boards.

The box picture is generated, cut into cells, and the pieces laid on a drawn
board with its calibration markers, then seen through a tilted camera with a
lighting change and sensor noise. The truth is known, so every reading is
checked against it.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import pytest

from thirdeye.compiler.jigsaw import grid_spec
from thirdeye.compiler.schema import Board, PieceRef
from thirdeye.perception.calibrate import MarkerLayout
from thirdeye.perception.pieces import UNKNOWN, PieceMatcher, main
from thirdeye.session import Session

COLS, ROWS = 4, 3
BOARD = Board(cols=COLS, rows=ROWS)
CP = 80            # flat-drawing pixels per cell
MARGIN = 2         # cells of margin around the board
LAYOUT = MarkerLayout.around_plate(COLS, ROWS, side_studs=1.0, gap_studs=0.25)
TILT = np.array([[0.9, 0.1, 30.0], [-0.07, 0.88, 60.0], [0.00008, 0.0001, 1.0]])
BOARD_BGR = (70, 110, 60)

#: (piece home (row, col), where it lies (row, col), clockwise turn)
Placed = tuple[tuple[int, int], tuple[int, int], int]


def box_picture(seed: int = 7) -> np.ndarray:
    """A picture with something in every cell: soft colour blobs over a gradient,
    plus a few hard strokes, as printed jigsaw art has."""
    rng = np.random.default_rng(seed)
    h, w = ROWS * CP, COLS * CP
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img += np.stack([xx / w * 120, yy / h * 120, (1 - xx / w) * 90], axis=2)
    for _ in range(25):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(15, 60)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
        img += blob[..., None] * rng.uniform(-120, 160, 3)
    out = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(12):
        p1 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        p2 = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.line(out, p1, p2, colour, int(rng.integers(2, 6)))
    return out


REFERENCE = box_picture()


def tile(ref: np.ndarray, row: int, col: int) -> np.ndarray:
    return ref[row * CP:(row + 1) * CP, col * CP:(col + 1) * CP]


def flat_scene(placed: Sequence[Placed], ref: np.ndarray = REFERENCE,
               foreign: Sequence[tuple[int, int]] = ()) -> np.ndarray:
    size = ((COLS + 2 * MARGIN) * CP, (ROWS + 2 * MARGIN) * CP)
    img = np.full((size[1], size[0], 3), 245, np.uint8)
    x0, y0 = MARGIN * CP, MARGIN * CP
    img[y0:y0 + ROWS * CP, x0:x0 + COLS * CP] = BOARD_BGR
    dictionary = cv2.aruco.getPredefinedDictionary(LAYOUT.dictionary)
    side = int(LAYOUT.side_studs * CP)
    for mid, (sx, sy) in LAYOUT.centres.items():
        cx, cy = (sx + MARGIN) * CP, (sy + MARGIN) * CP
        mx, my = round(cx - side / 2), round(cy - side / 2)
        marker = cv2.aruco.generateImageMarker(dictionary, mid, side)
        img[my:my + side, mx:mx + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    for (hr, hc), (r, c), rot in placed:
        piece = np.rot90(tile(ref, hr, hc), -rot // 90)   # clockwise
        img[y0 + r * CP:y0 + (r + 1) * CP, x0 + c * CP:x0 + (c + 1) * CP] = piece
    other = box_picture(seed=99)
    for r, c in foreign:
        img[y0 + r * CP:y0 + (r + 1) * CP, x0 + c * CP:x0 + (c + 1) * CP] = tile(other, 1, 1)
    return img


def camera(flat: np.ndarray, gain: float = 1.0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    h, w = flat.shape[:2]
    seen = cv2.warpPerspective(flat, TILT, (w + 60, h + 120), borderValue=(245, 245, 245))
    out = seen.astype(np.float32) * gain
    if noise:
        out += np.random.default_rng(seed).normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


#: Frame pixels -> cell coordinates, the map calibration would recover.
CELL_FROM_FRAME = np.linalg.inv(
    TILT @ np.array([[CP, 0, MARGIN * CP], [0, CP, MARGIN * CP], [0, 0, 1]], float)
)


def matcher(background: bool = True) -> PieceMatcher:
    m = PieceMatcher(REFERENCE, BOARD)
    if background:
        m.set_background(camera(flat_scene([])), CELL_FROM_FRAME)
    return m


def found(matches) -> set[tuple[str, int, int, int]]:
    return {(m.name, m.row, m.col, m.rot) for m in matches}


# --- identification ----------------------------------------------------------------------

def test_every_piece_in_place_is_named_for_its_own_cell() -> None:
    placed = [((r, c), (r, c), 0) for r in range(ROWS) for c in range(COLS)]
    matches = matcher().read(camera(flat_scene(placed)), CELL_FROM_FRAME)
    assert found(matches) == {(f"r{r}c{c}", r, c, 0) for (r, c), _, _ in placed}
    assert min(m.confidence for m in matches) > 0.5


@pytest.mark.parametrize("rot", [90, 180, 270])
def test_a_turned_piece_reports_its_turn_clockwise(rot: int) -> None:
    matches = matcher().read(camera(flat_scene([((1, 2), (1, 2), rot)])), CELL_FROM_FRAME)
    assert found(matches) == {("r1c2", 1, 2, rot)}


def test_a_shuffled_board_is_read_exactly() -> None:
    rng = np.random.default_rng(3)
    homes = [(r, c) for r in range(ROWS) for c in range(COLS)]
    spots = [homes[i] for i in rng.permutation(len(homes))]
    turns = rng.choice([0, 90, 180, 270], len(homes))
    placed = [(h, s, int(t)) for h, s, t in zip(homes, spots, turns, strict=True)]
    frame = camera(flat_scene(placed), gain=0.8, noise=6.0)
    got = found(matcher().read(frame, CELL_FROM_FRAME))
    assert got == {(f"r{h[0]}c{h[1]}", s[0], s[1], t) for h, s, t in placed}


def test_lighting_and_noise_do_not_change_the_answer() -> None:
    placed = [((0, 0), (0, 0), 0), ((2, 3), (1, 1), 90)]
    for gain, noise in [(0.7, 0.0), (1.2, 0.0), (1.0, 10.0)]:
        frame = camera(flat_scene(placed), gain=gain, noise=noise, seed=int(gain * 10))
        assert found(matcher().read(frame, CELL_FROM_FRAME)) == {
            ("r0c0", 0, 0, 0), ("r2c3", 1, 1, 90)}, (gain, noise)


def test_empty_cells_are_not_read() -> None:
    matches = matcher().read(camera(flat_scene([((0, 1), (0, 1), 0)])), CELL_FROM_FRAME)
    assert [(m.row, m.col) for m in matches] == [(0, 1)]


def test_a_piece_from_another_puzzle_is_unknown_and_not_confident() -> None:
    frame = camera(flat_scene([], foreign=[(1, 1)]))
    [m] = matcher().read(frame, CELL_FROM_FRAME)
    assert m.name == UNKNOWN and m.confidence < 0.5


def test_without_a_background_only_good_matches_are_reported() -> None:
    frame = camera(flat_scene([((2, 0), (2, 0), 0)], foreign=[(0, 3)]))
    assert found(matcher(background=False).read(frame, CELL_FROM_FRAME)) == {("r2c0", 2, 0, 0)}


def test_plain_regions_are_matched_with_low_confidence() -> None:
    """Two cells of the same flat colour cannot be told apart, and the matcher
    must say so rather than pick one confidently."""
    flat = REFERENCE.copy()
    flat[0:CP, 0:2 * CP] = (200, 150, 90)       # r0c0 and r0c1: the same sky
    m = PieceMatcher(flat, BOARD)
    m.set_background(camera(flat_scene([], ref=flat)), CELL_FROM_FRAME)
    frame = camera(flat_scene([((0, 0), (2, 2), 0)], ref=flat))
    [match] = m.read(frame, CELL_FROM_FRAME)
    assert match.name in {"r0c0", "r0c1"} and match.confidence < 0.5


def test_readings_feed_the_jigsaw_session() -> None:
    session = Session(grid_spec(COLS, ROWS))
    m = matcher()
    frame = camera(flat_scene([((0, 0), (0, 0), 0), ((0, 3), (0, 1), 0)]))
    readings = m.readings(frame, CELL_FROM_FRAME)
    assert sorted(str(r.part) for r in readings) == [str(PieceRef(piece="r0c0")),
                                                     str(PieceRef(piece="r0c3"))]
    alerts = []
    for i in range(10):
        session.state.observe(i * 0.1, readings)  # type: ignore[attr-defined]
        alerts += session.pump()
    assert [(a.code, a.target) for a in alerts] == [("wrong_position", "r0c3")]


# --- the command line, calibration included -------------------------------------------

def test_the_command_line_calibrates_from_the_markers(tmp_path, capsys) -> None:
    ref, photo, empty = tmp_path / "box.png", tmp_path / "photo.png", tmp_path / "empty.png"
    cv2.imwrite(str(ref), REFERENCE)
    cv2.imwrite(str(photo), camera(flat_scene([((1, 1), (1, 1), 0), ((0, 2), (2, 3), 180)])))
    cv2.imwrite(str(empty), camera(flat_scene([])))
    out = tmp_path / "annotated.png"
    assert main([str(photo), "--reference", str(ref), "--board", "4x3", "--empty", str(empty),
                 "--marker-cells", "1.0", "--gap-cells", "0.25", "-o", str(out)]) == 0
    text = capsys.readouterr().out
    assert "row 1 col 1: r1c1 rot   0" in text and "ok" in text
    assert "row 2 col 3: r0c2 rot 180" in text and "MISPLACED" in text
    assert out.exists()
