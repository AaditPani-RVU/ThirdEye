"""Jenga slot occupancy from two corner depth cameras, on ray-cast towers.

The synthetic depth maps are rendered from exact geometry, so the truth is
known; sensor noise is added where the claim is that it does not matter.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic_tower import CORNER_A, CORNER_B, H, K, W, views

from thirdeye.compiler.schema import load_spec
from thirdeye.events import Slot
from thirdeye.perception.tower import (
    MARKER_MM,
    DepthView,
    TowerGeometry,
    _marker_corners,
    merge,
    outline,
    read_view,
    tower_markers,
    tower_pose,
)
from thirdeye.session import Session
from thirdeye.state.jenga.lattice import full_tower

SPECS = Path(__file__).resolve().parents[1] / "specs"


def tower(*holes: tuple[int, int], extra: list[list[int]] | None = None) -> list[list[int]]:
    rows = [list(r) for r in full_tower()]
    for layer, slot in holes:
        rows[layer][slot] = 0
    return rows + (extra or [])


def read(occ, pushed=None, noise_mm: float = 0.0, seed: int = 0, mask_a=None):
    a, b = views(occ, pushed, mask_a=mask_a)
    if noise_mm:
        rng = np.random.default_rng(seed)
        for v in (a, b):
            valid = v.depth_mm > 0
            v.depth_mm = v.depth_mm + valid * rng.normal(0, noise_mm, v.depth_mm.shape)
    return merge(read_view(a), read_view(b))


def test_geometry_every_end_is_seen_by_exactly_one_corner() -> None:
    (ra, ta), (rb, tb) = CORNER_A, CORNER_B
    a = DepthView(np.zeros((H, W)), K, ra, ta)
    b = DepthView(np.zeros((H, W)), K, rb, tb)
    g = TowerGeometry()
    for layer in range(18):
        for slot in range(3):
            ends = g.ends(layer, slot).values()
            assert sum(a.faces(n, c) for c, n in ends) == 1
            assert sum(b.faces(n, c) for c, n in ends) == 1


def test_a_full_tower_reads_full_and_settled() -> None:
    r = read(tower())
    assert r.settled and r.unknown == []
    assert r.occupancy() == [[1, 1, 1]] * 18
    assert r.height_mm() == 18 * 15


@pytest.mark.parametrize("hole", [(0, 1), (5, 1), (6, 0), (11, 2), (16, 1), (17, 0)])
def test_every_kind_of_hole_is_found(hole) -> None:
    r = read(tower(hole))
    occ = r.occupancy()
    assert occ[hole[0]][hole[1]] == 0
    assert sum(map(sum, occ)) == 18 * 3 - 1, "and nothing else"


def test_a_growing_top_layer_is_read() -> None:
    r = read(tower((4, 1), (9, 0), extra=[[1, 0, 1], [0, 1, 0]]))
    occ = r.occupancy()
    assert len(occ) == 20 and occ[18] == [1, 0, 1] and occ[19] == [0, 1, 0]
    assert occ[4][1] == 0 and occ[9][0] == 0


@pytest.mark.parametrize("push", [-30.0, -15.0, 20.0, 40.0])
def test_a_block_partway_out_is_moving_not_empty(push) -> None:
    r = read(tower(), pushed={(8, 1): push})
    assert r.states[Slot(8, 1)] == "moving"
    assert not r.settled


@pytest.mark.parametrize("seed", range(4))
def test_three_millimetres_of_sensor_noise_changes_nothing(seed) -> None:
    occ = tower((3, 1), (7, 1), (12, 0), (14, 2))
    r = read(occ, noise_mm=3.0, seed=seed)
    assert r.settled and r.occupancy()[:18] == [row for row in occ]


def test_a_hand_in_front_of_one_camera_leaves_the_other_to_decide() -> None:
    occ = tower((10, 1))
    mask = np.zeros((H, W), bool)
    mask[:, : W // 2 + 30] = True      # a hand covering most of camera A's view
    r = read(occ, mask_a=mask)
    assert r.occupancy()[10][1] == 0
    assert r.settled


def test_a_blocked_view_is_no_reading_not_an_empty_tower() -> None:
    mask = np.ones((H, W), bool)
    a, b = views(tower((6, 1)))
    a.mask, b.mask = mask, mask
    r = merge(read_view(a), read_view(b))
    assert not r.usable and r.height_mm() is None
    previous = tuple(tuple(row) for row in tower((6, 1)))
    assert r.occupancy(previous) == [list(row) for row in previous]


def test_a_hand_over_the_top_layers_does_not_shorten_the_tower() -> None:
    occ = tower((2, 1))
    a, b = views(occ)
    top_rows = np.zeros((H, W), bool)
    top_rows[: H // 3] = True          # the top third of both images
    a.mask, b.mask = top_rows, top_rows
    r = merge(read_view(a), read_view(b))
    previous = tuple(tuple(row) for row in occ)
    assert r.usable and r.occupancy(previous) == [list(row) for row in previous]


def test_a_fallen_tower_is_much_lower() -> None:
    r = read([[1, 1, 1], [1, 0, 1], [0, 1, 0]])
    assert r.height_mm() == 3 * 15


def test_readings_drive_a_jenga_session_to_the_foul() -> None:
    """Pull from the top complete layer, then place it on top: one alert."""
    session = Session(load_spec(SPECS / "example_constraints.json"))
    steps = [tower((17, 1)), tower((17, 1), extra=[[1, 0, 0]])]
    alerts = []
    for i, occ in enumerate(steps):
        r = read(occ)
        assert r.settled
        session.state.observe(r.occupancy(session.state.current()), t=float(i), hands_in_contact=1)
        alerts += session.pump()
    assert [a.code for a in alerts] == ["illegal_source_layer"]


def _mat_photo(skip: set[int] = frozenset()) -> tuple[np.ndarray, np.ndarray]:
    """What camera A's colour camera sees of the mat markers (4x the depth
    resolution), with the markers in `skip` hidden, as the tower hides one."""
    R, t = CORNER_A
    K4 = np.diag([4.0, 4.0, 1.0]) @ K
    img = np.full((H * 4 + 240, W * 4), 200, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    view = DepthView(np.zeros((H, W)), K4, R, t)
    for mid, (x, y) in tower_markers().items():
        if mid in skip:
            continue
        uv, _ = view.project(_marker_corners(x, y, MARKER_MM))
        marker = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(dictionary, mid, 160),
                                    20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        inner = np.float32([[20, 20], [179, 20], [179, 179], [20, 179]])
        warp = cv2.getPerspectiveTransform(inner, uv.astype(np.float32))
        drawn = cv2.warpPerspective(marker, warp, img.shape[::-1], borderValue=0)
        region = cv2.warpPerspective(np.full_like(marker, 255), warp, img.shape[::-1])
        img[region > 0] = drawn[region > 0]
    return img, K4


@pytest.mark.parametrize("hidden", [set(), {0}, {0, 3}])
def test_the_tower_pose_is_recovered_from_the_mat_markers(hidden) -> None:
    img, K4 = _mat_photo(skip=hidden)
    pose = tower_pose(img, K4)
    assert pose is not None
    R, t = CORNER_A
    assert np.allclose(pose[0], R, atol=0.03) and np.allclose(pose[1], t, atol=8.0)


def test_one_marker_is_not_enough() -> None:
    img, K4 = _mat_photo(skip={0, 1, 3})
    assert tower_pose(img, K4) is None


def test_the_outline_contains_the_tower() -> None:
    R, t = CORNER_A
    view = DepthView(np.zeros((H, W)), K, R, t)
    poly = outline(view, 18)
    uv, _ = view.project(np.array([[0.0, 0.0, 135.0]]))
    assert cv2.pointPolygonTest(poly.reshape(-1, 1, 2), tuple(map(float, uv[0])), False) > 0
