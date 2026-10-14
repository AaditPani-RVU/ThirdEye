"""The colour-only Jenga baseline, on shaded renders of the synthetic tower."""

from __future__ import annotations

import numpy as np
from synthetic_tower import CORNER_A, K, render

from thirdeye.events import Slot
from thirdeye.perception.tower_rgb import RgbTowerReader
from thirdeye.state.jenga.lattice import full_tower


def shade(occ) -> np.ndarray:
    """Nearer is brighter: a lamp at the camera, the crudest colour image there is."""
    depth = render(occ, *CORNER_A)
    return np.clip(255 - (depth - 300) / 2.5, 0, 255).astype(np.uint8)


def tower(*holes, extra=None):
    rows = [list(r) for r in full_tower()]
    for layer, slot in holes:
        rows[layer][slot] = 0
    return rows + (extra or [])


def test_a_hole_and_a_new_block_are_read_from_colour() -> None:
    reader = RgbTowerReader(K, *CORNER_A, reference=shade(tower()))
    r = reader.read(shade(tower((5, 0), extra=[[1, 0, 0]])))
    assert r.states[Slot(5, 0)] == "empty"
    assert r.states[Slot(4, 1)] == "block"
    assert r.states[Slot(18, 0)] == "block"
    assert r.occupancy()[5] == [0, 1, 1]


def test_an_unchanged_tower_reads_full() -> None:
    img = shade(tower())
    r = RgbTowerReader(K, *CORNER_A, reference=img).read(img)
    assert r.occupancy() == [list(row) for row in full_tower()]
