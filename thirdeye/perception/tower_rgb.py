"""Baseline: Jenga slot occupancy from colour alone, no depth (CS3235 doc, section 7).

The depth reader (`tower.py`) asks "is there a block end at this exact
distance?". This asks the only question a single colour camera can: "does this
slot's end face still look the way it did when the tower was full?". Each end
face the camera sees is sampled at the same points `tower.py` uses, in the
grey image, and compared with the same points in a reference frame of the full
tower; a slot whose face changed by more than `threshold` grey levels (median
over the samples) reads empty.

It is a deliberately simple baseline, there to be beaten: an empty slot shows
the dark inside of the tower, which differs from a lit end face, so it should
catch clean holes in good light -- and it cannot tell a block pushed partway in
(the depth reader's "moving") from a hole, nor a shadow from either. The
comparison is the point of the experiment: does depth earn its keep?

Its output is a `TowerReading`, so the same lattice and checker consume it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from thirdeye.events import Slot
from thirdeye.perception.tower import STANDARD, DepthView, SlotState, TowerGeometry, TowerReading
from thirdeye.state.jenga.lattice import STANDARD_LAYERS


@dataclass
class RgbTowerReader:
    """One colour camera and a reference frame of the full tower."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    reference: np.ndarray
    geom: TowerGeometry = STANDARD
    #: Median grey-level change at which an end face counts as changed.
    threshold: float = 25.0
    #: Layers in the reference frame (a full standard tower). Below, a change
    #: is a block gone; above, where the reference saw air, a block arrived.
    ref_layers: int = STANDARD_LAYERS

    def __post_init__(self) -> None:
        self._ref = self._grey(self.reference)
        self._view = DepthView(np.zeros(self._ref.shape), np.asarray(self.K, float),
                               np.asarray(self.R, float), np.asarray(self.t, float).reshape(3))

    @staticmethod
    def _grey(img: np.ndarray) -> np.ndarray:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return g.astype(np.float32)

    def read(self, image: np.ndarray, mask: np.ndarray | None = None) -> TowerReading:
        grey = self._grey(image)
        h, w = grey.shape
        states: dict[Slot, SlotState] = {}
        for layer in range(self.geom.layers):
            for slot in range(self.geom.slots):
                verdict: SlotState = "unknown"
                for face, (centre, normal) in self.geom.ends(layer, slot).items():
                    if not self._view.faces(normal, centre):
                        continue
                    uv, z = self._view.project(self.geom.samples(layer, slot, face))
                    u = np.round(uv[:, 0]).astype(int)
                    v = np.round(uv[:, 1]).astype(int)
                    if not ((u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)).all():
                        continue
                    if mask is not None and mask[v, u].mean() > 0.5:
                        continue
                    changed = float(np.median(np.abs(grey[v, u] - self._ref[v, u]))) \
                        > self.threshold
                    full_before = layer < self.ref_layers
                    verdict = "block" if changed != full_before else "empty"
                states[Slot(layer, slot)] = verdict
        top = max((s.layer for s, st in states.items() if st == "block"), default=-1)
        unknown = [s for s, st in states.items() if st == "unknown" and s.layer <= top]
        return TowerReading(states, settled=True, unknown=unknown)
