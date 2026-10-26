"""Collapse detection (FR-STA-8).

A collapse is the one Jenga event the lattice cannot express -- there is no
occupancy grid for a heap -- and the one that ends the session. It is detected
from the tower's measured height, which the side views give directly: normal
play moves the top by at most one layer at a time, while a collapse takes it
down by many layers inside a single frame. The two are separated by an order of
magnitude, which is why the plan calls this trivially separable.

Two guards keep it from firing on a bad frame rather than a fallen tower:

*The reference height is a median* of the last few frames, so a single glitchy
reading cannot become the "before" a later frame is compared with.

*The drop must hold* for `confirm_frames` consecutive frames. A silhouette
briefly cut short -- a hand across the view, a dropped frame -- recovers on the
next one; a fallen tower does not. The collapse is timestamped at the first
low frame, not the confirming one, so the report's reconstruction of the last
moves is anchored to when it actually happened.

*And for `confirm_s` seconds*, where frames come fast: an arm reaching across
a single camera's view can hide the whole tower for several seconds (game_01
of the Oct 15 recordings: 5.7 s), but it leaves; a heap does not get up.

*A jump up stays out of the reference.* Play raises the tower one layer at a
time; a reading several layers higher is a hand or arm over the top. Let into
the median, it makes the return to the true height look like a fall. A rise
that holds for `rise_frames` is believed (the reference started low).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median

from thirdeye.state.jenga.lattice import BLOCK_MM, TowerLattice

#: A layer is one block thick.
LAYER_MM = BLOCK_MM[2]


@dataclass
class CollapseDetector:
    """Watches tower height and marks the lattice when the tower comes down."""

    lattice: TowerLattice
    #: A drop of at least this many layers in one frame is a collapse candidate.
    #: Normal play changes height by at most one.
    min_drop_layers: float = 4.0
    confirm_frames: int = 3
    confirm_s: float = 0.0
    history: int = 5
    #: A frame more than this many layers above the reference is not added to it...
    max_rise_layers: float = 1.5
    #: ...unless the rise holds this many frames.
    rise_frames: int = 20

    t_collapse: float | None = field(default=None, init=False)
    _recent: deque[float] = field(init=False)
    _pending: tuple[float, float] | None = field(default=None, init=False)
    _low: int = field(default=0, init=False)
    _high: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.history)

    @property
    def collapsed(self) -> bool:
        return self.t_collapse is not None

    @property
    def pending(self) -> bool:
        """A drop is being confirmed: the tower fell, or something hides it."""
        return self._pending is not None

    def feed(self, t: float, height_mm: float | None) -> bool:
        """One frame's measured tower height. True on the frame a collapse is confirmed.

        `None` is a frame where the height could not be measured; it neither
        advances nor resets a pending confirmation.
        """
        if self.collapsed or height_mm is None:
            return False

        if self._pending is None and self._recent:
            reference = median(self._recent)
            if self._fell(reference, height_mm):
                self._pending, self._low = (t, reference), 0

        if self._pending is not None:
            t0, reference = self._pending
            if self._fell(reference, height_mm):
                self._low += 1
                if self._low >= self.confirm_frames and t - t0 >= self.confirm_s:
                    self.t_collapse = t0
                    self.lattice.collapse_detected = True
                    return True
                return False
            # It came back: a glitch, not a collapse. The low frames are
            # discarded so they never pollute the reference.
            self._pending, self._low = None, 0

        if self._recent and height_mm - median(self._recent) > self.max_rise_layers * LAYER_MM:
            self._high += 1
            if self._high < self.rise_frames:
                return False
            self._recent.clear()
        self._high = 0
        self._recent.append(height_mm)
        return False

    def _fell(self, reference: float, height_mm: float) -> bool:
        return reference - height_mm >= self.min_drop_layers * LAYER_MM
