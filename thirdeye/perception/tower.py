"""Jenga slot occupancy from depth, seen from two opposite corners (FR-PER-8, FR-STA-7).

    per camera:  depth map + tower pose  ->  for every slot, how far its end face
                                            sits from where a block's end should be
    merged:      both ends of every slot ->  occupancy rows for `TowerLattice`,
                                            or "not settled" while a block is moving

The rig is two depth cameras (an iPhone Pro's LiDAR, and a second one) at
opposite corners of the tower, each looking at two faces. Geometry does the
work, not a detector: every block's size and place in a standard tower is
known, so each camera is asked a precise question -- is there a block end at
this exact spot? -- by projecting points on that end face into the depth map
and comparing the depth measured there with the depth a block end would have.
All blocks look alike, which defeats appearance but not geometry.

*Why every slot is seen, and seen twice.* Layers alternate direction, so an
even layer shows its three block ends on the +x and -x faces and an odd layer
on +y and -y. A camera at the (+x, +y) corner sees +x and +y, so it sees one
end of *every* block; the opposite camera sees the other end. Each slot gets
two readings, one per end.

*Why the merge decides, not either camera.* From a corner, a ray into an empty
slot soon hits the hole's side wall, so a hole looks only a few centimetres
deep -- about what a block pushed partway in looks like. The other end breaks
the tie. A block is rigid: pushed in by d at one end, it sticks out by d at the
other. So:

    both ends at the face          -> a block, settled
    both ends further than a face  -> no block (a hole seen from both sides)
    one end in, the other out      -> a block on the move: not settled

A reading with a block on the move is not handed to the lattice, which judges
settled towers only (see `TowerLattice`). A slot neither camera can see (a hand
in front, a dropout in the depth) keeps its previous state, and says so.

Units are millimetres throughout; the tower frame has its origin at the centre
of the base, z up. Record3D depth maps are in metres and come in at lower
resolution than the colour image; `DepthView` takes care of both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from thirdeye.events import Slot
from thirdeye.state.jenga.lattice import BLOCK_MM, SLOTS_PER_LAYER, STANDARD_LAYERS

LENGTH, WIDTH, HEIGHT = BLOCK_MM
HALF = LENGTH / 2


# --- the tower's geometry ---------------------------------------------------------------


@dataclass(frozen=True)
class TowerGeometry:
    """Where every block of a tower is, in the tower frame."""

    #: Layers to look at: more than the tower starts with, since it grows.
    layers: int = STANDARD_LAYERS + 8
    slots: int = SLOTS_PER_LAYER

    def box(self, layer: int, slot: int) -> tuple[np.ndarray, np.ndarray]:
        """(min corner, max corner) of the block in this slot."""
        across = -HALF + WIDTH * slot, -HALF + WIDTH * (slot + 1)
        z = HEIGHT * layer, HEIGHT * (layer + 1)
        if layer % 2 == 0:  # runs along x, slots side by side in y
            return np.array([-HALF, across[0], z[0]]), np.array([HALF, across[1], z[1]])
        return np.array([across[0], -HALF, z[0]]), np.array([across[1], HALF, z[1]])

    def ends(self, layer: int, slot: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """The block's two end faces: name -> (centre, outward normal)."""
        lo, hi = self.box(layer, slot)
        c = (lo + hi) / 2
        axis = 0 if layer % 2 == 0 else 1
        out = {}
        for sign, name in ((1, "+"), (-1, "-")):
            centre = c.copy()
            centre[axis] = sign * HALF
            normal = np.zeros(3)
            normal[axis] = sign
            out[f"{name}{'xy'[axis]}"] = (centre, normal)
        return out

    def samples(self, layer: int, slot: int, face: str, n: int = 4,
                inset: float = 0.3) -> np.ndarray:
        """n x n points on an end face, kept `inset` away from its edges so a
        neighbour's edge or a chamfer does not land in the sample."""
        centre, normal = self.ends(layer, slot)[face]
        across = np.array([0.0, 1.0, 0.0]) if normal[0] else np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        ticks = np.linspace(-(0.5 - inset), 0.5 - inset, n)
        return np.array([centre + a * WIDTH * across + b * HEIGHT * up
                         for a in ticks for b in ticks])


#: The standard tower, looked at up to 8 layers above its starting height.
STANDARD = TowerGeometry()


# --- one camera -----------------------------------------------------------------------------


@dataclass
class DepthView:
    """One depth frame and where the camera was, relative to the tower.

    `K` is for the depth map's own resolution. `R`, `t` take a tower-frame
    point to the camera frame: `X_cam = R @ X + t`, in millimetres.
    """

    depth_mm: np.ndarray
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    #: Pixels to ignore -- a hand, typically. Same shape as the depth map.
    mask: np.ndarray | None = None

    @classmethod
    def from_record3d(cls, depth_m: np.ndarray, K_rgb: np.ndarray, rgb_size: tuple[int, int],
                      R: np.ndarray, t: np.ndarray) -> DepthView:
        """Record3D gives metres, and intrinsics for the colour image; the
        depth map is smaller, so the intrinsics are scaled down to it."""
        h, w = depth_m.shape
        sx, sy = w / rgb_size[0], h / rgb_size[1]
        K = np.diag([sx, sy, 1.0]) @ np.asarray(K_rgb, float)
        return cls(np.asarray(depth_m, np.float64) * 1000.0, K, np.asarray(R, float),
                   np.asarray(t, float).reshape(3))

    @property
    def centre(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel (u, v) and camera depth of tower-frame points."""
        cam = pts @ self.R.T + self.t
        uvw = cam @ self.K.T
        return uvw[:, :2] / uvw[:, 2:3], cam[:, 2]

    def faces(self, normal: np.ndarray, centre: np.ndarray, grazing: float = 0.15) -> bool:
        """Does this camera see a face with this outward normal? Not edge-on."""
        to_cam = self.centre - centre
        return float(normal @ to_cam) > grazing * float(np.linalg.norm(to_cam))


EndState = Literal["seen", "occluded", "unseen"]


@dataclass(frozen=True)
class EndReading:
    """One end of one slot, from one camera."""

    state: EndState
    #: Median of measured minus expected depth over the samples, in mm.
    #: Positive: further than a block end would be (recessed, or a hole).
    #: Negative: nearer (a block sticking out).
    offset_mm: float | None = None


#: Nearer than this, a sample is something in front of the tower, not a block.
OCCLUDED_MM = 60.0


def _offset(delta: np.ndarray) -> float:
    """One number for an end face's samples, leaning away from zero.

    Seen from a corner, a hole is shallow: a ray into an empty *middle* slot
    meets a neighbour's side within 10-20 mm. The samples on the side of the
    face towards the camera see deepest, so the 75th percentile of a recess
    (25th of a protrusion) keeps roughly twice the median's margin over the
    tolerance, while a flush block end stays near zero either way.
    """
    return float(np.percentile(delta, 75 if np.median(delta) >= 0 else 25))


def read_view(view: DepthView, geom: TowerGeometry = STANDARD,
              min_valid: float = 0.5) -> dict[Slot, EndReading]:
    """Every slot end this camera faces, measured against where it should be."""
    h, w = view.depth_mm.shape
    out: dict[Slot, EndReading] = {}
    for layer in range(geom.layers):
        for slot in range(geom.slots):
            for face, (centre, normal) in geom.ends(layer, slot).items():
                if not view.faces(normal, centre):
                    continue
                pts = geom.samples(layer, slot, face)
                uv, expected = view.project(pts)
                u = np.round(uv[:, 0]).astype(int)
                v = np.round(uv[:, 1]).astype(int)
                inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (expected > 0)
                if not inside.all():
                    out[Slot(layer, slot)] = EndReading("unseen")
                    continue
                measured = view.depth_mm[v, u]
                ok = np.isfinite(measured) & (measured > 0)
                if view.mask is not None:
                    ok &= ~view.mask[v, u].astype(bool)
                if ok.mean() < min_valid:
                    out[Slot(layer, slot)] = EndReading("occluded")
                    continue
                delta = measured[ok] - expected[ok]
                if np.mean(delta < -OCCLUDED_MM) > 0.5:
                    out[Slot(layer, slot)] = EndReading("occluded")
                    continue
                out[Slot(layer, slot)] = EndReading("seen", _offset(delta))
    return out


# --- both cameras ---------------------------------------------------------------------------

SlotState = Literal["block", "empty", "moving", "unknown"]


@dataclass
class TowerReading:
    """The merged verdict on every slot, and what it means for the lattice."""

    states: dict[Slot, SlotState]
    #: False while any block is partway through the tower. The lattice takes
    #: settled readings only.
    settled: bool
    unknown: list[Slot] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Did either camera see any slot at all? A fully blocked view is no
        reading -- not an empty tower."""
        return any(st != "unknown" for st in self.states.values())

    def occupancy(self, previous: tuple[tuple[int, ...], ...] | None = None,
                  slots: int = SLOTS_PER_LAYER) -> list[list[int]]:
        """Rows up to the top block. A slot neither camera could see keeps its
        previous value -- or counts as a block, with no previous reading,
        since a tower is built full -- and that includes the top layers: a
        hand across the top of the tower does not make it shorter."""
        prev = previous or ()

        def was(layer: int, slot: int) -> int:
            return int(prev[layer][slot]) if layer < len(prev) and slot < len(prev[layer]) else 0

        top = max((s.layer for s, st in self.states.items()
                   if st in ("block", "moving") or (st == "unknown" and was(s.layer, s.slot))),
                  default=-1)
        rows = []
        for layer in range(top + 1):
            row = []
            for slot in range(slots):
                st = self.states.get(Slot(layer, slot), "unknown")
                if st == "unknown":
                    row.append(was(layer, slot) if prev else 1)
                else:
                    row.append(0 if st == "empty" else 1)
            rows.append(row)
        return rows

    def height_mm(self) -> float | None:
        """Top of the highest block, for collapse detection; None if nothing
        was seen at all."""
        seen = [s.layer for s, st in self.states.items() if st in ("block", "moving")]
        if not seen and all(st == "unknown" for st in self.states.values()):
            return None
        return (max(seen, default=-1) + 1) * HEIGHT


def baseline(readings: dict[Slot, EndReading], max_mm: float = 40.0) -> dict[Slot, float]:
    """The untouched tower's per-slot offsets, for `rebase`. A slot read deeper
    than `max_mm` is empty (air above the tower), not a biased block; rebased,
    it would read as a block at offset 0 for good."""
    return {s: r.offset_mm for s, r in readings.items()
            if r.state == "seen" and r.offset_mm is not None and r.offset_mm <= max_mm}


def rebase(readings: dict[Slot, EndReading],
           baseline: dict[Slot, float]) -> dict[Slot, EndReading]:
    """Offsets measured against a baseline reading (the untouched tower) instead
    of the ideal geometry, so a per-slot bias -- a chamfer, a block not quite
    flush, a few mm of pose error -- does not read as a hole. Slots the baseline
    has no number for keep their absolute offset."""
    return {s: (EndReading("seen", r.offset_mm - baseline[s])
                if r.state == "seen" and r.offset_mm is not None and s in baseline else r)
            for s, r in readings.items()}


def merge(a: dict[Slot, EndReading], b: dict[Slot, EndReading],
          tol_mm: float = 6.0, alone_mm: float = 20.0,
          protrude_mm: float | None = None) -> TowerReading:
    """Two cameras' end readings -> one verdict per slot (see the module doc).

    With only one end seen, a block end at the face is still a block, but a
    recess must be deep (`alone_mm`) to count as a hole: shallower, it could be
    a block pushed in, and the other end is not there to say. `protrude_mm`
    (default `tol_mm`) is how far a lone end must stick out to count as moving.
    """
    protrude_mm = tol_mm if protrude_mm is None else protrude_mm
    states: dict[Slot, SlotState] = {}
    for slot in sorted(set(a) | set(b)):
        ends = [r.offset_mm for r in (a.get(slot), b.get(slot))
                if r is not None and r.state == "seen" and r.offset_mm is not None]
        if not ends:
            states[slot] = "unknown"
        elif len(ends) == 1:
            (o,) = ends
            states[slot] = ("block" if abs(o) <= tol_mm else
                            "empty" if o >= alone_mm else
                            "moving" if o < -protrude_mm else "unknown")
        else:
            o1, o2 = ends
            if abs(o1) <= tol_mm and abs(o2) <= tol_mm:
                states[slot] = "block"
            elif o1 > tol_mm and o2 > tol_mm:
                states[slot] = "empty"
            elif (o1 > tol_mm and o2 < -tol_mm) or (o1 < -tol_mm and o2 > tol_mm):
                states[slot] = "moving"
            else:
                # One end at the face, the other off it: a block and a hole at
                # once. Not physical -- a misread -- so no claim either way.
                states[slot] = "unknown"
    unknown = [s for s, st in states.items() if st == "unknown"]
    # Unknown slots above the top of the tower are just air nobody could see.
    top = max((s.layer for s, st in states.items() if st in ("block", "moving")), default=-1)
    unknown = [s for s in unknown if s.layer <= top]
    settled = not any(st == "moving" for st in states.values())
    return TowerReading(states, settled, unknown)


# --- where the tower is ---------------------------------------------------------------------


#: Printed side of each mat marker, mm.
MARKER_MM = 40.0


def tower_markers(side_mm: float = MARKER_MM, gap_mm: float = 25.0) -> dict[int, tuple[float, float]]:
    """Centres of the four ArUco markers on the mat around the tower's base,
    in the tower frame: one off each corner of the 75 mm footprint.

    Print each the right way up: the top edge of the printed marker faces +y.
    From a corner camera the marker at the far corner is behind the tower, so
    `tower_pose` needs only two of the four.
    """
    o = HALF + gap_mm + side_mm / 2
    return {0: (-o, -o), 1: (o, -o), 2: (o, o), 3: (-o, o)}


def _marker_corners(x: float, y: float, side: float) -> np.ndarray:
    """A marker's corners in the tower frame, in ArUco's order (top-left,
    top-right, bottom-right, bottom-left of the printed marker, top = +y)."""
    h = side / 2
    return np.array([[x - h, y + h, 0], [x + h, y + h, 0], [x + h, y - h, 0], [x - h, y - h, 0]],
                    float)


def tower_pose(image: np.ndarray, K: np.ndarray,
               markers: dict[int, tuple[float, float]] | None = None,
               side_mm: float = MARKER_MM, min_markers: int = 2,
               dictionary: int = cv2.aruco.DICT_4X4_50) -> tuple[np.ndarray, np.ndarray] | None:
    """(R, t) of the tower frame in this camera, from the mat markers it can
    see; None if fewer than `min_markers`. Fixed cameras: do this once."""
    markers = markers or tower_markers(side_mm)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary))
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    obj, img = [], []
    for mid, c in zip(ids.ravel(), corners, strict=True):
        if int(mid) in markers:
            obj.append(_marker_corners(*markers[int(mid)], side_mm))
            img.append(c.reshape(4, 2))
    if len(obj) < min_markers:
        return None
    ok, rvec, tvec = cv2.solvePnP(np.concatenate(obj), np.concatenate(img).astype(np.float64),
                                  np.asarray(K, np.float64), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def outline(view: DepthView, layers: int) -> np.ndarray:
    """The tower's silhouette in this camera, as a pixel polygon -- the region
    `hands_in_contact` counts fingertips in."""
    top = HEIGHT * max(layers, 1)
    corners = np.array([[x, y, z] for x in (-HALF, HALF) for y in (-HALF, HALF)
                        for z in (0.0, top)])
    uv, _ = view.project(corners)
    return cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)


def refine_pose(view: DepthView, footprint_mm: float = 2 * HALF) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) corrected against the depth map itself, for a mat whose markers sit
    a few mm off the printed layout (a different print size, a curled sheet).

    The marker pose fixes the rotation well but can be off in translation by a
    few centimetres, and every end-face reading then inherits that as a bias.
    Two things in the depth map pin it down, provided the first frames show the
    untouched tower: the floor beside it must lie at z = 0, and the tower's
    footprint must be centred on the origin. Both are measured as medians of
    the back-projected depth in the tower frame, so a hand or a few pulled
    blocks do not move them. Returns the pose unchanged if too little depth
    is seen to say.
    """
    d = view.depth_mm
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    ok = d > 0
    cam = np.stack([(u - view.K[0, 2]) / view.K[0, 0] * d,
                    (v - view.K[1, 2]) / view.K[1, 1] * d, d], -1)[ok]
    p = (cam - view.t) @ view.R  # R^T (cam - t)
    r = 2.0 * footprint_mm
    near = p[(np.abs(p[:, 0]) < r) & (np.abs(p[:, 1]) < r)]
    side = near[(np.maximum(np.abs(near[:, 0]), np.abs(near[:, 1])) > footprint_mm)
                & (np.abs(near[:, 2]) < 60)]
    tower = near[near[:, 2] > 60]
    if len(side) < 200 or len(tower) < 200:
        return view.R, view.t
    shift = np.array([-np.mean(np.percentile(tower[:, 0], [2, 98])),
                      -np.mean(np.percentile(tower[:, 1], [2, 98])),
                      -float(np.median(side[:, 2]))])
    return view.R, view.t - view.R @ shift
