"""Synthetic depth maps of a Jenga tower, by ray casting (for the tower perception tests).

Blocks are axis-aligned boxes in the tower frame (mm); the table is z = 0.
Each pixel's depth is the camera-frame z of the nearest hit, as a depth camera
reports it. A block can be offset along its length to simulate a push.
"""

from __future__ import annotations

import numpy as np

from thirdeye.perception.tower import STANDARD, DepthView

W, H = 256, 192          # the iPhone LiDAR depth map's resolution
WALL_MM = 1500.0
K = np.array([[210.0, 0, W / 2], [0, 210.0, H / 2], [0, 0, 1]])


def look_at(eye, target=(0.0, 0.0, 130.0)) -> tuple[np.ndarray, np.ndarray]:
    """R, t for a camera at `eye` looking at `target`, image y down, z up in the world."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return R, -R @ eye


#: Opposite corners, 45 cm out and a little above the middle of the tower.
CORNER_A = look_at((320.0, 320.0, 260.0))
CORNER_B = look_at((-320.0, -320.0, 260.0))


def render(occupancy, R, t, pushed: dict | None = None, geom=STANDARD) -> np.ndarray:
    """Depth in mm. `pushed` maps (layer, slot) -> mm moved along the block's
    length, towards +x (even layer) or +y (odd layer)."""
    pushed = pushed or {}
    los, his = [], []
    for layer, row in enumerate(occupancy):
        for slot, filled in enumerate(row):
            if not filled:
                continue
            lo, hi = geom.box(layer, slot)
            d = pushed.get((layer, slot), 0.0)
            axis = 0 if layer % 2 == 0 else 1
            lo, hi = lo.copy(), hi.copy()
            lo[axis] += d
            hi[axis] += d
            los.append(lo)
            his.append(hi)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays_cam = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], axis=1) @ np.linalg.inv(K).T
    dirs = rays_cam @ R          # world direction per unit of camera depth
    origin = -R.T @ t
    depth = np.full(u.size, np.inf)
    if los:
        lo, hi = np.array(los), np.array(his)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / dirs[:, None, :]
            t1 = (lo[None] - origin) * inv
            t2 = (hi[None] - origin) * inv
        tmin = np.nanmax(np.minimum(t1, t2), axis=2)
        tmax = np.nanmin(np.maximum(t1, t2), axis=2)
        hit = (tmax >= tmin) & (tmax > 0)
        tnear = np.where(hit, np.maximum(tmin, 0), np.inf)
        depth = np.minimum(depth, tnear.min(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_table = -origin[2] / dirs[:, 2]
    depth = np.where((t_table > 0) & (t_table < depth), t_table, depth)
    # Nothing hit: a room wall behind, as indoors there always is.
    depth[~np.isfinite(depth)] = WALL_MM
    return depth.reshape(H, W)


def views(occupancy, pushed=None, mask_a=None) -> tuple[DepthView, DepthView]:
    (ra, ta), (rb, tb) = CORNER_A, CORNER_B
    return (DepthView(render(occupancy, ra, ta, pushed), K, ra, ta, mask=mask_a),
            DepthView(render(occupancy, rb, tb, pushed), K, rb, tb))
