"""Which layer was a block pulled from, from colour alone: the crop-difference finder.

Input is the teammate capture app's training crops (`face_crops.py`): every
visible block end face, cut out square-on and saved per frame as
``<ms>_L<layer>_S<slot>_<p|m>[_side].jpg`` under ``<session>/<camera>/``.

The player's tap names the slot (left/middle/right = 0/1/2), so only the layer
is open. For each layer of that slot and each camera, the median end-face crop
of a calm stretch *before* is compared with one *after*; the layer whose face
changed most is the pull. Scores are z-scored per camera and summed, so a
camera that sees everything a little blurrier does not outvote the other.

Pulls already confirmed this game rule out layers where the tapped slot is
empty or holds the layer's last block. On the labelled Oct 20 sessions (49
pulls) this gets 78 % exact, 92 % top-2 (73 % / 86 % without that); see
`scripts/eval_layerfinder.py`. Layers here count from 0, as the crop names
do; the teammate's finder and the player's typed corrections count from 1.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SLOT = {"left": 0, "middle": 1, "right": 2}
NAME = re.compile(r"^(\d+)_L(\d+)_S(\d+)_(p|m)(_side)?\.jpg$")

#: (camera, layer, slot) -> [(ms, crop file)], oldest first.
CropIndex = dict[tuple[str, int, int], list[tuple[int, Path]]]


def index_crops(session: Path, cameras: tuple[str, ...] = ("android", "iphone")) -> CropIndex:
    """End-face crops of a session (side faces are left out)."""
    idx: CropIndex = defaultdict(list)
    for cam in cameras:
        folder = Path(session) / cam
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            m = NAME.match(p.name)
            if m and not m[5]:
                idx[(cam, int(m[2]), int(m[3]))].append((int(m[1]), p))
    for v in idx.values():
        v.sort()
    return dict(idx)


def _median_face(files: list[tuple[int, Path]], lo: int, hi: int) -> np.ndarray | None:
    got = [cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (32, 32)).astype(np.float32)
           for t, p in files if lo <= t <= hi]
    if not got:
        return None
    img = np.median(got, axis=0)
    return (img - img.mean()) / (img.std() + 1e-6)


def layer_scores(idx: CropIndex, mark_ms: int, slot: int, prev_ms: int | None = None,
                 pre_s: float = 3.5, post_s: tuple[float, float] = (0.8, 4.5)) -> dict[int, float]:
    """Layer -> how much its face in `slot` changed around the tap at `mark_ms`.

    `before` is the calm stretch just after the previous tap (`prev_ms`) -- the
    hand is away, the tower still -- not the seconds before this tap, which
    hold this move's own reaching. With no previous tap it starts 9 s back.
    """
    lo = (prev_ms + 1000) if prev_ms else mark_ms - 9000
    hi = min(lo + int(pre_s * 1000), mark_ms - 2000)
    per_cam: dict[str, dict[int, float]] = defaultdict(dict)
    for (cam, layer, s), files in idx.items():
        if s != slot:
            continue
        a = _median_face(files, lo, hi)
        b = _median_face(files, mark_ms + int(post_s[0] * 1000), mark_ms + int(post_s[1] * 1000))
        if a is not None and b is not None:
            per_cam[cam][layer] = float(np.abs(a - b).mean())
    score: dict[int, float] = defaultdict(float)
    for d in per_cam.values():
        v = np.array(list(d.values()))
        for layer, x in d.items():
            score[layer] += (x - v.mean()) / (v.std() + 1e-6)
    return dict(score)


def possible(layer: int, slot: int, removed: set[tuple[int, int]]) -> bool:
    """Could `slot` of `layer` be pulled with the tower still standing?

    Not if it is already out, or if it is the layer's last block. These are
    physical limits, not the game's rules: a pull the rules forbid (the top
    complete layer, say) is still a candidate, because flagging it is the
    checker's job. A pull leaving one edge block is kept too: near the top,
    where little weight rests on it, friction can hold that (Oct 20, 17:00).
    """
    if (layer, slot) in removed:
        return False
    return any((layer, s) not in removed for s in range(3) if s != slot)


def find_layer(idx: CropIndex, mark_ms: int, slot: int, prev_ms: int | None = None,
               removed: set[tuple[int, int]] | None = None, **kw) -> list[int]:
    """Layers of `slot`, most likely pulled first; empty if no crops cover the tap.

    `removed` holds the (layer, slot) pulls already confirmed this game; layers
    where this pull is physically impossible (see `possible`) are left out.
    """
    sc = layer_scores(idx, mark_ms, slot, prev_ms, **kw)
    if removed:
        sc = {layer: x for layer, x in sc.items() if possible(layer, slot, removed)}
    return sorted(sc, key=sc.get, reverse=True)
