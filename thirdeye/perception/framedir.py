"""A folder of per-frame files from the iPhone capture app, read as `Frame`s.

Each frame is four files sharing a millisecond timestamp::

    <ms>_rgb.jpg                  colour, e.g. 1920x1440
    <ms>_depth_256x192.bin        float32 metres, row-major (h x w)
    <ms>_conf_256x192.bin         uint8 0/1/2
    <ms>_intrinsics.json          fx, fy, cx, cy for the colour image

`<ms>_mark.json` files in the same folder hold the player's "I pulled a block"
taps (`{"mark": n, "t_ms": ...}`); `marks()` returns their times in seconds
from the first frame.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from thirdeye.perception.record3d import Frame

_DEPTH = re.compile(r"^(\d+)_depth_(\d+)x(\d+)\.bin$")


@dataclass
class FrameDir:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        found = [(int(m[1]), int(m[2]), int(m[3]))
                 for p in self.path.iterdir() if (m := _DEPTH.match(p.name))]
        if not found:
            raise FileNotFoundError(f"no *_depth_WxH.bin frames in {self.path}")
        found.sort()
        self.stamps = [f[0] for f in found]
        self.depth_size = (found[0][1], found[0][2])
        self.t0_ms = self.stamps[0]

    def __len__(self) -> int:
        return len(self.stamps)

    def frame(self, i: int) -> Frame:
        ms, (dw, dh) = self.stamps[i], self.depth_size
        stem = self.path / str(ms)
        colour = cv2.imread(f"{stem}_rgb.jpg")
        if colour is None:
            raise FileNotFoundError(f"{stem}_rgb.jpg")
        depth = np.fromfile(f"{stem}_depth_{dw}x{dh}.bin", np.float32).reshape(dh, dw)
        conf_path = Path(f"{stem}_conf_{dw}x{dh}.bin")
        conf = np.fromfile(conf_path, np.uint8).reshape(dh, dw) if conf_path.exists() else None
        k = json.loads(Path(f"{stem}_intrinsics.json").read_text())
        K = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], float)
        return Frame(t=(ms - self.t0_ms) / 1000.0, colour=colour, depth_m=depth, K=K,
                     confidence=conf)

    def marks(self) -> list[float]:
        """Seconds from the first frame at which the player tapped 'pulled'."""
        out = [json.loads(p.read_text())["t_ms"] for p in self.path.glob("*_mark.json")]
        return sorted((ms - self.t0_ms) / 1000.0 for ms in out)
