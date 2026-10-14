"""Colour + depth from an iPhone Pro, through the Record3D app (FR-PER-8).

    python -m thirdeye.perception.record3d game07.r3d            # what is in a recording
    python -m thirdeye.perception.record3d game07.r3d --frame 40 -o f40.png

Two ways in, one `Frame` out:

*A recording* (`.r3d`, exported from the app). A zip holding a `metadata`
JSON -- colour size `w`/`h`, depth size `dw`/`dh`, `fps`, and the colour
intrinsics `K` as nine numbers, column-major -- and, per frame `i`,
`rgbd/i.jpg` (colour), `rgbd/i.depth` (float32 metres, `dh` x `dw`, LZFSE-
compressed) and optionally `rgbd/i.conf` (uint8 confidence 0-2, same).

*A live USB stream*, through the `record3d` package's `Record3DStream`.

**Check against a real file before relying on this.** The `.r3d` layout above
is Record3D's documented export as best known when this was written, and the
tests exercise this reader against archives built to that description -- they
prove the parsing, not the format. The first real recording is the test that
matters: `python -m thirdeye.perception.record3d some.r3d` prints what it
found.

Needs `pyliblzfse` for recordings and `record3d` for streaming, both in the
`live` extra; each is imported only when used.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from thirdeye.perception.tower import DepthView


@dataclass
class Frame:
    t: float
    #: BGR, as OpenCV expects.
    colour: np.ndarray
    #: Metres, (dh, dw). 0 or NaN where the sensor had no return.
    depth_m: np.ndarray
    #: Colour-image intrinsics.
    K: np.ndarray
    #: 0 low, 1 medium, 2 high; None if the recording has none.
    confidence: np.ndarray | None = None

    def depth_view(self, R: np.ndarray, t: np.ndarray, min_confidence: int = 1) -> DepthView:
        """This frame as the tower perception reads it, in millimetres, with
        low-confidence depth (object edges, mostly) blanked out."""
        depth = self.depth_m.astype(np.float64).copy()
        if self.confidence is not None:
            depth[self.confidence < min_confidence] = 0.0
        h, w = self.colour.shape[:2]
        return DepthView.from_record3d(depth, self.K, (w, h), R, t)


def _lzfse() -> Callable[[bytes], bytes]:
    try:
        import liblzfse
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError("reading .r3d needs pyliblzfse: pip install pyliblzfse") from exc
    return liblzfse.decompress


@dataclass
class R3DRecording:
    """A `.r3d` export, read lazily frame by frame."""

    path: Path
    decompress: Callable[[bytes], bytes] | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._zip = zipfile.ZipFile(self.path)
        self.meta = json.loads(self._zip.read("metadata"))
        self.decompress = self.decompress or _lzfse()
        names = set(self._zip.namelist())
        self._count = sum(1 for n in names if n.startswith("rgbd/") and n.endswith(".depth"))
        self._has_conf = any(n.endswith(".conf") for n in names)

    @property
    def K(self) -> np.ndarray:
        """Colour intrinsics. Stored column-major, so the transpose of the
        row-major reshape."""
        return np.asarray(self.meta["K"], float).reshape(3, 3).T

    @property
    def fps(self) -> float:
        return float(self.meta.get("fps", 30))

    @property
    def depth_size(self) -> tuple[int, int]:
        return int(self.meta["dw"]), int(self.meta["dh"])

    def __len__(self) -> int:
        return self._count

    def frame(self, i: int) -> Frame:
        dw, dh = self.depth_size
        colour = cv2.imdecode(np.frombuffer(self._zip.read(f"rgbd/{i}.jpg"), np.uint8),
                              cv2.IMREAD_COLOR)
        assert self.decompress is not None
        depth = np.frombuffer(self.decompress(self._zip.read(f"rgbd/{i}.depth")),
                              np.float32).reshape(dh, dw)
        conf = None
        if self._has_conf:
            conf = np.frombuffer(self.decompress(self._zip.read(f"rgbd/{i}.conf")),
                                 np.uint8).reshape(dh, dw)
        return Frame(t=i / self.fps, colour=colour, depth_m=depth, K=self.K, confidence=conf)

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self.frame(i)

    def close(self) -> None:
        self._zip.close()


class LiveStream:  # pragma: no cover - needs a phone on USB
    """Frames from the Record3D app over USB ("USB streaming" mode in the app).

    Wraps the `record3d` package's callback API as a blocking `next_frame()`.
    """

    def __init__(self, device_index: int = 0):
        from record3d import Record3DStream

        devices = Record3DStream.get_connected_devices()
        if len(devices) <= device_index:
            raise RuntimeError(f"{len(devices)} Record3D device(s) connected; is the app "
                               "open in USB streaming mode?")
        self._event = threading.Event()
        self._stream = Record3DStream()
        self._stream.on_new_frame = self._event.set
        self._stream.on_stream_stopped = self._event.set
        self._stream.connect(devices[device_index])
        self._t0: float | None = None

    def next_frame(self, timeout: float = 5.0) -> Frame | None:
        import time

        if not self._event.wait(timeout):
            return None
        self._event.clear()
        s = self._stream
        k = s.get_intrinsic_mat()
        K = np.array([[k.fx, 0, k.tx], [0, k.fy, k.ty], [0, 0, 1.0]])
        colour = cv2.cvtColor(np.asarray(s.get_rgb_frame()), cv2.COLOR_RGB2BGR)
        now = time.monotonic()
        self._t0 = self._t0 if self._t0 is not None else now
        conf = s.get_confidence_frame() if hasattr(s, "get_confidence_frame") else None
        return Frame(t=now - self._t0, colour=colour, depth_m=np.asarray(s.get_depth_frame()),
                     K=K, confidence=None if conf is None or not np.size(conf) else conf)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", type=Path, help="a .r3d export")
    ap.add_argument("--frame", type=int, help="save this frame's colour and depth side by side")
    ap.add_argument("-o", "--out", type=Path, default=Path("frame.png"))
    args = ap.parse_args(argv)

    rec = R3DRecording(args.recording)
    dw, dh = rec.depth_size
    print(f"{len(rec)} frames at {rec.fps:g} fps; colour {rec.meta.get('w')}x{rec.meta.get('h')}, "
          f"depth {dw}x{dh}; confidence maps: {'yes' if rec._has_conf else 'no'}")
    print(f"K (colour):\n{np.array2string(rec.K, precision=1)}")
    first = rec.frame(0)
    valid = first.depth_m[np.isfinite(first.depth_m) & (first.depth_m > 0)]
    if valid.size:
        print(f"frame 0 depth: {valid.min():.3f}-{valid.max():.3f} m, median {np.median(valid):.3f}"
              f"; {100 * valid.size / first.depth_m.size:.0f}% valid")
    if args.frame is not None:
        f = rec.frame(args.frame)
        d = np.nan_to_num(f.depth_m)
        vis = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=255 / max(float(d.max()), 1e-6)),
                                cv2.COLORMAP_TURBO)
        vis = cv2.resize(vis, (f.colour.shape[1], f.colour.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(args.out), np.hstack([f.colour, vis]))
        print(f"frame {args.frame} -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
