"""A phone streaming its camera to this server -- the teammate's Android capture app.

The app (`third-eye-testing/android-capture`) needs nothing but this machine's
address typed in. It speaks a small protocol on port 8000, which this module
answers so the same app works against Third Eye unchanged:

    POST /android        body = one JPEG; header X-Frame = the app's frame counter
                         -> {"markers": [ids seen], "recording": bool, "saved": n}
    GET|POST /cmd/start  begin saving frames (and taps) to data/phone/<time>/
    GET|POST /cmd/stop
    GET|POST /cmd/mark   a "block pulled" tap, saved as <ms>_mark.json

Saved folders use the capture app's file names (`<ms>_android.jpg`,
`<ms>_mark.json`), so the teammate's tools and ours read them alike.

Markers are looked for at most `detect_hz` times a second -- the app can send
60 frames a second and the reply must not wait on the detector.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


@dataclass
class PhoneFeed:
    save_root: Path
    #: Frames written per second while recording.
    save_fps: float = 5.0
    detect_hz: float = 10.0
    preview_width: int = 960

    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    raw: bytes | None = None
    preview: bytes | None = None
    markers: list[int] = field(default_factory=list)
    size: tuple[int, int] | None = None
    last_t: float = 0.0
    last_seq: int = 0
    times: collections.deque = field(default_factory=lambda: collections.deque(maxlen=40))
    folder: Path | None = None
    saved: int = 0
    marks: int = 0
    _detect_t: float = 0.0
    _save_t: float = 0.0

    # --- the app's side -----------------------------------------------------------------

    def push(self, jpeg: bytes, seq: int = 0) -> dict:
        now = time.time()
        with self.lock:
            # the app encodes in parallel, so frames can overtake; a big jump back is a restart
            stale = seq > 0 and 0 < self.last_seq - seq < 100
            if not stale:
                self.last_seq = seq
                self.raw, self.last_t = jpeg, now
                self.times.append(now)
            detect = not stale and now - self._detect_t >= 1.0 / self.detect_hz
            if detect:
                self._detect_t = now
            save = (not stale and self.folder is not None
                    and now - self._save_t >= 1.0 / self.save_fps)
            if save:
                self._save_t = now
                folder = self.folder
        if detect:
            self._detect(jpeg)
        if save:
            (folder / f"{int(now * 1000)}_android.jpg").write_bytes(jpeg)
            with self.lock:
                self.saved += 1
        with self.lock:
            return {"markers": list(self.markers), "recording": self.folder is not None,
                    "saved": self.saved}

    def command(self, cmd: str) -> bool:
        """start / stop / mark; False for anything else."""
        now = time.time()
        with self.lock:
            if cmd == "start":
                if self.folder is None:
                    self.folder = self.save_root / time.strftime("%Y%m%d_%H%M%S")
                    self.folder.mkdir(parents=True, exist_ok=True)
                    self.saved = self.marks = 0
            elif cmd == "stop":
                self.folder = None
            elif cmd == "mark":
                self.marks += 1
                if self.folder is not None:
                    ms = int(now * 1000)
                    (self.folder / f"{ms}_mark.json").write_text(
                        json.dumps({"mark": self.marks, "t_ms": ms}))
            else:
                return False
        return True

    # --- the browser's side ---------------------------------------------------------------

    def status(self) -> dict:
        now = time.time()
        with self.lock:
            t = list(self.times)
            fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 2 and t[-1] > t[0] and now - t[-1] < 2 else 0.0
            return {"live": bool(self.last_t) and now - self.last_t < 3.0,
                    "age": round(now - self.last_t, 1) if self.last_t else None,
                    "fps": round(fps, 1), "markers": list(self.markers), "size": self.size,
                    "recording": self.folder is not None,
                    "folder": self.folder.name if self.folder else None,
                    "saved": self.saved, "marks": self.marks}

    def frame(self) -> bytes | None:
        with self.lock:
            return self.preview or self.raw

    def _detect(self, jpeg: bytes) -> None:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        corners, ids, _ = _DETECTOR.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        found = sorted(int(i) for i in ids.ravel()) if ids is not None else []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(img, corners, ids, (60, 200, 60))
        h, w = img.shape[:2]
        if w > self.preview_width:
            img = cv2.resize(img, (self.preview_width, round(h * self.preview_width / w)),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with self.lock:
            self.markers, self.size = found, (w, h)
            if ok:
                self.preview = buf.tobytes()
