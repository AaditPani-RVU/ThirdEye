"""One session at a time, run on a worker thread, reported as protocol messages.

Two sources feed a `Session` the same way the command-line tools do:

*Replay* -- a scripted scenario through `thirdeye.replay`, paced to the wall
clock (`speed` x real time, 0 for as fast as possible). This is the demo that
needs no hardware, and the one the tests drive.

*Camera* -- a webcam or a recorded video through the live perception pipeline
of `thirdeye.live` (LEGO or jigsaw). The camera is read here, in this process
(FR-UI-4); only overlaid, downscaled JPEGs leave it, a few times a second.

The runner never touches the network. It calls `publish` with each message,
and the app decides who hears it.
"""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from thirdeye.explain.report import plain
from thirdeye.replay import Noise, Replay, Stopped
from thirdeye.server import protocol
from thirdeye.session import Session
from thirdeye.sessionlog import SessionRecord

Publish = Callable[[dict[str, Any]], None]


@dataclass
class Runner:
    """A session on a thread. `start()` once; `stop()` ends it early."""

    publish: Publish
    log_dir: Path | None = None
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, wait: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(wait)

    def _launch(self, target: Callable[[], None]) -> None:
        def guarded() -> None:
            try:
                target()
            except Exception as exc:  # reported to the UI, never swallowed silently
                self.publish(protocol.error(f"{type(exc).__name__}: {exc}"))
                raise

        self._thread = threading.Thread(target=guarded, daemon=True, name="thirdeye-session")
        self._thread.start()

    # --- bookkeeping shared by both sources ------------------------------------------

    def _watch(self, session: Session) -> Callable[[float, bool], None]:
        """A per-frame callback that publishes new alerts and changed state."""
        sent_alerts = 0
        last: dict[str, Any] = {"rows": None, "t": -1.0}

        def tick(t: float, force: bool = False) -> None:
            nonlocal sent_alerts
            for a in session.alerts[sent_alerts:]:
                self.publish(protocol.alert(a))
            sent_alerts = len(session.alerts)
            msg = protocol.state(session, t)
            key = (msg["checklist"], msg.get("tower"))
            if force or key != last["rows"] or t - last["t"] >= 1.0:
                self.publish(msg)
                last["rows"], last["t"] = key, t

        return tick

    def _finish(self, session: Session, t: float, tick, meta: dict[str, Any]) -> None:
        session.finish(t)
        tick(t, True)
        self._publish_end(session, SessionRecord.of(session, meta=meta))

    def _publish_end(self, session: Session, record: SessionRecord) -> None:
        if self.log_dir is not None:
            name = record.meta.get("run") or f"session_{int(time.time())}"
            record.write(self.log_dir / f"{name}.json")
        self.publish(protocol.end(session.summary(), plain(record).markdown(), record.outcome))

    # --- sources ------------------------------------------------------------------------

    def start_replay(self, script: Path, speed: float = 1.0, noise: Noise | None = None) -> None:
        replay = Replay.load(script, noise=noise)
        session = replay.session
        tick = self._watch(session)
        wall0 = time.monotonic()

        def on_frame(t: float) -> None:
            if self._stop.is_set():
                raise Stopped
            if speed > 0:
                delay = t / speed - (time.monotonic() - wall0)
                if delay > 0:
                    self._stop.wait(delay)
            tick(t)

        replay.on_frame = on_frame

        def run() -> None:
            self.publish(protocol.hello(session, source=f"replay:{script.stem}"))
            record = replay.run()   # finishes the session, stopped early or not
            tick(replay.t, True)
            record.meta.update({"run": script.stem, "script": str(script), "source": "replay"})
            self._publish_end(session, record)

        self._launch(run)

    def start_camera(self, session: Session, pipeline, capture: cv2.VideoCapture,
                     source: str, video: bool, frame_every_s: float = 0.2,
                     meta: dict[str, Any] | None = None) -> None:
        from thirdeye.live import draw

        tick = self._watch(session)

        def run() -> None:
            self.publish(protocol.hello(session, source=source))
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            start, frames, t, sent_at, recent = time.monotonic(), 0, 0.0, -1.0, []
            try:
                while not self._stop.is_set():
                    ok, img = capture.read()
                    if not ok:
                        break
                    t = frames / fps if video else time.monotonic() - start
                    frames += 1
                    result = pipeline.process(img, t)
                    if result.usable:
                        session.state.observe(t, result.readings,
                                              hands_present=result.hands_present)
                        recent += [str(a) for a in session.pump()]
                    tick(t)
                    if t - sent_at >= frame_every_s:
                        sent_at = t
                        self.publish(protocol.frame(t, _jpeg(
                            draw(img, result, session, recent, pipeline.grid))))
            finally:
                capture.release()
            self._finish(session, t, tick, {"source": source, **(meta or {})})

        self._launch(run)


def _jpeg(img, width: int = 640) -> str:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
