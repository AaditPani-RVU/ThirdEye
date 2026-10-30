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
    #: Called with the finished session's record and its plain report, to keep them.
    on_end: Callable[[SessionRecord, str], object] | None = None
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _started_at: str = field(default="", init=False)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, wait: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(wait)

    def _launch(self, target: Callable[[], None]) -> None:
        self._started_at = _stamp()
        def guarded() -> None:
            try:
                target()
            except Exception as exc:  # reported to the UI, never swallowed silently
                self.publish(protocol.error(f"{type(exc).__name__}: {exc}"))
                raise

        self._thread = threading.Thread(target=guarded, daemon=True, name="thirdeye-session")
        self._thread.start()

    # --- bookkeeping shared by both sources ------------------------------------------

    def _watch(self, session: Session) -> Callable[..., None]:
        """A per-frame callback that publishes new alerts and changed state."""
        sent_alerts = 0
        last: dict[str, Any] = {"rows": None, "t": -1.0}

        def tick(t: float, force: bool = False, perception: str | None = None) -> None:
            nonlocal sent_alerts
            for a in session.alerts[sent_alerts:]:
                self.publish(protocol.alert(a))
            sent_alerts = len(session.alerts)
            msg = protocol.state(session, t, perception)
            key = (msg["checklist"], msg.get("tower"), perception)
            if force or key != last["rows"] or t - last["t"] >= 1.0:
                self.publish(msg)
                last["rows"], last["t"] = key, t

        return tick

    def _finish(self, session: Session, t: float, tick, meta: dict[str, Any]) -> None:
        session.finish(t)
        tick(t, True)
        self._publish_end(session, SessionRecord.of(session, meta=meta))

    def _publish_end(self, session: Session, record: SessionRecord) -> None:
        record.meta.setdefault("started_at", self._started_at)
        record.meta["ended_at"] = _stamp()
        if self.log_dir is not None:
            name = record.meta.get("run") or f"session_{int(time.time())}"
            record.write(self.log_dir / f"{name}.json")
        report = plain(record).markdown()
        if self.on_end is not None:
            try:
                self.on_end(record, report)
            except Exception as exc:  # noqa: BLE001 -- storage trouble must not hide the result
                self.publish(protocol.error(f"session not saved: {type(exc).__name__}: {exc}"))
        self.publish(protocol.end(session.summary(), report, record.outcome))

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
            blind, lost_after = 0, max(int(fps), 1)   # about a second of unusable frames
            try:
                while not self._stop.is_set():
                    ok, img = capture.read()
                    if not ok:
                        break
                    t = frames / fps if video else time.monotonic() - start
                    frames += 1
                    t0 = time.perf_counter()
                    result = pipeline.process(img, t)
                    blind = 0 if result.usable else blind + 1
                    if result.usable:
                        session.state.observe(t, result.readings,
                                              hands_present=result.hands_present)
                        recent += [str(a) for a in session.pump()]
                    pipeline.timings.add("frame", time.perf_counter() - t0)
                    tick(t, perception="lost" if blind >= lost_after else "ok")
                    if t - sent_at >= frame_every_s:
                        sent_at = t
                        self.publish(protocol.frame(t, _jpeg(
                            draw(img, result, session, recent, pipeline.grid))))
            finally:
                capture.release()
            fps = frames / max(time.monotonic() - start, 1e-9)
            self._finish(session, t, tick, {"source": source, **(meta or {}),
                                            "latency": pipeline.timings.summary(),
                                            "fps": round(fps, 2), "frames": frames})

        self._launch(run)


    def start_jenga(self, session: Session, pipeline, feed, source: str,
                    frame_every_s: float = 0.2, send_frames: bool = True,
                    meta: dict[str, Any] | None = None) -> None:
        """Corner depth cameras through `live_jenga.JengaPipeline`. `feed.next()`
        gives (frames, t) or None at the end."""
        from thirdeye.live_jenga import draw

        tick = self._watch(session)

        def run() -> None:
            self.publish(protocol.hello(session, source=source))
            t, sent_at, recent = 0.0, -1.0, []
            start, frames = time.monotonic(), 0
            while not self._stop.is_set() and (got := feed.next()) is not None:
                batch, t = got
                frames += 1
                t0 = time.perf_counter()
                result = pipeline.process(batch, t)
                if result.observed:
                    recent += [str(a) for a in session.pump()]
                pipeline.timings.add("frame", time.perf_counter() - t0)
                tick(t, perception="ok" if result.posed else "lost")
                if send_frames and t - sent_at >= frame_every_s and batch[0] is not None:
                    sent_at = t
                    self.publish(protocol.frame(t, _jpeg(draw(
                        batch[0], pipeline.poses[0], result, session, recent,
                        len(session.state.current()) + 1))))
                if session.outcome is not None:
                    break
            fps = frames / max(time.monotonic() - start, 1e-9)
            self._finish(session, t, tick, {"source": source, **(meta or {}),
                                            "latency": pipeline.timings.summary(),
                                            "fps": round(fps, 2), "frames": frames})

        self._launch(run)

    def start_follow(self, spec, link, poll_s: float = 0.5,
                     meta: dict[str, Any] | None = None) -> None:
        """The game server's confirmed pulls (`gamelink.py`), as a session. Runs until stopped
        or the game server reports the game over."""
        from thirdeye.server.gamelink import ASSUMED_PLACEMENT, LinkError, apply_pull, full
        from thirdeye.session import Session
        from thirdeye.state.jenga.lattice import TowerLattice

        first = link.status()   # fail now, in the request, if the game server is not there
        source = f"game:{link.url}"
        meta = {"source": source, "note": ASSUMED_PLACEMENT, **(meta or {})}

        def run() -> None:
            start = time.monotonic()
            st, session, tick, snap = first, None, None, ()
            while True:
                t = time.monotonic() - start
                rebuild, new = link.plan(st)
                if rebuild or session is None:
                    snap = full(st.layers)
                    session = Session(spec, state=TowerLattice(initial=snap))
                    tick = self._watch(session)
                    self.publish(protocol.hello(session, source=source))
                for pull in new:
                    snap = apply_pull(snap, pull)
                    session.state.observe(snap, t)
                    session.pump()
                if st.game_over and not session.state.collapse_detected:
                    session.state.collapse_detected = True
                    session.state.observe(snap, t)
                    session.pump()
                tick(t, bool(new) or rebuild)
                if session.outcome is not None or self._stop.wait(poll_s):
                    break
                try:
                    st = link.status()
                except LinkError:
                    continue   # a blip on the Wi-Fi; the next poll catches up
            self._finish(session, t, tick, meta)

        self._launch(run)


def _stamp() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _jpeg(img, width: int = 640) -> str:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
