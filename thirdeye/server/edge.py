"""The edge laptop: runs the cameras and the checker here, sends text to the cloud.

    # jigsaw, webcam
    python -m thirdeye.server.edge --cloud https://thirdeye-xyz.a.run.app \
        --spec specs/jigsaw_4x3.json --reference box.jpg --camera 0
    # Jenga, one or two iPhones over USB (Record3D), or recordings
    python -m thirdeye.server.edge --cloud https://... --spec specs/example_constraints.json --stream 0
    python -m thirdeye.server.edge --cloud https://... --spec specs/example_constraints.json \
        --r3d cornerA.r3d --r3d cornerB.r3d
    # Jenga, the game server's confirmed pulls (both phones, MARK taps; see gamelink.py)
    python -m thirdeye.server.edge --cloud https://... --spec specs/example_constraints.json \
        --follow http://127.0.0.1:8000
    # no hardware: a scripted scenario, to test the link
    python -m thirdeye.server.edge --cloud https://... --scenario jenga_fouls

Perception never runs in the cloud (CS3235 doc, section 2): the session runs
in this process exactly as `thirdeye.live` would run it, and every protocol
message it produces is forwarded to the server's `/api/edge/publish`; the
finished session's log goes to `/api/edge/log`, where it is stored. Video
stays here: overlay frames are dropped unless `--send-frames` is given.

The edge proves who it is with THIRDEYE_EDGE_TOKEN (environment or `.env`).
It sends a heartbeat every few seconds so the UI can show "edge offline".
Messages are batched and sent from a background thread; if the cloud cannot
be reached they are kept and retried, and the session carries on regardless.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from thirdeye.compiler.llm import load_env
from thirdeye.sessionlog import SessionRecord

ROOT = Path(__file__).resolve().parents[2]


class CloudLink:
    """Batches protocol messages to the cloud; heartbeats while idle."""

    def __init__(self, base: str, token: str, name: str = "edge", send_frames: bool = False,
                 heartbeat_s: float = 5.0, post=None):
        self.base = base.rstrip("/")
        self.token = token
        self.name = name
        self.send_frames = send_frames
        self.heartbeat_s = heartbeat_s
        self.post = post or self._post
        self.q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.failures = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="cloud-link")

    def start(self) -> CloudLink:
        self._thread.start()
        return self

    def publish(self, msg: dict[str, Any]) -> None:
        if msg.get("type") == "frame" and not self.send_frames:
            return
        self.q.put(msg)

    def upload_log(self, record: SessionRecord, report: str) -> str | None:
        try:
            return self.post("/api/edge/log", {"record": record.to_json(), "report": report})["id"]
        except Exception as exc:  # noqa: BLE001 -- the log is also on disk; say so, carry on
            print(f"could not upload the session log: {exc}", file=sys.stderr)
            return None

    def close(self, wait: float = 10.0) -> None:
        deadline = time.monotonic() + wait
        while not self.q.empty() and time.monotonic() < deadline:
            time.sleep(0.1)
        self._stop.set()
        self._thread.join(2.0)

    def _loop(self) -> None:
        pending: list[dict[str, Any]] = []
        last = 0.0
        while not self._stop.is_set():
            try:
                pending.append(self.q.get(timeout=0.2))
                while len(pending) < 200:
                    pending.append(self.q.get_nowait())
            except queue.Empty:
                pass
            if not pending and time.monotonic() - last < self.heartbeat_s:
                continue
            try:
                self.post("/api/edge/publish", {"messages": pending})
                pending, last, self.failures = [], time.monotonic(), 0
            except Exception as exc:  # noqa: BLE001 -- any failure: keep the messages, retry
                self.failures += 1
                if self.failures in (1, 10, 100):
                    print(f"cloud unreachable ({exc}); keeping {len(pending)} message(s)",
                          file=sys.stderr)
                pending = [m for m in pending if m.get("type") != "frame"][-5000:]
                self._stop.wait(min(2.0 * self.failures, 10.0))

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}",
                     "X-Edge-Name": self.name})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}: {exc.read()[:200]!r}") from exc


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a server + cameras
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cloud", default=os.environ.get("THIRDEYE_CLOUD"),
                    help="the server's base URL (or THIRDEYE_CLOUD)")
    ap.add_argument("--name", default=os.uname().nodename)
    ap.add_argument("--spec", type=Path)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int)
    src.add_argument("--video", type=Path)
    src.add_argument("--r3d", type=Path, action="append")
    src.add_argument("--stream", type=int, action="append")
    src.add_argument("--scenario", help="a scripted scenario name, to test the link")
    src.add_argument("--follow", metavar="URL",
                     help="Jenga: follow the game server's confirmed pulls (gamelink.py)")
    ap.add_argument("--reference", type=Path, help="jigsaw: the box picture")
    ap.add_argument("--empty", type=Path, help="jigsaw: the empty board photo")
    ap.add_argument("--no-hands", action="store_true")
    ap.add_argument("--send-frames", action="store_true",
                    help="also send overlay JPEGs (video leaves the laptop)")
    ap.add_argument("--logs", type=Path, default=ROOT / "eval" / "sessions")
    args = ap.parse_args(argv)

    load_env()
    token = os.environ.get("THIRDEYE_EDGE_TOKEN", "")
    if not args.cloud:
        ap.error("--cloud or THIRDEYE_CLOUD is required")
    if not token:
        print("warning: THIRDEYE_EDGE_TOKEN is not set; a cloud with sign-in on will refuse us",
              file=sys.stderr)
    link = CloudLink(args.cloud, token, args.name, send_frames=args.send_frames).start()

    from thirdeye.server.runner import Runner

    done = threading.Event()

    def on_end(record: SessionRecord, report: str) -> None:
        sid = link.upload_log(record, report)
        print(f"session saved in the cloud as {sid}" if sid else "session kept locally only")

    def publish(msg: dict[str, Any]) -> None:
        link.publish(msg)
        if msg["type"] == "alert":
            print(f"[{msg['t']:7.2f}] {msg['severity']} {msg['code']}: {msg['text']}", flush=True)
        if msg["type"] in ("end", "error"):
            print(msg.get("summary") or msg.get("message"), flush=True)
            done.set()

    runner = Runner(publish, log_dir=args.logs, on_end=on_end)

    if args.scenario:
        runner.start_replay(ROOT / "scripts" / "scenarios" / f"{args.scenario}.yaml", speed=1.0)
    else:
        if args.spec is None:
            ap.error("--spec is required with a camera source")
        from thirdeye.compiler.schema import ConstraintSpec, load_spec
        from thirdeye.session import Session

        spec = load_spec(args.spec)
        session = Session(spec)
        if args.follow:
            from thirdeye.server.gamelink import GameLink

            if not isinstance(spec, ConstraintSpec):
                ap.error("--follow needs a Jenga rulebook spec")
            runner.start_follow(spec, GameLink(args.follow), meta={"spec": str(args.spec)})
        elif isinstance(spec, ConstraintSpec):
            from thirdeye.live_jenga import JengaPipeline, _Recordings, _Streams
            from thirdeye.perception.hands import HandDetector

            sources = args.r3d or args.stream
            if not sources:
                ap.error("Jenga needs --r3d or --stream")
            feed = _Recordings(args.r3d) if args.r3d else _Streams(args.stream)
            from thirdeye.state.jenga.lattice import TowerLattice

            hands = None if args.no_hands else [HandDetector() for _ in sources]
            assert isinstance(session.state, TowerLattice)
            tower = JengaPipeline(session.state, cameras=len(sources), hands=hands)
            runner.start_jenga(session, tower, feed, source=f"edge:{args.name}",
                               send_frames=args.send_frames, meta={"spec": str(args.spec)})
        else:
            import cv2

            from thirdeye.live import build_pipeline

            pipeline = build_pipeline(spec, reference=args.reference, empty=args.empty,
                                      hands=not args.no_hands)
            capture = cv2.VideoCapture(str(args.video) if args.video else args.camera)
            if not capture.isOpened():
                print("cannot open the camera or video", file=sys.stderr)
                return 1
            runner.start_camera(session, pipeline, capture, source=f"edge:{args.name}",
                                video=args.video is not None, meta={"spec": str(args.spec)})
    try:
        while not done.wait(0.5):
            pass
    except KeyboardInterrupt:
        runner.stop()
        done.wait(10)
    link.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
