"""The demo server: HTTP to start and stop a session, a WebSocket to watch it (FR-UI-*).

    python -m thirdeye.server.app                 # http://127.0.0.1:8000
    python -m thirdeye.server.app --host 0.0.0.0  # let another machine on the LAN watch

One session at a time, run by `runner.Runner` on a worker thread. Every message
it produces goes to every connected browser, and is kept, so a browser that
connects mid-session is sent the story so far first. The message schema is in
`protocol.py`.

HTTP:
    GET  /                    the UI (ui/index.html)
    GET  /api/scenarios       scripted scenarios that can be replayed
    GET  /api/specs           procedure specs
    POST /api/replay          {"scenario": "jigsaw_rotated", "speed": 2.0, "noise": "drop=0.1"}
    POST /api/camera          {"spec": "jigsaw_4x3", "camera": 0 | "video": "data/raw/R1.mp4",
                               "reference": "data/box.jpg", "empty": null, "hands": true}
    POST /api/stop
    WS   /ws                  the session, as protocol messages

Files are named, not pathed: a scenario or spec by its name, a video or
picture by a path inside the repository. The server reads nothing outside it.
Binds to localhost unless told otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from thirdeye.replay import Noise, ScriptError
from thirdeye.server import protocol
from thirdeye.server.runner import Runner

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "scripts" / "scenarios"
SPECS = ROOT / "specs"
UI = ROOT / "ui" / "index.html"


class ReplayRequest(BaseModel):
    scenario: str
    speed: float = 1.0
    noise: str = ""


class CameraRequest(BaseModel):
    spec: str
    camera: int | None = 0
    video: str | None = None
    reference: str | None = None
    empty: str | None = None
    hands: bool = True
    weights: str | None = None
    zero_shot: bool = False


class Hub:
    """Who is listening, and what they have missed."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.runner: Runner | None = None
        self.log_dir: Path | None = None

    def publish(self, msg: dict[str, Any]) -> None:
        """Called from the session thread."""
        if msg["type"] == "hello":
            self.history = []
        if msg["type"] != "frame":      # frames are for now, not for late joiners
            self.history.append(msg)
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self._send(msg), self.loop)

    async def _send(self, msg: dict[str, Any]) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except (WebSocketDisconnect, RuntimeError):  # a closed tab; forget it
                self.clients.discard(ws)

    def fresh_runner(self) -> Runner:
        if self.runner is not None and self.runner.running:
            self.runner.stop()
        self.runner = Runner(self.publish, log_dir=self.log_dir)
        return self.runner


def _inside(rel: str | None, what: str) -> Path | None:
    """A repo-relative path that must stay inside the repo."""
    if rel is None:
        return None
    path = (ROOT / rel).resolve()
    if ROOT not in path.parents or not path.exists():
        raise HTTPException(400, f"{what} must be an existing file inside the repository")
    return path


def _named(folder: Path, name: str, suffix: str) -> Path:
    path = folder / f"{name}{suffix}"
    if "/" in name or "\\" in name or ".." in name or not path.exists():
        raise HTTPException(404, f"no {suffix} named {name!r} in {folder.name}/")
    return path


def create_app(log_dir: Path | None = None) -> FastAPI:
    hub = Hub()
    hub.log_dir = log_dir

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        hub.loop = asyncio.get_running_loop()
        yield
        if hub.runner is not None:
            hub.runner.stop()

    app = FastAPI(title="Third Eye", lifespan=lifespan)
    app.state.hub = hub

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(UI)

    @app.get("/api/scenarios")
    def scenarios() -> list[str]:
        return sorted(p.stem for p in SCENARIOS.glob("*.yaml"))

    @app.get("/api/specs")
    def specs() -> list[str]:
        return sorted(p.stem for p in SPECS.glob("*.json"))

    @app.post("/api/replay")
    def replay(req: ReplayRequest) -> dict[str, str]:
        script = _named(SCENARIOS, req.scenario, ".yaml")
        try:
            noise = Noise.parse(req.noise)
            hub.fresh_runner().start_replay(script, speed=max(req.speed, 0.0), noise=noise)
        except ScriptError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"started": req.scenario}

    @app.post("/api/camera")
    def camera(req: CameraRequest) -> dict[str, str]:
        import cv2

        from thirdeye.compiler.schema import DagSpec, load_spec
        from thirdeye.live import build_pipeline
        from thirdeye.session import Session

        spec = load_spec(_named(SPECS, req.spec, ".json"))
        if not isinstance(spec, DagSpec):
            raise HTTPException(400, "the camera path runs placement specs (LEGO, jigsaw)")
        video = _inside(req.video, "video")
        try:
            pipeline = build_pipeline(
                spec, reference=_inside(req.reference, "reference"),
                empty=_inside(req.empty, "empty"), weights=req.weights,
                zero_shot=req.zero_shot, hands=req.hands)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(400, str(exc)) from exc
        capture = cv2.VideoCapture(str(video) if video else int(req.camera or 0))
        if not capture.isOpened():
            raise HTTPException(400, "cannot open the camera or video")
        source = f"video:{video.name}" if video else f"camera:{req.camera}"
        hub.fresh_runner().start_camera(Session(spec), pipeline, capture, source=source,
                                        video=video is not None)
        return {"started": source}

    @app.post("/api/stop")
    def stop() -> dict[str, bool]:
        running = hub.runner is not None and hub.runner.running
        if hub.runner is not None:
            hub.runner.stop()
        return {"stopped": running}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        for msg in list(hub.history):
            await socket.send_json(msg)
        hub.clients.add(socket)
        try:
            while True:  # nothing inbound is acted on (IF-WS-1); just notice the close
                await socket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(socket)

    return app


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - starts a server
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--logs", type=Path, default=ROOT / "eval" / "sessions",
                    help="write each session's log here")
    args = ap.parse_args(argv)
    print(f"Third Eye on http://{args.host}:{args.port}  (protocol v{protocol.VERSION})")
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(create_app(args.logs), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
