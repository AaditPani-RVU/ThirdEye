"""The Third Eye server: sessions, specs, history and the live WebSocket (FR-UI-*, FR-CMP-9).

    python -m thirdeye.server.app                 # http://127.0.0.1:8000, no sign-in
    python -m thirdeye.server.app --host 0.0.0.0  # let another machine on the LAN watch

The same app runs on a laptop and on Cloud Run; the environment decides which
store (`store.py`) and which sign-in (`auth.py`) it uses. On the laptop it also
reads the camera; in the cloud the camera stays on the edge laptop, which
pushes its session's messages and final log here (`edge.py`) -- text only, no
video unless the edge is told to send overlay frames.

One session at a time. Every message goes to every connected browser and is
kept, so a browser that connects mid-session is sent the story so far first.
The message schema is in `protocol.py`.

HTTP (sign-in required when THIRDEYE_AUTH=firebase; * = editors only):
    GET  /healthz                        liveness, no sign-in
    GET  /api/config                     what the UI needs to know (auth on/off)
    GET  /api/me                         the signed-in user
    GET  /api/scenarios                  scripted scenarios that can be replayed
    GET  /api/specs                      procedure specs, with metadata
    GET  /api/specs/{name}               one spec
    PUT  /api/specs/{name}            *  {"spec": {...}} -> validated, then saved
    POST /api/validate                   {"spec": {...}} -> problems
    POST /api/compile                 *  {"text", "kind": dag|constraints|jigsaw, "cols", "rows",
                                          "model"} -> a validated spec, not yet saved
    POST /api/replay                     {"scenario": "jigsaw_rotated", "speed": 2.0, "noise": ""}
    POST /api/camera                  *  local camera or video (see CameraRequest)
    POST /api/stop
    GET  /api/sessions                   history, newest first
    GET  /api/sessions/{id}              summary, report and full log
    POST /api/sessions/{id}/report    *  rewrite the report with the LLM
    GET  /api/edge                       is an edge laptop connected?
    POST /api/edge/publish               edge token: {"messages": [...]}
    POST /api/edge/log                   edge token: {"record": {...}, "report": "..."}
    WS   /ws?token=                      the session, as protocol messages

Files are named, not pathed: a scenario or spec by its name, a video or
picture by a path inside the repository. Binds to localhost unless told otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from thirdeye.replay import Noise, ScriptError
from thirdeye.server import protocol
from thirdeye.server.auth import Auth, User, bearer
from thirdeye.server.runner import Runner
from thirdeye.server.store import Store, safe_name, store_from_env
from thirdeye.sessionlog import SessionRecord

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "scripts" / "scenarios"
SPECS = ROOT / "specs"
UI = ROOT / "ui" / "index.html"
WEB = ROOT / "web" / "dist"
DEMO = ROOT / "eval" / "demo"

#: An edge that has not been heard from for this long is shown as offline.
EDGE_TIMEOUT_S = 15.0


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


class SpecBody(BaseModel):
    spec: dict[str, Any]


class CompileRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    kind: Literal["dag", "constraints", "jigsaw"]
    cols: int | None = Field(default=None, ge=2, le=40)
    rows: int | None = Field(default=None, ge=2, le=40)
    model: str | None = None
    name: str | None = None


class EdgeMessages(BaseModel):
    messages: list[dict[str, Any]] = Field(max_length=500)


class EdgeLog(BaseModel):
    record: dict[str, Any]
    report: str = ""


class Hub:
    """Who is listening, and what they have missed."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.runner: Runner | None = None
        self.log_dir: Path | None = None
        self.store: Store | None = None
        self.edge_seen: float | None = None
        self.edge_name: str = ""

    def publish(self, msg: dict[str, Any]) -> None:
        """Called from the session thread, or from an edge's request."""
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

    def keep(self, record: SessionRecord, report: str, by: str = "server") -> str:
        """Save a finished session; returns its id."""
        run = record.meta.get("run") or record.procedure
        sid = f"{_slug(run)}_{time.strftime('%Y%m%d-%H%M%S')}"
        if self.store is not None:
            self.store.put_session(sid, record, report, by)
        return sid

    def fresh_runner(self, by: str = "server") -> Runner:
        if self.runner is not None and self.runner.running:
            self.runner.stop()
        self.runner = Runner(self.publish, log_dir=self.log_dir,
                             on_end=lambda rec, rep: self.keep(rec, rep, by))
        return self.runner

    @property
    def edge_online(self) -> bool:
        return self.edge_seen is not None and time.monotonic() - self.edge_seen < EDGE_TIMEOUT_S


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))[:50].strip("_.") or "session"


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


def _problems(spec: dict[str, Any]) -> tuple[Any, list[dict[str, str]]]:
    """Parse and validate; the parsed spec (or None) and every problem found."""
    from thirdeye.compiler.schema import parse_spec
    from thirdeye.compiler.validate import validate

    try:
        parsed = parse_spec(spec)
    except ValidationError as exc:
        return None, [{"severity": "error", "code": "schema", "target": "",
                       "message": f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"}
                      for e in exc.errors()]
    return parsed, [{"severity": p.severity, "code": p.code, "target": p.target,
                     "message": p.message} for p in validate(parsed)]


def _user(request: Request) -> User:
    return request.app.state.auth.user_from_token(bearer(request))


def _editor(u: Annotated[User, Depends(_user)]) -> User:
    if not u.editor:
        raise HTTPException(403, "this account may watch but not edit")
    return u


def _edge(request: Request) -> None:
    if not request.app.state.auth.edge_ok(bearer(request)):
        raise HTTPException(401, "edge token missing or wrong")
    hub = request.app.state.hub
    hub.edge_seen = time.monotonic()
    hub.edge_name = request.headers.get("x-edge-name", "edge")[:60]


UserDep = Annotated[User, Depends(_user)]
EditorDep = Annotated[User, Depends(_editor)]


def create_app(log_dir: Path | None = None, store: Store | None = None,
               auth: Auth | None = None) -> FastAPI:
    hub = Hub()
    hub.log_dir = log_dir
    db: Store = store if store is not None else store_from_env()
    hub.store = db
    auth = auth or Auth.from_env()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        hub.loop = asyncio.get_running_loop()
        yield
        if hub.runner is not None:
            hub.runner.stop()

    app = FastAPI(title="Third Eye", lifespan=lifespan)
    app.state.hub = hub
    app.state.auth = auth
    origins = [o.strip() for o in
               os.environ.get("THIRDEYE_CORS_ORIGINS", "").replace(";", ",").split(",")
               if o.strip()]
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"],
                           allow_headers=["Authorization", "Content-Type"])


    # --- open ---------------------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return {"auth": auth.enabled, "protocol": protocol.VERSION}

    # --- signed in ----------------------------------------------------------------

    @app.get("/api/me")
    def me(u: UserDep) -> dict[str, Any]:
        return {"uid": u.uid, "email": u.email, "name": u.name, "editor": u.editor}

    @app.get("/api/scenarios")
    def scenarios(_: UserDep) -> list[str]:
        return sorted(p.stem for p in SCENARIOS.glob("*.yaml"))

    @app.get("/api/specs")
    def specs(_: UserDep) -> list[dict[str, Any]]:
        return db.list_specs()

    @app.get("/api/specs/{name}")
    def get_spec(name: str, _: UserDep) -> dict[str, Any]:
        try:
            spec = db.get_spec(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if spec is None:
            raise HTTPException(404, f"no spec named {name!r}")
        return spec

    @app.put("/api/specs/{name}")
    def put_spec(name: str, body: SpecBody, u: EditorDep) -> dict[str, Any]:
        try:
            safe_name(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        parsed, problems = _problems(body.spec)
        if parsed is None or any(p["severity"] == "error" for p in problems):
            raise HTTPException(422, {"message": "the spec does not validate",
                                      "problems": problems})
        db.put_spec(name, parsed.model_dump(mode="json", exclude_none=True),
                           by=u.label, valid=True)
        return {"saved": name, "problems": problems}

    @app.post("/api/validate")
    def validate_spec(body: SpecBody, _: UserDep) -> dict[str, Any]:
        parsed, problems = _problems(body.spec)
        return {"valid": parsed is not None and not any(p["severity"] == "error"
                                                        for p in problems),
                "problems": problems}

    @app.post("/api/compile")
    def compile_(req: CompileRequest, _: EditorDep) -> dict[str, Any]:
        from thirdeye.compiler.compile import CompileError, compile_source
        from thirdeye.compiler.jigsaw import compile_instructions
        from thirdeye.compiler.llm import LLMError, make_llm

        try:
            llm = make_llm(req.model)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        spec: Any
        try:
            if req.kind == "jigsaw":
                if not (req.cols and req.rows):
                    raise HTTPException(400, "a jigsaw needs cols and rows")
                spec, warnings = compile_instructions(req.text, req.cols, req.rows, llm,
                                                      source_name="ui", procedure=req.name)
            else:
                spec, warnings = compile_source(req.text, req.kind, llm, source_name="ui")
        except LLMError as exc:
            raise HTTPException(502, f"the LLM call failed: {exc}") from exc
        except CompileError as exc:
            raise HTTPException(422, {"message": str(exc), "problems": exc.problems,
                                      "raw": exc.raw[:20_000]}) from exc
        return {"spec": spec.model_dump(mode="json", exclude_none=True),
                "warnings": [str(w) for w in warnings]}

    @app.post("/api/replay")
    def replay(req: ReplayRequest, u: UserDep) -> dict[str, str]:
        script = _named(SCENARIOS, req.scenario, ".yaml")
        try:
            noise = Noise.parse(req.noise)
            hub.fresh_runner(u.label).start_replay(script, speed=max(req.speed, 0.0),
                                                   noise=noise)
        except ScriptError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"started": req.scenario}

    @app.post("/api/camera")
    def camera(req: CameraRequest, u: EditorDep) -> dict[str, str]:
        import cv2

        from thirdeye.compiler.schema import DagSpec, parse_spec
        from thirdeye.live import build_pipeline
        from thirdeye.session import Session

        raw = db.get_spec(req.spec)
        if raw is None:
            raise HTTPException(404, f"no spec named {req.spec!r}")
        parsed = parse_spec(raw)
        if not isinstance(parsed, DagSpec):
            raise HTTPException(400, "the camera path runs placement specs (jigsaw, LEGO); "
                                "Jenga runs on the edge with live_jenga")
        video = _inside(req.video, "video")
        try:
            pipeline = build_pipeline(
                parsed, reference=_inside(req.reference, "reference"),
                empty=_inside(req.empty, "empty"), weights=req.weights,
                zero_shot=req.zero_shot, hands=req.hands)
        except (ValueError, FileNotFoundError, ImportError) as exc:
            raise HTTPException(400, str(exc)) from exc
        capture = cv2.VideoCapture(str(video) if video else int(req.camera or 0))
        if not capture.isOpened():
            raise HTTPException(400, "cannot open the camera or video")
        source = f"video:{video.name}" if video else f"camera:{req.camera}"
        hub.fresh_runner(u.label).start_camera(Session(parsed), pipeline, capture, source=source,
                                               video=video is not None,
                                               meta={"spec": req.spec})
        return {"started": source}

    @app.post("/api/stop")
    def stop(_: UserDep) -> dict[str, bool]:
        running = hub.runner is not None and hub.runner.running
        if hub.runner is not None:
            hub.runner.stop()
        return {"stopped": running}

    @app.get("/api/sessions")
    def sessions(_: UserDep) -> list[dict[str, Any]]:
        return db.list_sessions()

    @app.get("/api/sessions/{sid}")
    def session(sid: str, _: UserDep) -> dict[str, Any]:
        try:
            found = db.get_session(sid)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if found is None:
            raise HTTPException(404, f"no session {sid!r}")
        return found

    @app.post("/api/sessions/{sid}/report")
    def llm_report(sid: str, _: EditorDep, model: str | None = None) -> dict[str, str]:
        from thirdeye.compiler.llm import LLMError, make_llm
        from thirdeye.explain.report import with_llm

        found = session(sid, _)
        if not found.get("log"):
            raise HTTPException(404, "this session has no log")
        record = SessionRecord.from_json(found["log"])
        try:
            report = with_llm(record, make_llm(model, max_tokens=4000)).markdown()
        except (LLMError, ValueError) as exc:
            raise HTTPException(502, f"the LLM call failed: {exc}") from exc
        db.set_report(sid, report)
        return {"report": report}

    # --- the edge laptop ----------------------------------------------------------

    @app.get("/api/edge")
    def edge_status(_: UserDep) -> dict[str, Any]:
        return {"online": hub.edge_online, "name": hub.edge_name if hub.edge_online else "",
                "running": hub.runner is not None and hub.runner.running}

    @app.post("/api/edge/publish", dependencies=[Depends(_edge)])
    def edge_publish(body: EdgeMessages) -> dict[str, int]:
        for msg in body.messages:
            if msg.get("v") != protocol.VERSION or not isinstance(msg.get("type"), str):
                raise HTTPException(400, f"not a protocol v{protocol.VERSION} message")
            hub.publish(msg)
        return {"accepted": len(body.messages)}

    @app.post("/api/edge/log", dependencies=[Depends(_edge)])
    def edge_log(body: EdgeLog) -> dict[str, str]:
        try:
            record = SessionRecord.from_json(body.record)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"not a session log: {exc}") from exc
        return {"id": hub.keep(record, body.report, by=f"edge:{hub.edge_name}")}

    # --- the live socket ----------------------------------------------------------

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        try:
            auth.user_from_token(bearer(socket))
        except HTTPException:
            await socket.close(code=4401)
            return
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

    # --- a phone camera (the capture app; see phone.py) ------------------------------
    # The app cannot sign in, so these exist only when sign-in is off: on a laptop
    # or a tailnet, never on Cloud Run.

    if not auth.enabled:
        from fastapi.concurrency import run_in_threadpool
        from fastapi.responses import Response

        from thirdeye.server.phone import PhoneFeed

        phone = PhoneFeed(ROOT / "data" / "phone")
        app.state.phone = phone

        @app.post("/android")
        async def android(request: Request) -> dict[str, Any]:
            body = await request.body()
            seq = int(request.headers.get("x-frame", 0) or 0)
            return await run_in_threadpool(phone.push, body, seq)

        @app.api_route("/cmd/{cmd}", methods=["GET", "POST"])
        def phone_cmd(cmd: str) -> Response:
            if not phone.command(cmd):
                raise HTTPException(404, "unknown command")
            return Response("OK", media_type="text/plain")

        @app.get("/api/phone")
        def phone_status() -> dict[str, Any]:
            return phone.status()

        @app.get("/api/phone/frame.jpg")
        def phone_frame() -> Response:
            jpeg = phone.frame()
            if jpeg is None:
                raise HTTPException(404, "no frame yet")
            return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # --- the UI -------------------------------------------------------------------

    # Real-game pictures for the "Real games" page (scripts/make_demo_data.py):
    # block-face crops only, built locally, never present on Cloud Run.
    if DEMO.exists():
        app.mount("/demo", StaticFiles(directory=DEMO), name="demo")
    if WEB.exists():
        app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    else:
        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(UI)

    return app


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - starts a server
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    ap.add_argument("--logs", type=Path, default=ROOT / "eval" / "sessions",
                    help="also write each session's log here")
    args = ap.parse_args(argv)
    from thirdeye.compiler.llm import load_env

    load_env()
    print(f"Third Eye on http://{args.host}:{args.port}  (protocol v{protocol.VERSION})")
    with contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(create_app(args.logs), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
