"""The /game/... pass-through to the game server: only the listed paths, only to the configured address."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from thirdeye.server import gameproxy


class _Fake(BaseHTTPRequestHandler):
    seen: list = []

    def do_GET(self):
        body = json.dumps({"tower": {"layers": 9, "removed": []}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        _Fake.seen.append((self.path, self.rfile.read(n)))
        self.send_response(200); self.end_headers(); self.wfile.write(b"OK")

    def log_message(self, *a):  # quiet
        pass


@pytest.fixture()
def fake(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(gameproxy, "GAME_URL", f"http://127.0.0.1:{srv.server_port}")
    _Fake.seen.clear()
    yield srv
    srv.shutdown()


def test_status_and_command_pass_through(fake):
    code, body, ctype = gameproxy.forward("GET", "status.json")
    assert code == 200 and json.loads(body)["tower"]["layers"] == 9 and "json" in ctype
    code, body, _ = gameproxy.forward("POST", "cmd/tower", b'{"layers": 12, "removed": []}')
    assert code == 200 and _Fake.seen == [("/cmd/tower", b'{"layers": 12, "removed": []}')]


@pytest.mark.parametrize("method,path", [("GET", "../../etc/passwd"), ("GET", "upload"), ("POST", "cmd/anything"),
                                         ("POST", "status.json"), ("GET", "cmd/start")])
def test_other_paths_are_refused(fake, method, path):
    with pytest.raises(KeyError):
        gameproxy.forward(method, path)


def test_unreachable_game_server_is_reported(monkeypatch):
    monkeypatch.setattr(gameproxy, "GAME_URL", "http://127.0.0.1:9")
    with pytest.raises(gameproxy.GameUnreachable):
        gameproxy.forward("GET", "status.json", timeout=1.0)


def test_routes_are_mounted_and_refuse_other_paths(fake, tmp_path):
    from fastapi.testclient import TestClient
    from thirdeye.server.app import create_app
    from thirdeye.server.auth import Auth
    from thirdeye.server.store import LocalStore
    app = create_app(log_dir=tmp_path, store=LocalStore(tmp_path / "store"), auth=Auth())
    with TestClient(app) as c:
        assert c.get("/game/status.json").json()["tower"]["layers"] == 9
        assert c.get("/game/secrets").status_code == 404
        assert c.post("/game/cmd/mark").status_code == 200
        assert c.get("/game_info").json()["url"].startswith("http://127.0.0.1:")
