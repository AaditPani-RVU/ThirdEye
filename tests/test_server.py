"""The demo server: HTTP to start a session, the WebSocket to watch it (FR-UI-*, IF-WS-*)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from thirdeye.server import protocol
from thirdeye.server.app import create_app
from thirdeye.server.auth import Auth
from thirdeye.server.store import LocalStore


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(log_dir=tmp_path, store=LocalStore(tmp_path / "store"), auth=Auth())
    with TestClient(app) as c:
        c.logs = tmp_path  # type: ignore[attr-defined]
        yield c


def _until_end(ws) -> list[dict]:
    out = []
    while True:
        msg = ws.receive_json()
        out.append(msg)
        if msg["type"] in ("end", "error"):
            return out


def test_the_page_and_the_lists_are_served(client) -> None:
    assert "Third Eye" in client.get("/").text
    assert "jenga_fouls" in client.get("/api/scenarios").json()
    assert "jigsaw_4x3" in [s["name"] for s in client.get("/api/specs").json()]


def test_a_replay_streams_hello_state_alerts_and_the_report(client) -> None:
    with client.websocket_connect("/ws") as ws:
        assert client.post("/api/replay", json={"scenario": "jigsaw_rotated",
                                                "speed": 0}).status_code == 200
        msgs = _until_end(ws)
    assert all(m["v"] == protocol.VERSION for m in msgs)
    kinds = [m["type"] for m in msgs]
    assert kinds[0] == "hello" and kinds[-1] == "end" and "state" in kinds
    alerts = [m for m in msgs if m["type"] == "alert"]
    assert [(a["code"], a["target"]) for a in alerts] == [("wrong_rotation", "r2c2")]
    last_state = [m for m in msgs if m["type"] == "state"][-1]
    assert last_state["progress"] == "11/12"
    assert "wrong rotation" in msgs[-1]["report"]
    assert (client.logs / "jigsaw_rotated.json").exists()


def test_a_jenga_replay_carries_the_legality_panel(client) -> None:
    with client.websocket_connect("/ws") as ws:
        client.post("/api/replay", json={"scenario": "jenga_fouls", "speed": 0})
        msgs = _until_end(ws)
    towers = [m["tower"] for m in msgs if m["type"] == "state" and "tower" in m]
    assert towers and towers[0]["top_complete_layer"] == 17
    assert [18, 0] in towers[0]["legal_slots"]
    assert msgs[-1]["outcome"] == "game_over"


def test_a_late_joiner_gets_the_story_so_far(client) -> None:
    with client.websocket_connect("/ws") as ws:
        client.post("/api/replay", json={"scenario": "model_a_skip", "speed": 0})
        _until_end(ws)
    with client.websocket_connect("/ws") as late:
        msgs = _until_end(late)
    assert msgs[0]["type"] == "hello" and [m["code"] for m in msgs if m["type"] == "alert"]


def test_stop_ends_a_real_time_replay_early(client) -> None:
    with client.websocket_connect("/ws") as ws:
        client.post("/api/replay", json={"scenario": "model_a_correct", "speed": 1})
        assert ws.receive_json()["type"] == "hello"
        assert client.post("/api/stop").json() == {"stopped": True}
        msgs = _until_end(ws)
    assert msgs[-1]["type"] == "end"
    assert "12/12" not in msgs[-1]["summary"]   # stopped well before the build was done


@pytest.mark.parametrize("body", [
    {"scenario": "../../pyproject"},
    {"scenario": "no_such_scenario"},
])
def test_only_named_scenarios_can_be_run(client, body) -> None:
    assert client.post("/api/replay", json=body).status_code == 404


def test_bad_noise_is_refused(client) -> None:
    assert client.post("/api/replay", json={"scenario": "jigsaw_rotated",
                                            "noise": "blur=1"}).status_code == 400


def test_a_camera_request_cannot_read_outside_the_repository(client) -> None:
    r = client.post("/api/camera", json={"spec": "jigsaw_4x3", "video": "../../../etc/passwd",
                                         "reference": "../../../etc/hosts", "hands": False})
    assert r.status_code == 400
