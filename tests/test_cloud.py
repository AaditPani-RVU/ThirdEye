"""The cloud side of the server: storage, sign-in, spec editing, history, the edge link."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from thirdeye.server import protocol
from thirdeye.server.app import create_app
from thirdeye.server.auth import Auth
from thirdeye.server.edge import CloudLink
from thirdeye.server.store import CloudStore, LocalStore, safe_name
from thirdeye.sessionlog import SessionRecord

SPECS = Path(__file__).resolve().parents[1] / "specs"


def _until_end(ws) -> list[dict]:
    out = []
    while True:
        msg = ws.receive_json()
        out.append(msg)
        if msg["type"] in ("end", "error"):
            return out


# --- an in-memory Firestore + GCS -----------------------------------------------------


class FakeDoc:
    def __init__(self, coll, id_):
        self.coll, self.id = coll, id_

    def set(self, data):
        self.coll.docs[self.id] = dict(data)

    def update(self, data):
        self.coll.docs[self.id].update(data)

    def get(self):
        data = self.coll.docs.get(self.id)
        return SimpleNamespace(exists=data is not None, to_dict=lambda: dict(data or {}),
                               id=self.id)

    def collection(self, name):
        return self.coll.db.collection(f"{self.coll.name}/{self.id}/{name}")


class FakeColl:
    def __init__(self, db, name):
        self.db, self.name, self.docs = db, name, {}

    def document(self, id_):
        return FakeDoc(self, id_)

    def stream(self):
        return [FakeDoc(self, k).get() for k in self.docs]


class FakeDB:
    def __init__(self):
        self.colls = {}

    def collection(self, name):
        return self.colls.setdefault(name, FakeColl(self, name))


class FakeBucket:
    name = "bucket"

    def __init__(self):
        self.blobs = {}

    def blob(self, path):
        bucket = self

        class Blob:
            def upload_from_string(self, data, content_type=None):
                bucket.blobs[path] = data

            def exists(self):
                return path in bucket.blobs

            def download_as_text(self):
                return bucket.blobs[path]

        return Blob()


def _record() -> SessionRecord:
    from thirdeye.replay import Replay

    root = Path(__file__).resolve().parents[1]
    return Replay.load(root / "scripts" / "scenarios" / "jigsaw_rotated.yaml").run()


@pytest.fixture(params=["local", "cloud"])
def store(request, tmp_path):
    if request.param == "local":
        return LocalStore(tmp_path / "store")
    return CloudStore(FakeDB(), FakeBucket())


def test_a_store_keeps_specs_and_sessions(store) -> None:
    spec = json.loads((SPECS / "jigsaw_4x3.json").read_text())
    store.put_spec("mine", spec, by="a@b.c", valid=True)
    assert store.get_spec("mine") == spec
    listed = {s["name"]: s for s in store.list_specs()}
    assert listed["mine"]["kind"] == "dag" and listed["mine"]["created_by"] == "a@b.c"

    record = _record()
    store.put_session("run1", record, "# report", by="a@b.c")
    (summary,) = [s for s in store.list_sessions() if s["id"] == "run1"]
    assert summary["alert_count"] == 1 and summary["report"] == "# report"
    full = store.get_session("run1")
    assert SessionRecord.from_json(full["log"]).alerts[0].code == "wrong_rotation"
    store.set_report("run1", "# better")
    assert store.get_session("run1")["report"] == "# better"
    assert store.get_session("nope") is None


def test_the_cloud_store_writes_every_alert_as_a_document() -> None:
    db = FakeDB()
    CloudStore(db, FakeBucket()).put_session("run1", _record(), "", by="x")
    first = next(iter(db.colls["sessions/run1/alerts"].docs.values()))
    assert first["code"] == "wrong_rotation"


@pytest.mark.parametrize("name", ["../x", "a/b", "", ".hidden", "x" * 81])
def test_names_are_held_to_a_plain_alphabet(name) -> None:
    with pytest.raises(ValueError):
        safe_name(name)


# --- sign-in ------------------------------------------------------------------------


def _claims(token: str) -> dict:
    if token == "bad":
        raise ValueError("expired")
    email, _, verified = token.partition(":")
    return {"uid": email, "email": email, "email_verified": verified != "unverified"}


@pytest.fixture
def secured(tmp_path):
    auth = Auth("firebase", _claims, frozenset({"lead@x.org"}), edge_token="edge-secret")
    app = create_app(store=LocalStore(tmp_path / "store"), auth=auth)
    with TestClient(app) as c:
        yield c


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_sign_in_is_required_when_on(secured) -> None:
    assert secured.get("/healthz").status_code == 200
    assert secured.get("/api/config").json()["auth"] is True
    assert secured.get("/api/specs").status_code == 401
    assert secured.get("/api/specs", headers=_h("bad")).status_code == 401
    assert secured.get("/api/specs", headers=_h("lead@x.org")).status_code == 200


def test_only_listed_accounts_edit(secured) -> None:
    spec = json.loads((SPECS / "jigsaw_4x3.json").read_text())
    viewer, lead = _h("guest@y.org"), _h("lead@x.org")
    assert secured.get("/api/me", headers=viewer).json()["editor"] is False
    assert secured.put("/api/specs/s1", json={"spec": spec}, headers=viewer).status_code == 403
    assert secured.put("/api/specs/s1", json={"spec": spec}, headers=lead).status_code == 200
    # an unverified address that matches the list is still only a viewer
    assert secured.get("/api/me", headers=_h("lead@x.org:unverified")).json()["editor"] is False


def test_the_socket_refuses_a_missing_token(secured) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), secured.websocket_connect("/ws") as ws:
        ws.receive_json()
    with secured.websocket_connect("/ws?token=lead@x.org"):
        pass


def test_the_edge_needs_its_token_and_its_messages_reach_browsers(secured) -> None:
    hello = {"v": protocol.VERSION, "type": "hello", "procedure": "p", "kind": "dag",
             "source": "edge", "checklist": [], "rules": []}
    assert secured.post("/api/edge/publish", json={"messages": [hello]}).status_code == 401
    assert secured.post("/api/edge/publish", json={"messages": [hello]},
                        headers=_h("wrong")).status_code == 401
    with secured.websocket_connect("/ws?token=lead@x.org") as ws:
        r = secured.post("/api/edge/publish", json={"messages": [hello]},
                         headers={**_h("edge-secret"), "X-Edge-Name": "lab-laptop"})
        assert r.json() == {"accepted": 1}
        assert ws.receive_json()["procedure"] == "p"
    status = secured.get("/api/edge", headers=_h("lead@x.org")).json()
    assert status["online"] and status["name"] == "lab-laptop"
    bad = secured.post("/api/edge/publish", json={"messages": [{"type": "hello"}]},
                       headers=_h("edge-secret"))
    assert bad.status_code == 400


def test_an_edge_log_lands_in_history(secured) -> None:
    rec = _record()
    r = secured.post("/api/edge/log", json={"record": rec.to_json(), "report": "r"},
                     headers=_h("edge-secret"))
    sid = r.json()["id"]
    got = secured.get(f"/api/sessions/{sid}", headers=_h("guest@y.org")).json()
    assert got["alert_count"] == 1 and got["log"]["procedure"] == rec.procedure


# --- specs, compile, history (no sign-in) ----------------------------------------------


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(store=LocalStore(tmp_path / "store"), auth=Auth())) as c:
        yield c


def test_a_spec_is_validated_before_it_is_saved(client) -> None:
    spec = json.loads((SPECS / "jigsaw_4x3.json").read_text())
    assert client.post("/api/validate", json={"spec": spec}).json()["valid"] is True
    spec["steps"][1]["hard_depends_on"] = [spec["steps"][1]["id"]]   # depends on itself
    checked = client.post("/api/validate", json={"spec": spec}).json()
    assert not checked["valid"] and checked["problems"]
    assert client.put("/api/specs/broken", json={"spec": spec}).status_code == 422
    assert client.post("/api/validate", json={"spec": {"kind": "dag"}}).json()["valid"] is False
    assert client.put("/api/specs/bad name!", json={"spec": spec}).status_code == 400


def test_compile_returns_a_spec_without_saving_it(client, monkeypatch) -> None:
    from test_compile import Scripted

    text = (SPECS / "example_constraints.json").read_text()
    monkeypatch.setattr("thirdeye.compiler.llm.make_llm",
                        lambda model=None, max_tokens=16000: Scripted([text]))
    r = client.post("/api/compile", json={"text": "the rules", "kind": "constraints"})
    assert r.status_code == 200, r.text
    assert r.json()["spec"]["kind"] == "constraints"
    assert "example" not in [s["name"] for s in client.get("/api/specs").json()
                             if s.get("created_by") != "repository"]


def test_a_failed_compile_lists_the_problems(client, monkeypatch) -> None:
    from test_compile import Scripted

    monkeypatch.setattr("thirdeye.compiler.llm.make_llm",
                        lambda model=None, max_tokens=16000: Scripted(["{", "{"]))
    r = client.post("/api/compile", json={"text": "x", "kind": "dag"})
    assert r.status_code == 422 and r.json()["detail"]["problems"]
    r = client.post("/api/compile", json={"text": "x", "kind": "jigsaw"})
    assert r.status_code == 400


def test_a_replay_is_kept_in_history(client) -> None:
    with client.websocket_connect("/ws") as ws:
        client.post("/api/replay", json={"scenario": "jigsaw_rotated", "speed": 0})
        _until_end(ws)
    (s,) = client.get("/api/sessions").json()
    assert s["alert_count"] == 1 and s["started_at"] and "wrong rotation" in s["report"]
    assert client.get(f"/api/sessions/{s['id']}").json()["log"]["alerts"]


# --- the edge's link ---------------------------------------------------------------


def test_the_link_batches_drops_frames_and_retries() -> None:
    sent, fail = [], {"n": 1}

    def post(path, body):
        if fail["n"]:
            fail["n"] -= 1
            raise OSError("offline")
        sent.append(body["messages"])
        return {}

    link = CloudLink("http://x", "t", post=post, heartbeat_s=60).start()
    link.publish({"type": "hello"})
    link.publish({"type": "frame", "jpeg": "..."})
    link.publish({"type": "alert"})
    import time

    deadline = time.monotonic() + 10
    while sum(len(b) for b in sent) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    link.close()
    assert [m["type"] for b in sent for m in b] == ["hello", "alert"]
