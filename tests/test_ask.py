"""Ask the referee: the session as text, and answers held to it."""

from __future__ import annotations

import json

import pytest

from thirdeye.explain.ask import ask, context


class FakeLLM:
    name = "fake"

    def __init__(self, answer: str, cites: list[str]) -> None:
        self.reply = json.dumps({"answer": answer, "cites": cites})
        self.seen = ""

    def complete(self, system: str, user: str, schema) -> str:
        self.seen = user
        return self.reply


HISTORY = [
    {"v": 1, "type": "hello", "procedure": "jenga_game", "kind": "constraints",
     "source": "replay:x", "checklist": [],
     "rules": [{"id": "c1", "text": "One hand at a time", "code": "two_handed_move"}]},
    {"v": 1, "type": "state", "t": 12.0, "checklist": [], "next": None, "progress": "2 moves",
     "tower": {"rows": [[1, 1, 1], [1, 0, 1], [0, 1, 1]], "top_layer": 2,
               "top_complete_layer": 0, "legal_slots": [[2, 0]]}},
    {"v": 1, "type": "alert", "t": 11.0, "code": "two_handed_move", "severity": "alert",
     "text": "One hand at a time", "target": "move", "rule": "c1"},
]


def test_the_context_is_what_the_camera_and_checker_produced() -> None:
    text = context(HISTORY)
    assert "two_handed_move: One hand at a time" in text
    assert "layer 1: #.#" in text and "layer 2: .##" in text
    assert "layer 2 slot 0" in text
    assert "t=11.0 two_handed_move" in text


def test_no_session_means_no_question() -> None:
    assert context([]) == ""
    with pytest.raises(ValueError):
        ask([], "anything?", FakeLLM("", []))


def test_citations_that_name_nothing_in_the_session_are_dropped() -> None:
    llm = FakeLLM("Two hands were used.",
                  ["two_handed_move", "two_handed_move@11.0", "two_handed_move@30.0", "made_up"])
    a = ask(HISTORY, "Why the alert?", llm)
    assert a.cites == ["two_handed_move", "two_handed_move@11.0"]
    assert a.dropped == ["two_handed_move@30.0", "made_up"]
    assert "Question: Why the alert?" in llm.seen
    assert a.context in llm.seen


def test_an_unreadable_answer_is_reported_not_raised() -> None:
    class Garbled(FakeLLM):
        def complete(self, system, user, schema):
            return "not json"

    a = ask(HISTORY, "?", Garbled("", []))
    assert "could not be read" in a.answer and a.cites == []


def test_the_server_answers_about_the_live_session(monkeypatch, tmp_path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import thirdeye.compiler.llm as llm_mod
    from thirdeye.server.app import create_app
    from thirdeye.server.auth import Auth
    from thirdeye.server.store import LocalStore

    monkeypatch.setattr(llm_mod, "make_llm",
                        lambda *a, **k: FakeLLM("Layer 2 slot 0.", ["two_handed_move"]))
    app = create_app(log_dir=tmp_path, store=LocalStore(tmp_path / "s"), auth=Auth())
    with TestClient(app) as c:
        assert c.post("/api/ask", json={"question": "where next?"}).status_code == 400
        for m in HISTORY:
            app.state.hub.publish(m)
        r = c.post("/api/ask", json={"question": "where next?"}).json()
    assert r["answer"] == "Layer 2 slot 0." and r["cites"] == ["two_handed_move"]
    assert "layer 1: #.#" in r["context"]
