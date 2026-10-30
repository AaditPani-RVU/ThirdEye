"""Following the game server: its pull record, as moves the rules can judge."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from thirdeye.compiler.schema import load_spec
from thirdeye.server.gamelink import GameLink, GameStatus, LinkError, apply_pull, full
from thirdeye.server.runner import Runner

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "example_constraints.json"


def status(layers=10, removed=(), epoch=0, over=None) -> dict:
    return {"tower": {"layers": layers, "removed": [list(p) for p in removed]},
            "tower_epoch": epoch, "game_over": over, "markers": [0, 1], "android_markers": [3]}


def test_the_status_is_read_as_layers_and_pulls() -> None:
    st = GameStatus.parse(status(removed=[(3, 1)], over={"reason": "fell"}))
    assert (st.layers, st.pulls, st.game_over, st.markers) == (10, ((3, 1),), "fell", (0, 1, 3))
    with pytest.raises(LinkError):
        GameStatus.parse({"markers": []})


def test_a_pull_goes_on_top_in_the_first_legal_slot() -> None:
    snap = apply_pull(full(3), (0, 1))
    assert snap == ((1, 0, 1), (1, 1, 1), (1, 1, 1), (1, 0, 0))
    snap = apply_pull(snap, (1, 0))
    assert snap[3] == (1, 1, 0) and snap[1] == (0, 1, 1)


def test_new_pulls_extend_and_a_correction_rebuilds() -> None:
    link = GameLink("http://x")
    assert link.plan(GameStatus.parse(status(removed=[(3, 1)]))) == (True, ((3, 1),))
    assert link.plan(GameStatus.parse(status(removed=[(3, 1), (4, 0)]))) == (False, ((4, 0),))
    assert link.plan(GameStatus.parse(status(removed=[(3, 1), (4, 0)]))) == (False, ())
    rebuilt = link.plan(GameStatus.parse(status(removed=[(3, 1), (5, 0)])))   # the player corrected
    assert rebuilt == (True, ((3, 1), (5, 0)))
    assert link.plan(GameStatus.parse(status(removed=[(3, 1), (5, 0)], epoch=1)))[0]


class _Game:
    """A stand-in game server whose status the test changes."""

    def __init__(self) -> None:
        self.body = status()
        game = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                data = json.dumps(game.body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def test_the_runner_judges_each_confirmed_pull_and_ends_on_game_over() -> None:
    game = _Game()
    msgs: list[dict] = []
    done = threading.Event()

    def publish(m: dict) -> None:
        msgs.append(m)
        if m["type"] in ("end", "error"):
            done.set()

    runner = Runner(publish)
    runner.start_follow(load_spec(SPEC), GameLink(game.url), poll_s=0.05)
    game.body = status(removed=[(2, 1)])                 # a legal pull
    _wait(lambda: _tower(msgs) and _tower(msgs)["rows"][2] == [1, 0, 1])
    game.body = status(removed=[(2, 1), (9, 0)])        # from the top complete layer: against the rules
    _wait(lambda: any(m["type"] == "alert" for m in msgs))
    game.body = status(removed=[(2, 1), (9, 0)], over={"reason": "you reported that pieces fell"})
    assert done.wait(5)
    game.server.shutdown()

    assert msgs[0]["type"] == "hello" and msgs[0]["source"] == f"game:{game.url}"
    codes = [m["code"] for m in msgs if m["type"] == "alert"]
    assert codes[0] == "illegal_source_layer"
    assert msgs[-1]["type"] == "end" and msgs[-1]["outcome"]


def test_an_unreachable_game_server_fails_at_once() -> None:
    with pytest.raises(LinkError):
        Runner(lambda m: None).start_follow(load_spec(SPEC), GameLink("http://127.0.0.1:9", timeout=0.5))


def _tower(msgs):
    states = [m for m in msgs if m["type"] == "state" and m.get("tower")]
    return states[-1]["tower"] if states else None


def _wait(cond, timeout: float = 5.0) -> None:
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")
