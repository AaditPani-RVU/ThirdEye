"""A pass-through to the game server (third-eye-testing/iphone_saved.py), so the React "Game" tab can use
its camera pictures and its controls from this server's own address (no CORS, nothing to configure).

    GET  /game/<picture or status>     preview.jpg  android.jpg  status.json  coverage.txt
    POST /game/cmd/<command>           start stop mark label tower redetect dataset/on dataset/off

Only those paths go through, and only to the one address set by GAME_SERVER (default http://127.0.0.1:8000):
a browser cannot point this at anything else.  Mounted only when sign-in is off, like the phone feed.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

GAME_URL = os.environ.get("GAME_SERVER", "http://127.0.0.1:8000").rstrip("/")

GET_OK = {"preview.jpg", "android.jpg", "status.json", "coverage.txt"}
POST_OK = {"cmd/start", "cmd/stop", "cmd/mark", "cmd/label", "cmd/tower", "cmd/redetect",
           "cmd/dataset/on", "cmd/dataset/off"}


class GameUnreachable(RuntimeError):
    pass


def forward(method: str, path: str, body: bytes | None = None, timeout: float = 4.0) -> tuple[int, bytes, str]:
    """(status, body, content-type) from the game server; raises GameUnreachable if it is not there."""
    allowed = GET_OK if method == "GET" else POST_OK
    if path not in allowed:
        raise KeyError(path)
    req = urllib.request.Request(f"{GAME_URL}/{path}", data=body if method == "POST" else None, method=method,
                                 headers={"Content-Type": "application/json"} if method == "POST" else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "application/octet-stream")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get("Content-Type", "text/plain")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GameUnreachable(f"cannot reach the game server at {GAME_URL}: {exc}") from exc
