"""Follow the teammate's game server: its confirmed pulls drive a Third Eye session.

    POST /api/follow {"url": "http://127.0.0.1:8000"}     # from the Live page
    python -m thirdeye.server.edge --cloud ... --follow http://127.0.0.1:8000

The game server (`third-eye-testing/iphone_saved.py`) takes both phones. After
each MARK tap and the player's left / middle / right, its finder names the
pulled block from the face crops, the player may correct it, and the result
lands in `/status.json` as ``tower = {layers, removed: [[layer, slot], ...]}``
(0-based, bottom up). That list is the most reliable pull record we have --
the one-camera depth path scored 0/10 on real games, the crop finder 78 %
exact before the player's corrections -- so this module trusts it and reads
nothing else.

The game server records pulls, not placements. The rules judge a whole move,
so each pull is paired with a placement in the first legal top slot, and the
session says so (`ASSUMED_PLACEMENT`): the pull is measured, the placement is
not. Rules about placement therefore cannot fire here, and the one-hand rule
is not in force (the hand count is unknown, not assumed fine).

A pull list that is not the previous one plus new pulls (a correction, or the
tower re-detected) rebuilds the session from scratch, so the judgement always
matches the corrected record.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from thirdeye.state.jenga.lattice import Snapshot, legal_top_slots

ASSUMED_PLACEMENT = "placements are not observed; each is assumed in the first legal top slot"

Pull = tuple[int, int]


class LinkError(RuntimeError):
    """The game server could not be reached or answered something unexpected."""


@dataclass(frozen=True)
class GameStatus:
    layers: int
    pulls: tuple[Pull, ...]
    epoch: int = 0
    game_over: str | None = None
    markers: tuple[int, ...] = ()

    @classmethod
    def parse(cls, body: dict[str, Any]) -> GameStatus:
        tower = body.get("tower")
        if not isinstance(tower, dict) or "layers" not in tower:
            raise LinkError("no tower in the game server's status; is this the game server?")
        over = body.get("game_over")
        return cls(layers=int(tower["layers"]),
                   pulls=tuple((int(L), int(s)) for L, s in tower.get("removed", [])),
                   epoch=int(body.get("tower_epoch") or 0),
                   game_over=(over.get("reason") or "game over") if isinstance(over, dict) else None,
                   markers=tuple(sorted(set(body.get("markers") or []) | set(body.get("android_markers") or []))))


def full(layers: int, slots: int = 3) -> Snapshot:
    return tuple((1,) * slots for _ in range(layers))


def apply_pull(snap: Snapshot, pull: Pull) -> Snapshot:
    """The tower after taking `pull` and putting it down in the first legal top slot."""
    rows = [list(r) for r in snap]
    L, s = pull
    if 0 <= L < len(rows):
        rows[L][s] = 0
    after = tuple(tuple(r) for r in rows)
    dst = min(legal_top_slots(after))
    while len(rows) <= dst.layer:
        rows.append([0] * len(rows[0]))
    rows[dst.layer][dst.slot] = 1
    return tuple(tuple(r) for r in rows)


@dataclass
class GameLink:
    url: str
    timeout: float = 3.0
    #: What this session has applied so far, to tell new pulls from a correction.
    applied: tuple[Pull, ...] = field(default=(), init=False)
    layers: int = field(default=0, init=False)
    epoch: int = field(default=-1, init=False)

    def status(self) -> GameStatus:
        url = self.url.rstrip("/") + "/status.json"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                return GameStatus.parse(json.loads(resp.read()))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LinkError(f"cannot reach the game server at {self.url}: {exc}") from exc
        except ValueError as exc:
            raise LinkError(f"{url} did not answer JSON: {exc}") from exc

    def plan(self, st: GameStatus) -> tuple[bool, tuple[Pull, ...]]:
        """(rebuild?, pulls to apply). Rebuild when the record changed rather than grew."""
        grew = (st.layers == self.layers and st.epoch == self.epoch
                and st.pulls[:len(self.applied)] == self.applied)
        if grew:
            new = st.pulls[len(self.applied):]
        else:
            new = st.pulls
        self.applied, self.layers, self.epoch = st.pulls, st.layers, st.epoch
        return not grew, new
