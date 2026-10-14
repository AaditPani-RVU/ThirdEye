"""The WebSocket message schema, versioned (IF-WS-1, IF-WS-2).

Every message is one JSON object with `v` (this schema's version) and `type`.
Outbound only: the browser never sends frames or judgements, it only starts
and stops sessions through the HTTP API (FR-UI-4). Bump `VERSION` on any change
a UI written against the previous version would misread.

    hello   a session started    {procedure, kind, source, checklist, rules}
    state   progress changed     {t, checklist, next, progress, tower?}
    alert   the checker spoke    {t, code, severity, text, target, rule}
    frame   the camera, overlaid {t, jpeg}        -- base64 JPEG, camera sources only
    end     the session is over  {summary, report, outcome}
    error   it could not run     {message}

`checklist` rows are {id, status, instruction}, status one of pending, active,
done, error. `rules` (rulebooks) are {id, text, code}: what the checker
enforces, in the rulebook's words. `progress` is "done/total" steps, or
"N moves" for a rulebook. `tower` (Jenga) is {rows, top_layer, top_complete_layer,
legal_slots}: the occupancy the checker is judging against and where the next
block may legally go -- the live legality panel of FR-UI-2.
"""

from __future__ import annotations

from typing import Any

from thirdeye.events import Alert
from thirdeye.session import Session
from thirdeye.state.jenga.lattice import (
    TowerLattice,
    legal_top_slots,
    top_complete_layer,
    top_layer,
)

VERSION = 1


def _msg(type_: str, **body: Any) -> dict[str, Any]:
    return {"v": VERSION, "type": type_, **body}


def checklist(session: Session) -> list[dict[str, str]]:
    return [{"id": i, "status": s, "instruction": text} for i, s, text in session.checklist()]


def rules(session: Session) -> list[dict[str, str]]:
    return [{"id": p.id, "text": p.text, "code": p.violation}
            for a in session.checker.actions if a.repeatable for p in a.alerting()]


def hello(session: Session, source: str) -> dict[str, Any]:
    return _msg("hello", procedure=session.spec.procedure, kind=session.spec.kind,
                source=source, checklist=checklist(session), rules=rules(session))


def state(session: Session, t: float) -> dict[str, Any]:
    rows = checklist(session)
    nxt = session.checker.next_expected()
    if all(a.repeatable for a in session.checker.actions):
        moves = sum(e.kind == "event" and e.detail.startswith("MOVE") for e in session.log)
        progress = f"{moves} moves"
    else:
        progress = f"{sum(r['status'] == 'done' for r in rows)}/{len(rows)}"
    body: dict[str, Any] = {
        "t": round(t, 2),
        "checklist": rows,
        "next": None if nxt is None or nxt.repeatable else nxt.id,
        "progress": progress,
    }
    if isinstance(session.state, TowerLattice):
        snap = session.state.current()
        body["tower"] = {
            "rows": [list(r) for r in snap],
            "top_layer": top_layer(snap),
            "top_complete_layer": top_complete_layer(snap),
            "legal_slots": sorted([s.layer, s.slot] for s in legal_top_slots(snap)),
        }
    return _msg("state", **body)


def alert(a: Alert) -> dict[str, Any]:
    return _msg("alert", t=round(a.t, 2), code=a.code, severity=a.severity, text=a.text,
                target=a.target, rule=a.rule)


def frame(t: float, jpeg_b64: str) -> dict[str, Any]:
    return _msg("frame", t=round(t, 2), jpeg=jpeg_b64)


def end(summary: str, report: str, outcome: str | None) -> dict[str, Any]:
    return _msg("end", summary=summary, report=report, outcome=outcome)


def error(message: str) -> dict[str, Any]:
    return _msg("error", message=message)
