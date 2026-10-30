"""Ask the referee: a question about the session, answered from what the camera saw.

    python -m thirdeye.explain.ask "Which blocks can I still take?"   # against a scripted game

The perception and checker side has already turned the video into text: the
rules in force, the tower's occupancy, where the next block may go, and the
alerts so far. This module hands that text to the LLM with the player's
question. The model never sees a frame, and it does not judge anything live --
the checker still decides what is legal; the model only explains it.

It is held to the session the same way the report is: every answer cites the
rules and alerts it rests on, and a citation that names nothing in the session
is dropped and shown as such, so the UI can show which parts are grounded.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

#: How far a cited alert time may be from the alert's own (the context rounds to 0.1 s).
TIME_SLACK = 0.1


class AnswerBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    #: Rule codes (`illegal_source_layer`) and alerts as `code@t` (`two_hands@12.4`).
    cites: list[str]


@dataclass
class Answer:
    question: str
    answer: str
    cites: list[str]
    #: Citations that name no rule or alert in the session.
    dropped: list[str] = field(default_factory=list)
    context: str = ""
    author: str = ""
    seconds: float = 0.0

    def json(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "cites": self.cites,
                "dropped": self.dropped, "context": self.context, "author": self.author,
                "seconds": round(self.seconds, 2)}


def _last(history: list[dict[str, Any]], type_: str) -> dict[str, Any] | None:
    return next((m for m in reversed(history) if m.get("type") == type_), None)


def context(history: list[dict[str, Any]]) -> str:
    """The session so far as text, from its protocol messages: the model's only input."""
    hello = _last(history, "hello")
    if hello is None:
        return ""
    state = _last(history, "state") or {}
    alerts = [m for m in history if m.get("type") == "alert"]
    end = _last(history, "end")
    rulebook = hello.get("kind") == "constraints"
    lines = [
        (f"Procedure: {hello['procedure']} ({'rulebook' if rulebook else 'steps'}), "
         f"source {hello.get('source') or '-'}"),
        f"Time: t={state.get('t', 0):.1f}s, progress {state.get('progress') or '-'}"
        + (", the session is over" if end else ", in progress"),
    ]
    if state.get("perception") == "lost":
        lines.append("Perception: markers out of view, alerts paused")
    if hello.get("rules"):
        lines += ["", "Rules in force (code: the rule's own words):"]
        lines += [f"- {r['code']}: {r['text']}" for r in hello["rules"]]
    tower = state.get("tower")
    if tower:
        lines += ["", ("Tower as the cameras read it (layer 0 at the bottom; slots 0 1 2; "
                       "# = block there, . = empty):")]
        for layer in range(len(tower["rows"]) - 1, -1, -1):
            row = "".join("#" if v else "." for v in tower["rows"][layer])
            lines.append(f"- layer {layer}: {row}")
        lines.append(f"Top layer: {tower['top_layer']}; top complete layer: "
                     f"{tower['top_complete_layer']} (blocks may be taken only below it)")
        legal = ", ".join(f"layer {L} slot {s}" for L, s in tower["legal_slots"]) or "none"
        lines.append(f"Where the next block may be placed: {legal}")
    rows = state.get("checklist") or hello.get("checklist") or []
    if rows and not rulebook:
        lines += ["", "Steps (status: instruction):"]
        lines += [f"- {r['id']} {r['status']}: {r['instruction']}" for r in rows]
    lines += ["", "Alerts so far (time, code, severity, the checker's words):"]
    lines += [f"- t={a['t']:.1f} {a['code']} [{a['severity']}]: {a['text']}"
              for a in alerts] or ["- none"]
    return "\n".join(lines)


_SYSTEM = """\
You are the referee's voice for a game or task watched by cameras (Jenga, a
jigsaw, a building model). A computer-vision system has already read the scene
and a rule checker has judged it; you are given their output as text, and a
question from the player or an onlooker. You never saw the video.

Rules:
- Answer only from the session text, plus general knowledge of how the game is
  played. Never invent blocks, moves or alerts the text does not show.
- Jenga stability: a layer stands if it keeps its middle block, or both side
  blocks. So from a full layer (###) any block can go; from .## or ##. only the
  side block may go; from #.# or a lone block, nothing may go. When asked what
  is safe, apply this on top of the rules, and say which is which.
- If the text cannot answer the question, say so in one sentence and say what
  would be needed.
- Two to four short sentences, plain words, no lists or Markdown.
- Name layers and slots the way the text does ("layer 3, slot 1").
- `cites`: every rule code and alert your answer rests on. A rule is cited by
  its code (e.g. illegal_source_layer); an alert by code@time as in the text
  (e.g. two_hands@12.4). Empty if the answer rests on the tower state alone.
"""


def _grounded(cite: str, rules: set[str], alerts: list[tuple[str, float]]) -> bool:
    code, at, t = cite.strip().partition("@")
    if not at:
        return code in rules or any(c == code for c, _ in alerts)
    try:
        when = float(t.rstrip("s"))
    except ValueError:
        return False
    return any(c == code and abs(a - when) <= TIME_SLACK for c, a in alerts)


def ask(history: list[dict[str, Any]], question: str, llm) -> Answer:
    """The model's answer to `question` about the session in `history`, its citations checked."""
    from thirdeye.compiler.compile import _strict

    text = context(history)
    if not text:
        raise ValueError("no session yet: start a replay or a game first")
    t0 = time.monotonic()
    raw = llm.complete(_SYSTEM, f"{text}\n\nQuestion: {question}",
                       _strict(AnswerBody.model_json_schema()))
    author = getattr(llm, "name", getattr(llm, "model", ""))
    try:
        body = AnswerBody.model_validate_json(raw)
    except ValidationError:
        return Answer(question, "The model's answer could not be read; try again.", [],
                      context=text, author=author, seconds=time.monotonic() - t0)
    hello = _last(history, "hello") or {}
    rules = {r["code"] for r in hello.get("rules", [])}
    alerts = [(m["code"], m["t"]) for m in history if m.get("type") == "alert"]
    kept = [c for c in body.cites if _grounded(c, rules, alerts)]
    dropped = [c for c in body.cites if c not in kept]
    return Answer(question, body.answer, kept, dropped, context=text, author=author,
                  seconds=time.monotonic() - t0)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - makes an API call
    from thirdeye.compiler.llm import make_llm
    from thirdeye.server.app import SCENARIOS
    from thirdeye.server.runner import Runner

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("question")
    ap.add_argument("--scenario", default="jenga_fouls")
    ap.add_argument("--model", default=None)
    ap.add_argument("--context", action="store_true", help="print what the model would see")
    args = ap.parse_args(argv)

    history: list[dict[str, Any]] = []
    runner = Runner(history.append)
    runner.start_replay(SCENARIOS / f"{args.scenario}.yaml", speed=0)
    while runner.running or not any(m["type"] in ("end", "error") for m in history):
        time.sleep(0.05)
    if args.context:
        print(context(history))
        return 0
    a = ask(history, args.question, make_llm(args.model, max_tokens=2000))
    print(a.answer)
    print(f"cites: {', '.join(a.cites) or '-'}" + (f"; dropped: {', '.join(a.dropped)}"
                                                    if a.dropped else ""))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
