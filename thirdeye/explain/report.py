"""The end-of-session report (FR-EXP-1, FR-EXP-2).

    python -m thirdeye.explain.report run.json               # plain report, no LLM
    python -m thirdeye.explain.report run.json --llm -o run.md

Reads a session log -- never the session, never the video (FR-EXP-2) -- and
writes a short report a person can read: what went wrong, when, which step,
and what it means.

Two routes, one shape. The plain route writes the report from the checker's own
alert text and needs nothing. The LLM route hands the model a text digest of the
log and asks for the same report in friendlier words. The model is held to the
log in two ways, because an explainer that invents a mistake is worse than none:

- every finding it writes must cite an alert that is in the log, by code and
  time; a finding that cites nothing is dropped and counted;
- every red alert in the log appears in the report; one the model left out is
  added back in the checker's own words.

So the LLM decides how the report reads, and the log decides what it says.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from thirdeye.events import Alert
from thirdeye.sessionlog import SessionRecord

#: How far a cited time may be from the alert's own (the digest rounds to 0.1 s).
TIME_SLACK = 0.1


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t: float
    code: str
    what_happened: str
    consequence: str


class ReportBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str
    findings: list[Finding]
    went_well: str
    advice: list[str]


@dataclass
class Report:
    record: SessionRecord
    body: ReportBody
    #: Model findings dropped for citing no alert in the log.
    rejected: list[Finding] = field(default_factory=list)
    #: Alerts the model left out, restored from the checker's text.
    restored: list[Alert] = field(default_factory=list)
    author: str = "plain"

    def markdown(self) -> str:
        r, b = self.record, self.body
        lines = [f"# Session report: {r.procedure}", "", b.headline, ""]
        if b.findings:
            lines += ["## What went wrong", ""]
            for f in sorted(b.findings, key=lambda f: f.t):
                lines.append(f"- **{_clock(f.t)} · {f.code.replace('_', ' ')}**: "
                             f"{f.what_happened} {f.consequence}".rstrip())
            lines.append("")
        if b.went_well:
            lines += ["## What went well", "", b.went_well, ""]
        if b.advice:
            lines += ["## Next time", ""] + [f"- {a}" for a in b.advice] + [""]
        lines += ["---", "", _footer(self)]
        return "\n".join(lines) + "\n"


def _clock(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m)}:{s:04.1f}"


def _footer(report: Report) -> str:
    r = report.record
    bits = [f"Written by: {report.author}, from the session log only."]
    if r.perception_misses or r.unevaluable:
        bits.append(
            f"Not counted against the player: {len(r.perception_misses)} of our own "
            f"detection misses, {len(r.unevaluable)} rule checks we could not run."
        )
    if report.rejected:
        bits.append(f"{len(report.rejected)} model statement(s) dropped for citing no alert.")
    if report.restored:
        bits.append(f"{len(report.restored)} alert(s) the model left out were added back.")
    return " ".join(bits)


# --- the digest: all the model ever sees ------------------------------------------------


def digest(record: SessionRecord) -> str:
    """The session log as text: the model's only input (FR-EXP-2)."""
    steps = record.checklist
    done = [c for c in steps if c.status in ("done", "error")]
    lines = [
        f"Procedure: {record.procedure} ({'rulebook' if record.kind == 'constraints' else 'steps'})",
        f"Outcome: {record.outcome or 'finished'}",
    ]
    if record.kind != "constraints":
        lines.append(f"Steps completed: {len(done)} of {len(steps)}")
    moves = sum(1 for e in record.log if e.kind == "event" and e.detail.startswith("MOVE"))
    if moves:
        lines.append(f"Moves made: {moves}")
    lines += ["", "Alerts (time in seconds, code, step, the rule's own words):"]
    lines += [
        f"- t={a.t:.1f} {a.code} {a.target or '-'} [{a.severity}]: {a.text}"
        for a in record.alerts
    ] or ["- none"]
    events = [e for e in record.log if e.kind == "event"]
    if events:
        lines += ["", "What the camera saw, in order:"]
        lines += [f"- t={e.t:.1f} {e.detail}" for e in events[:80]]
        if len(events) > 80:
            lines.append(f"- ... and {len(events) - 80} more")
    return "\n".join(lines)


_SYSTEM = """\
You write a short, friendly report for someone who just finished a task in front
of a camera (a jigsaw, a Jenga game, a building model). You are given the session
log as text. You never saw the video.

Rules:
- One finding per alert. Cite the alert's exact time `t` and `code`. Never write a
  finding that is not in the alert list, and never guess about anything the log
  does not say.
- `what_happened`: one plain sentence, naming the step or piece.
- `consequence`: one sentence on why it matters for the task.
- `headline`: one sentence on how the session went overall.
- `went_well`: one or two sentences, or empty if nothing did.
- `advice`: at most three short, specific tips drawn from the findings.
- Warnings are suspicions, not certainties: say "may have".
"""


def plain(record: SessionRecord) -> Report:
    """The report from the checker's own words. Needs no model."""
    red = record.red()
    findings = [
        Finding(t=a.t, code=a.code,
                what_happened=a.text if a.severity == "alert" else f"Possibly: {a.text}",
                consequence="")
        for a in record.alerts
    ]
    steps = record.checklist
    clean = sum(c.status == "done" for c in steps)
    fouls = [a for a in red if a.code != record.outcome]   # the ending is not a foul
    if record.kind == "constraints":
        headline = f"{len(fouls)} rule(s) broken."
    else:
        headline = f"{clean} of {len(steps)} steps done correctly, {len(fouls)} mistake(s) flagged."
    if record.outcome:
        headline += f" The session ended: {record.outcome.replace('_', ' ')}."
    went_well = "" if red else "No mistakes were flagged."
    return Report(record, ReportBody(headline=headline, findings=findings,
                                     went_well=went_well, advice=[]))


def _grounded(f: Finding, alerts: list[Alert]) -> Alert | None:
    return next((a for a in alerts
                 if a.code == f.code and abs(a.t - f.t) <= TIME_SLACK), None)


def with_llm(record: SessionRecord, llm) -> Report:
    """The report in the model's words, held to the log (see the module doc)."""
    schema = _schema()
    raw = llm.complete(_SYSTEM, digest(record), schema)
    try:
        body = ReportBody.model_validate_json(raw)
    except ValidationError:
        report = plain(record)
        report.author = f"{llm.model} (unreadable answer; plain report used)"
        return report
    kept, rejected, cited = [], [], []
    for f in body.findings:
        hit = _grounded(f, [a for a in record.alerts if a not in cited])
        if hit is None:
            rejected.append(f)
        else:
            cited.append(hit)
            kept.append(f.model_copy(update={"t": hit.t}))
    restored = [a for a in record.red() if a not in cited]
    kept += [Finding(t=a.t, code=a.code, what_happened=a.text, consequence="")
             for a in restored]
    body = body.model_copy(update={"findings": kept})
    return Report(record, body, rejected=rejected, restored=restored, author=llm.model)


def _schema() -> dict[str, Any]:
    from thirdeye.compiler.compile import _strict

    return _strict(ReportBody.model_json_schema())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path, help="a session log (IF-LOG-1)")
    ap.add_argument("--llm", action="store_true", help="write it with the LLM (one API call)")
    ap.add_argument("--model", default=None)
    ap.add_argument("-o", "--out", type=Path, help="write the Markdown here")
    ap.add_argument("--digest", action="store_true", help="print what the model would see")
    args = ap.parse_args(argv)

    record = SessionRecord.read(args.log)
    if args.digest:
        print(digest(record))
        return 0
    if args.llm:
        from thirdeye.compiler.compile import DEFAULT_MODEL, ClaudeLLM

        report = with_llm(record, ClaudeLLM(model=args.model or DEFAULT_MODEL, max_tokens=2000))
    else:
        report = plain(record)
    text = report.markdown()
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

