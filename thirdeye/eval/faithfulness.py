"""Does the LLM report say only what the log says? (CS3235 doc, section 7: report faithfulness)

    python -m thirdeye.eval.faithfulness                       # every scripted scenario
    python -m thirdeye.eval.faithfulness --logs data/logs      # real session logs
    python -m thirdeye.eval.faithfulness --model groq:openai/gpt-oss-120b --judge gemini:gemini-2.5-flash

For each session log: write the LLM report (`explain.report.with_llm`), then
ask a judge model to split the report into its factual claims and mark each
one supported or not by the log digest -- the same text the writer saw.
Target: at most 5% of claims unsupported.

Two numbers come from the report's own guard rails and need no judge: how many
findings the writer made that cited no logged alert (dropped before anyone
reads them) and how many red alerts it left out (restored in the checker's
words). The judge covers what those cannot: the headline, "went well", the
advice, and the wording of every finding.

A judge is a model too, so every report and verdict is written to `--out` for
the manual check of ten reports the doc asks for.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from thirdeye.explain.report import digest, with_llm
from thirdeye.sessionlog import SessionRecord

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "scripts" / "scenarios"


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str
    supported: bool
    evidence: str


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[Claim]


_JUDGE = """\
You check a written report against the session log it was written from. List
every factual claim the report makes about the session -- what happened, when,
which step or piece, how many, how it ended, what went well. Leave out advice
that states no fact, and the report's footer about how it was written.

For each claim, `supported` is true only if the log states it or it follows
directly from the log; `evidence` quotes the log line, or says what is missing.
A claim that goes beyond the log (a cause, an intention, a detail the log does
not record) is not supported, even if plausible.
"""


@dataclass
class Result:
    name: str
    claims: list[Claim] = field(default_factory=list)
    dropped: int = 0
    restored: int = 0
    error: str = ""

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if not c.supported]


def judge(record: SessionRecord, report_md: str, llm) -> list[Claim]:
    from thirdeye.compiler.compile import _strict

    user = f"SESSION LOG\n\n{digest(record)}\n\n---\nREPORT\n\n{report_md}"
    raw = llm.complete(_JUDGE, user, _strict(Verdict.model_json_schema()))
    return Verdict.model_validate_json(raw).claims


def evaluate(records: dict[str, SessionRecord], writer, judge_llm, out: Path | None) -> list[Result]:
    from thirdeye.compiler.llm import LLMError

    results = []
    for name, record in records.items():
        res = Result(name)
        try:
            report = with_llm(record, writer)
            md = report.markdown()
            res.dropped, res.restored = len(report.rejected), len(report.restored)
            res.claims = judge(record, md, judge_llm)
        except (LLMError, ValueError) as exc:
            res.error = str(exc)[:300]
            md = ""
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{name}.md").write_text(md)
            (out / f"{name}.verdict.json").write_text(json.dumps(
                {"dropped": res.dropped, "restored": res.restored, "error": res.error,
                 "claims": [c.model_dump() for c in res.claims]}, indent=1))
        results.append(res)
    return results


def summary(results: list[Result]) -> dict[str, Any]:
    claims = sum(len(r.claims) for r in results)
    bad = sum(len(r.unsupported) for r in results)
    return {
        "reports": len(results),
        "failed": sum(bool(r.error) for r in results),
        "claims": claims,
        "unsupported": bad,
        "unsupported_rate": round(bad / claims, 4) if claims else None,
        "findings_dropped": sum(r.dropped for r in results),
        "alerts_restored": sum(r.restored for r in results),
    }


def table(results: list[Result]) -> str:
    rows = [f"{'session':<26} {'claims':>6} {'unsup':>6} {'dropped':>8} {'restored':>9}"]
    for r in results:
        if r.error:
            rows.append(f"{r.name:<26} failed: {r.error[:60]}")
            continue
        rows.append(f"{r.name:<26} {len(r.claims):>6} {len(r.unsupported):>6} {r.dropped:>8} "
                    f"{r.restored:>9}")
        rows += [f"    unsupported: {c.claim} ({c.evidence})" for c in r.unsupported]
    s = summary(results)
    rate = "n/a" if s["unsupported_rate"] is None else f"{100 * s['unsupported_rate']:.1f}%"
    rows += ["", (f"{s['unsupported']} of {s['claims']} claims unsupported ({rate}; target <= 5%)"
                  f"; {s['findings_dropped']} finding(s) dropped, {s['alerts_restored']} alert(s) "
                  f"restored; {s['failed']} report(s) failed")]
    return "\n".join(rows)


def load(logs: Path | None) -> dict[str, SessionRecord]:
    if logs is not None:
        return {p.stem: SessionRecord.read(p) for p in sorted(logs.glob("*.json"))}
    from thirdeye.replay import Replay

    return {p.stem: Replay.load(p).run() for p in sorted(SCENARIOS.glob("*.yaml"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs", type=Path, help="folder of session logs (default: replay "
                    "every scripted scenario)")
    ap.add_argument("--model", default=None, help="the report writer (provider:model)")
    ap.add_argument("--judge", default=None, help="the judge (default: same as --model)")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "faithfulness",
                    help="write each report and verdict here, for the manual check")
    ap.add_argument("--json", type=Path, help="write the summary here")
    args = ap.parse_args(argv)

    from thirdeye.compiler.llm import make_llm

    writer = make_llm(args.model, max_tokens=4000)
    judge_llm = make_llm(args.judge or args.model, max_tokens=8000)
    results = evaluate(load(args.logs), writer, judge_llm, args.out)
    print(table(results))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"writer": writer.name, "judge": judge_llm.name,
                                         **summary(results)}, indent=1) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
