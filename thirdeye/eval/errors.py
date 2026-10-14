"""Are errors caught, and is it quiet when nothing is wrong? (EV-ERR-1..4, EV-JEN-1)

    python -m thirdeye.eval.errors --labels data/jigsaw/labels --logs data/jigsaw/logs
    python -m thirdeye.eval.errors --scenarios scripts/scenarios

Compares the alerts in each session log against that run's label file and
reports, per alert code: true and false positives, misses, precision and
recall; false alerts per correct run; and the median delay from the mistake to
its alert. `--scenarios` replays every scripted scenario and scores it against
its own `expect:` block, so the scorer is exercised end to end before any real
run is labelled.

*Matching.* An alert matches a labelled error when the code agrees, the target
agrees (if both name one), and -- when the label has a time -- the alert falls
between `early` seconds before and `late` seconds after it. Each alert and each
label matches at most once, earliest first. An alert that matches nothing is a
false positive; a label nothing matched is a miss.

*What counts.* Red alerts only, by default: a yellow warning is a suspicion the
UI shows differently (FR-CHK-9). `--warnings` counts them too, which is the
with/without-soft-warnings ablation (EV-AB-6). Our own failures -- perception
misses, unevaluable rules -- are reported beside the score and never inside it
(FR-CHK-8).

*One operating point.* This reports precision and recall at the checker's
current confidence threshold. "Precision at fixed recall" (EV-ERR-1) needs that
threshold swept, which needs recorded runs; the sweep is a loop over this.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from thirdeye.eval.labels import ErrorLabel, RunLabels, parse_error, read_all
from thirdeye.events import Alert
from thirdeye.sessionlog import SessionRecord

#: A labeller clicks a little late or early; an alert raised a moment before the
#: labelled time is the same mistake, not a false alarm.
EARLY = 1.0
#: Beyond this an alert is too late to be about that mistake. Generous, because
#: a missed step is only known at the end and a labeller marks it there too.
LATE = 10.0


@dataclass
class RunScore:
    run: str
    correct: bool
    tp: list[tuple[ErrorLabel, Alert]] = field(default_factory=list)
    fp: list[Alert] = field(default_factory=list)
    fn: list[ErrorLabel] = field(default_factory=list)
    perception_misses: int = 0
    unevaluable: int = 0

    @property
    def delays(self) -> list[float]:
        return [a.t - e.t for e, a in self.tp if e.t is not None]


def _fits(label: ErrorLabel, alert: Alert, early: float, late: float) -> bool:
    if label.code != alert.code:
        return False
    if label.target and alert.target and label.target != alert.target:
        return False
    return label.t is None or label.t - early <= alert.t <= label.t + late


def score_run(
    labels: RunLabels,
    record: SessionRecord,
    *,
    warnings: bool = False,
    early: float = EARLY,
    late: float = LATE,
) -> RunScore:
    if labels.procedure != record.procedure:
        raise ValueError(
            f"run {labels.run}: labelled as {labels.procedure!r} but the log is "
            f"{record.procedure!r} -- the wrong log is paired with this label"
        )
    alerts = sorted(record.alerts if warnings else record.red(), key=lambda a: a.t)
    todo = sorted(labels.errors, key=lambda e: (e.t is None, e.t or 0.0))
    score = RunScore(
        run=labels.run,
        correct=labels.correct,
        perception_misses=len(record.perception_misses),
        unevaluable=len(record.unevaluable),
    )
    free = list(alerts)
    for label in todo:
        hit = next((a for a in free if _fits(label, a, early, late)), None)
        if hit is None:
            score.fn.append(label)
        else:
            free.remove(hit)
            score.tp.append((label, hit))
    score.fp = free
    return score


@dataclass
class Report:
    runs: list[RunScore]

    def codes(self) -> list[str]:
        seen = {e.code for r in self.runs for e, _ in r.tp}
        seen |= {a.code for r in self.runs for a in r.fp}
        seen |= {e.code for r in self.runs for e in r.fn}
        return sorted(seen)

    def counts(self, code: str | None = None) -> tuple[int, int, int]:
        def keep(c: str) -> bool:
            return code is None or c == code
        tp = sum(keep(e.code) for r in self.runs for e, _ in r.tp)
        fp = sum(keep(a.code) for r in self.runs for a in r.fp)
        fn = sum(keep(e.code) for r in self.runs for e in r.fn)
        return tp, fp, fn

    def false_alerts_per_correct_run(self) -> float | None:
        correct = [r for r in self.runs if r.correct]
        return None if not correct else sum(len(r.fp) for r in correct) / len(correct)

    def median_delay(self) -> float | None:
        delays = [d for r in self.runs for d in r.delays]
        return statistics.median(delays) if delays else None

    def to_json(self) -> dict:
        rows = {}
        for code in [*self.codes(), None]:
            tp, fp, fn = self.counts(code)
            rows[code or "all"] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn),
            }
        return {
            "runs": len(self.runs),
            "correct_runs": sum(r.correct for r in self.runs),
            "per_code": rows,
            "false_alerts_per_correct_run": self.false_alerts_per_correct_run(),
            "median_delay_s": self.median_delay(),
            "perception_misses": sum(r.perception_misses for r in self.runs),
            "unevaluable": sum(r.unevaluable for r in self.runs),
        }

    def table(self) -> str:
        lines = [f"{'code':<24}{'tp':>5}{'fp':>5}{'fn':>5}{'precision':>11}{'recall':>9}"]
        for code in [*self.codes(), None]:
            tp, fp, fn = self.counts(code)
            if code is None:
                lines.append("-" * 59)
            lines.append(
                f"{code or 'all':<24}{tp:>5}{fp:>5}{fn:>5}"
                f"{_fmt(_ratio(tp, tp + fp)):>11}{_fmt(_ratio(tp, tp + fn)):>9}"
            )
        data = self.to_json()
        lines += [
            "",
            f"runs: {data['runs']} ({data['correct_runs']} correct)",
            f"false alerts per correct run: {_fmt(data['false_alerts_per_correct_run'])}",
            f"median alert delay: {_fmt(data['median_delay_s'], 's')}",
            (
                f"ours, not the user's: {data['perception_misses']} perception misses, "
                f"{data['unevaluable']} unevaluable rule checks"
            ),
        ]
        return "\n".join(lines)

    def mistakes(self) -> str:
        """Every false positive and miss, run by run -- what to look at next."""
        lines = []
        for r in self.runs:
            for a in r.fp:
                lines.append(f"{r.run}: false alert  {a.code} {a.target} @ {a.t:.2f}".rstrip())
            for e in r.fn:
                lines.append(f"{r.run}: missed       {e}")
        return "\n".join(lines)


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def _fmt(value: float | None, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{unit}"


# --- inputs ----------------------------------------------------------------------


def run_id(record: SessionRecord, path: Path) -> str:
    """A log names its run in `meta.run`; failing that, by its file name."""
    return str(record.meta.get("run") or path.stem)


def pair(labels: dict[str, RunLabels], logs: Path) -> list[tuple[RunLabels, SessionRecord]]:
    """Join label files to session logs by run id. An unlabelled log, or a label
    with no log, is an error: silently dropping either would bias the score."""
    records: dict[str, SessionRecord] = {}
    for path in sorted(logs.glob("*.json")):
        record = SessionRecord.read(path)
        records[run_id(record, path)] = record
    missing = sorted(set(labels) - set(records))
    extra = sorted(set(records) - set(labels))
    if missing or extra:
        raise ValueError(
            "labels and logs do not pair up: "
            + "; ".join(
                part for part in (
                    f"no log for {', '.join(missing)}" if missing else "",
                    f"no label for {', '.join(extra)}" if extra else "",
                ) if part
            )
        )
    return [(labels[k], records[k]) for k in sorted(labels)]


def scenario_pairs(folder: Path) -> list[tuple[RunLabels, SessionRecord]]:
    """Replay every scenario and label it from its own `expect:` block."""
    from thirdeye.replay import Replay  # replay pulls in every state model

    out = []
    for path in sorted(folder.glob("*.yaml")):
        script = yaml.safe_load(path.read_text())
        record = Replay.load(path).run()
        labels = RunLabels(
            run=path.stem,
            procedure=record.procedure,
            split="scripted",
            errors=[parse_error(e) for e in script.get("expect") or []],
        )
        out.append((labels, record))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels", type=Path, help="folder of label files (one YAML per run)")
    src.add_argument("--scenarios", type=Path, help="replay and score scripted scenarios")
    ap.add_argument("--logs", type=Path, help="folder of session logs, with --labels")
    ap.add_argument("--split", choices=["tune", "test"], help="score only this split")
    ap.add_argument("--procedure", help="score only runs of this procedure")
    ap.add_argument("--warnings", action="store_true", help="count yellow warnings too (EV-AB-6)")
    ap.add_argument("--early", type=float, default=EARLY)
    ap.add_argument("--late", type=float, default=LATE)
    ap.add_argument("--json", type=Path, help="also write the numbers here")
    ap.add_argument("-v", "--verbose", action="store_true", help="list every miss and false alert")
    args = ap.parse_args(argv)

    if args.labels:
        if args.logs is None:
            ap.error("--labels needs --logs")
        pairs = pair(read_all(args.labels), args.logs)
    else:
        pairs = scenario_pairs(args.scenarios)
    pairs = [
        (lab, rec) for lab, rec in pairs
        if (args.split is None or lab.split == args.split)
        and (args.procedure is None or lab.procedure == args.procedure)
    ]
    if not pairs:
        print("no runs to score", file=sys.stderr)
        return 1

    report = Report([
        score_run(lab, rec, warnings=args.warnings, early=args.early, late=args.late)
        for lab, rec in pairs
    ])
    print(report.table())
    if args.verbose and (text := report.mistakes()):
        print("\n" + text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_json(), indent=1) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
