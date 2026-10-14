"""Is the compiler right? (EV-CMP-1..3, plan 5.3)

    python -m thirdeye.eval.compiler --compile --model claude-sonnet-5   # spends ~20 calls
    python -m thirdeye.eval.compiler --compiled eval/compiled/claude-sonnet-5

Compiles every manual in `manuals/eval_corpus/` (or reads specs compiled
earlier) and scores each against its hand-written ground truth. Compiled specs
are saved, so a score is regenerated without spending another call (NFR-REP-1)
and the same scorer can grade a different model's specs for comparison.

*Steps* (placement specs) are matched by what they place and where -- part and
footprint on a layer -- never by id, because a model may number its steps
differently and still be right. A ground-truth step with no exact match is
classified by the plan's error taxonomy: *wrong part* (right place, wrong
brick), *wrong pose* (right brick, wrong place), or *missed*. A compiled step
matching nothing is *hallucinated*.

*Order* is compared as the set of "A must come before B" pairs each spec
implies, from its dependencies and phases together, transitively. That
compares what the checker will enforce, not how it was written down: "corners,
then edges" as phases and the same order as per-step dependencies are the same
answer. Hard (physical-support) dependencies are also scored on their own,
because the checker treats them differently (FR-CHK-7).

*Constraints* are compared by behaviour, not text: each precondition is
evaluated over a bank of sampled towers and moves, and two preconditions are
the same rule when they agree on every sample. So `src.layer <
top_complete_layer` and `src.layer <= top_complete_layer - 1` match, and
`dst in legal_top_slots()` does not match "finish the top level first". The
violation code a matched rule carries is reported beside the score, since the
evaluation groups alerts by it, but a differently spelt code is not a wrong rule.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from thirdeye.checker.evaluate import Environment, EvalError, evaluate
from thirdeye.compiler.schema import ConstraintSpec, DagSpec, Step, load_spec
from thirdeye.events import Slot
from thirdeye.state.jenga.lattice import TowerLattice

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "manuals" / "eval_corpus"

# --- placement specs ------------------------------------------------------------------


def _where(step: Step) -> tuple[int, frozenset[tuple[int, int]]]:
    return step.pose.layer, frozenset(step.footprint)


def _pairs(spec: DagSpec, rename: dict[str, str]) -> set[tuple[str, str]]:
    """Every (before, after) the spec enforces, in ground-truth ids, transitively."""
    edges: dict[str, set[str]] = {}

    def add(before: str, after: str) -> None:
        if before in rename and after in rename:
            edges.setdefault(rename[after], set()).add(rename[before])

    for step in spec.steps:
        for dep in {*step.soft_depends_on, *step.hard_depends_on}:
            add(dep, step.id)
    members: dict[str, list[str]] = {}
    for step in spec.steps:
        if step.phase:
            members.setdefault(step.phase, []).append(step.id)
    for phase in spec.phases:
        for earlier in phase.after:
            for b in members.get(earlier, []):
                for a in members.get(phase.id, []):
                    add(b, a)
    closed: set[tuple[str, str]] = set()
    for node, befores in edges.items():
        stack, seen = list(befores), set()
        while stack:
            prev = stack.pop()
            if prev in seen:
                continue
            seen.add(prev)
            closed.add((prev, node))
            stack += edges.get(prev, ())
    return closed


def _hard(spec: DagSpec, rename: dict[str, str]) -> set[tuple[str, str]]:
    return {
        (rename[d], rename[s.id]) for s in spec.steps for d in s.hard_depends_on
        if d in rename and s.id in rename
    }


def _pr(got: set, want: set) -> dict[str, float | int | None]:
    tp = len(got & want)
    return {
        "tp": tp, "fp": len(got - want), "fn": len(want - got),
        "precision": tp / len(got) if got else (1.0 if not want else 0.0),
        "recall": tp / len(want) if want else 1.0,
    }


def score_dag(got: DagSpec, want: DagSpec) -> dict[str, Any]:
    left = list(got.steps)
    match: dict[str, str] = {}      # got id -> want id, for order scoring
    errors: dict[str, list[str]] = {"wrong_part": [], "wrong_pose": [], "missed": []}
    exact = 0
    todo = []
    for w in want.steps:
        hit = next((g for g in left if str(g.part) == str(w.part) and _where(g) == _where(w)), None)
        if hit is None:
            todo.append(w)
            continue
        left.remove(hit)
        match[hit.id] = w.id
        exact += 1
    for w in todo:
        hit = next((g for g in left if _where(g) == _where(w)), None)
        kind = "wrong_part"
        if hit is None:
            hit = next((g for g in left if str(g.part) == str(w.part)), None)
            kind = "wrong_pose"
        if hit is None:
            errors["missed"].append(w.id)
            continue
        left.remove(hit)
        match[hit.id] = w.id
        errors[kind].append(w.id)
    optional_right = sum(
        next(s for s in got.steps if s.id == g).optional == next(
            s for s in want.steps if s.id == w).optional
        for g, w in match.items()
    )
    identity = {s.id: s.id for s in want.steps}
    return {
        "steps": len(want.steps),
        "step_accuracy": exact / len(want.steps),
        "errors": {**errors, "hallucinated": [g.id for g in left]},
        "order": _pr(_pairs(got, match), _pairs(want, identity)),
        "hard_dependencies": _pr(_hard(got, match), _hard(want, identity)),
        "optional_accuracy": optional_right / len(match) if match else None,
    }


# --- constraint specs -----------------------------------------------------------------


@dataclass
class Sample:
    env: Environment


def sample_moves(n: int = 1500, seed: int = 0, layers: int = 18, slots: int = 3) -> list[Sample]:
    """Towers and moves a real game could produce.

    Heights from a fresh tower to a tall one; every layer below the top holds
    at least one block, the top any non-empty subset; the block comes from any
    filled slot and goes to any empty slot at most one layer above the top,
    since nothing can be put down in mid-air. One or two hands; a collapse on
    some. Seeded, so a score is reproducible.
    """
    rng = random.Random(seed)
    out = []
    subsets = [s for s in range(1, 2 ** slots)]
    for _ in range(n):
        height = rng.randint(4, layers + 6)
        rows = []
        for layer in range(height):
            full = rng.random() < (0.55 if layer < height - 1 else 0.4)
            mask = 2 ** slots - 1 if full else rng.choice(subsets)
            rows.append([(mask >> s) & 1 for s in range(slots)])
        lattice = TowerLattice(initial=rows, slots_per_layer=slots)
        lattice.collapse_detected = rng.random() < 0.1
        filled = [Slot(i, s) for i, r in enumerate(rows) for s in range(slots) if r[s]]
        empty = [Slot(i, s) for i, r in enumerate(rows) for s in range(slots) if not r[s]]
        empty += [Slot(height, s) for s in range(slots)]
        names, funcs = lattice.predicates()
        env = Environment(dict(names), dict(funcs)).merge({
            "src": rng.choice(filled),
            "dst": rng.choice(empty),
            "hands_in_contact": rng.choice([1, 1, 1, 2]),
        })
        out.append(Sample(env))
    return out


def _truth(expr: str, samples: list[Sample]) -> tuple[bool | None, ...]:
    out = []
    for s in samples:
        try:
            out.append(evaluate(expr, s.env))
        except EvalError:
            out.append(None)
    return tuple(out)


def score_constraints(got: ConstraintSpec, want: ConstraintSpec,
                      samples: list[Sample] | None = None) -> dict[str, Any]:
    samples = samples or sample_moves()
    g_pre = [p for a in got.actions for p in a.preconditions if p.on_violation == "alert"]
    w_pre = [p for a in want.actions for p in a.preconditions if p.on_violation == "alert"]
    g_truth = {p.id: _truth(p.expr, samples) for p in g_pre}
    w_truth = {p.id: _truth(p.expr, samples) for p in w_pre}

    matched: list[tuple[str, str]] = []
    free = list(g_pre)
    for w in w_pre:
        hit = next((g for g in free if g_truth[g.id] == w_truth[w.id]), None)
        if hit is not None:
            free.remove(hit)
            matched.append((hit.id, w.id))
    missed = [w.id for w in w_pre if w.id not in {m for _, m in matched}]

    def legal(truths: dict[str, tuple], i: int) -> bool:
        return all(t[i] is True for t in truths.values())

    agree = sum(legal(g_truth, i) == legal(w_truth, i) for i in range(len(samples)))
    code = {p.id: p.violation for p in g_pre} | {("w", p.id): p.violation for p in w_pre}
    g_term = {t.outcome: _truth(t.expr, samples) for t in got.terminal}
    w_term = {t.outcome: _truth(t.expr, samples) for t in want.terminal}
    return {
        "rules": len(w_pre),
        "constraint_accuracy": len(matched) / len(w_pre) if w_pre else 1.0,
        "constraint_precision": len(matched) / len(g_pre) if g_pre else 1.0,
        "errors": {
            "missed_rule": [f"{w} ({code[('w', w)]})" for w in missed],
            "hallucinated_rule": [f"{g.id}: {g.expr}" for g in free],
        },
        "same_code": sum(code[g] == code[("w", w)] for g, w in matched),
        "move_agreement": agree / len(samples),
        "terminal_match": sorted(o for o, t in w_term.items() if g_term.get(o) == t),
        "terminal_expected": sorted(w_term),
    }


def score(got: DagSpec | ConstraintSpec, want: DagSpec | ConstraintSpec,
          samples: list[Sample] | None = None) -> dict[str, Any]:
    if isinstance(want, DagSpec):
        if not isinstance(got, DagSpec) or got.state_model != want.state_model:
            return {"wrong_form": True}
        return score_dag(got, want)
    if not isinstance(got, ConstraintSpec):
        return {"wrong_form": True}
    return score_constraints(got, want, samples)


# --- the corpus -----------------------------------------------------------------------


@dataclass
class Entry:
    name: str
    route: str
    board: tuple[int, int] | None = None

    @property
    def manual(self) -> Path:
        return CORPUS / f"{self.name}.txt"

    @property
    def truth(self) -> Path:
        return CORPUS / f"{self.name}.spec.json"


def entries(corpus: Path = CORPUS) -> list[Entry]:
    data = yaml.safe_load((corpus / "corpus.yaml").read_text())
    return [Entry(e["name"], e["route"], tuple(e["board"]) if "board" in e else None)
            for e in data["entries"]]


def compile_entry(entry: Entry, llm) -> DagSpec | ConstraintSpec:
    from thirdeye.compiler.compile import compile_source
    from thirdeye.compiler.jigsaw import compile_instructions

    source = entry.manual.read_text()
    if entry.route == "jigsaw":
        assert entry.board is not None
        spec, _ = compile_instructions(source, *entry.board, llm, source_name=entry.name,
                                       procedure=entry.name)
    else:
        spec, _ = compile_source(source, entry.route, llm, source_name=entry.name)  # type: ignore[arg-type]
    return spec


@dataclass
class CorpusResult:
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        def mean(key: str, sub: str | None = None) -> float | None:
            vals = []
            for s in self.scores.values():
                v = s.get(key)
                if isinstance(v, dict) and sub:
                    v = v.get(sub)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            return sum(vals) / len(vals) if vals else None

        taxonomy: dict[str, int] = {}
        for s in self.scores.values():
            for kind, items in s.get("errors", {}).items():
                taxonomy[kind] = taxonomy.get(kind, 0) + len(items)
        return {
            "entries": len(self.scores) + len(self.failed),
            "compile_failures": len(self.failed),
            "step_accuracy": mean("step_accuracy"),
            "order_precision": mean("order", "precision"),
            "order_recall": mean("order", "recall"),
            "hard_dependency_recall": mean("hard_dependencies", "recall"),
            "constraint_accuracy": mean("constraint_accuracy"),
            "constraint_precision": mean("constraint_precision"),
            "move_agreement": mean("move_agreement"),
            "error_taxonomy": taxonomy,
        }

    def table(self) -> str:
        head = f"{'entry':<22}{'steps':>7}{'order P/R':>13}{'hard R':>8}{'rules':>7}{'moves':>7}"
        lines = [head]
        for name, s in self.scores.items():
            if s.get("wrong_form"):
                lines.append(f"{name:<22}  compiled to the wrong form")
                continue
            if "step_accuracy" in s:
                o, h = s["order"], s["hard_dependencies"]
                lines.append(f"{name:<22}{s['step_accuracy']:>7.2f}"
                             f"{o['precision']:>7.2f}/{o['recall']:<5.2f}{h['recall']:>8.2f}")
            else:
                lines.append(f"{name:<22}{'':>7}{'':>13}{'':>8}"
                             f"{s['constraint_accuracy']:>7.2f}{s['move_agreement']:>7.2f}")
        for name, why in self.failed.items():
            lines.append(f"{name:<22}  compile failed: {why}")
        sm = self.summary()

        def f(v: float | None) -> str:
            return "n/a" if v is None else f"{v:.2f}"

        lines += [
            "",
            (
                f"step accuracy {f(sm['step_accuracy'])} (target 0.9)   "
                f"order P/R {f(sm['order_precision'])}/{f(sm['order_recall'])} (target 0.8)"
            ),
            (
                f"constraint accuracy {f(sm['constraint_accuracy'])}, precision "
                f"{f(sm['constraint_precision'])}; move agreement {f(sm['move_agreement'])}"
            ),
            f"errors: {sm['error_taxonomy']}; compile failures: {sm['compile_failures']}",
        ]
        return "\n".join(lines)


def run(compiled: Path, corpus: Path = CORPUS, llm=None) -> CorpusResult:
    """Score every corpus entry, compiling first with `llm` if one is given."""
    from thirdeye.compiler.compile import CompileError

    result = CorpusResult()
    samples = sample_moves()
    compiled.mkdir(parents=True, exist_ok=True)
    for entry in entries(corpus):
        out = compiled / f"{entry.name}.json"
        if llm is not None:
            try:
                spec = compile_entry(entry, llm)
            except CompileError as exc:
                (compiled / f"{entry.name}.failed.txt").write_text(
                    f"{exc}\n" + "\n".join(exc.problems) + f"\n\n{exc.raw}\n")
                result.failed[entry.name] = str(exc)
                continue
            out.write_text(spec.model_dump_json(indent=1, exclude_none=True) + "\n")
        if not out.exists():
            result.failed[entry.name] = "no compiled spec"
            continue
        result.scores[entry.name] = score(load_spec(out), load_spec(entry.truth), samples)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--compile", action="store_true", help="compile the corpus first (API calls)")
    ap.add_argument("--model", default=None, help="model to compile with")
    ap.add_argument("--compiled", type=Path, help="folder of compiled specs "
                    "(default eval/compiled/<model>)")
    ap.add_argument("--json", type=Path, help="write the full scores here")
    args = ap.parse_args(argv)

    from thirdeye.compiler.compile import DEFAULT_MODEL, ClaudeLLM

    model = args.model or DEFAULT_MODEL
    compiled = args.compiled or ROOT / "eval" / "compiled" / model
    llm = ClaudeLLM(model=model) if args.compile else None
    result = run(compiled, llm=llm)
    print(result.table())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {"summary": result.summary(), "entries": result.scores, "failed": result.failed},
            indent=1) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
