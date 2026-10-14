"""The LLM procedure compiler (FR-CMP-1, FR-CMP-4, FR-CMP-5, FR-CMP-7, IF-SPEC-5).

    python -m thirdeye.compiler.compile manuals/model_a.txt --kind dag -o specs/model_a.compiled.json

Text in, validated spec out, once per procedure and never on the live path
(FR-CMP-10). The model sees only text (FR-CMP-11): the manual or rulebook, the
schema, the fixed predicate vocabulary, and one worked example.

The output is constrained twice. The API's structured-output mode holds the
model to the spec's JSON schema, so a missing field or an invented key cannot
happen; then the same `validate()` a hand-written spec goes through checks what
a schema cannot -- cycles, collisions, floating parts, illegal expressions. If
either fails, the model gets **one** repair round with the exact problems
listed (FR-CMP-7), and if that fails too the compiler stops and says why. A
second automatic round would mostly teach us how persuasive the model is.

The LLM is behind a one-method protocol, so the compile loop -- and the repair
round, which is the part worth testing -- runs in the test suite against a
scripted stand-in and never spends a call.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from thirdeye.compiler.expr import vocabulary_prompt
from thirdeye.compiler.llm import DEFAULT_MODEL, LLMError, make_llm
from thirdeye.compiler.schema import ConstraintSpec, DagSpec, Provenance, parse_spec
from thirdeye.compiler.validate import Problem, errors, validate

COMPILER_VERSION = "0.2.0"
#: Claude's default when it is picked; the project default is `llm.DEFAULT_MODEL`.
CLAUDE_MODEL = "claude-sonnet-5"

Kind = Literal["dag", "constraints"]

SPECS = Path(__file__).resolve().parents[2] / "specs"
EXAMPLES: dict[str, Path] = {
    "dag": SPECS / "example_dag.json",
    "constraints": SPECS / "example_constraints.json",
}


class LLM(Protocol):
    """Anything that can turn a prompt into JSON text matching a schema."""

    model: str

    def complete(self, system: str, user: str, schema: dict[str, Any] | None) -> str: ...


class CompileError(Exception):
    """The spec did not validate after the repair round. Needs a human (FR-CMP-9)."""

    def __init__(self, message: str, problems: list[str], raw: str):
        super().__init__(message)
        self.problems = problems
        self.raw = raw


# --- the schema the model is held to -------------------------------------------------

#: JSON-schema keywords that describe rather than constrain, or that
#: structured outputs may not accept. Bounds are dropped here and enforced by
#: Pydantic on the way back in, so nothing is lost.
_DROP = {"title", "description", "default", "minimum", "maximum", "exclusiveMinimum",
         "exclusiveMaximum", "pattern", "minItems", "maxItems"}


def output_schema(kind: Kind) -> dict[str, Any]:
    """The spec's JSON schema, reduced to what the model should fill in.

    Every property is made required, so the model states each default rather
    than leaving us to guess whether it meant it; `provenance` is removed,
    because that is ours to write.
    """
    model = DagSpec if kind == "dag" else ConstraintSpec
    schema = _strict(model.model_json_schema())
    schema.get("$defs", {}).pop("Provenance", None)
    return schema


def _strict(node: Any) -> Any:
    if isinstance(node, list):
        return [_strict(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {k: _strict(v) for k, v in node.items() if k not in _DROP}
    if "const" in out:
        out["enum"] = [out.pop("const")]
    if "prefixItems" in out:
        # A pair like expected_duration_s: an array of numbers; Pydantic
        # checks the length when the spec is parsed.
        items = out.pop("prefixItems")
        out["items"] = items[0] if items else {}
    if out.get("type") == "object" and "properties" in out:
        out["properties"].pop("provenance", None)
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return out


# --- prompts --------------------------------------------------------------------------

_DAG_RULES = """\
You compile a LEGO building manual into a dependency-graph procedure spec.

Coordinates are studs on the baseplate. x runs left to right, y runs from the
builder's edge away from them, and (0, 0) is the bottom-left stud. A pose gives
the brick's lowest-x, lowest-y stud. A part of size "WxH" at rot 0 covers W studs
along x and H studs along y; at rot 90 it covers H along x and W along y. Layer 0
sits on the plate; layer n+1 sits on bricks in layer n.

For every step record two kinds of dependency, independently:
- hard_depends_on: the steps whose bricks physically hold this one up -- every
  step on the layer directly below whose studs this brick covers. Nothing else.
- soft_depends_on: the order the manual states. A numbered manual means each
  step softly depends on the one before it. Every hard dependency is also soft.

Use one step per manual instruction, ids s1, s2, ... in manual order, and copy
each instruction verbatim. Colours and sizes are lower-case, sizes like "2x4".
Mark a step optional only if the manual says it is optional. Use
expected_duration_s [3, 20] unless the manual suggests otherwise.
"""

_CONSTRAINT_RULES = """\
You compile a game rulebook into a constraint-form procedure spec: actions with
preconditions, no fixed order. Each rule the player can break becomes one
precondition, with a machine expression, the rule in plain words as `text`,
and a snake_case `violation` code naming the foul. Ending conditions go in
`terminal`.

Expressions may use only this vocabulary:

{vocabulary}
"""


def _system(kind: Kind, example: bool = True) -> str:
    rules = _DAG_RULES if kind == "dag" else _CONSTRAINT_RULES.format(
        vocabulary=vocabulary_prompt()
    )
    if not example:
        return rules
    shown = EXAMPLES[kind].read_text()
    return f"{rules}\nA correctly compiled example, for format only:\n\n{shown}"


def _repair_prompt(source: str, raw: str, problems: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        f"{source}\n\n---\nYour previous spec for this source was:\n\n{raw}\n\n"
        f"It failed validation:\n{listed}\n\n"
        "Return the whole corrected spec. Change only what these problems require."
    )


# --- the compile loop -----------------------------------------------------------------

@dataclass
class Attempt:
    raw: str
    problems: list[str] = field(default_factory=list)
    warnings: list[Problem] = field(default_factory=list)
    spec: DagSpec | ConstraintSpec | None = None


def check(raw: str, kind: Kind) -> Attempt:
    """Parse and validate one model output. Never raises on bad content."""
    attempt = Attempt(raw=raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        attempt.problems = [f"not valid JSON: {exc}"]
        return attempt
    if isinstance(data, dict):
        data.pop("provenance", None)
    try:
        spec = parse_spec(data)
    except ValidationError as exc:
        attempt.problems = [
            f"schema: {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        ]
        return attempt
    if spec.kind != kind:
        attempt.problems = [f"expected a {kind!r} spec, got {spec.kind!r}"]
        return attempt
    problems = validate(spec)
    attempt.problems = [str(p) for p in errors(problems)]
    attempt.warnings = [p for p in problems if p.severity == "warning"]
    attempt.spec = spec
    return attempt


def repair_loop(
    llm: LLM, system: str, source: str, schema: dict[str, Any] | None,
    check_fn: Callable[[str], Attempt], repair: bool = True,
) -> tuple[Attempt, bool]:
    """Ask once; if the answer has problems, ask once more with them listed.

    Returns the accepted attempt and whether it took the repair round. Raises
    `CompileError` if the second answer is still wrong. Shared by every route
    that puts an LLM in front of a spec, so they all get exactly one repair.
    """
    first = check_fn(llm.complete(system, source, schema))
    final, repaired = first, False
    if first.problems and repair:
        final = check_fn(llm.complete(system, _repair_prompt(source, first.raw, first.problems),
                                      schema))
        repaired = True
    if final.problems or final.spec is None:
        raise CompileError(
            f"spec failed validation {'after one repair round' if repaired else 'with no repair round'}"
            f" ({len(final.problems)} problem(s))",
            final.problems,
            final.raw,
        )
    return final, repaired


def stamp(spec, model: str, source_name: str, repaired: bool):
    """Write the provenance the model is never asked for (IF-SPEC-5)."""
    return spec.model_copy(
        update={
            "provenance": Provenance(
                compiler_version=COMPILER_VERSION,
                model=model,
                source=source_name,
                compiled_at=datetime.now(UTC).isoformat(timespec="seconds"),
                repaired=repaired,
            )
        }
    )


def compile_source(
    source: str, kind: Kind, llm: LLM, source_name: str = "", *,
    schema: bool = True, example: bool = True, repair: bool = True,
) -> tuple[DagSpec | ConstraintSpec, list[Problem]]:
    """Compile a manual or rulebook. Returns the spec and its validation warnings.

    Raises `CompileError` if the spec still fails after one repair round.
    `schema`, `example` and `repair` exist for the ablation baseline only
    (`eval.compiler --bare`): the schema is given as text in the prompt but
    not enforced in decoding, there is no worked example and no repair round.
    """
    system = _system(kind, example)
    if not schema:
        system = with_schema_text(system, output_schema(kind))
    final, repaired = repair_loop(
        llm, system, source, output_schema(kind) if schema else None,
        lambda raw: check(raw, kind), repair=repair,
    )
    return stamp(final.spec, model_name(llm), source_name, repaired), final.warnings


def with_schema_text(system: str, schema: dict[str, Any]) -> str:
    """For unconstrained (plain JSON) decoding: the schema, as text."""
    return (f"{system}\nAnswer with one JSON object, and nothing else, matching this JSON "
            f"schema:\n\n{json.dumps(schema)}")


def model_name(llm: LLM) -> str:
    """`provider:model` when the backend knows it, for provenance."""
    return getattr(llm, "name", llm.model)


# --- the real model -------------------------------------------------------------------

@dataclass
class ClaudeLLM:
    """Claude through the Anthropic SDK (the `llm` extra).

    The key comes from the environment or an `ant auth login` profile, never
    from code or config (NFR-PRIV-2).
    """

    model: str = CLAUDE_MODEL
    max_tokens: int = 16000
    client: Any = None

    def complete(self, system: str, user: str, schema: dict[str, Any] | None) -> str:
        if self.client is None:
            import anthropic  # the `llm` extra; imported here so nothing else needs it

            self.client = anthropic.Anthropic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            **({"output_config": {"format": {"type": "json_schema", "schema": schema}}}
               if schema is not None else {}),
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise CompileError(f"the model stopped early: {response.stop_reason}", [], "")
        return next(b.text for b in response.content if b.type == "text")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="manual or rulebook, plain text")
    ap.add_argument("--kind", choices=["dag", "constraints"], required=True)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--model", default=None,
                    help=f"provider:model (default $THIRDEYE_LLM or {DEFAULT_MODEL})")
    args = ap.parse_args(argv)

    try:
        spec, warnings = compile_source(
            args.source.read_text(), args.kind, make_llm(args.model),
            source_name=args.source.as_posix(),
        )
    except LLMError as exc:
        print(f"compile failed: {exc}", file=sys.stderr)
        return 1
    except CompileError as exc:
        print(f"compile failed: {exc}", file=sys.stderr)
        for p in exc.problems:
            print(f"  {p}", file=sys.stderr)
        if exc.raw:
            rejected = args.out.with_suffix(".rejected.json")
            rejected.write_text(exc.raw)
            print(f"last attempt kept at {rejected} for hand repair", file=sys.stderr)
        return 1
    args.out.write_text(spec.model_dump_json(indent=2, exclude_none=True) + "\n")
    for w in warnings:
        print(w)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
