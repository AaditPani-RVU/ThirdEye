"""A jigsaw spec from the board's size alone.

    python -m thirdeye.compiler.jigsaw 4 3 -o specs/jigsaw_4x3.json

A rectangular jigsaw's geometry is fully determined by its size: piece `rRcC`
belongs at row R, column C, turned 0. So the steps need no manual, and building
them is not the LLM's job. The instructions carry the order, and the usual
order is written here as phases: corners, then edges, then the interior. This
is the "hand" route of IF-SPEC-5.

The LLM route reads a real instruction sheet into phases:

    python -m thirdeye.compiler.jigsaw 4 3 --instructions manuals/jigsaw_border.txt -o spec.json

The model never writes a coordinate. It answers with a *plan*: named phases,
their order, and which pieces each holds, in a small selector vocabulary
(`corners`, `row 0`, `columns 2-3`, `rest`, ...). The geometry comes from the
board size, as above, and the plan only orders it. A 6x8 puzzle is 48 steps;
asking a model to type 48 poses is asking for a typo, and a typo is a piece the
checker would send to the wrong cell.

Pieces are named for their home cell (see `PieceRef`), so the name perception
reports for a piece is decided by where it matches in the box picture.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from thirdeye.compiler.compile import (
    COMPILER_VERSION,
    LLM,
    Attempt,
    CompileError,
    _strict,
    model_name,
    repair_loop,
    stamp,
    with_schema_text,
)
from thirdeye.compiler.llm import DEFAULT_MODEL, LLMError, make_llm
from thirdeye.compiler.schema import (
    Board,
    DagSpec,
    Phase,
    PieceRef,
    Pose,
    Provenance,
    Step,
    Strict,
)
from thirdeye.compiler.validate import Problem, errors, validate

#: Phase id -> (name in alert text, phases it comes after).
BORDER_FIRST: dict[str, tuple[str, list[str]]] = {
    "corners": ("the corners", []),
    "edges": ("the edges", ["corners"]),
    "interior": ("the interior", ["corners", "edges"]),
}


def piece_name(row: int, col: int) -> str:
    return f"r{row}c{col}"


def _kind(row: int, col: int, rows: int, cols: int) -> str:
    on_row_edge = row in (0, rows - 1)
    on_col_edge = col in (0, cols - 1)
    if on_row_edge and on_col_edge:
        return "corners"
    if on_row_edge or on_col_edge:
        return "edges"
    return "interior"


def _where(row: int, col: int, rows: int, cols: int) -> str:
    vertical = {0: "top", rows - 1: "bottom"}.get(row, "")
    horizontal = {0: "left", cols - 1: "right"}.get(col, "")
    if vertical and horizontal:
        return f"the {vertical}-{horizontal} corner"
    if vertical or horizontal:
        return f"the {vertical or horizontal} edge"
    return f"row {row}, column {col}"


def grid_spec(cols: int, rows: int, procedure: str | None = None, phased: bool = True) -> DagSpec:
    """Every piece of a `cols` x `rows` jigsaw, border first if `phased`.

    A puzzle with no interior (two rows or two columns) just has no interior
    phase. Steps are listed corners, edges, interior, each in reading order, so
    the checklist reads the way the instructions do.
    """
    if cols < 2 or rows < 2:
        raise ValueError("a jigsaw needs at least two rows and two columns")
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    order = list(BORDER_FIRST)
    cells.sort(key=lambda rc: order.index(_kind(rc[0], rc[1], rows, cols)))

    steps = [
        Step(
            id=piece_name(r, c),
            instruction=f"Place piece {piece_name(r, c)} at {_where(r, c, rows, cols)}",
            part=PieceRef(piece=piece_name(r, c)),
            pose=Pose(x=c, y=r),
            phase=_kind(r, c, rows, cols) if phased else None,
            expected_duration_s=(2.0, 60.0),
        )
        for r, c in cells
    ]
    used = {s.phase for s in steps}
    phases = [
        Phase(id=pid, name=name, after=[a for a in after if a in used])
        for pid, (name, after) in BORDER_FIRST.items()
        if phased and pid in used
    ]
    return DagSpec(
        procedure=procedure or f"jigsaw_{cols}x{rows}",
        state_model="piece_grid",
        board=Board(cols=cols, rows=rows),
        steps=steps,
        phases=phases,
        provenance=Provenance(compiler_version=COMPILER_VERSION, model="hand",
                              source="thirdeye.compiler.jigsaw"),
    )


# --- the LLM route: instructions -> a plan of phases ----------------------------------

class PhasePlan(Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    #: How an alert names it: "the border", "the top row".
    name: str
    after: list[str] = Field(default_factory=list)
    #: Selectors; see `SELECTORS`.
    members: list[str]


class JigsawPlan(Strict):
    """What the model answers. An empty `phases` means "any order"."""

    phases: list[PhasePlan]


#: The selector vocabulary, as the prompt shows it. Rows count from 0 at the
#: top of the picture, columns from 0 at the left; `last` is the final one.
SELECTORS = """  corners              the four corner pieces
  edges                border pieces that are not corners
  border               corners and edges together
  interior             every piece not on the border
  row N / rows A-B     whole rows; N may be `last`
  column N / columns A-B
                       whole columns; N may be `last`
  piece rRcC           one piece, e.g. piece r0c3
  rest                 every piece no other phase selects (at most one phase)"""

_RANGE = re.compile(r"^(row|rows|column|columns|col|cols)\s+(\d+|last)(?:\s*-\s*(\d+|last))?$")


def select(selector: str, cols: int, rows: int) -> set[tuple[int, int]]:
    """The (row, col) cells a selector names. Raises ValueError for anything else."""
    sel = " ".join(selector.lower().split())
    cells = {(r, c) for r in range(rows) for c in range(cols)}
    kinds = {rc: _kind(*rc, rows, cols) for rc in cells}
    if sel in ("corners", "edges", "interior"):
        return {rc for rc, k in kinds.items() if k == sel}
    if sel == "border":
        return {rc for rc, k in kinds.items() if k != "interior"}
    if m := re.fullmatch(r"piece r(\d+)c(\d+)", sel):
        r, c = int(m[1]), int(m[2])
        if (r, c) not in cells:
            raise ValueError(f"{selector!r} is off a {cols}x{rows} board")
        return {(r, c)}
    if m := _RANGE.match(sel):
        axis = "row" if m[1].startswith("row") else "col"
        size = rows if axis == "row" else cols

        def index(text: str) -> int:
            n = size - 1 if text == "last" else int(text)
            if not 0 <= n < size:
                raise ValueError(f"{selector!r} is off a {cols}x{rows} board")
            return n

        lo = index(m[2])
        hi = index(m[3]) if m[3] else lo
        lo, hi = min(lo, hi), max(lo, hi)
        return {(r, c) for r, c in cells if lo <= (r if axis == "row" else c) <= hi}
    raise ValueError(f"unknown selector {selector!r}")


def plan_spec(
    plan: JigsawPlan, cols: int, rows: int, procedure: str | None = None
) -> tuple[DagSpec, list[str]]:
    """Order the board's pieces by a plan. Returns the spec and every problem
    with the plan itself (the spec validator then checks the rest)."""
    problems: list[str] = []
    owner: dict[tuple[int, int], str] = {}
    rest: list[str] = []
    for phase in plan.phases:
        for sel in phase.members:
            if " ".join(sel.lower().split()) == "rest":
                rest.append(phase.id)
                continue
            try:
                chosen = select(sel, cols, rows)
            except ValueError as exc:
                problems.append(f"phase {phase.id}: {exc}")
                continue
            for rc in sorted(chosen):
                if rc in owner and owner[rc] != phase.id:
                    problems.append(
                        f"piece {piece_name(*rc)} is in both {owner[rc]} and {phase.id}; "
                        "give each piece one phase (use `rest` for 'everything else')"
                    )
                owner.setdefault(rc, phase.id)
    if len(rest) > 1:
        problems.append(f"`rest` is used by {', '.join(rest)}; at most one phase may use it")
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    left = [rc for rc in cells if rc not in owner]
    if plan.phases and left:
        if rest:
            owner.update({rc: rest[0] for rc in left})
        else:
            names = ", ".join(piece_name(*rc) for rc in left[:6])
            problems.append(
                f"{len(left)} piece(s) are in no phase ({names}{', ...' if len(left) > 6 else ''}); "
                "add them to a phase or add a `rest` phase"
            )

    order = [p.id for p in plan.phases]
    cells.sort(key=lambda rc: order.index(owner[rc]) if rc in owner else len(order))
    steps = [
        Step(
            id=piece_name(r, c),
            instruction=f"Place piece {piece_name(r, c)} at {_where(r, c, rows, cols)}",
            part=PieceRef(piece=piece_name(r, c)),
            pose=Pose(x=c, y=r),
            phase=owner.get((r, c)),
            expected_duration_s=(2.0, 60.0),
        )
        for r, c in cells
    ]
    spec = DagSpec(
        procedure=procedure or f"jigsaw_{cols}x{rows}",
        state_model="piece_grid",
        board=Board(cols=cols, rows=rows),
        steps=steps,
        phases=[Phase(id=p.id, name=p.name, after=list(p.after)) for p in plan.phases],
    )
    return spec, problems


_JIGSAW_RULES = """You read the instructions for a jigsaw puzzle and say in what order its pieces
go down, as phases. The board is {cols} columns by {rows} rows. Rows count from 0
at the top of the picture; columns from 0 at the left.

A phase is a group of pieces the instructions ask for together, in any order
within the group. `after` lists the phases that must be finished before this
one starts. Select pieces only with these selectors:

{selectors}

Every piece must be in exactly one phase. Use snake_case ids and a short `name`
the player would recognise ("the border", "the top row"). If the instructions
give no order at all, answer with no phases. Do not add an order the
instructions do not state.

Example. "Find the four corners first. Then build the rest of the frame. Fill
in the middle last." becomes:

{{"phases": [
  {{"id": "corners", "name": "the corners", "after": [], "members": ["corners"]}},
  {{"id": "edges", "name": "the edges", "after": ["corners"], "members": ["edges"]}},
  {{"id": "interior", "name": "the interior", "after": ["corners", "edges"], "members": ["interior"]}}
]}}
"""


def plan_schema() -> dict[str, Any]:
    return _strict(JigsawPlan.model_json_schema())


def check_plan(raw: str, cols: int, rows: int, procedure: str | None = None) -> Attempt:
    """Parse, build and validate one model answer. Never raises on bad content."""
    attempt = Attempt(raw=raw)
    try:
        plan = JigsawPlan.model_validate_json(raw)
    except ValidationError as exc:
        attempt.problems = [
            f"schema: {'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        ]
        return attempt
    spec, problems = plan_spec(plan, cols, rows, procedure)
    checked = validate(spec)
    attempt.problems = problems + [str(p) for p in errors(checked)]
    attempt.warnings = [p for p in checked if p.severity == "warning"]
    attempt.spec = spec
    return attempt


def compile_instructions(
    source: str, cols: int, rows: int, llm: LLM, source_name: str = "",
    procedure: str | None = None, *, schema: bool = True, repair: bool = True,
) -> tuple[DagSpec, list[Problem]]:
    """Jigsaw instructions -> a phased piece-grid spec, with one repair round.
    (`schema` and `repair` off: the ablation baseline only.)"""
    system = _JIGSAW_RULES.format(cols=cols, rows=rows, selectors=SELECTORS)
    if not schema:
        system = with_schema_text(system, plan_schema())
    final, repaired = repair_loop(
        llm, system, source, plan_schema() if schema else None,
        lambda raw: check_plan(raw, cols, rows, procedure), repair=repair,
    )
    return stamp(final.spec, model_name(llm), source_name, repaired), final.warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cols", type=int)
    ap.add_argument("rows", type=int)
    ap.add_argument("-o", "--out", type=Path, help="write here instead of stdout")
    ap.add_argument("--any-order", action="store_true", help="no phases: pieces in any order")
    ap.add_argument("--instructions", type=Path,
                    help="read the order from this instruction sheet with the LLM")
    ap.add_argument("--model", default=None,
                    help=f"provider:model (default $THIRDEYE_LLM or {DEFAULT_MODEL})")
    ap.add_argument("--procedure", help="spec name (default jigsaw_COLSxROWS)")
    args = ap.parse_args(argv)

    if args.instructions:
        try:
            spec, warnings = compile_instructions(
                args.instructions.read_text(), args.cols, args.rows, make_llm(args.model),
                source_name=args.instructions.as_posix(), procedure=args.procedure,
            )
        except LLMError as exc:
            print(f"compile failed: {exc}", file=sys.stderr)
            return 1
        except CompileError as exc:
            print(f"compile failed: {exc}", file=sys.stderr)
            for p in exc.problems:
                print(f"  {p}", file=sys.stderr)
            return 1
        for w in warnings:
            print(w, file=sys.stderr)
    else:
        spec = grid_spec(args.cols, args.rows, procedure=args.procedure,
                         phased=not args.any_order)
    text = json.dumps(spec.model_dump(mode="json", exclude_none=True), indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"{len(spec.steps)} pieces -> {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
