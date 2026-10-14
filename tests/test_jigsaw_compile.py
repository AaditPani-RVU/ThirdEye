"""The jigsaw LLM route: instructions -> a plan of phases -> a piece-grid spec.

The model is a scripted stand-in, as in test_compile.py; no API call is spent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from thirdeye.checker.engine import Checker
from thirdeye.compiler.compile import CompileError
from thirdeye.compiler.jigsaw import check_plan, compile_instructions, grid_spec, select
from thirdeye.compiler.schema import PieceRef, Pose
from thirdeye.events import Event


@dataclass
class Scripted:
    outputs: list[str]
    model: str = "scripted"
    prompts: list[str] = field(default_factory=list)

    def complete(self, system: str, user: str, schema: dict) -> str:
        self.prompts.append(user)
        return self.outputs.pop(0)


def _plan(*phases: tuple[str, list[str], list[str]]) -> str:
    return json.dumps({"phases": [
        {"id": pid, "name": f"the {pid}", "after": after, "members": members}
        for pid, members, after in phases
    ]})


BORDER_FIRST = _plan(
    ("corners", ["corners"], []),
    ("edges", ["edges"], ["corners"]),
    ("interior", ["interior"], ["corners", "edges"]),
)


def test_selectors_name_the_cells_they_say() -> None:
    assert select("corners", 4, 3) == {(0, 0), (0, 3), (2, 0), (2, 3)}
    assert len(select("border", 4, 3)) == 10 and select("interior", 4, 3) == {(1, 1), (1, 2)}
    assert select("row last", 4, 3) == {(2, c) for c in range(4)}
    assert select("Columns 2 - last", 4, 3) == {(r, c) for r in range(3) for c in (2, 3)}
    assert select("piece r1c2", 4, 3) == {(1, 2)}
    for bad in ("row 3", "piece r9c9", "the sky", "diagonal"):
        with pytest.raises(ValueError):
            select(bad, 4, 3)


def test_a_border_first_plan_rebuilds_the_hand_spec() -> None:
    attempt = check_plan(BORDER_FIRST, 4, 3)
    assert attempt.problems == []
    hand = grid_spec(4, 3)
    assert {(s.id, s.phase) for s in attempt.spec.steps} == {(s.id, s.phase) for s in hand.steps}
    assert [p.after for p in attempt.spec.phases] == [p.after for p in hand.phases]


def test_rest_takes_what_no_other_phase_selects() -> None:
    attempt = check_plan(_plan(("top", ["row 0"], []), ("others", ["rest"], ["top"])), 4, 3)
    assert attempt.problems == []
    phases = {s.id: s.phase for s in attempt.spec.steps}
    assert phases["r0c2"] == "top" and phases["r2c2"] == "others"


@pytest.mark.parametrize(("plan", "complaint"), [
    (_plan(("a", ["row 0"], []), ("b", ["corners"], ["a"]), ("c", ["rest"], [])), "in both"),
    (_plan(("a", ["border"], [])), "in no phase"),
    (_plan(("a", ["row 0"], []), ("b", ["rest"], []), ("c", ["rest"], [])), "at most one"),
    (_plan(("a", ["the sky"], [])), "unknown selector"),
    (_plan(("a", ["border"], ["b"]), ("b", ["rest"], ["a"])), "cycle"),
])
def test_a_bad_plan_is_a_problem_the_repair_round_can_read(plan: str, complaint: str) -> None:
    assert any(complaint in p for p in check_plan(plan, 4, 3).problems)


def test_no_phases_means_any_order() -> None:
    attempt = check_plan('{"phases": []}', 4, 3)
    assert attempt.problems == [] and all(s.phase is None for s in attempt.spec.steps)


def test_a_bad_first_answer_is_repaired_once_and_stamped() -> None:
    llm = Scripted([_plan(("a", ["border"], [])), BORDER_FIRST])
    spec, _ = compile_instructions("Corners, then edges, then the middle.", 4, 3, llm,
                                   source_name="sheet.txt")
    assert spec.provenance.repaired and spec.provenance.source == "sheet.txt"
    assert "in no phase" in llm.prompts[1]


def test_two_bad_answers_stop_the_compiler() -> None:
    bad = _plan(("a", ["border"], []))
    with pytest.raises(CompileError):
        compile_instructions("...", 4, 3, Scripted([bad, bad]))


def test_a_compiled_plan_drives_the_checker() -> None:
    spec, _ = compile_instructions("top row first", 4, 3, Scripted([
        _plan(("top", ["row 0"], []), ("others", ["rest"], ["top"])),
    ]))
    checker = Checker(spec)
    alerts = checker.on_event(Event(t=1.0, kind="ADD", part=PieceRef(piece="r2c2"),
                                    pose=Pose(x=2, y=2)))
    assert [a.code for a in alerts] == ["out_of_order"]
    assert "the top" in alerts[0].text
