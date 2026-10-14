"""The jigsaw path: piece-grid spec, phases, tracker and match policy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from thirdeye.checker.engine import Checker
from thirdeye.compiler.jigsaw import grid_spec, main
from thirdeye.compiler.schema import (
    Baseplate,
    Board,
    DagSpec,
    PartRef,
    Phase,
    PieceRef,
    Pose,
    Step,
    parse_spec,
)
from thirdeye.compiler.validate import errors, validate
from thirdeye.events import Event
from thirdeye.replay import brick, part
from thirdeye.session import Session
from thirdeye.state.jigsaw.board import PieceBoard
from thirdeye.state.lego.grid import Reading, rotations_indistinguishable

CORNERS = ["r0c0", "r0c3", "r2c0", "r2c3"]
EDGES = ["r0c1", "r0c2", "r1c0", "r1c3", "r2c1", "r2c2"]


def piece(name: str, x: int, y: int, rot: int = 0, t: float = 1.0, kind: str = "ADD") -> Event:
    return Event(t=t, kind=kind, part=PieceRef(piece=name),  # type: ignore[arg-type]
                 pose=Pose(x=x, y=y, rot=rot))  # type: ignore[arg-type]


def home(name: str, **kw) -> Event:
    return piece(name, int(name[3]), int(name[1]), **kw)


def reading(name: str, x: int, y: int, rot: int = 0) -> Reading:
    return Reading(part=PieceRef(piece=name), pose=Pose(x=x, y=y, rot=rot),  # type: ignore[arg-type]
                   residual=0.0)


@pytest.fixture
def spec() -> DagSpec:
    return grid_spec(4, 3)


def codes(alerts) -> list[str]:
    return [f"{a.code} {a.target}".rstrip() for a in alerts]


# --- spec -----------------------------------------------------------------------------

def test_the_generated_spec_is_valid_and_border_first(spec: DagSpec) -> None:
    assert validate(spec) == []
    assert [s.id for s in spec.steps[:4]] == CORNERS
    assert spec.state_model == "piece_grid" and spec.extent == (4, 3)
    assert parse_spec(spec.model_dump(mode="json")) == spec


def test_a_puzzle_without_an_interior_has_no_interior_phase() -> None:
    small = grid_spec(2, 2)
    assert [p.id for p in small.phases] == ["corners"]
    assert validate(small) == []
    with pytest.raises(ValueError):
        grid_spec(1, 5)


def test_the_command_line_writes_a_spec(tmp_path) -> None:
    out = tmp_path / "j.json"
    assert main(["3", "3", "-o", str(out), "--any-order"]) == 0
    loaded = parse_spec(__import__("json").loads(out.read_text()))
    assert isinstance(loaded, DagSpec) and loaded.phases == []


def test_each_state_model_takes_its_own_grid_and_parts() -> None:
    step = Step(id="a", instruction="x", part=PieceRef(piece="r0c0"), pose=Pose(x=0, y=0))
    with pytest.raises(ValidationError, match="needs a board"):
        DagSpec(procedure="p", state_model="piece_grid", steps=[step])
    with pytest.raises(ValidationError, match="places PartRefs only"):
        DagSpec(procedure="p", baseplate=Baseplate(studs_x=4, studs_y=4), steps=[step])
    brick_step = Step(id="b", instruction="x", part=PartRef(color="red", size="2x2"),
                      pose=Pose(x=0, y=0))
    with pytest.raises(ValidationError, match="places PieceRefs only"):
        DagSpec(procedure="p", state_model="piece_grid", board=Board(cols=2, rows=2),
                steps=[brick_step])


def test_the_validator_catches_jigsaw_mistakes(spec: DagSpec) -> None:
    steps = list(spec.steps)
    steps[1] = steps[1].model_copy(update={"part": PieceRef(piece="r0c0")})
    steps[2] = steps[2].model_copy(update={"pose": Pose(x=9, y=0), "phase": "middle"})
    bad = spec.model_copy(update={"steps": steps})
    got = {(p.code, p.target) for p in errors(validate(bad))}
    assert ("duplicate_piece", "r0c3") in got
    assert ("out_of_bounds", "r2c0") in got
    assert ("unknown_phase", "r2c0") in got


def test_phase_cycles_and_empty_phases_are_reported(spec: DagSpec) -> None:
    looped = [p.model_copy(update={"after": ["interior"]}) if p.id == "corners" else p
              for p in spec.phases]
    problems = validate(spec.model_copy(update={"phases": looped}))
    assert "cycle" in {p.code for p in errors(problems)}
    spare = [*spec.phases, Phase(id="sky", name="the sky")]
    problems = validate(spec.model_copy(update={"phases": spare}))
    assert [(p.code, p.severity) for p in problems] == [("empty_phase", "warning")]


def test_a_phase_lowers_to_one_precondition_per_step(spec: DagSpec) -> None:
    lowered = {a.id: a for a in spec.lower()}
    assert lowered["r0c0"].preconditions == []
    [pre] = lowered["r1c1"].preconditions
    assert pre.expr == "phase_done('corners') and phase_done('edges')"
    assert "the corners and the edges" in pre.text


# --- checker ----------------------------------------------------------------------------

def test_an_early_piece_is_one_alert_however_much_is_left(spec: DagSpec) -> None:
    checker = Checker(spec)
    assert codes(checker.on_event(home("r1c1"))) == ["out_of_order r1c1"]
    # Performed, though flagged: it is not reported again at the end.
    assert "missed_step r1c1" not in codes(checker.finish())


def test_a_misplaced_border_piece_does_not_make_the_interior_early(spec: DagSpec) -> None:
    checker = Checker(spec)
    for name in CORNERS + EDGES[1:]:
        assert checker.on_event(home(name)) == []
    assert codes(checker.on_event(piece("r0c1", 1, 1))) == ["wrong_position r0c1"]
    assert checker.on_event(home("r1c2")) == []


def test_an_optional_step_does_not_hold_its_phase_open() -> None:
    spec = grid_spec(3, 3)
    steps = [s.model_copy(update={"optional": True}) if s.id == "r0c1" else s
             for s in spec.steps]
    checker = Checker(spec.model_copy(update={"steps": steps}))
    for name in ["r0c0", "r0c2", "r2c0", "r2c2", "r1c0", "r1c2", "r2c1"]:
        assert checker.on_event(home(name)) == []
    assert checker.on_event(home("r1c1")) == []


def test_a_piece_is_judged_by_where_it_belongs(spec: DagSpec) -> None:
    checker = Checker(spec)
    [alert] = checker.on_event(piece("r0c0", 3, 0))
    assert alert.code == "wrong_position" and alert.target == "r0c0"
    assert "belongs at row 0, column 0" in alert.text
    # The cell's own piece (r0c3) was not attempted and is not blamed.
    assert checker.status["r0c3"] == "pending"


def test_wrong_rotation_is_reported_with_the_turn(spec: DagSpec) -> None:
    [alert] = Checker(spec).on_event(home("r0c0", rot=270))
    assert alert.code == "wrong_rotation" and "turned 270 degrees" in alert.text


def test_a_piece_from_another_puzzle(spec: DagSpec) -> None:
    [alert] = Checker(spec).on_event(piece("zz", 1, 1))
    assert alert.code == "wrong_part" and alert.target == ""


def test_taking_out_a_placed_piece_reopens_it(spec: DagSpec) -> None:
    checker = Checker(spec)
    checker.on_event(home("r0c0"))
    assert codes(checker.on_event(home("r0c0", t=2.0, kind="REMOVE"))) == ["removed_part r0c0"]
    assert checker.status["r0c0"] == "pending"


# --- through the tracker ----------------------------------------------------------------

def _settle(session: Session, t: float, pieces: list[Reading]) -> tuple[float, list]:
    """Hand over the board, then half a second and more of a clear view."""
    board = session.state
    assert isinstance(board, PieceBoard)
    alerts = []
    board.observe(t, pieces, hands_present=True)
    for i in range(1, 12):
        board.observe(t + i * 0.1, pieces)
        alerts += session.pump()
    return t + 1.2, alerts


def test_session_builds_a_piece_board(spec: DagSpec) -> None:
    assert isinstance(Session(spec).state, PieceBoard)


def test_turning_a_piece_the_right_way_completes_it_quietly(spec: DagSpec) -> None:
    session = Session(spec)
    t, alerts = _settle(session, 0.0, [reading("r0c0", 0, 0, rot=90)])
    assert codes(alerts) == ["wrong_rotation r0c0"]
    t, alerts = _settle(session, t, [reading("r0c0", 0, 0)])
    assert alerts == [] and session.checker.status["r0c0"] == "done"


def test_a_piece_moved_behind_a_hand_is_one_alert(spec: DagSpec) -> None:
    """Lifted from home and dropped elsewhere while occluded: the tracker
    commits the arrival before the departure. That is one misplacement, not a
    misplacement and a removal."""
    session = Session(spec)
    t, alerts = _settle(session, 0.0, [reading("r0c0", 0, 0)])
    assert alerts == []
    t, alerts = _settle(session, t, [reading("r0c0", 1, 1)])
    assert codes(alerts) == ["wrong_position r0c0"]
    t, alerts = _settle(session, t, [reading("r0c0", 0, 0)])
    assert alerts == [] and session.checker.status["r0c0"] == "done"


def test_a_misplaced_piece_moved_home_completes_its_step(spec: DagSpec) -> None:
    session = Session(spec)
    t, alerts = _settle(session, 0.0, [reading("r0c0", 3, 0)])
    assert codes(alerts) == ["wrong_position r0c0"]
    t, alerts = _settle(session, t, [reading("r0c0", 0, 0)])
    assert alerts == [] and session.checker.status["r0c0"] == "done"


# --- small pieces -----------------------------------------------------------------------

def test_every_turn_of_a_piece_is_visible() -> None:
    p = PieceRef(piece="r0c0")
    assert not rotations_indistinguishable(p, 0, 180)
    assert rotations_indistinguishable(p, 90, 90)
    assert rotations_indistinguishable(PartRef(color="red", size="2x2"), 0, 90)


def test_the_replayer_reads_pieces() -> None:
    r = brick("r1c2 @ 2,1,180 conf=0.4")
    assert r.part == PieceRef(piece="r1c2") and r.pose.rot == 180 and r.confidence == 0.4
    assert part("r1c2") == PieceRef(piece="r1c2")
    assert part("red_2x4") == PartRef(color="red", size="2x4")
