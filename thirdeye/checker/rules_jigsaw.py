"""Matching and alert text for the jigsaw board (FR-CHK-3..6).

A jigsaw differs from LEGO in one way that decides everything here: every piece
is unique. A brick's identity is only its colour and size, so a wrong brick in a
cell is judged against the *cell* -- what was wanted there. A piece's identity is
its home, so a misplaced piece is judged against the *piece* -- where it belongs:

    the piece's step is open, same cell, rot differs -> wrong rotation (FR-CHK-6)
    the piece's step is open, different cell         -> wrong position (FR-CHK-6)
    no step places this piece                        -> wrong part     (FR-CHK-5)
    a completed piece leaves its cell                -> removed part   (FR-CHK-6)

A piece in someone else's cell raises one alert, about itself. The cell's own
piece has not been attempted yet, and blaming it too would be two alerts for
one mistake (NFR-REL-2).

As with LEGO, a misplaced piece is an *attempt* at its step: the step goes to
`error`, which counts as performed for `phase_done()`, so one misplaced border
piece does not also make every interior piece out of order. Moving it to the
right cell, or turning it the right way, completes the step quietly.

Out-of-order and missed-step are not here: they come from the engine's generic
precondition loop and session-end sweep, the same code as for LEGO.
"""

from __future__ import annotations

from typing import NamedTuple

from thirdeye.compiler.schema import LoweredAction, PieceRef, Pose, Precondition
from thirdeye.events import COMPLETED, OPEN, Alert, Event, Status

WRONG_POSITION = "wrong_position"
WRONG_ROTATION = "wrong_rotation"
WRONG_PART = "wrong_part"
REMOVED_PART = "removed_part"

Cell = tuple[int, int]


def cell_str(p: Pose) -> str:
    return f"row {p.y}, column {p.x}"


def _cell(p: Pose) -> Cell:
    return (p.x, p.y)


class _Placement(NamedTuple):
    """An action that puts a known piece in a known cell."""

    action: LoweredAction
    piece: PieceRef
    pose: Pose


class PieceGridPolicy:
    """The jigsaw half of the checker."""

    def __init__(self) -> None:
        #: Cells holding a misplaced piece, and the step it was an attempt at.
        self._stand_ins: dict[Cell, str] = {}

    def match(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> LoweredAction | None:
        if ev.part is None or ev.pose is None:
            return None
        for cand in self._placements(actions):
            if (
                cand.piece == ev.part
                and _cell(cand.pose) == _cell(ev.pose)
                and cand.pose.rot == ev.pose.rot
                and self._is_open(cand.action.id, status)
            ):
                return cand.action
        return None

    def unmatched(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> list[Alert]:
        part, pose = ev.part, ev.pose
        if part is None or pose is None:
            return []
        home = next((c for c in self._placements(actions) if c.piece == part), None)

        if home is None:
            return [
                Alert(
                    t=ev.t,
                    code=WRONG_PART,
                    rule="jigsaw.wrong_part",
                    text=(
                        f"The {part} at {cell_str(pose)} is not part of this puzzle."
                    ),
                )
            ]

        step = home.action
        if _cell(pose) == _cell(home.pose) and pose.rot == home.pose.rot:
            # Exactly where it belongs, and its step is already credited: the
            # same piece seen again, e.g. a re-read's confirmation arriving
            # before the tracker's commit. Not an error.
            return []
        # Anything else is a misplacement -- including a piece already credited
        # at home and now seen elsewhere. That is a piece moved while a hand
        # covered the board: the tracker commits the arrival before the
        # departure, and the removal that follows is recognised as part of it.

        if _cell(pose) == _cell(home.pose):
            alert = Alert(
                t=ev.t,
                code=WRONG_ROTATION,
                rule="jigsaw.wrong_rotation",
                target=step.id,
                text=(
                    f"The {part} is in the right place, {cell_str(pose)}, but turned "
                    f"{(pose.rot - home.pose.rot) % 360} degrees: \"{step.instruction}\""
                ),
            )
        else:
            alert = Alert(
                t=ev.t,
                code=WRONG_POSITION,
                rule="jigsaw.wrong_position",
                target=step.id,
                text=(
                    f"The {part} is at {cell_str(pose)}, but it belongs at "
                    f"{cell_str(home.pose)}: \"{step.instruction}\""
                ),
            )
        status[step.id] = "error"
        self._stand_ins[_cell(pose)] = step.id
        return [alert]

    def on_remove(
        self, ev: Event, actions: list[LoweredAction], status: dict[str, Status]
    ) -> list[Alert]:
        """A misplaced piece lifted out withdraws its attempt without comment; a
        correctly placed one lifted out undoes its step, and says so."""
        if ev.pose is None:
            return []
        attempted = self._stand_ins.pop(_cell(ev.pose), None)
        if attempted is not None:
            if status.get(attempted) == "error" and attempted not in self._stand_ins.values():
                status[attempted] = "pending"
            return []
        for cand in self._placements(actions):
            if _cell(cand.pose) != _cell(ev.pose) or status.get(cand.action.id) not in COMPLETED:
                continue
            if cand.action.id in self._stand_ins.values():
                # The piece has already been seen somewhere else, and that
                # misplacement was reported. Its leaving home is the same move.
                return []
            status[cand.action.id] = "pending"
            return [
                Alert(
                    t=ev.t,
                    code=REMOVED_PART,
                    rule="jigsaw.removed_part",
                    target=cand.action.id,
                    text=(
                        f"The {cand.piece} has been taken out of {cell_str(cand.pose)}; "
                        f"step {cand.action.id} is no longer done."
                    ),
                )
            ]
        return []

    def matches_hint(self, ev: Event, action: LoweredAction) -> bool:
        return ev.part is not None and action.part == ev.part

    def explain(self, pre: Precondition, action: LoweredAction, ev: Event) -> str:
        where = "" if action.pose is None else f" at {cell_str(action.pose)}"
        return f"{pre.text.capitalize()}. Step {action.id}{where} is \"{action.instruction}\"."

    def missed_text(self, action: LoweredAction | None, action_id: str) -> str:
        if action is None or action.part is None or action.pose is None:
            return f"Step {action_id} was never completed."
        return (
            f"The {action.part} was never placed at {cell_str(action.pose)} -- "
            f"\"{action.instruction}\""
        )

    # --- helpers --------------------------------------------------------------------

    def _is_open(self, action_id: str, status: dict[str, Status]) -> bool:
        """Open to being completed: not yet done, or attempted wrongly so far.
        `active` counts, as a temporal hint may have got there first."""
        return status.get(action_id) in OPEN or (
            status.get(action_id) == "error" and action_id in self._stand_ins.values()
        )

    @staticmethod
    def _placements(actions: list[LoweredAction]) -> list[_Placement]:
        return [
            _Placement(a, a.part, a.pose)
            for a in actions
            if isinstance(a.part, PieceRef) and a.pose is not None
        ]
