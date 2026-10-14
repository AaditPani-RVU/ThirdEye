"""The jigsaw board tracker (FR-STA-2, FR-STA-3, FR-STA-4).

A jigsaw on a board is a stud grid with one-cell parts: each cell holds at most
one piece, a piece is a `PieceRef` named by its home cell, and a reading is that
piece snapped to the cell it lies in, turned some multiple of 90 degrees. So the
LEGO tracker already does everything this needs -- freeze while a hand is over
the board, commit a piece only once it has held still, report a turn in place as
a removal and a fresh placement -- and this class exists to say so and to build
it from a `Board`, not to do anything different.

What does differ lives elsewhere: identifying a piece is perception's job
(matching against the box picture), and what a misplaced piece *means* is the
checker's (`checker/rules_jigsaw.py`).

Coordinates are cells: `x` is the column and `y` the row, both counted from the
top-left of the finished picture, the way the box image is read.
"""

from __future__ import annotations

from thirdeye.compiler.schema import Board
from thirdeye.state.lego.build_state import BuildState
from thirdeye.state.lego.grid import StudGrid


class PieceBoard(BuildState):
    """Which piece is in which cell of the board, and what changed."""

    @classmethod
    def from_board(cls, board: Board, **kw) -> PieceBoard:
        return cls(StudGrid(board.cols, board.rows), **kw)
