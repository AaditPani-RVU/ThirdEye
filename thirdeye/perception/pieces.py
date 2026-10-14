"""Jigsaw piece identification by matching against the box picture (FR-PER-2).

    python -m thirdeye.perception.pieces photo.jpg --reference box.jpg --board 4x3

No detector and no training. A jigsaw piece's identity is where its picture sits
in the finished image, so identifying a piece is asking "which cell of the box
picture does this look like, and which way round". Per frame:

    frame --homography--> the board, straight on, one square per cell
          --per cell-->   the centre of the cell: empty, or a piece
          --per piece-->  best (cell, turn) of the box picture

*The centre only.* A piece's tabs reach roughly a quarter of a cell into its
neighbours, and a neighbour's tabs reach into it, so the edges of a cell show
parts of other pieces. The middle `inner` fraction belongs to one piece.

*Texture and colour together.* Texture is compared by normalised correlation,
which ignores brightness and contrast, so a print under a desk lamp still
matches the box. Colour is compared separately, by mean Lab chroma (a, b but
not lightness, which is what a lighting change mostly moves), because a patch
of plain sky has no texture to correlate and colour is all that tells it apart.

*Occupancy by what changed.* With a photo of the empty board, a cell is
occupied when its chromaticity or its relative texture differs from the empty
cell. Both are unchanged by overall brightness, so a cloud passing the window
does not fill the board with phantom pieces. Take that photo: without it an
empty cell of a plain board looks just like a plain piece of sky, and only
matching quality is left to tell them apart.

*Honest confidence.* The confidence is the gap between the best match and the
runner-up. Two pieces of the same blue sky match each other nearly as well as
themselves, and the reading says so: its confidence is low, so the checker can
only raise a yellow warning from it (FR-CHK-9), never a red alert.

Rotation: a reading's `rot` is how far the piece is turned *clockwise* from
the way it sits in the finished picture, in 90-degree steps, matching `Pose.rot`.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from thirdeye.compiler.jigsaw import piece_name
from thirdeye.compiler.schema import Board, PieceRef, Pose
from thirdeye.state.lego.grid import Reading

#: Name reported for a piece that matches nothing in the box picture well
#: enough. The checker calls it a wrong part, never a missing one.
UNKNOWN = "unknown"


@dataclass(frozen=True)
class PieceMatch:
    """The best explanation of one cell's contents."""

    col: int
    row: int
    #: Home cell of the matched piece, as (row, col) in the box picture.
    home: tuple[int, int] | None
    rot: int
    score: float
    confidence: float

    @property
    def name(self) -> str:
        return UNKNOWN if self.home is None else piece_name(*self.home)

    def reading(self) -> Reading:
        return Reading(
            part=PieceRef(piece=self.name),
            pose=Pose(x=self.col, y=self.row, rot=self.rot),  # type: ignore[arg-type]
            residual=0.0,
            confidence=self.confidence,
        )


@dataclass
class PieceMatcher:
    """Identifies the pieces on a board against its box picture."""

    reference: np.ndarray
    board: Board
    #: Rectified pixels per cell. Enough detail to match, cheap to warp.
    cell_px: int = 64
    #: Side of the square descriptor each cell centre is shrunk to.
    patch: int = 16
    #: Fraction of the cell, centred, that belongs to one piece.
    inner: float = 0.6
    #: Weight of colour against texture in the score. Both terms are in [0, 1].
    colour_weight: float = 0.25
    #: Lab chroma distance at which colour stops counting at all.
    colour_scale: float = 40.0
    #: Below this score an occupied cell is a piece from somewhere else. True
    #: matches score near 1 on synthetic boards and foreign pieces up to ~0.6;
    #: to be re-measured on real photos, where print and camera differ.
    min_score: float = 0.7
    #: A best-vs-runner-up gap this large is full confidence.
    full_margin: float = 0.15
    #: Difference from the empty board, in chromaticity and relative-texture
    #: percent, above which a cell is occupied.
    occupied_delta: float = 12.0
    #: Lab standard deviation below which a patch counts as flat: no texture
    #: to correlate, so two flat patches agree on texture by default.
    flat_std: float = 4.0

    _tex: np.ndarray = field(init=False, repr=False)
    _col: np.ndarray = field(init=False, repr=False)
    _flat: np.ndarray = field(init=False, repr=False)
    _background: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        cols, rows = self.board.cols, self.board.rows
        ref = cv2.resize(self.reference, (cols * self.cell_px, rows * self.cell_px),
                         interpolation=cv2.INTER_AREA)
        crops = [self._centre(ref, c, r) for r in range(rows) for c in range(cols)]
        described = [self._describe(crop) for crop in crops]
        self._tex = np.stack([t for t, _, _ in described])
        self._col = np.stack([c for _, c, _ in described])
        self._flat = np.array([f for _, _, f in described])

    # --- geometry --------------------------------------------------------------------

    def rectify(self, frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """The board straight on, `cell_px` per cell. `homography` maps frame
        pixels to cell coordinates, as calibration produces for the board."""
        scale = np.diag([self.cell_px, self.cell_px, 1.0])
        size = (self.board.cols * self.cell_px, self.board.rows * self.cell_px)
        return cv2.warpPerspective(frame, scale @ np.asarray(homography, float), size,
                                   flags=cv2.INTER_AREA)

    def _centre(self, img: np.ndarray, col: int, row: int) -> np.ndarray:
        lo = round(self.cell_px * (1 - self.inner) / 2)
        hi = self.cell_px - lo
        y0, x0 = row * self.cell_px, col * self.cell_px
        return img[y0 + lo:y0 + hi, x0 + lo:x0 + hi]

    # --- description -----------------------------------------------------------------

    def _describe(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """(texture vector, mean Lab chroma, flatness). Texture is zero-mean
        and unit-norm, so correlation is a dot product. Flatness runs from 0,
        textured, to 1, a plain patch whose texture vector is only noise."""
        small = cv2.resize(crop, (self.patch, self.patch), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        # OpenCV's 8-bit Lab stores L in 0..255; bring it to the usual 0..100.
        lab[..., 0] *= 100.0 / 255.0
        mean = lab.reshape(-1, 3).mean(axis=0)
        tex = (lab - mean).ravel()
        norm = float(np.linalg.norm(tex))
        std = norm / np.sqrt(tex.size)
        flat = 1.0 - min(std / self.flat_std, 1.0)
        return (tex / norm if norm > 1e-6 else tex), mean[1:], flat

    def _scores(self, crop: np.ndarray) -> np.ndarray:
        """Score of every (turn, reference cell): shape (4, cells)."""
        out = []
        for k in range(4):
            # Turning the crop anticlockwise k quarter-turns undoes a piece
            # that sits k quarter-turns clockwise.
            tex, col, flat = self._describe(np.ascontiguousarray(np.rot90(crop, k)))
            # Correlation where both have texture; agreement where both are
            # plain; nothing where one is plain and the other is not.
            texture = ((1 - flat) * (1 - self._flat) * np.clip(self._tex @ tex, 0.0, 1.0)
                       + flat * self._flat)
            colour = 1.0 - np.clip(np.linalg.norm(self._col - col, axis=1) / self.colour_scale,
                                   0.0, 1.0)
            out.append((1 - self.colour_weight) * texture + self.colour_weight * colour)
        return np.stack(out)

    # --- the per-frame path ------------------------------------------------------------

    @property
    def has_background(self) -> bool:
        return self._background is not None

    def set_background(self, frame: np.ndarray, homography: np.ndarray) -> None:
        """Remember the empty board, so an occupied cell can be told from an
        empty one by what changed rather than by what matched."""
        self._background = self._cell_features(self.rectify(frame, homography))

    def _cell_features(self, board: np.ndarray) -> np.ndarray:
        """(rows, cols, 4): chromaticity b, g, r and relative texture, in
        percent. All four are ratios, so a change of overall gain leaves them."""
        px = board.astype(np.float32) + 1.0
        chroma = px / px.sum(axis=2, keepdims=True)
        grey = px.mean(axis=2)
        out = np.empty((self.board.rows, self.board.cols, 4), np.float32)
        for r in range(self.board.rows):
            for c in range(self.board.cols):
                g = self._centre(grey, c, r)
                out[r, c, :3] = self._centre(chroma, c, r).reshape(-1, 3).mean(axis=0)
                out[r, c, 3] = g.std() / g.mean()
        return out * 100.0

    def occupied(self, board: np.ndarray) -> np.ndarray:
        """(rows, cols) booleans. Without a background, every cell is a
        candidate and `min_score` alone decides."""
        if self._background is None:
            return np.ones((self.board.rows, self.board.cols), bool)
        delta = np.linalg.norm(self._cell_features(board) - self._background, axis=2)
        return delta > self.occupied_delta

    def match_cell(self, board: np.ndarray, col: int, row: int) -> PieceMatch:
        scores = self._scores(self._centre(board, col, row))
        k, cell = np.unravel_index(int(np.argmax(scores)), scores.shape)
        best = float(scores[k, cell])
        runner_up = float(np.partition(scores.ravel(), -2)[-2])
        confidence = float(np.clip((best - runner_up) / self.full_margin, 0.0, 1.0))
        cols = self.board.cols
        home = (int(cell) // cols, int(cell) % cols) if best >= self.min_score else None
        return PieceMatch(col=col, row=row, home=home, rot=90 * int(k), score=best,
                          confidence=confidence if home is not None else 0.3)

    def read(self, frame: np.ndarray, homography: np.ndarray) -> list[PieceMatch]:
        """Every piece on the board this frame.

        A cell that matches nothing is reported only when the background says
        something is there; without a background it is just an empty cell.
        """
        board = self.rectify(frame, homography)
        mask = self.occupied(board)
        out = []
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                if not mask[row, col]:
                    continue
                m = self.match_cell(board, col, row)
                if m.home is None and self._background is None:
                    continue
                out.append(m)
        return out

    def readings(self, frame: np.ndarray, homography: np.ndarray) -> list[Reading]:
        return [m.reading() for m in self.read(frame, homography)]


# --- trying it on a photo ------------------------------------------------------------

def annotate(board: np.ndarray, matches: list[PieceMatch], cell_px: int) -> np.ndarray:
    out = board.copy()
    for m in matches:
        x, y = m.col * cell_px, m.row * cell_px
        right = m.home == (m.row, m.col) and m.rot == 0
        colour = (60, 180, 60) if right else (40, 40, 220)
        cv2.rectangle(out, (x + 2, y + 2), (x + cell_px - 3, y + cell_px - 3), colour, 2)
        label = f"{m.name}/{m.rot}"
        cv2.putText(out, label, (x + 4, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
        cv2.putText(out, f"{m.confidence:.2f}", (x + 4, y + cell_px - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    return out


def main(argv: list[str] | None = None) -> int:
    from thirdeye.perception.calibrate import Calibrator, MarkerLayout

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("photo", type=Path, help="top-down photo of the board with its markers")
    ap.add_argument("--reference", type=Path, required=True, help="the box picture")
    ap.add_argument("--board", required=True, help="COLSxROWS, e.g. 4x3")
    ap.add_argument("--empty", type=Path, help="photo of the empty board, same camera position")
    ap.add_argument("--marker-cells", type=float, default=1.0,
                    help="printed marker side, in cells (see scripts/make_markers.py)")
    ap.add_argument("--gap-cells", type=float, default=0.25)
    ap.add_argument("-o", "--out", type=Path, help="write the annotated board here")
    args = ap.parse_args(argv)

    cols, rows = (int(n) for n in args.board.lower().split("x"))
    board = Board(cols=cols, rows=rows)
    reference = cv2.imread(str(args.reference))
    photo = cv2.imread(str(args.photo))
    if reference is None or photo is None:
        ap.error("could not read the reference or the photo")

    layout = MarkerLayout.around_plate(cols, rows, side_studs=args.marker_cells,
                                       gap_studs=args.gap_cells)
    cal = Calibrator(layout).refresh(photo)
    if not cal.usable or cal.homography is None:
        ap.error(f"calibration failed ({cal.status}); saw markers {sorted(cal.seen)}")

    matcher = PieceMatcher(reference, board)
    if not args.empty:
        print("note: no --empty photo; plain pieces cannot be told from empty cells",
              file=sys.stderr)
    if args.empty:
        empty = cv2.imread(str(args.empty))
        if empty is None:
            ap.error("could not read the empty-board photo")
        matcher.set_background(empty, cal.homography)
    matches = matcher.read(photo, cal.homography)
    for m in matches:
        verdict = "ok" if m.home == (m.row, m.col) and m.rot == 0 else "MISPLACED"
        print(f"row {m.row} col {m.col}: {m.name} rot {m.rot:>3}  "
              f"score {m.score:.2f} conf {m.confidence:.2f}  {verdict}")
    if args.out:
        cv2.imwrite(str(args.out),
                    annotate(matcher.rectify(photo, cal.homography), matches, matcher.cell_px))
        print(f"annotated -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
