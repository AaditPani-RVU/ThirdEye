"""The ground-truth label file for one recorded run (DR-JEN-3, EV-ERR-*).

One YAML file per run, written by whoever labels the video. It says what really
happened, so the evaluation can compare it against what the checker said. The
format is fixed *before* recording so the labelling produces exactly what the
scorer reads::

    run: jigsaw_R012              # unique; the session log for this run carries the same id
    procedure: jigsaw_4x3         # must equal the spec's `procedure`
    split: tune                   # tune | test (NFR-REP-2: fixed once, never moved)
    person: B                     # who performed it; splits are made by person
    video: raw/jigsaw_R012_personB.mp4
    planted: out_of_order r1c1    # what the run sheet asked for, or "none"
    errors:                       # every mistake that happened, planted or not
      - {code: out_of_order, target: r1c1, t: 11.2}
    # optional, for state accuracy (EV-STA-*); one of the two, per procedure
    placements:                   # jigsaw: one line per piece put down
      - {t: 3.4, piece: r0c0, cell: [0, 0], rot: 0}
    moves:                        # Jenga: one line per move
      - {start: 2.0, end: 5.1, src: [5, 1], dst: [18, 0], hands: 1}

An error's `code` is the checker's alert code (`missed_step`, `out_of_order`,
`wrong_rotation`, `illegal_source_layer`, ... and `game_over` for a collapse), so
the labeller and the checker speak one vocabulary. `target` is the step or piece
the alert would name; leave it out where the checker names none (a Jenga move is
`move`, and matching on it adds nothing). `t` is when the mistake became visible
-- the moment the piece settled, the block came out -- and, for a missed step,
the end of the run. Leave `t` out only for scripted runs, which have no clock
worth scoring a delay against.

An empty `errors` list is a claim too: it says the run was performed correctly,
and every alert on it is a false alarm (EV-ERR-3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    # A misspelt key in a hand-written label file must fail loudly, not be
    # dropped -- `taget:` silently ignored would turn a labelled error into a miss.
    model_config = ConfigDict(extra="forbid", frozen=True)


class ErrorLabel(_Strict):
    code: str
    target: str = ""
    t: float | None = None

    def __str__(self) -> str:
        at = "" if self.t is None else f" @ {self.t:.2f}"
        return f"{self.code} {self.target}".rstrip() + at


class Placement(_Strict):
    t: float
    piece: str
    cell: tuple[int, int]
    rot: Literal[0, 90, 180, 270] = 0


class Move(_Strict):
    start: float
    end: float
    src: tuple[int, int]
    dst: tuple[int, int] | None = None  # None: the block was dropped or the tower fell
    hands: int = 1


class RunLabels(_Strict):
    run: str
    procedure: str
    split: Literal["tune", "test", "scripted"] = "test"
    person: str = ""
    video: str = ""
    planted: str = "none"
    errors: list[ErrorLabel] = Field(default_factory=list)
    placements: list[Placement] = Field(default_factory=list)
    moves: list[Move] = Field(default_factory=list)

    @field_validator("errors", mode="before")
    @classmethod
    def _short_form(cls, value):
        """Accept `"out_of_order r1c1 @ 11.2"` as well as the mapping form, so a
        label file can be typed as fast as a scenario's `expect:` block."""
        return [parse_error(v) if isinstance(v, str) else v for v in value or []]

    @property
    def correct(self) -> bool:
        return not self.errors

    @classmethod
    def read(cls, path: str | Path) -> RunLabels:
        data = yaml.safe_load(Path(path).read_text())
        if not isinstance(data, dict):
            raise TypeError(f"{path}: a label file is a mapping with at least 'run' and 'procedure'")
        return cls.model_validate(data)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.model_dump(exclude_defaults=True), sort_keys=False))
        return path


def parse_error(text: str) -> ErrorLabel:
    """`code [target] [@ t]` -> ErrorLabel."""
    body, _, at = text.partition("@")
    words = body.split()
    if not 1 <= len(words) <= 2:
        raise ValueError(f"cannot read error label {text!r}; expected 'code [target] [@ t]'")
    return ErrorLabel(
        code=words[0],
        target=words[1] if len(words) == 2 else "",
        t=float(at) if at.strip() else None,
    )


def read_all(folder: str | Path) -> dict[str, RunLabels]:
    """Every `*.yaml` label file under `folder`, keyed by run id."""
    out: dict[str, RunLabels] = {}
    for path in sorted(Path(folder).glob("*.yaml")):
        labels = RunLabels.read(path)
        if labels.run in out:
            raise ValueError(f"{path}: run id {labels.run!r} is labelled twice")
        out[labels.run] = labels
    return out
