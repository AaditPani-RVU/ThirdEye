"""Write the ground-truth specs for the compiler eval corpus (plan 5.3, EV-CMP-*).

    python scripts/make_eval_corpus.py            # writes manuals/eval_corpus/*.spec.json + corpus.yaml

The manuals in `manuals/eval_corpus/*.txt` are written by hand, as a person
would write instructions. Their correct specs are written here, compactly,
rather than as JSON by hand: a brick is one line, and the parts of a spec that
follow from geometry -- which bricks physically hold up which -- are computed,
so they cannot be mistyped. Every ground truth must pass the validator with no
errors *and* no warnings; a ground truth the validator doubts is not one.

What is written by hand, and so is the actual answer key: each brick's colour,
size and pose, and the order the manual states (`soft`), where it is anything
other than "each step after the one before".
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

from thirdeye.compiler.jigsaw import JigsawPlan, plan_spec
from thirdeye.compiler.schema import (
    Baseplate,
    ConstraintSpec,
    DagSpec,
    PartRef,
    Pose,
    Provenance,
    Step,
    parse_spec,
)
from thirdeye.compiler.validate import validate

CORPUS = Path(__file__).resolve().parents[1] / "manuals" / "eval_corpus"
HAND = Provenance(compiler_version="ground-truth", model="hand")

# --- LEGO: (colour, size, x, y, layer, rot), one per numbered manual line ----------

LEGO: dict[str, dict] = {
    "lego_bench": {"plate": (8, 8), "steps": [
        ("blue", "2x2", 0, 0, 0, 0),
        ("blue", "2x2", 4, 0, 0, 0),
        ("red", "2x6", 0, 0, 1, 90),
        ("white", "1x2", 5, 0, 2, 0),
    ]},
    "lego_tower": {"plate": (8, 8), "steps": [
        ("yellow", "2x2", 3, 3, 0, 0),
        ("red", "2x2", 3, 3, 1, 0),
        ("blue", "2x2", 3, 3, 2, 0),
        ("green", "2x2", 3, 3, 3, 0),
        ("white", "2x2", 3, 3, 4, 0),
    ]},
    "lego_bridge": {"plate": (12, 8), "steps": [
        ("black", "2x2", 2, 3, 0, 0),
        ("black", "2x2", 2, 3, 1, 0),
        ("black", "2x2", 8, 3, 0, 0),
        ("black", "2x2", 8, 3, 1, 0),
        ("red", "2x8", 2, 3, 2, 90),
        ("yellow", "1x2", 5, 3, 3, 0),
    ], "soft": {"s3": [], "s5": ["s2", "s4"]}},
    "lego_letter_l": {"plate": (10, 10), "steps": [
        ("green", "2x4", 0, 0, 0, 0),
        ("green", "2x4", 0, 4, 0, 0),
        ("green", "2x4", 2, 0, 0, 90),
        ("green", "2x2", 6, 0, 0, 0),
    ]},
    "lego_wall": {"plate": (16, 16), "steps": [
        ("red", "2x4", 0, 0, 0, 90),
        ("red", "2x4", 4, 0, 0, 90),
        ("red", "2x4", 8, 0, 0, 90),
        ("white", "2x4", 2, 0, 1, 90),
        ("white", "2x4", 6, 0, 1, 90),
        ("white", "2x2", 0, 0, 1, 0),
        ("white", "2x2", 10, 0, 1, 0),
    ]},
    "lego_table": {"plate": (10, 10), "steps": [
        ("black", "2x2", 2, 2, 0, 0),
        ("black", "2x2", 6, 2, 0, 0),
        ("black", "2x2", 2, 6, 0, 0),
        ("black", "2x2", 6, 6, 0, 0),
        ("brown", "2x6", 2, 2, 1, 90),
        ("brown", "2x6", 2, 6, 1, 90),
        ("yellow", "1x1", 4, 2, 2, 0),
    ], "soft": {"s2": [], "s3": [], "s4": [], "s5": ["s1", "s2", "s3", "s4"]},
       "optional": ["s7"]},
    "lego_frame": {"plate": (8, 8), "steps": [
        ("red", "1x4", 0, 0, 0, 0),
        ("red", "1x4", 3, 0, 0, 0),
        ("blue", "1x4", 0, 3, 1, 90),
        ("blue", "1x4", 0, 0, 1, 90),
        ("white", "2x2", 1, 1, 0, 0),
    ]},
    "lego_stairs": {"plate": (12, 6), "steps": [
        ("grey", "2x6", 0, 0, 0, 90),
        ("grey", "2x4", 2, 0, 1, 90),
        ("grey", "2x2", 4, 0, 2, 0),
        ("red", "2x2", 4, 0, 3, 0),
    ]},
}

# --- jigsaw: board size and the phase plan the instructions state -----------------

JIGSAW: dict[str, dict] = {
    "jigsaw_border_first": {"board": (6, 4), "phases": [
        ("corners", "the corners", [], ["corners"]),
        ("edges", "the edges", ["corners"], ["edges"]),
        ("interior", "the middle", ["corners", "edges"], ["interior"]),
    ]},
    "jigsaw_rows": {"board": (5, 4), "phases": [
        ("row0", "the top row", [], ["row 0"]),
        ("row1", "the second row", ["row0"], ["row 1"]),
        ("row2", "the third row", ["row1"], ["row 2"]),
        ("row3", "the bottom row", ["row2"], ["row 3"]),
    ]},
    "jigsaw_sky_first": {"board": (6, 5), "phases": [
        ("sky", "the sky", [], ["rows 0-1"]),
        ("ground", "the ground", ["sky"], ["row last"]),
        ("trees", "the trees", ["sky", "ground"], ["rest"]),
    ]},
    "jigsaw_free": {"board": (4, 3), "phases": []},
}

# --- Jenga: the rulebook's rules as preconditions -----------------------------------

SRC = ("src.layer <= top_complete_layer - 1", "Take only from below the top complete level",
       "illegal_source_layer")
HANDS = ("hands_in_contact == 1", "One hand at a time", "two_handed_move")
FINISH = ("dst.layer == top_layer or layer_complete(top_layer)",
          "Finish the top level before starting a new one", "layer_started_early")
ON_TOP = ("dst in legal_top_slots()", "Place blocks only in an empty spot on top",
          "wrong_replacement")

JENGA: dict[str, list[tuple[str, str, str]]] = {
    "jenga_standard": [SRC, HANDS, FINISH, ON_TOP],
    "jenga_house_rules": [
        ("src.layer <= top_complete_layer - 2", "The top two complete levels are off limits",
         "illegal_source_layer"),
        FINISH, ON_TOP,
    ],
    "jenga_sides_only": [
        SRC, HANDS, FINISH, ON_TOP,
        ("src.slot != 1", "Only side blocks may be taken", "middle_block_taken"),
    ],
    "jenga_free_top": [
        SRC, HANDS,
        ("dst.layer >= top_layer", "Put the block anywhere on top of the tower",
         "not_on_top"),
    ],
}


def manual_lines(name: str) -> list[str]:
    text = (CORPUS / f"{name}.txt").read_text()
    return [m[1].strip() for m in re.finditer(r"^\s*\d+\.\s+(.+)$", text, re.MULTILINE)]


def _supporters(i: int, steps: list[Step]) -> list[str]:
    me = steps[i]
    if me.pose.layer == 0:
        return []
    return [
        s.id for s in steps
        if s.id != me.id and s.pose.layer == me.pose.layer - 1 and s.footprint & me.footprint
    ]


def lego_spec(name: str, entry: dict) -> DagSpec:
    lines = manual_lines(name)
    if len(lines) != len(entry["steps"]):
        raise SystemExit(f"{name}: {len(lines)} manual steps but {len(entry['steps'])} bricks")
    steps = [
        Step(id=f"s{i + 1}", instruction=line, part=PartRef(color=c, size=size),
             pose=Pose(x=x, y=y, layer=layer, rot=rot))  # type: ignore[arg-type]
        for i, (line, (c, size, x, y, layer, rot)) in enumerate(
            zip(lines, entry["steps"], strict=True))
    ]
    soft_override = entry.get("soft", {})
    for i, step in enumerate(steps):
        step.hard_depends_on = _supporters(i, steps)
        order = soft_override.get(step.id, [f"s{i}"] if i else [])
        step.soft_depends_on = sorted(set(order) | set(step.hard_depends_on),
                                      key=lambda s: int(s[1:]))
        step.optional = step.id in entry.get("optional", [])
    return DagSpec(procedure=name, baseplate=Baseplate(studs_x=entry["plate"][0],
                                                       studs_y=entry["plate"][1]),
                   steps=steps, provenance=HAND)


def jigsaw_spec(name: str, entry: dict) -> DagSpec:
    plan = JigsawPlan.model_validate({"phases": [
        {"id": pid, "name": pname, "after": after, "members": members}
        for pid, pname, after, members in entry["phases"]
    ]})
    cols, rows = entry["board"]
    spec, problems = plan_spec(plan, cols, rows, procedure=name)
    if problems:
        raise SystemExit(f"{name}: {problems}")
    return spec.model_copy(update={"provenance": HAND})


def jenga_spec(name: str, rules: list[tuple[str, str, str]]) -> ConstraintSpec:
    return parse_spec({
        "procedure": name, "kind": "constraints", "state_model": "tower_lattice",
        "tower": {"layers": 18, "slots_per_layer": 3},
        "actions": [{
            "id": "move",
            "instruction": "Take one block and place it on top",
            "preconditions": [
                {"id": f"c{i + 1}", "expr": expr, "text": text, "violation": code}
                for i, (expr, text, code) in enumerate(rules)
            ],
            "effects": ["occupancy[src] = 0", "occupancy[dst] = 1"],
        }],
        "terminal": [{"id": "t1", "expr": "collapse_detected", "outcome": "game_over"}],
        "provenance": HAND.model_dump(),
    })  # type: ignore[return-value]


def main() -> int:
    index = []
    built = (
        [(n, "dag", lego_spec(n, e), None) for n, e in LEGO.items()]
        + [(n, "jigsaw", jigsaw_spec(n, e), list(e["board"])) for n, e in JIGSAW.items()]
        + [(n, "constraints", jenga_spec(n, r), None) for n, r in JENGA.items()]
    )
    bad = False
    for name, route, spec, board in built:
        problems = validate(spec)
        if problems:
            bad = True
            print(f"{name}: ground truth has problems:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
        if not (CORPUS / f"{name}.txt").exists():
            raise SystemExit(f"{name}: no manual at {CORPUS / (name + '.txt')}")
        out = CORPUS / f"{name}.spec.json"
        out.write_text(json.dumps(spec.model_dump(mode="json", exclude_none=True), indent=1) + "\n")
        entry = {"name": name, "route": route}
        if board:
            entry["board"] = board
        index.append(entry)
    (CORPUS / "corpus.yaml").write_text(
        "# The compiler eval corpus: one manual, one ground-truth spec each.\n"
        "# Written by scripts/make_eval_corpus.py; route = how the manual is compiled.\n"
        + yaml.safe_dump({"entries": index}, sort_keys=False)
    )
    print(f"{len(built)} ground-truth specs written to {CORPUS}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
