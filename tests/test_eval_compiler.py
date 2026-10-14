"""The compiler scorer and the eval corpus it grades against (EV-CMP-1..3)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from thirdeye.compiler.schema import load_spec, parse_spec
from thirdeye.compiler.validate import validate
from thirdeye.eval.compiler import (
    CORPUS,
    entries,
    run,
    sample_moves,
    score,
)

SAMPLES = sample_moves(600)


def _truth(name: str):
    return load_spec(CORPUS / f"{name}.spec.json")


def _edited(name: str, edit) -> object:
    data = json.loads((CORPUS / f"{name}.spec.json").read_text())
    edit(data)
    return parse_spec(data)


# --- the corpus itself ----------------------------------------------------------------


def test_the_corpus_has_every_route_and_each_entry_has_a_manual_and_a_valid_truth() -> None:
    corpus = entries()
    assert 15 <= len(corpus) <= 20
    assert {e.route for e in corpus} == {"dag", "jigsaw", "constraints"}
    for e in corpus:
        assert e.manual.exists(), e.name
        assert validate(load_spec(e.truth)) == [], e.name


def test_the_ground_truth_is_what_the_generator_writes() -> None:
    """The committed specs must not drift from scripts/make_eval_corpus.py."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "make_eval_corpus.py"
    module = importlib.util.spec_from_file_location("make_eval_corpus", path)
    gen = importlib.util.module_from_spec(module)
    module.loader.exec_module(gen)

    for name, entry in gen.LEGO.items():
        assert gen.lego_spec(name, entry).model_dump(mode="json", exclude_none=True) == \
            json.loads((CORPUS / f"{name}.spec.json").read_text())


# --- placement specs -------------------------------------------------------------------


def test_renumbered_steps_are_still_right() -> None:
    def renumber(d):
        ids = {s["id"]: f"step_{s['id']}" for s in d["steps"]}
        for s in d["steps"]:
            s["id"] = ids[s["id"]]
            s["hard_depends_on"] = [ids[x] for x in s["hard_depends_on"]]
            s["soft_depends_on"] = [ids[x] for x in s["soft_depends_on"]]
    got = _edited("lego_bridge", renumber)
    s = score(got, _truth("lego_bridge"))
    assert s["step_accuracy"] == 1.0 and s["order"]["recall"] == 1.0


def test_wrong_colour_and_wrong_place_are_told_apart() -> None:
    def edit(d):
        d["steps"][0]["part"]["color"] = "orange"   # wrong part, right place
        d["steps"][3]["pose"]["x"] = 7               # right part, wrong place
    s = score(_edited("lego_letter_l", edit), _truth("lego_letter_l"))
    assert s["errors"]["wrong_part"] == ["s1"] and s["errors"]["wrong_pose"] == ["s4"]
    assert s["step_accuracy"] == 0.5


def test_an_extra_step_is_hallucinated_and_a_dropped_one_missed() -> None:
    def edit(d):
        d["steps"].pop()
        d["steps"].append({**d["steps"][0], "id": "sx", "pose": {"x": 7, "y": 7, "layer": 0,
                                                                  "rot": 0}})
    s = score(_edited("lego_tower", edit), _truth("lego_tower"))
    assert s["errors"]["missed"] == ["s5"] and s["errors"]["hallucinated"] == ["sx"]


def test_a_forgotten_either_order_note_costs_order_precision() -> None:
    """The bridge manual lets the pillars go in either order; a strict chain
    adds an order the manual did not state."""
    def chain(d):
        for i, s in enumerate(d["steps"][1:], start=1):
            s["soft_depends_on"] = sorted({*s["soft_depends_on"], f"s{i}"})
    s = score(_edited("lego_bridge", chain), _truth("lego_bridge"))
    assert s["order"]["recall"] == 1.0 and s["order"]["precision"] < 1.0


def test_missing_support_costs_hard_dependency_recall() -> None:
    def edit(d):
        d["steps"][4]["hard_depends_on"] = ["s2"]
    s = score(_edited("lego_bridge", edit), _truth("lego_bridge"))
    assert s["hard_dependencies"]["recall"] == pytest.approx(4 / 5)


def test_phases_and_the_same_order_as_dependencies_score_the_same() -> None:
    def as_deps(d):
        corners = [s["id"] for s in d["steps"] if s.get("phase") == "corners"]
        border = [s["id"] for s in d["steps"] if s.get("phase") in ("corners", "edges")]
        for s in d["steps"]:
            s["soft_depends_on"] = {"edges": corners, "interior": border}.get(s.get("phase"), [])
            s.pop("phase", None)
        d["phases"] = []
    s = score(_edited("jigsaw_border_first", as_deps), _truth("jigsaw_border_first"))
    assert s["order"] == {"tp": s["order"]["tp"], "fp": 0, "fn": 0,
                          "precision": 1.0, "recall": 1.0}


def test_a_jigsaw_with_no_order_given_is_scored_on_imposed_order() -> None:
    s = score(_truth("jigsaw_border_first").model_copy(
        update={"procedure": "jigsaw_free"}), _truth("jigsaw_border_first"))
    assert s["order"]["precision"] == 1.0
    wrong = score(load_spec(CORPUS / "jigsaw_border_first.spec.json"),
                  _edited("jigsaw_border_first", lambda d: d.update(phases=[]) or [
                      s.pop("phase", None) for s in d["steps"]]))
    assert wrong["order"]["precision"] == 0.0


# --- constraint specs ------------------------------------------------------------------


def test_a_rule_written_differently_but_meaning_the_same_matches() -> None:
    def edit(d):
        d["actions"][0]["preconditions"][0]["expr"] = "src.layer < top_complete_layer"
        d["actions"][0]["preconditions"][0]["violation"] = "took_from_top"
    s = score(_edited("jenga_standard", edit), _truth("jenga_standard"), SAMPLES)
    assert s["constraint_accuracy"] == 1.0 and s["move_agreement"] == 1.0
    assert s["same_code"] == 3


def test_the_sampler_tells_the_standard_rules_apart() -> None:
    """If two different rules agreed on every sample, a model could swap them
    unnoticed. Every pair of distinct corpus rules must disagree somewhere."""
    from thirdeye.eval.compiler import _truth as truth

    exprs = {p.expr for e in entries() if e.route == "constraints"
             for a in load_spec(e.truth).actions for p in a.preconditions}
    vectors = {x: truth(x, SAMPLES) for x in exprs}
    assert len(set(vectors.values())) == len(exprs)


def test_a_missing_rule_and_an_invented_one_are_both_counted() -> None:
    def edit(d):
        pres = d["actions"][0]["preconditions"]
        pres.pop(1)  # forget "one hand at a time"
        pres.append({"id": "cx", "expr": "src.slot != 1", "text": "sides only",
                     "violation": "middle"})
    s = score(_edited("jenga_standard", edit), _truth("jenga_standard"), SAMPLES)
    assert s["constraint_accuracy"] == 0.75 and s["constraint_precision"] == 0.75
    assert s["errors"]["missed_rule"] == ["c2 (two_handed_move)"]
    assert s["move_agreement"] < 1.0


def test_a_spec_of_the_wrong_form_scores_as_such() -> None:
    assert score(_truth("jenga_standard"), _truth("lego_tower")) == {"wrong_form": True}


# --- the corpus run -----------------------------------------------------------------------


@dataclass
class Echo:
    """A stand-in model that answers every manual with its own ground truth."""

    model: str = "echo"

    def complete(self, system: str, user: str, schema: dict) -> str:
        for e in entries():
            if e.manual.read_text() == user:
                data = json.loads(e.truth.read_text())
                if e.route == "jigsaw":
                    return json.dumps({"phases": [
                        {"id": p["id"], "name": p["name"], "after": p["after"],
                         "members": [f"piece {s['id']}" for s in data["steps"]
                                     if s.get("phase") == p["id"]]}
                        for p in data.get("phases", [])
                    ]})
                return json.dumps(data)
        raise AssertionError("unknown manual")


def test_a_perfect_model_scores_perfectly_through_the_real_compile_loop(tmp_path: Path) -> None:
    result = run(tmp_path, llm=Echo())
    assert result.failed == {}
    sm = result.summary()
    assert sm["step_accuracy"] == 1.0 and sm["order_recall"] == 1.0
    assert sm["constraint_accuracy"] == 1.0
    assert (tmp_path / "lego_wall.json").exists()
    # and the saved specs re-score without the model
    assert run(tmp_path).summary() == sm
    shutil.rmtree(tmp_path)
