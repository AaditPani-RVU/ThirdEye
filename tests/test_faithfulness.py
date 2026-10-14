"""Report faithfulness: the writer's report is split into claims and judged against the log."""

from __future__ import annotations

import json
from pathlib import Path

from thirdeye.eval.faithfulness import evaluate, summary, table
from thirdeye.replay import Replay

SCENARIOS = Path(__file__).resolve().parents[1] / "scripts" / "scenarios"


class Writer:
    model = name = "scripted-writer"

    def complete(self, system, user, schema):
        return json.dumps({
            "headline": "One piece was turned.",
            "findings": [{"t": 99.0, "code": "made_up", "what_happened": "x", "consequence": ""}],
            "went_well": "", "advice": []})


class Judge:
    model = name = "scripted-judge"

    def __init__(self):
        self.seen = []

    def complete(self, system, user, schema):
        self.seen.append(user)
        assert "SESSION LOG" in user and "REPORT" in user
        return json.dumps({"claims": [
            {"claim": "one piece was turned", "supported": True, "evidence": "wrong_rotation"},
            {"claim": "the player was tired", "supported": False, "evidence": "not in the log"},
        ]})


def test_claims_are_judged_and_guard_rails_counted(tmp_path) -> None:
    record = Replay.load(SCENARIOS / "jigsaw_rotated.yaml").run()
    judge = Judge()
    (res,) = evaluate({"jigsaw_rotated": record}, Writer(), judge, tmp_path)
    assert res.dropped == 1 and res.restored == 1      # invented finding out, real alert back
    assert len(res.unsupported) == 1
    s = summary([res])
    assert s["unsupported_rate"] == 0.5 and s["claims"] == 2
    assert "the player was tired" in table([res])
    assert (tmp_path / "jigsaw_rotated.md").read_text().startswith("# Session report")
    assert json.loads((tmp_path / "jigsaw_rotated.verdict.json").read_text())["dropped"] == 1
