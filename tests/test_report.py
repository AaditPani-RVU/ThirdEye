"""The session report is written from the log only, and held to it (FR-EXP-1, FR-EXP-2)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from thirdeye.explain.report import digest, main, plain, with_llm
from thirdeye.replay import Replay

SCENARIOS = Path(__file__).resolve().parents[1] / "scripts" / "scenarios"


@pytest.fixture(scope="module")
def jenga():
    return Replay.load(SCENARIOS / "jenga_fouls.yaml").run()


@pytest.fixture(scope="module")
def jigsaw_ok():
    return Replay.load(SCENARIOS / "jigsaw_correct.yaml").run()


@dataclass
class Canned:
    answer: dict
    model: str = "canned"
    seen: list[str] = field(default_factory=list)

    def complete(self, system: str, user: str, schema: dict) -> str:
        self.seen.append(user)
        return json.dumps(self.answer)


def _finding(t: float, code: str) -> dict:
    return {"t": t, "code": code, "what_happened": "x.", "consequence": "y."}


def test_the_digest_is_text_from_the_log(jenga) -> None:
    text = digest(jenga)
    assert "t=11.0 illegal_source_layer" in text and "Outcome: game_over" in text
    assert "Moves made: 6" in text


def test_the_plain_report_lists_every_alert_in_time_order(jenga) -> None:
    md = plain(jenga).markdown()
    assert md.count("\n- **") == len(jenga.alerts)
    assert md.index("illegal source layer") < md.index("· game over")


def test_a_clean_run_says_so(jigsaw_ok) -> None:
    md = plain(jigsaw_ok).markdown()
    assert "What went wrong" not in md and "No mistakes were flagged" in md


def test_a_finding_citing_no_alert_is_dropped(jenga) -> None:
    llm = Canned({"headline": "h", "went_well": "", "advice": [], "findings": [
        *[_finding(a.t, a.code) for a in jenga.alerts],
        _finding(3.0, "dropped_block"),                    # never happened
    ]})
    report = with_llm(jenga, llm)
    assert [f.code for f in report.rejected] == ["dropped_block"]
    assert "dropped_block" not in report.markdown()
    assert "1 model statement(s) dropped" in report.markdown()


def test_an_alert_the_model_left_out_is_added_back(jenga) -> None:
    llm = Canned({"headline": "h", "went_well": "", "advice": [],
                  "findings": [_finding(11.0, "illegal_source_layer")]})
    report = with_llm(jenga, llm)
    assert len(report.body.findings) == len(jenga.red())
    assert len(report.restored) == len(jenga.red()) - 1


def test_two_alerts_at_one_time_each_need_their_own_citation(jenga) -> None:
    """One alert backs one finding: saying the same thing three times cites it once."""
    llm = Canned({"headline": "h", "went_well": "", "advice": [],
                  "findings": [_finding(35.0, "wrong_replacement")] * 3})
    report = with_llm(jenga, llm)
    assert len(report.rejected) == 2


def test_an_unreadable_answer_falls_back_to_the_plain_report(jenga) -> None:
    class Garbage:
        model = "garbage"

        def complete(self, *_):
            return "not json"
    report = with_llm(jenga, Garbage())
    assert "plain report used" in report.author
    assert len(report.body.findings) == len(jenga.alerts)


def test_the_command_line_writes_markdown(tmp_path: Path, jenga, capsys) -> None:
    log = jenga.write(tmp_path / "log.json")
    assert main([str(log), "-o", str(tmp_path / "r.md")]) == 0
    assert (tmp_path / "r.md").read_text().startswith("# Session report: jenga_game")
    assert main([str(log), "--digest"]) == 0
    assert "What the camera saw" in capsys.readouterr().out
