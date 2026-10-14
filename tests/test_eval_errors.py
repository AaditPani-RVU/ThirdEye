"""The label format and the error scorer (EV-ERR-1..4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from thirdeye.eval.errors import Report, main, pair, score_run
from thirdeye.eval.labels import ErrorLabel, RunLabels, parse_error, read_all
from thirdeye.events import Alert
from thirdeye.sessionlog import SessionRecord

SCENARIOS = Path(__file__).resolve().parents[1] / "scripts" / "scenarios"


def _record(*alerts: Alert, procedure: str = "jigsaw_4x3", run: str = "") -> SessionRecord:
    return SessionRecord(
        procedure=procedure, kind="dag", alerts=list(alerts), log=[], checklist=[],
        meta={"run": run} if run else {},
    )


def _alert(code: str, target: str = "", t: float = 0.0, severity="alert") -> Alert:
    return Alert(t=t, code=code, rule="r", text="x", target=target, severity=severity)


def _labels(*errors: str, run: str = "R1", procedure: str = "jigsaw_4x3") -> RunLabels:
    return RunLabels(run=run, procedure=procedure, errors=list(errors))


# --- labels ----------------------------------------------------------------------


def test_the_short_form_reads_code_target_and_time() -> None:
    assert parse_error("out_of_order r1c1 @ 11.2") == ErrorLabel(
        code="out_of_order", target="r1c1", t=11.2
    )
    assert parse_error("game_over") == ErrorLabel(code="game_over")
    with pytest.raises(ValueError):
        parse_error("wrong piece in r1c1")


def test_a_misspelt_key_is_refused_not_dropped() -> None:
    with pytest.raises(ValidationError):
        RunLabels.model_validate({"run": "R1", "procedure": "p", "erors": []})
    with pytest.raises(ValidationError):
        RunLabels.model_validate(
            {"run": "R1", "procedure": "p", "errors": [{"code": "x", "taget": "r1c1"}]}
        )


def test_a_label_file_round_trips_and_a_duplicate_run_is_refused(tmp_path: Path) -> None:
    labels = RunLabels(
        run="R1", procedure="jigsaw_4x3", split="tune",
        errors=["wrong_rotation r2c2 @ 20.5"],
        placements=[{"t": 3.4, "piece": "r0c0", "cell": [0, 0]}],
    )
    labels.write(tmp_path / "a.yaml")
    assert RunLabels.read(tmp_path / "a.yaml") == labels
    assert not labels.correct
    labels.write(tmp_path / "b.yaml")
    with pytest.raises(ValueError, match="labelled twice"):
        read_all(tmp_path)


# --- matching --------------------------------------------------------------------


def test_an_alert_within_the_window_is_a_hit_and_its_delay_is_measured() -> None:
    s = score_run(_labels("wrong_rotation r2c2 @ 20.0"), _record(_alert("wrong_rotation", "r2c2", 21.5)))
    assert len(s.tp) == 1 and not s.fp and not s.fn
    assert s.delays == [1.5]


def test_a_late_alert_is_both_a_miss_and_a_false_alarm() -> None:
    s = score_run(_labels("wrong_rotation r2c2 @ 20.0"), _record(_alert("wrong_rotation", "r2c2", 45.0)))
    assert not s.tp and len(s.fp) == 1 and len(s.fn) == 1


def test_the_wrong_target_does_not_match() -> None:
    s = score_run(_labels("wrong_rotation r2c2"), _record(_alert("wrong_rotation", "r0c0")))
    assert not s.tp and len(s.fp) == 1 and len(s.fn) == 1


def test_an_alert_matches_at_most_one_label() -> None:
    s = score_run(
        _labels("wrong_replacement @ 5", "wrong_replacement @ 6"),
        _record(_alert("wrong_replacement", "move", 5.5)),
    )
    assert len(s.tp) == 1 and len(s.fn) == 1


def test_warnings_count_only_when_asked() -> None:
    record = _record(_alert("out_of_order", "s3", 4.0, severity="warning"))
    assert not score_run(_labels("out_of_order s3"), record).tp
    assert score_run(_labels("out_of_order s3"), record, warnings=True).tp


def test_a_log_of_another_procedure_is_refused() -> None:
    with pytest.raises(ValueError, match="wrong log"):
        score_run(_labels(), _record(procedure="jenga_game"))


# --- the report ------------------------------------------------------------------


def test_false_alerts_are_averaged_over_correct_runs_only() -> None:
    report = Report([
        score_run(_labels(run="A"), _record(_alert("wrong_part"), _alert("wrong_part"))),
        score_run(_labels(run="B"), _record()),
        score_run(_labels("missed_step r1c2", run="C"), _record(_alert("extra_part"))),
    ])
    assert report.false_alerts_per_correct_run() == 1.0
    tp, fp, fn = report.counts()
    assert (tp, fp, fn) == (0, 3, 1)
    assert report.to_json()["per_code"]["all"]["precision"] == 0.0
    assert "missed       missed_step r1c2" in report.mistakes()


def test_labels_and_logs_pair_by_run_id_and_a_stray_is_an_error(tmp_path: Path) -> None:
    _record(_alert("wrong_part"), run="R1").write(tmp_path / "whatever.json")
    pairs = pair({"R1": _labels(run="R1")}, tmp_path)
    assert pairs[0][0].run == "R1"
    _record().write(tmp_path / "R2.json")
    with pytest.raises(ValueError, match="no label for R2"):
        pair({"R1": _labels(run="R1")}, tmp_path)


def test_every_scenario_scores_perfectly(tmp_path: Path, capsys) -> None:
    out = tmp_path / "scores.json"
    assert main(["--scenarios", str(SCENARIOS), "--json", str(out)]) == 0
    assert "false alerts per correct run: 0.00" in capsys.readouterr().out
    assert '"recall": 1.0' in out.read_text()


def test_labelled_logs_are_scored_from_the_command_line(tmp_path: Path, capsys) -> None:
    (tmp_path / "labels").mkdir()
    _labels("wrong_part @ 3.0", run="R1").write(tmp_path / "labels" / "R1.yaml")
    _record(_alert("wrong_part", t=3.2), run="R1").write(tmp_path / "logs" / "R1.json")
    assert main(["--labels", str(tmp_path / "labels"), "--logs", str(tmp_path / "logs")]) == 0
    assert "median alert delay: 0.20s" in capsys.readouterr().out
