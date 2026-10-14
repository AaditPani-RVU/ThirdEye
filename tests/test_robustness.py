"""Scenarios under simulated detector noise (NFR-REL-2, EV-ERR-3).

These pin the margin the stability filter and the re-read paths buy. A change
that makes the checker jumpier under noise fails here before it fails a demo.
"""

from __future__ import annotations

import pytest

from thirdeye.eval.robustness import LEVELS, placement_scenarios, sweep
from thirdeye.replay import Noise, Replay, ScriptError


@pytest.fixture(scope="module")
def reports():
    levels = {k: LEVELS[k] for k in ("none", "mild", "real")}
    return sweep(placement_scenarios(), seeds=5, levels=levels)


def test_noise_parses_and_refuses_what_it_does_not_know() -> None:
    assert Noise.parse("drop=0.1, ghost=0.05,seed=3") == Noise(drop=0.1, ghost=0.05, seed=3)
    assert not Noise.parse("")
    with pytest.raises(ScriptError):
        Noise.parse("blur=0.3")


def test_the_same_seed_films_the_same_session() -> None:
    path = next(p for p in placement_scenarios() if p.stem == "model_a_correct")
    noise = Noise(drop=0.2, ghost=0.1, seed=7)
    assert Replay.load(path, noise=noise).run() == Replay.load(path, noise=noise).run()


def test_without_noise_every_scenario_is_perfect(reports) -> None:
    assert reports["none"].counts() == (reports["none"].counts()[0], 0, 0)


@pytest.mark.parametrize("level", ["mild", "real"])
def test_realistic_noise_raises_no_false_alarm_on_a_correct_run(reports, level) -> None:
    assert reports[level].false_alerts_per_correct_run() == 0.0


def test_realistic_noise_keeps_precision_and_most_recall(reports) -> None:
    tp, fp, fn = reports["real"].counts()
    assert tp / (tp + fp) >= 0.95
    assert tp / (tp + fn) >= 0.9
