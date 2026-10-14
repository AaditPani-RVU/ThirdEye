"""Latency pooling over session logs."""

from __future__ import annotations

from pathlib import Path

from thirdeye.eval.latency import pool, table
from thirdeye.perception.pipeline import Timings
from thirdeye.replay import Replay

SCENARIOS = Path(__file__).resolve().parents[1] / "scripts" / "scenarios"


def _timed(frame_ms: float, fps: float):
    record = Replay.load(SCENARIOS / "jigsaw_correct.yaml").run()
    t = Timings()
    for _ in range(20):
        t.add("frame", frame_ms / 1000)
        t.add("match", frame_ms / 2000)
    record.meta.update(latency=t.summary(), fps=fps)
    return record


def test_sessions_pool_against_the_budget() -> None:
    p = pool([_timed(20, 25), _timed(30, 18), Replay.load(SCENARIOS / "jigsaw_correct.yaml").run()])
    assert p["sessions"] == 3 and p["timed"] == 2
    assert p["stages"]["frame"]["median_ms"] == 25.0 and p["stages"]["frame"]["frames"] == 40
    assert p["fps_min"] == 18 and p["meets_budget"] is True
    assert "budget: met" in table(p)
    assert pool([_timed(60, 10)])["meets_budget"] is False
