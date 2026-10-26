"""Collapse detection from tower height (FR-STA-8)."""

from __future__ import annotations

from thirdeye.session import Session
from thirdeye.state.jenga.collapse import LAYER_MM, CollapseDetector
from thirdeye.state.jenga.lattice import TowerLattice, full_tower

TALL = 18 * LAYER_MM


def feed(det: CollapseDetector, heights, t0: float = 0.0, dt: float = 0.1) -> list[bool]:
    return [det.feed(t0 + i * dt, h) for i, h in enumerate(heights)]


def test_normal_play_is_not_a_collapse() -> None:
    det = CollapseDetector(TowerLattice())
    # The tower grows a layer, wobbles a few millimetres, never falls.
    heights = [TALL] * 10 + [TALL + LAYER_MM] * 10 + [TALL + LAYER_MM - 3] * 5
    assert not any(feed(det, heights))
    assert not det.collapsed


def test_a_fallen_tower_is_confirmed_and_timestamped_where_it_fell() -> None:
    lattice = TowerLattice()
    det = CollapseDetector(lattice, confirm_frames=3)
    fired = feed(det, [TALL] * 5 + [3 * LAYER_MM] * 5)
    assert fired.index(True) == 7          # confirmed on the third low frame...
    assert det.t_collapse == 0.5           # ...but dated to the first
    assert lattice.collapse_detected


def test_a_one_frame_glitch_recovers_and_is_forgotten() -> None:
    det = CollapseDetector(TowerLattice())
    heights = [TALL] * 5 + [0.0] + [TALL] * 5
    assert not any(feed(det, heights))
    # The glitch never became part of the reference height.
    assert min(det._recent) == TALL


def test_unmeasurable_frames_do_not_break_a_confirmation() -> None:
    det = CollapseDetector(TowerLattice(), confirm_frames=3)
    fired = feed(det, [TALL] * 5 + [LAYER_MM, None, LAYER_MM, None, LAYER_MM])
    assert fired[-1] is True


def test_a_collapse_fires_once() -> None:
    det = CollapseDetector(TowerLattice(), confirm_frames=1)
    assert sum(feed(det, [TALL] * 5 + [0.0] * 5)) == 1


def test_a_detected_collapse_ends_the_session(constraint_spec) -> None:
    session = Session(constraint_spec)
    assert isinstance(session.state, TowerLattice)
    det = CollapseDetector(session.state, confirm_frames=2)
    feed(det, [TALL] * 5 + [0.0] * 2)
    session.state.observe(full_tower(), t=0.6)
    alerts = session.pump()
    assert [a.code for a in alerts] == ["game_over"]
    assert session.outcome == "game_over"


def test_a_hand_over_the_top_and_back_is_not_a_collapse() -> None:
    # game_04 at 59-62 s: an arm above the tower reads four layers too tall for
    # three seconds, then the true height returns. That return is not a fall.
    det = CollapseDetector(TowerLattice())
    heights = [12 * LAYER_MM] * 10 + [16 * LAYER_MM] * 9 + [12 * LAYER_MM] * 10
    assert not any(feed(det, heights))
    # ...and the real fall afterwards is still caught.
    assert any(feed(det, [5 * LAYER_MM] * 5, t0=10.0))


def test_a_rise_that_holds_becomes_the_reference() -> None:
    # A reference that started low (tower half-seen) is replaced once the true
    # height has held long enough, so a later fall from it is measured right.
    det = CollapseDetector(TowerLattice(), rise_frames=10)
    feed(det, [4 * LAYER_MM] * 5 + [TALL] * 12)
    assert min(det._recent) == TALL


def test_a_drop_must_hold_for_confirm_s() -> None:
    # An arm across a single camera hides the tower for seconds, then leaves.
    det = CollapseDetector(TowerLattice(), confirm_s=8.0)
    hidden = feed(det, [TALL] * 5 + [LAYER_MM] * 50 + [TALL] * 5, dt=0.1)
    assert not any(hidden) and not det.pending
    fell = feed(det, [LAYER_MM] * 90, t0=10.0, dt=0.1)
    assert fell.index(True) == 80          # 8 s after the first low frame...
    assert det.t_collapse == 10.0          # ...and dated to it
