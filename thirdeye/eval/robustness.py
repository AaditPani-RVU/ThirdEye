"""How much detector noise before the checker starts crying wolf? (NFR-REL-2, EV-ERR-3)

    python -m thirdeye.eval.robustness                 # every placement scenario, 5 seeds
    python -m thirdeye.eval.robustness --seeds 20 -v

Replays every placement scenario (LEGO and jigsaw) under rising levels of
simulated detector noise and scores each against its own `expect:` block. The
answer is the level at which false alerts and missed errors begin -- the margin
the stability filter and occlusion freeze buy, measured before a real detector
exists, so a real detector's measured flicker rate can be read straight off it.

Jenga scenarios are left out: the lattice judges a settled reading, and its
failure modes are perception's, measured on recordings (EV-STA-3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from thirdeye.eval.errors import Report, score_run
from thirdeye.eval.labels import RunLabels, parse_error
from thirdeye.replay import Noise, Replay
from thirdeye.state.lego.build_state import BuildState

SCENARIOS = Path(__file__).resolve().parents[2] / "scripts" / "scenarios"

#: name -> per-frame probabilities (drop, ghost, hands, skip). "real" is a guess at
#: a decent detector at 10 FPS; the others bracket it.
LEVELS: dict[str, tuple[float, float, float, float]] = {
    "none": (0.0, 0.0, 0.0, 0.0),
    "mild": (0.05, 0.02, 0.01, 0.02),
    "real": (0.10, 0.05, 0.02, 0.05),
    "rough": (0.20, 0.10, 0.05, 0.10),
    "awful": (0.35, 0.25, 0.10, 0.20),
}


def placement_scenarios(folder: Path = SCENARIOS) -> list[Path]:
    return [p for p in sorted(folder.glob("*.yaml"))
            if isinstance(Replay.load(p).session.state, BuildState)]


def sweep(paths: list[Path], seeds: int, levels: dict = LEVELS) -> dict[str, Report]:
    out = {}
    for name, (drop, ghost, hands, skip) in levels.items():
        runs = []
        for path in paths:
            expect = [parse_error(e) for e in yaml.safe_load(path.read_text()).get("expect") or []]
            for seed in range(seeds if name != "none" else 1):
                noise = Noise(drop=drop, ghost=ghost, hands=hands, skip=skip, seed=seed)
                record = Replay.load(path, noise=noise).run()
                labels = RunLabels(run=f"{path.stem}#{seed}", procedure=record.procedure,
                                   split="scripted", errors=expect)
                runs.append(score_run(labels, record))
        out[name] = Report(runs)
    return out


def table(reports: dict[str, Report]) -> str:
    lines = [(f"{'noise':<8}{'drop':>6}{'ghost':>7}{'hands':>7}{'skip':>6}"
              f"{'recall':>9}{'precision':>11}{'false/ok run':>14}")]
    for name, report in reports.items():
        tp, fp, fn = report.counts()
        d, g, h, s = LEVELS.get(name, (0, 0, 0, 0))
        fa = report.false_alerts_per_correct_run()
        lines.append(
            f"{name:<8}{d:>6.2f}{g:>7.2f}{h:>7.2f}{s:>6.2f}"
            f"{tp / (tp + fn) if tp + fn else 1:>9.2f}{tp / (tp + fp) if tp + fp else 1:>11.2f}"
            f"{'n/a' if fa is None else f'{fa:.2f}':>14}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scenarios", type=Path, default=SCENARIOS)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("-v", "--verbose", action="store_true", help="list what went wrong")
    args = ap.parse_args(argv)

    reports = sweep(placement_scenarios(args.scenarios), args.seeds)
    print(table(reports))
    if args.verbose:
        for name, report in reports.items():
            if text := report.mistakes():
                print(f"\n[{name}]\n{text}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
