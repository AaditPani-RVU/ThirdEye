"""Is it usable live? Per-stage latency and FPS over recorded sessions (NFR-PERF-*, EV-LAT-1).

    python -m thirdeye.eval.latency data/logs          # every session log in the folder

Every live session (`thirdeye.live`, `thirdeye.live_jenga`, the server's camera
path, the edge) writes its per-stage timings and its frame rate into the log's
`meta`. This pools them: for each stage, the median of the sessions' medians
and the worst p95, against the budget -- 15 FPS or better, and a whole frame
(`frame`: perception, tracker and checker together) in 40 ms or less.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

from thirdeye.sessionlog import SessionRecord

FPS_TARGET = 15.0
FRAME_MS_TARGET = 40.0


def pool(records: list[SessionRecord]) -> dict[str, Any]:
    stages: dict[str, list[dict[str, float]]] = {}
    fps = []
    for r in records:
        for stage, s in (r.meta.get("latency") or {}).items():
            stages.setdefault(stage, []).append(s)
        if r.meta.get("fps"):
            fps.append(float(r.meta["fps"]))
    out: dict[str, Any] = {
        "sessions": len(records),
        "timed": sum(bool(r.meta.get("latency")) for r in records),
        "fps_median": round(median(fps), 2) if fps else None,
        "fps_min": round(min(fps), 2) if fps else None,
        "stages": {stage: {"median_ms": round(median(x["median_ms"] for x in xs), 2),
                           "p95_ms": round(max(x["p95_ms"] for x in xs), 2),
                           "frames": int(sum(x["n"] for x in xs))}
                   for stage, xs in sorted(stages.items())},
    }
    frame = out["stages"].get("frame")  # type: ignore[union-attr]
    out["meets_budget"] = (None if frame is None or not fps else
                           median(fps) >= FPS_TARGET and frame["median_ms"] <= FRAME_MS_TARGET)
    return out


def table(p: dict[str, Any]) -> str:
    rows = [f"{p['timed']} of {p['sessions']} session(s) carry timings",
            f"{'stage':<10} {'median ms':>10} {'worst p95':>10} {'frames':>8}"]
    rows += [f"{k:<10} {v['median_ms']:>10.1f} {v['p95_ms']:>10.1f} {v['frames']:>8}"
             for k, v in p["stages"].items()]
    if p["fps_median"] is not None:
        rows.append(f"FPS: median {p['fps_median']}, worst {p['fps_min']} "
                    f"(target >= {FPS_TARGET:g}; whole frame <= {FRAME_MS_TARGET:g} ms)")
    if p["meets_budget"] is not None:
        rows.append("budget: " + ("met" if p["meets_budget"] else "NOT met"))
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", type=Path, help="folder of session logs")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    records = [SessionRecord.read(p) for p in sorted(args.logs.glob("*.json"))]
    p = pool(records)
    print(table(p))
    if args.json:
        args.json.write_text(json.dumps(p, indent=1) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
