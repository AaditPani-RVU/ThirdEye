"""Which layer was the block pulled from? Score two answers against the player's
confirmed layers, on the labelled `live_*` sessions of the teammate bundle.

    .venv/bin/python scripts/eval_layerfinder.py <dataset-dir> [--pre 3.5 --post 0.8:4.5]

Ground truth per tap: the typed `correction` if the player made one, else the
layer the finder reported (the player saw it and let it stand -- so this is
optimistic for the teammate's finder, which is why its row is marked "shown").

  teammate   the finder's own answer, as logged in labels.jsonl
  crop-diff  ours, `thirdeye.perception.layerfinder`: the tap names the slot;
             the layer whose end face changed most across the tap is the pull
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from thirdeye.perception.layerfinder import SLOT, find_layer, index_crops


def truth(session: Path) -> list[dict]:
    """[{seq, mark_ms, slot, said, layer, confident}] for this session."""
    said, fixed = {}, {}
    for line in (session / "labels.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "correction" in r:
            fixed[r["seq"]] = r["correction"]
        else:
            said[r["seq"]] = r
    out = []
    for seq, r in sorted(said.items()):
        res = r["result"]
        if not res.get("ok"):
            continue
        out.append({"seq": seq, "mark_ms": r["mark_ms"], "slot": SLOT[r["pos"]],
                    "said": res["layer"], "layer": fixed.get(seq, res["layer"]),
                    "corrected": seq in fixed, "confident": res["confident"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--pre", type=float, default=3.5, help="seconds of 'before' after the previous tap")
    ap.add_argument("--post", default="0.8:4.5")
    ap.add_argument("--no-physical", dest="physical", action="store_false",
                    help="don't drop pulls the tower could not survive, given the earlier confirmed ones")
    a = ap.parse_args()
    post = tuple(float(x) for x in a.post.split(":"))
    tot = {"n": 0, "teammate": 0, "ours": 0, "ours_top2": 0, "corrected": 0, "conf": 0, "conf_ok": 0}
    for session in sorted(a.dataset.glob("live_*")):
        if not (session / "labels.jsonl").exists():
            continue
        gt = truth(session)
        idx = index_crops(session)
        hit = ours = 0
        prev = None
        removed: set[tuple[int, int]] = set()
        for g in gt:
            # The finder counts layers from 0; the labels count from 1.
            rank = [x + 1 for x in find_layer(idx, g["mark_ms"], g["slot"], prev,
                                              removed if a.physical else None,
                                              pre_s=a.pre, post_s=post)]
            prev = g["mark_ms"]
            removed.add((g["layer"] - 1, g["slot"]))
            g["ours"] = rank[0] if rank else None
            ours += g["ours"] == g["layer"]
            tot["ours_top2"] += g["layer"] in rank[:2]
            hit += g["said"] == g["layer"]
            tot["corrected"] += g["corrected"]
            tot["conf"] += g["confident"]
            tot["conf_ok"] += g["confident"] and g["said"] == g["layer"]
        tot["n"] += len(gt)
        tot["teammate"] += hit
        tot["ours"] += ours
        print(f"{session.name}: n={len(gt)} teammate {hit}/{len(gt)} ours {ours}/{len(gt)}  "
              + " ".join(f"{g['layer']}{'*' if g['corrected'] else ''}:{g['said']}/{g['ours']}" for g in gt))
    n = max(tot["n"], 1)
    print(f"\nall: n={tot['n']}  teammate {tot['teammate']/n:.0%} (shown, optimistic)  "
          f"crop-diff top1 {tot['ours']/n:.0%} top2 {tot['ours_top2']/n:.0%}  "
          f"corrected {tot['corrected']}  confident {tot['conf']} of which right {tot['conf_ok']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
