"""Build the "Real games" page's data from the teammate bundle's labelled sessions.

    .venv/bin/python scripts/make_demo_data.py "jenga-share(1)/jenga-share/dataset" eval/demo

For every labelled pull: the crop-diff finder's scores per layer, the teammate's
answer, the confirmed layer, and before/after pictures of the tapped column from
the camera that saw the change best. Writes `pulls.json` and `crops/*.jpg`.
Block faces only -- no frames with people in them leave the bundle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from thirdeye.perception.layerfinder import _median_face, index_crops, layer_scores, possible

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_layerfinder import truth

PRE, POST = 3.5, (0.8, 4.5)


def median_colour(files, lo, hi):
    got = [cv2.imread(str(p)) for t, p in files if lo <= t <= hi]
    got = [g for g in got if g is not None]
    return np.median(got, axis=0).astype(np.uint8) if got else None


def main() -> int:
    dataset, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "crops").mkdir(parents=True, exist_ok=True)
    pulls, n, ours_ok, theirs_ok = [], 0, 0, 0
    for session in sorted(dataset.glob("live_*")):
        if not (session / "labels.jsonl").exists():
            continue
        idx = index_crops(session)
        prev = None
        removed: set[tuple[int, int]] = set()
        for g in truth(session):
            mark = g["mark_ms"]
            lo = (prev + 1000) if prev else mark - 9000
            hi = min(lo + int(PRE * 1000), mark - 2000)
            alo, ahi = mark + int(POST[0] * 1000), mark + int(POST[1] * 1000)
            prev = mark
            # layer_scores counts from 0, the labels from 1
            score = {k + 1: v for k, v in layer_scores(idx, mark, g["slot"],
                                                         None if lo == mark - 9000 else lo - 1000,
                                                         pre_s=PRE, post_s=POST).items()}
            # as find_layer: no layer where this pull is physically impossible
            fits = [k for k in score if possible(k - 1, g["slot"], removed)]
            ours = max(fits, key=score.get) if fits else None
            removed.add((g["layer"] - 1, g["slot"]))
            # the camera whose picture of our chosen block changed most shows the move best
            best_cam, best = None, -1.0
            for cam in ("android", "iphone"):
                files = idx.get((cam, (ours or 1) - 1, g["slot"]), [])
                a, b = _median_face(files, lo, hi), _median_face(files, alo, ahi)
                if a is not None and b is not None and (d := float(np.abs(a - b).mean())) > best:
                    best_cam, best = cam, d
            column = []
            if best_cam:
                for (cam, layer, slot), files in sorted(idx.items(), key=lambda kv: kv[0][1]):
                    if cam != best_cam or slot != g["slot"]:
                        continue
                    a, b = median_colour(files, lo, hi), median_colour(files, alo, ahi)
                    if a is None or b is None:
                        continue
                    stem = f"{session.name[-6:]}_{g['seq']}_{layer + 1}"
                    cv2.imwrite(str(out / "crops" / f"{stem}_b.jpg"), a, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    cv2.imwrite(str(out / "crops" / f"{stem}_a.jpg"), b, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    column.append({"layer": layer + 1, "before": f"crops/{stem}_b.jpg",
                                   "after": f"crops/{stem}_a.jpg",
                                   "z": round(score.get(layer + 1, 0.0), 2)})
            n += 1
            ours_ok += ours == g["layer"]
            theirs_ok += g["said"] == g["layer"]
            pulls.append({"session": session.name, "seq": g["seq"],
                          "pos": ["left", "middle", "right"][g["slot"]],
                          "truth": g["layer"], "corrected": g["corrected"], "teammate": g["said"],
                          "teammate_confident": g["confident"], "ours": ours, "camera": best_cam,
                          "column": column[::-1]})
    summary = {"pulls": n, "ours_top1": round(ours_ok / max(n, 1), 3),
               "teammate_top1": round(theirs_ok / max(n, 1), 3),
               "sessions": len({p["session"] for p in pulls})}
    (out / "pulls.json").write_text(json.dumps({"summary": summary, "pulls": pulls}))
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
