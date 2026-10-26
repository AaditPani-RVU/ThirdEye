"""Log pull times while a game is recorded, for `scripts/eval_framedir.py --marks`.

    .venv/bin/python scripts/mark_pulls.py pulls.json

Press Enter at the moment you start the Record3D recordings: that is t = 0.
Then press Enter each time a block is fully out (type `u` + Enter to undo the
last one, `q` + Enter to finish). Writes a JSON list of seconds from t = 0,
saved after every tap so a crash loses nothing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    out = Path(args[0])
    input("Enter = recordings started (t = 0) ")
    t0, marks = time.monotonic(), []
    while (line := input(f"[{len(marks)} pulls] Enter = pulled, u = undo, q = done: ").strip()) != "q":
        if line == "u":
            if marks:
                print(f"  undone {marks.pop():.1f} s")
        else:
            marks.append(round(time.monotonic() - t0, 2))
            print(f"  pull {len(marks)} at {marks[-1]:.1f} s")
        out.write_text(json.dumps(marks))
    out.write_text(json.dumps(marks))
    print(f"{len(marks)} pulls -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
