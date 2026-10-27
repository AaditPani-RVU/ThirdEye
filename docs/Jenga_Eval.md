# Jenga on real recordings: what we measured, and the next recording

*October 27, 2026.* Our Jenga pipeline was first tested on real phone data using the
teammate bundle (`jenga-share(1)/`, git-ignored, about 40,000 files: do not commit it). Their
capture code lives in `github.com/switchtwitch12345/third-eye-testing`.

## What the bundle can test

| Recordings | What they hold | Can test |
| --- | --- | --- |
| `game_01_far`, `game_02_moving`, `game_04` (Oct 15) | Raw iPhone colour + LiDAR depth per frame, with pull taps (`*_mark.json`); `game_01_far` has no taps | Our depth pipeline, one camera |
| `dataset/live_*` (Oct 20) | Cropped block faces from both phones, with the teammate finder's answer per tap and the player's typed corrections (`labels.jsonl`, 6 sessions, 49 pulls) | Which-layer finders, colour only |

The bundle has no two-camera depth recording. One is needed; see the last section.

## Depth pipeline, one iPhone (Oct 15 games)

```bash
.venv/bin/python scripts/eval_framedir.py "jenga-share(1)/jenga-share/game_04" \
    --refine --relative --repose --alone 18 --protrude 20
```

| Recording | Pulls found | Removals reported | Collapse |
| --- | --- | --- | --- |
| `game_04` | 3 of 10 | 4 (75 % precision) | caught at 186.7 s |
| `game_02_moving` | 0–1 of 9 | 2–3 | the recording ends 1 s after the fall, before the 8 s hold |
| `game_01_far` | no taps | 4 | none (correct) |

The plain pipeline found 0 of 10 on `game_04`. Each fix below is an option, off by default,
and has its own flag on `thirdeye.live_jenga`:

- **`refine`:** the marker pose was about 35 mm off in depth, so every slot read as a hole.
  The pose is now corrected against the floor and the tower footprint.
- **`relative`:** each slot is read against the first frame. Slots more than 40 mm deep in
  that frame (`base_max_mm`) are air above the tower. They used to rebase to 0 mm, read as
  blocks, and keep the tower 17 layers tall forever.
- **`repose`:** the cameras get bumped (`game_04` at 30 s and 180 s, `game_01_far` at 75 s
  and 105 s, `game_02_moving` at 15 s). The camera is followed once its marker pose moves
  more than 8 mm or 1.5° for 3 frames in a row.
- **Always on:** a block stuck out for a while counts as a block, not as "moving" forever,
  and no longer stalls the tower.
- **Always on, collapse:** a drop must hold for 8 s (an arm across one camera hid the tower
  for 5.7 s). A jump *up* is kept out of the reference (a hand over the top). Readings are
  not used while a drop is unconfirmed.

**Limit: one camera.** From one corner, a pulled block shows only as a 15–35 mm dent in the
depth, close to the noise. The pipeline was designed for two diagonal corners, and this data
cannot measure that.

## Which layer: colour-only finder (Oct 20 sessions)

```bash
.venv/bin/python scripts/eval_layerfinder.py "jenga-share(1)/jenga-share/dataset"
```

| Finder | Exact layer | Top 2 |
| --- | --- | --- |
| Teammate's (`move_infer.py`) | 82 %, optimistic: uncorrected answers count as right | – |
| Ours (`thirdeye/perception/layerfinder.py`) | 78 % | 92 % |
| Ours, without the earlier pulls (`--no-physical`) | 73 % | 86 % |

The tap gives the slot (left/middle/right). Our finder picks the layer whose end face changed
most between a calm stretch after the previous tap and the seconds after this one. Layers
where the tapped slot is already empty, or holds the layer's last block, are ruled out from the
pulls confirmed earlier in the game (Oct 28: +5 points exact, +6 top 2). A pull that leaves one
edge block is *not* ruled out: one game did that near the top and the tower stood. A "the face
went dark" cue was also tried and lost 3–14 points, since a hole shows whatever is behind it. The
teammate's "confident" flag is right 32 times out of 35. Layers count from 0 in the crops and
from 1 in their labels.

## The next recording: two corners, our setup

This is the measurement still missing.

1. Lay the mat: ArUco `DICT_4X4_50` markers 0–3, each **40 mm** printed. The teammate's
   `aruco_markers_A4.png` is this set, and the Oct 15 mat matched our layout. (Don't use
   `scripts/make_markers.py`: it sizes markers for the LEGO plate.) With the tower centre at
   the origin, put the marker centres 82.5 mm out on each axis (`tower_markers`): 0 at
   (−, −), 1 at (+, −), 2 at (+, +), 3 at (−, +). Each marker's top edge faces +y, toward 2
   and 3. `--refine` absorbs a few mm of error. Build a fresh 18-layer tower in the middle.
2. Place two iPhones on stands at **diagonally opposite corners**, about 50–70 cm away. Each
   one must see at least two markers and the whole tower. Don't touch them during the game
   (`--repose` copes if you do).
3. Start Record3D on both phones, and at the same moment press Enter in
   `.venv/bin/python scripts/mark_pulls.py data/pulls_R001.json`.
4. Play. Press Enter each time a block is **fully out** (`u` undoes, `q` finishes). Play until
   the tower falls, then leave it on the floor for 10 s.
5. Export both recordings as `.r3d` and score them:

```bash
.venv/bin/python scripts/eval_framedir.py --r3d A.r3d --r3d B.r3d \
    --marks data/pulls_R001.json --refine --relative --repose
```

The two recordings are replayed frame by frame together, so start them within a fraction of a
second of each other. Two or three games of about 15 pulls each are enough to tell whether two
corners fix what one could not. Then run the same games through `live_jenga` with `--log` for
the full rule check.
