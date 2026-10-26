"""The colour-only crop-difference layer finder."""

from __future__ import annotations

import cv2
import numpy as np

from thirdeye.perception.layerfinder import find_layer, index_crops

TAP = 20_000


def face(seed: int) -> np.ndarray:
    """A block end face: wood-ish noise, the same every frame for one seed."""
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(90, 200, (48, 48), dtype=np.uint8), (5, 5), 0)


def write_session(root, *pulled: tuple[int, int], cams=("iphone", "android"), layers=6):
    """Crops every 500 ms for every slot; pulled ones turn to a dark hole at TAP."""
    for cam in cams:
        (root / cam).mkdir(parents=True)
        for ms in range(10_000, 26_000, 500):
            for layer in range(layers):
                for slot in range(3):
                    img = face(100 * layer + slot)
                    if (layer, slot) in pulled and ms > TAP:
                        img = np.full_like(img, 25)
                    noise = np.random.default_rng(ms + layer).normal(0, 3, img.shape)
                    img = np.clip(img + noise, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(root / cam / f"{ms}_L{layer}_S{slot}_p.jpg"), img)
            # side faces are not end faces and are ignored
            cv2.imwrite(str(root / cam / f"{ms}_L0_S0_p_side.jpg"), face(0))


def test_index_keeps_end_faces_only(tmp_path):
    write_session(tmp_path, (2, 1), cams=("iphone",), layers=2)
    idx = index_crops(tmp_path)
    assert set(idx) == {("iphone", layer, slot) for layer in range(2) for slot in range(3)}
    stamps = [t for t, _ in idx[("iphone", 0, 0)]]
    assert stamps == sorted(stamps) and len(stamps) == 32


def test_the_layer_whose_face_changed_is_found(tmp_path):
    write_session(tmp_path, (3, 1))
    rank = find_layer(index_crops(tmp_path), TAP, slot=1, prev_ms=11_000)
    assert rank[0] == 3          # counted from 0, as the crop names are
    assert sorted(rank) == list(range(6))


def test_the_tap_slot_limits_the_search(tmp_path):
    # Two faces change at once, in different slots: each slot's tap finds its own.
    write_session(tmp_path, (3, 1), (1, 0))
    idx = index_crops(tmp_path)
    assert find_layer(idx, TAP, slot=1, prev_ms=11_000)[0] == 3
    assert find_layer(idx, TAP, slot=0, prev_ms=11_000)[0] == 1


def test_no_crops_around_the_tap_gives_no_answer(tmp_path):
    write_session(tmp_path, (3, 1), cams=("iphone",), layers=2)
    assert find_layer(index_crops(tmp_path), 90_000, slot=1) == []
