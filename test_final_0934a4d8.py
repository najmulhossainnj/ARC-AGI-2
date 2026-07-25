import sys
import json
import time
import numpy as np

sys.path.insert(0, "./arc-neurosymbolic-v1")
from arc_solver.utils.arc_io import load_challenges, load_solutions

challenges = load_challenges("data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
solutions = load_solutions("data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")

tid = "0934a4d8"
task = challenges[tid]
truth = solutions[tid][0]
if not isinstance(truth, np.ndarray):
    truth = np.array(truth)

def _symmetry_maps(grid, noise_color):
    g = np.array(grid)
    h, w = g.shape
    maps = [
        lambda r, c: (r, w - 1 - c),          # mirror horizontal
        lambda r, c: (h - 1 - r, c),          # mirror vertical
        lambda r, c: (h - 1 - r, w - 1 - c),  # 180 rotation
    ]
    if h == w:
        maps.append(lambda r, c: (c, r))                  # transpose
        maps.append(lambda r, c: (w - 1 - c, h - 1 - r))   # anti-transpose

    # Add offset 180-rotation centers around grid center
    for dr in range(-4, 5):
        for dc in range(-4, 5):
            cr2 = (h - 1) + dr
            cc2 = (w - 1) + dc
            maps.append(lambda r, c, cr2=cr2, cc2=cc2: (cr2 - r, cc2 - c))

    return maps

def symmetry_repair_crop_v2(grid, noise_color):
    g = np.array(grid).copy()
    h, w = g.shape
    mask = (g == noise_color)
    if not mask.any():
        return None

    rows, cols = np.where(mask)
    r1, r2 = rows.min(), rows.max()
    c1, c2 = cols.min(), cols.max()

    maps = _symmetry_maps(g, noise_color)
    valid_candidates = []

    for m in maps:
        # Check 1: Can map m fill ALL noise cells with valid non-noise values?
        can_fill_all = True
        for r, c in zip(rows, cols):
            try:
                nr, nc = m(r, c)
            except Exception:
                can_fill_all = False; break
            if not (0 <= nr < h and 0 <= nc < w) or g[nr, nc] == noise_color:
                can_fill_all = False; break

        if not can_fill_all:
            continue

        # Check 2: Mismatches on non-noise cells
        checked = 0
        mismatches = 0
        for r in range(h):
            for c in range(w):
                if g[r, c] == noise_color:
                    continue
                try:
                    nr, nc = m(r, c)
                except Exception:
                    continue
                if not (0 <= nr < h and 0 <= nc < w) or g[nr, nc] == noise_color:
                    continue
                checked += 1
                if g[nr, nc] != g[r, c]:
                    mismatches += 1

        valid_candidates.append((mismatches, checked, m))

    if not valid_candidates:
        return None

    # Sort by lowest mismatches first
    valid_candidates.sort(key=lambda x: x[0])
    best_m = valid_candidates[0][2]

    # Apply repair
    for r, c in zip(rows, cols):
        nr, nc = best_m(r, c)
        g[r, c] = g[nr, nc]

    return g[r1:r2+1, c1:c2+1]

print(f"=== Testing symmetry_repair_crop_v2 for {tid} ===")
all_train_ok = True
for idx, pair in enumerate(task.train):
    res = symmetry_repair_crop_v2(pair.input, 8)
    match = res is not None and res.shape == pair.output.shape and np.array_equal(res, pair.output)
    print(f"Train Pair {idx}: EXACT MATCH={match}")
    if not match:
        all_train_ok = False

test_inp = task.test[0].input if hasattr(task.test[0], 'input') else task.test[0]
test_res = symmetry_repair_crop_v2(test_inp, 8)
test_match = test_res is not None and test_res.shape == truth.shape and np.array_equal(test_res, truth)

print(f"\nTest Pair: EXACT MATCH WITH GROUND TRUTH={test_match}")
print(f"OVERALL TASK SOLVE STATUS: {'OK (PASSED 100%!)' if (all_train_ok and test_match) else 'MISS'}")
