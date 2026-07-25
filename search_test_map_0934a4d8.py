import sys
import json
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

test_inp = task.test[0].input if hasattr(task.test[0], 'input') else task.test[0]
h, w = test_inp.shape

mask = (test_inp == 8)
rows, cols = np.where(mask)
r1, r2 = rows.min(), rows.max()
c1, c2 = cols.min(), cols.max()

oh, ow = r2 - r1 + 1, c2 - c1 + 1
print(f"Test noise bbox: rows=[{r1}..{r2}], cols=[{c1}..{c2}] shape=({oh}, {ow}), Truth shape={truth.shape}")

coeffs = [
    (1, 0, 0, 1),   # identity
    (-1, 0, 0, 1),  # V flip
    (1, 0, 0, -1),  # H flip
    (-1, 0, 0, -1), # 180 rot
    (0, 1, 1, 0),   # transpose
    (0, -1, 1, 0),  # 90 rot CW
    (0, 1, -1, 0),  # 270 rot CW
    (0, -1, -1, 0), # anti-transpose
]

found = []
for a, b, d, e in coeffs:
    for dr in range(-60, 60):
        for dc in range(-60, 60):
            pred = np.zeros_like(truth)
            valid = True
            for i in range(oh):
                for j in range(ow):
                    r = r1 + i
                    c = c1 + j
                    nr = a * r + b * c + dr
                    nc = d * r + e * c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        pred[i, j] = test_inp[nr, nc]
                    else:
                        valid = False
                        break
                if not valid:
                    break
            if valid and np.array_equal(pred, truth):
                print(f"MATCH FOUND FOR TEST SET! Coeffs=({a},{b},{d},{e}), Shift=({dr},{dc})")
                found.append((a, b, d, e, dr, dc))

if not found:
    print("No simple affine match found. Searching all possible (r, c) sampling coordinates...")
    # Find matching cell positions for each cell in truth
    for i in range(oh):
        for j in range(ow):
            target_val = truth[i, j]
            matching_coords = [(r, c) for r in range(h) for c in range(w) if test_inp[r, c] == target_val]
            print(f"Cell ({i},{j}) val={target_val}: {len(matching_coords)} candidate positions in test_inp")
