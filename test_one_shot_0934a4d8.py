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

test_inp = task.test[0].input if hasattr(task.test[0], 'input') else task.test[0]

h, w = test_inp.shape
mask = (test_inp == 8)
rows, cols = np.where(mask)
r1, r2 = rows.min(), rows.max()
c1, c2 = cols.min(), cols.max()

print(f"Test Input Color 8 Bounding Box: rows=[{r1}..{r2}], cols=[{c1}..{c2}] shape=({r2-r1+1}, {c2-c1+1})")

# Test standard 180 rotation (H-1-r, W-1-c)
pred_crop = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=int)
for r in range(r1, r2 + 1):
    for c in range(c1, c2 + 1):
        nr = (h - 1) - r
        nc = (w - 1) - c
        pred_crop[r - r1, c - c1] = test_inp[nr, nc]

match = np.array_equal(pred_crop, truth)
print(f"Standard 180 rotation (H-1-r, W-1-c): shape={pred_crop.shape}, matches_truth={match}")

if not match and pred_crop.shape == truth.shape:
    diff = (pred_crop != truth).sum()
    print(f"Diff count: {diff}")
    print("Pred:")
    print(pred_crop)
    print("Truth:")
    print(truth)
