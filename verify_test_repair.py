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
g = test_inp.copy()
h, w = g.shape
noise_color = 8

mask = (g == noise_color)
rows, cols = np.where(mask)
r1, r2 = rows.min(), rows.max()
c1, c2 = cols.min(), cols.max()

print(f"Noise bbox: rows=[{r1}..{r2}], cols=[{c1}..{c2}] shape=({r2-r1+1}, {c2-c1+1}), Truth shape={truth.shape}")

# Test Transpose map (nr = c, nc = r)
crop = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=int)
for r in range(r1, r2 + 1):
    for c in range(c1, c2 + 1):
        nr = c
        nc = r
        crop[r - r1, c - c1] = g[nr, nc]

match = np.array_equal(crop, truth)
print(f"Transpose map (nr=c, nc=r): shape={crop.shape}, EXACT MATCH WITH TRUTH={match}")

if not match and crop.shape == truth.shape:
    diff = (crop != truth).sum()
    print(f"Diff count: {diff}")
    print("Pred Crop:")
    print(crop)
    print("Truth:")
    print(truth)
