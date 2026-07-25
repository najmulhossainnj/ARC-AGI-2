import sys
import json
import numpy as np

sys.path.insert(0, "./arc-neurosymbolic-v1")
from arc_solver.utils.arc_io import load_challenges, load_solutions

challenges = load_challenges("data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
solutions = load_solutions("data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")

tid = "0934a4d8"
task = challenges[tid]

for idx, pair in enumerate(task.train):
    inp = pair.input
    out = pair.output
    
    # Bounding box of noise color (8 for pair 0)
    mask = (inp == 8)
    if not mask.any():
        continue
    rows, cols = np.where(mask)
    r1, r2 = rows.min(), rows.max()
    c1, c2 = cols.min(), cols.max()
    
    # 180 rotation around panel center (31 - r, 29 - c)
    repaired_crop = np.zeros((r2 - r1 + 1, c2 - c1 + 1), dtype=int)
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            nr = 31 - r
            nc = 29 - c
            repaired_crop[r - r1, c - c1] = inp[nr, nc]
            
    match = np.array_equal(repaired_crop, out)
    print(f"Train Pair {idx}: noise crop shape={repaired_crop.shape}, out shape={out.shape}, EXACT MATCH={match}")
    if not match and repaired_crop.shape == out.shape:
        print("  Diff count:", (repaired_crop != out).sum())
