import sys
import json
import numpy as np

sys.path.insert(0, "./arc-neurosymbolic-v1")
from arc_solver.utils.arc_io import load_challenges, load_solutions
from arc_solver.dsl.advanced_transforms import symmetry_repair

challenges = load_challenges("data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
solutions = load_solutions("data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")

tid = "0934a4d8"
task = challenges[tid]
inp0 = task.train[0].input
out0 = task.train[0].output

repaired = symmetry_repair(inp0, 8)
print(f"Repaired is None? {repaired is None}")

if repaired is not None:
    mask = (inp0 == 8)
    rows, cols = np.where(mask)
    r1, r2 = rows.min(), rows.max()
    c1, c2 = cols.min(), cols.max()
    crop = repaired[r1:r2+1, c1:c2+1]
    
    print("Crop shape:", crop.shape, "Output shape:", out0.shape)
    print("Crop matches out0?", np.array_equal(crop, out0))
    if crop.shape == out0.shape:
        diff = (crop != out0)
        print("Diff count:", diff.sum())
        print("Crop:")
        print(crop)
        print("Output ground truth:")
        print(out0)
