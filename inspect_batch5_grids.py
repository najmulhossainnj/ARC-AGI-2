import sys
import json
import numpy as np

sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges

train_ch = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json')

def show(tid):
    task = train_ch[tid]
    print(f"\n=== TASK {tid} ===")
    inp0 = task.train[0].input
    out0 = task.train[0].output
    print("INP 0 shape:", inp0.shape, "colors:", np.unique(inp0))
    print(inp0)
    print("OUT 0 shape:", out0.shape, "colors:", np.unique(out0))
    print(out0)

show('0c786b71')
show('0b148d64')
