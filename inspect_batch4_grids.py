import sys
import json
import numpy as np

sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges

train_ch = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json')
eval_ch = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json')

def show(tid, ch_dict):
    task = ch_dict[tid]
    print(f"\n=== TASK {tid} ===")
    inp0 = task.train[0].input
    out0 = task.train[0].output
    print("INP 0 shape:", inp0.shape, "colors:", np.unique(inp0))
    print(inp0)
    print("OUT 0 shape:", out0.shape, "colors:", np.unique(out0))
    print(out0)

show('0692e18c', train_ch)
show('13e47133', eval_ch)
