import sys
import json
import numpy as np

sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges, load_solutions

train_ch = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json')
train_sol = load_solutions('data/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json')

def analyze(tid):
    task = train_ch[tid]
    truth = np.array(train_sol[tid][0])
    print(f"\n==================== EMPIRICAL ANALYSIS OF MISS TASK {tid} ====================")
    for idx, pair in enumerate(task.train):
        inp, out = pair.input, pair.output
        print(f"\n--- TRAIN PAIR {idx} ---")
        print("INPUT shape:", inp.shape, "colors:", np.unique(inp))
        print(inp)
        print("\nOUTPUT shape:", out.shape, "colors:", np.unique(out))
        print(out)
        diff = (inp != out)
        print(f"Diff cells count: {diff.sum()} / {inp.size} ({diff.sum()/inp.size*100:.1f}%)")

analyze('070dd51e')
analyze('09c534e7')
