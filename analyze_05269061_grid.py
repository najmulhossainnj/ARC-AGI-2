import sys
import json
import numpy as np

sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges, load_solutions

train_ch = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json')
train_sol = load_solutions('data/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json')

tid = '05269061'
task = train_ch[tid]
truth = np.array(train_sol[tid][0])

np.set_printoptions(threshold=2000, linewidth=120)

print(f"==================== DEEP DIAGNOSTIC OF TASK {tid} ====================")

for idx, pair in enumerate(task.train):
    inp, out = pair.input, pair.output
    print(f"\n--- TRAIN PAIR {idx} ---")
    print("INPUT shape:", inp.shape, "colors:", np.unique(inp))
    print(inp)
    print("\nOUTPUT shape:", out.shape, "colors:", np.unique(out))
    print(out)
    diff = (inp != out)
    print(f"Diff cells count: {diff.sum()} / {inp.size} ({diff.sum()/inp.size*100:.1f}%)")
    diff_r, diff_c = np.where(diff)
    print(f"Diff locations: rows={diff_r.tolist()}, cols={diff_c.tolist()}")

test_inp = task.test[0].input if hasattr(task.test[0], 'output') else task.test[0]
print("\n--- TEST PAIR ---")
print("TEST INPUT shape:", test_inp.shape, "colors:", np.unique(test_inp))
print(test_inp)
print("\nTEST TRUTH shape:", truth.shape, "colors:", np.unique(truth))
print(truth)
