import sys
import json
import numpy as np

sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges, load_solutions
from arc_solver.synthesis.classifier import classify_task

challenges = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json')
solutions = load_solutions('data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json')

target_ids = ['135a2760', '136b0064', '13e47133', '142ca369']

np.set_printoptions(threshold=1000, linewidth=120)

for tid in target_ids:
    if tid not in challenges:
        continue
    task = challenges[tid]
    train_pairs = [(p.input, p.output) for p in task.train]
    cat = classify_task(train_pairs)
    print("="*60)
    print(f"TASK: {tid} | CLASSIFIED CATEGORY: {cat}")
    print("="*60)
    for idx, (inp, out) in enumerate(train_pairs[:2]):
        print(f"\n--- Train Pair {idx} ---")
        print(f"Input shape: {inp.shape}, Unique colors: {np.unique(inp)}")
        print(f"Output shape: {out.shape}, Unique colors: {np.unique(out)}")
        if inp.shape == out.shape and inp.size <= 400:
            diff = (inp != out)
            print(f"Diff cells count: {diff.sum()} / {inp.size} ({diff.sum()/inp.size*100:.1f}%)")
