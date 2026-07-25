import sys
import json
import numpy as np

sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges, load_solutions

challenges = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json')
solutions = load_solutions('data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json')

def show_task(tid):
    task = challenges[tid]
    print(f"\n==================== TASK {tid} ====================")
    inp0 = task.train[0].input
    out0 = task.train[0].output
    print("--- TRAIN PAIR 0 INPUT ---")
    print(inp0)
    print("\n--- TRAIN PAIR 0 OUTPUT ---")
    print(out0)

show_task('136b0064')
show_task('13e47133')
show_task('142ca369')
