import json
import time
import sys
import os

# Add codebase to path
sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges, load_solutions
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

def main():
    challenges_path = 'data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json'
    solutions_path = 'data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json'
    
    print(f"Loading challenges from {challenges_path}...")
    challenges = load_challenges(challenges_path)
    solutions = load_solutions(solutions_path)

    tid = '0934a4d8'
    if tid not in challenges:
        print(f"Task {tid} not found in challenges!")
        return

    task = challenges[tid]
    print(f"\n--- Fast Pipeline Test (ranker=None) for Puzzle {tid} on Remote Colab VM ---")
    print(f"Train pairs count: {len(task.train)}, Test pairs count: {len(task.test)}")
    
    # Disable neural ranker overhead for fast symbolic resolution
    solver = NeuroSymbolicARCSolver(beam_width=50, max_depth=3, ranker=None)

    t0 = time.time()
    train_pairs = [(p.input, p.output) for p in task.train]
    preds, programs = solver.solve_task(train_pairs, task.test)
    t1 = time.time()

    truth = solutions[tid]
    ok = all(truth[i] in preds[i] for i in range(len(truth)))
    status = 'OK (SOLVED!)' if ok else 'MISS'
    
    print(f"\nBENCHMARK RESULTS FOR {tid}:")
    print(f"  Solve Status: {status}")
    print(f"  Execution Time: {t1-t0:.4f} seconds")
    if programs:
        print("  Discovered Programs:")
        for idx, prog in enumerate(programs[:3], 1):
            print(f"    [{idx}] {prog.program} (error: {prog.error:.4f}, score: {prog.score:.4f})")

if __name__ == '__main__':
    main()
