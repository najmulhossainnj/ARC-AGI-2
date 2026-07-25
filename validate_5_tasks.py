import sys
import time
import json
import numpy as np

sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges, load_solutions
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

def main():
    challenges_path = 'data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json'
    solutions_path = 'data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json'
    
    print("Loading evaluation dataset...")
    challenges = load_challenges(challenges_path)
    solutions = load_solutions(solutions_path)

    target_ids = ['0934a4d8', '135a2760', '136b0064', '13e47133', '142ca369']
    
    print("\n" + "="*70)
    print("=== LOCAL BENCHMARK ON 5 EVALUATION PUZZLES ===")
    print("="*70)

    solver = NeuroSymbolicARCSolver(beam_width=20, max_depth=2, ranker=None)
    
    solved_count = 0
    total_time = 0.0

    for tid in target_ids:
        if tid not in challenges:
            print(f"[{tid}] NOT FOUND in challenges dataset!")
            continue

        task = challenges[tid]
        train_pairs = [(p.input, p.output) for p in task.train]
        
        t0 = time.time()
        preds, programs = solver.solve_task(train_pairs, task.test)
        t1 = time.time()
        dt = t1 - t0
        total_time += dt

        truth = solutions[tid]
        
        # Check if attempt 1 or attempt 2 matches truth
        ok = False
        if preds and len(preds) > 0:
            for attempt_idx, attempt in enumerate(preds[0]):
                if np.array_equal(np.array(attempt), np.array(truth[0])):
                    ok = True
                    break
        
        status = "OK (SOLVED!)" if ok else "MISS"
        if ok:
            solved_count += 1

        print(f"\nPuzzle [{tid}]: {status} in {dt:.2f}s")
        if programs:
            print("  Top Discovered Program(s):")
            for idx, prog in enumerate(programs[:2], 1):
                print(f"    [{idx}] {prog.program} (error: {prog.error:.4f}, score: {prog.score:.4f})")
        else:
            print("  No program candidate discovered (used fallback)")

    print("\n" + "="*70)
    print(f"SUMMARY: Solved {solved_count}/{len(target_ids)} puzzles in {total_time:.2f} seconds total!")
    print("="*70)

if __name__ == '__main__':
    main()
