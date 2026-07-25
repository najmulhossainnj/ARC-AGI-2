import sys
import time
import json
import numpy as np

sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges, load_solutions
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

def run_batch_next_10():
    train_ch = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json')
    train_sol = load_solutions('data/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json')
    eval_ch = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json')
    eval_sol = load_solutions('data/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json')

    train_keys = list(train_ch.keys())
    eval_keys = list(eval_ch.keys())

    # Next Batch of 10: 8 Training tasks + 2 Evaluation tasks
    batch_tasks = [
        (train_keys[21], train_ch, train_sol, 'TRAIN'),  # 0d3d2d70
        (train_keys[22], train_ch, train_sol, 'TRAIN'),  # 0e67414d
        (train_keys[23], train_ch, train_sol, 'TRAIN'),  # 150deff5
        (train_keys[24], train_ch, train_sol, 'TRAIN'),  # 1190525d
        (train_keys[25], train_ch, train_sol, 'TRAIN'),  # 17864aa1
        (train_keys[26], train_ch, train_sol, 'TRAIN'),  # 1b60fb0c
        (train_keys[27], train_ch, train_sol, 'TRAIN'),  # 1bfc4729
        (train_keys[28], train_ch, train_sol, 'TRAIN'),  # 1c786137
        (eval_keys[3], eval_ch, eval_sol, 'EVAL'),     # 17cae0c1
        (eval_keys[4], eval_ch, eval_sol, 'EVAL'),     # 1a6449f1
    ]

    print("="*75)
    print("=== RUNNING NEXT 10-PUZZLE BENCHMARK (8 TRAIN : 2 EVAL) ===")
    print("="*75)
    sys.stdout.flush()

    solver = NeuroSymbolicARCSolver(beam_width=20, max_depth=2, ranker=None)
    
    solved_count = 0
    total_time = 0.0

    for tid, ch_dict, sol_dict, split_tag in batch_tasks:
        if tid not in ch_dict:
            continue

        task = ch_dict[tid]
        truth = sol_dict[tid]
        train_pairs = [(p.input, p.output) for p in task.train]

        t0 = time.time()
        preds, programs = solver.solve_task(train_pairs, task.test)
        dt = time.time() - t0
        total_time += dt

        ok = False
        if preds and len(preds) > 0:
            for attempt in preds[0]:
                if np.array_equal(np.array(attempt), np.array(truth[0])):
                    ok = True
                    break

        status = "OK (SOLVED!)" if ok else "MISS"
        if ok:
            solved_count += 1

        print(f"\n[{split_tag}] Puzzle {tid}: {status} in {dt:.2f}s")
        if programs:
            print("  Discovered Program(s):")
            for idx, prog in enumerate(programs[:2], 1):
                p_str = prog.program if hasattr(prog, 'program') else prog
                p_err = getattr(prog, 'error', 0.0)
                p_score = getattr(prog, 'score', 0.0)
                print(f"    [{idx}] {p_str} (error: {p_err:.4f}, score: {p_score:.4f})")
        else:
            print("  No symbolic program candidate discovered (used fallback)")
        sys.stdout.flush()

    print("\n" + "="*75)
    print(f"NEXT 10-PUZZLE BATCH SUMMARY: Solved {solved_count}/{len(batch_tasks)} puzzles in {total_time:.2f}s total!")
    print("="*75)
    sys.stdout.flush()

if __name__ == '__main__':
    run_batch_next_10()
