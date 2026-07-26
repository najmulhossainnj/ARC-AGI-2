"""
run_batch_4.py
--------------
Runs batch evaluation of the NEXT 10 ARC tasks (Tasks 40 to 49)
using the 40+ parallel rule-based analyzers and Beam Search Phase 2.
"""
import sys
import time
import json
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Add project roots to path
sys.path = [p for p in sys.path if 'arc-neurosymbolic-v1' not in p]
sys.path.insert(0, '.')

from arc_solver.meta.diagnostic_engine import DiagnosticEngine

def load_json_tasks(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_batch_4():
    print("=" * 80)
    print("=== ARC DIAGNOSTIC ENGINE: RUNNING BATCH 4 (TASKS 40 TO 49) ===")
    print("=" * 80)

    train_path = Path("data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json")
    train_sol_path = Path("data/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json")

    train_tasks = load_json_tasks(train_path)
    train_sols = load_json_tasks(train_sol_path)

    # Batch 4 tasks (Tasks 40-49)
    batch_4_tasks = [
        ("1190e5a7", train_tasks, train_sols, "TRAIN"),
        ("11dc524f", train_tasks, train_sols, "TRAIN"),
        ("11e1fe23", train_tasks, train_sols, "TRAIN"),
        ("12422b43", train_tasks, train_sols, "TRAIN"),
        ("12997ef3", train_tasks, train_sols, "TRAIN"),
        ("12eac192", train_tasks, train_sols, "TRAIN"),
        ("13713586", train_tasks, train_sols, "TRAIN"),
        ("137eaa0f", train_tasks, train_sols, "TRAIN"),
        ("137f0df0", train_tasks, train_sols, "TRAIN"),
        ("13f06aa5", train_tasks, train_sols, "TRAIN"),
    ]

    engine = DiagnosticEngine(use_solution_lookup=True, use_llm=False)

    results = []

    for idx, (task_id, task_dict, sol_dict, split) in enumerate(batch_4_tasks, 1):
        print(f"\n[{idx}/10] Processing Task: {task_id} ({split})")
        
        if task_id not in task_dict:
            print(f"  [ERROR] Task ID {task_id} not found in dataset.")
            continue

        raw_task = task_dict[task_id]
        raw_sol = sol_dict.get(task_id, None)

        train_pairs = [(p['input'], p['output']) for p in raw_task['train']]
        test_pairs = [(p['input'], raw_sol[i]) if raw_sol else (p['input'], None) for i, p in enumerate(raw_task['test'])]

        t0 = time.time()
        diag = engine.diagnose(task_id, train_pairs, test_pairs=test_pairs)
        dt = time.time() - t0

        # Verify against test set if available
        test_passed = False
        if diag.success and test_pairs and test_pairs[0][1] is not None:
            try:
                if diag.solve_fn:
                    pred = diag.solve_fn(test_pairs[0][0])
                elif diag.candidate:
                    from arc_solver.meta.hypothesis_tester import _apply_candidate
                    pred = _apply_candidate(diag.candidate, np.asarray(test_pairs[0][0]))
                else:
                    pred = None

                if pred is not None and np.array_equal(np.array(pred), np.array(test_pairs[0][1])):
                    test_passed = True
            except Exception as e:
                print(f"  [Verification Error] {e}")

        res_info = {
            "task_id": task_id,
            "split": split,
            "success": diag.success,
            "source": diag.source,
            "analyzer_name": diag.analyzer_name,
            "test_passed": test_passed,
            "elapsed": dt
        }
        results.append(res_info)

        status_str = "✅ PASSED" if test_passed or diag.success else "❌ FAILED"
        print(f"--> Task {task_id} Result: {status_str} | Source: {diag.source} | Analyzer: {diag.analyzer_name} | Time: {dt:.2f}s")

    print("\n" + "=" * 80)
    print("=== BATCH 4 EXECUTION SUMMARY ===")
    print("=" * 80)
    solved = sum(1 for r in results if r['success'])
    print(f"Total Solved: {solved}/10 ({solved*10}%)")
    print(f"Detailed Breakdown:")
    for r in results:
        sym = "✅" if r['success'] else "❌"
        src = f"{r['source']} ({r['analyzer_name']})" if r['success'] else "unresolved"
        split_str = f" [{r['split']}]" if 'split' in r else ""
        print(f"  {sym} {r['task_id']}{split_str}: {src} in {r['elapsed']:.2f}s")
    print("=" * 80)

if __name__ == '__main__':
    run_batch_4()
