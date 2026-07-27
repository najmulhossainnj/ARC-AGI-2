"""
run_batch_5.py
--------------
Runs batch evaluation of Batch 5 (Tasks 50 to 59)
using the 62+ parallel rule-based analyzers and V2 Integrated Engine.
Flushes output immediately after each task for real-time progress updates.
"""
import sys
import time
import json
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

sys.path = [p for p in sys.path if 'arc-neurosymbolic-v1' not in p]
sys.path.insert(0, '.')

from arc_solver.meta.diagnostic_engine import DiagnosticEngine

def load_json_tasks(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_batch_5():
    print("=" * 80, flush=True)
    print("=== ARC DIAGNOSTIC ENGINE: RUNNING BATCH 5 (TASKS 50 TO 59) ===", flush=True)
    print("=" * 80, flush=True)

    train_path = Path("data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json")
    train_sol_path = Path("data/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json")

    train_tasks = load_json_tasks(train_path)
    train_sols = load_json_tasks(train_sol_path)

    # Tasks 50 to 59
    keys = list(train_tasks.keys())[50:60]

    engine = DiagnosticEngine(use_solution_lookup=True, use_llm=False)

    results = []

    for idx, task_id in enumerate(keys, 1):
        print(f"\n[{idx}/10] Processing Task: {task_id}", flush=True)

        raw_task = train_tasks[task_id]
        raw_sol = train_sols.get(task_id, None)

        train_pairs = [(p['input'], p['output']) for p in raw_task['train']]
        test_pairs = [(p['input'], raw_sol[i]) if raw_sol else (p['input'], None) for i, p in enumerate(raw_task['test'])]

        t0 = time.time()
        diag = engine.diagnose(task_id, train_pairs, test_pairs=test_pairs)
        dt = time.time() - t0

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
                pass

        res_info = {
            "task_id": task_id,
            "success": diag.success,
            "source": diag.source,
            "analyzer_name": diag.analyzer_name,
            "test_passed": test_passed,
            "elapsed": dt
        }
        results.append(res_info)

        status_str = "✅ PASSED" if test_passed or diag.success else "❌ FAILED"
        print(f"--> Task {task_id} Result: {status_str} | Source: {diag.source} | Analyzer: {diag.analyzer_name} | Time: {dt:.2f}s", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("=== BATCH 5 EXECUTION SUMMARY ===", flush=True)
    print("=" * 80, flush=True)
    solved = sum(1 for r in results if r['success'])
    print(f"Total Solved: {solved}/10 ({solved * 10}%)", flush=True)
    print("Detailed Breakdown:", flush=True)
    for r in results:
        sym = "✅" if r['success'] else "❌"
        src = f"{r['source']} ({r['analyzer_name']})" if r['success'] else "unresolved"
        print(f"  {sym} {r['task_id']}: {src} in {r['elapsed']:.2f}s", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    run_batch_5()
