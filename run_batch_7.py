"""
run_batch_7.py
--------------
Benchmark script for Batch 7 (Tasks 70 to 79) with real-time output flushing.
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from arc_solver.meta.diagnostic_engine import DiagnosticEngine

BATCH7_IDS = [
    '1990f7a8', '19bb5feb', '1a07d186', '1a244afd', '1a2e2828',
    '1a6449f1', '1acc24af', '1b2d62fb', '1b59e163', '1b60fb0c'
]

def main():
    print("=" * 80)
    print("=== ARC DIAGNOSTIC ENGINE: RUNNING BATCH 7 (TASKS 70 TO 79) ===")
    print("=" * 80)
    sys.stdout.flush()

    challenges = json.load(open('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json'))
    engine = DiagnosticEngine()

    solved_count = 0
    results = {}

    for idx, tid in enumerate(BATCH7_IDS, 1):
        print(f"\n[{idx}/10] Processing Task: {tid}")
        sys.stdout.flush()

        task_data = challenges[tid]
        train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task_data['train']]

        res = engine.diagnose(tid, train_pairs)
        results[tid] = res

        if res.success:
            solved_count += 1
            print(f"--> Task {tid} Result: ✅ PASSED | Source: {res.source} | Analyzer: {res.analyzer_name} | Time: {res.elapsed:.2f}s")
        else:
            print(f"--> Task {tid} Result: ❌ FAILED | Source: {res.source} | Analyzer: {res.analyzer_name} | Time: {res.elapsed:.2f}s")
        sys.stdout.flush()

    print("\n" + "=" * 80)
    print("=== BATCH 7 EXECUTION SUMMARY ===")
    print("=" * 80)
    print(f"Total Solved: {solved_count}/10 ({solved_count * 10}%)")
    print("Detailed Breakdown:")
    for tid in BATCH7_IDS:
        r = results[tid]
        status = "✅ PASSED" if r.success else "❌ FAILED"
        print(f"  {status} {tid}: {r.source} ({r.analyzer_name}) in {r.elapsed:.2f}s")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
