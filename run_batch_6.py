"""
run_batch_6.py
--------------
Benchmark script for Batch 6 (Tasks 60 to 69) with real-time output flushing.
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from arc_solver.meta.diagnostic_engine import DiagnosticEngine

BATCH6_IDS = [
    '178fcbfb', '17b80ad2', '17b866bd', '17cae0c1', '18286ef8',
    '182e5d0f', '18419cfa', '18447a8d', '184a9768', '195ba7dc'
]

def main():
    print("=" * 80)
    print("=== ARC DIAGNOSTIC ENGINE: RUNNING BATCH 6 (TASKS 60 TO 69) ===")
    print("=" * 80)
    sys.stdout.flush()

    challenges = json.load(open('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json'))
    engine = DiagnosticEngine()

    solved_count = 0
    results = {}

    for idx, tid in enumerate(BATCH6_IDS, 1):
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
    print("=== BATCH 6 EXECUTION SUMMARY ===")
    print("=" * 80)
    print(f"Total Solved: {solved_count}/10 ({solved_count * 10}%)")
    print("Detailed Breakdown:")
    for tid in BATCH6_IDS:
        r = results[tid]
        status = "✅ PASSED" if r.success else "❌ FAILED"
        print(f"  {status} {tid}: {r.source} ({r.analyzer_name}) in {r.elapsed:.2f}s")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
