"""
run_batch_100.py
----------------
Benchmark script for the first 100 tasks with real-time output flushing.
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from arc_solver.meta.diagnostic_engine import DiagnosticEngine

BATCH100_IDS = [
    '00576224', '007bbfb7', '009d5c81', '00d62c1b', '00dbd492',
    '017c7c7b', '025d127b', '03560426', '045e512c', '0520fde7',
    '05269061', '05a7bcf2', '05f2a901', '0607ce86', '0692e18c',
    '06df4c85', '070dd51e', '08ed6ac7', '09629e4f', '0962bcdd',
    '09c534e7', '0a1d4ef5', '0a2355a6', '0a938d79', '0b148d64',
    '0b17323b', '0bb8deee', '0becf7df', '0c786b71', '0c9aba6e',
    '0ca9ddb6', '0d3d703e', '0d87d2a6', '0e206a2e', '0e671a1a',
    '0f63c0b9', '103eff5b', '10fcaaa3', '11852cab', '1190bc91',
    '1190e5a7', '11dc524f', '11e1fe23', '12422b43', '12997ef3',
    '12eac192', '13713586', '137eaa0f', '137f0df0', '13f06aa5',
    '140c817e', '14754a24', '1478ab18', '14b8e18c', '150deff5',
    '15113be4', '15660dd6', '15663ba9', '15696249', '17829a00',
    '178fcbfb', '17b80ad2', '17b866bd', '17cae0c1', '18286ef8',
    '182e5d0f', '18419cfa', '18447a8d', '184a9768', '195ba7dc',
    '1990f7a8', '19bb5feb', '1a07d186', '1a244afd', '1a2e2828',
    '1a6449f1', '1acc24af', '1b2d62fb', '1b59e163', '1b60fb0c',
    '1b8318e3', '1be83260', '1bfc4729', '1c02dbbe', '1c0d0a4b',
    '1c56ad9f', '1c786137', '1caeab9d', '1cf80156', '1d0a4b61',
    '1d398264', '1d61978c', '1da012fc', '1e0a9b12', '1e32b0e9',
    '1e5d6875', '1e81d6f9', '1efba499', '1f0c79e5', '1f642eb9'
]

def main():
    print("=" * 80)
    print("=== ARC DIAGNOSTIC ENGINE: RUNNING FIRST 100 TASKS ===")
    print("=" * 80)
    sys.stdout.flush()

    challenges = json.load(open('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json'))
    engine = DiagnosticEngine()

    solved_count = 0
    failed_tasks = []
    results = {}
    n = len(BATCH100_IDS)

    for idx, tid in enumerate(BATCH100_IDS, 1):
        print(f"\n[{idx}/{n}] Processing Task: {tid}")
        sys.stdout.flush()

        task_data = challenges[tid]
        train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task_data['train']]

        res = engine.diagnose(tid, train_pairs)
        results[tid] = res

        if res.success:
            solved_count += 1
            print(f"--> Task {tid} Result: ✅ PASSED | Source: {res.source} | Analyzer: {res.analyzer_name} | Time: {res.elapsed:.2f}s")
        else:
            failed_tasks.append(tid)
            print(f"--> Task {tid} Result: ❌ FAILED | Source: {res.source} | Analyzer: {res.analyzer_name} | Time: {res.elapsed:.2f}s")
        sys.stdout.flush()

    print("\n" + "=" * 80)
    print("=== BATCH 100 EXECUTION SUMMARY ===")
    print("=" * 80)
    print(f"Total Solved: {solved_count}/{n} ({solved_count * 100 // n}%)")
    print(f"\nFailed Tasks ({len(failed_tasks)}):")
    for tid in failed_tasks:
        print(f"  ❌ {tid}")
    print("\nDetailed Breakdown:")
    for tid in BATCH100_IDS:
        r = results[tid]
        status = "✅" if r.success else "❌"
        print(f"  {status} {tid}: {r.source} ({r.analyzer_name}) in {r.elapsed:.2f}s")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
