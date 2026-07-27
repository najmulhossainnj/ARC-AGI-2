"""
run_batch_100_3.py
------------------
Benchmark script for Tasks 200-299 with real-time output flushing.
"""
import sys, json
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from arc_solver.meta.diagnostic_engine import DiagnosticEngine

BATCH_IDS = [
    '36d67576', '36fdfd69', '37ce87bb', '37d3e8b2', '3906de3d',
    '396d80d7', '3979b1a8', '39a8645d', '39e1d7f9', '3a301edc',
    '3aa6fb7a', '3ac3eb23', '3ad05f52', '3af2c5a8', '3b4c2228',
    '3bd292e8', '3bd67248', '3bdb4ada', '3befdf3e', '3c9b0459',
    '3cd86f4f', '3d31c5b3', '3d588dc9', '3d6c6e23', '3de23699',
    '3e980e27', '3eda0437', '3ee1011a', '3f23242b', '3f7978a0',
    '4093f84a', '40f6cd08', '412b6263', '414297c0', '41ace6b5',
    '41e4d17e', '423a55dc', '4258a5f9', '4290ef0e', '42918530',
    '42a15761', '42a50994', '42f14c03', '42f83767', '4347f46a',
    '4364c1c4', '444801d8', '445eab21', '447fd412', '44d8ac46',
    '44f52bb0', '4522001f', '456873bc', '45737921', '458e3a53',
    '45bbe264', '4612dd53', '46442a0e', '465b7d93', '469497ad',
    '46c35fc7', '46f33fce', '470c91de', '47c1f68c', '48131b3c',
    '484b58aa', '4852f2fa', '48634b99', '48d8fb45', '48f8583b',
    '4938f0c2', '494ef9d7', '496994bd', '49d1d64f', '4a1cacc2',
    '4acc7107', '4b6b68e5', '4be741c5', '4c177718', '4c4377d9',
    '4c5c2cf0', '4cd1b7b2', '4df5b0ae', '4e45f183', '4e469f39',
    '4e7e0eb9', '4f537728', '4ff4c9da', '5034a0b5', '505fff84',
    '506d28a5', '50846271', '508bd3b6', '50a16a69', '50aad11f',
    '50c07299', '50cb2852', '50f325b5', '5117e062', '5168d44c'
]

def main():
    print("=" * 80)
    print("=== ARC DIAGNOSTIC ENGINE: RUNNING TASKS 200-299 ===")
    print("=" * 80)
    sys.stdout.flush()

    challenges = json.load(open('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json'))
    engine = DiagnosticEngine()

    solved_count = 0
    failed_tasks = []
    results = {}
    n = len(BATCH_IDS)

    for idx, tid in enumerate(BATCH_IDS, 1):
        print(f"\n[{idx}/{n}] Processing Task: {tid}")
        sys.stdout.flush()

        task_data = challenges[tid]
        train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task_data['train']]

        res = engine.diagnose(tid, train_pairs)
        results[tid] = res

        if res.success:
            solved_count += 1
            print(f"--> Task {tid} Result: ✅ PASSED | Analyzer: {res.analyzer_name} | Time: {res.elapsed:.2f}s")
        else:
            failed_tasks.append(tid)
            print(f"--> Task {tid} Result: ❌ FAILED | Analyzer: {res.analyzer_name} | Time: {res.elapsed:.2f}s")
        sys.stdout.flush()

    print("\n" + "=" * 80)
    print("=== TASKS 200-299 SUMMARY ===")
    print("=" * 80)
    print(f"Total Solved: {solved_count}/{n} ({solved_count * 100 // n}%)")
    print(f"\nFailed Tasks ({len(failed_tasks)}):")
    for tid in failed_tasks:
        print(f"  ❌ {tid}")
    print("\nDetailed Breakdown:")
    for tid in BATCH_IDS:
        r = results[tid]
        status = "✅" if r.success else "❌"
        print(f"  {status} {tid}: ({r.analyzer_name}) in {r.elapsed:.2f}s")
    print("=" * 80)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
