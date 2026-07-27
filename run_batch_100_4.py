"""
run_batch_100_4.py
------------------
Benchmark script for Tasks 300-399 with real-time output flushing.
"""
import sys, json
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from arc_solver.meta.diagnostic_engine import DiagnosticEngine

BATCH_IDS = [
    '516b51b7', '5207a7b5', '522fdd07', '52364a65', '5289ad53',
    '52df9849', '52fd389e', '538b439f', '539a4f51', '53b68214',
    '543a7ed5', '54d82841', '54d9e175', '54db823b', '54dc2872',
    '55059096', '551d5bf1', '5521c0d9', '5582e5ca', '5587a8d0',
    '5614dbcf', '5623160b', '56dc2b01', '56ff96f3', '5751f35e',
    '575b1a71', '5783df64', '5792cb4d', '57aa92db', '57edb29d',
    '5833af48', '58743b76', '58c02a16', '58e15b12', '59341089',
    '5a5a2103', '5a719d11', '5ad4f10b', '5ad8a7c0', '5adee1b2',
    '5af49b42', '5b37cb25', '5b526a93', '5b692c0f', '5b6cbef5',
    '5bd6f4ac', '5c0a986e', '5c2c9af4', '5d2a5c43', '5d588b4d',
    '5daaa586', '5e6bbc0b', '5ecac7f7', '5ffb2104', '60a26a3e',
    '60b61512', '60c09cac', '60d73be6', '6150a2bd', '6165ea8f',
    '623ea044', '626c0bcc', '62ab2642', '62b74c02', '62c24649',
    '6350f1f4', '63613498', '639f5a19', '642248e4', '642d658d',
    '6430c8c4', '6455b5f5', '64a7c07e', '652646ff', '662c240a',
    '668eec9a', '66ac4c3b', '66e6c45b', '66f2d22f', '67385a82',
    '673ef223', '67636eac', '6773b310', '67a3c6ac', '67a423a3',
    '67c52801', '67e8384a', '681b3aeb', '6855a6e4', '689c358e',
    '68b16354', '68b67ca3', '68bc2e87', '692cd3b6', '694f12f3',
    '695367ec', '696d4842', '69889d6e', '6a11f6da', '6a1e5592'
]

def main():
    print("=" * 80)
    print("=== ARC DIAGNOSTIC ENGINE: RUNNING TASKS 300-399 ===")
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
    print("=== TASKS 300-399 SUMMARY ===")
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
