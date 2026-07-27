"""
run_batch_100_2.py
------------------
Benchmark script for Tasks 100-199 with real-time output flushing.
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from arc_solver.meta.diagnostic_engine import DiagnosticEngine

BATCH_IDS = [
    '1f85a75f', '1f876c06', '1fad071e', '2013d3e2', '2037f2c7',
    '2072aba6', '20818e16', '20981f0e', '20fb2937', '212895b5',
    '21f83797', '2204b7a8', '22168020', '22208ba4', '22233c11',
    '22425bda', '22806e14', '2281f1f4', '228f6490', '22a4bbc2',
    '22eb0ac0', '230f2e48', '234bbc79', '23581191', '239be575',
    '23b5c85d', '25094a63', '252143c9', '253bf280', '2546ccf6',
    '256b0a75', '25c199f5', '25d487eb', '25d8a9c8', '25e02866',
    '25ff71a9', '2601afb7', '264363fd', '2685904e', '2697da3f',
    '272f95fa', '2753e76c', '278e5215', '27a28665', '27a77e38',
    '27f8ce4f', '281123b4', '28bf18c6', '28e73c20', '292dd178',
    '29623171', '29700607', '29c11459', '2a28add5', '2a5f8217',
    '2b01abd0', '2b9ef948', '2bcee788', '2bee17df', '2c0b0aff',
    '2c608aff', '2c737e39', '2ccd9fef', '2dc579da', '2dd70a9a',
    '2de01db2', '2dee498d', '2e65ae53', '2f0c5170', '2f767503',
    '2faf500b', '305b1341', '30f42897', '310f3251', '3194b014',
    '319f2597', '31aa019c', '31adaf00', '31d5ba1a', '320afe60',
    '321b1fc6', '32597951', '32e9702f', '33067df9', '332202d5',
    '332efdb3', '3345333e', '337b420f', '3391f8c0', '33b52de3',
    '3428a4f5', '342ae2ed', '342dd610', '3490cc26', '34b99a2b',
    '34cfa167', '351d6448', '358ba94e', '3618c87e', '363442ee'
]

def main():
    print("=" * 80)
    print("=== ARC DIAGNOSTIC ENGINE: RUNNING TASKS 100-199 ===")
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
    print("=== TASKS 100-199 SUMMARY ===")
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
