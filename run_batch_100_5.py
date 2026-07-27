"""
run_batch_100_5.py
------------------
Benchmark script for Tasks 400-499 with real-time output flushing.
"""
import sys, json
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from arc_solver.meta.diagnostic_engine import DiagnosticEngine

BATCH_IDS = [
    '6a980be1', '6aa20dc0', '6ad5bdfd', '6b9890af', '6bcdb01e',
    '6c434453', '6ca952ad', '6cbe9eb8', '6cdd2623', '6cf79266',
    '6d0160f0', '6d0aefbc', '6d1d5c90', '6d58a25d', '6d75e8bb',
    '6df30ad6', '6e02f1e3', '6e19193c', '6e82a1ae', '6ea4a07e',
    '6ecd11f4', '6f473927', '6f8cd79b', '6fa7a44f', '6ffe8f07',
    '7039b2d7', '705a3229', '712bf12e', '72207abc', '72322fa7',
    '72a961c9', '72ca375d', '73182012', '73c3b0d8', '73ccf9c2',
    '7447852a', '7468f01a', '746b3537', '74dd1130', '753ea09b',
    '758abdf0', '759f3fd3', '75b8110e', '760b3cac', '762cd429',
    '770cc55f', '776ffc46', '77fdfe62', '780d0b14', '782b5218',
    '7837ac64', '78e78cff', '79369cc6', '794b24be', '7953d61e',
    '79cce52d', '7acdf6d3', '7b6016b9', '7b7f7511', '7bb29440',
    '7c008303', '7c8af763', '7c9b52a0', '7d18a6fb', '7d1f7ee8',
    '7d419a02', '7d7772cc', '7ddcd7ec', '7df24a62', '7e02026e',
    '7e0986d6', '7e2bad24', '7e4d4f7c', '7e576d6e', '7ec998c9',
    '7ee1c6ea', '7f4411dc', '7fe24cdd', '80214e03', '80af3007',
    '810b9b61', '817e6c09', '81c0276b', '825aa9e9', '82819916',
    '83302e8f', '833966f4', '833dafe3', '834ec97d', '83b6b474',
    '83eb0a57', '8403a5d5', '84551f4c', '845d6e51', '846bdb03',
    '84ba50d3', '84db8fc4', '84f2aca1', '855e0971', '8597cfd7'
]

def main():
    print("=" * 80)
    print("=== ARC DIAGNOSTIC ENGINE: RUNNING TASKS 400-499 ===")
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
    print("=== TASKS 400-499 SUMMARY ===")
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
