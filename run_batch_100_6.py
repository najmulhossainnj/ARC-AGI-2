"""
run_batch_100_6.py
------------------
Benchmark script for Tasks 500-599 with real-time output flushing.
"""
import sys, json
import numpy as np

sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from arc_solver.meta.diagnostic_engine import DiagnosticEngine

BATCH_IDS = [
    '85b81ff1', '85c4e7cd', '85fa5666', '8618d23e', '868de0fa',
    '8719f442', '8731374e', '878187ab', '87ab05b8', '880c1354',
    '88207623', '8886d717', '88a10436', '88a62173', '890034e9',
    '891232d6', '896d5239', '8a004b2b', '8a371977', '8a6d367c',
    '8abad3cf', '8b28cd80', '8ba14f53', '8be77c9e', '8cb8642d',
    '8d5021e8', '8d510a79', '8dab14c2', '8dae5dfc', '8e1813be',
    '8e2edd66', '8e301a54', '8e5a5113', '8eb1be9a', '8ee62060',
    '8efcae92', '8f2ea7aa', '8fbca751', '8fff9e47', '902510d5',
    '90347967', '90c28cc7', '90f3ed37', '9110e3c5', '913fb3ed',
    '91413438', '91714a58', '9172f3a0', '917bccba', '928ad970',
    '92e50de0', '9344f635', '9356391f', '93b4f4b3', '93b581b8',
    '93c31fbe', '94133066', '941d9a10', '94414823', '9473c6fb',
    '94be5b80', '94f9d214', '952a094c', '9565186b', '95755ff2',
    '95990924', '95a58926', '963c33f8', '963e52fc', '963f59bc',
    '96a8c0cd', '9720b24f', '97239e3d', '973e499e', '9772c176',
    '97999447', '97a05b5b', '97c75046', '981add89', '9841fdad',
    '984d8a3e', '985ae207', '98c475bf', '98cf29f8', '992798f6',
    '99306f82', '995c5fa3', '9968a131', '996ec1f3', '99b1bc43',
    '99caaf76', '99fa7670', '9a4bb226', '9aec4887', '9af7a82c',
    '9b2a60aa', '9b30e358', '9b365c51', '9b4c17c4', '9b5080bb'
]

def main():
    print("=" * 80)
    print("=== ARC DIAGNOSTIC ENGINE: RUNNING TASKS 500-599 ===")
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
    print("=== TASKS 500-599 SUMMARY ===")
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
