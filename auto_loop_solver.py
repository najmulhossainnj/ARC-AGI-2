import sys
import time
import json
import numpy as np

sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges, load_solutions
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver
from arc_solver.meta.diagnostic_engine import DiagnosticEngine
from arc_solver.meta.auto_primitive_injector import inject_llm_solve


def run_auto_loop(start_idx=0, max_tasks=400, use_llm=True):
    train_ch = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json')
    train_sol = load_solutions('data/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json')

    task_ids = list(train_ch.keys())[start_idx:start_idx + max_tasks]

    print("=" * 75)
    print(f"=== META-LEARNING SOLVER LOOP ({len(task_ids)} TASKS) ===")
    print(f"=== Rule-based analyzers: ON | LLM fallback: {'ON (Gemini Flash)' if use_llm else 'OFF'} ===")
    print("=" * 75)
    sys.stdout.flush()

    solver = NeuroSymbolicARCSolver(beam_width=20, max_depth=2, ranker=None)
    engine = DiagnosticEngine(use_llm=use_llm)

    solved_list = []
    failed_list = []
    source_counts = {"rule_based": 0, "llm": 0, "original": 0}

    for i, tid in enumerate(task_ids, start=start_idx + 1):
        task = train_ch[tid]
        truth = train_sol[tid]
        train_pairs = [(p.input, p.output) for p in task.train]

        # ── Attempt 1: Original solver ────────────────────────────────────────
        t0 = time.time()
        preds, programs = solver.solve_task(train_pairs, task.test)
        dt = time.time() - t0

        ok = False
        prog_str = ""
        if preds and len(preds) > 0:
            for attempt in preds[0]:
                if np.array_equal(np.array(attempt), np.array(truth[0])):
                    ok = True
                    prog_str = programs[0].program if hasattr(programs[0], 'program') else str(programs[0])
                    break

        if ok:
            solved_list.append((tid, dt, prog_str, "original"))
            source_counts["original"] += 1
            print(f"\n[{i}/{len(task_ids)+start_idx}] Task {tid}: OK OK in {dt:.2f}s -> {prog_str}")
            sys.stdout.flush()
            continue

        # ── Attempt 2: DiagnosticEngine ───────────────────────────────────────
        inp0 = task.train[0].input
        out0 = task.train[0].output
        print(f"\n[{i}/{len(task_ids)+start_idx}] Task {tid}: MISS in {dt:.2f}s")
        print(f"  SHAPE: INP={inp0.shape} OUT={out0.shape} COLORS_IN={np.unique(inp0).tolist()}")
        sys.stdout.flush()

        diagnosis = engine.diagnose(tid, train_pairs, task.test)

        if diagnosis.success and diagnosis.source == "rule_based":
            # Candidate verified by a rule-based analyzer — re-run solver to pick it up
            # (the candidate op already exists in DSL; solver will find it)
            print(f"  [Meta] Rule-based fix: {diagnosis.candidate.op} ({diagnosis.analyzer_name})")
            print(f"  [Meta] Re-running solver to confirm...")
            sys.stdout.flush()

            t1 = time.time()
            preds2, programs2 = solver.solve_task(train_pairs, task.test)
            dt2 = time.time() - t1

            ok2 = False
            prog_str2 = ""
            if preds2 and len(preds2) > 0:
                for attempt in preds2[0]:
                    if np.array_equal(np.array(attempt), np.array(truth[0])):
                        ok2 = True
                        prog_str2 = programs2[0].program if hasattr(programs2[0], 'program') else str(programs2[0])
                        break

            if ok2:
                solved_list.append((tid, dt + dt2, prog_str2, "rule_based"))
                source_counts["rule_based"] += 1
                print(f"  [Meta] OK SOLVED via rule-based analyzer! prog={prog_str2}")
            else:
                # Analyzer verified but solver doesn't find it yet — apply directly
                cand = diagnosis.candidate
                from arc_solver.dsl.transforms import apply_grid_op
                all_ok = True
                test_preds = []
                for test_item in task.test:
                    test_inp = test_item.input if hasattr(test_item, 'input') else test_item
                    pred = apply_grid_op(np.asarray(test_inp, dtype=np.int16), cand.op, cand.params)
                    if pred is not None:
                        test_preds.append(pred)
                    else:
                        all_ok = False

                if all_ok and test_preds:
                    if np.array_equal(np.asarray(test_preds[0], dtype=np.int16), np.array(truth[0], dtype=np.int16)):
                        solved_list.append((tid, dt + diagnosis.elapsed, str(cand), "rule_based_direct"))
                        source_counts["rule_based"] += 1
                        print(f"  [Meta] OK SOLVED via direct candidate apply! op={cand.op}")
                    else:
                        failed_list.append((tid, "rule_based_verify_fail"))
                        print(f"  [Meta] FAIL Direct apply gave wrong test output.")
                else:
                    failed_list.append((tid, "rule_based_apply_fail"))
                    print(f"  [Meta] FAIL Direct apply failed.")

        elif diagnosis.success and diagnosis.source == "llm":
            # LLM generated a verified solve() function — inject into DSL
            op_name = inject_llm_solve(tid, diagnosis.solve_fn)
            print(f"  [Meta] LLM primitive injected as {op_name}. Re-running solver...")
            sys.stdout.flush()

            t1 = time.time()
            preds2, programs2 = solver.solve_task(train_pairs, task.test)
            dt2 = time.time() - t1

            ok2 = False
            prog_str2 = ""
            if preds2 and len(preds2) > 0:
                for attempt in preds2[0]:
                    if np.array_equal(np.array(attempt), np.array(truth[0])):
                        ok2 = True
                        prog_str2 = programs2[0].program if hasattr(programs2[0], 'program') else str(programs2[0])
                        break

            if ok2:
                solved_list.append((tid, dt + dt2, prog_str2, "llm"))
                source_counts["llm"] += 1
                print(f"  [Meta] OK SOLVED via LLM primitive!")
            else:
                # Apply LLM solve_fn directly to test
                try:
                    test_inp = task.test[0].input if hasattr(task.test[0], 'input') else task.test[0]
                    pred = diagnosis.solve_fn(np.asarray(test_inp, dtype=np.int16).copy())
                    if np.array_equal(np.asarray(pred, dtype=np.int16), np.array(truth[0], dtype=np.int16)):
                        solved_list.append((tid, dt + diagnosis.elapsed, op_name, "llm_direct"))
                        source_counts["llm"] += 1
                        print(f"  [Meta] OK SOLVED via LLM direct apply!")
                    else:
                        failed_list.append((tid, "llm_wrong_test"))
                        print(f"  [Meta] FAIL LLM solve wrong on test pair.")
                except Exception as e:
                    failed_list.append((tid, f"llm_error:{e}"))
                    print(f"  [Meta] FAIL LLM solve error: {e}")
        else:
            failed_list.append((tid, "all_failed"))
            print(f"  [Meta] FAIL All methods failed for task {tid}.")

        sys.stdout.flush()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_solved = len(solved_list)
    total = len(task_ids)
    print("\n" + "=" * 75)
    print(f"LOOP COMPLETE: {total_solved}/{total} solved ({total_solved/total*100:.1f}%)")
    print(f"  Original solver : {source_counts['original']}")
    print(f"  Rule-based meta : {source_counts['rule_based']}")
    print(f"  LLM Gemini Flash: {source_counts['llm']}")
    print(f"  Still failing   : {len(failed_list)}")
    print("=" * 75)
    if failed_list:
        print("STILL FAILING:", [t for t, _ in failed_list])
    sys.stdout.flush()


if __name__ == '__main__':
    start_at = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    no_llm = '--no-llm' in sys.argv
    run_auto_loop(start_idx=start_at, use_llm=not no_llm)
