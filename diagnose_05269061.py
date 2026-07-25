import sys
import json
import numpy as np

sys.path.insert(0, './arc-neurosymbolic-v1')
from arc_solver.utils.arc_io import load_challenges, load_solutions
from arc_solver.dsl.ast import Instruction, Program
from arc_solver.dsl.executor import execute
from arc_solver.verification.exact import program_error

train_ch = load_challenges('data/arc-prize-2026-arc-agi-2/arc-agi_training_challenges.json')
train_sol = load_solutions('data/arc-prize-2026-arc-agi-2/arc-agi_training_solutions.json')

tid = '05269061'
task = train_ch[tid]
truth = np.array(train_sol[tid][0])
train_pairs = [(p.input, p.output) for p in task.train]

prog = Program((
    Instruction('PATTERN_COMPLETE', (3, 6)),
    Instruction('PATTERN_COMPLETE', (3, 3))
))

print(f"=== DIAGNOSING TASK {tid} ===")
print("Program:", prog)
print("Training Pairs Count:", len(train_pairs))

err = program_error(prog, train_pairs)
print("program_error on train_pairs:", err)

for idx, (inp, out) in enumerate(train_pairs):
    res = execute(prog, inp)
    match = res is not None and np.array_equal(res, out)
    diff = (res != out).sum() if res is not None else -1
    print(f"  Train pair {idx}: match={match}, diff_cells={diff}")

# Test pair execution
test_inp = task.test[0].input if hasattr(task.test[0], 'input') else task.test[0]
test_res = execute(prog, test_inp)
test_match = test_res is not None and np.array_equal(test_res, truth)
test_diff = (test_res != truth).sum() if test_res is not None else -1
print(f"  TEST PAIR: match={test_match}, diff_cells={test_diff}")
