from ..dsl.executor import execute
from ..core.grid import grid_equal

def program_error(program,train_pairs):
    total=0
    for inp,out in train_pairs:
        pred=execute(program,inp)
        if pred is None or pred.shape!=out.shape:
            return float("inf")
        total += int((pred!=out).sum())
    return total

def exact(program,train_pairs):
    return program_error(program,train_pairs)==0
