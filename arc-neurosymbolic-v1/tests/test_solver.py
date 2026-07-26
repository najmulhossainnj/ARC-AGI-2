import numpy as np
from arc_solver.core.task import ARCTask,TaskPair
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

def test_solver():
    g=np.array([[1,0],[0,0]])
    t=ARCTask("x",[TaskPair(g,g)],[g])
    preds,_=NeuroSymbolicARCSolver(20,2).solve(t)
    assert preds
