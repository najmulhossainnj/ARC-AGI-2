import numpy as np
from arc_solver.dsl.ast import Program,Instruction
from arc_solver.dsl.executor import execute

def test_rotate():
    g=np.array([[1,0],[0,0]])
    p=Program((Instruction("ROTATE",(90,)),))
    out=execute(p,g)
    assert out.tolist()==[[0,1],[0,0]]
