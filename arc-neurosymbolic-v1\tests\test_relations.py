import numpy as np
from arc_solver.core.scene import Scene

def test_left_right():
    g=np.array([[1,0,2]])
    s=Scene.from_grid(g)
    assert s.graph.has(0,"left_of",1)
