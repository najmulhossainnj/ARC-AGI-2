import numpy as np
from arc_solver.perception.extractor import extract_objects

def test_components():
    g=np.array([[0,1,0],[0,1,0],[2,0,2]])
    objs=extract_objects(g)
    assert len(objs)==3
