from collections import Counter
from ..dsl.executor import execute
from ..core.grid import grid_hash

def select_diverse(candidates,test_grid,top_k=2):
    groups={}
    for c in candidates:
        out=execute(c.program,test_grid)
        if out is not None:
            groups.setdefault(grid_hash(out),[]).append((c,out))
    ranked=sorted(groups.values(),key=lambda g:min(x[0].score for x in g))
    return [min(g,key=lambda x:x[0].score) for g in ranked[:top_k]]
