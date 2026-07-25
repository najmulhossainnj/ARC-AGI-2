from ..dsl.executor import execute
from ..core.grid import grid_hash

def output_consistency(programs,test_grid):
    groups={}
    for p in programs:
        out=execute(p.program if hasattr(p,"program") else p,test_grid)
        if out is None: continue
        groups.setdefault(grid_hash(out),[]).append((p,out))
    return sorted(groups.values(),key=len,reverse=True)
