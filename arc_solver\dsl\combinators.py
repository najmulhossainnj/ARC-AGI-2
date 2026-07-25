import numpy as np

def union_grids(grids, background=0):
    if not grids: return None
    shape=grids[0].shape
    out=np.zeros(shape,dtype=grids[0].dtype)
    for g in grids:
        if g.shape!=shape: continue
        mask=g!=background
        out[mask]=g[mask]
    return out

def difference(a,b,background=0):
    out=a.copy()
    mask=(b!=background)
    out[mask]=background
    return out
