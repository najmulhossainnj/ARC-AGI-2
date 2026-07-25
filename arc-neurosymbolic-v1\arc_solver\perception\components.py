from collections import deque
import numpy as np

def connected_components(grid, background=0, diagonal=False):
    g = np.asarray(grid)
    h, w = g.shape
    seen = np.zeros((h,w), dtype=bool)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if diagonal:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r,c] or g[r,c] == background:
                continue
            q = deque([(r,c)])
            seen[r,c] = True
            cells = []
            while q:
                cr,cc = q.popleft()
                cells.append((cr,cc))
                for dr,dc in dirs:
                    nr,nc = cr+dr,cc+dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr,nc] and g[nr,nc] != background:
                        seen[nr,nc] = True
                        q.append((nr,nc))
            components.append(cells)
    return components

def same_color_components(grid, background=0, diagonal=False):
    g = np.asarray(grid)
    h,w = g.shape
    seen = np.zeros((h,w), dtype=bool)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if diagonal:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    result = []
    for r in range(h):
        for c in range(w):
            color = int(g[r,c])
            if seen[r,c] or color == background:
                continue
            q=deque([(r,c)]); seen[r,c]=True; cells=[]
            while q:
                cr,cc=q.popleft(); cells.append((cr,cc))
                for dr,dc in dirs:
                    nr,nc=cr+dr,cc+dc
                    if 0<=nr<h and 0<=nc<w and not seen[nr,nc] and int(g[nr,nc])==color:
                        seen[nr,nc]=True; q.append((nr,nc))
            result.append((color,cells))
    return result
