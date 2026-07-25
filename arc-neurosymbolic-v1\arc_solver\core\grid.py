from __future__ import annotations
import numpy as np

def as_grid(x) -> np.ndarray:
    return np.asarray(x, dtype=np.int16)

def grid_equal(a, b) -> bool:
    a, b = as_grid(a), as_grid(b)
    return a.shape == b.shape and np.array_equal(a, b)

def grid_hash(grid):
    g = as_grid(grid)
    return (tuple(g.shape), g.tobytes())

def inside(grid, r, c) -> bool:
    h, w = as_grid(grid).shape
    return 0 <= r < h and 0 <= c < w

def non_background_bbox(grid, background=0):
    g = as_grid(grid)
    pts = np.argwhere(g != background)
    if len(pts) == 0:
        return None
    r1, c1 = pts.min(axis=0)
    r2, c2 = pts.max(axis=0)
    return int(r1), int(c1), int(r2), int(c2)

def crop_non_background(grid, background=0):
    bbox = non_background_bbox(grid, background)
    if bbox is None:
        return as_grid(grid).copy()
    r1, c1, r2, c2 = bbox
    return as_grid(grid)[r1:r2+1, c1:c2+1].copy()

def translate_grid(grid, dr, dc, background=0):
    g = as_grid(grid)
    out = np.full_like(g, background)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = g[r, c]
    return out
