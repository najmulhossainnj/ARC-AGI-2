"""
frame_corner_marker.py
----------------------
Analyzer for task 15663ba9:
- Endpoints and outer corners of line segments/frames are marked with color 4.
- Inner stair-step corners of adjacent turns are marked with color 2.
- 100% verified across all training pairs.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from scipy.ndimage import label
from .base import Analyzer, ProgramCandidate


class FrameCornerMarkerAnalyzer(Analyzer):
    """Mark endpoints/outer corners with color 4 and inner stair-step corners with color 2."""
    name = "frame_corner_marker"
    priority = 18

    def analyze(self, train_pairs, features):
        results = []
        for inp, out in train_pairs:
            inp = np.array(inp)
            out = np.array(out)
            r = self._analyze_pair(inp, out)
            if r is None:
                return None
            results.append(r)
        if not results:
            return None

        def make_solve_fn():
            def solve_fn(grid):
                g = np.asarray(grid)
                pred = FrameCornerMarkerAnalyzer()._compute(g)
                if pred is None:
                    return g.tolist()
                return pred.tolist()
            return solve_fn

        return ProgramCandidate(
            op="FRAME_CORNER_MARKER",
            params=(),
            description="Mark outer corners/endpoints with 4 and inner stair corners with 2",
            solve_fn=make_solve_fn()
        )

    def _compute(self, inp):
        h, w = inp.shape
        out = inp.copy().astype(int)
        bg = 0
        struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

        for color in np.unique(inp):
            if color == bg:
                continue
            mask = (inp == color)
            lbl, num = label(mask, structure=struct4)
            for k in range(1, num + 1):
                comp = (lbl == k)
                rs, cs = np.where(comp)
                points = set(zip(rs.tolist(), cs.tolist()))

                corners = []
                for r, c in points:
                    nbrs = [(dr, dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (r + dr, c + dc) in points]
                    if len(nbrs) == 1:
                        out[r, c] = 4  # Endpoint -> 4
                    elif len(nbrs) == 2:
                        (dr1, dc1), (dr2, dc2) = nbrs
                        if dr1 != -dr2 or dc1 != -dc2:
                            corners.append((r, c))

                for r, c in corners:
                    adj_corners = [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (r + dr, c + dc) in corners]
                    if adj_corners:
                        nr, nc = adj_corners[0]
                        if (r, c) < (nr, nc):
                            out[r, c] = 4
                        else:
                            out[r, c] = 2
                    else:
                        out[r, c] = 4

        return out

    def _analyze_pair(self, inp, out):
        pred = self._compute(inp)
        if pred is not None and np.array_equal(pred, out):
            return True
        return None
