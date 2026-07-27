"""
segment_extend_to_boundary.py
Analyzer for task 13713586:
- Detect solid boundary row/column (e.g., color 5).
- Extend non-boundary colored segments towards the boundary row/col.
- 100% verified across all training pairs.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate


class SegmentExtendToBoundaryAnalyzer(Analyzer):
    """Extend color segments toward solid boundary row/col."""
    name = "segment_extend_to_boundary"
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
                out = self._compute(g)
                if out is None:
                    return g.tolist()
                return out.tolist()
            return solve_fn

        return ProgramCandidate(
            op="SEGMENT_EXTEND_TO_BOUNDARY",
            params=(),
            description="Extend non-bg segments toward solid boundary row/col",
            solve_fn=make_solve_fn()
        )

    def _compute(self, inp):
        h, w = inp.shape
        bg = 0
        bc_col = None
        bc_row = None
        for c in range(w):
            if len(np.unique(inp[:, c])) == 1 and inp[0, c] != bg:
                bc_col = c
                break
        if bc_col is None:
            for r in range(h):
                if len(np.unique(inp[r, :])) == 1 and inp[r, 0] != bg:
                    bc_row = r
                    break

        if bc_col is None and bc_row is None:
            return None

        out = inp.copy().astype(int)
        if bc_col is not None:
            for r in range(h):
                row = inp[r, :]
                non_bg = [(c, row[c]) for c in range(w) if row[c] != bg and c != bc_col]
                if not non_bg:
                    continue
                if bc_col > max(c for c, v in non_bg):
                    # Boundary is to the right -> extend rightward to bc_col
                    non_bg.sort()
                    for i in range(len(non_bg)):
                        c_curr, val_curr = non_bg[i]
                        c_next = non_bg[i+1][0] if i+1 < len(non_bg) else bc_col
                        for c in range(c_curr + 1, c_next):
                            if out[r, c] == bg:
                                out[r, c] = val_curr
                else:
                    # Boundary is to the left -> extend leftward to bc_col
                    non_bg.sort(reverse=True)
                    for i in range(len(non_bg)):
                        c_curr, val_curr = non_bg[i]
                        c_next = non_bg[i+1][0] if i+1 < len(non_bg) else bc_col
                        for c in range(c_next + 1, c_curr):
                            if out[r, c] == bg:
                                out[r, c] = val_curr

        elif bc_row is not None:
            for c in range(w):
                col = inp[:, c]
                non_bg = [(r, col[r]) for r in range(h) if col[r] != bg and r != bc_row]
                if not non_bg:
                    continue
                if bc_row > max(r for r, v in non_bg):
                    # Boundary is at bottom -> extend downward to bc_row
                    non_bg.sort()
                    for i in range(len(non_bg)):
                        r_curr, val_curr = non_bg[i]
                        r_next = non_bg[i+1][0] if i+1 < len(non_bg) else bc_row
                        for r in range(r_curr + 1, r_next):
                            if out[r, c] == bg:
                                out[r, c] = val_curr
                else:
                    # Boundary is at top -> extend upward to bc_row
                    non_bg.sort(reverse=True)
                    for i in range(len(non_bg)):
                        r_curr, val_curr = non_bg[i]
                        r_next = non_bg[i+1][0] if i+1 < len(non_bg) else bc_row
                        for r in range(r_next + 1, r_curr):
                            if out[r, c] == bg:
                                out[r, c] = val_curr

        return out

    def _analyze_pair(self, inp, out):
        pred = self._compute(inp)
        if pred is not None and np.array_equal(pred, out):
            return True
        return None
