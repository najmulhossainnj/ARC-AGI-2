"""
dot_center_diag_fill.py
Analyzer for task 11e1fe23:
- Each colored dot moves 2 steps diagonally inward toward the centroid of the dots.
- Color 5 is placed at the center (arithmetic mean) of the endpoints.
- 100% verified across all training pairs with zero error.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate


class DotCenterDiagFillAnalyzer(Analyzer):
    """Move dots 2 steps diagonally inward toward centroid, place color 5 at center of endpoints."""
    name = "dot_center_diag_fill"
    priority = 20

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
                pred = DotCenterDiagFillAnalyzer()._compute(g)
                if pred is None:
                    return g.tolist()
                return pred.tolist()
            return solve_fn

        return ProgramCandidate(
            op="DOT_CENTER_DIAG_FILL",
            params=(),
            description="Move dots 2 steps diagonally inward, place color 5 at center",
            solve_fn=make_solve_fn()
        )

    def _compute(self, inp):
        h, w = inp.shape
        bg = 0
        dots = list(zip(*np.where(inp != bg)))
        if len(dots) < 2:
            return None

        cr_in = np.mean([r for r, c in dots])
        cc_in = np.mean([c for r, c in dots])

        out = inp.copy().astype(int)
        endpoints = []

        for r, c in dots:
            dr = 1 if cr_in > r else -1
            dc = 1 if cc_in > c else -1
            nr, nc = int(r + 2 * dr), int(c + 2 * dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = int(inp[r, c])
                endpoints.append((nr, nc))

        if endpoints:
            center_r = int(round(np.mean([r for r, c in endpoints])))
            center_c = int(round(np.mean([c for r, c in endpoints])))
            if 0 <= center_r < h and 0 <= center_c < w:
                out[center_r, center_c] = 5

        return out

    def _analyze_pair(self, inp, out):
        pred = self._compute(inp)
        if pred is not None and np.array_equal(pred, out):
            return True
        return None
