"""
uniform_line_tile_3x3.py
------------------------
Analyzer for task 15696249:
- 3x3 input has a row or column of a single uniform color.
- If row R is uniform, tile 3x3 input horizontally 3 times at row block R of 9x9 grid.
- If col C is uniform, tile 3x3 input vertically 3 times at col block C of 9x9 grid.
- 100% verified across all training pairs with zero error.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate


class UniformLineTile3x3Analyzer(Analyzer):
    """Tile 3x3 input pattern 3 times along row/col block matching uniform line."""
    name = "uniform_line_tile_3x3"
    priority = 15

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
                pred = UniformLineTile3x3Analyzer()._compute(g)
                if pred is None:
                    return g.tolist()
                return pred.tolist()
            return solve_fn

        return ProgramCandidate(
            op="UNIFORM_LINE_TILE_3X3",
            params=(),
            description="Tile 3x3 grid 3 times along block matching uniform row/col line",
            solve_fn=make_solve_fn()
        )

    def _compute(self, inp):
        if inp.shape != (3, 3):
            return None

        out = np.zeros((9, 9), dtype=int)

        # Check rows
        for r in range(3):
            if len(np.unique(inp[r, :])) == 1:
                r_start = r * 3
                out[r_start:r_start + 3, 0:3] = inp
                out[r_start:r_start + 3, 3:6] = inp
                out[r_start:r_start + 3, 6:9] = inp
                return out

        # Check cols
        for c in range(3):
            if len(np.unique(inp[:, c])) == 1:
                c_start = c * 3
                out[0:3, c_start:c_start + 3] = inp
                out[3:6, c_start:c_start + 3] = inp
                out[6:9, c_start:c_start + 3] = inp
                return out

        return None

    def _analyze_pair(self, inp, out):
        pred = self._compute(inp)
        if pred is not None and np.array_equal(pred, out):
            return True
        return None
