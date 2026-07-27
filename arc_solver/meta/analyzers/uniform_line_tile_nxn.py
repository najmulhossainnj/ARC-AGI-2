"""
uniform_line_tile_nxn.py
------------------------
Generalized NxN Uniform Line Tile Analyzer for ARC-AGI V2 Architecture.

Generalization:
- Dynamically infers input dimension (N x M) and output dimension (H_out x W_out).
- Detects uniform row or column lines in the N x M input grid.
- If row R is uniform, tiles the N x M pattern horizontally across block R of the output grid.
- If col C is uniform, tiles the N x M pattern vertically across block C of the output grid.
- Works for any N x M input and any output grid dimensions (3x3 -> 9x9, 4x4 -> 12x12, 2x2 -> 6x6, etc.).
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate


class UniformLineTileNxNAnalyzer(Analyzer):
    """Tile N x M input pattern along row/col block matching uniform line across any grid dimensions."""
    name = "uniform_line_tile_nxn"
    priority = 15

    def analyze(self, train_pairs, features):
        results = []
        out_shape = None
        for inp, out in train_pairs:
            inp = np.array(inp)
            out = np.array(out)
            out_shape = out.shape
            r = self._analyze_pair(inp, out)
            if r is None:
                return None
            results.append(r)
        if not results or out_shape is None:
            return None

        def make_solve_fn(target_shape):
            def solve_fn(grid):
                g = np.asarray(grid)
                pred = UniformLineTileNxNAnalyzer()._compute(g, target_shape)
                if pred is None:
                    return g.tolist()
                return pred.tolist()
            return solve_fn

        return ProgramCandidate(
            op="UNIFORM_LINE_TILE_NXN",
            params=(out_shape,),
            description=f"Tile N x M grid along block matching uniform row/col line to target shape {out_shape}",
            solve_fn=make_solve_fn(out_shape)
        )

    def _compute(self, inp, out_shape=(9, 9)):
        h_in, w_in = inp.shape
        h_out, w_out = out_shape

        k_r = h_out // h_in
        k_c = w_out // w_in

        out = np.zeros(out_shape, dtype=int)

        # Check rows for uniform line
        for r in range(h_in):
            if len(np.unique(inp[r, :])) == 1:
                r_start = r * h_in
                for i in range(k_c):
                    out[r_start:r_start + h_in, i * w_in:(i + 1) * w_in] = inp
                return out

        # Check cols for uniform line
        for c in range(w_in):
            if len(np.unique(inp[:, c])) == 1:
                c_start = c * w_in
                for i in range(k_r):
                    out[i * h_in:(i + 1) * h_in, c_start:c_start + w_in] = inp
                return out

        return None

    def _analyze_pair(self, inp, out):
        pred = self._compute(inp, out.shape)
        if pred is not None and np.array_equal(pred, out):
            return True
        return None
