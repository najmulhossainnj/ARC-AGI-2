"""
pattern_tile_downward.py
Analyzer for task 12422b43:
- Input has a marker column with color 5 spanning N rows.
- Output tiles rows 0..N-1 (without the 5 column) downward into empty rows below.
- 100% verified across all training pairs with 0 error.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate


class PatternTileDownwardAnalyzer(Analyzer):
    """Tile top N rows (where N = count of marker color 5) downward into empty rows below."""
    name = "pattern_tile_downward"
    priority = 19

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
                pred = PatternTileDownwardAnalyzer()._compute(g)
                if pred is None:
                    return g.tolist()
                return pred.tolist()
            return solve_fn

        return ProgramCandidate(
            op="PATTERN_TILE_DOWNWARD",
            params=(),
            description="Tile rows 0..N-1 (N=marker count) downward into empty rows",
            solve_fn=make_solve_fn()
        )

    def _compute(self, inp):
        h, w = inp.shape
        bg = 0
        marker_col = None
        for c in range(w):
            if (inp[:, c] == 5).sum() > 0:
                marker_col = c
                break
        if marker_col is None:
            return None

        marker_rows = np.where(inp[:, marker_col] == 5)[0]
        tile_h = len(marker_rows)
        if tile_h == 0:
            return None

        # Extract tile from rows 0..tile_h-1 (setting marker_col to bg)
        tile = inp[0:tile_h, :].copy().astype(int)
        tile[:, marker_col] = bg

        # Find last row in input that has non-bg content
        non_bg_rows = [r for r in range(h) if (inp[r, :] != bg).any()]
        if not non_bg_rows:
            return None
        last_inp_row = max(non_bg_rows)

        pred = inp.copy().astype(int)
        pos = last_inp_row + 1
        while pos < h:
            rem = min(tile_h, h - pos)
            pred[pos:pos+rem, :] = tile[:rem, :]
            pos += tile_h

        return pred

    def _analyze_pair(self, inp, out):
        pred = self._compute(inp)
        if pred is not None and np.array_equal(pred, out):
            return True
        return None
