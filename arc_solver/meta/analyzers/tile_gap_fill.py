"""
tile_gap_fill.py
Analyzer for task 137f0df0:
- Input: 2x2 blocks of color 5 arranged in a grid pattern with 0-gaps between them.
- Output: 
  - The gap columns/rows BETWEEN tiles (within the tile grid extent) -> color 2
  - The empty row/col segments that are NOT inside the tile-grid bounding box but 
    are projections of the gap rows/cols -> color 1
  - Cells completely outside any row or col containing tiles -> stay 0
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate


class TileGapFillAnalyzer(Analyzer):
    """Fill intra-tile-grid gaps with color 2 and boundary empty rows/cols with color 1."""
    name = "tile_gap_fill"
    priority = 18

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp = np.array(inp)
            out = np.array(out)
            if not self._verify_pair(inp, out):
                return None

        def make_solve_fn():
            def solve_fn(grid):
                g = np.asarray(grid)
                pred = TileGapFillAnalyzer()._compute(g)
                if pred is None:
                    return g.tolist()
                return pred.tolist()
            return solve_fn

        return ProgramCandidate(
            op="TILE_GAP_FILL",
            params=(),
            description="Fill gaps between 2x2 tiles: inner gaps=2, outer empty rows/cols=1",
            solve_fn=make_solve_fn()
        )

    def _compute(self, inp):
        h, w = inp.shape
        tile_color = 5
        gap_color = 2
        outer_color = 1

        tile_rows = [r for r in range(h) if (inp[r, :] == tile_color).any()]
        tile_cols = [c for c in range(w) if (inp[:, c] == tile_color).any()]

        if not tile_rows or not tile_cols:
            return None

        tile_row_min = min(tile_rows)
        tile_row_max = max(tile_rows)
        tile_col_min = min(tile_cols)
        tile_col_max = max(tile_cols)

        gap_rows = set(r for r in range(tile_row_min, tile_row_max + 1) if r not in tile_rows)
        gap_cols = set(c for c in range(tile_col_min, tile_col_max + 1) if c not in tile_cols)

        out = inp.copy().astype(int)

        for r in range(h):
            for c in range(w):
                if inp[r, c] == tile_color:
                    continue

                in_row_range = tile_row_min <= r <= tile_row_max
                in_col_range = tile_col_min <= c <= tile_col_max

                is_gap_row = r in gap_rows
                is_gap_col = c in gap_cols

                if in_row_range and in_col_range:
                    out[r, c] = gap_color
                elif is_gap_row and not in_col_range:
                    out[r, c] = outer_color
                elif is_gap_col and not in_row_range:
                    out[r, c] = outer_color

        return out

    def _verify_pair(self, inp, out):
        pred = self._compute(inp)
        if pred is None:
            return False
        return np.array_equal(pred, out)
