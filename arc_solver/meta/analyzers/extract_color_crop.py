"""
extract_color_crop.py
Analyzer for task 1190e5a7:
- Input grid has a background color and a 'line' color.
- The line color forms full rows and full columns, dividing the grid.
- Output: background-colored grid with shape = (num_full_row_gaps, num_full_col_gaps).
- i.e., count the row-sections and col-sections between the full-line rows/cols.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate


class ExtractColorCropAnalyzer(Analyzer):
    """Output shape = (#row sections, #col sections) filled with background color."""
    name = "extract_color_crop"
    priority = 18

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp = np.array(inp)
            out = np.array(out)
            r = self._analyze_pair(inp, out)
            if r is None:
                return None

        def make_solve_fn():
            def solve_fn(grid):
                g = np.asarray(grid)
                h, w = g.shape
                counts = np.bincount(g.flatten().astype(int), minlength=20)
                bg = int(np.argmax(counts))
                lcs = [c for c in np.unique(g) if c != bg]
                if len(lcs) != 1:
                    return g.tolist()
                lc = int(lcs[0])
                full_rows = [r for r in range(h) if (g[r, :] == lc).all()]
                full_cols = [c for c in range(w) if (g[:, c] == lc).all()]
                n_row_sections = len(full_rows) + 1
                n_col_sections = len(full_cols) + 1
                return np.full((n_row_sections, n_col_sections), bg, dtype=int).tolist()
            return solve_fn

        return ProgramCandidate(
            op="EXTRACT_COLOR_CROP",
            params=(),
            description="Output shape (#row_sections, #col_sections) filled with bg",
            solve_fn=make_solve_fn()
        )

    def _analyze_pair(self, inp, out):
        h, w = inp.shape
        counts = np.bincount(inp.flatten().astype(int), minlength=20)
        bg = int(np.argmax(counts))
        line_colors = [c for c in np.unique(inp) if c != bg]
        if len(line_colors) != 1:
            return None
        lc = int(line_colors[0])

        full_rows = [r for r in range(h) if (inp[r, :] == lc).all()]
        full_cols = [c for c in range(w) if (inp[:, c] == lc).all()]

        n_row_sections = len(full_rows) + 1
        n_col_sections = len(full_cols) + 1

        expected = np.full((n_row_sections, n_col_sections), bg, dtype=int)
        if np.array_equal(expected, out):
            return lc
        return None
