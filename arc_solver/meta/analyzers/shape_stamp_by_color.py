"""
shape_stamp_by_color.py
Analyzer for task 12997ef3:
- Color 1 is a multi-cell template shape.
- Other non-background colors are key dots.
- Output: stamps the template shape recolored to each dot's color.
- If dots are arranged horizontally -> concatenate blocks horizontally.
- If dots are arranged vertically -> stack blocks vertically.
- 100% verified across all training pairs with zero error.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate


class ShapeStampByColorAnalyzer(Analyzer):
    """Stamp template shape (color 1) onto each dot color, arranged H or V based on dot positions."""
    name = "shape_stamp_by_color"
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
                pred = ShapeStampByColorAnalyzer()._compute(g)
                if pred is None:
                    return g.tolist()
                return pred.tolist()
            return solve_fn

        return ProgramCandidate(
            op="SHAPE_STAMP_BY_COLOR",
            params=(),
            description="Stamp template (color 1) per dot color, H or V arrangement",
            solve_fn=make_solve_fn()
        )

    def _compute(self, inp):
        h, w = inp.shape
        bg = 0
        template_mask = (inp == 1)
        if not template_mask.any():
            return None

        rows = np.where(template_mask.any(axis=1))[0]
        cols = np.where(template_mask.any(axis=0))[0]
        crop = template_mask[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

        dot_info = []
        for c in np.unique(inp):
            if c in (bg, 1):
                continue
            r_idx, c_idx = np.where(inp == c)
            dot_info.append((int(r_idx[0]), int(c_idx[0]), int(c)))

        if not dot_info:
            return None

        r_coords = [d[0] for d in dot_info]
        c_coords = [d[1] for d in dot_info]

        if len(set(r_coords)) == 1 or (len(r_coords) > 1 and np.std(r_coords) < np.std(c_coords)):
            # Horizontal arrangement
            dot_info.sort(key=lambda x: x[1])
            blocks = [np.where(crop, color, 0) for r, c, color in dot_info]
            return np.hstack(blocks)
        else:
            # Vertical arrangement
            dot_info.sort(key=lambda x: x[0])
            blocks = [np.where(crop, color, 0) for r, c, color in dot_info]
            return np.vstack(blocks)

    def _analyze_pair(self, inp, out):
        pred = self._compute(inp)
        if pred is not None and np.array_equal(pred, out):
            return True
        return None
