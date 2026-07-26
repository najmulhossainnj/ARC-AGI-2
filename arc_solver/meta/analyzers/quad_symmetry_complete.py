from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid

class QuadSymmetryCompleteAnalyzer(Analyzer):
    """Detect 4-way (horizontal + vertical) symmetry completion around content centroid/bbox center (like 11852cab)."""
    name = "quad_symmetry_complete"
    priority = 10

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            pred = self._transform(inp)
            if pred is None or not np.array_equal(pred, out):
                return None

        return ProgramCandidate(
            op="QUAD_SYMMETRY_COMPLETE",
            params=(),
            description="Complete 4-way symmetry around content center"
        )

    def _transform(self, inp):
        h, w = inp.shape
        pts = np.argwhere(inp != 0)
        if len(pts) == 0:
            return None
        r_center = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
        c_center = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
        
        out = inp.copy()
        for r, c in pts:
            color = inp[r, c]
            dr = r - r_center
            dc = c - c_center
            for sign_r in [1, -1]:
                for sign_c in [1, -1]:
                    nr = int(round(r_center + sign_r * dr))
                    nc = int(round(c_center + sign_c * dc))
                    if 0 <= nr < h and 0 <= nc < w:
                        out[nr, nc] = color
        return out
