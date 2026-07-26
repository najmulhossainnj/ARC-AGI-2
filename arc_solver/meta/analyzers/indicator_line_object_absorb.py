from __future__ import annotations
import numpy as np
from scipy.ndimage import label as _label
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid

class IndicatorLineObjectAbsorbAnalyzer(Analyzer):
    """Detect aligned indicator dots that draw connecting lines and recolor touched objects (like 0d87d2a6)."""
    name = "indicator_line_object_absorb"
    priority = 12

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            pred = self._transform(inp)
            if pred is None or not np.array_equal(pred, out):
                return None
                
        return ProgramCandidate(
            op="INDICATOR_LINE_OBJECT_ABSORB",
            params=(),
            description="Connect aligned indicator dots and absorb touched objects"
        )

    def _transform(self, inp):
        h, w = inp.shape
        out = inp.copy()
        
        # Indicator dots: find non-bg color that appears in aligned pairs
        colors = set(np.unique(inp)) - {0}
        ind_c = None
        for c in colors:
            pts = np.argwhere(inp == c)
            if len(pts) >= 2:
                # check if any two share row or col
                rs = [p[0] for p in pts]
                cs = [p[1] for p in pts]
                if len(set(rs)) < len(rs) or len(set(cs)) < len(cs):
                    ind_c = c
                    break
        if ind_c is None:
            return None
            
        pts = np.argwhere(inp == ind_c)
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[i]
                r2, c2 = pts[j]
                if r1 == r2:
                    for c_ in range(min(c1, c2), max(c1, c2) + 1):
                        if out[r1, c_] == 0:
                            out[r1, c_] = ind_c
                if c1 == c2:
                    for r_ in range(min(r1, r2), max(r1, r2) + 1):
                        if out[r_, c1] == 0:
                            out[r_, c1] = ind_c
                            
        target_colors = colors - {ind_c}
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        for tc in target_colors:
            lbl, num = _label(inp == tc, structure=struct)
            for obj_id in range(1, num + 1):
                obj_mask = (lbl == obj_id)
                touch = False
                rs, cs = np.where(obj_mask)
                for r, c in zip(rs, cs):
                    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == ind_c:
                            touch = True
                            break
                    if touch:
                        break
                if touch:
                    out[obj_mask] = ind_c
        return out
