from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid


class TranslationAnalyzer(Analyzer):
    """Detect uniform translation of all non-background cells by same (dr, dc)."""
    name = "translation"
    priority = 10

    def analyze(self, train_pairs, features):
        shifts = []
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            in_pts = list(map(tuple, np.argwhere(inp != 0).tolist()))
            out_pts = set(map(tuple, np.argwhere(out != 0).tolist()))
            if not in_pts or len(in_pts) != len(out_pts):
                return None
            # Try shift from first point
            r0, c0 = in_pts[0]
            for dr_cand in range(-inp.shape[0], inp.shape[0]):
                for dc_cand in range(-inp.shape[1], inp.shape[1]):
                    nr, nc = r0 + dr_cand, c0 + dc_cand
                    if (nr, nc) not in out_pts:
                        continue
                    # Verify all points
                    shifted = {(r + dr_cand, c + dc_cand) for r, c in in_pts}
                    if shifted == out_pts:
                        shifts.append((dr_cand, dc_cand))
                        break
                else:
                    continue
                break
            else:
                return None
        if not shifts or len(set(shifts)) != 1:
            return None
        dr, dc = shifts[0]
        return ProgramCandidate(
            op="TRANSLATE",
            params=(int(dr), int(dc)),
            description=f"Translate all objects by dr={dr}, dc={dc}",
        )
