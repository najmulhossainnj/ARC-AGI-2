from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid

class TileDiagonalMarkAnalyzer(Analyzer):
    """Detect 2x2 tile repetition where non-background seeds are marked at diagonal 1-step neighbors with a specific color."""
    name = "tile_diagonal_mark"
    priority = 18

    def analyze(self, train_pairs, features):
        mark_colors = []
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            h, w = inp.shape
            if out.shape != (2 * h, 2 * w):
                return None
            tiled = np.tile(inp, (2, 2))
            # Check what colors are in out but not in tiled
            diff_mask = (out != tiled)
            if not diff_mask.any():
                return None
            mark_vals = np.unique(out[diff_mask])
            if len(mark_vals) != 1:
                return None
            mark_c = int(mark_vals[0])
            
            # Predict and verify
            pred = tiled.copy()
            dots = np.argwhere(tiled != 0)
            for r, c in dots:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 2 * h and 0 <= nc < 2 * w and pred[nr, nc] == 0:
                        pred[nr, nc] = mark_c
            if not np.array_equal(pred, out):
                return None
            mark_colors.append(mark_c)
            
        if not mark_colors or len(set(mark_colors)) != 1:
            return None
            
        mark_color = mark_colors[0]
        return ProgramCandidate(
            op="TILE_2X2_DIAGONAL_MARK",
            params=(mark_color,),
            description=f"Tile 2x2 and mark diagonal neighbors of non-bg cells with color {mark_color}"
        )
