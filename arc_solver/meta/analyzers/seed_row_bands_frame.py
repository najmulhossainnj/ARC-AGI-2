from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid

class SeedRowBandsFrameAnalyzer(Analyzer):
    """Detect single-cell seeds driving horizontal region partitioning and border framing (like 0f63c0b9)."""
    name = "seed_row_bands_frame"
    priority = 15

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            pred = self._transform(inp)
            if pred is None or not np.array_equal(pred, out):
                return None

        return ProgramCandidate(
            op="SEED_ROW_BANDS_FRAME",
            params=(),
            description="Partition grid by seed row midpoints and draw border frame per band"
        )

    def _transform(self, inp):
        h, w = inp.shape
        seeds = np.argwhere(inp != 0)
        if len(seeds) < 2:
            return None
            
        seeds = seeds[np.argsort(seeds[:, 0])]
        out = np.zeros_like(inp)
        n_seeds = len(seeds)
        
        r_starts = [0] * n_seeds
        r_ends = [h - 1] * n_seeds
        
        for i in range(n_seeds - 1):
            mid = (seeds[i][0] + seeds[i + 1][0]) // 2
            r_ends[i] = mid
            r_starts[i + 1] = mid + 1
            
        for i in range(n_seeds):
            r_seed, c_seed = seeds[i]
            c = inp[r_seed, c_seed]
            rs, re = r_starts[i], r_ends[i]
            
            out[rs:re + 1, 0] = c
            out[rs:re + 1, w - 1] = c
            out[r_seed, :] = c
            if i == 0:
                out[0, :] = c
            if i == n_seeds - 1:
                out[h - 1, :] = c
                
        return out
