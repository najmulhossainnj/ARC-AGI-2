from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid

class SeedSurroundRuleAnalyzer(Analyzer):
    """Detect single-cell seed colors that get surrounded by specific colors in orthogonal or diagonal patterns (like 0ca9ddb6)."""
    name = "seed_surround_rule"
    priority = 10

    def analyze(self, train_pairs, features):
        rules = []  # tuple of (seed_color, shape_mode, surround_color)
        
        # Check first pair to propose rules
        inp0, out0 = as_grid(train_pairs[0][0]), as_grid(train_pairs[0][1])
        if inp0.shape != out0.shape:
            return None
            
        h, w = inp0.shape
        diff_mask = (inp0 != out0) & (out0 != 0) & (inp0 == 0)
        if not diff_mask.any():
            return None
            
        # For each unique newly added color in out0
        new_colors = set(np.unique(out0[diff_mask]))
        seed_colors = set(np.unique(inp0)) - {0}
        
        candidate_rules = []
        for sc in seed_colors:
            s_pts = set(map(tuple, np.argwhere(inp0 == sc).tolist()))
            for nc in new_colors:
                # Test ORTHOGONAL
                ortho_pts = set()
                for r, c in s_pts:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc_ = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc_ < w and inp0[nr, nc_] == 0:
                            ortho_pts.add((nr, nc_))
                            
                out_nc_pts = set(map(tuple, np.argwhere((out0 == nc) & (inp0 == 0)).tolist()))
                if ortho_pts and ortho_pts == out_nc_pts:
                    candidate_rules.append((int(sc), "ORTHOGONAL", int(nc)))
                    continue
                    
                # Test DIAGONAL
                diag_pts = set()
                for r, c in s_pts:
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc_ = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc_ < w and inp0[nr, nc_] == 0:
                            diag_pts.add((nr, nc_))
                            
                if diag_pts and diag_pts == out_nc_pts:
                    candidate_rules.append((int(sc), "DIAGONAL", int(nc)))
                    
        if not candidate_rules:
            return None
            
        # Verify candidate rules across all train pairs
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            pred = inp.copy()
            h_i, w_i = inp.shape
            for sc, mode, nc in candidate_rules:
                for r, c in np.argwhere(inp == sc):
                    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] if mode == "ORTHOGONAL" else [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    for dr, dc in dirs:
                        nr, nc_ = r + dr, c + dc
                        if 0 <= nr < h_i and 0 <= nc_ < w_i and pred[nr, nc_] == 0:
                            pred[nr, nc_] = nc
            if not np.array_equal(pred, out):
                return None
                
        return ProgramCandidate(
            op="SEED_SURROUND_MARK",
            params=(tuple(candidate_rules),),
            description=f"Surround seed colors according to rules: {candidate_rules}"
        )
