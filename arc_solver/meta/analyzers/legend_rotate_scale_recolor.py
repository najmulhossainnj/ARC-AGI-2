from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid

class LegendRotateScaleRecolorAnalyzer(Analyzer):
    """Detect legend rotated by 270 deg and block-scale recolored onto target structure (like 103eff5b)."""
    name = "legend_rotate_scale_recolor"
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
            op="LEGEND_ROTATE_SCALE_RECOLOR",
            params=(),
            description="Rotate legend 270 deg and block-scale recolor target grey structure"
        )

    def _transform(self, inp, target_color=8):
        h, w = inp.shape
        legend_mask = (inp != 0) & (inp != target_color)
        if not legend_mask.any():
            return None
        l_rs, l_cs = np.where(legend_mask)
        lr1, lr2 = l_rs.min(), l_rs.max()
        lc1, lc2 = l_cs.min(), l_cs.max()
        legend = inp[lr1:lr2 + 1, lc1:lc2 + 1]
        
        target_mask = (inp == target_color)
        if not target_mask.any():
            return None
        rs, cs = np.where(target_mask)
        r1, r2 = rs.min(), rs.max()
        c1, c2 = cs.min(), cs.max()
        gh, gw = r2 - r1 + 1, c2 - c1 + 1
        
        # Rotate 270 deg (np.rot90 k=3)
        legend_rot = np.rot90(legend, 3)
        lh, lw = legend_rot.shape
        if gh % lh != 0 or gw % lw != 0:
            return None
            
        scale_h = gh // lh
        scale_w = gw // lw
        
        out = inp.copy()
        for lr in range(lh):
            for lc in range(lw):
                color = legend_rot[lr, lc]
                if color != 0:
                    sub_r1 = r1 + lr * scale_h
                    sub_r2 = r1 + (lr + 1) * scale_h
                    sub_c1 = c1 + lc * scale_w
                    sub_c2 = c1 + (lc + 1) * scale_w
                    
                    b_mask = (inp[sub_r1:sub_r2, sub_c1:sub_c2] == target_color)
                    out[sub_r1:sub_r2, sub_c1:sub_c2][b_mask] = color
                    
        return out
