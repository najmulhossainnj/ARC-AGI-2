from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid

class LPathDotConnectAnalyzer(Analyzer):
    """Detect 3-dot color-chained L-path connections (like 0e671a1a)."""
    name = "l_path_dot_connect"
    priority = 14

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            pred = self._transform(inp)
            if pred is None or not np.array_equal(pred, out):
                return None

        return ProgramCandidate(
            op="L_PATH_DOT_CONNECT",
            params=(),
            description="Connect 3 key dots with chained horizontal-first L-paths using fill color"
        )

    def _transform(self, inp, fill_color=5):
        h, w = inp.shape
        out = inp.copy()
        
        # Check colors
        colors = set(np.unique(inp)) - {0}
        if len(colors) != 3:
            return None
            
        p2 = np.argwhere(inp == 2)
        p4 = np.argwhere(inp == 4)
        p3 = np.argwhere(inp == 3)
        if len(p2) != 1 or len(p4) != 1 or len(p3) != 1:
            return None
            
        r2, c2 = p2[0]
        r4, c4 = p4[0]
        r3, c3 = p3[0]
        
        # p2 -> p4
        out[r2, min(c2, c4):max(c2, c4)+1] = np.where(out[r2, min(c2, c4):max(c2, c4)+1] == 0, fill_color, out[r2, min(c2, c4):max(c2, c4)+1])
        out[min(r2, r4):max(r2, r4)+1, c4] = np.where(out[min(r2, r4):max(r2, r4)+1, c4] == 0, fill_color, out[min(r2, r4):max(r2, r4)+1, c4])
        
        # p4 -> p3
        out[r4, min(c4, c3):max(c4, c3)+1] = np.where(out[r4, min(c4, c3):max(c4, c3)+1] == 0, fill_color, out[r4, min(c4, c3):max(c4, c3)+1])
        out[min(r4, r3):max(r4, r3)+1, c3] = np.where(out[min(r4, r3):max(r4, r3)+1, c3] == 0, fill_color, out[min(r4, r3):max(r4, r3)+1, c3])
        
        return out
