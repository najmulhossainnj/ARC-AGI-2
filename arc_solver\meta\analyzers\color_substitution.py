from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid


class ColorSubstitutionAnalyzer(Analyzer):
    """Detect consistent color-to-color remapping across all train pairs."""
    name = "color_substitution"
    priority = 5

    def analyze(self, train_pairs, features):
        mapping = {}
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            for v in np.unique(inp):
                mask = (inp == v)
                out_vals = np.unique(out[mask])
                if len(out_vals) != 1:
                    return None
                ov, cv = int(out_vals[0]), int(v)
                if cv in mapping and mapping[cv] != ov:
                    return None
                mapping[cv] = ov
        nontrivial = {k: v for k, v in mapping.items() if k != v}
        if not nontrivial:
            return None
        return ProgramCandidate(
            op="COLORMAP",
            params=(tuple(sorted(nontrivial.items())),),
            description=f"Color substitution: {nontrivial}",
        )
