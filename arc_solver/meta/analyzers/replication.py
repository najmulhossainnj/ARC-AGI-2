from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid


class ArrowReplicateAnalyzer(Analyzer):
    """Detect arrow-driven template replication (like task 045e512c).

    Pattern: one multi-cell 'template' object of color T stays in place.
    Other color objects are 'arrows': same-height (for RIGHT/LEFT) or
    same-width (for UP/DOWN) segments adjacent to the template.
    Their length = number of repetitions.  The template is replicated
    N times in that direction with a step = template_dim + gap.
    """

    name = "arrow_replicate"
    priority = 20

    def analyze(self, train_pairs, features):
        sig = None
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            s = self._extract_signature(inp, out)
            if s is None:
                return None
            if sig is None:
                sig = s
            elif s != sig:
                return None
        if sig is None:
            return None
        template_color, arrow_specs = sig
        return ProgramCandidate(
            op="ARROW_REPLICATE",
            params=(template_color, arrow_specs),
            description=f"Arrow replication: template={template_color} arrows={arrow_specs}",
        )

    def _extract_signature(self, inp, out):
        colors = sorted(set(int(v) for v in np.unique(inp)) - {0})
        if len(colors) < 2:
            return None

        # Identify template: largest multi-cell object that stays in same location in output
        template_color = None
        template_bbox = None
        for c in colors:
            in_pts = set(map(tuple, np.argwhere(inp == c).tolist()))
            out_pts = set(map(tuple, np.argwhere(out == c).tolist()))
            if len(in_pts) >= 3 and in_pts.issubset(out_pts):
                rs = [r for r, _ in in_pts]
                cs = [c_ for _, c_ in in_pts]
                template_bbox = (min(rs), min(cs), max(rs), max(cs))
                template_color = c
                break
        if template_color is None:
            return None

        tr1, tc1, tr2, tc2 = template_bbox
        th, tw = tr2 - tr1 + 1, tc2 - tc1 + 1

        arrow_specs = []
        for c in colors:
            if c == template_color:
                continue
            in_rs, in_cs = np.where(inp == c)
            if len(in_rs) == 0:
                continue
            ar1, ac1 = int(in_rs.min()), int(in_cs.min())
            ar2, ac2 = int(in_rs.max()), int(in_cs.max())
            arrow_h = ar2 - ar1 + 1
            arrow_w = ac2 - ac1 + 1

            direction = None
            n_copies = None
            step = None

            if ac1 > tc2 and arrow_h == th:  # RIGHT of template, same height
                direction = "RIGHT"
                gap = ac1 - tc2 - 1
                step = tw + gap + 1
                n_copies = arrow_h
            elif ac2 < tc1 and arrow_h == th:  # LEFT
                direction = "LEFT"
                gap = tc1 - ac2 - 1
                step = tw + gap + 1
                n_copies = arrow_h
            elif ar1 > tr2 and arrow_w == tw:  # DOWN
                direction = "DOWN"
                gap = ar1 - tr2 - 1
                step = th + gap + 1
                n_copies = arrow_w
            elif ar2 < tr1 and arrow_w == tw:  # UP
                direction = "UP"
                gap = tr1 - ar2 - 1
                step = th + gap + 1
                n_copies = arrow_w

            if direction:
                arrow_specs.append((int(c), direction, int(n_copies), int(step)))

        if not arrow_specs:
            return None
        return (int(template_color), tuple(sorted(arrow_specs)))
