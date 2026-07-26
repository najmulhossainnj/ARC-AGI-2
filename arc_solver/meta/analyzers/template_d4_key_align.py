from __future__ import annotations
import numpy as np
from scipy.ndimage import label as _label, binary_dilation as _binary_dilation
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid

class TemplateD4KeyAlignAnalyzer(Analyzer):
    """Detect D4-transformed template alignment onto target key dots and deletion of unaligned background (like 0e206a2e)."""
    name = "template_d4_key_align"
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
            op="TEMPLATE_D4_KEY_ALIGN",
            params=(),
            description="Align D4-transformed template onto target key dots and keep only matched objects"
        )

    def _transform(self, inp):
        h, w = inp.shape
        struct = np.ones((3, 3), dtype=int)
        lbl, num = _label(inp != 0, structure=struct)
        if num < 2:
            return None

        comps = []
        for i in range(1, num + 1):
            mask = (lbl == i)
            rs, cs = np.where(mask)
            r1, r2, c1, c2 = rs.min(), rs.max(), cs.min(), cs.max()
            crop = inp[r1:r2 + 1, c1:c2 + 1].copy()
            crop[~mask[r1:r2 + 1, c1:c2 + 1]] = 0
            comps.append(crop)

        out = np.zeros_like(inp)
        templates = [c for c in comps if (c != 0).sum() >= 4]
        if not templates:
            return None

        dots_mask = np.zeros_like(inp, dtype=bool)
        for i in range(1, num + 1):
            if (lbl == i).sum() < 4:
                dots_mask |= (lbl == i)

        if not dots_mask.any():
            return None

        expanded_dots = _binary_dilation(dots_mask, iterations=4)
        target_lbl, n_t = _label(expanded_dots, structure=struct)

        for t_id in range(1, n_t + 1):
            t_mask = (target_lbl == t_id) & dots_mask
            t_dots = np.argwhere(t_mask)
            if len(t_dots) == 0:
                continue
            t_colors = [inp[r, c] for r, c in t_dots]

            matched_group = False
            for tmpl in templates:
                tmpl_colors = set(tmpl[tmpl != 0])
                if set(t_colors).issubset(tmpl_colors):
                    for k in range(4):
                        for flip in [False, True]:
                            t_trans = np.rot90(tmpl, k)
                            if flip:
                                t_trans = np.fliplr(t_trans)

                            for dot_idx in range(len(t_dots)):
                                cdot = t_colors[dot_idx]
                                dots_in_trans = np.argwhere(t_trans == cdot)
                                for r_tr, c_tr in dots_in_trans:
                                    r_target, c_target = t_dots[dot_idx]
                                    top_r = r_target - r_tr
                                    top_c = c_target - c_tr
                                    th, tw = t_trans.shape

                                    if 0 <= top_r and top_r + th <= h and 0 <= top_c and top_c + tw <= w:
                                        valid = True
                                        for r_td, c_td in t_dots:
                                            rel_r = r_td - top_r
                                            rel_c = c_td - top_c
                                            if not (0 <= rel_r < th and 0 <= rel_c < tw and t_trans[rel_r, rel_c] == inp[r_td, c_td]):
                                                valid = False
                                                break
                                        if valid:
                                            out[top_r:top_r + th, top_c:top_c + tw] = np.where(
                                                t_trans != 0, t_trans, out[top_r:top_r + th, top_c:top_c + tw]
                                            )
                                            matched_group = True
                                            break
                                if matched_group:
                                    break
                            if matched_group:
                                break
                        if matched_group:
                            break
                    if matched_group:
                        break

        return out
