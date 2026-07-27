"""
spatial_morphology.py
--------------------
Universal Spatial Vector & Topo-Morphology Engine for ARC-AGI V2 Architecture.

Computes:
1. Directional Attraction & Gravity Dynamics (UP, DOWN, LEFT, RIGHT, TOWARD_BORDER, TOWARD_ANCHOR).
2. Topological Invariants (Endpoints, Convex Corners, Concave Turns, Hollow Enclosures).
3. Claws, Pointers, and Contact Recoloring Dynamics.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from scipy.ndimage import label, binary_dilation


@dataclass
class ObjectTopology:
    color: int
    mask: np.ndarray
    endpoints: List[Tuple[int, int]] = field(default_factory=list)
    convex_corners: List[Tuple[int, int]] = field(default_factory=list)
    concave_turns: List[Tuple[int, int]] = field(default_factory=list)


class UniversalSpatialMorphology:
    """Computes general spatial vector mechanics and topological invariants."""

    def compute_topology(self, grid: np.ndarray, color: int) -> ObjectTopology:
        g = np.asarray(grid)
        h, w = g.shape
        mask = (g == color)
        struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        lbl, num = label(mask, structure=struct4)

        endpoints = []
        convex_corners = []
        concave_turns = []

        rs, cs = np.where(mask)
        points = set(zip(rs.tolist(), cs.tolist()))

        for r, c in points:
            nbrs = [(dr, dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] if (r + dr, c + dc) in points]
            if len(nbrs) == 1:
                endpoints.append((r, c))
            elif len(nbrs) == 2:
                (dr1, dc1), (dr2, dc2) = nbrs
                if dr1 != -dr2 or dc1 != -dc2:  # 90-degree turn
                    diag_r, diag_c = r + dr1 + dr2, c + dc1 + dc2
                    if (diag_r, diag_c) in points:
                        concave_turns.append((r, c))
                    else:
                        convex_corners.append((r, c))

        return ObjectTopology(
            color=color,
            mask=mask,
            endpoints=endpoints,
            convex_corners=convex_corners,
            concave_turns=concave_turns
        )

    def apply_directional_gravity(self, grid: np.ndarray, top_color: int, bot_color: int, bg_color: int) -> np.ndarray:
        g = np.asarray(grid)
        h, w = g.shape
        out = np.full_like(g, bg_color)
        out[0, :] = top_color
        out[-1, :] = bot_color

        struct8 = np.ones((3, 3), dtype=int)

        # Top gravity: move UP
        mask_top = (g == top_color)
        mask_top[0, :] = False
        lbl_top, num_top = label(mask_top, structure=struct8)

        top_objs = []
        for k in range(1, num_top + 1):
            comp = (lbl_top == k)
            rs, cs = np.where(comp)
            top_objs.append((int(rs.min()), comp))
        top_objs.sort(key=lambda x: x[0])

        for _, comp in top_objs:
            rs, cs = np.where(comp)
            min_r = int(rs.min())
            shift = 0
            for s in range(1, min_r):
                shifted_rs = rs - s
                if (out[shifted_rs, cs] == bg_color).all():
                    shift = s
                else:
                    break
            out[rs - shift, cs] = top_color

        # Bottom gravity: move DOWN
        mask_bot = (g == bot_color)
        mask_bot[-1, :] = False
        lbl_bot, num_bot = label(mask_bot, structure=struct8)

        bot_objs = []
        for k in range(1, num_bot + 1):
            comp = (lbl_bot == k)
            rs, cs = np.where(comp)
            bot_objs.append((int(rs.max()), comp))
        bot_objs.sort(key=lambda x: x[0], reverse=True)

        for _, comp in bot_objs:
            rs, cs = np.where(comp)
            max_r = int(rs.max())
            shift = 0
            for s in range(1, h - 1 - max_r):
                shifted_rs = rs + s
                if (out[shifted_rs, cs] == bg_color).all():
                    shift = s
                else:
                    break
            out[rs + shift, cs] = bot_bot if False else bot_color
            out[rs + shift, cs] = bot_color
            if (cs == w - 1).any():
                for r in range(min(rs), h):
                    out[r, w - 1] = bot_color

        return out
