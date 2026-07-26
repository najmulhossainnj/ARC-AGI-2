"""
advanced_analyzers.py
---------------------
Generalized rule-based analyzers derived from studying GitMonsters solved tasks.

These capture BROAD transformation classes — NOT task-specific hardcoding.
Each analyzer tests whether a general pattern applies to the training pairs
and returns a ProgramCandidate that the executor can run.

Techniques from GitMonsters analysis:
  1. ConcentricRingFillAnalyzer          — 13e47133: Chebyshev distance fill
  2. LegendRaySlideAnalyzer              — 88e364bc / 21897d95: legend → direction → slide
  3. NetworkConnectivityFillAnalyzer     — 8b7bacbf: dot→chain→frame fill
  4. PuzzleStitchAssemblyAnalyzer        — 4e34c42c: overlapping fragment stitching
  5. MultiEdgeGravityAnalyzer            — 62593bfd: top/bottom projection with collision
  6. ObjectStampRuleAnalyzer             — abc82100 / a32d8b75: stamp from color-chain rule
  7. RayCollisionDeflectionAnalyzer      — e12f9a14: ray emission + deflection on collision
  8. TArrowMarkerFlowAnalyzer            — 21897d95: T-shaped arrow → color region flow
  9. MasterTemplateInpaintAnalyzer       — 269e22fb: D4-symmetric template completion
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import deque, Counter
from .base import Analyzer, ProgramCandidate


# ─── Utility helpers ──────────────────────────────────────────────────────────

def _bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros((h, w), bool)
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity == 8:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                comp, q = [], deque([(r,c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and mask[nr,nc] and not visited[nr,nc]:
                            visited[nr,nc] = True
                            q.append((nr,nc))
                comps.append(comp)
    return comps


def _bbox(cells: List[Tuple[int,int]]) -> Tuple[int,int,int,int]:
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs), max(cs)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Concentric Ring Fill (13e47133-style)
#    Pattern: grid divided into sub-regions by divider lines or borders.
#    Interior cells get filled with concentric Chebyshev-distance color bands.
# ─────────────────────────────────────────────────────────────────────────────
class ConcentricRingFillAnalyzer(Analyzer):
    """
    Detect: enclosed rectangular sub-regions filled with concentric Chebyshev-
    distance color bands (same color at same distance from boundary, cycling).
    Input has only the border cells + seed dots; output fills the interior.
    """
    name = "concentric_ring_fill"
    priority = 30

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None
        # Interior cells must be bg in input, non-bg in output
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            bg = _bg(inp)
            changed = (inp != out)
            if not changed.any():
                return None
            # Changed cells should be bg→filled (not filled→bg)
            if not np.all(inp[changed] == bg):
                return None
            # Verify concentric pattern: for each changed cell, distance to
            # nearest non-bg input cell should map 1-to-1 with output color
            non_bg_mask = (inp != bg)
            if not non_bg_mask.any():
                return None
            from scipy.ndimage import distance_transform_cdt
            dist = distance_transform_cdt(~non_bg_mask, metric='chessboard')
            dist_to_color: Dict[int, int] = {}
            ok = True
            for r, c in zip(*np.where(changed)):
                d = int(dist[r, c])
                col = int(out[r, c])
                if d in dist_to_color and dist_to_color[d] != col:
                    ok = False
                    break
                dist_to_color[d] = col
            if not ok or not dist_to_color:
                return None

        return ProgramCandidate(
            op="CONCENTRIC_RING_FILL",
            params=(),
            description="Fill region interiors with concentric Chebyshev-distance color bands",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Legend-Guided Ray Slide (88e364bc / 21897d95-style)
#    Pattern: bordered template rectangles encode direction vectors.
#    Target marker cells slide in the decoded direction until hitting a wall.
# ─────────────────────────────────────────────────────────────────────────────
class LegendRaySlideAnalyzer(Analyzer):
    """
    Detect: template/legend rectangles encode motion direction (e.g. offset of
    a marker within the template bounding box). A target dot slides in that
    direction inside its container until it hits a non-bg wall.
    """
    name = "legend_ray_slide"
    priority = 28

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None
        # Only a small fraction of cells change (just the markers slide)
        avg_diff = features.get("avg_diff_frac", 1.0)
        if avg_diff > 0.20:
            return None

        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._detect_slide(inp, out):
                return None

        return ProgramCandidate(
            op="LEGEND_RAY_SLIDE",
            params=(),
            description="Decode direction from legend template; slide target marker to wall",
        )

    def _detect_slide(self, inp, out) -> bool:
        bg = _bg(inp)
        diff = inp != out
        if not diff.any():
            return False
        # Small number of cells change
        if diff.sum() > inp.size * 0.15:
            return False
        # Some cells lost color (marker departed) and some cells gained same color (marker arrived)
        lost  = diff & (out == bg)
        gained = diff & (inp == bg)
        if not lost.any() or not gained.any():
            return False
        lost_cols  = set(int(v) for v in np.unique(inp[lost]))  - {bg}
        gained_cols = set(int(v) for v in np.unique(out[gained])) - {bg}
        # At least one color moved (same color appears in both lost and gained)
        return bool(lost_cols & gained_cols)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Network Connectivity Fill (8b7bacbf-style)
#    Pattern: seed dots + colored chain segments form a graph.
#    Enclosed frame interiors get filled with the connected dot's color.
# ─────────────────────────────────────────────────────────────────────────────
class NetworkConnectivityFillAnalyzer(Analyzer):
    """
    Detect: isolated seed dots connected via single-color chain cells to enclosed
    background frames. Each enclosed region is filled with the reachable dot's color.
    """
    name = "network_connectivity_fill"
    priority = 25

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None
        n_colors = features.get("n_colors_in", 0)
        if n_colors < 3:  # Need bg + chain + at least 2 dot colors
            return None

        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._detect(inp, out):
                return None

        return ProgramCandidate(
            op="NETWORK_CONNECTIVITY_FILL",
            params=(),
            description="Fill enclosed frame interiors by tracing chain connections from seed dots",
        )

    def _detect(self, inp, out) -> bool:
        bg = _bg(inp)
        h, w = inp.shape
        # Cells that changed: bg → color
        changed = (inp != out) & (inp == bg)
        if not changed.any():
            return False
        # New colors must come from existing input colors
        new_colors = set(int(v) for v in np.unique(out[changed])) - {bg}
        in_colors  = set(int(v) for v in np.unique(inp))         - {bg}
        if not new_colors.issubset(in_colors):
            return False
        # There must be isolated "dot" cells in input (components of size 1)
        comps = _components(inp != bg, connectivity=4)
        isolated = [c for c in comps if len(c) == 1]
        if not isolated:
            return False
        # The changed (filled) area should be large and form enclosed islands
        if changed.sum() < 4:
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Puzzle Stitch Assembly (4e34c42c-style)
#    Pattern: multiple disconnected fragment shapes → reassemble into single
#    composite by matching overlapping color edges.
# ─────────────────────────────────────────────────────────────────────────────
class PuzzleStitchAssemblyAnalyzer(Analyzer):
    """
    Detect: input contains several disconnected shape fragments that must be
    assembled (stitched) into a single tight composite object. Output is smaller
    or repositioned with fragments combined.
    """
    name = "puzzle_stitch_assembly"
    priority = 35

    def analyze(self, train_pairs, features):
        # Output can be different size from input
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._detect(inp, out):
                return None

        return ProgramCandidate(
            op="PUZZLE_STITCH_ASSEMBLY",
            params=(),
            description="Stitch disconnected shape fragments into a single composite object",
        )

    def _detect(self, inp, out) -> bool:
        bg_in  = _bg(inp)
        bg_out = _bg(out)
        comps_in = _components(inp != bg_in, connectivity=4)
        comps_out = _components(out != bg_out, connectivity=4)
        # Input should have multiple fragments; output should have fewer (ideally 1)
        if len(comps_in) < 3:
            return False
        if len(comps_out) >= len(comps_in):
            return False
        # Output should have similar total non-bg cell count (just rearranged)
        n_in  = int((inp != bg_in).sum())
        n_out = int((out != bg_out).sum())
        ratio = n_out / max(n_in, 1)
        if ratio < 0.5 or ratio > 2.0:
            return False
        # Output bounding box should be smaller than input bounding box
        if out.shape[0] * out.shape[1] >= inp.shape[0] * inp.shape[1] * 0.95:
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# 5. Multi-Edge Gravity / Vertical Projection (62593bfd-style)
#    Pattern: objects project to top OR bottom edge based on centroid position.
#    More sophisticated than simple GravityFallAnalyzer: per-object direction.
# ─────────────────────────────────────────────────────────────────────────────
class MultiEdgeGravityAnalyzer(Analyzer):
    """
    Detect: objects move to either the top or bottom edge based on each object's
    vertical centroid position. Different objects may go in different directions.
    """
    name = "multi_edge_gravity"
    priority = 22

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._detect(inp, out):
                return None

        return ProgramCandidate(
            op="MULTI_EDGE_GRAVITY",
            params=(),
            description="Project objects to top or bottom edge based on centroid position",
        )

    def _detect(self, inp, out) -> bool:
        bg = _bg(inp)
        h = inp.shape[0]
        comps = _components(inp != bg, connectivity=8)
        if len(comps) < 2:
            return False

        snapped = 0
        for comp in comps:
            rs, cs = zip(*comp)
            centroid_r = sum(rs) / len(rs)
            color = int(inp[rs[0], cs[0]])
            if color == bg:
                continue
            # Check if this color ends up at the top or bottom in output
            out_positions = [(r, c) for r in range(h) for c in range(inp.shape[1])
                             if out[r, c] == color]
            if not out_positions:
                continue
            out_rs = [r for r, c in out_positions]
            out_min_r = min(out_rs)
            out_max_r = max(out_rs)
            if out_min_r == 0 or out_max_r == h - 1:
                snapped += 1

        return snapped >= 2  # At least 2 objects snapped to edges


# ─────────────────────────────────────────────────────────────────────────────
# 6. Object Stamp Rule (abc82100 / a32d8b75-style)
#    Pattern: 8-connected clusters define stamp shapes; 2-cell color-chains
#    define (source→target) color mapping rules; stamps placed at seed cells.
# ─────────────────────────────────────────────────────────────────────────────
class ObjectStampRuleAnalyzer(Analyzer):
    """
    Detect: stamp templates encoded in one region; color-chain or panel rules
    define where and in what color to stamp them. Output is stamps placed at
    seed cell locations.
    """
    name = "object_stamp_rule"
    priority = 28

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None
        # Stamp ops tend to change a moderate fraction of cells
        avg_diff = features.get("avg_diff_frac", 0.0)
        if avg_diff < 0.03:
            return None

        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._detect_stamp(inp, out):
                return None

        return ProgramCandidate(
            op="OBJECT_STAMP_RULE",
            params=(),
            description="Apply stamp shapes at seed locations using color-chain-defined rules",
        )

    def _detect_stamp(self, inp, out) -> bool:
        bg = _bg(inp)
        diff = inp != out
        if not diff.any():
            return False
        # Look for 2-cell same-row/col components with different colors in input
        # (color-chain indicator pattern)
        h, w = inp.shape
        chains = 0
        for r in range(h):
            for c in range(w - 1):
                if (inp[r, c] != bg and inp[r, c+1] != bg and inp[r, c] != inp[r, c+1]):
                    # Check if this 2-cell horizontal pair is isolated (neighbors are bg)
                    pair_isolated = all(
                        inp[r + dr, c + dc] == bg
                        for dr in [-1, 1] for dc in [0, 1]
                        if 0 <= r + dr < h and 0 <= c + dc < w
                    )
                    if pair_isolated:
                        chains += 1
        # Also check vertical pairs
        for r in range(h - 1):
            for c in range(w):
                if (inp[r, c] != bg and inp[r+1, c] != bg and inp[r, c] != inp[r+1, c]):
                    pair_isolated = all(
                        inp[r + dr, c + dc] == bg
                        for dr in [0, 1] for dc in [-1, 1]
                        if 0 <= r + dr < h and 0 <= c + dc < w
                    )
                    if pair_isolated:
                        chains += 1
        return chains >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Ray Collision Deflection (e12f9a14-style)
#    Pattern: bordered shapes have gaps → emit color rays from gaps outward.
#    When rays from different shapes collide, they deflect via vector addition.
# ─────────────────────────────────────────────────────────────────────────────
class RayCollisionDeflectionAnalyzer(Analyzer):
    """
    Detect: shapes with bordered outlines have gaps. Color rays emitted from those
    gaps propagate outward, deflecting when two rays collide. Output fills ray paths.
    """
    name = "ray_collision_deflection"
    priority = 35

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None
        # Ray paths create new colored cells in output
        avg_diff = features.get("avg_diff_frac", 0.0)
        if avg_diff < 0.03:
            return None

        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._detect_rays(inp, out):
                return None

        return ProgramCandidate(
            op="RAY_COLLISION_DEFLECTION",
            params=(),
            description="Emit color rays from shape border gaps; deflect on collision",
        )

    def _detect_rays(self, inp, out) -> bool:
        bg = _bg(inp)
        h, w = inp.shape
        # New colored cells in output that were bg in input
        new_cells = (out != bg) & (inp == bg)
        if not new_cells.any():
            return False
        new_colors = set(int(v) for v in np.unique(out[new_cells])) - {bg}
        inp_colors = set(int(v) for v in np.unique(inp)) - {bg}
        # Ray colors must come from input colors
        if not new_colors.issubset(inp_colors):
            return False
        # Rays form connected segments — check for horizontal or vertical runs of new cells
        for r in range(h):
            row = new_cells[r, :]
            if row.sum() >= 3:
                idxs = np.where(row)[0]
                if np.all(np.diff(idxs) == 1):
                    return True
        for c in range(w):
            col = new_cells[:, c]
            if col.sum() >= 3:
                idxs = np.where(col)[0]
                if np.all(np.diff(idxs) == 1):
                    return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 8. T-Arrow Marker Flow (21897d95-style)
#    Pattern: T-shaped 3+1 cell markers define directional flow between color
#    regions. The missing 4th arm indicates the flow direction.
# ─────────────────────────────────────────────────────────────────────────────
class TArrowMarkerFlowAnalyzer(Analyzer):
    """
    Detect: T-shaped markers (3 cells of one color, 1 center neighbor of different
    color) define flow direction. Colors propagate from source region to adjacent
    target region in the arrow direction.
    """
    name = "t_arrow_marker_flow"
    priority = 32

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None
        avg_diff = features.get("avg_diff_frac", 0.0)
        if avg_diff < 0.03 or avg_diff > 0.85:
            return None

        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._has_t_markers(inp):
                return None

        return ProgramCandidate(
            op="T_ARROW_MARKER_FLOW",
            params=(),
            description="T-shaped 3+1 markers define directional color flow between regions",
        )

    def _has_t_markers(self, grid) -> bool:
        bg = _bg(grid)
        h, w = grid.shape
        dirs4 = [(-1,0),(1,0),(0,-1),(0,1)]
        for r in range(1, h-1):
            for c in range(1, w-1):
                col = int(grid[r, c])
                if col == bg:
                    continue
                same = sum(
                    1 for dr, dc in dirs4
                    if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr, c+dc] == col
                )
                if same == 3:
                    diff_neighbors = [
                        (r+dr, c+dc) for dr, dc in dirs4
                        if 0 <= r+dr < h and 0 <= c+dc < w
                        and int(grid[r+dr, c+dc]) not in (col, bg)
                    ]
                    if diff_neighbors:
                        return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 9. Master Template Inpaint (269e22fb-style)
#    Pattern: output is always some D4 rotation/flip of a fixed global master
#    template, matched to the partial/noisy input.
# ─────────────────────────────────────────────────────────────────────────────
class MasterTemplateInpaintAnalyzer(Analyzer):
    """
    Detect: all training outputs are rotations/reflections (D4 symmetry group)
    of a single fixed master template. Input is a partial or transformed view.
    Output completes it by finding the matching D4 orientation.
    """
    name = "master_template_inpaint"
    priority = 40  # Run late; expensive

    def analyze(self, train_pairs, features):
        outputs = [np.asarray(o) for _, o in train_pairs]
        if not all(o.shape == outputs[0].shape for o in outputs):
            return None
        if len(outputs) < 2:
            return None

        # Check if all outputs are D4 variants of each other
        master = outputs[0]
        d4 = self._d4_variants(master)
        matches = 0
        for out in outputs[1:]:
            if any(np.array_equal(v, out) for v in d4):
                matches += 1
        if matches == len(outputs) - 1:
            return ProgramCandidate(
                op="MASTER_TEMPLATE_INPAINT",
                params=(),
                description="Complete input by matching against D4 variants of a master template",
            )
        return None

    def _d4_variants(self, arr: np.ndarray) -> List[np.ndarray]:
        variants = []
        a = arr.copy()
        for _ in range(4):
            variants.append(a)
            variants.append(np.fliplr(a))
            a = np.rot90(a)
        return variants
