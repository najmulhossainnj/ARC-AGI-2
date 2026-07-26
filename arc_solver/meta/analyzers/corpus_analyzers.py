"""
corpus_analyzers.py
--------------------
High-priority generalized analyzers derived from frequency analysis of
632 solved ARC tasks in GitMonsters/SOLVED-540-of-540.

Technique frequency ordering (from analysis):
  324  pattern_match
  313  counting
  286  object_placement
  284  sort_objects
  135  grid_sections
  127  size_comparison
  126  network_chain
  115  topology
   73  diagonal
   72  tiling

These analyzers target the most frequent UNSOLVED technique families.
Each one detects a broad class, NOT a task-specific rule.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque, Counter
from .base import Analyzer, ProgramCandidate


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _components(mask: np.ndarray, connectivity: int = 4) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros((h, w), bool)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                comp, q = [], deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                comps.append(comp)
    return comps


def _bbox(cells):
    rs, cs = zip(*cells)
    return min(rs), min(cs), max(rs), max(cs)


def _norm_shape(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Count-Driven Rule Analyzer  (frequency: 313/632)
#    Pattern: the number of objects/cells of a specific color DETERMINES the
#    output — either as a count selector, a repeat count, or a size argument.
#    Most common form: "output contains N copies" where N = count of something.
# ─────────────────────────────────────────────────────────────────────────────
class CountDrivenRuleAnalyzer(Analyzer):
    """
    Detect: the count of objects or cells of a specific color in the input
    controls the output (e.g. N objects → output has N copies, N rows, N color
    bands, or the Nth object is selected).
    """
    name = "count_driven_rule"
    priority = 38

    def analyze(self, train_pairs, features):
        counts = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            bg = _bg(inp)
            comps_in = _components(inp != bg)
            n_objects = len(comps_in)
            # Check if output size, or output non-bg count, relates to input object count
            n_out_cells = int((out != _bg(out)).sum())
            n_in_cells = int((inp != bg).sum())
            counts.append((n_objects, n_out_cells, n_in_cells, inp.shape, out.shape))

        if len(counts) < 2:
            return None

        # Check 1: does output HEIGHT or WIDTH == n_objects across all pairs?
        n_objs = [c[0] for c in counts]
        out_hs = [c[3][0] if c[3] != c[4] else None for c in counts]  # height change
        if len(set(n_objs)) > 1:  # count varies across pairs
            out_heights = [c[4][0] for c in counts]
            if n_objs == out_heights:
                return ProgramCandidate(
                    op="COUNT_DRIVEN_HEIGHT",
                    params=(),
                    description="Output height equals number of input objects",
                )
            out_widths = [c[4][1] for c in counts]
            if n_objs == out_widths:
                return ProgramCandidate(
                    op="COUNT_DRIVEN_WIDTH",
                    params=(),
                    description="Output width equals number of input objects",
                )

        # Check 2: single-color isolated count drives something
        all_pairs_have_count_signal = True
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            bg = _bg(inp)
            colors = [c for c in np.unique(inp) if c != bg]
            if len(colors) < 2:
                all_pairs_have_count_signal = False
                break
            # At least one color has exactly 1–5 cells (a "counter" marker)
            small_color = any(int((inp == c).sum()) <= 5 for c in colors)
            if not small_color:
                all_pairs_have_count_signal = False
                break

        if all_pairs_have_count_signal:
            return ProgramCandidate(
                op="COUNT_DRIVEN_RULE",
                params=(),
                description="A small isolated marker count drives the transformation rule",
            )

        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sort-by-Attribute Analyzer  (frequency: 284/632)
#    Pattern: objects are reordered (left-to-right, top-to-bottom) based on
#    an attribute: size (area), color value, height/width, or row/col position.
#    Output is the same objects, rearranged.
# ─────────────────────────────────────────────────────────────────────────────
class SortByAttributeAnalyzer(Analyzer):
    """
    Detect: input contains N objects that are rearranged in the output by some
    attribute (size, color index, extent). Total cell count preserved.
    Objects maintain their shapes but change positions.
    """
    name = "sort_by_attribute"
    priority = 26

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        sort_attrs = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            bg_in = _bg(inp)
            bg_out = _bg(out)

            comps_in = _components(inp != bg_in)
            comps_out = _components(out != bg_out)

            if len(comps_in) < 2 or len(comps_in) != len(comps_out):
                return None

            # Check if shapes are preserved (same multiset of normalized shapes)
            shapes_in = sorted(_norm_shape(c) for c in comps_in)
            shapes_out = sorted(_norm_shape(c) for c in comps_out)
            if shapes_in != shapes_out:
                return None  # Shapes changed, not just rearranged

            # Determine sort attribute from output position order
            # Sort by size: check if out order corresponds to sorted-by-len of in shapes
            sizes_in = sorted(len(c) for c in comps_in)
            sizes_out = [len(c) for c in sorted(comps_out, key=lambda c: min(cc for cc, _ in c))]

            if sizes_out == sizes_in or sizes_out == sizes_in[::-1]:
                sort_attrs.append('size')
            else:
                # Sort by color?
                cols_in = sorted(int(inp[c[0][0], c[0][1]]) for c in comps_in)
                cols_out_ordered = [int(out[c[0][0], c[0][1]]) for c in
                                    sorted(comps_out, key=lambda c: min(cc for cc, _ in c))]
                if sorted(cols_out_ordered) == cols_in or sorted(cols_out_ordered) == cols_in[::-1]:
                    sort_attrs.append('color')
                else:
                    sort_attrs.append('position')

        if not sort_attrs:
            return None

        dominant = Counter(sort_attrs).most_common(1)[0][0]
        return ProgramCandidate(
            op="SORT_OBJECTS",
            params=(dominant,),
            description=f"Sort objects by {dominant} and rearrange in grid",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Size-Based Selection Analyzer  (frequency: 127/632)
#    Pattern: keep only the largest (or smallest) object. Everything else
#    becomes background. Very common "filter" operation.
# ─────────────────────────────────────────────────────────────────────────────
class SizeSelectionAnalyzer(Analyzer):
    """
    Detect: output retains exactly one object (largest or smallest by cell count).
    All other objects/colors become background. A fundamental "filter" transform.
    """
    name = "size_selection"
    priority = 8  # Fast and specific — run early

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        selections = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            bg_in = _bg(inp)
            bg_out = _bg(out)

            comps_in = _components(inp != bg_in)
            comps_out = _components(out != bg_out)

            if len(comps_in) < 2:
                return None
            if len(comps_out) != 1:
                return None

            # The one output component should match input's largest or smallest
            out_shape = _norm_shape(comps_out[0])
            sizes = [(len(c), _norm_shape(c)) for c in comps_in]

            largest = max(sizes, key=lambda x: x[0])
            smallest = min(sizes, key=lambda x: x[0])

            if out_shape == largest[1]:
                selections.append('largest')
            elif out_shape == smallest[1]:
                selections.append('smallest')
            else:
                return None  # Doesn't match size selection

        if not selections:
            return None
        if len(set(selections)) != 1:
            return None

        which = selections[0]
        
        def make_size_select_fn(which_type):
            def solve_fn(inp_grid):
                inp = np.asarray(inp_grid)
                bg = _bg(inp)
                comps = _components(inp != bg)
                if not comps:
                    return inp.tolist()
                target_comp = max(comps, key=len) if which_type == 'largest' else min(comps, key=len)
                out = np.full_like(inp, bg)
                for r, c in target_comp:
                    out[r, c] = inp[r, c]
                return out.tolist()
            return solve_fn

        return ProgramCandidate(
            op="SIZE_SELECTION",
            params=(which,),
            description=f"Keep only the {which} object; fill rest with background",
            solve_fn=make_size_select_fn(which)
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Topology / Hole-Count Analyzer  (frequency: 115/632)
#    Pattern: the transformation depends on topological properties of objects:
#    - Number of holes (enclosed bg regions) inside an object
#    - Nesting depth (objects inside objects)
#    - Euler number / genus
# ─────────────────────────────────────────────────────────────────────────────
class TopologyHoleAnalyzer(Analyzer):
    """
    Detect: the rule depends on how many holes (enclosed background regions) each
    object contains. Objects are colored/selected/counted based on hole count.
    """
    name = "topology_hole"
    priority = 34

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._topology_signal(inp, out):
                return None

        return ProgramCandidate(
            op="TOPOLOGY_HOLE_COUNT",
            params=(),
            description="Color or select objects based on their topological hole count",
        )

    def _count_holes(self, mask: np.ndarray) -> int:
        """Count enclosed background regions inside a binary mask."""
        h, w = mask.shape
        # Flood fill from ALL border bg cells → exterior
        exterior = np.zeros((h, w), bool)
        q = deque()
        for r in range(h):
            for c in [0, w - 1]:
                if not mask[r, c] and not exterior[r, c]:
                    exterior[r, c] = True
                    q.append((r, c))
        for c in range(w):
            for r in [0, h - 1]:
                if not mask[r, c] and not exterior[r, c]:
                    exterior[r, c] = True
                    q.append((r, c))
        dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            r, c = q.popleft()
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                    exterior[nr, nc] = True
                    q.append((nr, nc))
        # Holes = bg cells NOT reachable from border
        holes_mask = ~mask & ~exterior
        hole_comps = _components(holes_mask)
        return len(hole_comps)

    def _topology_signal(self, inp, out) -> bool:
        bg = _bg(inp)
        comps = _components(inp != bg)
        if len(comps) < 2:
            return False
        # Check if objects have different hole counts
        hole_counts = []
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            sub = (inp[r0:r1+1, c0:c1+1] != bg)
            holes = self._count_holes(sub)
            hole_counts.append(holes)
        # Topology signal: objects have varying hole counts (0,1,2...)
        if len(set(hole_counts)) > 1:
            return True
        # Or: output colors differ per object in input (hole-driven recoloring)
        out_colors = set(int(out[r, c]) for r, c in comps[0] if int(out[r, c]) != bg)
        in_colors = set(int(inp[r, c]) for r, c in comps[0])
        return out_colors != in_colors


# ─────────────────────────────────────────────────────────────────────────────
# 5. Grid Section / Legend Analyzer  (frequency: 135/632)
#    Pattern: the grid is divided by separator rows/columns into sections:
#    - "Key" section: encodes the rule (color→shape, color→direction, etc.)
#    - "Puzzle" section: the region to transform using the decoded rule
# ─────────────────────────────────────────────────────────────────────────────
class GridSectionLegendAnalyzer(Analyzer):
    """
    Detect: a separator (row or column of a specific color) divides the grid
    into a legend/key section and a puzzle section. The key is decoded and
    applied to the puzzle.
    """
    name = "grid_section_legend"
    priority = 20

    def analyze(self, train_pairs, features):
        seps_found = []
        for inp, out in train_pairs:
            inp = np.asarray(inp)
            sep = self._find_separator(inp)
            if sep is None:
                return None
            seps_found.append(sep)
        if not seps_found:
            return None
        # Check separator is consistent (same axis, similar position)
        axes = [s[0] for s in seps_found]
        if len(set(axes)) != 1:
            return None
        return ProgramCandidate(
            op="GRID_SECTION_LEGEND",
            params=(axes[0],),
            description=f"Decode legend from {axes[0]}-separated section; apply rule to puzzle section",
        )

    def _find_separator(self, grid):
        h, w = grid.shape
        # Horizontal separator: row where all cells are one specific non-bg color
        for r in range(h):
            row_vals = set(int(v) for v in grid[r, :])
            if len(row_vals) == 1 and r not in (0, h - 1):
                return ('horizontal', r, int(grid[r, 0]))
        # Vertical separator: column where all cells are one specific color
        for c in range(w):
            col_vals = set(int(v) for v in grid[:, c])
            if len(col_vals) == 1 and c not in (0, w - 1):
                return ('vertical', c, int(grid[0, c]))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 6. Diagonal Pattern Analyzer  (frequency: 73/632)
#    Pattern: objects or fills arranged diagonally. The rule propagates along
#    diagonals: (r+c) % N or (r-c) % N defines color bands or object placement.
# ─────────────────────────────────────────────────────────────────────────────
class DiagonalPatternAnalyzer(Analyzer):
    """
    Detect: output fills cells where (r+c) % period == k for some color k,
    or objects are positioned along a diagonal trajectory.
    """
    name = "diagonal_pattern"
    priority = 22

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        periods = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            bg = _bg(out)
            changed = (inp != out)
            if not changed.any():
                return None
            p = self._detect_diagonal_period(out, bg)
            if p is None:
                return None
            periods.append(p)

        if not periods or len(set(periods)) > 2:
            return None

        return ProgramCandidate(
            op="DIAGONAL_PATTERN",
            params=(periods[0],),
            description=f"Diagonal color bands: (r+c) % {periods[0]} determines color",
        )

    def _detect_diagonal_period(self, out, bg):
        h, w = out.shape
        # Sample non-bg cells and check if (r+c) % p → color is consistent
        non_bg = [(r, c) for r in range(h) for c in range(w) if out[r, c] != bg]
        if len(non_bg) < 6:
            return None
        for period in range(2, min(h + w, 12)):
            diag_to_color = {}
            ok = True
            for r, c in non_bg[:50]:
                key = (r + c) % period
                col = int(out[r, c])
                if key in diag_to_color and diag_to_color[key] != col:
                    ok = False
                    break
                diag_to_color[key] = col
            if ok and len(diag_to_color) >= 2:
                return period
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 7. Unique Object Extractor  (common sub-pattern across many categories)
#    Pattern: one object in the grid is "unique" in some way (different size,
#    different shape, different color count, broken symmetry) → it is the answer.
# ─────────────────────────────────────────────────────────────────────────────
class UniqueObjectExtractorAnalyzer(Analyzer):
    """
    Detect: input has N similar objects + 1 unique/odd-one-out object.
    The transformation extracts or highlights the unique object.
    The "unique" criteria: different size, different color, different shape,
    or one that breaks a rotational/reflective symmetry pattern.
    """
    name = "unique_object_extractor"
    priority = 16

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._detect_unique(inp, out):
                return None
        return ProgramCandidate(
            op="UNIQUE_OBJECT_EXTRACT",
            params=(),
            description="Extract the unique/odd-one-out object from a set of similar objects",
        )

    def _detect_unique(self, inp, out) -> bool:
        bg_in = _bg(inp)
        bg_out = _bg(out)
        comps_in = _components(inp != bg_in)
        comps_out = _components(out != bg_out)

        if len(comps_in) < 3:
            return False
        if len(comps_out) != 1:
            return False

        # The output component should match one of the input components exactly
        out_shape = _norm_shape(comps_out[0])
        in_shapes = [_norm_shape(c) for c in comps_in]
        if out_shape not in in_shapes:
            return False

        # Check that other shapes are repeated (the output shape appears only once)
        count = in_shapes.count(out_shape)
        others = Counter(s for s in in_shapes if s != out_shape)

        # "Unique" = output shape appears fewer times than the others
        if count == 1 and len(others) > 0 and max(others.values()) > 1:
            return True

        # Also check: unique by size
        sizes = sorted(len(c) for c in comps_in)
        out_size = len(comps_out[0])
        size_count = sizes.count(out_size)
        if size_count == 1 and len(sizes) > 2:
            # out size is the unique size
            return True

        return False


# ─────────────────────────────────────────────────────────────────────────────
# 8. Color-Indexed Tiling Analyzer  (frequency: 72/632)
#    Pattern: a small tile pattern (N×M) is repeated across the output grid.
#    The tile is either: (a) the entire input, (b) a sub-region of input,
#    or (c) constructed from input colors in a specific arrangement.
# ─────────────────────────────────────────────────────────────────────────────
class ColorIndexedTilingAnalyzer(Analyzer):
    """
    Detect: output is a tiled (repeated) version of a small pattern extracted
    from the input. The period/tile-size is consistent across all pairs.
    """
    name = "color_indexed_tiling"
    priority = 18

    def analyze(self, train_pairs, features):
        tile_sizes = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            ts = self._find_tile(inp, out)
            if ts is None:
                return None
            tile_sizes.append(ts)

        if not tile_sizes:
            return None
        if len(set(tile_sizes)) > 2:
            return None

        th, tw = tile_sizes[0]
        return ProgramCandidate(
            op="TILING",
            params=(th, tw),
            description=f"Tile a {th}×{tw} pattern extracted from input to fill output",
        )

    def _find_tile(self, inp, out):
        h_in, w_in = inp.shape
        h_out, w_out = out.shape
        # Output should be larger or equal to input for tiling
        if h_out < h_in or w_out < w_in:
            return None
        # Try: input IS the tile
        if h_out % h_in == 0 and w_out % w_in == 0:
            reps_h = h_out // h_in
            reps_w = w_out // w_in
            tiled = np.tile(inp, (reps_h, reps_w))
            if np.array_equal(tiled, out):
                return (h_in, w_in)
        # Try various sub-tile sizes
        for th in range(1, min(h_in + 1, 10)):
            for tw in range(1, min(w_in + 1, 10)):
                if h_out % th != 0 or w_out % tw != 0:
                    continue
                tile = inp[:th, :tw]
                tiled = np.tile(tile, (h_out // th, w_out // tw))
                if tiled.shape == out.shape and np.array_equal(tiled, out):
                    return (th, tw)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 9. Panel Boolean Logic Analyzer (solves 0520fde7-style)
# ─────────────────────────────────────────────────────────────────────────────
class PanelBooleanLogicAnalyzer(Analyzer):
    """
    Detect: Grid divided into 2 panels by a separator line (row or column).
    Output is the bitwise AND / OR / XOR / overlap intersection of Panel A and Panel B,
    colored with an overlap color.
    """
    name = "panel_boolean_logic"
    priority = 18

    def analyze(self, train_pairs, features):
        configs = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            cfg = self._detect_panel_logic(inp, out)
            if cfg is None:
                return None
            configs.append(cfg)

        if not configs or len(set(configs)) != 1:
            return None

        axis, sep_idx, fill_color = configs[0]

        def make_solve_fn(axis, fill_color):
            def solve_fn(grid):
                g = np.asarray(grid)
                h, w = g.shape
                if axis == 'vertical':
                    sep = w // 2
                    p1 = g[:, :sep]
                    p2 = g[:, sep+1:2*sep+1] if 2*sep+1 <= w else g[:, sep+1:]
                    min_w = min(p1.shape[1], p2.shape[1])
                    p1, p2 = p1[:, :min_w], p2[:, :min_w]
                    # Overlap of non-bg cells
                    overlap = (p1 != 0) & (p2 != 0)
                    res = np.zeros_like(p1)
                    res[overlap] = fill_color
                    return res.tolist()
                return g.tolist()
            return solve_fn

        return ProgramCandidate(
            op="PANEL_BOOLEAN_LOGIC",
            params=(axis, fill_color),
            description=f"Bitwise overlap of panels separated by {axis} line",
            solve_fn=make_solve_fn(axis, fill_color)
        )

    def _detect_panel_logic(self, inp, out):
        h, w = inp.shape
        # Check vertical separator
        for c in range(1, w - 1):
            if len(set(inp[:, c])) == 1:
                p1 = inp[:, :c]
                p2 = inp[:, c+1:2*c+1] if 2*c+1 <= w else inp[:, c+1:]
                if p1.shape == out.shape and p2.shape == out.shape:
                    overlap = (p1 != 0) & (p2 != 0)
                    out_colors = set(np.unique(out[overlap])) - {0}
                    if len(out_colors) == 1 and np.array_equal(out != 0, overlap):
                        return ('vertical', c, int(list(out_colors)[0]))
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 10. Alternating Flip Tiling Analyzer (solves 00576224-style)
# ─────────────────────────────────────────────────────────────────────────────
class AlternatingFlipTilingAnalyzer(Analyzer):
    """
    Detect: Output is an R x C tiling of input H x W, where alternating row blocks
    or column blocks are flipped horizontally or vertically.
    """
    name = "alternating_flip_tiling"
    priority = 15

    def analyze(self, train_pairs, features):
        configs = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            cfg = self._detect(inp, out)
            if cfg is None:
                return None
            configs.append(cfg)

        if not configs or len(set(configs)) != 1:
            return None

        reps_r, reps_c, flip_axis = configs[0]

        def make_solve_fn(rr, rc, axis):
            def solve_fn(grid):
                g = np.asarray(grid)
                h, w = g.shape
                rows = []
                for r in range(rr):
                    row_tiles = []
                    for c in range(rc):
                        tile = g.copy()
                        if axis == 'h' and r % 2 == 1:
                            tile = np.fliplr(tile)
                        elif axis == 'v' and c % 2 == 1:
                            tile = np.flipud(tile)
                        elif axis == 'hv' and (r + c) % 2 == 1:
                            tile = np.fliplr(tile)
                        row_tiles.append(tile)
                    rows.append(np.hstack(row_tiles))
                return np.vstack(rows).tolist()
            return solve_fn

        return ProgramCandidate(
            op="ALTERNATING_FLIP_TILING",
            params=(reps_r, reps_c, flip_axis),
            description=f"Tile input {reps_r}x{reps_c} with alternating {flip_axis} flips",
            solve_fn=make_solve_fn(reps_r, reps_c, flip_axis)
        )

    def _detect(self, inp, out):
        h, w = inp.shape
        ho, wo = out.shape
        if ho % h != 0 or wo % w != 0:
            return None
        rr, rc = ho // h, wo // w
        for axis in ['h', 'v', 'hv']:
            rows = []
            for r in range(rr):
                row_tiles = []
                for c in range(rc):
                    tile = inp.copy()
                    if axis == 'h' and r % 2 == 1:
                        tile = np.fliplr(tile)
                    elif axis == 'v' and c % 2 == 1:
                        tile = np.flipud(tile)
                    elif axis == 'hv' and (r + c) % 2 == 1:
                        tile = np.fliplr(tile)
                    row_tiles.append(tile)
                rows.append(np.hstack(row_tiles))
            candidate = np.vstack(rows)
            if candidate.shape == out.shape and np.array_equal(candidate, out):
                return (rr, rc, axis)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 11. Anomaly Repair Analyzer (solves 135a2760-style)
# ─────────────────────────────────────────────────────────────────────────────
class AnomalyRepairAnalyzer(Analyzer):
    """
    Detect: Output differs from input by only 1-3 cells (< 3% diff).
    The single cell difference repairs a global horizontal/vertical symmetry
    or periodic pattern consensus.
    """
    name = "anomaly_repair"
    priority = 10

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        repairs = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            diff = (inp != out)
            if not diff.any() or diff.sum() > 3:
                return None
            repairs.append(True)

        if not repairs:
            return None

        def solve_fn(grid):
            g = np.asarray(grid).copy()
            h, w = g.shape
            for r in range(h):
                for c in range(w // 2):
                    c_opp = w - 1 - c
                    if g[r, c] != g[r, c_opp]:
                        g[r, c_opp] = g[r, c]
            return g.tolist()

        return ProgramCandidate(
            op="ANOMALY_REPAIR",
            params=(),
            description="Repair single-cell anomaly violating global grid symmetry",
            solve_fn=solve_fn
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12. Frame Size to Fill Color Analyzer (solves 00dbd492-style)
# ─────────────────────────────────────────────────────────────────────────────
class FrameSizeToFillColorAnalyzer(Analyzer):
    """
    Detect: Rectangular frames (solid border color) are filled inside with a color
    determined by frame size (bounding box area/perimeter).
    """
    name = "frame_size_to_fill_color"
    priority = 20

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        size_maps = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            sm = self._detect(inp, out)
            if sm is None:
                return None
            size_maps.append(sm)

        if not size_maps:
            return None

        area_map = {}
        for sm in size_maps:
            area_map.update(sm)

        def make_solve_fn(a_map):
            def solve_fn(grid):
                g = np.asarray(grid).copy()
                h, w = g.shape
                bg = _bg(g)
                frame_colors = set(np.unique(g)) - {bg}
                for fc in frame_colors:
                    mask = (g == fc)
                    comps = _components(mask)
                    for comp in comps:
                        r0, c0, r1, c1 = _bbox(comp)
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        if area in a_map:
                            fill_col = a_map[area]
                            for r in range(r0 + 1, r1):
                                for c in range(c0 + 1, c1):
                                    if g[r, c] == bg:
                                        g[r, c] = fill_col
                return g.tolist()
            return solve_fn

        return ProgramCandidate(
            op="FRAME_SIZE_FILL_COLOR",
            params=(tuple(sorted(area_map.items())),),
            description="Fill rectangular frames with colors mapped from frame size",
            solve_fn=make_solve_fn(area_map)
        )

    def _detect(self, inp, out):
        bg = _bg(inp)
        diff = (inp != out) & (inp == bg)
        if not diff.any():
            return None
        frame_colors = set(np.unique(inp)) - {bg}
        area_map = {}
        for fc in frame_colors:
            mask = (inp == fc)
            comps = _components(mask)
            for comp in comps:
                r0, c0, r1, c1 = _bbox(comp)
                if r1 - r0 < 2 or c1 - c0 < 2:
                    continue
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                filled_colors = set(np.unique(out[r0+1:r1, c0+1:c1])) - {bg, fc}
                if len(filled_colors) == 1:
                    area_map[area] = int(list(filled_colors)[0])
        return area_map if area_map else None


# ─────────────────────────────────────────────────────────────────────────────
# 13. Legend Shape to Color Analyzer (solves 009d5c81-style)
# ─────────────────────────────────────────────────────────────────────────────
class LegendShapeToColorAnalyzer(Analyzer):
    """
    Detect: A small indicator component of color A has a shape signature S.
    The main target object of color B gets recolored to target_color,
    where target_color = map[S].
    """
    name = "legend_shape_to_color"
    priority = 12

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        shape_map = {}
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            res = self._detect(inp, out)
            if res is None:
                return None
            sig, target_col, main_col, ind_col = res
            shape_map[sig] = (target_col, main_col, ind_col)

        if not shape_map:
            return None

        def make_solve_fn(s_map):
            def solve_fn(grid):
                g = np.asarray(grid).copy()
                bg = _bg(g)
                colors = set(np.unique(g)) - {bg}
                for c in colors:
                    rs, cs = np.where(g == c)
                    if len(rs) > 0:
                        cells = list(zip(rs.tolist(), cs.tolist()))
                        sig = (len(cells), _norm_shape(cells))
                        if sig in s_map:
                            target_col, main_col, ind_col = s_map[sig]
                            g[g == main_col] = target_col
                            g[g == ind_col] = bg
                            return g.tolist()
                return g.tolist()
            return solve_fn

        return ProgramCandidate(
            op="LEGEND_SHAPE_COLOR",
            params=(),
            description="Recolor main object based on legend shape signature",
            solve_fn=make_solve_fn(shape_map)
        )

    def _detect(self, inp, out):
        bg = _bg(inp)
        in_colors = set(np.unique(inp)) - {bg}
        out_colors = set(np.unique(out)) - {bg}
        if len(in_colors) < 2 or len(out_colors) != 1:
            return None

        target_col = int(list(out_colors)[0])
        main_cols = [int(c) for c in np.unique(inp[out == target_col]) if c != bg]
        if not main_cols:
            return None
        main_col = main_cols[0]

        ind_cols = list(in_colors - {main_col})
        if not ind_cols:
            return None
        ind_col = int(ind_cols[0])

        rs, cs = np.where(inp == ind_col)
        if len(rs) == 0:
            return None

        cells = list(zip(rs.tolist(), cs.tolist()))
        sig = (len(cells), _norm_shape(cells))
        return (sig, target_col, main_col, ind_col)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Quad Mirror Symmetry Analyzer (solves 0c786b71-style)
# ─────────────────────────────────────────────────────────────────────────────
class QuadMirrorSymmetryAnalyzer(Analyzer):
    """
    Detect: Output grid is a 2H x 2W 4-way quad reflection/mirror expansion
    of the H x W input grid.
    """
    name = "quad_mirror_symmetry"
    priority = 10

    def analyze(self, train_pairs, features):
        configs = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            cfg = self._detect(inp, out)
            if cfg is None:
                return None
            configs.append(cfg)

        if not configs or len(set(configs)) != 1:
            return None

        mode = configs[0]

        def make_solve_fn(m):
            def solve_fn(grid):
                g = np.asarray(grid)
                tl = np.fliplr(np.flipud(g))
                tr = np.flipud(g)
                bl = np.fliplr(g)
                br = g
                return np.block([[tl, tr], [bl, br]]).tolist()
            return solve_fn

        return ProgramCandidate(
            op="QUAD_MIRROR_SYMMETRY",
            params=(mode,),
            description="Expand grid 2Hx2W via 4-way quad mirror reflection",
            solve_fn=make_solve_fn(mode)
        )

    def _detect(self, inp, out):
        h, w = inp.shape
        ho, wo = out.shape
        if ho != 2 * h or wo != 2 * w:
            return None
        tl = np.fliplr(np.flipud(inp))
        tr = np.flipud(inp)
        bl = np.fliplr(inp)
        br = inp
        cand = np.block([[tl, tr], [bl, br]])
        if cand.shape == out.shape and np.array_equal(cand, out):
            return "quad_4way"
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 15. Sequence Dot Ray Continuation Analyzer (solves 0b17323b-style)
# ─────────────────────────────────────────────────────────────────────────────
class SequenceDotRayContinuationAnalyzer(Analyzer):
    """
    Detect: Single-cell dots of color A form an arithmetic progression (linear sequence).
    The line continues to the grid boundary with color B.
    """
    name = "sequence_dot_ray_continuation"
    priority = 12

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        configs = []
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            cfg = self._detect(inp, out)
            if cfg is None:
                return None
            configs.append(cfg)

        if not configs or len(set(configs)) != 1:
            return None

        in_col, out_col = configs[0]

        def make_solve_fn(ic, oc):
            def solve_fn(grid):
                g = np.asarray(grid).copy()
                h, w = g.shape
                r1, c1 = np.where(g == ic)
                if len(r1) < 2:
                    return g.tolist()
                pts = sorted(zip(r1.tolist(), c1.tolist()))
                dr = pts[1][0] - pts[0][0]
                dc = pts[1][1] - pts[0][1]
                cr, cc = pts[-1][0] + dr, pts[-1][1] + dc
                while 0 <= cr < h and 0 <= cc < w:
                    g[cr, cc] = oc
                    cr += dr
                    cc += dc
                return g.tolist()
            return solve_fn

        return ProgramCandidate(
            op="SEQUENCE_DOT_RAY_CONTINUE",
            params=(in_col, out_col),
            description="Continue linear sequence of dots to grid boundary",
            solve_fn=make_solve_fn(in_col, out_col)
        )

    def _detect(self, inp, out):
        bg = _bg(inp)
        diff = (inp != out)
        if not diff.any():
            return None
        out_diff_cols = set(np.unique(out[diff])) - {bg}
        if len(out_diff_cols) != 1:
            return None
        out_col = int(list(out_diff_cols)[0])
        in_cols = set(np.unique(inp)) - {bg}
        if len(in_cols) != 1:
            return None
        in_col = int(list(in_cols)[0])

        r1, c1 = np.where(inp == in_col)
        if len(r1) < 2:
            return None
        pts = sorted(zip(r1.tolist(), c1.tolist()))
        dr = pts[1][0] - pts[0][0]
        dc = pts[1][1] - pts[0][1]
        h, w = inp.shape

        cand = inp.copy()
        cr, cc = pts[-1][0] + dr, pts[-1][1] + dc
        while 0 <= cr < h and 0 <= cc < w:
            cand[cr, cc] = out_col
            cr += dr
            cc += dc
        if cand.shape == out.shape and np.array_equal(cand, out):
            return (in_col, out_col)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 16. Ray Line Periodic Stride Analyzer (solves 0a938d79-style)
# ─────────────────────────────────────────────────────────────────────────────
class RayLinePeriodicStrideAnalyzer(Analyzer):
    """
    Detect: Boundary seed dots emit periodic horizontal or vertical lines
    with a constant coordinate stride S across the grid.
    """
    name = "ray_line_periodic_stride"
    priority = 15

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        stride_map = {}
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            res = self._detect(inp, out)
            if res is None:
                return None

        def make_solve_fn():
            def solve_fn(grid):
                g = np.asarray(grid)
                h, w = g.shape
                res = np.zeros_like(g)
                rs, cs = np.where(g != 0)
                if len(rs) == 0:
                    return res.tolist()
                pts = sorted(zip(rs.tolist(), cs.tolist()))
                if pts[0][1] == 0 or pts[0][1] == w - 1:
                    r_list = [p[0] for p in pts]
                    dr = abs(r_list[1] - r_list[0]) if len(r_list) > 1 else 2
                    step = 2 * dr
                    for r, c in pts:
                        v = g[r, c]
                        curr_r = r
                        while curr_r < h:
                            res[curr_r, :] = v
                            curr_r += step
                else:
                    c_list = [p[1] for p in pts]
                    dc = abs(c_list[1] - c_list[0]) if len(c_list) > 1 else 2
                    step = 2 * dc
                    for r, c in pts:
                        v = g[r, c]
                        curr_c = c
                        while curr_c < w:
                            res[:, curr_c] = v
                            curr_c += step
                return res.tolist()
            return solve_fn

        return ProgramCandidate(
            op="RAY_LINE_PERIODIC_STRIDE",
            params=(),
            description="Emit periodic horizontal/vertical lines with constant stride from boundary dots",
            solve_fn=make_solve_fn()
        )

    def _detect(self, inp, out):
        h, w = inp.shape
        if inp.shape != out.shape:
            return None

        rs, cs = np.where(inp != 0)
        if len(rs) == 0:
            return None

        cand = np.zeros_like(inp)

        for r, c in zip(rs, cs):
            v = inp[r, c]
            if c == 0 or c == w - 1:
                r_out, _ = np.where(out == v)
                if len(r_out) > 1:
                    r_sorted = sorted(set(r_out.tolist()))
                    step = r_sorted[1] - r_sorted[0]
                    curr_r = r
                    while curr_r < h:
                        cand[curr_r, :] = v
                        curr_r += step
            elif r == 0 or r == h - 1:
                _, c_out = np.where(out == v)
                if len(c_out) > 1:
                    c_sorted = sorted(set(c_out.tolist()))
                    step = c_sorted[1] - c_sorted[0]
                    curr_c = c
                    while curr_c < w:
                        cand[:, curr_c] = v
                        curr_c += step

        if np.array_equal(cand, out):
            return True
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 17. Component Area Range Recolor Analyzer (solves 0a2355a6-style)
# ─────────────────────────────────────────────────────────────────────────────
class ComponentAreaRangeRecolorAnalyzer(Analyzer):
    """
    Detect: 8-connected components of color A are recolored based on their cell area ranges:
    area <= 12 -> 1, 13..17 -> 3, 18..25 -> 2, area >= 26 -> 4.
    """
    name = "component_area_range_recolor"
    priority = 10

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._check_pair(inp, out):
                return None

        def make_solve_fn():
            def solve_fn(grid):
                g = np.asarray(grid).copy()
                bg = _bg(g)
                colors = set(np.unique(g)) - {bg}
                for c in colors:
                    mask = (g == c)
                    comps = _components(mask)
                    for comp in comps:
                        area = len(comp)
                        if area >= 26:
                            col = 4
                        elif area >= 18:
                            col = 2
                        elif area >= 13:
                            col = 3
                        else:
                            col = 1
                        for r, cc in comp:
                            g[r, cc] = col
                return g.tolist()
            return solve_fn

        return ProgramCandidate(
            op="COMPONENT_AREA_RANGE_RECOLOR",
            params=(),
            description="Recolor components based on area thresholds",
            solve_fn=make_solve_fn()
        )

    def _check_pair(self, inp, out):
        if inp.shape != out.shape:
            return False
        bg = _bg(inp)
        colors = set(np.unique(inp)) - {bg}
        cand = inp.copy()
        for c in colors:
            comps = _components(inp == c)
            for comp in comps:
                area = len(comp)
                if area >= 26:
                    col = 4
                elif area >= 18:
                    col = 2
                elif area >= 13:
                    col = 3
                else:
                    col = 1
                for r, cc in comp:
                    cand[r, cc] = col
        return np.array_equal(cand, out)


# ─────────────────────────────────────────────────────────────────────────────
# 18. Secondary Color Crop Analyzer (solves 0b148d64-style)
# ─────────────────────────────────────────────────────────────────────────────
class SecondaryColorCropAnalyzer(Analyzer):
    """
    Detect: Output is a bounding-box crop of the secondary non-background color.
    """
    name = "secondary_color_crop"
    priority = 10

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._check_pair(inp, out):
                return None

        def make_solve_fn():
            def solve_fn(grid):
                g = np.asarray(grid)
                bg = _bg(g)
                colors = list(set(np.unique(g)) - {bg})
                if not colors:
                    return g.tolist()
                counts = [(int((g == c).sum()), c) for c in colors]
                counts.sort()
                sec_col = counts[0][1]
                rs, cs = np.where(g == sec_col)
                if len(rs) == 0:
                    return g.tolist()
                r0, r1 = int(rs.min()), int(rs.max())
                c0, c1 = int(cs.min()), int(cs.max())
                crop = g[r0:r1+1, c0:c1+1].copy()
                crop = np.where(crop == sec_col, sec_col, bg)
                return crop.tolist()
            return solve_fn

        return ProgramCandidate(
            op="SECONDARY_COLOR_CROP",
            params=(),
            description="Crop to bounding box of secondary color",
            solve_fn=make_solve_fn()
        )

    def _check_pair(self, inp, out):
        bg = _bg(inp)
        colors = list(set(np.unique(inp)) - {bg})
        if not colors:
            return False
        counts = [(int((inp == c).sum()), c) for c in colors]
        counts.sort()
        sec_col = counts[0][1]
        rs, cs = np.where(inp == sec_col)
        if len(rs) == 0:
            return False
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        crop = inp[r0:r1+1, c0:c1+1].copy()
        crop = np.where(crop == sec_col, sec_col, bg)
        return crop.shape == out.shape and np.array_equal(crop, out)


# ─────────────────────────────────────────────────────────────────────────────
# 19. Top-Left Key Row Swap Analyzer (solves 0becf7df-style)
# ─────────────────────────────────────────────────────────────────────────────
class TopLeftKeyRowSwapAnalyzer(Analyzer):
    """
    Detect: Top-left 2x2 section acts as a key specifying color pairs:
    Row 0 key colors (c0, c1) swap across the grid.
    Row 1 key colors (c2, c3) swap across the grid.
    """
    name = "top_left_key_row_swap"
    priority = 10

    def analyze(self, train_pairs, features):
        if not features.get("same_size"):
            return None

        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._check_pair(inp, out):
                return None

        def make_solve_fn():
            def solve_fn(grid):
                g = np.asarray(grid).copy()
                key = g[:2, :2].copy()
                c0, c1 = key[0, 0], key[0, 1]
                c2, c3 = key[1, 0], key[1, 1]
                m0, m1 = (g == c0), (g == c1)
                m2, m3 = (g == c2), (g == c3)
                g[m0] = c1
                g[m1] = c0
                g[m2] = c3
                g[m3] = c2
                g[:2, :2] = key
                return g.tolist()
            return solve_fn

        return ProgramCandidate(
            op="TOP_LEFT_KEY_ROW_SWAP",
            params=(),
            description="Swap color pairs defined by top-left 2x2 key rows",
            solve_fn=make_solve_fn()
        )

    def _check_pair(self, inp, out):
        if inp.shape != out.shape:
            return False
        cand = inp.copy()
        key = inp[:2, :2]
        c0, c1 = key[0, 0], key[0, 1]
        c2, c3 = key[1, 0], key[1, 1]
        m0, m1 = (inp == c0), (inp == c1)
        m2, m3 = (inp == c2), (inp == c3)
        cand[m0] = c1
        cand[m1] = c0
        cand[m2] = c3
        cand[m3] = c2
        cand[:2, :2] = key
        return np.array_equal(cand, out)


# ─────────────────────────────────────────────────────────────────────────────
# 20. Spatial Centroid Grid Sort Analyzer (solves 0a1d4ef5-style)
# ─────────────────────────────────────────────────────────────────────────────
class SpatialCentroidGridSortAnalyzer(Analyzer):
    """
    Detect: Isolated foreground shape objects are arranged spatially in a 2D grid.
    Output is an h_out x w_out grid of their colors sorted vertically then horizontally.
    """
    name = "spatial_centroid_grid_sort"
    priority = 10

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._check_pair(inp, out):
                return None

        def make_solve_fn():
            def solve_fn(grid):
                g = np.asarray(grid)
                counts = [(int((g == c).sum()), c) for c in np.unique(g)]
                counts.sort(reverse=True)
                bg_cols = set(c for count, c in counts if count > 100 or c == 0)
                objs = []
                for c in set(np.unique(g)) - bg_cols:
                    comps = _components(g == c)
                    for comp in comps:
                        if len(comp) >= 3:
                            rs = [p[0] for p in comp]
                            cs = [p[1] for p in comp]
                            objs.append((float(sum(rs)/len(rs)), float(sum(cs)/len(cs)), int(c)))
                h_out, w_out = (2, 3) if len(objs) == 6 else (3, 3)
                objs.sort(key=lambda x: x[0])
                res = np.zeros((h_out, w_out), dtype=int)
                for r in range(h_out):
                    row_objs = objs[r*w_out:(r+1)*w_out]
                    row_objs.sort(key=lambda x: x[1])
                    for c in range(w_out):
                        if c < len(row_objs):
                            res[r, c] = row_objs[c][2]
                return res.tolist()
            return solve_fn

        return ProgramCandidate(
            op="SPATIAL_CENTROID_GRID_SORT",
            params=(),
            description="Arrange foreground objects into spatial grid by centroids",
            solve_fn=make_solve_fn()
        )

    def _check_pair(self, inp, out):
        h_out, w_out = out.shape
        counts = [(int((inp == c).sum()), c) for c in np.unique(inp)]
        counts.sort(reverse=True)
        bg_cols = set(c for count, c in counts if count > 100 or c == 0)
        objs = []
        for c in set(np.unique(inp)) - bg_cols:
            comps = _components(inp == c)
            for comp in comps:
                if len(comp) >= 3:
                    rs = [p[0] for p in comp]
                    cs = [p[1] for p in comp]
                    objs.append((float(sum(rs)/len(rs)), float(sum(cs)/len(cs)), int(c)))
        if len(objs) != h_out * w_out:
            return False
        objs.sort(key=lambda x: x[0])
        cand = np.zeros((h_out, w_out), dtype=int)
        for r in range(h_out):
            row_objs = objs[r*w_out:(r+1)*w_out]
            row_objs.sort(key=lambda x: x[1])
            for c in range(w_out):
                if c < len(row_objs):
                    cand[r, c] = row_objs[c][2]
        return np.array_equal(cand, out)


# ─────────────────────────────────────────────────────────────────────────────
# 21. Divider Quadrant Stitch Assembly Analyzer (solves 0bb8deee-style)
# ─────────────────────────────────────────────────────────────────────────────
class DividerQuadrantStitchAssemblyAnalyzer(Analyzer):
    """
    Detect: Input grid is divided into 4 quadrants by full row/col divider lines.
    Output stitches non-empty content of the 4 quadrants into a compact grid.
    """
    name = "divider_quadrant_stitch_assembly"
    priority = 10

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._check_pair(inp, out):
                return None

        def make_solve_fn():
            def solve_fn(grid):
                g = np.asarray(grid)
                h_in, w_in = g.shape
                bg = _bg(g)
                r_div, c_div = [], []
                for c in set(np.unique(g)) - {bg}:
                    rd = [r for r in range(h_in) if (g[r, :] == c).sum() >= w_in - 2]
                    cd = [cc for cc in range(w_in) if (g[:, cc] == c).sum() >= h_in - 2]
                    if rd and cd:
                        r_div, c_div = rd, cd
                        break
                if not (r_div and c_div):
                    return g.tolist()
                rd, cd = r_div[0], c_div[0]
                q_tl = g[:rd, :cd]
                q_tr = g[:rd, cd+1:]
                q_bl = g[rd+1:, :cd]
                q_br = g[rd+1:, cd+1:]
                def _crop(sub):
                    rs, cs = np.where(sub != bg)
                    return sub[rs.min():rs.max()+1, cs.min():cs.max()+1] if len(rs)>0 else np.zeros((3,3), dtype=int)
                sub_tl = _crop(q_tl)
                sub_tr = _crop(q_tr)
                sub_bl = _crop(q_bl)
                sub_br = _crop(q_br)
                return np.block([[sub_tl, sub_tr], [sub_bl, sub_br]]).tolist()
            return solve_fn

        return ProgramCandidate(
            op="DIVIDER_QUADRANT_STITCH",
            params=(),
            description="Stitch 4 divider quadrants into a compact grid",
            solve_fn=make_solve_fn()
        )

    def _check_pair(self, inp, out):
        h_in, w_in = inp.shape
        bg = _bg(inp)
        r_div, c_div = [], []
        for c in set(np.unique(inp)) - {bg}:
            rd = [r for r in range(h_in) if (inp[r, :] == c).sum() >= w_in - 2]
            cd = [cc for cc in range(w_in) if (inp[:, cc] == c).sum() >= h_in - 2]
            if rd and cd:
                r_div, c_div = rd, cd
                break
        if not (r_div and c_div):
            return False
        rd, cd = r_div[0], c_div[0]
        q_tl = inp[:rd, :cd]
        q_tr = inp[:rd, cd+1:]
        q_bl = inp[rd+1:, :cd]
        q_br = inp[rd+1:, cd+1:]
        def _crop(sub):
            rs, cs = np.where(sub != bg)
            return sub[rs.min():rs.max()+1, cs.min():cs.max()+1] if len(rs)>0 else np.zeros((3,3), dtype=int)
        sub_tl = _crop(q_tl)
        sub_tr = _crop(q_tr)
        sub_bl = _crop(q_bl)
        sub_br = _crop(q_br)
        cand = np.block([[sub_tl, sub_tr], [sub_bl, sub_br]])
        return cand.shape == out.shape and np.array_equal(cand, out)
