from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate
from ...core.grid import as_grid


class SymmetryCompleteAnalyzer(Analyzer):
    """Detect if output is a symmetrized/completed version of input."""
    name = "symmetry_complete"
    priority = 15

    def analyze(self, train_pairs, features):
        axes = []
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            axis = self._detect_axis(inp, out)
            if axis is None:
                return None
            axes.append(axis)
        if not axes or len(set(axes)) != 1:
            return None
        return ProgramCandidate(
            op="SYMMETRY_REPAIR",
            params=(axes[0],),
            description=f"Symmetry completion along axis {axes[0]}",
        )

    def _detect_axis(self, inp, out):
        for axis in ["H", "V", "HV", "D1", "D2"]:
            sym = self._make_symmetric(inp, axis)
            if sym is not None and np.array_equal(sym, out):
                return axis
        return None

    def _make_symmetric(self, g, axis):
        g = as_grid(g).copy()
        if axis == "H":
            return np.maximum(g, np.flipud(g))
        if axis == "V":
            return np.maximum(g, np.fliplr(g))
        if axis == "HV":
            t = np.maximum(g, np.flipud(g))
            return np.maximum(t, np.fliplr(t))
        return None


class GravityFallAnalyzer(Analyzer):
    """Detect if objects fall toward an edge (gravity effect)."""
    name = "gravity_fall"
    priority = 12

    def analyze(self, train_pairs, features):
        directions = []
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            d = None
            for direction in ["down", "up", "left", "right"]:
                if np.array_equal(self._apply_gravity(inp, direction), out):
                    d = direction
                    break
            if d is None:
                return None
            directions.append(d)
        if not directions or len(set(directions)) != 1:
            return None
        return ProgramCandidate(
            op="GRAVITY",
            params=(directions[0],),
            description=f"Gravity toward {directions[0]}",
        )

    def _apply_gravity(self, grid, direction, background=0):
        g = as_grid(grid).copy()
        h, w = g.shape
        out = np.full_like(g, background)
        if direction == "down":
            for c in range(w):
                col = g[:, c]
                non_bg = col[col != background]
                if len(non_bg):
                    out[h - len(non_bg):, c] = non_bg
        elif direction == "up":
            for c in range(w):
                col = g[:, c]
                non_bg = col[col != background]
                if len(non_bg):
                    out[: len(non_bg), c] = non_bg
        elif direction == "right":
            for r in range(h):
                row = g[r, :]
                non_bg = row[row != background]
                if len(non_bg):
                    out[r, w - len(non_bg):] = non_bg
        elif direction == "left":
            for r in range(h):
                row = g[r, :]
                non_bg = row[row != background]
                if len(non_bg):
                    out[r, : len(non_bg)] = non_bg
        return out


class BorderCropAnalyzer(Analyzer):
    """Detect if output is cropped to a color cluster bounding box."""
    name = "border_crop"
    priority = 25

    def analyze(self, train_pairs, features):
        crop_colors = []
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            found = False
            for c in sorted(set(int(v) for v in np.unique(inp)) - {0}):
                rs, cs = np.where(inp == c)
                if len(rs) == 0:
                    continue
                r1, r2 = int(rs.min()), int(rs.max())
                c1, c2 = int(cs.min()), int(cs.max())
                crop = inp[r1 : r2 + 1, c1 : c2 + 1]
                if crop.shape == out.shape and np.array_equal(crop, out):
                    crop_colors.append(c)
                    found = True
                    break
            if not found:
                return None
        if not crop_colors or len(set(crop_colors)) != 1:
            return None
        return ProgramCandidate(
            op="CROP_TO_COLOR",
            params=(crop_colors[0],),
            description=f"Crop to bounding box of color {crop_colors[0]}",
        )


class PatternExtensionAnalyzer(Analyzer):
    """Detect diagonal periodic pattern completion (05269061-style)."""
    name = "pattern_extension"
    priority = 22

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            non_bg = [(int(r), int(c), int(inp[r, c])) for r, c in zip(*np.where(inp != 0))]
            if not non_bg:
                return None
            matched_period = None
            for period in range(1, 10):
                seq = {}
                ok = True
                for r, c, v in non_bg:
                    k = (r + c) % period
                    if k in seq and seq[k] != v:
                        ok = False
                        break
                    seq[k] = v
                if ok and len(seq) == period:
                    test = np.zeros_like(inp)
                    for r in range(inp.shape[0]):
                        for c_ in range(inp.shape[1]):
                            k = (r + c_) % period
                            if k in seq:
                                test[r, c_] = seq[k]
                    if np.array_equal(test, out):
                        matched_period = period
                        break
            if matched_period is None:
                return None
        return ProgramCandidate(
            op="DIAGONAL_PATTERN_COMPLETE",
            params=(),
            description=f"Diagonal pattern completion period={matched_period}",
        )


class BlockCycleAnalyzer(Analyzer):
    """Detect vertical block cycling + optional recoloring (017c7c7b-style)."""
    name = "block_cycle"
    priority = 18

    def analyze(self, train_pairs, features):
        params_list = []
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            p = self._detect(inp, out)
            if p is None:
                return None
            params_list.append(p)
        if not params_list:
            return None
        first = params_list[0]
        if not all(p == first for p in params_list[1:]):
            return None
        block_h, num_blocks, recolor_tuples = first
        return ProgramCandidate(
            op="CYCLE_BLOCK_EXTEND",
            params=(block_h, num_blocks, recolor_tuples),
            description=f"Block cycle block_h={block_h} recolor={recolor_tuples}",
        )

    def _detect(self, inp, out, block_h=3):
        h, w = inp.shape
        if h < block_h * 2 or out.shape[0] != block_h * 3:
            return None
        p1 = inp[:block_h, :]
        p2 = inp[block_h : 2 * block_h, :]
        p3 = p1.copy() if np.array_equal(p1, p2) else np.fliplr(p1)
        stacked = np.vstack([p1, p2, p3])
        in_cols = set(int(v) for v in np.unique(stacked)) - {0}
        out_cols = set(int(v) for v in np.unique(out)) - {0}
        if len(in_cols) == 1 and len(out_cols) == 1:
            ic = next(iter(in_cols))
            oc = next(iter(out_cols))
            recolor_tuples = ((ic, oc),) if ic != oc else ()
            mapped = stacked.copy()
            if ic != oc:
                mapped[mapped == ic] = oc
            if np.array_equal(mapped, out):
                return (block_h, 3, recolor_tuples)
        elif in_cols == out_cols and np.array_equal(stacked, out):
            return (block_h, 3, ())
        return None


class ParallelogramAlignAnalyzer(Analyzer):
    """Detect slanted parallelogram alignment (025d127b-style)."""
    name = "parallelogram_align"
    priority = 28

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            try:
                from ...dsl.advanced_transforms import shift_parallelogram_fix_right
                pred = shift_parallelogram_fix_right(inp)
                if pred is None or not np.array_equal(pred, out):
                    return None
            except Exception:
                return None
        return ProgramCandidate(
            op="SHIFT_PARALLELOGRAM",
            params=(),
            description="Shift parallelogram objects to align (right anchor)",
        )


class DiagonalChainAnalyzer(Analyzer):
    """Detect diagonal object chaining from top-left (03560426-style)."""
    name = "diagonal_chain"
    priority = 28

    def analyze(self, train_pairs, features):
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                return None
            try:
                from ...dsl.advanced_transforms import diagonal_stack_chain
                pred = diagonal_stack_chain(inp)
                if pred is None or not np.array_equal(pred, out):
                    return None
            except Exception:
                return None
        return ProgramCandidate(
            op="DIAGONAL_STACK_CHAIN",
            params=(),
            description="Chain objects diagonally from top-left sorted by input column",
        )
