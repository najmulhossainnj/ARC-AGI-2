from __future__ import annotations
"""
param_learning.py
-------------------
Automatic Parameter Learning for DSL Instructions.
Deduces specific parameters (colors, offsets, shift vectors, scale factors)
directly from input/output training pairs to avoid exhaustive grid searches.
"""
import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from ..core.grid import as_grid

def learn_color_map(train_pairs, background=0) -> List[Dict[int, int]]:
    """Learn a direct color mapping table from train pairs."""
    mapping = {}
    for inp, out in train_pairs:
        inp, out = as_grid(inp), as_grid(out)
        if inp.shape != out.shape:
            return []
        for v in np.unique(inp):
            out_vals = np.unique(out[inp == v])
            if len(out_vals) != 1:
                return []
            ov = int(out_vals[0])
            cv = int(v)
            if cv in mapping and mapping[cv] != ov:
                return []
            mapping[cv] = ov
    nontrivial = {k: v for k, v in mapping.items() if k != v}
    return [nontrivial] if nontrivial else []

# Function alias for backwards compatibility with grammar.py
learn_colormap = learn_color_map

def learn_translate_shift(train_pairs, background=0) -> List[Tuple[int, int]]:
    """Learn uniform translation (dr, dc) shift for non-background pixels."""
    shifts = []
    for inp, out in train_pairs:
        inp, out = as_grid(inp), as_grid(out)
        if inp.shape != out.shape:
            return []
        in_pts = set(map(tuple, np.argwhere(inp != background).tolist()))
        out_pts = set(map(tuple, np.argwhere(out != background).tolist()))
        if len(in_pts) != len(out_pts) or len(in_pts) == 0:
            return []
        in_arr = np.array(sorted(in_pts))
        out_arr = np.array(sorted(out_pts))
        cand_shifts = set()
        for ip, op in zip(in_arr[:5], out_arr[:5]):
            cand_shifts.add((int(op[0] - ip[0]), int(op[1] - ip[1])))
        if len(cand_shifts) != 1:
            return []
        dr, dc = next(iter(cand_shifts))
        shifted = {(r + dr, c + dc) for r, c in in_pts}
        if shifted != out_pts:
            return []
        shifts.append((dr, dc))
    if not shifts or len(set(shifts)) != 1:
        return []
    return [shifts[0]]

def learn_crop_to_color(train_pairs, background=0) -> List[int]:
    """Learn color target whose bounding box matches the output shape & content."""
    crop_colors = []
    if not train_pairs:
        return []
    inp0 = as_grid(train_pairs[0][0])
    for c in set(int(v) for v in np.unique(inp0)) - {background}:
        valid = True
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            rs, cs = np.where(inp == c)
            if len(rs) == 0:
                valid = False
                break
            r1, r2 = int(rs.min()), int(rs.max())
            c1, c2 = int(cs.min()), int(cs.max())
            cropped = inp[r1:r2+1, c1:c2+1]
            if cropped.shape != out.shape or not np.array_equal(cropped, out):
                valid = False
                break
        if valid:
            crop_colors.append(c)
    return crop_colors

def learn_fill_enclosed(train_pairs, background=0) -> List[Tuple[int, int]]:
    """Learn fill color for enclosed regions of a border color."""
    if not train_pairs:
        return []
    from ..dsl.advanced_transforms import find_enclosed_regions
    inp0 = as_grid(train_pairs[0][0])
    colors = set(int(v) for v in np.unique(inp0)) - {background}
    valid_pairs = []
    for border_c in colors:
        for fill_c in colors:
            if border_c == fill_c:
                continue
            valid = True
            for inp, out in train_pairs:
                inp, out = as_grid(inp), as_grid(out)
                if inp.shape != out.shape:
                    valid = False; break
                enc = find_enclosed_regions(inp, border_c=border_c, background=background)
                if enc is None or not enc.any():
                    valid = False; break
                expected = inp.copy()
                expected[enc] = fill_c
                if not np.array_equal(expected, out):
                    valid = False; break
            if valid:
                valid_pairs.append((border_c, fill_c))
    return valid_pairs

def learn_gravity_direction(train_pairs, background=0) -> List[str]:
    """Learn gravity fall direction across all training pairs."""
    from ..dsl.advanced_transforms import apply_gravity
    for direction in ['down', 'up', 'left', 'right']:
        valid = True
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                valid = False; break
            g = apply_gravity(inp, direction, background=background)
            if g is None or not np.array_equal(g, out):
                valid = False; break
        if valid:
            return [direction]
    return []

def learn_arrow_replication(train_pairs, background=0) -> List[Tuple[int, Tuple]]:
    """Learn arrow-driven template replication parameters."""
    if not train_pairs:
        return []
    inp0, out0 = as_grid(train_pairs[0][0]), as_grid(train_pairs[0][1])
    if inp0.shape != out0.shape:
        return []
    colors = sorted(set(int(v) for v in np.unique(inp0)) - {background})
    if len(colors) < 2:
        return []

    for template_color in colors:
        in_pts = set(map(tuple, np.argwhere(inp0 == template_color).tolist()))
        out_pts = set(map(tuple, np.argwhere(out0 == template_color).tolist()))
        if len(in_pts) >= 3 and in_pts.issubset(out_pts):
            rs = [r for r, _ in in_pts]
            cs = [c for _, c in in_pts]
            tr1, tc1, tr2, tc2 = min(rs), min(cs), max(rs), max(cs)
            th, tw = tr2 - tr1 + 1, tc2 - tc1 + 1

            arrow_specs = []
            for c in colors:
                if c == template_color:
                    continue
                in_rs, in_cs = np.where(inp0 == c)
                if len(in_rs) == 0:
                    continue
                ar1, ac1 = int(in_rs.min()), int(in_cs.min())
                ar2, ac2 = int(in_rs.max()), int(in_cs.max())
                arrow_h, arrow_w = ar2 - ar1 + 1, ac2 - ac1 + 1

                direction = None
                if ac1 > tc2 and arrow_h == th:
                    direction = 'RIGHT'
                    step = tw + (ac1 - tc2 - 1) + 1
                    n_copies = arrow_h
                elif ac2 < tc1 and arrow_h == th:
                    direction = 'LEFT'
                    step = tw + (tc1 - ac2 - 1) + 1
                    n_copies = arrow_h
                elif ar1 > tr2 and arrow_w == tw:
                    direction = 'DOWN'
                    step = th + (ar1 - tr2 - 1) + 1
                    n_copies = arrow_w
                elif ar2 < tr1 and arrow_w == tw:
                    direction = 'UP'
                    step = th + (tr1 - ar2 - 1) + 1
                    n_copies = arrow_w

                if direction:
                    arrow_specs.append((c, direction, n_copies, step))

            if arrow_specs:
                return [(template_color, tuple(sorted(arrow_specs)))]
    return []

def learn_remove_small_objects(train_pairs, background=0) -> List[int]:
    """Learn minimum object size threshold to remove small isolated components."""
    from scipy.ndimage import label
    struct = np.ones((3, 3), dtype=int)
    thresholds = []
    if not train_pairs:
        return []
    for min_size in range(1, 25):
        valid = True
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            if inp.shape != out.shape:
                valid = False; break
            test = inp.copy()
            for c in set(int(v) for v in np.unique(inp)) - {background}:
                mask = (inp == c)
                lbl, num = label(mask, structure=struct)
                for i in range(1, num + 1):
                    if (lbl == i).sum() < min_size:
                        test[lbl == i] = background
            if not np.array_equal(test, out):
                valid = False; break
        if valid:
            thresholds.append(min_size)
            break
    return thresholds

def learn_recolor_by_indicator_feature(train_pairs, background=0) -> List[Tuple[int, int, Dict]]:
    """Learn recoloring rule targeting target_c based on indicator color feature."""
    if not train_pairs:
        return []
    inp0 = as_grid(train_pairs[0][0])
    from ..dsl.advanced_transforms import count_enclosed_holes

    colors_in = set(int(v) for v in np.unique(inp0)) - {background}
    candidates = []

    for target_c in colors_in:
        for ind_c in colors_in:
            if target_c == ind_c:
                continue
            feat_map = {}
            valid = True
            for inp, out in train_pairs:
                inp, out = as_grid(inp), as_grid(out)
                if inp.shape != out.shape:
                    valid = False; break
                mask_ind = (inp == ind_c)
                if not mask_ind.any():
                    valid = False; break
                cnt = int(mask_ind.sum())
                rows, cols = np.where(mask_ind)
                r1, r2 = rows.min(), rows.max()
                c1, c2 = cols.min(), cols.max()
                sub = inp[r1:r2+1, c1:c2+1]
                holes = count_enclosed_holes(sub, background=background)

                target_mask = (inp == target_c)
                if not target_mask.any():
                    valid = False; break
                
                # Check dimensions before indexing output grid
                if out.shape != inp.shape:
                    valid = False; break
                
                out_cols = np.unique(out[target_mask])
                if len(out_cols) != 1:
                    valid = False; break

                feat_key = (int(cnt), int(holes))
                feat_map[feat_key] = int(out_cols[0])
            if valid and len(feat_map) > 0:
                candidates.append((target_c, ind_c, feat_map))
    return candidates

def learn_object_translation(train_pairs, background=0, max_colors=9, max_objects=20):
    """Learn per-object translation parameters."""
    return []

def learn_cycle_block_extend(train_pairs, background=0) -> List[Tuple[int, int, Tuple]]:
    """Learn block height, num blocks, and recolor mapping for block cycling."""
    if not train_pairs:
        return []
    for block_h in [2, 3, 4, 5]:
        valid = True
        recolor_tuples = ()
        for inp, out in train_pairs:
            inp, out = as_grid(inp), as_grid(out)
            h, w = inp.shape
            if h < block_h * 2 or out.shape[0] != block_h * 3:
                valid = False; break
            p1 = inp[:block_h, :]
            p2 = inp[block_h:2*block_h, :]
            p3 = p1.copy() if np.array_equal(p1, p2) else np.fliplr(p1)
            stacked = np.vstack([p1, p2, p3])
            in_cols = set(int(v) for v in np.unique(stacked)) - {background}
            out_cols = set(int(v) for v in np.unique(out)) - {background}
            if len(in_cols) == 1 and len(out_cols) == 1:
                ic, oc = next(iter(in_cols)), next(iter(out_cols))
                if ic != oc:
                    recolor_tuples = ((ic, oc),)
                mapped = stacked.copy()
                if ic != oc:
                    mapped[mapped == ic] = oc
                if not np.array_equal(mapped, out):
                    valid = False; break
            elif in_cols == out_cols and np.array_equal(stacked, out):
                pass
            else:
                valid = False; break
        if valid:
            return [(block_h, 3, recolor_tuples)]
    return []
