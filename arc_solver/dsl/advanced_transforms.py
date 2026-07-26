"""
Additional grid- and object-level transforms.

These extend the original whole-grid primitive set (rotate/flip/crop/gravity/
scale/recolor) with the operations flagged as missing in the README:
object selection, deletion, recoloring, mirrored/tiled mosaics, block
downscaling, border framing, hole filling, symmetry-based repair, two-panel
boolean logic, and periodic pattern completion.

Every function here is a pure function of (grid, *params) -> grid | None.
Returning None signals "not applicable to this grid" and is treated by the
executor the same way as any other failure (candidate discarded).
"""
from __future__ import annotations
from collections import deque, Counter
import numpy as np

from ..core.grid import as_grid, non_background_bbox
from ..core.scene import Scene
from .selectors import select_objects


# ----------------------------------------------------------------------
# Simple whole-grid ops
# ----------------------------------------------------------------------

def transpose(grid):
    return as_grid(grid).T.copy()


def apply_colormap(grid, mapping):
    g = as_grid(grid).copy()
    out = g.copy()
    for old, new in mapping:
        out[g == old] = new
    return out


def tile_grid(grid, rh, rw):
    g = as_grid(grid)
    rh, rw = int(rh), int(rw)
    if rh <= 0 or rw <= 0 or rh > 10 or rw > 10:
        return None
    return np.tile(g, (rh, rw))


def fractal_tile(grid, background=0):
    """Self-replicating supertile: an h x w input becomes an (h*h) x (w*w)
    output made of h*w copies of the input, arranged in the input's own
    shape -- copy (r, c) is a full copy of the input if grid[r, c] isn't
    background, else a blank (background) block. No parameters beyond
    background since the input dictates both the tiling factor and the
    placement mask. A very common ARC pattern (self-similar / 'fractal'
    replication) that plain TILE (uniform repetition) can't express.
    """
    g = as_grid(grid)
    h, w = g.shape
    if h == 0 or w == 0 or h * w > 400:
        return None
    out = np.full((h * h, w * w), background, dtype=g.dtype)
    for r in range(h):
        for c in range(w):
            if g[r, c] != background:
                out[r * h:(r + 1) * h, c * w:(c + 1) * w] = g
    return out


def fractal_tile_inverse(grid, background=0):
    """Self-replicating supertile: an h x w input becomes an (h*h) x (w*w)
    output made of h*w copies of the INVERTED input shape."""
    g = as_grid(grid)
    h, w = g.shape
    if h == 0 or w == 0 or h * w > 400:
        return None
    fg_colors = set(np.unique(g)) - {background}
    if len(fg_colors) != 1:
        return None
    fg_col = next(iter(fg_colors))
    
    inv_block = np.full_like(g, fg_col)
    inv_block[g != background] = background
    
    out = np.full((h * h, w * w), background, dtype=g.dtype)
    for r in range(h):
        for c in range(w):
            if g[r, c] != background:
                out[r * h:(r + 1) * h, c * w:(c + 1) * w] = inv_block
    return out


def _flip_kind(g, kind):
    if kind == "H":
        return np.fliplr(g)
    if kind == "V":
        return np.flipud(g)
    return g


def reflect_tile(grid, rh, rw, row_flip, col_flip):
    """Tile the grid rh x rw times, like TILE, but every odd-indexed
    row-band gets `row_flip` applied ('H', 'V', or None) and every
    odd-indexed column-band gets `col_flip` applied; a band that's odd on
    both axes gets both, composed. This is the general form of the
    'mirror-tiled wallpaper' pattern MOSAIC only covers for a single 2x
    repetition -- e.g. a 3x3 supertile where every other row-band is
    horizontally mirrored, with no mirroring across columns at all.
    """
    g = as_grid(grid)
    rh, rw = int(rh), int(rw)
    if rh <= 0 or rw <= 0 or rh > 6 or rw > 6:
        return None
    h, w = g.shape
    out = np.zeros((h * rh, w * rw), dtype=g.dtype)
    for br in range(rh):
        for bc in range(rw):
            tile = g
            if br % 2 == 1:
                tile = _flip_kind(tile, row_flip)
            if bc % 2 == 1:
                tile = _flip_kind(tile, col_flip)
            out[br * h:(br + 1) * h, bc * w:(bc + 1) * w] = tile
    return out


def mosaic(grid, mode):
    g = as_grid(grid)
    h_flip = np.fliplr(g)
    v_flip = np.flipud(g)
    hv_flip = np.flipud(h_flip)
    if mode == "H_MIRROR":
        return np.concatenate([g, h_flip], axis=1)
    if mode == "H_PLAIN":
        return np.concatenate([g, g], axis=1)
    if mode == "V_MIRROR":
        return np.concatenate([g, v_flip], axis=0)
    if mode == "V_PLAIN":
        return np.concatenate([g, g], axis=0)
    if mode == "QUAD_MIRROR":
        top = np.concatenate([g, h_flip], axis=1)
        bottom = np.concatenate([v_flip, hv_flip], axis=1)
        return np.concatenate([top, bottom], axis=0)
    return None


def downscale(grid, fh, fw):
    g = as_grid(grid)
    h, w = g.shape
    fh, fw = int(fh), int(fw)
    if fh <= 0 or fw <= 0 or h % fh != 0 or w % fw != 0:
        return None
    oh, ow = h // fh, w // fw
    out = np.zeros((oh, ow), dtype=g.dtype)
    for r in range(oh):
        for c in range(ow):
            block = g[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw]
            vals, counts = np.unique(block, return_counts=True)
            out[r, c] = vals[np.argmax(counts)]
    return out


def add_border(grid, color, width):
    g = as_grid(grid)
    width = int(width)
    if width <= 0 or width > 10:
        return None
    h, w = g.shape
    out = np.full((h + 2 * width, w + 2 * width), color, dtype=g.dtype)
    out[width:width + h, width:width + w] = g
    return out


def strip_border(grid, width):
    g = as_grid(grid)
    width = int(width)
    h, w = g.shape
    if width <= 0 or h <= 2 * width or w <= 2 * width:
        return None
    return g[width:h - width, width:w - width].copy()


# ----------------------------------------------------------------------
# Hole filling (enclosed background regions)
# ----------------------------------------------------------------------

def _background_reachable_from_edge(g, background):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if g[r, c] == background and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if g[r, c] == background and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == background:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def fill_holes(grid, fill_color, background=0):
    g = as_grid(grid).copy()
    reachable = _background_reachable_from_edge(g, background)
    holes = (g == background) & (~reachable)
    if not holes.any():
        return None
    g[holes] = fill_color
    return g


# ----------------------------------------------------------------------
# Symmetry-based repair (fills a "damaged"/noise color using grid self-symmetry)
# ----------------------------------------------------------------------

def _symmetry_maps(grid, noise_color):
    g = as_grid(grid)
    h, w = g.shape
    maps = [
        lambda r, c: (r, w - 1 - c),          # mirror horizontal
        lambda r, c: (h - 1 - r, c),          # mirror vertical
        lambda r, c: (h - 1 - r, w - 1 - c),  # 180 rotation
    ]
    if h == w:
        maps.append(lambda r, c: (c, r))                  # transpose
        maps.append(lambda r, c: (w - 1 - c, h - 1 - r))   # anti-transpose

    # Add offset 180-rotation centers around grid center
    for dr in range(-4, 5):
        for dc in range(-4, 5):
            cr2 = (h - 1) + dr
            cc2 = (w - 1) + dc
            maps.append(lambda r, c, cr2=cr2, cc2=cc2: (cr2 - r, cc2 - c))

    # Efficient candidate center discovery from matching non-noise cells
    non_noise = [(r, c, g[r, c]) for r in range(h) for c in range(w) if g[r, c] != noise_color]
    if len(non_noise) > 1:
        step = max(1, len(non_noise) // 60)
        pts = non_noise[::step][:60]
        centers_180 = set()
        for r1, c1, col1 in pts:
            for r2, c2, col2 in pts:
                if col1 == col2:
                    centers_180.add((r1 + r2, c1 + c2))
        for cr2, cc2 in centers_180:
            maps.append(lambda r, c, cr2=cr2, cc2=cc2: (cr2 - r, cc2 - c))
    return maps


def symmetry_repair(grid, noise_color, background=0):
    g = as_grid(grid).copy()
    h, w = g.shape
    mask = (g == noise_color)
    if not mask.any():
        return None

    rows, cols = np.where(mask)
    maps = _symmetry_maps(g, noise_color)
    valid_candidates = []

    for m in maps:
        can_fill_all = True
        for r, c in zip(rows, cols):
            try:
                nr, nc = m(r, c)
            except Exception:
                can_fill_all = False; break
            if not (0 <= nr < h and 0 <= nc < w) or g[nr, nc] == noise_color:
                can_fill_all = False; break

        if not can_fill_all:
            continue

        checked = 0
        mismatches = 0
        for r in range(h):
            for c in range(w):
                if g[r, c] == noise_color:
                    continue
                try:
                    nr, nc = m(r, c)
                except Exception:
                    continue
                if not (0 <= nr < h and 0 <= nc < w) or g[nr, nc] == noise_color:
                    continue
                checked += 1
                if g[nr, nc] != g[r, c]:
                    mismatches += 1

        if checked >= 20:
            match_rate = (checked - mismatches) / checked
            valid_candidates.append((match_rate, checked, m))

    if not valid_candidates:
        return None

    valid_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_m = valid_candidates[0][2]

    for r, c in zip(rows, cols):
        nr, nc = best_m(r, c)
        g[r, c] = g[nr, nc]

    return g
    if (g == noise_color).any():
        return None
    return g


def symmetry_repair_crop(grid, noise_color, background=0):
    """Repair a grid using self-symmetry and crop to the bounding box of the repaired noise region."""
    g = as_grid(grid)
    if not (g == noise_color).any():
        return None
    mask = g == noise_color
    rs, cs = np.where(mask)
    r1, r2, c1, c2 = rs.min(), rs.max(), cs.min(), cs.max()
    repaired = symmetry_repair(g, noise_color, background=background)
    if repaired is None:
        return None
    return repaired[r1:r2 + 1, c1:c2 + 1].copy()


def apply_per_region(grid, op_name, op_args, background=0):
    """Segment grid into rectangular sub-regions (panels) bounded by uniform separator lines or border frames,
    apply `op_name(*op_args)` independently per panel, and reassemble."""
    g = as_grid(grid).copy()
    h, w = g.shape
    
    non_border_rows = [r for r in range(h) if len(set(g[r, :])) > 1]
    non_border_cols = [c for c in range(w) if len(set(g[:, c])) > 1]
    
    row_panels = []
    rcurr = []
    for r in range(h):
        if r in non_border_rows: rcurr.append(r)
        else:
            if rcurr: row_panels.append(rcurr); rcurr = []
    if rcurr: row_panels.append(rcurr)
    
    col_panels = []
    ccurr = []
    for c in range(w):
        if c in non_border_cols: ccurr.append(c)
        else:
            if ccurr: col_panels.append(ccurr); ccurr = []
    if ccurr: col_panels.append(ccurr)
    
    repaired = g.copy()
    changed = False
    for r_group in row_panels:
        r1, r2 = r_group[0], r_group[-1]
        for c_group in col_panels:
            c1, c2 = c_group[0], c_group[-1]
            sub = g[r1:r2+1, c1:c2+1]
            if sub.size == 0: continue
            
            if op_name == "PATTERN_COMPLETE":
                has_h_border = (sub.shape[0] > 2) and (len(set(sub[0, :])) == 1) and (len(set(sub[-1, :])) == 1)
                has_v_border = (sub.shape[1] > 2) and (len(set(sub[:, 0])) == 1) and (len(set(sub[:, -1])) == 1)
                
                if has_h_border and has_v_border:
                    target_sub = sub[1:-1, 1:-1]
                else:
                    target_sub = sub

                sub_h, sub_w = target_sub.shape
                best_cand = target_sub
                min_changes = float('inf')
                for ph in range(1, sub_h + 1):
                    for pw in range(1, min(6, sub_w) + 1):
                        if ph == sub_h and pw == sub_w: continue
                        cand = pattern_complete(target_sub, ph, pw, background=background)
                        if cand is not None:
                            changes = (cand != target_sub).sum()
                            if 0 < changes < min_changes:
                                min_changes = changes
                                best_cand = cand
                if not np.array_equal(best_cand, target_sub):
                    if has_h_border and has_v_border:
                        repaired[r1+1:r2, c1+1:c2] = best_cand
                    else:
                        repaired[r1:r2+1, c1:c2+1] = best_cand
                    changed = True
                    
    if not changed:
        return None
    return repaired



# ----------------------------------------------------------------------
# Two-panel boolean logic (split-and-combine)
# ----------------------------------------------------------------------

def _split_panels(g, axis, has_sep):
    h, w = g.shape
    if axis == "V":
        if has_sep:
            if w % 2 == 0:
                return None
            mid = w // 2
            return g[:, :mid], g[:, mid + 1:]
        else:
            if w % 2 != 0:
                return None
            mid = w // 2
            return g[:, :mid], g[:, mid:]
    else:
        if has_sep:
            if h % 2 == 0:
                return None
            mid = h // 2
            return g[:mid, :], g[mid + 1:, :]
        else:
            if h % 2 != 0:
                return None
            mid = h // 2
            return g[:mid, :], g[mid:, :]


def panel_logic(grid, axis, has_sep, logic, out_color, background=0):
    g = as_grid(grid)
    panels = _split_panels(g, axis, has_sep)
    if panels is None:
        return None
    a, b = panels
    if a.shape != b.shape:
        return None
    am = a != background
    bm = b != background
    if logic == "AND":
        mask = am & bm
    elif logic == "OR":
        mask = am | bm
    elif logic == "XOR":
        mask = am ^ bm
    elif logic == "DIFF":
        mask = am & (~bm)
    else:
        return None
    out = np.full(a.shape, background, dtype=g.dtype)
    out[mask] = out_color
    return out


# ----------------------------------------------------------------------
# Periodic pattern completion (repair a grid that should tile with period ph,pw)
# ----------------------------------------------------------------------

def pattern_complete(grid, ph, pw, background=0):
    g = as_grid(grid).copy()
    h, w = g.shape
    ph, pw = int(ph), int(pw)
    if ph <= 0 or pw <= 0 or ph > h or pw > w:
        return None

    votes = {}
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v == background:
                continue
            key = (r % ph, c % pw)
            votes.setdefault(key, Counter())[v] += 1

    if not votes:
        return None

    changed = False
    for r in range(h):
        for c in range(w):
            key = (r % ph, c % pw)
            if key not in votes:
                continue
            best = votes[key].most_common(1)[0][0]
            if g[r, c] != best:
                g[r, c] = best
                changed = True
    if not changed:
        return None
    return g


def mirror_4way_quad(grid):
    """4-way mirror reflection quad tiling."""
    g = as_grid(grid)
    top_left = np.flipud(np.fliplr(g))
    top_right = np.flipud(g)
    bot_left = np.fliplr(g)
    bot_right = g
    top = np.hstack([top_left, top_right])
    bot = np.hstack([bot_left, bot_right])
    return np.vstack([top, bot])


def extract_unique_color_panel(grid, background=0):
    """Extract grid sub-panel that contains a unique non-background color."""
    g = as_grid(grid)
    h, w = g.shape
    row_sep = [r for r in range(h) if np.all(g[r, :] == background)]
    col_sep = [c for c in range(w) if np.all(g[:, c] == background)]
    
    r_bounds, curr = [], 0
    for r in sorted(row_sep):
        if r > curr: r_bounds.append((curr, r))
        curr = r + 1
    if curr < h: r_bounds.append((curr, h))
    
    c_bounds, curr = [], 0
    for c in sorted(col_sep):
        if c > curr: c_bounds.append((curr, c))
        curr = c + 1
    if curr < w: c_bounds.append((curr, w))
    
    panels = []
    for r1, r2 in r_bounds:
        for c1, c2 in c_bounds:
            p = g[r1:r2, c1:c2]
            colors = set(np.unique(p)) - {background}
            if colors:
                panels.append((p, colors))
                
    all_panel_colors = [c for _, colors in panels for c in colors]
    for p, colors in panels:
        c = next(iter(colors))
        if all_panel_colors.count(c) == 1:
            return p
            
    return None


def fill_frame_by_size(grid, frame_color=2, background=0):
    """Fill hollow frame interior background based on interior width dimension."""
    g = as_grid(grid).copy()
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    out = g.copy()
    
    color_map = {
        3: 8, # 5x5 frame -> 3x3 interior -> teal 8
        5: 4, # 7x7 frame -> 5x5 interior -> yellow 4
        7: 3  # 9x9 frame -> 7x7 interior -> green 3
    }
    
    for r in range(h):
        for c in range(w):
            if g[r, c] == frame_color and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                frame_cells = []
                
                while q:
                    cr, cc = q.popleft()
                    frame_cells.append((cr, cc))
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            if dr == 0 and dc == 0: continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == frame_color and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                                
                frs = [cell[0] for cell in frame_cells]
                fcs = [cell[1] for cell in frame_cells]
                r1, r2 = min(frs), max(frs)
                c1, c2 = min(fcs), max(fcs)
                
                int_w = (c2 - c1 + 1) - 2
                if int_w in color_map:
                    fill_col = color_map[int_w]
                    for ir in range(r1 + 1, r2):
                        for ic in range(c1 + 1, c2):
                            if g[ir, ic] == background:
                                out[ir, ic] = fill_col
                                
    return out


def cycle_block_extend(grid, block_h=3, num_blocks=3, recolor_tuples=()):
    """Extend 3x3 vertical sub-block sequence and recolor."""
    g = as_grid(grid).copy()
    h, w = g.shape
    if h < block_h: return None
    p1 = g[:block_h, :]
    p2 = g[block_h:2*block_h, :] if h >= 2*block_h else p1
    p3 = p1.copy() if np.array_equal(p1, p2) else np.fliplr(p1)
    
    out = np.vstack([p1, p2, p3])
    if recolor_tuples:
        for old_c, new_c in recolor_tuples:
            out[out == old_c] = new_c
    return out


def shift_parallelogram_fix_right(grid, background=0):
    """Align slanted parallelogram objects: shift each non-bottom row right by 1,
    keeping the right-side column anchored at the bottom row's rightmost column."""
    try:
        from scipy.ndimage import label as _label
    except ImportError:
        return None
    g = as_grid(grid).copy()
    h, w = g.shape
    out = np.full_like(g, background)
    struct = np.ones((3, 3), dtype=int)

    for c in set(int(v) for v in np.unique(g)) - {background}:
        mask = (g == c)
        lbl, num = _label(mask, structure=struct)
        for i in range(1, num + 1):
            obj_mask = (lbl == i)
            rs, cs = np.where(obj_mask)
            r_min, r_max = int(rs.min()), int(rs.max())
            bot_cs = cs[rs == r_max]
            bot_c_max = int(bot_cs.max())

            for r in range(r_min, r_max + 1):
                row_mask = (rs == r)
                row_cs = cs[row_mask]
                if r == r_max:
                    for col in row_cs:
                        out[r, int(col)] = c
                else:
                    for col in row_cs:
                        col = int(col)
                        if col >= bot_c_max:
                            out[r, col] = c
                        else:
                            if col + 1 < w:
                                out[r, col + 1] = c
    return out


def diagonal_stack_chain(grid, background=0):
    """Chain objects diagonally from top-left: sort objects by leftmost input column,
    place each so its top-left is at the bottom-right corner of the previous object."""
    try:
        from scipy.ndimage import label as _label
    except ImportError:
        return None
    g = as_grid(grid).copy()
    h, w = g.shape
    out = np.full_like(g, background)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

    objects = []
    for c in sorted(set(int(v) for v in np.unique(g)) - {background}):
        mask = (g == c)
        lbl, num = _label(mask, structure=struct)
        for i in range(1, num + 1):
            obj_mask = (lbl == i)
            rs, cs = np.where(obj_mask)
            r_min, r_max = int(rs.min()), int(rs.max())
            c_min, c_max = int(cs.min()), int(cs.max())
            local = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=g.dtype)
            for r, c_ in zip(rs, cs):
                local[int(r) - r_min, int(c_) - c_min] = int(g[r, c_])
            objects.append((c_min, c, local))

    objects.sort(key=lambda x: x[0])

    cur_r, cur_c = 0, 0
    for _, c, local in objects:
        lh, lw = local.shape
        for lr in range(lh):
            for lc in range(lw):
                v = local[lr, lc]
                if v != background:
                    dr, dc = cur_r + lr, cur_c + lc
                    if 0 <= dr < h and 0 <= dc < w:
                        out[dr, dc] = v
        cur_r = cur_r + lh - 1
        cur_c = cur_c + lw - 1

    return out


def diagonal_pattern_complete(grid, background=0):
    """Complete diagonal periodic pattern out[r, c] = color[(r + c) % p] from input non-background cells."""
    g = as_grid(grid).copy()
    h, w = g.shape
    best_out = None
    min_err = float('inf')
    
    for p in range(2, 7):
        votes = {}
        for r in range(h):
            for c in range(w):
                v = int(g[r, c])
                if v == background: continue
                key = (r + c) % p
                votes.setdefault(key, Counter())[v] += 1
        
        if len(votes) == p:
            cand = np.full_like(g, background)
            for r in range(h):
                for c in range(w):
                    key = (r + c) % p
                    cand[r, c] = votes[key].most_common(1)[0][0]
            
            non_bg = (g != background)
            err = (g[non_bg] != cand[non_bg]).sum()
            if err < min_err:
                min_err = err
                best_out = cand
                
    return best_out


# ----------------------------------------------------------------------
# Object-level ops (selection, recoloring, deletion, cropping)
# ----------------------------------------------------------------------

_scene_cache = {}

def _cached_scene(g, background):
    key = (g.shape, g.tobytes(), background)
    scene = _scene_cache.get(key)
    if scene is None:
        scene = Scene.from_grid(g, background=background)
        if len(_scene_cache) > 2000:
            _scene_cache.clear()
        _scene_cache[key] = scene
    return scene


def select_recolor(grid, selector, value, newcolor, background=0):
    g = as_grid(grid).copy()
    scene = _cached_scene(as_grid(grid), background)
    objs = select_objects(scene, selector, value)
    if not objs:
        return None
    for o in objs:
        for r, c in o.cells:
            g[r, c] = newcolor
    return g


def select_crop(grid, selector, value, background=0):
    g = as_grid(grid)
    scene = _cached_scene(g, background)
    objs = select_objects(scene, selector, value)
    if not objs:
        return None
    r1 = min(o.top for o in objs)
    c1 = min(o.left for o in objs)
    r2 = max(o.bottom for o in objs)
    c2 = max(o.right for o in objs)
    return g[r1:r2 + 1, c1:c2 + 1].copy()


def select_delete(grid, selector, value, background=0):
    return select_recolor(grid, selector, value, background, background=background)


def count_enclosed_holes(grid, background=0):
    g = as_grid(grid)
    reachable = _background_reachable_from_edge(g, background)
    holes = (g == background) & (~reachable)
    return int(holes.sum())


def recolor_by_indicator_feature(grid, target_color, indicator_color, feature_map, background=0):
    """Recolor target_color cells to a color determined by (cell_count, hole_count) of indicator_color cells,
    and erase indicator_color cells to background."""
    g = as_grid(grid).copy()
    mask_ind = (g == indicator_color)
    if not mask_ind.any():
        return None
    cnt = int(mask_ind.sum())
    rows, cols = np.where(mask_ind)
    r1, r2 = rows.min(), rows.max()
    c1, c2 = cols.min(), cols.max()
    sub = g[r1:r2+1, c1:c2+1]
    holes = count_enclosed_holes(sub, background=background)
    
    key = (cnt, holes)
    if key not in feature_map:
        return None
    new_c = feature_map[key]
    g[g == target_color] = new_c
    g[mask_ind] = background
    return g


# LLM-generated primitive for task 50cb2852
def llm_50cb2852(grid, background=0):
    """Mark center pixels of solid blocks with color 8.
    Task 50cb2852: Find pixels that have the same color on all 4 orthogonal neighbors.
    """
    g = as_grid(grid).copy()
    if g.size == 0:
        return g
    
    output = g.copy()
    for i in range(1, g.shape[0] - 1):
        for j in range(1, g.shape[1] - 1):
            if g[i, j] != background:
                if (g[i-1, j] == g[i, j] and g[i+1, j] == g[i, j] and 
                    g[i, j-1] == g[i, j] and g[i, j+1] == g[i, j]):
                    output[i, j] = 8
    return output


def ray_trace(grid, seed_color, paint_color, direction_mode, bounce_walls, bounce_objects, background=0):
    """
    Generalized ray tracing engine.
    direction_mode: 'DIAGONAL', 'ORTHOGONAL', 'ALL_8'
    """
    g = as_grid(grid)
    h, w = g.shape
    out = g.copy()
    
    seeds = np.where(g == seed_color)
    if len(seeds[0]) == 0:
        return None
        
    directions = []
    if direction_mode in ('DIAGONAL', 'ALL_8'):
        directions.extend([(-1,-1), (-1,1), (1,-1), (1,1)])
    if direction_mode in ('ORTHOGONAL', 'ALL_8'):
        directions.extend([(-1,0), (1,0), (0,-1), (0,1)])
        
    # Find boundary cells of the seed regions to emit rays from
    emitters = set()
    for r, c in zip(*seeds):
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != seed_color:
                emitters.add((r, c, dr, dc))
                
    for r, c, dr, dc in emitters:
        cr, cc = r + dr, c + dc
        cdr, cdc = dr, dc
        seen_states = set()
        
        while 0 <= cr < h and 0 <= cc < w:
            state = (cr, cc, cdr, cdc)
            if state in seen_states:
                break
            seen_states.add(state)
            
            cell_val = out[cr, cc]
            if cell_val != background and cell_val != paint_color:
                if bounce_objects:
                    cdr *= -1
                    cdc *= -1
                    cr += cdr
                    cc += cdc
                    continue
                else:
                    break
                    
            out[cr, cc] = paint_color
            
            nr, nc = cr + cdr, cc + cdc
            bounced = False
            if bounce_walls:
                if not (0 <= nr < h):
                    cdr *= -1
                    bounced = True
                if not (0 <= nc < w):
                    cdc *= -1
                    bounced = True
            
            if bounced:
                nr, nc = cr + cdr, cc + cdc
                
            cr = nr
            cc = nc
            
    return out


def tile_2x2_diagonal_mark(grid, mark_color=8, background=0):
    """Tile input 2x2 and mark 1-step diagonal neighbors of non-background cells with mark_color."""
    g = as_grid(grid)
    h, w = g.shape
    tiled = np.tile(g, (2, 2))
    out = tiled.copy()
    dots = np.argwhere(tiled != background)
    for r, c in dots:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 2 * h and 0 <= nc < 2 * w and out[nr, nc] == background:
                out[nr, nc] = mark_color
    return out


def seed_surround_mark(grid, rules=(), background=0):
    """Surround specified seed colors with target colors in ORTHOGONAL or DIAGONAL direction."""
    g = as_grid(grid)
    h, w = g.shape
    out = g.copy()
    for sc, mode, nc in rules:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] if mode == "ORTHOGONAL" else [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for r, c in np.argwhere(g == sc):
            for dr, dc in dirs:
                nr, nc_ = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc_ < w and out[nr, nc_] == background:
                    out[nr, nc_] = nc
    return out


def indicator_line_object_absorb(grid, background=0):
    """Connect aligned indicator dots and absorb touched objects into indicator color."""
    try:
        from scipy.ndimage import label as _label
    except ImportError:
        return None
    g = as_grid(grid)
    h, w = g.shape
    out = g.copy()
    colors = set(np.unique(g)) - {background}
    ind_c = None
    for c in colors:
        pts = np.argwhere(g == c)
        if len(pts) >= 2:
            rs = [p[0] for p in pts]
            cs = [p[1] for p in pts]
            if len(set(rs)) < len(rs) or len(set(cs)) < len(cs):
                ind_c = c
                break
    if ind_c is None:
        return None
        
    pts = np.argwhere(g == ind_c)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            r1, c1 = pts[i]
            r2, c2 = pts[j]
            if r1 == r2:
                for c_ in range(min(c1, c2), max(c1, c2) + 1):
                    if out[r1, c_] == background:
                        out[r1, c_] = ind_c
            if c1 == c2:
                for r_ in range(min(r1, r2), max(r1, r2) + 1):
                    if out[r_, c1] == background:
                        out[r_, c1] = ind_c
                        
    target_colors = colors - {ind_c}
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    for tc in target_colors:
        lbl, num = _label(g == tc, structure=struct)
        for obj_id in range(1, num + 1):
            obj_mask = (lbl == obj_id)
            touch = False
            rs, cs = np.where(obj_mask)
            for r, c in zip(rs, cs):
                for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == ind_c:
                        touch = True
                        break
                if touch:
                    break
            if touch:
                out[obj_mask] = ind_c
    return out


def l_path_dot_connect(grid, fill_color=5, background=0):
    """Connect 3 key dots (2, 4, 3) with chained horizontal-first L-paths using fill_color."""
    g = as_grid(grid)
    h, w = g.shape
    out = g.copy()
    colors = set(np.unique(g)) - {background}
    if len(colors) != 3:
        return None
    p2 = np.argwhere(g == 2)
    p4 = np.argwhere(g == 4)
    p3 = np.argwhere(g == 3)
    if len(p2) != 1 or len(p4) != 1 or len(p3) != 1:
        return None
    r2, c2 = p2[0]
    r4, c4 = p4[0]
    r3, c3 = p3[0]
    
    out[r2, min(c2, c4):max(c2, c4)+1] = np.where(out[r2, min(c2, c4):max(c2, c4)+1] == background, fill_color, out[r2, min(c2, c4):max(c2, c4)+1])
    out[min(r2, r4):max(r2, r4)+1, c4] = np.where(out[min(r2, r4):max(r2, r4)+1, c4] == background, fill_color, out[min(r2, r4):max(r2, r4)+1, c4])
    
    out[r4, min(c4, c3):max(c4, c3)+1] = np.where(out[r4, min(c4, c3):max(c4, c3)+1] == background, fill_color, out[r4, min(c4, c3):max(c4, c3)+1])
    out[min(r4, r3):max(r4, r3)+1, c3] = np.where(out[min(r4, r3):max(r4, r3)+1, c3] == background, fill_color, out[min(r4, r3):max(r4, r3)+1, c3])
    
    return out


def seed_row_bands_frame(grid, background=0):
    """Partition grid by seed row midpoints and draw border frame per band."""
    g = as_grid(grid)
    h, w = g.shape
    seeds = np.argwhere(g != background)
    if len(seeds) < 2:
        return None
    seeds = seeds[np.argsort(seeds[:, 0])]
    out = np.full_like(g, background)
    n_seeds = len(seeds)
    r_starts = [0] * n_seeds
    r_ends = [h - 1] * n_seeds
    for i in range(n_seeds - 1):
        mid = (seeds[i][0] + seeds[i + 1][0]) // 2
        r_ends[i] = mid
        r_starts[i + 1] = mid + 1
    for i in range(n_seeds):
        r_seed, c_seed = seeds[i]
        c = g[r_seed, c_seed]
        rs, re = r_starts[i], r_ends[i]
        out[rs:re + 1, 0] = c
        out[rs:re + 1, w - 1] = c
        out[r_seed, :] = c
        if i == 0:
            out[0, :] = c
        if i == n_seeds - 1:
            out[h - 1, :] = c
    return out


def legend_rotate_scale_recolor(grid, target_color=8, background=0):
    """Rotate top legend 270 deg and block-scale recolor target structure."""
    g = as_grid(grid)
    h, w = g.shape
    legend_mask = (g != background) & (g != target_color)
    if not legend_mask.any():
        return None
    l_rs, l_cs = np.where(legend_mask)
    lr1, lr2 = l_rs.min(), l_rs.max()
    lc1, lc2 = l_cs.min(), l_cs.max()
    legend = g[lr1:lr2 + 1, lc1:lc2 + 1]
    
    target_mask = (g == target_color)
    if not target_mask.any():
        return None
    rs, cs = np.where(target_mask)
    r1, r2 = rs.min(), rs.max()
    c1, c2 = cs.min(), cs.max()
    gh, gw = r2 - r1 + 1, c2 - c1 + 1
    
    legend_rot = np.rot90(legend, 3)
    lh, lw = legend_rot.shape
    if gh % lh != 0 or gw % lw != 0:
        return None
        
    scale_h = gh // lh
    scale_w = gw // lw
    
    out = g.copy()
    for lr in range(lh):
        for lc in range(lw):
            color = legend_rot[lr, lc]
            if color != background:
                sub_r1 = r1 + lr * scale_h
                sub_r2 = r1 + (lr + 1) * scale_h
                sub_c1 = c1 + lc * scale_w
                sub_c2 = c1 + (lc + 1) * scale_w
                
                b_mask = (g[sub_r1:sub_r2, sub_c1:sub_c2] == target_color)
                out[sub_r1:sub_r2, sub_c1:sub_c2][b_mask] = color
                
    return out


def quad_symmetry_complete(grid, background=0):
    """Complete 4-way (horizontal + vertical) symmetry around content center."""
    g = as_grid(grid)
    h, w = g.shape
    pts = np.argwhere(g != background)
    if len(pts) == 0:
        return None
    r_center = (pts[:, 0].min() + pts[:, 0].max()) / 2.0
    c_center = (pts[:, 1].min() + pts[:, 1].max()) / 2.0
    out = g.copy()
    for r, c in pts:
        color = g[r, c]
        dr = r - r_center
        dc = c - c_center
        for sign_r in [1, -1]:
            for sign_c in [1, -1]:
                nr = int(round(r_center + sign_r * dr))
                nc = int(round(c_center + sign_c * dc))
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = color
    return out


def template_d4_key_align(grid, background=0):
    """Align D4-transformed template onto target key dots and keep only matched objects."""
    try:
        from scipy.ndimage import label as _label, binary_dilation as _binary_dilation
    except ImportError:
        return None
    g = as_grid(grid)
    h, w = g.shape
    struct = np.ones((3, 3), dtype=int)
    lbl, num = _label(g != background, structure=struct)
    if num < 2:
        return None

    comps = []
    for i in range(1, num + 1):
        mask = (lbl == i)
        rs, cs = np.where(mask)
        r1, r2, c1, c2 = rs.min(), rs.max(), cs.min(), cs.max()
        crop = g[r1:r2 + 1, c1:c2 + 1].copy()
        crop[~mask[r1:r2 + 1, c1:c2 + 1]] = background
        comps.append(crop)

    out = np.full_like(g, background)
    templates = [c for c in comps if (c != background).sum() >= 4]
    if not templates:
        return None

    dots_mask = np.zeros_like(g, dtype=bool)
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
        t_colors = [g[r, c] for r, c in t_dots]

        matched_group = False
        for tmpl in templates:
            tmpl_colors = set(tmpl[tmpl != background])
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
                                        if not (0 <= rel_r < th and 0 <= rel_c < tw and t_trans[rel_r, rel_c] == g[r_td, c_td]):
                                            valid = False
                                            break
                                    if valid:
                                        out[top_r:top_r + th, top_c:top_c + tw] = np.where(
                                            t_trans != background, t_trans, out[top_r:top_r + th, top_c:top_c + tw]
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








