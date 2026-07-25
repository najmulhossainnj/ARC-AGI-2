"""
Data-driven parameter proposal for the advanced transform families.

Rather than blindly enumerating every possible argument combination (which
would blow up the search space), each `learn_*` function inspects the
training pairs and proposes a small, bounded list of plausible parameter
values. The beam search still verifies candidates by exact match on all
training pairs, so a wrong guess here just costs one wasted candidate
rather than a wrong answer.
"""
from __future__ import annotations
from collections import Counter
import numpy as np

from ..core.grid import as_grid


def _unique_colors(grid):
    return set(int(x) for x in np.unique(as_grid(grid)))


def learn_colormap(train_pairs):
    """A single consistent input-color -> output-color mapping, if one exists."""
    mapping = {}
    for inp, out in train_pairs:
        inp, out = as_grid(inp), as_grid(out)
        if inp.shape == out.shape:
            for old, new in zip(inp.flatten().tolist(), out.flatten().tolist()):
                if old in mapping and mapping[old] != new:
                    return []
                mapping[old] = new
        else:
            in_cols = set(np.unique(inp)) - {0}
            out_cols = set(np.unique(out)) - {0}
            if len(in_cols) == 1 and len(out_cols) == 1:
                mapping[next(iter(in_cols))] = next(iter(out_cols))
            else:
                return []
    if all(k == v for k, v in mapping.items()):
        return []
    return [tuple(sorted(mapping.items()))]


def learn_tile_factors(train_pairs):
    factors = set()
    for inp, out in train_pairs:
        inp, out = as_grid(inp), as_grid(out)
        ih, iw = inp.shape
        oh, ow = out.shape
        if ih == 0 or iw == 0 or oh % ih != 0 or ow % iw != 0:
            return []
        factors.add((oh // ih, ow // iw))
    if len(factors) != 1:
        return []
    rh, rw = next(iter(factors))
    if rh == 1 and rw == 1:
        return []
    if rh > 6 or rw > 6:
        return []
    return [(rh, rw)]


def learn_fractal_tile(train_pairs, background=0):
    """FRACTAL_TILE has no parameters beyond background, so this just
    checks whether the shape ratio is consistent with self-replication
    (output is exactly input-shape-squared) before letting the beam
    search verify the single candidate end to end."""
    for inp, out in train_pairs:
        inp, out = as_grid(inp), as_grid(out)
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh != ih * ih or ow != iw * iw:
            return []
    return [background]


def learn_reflect_tile(train_pairs):
    """Propose (rh, rw, row_flip, col_flip) candidates for REFLECT_TILE:
    reuse the shape-ratio detection from `learn_tile_factors`, then try
    all 9 (row_flip, col_flip) combinations -- None/'H'/'V' on each axis
    -- verifying each end to end. Cheap: at most 9 candidates, each just a
    handful of array comparisons.
    """
    from ..dsl.advanced_transforms import reflect_tile

    factors = learn_tile_factors(train_pairs)
    if not factors:
        return []
    rh, rw = factors[0]

    found = []
    for row_flip in (None, "H", "V"):
        for col_flip in (None, "H", "V"):
            if row_flip is None and col_flip is None:
                continue  # that's just plain TILE, already its own family
            ok = True
            for inp, out in train_pairs:
                pred = reflect_tile(as_grid(inp), rh, rw, row_flip, col_flip)
                if pred is None or pred.shape != as_grid(out).shape or \
                   not np.array_equal(pred, as_grid(out)):
                    ok = False
                    break
            if ok:
                found.append((rh, rw, row_flip, col_flip))
    return found


def learn_downscale_factors(train_pairs):
    factors = set()
    for inp, out in train_pairs:
        inp, out = as_grid(inp), as_grid(out)
        ih, iw = inp.shape
        oh, ow = out.shape
        if oh == 0 or ow == 0 or ih % oh != 0 or iw % ow != 0:
            return []
        factors.add((ih // oh, iw // ow))
    if len(factors) != 1:
        return []
    fh, fw = next(iter(factors))
    if fh == 1 and fw == 1:
        return []
    return [(fh, fw)]


def learn_border(train_pairs):
    widths = set()
    colors = set()
    for inp, out in train_pairs:
        inp, out = as_grid(inp), as_grid(out)
        ih, iw = inp.shape
        oh, ow = out.shape
        dh, dw = oh - ih, ow - iw
        if dh <= 0 or dw <= 0 or dh % 2 != 0 or dw % 2 != 0:
            return []
        wh, ww = dh // 2, dw // 2
        if wh != ww:
            return []
        if not np.array_equal(out[wh:wh + ih, ww:ww + iw], inp):
            return []
        widths.add(wh)
        colors.add(int(out[0, 0]))
    if len(widths) != 1 or len(colors) != 1:
        return []
    return [(next(iter(colors)), next(iter(widths)))]


def learn_fill_holes_color(train_pairs):
    from ..dsl.advanced_transforms import _background_reachable_from_edge

    candidates = set()
    for inp, out in train_pairs:
        inp, out = as_grid(inp), as_grid(out)
        if inp.shape != out.shape:
            return []
        reachable = _background_reachable_from_edge(inp, 0)
        holes = (inp == 0) & (~reachable)
        if not holes.any():
            continue
        hole_vals = set(int(v) for v in out[holes])
        if len(hole_vals) != 1:
            return []
        candidates.add(next(iter(hole_vals)))
        # everything outside holes must be unchanged
        unchanged = ~holes
        if not np.array_equal(inp[unchanged], out[unchanged]):
            return []
    if len(candidates) != 1:
        return []
    return [next(iter(candidates))]


def symmetry_noise_candidates(train_pairs, limit=10):
    from collections import Counter
    counts = Counter()
    for inp, _ in train_pairs:
        counts.update(as_grid(inp).flatten())
    sorted_colors = [col for col, _ in counts.most_common() if col != 0][::-1]
    if 0 in counts:
        sorted_colors.append(0)
    return sorted_colors[:limit]


def learn_panel_logic_layout(train_pairs):
    """Detect a consistent split axis / separator presence from the inputs."""
    layouts = set()
    for inp, _ in train_pairs:
        inp = as_grid(inp)
        h, w = inp.shape
        found = []
        if w % 2 == 1:
            found.append(("V", True))
        if w % 2 == 0:
            found.append(("V", False))
        if h % 2 == 1:
            found.append(("H", True))
        if h % 2 == 0:
            found.append(("H", False))
        layouts.add(tuple(found))
    # intersect possibilities across all pairs
    common = set(layouts.pop())
    for l in layouts:
        common &= set(l)
    return list(common)


def learn_pattern_periods(train_pairs, max_period=6):
    """Small set of (ph, pw) periods worth trying, from the first same-shape pair."""
    periods = set()
    for inp, out in train_pairs:
        inp, out = as_grid(inp), as_grid(out)
        if inp.shape != out.shape:
            continue
        h, w = inp.shape
        for ph in range(1, min(max_period, h) + 1):
            for pw in range(1, min(max_period, w) + 1):
                if ph == h and pw == w:
                    continue
                periods.add((ph, pw))
    return list(periods)


def learn_recolor_by_indicator_feature(train_pairs, background=0):
    """Learn target_color, indicator_color, and feature_map mapping (count, holes) -> new_color."""
    candidates = []
    inp0, out0 = train_pairs[0]
    inp0, out0 = as_grid(inp0), as_grid(out0)
    if inp0.shape != out0.shape:
        return candidates

    from ..dsl.advanced_transforms import count_enclosed_holes

    colors_in = set(np.unique(inp0)) - {background}

    for target_c in colors_in:
        for ind_c in colors_in:
            if target_c == ind_c: continue
            feat_map = {}
            valid = True
            for inp, out in train_pairs:
                inp, out = as_grid(inp), as_grid(out)
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
                out_cols = np.unique(out[target_mask])
                if len(out_cols) != 1:
                    valid = False; break
                
                feat_key = (int(cnt), int(holes))
                feat_map[feat_key] = int(out_cols[0])
            if valid and len(feat_map) > 0:
                candidates.append((target_c, ind_c, feat_map))
    return candidates


def learn_object_translation(train_pairs, background=0, max_colors=9, max_objects=20):
    """Propose (selector, dr, dc, erase) combos for TRANSLATE_OBJECT: every
    matching object shifts by the same fixed vector in every training pair.

    Unlike `learn_object_relocation` (anchor-relative sliding), the offset
    here is inferred directly from the correspondence matches rather than
    searched combinatorially, so this stays cheap regardless of how many
    colors/objects are in the scene.
    """
    from ..perception.extractor import extract_objects
    from ..correspondence.matcher import match_objects
    from ..correspondence.changes import translation
    from ..dsl.selectors import select_objects
    from ..dsl.object_ops import translate_object

    class _ObjBag:
        def __init__(self, objects):
            self.objects = objects

    pairs_np = [(as_grid(inp), as_grid(out)) for inp, out in train_pairs]
    if any(inp.shape != out.shape for inp, out in pairs_np):
        return []

    colors = set()
    per_pair = []  # (in_objs, {obj_id: (dr,dc) or None})
    max_objects_seen = 0
    for inp, out in pairs_np:
        colors |= _unique_colors(inp)
        in_objs = extract_objects(inp, background=background)
        out_objs = extract_objects(out, background=background)
        max_objects_seen = max(max_objects_seen, len(in_objs), len(out_objs))
        matches = match_objects(in_objs, out_objs, min_score=0.5)
        vec_by_id = {a.id: translation(a, b) for a, b, _ in matches}
        per_pair.append((in_objs, vec_by_id))

    if max_objects_seen > max_objects:
        return []

    colors = sorted(colors - {background})[:max_colors]
    selectors = [("COLOR", c) for c in colors]
    selectors += [("LARGEST", None), ("SMALLEST", None), ("UNIQUE_COLOR", None)]

    found = []
    for sel, val in selectors:
        vecs = set()
        valid = True
        any_selected = False
        for in_objs, vec_by_id in per_pair:
            selected = select_objects(_ObjBag(in_objs), sel, val)
            if not selected:
                continue
            any_selected = True
            for o in selected:
                v = vec_by_id.get(o.id)
                if v is None:
                    valid = False
                    break
                vecs.add(v)
            if not valid:
                break
        if not (valid and any_selected and len(vecs) == 1):
            continue
        dr, dc = next(iter(vecs))
        if (dr, dc) == (0, 0):
            continue
        for erase in (True, False):
            ok = True
            for inp, out in pairs_np:
                pred = translate_object(inp, sel, val, dr, dc, erase, background)
                if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
                    ok = False
                    break
            if ok:
                found.append((sel, val, dr, dc, erase))
    return found


def learn_object_relocation(train_pairs, background=0, max_colors=5):
    """Propose (mover_selector, anchor_selector, relation, erase) combos for
    RELOCATE_OBJECT / COPY_OBJECT.

    This mirrors the other `learn_*` functions: rather than reasoning
    abstractly about "where did the object go", it proposes a bounded set of
    (selector, anchor, relation) combos built from the relation vocabulary in
    `dsl.object_ops.RELOCATE_RELATIONS`, and only keeps the ones that
    reproduce every training pair exactly. The correspondence matcher is used
    first as a filter: if no input object plausibly maps to a moved output
    object in *any* training pair (i.e. nothing actually moved), there's no
    point trying every selector/anchor/relation combination.
    """
    from ..perception.extractor import extract_objects
    from ..correspondence.matcher import match_objects
    from ..correspondence.changes import translation
    from .. import dsl as _dsl  # noqa: F401  (avoid unused warnings in some linters)
    from ..dsl.object_ops import RELOCATE_RELATIONS, relocate_object, copy_object

    pairs_np = [(as_grid(inp), as_grid(out)) for inp, out in train_pairs]

    # RELOCATE_OBJECT/COPY_OBJECT only remap within a fixed grid shape.
    if any(inp.shape != out.shape for inp, out in pairs_np):
        return []

    something_moved = False
    colors = set()
    max_objects_seen = 0
    for inp, out in pairs_np:
        colors |= _unique_colors(inp)
        in_objs = extract_objects(inp, background=background)
        out_objs = extract_objects(out, background=background)
        max_objects_seen = max(max_objects_seen, len(in_objs), len(out_objs))
        matches = match_objects(in_objs, out_objs, min_score=0.5)
        for a, b, _ in matches:
            if translation(a, b) not in (None, (0, 0)):
                something_moved = True
    if not something_moved:
        return []
    # This relation pattern (slide/copy one object relative to another) is a
    # small, targeted family. Dense/noisy scenes blow up the selector/anchor
    # combinatorics for no real benefit -- real "relocate toward X" tasks
    # have a handful of objects, not dozens.
    if max_objects_seen > 12:
        return []

    colors = sorted(colors - {background})[:max_colors]
    mover_selectors = [("COLOR", c) for c in colors]
    mover_selectors += [("LARGEST", None), ("SMALLEST", None), ("UNIQUE_COLOR", None)]
    anchor_selectors = [("COLOR", c) for c in colors]
    anchor_selectors += [("LARGEST", None), ("SMALLEST", None)]

    found = []
    tried = 0
    max_tried = 400
    for m_sel, m_val in mover_selectors:
        for a_sel, a_val in anchor_selectors:
            if m_sel == a_sel and m_val == a_val:
                continue
            for relation in RELOCATE_RELATIONS:
                for erase, fn in ((True, relocate_object), (False, copy_object)):
                    tried += 1
                    if tried > max_tried:
                        return found
                    ok = True
                    for inp, out in pairs_np:
                        pred = fn(inp, m_sel, m_val, a_sel, a_val, relation, background)
                        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
                            ok = False
                            break
                    if ok:
                        found.append((m_sel, m_val, a_sel, a_val, relation, erase))
    return found


def select_recolor_candidates(train_pairs):
    """Bounded (selector, value) list plus plausible new colors."""
    selectors = [("LARGEST", None), ("SMALLEST", None), ("UNIQUE_COLOR", None)]
    in_colors = set()
    out_colors = set()
    for inp, out in train_pairs:
        in_colors |= _unique_colors(inp)
        out_colors |= _unique_colors(out)
    for c in sorted(in_colors)[:9]:
        selectors.append(("COLOR", c))

    new_colors = sorted(out_colors - in_colors)
    new_colors.append(0)
    new_colors = sorted(set(new_colors))[:4]
    return selectors, new_colors


def _bounded_selectors(train_pairs, background=0, max_colors=9):
    """Shared (selector, value) candidate list for the object-level
    learners below: LARGEST/SMALLEST/UNIQUE_COLOR plus one per color
    actually present. Kept separate from `select_recolor_candidates`
    (which also proposes target colors) since these callers don't need
    that half."""
    colors = set()
    for inp, _ in train_pairs:
        colors |= _unique_colors(inp)
    colors = sorted(colors - {background})[:max_colors]
    selectors = [("LARGEST", None), ("SMALLEST", None), ("UNIQUE_COLOR", None)]
    selectors += [("COLOR", c) for c in colors]
    return selectors


def learn_delete_objects(train_pairs, background=0):
    """Propose (selector, value) candidates for DELETE_OBJECTS: every
    object matching the selector is consistently erased in every training
    pair, and nothing else in the grid changes. This is the generalization
    of `match_objects`' 'removed' list -- rather than reasoning about which
    *specific* objects were removed, it looks for a selector that would
    have picked out exactly those objects every time.
    """
    from ..dsl.object_ops import delete_objects

    pairs_np = [(as_grid(inp), as_grid(out)) for inp, out in train_pairs]
    if any(inp.shape != out.shape for inp, out in pairs_np):
        return []

    found = []
    for sel, val in _bounded_selectors(train_pairs, background):
        ok = True
        for inp, out in pairs_np:
            pred = delete_objects(inp, sel, val, background)
            if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
                ok = False
                break
        if ok:
            found.append((sel, val))
    return found


def learn_rank_recolor(train_pairs, background=0, max_rank=8):
    """Propose a (rank_key, mapping) candidate for RANK_RECOLOR.

    Uses `match_objects` + `classify_change` (the same correspondence
    machinery `learn_object_translation` uses for 'move') to find objects
    whose change kind is 'recolor', then checks whether the new color is
    consistent for every object at the same size-rank across every
    training pair -- e.g. "the largest object always turns red, the
    second-largest always turns blue". Bails (returns []) the moment any
    pair needs something this family can't express (an add/delete, a
    recolor to more than one color, a non-recolor change), rather than
    guessing; the final candidate is still re-verified end to end before
    being returned.
    """
    from ..perception.extractor import extract_objects
    from ..correspondence.matcher import match_objects
    from ..correspondence.changes import classify_change
    from ..dsl.object_ops import _size_rank, recolor_by_rank

    pairs_np = [(as_grid(inp), as_grid(out)) for inp, out in train_pairs]
    if any(inp.shape != out.shape for inp, out in pairs_np):
        return []

    mapping = {}
    for inp, out in pairs_np:
        in_objs = extract_objects(inp, background=background)
        out_objs = extract_objects(out, background=background)
        if len(in_objs) > max_rank * 3 or not in_objs:
            return []
        ranks = _size_rank(in_objs)
        matches = match_objects(in_objs, out_objs, min_score=0.5)
        if len(matches) != len(in_objs):
            return []  # something was added/removed -- not a pure recolor
        for a, b, _ in matches:
            kind = classify_change(a, b)
            if kind == "unchanged":
                continue
            if kind != "recolor" or len(b.colors) != 1:
                return []
            r = ranks[a.id]
            if r >= max_rank:
                return []
            new_color = b.colors[0]
            if mapping.get(r, new_color) != new_color:
                return []
            mapping[r] = new_color
    if not mapping:
        return []

    candidate = ("size_desc", tuple(sorted(mapping.items())))
    for inp, out in pairs_np:
        pred = recolor_by_rank(inp, *candidate, background)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return []
    return [candidate]


def learn_rank_resize(train_pairs, background=0, max_rank=8, max_factor=4):
    """Propose a (rank_key, mapping) candidate for RANK_RESIZE. Mirrors
    `learn_rank_recolor` but for the 'resize' change kind: objects at the
    same size-rank grow by a consistent integer factor, anchored at their
    top-left corner, across every training pair -- e.g. "the largest
    object doubles in size, the smallest triples".
    """
    from ..perception.extractor import extract_objects
    from ..correspondence.matcher import match_objects
    from ..correspondence.changes import classify_change
    from ..dsl.object_ops import _size_rank, resize_objects_by_rank

    pairs_np = [(as_grid(inp), as_grid(out)) for inp, out in train_pairs]
    if any(inp.shape != out.shape for inp, out in pairs_np):
        return []

    mapping = {}
    for inp, out in pairs_np:
        in_objs = extract_objects(inp, background=background)
        out_objs = extract_objects(out, background=background)
        if len(in_objs) > max_rank * 3 or not in_objs:
            return []
        ranks = _size_rank(in_objs)
        matches = match_objects(in_objs, out_objs, min_score=0.3)
        if len(matches) != len(in_objs):
            return []
        for a, b, _ in matches:
            kind = classify_change(a, b)
            if kind == "unchanged":
                continue
            if kind != "resize" or a.colors != b.colors:
                return []
            if b.top != a.top or b.left != a.left:
                return []
            if a.height == 0 or a.width == 0:
                return []
            if b.height % a.height or b.width % a.width:
                return []
            fh, fw = b.height // a.height, b.width // a.width
            if fh != fw or fh < 2 or fh > max_factor:
                return []
            r = ranks[a.id]
            if r >= max_rank:
                return []
            if mapping.get(r, fh) != fh:
                return []
            mapping[r] = fh
    if not mapping:
        return []

    candidate = ("size_desc", tuple(sorted(mapping.items())))
    for inp, out in pairs_np:
        pred = resize_objects_by_rank(inp, *candidate, background)
        if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
            return []
    return [candidate]


def learn_objects_to_strip(train_pairs, background=0):
    """Propose (axis, order) candidates for OBJECTS_TO_STRIP: the output
    is a 1 x N or N x 1 grid, one cell per input object holding that
    object's own color. Only 4 (axis, order) combinations exist, so each
    is just tried directly and verified end to end -- no piecewise
    inference needed, unlike the rank-based learners above.
    """
    from ..dsl.object_ops import objects_to_strip

    pairs_np = [(as_grid(inp), as_grid(out)) for inp, out in train_pairs]
    candidates = []
    for axis in ("ROW", "COL"):
        for order in ("size_desc", "reading"):
            ok = True
            for inp, out in pairs_np:
                pred = objects_to_strip(inp, axis, order, background)
                if pred is None or pred.shape != out.shape or not np.array_equal(pred, out):
                    ok = False
                    break
            if ok:
                candidates.append((axis, order))
    return candidates


def learn_cycle_block_extend(train_pairs):
    res = []
    mapping = {}
    for inp, out in train_pairs:
        inp, out = as_grid(inp), as_grid(out)
        in_cols = set(np.unique(inp)) - {0}
        out_cols = set(np.unique(out)) - {0}
        if len(in_cols) == 1 and len(out_cols) == 1:
            mapping[next(iter(in_cols))] = next(iter(out_cols))
        else:
            return []
    if mapping:
        res.append((3, 3, tuple(sorted(mapping.items()))))
    return res
