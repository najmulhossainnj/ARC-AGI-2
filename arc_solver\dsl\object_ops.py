"""
Object-level program ops: move or copy an object relative to another object.

These are the compositional counterpart to the whole-grid transforms in
`transforms.py`. Instead of remapping every pixel, they operate on the
object graph: pick a "mover" (or movers) with a selector, pick an "anchor"
with another selector, and place the mover so it satisfies a spatial
relation to the anchor (touching one of its sides, or centered on it).

The relation vocabulary intentionally mirrors the predicates already
computed in `relations/predicates.py` (touching, aligned_*, etc.) so the
same relational vocabulary is used for both perception (the relation
graph) and action (this module) rather than inventing a second one.
"""
from __future__ import annotations
import numpy as np

from ..core.grid import as_grid
from ..core.scene import Scene
from .selectors import select_objects

# Relation catalog for RELOCATE_OBJECT / COPY_OBJECT. Each "touch_*" relation
# slides the mover straight toward the anchor along one axis, keeping the
# mover's coordinate on the other axis unchanged (a "gravity toward anchor"
# pattern). "center_on" recenters the mover on the anchor on both axes.
RELOCATE_RELATIONS = (
    "touch_left", "touch_right", "touch_top", "touch_bottom", "center_on",
)

_scene_cache: dict = {}


def _cached_scene(g, background):
    key = (g.shape, g.tobytes(), background)
    scene = _scene_cache.get(key)
    if scene is None:
        scene = Scene.from_grid(g, background=background)
        if len(_scene_cache) > 2000:
            _scene_cache.clear()
        _scene_cache[key] = scene
    return scene


def _anchor_bbox(scene, anchor_selector, anchor_value, exclude_ids=()):
    anchors = [
        o for o in select_objects(scene, anchor_selector, anchor_value)
        if o.id not in exclude_ids
    ]
    if not anchors:
        return None
    r1 = min(o.top for o in anchors)
    c1 = min(o.left for o in anchors)
    r2 = max(o.bottom for o in anchors)
    c2 = max(o.right for o in anchors)
    return r1, c1, r2, c2


def _target_top_left(obj, anchor_bbox, relation):
    at, al, ab, ar = anchor_bbox
    h, w = obj.height, obj.width
    if relation == "touch_right":
        return obj.top, ar + 1
    if relation == "touch_left":
        return obj.top, al - w
    if relation == "touch_bottom":
        return ab + 1, obj.left
    if relation == "touch_top":
        return at - h, obj.left
    if relation == "center_on":
        acr, acc = (at + ab) / 2.0, (al + ar) / 2.0
        return (
            int(round(acr - (h - 1) / 2.0)),
            int(round(acc - (w - 1) / 2.0)),
        )
    raise ValueError(f"unknown relation {relation}")


def _place(grid, movers, anchor_bbox, relation, erase, background):
    g = grid
    h, w = g.shape
    exclude = {o.id for o in movers}
    if anchor_bbox is None:
        return None
    for obj in movers:
        nt, nl = _target_top_left(obj, anchor_bbox, relation)
        dr, dc = nt - obj.top, nl - obj.left
        cell_colors = [(r, c, g[r, c]) for r, c in obj.cells]
        new_cells = []
        for r, c, color in cell_colors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                return None  # would move off-grid: this relation doesn't apply
            new_cells.append((nr, nc, color))
        if erase:
            for r, c, _ in cell_colors:
                g[r, c] = background
        for nr, nc, color in new_cells:
            g[nr, nc] = color
    return g


def translate_object(grid, selector, value, dr, dc, erase=True, background=0):
    """Move the selected object(s) by a fixed (dr, dc), the same offset for
    every matching object. This is the "shifts by a constant vector every
    time" pattern, distinct from RELOCATE_OBJECT's anchor-relative sliding."""
    g = as_grid(grid).copy()
    scene = _cached_scene(as_grid(grid), background)
    movers = select_objects(scene, selector, value)
    if not movers:
        return None
    h, w = g.shape
    placements = []
    for obj in movers:
        for r, c in obj.cells:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                return None
            placements.append((nr, nc, g[r, c]))
    if erase:
        for obj in movers:
            for r, c in obj.cells:
                g[r, c] = background
    for nr, nc, color in placements:
        g[nr, nc] = color
    return g


def relocate_object(grid, selector, value, anchor_selector, anchor_value,
                     relation, background=0):
    """Move the selected object(s) so each touches/centers-on the anchor."""
    g = as_grid(grid).copy()
    scene = _cached_scene(as_grid(grid), background)
    movers = select_objects(scene, selector, value)
    if not movers:
        return None
    anchor_bbox = _anchor_bbox(
        scene, anchor_selector, anchor_value,
        exclude_ids={o.id for o in movers},
    )
    return _place(g, movers, anchor_bbox, relation, erase=True, background=background)


def copy_object(grid, selector, value, anchor_selector, anchor_value,
                 relation, background=0):
    """Stamp a copy of the selected object(s) next to the anchor, leaving
    the original object(s) in place."""
    g = as_grid(grid).copy()
    scene = _cached_scene(as_grid(grid), background)
    movers = select_objects(scene, selector, value)
    if not movers:
        return None
    anchor_bbox = _anchor_bbox(scene, anchor_selector, anchor_value)
    return _place(g, movers, anchor_bbox, relation, erase=False, background=background)


# ----------------------------------------------------------------------
# Rank-based ops.
#
# `translate_object`/`relocate_object`/`copy_object` above cover the "move"
# change kind from `correspondence.changes.classify_change`. These three
# cover "recolor" and "resize", using the same idea of matching input and
# output objects and inferring one consistent rule -- here, keyed off each
# object's size rank rather than its color, so "the two largest objects
# swap to red/blue" or "each object grows in proportion to its rank" are
# reachable without a dedicated primitive per task.
# ----------------------------------------------------------------------

def _size_rank(objects):
    """0-indexed rank by size (largest first), ties broken by reading order
    (top, then left) for determinism. A mapping keyed by rank means the
    same thing on a test grid with a different object count than any
    training pair had."""
    ordered = sorted(objects, key=lambda o: (-o.size, o.top, o.left))
    return {o.id: i for i, o in enumerate(ordered)}


def delete_objects(grid, selector, value, background=0):
    """Erase every object matching `selector` (fill with background),
    leaving everything else untouched. The generalization of the
    'unmatched input object' case classify_change/match_objects surface:
    an object with no corresponding output object was deleted."""
    g = as_grid(grid).copy()
    scene = _cached_scene(as_grid(grid), background)
    victims = select_objects(scene, selector, value)
    if not victims:
        return None
    for obj in victims:
        for r, c in obj.cells:
            g[r, c] = background
    return g


def recolor_by_rank(grid, rank_key, mapping, background=0):
    """Recolor each object according to its rank (see `_size_rank`);
    objects whose rank isn't a key in `mapping` are left unchanged.
    `mapping` is a tuple of (rank, new_color) pairs, the same
    hashable-tuple convention `COLORMAP` uses for its dict-shaped
    argument. `rank_key` is currently always 'size_desc' -- kept as an
    explicit argument so other rank orderings can be added later without
    changing the op's arity."""
    g = as_grid(grid).copy()
    scene = _cached_scene(as_grid(grid), background)
    if not scene.objects:
        return None
    ranks = _size_rank(scene.objects)
    mapping = dict(mapping)
    for obj in scene.objects:
        new_color = mapping.get(ranks[obj.id])
        if new_color is None:
            continue
        for r, c in obj.cells:
            g[r, c] = new_color
    return g


def resize_objects_by_rank(grid, rank_key, mapping, background=0):
    """Block-scale each object by an integer factor according to its rank
    (see `_size_rank`), anchored at the object's top-left corner. Objects
    whose rank isn't a key in `mapping` are left unchanged. `mapping` is a
    tuple of (rank, factor) pairs. Growing an object past the grid edge,
    or two grown objects disagreeing on an overlapping cell, fails the op
    (returns None) rather than silently clipping or overwriting."""
    src = as_grid(grid)
    scene = _cached_scene(src, background)
    if not scene.objects:
        return None
    ranks = _size_rank(scene.objects)
    mapping = dict(mapping)
    h, w = src.shape
    out = src.copy()
    for obj in scene.objects:
        factor = mapping.get(ranks[obj.id])
        if factor is None or factor == 1:
            continue
        sub = src[obj.top:obj.bottom + 1, obj.left:obj.right + 1]
        scaled = np.repeat(np.repeat(sub, factor, axis=0), factor, axis=1)
        for r, c in obj.cells:
            out[r, c] = background
        nh, nw = scaled.shape
        if obj.top + nh > h or obj.left + nw > w:
            return None
        target = out[obj.top:obj.top + nh, obj.left:obj.left + nw]
        collision = (target != background) & (scaled != background) & (target != scaled)
        if collision.any():
            return None
        mask = scaled != background
        target[mask] = scaled[mask]
    return out


def objects_to_strip(grid, axis, order, background=0):
    """Summarize the input's objects as a strip whose *length* is derived
    from the object count -- one cell per object, holding that object's
    own color -- rather than preserving or uniformly scaling the input's
    shape the way every other op in this module does. `axis` is 'ROW'
    (output shape 1 x N) or 'COL' (N x 1). `order` sequences the objects
    along the strip: 'size_desc' (largest first, ties by reading order) or
    'reading' (top-left to bottom-right, row-major). Multi-colored objects
    have no single color to place, so they fail the op (returns None).

    This is the "output size = a property of the objects, not of the grid"
    family: a well-known ARC pattern (count the objects / read off their
    colors in rank order) that every shape-preserving or shape-scaling op
    elsewhere in the DSL is structurally unable to produce.
    """
    scene = _cached_scene(as_grid(grid), background)
    objs = scene.objects
    if not objs:
        return None
    if order == "size_desc":
        ranked = sorted(objs, key=lambda o: (-o.size, o.top, o.left))
    elif order == "reading":
        ranked = sorted(objs, key=lambda o: (o.top, o.left))
    else:
        return None
    colors = []
    for o in ranked:
        if len(o.colors) != 1:
            return None
        colors.append(o.colors[0])
    if axis == "ROW":
        return np.array([colors], dtype=np.int16)
    if axis == "COL":
        return np.array([[c] for c in colors], dtype=np.int16)
    return None
