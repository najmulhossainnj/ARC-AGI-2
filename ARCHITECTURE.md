# ARC Neuro-Symbolic Solver — Architecture Reference

Purpose of this doc: paste this + the project zip (and dataset files) into a
new conversation to resume work without re-explaining the codebase.

## What this is

A symbolic (DSL + beam search) ARC-AGI solver, packaged for Kaggle. Package
root: `arc_solver/`. Dataset format is the 2025 ARC Prize consolidated JSON
(`arc-agi_{split}_challenges.json` / `arc-agi_{split}_solutions.json`,
`sample_submission.json`). Working dataset used in dev: 1000 training tasks,
120 evaluation tasks (with solutions), 240 test tasks (no solutions) — i.e.
this is the **ARC-AGI-2** data, not ARC-AGI-1 (whose eval set is 400).

## Pipeline (top to bottom)

1. **`core/`** — `grid.py` (numpy grid helpers, `grid_hash`), `objects.py`
   (frozen `ARCObject`: cells, colors, bbox, shape signature), `scene.py`
   (`Scene.from_grid`: runs perception + relation-graph building over one
   grid), `types.py`, `task.py` (train/test pair containers).
2. **`perception/`** — `extractor.py` (connected-component extraction into
   `ARCObject`s, background-aware, optional diagonal connectivity),
   `components.py`, `shapes.py`.
3. **`relations/`** — `builder.py`/`graph.py`/`predicates.py`: builds a
   relation graph between objects in a scene (adjacency, alignment, etc.)
   for use by selectors/predicates.
4. **`correspondence/`** — `matcher.py` (matches objects between an input
   scene and output scene across a train pair), `changes.py` (infers
   per-object deltas — translation, recolor, deletion), `similarity.py`.
   This is what lets e.g. `learn_object_translation` figure out "objects of
   color X move by (dr, dc)" from paired input/output scenes.
5. **`dsl/`** — the instruction set:
   - `primitives.py` — enums for rotations/flips/gravity directions/scales.
   - `transforms.py`, `advanced_transforms.py` — whole-grid transform
     functions (rotate, flip, crop, gravity, scale, recolor, transpose,
     colormap, tile, fractal_tile, reflect_tile, mosaic, downscale, border,
     fill_holes, **symmetry_repair**, panel_logic, **pattern_complete**).
   - `object_ops.py` — object-level ops: select/recolor/crop/delete,
     translate/relocate/copy objects, rank-based recolor/resize,
     objects-to-strip.
   - `selectors.py` — object selection predicates (by color, size, rank...).
   - `ast.py` — `Instruction`, `Program` (tuple of instructions).
   - `executor.py` — runs a `Program` against a grid.
   - `type_checker.py`, `combinators.py`.
6. **`synthesis/`** — program search:
   - `grammar.py` — **this is the key file**. Defines two kinds of op
     family (see below) and `primitive_programs()`, which enumerates
     candidate `Program`s per family for a given set of train pairs.
   - `param_learning.py` — per-family parameter inference from train pairs
     (e.g. `learn_colormap`, `learn_tile_factors`, `symmetry_noise_candidates`,
     `learn_pattern_periods`, `learn_object_translation`...). This is where
     each family's "guess the parameters from examples" logic lives.
   - `beam_search.py` — `BeamSearcher`. Two candidate classes:
     - **composable** families (identity, rotate, flip, crop, gravity,
       scale, recolor, transpose): cheap, near parameter-free, chained
       across search depth.
     - **solver** families (colormap, tile, mosaic, downscale, border,
       fill_holes, symmetry_repair, panel_logic, pattern_complete,
       select_recolor, select_crop, object_relocate, object_translate,
       delete_objects, rank_recolor, rank_resize, objects_to_strip,
       fractal_tile, reflect_tile): higher-arity, data-driven, each one
       tries to explain the whole input→output gap in one instruction.
       Re-derived against the *residual* (train pairs passed through the
       current beam prefix) at each depth, so e.g. "rotate, then
       translate-the-largest-object" is reachable even though rotate and
       translate are learned/searched independently.
     - `program_cache.py`, `constraints.py`.
   - `verification/exact.py` — exact match against all train pairs (a
     program only counts as a real solution if it reproduces every train
     output exactly); `verification/mdl.py`, `verification/consistency.py`
     score/tie-break.
7. **`neural/`** — optional candidate ranker (`ranker.py`, `numpy_mlp.py`,
   pretrained weights in `neural/weights/ranker.npz`) that reorders beam
   candidates by learned score; `trainer.py`/`dataset.py`/`encoders.py` for
   (re)training it. Disabled by default (`neural.enabled: false` in
   `configs/default.yaml`).
8. **`solver/pipeline.py`** — `NeuroSymbolicARCSolver.solve_task(train_pairs,
   test_inputs)`: runs the beam searcher to get a ranked list of candidate
   `Program`s that solve (or best-fit) the train pairs, executes each on
   every test input, groups by output-grid hash, and returns the top 2
   distinct outputs per test input as `[attempt_1, attempt_2]` (ARC's
   required submission shape). Falls back to identity if nothing executes.

## Running it

```bash
# from arc-neurosymbolic-v1/, no install needed if PYTHONPATH includes .
python scripts/validate.py --challenges <eval_challenges.json> --solutions <eval_solutions.json>
python scripts/kaggle_solve.py --challenges <test_challenges.json> --output submission.json
```

`validate.py` reports exact-match accuracy (attempt_1 or attempt_2 correct)
over a challenge/solution pair. Default `beam_width=100, max_depth=3`
(note: `configs/default.yaml` says `max_depth: 5`, but `validate.py`'s own
argparse default is 3 — check which one you're actually invoking).

**Runtime**: ~15–60s/task at default settings on plain CPU/numpy, so a full
120-task eval pass is ~30–60 min. Budget for that (background process +
per-task timeout) rather than running inline.

## Known architectural limitation (as of this version)

The `SOLVER_FAMILIES` in `grammar.py` are **hand-built, single-shot,
whole-task solvers** — one Python function + one param-learning routine per
"pattern shape" (tiling, mosaics, panel logic, symmetry repair, pattern
completion, object translation, ...). This is enumerable but not
compositional in the way that matters most: each family (a) assumes its
transform explains the *entire* grid with one global parameter set, and
(b) has no way to invoke another family as a sub-step over a sub-region.

Concretely, diagnosed against 5 missed ARC-AGI-2 eval tasks
(`0934a4d8`, `135a2760`, `136b0064`, `13e47133`, `142ca369`):

- **`0934a4d8`** (recover an occluded patch of a symmetric grid, output =
  just the patch): `symmetry_repair` exists and is the right idea, but (a)
  it only tests symmetry about the exact grid center (h-1-r / w-1-c /
  transpose), and this task's true symmetry axis is offset — the outer 1–2
  border rows/cols sit outside the mirror-symmetric core, so the rigid
  center-based map never validates; and (b) even if repair worked, there is
  no family that then **crops to the bounding box of the repaired
  region** — the DSL always returns whole-grid or object-bbox crops, not
  "bbox of wherever the noise color was."
- **`135a2760`** (fix glitches in a periodic pattern): `pattern_complete`
  exists but assumes a single global `(ph, pw)` period across the *whole*
  grid. This task's grid is partitioned into several framed panels by
  separator rows/colors, and each panel has its own local period and
  background — a single global period can't fit all panels at once. Needs
  panel segmentation + independent per-panel period-fit, not a bigger
  global period search.
- **`136b0064`**: requires decoding small multi-color "icon" objects into a
  directional/path-tracing reconstruction. No primitive in the DSL family
  list does anything like this (nothing about paths, direction decoding
  from marker shapes, or icon-to-instruction mapping) — genuinely new
  primitive, not a tuning issue.
- **`13e47133`**: requires growing concentric/nested rectangles outward
  from small marker objects using a 2-color palette read off the marker.
  No "generative growth from a seed" primitive exists.
- **`142ca369`**: requires casting diagonal rays from small marker objects
  across the grid (with reflection at boundaries), repeating the marker's
  color pattern along the ray. No ray-casting primitive exists.

**Takeaway**: 2 of 5 (`0934a4d8`, `135a2760`) are cases where roughly the
right family already exists but is too rigid (global-only symmetry axis,
global-only period) — these would benefit from making existing families
*regional* (find symmetry/period per detected sub-region, not just
globally) and adding a generic "crop to where a `SELECT`-able condition
holds" op instead of only object-bbox crop. The other 3 need entirely new
primitive types (path/ray casting, generative growth from seed, icon
decoding) that no amount of parameter tuning on existing families will
reach.

This matches the README's own stated next step (object-level `SELECT` /
`FILTER` / `MOVE` / relational predicates / compositional programs) — the
fix is less "add family #19" and more "make the existing families
recursive/regional, and add a handful of genuinely general primitives
(ray-cast, seeded-growth, per-region-apply) that compose with everything
already here," rather than continuing to special-case one family per
surface pattern.

## Suggested next steps

1. Add a generic **`APPLY_PER_REGION`** combinator: segment a grid into
   regions (by separator color/connected background gaps), then apply any
   existing solver family independently per region with its own learned
   params, then reassemble. This alone would likely fix `135a2760`-style
   tasks without a new family.
2. Generalize `symmetry_repair`'s `_symmetry_maps` to search over
   candidate mirror axes/offsets (not just the exact grid center), and add
   a `CROP_TO_MASK` / `CROP_TO_CHANGED_REGION` op so "repair then extract
   just the repaired patch" becomes a 2-step composable program instead of
   needing a bespoke family.
3. New primitive families worth adding, in rough order of how often this
   shape recurs in ARC-AGI-2: ray/line casting from seed objects
   (with reflection), seeded concentric/nested-shape growth, and
   icon/legend decoding (small marker object → instruction, applied
   elsewhere in the grid).
4. Re-run `scripts/validate.py` on the full 120-eval set (budget ~30–60
   min, run in background with a per-task timeout) after each addition to
   track solve-rate delta, and keep a running list of missed task IDs with
   the same "what transform is needed / what's missing" format as above —
   that's the fastest way to see which primitive to build next.
