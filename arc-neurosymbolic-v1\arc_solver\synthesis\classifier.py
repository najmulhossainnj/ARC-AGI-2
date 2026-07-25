"""
Task Classifier & Search Space Pruner.

Classifies ARC tasks into structural transformation categories based on
input vs output dimension relations across training pairs. Prunes inapplicable
solver families from beam search to accelerate search by 5x-10x and prevent
overfitting/false positives.
"""
from __future__ import annotations
from typing import List, Tuple, Set
import numpy as np

SAME_SIZE_PRUNED = {
    "crop", "tile", "scale", "downscale", "symmetry_repair_crop",
    "select_crop", "fractal_tile", "reflect_tile"
}

DOWNSCALE_PRUNED = {
    "tile", "scale", "fractal_tile", "reflect_tile", "mosaic"
}

UPSCALE_PRUNED = {
    "crop", "downscale", "symmetry_repair_crop", "select_crop"
}

FEATURE_BOUNDED_PRUNED = {
    "tile", "scale", "fractal_tile", "reflect_tile", "mosaic", "panel_logic",
    "apply_per_region", "pattern_complete", "border", "fill_holes", "recolor",
    "gravity", "rotate", "flip"
}

def classify_task(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    """Determine the structural dimension transformation category of a task."""
    if not train_pairs:
        return "DYNAMIC"

    same_size = True
    downscale = True
    upscale = True

    for inp, out in train_pairs:
        inp_h, inp_w = inp.shape
        out_h, out_w = out.shape

        if (inp_h != out_h) or (inp_w != out_w):
            same_size = False

        if (out_h > inp_h) or (out_w > inp_w):
            downscale = False

        if (out_h < inp_h) or (out_w < inp_w):
            upscale = False

    if same_size:
        return "SAME_SIZE"

    # Stage 2: Check for FEATURE_BOUNDED_CROP (varying output dimensions matching a color cluster bbox)
    if downscale:
        out_shapes = [p[1].shape for p in train_pairs]
        if len(set(out_shapes)) > 1:
            inp0, out0 = train_pairs[0]
            colors0 = set(np.unique(inp0)) - {0}
            for c in colors0:
                match = True
                for inp, out in train_pairs:
                    r, cols = np.where(inp == c)
                    if len(r) == 0:
                        match = False; break
                    h_box = r.max() - r.min() + 1
                    w_box = cols.max() - cols.min() + 1
                    if (h_box, w_box) != out.shape:
                        match = False; break
                if match:
                    return "FEATURE_BOUNDED_CROP"

        return "DOWNSCALE_CROP"

    if upscale:
        return "UPSCALE_GROW"
    
    return "DYNAMIC"

def filter_families(category: str, families: Set[str] | List[str]) -> List[str]:
    """Filter out families that are mathematically incompatible with the task category."""
    fam_set = set(families)
    
    if category == "SAME_SIZE":
        fam_set -= SAME_SIZE_PRUNED
    elif category == "FEATURE_BOUNDED_CROP":
        fam_set -= FEATURE_BOUNDED_PRUNED
    elif category == "DOWNSCALE_CROP":
        fam_set -= DOWNSCALE_PRUNED
    elif category == "UPSCALE_GROW":
        fam_set -= UPSCALE_PRUNED

    return sorted(list(fam_set))
