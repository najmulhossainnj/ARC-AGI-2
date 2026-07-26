"""
Pattern Analyzer - Rule-based transformation detection.

This analyzer examines input-output grid pairs and detects common
transformation patterns using pure rule-based logic (no LLM required).

Supported patterns:
- Color mapping/substitution
- Rotation (90, 180, 270 degrees)
- Reflection (horizontal, vertical, diagonal)
- Cropping and padding
- Replication/tiling
- Symmetry detection
- Border operations
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from collections import Counter


def analyze_transformation(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]]
) -> Optional[Callable]:
    """
    Analyze input-output pairs to detect the transformation pattern.
    Returns a solve function if a pattern is detected, None otherwise.
    """
    if not train_pairs or len(train_pairs) < 1:
        return None
    
    # Try each pattern detector - more specific patterns first
    detectors = [
        detect_rotation,         # Most specific
        detect_reflection,       # Specific transformations
        detect_replication,      # Tiling
        detect_scale,           # Scaling
        detect_pixel_replacement, # Per-pixel changes
        detect_color_mapping,    # Color substitution
        detect_color_swap,       # Color swapping
        detect_symmetry,         # Symmetry ops
        detect_crop_and_pad,    # Cropping
        detect_border_operation, # Border ops
        detect_region_operation, # Region ops
    ]
    
    for detector in detectors:
        result = detector(train_pairs)
        if result is not None:
            return result
    
    return None


def detect_color_mapping(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect color substitution/mapping patterns."""
    if len(pairs) < 1:
        return None
    
    # Collect all color mappings
    mappings = []
    for inp, out in pairs:
        mapping = _compute_color_mapping(inp, out)
        if mapping:
            mappings.append(mapping)
    
    if not mappings:
        return None
    
    # Check if all mappings are consistent
    if len(mappings) == 1:
        mapping = mappings[0]
    else:
        # Find common mappings across all pairs
        common = mappings[0]
        for m in mappings[1:]:
            common = {k: v for k, v in common.items() if m.get(k) == v}
        if not common:
            return None
        mapping = common
    
    if not mapping:
        return None
    
    def solve(grid):
        g = grid.copy()
        for old, new in mapping.items():
            g[g == old] = new
        return g
    
    solve._pattern = f"color_mapping:{mapping}"
    return solve


def _compute_color_mapping(inp: np.ndarray, out: np.ndarray) -> Optional[Dict]:
    """Compute color mapping from input to output."""
    if inp.shape != out.shape:
        return None
    
    mapping = {}
    h, w = inp.shape
    for i in range(h):
        for j in range(w):
            in_color = int(inp[i, j])
            out_color = int(out[i, j])
            if in_color != out_color:
                if in_color in mapping and mapping[in_color] != out_color:
                    return None  # Inconsistent mapping
                mapping[in_color] = out_color
    
    return mapping


def detect_rotation(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect rotation patterns (90, 180, 270 degrees)."""
    for inp, out in pairs:
        if inp.shape != out.shape:
            continue
        
        # Try each rotation
        for angle in [90, 180, 270]:
            rotated = _rotate_grid(inp, angle)
            if np.array_equal(rotated, out):
                # Verify with other pairs
                valid = True
                for inp2, out2 in pairs[1:]:
                    if inp2.shape != out2.shape:
                        valid = False
                        break
                    if not np.array_equal(_rotate_grid(inp2, angle), out2):
                        valid = False
                        break
                
                if valid:
                    def make_solver(angle):
                        def solve(grid):
                            return _rotate_grid(grid, angle)
                        return solve
                    solve = make_solver(angle)
                    solve._pattern = f"rotation:{angle}"
                    return solve
    
    return None


def _rotate_grid(grid: np.ndarray, angle: int) -> np.ndarray:
    """Rotate grid by angle degrees."""
    if angle == 90:
        return np.rot90(grid)
    elif angle == 180:
        return np.rot90(grid, 2)
    elif angle == 270:
        return np.rot90(grid, 3)
    return grid.copy()


def detect_reflection(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect reflection patterns (horizontal, vertical, diagonal)."""
    reflections = [
        ("horizontal", lambda g: np.fliplr(g)),
        ("vertical", lambda g: np.flipud(g)),
        ("transpose", lambda g: g.T.copy()),
    ]
    
    for name, reflect_fn in reflections:
        valid = True
        for inp, out in pairs:
            if inp.shape != out.shape:
                valid = False
                break
            if not np.array_equal(reflect_fn(inp), out):
                valid = False
                break
        
        if valid:
            def make_solver(fn):
                def solve(grid):
                    return fn(grid.copy())
                return solve
            solve = make_solver(reflect_fn)
            solve._pattern = f"reflection:{name}"
            return solve
    
    return None


def detect_replication(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect replication/tiling patterns."""
    for inp, out in pairs:
        if inp.ndim != 2 or out.ndim != 2:
            continue
        
        h_in, w_in = inp.shape
        h_out, w_out = out.shape
        
        # Check if output is tiled
        if h_out % h_in == 0 and w_out % w_in == 0:
            times_h = h_out // h_in
            times_w = w_out // w_in
            
            # Verify tiling
            expected = np.tile(inp, (times_h, times_w))
            if np.array_equal(expected, out):
                def solve(grid):
                    return np.tile(grid, (times_h, times_w))
                solve._pattern = f"tile:{times_h}x{times_w}"
                return solve
    
    return None


def detect_symmetry(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect symmetry-related transformations."""
    for inp, out in pairs:
        if inp.shape != out.shape:
            continue
        
        h, w = inp.shape
        
        # Check for horizontal symmetry (left-right mirror)
        if np.array_equal(np.fliplr(inp), out):
            # Count how many columns changed
            changes = 0
            for j in range(w // 2):
                for i in range(h):
                    if inp[i, j] != out[i, j]:
                        changes += 1
            
            if changes > 0:
                def solve(grid):
                    return np.fliplr(grid)
                solve._pattern = "horizontal_mirror"
                return solve
        
        # Check for vertical symmetry
        if np.array_equal(np.flipud(inp), out):
            def solve(grid):
                return np.flipud(grid)
            solve._pattern = "vertical_mirror"
            return solve
    
    return None


def detect_border_operation(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect operations on borders/edges."""
    for inp, out in pairs:
        if inp.shape != out.shape:
            continue
        
        h, w = inp.shape
        
        # Check for border filling (fill non-border with background)
        bg = _detect_background(inp)
        if bg is not None:
            # Check if interior is filled with background
            filled = inp.copy()
            for i in range(1, h-1):
                for j in range(1, w-1):
                    filled[i, j] = bg
            
            if np.array_equal(filled, out):
                def solve(grid):
                    bg_color = _detect_background(grid)
                    if bg_color is None:
                        return grid.copy()
                    result = grid.copy()
                    for i in range(1, result.shape[0]-1):
                        for j in range(1, result.shape[1]-1):
                            result[i, j] = bg_color
                    return result
                solve._pattern = "fill_interior"
                return solve
        
        # Check for border extraction
        border_inp = inp.copy()
        border_out = out.copy()
        # Remove border
        for i in range(h):
            for j in range(w):
                if i == 0 or i == h-1 or j == 0 or j == w-1:
                    border_inp[i, j] = bg if bg else 0
                    border_out[i, j] = bg if bg else 0
        
        if np.array_equal(border_inp, border_out):
            # Border is same in input and output
            pass  # Could implement border-only extraction
    
    return None


def detect_pixel_replacement(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect pixel-level replacement patterns."""
    # Look for patterns like "replace center pixel of blocks"
    for inp, out in pairs:
        if inp.shape != out.shape:
            continue
        
        h, w = inp.shape
        
        # Look for center-of-block markers
        centers = []
        for i in range(1, h-1):
            for j in range(1, w-1):
                if out[i, j] != inp[i, j]:
                    # This pixel changed
                    color = int(out[i, j])
                    # Check if it's the center of a uniform block
                    if (_is_center_of_block(inp, i, j)):
                        centers.append((i, j, color))
        
        if centers:
            # Try to detect the pattern
            colors_changed = set(c for _, _, c in centers)
            if len(colors_changed) == 1:
                new_color = centers[0][2]
                bg = _detect_background(inp)
                
                def solve(grid):
                    result = grid.copy()
                    h, w = result.shape
                    for i in range(1, h-1):
                        for j in range(1, w-1):
                            if result[i, j] != bg:
                                # Check if center of uniform block
                                color = result[i, j]
                                if (result[i-1, j] == color and result[i+1, j] == color and
                                    result[i, j-1] == color and result[i, j+1] == color):
                                    result[i, j] = new_color
                    return result
                solve._pattern = f"mark_block_centers:{new_color}"
                return solve
    
    return None


def _is_center_of_block(grid: np.ndarray, i: int, j: int) -> bool:
    """Check if (i,j) is the center of a uniform-colored block."""
    color = grid[i, j]
    h, w = grid.shape
    
    # Check all 8 neighbors are same color
    neighbors = [
        (i-1, j-1), (i-1, j), (i-1, j+1),
        (i, j-1),           (i, j+1),
        (i+1, j-1), (i+1, j), (i+1, j+1)
    ]
    
    for ni, nj in neighbors:
        if 0 <= ni < h and 0 <= nj < w:
            if grid[ni, nj] != color:
                return False
    return True


def detect_region_operation(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect operations on regions/objects."""
    from scipy.ndimage import label
    
    for inp, out in pairs:
        if inp.shape != out.shape:
            continue
        
        # Count regions in input and output
        bg = _detect_background(inp)
        if bg is None:
            continue
        
        inp_mask = (inp != bg).astype(int)
        out_mask = (out != bg).astype(int)
        
        try:
            inp_labels, inp_n = label(inp_mask)
            out_labels, out_n = label(out_mask)
        except:
            continue
        
        # Check if region count changed
        if inp_n != out_n:
            # Some regions merged or split
            pass
        
        # Check if colors changed
        if not np.array_equal(inp, out):
            # Look for color changes
            color_changes = {}
            h, w = inp.shape
            for i in range(h):
                for j in range(w):
                    if inp[i, j] != out[i, j]:
                        key = (int(inp[i, j]), int(out[i, j]))
                        color_changes[key] = color_changes.get(key, 0) + 1
            
            if color_changes:
                # Find most common change
                most_common = max(color_changes.items(), key=lambda x: x[1])
                old_color, new_color = most_common[0]
                
                def solve(grid):
                    result = grid.copy()
                    result[result == old_color] = new_color
                    return result
                solve._pattern = f"recolor:{old_color}->{new_color}"
                return solve
    
    return None


def detect_crop_and_pad(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect crop and pad transformations."""
    for inp, out in pairs:
        h_in, w_in = inp.shape
        h_out, w_out = out.shape
        
        if h_out > h_in and w_out > w_in:
            # Output is larger - could be padding
            pass  # Could implement padding detection
        
        if h_out < h_in and w_out < w_in:
            # Output is smaller - could be cropping
            bg = _detect_background(out)
            if bg is not None:
                # Check if cropped to bounding box
                mask = (out != bg)
                if mask.any():
                    rows = np.where(mask.any(axis=1))[0]
                    cols = np.where(mask.any(axis=0))[0]
                    r1, r2 = rows.min(), rows.max()
                    c1, c2 = cols.min(), cols.max()
                    
                    cropped = out[r1:r2+1, c1:c2+1]
                    
                    if np.array_equal(cropped, inp):
                        def solve(grid):
                            bg_color = _detect_background(grid)
                            mask = (grid != bg_color)
                            rows = np.where(mask.any(axis=1))[0]
                            cols = np.where(mask.any(axis=0))[0]
                            r1, r2 = rows.min(), rows.max()
                            c1, c2 = cols.min(), cols.max()
                            return grid[r1:r2+1, c1:c2+1]
                        solve._pattern = "crop_to_content"
                        return solve
    
    return None


def detect_scale(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect scaling patterns."""
    for inp, out in pairs:
        h_in, w_in = inp.shape
        h_out, w_out = out.shape
        
        if h_in == 0 or w_in == 0:
            continue
        
        # Check for integer scaling (2x, 3x, etc.)
        if h_out % h_in == 0 and w_out % w_in == 0:
            scale_h = h_out // h_in
            scale_w = w_out // w_in
            
            if scale_h == scale_w and scale_h > 1:
                # Check if it's a simple upscale
                upscaled = np.kron(inp, np.ones((scale_h, scale_w), dtype=inp.dtype))
                if np.array_equal(upscaled, out):
                    def solve(grid, scale=scale_h):
                        return np.kron(grid, np.ones((scale, scale), dtype=grid.dtype))
                    solve._pattern = f"scale_up:{scale_h}"
                    return solve
    
    return None


def detect_color_swap(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Callable]:
    """Detect swap of two colors."""
    for inp, out in pairs:
        if inp.shape != out.shape:
            continue
        
        # Find all pixels where colors differ
        diff_mask = inp != out
        if not diff_mask.any():
            continue
        
        # Get unique colors in input and output at diff positions
        in_colors = set(int(c) for c in np.unique(inp[diff_mask]))
        out_colors = set(int(c) for c in np.unique(out[diff_mask]))
        
        if len(in_colors) == 2 and len(out_colors) == 2:
            # Two colors being swapped
            in_list = list(in_colors)
            out_list = list(out_colors)
            
            # Check if it's a swap
            if (in_list[0] in out_colors and in_list[1] in out_colors and
                out_list[0] in in_colors and out_list[1] in in_colors):
                
                color_a, color_b = in_list
                
                def solve(grid):
                    result = grid.copy()
                    temp = result.copy()
                    result[grid == color_a] = color_b
                    result[temp == color_b] = color_a
                    return result
                solve._pattern = f"swap_colors:{color_a}<->{color_b}"
                return solve
    
    return None


def _detect_background(grid: np.ndarray) -> Optional[int]:
    """Detect background color (most common color)."""
    if grid.size == 0:
        return None
    counts = Counter(grid.flatten())
    if not counts:
        return None
    return counts.most_common(1)[0][0]


# Alias for compatibility with diagnostic engine
PatternAnalyzer = type('PatternAnalyzer', (), {
    'analyze': staticmethod(lambda *args, **kwargs: analyze_transformation(*args, **kwargs))
})
