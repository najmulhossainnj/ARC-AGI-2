from __future__ import annotations
import textwrap
import numpy as np
from typing import List, Tuple


def format_grid(g: np.ndarray) -> str:
    """Format a 2D grid as a compact string for an LLM prompt."""
    rows = []
    for row in g:
        rows.append("[" + ", ".join(str(int(v)) for v in row) + "]")
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def build_prompt(
    task_id: str,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    features: dict,
    tried_ops: List[str],
    attempt_num: int = 1,
) -> str:
    """Build a structured LLM prompt for primitive code generation."""
    
    # Different strategies based on attempt number
    if attempt_num == 1:
        strategy_hint = "Try to identify the main pattern transformation."
    elif attempt_num == 2:
        strategy_hint = "Look for symmetry, rotation, or reflection patterns."
    elif attempt_num == 3:
        strategy_hint = "Consider counting objects or encoding counts as colors."
    else:
        strategy_hint = "Think step-by-step: identify objects, their relationships, then the transformation."
    
    lines = [
        "You are an expert ARC (Abstraction and Reasoning Corpus) puzzle solver.",
        "Given the following input→output grid pairs, write a Python function that",
        "transforms ANY input grid to its corresponding output grid.",
        "",
        f"Task ID: {task_id}",
        "",
        f"=== STRATEGY ({attempt_num}) ===",
        strategy_hint,
        "",
        "=== COMMON ARC TRANSFORMATIONS (for reference) ===",
        "- Translation: move pattern to new position",
        "- Rotation/Flip: rotate or mirror the grid",
        "- Color substitution: replace one color with another",
        "- Pattern replication: copy pattern to fill space",
        "- Bounding box: crop or expand around objects",
        "- Flood fill: fill enclosed regions",
        "- Pattern counting: count and encode as colors",
        "",
        "=== TRAIN PAIRS ===",
    ]

    for i, (inp, out) in enumerate(train_pairs):
        inp, out = np.asarray(inp), np.asarray(out)
        lines += [
            f"[PAIR {i}]",
            f"Input shape: {list(inp.shape)}, colors: {sorted(set(inp.flatten().tolist()))}",
            f"Input:\n{format_grid(inp)}",
            f"Output shape: {list(out.shape)}, colors: {sorted(set(out.flatten().tolist()))}",
            f"Output:\n{format_grid(out)}",
            f"Diff cells: {int((inp != out).sum())} / {inp.size}",
            "",
        ]

    lines += [
        "=== OBSERVED FEATURES ===",
        f"Same-size transformation: {features.get('same_size', 'unknown')}",
        f"Input color count: {features.get('n_colors_in', '?')}",
        f"Output color count: {features.get('n_colors_out', '?')}",
        f"Average non-background cells in input: {features.get('avg_nonbg_in', '?')}",
        f"Average diff fraction: {features.get('avg_diff_frac', '?'):.2%}" if features.get('avg_diff_frac') else "",
        "",
        "=== PRIMITIVES ALREADY TRIED (THAT FAILED) ===",
        ", ".join(tried_ops) if tried_ops else "(none)",
        "",
        "=== TASK ===",
        "Write a Python function with this EXACT signature:",
        "",
        "```python",
        "import numpy as np",
        "",
        "def solve(grid: np.ndarray, background: int = 0) -> np.ndarray:",
        '    """Transform input grid to output grid.',
        "    Args:",
        "        grid: 2D numpy array (int16) of the input",
        "        background: background color value (default 0)",
        "    Returns:",
        "        Transformed 2D numpy array",
        '    """',
        "    # YOUR IMPLEMENTATION HERE",
        "    ...",
        "```",
        "",
        "RULES:",
        "- Return a numpy array of the same dtype as input (int16 preferred)",
        "- PREFER numpy only - avoid scipy if possible",
        "- If scipy is needed, import it inside the function: from scipy.ndimage import label, find_objects",
        "- Keep the solution concise and general (must work for any valid input, not just training examples)",
        "- Do NOT hardcode specific values from the training examples",
        "- CRITICAL: Test your logic mentally against the training examples before returning",
        "- The background color may vary - detect it from the most common color in the grid",
        "- Output ONLY the Python code block, no explanation",
    ]

    return "\n".join(lines)
