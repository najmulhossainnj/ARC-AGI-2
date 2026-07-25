from __future__ import annotations
import textwrap
import numpy as np
from typing import List, Tuple, Optional


def format_grid(g: np.ndarray) -> str:
    """Format a 2D grid as a compact string for an LLM prompt."""
    rows = []
    for row in g:
        rows.append("[" + ", ".join(str(int(v)) for v in row) + "]")
    return "[\n  " + ",\n  ".join(rows) + "\n]"


def _detect_transformation_hints(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> List[str]:
    """Generate automatic diagnostic hints from the train pairs."""
    hints = []
    for i, (inp, out) in enumerate(train_pairs):
        inp, out = np.asarray(inp), np.asarray(out)
        # Size / shape
        if inp.shape == out.shape:
            diff = int((inp != out).sum())
            total = inp.size
            hints.append(f"Pair {i}: Same shape {list(inp.shape)}. {diff}/{total} cells changed ({100*diff/max(total,1):.0f}%).")
        else:
            h_ratio = out.shape[0] / max(inp.shape[0], 1)
            w_ratio = out.shape[1] / max(inp.shape[1], 1)
            hints.append(f"Pair {i}: Shape changed {list(inp.shape)} -> {list(out.shape)} (h_ratio={h_ratio:.2f}, w_ratio={w_ratio:.2f}).")

        # Color changes
        in_colors = set(int(v) for v in np.unique(inp)) - {0}
        out_colors = set(int(v) for v in np.unique(out)) - {0}
        added = out_colors - in_colors
        removed = in_colors - out_colors
        if added:
            hints.append(f"Pair {i}: New colors appeared in output: {sorted(added)}")
        if removed:
            hints.append(f"Pair {i}: Colors disappeared in output: {sorted(removed)}")

        # Non-background cell counts
        nb_in = int((inp != 0).sum())
        nb_out = int((out != 0).sum())
        if nb_in != nb_out:
            hints.append(f"Pair {i}: Non-bg cells: input={nb_in}, output={nb_out} (delta={nb_out - nb_in:+d}).")

    return hints


def build_prompt(
    task_id: str,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    features: dict,
    tried_ops: List[str],
) -> str:
    """Build a structured LLM prompt with step-by-step diagnostic analysis instructions."""
    hints = _detect_transformation_hints(train_pairs)

    lines = [
        "You are an expert ARC (Abstraction and Reasoning Corpus) puzzle solver and Python algorithm engineer.",
        "Your task is to figure out the EXACT transformation rule from the input/output pairs and implement it as a general Python function.",
        "",
        f"Task ID: {task_id}",
        "",
        "=== STEP 1: AUTOMATED DIAGNOSTIC ANALYSIS ===",
        "(These are observations automatically extracted from the training pairs. Use them as hints.)",
        *hints,
        "",
        "=== STEP 2: VISUAL ANALYSIS CHECKLIST ===",
        "Carefully examine each input/output pair and answer these questions mentally:",
        "  Q1: Do all pairs have the same input/output shape? If not, what determines the output size?",
        "  Q2: Which color is the background (most common, usually 0)?",
        "  Q3: Are there 'indicator' colors (small single-cell markers) that control where/how the main object is placed?",
        "  Q4: Does any object move (translation/gravity), grow (tiling/replication), shrink, rotate, or reflect?",
        "  Q5: Is there a color remapping? Do cells change value based on a rule?",
        "  Q6: What part of the grid is preserved vs changed?",
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
        "=== OBSERVED STATISTICAL FEATURES ===",
        f"Same-size transformation: {features.get('same_size', 'unknown')}",
        f"Input color count: {features.get('n_colors_in', '?')}",
        f"Output color count: {features.get('n_colors_out', '?')}",
        f"Average non-background cells in input: {features.get('avg_nonbg_in', '?')}",
        f"Average diff fraction: {features.get('avg_diff_frac', '?'):.2%}" if features.get('avg_diff_frac') else "",
        "",
        "=== PREVIOUSLY TRIED APPROACHES (THAT ALL FAILED) ===",
        ", ".join(tried_ops) if tried_ops else "(none)",
        "",
        "=== STEP 3: CODE GENERATION ===",
        "Based on your analysis above, write a correct, general Python `solve()` function.",
        "Output ONLY the Python code block (```python ... ```) with no prose before or after:",
        "",
        "```python",
        "import numpy as np",
        "from scipy.ndimage import label",
        "",
        "def solve(grid: np.ndarray, background: int = 0) -> np.ndarray:",
        '    """',
        "    Transform input grid to output grid.",
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
        "CRITICAL RULES:",
        "- The function MUST be named exactly `solve`",
        "- Return a numpy array of integer type (np.int16 or np.int64)",
        "- Only use numpy and scipy.ndimage — no other external libraries",
        "- Do NOT hardcode test outputs or specific index values derived only from training data",
        "- The solution must generalize to unseen test inputs that follow the same transformation rule",
    ]

    return "\n".join(lines)


def build_correction_prompt(
    task_id: str,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    features: dict,
    tried_ops: List[str],
    previous_code: str,
    failure_reason: str,
) -> str:
    """Build a self-correction prompt that feeds back the failed code and error."""
    base = build_prompt(task_id, train_pairs, features, tried_ops)

    correction = "\n".join([
        "",
        "=== PREVIOUS ATTEMPT (FAILED) ===",
        "You previously generated the following code, but it did NOT produce the correct output for all train pairs:",
        "",
        "```python",
        previous_code,
        "```",
        "",
        f"Failure reason: {failure_reason}",
        "",
        "=== CORRECTION REQUIRED ===",
        "Carefully re-examine the train pairs. Identify what went wrong in the previous attempt.",
        "Fix the logic and produce a corrected `solve()` function below (```python ... ``` block only):",
    ])

    return base + correction
