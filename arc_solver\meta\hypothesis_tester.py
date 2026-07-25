from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .analyzers.base import ProgramCandidate


def _apply_candidate(candidate: ProgramCandidate, inp: np.ndarray) -> Optional[np.ndarray]:
    """Apply a ProgramCandidate to an input grid and return the result."""
    from ..dsl.transforms import apply_grid_op
    try:
        result = apply_grid_op(inp, candidate.op, candidate.params)
        return np.asarray(result, dtype=np.int16) if result is not None else None
    except Exception as e:
        return None


def verify_100pct(
    candidate: ProgramCandidate,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> bool:
    """Return True if candidate achieves exact match on ALL training pairs."""
    if not train_pairs:
        return False
    for inp, out in train_pairs:
        inp = np.asarray(inp, dtype=np.int16)
        out = np.asarray(out, dtype=np.int16)
        pred = _apply_candidate(candidate, inp)
        if pred is None:
            return False
        if pred.shape != out.shape:
            return False
        if not np.array_equal(pred, out):
            return False
    return True


def verify_solve_fn(
    solve_fn,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> bool:
    """Return True if a solve() function achieves exact match on ALL training pairs."""
    for inp, out in train_pairs:
        inp = np.asarray(inp, dtype=np.int16)
        out = np.asarray(out, dtype=np.int16)
        try:
            pred = solve_fn(inp.copy())
            pred = np.asarray(pred, dtype=np.int16)
            if pred.shape != out.shape or not np.array_equal(pred, out):
                return False
        except Exception:
            return False
    return True
