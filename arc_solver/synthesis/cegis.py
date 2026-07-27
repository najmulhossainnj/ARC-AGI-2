"""
cegis.py
--------
Counterexample-Guided Inductive Synthesis (CEGIS) Refinement Engine for ARC.
Extracts failure diagnostics from counterexample training pairs to refine candidate programs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class Counterexample:
    """Diagnostic details of a counterexample failure."""
    pair_index: int
    predicted_grid: np.ndarray
    expected_grid: np.ndarray
    shape_mismatch: bool
    num_mismatch_cells: int
    mismatch_coords: List[Tuple[int, int]]
    mismatch_details: List[Tuple[int, int, int, int]]  # (r, c, pred_color, exp_color)
    refined_constraints: List[str] = field(default_factory=list)


class CEGISRefiner:
    """Analyzes counterexamples to generate refined synthesis constraints."""

    @classmethod
    def diagnose_failure(
        cls, pair_index: int, predicted: np.ndarray, expected: np.ndarray
    ) -> Counterexample:
        p = np.asarray(predicted, dtype=int)
        e = np.asarray(expected, dtype=int)

        if p.shape != e.shape:
            return Counterexample(
                pair_index=pair_index,
                predicted_grid=p,
                expected_grid=e,
                shape_mismatch=True,
                num_mismatch_cells=int(e.size),
                mismatch_coords=[],
                mismatch_details=[],
                refined_constraints=[f"REQUIRE_OUTPUT_SHAPE_{e.shape[0]}x{e.shape[1]}"],
            )

        diffs = np.argwhere(p != e)
        num_diffs = len(diffs)
        coords = [(int(r), int(c)) for r, c in diffs]
        details = [(int(r), int(c), int(p[r, c]), int(e[r, c])) for r, c in diffs[:20]]

        constraints = []
        if num_diffs > 0:
            # Analyze pattern of mismatches
            exp_colors = set(e[r, c] for r, c in coords)
            pred_colors = set(p[r, c] for r, c in coords)
            if len(exp_colors) == 1:
                target_c = next(iter(exp_colors))
                constraints.append(f"RECOLOR_TARGET_{target_c}")
            if len(coords) <= 3:
                constraints.append("REPAIR_ISOLATED_CELLS")

        return Counterexample(
            pair_index=pair_index,
            predicted_grid=p,
            expected_grid=e,
            shape_mismatch=False,
            num_mismatch_cells=num_diffs,
            mismatch_coords=coords,
            mismatch_details=details,
            refined_constraints=constraints,
        )

    @classmethod
    def refine_candidate(
        cls, counterexamples: List[Counterexample]
    ) -> List[str]:
        all_constraints = []
        for ce in counterexamples:
            all_constraints.extend(ce.refined_constraints)
        return list(set(all_constraints))
