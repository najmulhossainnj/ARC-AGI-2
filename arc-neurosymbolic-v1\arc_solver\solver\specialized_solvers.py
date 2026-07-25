"""
Category-Specialized Solvers.

Routes ARC tasks to specialized solver engines based on structural task category
(SAME_SIZE, DOWNSCALE_CROP, UPSCALE_GROW, DYNAMIC). Each solver focuses exclusively
on transformation families that match the task's mathematical invariants.
"""
from __future__ import annotations
from typing import List, Tuple, Any
import numpy as np

from ..synthesis.classifier import classify_task, filter_families
from ..synthesis.beam_search import BeamSearcher
from ..synthesis.grammar import SOLVER_FAMILIES, COMPOSABLE_FAMILIES

class CategorySpecializedARCSolver:
    """Multi-Solver Router that selects specialized solver configurations per task category."""

    def __init__(self, beam_width: int = 50, max_depth: int = 3, ranker: Any = None):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.ranker = ranker
        self.beam_searcher = BeamSearcher(beam_width=beam_width, max_depth=max_depth)

    def solve_task(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
    ) -> Tuple[List[List[np.ndarray]], List[Any]]:
        category = classify_task(train_pairs)
        
        # Route to specialized family set based on category
        specialized_solver_fams = set(filter_families(category, SOLVER_FAMILIES))
        specialized_comp_fams = set(filter_families(category, COMPOSABLE_FAMILIES))

        # Run beam search with category-specialized grammar families
        found, candidates = self.beam_searcher.search(
            train_pairs,
            families=specialized_solver_fams
        )

        # Only use verified exact-match programs (err == 0.0 on ALL train pairs)
        # to guarantee zero false positives.
        all_programs = found
        predictions = []

        if found:
            def get_prog(item):
                return item.program if hasattr(item, 'program') else item

            best_prog = get_prog(found[0])
            for test_inp in test_inputs:
                from ..dsl.executor import execute
                pred1 = execute(best_prog, test_inp)
                if pred1 is None:
                    pred1 = test_inp.copy()
                
                # Attempt 2 fallback
                pred2 = pred1
                if len(found) > 1:
                    attempt2_prog = get_prog(found[1])
                    p2 = execute(attempt2_prog, test_inp)
                    if p2 is not None:
                        pred2 = p2
                
                predictions.append([pred1, pred2])
        else:
            # Fallback identity predictions
            for test_inp in test_inputs:
                predictions.append([test_inp.copy(), test_inp.copy()])

        return predictions, all_programs
