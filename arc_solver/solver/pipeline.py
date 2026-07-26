"""
pipeline.py
-----------
Main entry point for NeuroSymbolicARCSolver.
Integrates:
  1. DiagnosticEngine (Phase 0 external lookup + Phase 1 rule-based analyzers + Phase 2 LLM fallback)
  2. CategorySpecializedARCSolver (Beam search synthesis)
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Any, Optional

from ..neural.ranker import load_ranker
from .specialized_solvers import CategorySpecializedARCSolver
from ..meta.diagnostic_engine import DiagnosticEngine
from ..meta.hypothesis_tester import _apply_candidate


class NeuroSymbolicARCSolver:
    """
    Unified ARC solver combining DiagnosticEngine (parallel rule-based analyzers)
    with Category-Specialized Beam Search synthesis.
    """

    def __init__(
        self,
        beam_width: int = 50,
        max_depth: int = 3,
        ranker: Any = None,
        ranker_weights: Any = None,
        use_diagnostic_engine: bool = True,
        use_solution_lookup: bool = True,
        use_llm: bool = True,
    ):
        self.ranker = ranker if ranker is not None else load_ranker(ranker_weights)
        self.specialized_router = CategorySpecializedARCSolver(beam_width, max_depth, self.ranker)
        self.use_diagnostic_engine = use_diagnostic_engine
        
        if self.use_diagnostic_engine:
            self.diagnostic_engine = DiagnosticEngine(
                use_solution_lookup=use_solution_lookup,
                use_llm=use_llm,
            )
        else:
            self.diagnostic_engine = None

    def solve_task(
        self,
        train_pairs: List[Tuple[Any, Any]],
        test_inputs: List[Any],
        task_id: str = "unknown_task",
    ) -> Tuple[List[List[np.ndarray]], List[Any]]:
        """
        Solve an ARC task using the unified pipeline.
        
        Order of execution:
          Step 1: DiagnosticEngine (External lookup + 33 Parallel Rule Analyzers)
          Step 2: If unresolved -> Beam Search Synthesis (CategorySpecializedARCSolver)
        """
        # Format train pairs as numpy arrays
        np_train_pairs = [
            (np.asarray(inp, dtype=np.int16), np.asarray(out, dtype=np.int16))
            for inp, out in train_pairs
        ]
        np_test_inputs = [np.asarray(t, dtype=np.int16) for t in test_inputs]

        # Step 1: Run Diagnostic Engine (Analyzers + Lookup)
        if self.use_diagnostic_engine and self.diagnostic_engine is not None:
            diag = self.diagnostic_engine.diagnose(task_id, np_train_pairs)
            
            if diag.success:
                predictions = []
                for test_inp in np_test_inputs:
                    try:
                        if diag.solve_fn:
                            pred1 = np.asarray(diag.solve_fn(test_inp.copy()), dtype=np.int16)
                        elif diag.candidate:
                            pred1 = _apply_candidate(diag.candidate, test_inp.copy())
                        else:
                            pred1 = test_inp.copy()
                    except Exception:
                        pred1 = test_inp.copy()
                    
                    if pred1 is None:
                        pred1 = test_inp.copy()
                    
                    predictions.append([pred1, pred1])
                
                # Return predictions and the winning program/analyzer candidate
                prog_repr = diag.candidate if diag.candidate else f"Analyzer:{diag.analyzer_name}"
                return predictions, [prog_repr]

        # Step 2: Fallback to Category-Specialized Beam Search Synthesis
        return self.specialized_router.solve_task(np_train_pairs, np_test_inputs)
