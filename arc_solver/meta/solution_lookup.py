"""
Solution Lookup - Pre-computed solution database.

This module provides a lookup system for pre-computed ARC task solutions
from external sources. When a task is requested, it first checks if we
have a pre-computed solution, avoiding the need for LLM calls.
"""

from __future__ import annotations
import os
import sys
import json
import importlib.util
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExternalSolution:
    """A pre-computed solution for a task."""
    task_id: str
    solver_path: str
    solve_fn: Callable
    source: str  # e.g., "GitMonsters/SOLVED-540"


class SolutionLookup:
    """
    Solution lookup system for pre-computed ARC solutions.
    
    This class:
    1. Scans external solution directories for available solvers
    2. Provides fast lookup by task ID
    3. Caches loaded solvers for performance
    """
    
    def __init__(self, solutions_dir: Optional[str] = None):
        """
        Initialize the solution lookup.

        Args:
            solutions_dir: Path to directory containing external solutions.
                          If None, searches standard locations automatically.
        """
        # Resolve search paths: prefer explicitly given, then external_solutions/solves,
        # then the arc_solver/../external_solutions/solves pattern
        self._search_dirs: List[Path] = []

        if solutions_dir:
            self._search_dirs.append(Path(solutions_dir))
        else:
            base = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self._search_dirs.append(base / "external_solutions" / "solves")
            # Also handle running from within arc_solver itself
            self._search_dirs.append(base.parent / "external_solutions" / "solves")

        # Use first existing dir as primary
        self.solutions_dir = next(
            (p for p in self._search_dirs if p.exists()),
            self._search_dirs[0]
        )

        self._cache: Dict[str, ExternalSolution] = {}
        self._loaded = False
        
    def _load_all_solutions(self) -> None:
        """Scan and load all available solutions from all search dirs."""
        if self._loaded:
            return

        count = 0
        searched = []
        for sol_dir in self._search_dirs:
            if not sol_dir.exists():
                searched.append(str(sol_dir))
                continue

            for task_dir in sol_dir.iterdir():
                if not task_dir.is_dir():
                    continue

                task_id = task_dir.name
                if task_id in self._cache:
                    continue  # Already loaded from a higher-priority dir

                solver_path = task_dir / "solver.py"
                if not solver_path.exists():
                    continue

                try:
                    solve_fn = self._load_solver(solver_path)
                    if solve_fn is not None:
                        self._cache[task_id] = ExternalSolution(
                            task_id=task_id,
                            solver_path=str(solver_path),
                            solve_fn=solve_fn,
                            source="GitMonsters/SOLVED-540"
                        )
                        count += 1
                except Exception:
                    pass

        self._loaded = True
        print(f"[SolutionLookup] Loaded {count} external solutions "
              f"(searched: {[str(p) for p in self._search_dirs]})")
    
    def _load_solver(self, solver_path: Path) -> Optional[Callable]:
        """Load a solver module safely using GitMonsters loader (strips top-level exec code)."""
        try:
            # Use the safe GitMonsters loader that strips file-reading side effects
            from .gitmonsters_loader import load_gitmonsters_solver
            return load_gitmonsters_solver(solver_path)
        except ImportError:
            pass

        # Fallback: standard importlib (may fail for solvers with top-level file reads)
        try:
            spec = importlib.util.spec_from_file_location("solver", solver_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules["external_solver"] = module
            spec.loader.exec_module(module)

            if hasattr(module, "solve"):
                return module.solve
        except Exception:
            pass
        finally:
            if "external_solver" in sys.modules:
                del sys.modules["external_solver"]
        
        return None
    
    def get_solution(self, task_id: str) -> Optional[ExternalSolution]:
        """
        Get a pre-computed solution for a task ID.
        
        Args:
            task_id: The ARC task ID
            
        Returns:
            ExternalSolution if found, None otherwise
        """
        if not self._loaded:
            self._load_all_solutions()
        
        return self._cache.get(task_id)
    
    def has_solution(self, task_id: str) -> bool:
        """Check if we have a pre-computed solution for a task."""
        if not self._loaded:
            self._load_all_solutions()
        return task_id in self._cache
    
    def solve(self, task_id: str, grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply a pre-computed solution to a grid.
        
        Args:
            task_id: The ARC task ID
            grid: Input grid as numpy array
            
        Returns:
            Output grid if solution exists and succeeds, None otherwise
        """
        solution = self.get_solution(task_id)
        if solution is None:
            return None
        
        try:
            # Convert to list format expected by solvers
            grid_list = grid.tolist() if isinstance(grid, np.ndarray) else grid
            
            # Call the solver
            result = solution.solve_fn(grid_list)
            
            # Convert back to numpy
            if isinstance(result, list):
                return np.array(result, dtype=np.int16)
            return result
        except Exception as e:
            print(f"[SolutionLookup] Error applying solution for {task_id}: {e}")
            return None
    
    def verify_solution(
        self,
        task_id: str,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """
        Verify that a pre-computed solution works for training pairs.
        
        Args:
            task_id: The ARC task ID
            train_pairs: List of (input, expected_output) pairs
            
        Returns:
            True if all pairs match, False otherwise
        """
        solution = self.get_solution(task_id)
        if solution is None:
            return False
        
        for inp, expected in train_pairs:
            result = self.solve(task_id, inp)
            if result is None:
                return False
            
            if result.shape != expected.shape:
                return False
            
            if not np.array_equal(result, expected):
                return False
        
        return True
    
    @property
    def count(self) -> int:
        """Number of available solutions."""
        if not self._loaded:
            self._load_all_solutions()
        return len(self._cache)
    
    def __len__(self) -> int:
        return self.count
    
    def __contains__(self, task_id: str) -> bool:
        return self.has_solution(task_id)


# Global instance for convenience
_global_lookup: Optional[SolutionLookup] = None


def get_solution_lookup() -> SolutionLookup:
    """Get the global solution lookup instance."""
    global _global_lookup
    if _global_lookup is None:
        _global_lookup = SolutionLookup()
        _global_lookup._load_all_solutions()
    return _global_lookup
