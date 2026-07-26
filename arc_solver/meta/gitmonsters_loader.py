"""
gitmonsters_loader.py
---------------------
Safely loads and wraps GitMonsters solver files for use in SolutionLookup.

Each solver file has:
  1. A function named solve_<task_id>(grid) -> list
  2. Top-level testing code that reads from a hardcoded path (UNSAFE to exec directly)
  3. A final line: solve = solve_<task_id>

This loader:
  - Strips the top-level testing/file-read code
  - Extracts only the function definitions
  - Exposes a clean solve(grid) -> list interface
"""
from __future__ import annotations
import re
import ast
import sys
import importlib.util
from pathlib import Path
from typing import Optional, Callable, Dict


# Known task IDs from GitMonsters/13-Impossible-ARC-Tasks-SOLVED
GITMONSTERS_TASK_IDS = [
    "abc82100", "21897d95", "e12f9a14", "a32d8b75", "9bbf930d",
    "4e34c42c", "88bcf3b4", "13e47133", "8b7bacbf", "62593bfd",
    "88e364bc", "2b83f449", "269e22fb",
]


def _strip_toplevel_exec_code(source: str) -> str:
    """
    Remove top-level code that reads files or runs tests (unsafe when exec'd).
    Keeps only: imports, function/class definitions, and the final 'solve = ...' assignment.
    """
    lines = source.splitlines()
    out_lines = []
    
    # Parse to find function defs and their end lines
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    
    # Collect ranges of function/class definitions
    safe_ranges = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, 'end_lineno'):
                for ln in range(node.lineno, node.end_lineno + 1):
                    safe_ranges.add(ln)
    
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        
        # Always keep: function/class definitions and their bodies
        if i in safe_ranges:
            out_lines.append(line)
            continue
        
        # Always keep: imports
        if stripped.startswith('import ') or stripped.startswith('from '):
            out_lines.append(line)
            continue
        
        # Always keep: solve = solve_<task_id> assignment
        if re.match(r'^solve\s*=\s*solve_\w+', stripped):
            out_lines.append(line)
            continue
        
        # Always keep: solve = lambda ... or other short solve assignments
        if re.match(r'^solve\s*=\s*', stripped) and 'open(' not in stripped:
            out_lines.append(line)
            continue
        
        # Skip everything else (file reads, test loops, print statements, etc.)
    
    return "\n".join(out_lines)


def load_gitmonsters_solver(solver_path: Path) -> Optional[Callable]:
    """
    Load a GitMonsters solver.py file safely and return its solve() function.
    
    Returns:
        callable solve(grid: list) -> list, or None on failure
    """
    try:
        source = solver_path.read_text(encoding='utf-8')
        clean_source = _strip_toplevel_exec_code(source)
        
        namespace = {}
        # Provide common stdlib modules the solvers may need
        import collections, copy, itertools, functools, math, heapq
        namespace.update({
            'collections': collections,
            'copy': copy,
            'itertools': itertools,
            'functools': functools,
            'math': math,
            'heapq': heapq,
            'deque': collections.deque,
            'defaultdict': collections.defaultdict,
            'Counter': collections.Counter,
        })
        
        exec(compile(clean_source, str(solver_path), 'exec'), namespace)
        
        solve_fn = namespace.get('solve')
        if callable(solve_fn):
            return solve_fn
        
        # Fallback: look for solve_<task_id> function
        task_id = solver_path.parent.name
        alt_fn = namespace.get(f'solve_{task_id}')
        if callable(alt_fn):
            return alt_fn
        
        return None
    
    except Exception as e:
        print(f"[GitMonstersLoader] Failed to load {solver_path}: {e}")
        return None


def load_all_gitmonsters_solvers(solvers_root: Path) -> Dict[str, Callable]:
    """
    Load all solver.py files from solvers_root/<task_id>/solver.py
    
    Returns:
        dict mapping task_id -> solve_fn
    """
    result = {}
    if not solvers_root.exists():
        print(f"[GitMonstersLoader] solvers_root not found: {solvers_root}")
        return result
    
    for task_dir in sorted(solvers_root.iterdir()):
        if not task_dir.is_dir():
            continue
        solver_file = task_dir / 'solver.py'
        if not solver_file.exists():
            continue
        
        task_id = task_dir.name
        fn = load_gitmonsters_solver(solver_file)
        if fn is not None:
            result[task_id] = fn
        else:
            print(f"[GitMonstersLoader] Could not load solve() from {task_id}/solver.py")
    
    return result
