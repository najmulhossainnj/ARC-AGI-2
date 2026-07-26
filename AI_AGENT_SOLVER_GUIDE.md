# AI Agent Development Manual & Prompt Guide: Generalizable Pattern Extraction for ARC-AGI

> **Document Designation**: AI Agent System Instructions & Operational Guide  
> **Target Audience**: AI Coding Assistants, Subagents, and Automated Solvers  
> **Core Goal**: Guide AI agents to analyze ARC-AGI tasks, extract non-overfitting algorithmic transformation rules, use web search for task solution writeups when stuck, and implement standalone `Analyzer` classes.

---

## 1. Executive Guidelines for AI Agents

When tasked with improving the ARC-AGI solver:

1. **NO OVERFITTING / NO HARDCODING**: Never hardcode specific grid dimensions (e.g. `grid.shape == (30,30)`), specific task IDs, or fixed pixel offsets unless mathematically inherent to the pattern class.
2. **PARALLEL STANDALONE ANALYZERS**: Implement all rules as `Analyzer` subclasses that return a `ProgramCandidate` containing a pure Python/NumPy closure `solve_fn(grid)`.
3. **ZERO FALSE POSITIVES**: Every candidate MUST pass $100\%$ exact-match verification (`verify_100pct`) across all training pairs before acceptance.
4. **COMPETITION OFFLINE COMPATIBILITY**: Analyzers must run in $<0.05\text{s}$ using local NumPy/SciPy operations without network API dependencies.

---

## 2. The 5-Phase Pattern Extraction Workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PHASE A: Visual & Invariant Inspection          │
│                        Inspect grid shapes, color deltas, & topology   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                         [Pattern Clear & Detected?]
                          ├── YES ──► Jump to PHASE C
                          │
                          └── NO  ──► PHASE B: Web Search Fallback
                                      Search web for writeups/articles on task ID
                                    │
┌───────────────────────────────────┴────────────────────────────────────┐
│                        PHASE C: Formulate Abstract Transformation Rule │
│                        Express invariant as parameterized NumPy logic  │
├────────────────────────────────────────────────(───────────────────────┤
│                        PHASE D: Implement Standalone Analyzer          │
│                        Write _detect() and solve_fn() closure          │
├────────────────────────────────────────────────────────────────────────┤
│                        PHASE E: Register & 100% Verification           │
│                        Add to ALL_ANALYZERS and verify all pairs       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase B: Web Search Fallback Strategy (When Visual Inspection Fails)

If an AI agent cannot deduce the transformation rule from raw grid inspection alone, **it must trigger a targeted web search for solution articles and writeups**.

### Standardized Web Search Queries
Use the exact task ID in the search query:

```bash
# Query Template 1: Search for task solution writeups
"ARC-AGI task <TASK_ID> solution"

# Query Template 2: Search for puzzle explanation articles
"ARC puzzle <TASK_ID> explanation"

# Query Template 3: Search GitHub repositories for solver code
site:github.com "<TASK_ID>" "solver.py" OR "solve"

# Query Template 4: Search Kaggle / Reddit discussions
"ARC-AGI" "<TASK_ID>" solver OR solution
```

### Digesting Web Articles into Generalizable Code
When an article or GitHub solver is retrieved:
1. **Extract the Core Concept**: Ignore hardcoded index numbers or specific colors in the writeup. Extract the high-level logic (e.g. *"The puzzle extracts 2D spatial centroids of isolated shapes and arranges them into a grid"*).
2. **Translate to NumPy Logic**: Replace specific color constants with dynamic detection (`_bg(grid)`, `np.unique(grid)`).
3. **Verify Generalization**: Ensure the logic works for any grid dimensions, not just the dimensions in the article.

---

## 4. Complete Code Template for Writing an `Analyzer` Class

Every new pattern discovered should be written using this template in `arc_solver/meta/analyzers/corpus_analyzers.py`:

```python
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional
from .base import Analyzer, ProgramCandidate

class <PatternName>Analyzer(Analyzer):
    """
    Detect: <ONE SENTENCE DESCRIPTION OF THE PATTERN CLASS>
    """
    name = "<pattern_name_snake_case>"
    priority = 15  # Priority: 5-15 (fast/specific), 20-30 (structural), 35+ (expensive)

    def analyze(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]], features: dict) -> Optional[ProgramCandidate]:
        # 1. Feature Pre-filtering (e.g. check same_size if required)
        if not features.get("same_size"):
            return None

        # 2. Check pattern condition across ALL training pairs
        for inp, out in train_pairs:
            inp, out = np.asarray(inp), np.asarray(out)
            if not self._check_pair(inp, out):
                return None

        # 3. Construct Parameterized solve_fn closure
        def make_solve_fn():
            def solve_fn(grid: np.ndarray) -> List[List[int]]:
                g = np.asarray(grid).copy()
                # === IMPLEMENT GENERALIZABLE TRANSFORMATION ALGORITHM ===
                # Use g.shape dynamically, extract colors dynamically
                return g.tolist()
            return solve_fn

        return ProgramCandidate(
            op="<UPPERCASE_OP_NAME>",
            params=(),
            description="<Human readable description>",
            solve_fn=make_solve_fn()
        )

    def _check_pair(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Verify if the transformation rule produces 100% exact match on pair."""
        if inp.shape != out.shape:
            return False
        
        # Compute predicted grid candidate
        cand = inp.copy()
        # ... apply transformation ...
        
        return np.array_equal(cand, out)
```

---

## 5. System Registration Checklist

After writing the `Analyzer` class:

1. **Import in `arc_solver/meta/analyzers/__init__.py`**:
   ```python
   from .corpus_analyzers import <PatternName>Analyzer
   ```
2. **Add to `ALL_ANALYZERS` List**:
   ```python
   ALL_ANALYZERS = [
       ...
       <PatternName>Analyzer(),
       ...
   ]
   ```
3. **Test Benchmark**: Run `python run_batch_next_10.py` to confirm instant solve verification.
4. **Push to GitHub**: Commit and push changes to `GitMonsters/ARC-AGI-2`.

---

## 6. Ready-to-Use AI Subagent Prompt

Copy and paste this prompt when invoking another subagent or AI system:

```markdown
You are an expert Neuro-Symbolic AI Coding Agent tasked with solving ARC-AGI puzzles.

YOUR TASK:
1. Inspect the provided ARC task training pairs (inputs and outputs).
2. If the pattern is not immediately obvious, run a web search using query: `"ARC puzzle <TASK_ID> solution"` or `"ARC-AGI task <TASK_ID> explanation"`.
3. Extract the underlying geometric/topological invariant from visual inspection or web writeups.
4. Convert the logic into a standalone, non-overfitting `Analyzer` subclass inside `arc_solver/meta/analyzers/corpus_analyzers.py`.
5. Ensure the analyzer defines a parameterized `solve_fn(grid)` closure operating in NumPy.
6. Register the analyzer in `ALL_ANALYZERS` in `arc_solver/meta/analyzers/__init__.py`.
7. Verify that `verify_100pct` succeeds on 100% of training pairs with zero false positives.
```
