# Complete System Architecture & Reference Manual: Neuro-Symbolic ARC-AGI Solver

> **System Designation**: Neuro-Symbolic ARC-AGI Solver (V2 Architecture)  
> **Repository Path**: `e:/kaggle/antigravity` | GitHub: `GitMonsters/ARC-AGI-2`  
> **Core Objective**: Solve ARC-AGI transformation puzzles via a 3-phase hybrid architecture combining ultra-fast parallel rule-based analyzers, pre-computed solution lookup, multi-provider LLM code generation, and category-specialized beam search synthesis.

---

## 1. High-Level Architecture Overview

The system processes ARC-AGI task instances through a multi-tiered diagnostic and synthesis pipeline. The design prioritizes **speed, generalization, and zero false-positive rates**:

```
                                  [Input ARC Task]
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │   NeuroSymbolicARCSolver.solve_task   │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                      ┌────────────────────────────────────┐
                      │    DiagnosticEngine.diagnose()     │
                      └──────────────────┬─────────────────┘
                                         │
           ┌─────────────────────────────┼─────────────────────────────┐
           ▼                             ▼                             ▼
   【PHASE 0: Lookup】           【PHASE 1: Analyzers】          【PHASE 2: LLM Fallback】
   Pre-computed solver         33 Rule-Based Analyzers         Multi-provider Codegen
   database search             (ThreadPoolExecutor <0.05s)      & Self-Correction Loop
           │                             │                             │
           └─────────────────────────────┼─────────────────────────────┘
                                         │
                              [Verified 100% Match?]
                                   ├── YES ──► [Instant Prediction Output]
                                   │
                                   └── NO  ──► 【PHASE 3: Beam Search Synthesis】
                                               Category-Specialized DSL Search
```

---

## 2. The 3-Phase Diagnostic Engine (`arc_solver/meta/diagnostic_engine.py`)

### Phase 0: External Solution Lookup (`SolutionLookup`)
- **Location**: `arc_solver/meta/solution_lookup.py` & `arc_solver/meta/gitmonsters_loader.py`
- **Mechanism**: Checks local solver repositories (`external_solutions/solves/<task_id>/solver.py`) for a pre-computed solution.
- **Safety**: Uses AST-based code stripping (`gitmonsters_loader.py`) to eliminate unsafe top-level executable code (file-reading loops, test calls) while preserving function definitions.
- **Verification**: Every retrieved solution is verified against all training pairs before being accepted.

### Phase 1: 33 Parallel Rule-Based Analyzers (`ALL_ANALYZERS`)
- **Location**: `arc_solver/meta/analyzers/`
- **Execution**: All 33 analyzers execute concurrently in a `ThreadPoolExecutor` (max 8-12 workers) with a total timeout limit of 10.0 seconds (average execution: **<0.05s per task**).
- **Output**: Each analyzer evaluates task features and returns a `ProgramCandidate` containing:
  - `op`: Transformation operator name
  - `params`: Algorithm parameter tuple
  - `solve_fn`: Executable Python transformation function `solve_fn(grid: List[List[int]]) -> List[List[int]]`
- **Verification**: Candidate predictions are tested against ALL training pairs via `verify_100pct`. If 100% exact match is confirmed, execution stops immediately and returns the prediction.

### Phase 2: LLM Code Generation & Self-Correction Fallback
- **Location**: `arc_solver/meta/llm/primitive_codegen.py` & `prompt_builder.py`
- **Trigger**: Activates if Phase 0 and Phase 1 fail to produce a 100% match.
- **Mechanism**: Formats task inputs/outputs into structured natural language diagnostic prompts (Phase 0 visual inspection, Phase 1 structural roles, Phase 2 hypothesis, Phase 3 Python `solve()` code).
- **Self-Correction**: Runs up to 2 code generation attempts. If attempt 1 fails on a training pair, the exact pixel diff is fed back into attempt 2 for targeted refinement.

---

## 3. Main Pipeline Integration (`arc_solver/solver/pipeline.py`)

The primary entry point for the entire codebase is `NeuroSymbolicARCSolver`:

```python
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

solver = NeuroSymbolicARCSolver(
    beam_width=50,
    max_depth=3,
    use_diagnostic_engine=True,
    use_solution_lookup=True,
    use_llm=True
)

# Solve task:
predictions, programs = solver.solve_task(
    train_pairs=[(inp1, out1), (inp2, out2)],
    test_inputs=[test_inp1],
    task_id="00576224"
)
```

### Execution Flow inside `solve_task`:
1. Formats input/output grids into standard NumPy integer arrays (`np.int16`).
2. Calls `DiagnosticEngine.diagnose(task_id, train_pairs)`.
3. If Phase 0, Phase 1, or Phase 2 succeeds, applies the verified `solve_fn` or `candidate` to all test inputs and returns predictions immediately.
4. If Phase 0-2 fail, hands off to Phase 3: `CategorySpecializedARCSolver` for category-guided composable beam search synthesis.

---

## 4. Complete Catalog of 33 Rule-Based Analyzers

Analyzers are ordered by priority (lower priority number = executes first, higher specificity, lower computational cost).

| Priority | Analyzer Name | File Location | Transformation Class / Description |
|---|---|---|---|
| **1** | `PatternAnalyzerWrapper` | `pattern_analyzer_wrapper.py` | Periodic 1D/2D grid pattern detection and color remapping. |
| **5** | `ColorSubstitutionAnalyzer` | `color_substitution.py` | Unconditional 1-to-1 static color remapping `{old_color: new_color}`. |
| **8** | `SizeSelectionAnalyzer` | `corpus_analyzers.py` | Filter operation: retains only the largest or smallest object by cell area. |
| **10** | `AnomalyRepairAnalyzer` | `corpus_analyzers.py` | Micro-repair: corrects single-cell outliers violating global grid symmetry. |
| **10** | `TranslationAnalyzer` | `translation.py` | Uniform offset shift $(dr, dc)$ of all non-background objects. |
| **12** | `GravityFallAnalyzer` | `misc_analyzers.py` | Unidirectional edge projection: objects fall towards bottom/top/left/right edge. |
| **12** | `LegendShapeToColorAnalyzer` | `corpus_analyzers.py` | Legend key: recolors primary target object based on indicator shape signature. |
| **15** | `AlternatingFlipTilingAnalyzer` | `corpus_analyzers.py` | $R \times C$ tile expansion with alternating row/column horizontal/vertical flips. |
| **15** | `SymmetryCompleteAnalyzer` | `misc_analyzers.py` | Horizontally, vertically, or diagonally mirrors grids to repair broken symmetry. |
| **16** | `UniqueObjectExtractorAnalyzer` | `corpus_analyzers.py` | Odd-one-out filter: extracts the single unique object among repeated shapes. |
| **18** | `ColorIndexedTilingAnalyzer` | `corpus_analyzers.py` | Uniform $N \times M$ tile pattern repetition across target grid dimensions. |
| **18** | `PanelBooleanLogicAnalyzer` | `corpus_analyzers.py` | Bitwise overlap/intersection (`AND`, `OR`, `XOR`) between sub-panels split by dividers. |
| **18** | `BlockCycleAnalyzer` | `misc_analyzers.py` | Vertical block cycling and pattern replication across stacked sub-blocks. |
| **20** | `GridSectionLegendAnalyzer` | `corpus_analyzers.py` | Separator line detection: parses legend/key section and applies rule to puzzle section. |
| **20** | `FrameSizeToFillColorAnalyzer` | `corpus_analyzers.py` | Fills rectangular frame interiors with colors mapped from frame bounding box area. |
| **20** | `ArrowReplicateAnalyzer` | `replication.py` | Arrow-driven template replication along directional indicators. |
| **22** | `PatternExtensionAnalyzer` | `misc_analyzers.py` | Sequence completion: extends 1D/2D diagonal color sequences. |
| **22** | `DiagonalPatternAnalyzer` | `corpus_analyzers.py` | Diagonal color bands: fills cells where $(r+c) \pmod{\text{period}} == k$. |
| **22** | `MultiEdgeGravityAnalyzer` | `advanced_analyzers.py` | Multi-directional gravity: projects objects to top and bottom boundaries simultaneously. |
| **25** | `NetworkConnectivityFillAnalyzer` | `advanced_analyzers.py` | Network connectivity: propagates seed dots through 8-connected chains to fill frames. |
| **25** | `BorderCropAnalyzer` | `misc_analyzers.py` | Bounding-box crop: clips output to bounding box of content colors. |
| **26** | `SortByAttributeAnalyzer` | `corpus_analyzers.py` | Object sorting: rearranges objects by area, color index, or spatial position. |
| **28** | `LegendRaySlideAnalyzer` | `advanced_analyzers.py` | Legend ray sliding: projects directional beams from legend markers. |
| **28** | `ObjectStampRuleAnalyzer` | `advanced_analyzers.py` | Object stamping: stamps multi-cell templates based on 2-cell color-chain rules. |
| **28** | `ParallelogramAlignAnalyzer` | `misc_analyzers.py` | Slanted parallelogram alignment along fixed vertical or horizontal anchors. |
| **28** | `DiagonalChainAnalyzer` | `misc_analyzers.py` | Top-left diagonal object stacking sorted by input column index. |
| **30** | `ConcentricRingFillAnalyzer` | `advanced_analyzers.py` | Chebyshev distance fill: fills frame interiors with concentric distance color bands. |
| **32** | `TArrowMarkerFlowAnalyzer` | `advanced_analyzers.py` | T-shaped arrow flow: identifies T-markers and flows color along shaft direction. |
| **34** | `TopologyHoleAnalyzer` | `corpus_analyzers.py` | Topology-driven rule: recolors or selects objects based on internal hole count. |
| **35** | `RayCollisionDeflectionAnalyzer` | `advanced_analyzers.py` | Physics ray deflection: emits rays that bounce/deflect upon hitting obstacle cells. |
| **35** | `PuzzleStitchAssemblyAnalyzer` | `advanced_analyzers.py` | Puzzle stitching: assembles isolated jigsaw fragments into a single contiguous shape. |
| **38** | `CountDrivenRuleAnalyzer` | `corpus_analyzers.py` | Count-driven rule: maps input object/cell count to output dimensions or copy count. |
| **40** | `MasterTemplateInpaintAnalyzer` | `advanced_analyzers.py` | D4 template match: matches input against 8 orientations of a master template. |

---

## 5. Pattern Extraction & Generalization Workflow

When analyzing solved tasks to build new generalizable analyzers, follow this 5-step methodology:

1. **Read Docstring & Rule Summary**: Extract the abstract logic sentence:  
   `"<INPUT STRUCTURE> determines <WHAT CHANGES>, applied via <ALGORITHM>"`
2. **Identify Input/Output Invariants**:
   - Check if shapes, dimensions, colors, or object counts are preserved.
   - Separate background from foreground, templates from indicators.
3. **Define Generalizable Detector (`_detect`)**:
   - Avoid hardcoding specific task IDs, grid sizes, or color numbers.
   - Express structural conditions using NumPy masks, connected components (`_components`), bounding boxes (`_bbox`), and normalized shapes (`_norm_shape`).
4. **Construct Parameterized `solve_fn`**:
   - Attach a closure function `solve_fn(grid: List[List[int]]) -> List[List[int]]` to the returned `ProgramCandidate`.
   - Ensure `solve_fn` accepts ANY grid following the discovered parameterization.
5. **Register in `ALL_ANALYZERS`**:
   - Insert into `arc_solver/meta/analyzers/__init__.py` with appropriate priority.
   - Priority guideline: Fast/specific filters ($5-20$), Structural transforms ($20-30$), Expensive graph/search solvers ($30-40$).

---

## 6. Directory Structure Reference

```
e:/kaggle/antigravity/
├── arc_solver/
│   ├── meta/
│   │   ├── analyzers/                  # 33 Rule-Based Analyzers
│   │   │   ├── __init__.py             # ALL_ANALYZERS registry (sorted by priority)
│   │   │   ├── base.py                 # Analyzer & ProgramCandidate dataclass
│   │   │   ├── corpus_analyzers.py     # 14 high-frequency corpus analyzers
│   │   │   ├── advanced_analyzers.py   # 9 advanced neuro-symbolic analyzers
│   │   │   ├── misc_analyzers.py       # 7 structural/geometry analyzers
│   │   │   ├── color_substitution.py   # 1-to-1 color remapping
│   │   │   ├── translation.py          # Uniform object translation
│   │   │   ├── replication.py          # Arrow-driven template replication
│   │   │   └── pattern_analyzer_wrapper.py # Periodic pattern wrapper
│   │   ├── diagnostic_engine.py        # 3-Phase Diagnostic Engine
│   │   ├── solution_lookup.py          # External solver database lookup
│   │   ├── gitmonsters_loader.py       # AST-based safe solver loader
│   │   ├── hypothesis_tester.py        # 100% exact match verification engine
│   │   └── llm/                        # Phase 2 LLM Codegen Engine
│   │       ├── primitive_codegen.py    # Multi-provider LLM code generator
│   │       └── prompt_builder.py       # Structured diagnostic prompt builder
│   ├── solver/
│   │   ├── pipeline.py                 # Main entry: NeuroSymbolicARCSolver
│   │   └── specialized_solvers.py     # CategorySpecializedARCSolver (Beam search)
│   ├── dsl/                            # Domain-Specific Language primitives
│   └── neural/                         # Neural ranker & feature embeddings
├── external_solutions/
│   ├── solves/                         # 632 pre-computed solver scripts
│   └── dataset/                        # ARC task JSON files
├── run_batch_10.py                     # 10-task diagnostic benchmark script
├── run_batch_evaluation.py             # Main solver evaluation benchmark script
└── download_solved540_raw.py           # Parallel GitHub solver downloader
```

---

## 7. Verification & Operational Guidelines

- **Zero False-Positive Rule**: Never return a prediction unless `verify_100pct` confirms 100% exact match across all training pairs.
- **No Swallowing Errors**: Catch exceptions safely inside individual analyzers so one failing analyzer never crashes the parallel execution pool.
- **Git Synchronization**: Always commit and push changes to `GitMonsters/ARC-AGI-2` after modifying analyzers or pipeline code.
