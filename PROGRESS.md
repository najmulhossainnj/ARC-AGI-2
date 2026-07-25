# ARC Neuro-Symbolic Solver — Task & Progress Tracker

## Core Methodology & Development Principle

1. **5-Task Batch Validation**: Test the solver against a batch of 5 evaluation tasks.
2. **Empirical Failure Diagnosis**: If a task fails (`MISS`), dump and visually inspect the exact grid inputs, expected outputs, and intermediate solver steps.
3. **Targeted Enhancement**: Identify whether the failure is due to:
   - Rigid global assumptions in an existing solver family (generalize it to be regional or search offset parameters).
   - Missing primitive operations (build the specific transform/combinator).
4. **Iterative Verification**: Apply findings to solve the target tasks, re-verify on the 5-task batch, and expand to larger evaluation runs.

---

## Directory Structure & Component Map

```text
e:/kaggle/antigravity/
├── ARCHITECTURE.md                  # Comprehensive architecture & module reference
├── PROGRESS.md                      # Workspace copy of live tracker
├── Validating task families on eval set.json # Prior conversation transcript & diagnostic logs
├── data/
│   └── arc-prize-2026-arc-agi-2/   # ARC Prize consolidated dataset
│       ├── arc-agi_evaluation_challenges.json
│       ├── arc-agi_evaluation_solutions.json
│       ├── arc-agi_test_challenges.json
│       ├── arc-agi_training_challenges.json
│       ├── arc-agi_training_solutions.json
│       └── sample_submission.json
└── arc-neurosymbolic-v1/            # Solver source code & package root
    ├── README.md                    # Setup and basic usage guide
    ├── pyproject.toml / requirements.txt
    ├── kaggle_notebook.py           # Standalone single-file Kaggle entrypoint
    ├── configs/
    │   └── default.yaml             # Solver hyperparameters (beam width, max depth)
    ├── scripts/
    │   ├── validate.py              # Local validation script over evaluation dataset
    │   ├── kaggle_solve.py          # Generate submission.json for test challenges
    │   ├── solve.py                 # Single-task solver CLI
    │   ├── make_submission.py       # Package submission
    │   └── train_ranker.py          # Neural candidate ranker training script
    ├── tests/                       # Pytest suite for perception, dsl, search, solver
    └── arc_solver/                  # Core solver package
        ├── core/                    # Grid data structures, Scene representation, ARCObject
        ├── perception/              # Connected-component object extraction (background/diagonal)
        ├── relations/               # Spatial relation graph builder & predicates
        ├── correspondence/          # Object matching across train pairs, delta inference
        ├── dsl/                     # AST, primitives, transforms, advanced transforms, object ops
        ├── synthesis/               # Grammar definitions, parameter learning, beam search engine
        ├── verification/            # Exact match verifier, MDL scorer, consistency checks
        ├── neural/                  # Optional MLP neural candidate ranker
        ├── analysis/                # Task signature & decomposition tools
        ├── utils/                   # IO, visualization, logging
        └── solver/                  # Pipeline integration (NeuroSymbolicARCSolver)
```

---

## Progress Log & Completed Tasks

- [x] **Initial Repository Setup & Architecture Reference**:
  - Analyzed and documented whole codebase pipeline in `ARCHITECTURE.md`.
  - Verified package layout, dependencies (NumPy, PyYAML, pytest), and Kaggle submission formatting.
- [x] **First 5-Task Validation & Failure Analysis (`0934a4d8`, `135a2760`, `136b0064`, `13e47133`, `142ca369`)**:
  - Ran `NeuroSymbolicARCSolver` (beam_width=100, max_depth=3) on the 5 evaluation tasks.
  - Result: 0/5 solved (`MISS`).
  - Inspected exact grid data and diagnosed specific failure modes for each task:
    - **`0934a4d8`**: `symmetry_repair` assumes center-aligned symmetry axis; grid axis is offset. Lacks `CROP_TO_CHANGED_REGION`.
    - **`135a2760`**: Multi-panel grid with local periods. `pattern_complete` assumes single global period.
    - **`142ca369`**: Missing diagonal ray casting primitive with boundary reflection.
    - **`13e47133`**: Missing seeded concentric/nested shape growth primitive.
    - **`136b0064`**: Missing icon decoding & path tracing primitive.

---

## Action Plan & Work Backlog

### Phase 1: High-Priority Fixes for First 5 Eval Tasks
- [ ] **Task `135a2760` — `APPLY_PER_REGION` Combinator**:
  - Implement panel/region segmentation by separator lines or background gaps.
  - Apply `pattern_complete` independently per region.
- [ ] **Task `0934a4d8` — Generalized Symmetry & Bounding Box Crop**:
  - Generalize `symmetry_repair` in `advanced_transforms.py` to evaluate offset mirror axes.
  - Add `CROP_TO_CHANGED_REGION` / `CROP_TO_MASK` op to `object_ops.py`.
- [ ] **Task `142ca369` — Ray Casting Primitive**:
  - Implement diagonal line/ray casting primitive from seed objects with wall reflection.
  - Wire parameter learner `learn_ray_cast` into `grammar.py` and `param_learning.py`.
- [ ] **Task `13e47133` — Seeded Concentric Rectangular Growth**:
  - Implement primitive for concentric shape expansion from marker seeds using palette colors.
  - Add parameter inference for growth rules.
- [ ] **Task `136b0064` — Icon / Marker Path Decoding**:
  - Implement icon-to-path transform primitive.

### Phase 2: Re-validation & Next 5-Task Iteration Cycle
- [ ] Re-run validation on the first 5 eval tasks to verify 5/5 or improved solve rate.
- [ ] Select next batch of 5 eval tasks (`140c817e`, `1436024a`, etc.).
- [ ] Analyze failure modes on missing tasks and implement required DSL primitives.
- [ ] Run full 120-task validation via `python scripts/validate.py`.
