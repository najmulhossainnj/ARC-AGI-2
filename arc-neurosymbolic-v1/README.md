# ARC Neuro-Symbolic Solver v1 — Kaggle/ARC Prize Fixed Version

This version uses the consolidated ARC Prize JSON format.

## Dataset files

Attach the competition dataset containing:

- `arc-agi_training-challenges.json`
- `arc-agi_training-solutions.json`
- `arc-agi_evaluation-challenges.json`
- `arc-agi_evaluation-solutions.json`
- `arc-agi_test-challenges.json`
- `sample_submission.json`

## Challenge format

```json
{
  "task_id": {
    "train": [
      {
        "input": [[0,1]],
        "output": [[1,0]]
      }
    ],
    "test": [
      {
        "input": [[1,0]]
      }
    ]
  }
}
```

`test` is a list because a small number of tasks have two test inputs.

## Submission format

The solver generates:

```json
{
  "task_id": [
    {
      "attempt_1": [[...]],
      "attempt_2": [[...]]
    }
  ]
}
```

For a task with two test inputs:

```json
{
  "task_id": [
    {
      "attempt_1": [[...]],
      "attempt_2": [[...]]
    },
    {
      "attempt_1": [[...]],
      "attempt_2": [[...]]
    }
  ]
}
```

The order is identical to the order of `test` in the challenge file.

## Local evaluation

```bash
python scripts/validate.py \
  --challenges /path/arc-agi_evaluation_challenges.json \
  --solutions /path/arc-agi_evaluation-solutions.json
```

## Generate test submission

```bash
python scripts/kaggle_solve.py \
  --challenges /path/arc-agi_test_challenges.json \
  --output submission.json
```

## Kaggle

`kaggle_notebook.py` is the single-entrypoint version. It:

1. Finds `arc-agi_test_challenges.json` automatically.
2. Loads all task IDs.
3. Preserves multiple test inputs.
4. Runs the solver independently on every task.
5. Produces exactly two attempts for every test input.
6. Validates grid shape and color values.
7. Writes `/kaggle/working/submission.json`.

## Architecture

- Connected-component object extraction
- Object representation
- Relation graph
- Object correspondence
- Transformation DSL
- Program grammar
- Beam search
- Exact training verification
- Candidate diversity
- Two-attempt prediction

## Important limitation

This is a corrected **data-format/Kaggle integration v1**. The solver's DSL is still a foundation and does not yet cover the full ARC transformation space. In particular, the next major improvement should add object-level program synthesis:

- `SELECT`
- `FILTER`
- `ARGMAX` / `ARGMIN`
- `MOVE`
- `COPY`
- `DELETE`
- `RECOLOR_OBJECT`
- object rotation/reflection
- relational predicates
- object grouping
- repeated transformations
- compositional programs
- neural candidate ranking

Those additions are expected to improve ARC solve rate substantially beyond the current primitive transformation baseline.
