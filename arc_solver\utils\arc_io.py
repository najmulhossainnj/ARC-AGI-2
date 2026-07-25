import json
from pathlib import Path
from ..core.task import ARCTask

def load_challenges(path):
    """Load consolidated ARC challenge JSON into {task_id: ARCTask}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        task_id: ARCTask.from_challenge(task_id, task_data)
        for task_id, task_data in raw.items()
    }

def load_solutions(path):
    """Load consolidated ARC solution JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_submission(challenges, solver):
    """
    Produce the ARC Prize submission schema.

    Each task maps to a list of test predictions.
    Each test prediction contains exactly two attempts.
    """
    submission = {}

    for task_id, task in challenges.items():
        train_pairs = [(p.input, p.output) for p in task.train]
        predictions, _ = solver.solve_task(train_pairs, task.test)

        submission[task_id] = [
            {
                "attempt_1": choices[0],
                "attempt_2": choices[1],
            }
            for choices in predictions
        ]

    return submission

def validate_submission(submission, challenges=None):
    """Validate basic ARC submission structure and rectangular grids."""
    if not isinstance(submission, dict):
        raise ValueError("Submission must be a JSON object.")

    if challenges is not None:
        missing = set(challenges) - set(submission)
        extra = set(submission) - set(challenges)
        if missing:
            raise ValueError(f"Missing task IDs: {sorted(missing)[:5]}")
        if extra:
            raise ValueError(f"Unexpected task IDs: {sorted(extra)[:5]}")

    for task_id, test_predictions in submission.items():
        if not isinstance(test_predictions, list):
            raise ValueError(f"{task_id}: predictions must be a list.")

        for i, prediction in enumerate(test_predictions):
            if not isinstance(prediction, dict):
                raise ValueError(f"{task_id} test {i}: prediction must be an object.")
            if set(prediction) != {"attempt_1", "attempt_2"}:
                raise ValueError(
                    f"{task_id} test {i}: expected attempt_1 and attempt_2."
                )

            for attempt_name in ("attempt_1", "attempt_2"):
                grid = prediction[attempt_name]
                if not isinstance(grid, list) or not grid:
                    raise ValueError(f"{task_id} test {i} {attempt_name}: invalid grid.")
                width = len(grid[0])
                if width == 0:
                    raise ValueError(f"{task_id} test {i} {attempt_name}: empty row.")
                if any(not isinstance(row, list) or len(row) != width for row in grid):
                    raise ValueError(
                        f"{task_id} test {i} {attempt_name}: non-rectangular grid."
                    )
                for row in grid:
                    for cell in row:
                        if not isinstance(cell, int) or not 0 <= cell <= 9:
                            raise ValueError(
                                f"{task_id} test {i} {attempt_name}: "
                                f"cells must be integers 0..9."
                            )
    return True

def write_submission(submission, path):
    validate_submission(submission)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(submission, f, separators=(",", ":"))
