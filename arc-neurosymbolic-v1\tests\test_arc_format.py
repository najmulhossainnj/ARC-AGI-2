import json
from arc_solver.utils.arc_io import load_challenges, validate_submission

def test_multiple_test_inputs(tmp_path):
    path = tmp_path / "challenges.json"
    data = {
        "task_x": {
            "train": [
                {"input": [[1]], "output": [[2]]}
            ],
            "test": [
                {"input": [[1]]},
                {"input": [[2]]}
            ],
        }
    }
    path.write_text(json.dumps(data))
    tasks = load_challenges(path)

    assert list(tasks) == ["task_x"]
    assert len(tasks["task_x"].train) == 1
    assert len(tasks["task_x"].test) == 2

def test_submission_schema():
    submission = {
        "task_x": [
            {
                "attempt_1": [[0, 1]],
                "attempt_2": [[1, 0]],
            },
            {
                "attempt_1": [[2]],
                "attempt_2": [[2]],
            },
        ]
    }
    assert validate_submission(submission)
