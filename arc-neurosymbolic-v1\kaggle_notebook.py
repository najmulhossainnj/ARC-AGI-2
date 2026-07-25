# ARC Prize / Kaggle single-notebook entrypoint.
#
# Recommended:
# 1. Upload this repository as a Kaggle Dataset.
# 2. Attach the competition data.
# 3. Paste this file's contents into notebook cells, or run it as a script.
#
# The hidden rerun environment swaps arc-agi_test_challenges.json,
# so this code discovers it dynamically.

import os
import sys

REPO = "/kaggle/input/arc-neurosymbolic-v1-kaggle"
if os.path.isdir(REPO):
    sys.path.insert(0, REPO)

from arc_solver.utils.arc_io import (
    load_challenges,
    make_submission,
    write_submission,
    validate_submission,
)
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

def find_file(filename):
    matches = []
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            matches.append(os.path.join(root, filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under /kaggle/input"
        )

    # Prefer a path containing "arc" or "prize" if multiple datasets contain it.
    matches.sort(
        key=lambda p: (
            "arc" not in p.lower(),
            "prize" not in p.lower(),
            len(p),
        )
    )
    return matches[0]

challenge_path = find_file("arc-agi_test_challenges.json")

print("Challenge file:", challenge_path)

challenges = load_challenges(challenge_path)

solver = NeuroSymbolicARCSolver(
    beam_width=100,
    max_depth=3,
)

submission = make_submission(challenges, solver)
validate_submission(submission, challenges)

output_path = "/kaggle/working/submission.json"
write_submission(submission, output_path)

print("Tasks:", len(submission))
print("Submission:", output_path)

first_task_id = next(iter(submission))
print("First task:", first_task_id)
print("Number of test inputs:", len(submission[first_task_id]))
