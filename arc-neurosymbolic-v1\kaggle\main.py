# Kaggle entrypoint example.
# Adjust DATA_DIR to the mounted ARC competition dataset.
import os
from arc_solver.utils.io import load_tasks
from arc_solver.solver.pipeline import NeuroSymbolicARCSolver

DATA_DIR=os.environ.get("ARC_DATA_DIR","/kaggle/input/arc-prize-2024")
solver=NeuroSymbolicARCSolver(beam_width=100,max_depth=5)
# Use scripts/make_submission.py or adapt this entrypoint to the exact competition format.
