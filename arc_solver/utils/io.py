import json
from pathlib import Path
from ..core.task import ARCTask

def load_tasks(path):
    path=Path(path)
    tasks={}
    for p in sorted(path.glob("*.json")):
        with open(p) as f:
            tasks[p.stem]=ARCTask.from_dict(p.stem,json.load(f))
    return tasks

def load_single(path):
    with open(path) as f:
        data=json.load(f)
    return ARCTask.from_dict(Path(path).stem,data)
