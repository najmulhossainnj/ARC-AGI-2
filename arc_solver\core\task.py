from dataclasses import dataclass
import numpy as np

@dataclass
class TaskPair:
    input: np.ndarray
    output: np.ndarray

@dataclass
class ARCTask:
    task_id: str
    train: list[TaskPair]
    test: list[np.ndarray]

    @classmethod
    def from_challenge(cls, task_id, data):
        train = [
            TaskPair(
                np.asarray(pair["input"], dtype=np.int16),
                np.asarray(pair["output"], dtype=np.int16),
            )
            for pair in data.get("train", [])
        ]
        test = [
            np.asarray(pair["input"], dtype=np.int16)
            for pair in data.get("test", [])
        ]
        return cls(task_id=task_id, train=train, test=test)
