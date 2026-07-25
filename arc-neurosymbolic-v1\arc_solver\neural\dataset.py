from dataclasses import dataclass
@dataclass
class RankExample:
    task_features: object
    program_features: object
    label: int
