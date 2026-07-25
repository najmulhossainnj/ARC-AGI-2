from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple
import numpy as np


@dataclass
class ProgramCandidate:
    op: str
    params: tuple = field(default_factory=tuple)
    description: str = ""

    def to_instruction_args(self):
        return self.params

    def __repr__(self):
        return f"ProgramCandidate(op={self.op!r}, params={self.params!r})"


class Analyzer(ABC):
    name: str = "base"
    priority: int = 50

    @abstractmethod
    def analyze(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        features: dict,
    ) -> Optional[ProgramCandidate]:
        """Return a ProgramCandidate if pattern detected, else None."""
        pass
