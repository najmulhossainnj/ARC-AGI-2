from __future__ import annotations
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

@dataclass(frozen=True)
class ARCObject:
    id: int
    cells: tuple[tuple[int, int], ...]
    colors: tuple[int, ...]
    bbox: tuple[int, int, int, int]
    component_id: int = 0

    @property
    def size(self): return len(self.cells)
    @property
    def top(self): return self.bbox[0]
    @property
    def left(self): return self.bbox[1]
    @property
    def bottom(self): return self.bbox[2]
    @property
    def right(self): return self.bbox[3]
    @property
    def height(self): return self.bottom - self.top + 1
    @property
    def width(self): return self.right - self.left + 1
    @property
    def center(self):
        return ((self.top+self.bottom)/2, (self.left+self.right)/2)

    def normalized_cells(self):
        return frozenset((r-self.top, c-self.left) for r,c in self.cells)

    def shape_signature(self):
        return (self.height, self.width, self.size, self.normalized_cells())

    def color_histogram(self):
        return Counter(self.colors)

@dataclass
class ObjectChange:
    input_id: int | None
    output_id: int | None
    kind: str
    score: float
    details: dict = field(default_factory=dict)
