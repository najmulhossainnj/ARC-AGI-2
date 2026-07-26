from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
from .grid import as_grid
from ..perception.extractor import extract_objects
from ..relations.builder import build_relation_graph

@dataclass
class Scene:
    grid: np.ndarray
    objects: list
    graph: object
    color_counts: dict
    background: int = 0

    @classmethod
    def from_grid(cls, grid, background=0, diagonal=False):
        g = as_grid(grid)
        objects = extract_objects(g, background=background, diagonal=diagonal)
        graph = build_relation_graph(g, objects)
        return cls(g, objects, graph, dict(Counter(g.flatten())), background)
