"""
relational_reasoner.py
----------------------
Relational Graph Reasoning Engine for ARC tasks using NetworkX.
Constructs directed graphs representing entity relationships (spatial, topological, structural).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import networkx as nx
from ..perception.multiview import GridEntity, PerceptionView, MultiViewPerception


class ARCRelationalGraph:
    """NetworkX directed graph representing grid entities and their relations."""

    def __init__(self, perception_view: PerceptionView):
        self.view = perception_view
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        entities = self.view.entities_4way
        
        # 1. Add Nodes
        for e in entities:
            self.graph.add_node(
                e.entity_id,
                color=e.color,
                area=e.area,
                bbox=e.bbox,
                centroid=e.centroid,
                aspect_ratio=e.aspect_ratio,
                is_solid=e.is_solid,
                is_line=e.is_line,
                is_frame=e.is_frame,
            )

        # 2. Add Relational Edges
        for i, e1 in enumerate(entities):
            r1_1, r2_1, c1_1, c2_1 = e1.bbox
            cr1, cc1 = e1.centroid
            for j, e2 in enumerate(entities):
                if i == j:
                    continue
                r1_2, r2_2, c1_2, c2_2 = e2.bbox
                cr2, cc2 = e2.centroid

                # Spatial relations
                if r2_1 < r1_2:
                    self.graph.add_edge(e1.entity_id, e2.entity_id, relation="ABOVE")
                elif r1_1 > r2_2:
                    self.graph.add_edge(e1.entity_id, e2.entity_id, relation="BELOW")

                if c2_1 < c1_2:
                    self.graph.add_edge(e1.entity_id, e2.entity_id, relation="LEFT_OF")
                elif c1_1 > c2_2:
                    self.graph.add_edge(e1.entity_id, e2.entity_id, relation="RIGHT_OF")

                # Topological relations (INSIDE / CONTAINED)
                if r1_1 <= r1_2 and r2_1 >= r2_2 and c1_1 <= c1_2 and c2_1 >= c2_2:
                    self.graph.add_edge(e2.entity_id, e1.entity_id, relation="INSIDE")

                # Adjacency / TOUCHING
                if cls_check_touching(e1.bbox, e2.bbox):
                    self.graph.add_edge(e1.entity_id, e2.entity_id, relation="TOUCHING")

                # Structural / Property equivalence
                if e1.color == e2.color:
                    self.graph.add_edge(e1.entity_id, e2.entity_id, relation="SAME_COLOR")
                if e1.area == e2.area:
                    self.graph.add_edge(e1.entity_id, e2.entity_id, relation="SAME_AREA")


def cls_check_touching(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
    r1_1, r2_1, c1_1, c2_1 = b1
    r1_2, r2_2, c1_2, c2_2 = b2
    row_touch = (r2_1 + 1 == r1_2) or (r2_2 + 1 == r1_1) or not (r2_1 < r1_2 or r1_1 > r2_2)
    col_touch = (c2_1 + 1 == c1_2) or (c2_2 + 1 == c1_1) or not (c2_1 < c1_2 or c1_1 > c2_2)
    adjacent = (r2_1 + 1 == r1_2 or r2_2 + 1 == r1_1) and not (c2_1 < c1_2 or c1_1 > c2_2)
    adjacent_col = (c2_1 + 1 == c1_2 or c2_2 + 1 == c1_1) and not (r2_1 < r1_2 or r1_1 > r2_2)
    return bool(adjacent or adjacent_col)


class RelationalGraphReasoner:
    """Analyzes relational graphs across train pairs."""

    @classmethod
    def analyze_pairs(cls, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict[str, Any]]:
        graph_transformations = []
        for inp, out in train_pairs:
            p_in = MultiViewPerception.analyze_grid(inp)
            p_out = MultiViewPerception.analyze_grid(out)

            g_in = ARCRelationalGraph(p_in)
            g_out = ARCRelationalGraph(p_out)

            transform_info = cls._compare_graphs(g_in.graph, g_out.graph)
            graph_transformations.append(transform_info)

        return graph_transformations

    @classmethod
    def _compare_graphs(cls, g_in: nx.DiGraph, g_out: nx.DiGraph) -> Dict[str, Any]:
        node_diff = len(g_out.nodes) - len(g_in.nodes)
        edge_diff = len(g_out.edges) - len(g_in.edges)

        in_colors = set(nx.get_node_attributes(g_in, "color").values())
        out_colors = set(nx.get_node_attributes(g_out, "color").values())

        return {
            "node_count_change": node_diff,
            "edge_count_change": edge_diff,
            "added_colors": list(out_colors - in_colors),
            "removed_colors": list(in_colors - out_colors),
            "is_isomorphic": nx.is_isomorphic(g_in, g_out) if len(g_in) == len(g_out) else False,
        }
