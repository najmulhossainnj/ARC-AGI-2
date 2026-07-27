"""
multiview.py
-------------
Multi-View Perception Engine for ARC Grids.
Extracts 6 parallel perception views (Pixel, Connected Component, Color Region, Contour/Geometry, Symmetry, Hierarchical).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.ndimage import label


@dataclass
class GridEntity:
    """Represents an extracted entity/object in the grid."""
    entity_id: int
    color: int
    mask: np.ndarray          # Boolean mask matching grid shape
    bbox: Tuple[int, int, int, int]  # (r_min, r_max, c_min, c_max)
    area: int
    centroid: Tuple[float, float]
    aspect_ratio: float
    is_solid: bool = False
    is_line: bool = False
    is_frame: bool = False
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)


@dataclass
class PerceptionView:
    """Container for multi-view perception results."""
    grid_shape: Tuple[int, int]
    bg_color: int
    unique_colors: List[int]
    color_histogram: Dict[int, int]
    entities_4way: List[GridEntity]
    entities_8way: List[GridEntity]
    color_entities: Dict[int, GridEntity]
    symmetries: Dict[str, bool]
    is_grid_divided: bool
    divider_rows: List[int]
    divider_cols: List[int]


class MultiViewPerception:
    """Computes multi-view perception for ARC grids."""

    @classmethod
    def analyze_grid(cls, grid: np.ndarray, bg_color: Optional[int] = None) -> PerceptionView:
        g = np.asarray(grid, dtype=int)
        h, w = g.shape

        # 1. Pixel & Color View
        flat = g.flatten()
        counts = np.bincount(flat, minlength=10)
        unique_colors = [int(c) for c in np.unique(g)]
        
        if bg_color is None:
            bg_color = int(np.argmax(counts))

        color_hist = {int(c): int(counts[c]) for c in unique_colors}

        # 2. Connected Component Views (4-way and 8-way)
        entities_4way = cls._extract_components(g, bg_color, connectivity=4)
        entities_8way = cls._extract_components(g, bg_color, connectivity=8)

        # 3. Color Segmentation View (per-color aggregated entities)
        color_entities = cls._extract_color_entities(g, bg_color, unique_colors)

        # 4. Symmetry View
        symmetries = cls._analyze_symmetry(g)

        # 5. Grid Division / Divider View
        is_divided, div_rows, div_cols = cls._analyze_dividers(g, bg_color)

        # 6. Hierarchical Object View (compute parent/children nesting)
        cls._compute_nesting(entities_4way)

        return PerceptionView(
            grid_shape=(h, w),
            bg_color=bg_color,
            unique_colors=unique_colors,
            color_histogram=color_hist,
            entities_4way=entities_4way,
            entities_8way=entities_8way,
            color_entities=color_entities,
            symmetries=symmetries,
            is_grid_divided=is_divided,
            divider_rows=div_rows,
            divider_cols=div_cols,
        )

    @classmethod
    def _extract_components(cls, grid: np.ndarray, bg_color: int, connectivity: int) -> List[GridEntity]:
        h, w = grid.shape
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int) if connectivity == 4 else np.ones((3, 3), dtype=int)
        entities = []
        entity_counter = 0

        for color in np.unique(grid):
            if color == bg_color:
                continue
            mask = (grid == color)
            lbl, num = label(mask, structure=struct)
            for i in range(1, num + 1):
                comp_mask = (lbl == i)
                rs, cs = np.where(comp_mask)
                r1, r2 = int(rs.min()), int(rs.max())
                c1, c2 = int(cs.min()), int(cs.max())
                area = int(comp_mask.sum())
                box_h = r2 - r1 + 1
                box_w = c2 - c1 + 1
                aspect_ratio = float(box_w / box_h) if box_h > 0 else 1.0

                is_solid = (area == box_h * box_w)
                is_line = (box_h == 1 or box_w == 1)
                is_frame = cls._check_is_frame(comp_mask, r1, r2, c1, c2)

                entity = GridEntity(
                    entity_id=entity_counter,
                    color=int(color),
                    mask=comp_mask,
                    bbox=(r1, r2, c1, c2),
                    area=area,
                    centroid=(float(rs.mean()), float(cs.mean())),
                    aspect_ratio=aspect_ratio,
                    is_solid=is_solid,
                    is_line=is_line,
                    is_frame=is_frame,
                )
                entities.append(entity)
                entity_counter += 1

        return entities

    @classmethod
    def _extract_color_entities(cls, grid: np.ndarray, bg_color: int, colors: List[int]) -> Dict[int, GridEntity]:
        color_entities = {}
        for c in colors:
            if c == bg_color:
                continue
            mask = (grid == c)
            rs, cs = np.where(mask)
            if len(rs) == 0:
                continue
            r1, r2 = int(rs.min()), int(rs.max())
            c1, c2 = int(cs.min()), int(cs.max())
            area = int(mask.sum())
            box_h = r2 - r1 + 1
            box_w = c2 - c1 + 1
            aspect_ratio = float(box_w / box_h) if box_h > 0 else 1.0

            color_entities[c] = GridEntity(
                entity_id=c,
                color=c,
                mask=mask,
                bbox=(r1, r2, c1, c2),
                area=area,
                centroid=(float(rs.mean()), float(cs.mean())),
                aspect_ratio=aspect_ratio,
                is_solid=(area == box_h * box_w),
                is_line=(box_h == 1 or box_w == 1),
            )
        return color_entities

    @classmethod
    def _analyze_symmetry(cls, grid: np.ndarray) -> Dict[str, bool]:
        return {
            "horizontal": bool(np.array_equal(grid, np.flipud(grid))),
            "vertical": bool(np.array_equal(grid, np.fliplr(grid))),
            "diagonal": bool(grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T)),
            "rotational_180": bool(np.array_equal(grid, np.rot90(grid, 2))),
        }

    @classmethod
    def _analyze_dividers(cls, grid: np.ndarray, bg_color: int) -> Tuple[bool, List[int], List[int]]:
        h, w = grid.shape
        full_rows = []
        full_cols = []
        
        for r in range(h):
            row_vals = np.unique(grid[r, :])
            if len(row_vals) == 1 and row_vals[0] != bg_color:
                full_rows.append(r)
                
        for c in range(w):
            col_vals = np.unique(grid[:, c])
            if len(col_vals) == 1 and col_vals[0] != bg_color:
                full_cols.append(c)

        is_divided = bool(full_rows or full_cols)
        return is_divided, full_rows, full_cols

    @classmethod
    def _check_is_frame(cls, mask: np.ndarray, r1: int, r2: int, c1: int, c2: int) -> bool:
        box_h = r2 - r1 + 1
        box_w = c2 - c1 + 1
        if box_h < 3 or box_w < 3:
            return False
        # Check border perimeter is filled but inner center has holes
        top_edge = mask[r1, c1:c2+1].all()
        bottom_edge = mask[r2, c1:c2+1].all()
        left_edge = mask[r1:r2+1, c1].all()
        right_edge = mask[r1:r2+1, c2].all()
        center_holes = (~mask[r1+1:r2, c1+1:c2]).any()
        return bool(top_edge and bottom_edge and left_edge and right_edge and center_holes)

    @classmethod
    def _compute_nesting(cls, entities: List[GridEntity]) -> None:
        for i, e1 in enumerate(entities):
            r1_1, r2_1, c1_1, c2_1 = e1.bbox
            for j, e2 in enumerate(entities):
                if i == j:
                    continue
                r1_2, r2_2, c1_2, c2_2 = e2.bbox
                if r1_1 <= r1_2 and r2_1 >= r2_2 and c1_1 <= c1_2 and c2_1 >= c2_2:
                    # e2 is inside e1
                    e2.parent_id = e1.entity_id
                    if e2.entity_id not in e1.children_ids:
                        e1.children_ids.append(e2.entity_id)
