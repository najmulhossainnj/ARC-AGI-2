"""
induction.py
------------
Symbolic Concept Induction Engine for ARC.
Discovers entity selection predicates and relational invariants across training pairs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from ..perception.multiview import MultiViewPerception, GridEntity, PerceptionView


@dataclass
class InductiveConcept:
    """Represents a discovered invariant concept across training pairs."""
    concept_type: str        # e.g., 'ENTITY_SELECTION', 'RELATIONAL', 'INVARIANT'
    name: str                # e.g., 'SMALLEST', 'UNIQUE_COLOR', 'SAME_SHAPE'
    confidence: float        # 1.0 if holds across 100% of training pairs
    params: Tuple[Any, ...] = ()
    description: str = ""


class ConceptInductionEngine:
    """Discovers concepts and invariants across train pairs."""

    @classmethod
    def induce_concepts(
        cls, train_pairs: List[Tuple[np.ndarray, np.ndarray]], bg_color: Optional[int] = None
    ) -> List[InductiveConcept]:
        concepts = []

        # Analyze perception views for all train pairs
        views_in = [MultiViewPerception.analyze_grid(p[0], bg_color=bg_color) for p in train_pairs]
        views_out = [MultiViewPerception.analyze_grid(p[1], bg_color=bg_color) for p in train_pairs]

        # 1. Test Entity Selection Predicates
        cls._induce_entity_selection_predicates(views_in, concepts)

        # 2. Test Invariants (e.g. preserved shapes, colors, dimensions)
        cls._induce_invariants(views_in, views_out, concepts)

        # 3. Test Global Layout Concepts (Dividers, Symmetries)
        cls._induce_layout_concepts(views_in, views_out, concepts)

        return concepts

    @classmethod
    def _induce_entity_selection_predicates(
        cls, views_in: List[PerceptionView], concepts: List[InductiveConcept]
    ) -> None:
        # Check SMALLEST, LARGEST, UNIQUE_COLOR consistency
        predicates = ["SMALLEST", "LARGEST", "UNIQUE_COLOR", "MOST_COMMON_COLOR", "SINGLE_CELL_DOTS"]

        for pred in predicates:
            valid = True
            for view in views_in:
                entities = view.entities_4way
                if not entities:
                    valid = False
                    break

                if pred == "SMALLEST":
                    min_area = min(e.area for e in entities)
                    smallest_count = sum(1 for e in entities if e.area == min_area)
                    if smallest_count != 1:
                        valid = False

                elif pred == "LARGEST":
                    max_area = max(e.area for e in entities)
                    largest_count = sum(1 for e in entities if e.area == max_area)
                    if largest_count != 1:
                        valid = False

                elif pred == "UNIQUE_COLOR":
                    # Check if there is an entity with a color that appears exactly once
                    color_counts = view.color_histogram
                    unique_c = [c for c, count in color_counts.items() if count == 1 and c != view.bg_color]
                    if not unique_c:
                        valid = False

                elif pred == "SINGLE_CELL_DOTS":
                    single_dots = [e for e in entities if e.area == 1]
                    if not single_dots:
                        valid = False

            if valid:
                concepts.append(
                    InductiveConcept(
                        concept_type="ENTITY_SELECTION",
                        name=pred,
                        confidence=1.0,
                        description=f"Entity selection predicate {pred} holds across all training inputs",
                    )
                )

    @classmethod
    def _induce_invariants(
        cls, views_in: List[PerceptionView], views_out: List[PerceptionView], concepts: List[InductiveConcept]
    ) -> None:
        # Check if output grid shape is invariant relative to input
        same_shape = all(vin.grid_shape == vout.grid_shape for vin, vout in zip(views_in, views_out))
        if same_shape:
            concepts.append(
                InductiveConcept(
                    concept_type="INVARIANT",
                    name="SAME_GRID_SHAPE",
                    confidence=1.0,
                    description="Output grid dimensions match input grid dimensions",
                )
            )

        # Check if output preserves background color
        same_bg = all(vin.bg_color == vout.bg_color for vin, vout in zip(views_in, views_out))
        if same_bg:
            concepts.append(
                InductiveConcept(
                    concept_type="INVARIANT",
                    name="SAME_BACKGROUND_COLOR",
                    confidence=1.0,
                    params=(views_in[0].bg_color,),
                    description=f"Background color is invariant (color={views_in[0].bg_color})",
                )
            )

    @classmethod
    def _induce_layout_concepts(
        cls, views_in: List[PerceptionView], views_out: List[PerceptionView], concepts: List[InductiveConcept]
    ) -> None:
        divided_in = all(vin.is_grid_divided for vin in views_in)
        if divided_in:
            concepts.append(
                InductiveConcept(
                    concept_type="LAYOUT",
                    name="LATTICE_DIVIDED_GRID",
                    confidence=1.0,
                    description="Input grids contain solid divider lines dividing sections",
                )
            )

        sym_h = all(vin.symmetries["horizontal"] for vin in views_in)
        if sym_h:
            concepts.append(
                InductiveConcept(
                    concept_type="LAYOUT",
                    name="HORIZONTAL_SYMMETRY",
                    confidence=1.0,
                    description="Input grids possess horizontal mirror symmetry",
                )
            )
