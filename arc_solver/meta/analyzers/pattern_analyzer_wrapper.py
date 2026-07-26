"""
Pattern Analyzer Wrapper - Wraps the rule-based pattern analyzer as an Analyzer.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict

from .base import Analyzer, ProgramCandidate
from .pattern_analyzer import analyze_transformation


class PatternAnalyzerWrapper(Analyzer):
    """
    Wrapper for the rule-based pattern analyzer.
    
    This analyzer detects common transformation patterns directly from
    input-output pairs using pure rule-based logic:
    - Color mapping/substitution
    - Rotation (90, 180, 270 degrees)
    - Reflection (horizontal, vertical, diagonal)
    - Replication/tiling
    - Symmetry operations
    - Border operations
    - Pixel replacement patterns
    """
    
    def __init__(self):
        self.name = "PatternAnalyzer"
        self.priority = 100  # High priority - runs early
    
    def analyze(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        features: Optional[Dict] = None,
    ) -> Optional[ProgramCandidate]:
        """
        Analyze training pairs and return a ProgramCandidate if pattern detected.
        """
        solve_fn = analyze_transformation(train_pairs)
        
        if solve_fn is None:
            return None
        
        # Get pattern info
        pattern_name = getattr(solve_fn, '_pattern', 'unknown')
        
        # Create a synthetic op name for this pattern
        op_name = f"PATTERN_{pattern_name.upper().replace(':', '_').replace(' ', '_')}"
        
        # Create ProgramCandidate
        candidate = ProgramCandidate(
            op=op_name,
            params=(),
            description=f"Pattern: {pattern_name}",
        )
        
        # Attach the solve function for later use
        candidate.solve_fn = solve_fn
        
        print(f"  [PatternAnalyzer] Detected pattern: {pattern_name}")
        
        return candidate
