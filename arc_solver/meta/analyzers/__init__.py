from .base import Analyzer, ProgramCandidate
from .color_substitution import ColorSubstitutionAnalyzer
from .translation import TranslationAnalyzer
from .replication import ArrowReplicateAnalyzer
from .misc_analyzers import (
    SymmetryCompleteAnalyzer,
    GravityFallAnalyzer,
    BorderCropAnalyzer,
    PatternExtensionAnalyzer,
    BlockCycleAnalyzer,
    ParallelogramAlignAnalyzer,
    DiagonalChainAnalyzer,
)
from .pattern_analyzer_wrapper import PatternAnalyzerWrapper
from .advanced_analyzers import (
    ConcentricRingFillAnalyzer,
    LegendRaySlideAnalyzer,
    NetworkConnectivityFillAnalyzer,
    PuzzleStitchAssemblyAnalyzer,
    MultiEdgeGravityAnalyzer,
    ObjectStampRuleAnalyzer,
    RayCollisionDeflectionAnalyzer,
    TArrowMarkerFlowAnalyzer,
    MasterTemplateInpaintAnalyzer,
)

ALL_ANALYZERS = [
    # ── Existing analyzers ──────────────────────────────────────────────────
    PatternAnalyzerWrapper(),       # priority=1  (runs first, high confidence)
    ColorSubstitutionAnalyzer(),    # priority=5
    TranslationAnalyzer(),          # priority=10
    GravityFallAnalyzer(),          # priority=12
    SymmetryCompleteAnalyzer(),     # priority=15
    BlockCycleAnalyzer(),           # priority=18
    PatternExtensionAnalyzer(),     # priority=22
    ArrowReplicateAnalyzer(),       # priority=20
    # ── New generalized analyzers (from GitMonsters pattern analysis) ────────
    MultiEdgeGravityAnalyzer(),     # priority=22 (generalizes GravityFallAnalyzer)
    NetworkConnectivityFillAnalyzer(), # priority=25
    LegendRaySlideAnalyzer(),       # priority=28
    ObjectStampRuleAnalyzer(),      # priority=28
    ConcentricRingFillAnalyzer(),   # priority=30
    TArrowMarkerFlowAnalyzer(),     # priority=32
    RayCollisionDeflectionAnalyzer(), # priority=35
    PuzzleStitchAssemblyAnalyzer(), # priority=35
    MasterTemplateInpaintAnalyzer(), # priority=40
    # ── Shape/geometry analyzers ─────────────────────────────────────────────
    ParallelogramAlignAnalyzer(),   # priority=28
    DiagonalChainAnalyzer(),        # priority=28
    BorderCropAnalyzer(),           # priority=25
]

ALL_ANALYZERS.sort(key=lambda a: a.priority)

__all__ = [
    "Analyzer", "ProgramCandidate", "ALL_ANALYZERS",
    # Original
    "ColorSubstitutionAnalyzer", "TranslationAnalyzer", "ArrowReplicateAnalyzer",
    "SymmetryCompleteAnalyzer", "GravityFallAnalyzer", "BorderCropAnalyzer",
    "PatternExtensionAnalyzer", "BlockCycleAnalyzer",
    "ParallelogramAlignAnalyzer", "DiagonalChainAnalyzer",
    "PatternAnalyzerWrapper",
    # New (from GitMonsters analysis)
    "ConcentricRingFillAnalyzer", "LegendRaySlideAnalyzer",
    "NetworkConnectivityFillAnalyzer", "PuzzleStitchAssemblyAnalyzer",
    "MultiEdgeGravityAnalyzer", "ObjectStampRuleAnalyzer",
    "RayCollisionDeflectionAnalyzer", "TArrowMarkerFlowAnalyzer",
    "MasterTemplateInpaintAnalyzer",
]
