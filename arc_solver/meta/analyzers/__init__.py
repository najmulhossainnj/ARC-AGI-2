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
from .corpus_analyzers import (
    CountDrivenRuleAnalyzer,
    SortByAttributeAnalyzer,
    SizeSelectionAnalyzer,
    TopologyHoleAnalyzer,
    GridSectionLegendAnalyzer,
    DiagonalPatternAnalyzer,
    UniqueObjectExtractorAnalyzer,
    ColorIndexedTilingAnalyzer,
    PanelBooleanLogicAnalyzer,
    AlternatingFlipTilingAnalyzer,
    AnomalyRepairAnalyzer,
    FrameSizeToFillColorAnalyzer,
)

# ── Single canonical ALL_ANALYZERS list ───────────────────────────────────────
# Priority ordering: lower = faster/cheaper/more specific (runs first)
ALL_ANALYZERS = [
    # ── Fast / high-precision filters ──────────────────────────────────────
    PatternAnalyzerWrapper(),           # priority=1   (periodic pattern, very fast)
    ColorSubstitutionAnalyzer(),        # priority=5   (1-to-1 color remap)
    SizeSelectionAnalyzer(),            # priority=8   (keep largest/smallest)
    AnomalyRepairAnalyzer(),            # priority=10  (repair single-cell anomaly)
    TranslationAnalyzer(),              # priority=10  (uniform shift)
    GravityFallAnalyzer(),              # priority=12  (objects fall to edge)
    AlternatingFlipTilingAnalyzer(),   # priority=15  (alternating flip tile expansion)
    SymmetryCompleteAnalyzer(),         # priority=15  (H/V/HV symmetry repair)
    UniqueObjectExtractorAnalyzer(),    # priority=16  (odd-one-out)
    ColorIndexedTilingAnalyzer(),       # priority=18  (tile small pattern)
    PanelBooleanLogicAnalyzer(),        # priority=18  (panel bitwise logic/AND/XOR)
    BlockCycleAnalyzer(),               # priority=18  (vertical block cycling)
    GridSectionLegendAnalyzer(),        # priority=20  (separator → key+puzzle)
    FrameSizeToFillColorAnalyzer(),     # priority=20  (frame size → fill color)
    ArrowReplicateAnalyzer(),           # priority=20  (arrow-driven replication)
    PatternExtensionAnalyzer(),         # priority=22  (diagonal pattern extension)
    DiagonalPatternAnalyzer(),          # priority=22  ((r+c)%period color bands)
    MultiEdgeGravityAnalyzer(),         # priority=22  (top/bottom projection)
    # ── Mid-priority: structural analyzers ─────────────────────────────────
    NetworkConnectivityFillAnalyzer(),  # priority=25  (dot→chain→frame fill)
    BorderCropAnalyzer(),               # priority=25  (crop to content bbox)
    SortByAttributeAnalyzer(),          # priority=26  (sort objects by attr)
    LegendRaySlideAnalyzer(),           # priority=28  (legend→direction→slide)
    ObjectStampRuleAnalyzer(),          # priority=28  (color-chain stamp rule)
    ParallelogramAlignAnalyzer(),       # priority=28  (parallelogram alignment)
    DiagonalChainAnalyzer(),            # priority=28  (diagonal object chain)
    ConcentricRingFillAnalyzer(),       # priority=30  (Chebyshev distance fill)
    TArrowMarkerFlowAnalyzer(),         # priority=32  (T-shaped arrow flow)
    TopologyHoleAnalyzer(),             # priority=34  (hole count driven)
    # ── Late / expensive analyzers ─────────────────────────────────────────
    RayCollisionDeflectionAnalyzer(),   # priority=35  (ray emit + deflect)
    PuzzleStitchAssemblyAnalyzer(),     # priority=35  (assemble fragments)
    CountDrivenRuleAnalyzer(),          # priority=38  (count→output size/color)
    MasterTemplateInpaintAnalyzer(),    # priority=40  (D4 template completion)
]

ALL_ANALYZERS.sort(key=lambda a: a.priority)

__all__ = [
    "Analyzer", "ProgramCandidate", "ALL_ANALYZERS",
    # Original analyzers
    "ColorSubstitutionAnalyzer", "TranslationAnalyzer", "ArrowReplicateAnalyzer",
    "SymmetryCompleteAnalyzer", "GravityFallAnalyzer", "BorderCropAnalyzer",
    "PatternExtensionAnalyzer", "BlockCycleAnalyzer",
    "ParallelogramAlignAnalyzer", "DiagonalChainAnalyzer",
    "PatternAnalyzerWrapper",
    # Analyzers from GitMonsters/13-Impossible analysis
    "ConcentricRingFillAnalyzer", "LegendRaySlideAnalyzer",
    "NetworkConnectivityFillAnalyzer", "PuzzleStitchAssemblyAnalyzer",
    "MultiEdgeGravityAnalyzer", "ObjectStampRuleAnalyzer",
    "RayCollisionDeflectionAnalyzer", "TArrowMarkerFlowAnalyzer",
    "MasterTemplateInpaintAnalyzer",
    # Analyzers from SOLVED-540 corpus frequency analysis
    "CountDrivenRuleAnalyzer", "SortByAttributeAnalyzer",
    "SizeSelectionAnalyzer", "TopologyHoleAnalyzer",
    "GridSectionLegendAnalyzer", "DiagonalPatternAnalyzer",
    "UniqueObjectExtractorAnalyzer", "ColorIndexedTilingAnalyzer",
]
