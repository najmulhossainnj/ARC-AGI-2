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
    LegendShapeToColorAnalyzer,
    QuadMirrorSymmetryAnalyzer,
    SequenceDotRayContinuationAnalyzer,
    RayLinePeriodicStrideAnalyzer,
    ComponentAreaRangeRecolorAnalyzer,
    SecondaryColorCropAnalyzer,
    TopLeftKeyRowSwapAnalyzer,
    SpatialCentroidGridSortAnalyzer,
    DividerQuadrantStitchAssemblyAnalyzer,
    SkeletonSolidBlockRecolorAnalyzer,
    SeedDiagonalRayProjectionAnalyzer,
    VerticalLineExtensionAnalyzer,
    BorderEdgeMarkingAnalyzer,
    CrossCenterExtractionAnalyzer,
)

from .tile_diagonal_mark import TileDiagonalMarkAnalyzer
from .seed_surround_rule import SeedSurroundRuleAnalyzer
from .indicator_line_object_absorb import IndicatorLineObjectAbsorbAnalyzer
from .l_path_dot_connect import LPathDotConnectAnalyzer
from .seed_row_bands_frame import SeedRowBandsFrameAnalyzer
from .legend_rotate_scale_recolor import LegendRotateScaleRecolorAnalyzer
from .quad_symmetry_complete import QuadSymmetryCompleteAnalyzer
from .template_d4_key_align import TemplateD4KeyAlignAnalyzer

# ── Single canonical ALL_ANALYZERS list ───────────────────────────────────────
# Priority ordering: lower = faster/cheaper/more specific (runs first)
ALL_ANALYZERS = [
    # ── Fast / high-precision filters ──────────────────────────────────────
    PatternAnalyzerWrapper(),           # priority=1   (periodic pattern, very fast)
    ColorSubstitutionAnalyzer(),        # priority=5   (1-to-1 color remap)
    SizeSelectionAnalyzer(),            # priority=8   (keep largest/smallest)
    SeedSurroundRuleAnalyzer(),         # priority=9   (seed color orthogonal/diagonal surround)
    QuadSymmetryCompleteAnalyzer(),     # priority=10  (4-way center symmetry completion)
    TileDiagonalMarkAnalyzer(),         # priority=10  (2x2 tile + diagonal neighbor mark)
    IndicatorLineObjectAbsorbAnalyzer(),# priority=10  (indicator line connect + object recolor)
    LPathDotConnectAnalyzer(),          # priority=10  (chained L-path dot connection)
    LegendRotateScaleRecolorAnalyzer(), # priority=15  (legend 270 deg rotate block scale recolor)
    TemplateD4KeyAlignAnalyzer(),       # priority=15  (D4 template alignment to key dots)
    SeedRowBandsFrameAnalyzer(),        # priority=15  (seed row bands frame)
    AnomalyRepairAnalyzer(),            # priority=10  (repair single-cell anomaly)
    TranslationAnalyzer(),              # priority=10  (uniform shift)
    GravityFallAnalyzer(),              # priority=12  (objects fall to edge)
    LegendShapeToColorAnalyzer(),       # priority=12  (legend shape → recolor target)
    QuadMirrorSymmetryAnalyzer(),        # priority=10  (2Hx2W 4-way quad mirror)
    SequenceDotRayContinuationAnalyzer(),# priority=12  (linear sequence of dots to boundary)
    RayLinePeriodicStrideAnalyzer(),    # priority=15  (periodic line stride from boundary dots)
    ComponentAreaRangeRecolorAnalyzer(),# priority=10  (recolor component by cell area range)
    SecondaryColorCropAnalyzer(),       # priority=10  (crop to secondary color bbox)
    TopLeftKeyRowSwapAnalyzer(),        # priority=10  (swap colors via top-left 2x2 key)
    SpatialCentroidGridSortAnalyzer(),  # priority=10  (2D centroid grid sort)
    DividerQuadrantStitchAssemblyAnalyzer(), # priority=10 (stitch 4 divider quadrants)
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
    # ── Batch 4 task analyzers ─────────────────────────────────────────────
    VerticalLineExtensionAnalyzer(),     # priority=12  (extend vertical lines horizontally)
    BorderEdgeMarkingAnalyzer(),       # priority=12  (mark border edges)
    CrossCenterExtractionAnalyzer(),    # priority=15  (extract center from cross pattern)
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
