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

# Batch 4 analyzers
from .extract_color_crop import ExtractColorCropAnalyzer
from .shape_move_to_anchor import ShapeMoveToAnchorAnalyzer
from .dot_center_diag_fill import DotCenterDiagFillAnalyzer
from .pattern_tile_downward import PatternTileDownwardAnalyzer
from .shape_stamp_by_color import ShapeStampByColorAnalyzer
from .isolated_to_color3 import IsolatedToColor3Analyzer
from .segment_extend_to_boundary import SegmentExtendToBoundaryAnalyzer
from .shape_packing import ShapePackingAnalyzer
from .tile_gap_fill import TileGapFillAnalyzer
from .ray_shoot_from_special import RayShootFromSpecialAnalyzer
from .cross_star_fill import CrossStarFillAnalyzer
from .frame_corner_extend import FrameCornerExtendAnalyzer
from .uniform_line_tile_3x3 import UniformLineTile3x3Analyzer
from .frame_corner_marker import FrameCornerMarkerAnalyzer
from .section_legend_mask_combine import SectionLegendMaskCombineAnalyzer
from .dual_gravity_separate import DualGravitySeparateAnalyzer

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
    DualGravitySeparateAnalyzer(),      # priority=14  (17829a00: dual gravity top/bottom separate)
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
    CrossStarFillAnalyzer(),            # priority=16  (140c817e: cross-star rays from seed dots)
    FrameCornerExtendAnalyzer(),        # priority=17  (14b8e18c: corner extensions for hollow frames)
    UniformLineTile3x3Analyzer(),       # priority=15  (15696249: tile 3x3 along uniform line)
    SectionLegendMaskCombineAnalyzer(), # priority=15  (15660dd6: mask legend + feature colors)
    FrameCornerMarkerAnalyzer(),        # priority=18  (15663ba9: mark corners/endpoints of frames)
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
    ExtractColorCropAnalyzer(),         # priority=18  (1190e5a7: crop to target color bbox)
    ShapeMoveToAnchorAnalyzer(),        # priority=19  (11dc524f: move shape to touch anchor)
    DotCenterDiagFillAnalyzer(),        # priority=20  (11e1fe23: diags from dots to center + color 5)
    PatternTileDownwardAnalyzer(),      # priority=19  (12422b43: tile pattern downward)
    ShapeStampByColorAnalyzer(),        # priority=18  (12997ef3: stamp template onto each dot color)
    IsolatedToColor3Analyzer(),         # priority=19  (12eac192: isolated non-anchor cells -> 3)
    SegmentExtendToBoundaryAnalyzer(),  # priority=18  (13713586: extend segments to boundary)
    ShapePackingAnalyzer(),             # priority=18  (137eaa0f: pack objects into 3x3 mosaic)
    TileGapFillAnalyzer(),              # priority=18  (137f0df0: fill tile gaps with 2, outer with 1)
    RayShootFromSpecialAnalyzer(),      # priority=22  (13f06aa5: rays from special cell to edges)
    VerticalLineExtensionAnalyzer(),    # priority=12  (extend vertical lines horizontally)
    BorderEdgeMarkingAnalyzer(),        # priority=12  (mark border edges)
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
    "ColorSubstitutionAnalyzer", "TranslationAnalyzer", "ArrowReplicateAnalyzer",
    "SymmetryCompleteAnalyzer", "GravityFallAnalyzer", "BorderCropAnalyzer",
    "PatternExtensionAnalyzer", "BlockCycleAnalyzer",
    "ParallelogramAlignAnalyzer", "DiagonalChainAnalyzer",
    "PatternAnalyzerWrapper", "CrossStarFillAnalyzer", "FrameCornerExtendAnalyzer",
    "ExtractColorCropAnalyzer", "DotCenterDiagFillAnalyzer",
    "PatternTileDownwardAnalyzer", "ShapeStampByColorAnalyzer",
    "IsolatedToColor3Analyzer", "SegmentExtendToBoundaryAnalyzer",
    "ShapePackingAnalyzer", "TileGapFillAnalyzer", "UniformLineTile3x3Analyzer",
    "FrameCornerMarkerAnalyzer", "SectionLegendMaskCombineAnalyzer",
    "DualGravitySeparateAnalyzer",
]
