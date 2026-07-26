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

ALL_ANALYZERS = [
    PatternAnalyzerWrapper(),  # High priority - runs first
    ColorSubstitutionAnalyzer(),
    TranslationAnalyzer(),
    SymmetryCompleteAnalyzer(),
    GravityFallAnalyzer(),
    PatternExtensionAnalyzer(),
    BlockCycleAnalyzer(),
    ArrowReplicateAnalyzer(),
    ParallelogramAlignAnalyzer(),
    DiagonalChainAnalyzer(),
    BorderCropAnalyzer(),
]

ALL_ANALYZERS.sort(key=lambda a: a.priority)

__all__ = [
    "Analyzer", "ProgramCandidate", "ALL_ANALYZERS",
    "ColorSubstitutionAnalyzer", "TranslationAnalyzer", "ArrowReplicateAnalyzer",
    "SymmetryCompleteAnalyzer", "GravityFallAnalyzer", "BorderCropAnalyzer",
    "PatternExtensionAnalyzer", "BlockCycleAnalyzer",
    "ParallelogramAlignAnalyzer", "DiagonalChainAnalyzer",
    "PatternAnalyzerWrapper",
]
