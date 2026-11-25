"""
CineLearn — Deterministic Compiler & Spatial Layout Engine
==========================================================
Package exposing the Hierarchical Box Model Layout Solver,
Spatial Ledger, and geometric layout data models.
"""

from src.compiler.card_compiler import (
    ConceptualCardCompiler,
    sanitize_latex,
    should_use_matching_tex,
)
from src.compiler.compiler_models import (
    CompilationDiagnostic,
    CompilationErrorCode,
    CompiledSceneOutput,
    ScheduledAction,
    ScheduledStep,
    ScheduledTimeline,
    canonicalize_time,
    is_safe_keyword_identifier,
)
from src.compiler.dimension_estimator import (
    EstimatedDimensions,
    IntrinsicDimensionEstimator,
)
from src.compiler.layout_engine import HierarchicalBoxModelLayoutSolver
from src.compiler.layout_models import (
    BoundingBox2D,
    CollisionCategory,
    CollisionRecord,
    ComputedEntityGeometry,
    ComputedViewportGeometry,
    CoordinateSpace,
    LayoutDiagnostic,
    LayoutErrorCode,
    LayoutFailureReport,
    SceneSpatialLayout,
)
from src.compiler.primitive_compiler import (
    PrimitiveCompiler,
    SUPPORTED_PRIMITIVES,
    SUPPORTED_SHAPE_TYPES,
)
from src.compiler.scene_compiler import DeterministicSceneCompiler
from src.compiler.spatial_ledger import (
    EntitySpatialSpan,
    SpatialLedger,
    SpatialTimeSlice,
)
from src.compiler.timeline_scheduler import (
    TimelineScheduler,
    build_disjoint_lifespan_intervals,
    deep_compare_parameters,
    determine_initial_presence,
    validate_tolerance,
)

__all__ = [
    # Geometric Models
    "BoundingBox2D",
    "CoordinateSpace",
    "CollisionCategory",
    "CollisionRecord",
    "LayoutErrorCode",
    "LayoutDiagnostic",
    "LayoutFailureReport",
    "ComputedEntityGeometry",
    "ComputedViewportGeometry",
    "SceneSpatialLayout",
    # Dimension Estimator
    "IntrinsicDimensionEstimator",
    "EstimatedDimensions",
    # Spatial Ledger
    "SpatialLedger",
    "SpatialTimeSlice",
    "EntitySpatialSpan",
    # Layout Engine / Solver
    "HierarchicalBoxModelLayoutSolver",
    # Milestone 4B Scene Compiler & Models
    "CompilationErrorCode",
    "CompilationDiagnostic",
    "CompiledSceneOutput",
    "PrimitiveCompiler",
    "SUPPORTED_PRIMITIVES",
    "SUPPORTED_SHAPE_TYPES",
    "ConceptualCardCompiler",
    "sanitize_latex",
    "should_use_matching_tex",
    "is_safe_keyword_identifier",
    "DeterministicSceneCompiler",
    # Milestone 4C.4 TimelineScheduler & Pacing Models
    "ScheduledAction",
    "ScheduledStep",
    "ScheduledTimeline",
    "canonicalize_time",
    "TimelineScheduler",
    "build_disjoint_lifespan_intervals",
    "determine_initial_presence",
    "deep_compare_parameters",
    "validate_tolerance",
]
