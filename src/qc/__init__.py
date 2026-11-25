"""
Milestone 6: Rendered-Video Quality Control (QC) Package.
"""

from src.qc.models import (
    QCDiagnosticLevel,
    QCLayer,
    QCDiagnosticCode,
    QCReportStatus,
    PixelBox,
    BlankCanvasEvidence,
    PerimeterClippingEvidence,
    SubtitleCollisionEvidence,
    ContrastEvidence,
    OverlapEvidence,
    QCDiagnostic,
    QCReport,
    RasterAnalyzerConfig,
)
from src.qc.keyframe_sampler import (
    SamplePhase,
    SamplingEventRef,
    KeyframeSamplerConfig,
    KeyframeSampleRequest,
    TimestampEvidence,
    KeyframeSamplingPlan,
    KeyframeSampleResult,
    KeyframeSampler,
)
from src.qc.spatial_projector import (
    ProjectionConfig,
    GeometricOverlapRecord,
    ProjectedEntityGeometry,
    SpatialProjectionRequest,
    ProjectedSpatialLayout,
    SpatialProjectionResult,
    SpatialCoordinateProjector,
    parse_aspect_ratio_to_float,
)
from src.qc.controller import (
    VideoQCConfig,
    RenderedVideoQualityController,
)

__all__ = [
    # M6.1 Core QC Models
    "QCDiagnosticLevel",
    "QCLayer",
    "QCDiagnosticCode",
    "QCReportStatus",
    "PixelBox",
    "BlankCanvasEvidence",
    "PerimeterClippingEvidence",
    "SubtitleCollisionEvidence",
    "ContrastEvidence",
    "OverlapEvidence",
    "QCDiagnostic",
    "QCReport",
    "RasterAnalyzerConfig",
    # M6.2 Keyframe Sampler
    "SamplePhase",
    "SamplingEventRef",
    "KeyframeSamplerConfig",
    "KeyframeSampleRequest",
    "TimestampEvidence",
    "KeyframeSamplingPlan",
    "KeyframeSampleResult",
    "KeyframeSampler",
    # M6.2 Spatial Projector
    "ProjectionConfig",
    "GeometricOverlapRecord",
    "ProjectedEntityGeometry",
    "SpatialProjectionRequest",
    "ProjectedSpatialLayout",
    "SpatialProjectionResult",
    "SpatialCoordinateProjector",
    "parse_aspect_ratio_to_float",
    # M6.3 QC Controller
    "VideoQCConfig",
    "RenderedVideoQualityController",
]
