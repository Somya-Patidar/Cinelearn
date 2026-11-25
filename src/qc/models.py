"""
Milestone 6.1: Quality Control (QC) Data Models.

Defines strict, typed Pydantic models for diagnostic codes, levels, evidence structures,
and aggregated QC reports for CineLearn rendered educational video verification.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Literal, Tuple
from pydantic import BaseModel, ConfigDict, Field, model_validator


class QCDiagnosticLevel(str, Enum):
    """Severity level of a QC diagnostic."""
    FATAL = "FATAL"        # Hard unrecoverable failure; immediate halt
    ERROR = "ERROR"        # Deterministic or structural defect; must be fixed or halted
    WARNING = "WARNING"    # Non-blocking advisory finding; recorded for review
    INFO = "INFO"          # Informational measurement or metric


class QCLayer(str, Enum):
    """Architectural layer responsible for a QC check."""
    LAYER_A = "Layer A"    # Pre-render structural and schema validation
    LAYER_B = "Layer B"    # Pre-render spatial layout and temporal scheduling
    LAYER_C = "Layer C"    # Post-render deterministic raster frame inspection
    LAYER_D = "Layer D"    # Post-render advisory multimodal VLM review


class QCDiagnosticCode(str, Enum):
    """Unique diagnostic codes spanning all QC layers."""
    # Layer A: Structural
    QC_SCHEMA_INVALID = "QC_SCHEMA_INVALID"
    QC_UNKNOWN_ENTITY = "QC_UNKNOWN_ENTITY"
    QC_UNSUPPORTED_COMPONENT = "QC_UNSUPPORTED_COMPONENT"
    QC_MISSING_NARRATION_ANCHOR = "QC_MISSING_NARRATION_ANCHOR"

    # Layer B: Deterministic Spatial & Temporal
    QC_LAYOUT_BOUNDARY_BREACH = "QC_LAYOUT_BOUNDARY_BREACH"
    QC_LAYOUT_OVERCROWDING = "QC_LAYOUT_OVERCROWDING"
    QC_TIMELINE_BUDGET_OVERRUN = "QC_TIMELINE_BUDGET_OVERRUN"
    QC_TIMELINE_TARGET_CONFLICT = "QC_TIMELINE_TARGET_CONFLICT"
    QC_LIFECYCLE_ANOMALY = "QC_LIFECYCLE_ANOMALY"

    # Layer C: Rendered Raster Inspection
    QC_BLANK_CANVAS = "QC_BLANK_CANVAS"
    QC_PERIMETER_CLIPPING = "QC_PERIMETER_CLIPPING"
    QC_SUBTITLE_COLLISION = "QC_SUBTITLE_COLLISION"
    QC_RASTER_OVERLAP = "QC_RASTER_OVERLAP"
    QC_LOW_CONTRAST = "QC_LOW_CONTRAST"
    QC_VISUAL_DISAPPEARANCE = "QC_VISUAL_DISAPPEARANCE"
    QC_FRAME_CORRUPTION = "QC_FRAME_CORRUPTION"
    QC_INCONCLUSIVE_MEASUREMENT = "QC_INCONCLUSIVE_MEASUREMENT"

    # Layer D: Advisory VLM
    QC_VLM_PEDAGOGICAL_CLARITY = "QC_VLM_PEDAGOGICAL_CLARITY"
    QC_VLM_VISUAL_CONFUSION = "QC_VLM_VISUAL_CONFUSION"


class QCReportStatus(str, Enum):
    """Aggregate status of a QC evaluation."""
    PASSED = "PASSED"      # Clean evaluation; zero errors or warnings
    WARNINGS = "WARNINGS"  # Non-blocking warnings present; video acceptable
    FAILED = "FAILED"      # Blocking errors or fatal defects present; rejected


class PixelBox(BaseModel):
    """Integer bounding box in frame pixel coordinates [x_min, y_min, x_max, y_max]."""
    model_config = ConfigDict(extra="forbid")

    x_min: int = Field(ge=0, description="Left coordinate (pixels)")
    y_min: int = Field(ge=0, description="Top coordinate (pixels)")
    x_max: int = Field(ge=0, description="Right coordinate (pixels)")
    y_max: int = Field(ge=0, description="Bottom coordinate (pixels)")

    @model_validator(mode="after")
    def validate_bounds(self) -> "PixelBox":
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be strictly greater than x_min ({self.x_min})")
        if self.y_max <= self.y_min:
            raise ValueError(f"y_max ({self.y_max}) must be strictly greater than y_min ({self.y_min})")
        return self

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height


class BlankCanvasEvidence(BaseModel):
    """Evidence metrics for blank or corrupted canvas detection."""
    model_config = ConfigDict(extra="forbid")

    mean_luminance: float = Field(ge=0.0, le=255.0)
    std_luminance: float = Field(ge=0.0)
    entropy: float = Field(ge=0.0)
    foreground_coverage: float = Field(ge=0.0, le=1.0)
    is_blank: bool
    std_threshold: float = Field(ge=0.0)
    entropy_threshold: float = Field(ge=0.0)


class PerimeterClippingEvidence(BaseModel):
    """Evidence metrics for perimeter clipping analysis on a specific edge."""
    model_config = ConfigDict(extra="forbid")

    edge: Literal["left", "right", "top", "bottom"]
    band_thickness_pixels: int = Field(ge=0)
    edge_pixel_density: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    is_clipped: bool


class SubtitleCollisionEvidence(BaseModel):
    """Evidence metrics for subtitle clearance band inspection."""
    model_config = ConfigDict(extra="forbid")

    subtitle_band_y_min: int = Field(ge=0)
    subtitle_band_y_max: int = Field(ge=0)
    collision_pixel_density: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    is_colliding: bool


class ContrastEvidence(BaseModel):
    """Evidence metrics for relative luminance and WCAG contrast ratio evaluation."""
    model_config = ConfigDict(extra="forbid")

    contrast_ratio: float = Field(ge=1.0, le=21.0)
    foreground_luminance: float = Field(ge=0.0, le=1.0)
    background_luminance: float = Field(ge=0.0, le=1.0)
    required_ratio: float = Field(ge=1.0)
    is_compliant: bool
    is_conclusive: bool
    is_large_text: bool = False
    method: str = "bimodal_clustering"


class OverlapEvidence(BaseModel):
    """Evidence metrics for geometric and raster ROI overlap inspection."""
    model_config = ConfigDict(extra="forbid")

    box_a: PixelBox
    box_b: PixelBox
    intersection_box: Optional[PixelBox] = None
    intersection_area: int = Field(ge=0)
    overlap_ratio_a: float = Field(ge=0.0, le=1.0)
    overlap_ratio_b: float = Field(ge=0.0, le=1.0)
    is_geometric_overlap: bool
    raster_edge_density: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_raster_overlap: Optional[bool] = None


class QCDiagnostic(BaseModel):
    """A single structured QC diagnostic finding."""
    model_config = ConfigDict(extra="forbid")

    code: QCDiagnosticCode
    level: QCDiagnosticLevel
    layer: QCLayer
    message: str
    scene_id: Optional[str] = None
    timestamp: Optional[float] = Field(default=None, ge=0.0)
    step_id: Optional[str] = None
    entity_id: Optional[str] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)


class QCReport(BaseModel):
    """Aggregated Quality Control report for a scene or video artifact."""
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    overall_status: QCReportStatus
    is_valid: bool
    sampled_frames_count: int = Field(ge=0)
    diagnostics: List[QCDiagnostic] = Field(default_factory=list)
    raster_metrics: Dict[str, Any] = Field(default_factory=dict)
    advisory_vlm: Optional[Dict[str, Any]] = None

    def has_errors(self) -> bool:
        """Returns True if any diagnostic is ERROR or FATAL."""
        return any(d.level in (QCDiagnosticLevel.ERROR, QCDiagnosticLevel.FATAL) for d in self.diagnostics)

    def has_warnings(self) -> bool:
        """Returns True if any diagnostic is WARNING."""
        return any(d.level == QCDiagnosticLevel.WARNING for d in self.diagnostics)

    def get_diagnostics_by_level(self, level: QCDiagnosticLevel) -> List[QCDiagnostic]:
        """Filters diagnostics by severity level."""
        return [d for d in self.diagnostics if d.level == level]

    def get_diagnostics_by_layer(self, layer: QCLayer) -> List[QCDiagnostic]:
        """Filters diagnostics by architectural layer."""
        return [d for d in self.diagnostics if d.layer == layer]

    def get_diagnostics_by_code(self, code: QCDiagnosticCode) -> List[QCDiagnostic]:
        """Filters diagnostics by diagnostic code."""
        return [d for d in self.diagnostics if d.code == code]


class RasterAnalyzerConfig(BaseModel):
    """Configurable thresholds for deterministic raster vision analysis."""
    model_config = ConfigDict(extra="forbid")

    blank_std_threshold: float = Field(default=4.0, ge=0.0)
    blank_entropy_threshold: float = Field(default=0.5, ge=0.0)
    min_fg_coverage: float = Field(default=0.015, ge=0.0, le=1.0)
    border_ratio: float = Field(default=0.025, ge=0.0, le=0.2)
    border_max_density: float = Field(default=0.02, ge=0.0, le=1.0)
    subtitle_band_y_min_ratio: float = Field(default=0.85, ge=0.5, le=1.0)
    subtitle_band_y_max_ratio: float = Field(default=0.98, ge=0.5, le=1.0)
    subtitle_max_density: float = Field(default=0.015, ge=0.0, le=1.0)
    normal_text_contrast_threshold: float = Field(default=4.5, ge=1.0, le=21.0)
    large_text_contrast_threshold: float = Field(default=3.0, ge=1.0, le=21.0)
    geometric_overlap_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    raster_overlap_density_threshold: float = Field(default=0.03, ge=0.0, le=1.0)
