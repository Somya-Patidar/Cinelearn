"""
CineLearn — Milestone 4A: Spatial Layout Models
===============================================
Domain-independent, typed spatial geometry models for CineLearn's
Hierarchical Box Model Layout Solver and Spatial Ledger.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.ir.models import Alignment, FlowDirection, ViewportType


class CoordinateSpace(str, Enum):
    """Coordinate reference spaces."""
    WORLD = "world"          # Manim canvas world coordinates (origin at center: [0, 0])
    VIEWPORT = "viewport"    # Local coordinates relative to viewport bounds
    PARENT = "parent"        # Local coordinates relative to parent entity bounds


class CollisionCategory(str, Enum):
    """
    Formal 7-category collision taxonomy from Scene IR & Spatial Composition Spec.
    """
    FORBIDDEN_OVERLAP = "forbidden_overlap"            # Category 1: Fatal defect (unrelated simultaneous overlap)
    INTENTIONAL_CONTAINMENT = "intentional_containment"  # Category 2: Valid (child within parent bounds)
    INTENTIONAL_LAYERING = "intentional_layering"        # Category 3: Valid (distinct z_indices with overlay/badge role)
    COMPOSITE_GROUPING = "composite_grouping"          # Category 4: Valid (declared members of same composite tree)
    TEMPORARY_TRANSITION = "temporary_transition"      # Category 5: Valid (short trajectory intersection <= 0.8s)
    SEQUENTIAL_REPLACEMENT = "sequential_replacement"  # Category 6: Valid (disjoint lifespan intervals)
    SHARED_SUB_SLOT = "shared_sub_slot"                # Category 7: Valid (distinct partitioned sub-slots)


class LayoutErrorCode(str, Enum):
    """Standardized layout error and diagnostic codes."""
    LAYOUT_OVERCROWDING_ERROR = "LAYOUT_OVERCROWDING_ERROR"
    LAYOUT_BOUNDARY_BREACH = "LAYOUT_BOUNDARY_BREACH"
    LAYOUT_FORBIDDEN_OVERLAP = "LAYOUT_FORBIDDEN_OVERLAP"
    LAYOUT_CYCLIC_HIERARCHY = "LAYOUT_CYCLIC_HIERARCHY"
    LAYOUT_PARENT_BOUNDS_EXCEEDED = "LAYOUT_PARENT_BOUNDS_EXCEEDED"
    LAYOUT_DIMENSION_UNCERTAINTY = "LAYOUT_DIMENSION_UNCERTAINTY"
    LAYOUT_INVALID_VIEWPORT = "LAYOUT_INVALID_VIEWPORT"
    LAYOUT_UNSATISFIABLE_CONSTRAINTS = "LAYOUT_UNSATISFIABLE_CONSTRAINTS"


# Shared spatial boundary and containment tolerances
VIEWPORT_BOUNDARY_TOLERANCE: float = 0.05  # Canvas and viewport containment tolerance (≈6.75px at 1080p)
CONTAINER_BORDER_TOLERANCE: float = 0.02   # Child-within-container stroke border tolerance (≈2.7px at 1080p)


class BoundingBox2D(BaseModel):
    """
    Axis-Aligned 2D Bounding Box (AABB) in Cartesian coordinates.
    Origin is at (0, 0), with width and height centered at (x, y).
    """
    model_config = ConfigDict(extra="forbid")

    x: float = Field(description="Center X coordinate")
    y: float = Field(description="Center Y coordinate")
    width: float = Field(gt=0, description="Box width (> 0)")
    height: float = Field(gt=0, description="Box height (> 0)")

    @classmethod
    def from_min_max(cls, left: float, right: float, bottom: float, top: float) -> "BoundingBox2D":
        """Constructs an AABB from boundary extremes."""
        if left >= right:
            raise ValueError(f"Invalid bounds: left ({left}) must be < right ({right})")
        if bottom >= top:
            raise ValueError(f"Invalid bounds: bottom ({bottom}) must be < top ({top})")
        w = float(right - left)
        h = float(top - bottom)
        cx = float((left + right) / 2.0)
        cy = float((bottom + top) / 2.0)
        return cls(x=cx, y=cy, width=w, height=h)

    @property
    def left(self) -> float:
        return self.x - self.width / 2.0

    @property
    def right(self) -> float:
        return self.x + self.width / 2.0

    @property
    def bottom(self) -> float:
        return self.y - self.height / 2.0

    @property
    def top(self) -> float:
        return self.y + self.height / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 1.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def contains_point(self, px: float, py: float, tolerance: float = 0.001) -> bool:
        """Checks if a point (px, py) lies within this box."""
        return (self.left - tolerance <= px <= self.right + tolerance and
                self.bottom - tolerance <= py <= self.top + tolerance)

    def contains_box(self, other: "BoundingBox2D", tolerance: float = 0.001) -> bool:
        """Checks if another box is strictly contained within this box."""
        eps = 1e-9
        return (other.left >= self.left - tolerance - eps and
                other.right <= self.right + tolerance + eps and
                other.bottom >= self.bottom - tolerance - eps and
                other.top <= self.top + tolerance + eps)

    def intersects(self, other: "BoundingBox2D", tolerance: float = 0.001) -> bool:
        """Checks if this box intersects another box in 2D."""
        x_overlap = min(self.right, other.right) - max(self.left, other.left)
        y_overlap = min(self.top, other.top) - max(self.bottom, other.bottom)
        return (x_overlap > -tolerance) and (y_overlap > -tolerance)

    def intersection_box(self, other: "BoundingBox2D") -> Optional["BoundingBox2D"]:
        """Computes the intersecting rectangle between two boxes, or None if disjoint."""
        i_left = max(self.left, other.left)
        i_right = min(self.right, other.right)
        i_bottom = max(self.bottom, other.bottom)
        i_top = min(self.top, other.top)

        if i_left < i_right and i_bottom < i_top:
            return BoundingBox2D.from_min_max(i_left, i_right, i_bottom, i_top)
        return None

    def intersection_area(self, other: "BoundingBox2D") -> float:
        """Computes the area of overlap between two boxes."""
        ibox = self.intersection_box(other)
        return ibox.area if ibox is not None else 0.0

    def overlap_ratio(self, other: "BoundingBox2D") -> float:
        """
        Computes overlap ratio relative to the smaller object's area.
        Returns value in [0.0, 1.0].
        """
        min_a = min(self.area, other.area)
        if min_a <= 0.0001:
            return 0.0
        return self.intersection_area(other) / min_a

    def scale(self, factor: float) -> "BoundingBox2D":
        """Returns a new box scaled uniformly around its center."""
        if factor <= 0:
            raise ValueError(f"Scale factor must be > 0, got {factor}")
        return BoundingBox2D(
            x=self.x,
            y=self.y,
            width=self.width * factor,
            height=self.height * factor,
        )

    def shift(self, dx: float, dy: float) -> "BoundingBox2D":
        """Returns a new box shifted by (dx, dy)."""
        return BoundingBox2D(
            x=self.x + dx,
            y=self.y + dy,
            width=self.width,
            height=self.height,
        )

    def to_dict(self) -> Dict[str, float]:
        """Serializes box to a dictionary with boundaries and center."""
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "left": round(self.left, 4),
            "right": round(self.right, 4),
            "bottom": round(self.bottom, 4),
            "top": round(self.top, 4),
        }


class CollisionRecord(BaseModel):
    """Record of a detected spatial intersection between two entities."""
    model_config = ConfigDict(extra="forbid")

    entity_a: str
    entity_b: str
    category: CollisionCategory
    is_fatal: bool
    overlap_area: float
    overlap_ratio: float
    time_range: Optional[Tuple[float, float]] = None
    description: str


class LayoutDiagnostic(BaseModel):
    """Structured diagnostic reporting a layout defect or warning."""
    model_config = ConfigDict(extra="forbid")

    error_code: LayoutErrorCode
    message: str
    viewport_id: Optional[str] = None
    entity_id: Optional[str] = None
    related_entities: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)

    @property
    def severity(self) -> str:
        """Severity classification for diagnostics compatibility."""
        return "ERROR"


class LayoutFailureReport(BaseModel):
    """
    Structured failure report produced when overcrowding mitigation hierarchy fails.
    Provides actionable guidance for upstream visual replanning.
    """
    model_config = ConfigDict(extra="forbid")

    status: str = "LAYOUT_OVERCROWDING_ERROR"
    scene_id: str
    viewport_id: str
    conflicting_entities: List[str]
    available_dimensions: Dict[str, float]
    required_dimensions: Dict[str, float]
    spatial_deficit: float
    mitigations_attempted: List[str]
    recommended_action: str = "SPLIT_INTO_SEQUENTIAL_BEATS"


class ComputedEntityGeometry(BaseModel):
    """
    Computed geometric layout allocation for a visual entity.
    Maintains both local coordinates (relative to parent or viewport) and
    absolute world coordinates (for direct Manim rendering).
    """
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    viewport_id: str
    parent_id: Optional[str] = None
    local_box: BoundingBox2D = Field(description="Bounding box in local parent/viewport space")
    world_box: BoundingBox2D = Field(description="Bounding box in Manim canvas world space")
    scale: float = Field(default=1.0, gt=0, description="Effective scale factor applied")
    z_index: int = Field(default=10, description="Rendering layering depth")
    visual_priority: int = Field(default=1, description="Priority for overcrowding demotion")
    slot: Optional[str] = Field(default=None, description="Assigned viewport sub-slot")
    layer_role: Optional[str] = Field(default=None, description="Role: base, overlay, badge, halo")
    is_uncertain: bool = Field(default=False, description="Flag indicating dimension estimation uncertainty")


class ComputedViewportGeometry(BaseModel):
    """
    Computed spatial geometry and partitioning for a declared viewport.
    """
    model_config = ConfigDict(extra="forbid")

    viewport_id: str
    viewport_type: ViewportType
    world_box: BoundingBox2D = Field(description="Viewport bounding box in canvas world coordinates")
    flow: FlowDirection = Field(default=FlowDirection.SINGLE)
    alignment: Alignment = Field(default=Alignment.CENTER)
    padding: float = Field(default=0.20, ge=0)
    sub_slots: Dict[str, BoundingBox2D] = Field(default_factory=dict)
    assigned_entities: List[str] = Field(default_factory=list)


class SceneSpatialLayout(BaseModel):
    """
    Complete, validated spatial layout solution for a Scene IR document.
    Deterministic output consumed by downstream compilers and quality control gates.
    """
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    canvas_box: BoundingBox2D = Field(description="Active visual canvas boundary")
    viewports: Dict[str, ComputedViewportGeometry] = Field(default_factory=dict)
    entities: Dict[str, ComputedEntityGeometry] = Field(default_factory=dict)
    is_valid: bool = Field(default=True, description="True if layout has zero fatal collision or boundary errors")
    errors: List[LayoutDiagnostic] = Field(default_factory=list)
    warnings: List[LayoutDiagnostic] = Field(default_factory=list)
    mitigations_applied: List[str] = Field(default_factory=list)
    failure_report: Optional[LayoutFailureReport] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def diagnostics(self) -> List[LayoutDiagnostic]:
        """Backward-compatible view combining errors and warnings."""
        return self.errors + self.warnings

    def get_entity_box(self, entity_id: str) -> Optional[BoundingBox2D]:
        """Returns the computed world bounding box for an entity."""
        geom = self.entities.get(entity_id)
        return geom.world_box if geom else None

    def get_viewport_box(self, viewport_id: str) -> Optional[BoundingBox2D]:
        """Returns the computed world bounding box for a viewport."""
        vp = self.viewports.get(viewport_id)
        return vp.world_box if vp else None
