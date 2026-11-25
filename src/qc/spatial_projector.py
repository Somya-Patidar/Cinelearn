"""
Milestone 6.2: Spatial Coordinate Projector.

Translates authoritative layout geometry (BoundingBox2D, ComputedEntityGeometry,
SceneSpatialLayout, SpatialLedger) into pixel-space bounding boxes (PixelBox)
for raster vision inspection.

Features:
- Deterministic world-to-pixel coordinate projection with vertical-axis inversion.
- Full support for arbitrary aspect ratios (16:9, 4:3, 1:1, 9:16) and resolutions.
- Safety padding and viewport boundary containment checking.
- Time-aware lifecycle filtering: projects only entities active at the sample timestamp.
- Strict semantic distinction between geometric bounding box intersection and verified collision.
- Structured diagnostics for out-of-bounds, degenerate, or missing geometry.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.compiler.layout_models import (
    BoundingBox2D,
    ComputedEntityGeometry,
    ComputedViewportGeometry,
    LayoutDiagnostic,
    LayoutErrorCode,
    SceneSpatialLayout,
)
from src.compiler.spatial_ledger import SpatialLedger, SpatialTimeSlice
from src.qc.models import (
    PixelBox,
    QCDiagnostic,
    QCDiagnosticCode,
    QCDiagnosticLevel,
    QCLayer,
)


# ============================================================================
# Enums and Typed Models for Spatial Projection
# ============================================================================

def parse_aspect_ratio_to_float(aspect_str: str) -> float:
    """Parses aspect ratio string (e.g. '16:9', '4:3', '1:1', '9:16') to float."""
    if not aspect_str:
        return 16.0 / 9.0
    s = str(aspect_str).strip()
    try:
        parts = s.split(":")
        if len(parts) == 2:
            w = float(parts[0].strip())
            h = float(parts[1].strip())
            if w > 0 and h > 0:
                return float(w / h)
        return float(s)
    except Exception:
        return 16.0 / 9.0


class ProjectionConfig(BaseModel):
    """
    Configurable parameters defining pixel resolution, camera world bounds,
    and containment tolerances for spatial projection.
    """
    model_config = ConfigDict(extra="forbid")

    frame_width: int = Field(default=1920, gt=0, description="Output frame width in pixels.")
    frame_height: int = Field(default=1080, gt=0, description="Output frame height in pixels.")
    world_height: float = Field(default=8.0, gt=0.0, description="Manim camera world height units.")
    aspect_ratio: str = Field(default="16:9", description="Camera aspect ratio.")
    safety_padding: float = Field(default=0.20, ge=0.0, description="Screen margin safety padding.")
    allow_partial_clipping: bool = Field(
        default=True,
        description="Whether to clamp partially out-of-bounds geometry to frame edges with a warning.",
    )

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect(cls, v: str) -> str:
        s = str(v).strip()
        if s not in ("16:9", "4:3", "9:16", "1:1"):
            # Check if it parses as float or x:y
            parse_aspect_ratio_to_float(s)
        return s

    @property
    def world_width(self) -> float:
        """Derived camera world width in Manim units."""
        aspect = parse_aspect_ratio_to_float(self.aspect_ratio)
        return float(self.world_height * aspect)

    @property
    def scale_x(self) -> float:
        """Horizontal pixel scaling factor (pixels per world unit)."""
        return self.frame_width / self.world_width

    @property
    def scale_y(self) -> float:
        """Vertical pixel scaling factor (pixels per world unit)."""
        return self.frame_height / self.world_height


class GeometricOverlapRecord(BaseModel):
    """
    Record of geometric axis-aligned bounding box (AABB) intersection between two entities.
    Preserves provenance and explicit semantic distinction from verified collision.
    """
    model_config = ConfigDict(extra="forbid")

    entity_a: str = Field(min_length=1)
    entity_b: str = Field(min_length=1)
    pixel_box_a: PixelBox
    pixel_box_b: PixelBox
    intersection_box: PixelBox
    intersection_area_pixels: int = Field(ge=0)
    overlap_ratio_a: float = Field(ge=0.0, le=1.0)
    overlap_ratio_b: float = Field(ge=0.0, le=1.0)
    semantic_note: str = (
        "Geometric bounding-box intersection only. Does NOT prove verified object-to-object collision "
        "without pixel-mask or edge-provenance analysis."
    )


class ProjectedEntityGeometry(BaseModel):
    """
    Pixel-space projection of an entity's layout allocation at a specific sample time.
    """
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(min_length=1)
    scene_id: Optional[str] = None
    viewport_id: str = Field(default="main")
    parent_id: Optional[str] = None
    world_box: BoundingBox2D
    pixel_box: PixelBox
    z_index: int = 10
    layer_role: Optional[str] = None
    is_clipped_by_canvas: bool = False
    is_clipped_by_viewport: bool = False
    active_intervals: List[Tuple[float, float]] = Field(default_factory=list)


class SpatialProjectionRequest(BaseModel):
    """
    Request to project active scene geometry into pixel space at a specific timestamp.
    """
    model_config = ConfigDict(extra="forbid")

    timestamp: float = Field(ge=0.0)
    scene_id: str = Field(min_length=1)
    filter_active_only: bool = True

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_finite(cls, v: float) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("timestamp must be numeric, not boolean.")
        if math.isnan(float(v)) or math.isinf(float(v)):
            raise ValueError("timestamp must be finite.")
        if float(v) < 0.0:
            raise ValueError("timestamp cannot be negative.")
        return round(float(v), 3)


class ProjectedSpatialLayout(BaseModel):
    """
    Complete pixel-space projected layout snapshot at a discrete timestamp.
    """
    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0.0)
    resolution: Tuple[int, int] = Field(description="(width, height) in pixels")
    projected_entities: Dict[str, ProjectedEntityGeometry] = Field(default_factory=dict)
    active_entity_ids: List[str] = Field(default_factory=list)
    inactive_entity_ids: List[str] = Field(default_factory=list)
    geometric_overlaps: List[GeometricOverlapRecord] = Field(default_factory=list)
    diagnostics: List[QCDiagnostic] = Field(default_factory=list)

    @property
    def active_count(self) -> int:
        return len(self.active_entity_ids)

    def get_pixel_box(self, entity_id: str) -> Optional[PixelBox]:
        """Returns the projected PixelBox for an active entity."""
        geom = self.projected_entities.get(entity_id)
        return geom.pixel_box if geom else None


class SpatialProjectionResult(BaseModel):
    """
    Aggregated projection evaluation result including validation status.
    """
    model_config = ConfigDict(extra="forbid")

    layout: ProjectedSpatialLayout
    is_valid: bool = True
    diagnostics: List[QCDiagnostic] = Field(default_factory=list)


# ============================================================================
# Spatial Coordinate Projector Engine
# ============================================================================

def _validate_finite_number(val: Any, name: str) -> float:
    """Rejects booleans, non-numerics, NaNs, and Infs."""
    if isinstance(val, bool):
        raise ValueError(f"{name} cannot be a boolean value.")
    if not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(val).__name__}.")
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{name} must be finite, got {val}.")
    return f


class SpatialCoordinateProjector:
    """
    Deterministic spatial coordinate projector for CineLearn QC.
    Maps Manim Cartesian world coordinates to raster pixel coordinates.
    """

    def __init__(self, config: Optional[ProjectionConfig] = None):
        self.config = config or ProjectionConfig()

    def project_point(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """
        Projects a 2D Cartesian world coordinate (origin at center) to pixel coordinate (origin at top-left).
        Applies vertical axis inversion.
        """
        wx = _validate_finite_number(world_x, "world_x")
        wy = _validate_finite_number(world_y, "world_y")

        cfg = self.config
        w_world = cfg.world_width
        h_world = cfg.world_height

        x_min_world = -w_world / 2.0
        y_max_world = h_world / 2.0

        # Affine mapping:
        # px = (wx - x_min_world) * (W_pixel / W_world)
        # py = (y_max_world - wy) * (H_pixel / H_world)
        px = int(round((wx - x_min_world) * cfg.scale_x))
        py = int(round((y_max_world - wy) * cfg.scale_y))

        return px, py

    def project_box(
        self,
        box: BoundingBox2D,
        scene_id: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> Tuple[Optional[PixelBox], bool, List[QCDiagnostic]]:
        """
        Projects a Cartesian world-space BoundingBox2D into an integer PixelBox.
        Handles coordinate origin differences, vertical-axis inversion, and screen boundary checks.

        Returns:
            (PixelBox or None, is_clipped_by_canvas, diagnostics)
        """
        diagnostics: List[QCDiagnostic] = []

        # Validate box integrity
        for attr in ("x", "y", "width", "height"):
            v = getattr(box, attr)
            try:
                _validate_finite_number(v, f"box.{attr}")
            except ValueError as e:
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_SCHEMA_INVALID,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_B,
                    message=f"Degenerate or non-finite geometry for entity '{entity_id}': {e}",
                    scene_id=scene_id,
                    entity_id=entity_id,
                )
                return None, False, [diag]

        if box.width <= 0.0 or box.height <= 0.0:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_SCHEMA_INVALID,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_B,
                message=f"Degenerate zero or negative box dimensions for entity '{entity_id}': width={box.width}, height={box.height}.",
                scene_id=scene_id,
                entity_id=entity_id,
            )
            return None, False, [diag]

        cfg = self.config
        w_world = cfg.world_width
        h_world = cfg.world_height
        x_min_world = -w_world / 2.0
        y_max_world = h_world / 2.0

        # World bounds
        w_left = box.left
        w_right = box.right
        w_bottom = box.bottom
        w_top = box.top

        # Convert to continuous pixel coordinates
        raw_px_min = (w_left - x_min_world) * cfg.scale_x
        raw_px_max = (w_right - x_min_world) * cfg.scale_x
        # Vertical inversion: top has smaller pixel Y, bottom has larger pixel Y
        raw_py_min = (y_max_world - w_top) * cfg.scale_y
        raw_py_max = (y_max_world - w_bottom) * cfg.scale_y

        is_clipped = False

        # Check canvas boundary breaches
        if (
            raw_px_min < 0.0
            or raw_py_min < 0.0
            or raw_px_max > cfg.frame_width
            or raw_py_max > cfg.frame_height
        ):
            is_clipped = True

        # Check completely off-screen geometry
        if (
            raw_px_max <= 0.0
            or raw_px_min >= cfg.frame_width
            or raw_py_max <= 0.0
            or raw_py_min >= cfg.frame_height
        ):
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_LAYOUT_BOUNDARY_BREACH,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_B,
                message=(
                    f"Entity '{entity_id}' is completely outside the visible canvas "
                    f"(pixels=[{raw_px_min:.1f}, {raw_py_min:.1f}, {raw_px_max:.1f}, {raw_py_max:.1f}] "
                    f"vs frame={cfg.frame_width}x{cfg.frame_height})."
                ),
                scene_id=scene_id,
                entity_id=entity_id,
            )
            diagnostics.append(diag)
            return None, True, diagnostics

        # Integer rounding and clamping
        if cfg.allow_partial_clipping and is_clipped:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_LAYOUT_BOUNDARY_BREACH,
                level=QCDiagnosticLevel.WARNING,
                layer=QCLayer.LAYER_B,
                message=(
                    f"Entity '{entity_id}' partially clipped by canvas edge: "
                    f"raw pixels=[{raw_px_min:.1f}, {raw_py_min:.1f}, {raw_px_max:.1f}, {raw_py_max:.1f}]."
                ),
                scene_id=scene_id,
                entity_id=entity_id,
            )
            diagnostics.append(diag)
            px_min = max(0, int(round(raw_px_min)))
            px_max = min(cfg.frame_width, int(round(raw_px_max)))
            py_min = max(0, int(round(raw_py_min)))
            py_max = min(cfg.frame_height, int(round(raw_py_max)))
        else:
            px_min = int(round(raw_px_min))
            px_max = int(round(raw_px_max))
            py_min = int(round(raw_py_min))
            py_max = int(round(raw_py_max))

        # Enforce minimum 1-pixel dimensions to satisfy PixelBox contract
        if px_max <= px_min:
            px_max = px_min + 1
        if py_max <= py_min:
            py_max = py_min + 1

        try:
            pixel_box = PixelBox(
                x_min=px_min,
                y_min=py_min,
                x_max=px_max,
                y_max=py_max,
            )
        except Exception as e:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_SCHEMA_INVALID,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_B,
                message=f"Failed to construct PixelBox for entity '{entity_id}': {e}",
                scene_id=scene_id,
                entity_id=entity_id,
            )
            return None, is_clipped, [diag]

        return pixel_box, is_clipped, diagnostics

    def project_entity(
        self,
        geometry: ComputedEntityGeometry,
        scene_id: Optional[str] = None,
        viewport_box: Optional[BoundingBox2D] = None,
    ) -> Tuple[Optional[ProjectedEntityGeometry], List[QCDiagnostic]]:
        """
        Projects a ComputedEntityGeometry into pixel space.
        Inspects viewport containment if viewport_box is provided.
        """
        eid = geometry.entity_id
        pixel_box, is_canvas_clipped, diags = self.project_box(
            box=geometry.world_box,
            scene_id=scene_id,
            entity_id=eid,
        )

        if pixel_box is None:
            return None, diags

        # Viewport boundary check
        is_vp_clipped = False
        if viewport_box is not None:
            # Check if entity world box is within viewport box (with 0.05 tolerance)
            if not viewport_box.contains_box(geometry.world_box, tolerance=0.05):
                is_vp_clipped = True
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_LAYOUT_BOUNDARY_BREACH,
                    level=QCDiagnosticLevel.WARNING,
                    layer=QCLayer.LAYER_B,
                    message=f"Entity '{eid}' extends beyond assigned viewport '{geometry.viewport_id}' boundaries.",
                    scene_id=scene_id,
                    entity_id=eid,
                )
                diags.append(diag)

        proj = ProjectedEntityGeometry(
            entity_id=eid,
            scene_id=scene_id,
            viewport_id=geometry.viewport_id,
            parent_id=geometry.parent_id,
            world_box=geometry.world_box,
            pixel_box=pixel_box,
            z_index=geometry.z_index,
            layer_role=geometry.layer_role,
            is_clipped_by_canvas=is_canvas_clipped,
            is_clipped_by_viewport=is_vp_clipped,
        )
        return proj, diags

    def project_layout_at_time(
        self,
        layout: SceneSpatialLayout,
        timestamp: float,
        ledger: Optional[SpatialLedger] = None,
    ) -> SpatialProjectionResult:
        """
        Projects an entire SceneSpatialLayout into pixel space at a specific timestamp.
        Filters active entities using ledger lifespan intervals or layout metadata.
        Computes pairwise geometric bounding-box intersections.
        """
        t = _validate_finite_number(timestamp, "timestamp")
        scene_id = layout.scene_id
        all_diags: List[QCDiagnostic] = []

        projected_map: Dict[str, ProjectedEntityGeometry] = {}
        active_ids: List[str] = []
        inactive_ids: List[str] = []

        # Determine active entities at timestamp t
        active_entities_set: Optional[Set[str]] = None
        intervals_by_entity: Dict[str, List[Tuple[float, float]]] = {}

        if ledger is not None:
            time_slice = ledger.query_time_slice(t)
            active_entities_set = set(time_slice.active_entities)
            for eid, span in ledger.spans.items():
                intervals_by_entity[eid] = span.intervals if span.intervals else [(span.t_start, span.t_end)]

        for eid, geom in layout.entities.items():
            # Check lifecycle active status
            is_active = True
            if active_entities_set is not None:
                is_active = eid in active_entities_set
            elif geom.entity_id in intervals_by_entity:
                ints = intervals_by_entity[geom.entity_id]
                is_active = any(t0 <= t <= t1 for t0, t1 in ints)

            if not is_active:
                inactive_ids.append(eid)
                continue

            active_ids.append(eid)

            # Resolve viewport box
            vp_box = None
            if geom.viewport_id in layout.viewports:
                vp_box = layout.viewports[geom.viewport_id].world_box

            proj, ent_diags = self.project_entity(geom, scene_id=scene_id, viewport_box=vp_box)
            all_diags.extend(ent_diags)

            if proj is not None:
                if eid in intervals_by_entity:
                    proj.active_intervals = intervals_by_entity[eid]
                projected_map[eid] = proj

        # Compute pairwise geometric bounding-box intersections
        overlaps = self._compute_geometric_overlaps(projected_map)

        proj_layout = ProjectedSpatialLayout(
            scene_id=scene_id,
            timestamp=t,
            resolution=(self.config.frame_width, self.config.frame_height),
            projected_entities=projected_map,
            active_entity_ids=sorted(active_ids),
            inactive_entity_ids=sorted(inactive_ids),
            geometric_overlaps=overlaps,
            diagnostics=all_diags,
        )

        is_valid = not any(d.level == QCDiagnosticLevel.ERROR for d in all_diags)

        return SpatialProjectionResult(
            layout=proj_layout,
            is_valid=is_valid,
            diagnostics=all_diags,
        )

    def project_ledger_at_time(
        self,
        ledger: SpatialLedger,
        timestamp: float,
        scene_id: str = "scene",
    ) -> SpatialProjectionResult:
        """
        Projects entities directly from a populated SpatialLedger at timestamp t.
        """
        t = _validate_finite_number(timestamp, "timestamp")
        time_slice = ledger.query_time_slice(t)
        all_diags: List[QCDiagnostic] = []
        projected_map: Dict[str, ProjectedEntityGeometry] = {}
        active_ids = list(time_slice.active_entities)
        inactive_ids = [eid for eid in ledger.spans.keys() if eid not in active_ids]

        for eid in active_ids:
            span = ledger.spans[eid]
            p_box, is_clipped, diags = self.project_box(span.box, scene_id=scene_id, entity_id=eid)
            all_diags.extend(diags)

            if p_box is not None:
                proj = ProjectedEntityGeometry(
                    entity_id=eid,
                    scene_id=scene_id,
                    viewport_id=span.viewport_id,
                    parent_id=span.parent_id,
                    world_box=span.box,
                    pixel_box=p_box,
                    z_index=span.z_index,
                    layer_role=span.layer_role,
                    is_clipped_by_canvas=is_clipped,
                    active_intervals=span.intervals if span.intervals else [(span.t_start, span.t_end)],
                )
                projected_map[eid] = proj

        overlaps = self._compute_geometric_overlaps(projected_map)

        proj_layout = ProjectedSpatialLayout(
            scene_id=scene_id,
            timestamp=t,
            resolution=(self.config.frame_width, self.config.frame_height),
            projected_entities=projected_map,
            active_entity_ids=sorted(active_ids),
            inactive_entity_ids=sorted(inactive_ids),
            geometric_overlaps=overlaps,
            diagnostics=all_diags,
        )

        is_valid = not any(d.level == QCDiagnosticLevel.ERROR for d in all_diags)

        return SpatialProjectionResult(
            layout=proj_layout,
            is_valid=is_valid,
            diagnostics=all_diags,
        )

    def _compute_geometric_overlaps(
        self,
        entities: Dict[str, ProjectedEntityGeometry],
    ) -> List[GeometricOverlapRecord]:
        """
        Detects pairwise geometric intersections between projected PixelBoxes.
        Documents provenance and preserves distinction from verified object collision.
        """
        records: List[GeometricOverlapRecord] = []
        keys = sorted(entities.keys())

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                id_a = keys[i]
                id_b = keys[j]
                ea = entities[id_a]
                eb = entities[id_b]
                ba = ea.pixel_box
                bb = eb.pixel_box

                # Compute integer pixel intersection
                ix0 = max(ba.x_min, bb.x_min)
                iy0 = max(ba.y_min, bb.y_min)
                ix1 = min(ba.x_max, bb.x_max)
                iy1 = min(ba.y_max, bb.y_max)

                if ix1 > ix0 and iy1 > iy0:
                    inter_box = PixelBox(x_min=ix0, y_min=iy0, x_max=ix1, y_max=iy1)
                    i_area = inter_box.area
                    ratio_a = round(i_area / ba.area, 4) if ba.area > 0 else 0.0
                    ratio_b = round(i_area / bb.area, 4) if bb.area > 0 else 0.0

                    rec = GeometricOverlapRecord(
                        entity_a=id_a,
                        entity_b=id_b,
                        pixel_box_a=ba,
                        pixel_box_b=bb,
                        intersection_box=inter_box,
                        intersection_area_pixels=i_area,
                        overlap_ratio_a=ratio_a,
                        overlap_ratio_b=ratio_b,
                    )
                    records.append(rec)

        return records
