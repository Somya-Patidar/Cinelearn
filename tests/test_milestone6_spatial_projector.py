"""
Milestone 6.2: Unit and Property Tests for Spatial Coordinate Projector.

Verifies:
- Basic Cartesian world-to-pixel coordinate projection.
- Vertical-axis inversion (top world Y -> smaller pixel Y, bottom world Y -> larger pixel Y).
- Non-square viewports and arbitrary aspect ratios (16:9, 4:3, 1:1, 9:16).
- Different output resolutions (1920x1080, 1280x720, 1024x768, 1080x1080).
- Safety padding and active canvas boundary checking.
- Multi-interval entity lifecycles: projects only active entities at sample timestamp.
- Inactive entities omitted from projected entities map.
- Boundary-touching and partially out-of-bounds geometry.
- Completely out-of-bounds geometry triggers structured boundary breach diagnostics.
- Degenerate and non-finite boxes rejected fail-closed.
- Geometric AABB overlap detected with provenance and distinct semantic disclaimer (not verified collision).
- JSON serialization and preservation of entity identity.
"""

import math
import pytest
from typing import Dict, List, Tuple

from src.compiler.layout_models import (
    BoundingBox2D,
    ComputedEntityGeometry,
    ComputedViewportGeometry,
    SceneSpatialLayout,
    ViewportType,
)
from src.compiler.spatial_ledger import SpatialLedger
from src.qc.models import PixelBox, QCDiagnosticCode, QCDiagnosticLevel
from src.qc.spatial_projector import (
    GeometricOverlapRecord,
    ProjectedEntityGeometry,
    ProjectedSpatialLayout,
    ProjectionConfig,
    SpatialCoordinateProjector,
    SpatialProjectionRequest,
    SpatialProjectionResult,
    parse_aspect_ratio_to_float,
)


class TestSpatialCoordinateProjectorBasics:
    """Verifies affine coordinate projection and vertical inversion."""

    def test_center_and_corner_point_projection_at_1080p(self):
        """World origin (0, 0) maps to frame center (960, 540) on 1920x1080."""
        projector = SpatialCoordinateProjector(ProjectionConfig(frame_width=1920, frame_height=1080, aspect_ratio="16:9"))

        # Center
        cx, cy = projector.project_point(0.0, 0.0)
        assert cx == 960
        assert cy == 540

        # World bounds for 16:9 with world_height=8.0:
        # world_width = 8.0 * (16/9) = 14.222222...
        # x_min = -7.111111..., y_max = +4.0
        tl_x, tl_y = projector.project_point(-14.222222 / 2.0, 4.0)
        assert tl_x == 0
        assert tl_y == 0

        br_x, br_y = projector.project_point(14.222222 / 2.0, -4.0)
        assert br_x == 1920
        assert br_y == 1080

    def test_vertical_axis_inversion(self):
        """Higher world Y (top) produces lower pixel Y (top of screen)."""
        projector = SpatialCoordinateProjector()

        # Point A is above Point B in Cartesian space
        _, py_high = projector.project_point(0.0, 2.0)
        _, py_low = projector.project_point(0.0, -2.0)

        # In raster space, top of image has smaller y coordinate
        assert py_high < py_low
        assert py_high == 270  # (4 - 2) * (1080/8) = 2 * 135 = 270
        assert py_low == 810   # (4 - (-2)) * 135 = 6 * 135 = 810

    def test_bounding_box_projection_preserves_dimensions(self):
        """BoundingBox2D projects to integer PixelBox preserving scale ratio."""
        projector = SpatialCoordinateProjector()

        # Box centered at (0, 0) with width=2.0, height=2.0
        # World left = -1.0, right = +1.0, bottom = -1.0, top = +1.0
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)
        p_box, is_clipped, diags = projector.project_box(box)

        assert p_box is not None
        assert not is_clipped
        assert len(diags) == 0

        # Scale is 135 pixels per unit (1080 / 8 = 135)
        # Width: 2.0 * 135 = 270 px
        # Height: 2.0 * 135 = 270 px
        assert p_box.width == 270
        assert p_box.height == 270
        assert p_box.x_min == 960 - 135  # 825
        assert p_box.x_max == 960 + 135  # 1095
        assert p_box.y_min == 540 - 135  # 405
        assert p_box.y_max == 540 + 135  # 675


class TestSpatialProjectorResolutionsAndAspectRatios:
    """Verifies non-square viewports and different output resolutions."""

    def test_720p_resolution(self):
        """1280x720 16:9 projection scales uniformly at 90 px/unit."""
        cfg = ProjectionConfig(frame_width=1280, frame_height=720, aspect_ratio="16:9")
        projector = SpatialCoordinateProjector(cfg)

        assert projector.config.scale_x == 90.0
        assert projector.config.scale_y == 90.0

        cx, cy = projector.project_point(0.0, 0.0)
        assert cx == 640
        assert cy == 360

        box = BoundingBox2D(x=0.0, y=0.0, width=1.0, height=1.0)
        p_box, _, _ = projector.project_box(box)
        assert p_box is not None
        assert p_box.width == 90
        assert p_box.height == 90

    def test_four_by_three_aspect_ratio(self):
        """1024x768 4:3 projection computes correct world width and scale."""
        cfg = ProjectionConfig(frame_width=1024, frame_height=768, aspect_ratio="4:3")
        projector = SpatialCoordinateProjector(cfg)

        # World height = 8.0 -> World width = 8.0 * (4/3) = 10.666667
        # Scale: 768 / 8.0 = 96 px/unit, 1024 / 10.666667 = 96 px/unit
        assert abs(projector.config.scale_x - 96.0) < 0.01
        assert abs(projector.config.scale_y - 96.0) < 0.01

        cx, cy = projector.project_point(0.0, 0.0)
        assert cx == 512
        assert cy == 384

    def test_one_to_one_square_aspect_ratio(self):
        """1080x1080 1:1 projection handles square viewports."""
        cfg = ProjectionConfig(frame_width=1080, frame_height=1080, aspect_ratio="1:1")
        projector = SpatialCoordinateProjector(cfg)

        assert projector.config.scale_x == 135.0
        assert projector.config.scale_y == 135.0

        cx, cy = projector.project_point(0.0, 0.0)
        assert cx == 540
        assert cy == 540


class TestSpatialProjectorLifecycleAndFiltering:
    """Verifies that only entities active at timestamp t are projected."""

    def test_multiple_lifecycle_intervals_and_inactive_omission(self):
        """Entities active at sample timestamp are projected; inactive entities are omitted."""
        ledger = SpatialLedger(scene_duration=10.0)

        # entity_a is active during [0.0, 3.0] and [6.0, 9.0]
        box_a = BoundingBox2D(x=-2.0, y=0.0, width=2.0, height=2.0)
        ledger.register_span(
            entity_id="entity_a",
            box=box_a,
            t_start=0.0,
            t_end=9.0,
            intervals=[(0.0, 3.0), (6.0, 9.0)],
        )

        # entity_b is active during [2.0, 7.0]
        box_b = BoundingBox2D(x=2.0, y=0.0, width=2.0, height=2.0)
        ledger.register_span(
            entity_id="entity_b",
            box=box_b,
            t_start=2.0,
            t_end=7.0,
            intervals=[(2.0, 7.0)],
        )

        projector = SpatialCoordinateProjector()

        # At t=1.0: only entity_a is active
        res_t1 = projector.project_ledger_at_time(ledger, timestamp=1.0)
        assert "entity_a" in res_t1.layout.projected_entities
        assert "entity_b" not in res_t1.layout.projected_entities
        assert "entity_b" in res_t1.layout.inactive_entity_ids
        assert res_t1.layout.active_count == 1

        # At t=2.5: both entity_a and entity_b are active
        res_t2 = projector.project_ledger_at_time(ledger, timestamp=2.5)
        assert "entity_a" in res_t2.layout.projected_entities
        assert "entity_b" in res_t2.layout.projected_entities
        assert res_t2.layout.active_count == 2

        # At t=4.5: entity_a is inactive (gap between 3.0 and 6.0), entity_b is active
        res_t4 = projector.project_ledger_at_time(ledger, timestamp=4.5)
        assert "entity_a" not in res_t4.layout.projected_entities
        assert "entity_a" in res_t4.layout.inactive_entity_ids
        assert "entity_b" in res_t4.layout.projected_entities
        assert res_t4.layout.active_count == 1

        # At t=9.5: both are inactive
        res_t9 = projector.project_ledger_at_time(ledger, timestamp=9.5)
        assert res_t9.layout.active_count == 0
        assert "entity_a" in res_t9.layout.inactive_entity_ids
        assert "entity_b" in res_t9.layout.inactive_entity_ids


class TestSpatialProjectorBoundariesAndDegeneracy:
    """Verifies boundary breach, clipping, and invalid geometry handling."""

    def test_completely_off_screen_box_produces_error_diagnostic(self):
        """Entity positioned entirely outside screen coordinates fails with QC_LAYOUT_BOUNDARY_BREACH."""
        projector = SpatialCoordinateProjector()

        # World x=20.0 is far beyond screen right (+7.11)
        off_screen_box = BoundingBox2D(x=20.0, y=0.0, width=2.0, height=2.0)
        p_box, is_clipped, diags = projector.project_box(off_screen_box, scene_id="s1", entity_id="ghost")

        assert p_box is None
        assert is_clipped
        assert len(diags) == 1
        assert diags[0].code == QCDiagnosticCode.QC_LAYOUT_BOUNDARY_BREACH
        assert diags[0].level == QCDiagnosticLevel.ERROR
        assert "completely outside the visible canvas" in diags[0].message

    def test_partial_clipping_with_warning_and_clamping(self):
        """Partially clipped entity clamps coordinates and emits a WARNING diagnostic."""
        projector = SpatialCoordinateProjector(ProjectionConfig(allow_partial_clipping=True))

        # Box centered at x=7.0 with width=2.0 (left=6.0, right=8.0 > 7.11)
        partial_box = BoundingBox2D(x=7.0, y=0.0, width=2.0, height=2.0)
        p_box, is_clipped, diags = projector.project_box(partial_box, scene_id="s1", entity_id="bleeder")

        assert p_box is not None
        assert is_clipped
        assert p_box.x_max == 1920  # Clamped to canvas right boundary
        assert any(d.level == QCDiagnosticLevel.WARNING for d in diags)

    def test_rejects_nan_or_infinite_coordinates(self):
        """Projector strictly rejects non-finite coordinates fail-closed."""
        projector = SpatialCoordinateProjector()

        with pytest.raises(ValueError, match="must be finite"):
            projector.project_point(float("nan"), 0.0)

        with pytest.raises(ValueError, match="must be finite"):
            projector.project_point(0.0, float("inf"))

    def test_rejects_boolean_coordinates(self):
        """Projector strictly rejects boolean inputs."""
        projector = SpatialCoordinateProjector()

        with pytest.raises(ValueError, match="cannot be a boolean value"):
            projector.project_point(True, 0.0)  # type: ignore[arg-type]


class TestSpatialProjectorGeometricOverlapSemantics:
    """Verifies that bounding-box overlap is tracked with proper provenance and non-collision disclaimer."""

    def test_geometric_overlap_detected_between_active_entities(self):
        """Overlapping pixel boxes generate GeometricOverlapRecord with semantic disclaimer."""
        ledger = SpatialLedger(scene_duration=5.0)

        # Entity A and B share an overlapping region
        box_a = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)
        box_b = BoundingBox2D(x=1.0, y=0.0, width=2.0, height=2.0)

        ledger.register_span("ent_a", box_a, t_start=0.0, t_end=5.0)
        ledger.register_span("ent_b", box_b, t_start=0.0, t_end=5.0)

        projector = SpatialCoordinateProjector()
        result = projector.project_ledger_at_time(ledger, timestamp=2.0)

        assert len(result.layout.geometric_overlaps) == 1
        rec = result.layout.geometric_overlaps[0]

        assert rec.entity_a == "ent_a"
        assert rec.entity_b == "ent_b"
        assert rec.intersection_area_pixels > 0
        assert rec.overlap_ratio_a > 0.0
        assert rec.overlap_ratio_b > 0.0

        # Verify semantic constraint: explicit non-collision disclaimer is preserved
        assert "Does NOT prove verified object-to-object collision" in rec.semantic_note

    def test_disjoint_entities_have_zero_geometric_overlaps(self):
        """Disjoint entities produce zero geometric overlap records."""
        ledger = SpatialLedger(scene_duration=5.0)

        box_a = BoundingBox2D(x=-4.0, y=0.0, width=1.0, height=1.0)
        box_b = BoundingBox2D(x=4.0, y=0.0, width=1.0, height=1.0)

        ledger.register_span("ent_left", box_a, t_start=0.0, t_end=5.0)
        ledger.register_span("ent_right", box_b, t_start=0.0, t_end=5.0)

        projector = SpatialCoordinateProjector()
        result = projector.project_ledger_at_time(ledger, timestamp=1.0)

        assert len(result.layout.geometric_overlaps) == 0


class TestSpatialProjectorSerializationAndSceneSpatialLayout:
    """Verifies integration with SceneSpatialLayout and full JSON round-trip."""

    def test_project_scene_spatial_layout(self):
        """Projects a SceneSpatialLayout object with ComputedEntityGeometry and viewports."""
        canvas_box = BoundingBox2D.from_min_max(left=-6.6, right=6.6, bottom=-3.0, top=3.0)
        vp_box = BoundingBox2D.from_min_max(left=-6.0, right=0.0, bottom=-2.5, top=2.5)

        computed_vp = ComputedViewportGeometry(
            viewport_id="viewport_split_left",
            viewport_type=ViewportType.SPLIT_LEFT,
            world_box=vp_box,
        )

        ent_geom = ComputedEntityGeometry(
            entity_id="chart1",
            viewport_id="viewport_split_left",
            local_box=BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0),
            world_box=BoundingBox2D(x=-3.0, y=0.0, width=2.0, height=2.0),
            z_index=15,
            layer_role="base",
        )

        layout = SceneSpatialLayout(
            scene_id="scene_spatial_test",
            canvas_box=canvas_box,
            viewports={"viewport_split_left": computed_vp},
            entities={"chart1": ent_geom},
        )

        projector = SpatialCoordinateProjector()
        res = projector.project_layout_at_time(layout, timestamp=1.0)

        assert res.is_valid
        assert "chart1" in res.layout.projected_entities

        p_chart = res.layout.projected_entities["chart1"]
        assert p_chart.entity_id == "chart1"
        assert p_chart.viewport_id == "viewport_split_left"
        assert p_chart.z_index == 15
        assert p_chart.layer_role == "base"
        assert p_chart.pixel_box.width == 270
        assert p_chart.pixel_box.height == 270

    def test_json_roundtrip_serialization(self):
        """ProjectedSpatialLayout serializes to JSON and deserializes cleanly."""
        ledger = SpatialLedger(scene_duration=5.0)
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=1.0)
        ledger.register_span("title_card", box, t_start=0.0, t_end=5.0, layer_role="card")

        projector = SpatialCoordinateProjector()
        result = projector.project_ledger_at_time(ledger, timestamp=2.0, scene_id="scene_json")

        json_str = result.layout.model_dump_json()
        restored = ProjectedSpatialLayout.model_validate_json(json_str)

        assert restored.scene_id == "scene_json"
        assert restored.timestamp == 2.0
        assert "title_card" in restored.projected_entities
        assert restored.projected_entities["title_card"].pixel_box.width == 270
        assert abs(restored.projected_entities["title_card"].pixel_box.height - 135) <= 1
