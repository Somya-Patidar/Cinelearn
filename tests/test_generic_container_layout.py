"""
Unit Tests for CineLearn Phase 1: Generic Container Layout Engine
=================================================================
Validates domain-independent bounded viewports, spatial docking,
clamping guarantees, and zero-overlap between primary visual canvas and inspector.
"""

import pytest
import numpy as np
from manim import Square, Rectangle, Circle, Text, VGroup

from src.domains.layout import RectangularViewport, GenericContainerLayout
from src.domains import VisualComponent, ProceduralComponent


class TestRectangularViewport:
    """Test suite for RectangularViewport boundary and clamping mechanics."""

    def test_viewport_initialization(self):
        vp = RectangularViewport("test_box", left=-2.0, right=2.0, bottom=-1.0, top=1.0)
        assert vp.name == "test_box"
        assert vp.width == 4.0
        assert vp.height == 2.0
        np.testing.assert_allclose(vp.center, np.array([0.0, 0.0, 0.0]))

    def test_invalid_viewport_bounds_raise(self):
        with pytest.raises(ValueError, match="left .* must be < right"):
            RectangularViewport("bad_x", left=2.0, right=-2.0, bottom=0.0, top=1.0)

        with pytest.raises(ValueError, match="bottom .* must be < top"):
            RectangularViewport("bad_y", left=0.0, right=2.0, bottom=3.0, top=1.0)

    def test_contains_point(self):
        vp = RectangularViewport("box", left=-1.0, right=1.0, bottom=-1.0, top=1.0)
        assert vp.contains_point(np.array([0.0, 0.0, 0.0]))
        assert vp.contains_point(np.array([1.0, 1.0, 0.0]))
        assert not vp.contains_point(np.array([1.5, 0.0, 0.0]))
        assert not vp.contains_point(np.array([0.0, -1.5, 0.0]))

    def test_clamp_mobject_shifts_inside(self):
        vp = RectangularViewport("box", left=-2.0, right=2.0, bottom=-1.0, top=1.0)
        # Create an object initially placed far to the top-right
        sq = Square(side_length=0.5).move_to(np.array([5.0, 5.0, 0.0]))
        vp.clamp_mobject(sq)
        # After clamping, object center and bounds must be within viewport
        assert sq.get_right()[0] <= vp.right + 0.01
        assert sq.get_top()[1] <= vp.top + 0.01

    def test_clamp_oversized_mobject_scales_down(self):
        vp = RectangularViewport("box", left=-1.0, right=1.0, bottom=-1.0, top=1.0)
        # Create an oversized object (10.0 x 10.0)
        sq = Square(side_length=10.0)
        vp.clamp_mobject(sq)
        assert sq.width <= vp.width
        assert sq.height <= vp.height


class TestGenericContainerLayout:
    """Test suite for GenericContainerLayout viewports and docking."""

    @pytest.fixture
    def layout(self):
        return GenericContainerLayout()

    def test_viewports_exist(self, layout):
        expected_viewports = ["header", "main_canvas", "inspector", "split_left", "split_right", "full_canvas"]
        for name in expected_viewports:
            assert name in layout.viewports
            assert isinstance(layout.viewports[name], RectangularViewport)

    def test_safe_zone_invariants(self, layout):
        """
        Critical safe buffer invariants:
        - Subtitles are at Y < -2.75.
        - The inspector must NEVER extend below Y = -2.65.
        - Active canvas top must not exceed Y = 2.55 (title buffer safe zone).
        """
        assert layout.inspector.bottom >= -2.65
        assert layout.main_canvas.top <= 2.55
        assert layout.full_canvas.bottom >= -2.65
        assert layout.full_canvas.top <= 2.55

    def test_disjoint_viewports_no_overlap(self, layout):
        """Main canvas and inspector must be mathematically disjoint in the Y axis."""
        assert layout.inspector.top <= layout.main_canvas.bottom
        # Specifically, inspector top is -1.65, main_canvas bottom is -1.55 (0.10 gap buffer)
        gap = layout.main_canvas.bottom - layout.inspector.top
        assert gap == pytest.approx(0.10, rel=1e-3)

    def test_split_viewports_no_overlap(self, layout):
        """Split left and split right must be mathematically disjoint in the X axis."""
        assert layout.split_left.right <= layout.split_right.left
        gap = layout.split_right.left - layout.split_left.right
        assert gap >= 0.20

    def test_dock_main(self, layout):
        obj = Square(side_length=2.0)
        layout.dock_main(obj)
        box = layout.get_bounding_box(obj)
        assert box["left"] >= layout.main_canvas.left - 0.05
        assert box["right"] <= layout.main_canvas.right + 0.05
        assert box["bottom"] >= layout.main_canvas.bottom - 0.05
        assert box["top"] <= layout.main_canvas.top + 0.05

    def test_dock_inspector_strict_subtitle_buffer(self, layout):
        """An oversized table or component docked to inspector must NEVER violate Y >= -2.65."""
        table_mock = Rectangle(width=12.0, height=8.0)
        layout.dock_inspector(table_mock)
        box = layout.get_bounding_box(table_mock)
        assert box["bottom"] >= -2.65
        assert box["top"] <= layout.inspector.top + 0.05
        assert box["height"] <= layout.inspector.height + 0.05

    def test_dock_split_left_and_right(self, layout):
        left_obj = Circle(radius=1.5)
        right_obj = Square(side_length=3.0)
        layout.dock_split_left(left_obj)
        layout.dock_split_right(right_obj)

        has_overlap, ratio = layout.check_overlap(left_obj, right_obj)
        assert not has_overlap
        assert ratio == 0.0

    def test_dock_main_and_inspector_zero_overlap(self, layout):
        """Simultaneous primary visual in main canvas and status table in inspector must have 0.0 overlap."""
        main_visual = Rectangle(width=6.0, height=3.5)
        inspector_table = Rectangle(width=6.0, height=1.0)

        layout.dock_main(main_visual)
        layout.dock_inspector(inspector_table)

        has_overlap, ratio = layout.check_overlap(main_visual, inspector_table)
        assert not has_overlap
        assert ratio == 0.0

    def test_dock_works_with_procedural_component(self, layout):
        """Phase 1 layout must seamlessly support Tier 3 ProceduralComponent."""
        group = VGroup(Circle(radius=0.8), Text("Step 1", font_size=18))
        proc = ProceduralComponent(group, name="ProceduralStep")
        layout.dock_main(proc)
        assert proc.get_top()[1] <= layout.main_canvas.top + 0.05
        assert proc.get_bottom()[1] >= layout.main_canvas.bottom - 0.05

    def test_dock_registers_with_scene(self):
        class MockScene:
            def __init__(self):
                self.canvas_mobjects = []

        scene = MockScene()
        layout = GenericContainerLayout(scene)
        sq = Square()
        layout.dock_main(sq)
        assert sq in scene.canvas_mobjects


class TestTeachingSceneIntegration:
    """Verifies that TeachingScene in base_class integrates GenericContainerLayout and clamps correctly."""

    def test_teaching_scene_has_layout_and_docking(self):
        from prompts import base_class
        scope = {}
        exec(base_class, scope)
        TeachingSceneCls = scope["TeachingScene"]

        scene = TeachingSceneCls()
        scene.setup_layout(
            title_text="Dijkstra's Algorithm: Shortest Paths",
            lecture_lines=["Initialize distances to infinity", "Select minimum tentative distance"]
        )

        assert hasattr(scene, "layout")
        assert isinstance(scene.layout, scope["GenericContainerLayout"])
        assert hasattr(scene, "dock_main")
        assert hasattr(scene, "dock_inspector")

        # Test dock_inspector from scene
        mock_table = Rectangle(width=5.0, height=2.0)
        scene.dock_inspector(mock_table)
        assert mock_table.get_bottom()[1] >= -2.65
        assert mock_table in scene.canvas_mobjects

    def test_place_at_grid_row_f_clamping(self):
        """Grid position F4 (bottom row) must clamp to Y >= -2.65 to protect subtitles."""
        from prompts import base_class
        scope = {}
        exec(base_class, scope)
        TeachingSceneCls = scope["TeachingScene"]

        scene = TeachingSceneCls()
        scene.setup_layout("Title", ["Lecture line 1"])

        sq = Square(side_length=1.0)
        scene.place_at_grid(sq, "F4")
        assert sq.get_bottom()[1] >= -2.65

    def test_rendered_frame_pixel_separation(self):
        """
        Validates intermediate rendered frame pixels directly from Manim renderer.
        Confirms strict spatial separation and zero pixel overlap between
        the primary visual in main_canvas and the inspector component,
        as well as clearance from the subtitle reserved zone.
        """
        from prompts import base_class
        scope = {}
        exec(base_class, scope)
        TeachingSceneCls = scope["TeachingScene"]

        scene = TeachingSceneCls()
        scene.setup_layout("Render Verification", ["Bullet point 1", "Bullet point 2"])

        # Main visual: Blue rectangle (#58C4DD)
        main_box = Rectangle(width=5.0, height=3.0, color="#58C4DD", fill_opacity=1.0)
        # Inspector visual: Green rectangle (#83C167)
        inspector_box = Rectangle(width=6.0, height=0.8, color="#83C167", fill_opacity=1.0)

        scene.dock_main(main_box)
        scene.dock_inspector(inspector_box)
        scene.add(main_box, inspector_box)

        # Render frame in memory
        scene.renderer.update_frame(scene)
        frame = scene.renderer.get_frame()  # Shape: (1080, 1920, 4)
        h, w, _ = frame.shape

        # Detect Blue component pixels: R < 120, G > 150, B > 180
        blue_mask = (frame[:, :, 0] < 120) & (frame[:, :, 1] > 150) & (frame[:, :, 2] > 180)
        # Detect Green component pixels: R > 100, G > 150, B < 130
        green_mask = (frame[:, :, 0] > 100) & (frame[:, :, 1] > 150) & (frame[:, :, 2] < 130)

        blue_ys, _ = np.where(blue_mask)
        green_ys, _ = np.where(green_mask)

        assert len(blue_ys) > 0, "Main canvas component was not rendered."
        assert len(green_ys) > 0, "Inspector component was not rendered."

        # Verify zero pixel overlap: main component bottom must be above inspector top
        main_bottom_px = blue_ys.max()
        inspector_top_px = green_ys.min()
        vertical_gap_px = inspector_top_px - main_bottom_px

        assert vertical_gap_px > 50, f"Insufficient vertical pixel separation: {vertical_gap_px}px"

        # Verify subtitle clearance: inspector bottom must be well above subtitle zone (Y > 930)
        inspector_bottom_px = green_ys.max()
        assert inspector_bottom_px < 920, f"Inspector encroaches on subtitle zone at Y={inspector_bottom_px}"

