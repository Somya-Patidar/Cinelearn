"""
Tests for Mathematics Domain Visual Components
==============================================
Validates FunctionCurve and NumberLineVisual functionality, semantic handles,
animation methods, container docking, zero overlap, and subtitle clearance.
"""

import sys
from pathlib import Path

# Ensure CineLearn root is in sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
import numpy as np
from manim import Scene, tempconfig, FadeIn

from src.domains import VisualRegistry
from src.domains.math import FunctionCurve, NumberLineVisual
from prompts import base_class

# Dynamically extract TeachingScene from base_class template
_scope = {}
exec(base_class.base_class if hasattr(base_class, "base_class") else base_class, _scope)
TeachingScene = _scope["TeachingScene"]


@pytest.fixture(autouse=True)
def ensure_math_discovered():
    VisualRegistry.discover_domains("src.domains")


class TestFunctionCurve:
    def test_registration(self):
        cls = VisualRegistry.get("math", "FunctionCurve")
        assert cls is FunctionCurve
        manifest = FunctionCurve.get_manifest()
        assert manifest["component_id"] == "FunctionCurve"
        assert manifest["domain"] == "math"
        assert "animate_plot_curve" in manifest["semantic_actions"]
        assert "animate_tangent" in manifest["semantic_actions"]
        assert "animate_shade_area" in manifest["semantic_actions"]
        assert "animate_riemann_rectangles" in manifest["semantic_actions"]

    def test_preset_functions(self):
        # 1. Polynomial cubic: x^3 - 3x
        cubic = FunctionCurve("cubic")
        assert cubic.get_semantic_handle("axes") is not None
        assert cubic.get_semantic_handle("curve") is not None
        assert cubic.get_semantic_handle("curve_label") is not None

        # 2. Trigonometric: sin(x)
        trig = FunctionCurve("sin(x)")
        assert trig.get_semantic_handle("curve") is not None

        # 3. Exponential: exp(x)
        expo = FunctionCurve("exp(x)")
        assert expo.get_semantic_handle("curve") is not None

    def test_custom_expression_and_callable(self):
        # String custom expression
        poly = FunctionCurve(function="x**2 - 2*x - 3", roots=[-1.0, 3.0])
        assert poly.get_semantic_handle("curve") is not None

        # Callable lambda
        gauss = FunctionCurve(function=lambda x: np.exp(-x**2), curve_label="Normal Distribution")
        assert gauss.get_semantic_handle("curve") is not None

    def test_secondary_function_and_intersection(self):
        # f(x) = x^2 - 1, g(x) = x + 1
        system = FunctionCurve(
            function="x**2 - 1",
            secondary_function="x + 1",
        )
        assert system.get_semantic_handle("secondary_curve") is not None

        anim_inter = system.animate_show_intersection()
        assert anim_inter is not None
        # Root of x^2 - x - 2 = 0 are x = -1 and x = 2
        assert system.get_semantic_handle("intersection_0") is not None

    def test_semantic_animations(self):
        curve_comp = FunctionCurve("cubic")

        anim_plot = curve_comp.animate_plot_curve()
        assert anim_plot is not None

        anim_tan = curve_comp.animate_tangent(x_val=1.0)
        assert anim_tan is not None
        assert curve_comp.get_semantic_handle("tangent_line") is not None
        assert curve_comp.get_semantic_handle("tangent_point") is not None

        anim_roots = curve_comp.animate_show_roots()
        assert anim_roots is not None

        anim_extrema = curve_comp.animate_show_extrema()
        assert anim_extrema is not None

        anim_area = curve_comp.animate_shade_area(0.0, 2.0)
        assert anim_area is not None
        assert curve_comp.get_semantic_handle("area") is not None

        anim_riemann = curve_comp.animate_riemann_rectangles(0.0, 2.0, dx=0.5)
        assert anim_riemann is not None
        assert curve_comp.get_semantic_handle("riemann_rectangles") is not None

        anim_trans = curve_comp.animate_transform_function("x**2")
        assert anim_trans is not None


class TestNumberLineVisual:
    def test_registration(self):
        cls = VisualRegistry.get("math", "NumberLineVisual")
        assert cls is NumberLineVisual
        manifest = NumberLineVisual.get_manifest()
        assert manifest["component_id"] == "NumberLineVisual"
        assert manifest["domain"] == "math"
        assert "animate_show_interval" in manifest["semantic_actions"]
        assert "animate_add_pointer" in manifest["semantic_actions"]

    def test_number_line_structure_and_intervals(self):
        nl = NumberLineVisual(
            x_range=(-4.0, 4.0, 1.0),
            intervals=[(-2.0, 3.0, True, False)],  # [-2, 3)
        )
        assert nl.get_semantic_handle("line") is not None
        assert nl.get_semantic_handle("ticks") is not None
        assert nl.get_semantic_handle("numbers") is not None
        assert nl.get_semantic_handle("interval_0") is not None

    def test_coordinate_mapping(self):
        nl = NumberLineVisual(x_range=(-5.0, 5.0, 1.0), length=6.0)
        pt_zero = nl.n2p(0.0)
        np.testing.assert_allclose(pt_zero[0], 0.0, atol=1e-3)

        pt_neg5 = nl.n2p(-5.0)
        np.testing.assert_allclose(pt_neg5[0], -3.0, atol=1e-3)

        pt_pos5 = nl.n2p(5.0)
        np.testing.assert_allclose(pt_pos5[0], 3.0, atol=1e-3)

    def test_semantic_animations(self):
        nl = NumberLineVisual(x_range=(-5.0, 5.0, 1.0))

        anim_high = nl.animate_highlight_number(2.0)
        assert anim_high is not None

        anim_int = nl.animate_show_interval(0.0, 4.0, closed_start=True, closed_end=True)
        assert anim_int is not None

        anim_ptr = nl.animate_add_pointer("x0", 1.5)
        assert anim_ptr is not None

        anim_mov = nl.animate_move_pointer("x0", 3.0)
        assert anim_mov is not None

        anim_lim = nl.animate_approach_limit(target_val=2.0, start_val=-3.0)
        assert anim_lim is not None


class TestMathDockingAndRendering:
    def test_function_curve_and_number_line_composition_docking(self):
        """
        Tests composing FunctionCurve in dock_main and NumberLineVisual in dock_inspector,
        verifying zero overlap and subtitle clearance.
        """
        scene = TeachingScene()
        scene.setup_layout(
            "Calculus: Derivatives and Tangent Slopes",
            [
                "The derivative represents the instantaneous rate of change",
                "Graphically, it is the slope of the tangent line at x",
                "The domain constraint x in [-2, 2] bounds the region of interest",
            ],
        )

        # 1. Main visual: FunctionCurve
        curve = FunctionCurve(function="x**3 - 3*x", x_range=(-3.0, 3.0, 1.0), y_range=(-4.0, 4.0, 1.0))
        scene.dock_main(curve)

        # 2. Inspector visual: NumberLineVisual (showing domain interval [-2, 2])
        nl = NumberLineVisual(x_range=(-3.0, 3.0, 1.0), intervals=[(-2.0, 2.0, True, True)])
        scene.dock_inspector(nl)

        # Verify zero overlap between curve and number line
        bb_curve = curve.get_bounding_box()
        bb_nl = nl.get_bounding_box()

        has_overlap, ratio = scene.layout.check_overlap(curve, nl)
        assert not has_overlap, f"Overlap detected between curve and number line! Ratio: {ratio}"
        assert ratio == 0.0

        # Verify vertical separation: curve bottom is strictly above number line top
        assert bb_curve["bottom"] > bb_nl["top"], (
            f"Curve bottom ({bb_curve['bottom']}) must be strictly above number line top ({bb_nl['top']})"
        )

        # Verify subtitle clearance: bottom of number line must be >= -2.65
        assert bb_nl["bottom"] >= -2.65, (
            f"Number line bottom ({bb_nl['bottom']}) encroaches on subtitle area (below -2.65)"
        )

        # Verify safe boundaries
        assert bb_curve["left"] >= -2.15
        assert bb_nl["left"] >= -2.15
        assert bb_curve["right"] <= 6.65
        assert bb_nl["right"] <= 6.65

    def test_render_scene_with_math_components(self, tmp_path):
        """
        Renders a Manim scene exercising FunctionCurve and NumberLineVisual animations.
        """
        class SampleMathScene(TeachingScene):
            def construct(self):
                self.setup_layout(
                    "Derivative & Tangent Line Demonstration",
                    [
                        "Plot the cubic function f(x) = x^3 - 3x",
                        "Evaluate local extrema and construct the tangent line at x = 1",
                    ],
                )

                curve = FunctionCurve(function="cubic", x_range=(-3.0, 3.0, 1.0))
                self.dock_main(curve)

                nl = NumberLineVisual(x_range=(-3.0, 3.0, 1.0))
                self.dock_inspector(nl)

                self.play(FadeIn(curve), FadeIn(nl))
                self.play(self.activate_caption(0))
                self.play(curve.animate_plot_curve())
                self.play(self.activate_caption(1))
                self.play(
                    curve.animate_tangent(x_val=1.0, color="#FFD700"),
                    nl.animate_add_pointer("x=1", 1.0),
                )

        with tempconfig({
            "quality": "low_quality",
            "preview": False,
            "format": "mp4",
            "media_dir": str(tmp_path),
            "disable_caching": True,
        }):
            scene = SampleMathScene()
            scene.render()

        rendered_files = list(tmp_path.glob("**/*.mp4"))
        assert len(rendered_files) > 0, "Expected at least one rendered MP4 file"
        assert rendered_files[0].stat().st_size > 1000, "Rendered video file is too small or corrupt"
