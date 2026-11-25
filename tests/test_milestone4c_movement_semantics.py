"""
CineLearn — Milestone 4C.3: Geometry-Aware Movement Semantics Test Suite
========================================================================
Comprehensive verification of deterministic, geometry-aware movement animations
for ActionType.MOVE across:
1. Valid movement operations (absolute destination, relative shift, sub-slot, viewport, pointer).
2. Viewport boundary validation and out-of-bounds detection (COMPILER_OUT_OF_BOUNDS).
3. Robust error handling and structured diagnostics (fail-closed, missing geometry, invalid params).
4. Sequential and repeated movements (cumulative position tracking across timeline events).
5. Easing function mapping (linear, ease_in, ease_out, smooth).
6. Determinism, AST safety, and valid Manim Python source generation.
7. Fixture compatibility and spatial ledger integration.
"""

import ast
import math
import numpy as np
import pytest
from typing import Dict, Any

from manim import Rectangle, linear, rate_functions

from src.compiler import (
    BoundingBox2D,
    CompilationDiagnostic,
    CompilationErrorCode,
    CompiledSceneOutput,
    DeterministicSceneCompiler,
    HierarchicalBoxModelLayoutSolver,
    SceneSpatialLayout,
)
from src.ir.fixtures import (
    get_binary_search_fixture,
    get_chemical_kinetics_fixture,
    get_pythagorean_theorem_fixture,
)
from src.ir.models import (
    ActionDeclaration,
    ActionType,
    Alignment,
    EasingType,
    EntityDeclaration,
    FlowDirection,
    LayoutConfiguration,
    NarrationAnchor,
    SceneIR,
    TimelineEvent,
    ViewportDeclaration,
    ViewportType,
    VisualTier,
)


def build_manim_animation_from_source(
    python_source: str,
    entity_var: str = "ent_box_1",
) -> Any:
    """
    Parses python_source, extracts the animation call passed to self.play(...),
    and executes it against real Manim CE runtime objects to construct and return
    the built Animation object via .build().
    Fails if the generated code raises AttributeError, TypeError, or construction errors.
    """
    tree = ast.parse(python_source)
    play_call = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "play"
        ):
            if ".animate" in ast.unparse(node):
                play_call = node
                break
    assert play_call is not None, "No self.play(...) movement call with .animate found in generated Python source."

    context = {
        entity_var: Rectangle(width=1.0, height=1.0),
        "np": np,
        "linear": linear,
        "rate_functions": rate_functions,
    }

    # Evaluate any keyword arguments (e.g. run_time, rate_func) in Manim runtime context
    for kw in play_call.keywords:
        eval(compile(ast.Expression(body=kw.value), "<test_kw>", "eval"), context)

    anim_expr = compile(ast.Expression(body=play_call.args[0]), "<test_anim>", "eval")
    builder = eval(anim_expr, context)
    return builder.build()


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def compiler() -> DeterministicSceneCompiler:
    return DeterministicSceneCompiler()


@pytest.fixture
def solver() -> HierarchicalBoxModelLayoutSolver:
    return HierarchicalBoxModelLayoutSolver()


@pytest.fixture
def movable_box_scene() -> SceneIR:
    """Scene containing a movable primitive box inside a main viewport."""
    return SceneIR(
        scene_id="MoveTestScene",
        title="Move Test Scene",
        duration_budget=12.0,
        layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
        viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
        entities=[
            EntityDeclaration(
                id="box_1",
                component_type="PrimitiveShape",
                visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                viewport="vp_main",
                initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0, "fill_color": "#336699"},
            )
        ],
        timeline=[
            TimelineEvent(
                step_id="step_1",
                narration_anchor=NarrationAnchor(
                    lecture_line_index=0,
                    text_cue="Introducing the box.",
                    audio_timestamp_range=(0.0, 3.0),
                ),
                actions=[
                    ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0),
                ],
            ),
            TimelineEvent(
                step_id="step_2",
                narration_anchor=NarrationAnchor(
                    lecture_line_index=1,
                    text_cue="Moving the box to the right.",
                    audio_timestamp_range=(3.0, 7.0),
                ),
                actions=[
                    ActionDeclaration(
                        type=ActionType.MOVE,
                        target_entity="box_1",
                        duration=1.5,
                        parameters={"shift": [1.5, 0.5]},
                    ),
                ],
            ),
        ],
    )


# =====================================================================
# 1. Valid Movement Compilation
# =====================================================================

class TestValidMovementCompilation:
    def test_relative_shift_list(self, compiler, solver, movable_box_scene):
        """Relative shift vector as list compiles to .animate.shift(np.array([dx, dy, 0.0]))."""
        movable_box_scene.timeline[1].actions[0].parameters = {"shift": [1.2, -0.4]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors, [d.message for d in out.diagnostics]
        code = out.python_source
        assert ast.parse(code) is not None
        assert "ent_box_1.animate.shift(np.array([1.2, -0.4, 0.0]))" in code
        assert "run_time=1.5" in code

    def test_relative_vector_dict(self, compiler, solver, movable_box_scene):
        """Relative vector specified as dictionary with dx, dy."""
        movable_box_scene.timeline[1].actions[0].parameters = {"relative_vector": {"dx": 0.8, "dy": 0.3}}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert ast.parse(code) is not None
        assert "ent_box_1.animate.shift(np.array([0.8, 0.3, 0.0]))" in code

    def test_absolute_destination_list(self, compiler, solver, movable_box_scene):
        """Absolute destination coordinates compile to .animate.move_to(np.array([x, y, 0.0]))."""
        layout = solver.solve(movable_box_scene)
        vp_box = layout.viewports["vp_main"].world_box
        # Choose a valid point within vp_main
        target_x = round(vp_box.x + 1.0, 2)
        target_y = round(vp_box.y - 0.5, 2)

        movable_box_scene.timeline[1].actions[0].parameters = {"destination": [target_x, target_y]}
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors, [d.message for d in out.diagnostics]
        code = out.python_source
        assert ast.parse(code) is not None
        assert f"ent_box_1.animate.move_to(np.array([{target_x}, {target_y}, 0.0]))" in code

    def test_absolute_destination_dict(self, compiler, solver, movable_box_scene):
        """Absolute target_position as dictionary with x, y."""
        layout = solver.solve(movable_box_scene)
        vp_box = layout.viewports["vp_main"].world_box
        target_x = round(vp_box.x - 0.8, 2)
        target_y = round(vp_box.y + 0.4, 2)

        movable_box_scene.timeline[1].actions[0].parameters = {"target_position": {"x": target_x, "y": target_y}}
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert ast.parse(code) is not None
        assert f"ent_box_1.animate.move_to(np.array([{target_x}, {target_y}, 0.0]))" in code

    def test_move_to_viewport_sub_slot(self, compiler, solver):
        """Moving entity to a partitioned sub-slot moves to the slot's center."""
        scene = SceneIR(
            scene_id="SlotMoveScene",
            title="Slot Move Scene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="token_a",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "circle", "radius": 0.4},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 3.0)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="token_a", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(3.0, 6.0)),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="token_a",
                            duration=1.0,
                            parameters={"slot": "slot_right"},
                        )
                    ],
                ),
            ],
        )
        layout = solver.solve(scene)
        # Mock a sub-slot inside vp_main for slot testing
        vp_geom = layout.viewports["vp_main"]
        vp_geom.sub_slots["slot_right"] = BoundingBox2D(x=2.5, y=0.0, width=2.0, height=2.0)

        out = compiler.compile(scene, layout)
        assert not out.has_errors
        code = out.python_source
        assert "ent_token_a.animate.move_to(np.array([2.5, 0.0, 0.0]))" in code

    def test_move_to_target_viewport_center(self, compiler, solver):
        """Moving entity to target_viewport moves to the target viewport's world center."""
        scene = SceneIR(
            scene_id="VpMoveScene",
            title="Viewport Move Scene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
            viewports={
                "vp_left": ViewportDeclaration(type=ViewportType.SPLIT_LEFT),
                "vp_right": ViewportDeclaration(type=ViewportType.SPLIT_RIGHT),
            },
            entities=[
                EntityDeclaration(
                    id="card_1",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_left",
                    initial_state={"card_type": "concept", "title": "Card 1", "body": "Initial text"},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 3.0)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="card_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(3.0, 6.0)),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="card_1",
                            duration=1.5,
                            parameters={"target_viewport": "vp_right"},
                        )
                    ],
                ),
            ],
        )
        layout = solver.solve(scene)
        vp_r_box = layout.viewports["vp_right"].world_box
        out = compiler.compile(scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert f"ent_card_1.animate.move_to(np.array([{round(vp_r_box.x, 4)}, {round(vp_r_box.y, 4)}, 0.0]))" in code

    def test_array_pointer_index_shift(self, compiler, solver):
        """Moving an array pointer via new_target_index shifts based on index delta."""
        scene = get_binary_search_fixture()
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert ast.parse(code) is not None
        # Binary search step 2 moves pointer_high to new_target_index 3
        assert "ent_pointer_high.animate.shift(" in code


# =====================================================================
# 2. Viewport Boundary Validation & Out-Of-Bounds Detection
# =====================================================================

class TestBoundaryAndOutOfBoundsValidation:
    def test_destination_out_of_viewport_bounds_rejected(self, compiler, solver, movable_box_scene):
        """Destination placing entity outside viewport boundary emits COMPILER_OUT_OF_BOUNDS."""
        # Intentionally move box far off to the right (+20.0)
        movable_box_scene.timeline[1].actions[0].parameters = {"destination": [20.0, 0.0]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_OUT_OF_BOUNDS]
        assert len(diags) >= 1
        assert "breaches viewport" in diags[0].message or "breaches canvas" in diags[0].message

    def test_relative_shift_breaching_boundary_rejected(self, compiler, solver, movable_box_scene):
        """Relative shift vector pushing entity out of viewport emits COMPILER_OUT_OF_BOUNDS."""
        movable_box_scene.timeline[1].actions[0].parameters = {"shift": [15.0, 0.0]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_OUT_OF_BOUNDS]
        assert len(diags) >= 1

    def test_near_boundary_within_tolerance_passes(self, compiler, solver, movable_box_scene):
        """Movement placing entity right up to the safety margin passes cleanly."""
        layout = solver.solve(movable_box_scene)
        vp_box = layout.viewports["vp_main"].world_box
        geom = layout.entities["box_1"]

        # Place right at right edge: x = vp_box.right - geom.world_box.width / 2.0
        safe_x = round(vp_box.right - geom.world_box.width / 2.0 - 0.02, 3)
        safe_y = round(geom.world_box.y, 3)

        movable_box_scene.timeline[1].actions[0].parameters = {"destination": [safe_x, safe_y]}
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors, [d.message for d in out.diagnostics]


# =====================================================================
# 3. Error Handling and Diagnostics
# =====================================================================

class TestErrorHandlingAndDiagnostics:
    def test_missing_geometry_diagnostic(self, compiler, solver, movable_box_scene):
        """Entity without computed geometry in layout emits COMPILER_MISSING_GEOMETRY."""
        layout = solver.solve(movable_box_scene)
        # Deliberately remove entity geometry from layout
        layout.entities.pop("box_1", None)

        out = compiler.compile(movable_box_scene, layout)
        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_MISSING_GEOMETRY]
        assert len(diags) >= 1

    def test_non_finite_coordinates_rejected(self, compiler, solver, movable_box_scene):
        """NaN or infinite coordinates produce COMPILER_INVALID_PARAMETER."""
        movable_box_scene.timeline[1].actions[0].parameters = {"destination": [float("nan"), 0.0]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1
        assert "non-finite" in diags[0].message

    def test_infinite_shift_rejected(self, compiler, solver, movable_box_scene):
        """Infinite shift coordinates produce COMPILER_INVALID_PARAMETER."""
        movable_box_scene.timeline[1].actions[0].parameters = {"shift": [float("inf"), 1.0]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1

    def test_non_numeric_coordinates_rejected(self, compiler, solver, movable_box_scene):
        """String values for coordinates produce COMPILER_INVALID_PARAMETER."""
        movable_box_scene.timeline[1].actions[0].parameters = {"destination": ["left", "top"]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1

    def test_empty_movement_parameters_rejected(self, compiler, solver, movable_box_scene):
        """Action MOVE with empty parameters produces COMPILER_INVALID_PARAMETER."""
        movable_box_scene.timeline[1].actions[0].parameters = {}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1
        assert "no recognized movement destination" in diags[0].message

    def test_invalid_parameter_key_rejected(self, compiler, solver, movable_box_scene):
        """Malicious parameter key produces COMPILER_INVALID_PARAMETER."""
        bad_key = "__import__('os').system('hack')"
        movable_box_scene.timeline[1].actions[0].parameters = {bad_key: [1, 2]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1
        assert "invalid parameter key" in diags[0].message

    def test_unknown_target_viewport_rejected(self, compiler, solver, movable_box_scene):
        """Referencing nonexistent target_viewport produces COMPILER_INVALID_PARAMETER."""
        movable_box_scene.timeline[1].actions[0].parameters = {"target_viewport": "vp_nonexistent"}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1
        assert "unknown target viewport" in diags[0].message

    def test_trajectory_request_produces_warning(self, compiler, solver, movable_box_scene):
        """Requesting continuous trajectory modeling produces a WARNING diagnostic."""
        movable_box_scene.timeline[1].actions[0].parameters = {"shift": [1.0, 0.0], "trajectory": "bezier"}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        # Should not be a fatal error (compiles shift safely)
        assert not out.has_errors
        warnings = [d for d in out.diagnostics if d.severity == "WARNING"]
        assert len(warnings) >= 1
        assert "trajectory modeling" in warnings[0].message


# =====================================================================
# 4. Sequential and Repeated Movements
# =====================================================================

class TestSequentialAndRepeatedMovements:
    def test_successive_shifts_track_cumulative_position(self, compiler, solver, movable_box_scene):
        """Multiple successive shifts update tracked position deterministically."""
        # Add Step 3: Shift UP by 0.5
        movable_box_scene.timeline.append(
            TimelineEvent(
                step_id="step_3",
                narration_anchor=NarrationAnchor(lecture_line_index=2, audio_timestamp_range=(7.0, 10.0)),
                actions=[
                    ActionDeclaration(
                        type=ActionType.MOVE,
                        target_entity="box_1",
                        duration=1.0,
                        parameters={"shift": [0.0, 0.5]},
                    )
                ],
            )
        )
        movable_box_scene.duration_budget = 14.0
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert ast.parse(code) is not None
        # Step 2 shift
        assert "ent_box_1.animate.shift(np.array([1.5, 0.5, 0.0]))" in code
        # Step 3 shift
        assert "ent_box_1.animate.shift(np.array([0.0, 0.5, 0.0]))" in code

    def test_second_move_breaching_boundary_fails_closed(self, compiler, solver, movable_box_scene):
        """A second movement that exceeds bounds based on cumulative position fails closed."""
        # Step 2 moves box right by +2.0 (safe)
        movable_box_scene.timeline[1].actions[0].parameters = {"shift": [2.0, 0.0]}
        # Step 3 moves box further right by +10.0 (breaches boundary!)
        movable_box_scene.timeline.append(
            TimelineEvent(
                step_id="step_3",
                narration_anchor=NarrationAnchor(lecture_line_index=2, audio_timestamp_range=(7.0, 10.0)),
                actions=[
                    ActionDeclaration(
                        type=ActionType.MOVE,
                        target_entity="box_1",
                        duration=1.0,
                        parameters={"shift": [10.0, 0.0]},
                    )
                ],
            )
        )
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_OUT_OF_BOUNDS]
        assert len(diags) >= 1


# =====================================================================
# 5. Easing Function Mapping & Manim Runtime Verification
# =====================================================================

class TestEasingFunctionMapping:
    def test_absolute_movement_linear_easing(self, compiler, solver, movable_box_scene):
        """Absolute movement with EasingType.LINEAR compiles and builds with linear rate_func."""
        movable_box_scene.timeline[1].actions[0].easing = EasingType.LINEAR
        movable_box_scene.timeline[1].actions[0].parameters = {"destination": [1.0, 0.5]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        assert "ent_box_1.animate(rate_func=linear).move_to(" in out.python_source
        anim = build_manim_animation_from_source(out.python_source, "ent_box_1")
        assert anim.rate_func is linear

    def test_absolute_movement_ease_in_easing(self, compiler, solver, movable_box_scene):
        """Absolute movement with EasingType.EASE_IN compiles and builds with ease_in_quad."""
        movable_box_scene.timeline[1].actions[0].easing = EasingType.EASE_IN
        movable_box_scene.timeline[1].actions[0].parameters = {"destination": [1.0, 0.5]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        assert "ent_box_1.animate(rate_func=rate_functions.ease_in_quad).move_to(" in out.python_source
        anim = build_manim_animation_from_source(out.python_source, "ent_box_1")
        assert anim.rate_func is rate_functions.ease_in_quad

    def test_absolute_movement_ease_out_easing(self, compiler, solver, movable_box_scene):
        """Absolute movement with EasingType.EASE_OUT compiles and builds with ease_out_quad."""
        movable_box_scene.timeline[1].actions[0].easing = EasingType.EASE_OUT
        movable_box_scene.timeline[1].actions[0].parameters = {"destination": [1.0, 0.5]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        assert "ent_box_1.animate(rate_func=rate_functions.ease_out_quad).move_to(" in out.python_source
        anim = build_manim_animation_from_source(out.python_source, "ent_box_1")
        assert anim.rate_func is rate_functions.ease_out_quad

    def test_relative_movement_linear_easing(self, compiler, solver, movable_box_scene):
        """Relative shift with EasingType.LINEAR compiles and builds with linear rate_func."""
        movable_box_scene.timeline[1].actions[0].easing = EasingType.LINEAR
        movable_box_scene.timeline[1].actions[0].parameters = {"shift": [1.5, 0.5]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        assert "ent_box_1.animate(rate_func=linear).shift(" in out.python_source
        anim = build_manim_animation_from_source(out.python_source, "ent_box_1")
        assert anim.rate_func is linear

    def test_relative_movement_ease_in_easing(self, compiler, solver, movable_box_scene):
        """Relative shift with EasingType.EASE_IN compiles and builds with ease_in_quad."""
        movable_box_scene.timeline[1].actions[0].easing = EasingType.EASE_IN
        movable_box_scene.timeline[1].actions[0].parameters = {"shift": [1.5, 0.5]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        assert "ent_box_1.animate(rate_func=rate_functions.ease_in_quad).shift(" in out.python_source
        anim = build_manim_animation_from_source(out.python_source, "ent_box_1")
        assert anim.rate_func is rate_functions.ease_in_quad

    def test_relative_movement_ease_out_easing(self, compiler, solver, movable_box_scene):
        """Relative shift with EasingType.EASE_OUT compiles and builds with ease_out_quad."""
        movable_box_scene.timeline[1].actions[0].easing = EasingType.EASE_OUT
        movable_box_scene.timeline[1].actions[0].parameters = {"shift": [1.5, 0.5]}
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        assert "ent_box_1.animate(rate_func=rate_functions.ease_out_quad).shift(" in out.python_source
        anim = build_manim_animation_from_source(out.python_source, "ent_box_1")
        assert anim.rate_func is rate_functions.ease_out_quad

    def test_smooth_easing_default(self, compiler, solver, movable_box_scene):
        """EasingType.SMOOTH omits rate_func argument and defaults to smooth in Manim."""
        movable_box_scene.timeline[1].actions[0].easing = EasingType.SMOOTH
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        assert "rate_func" not in out.python_source
        assert "ent_box_1.animate.shift(" in out.python_source
        anim = build_manim_animation_from_source(out.python_source, "ent_box_1")
        assert anim.rate_func is rate_functions.smooth

    def test_invalid_unsupported_easing_rejected(self, compiler, solver, movable_box_scene):
        """Unsupported easing value emits COMPILER_INVALID_PARAMETER diagnostic."""
        # Force an unsupported easing string on the action
        movable_box_scene.timeline[1].actions[0].easing = "custom_spring_bounce"
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1
        assert "unsupported easing" in diags[0].message
        assert "custom_spring_bounce" in diags[0].message

    def test_malformed_syntax_regression_verification(self):
        """Explicitly proves that the old malformed pattern fails runtime execution in Manim CE."""
        malformed_code = (
            "self.play(ent_box_1.animate, rate_func=linear.move_to(np.array([1.0, 0.5, 0.0])), run_time=1.0)"
        )
        with pytest.raises(AttributeError, match="has no attribute 'move_to'"):
            build_manim_animation_from_source(malformed_code, "ent_box_1")


# =====================================================================
# 6. Determinism & Output Stability
# =====================================================================

class TestDeterminismAndCodeValidity:
    def test_bit_identical_compilation(self, compiler, solver, movable_box_scene):
        """Repeated compilation yields bit-identical Python source."""
        layout1 = solver.solve(movable_box_scene)
        out1 = compiler.compile(movable_box_scene, layout1)

        layout2 = solver.solve(movable_box_scene)
        out2 = compiler.compile(movable_box_scene, layout2)

        assert out1.python_source == out2.python_source
        assert len(out1.diagnostics) == len(out2.diagnostics)

    def test_post_action_pause_emission(self, compiler, solver, movable_box_scene):
        """Action with post_action_pause emits self.wait."""
        movable_box_scene.timeline[1].actions[0].post_action_pause = 0.5
        layout = solver.solve(movable_box_scene)
        out = compiler.compile(movable_box_scene, layout)

        assert not out.has_errors
        assert "self.wait(0.5)" in out.python_source


# =====================================================================
# 7. Canonical Domain Fixtures Preservation
# =====================================================================

class TestCanonicalDomainFixturesPreservation:
    def test_binary_search_fixture(self, compiler, solver):
        scene = get_binary_search_fixture()
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert not out.has_errors, [d.message for d in out.diagnostics]
        assert ast.parse(out.python_source) is not None

    def test_chemical_kinetics_fixture(self, compiler, solver):
        scene = get_chemical_kinetics_fixture()
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert not out.has_errors, [d.message for d in out.diagnostics]
        assert ast.parse(out.python_source) is not None

    def test_pythagorean_theorem_fixture(self, compiler, solver):
        scene = get_pythagorean_theorem_fixture()
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert not out.has_errors, [d.message for d in out.diagnostics]
        assert ast.parse(out.python_source) is not None


# =====================================================================
# 8. Milestone 4C.8: Component Library Hardening (FINDING-4C3-02 & FINDING-4C3-06)
# =====================================================================

class TestComponentLibraryHardeningPointerSemantics:
    def _create_pointer_scene(
        self,
        parent_initial_state: Dict[str, Any],
        pointer_actions: list,
        parent_type: str = "CompositeNodeGraph",
        pointer_initial_index: int = 0,
        parent_id: str = "arr_1",
        pointer_parent_ref: str = "arr_1",
    ) -> SceneIR:
        entities = []
        if parent_id is not None:
            entities.append(
                EntityDeclaration(
                    id=parent_id,
                    component_type=parent_type,
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state=parent_initial_state,
                )
            )
        entities.append(
            EntityDeclaration(
                id="ptr_1",
                component_type="PointerIndicator",
                visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                viewport="main_vp",
                parent_id=pointer_parent_ref,
                initial_state={"target_index": pointer_initial_index, "label": "ptr"},
            )
        )
        return SceneIR(
            scene_id="PointerHardeningScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=entities,
            timeline=[
                TimelineEvent(
                    step_id=f"step_{idx + 1}",
                    actions=[act],
                )
                for idx, act in enumerate(pointer_actions)
            ],
            metadata={},
        )

    # -----------------------------------------------------------------
    # FINDING-4C3-02: Authoritative Geometry & Dynamic Cell Pitch
    # -----------------------------------------------------------------

    def test_pointer_default_cell_pitch(self, compiler, solver):
        """FINDING-4C3-02: Components specifying cell_pitch=0.8 preserve historical default."""
        act = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 4},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={"cell_pitch": 0.8, "capacity": 5},
            pointer_actions=[act],
            pointer_initial_index=0,
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors, [d.message for d in out.diagnostics]
        code = out.python_source
        assert ast.parse(code) is not None
        # Delta = 4 * 0.8 = 3.2
        assert "3.2" in code

    def test_pointer_non_default_cell_pitch(self, compiler, solver):
        """FINDING-4C3-02: Components specifying non-default pitch (1.25) derive destination correctly."""
        act = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 3},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={"cell_pitch": 1.25, "capacity": 5},
            pointer_actions=[act],
            pointer_initial_index=1,
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors, [d.message for d in out.diagnostics]
        code = out.python_source
        assert ast.parse(code) is not None
        # Delta = (3 - 1) * 1.25 = 2.5
        assert "2.5" in code

    def test_pointer_derived_from_table_cell_width(self, compiler, solver):
        """FINDING-4C3-02: Table components specifying cell_width=0.75 calculate destination from cell width."""
        act = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 2},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={"cell_width": 0.75, "capacity": 5},
            pointer_actions=[act],
            pointer_initial_index=0,
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors, [d.message for d in out.diagnostics]
        code = out.python_source
        assert ast.parse(code) is not None
        # Delta = 2 * 0.75 = 1.5
        assert "1.5" in code

    def test_pointer_unequal_cell_spacing(self, compiler, solver):
        """FINDING-4C3-02: Components specifying unequal cell widths correctly integrate cell intervals."""
        act = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 2},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={"cell_widths": [0.5, 1.0, 1.5, 0.8], "buff": 0.1},
            pointer_actions=[act],
            pointer_initial_index=0,
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors, [d.message for d in out.diagnostics]
        code = out.python_source
        assert ast.parse(code) is not None
        # Distance = (0.5+1.0)/2 + 0.1 + (1.0+1.5)/2 + 0.1 = 0.75 + 0.1 + 1.25 + 0.1 = 2.2
        assert "2.2" in code

    def test_pointer_first_and_last_valid_cells(self, compiler, solver):
        """FINDING-4C3-02: Pointer can move to the first (0) and last (4) valid cells of a 5-element parent."""
        act_last = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 4},
        )
        act_first = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 0},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={"elements": [{"index": i, "val": str(i)} for i in range(5)], "cell_pitch": 1.0},
            pointer_actions=[act_last, act_first],
            pointer_initial_index=0,
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors, [d.message for d in out.diagnostics]
        code = out.python_source
        assert ast.parse(code) is not None
        assert "4.0" in code
        assert "-4.0" in code

    def test_pointer_missing_parent_geometry_fails_closed(self, compiler, solver):
        """FINDING-4C3-02: Pointer referencing non-existent parent emits COMPILER_MISSING_GEOMETRY."""
        act = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 1, "target_array": "ghost_array"},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={},
            pointer_actions=[act],
            pointer_initial_index=0,
            parent_id=None,
            pointer_parent_ref=None,
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_MISSING_GEOMETRY for d in out.diagnostics)

    def test_pointer_insufficient_parent_geometry_fails_closed(self, compiler, solver):
        """FINDING-4C3-02: Parent providing no cell definitions or pitch emits COMPILER_MISSING_GEOMETRY."""
        act = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 1},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={"shape_type": "rectangle"},
            pointer_actions=[act],
            parent_type="PrimitiveShape",
            pointer_initial_index=0,
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_MISSING_GEOMETRY for d in out.diagnostics)

    # -----------------------------------------------------------------
    # FINDING-4C3-06: Transactional State Mutation & Boundary Validation
    # -----------------------------------------------------------------

    def test_pointer_negative_index_rejected(self, compiler, solver):
        """FINDING-4C3-06: Negative pointer index is rejected with COMPILER_OUT_OF_BOUNDS."""
        act = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": -1},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={"cell_pitch": 1.0, "capacity": 5},
            pointer_actions=[act],
            pointer_initial_index=0,
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_OUT_OF_BOUNDS for d in out.diagnostics)

    def test_pointer_out_of_range_index_rejected(self, compiler, solver):
        """FINDING-4C3-06: Index exceeding parent capacity is rejected with COMPILER_OUT_OF_BOUNDS."""
        act = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 5},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={"elements": [{"index": i} for i in range(5)], "cell_pitch": 1.0},
            pointer_actions=[act],
            pointer_initial_index=0,
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_OUT_OF_BOUNDS for d in out.diagnostics)

    def test_pointer_backward_movement(self, compiler, solver):
        """FINDING-4C3-06: Pointer moving backward computes negative displacement."""
        act = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 1},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={"cell_pitch": 1.0, "capacity": 5},
            pointer_actions=[act],
            pointer_initial_index=3,
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert ast.parse(code) is not None
        # Delta = (1 - 3) * 1.0 = -2.0
        assert "-2.0" in code

    def test_pointer_failed_operation_preserves_state_for_subsequent_valid_move(self, compiler, solver):
        """FINDING-4C3-06: A failed movement does not mutate pointer index, allowing subsequent move to succeed."""
        # Step 1: Valid move from 1 to 2
        act_valid_1 = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 2},
        )
        # Step 2: Invalid move to out-of-range index 99 (must fail)
        act_invalid = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 99},
        )
        # Step 3: Valid move from 2 to 3
        act_valid_2 = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ptr_1",
            duration=1.0,
            parameters={"new_target_index": 3},
        )
        scene = self._create_pointer_scene(
            parent_initial_state={"elements": [{"index": i} for i in range(5)], "cell_pitch": 1.0},
            pointer_actions=[act_valid_1, act_invalid, act_valid_2],
            pointer_initial_index=1,
        )
        layout = solver.solve(scene)

        # Track state explicitly across compile_action_elements
        tracker = {"ptr_1": {"target_index": 1}}
        pos_tracker = {"ptr_1": layout.entities["ptr_1"].world_box}
        entity_map = {e.id: e for e in scene.entities}

        # Step 1: Execute valid action 1 (1 -> 2)
        _, _, _, diags1 = compiler._compile_action_elements(
            action=act_valid_1,
            target_var="ent_ptr_1",
            layout=layout,
            entity_map=entity_map,
            entity_state_tracker=tracker,
            entity_position_tracker=pos_tracker,
            step_idx=0,
            a_idx=0,
        )
        assert len([d for d in diags1 if d.severity == "ERROR"]) == 0
        assert tracker["ptr_1"]["target_index"] == 2

        # Step 2: Execute invalid action 2 (2 -> 99)
        _, _, _, diags2 = compiler._compile_action_elements(
            action=act_invalid,
            target_var="ent_ptr_1",
            layout=layout,
            entity_map=entity_map,
            entity_state_tracker=tracker,
            entity_position_tracker=pos_tracker,
            step_idx=1,
            a_idx=0,
        )
        assert any(d.error_code == CompilationErrorCode.COMPILER_OUT_OF_BOUNDS for d in diags2)
        # Invariant check: tracker must NOT be corrupted to 99! It must remain committed at 2.
        assert tracker["ptr_1"]["target_index"] == 2

        # Step 3: Execute valid action 3 (2 -> 3)
        setup3, anim3, _, diags3 = compiler._compile_action_elements(
            action=act_valid_2,
            target_var="ent_ptr_1",
            layout=layout,
            entity_map=entity_map,
            entity_state_tracker=tracker,
            entity_position_tracker=pos_tracker,
            step_idx=2,
            a_idx=0,
        )
        assert len([d for d in diags3 if d.severity == "ERROR"]) == 0
        assert tracker["ptr_1"]["target_index"] == 3
        # Delta must be (3 - 2) * 1.0 = 1.0, NOT (3 - 99) * 1.0 = -96.0!
        assert any("1.0" in a for a in anim3)
        assert not any("-96.0" in a for a in anim3)
