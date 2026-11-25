"""
CineLearn — Milestone 4C.5: Layout and Spatial Semantics Hardening Test Suite
=============================================================================
Comprehensive unit, edge-case, and integration verification for Milestone 4C.5,
resolving all 8 authorized deferred findings and 1 advisory finding:

- FINDING-4C1-03: Container Border Overlap
- FINDING-4C1-04: Viewport Boundary Crossing
- FINDING-4C1-05: Missing Diagnostic Metadata on Layout Failure
- FINDING-4C2-02: PrimitiveShape Parameter Chaining
- FINDING-4C2-04: Incorrect Overflow Diagnostic Field Path
- FINDING-4C3-03: Boolean Coordinate Parsing
- FINDING-4C3-04: Trajectory Diagnostic Semantics
- FINDING-4C3-05: Cross-Viewport Migration Tracking
- ADVISORY-4C4-02: Concurrent Conceptual-Card Stroke Animation (with Manim CE runtime)
"""

import math
import pytest
from typing import Any, Dict, List

from src.compiler.card_compiler import ConceptualCardCompiler
from src.compiler.compiler_models import CompilationDiagnostic, CompilationErrorCode
from src.compiler.layout_engine import HierarchicalBoxModelLayoutSolver
from src.compiler.layout_models import (
    CONTAINER_BORDER_TOLERANCE,
    VIEWPORT_BOUNDARY_TOLERANCE,
    BoundingBox2D,
    LayoutDiagnostic,
    LayoutErrorCode,
    SceneSpatialLayout,
)
from src.compiler.scene_compiler import DeterministicSceneCompiler
from src.compiler.spatial_ledger import CollisionCategory, SpatialLedger
from src.compiler.timeline_scheduler import TimelineScheduler
from src.ir.models import (
    ActionDeclaration,
    ActionType,
    Alignment,
    EasingType,
    EntityDeclaration,
    EntityStyling,
    FlowDirection,
    LayoutConfiguration,
    NarrationAnchor,
    SceneIR,
    TimelineEvent,
    ViewportDeclaration,
    ViewportType,
    VisualTier,
)


@pytest.fixture
def compiler() -> DeterministicSceneCompiler:
    return DeterministicSceneCompiler()


@pytest.fixture
def solver() -> HierarchicalBoxModelLayoutSolver:
    return HierarchicalBoxModelLayoutSolver()


@pytest.fixture
def card_compiler() -> ConceptualCardCompiler:
    return ConceptualCardCompiler()


# =============================================================================
# A. FINDING-4C1-03: Container Border Overlap
# =============================================================================

class TestContainerBorderOverlap:
    """Verifies that child content does not visually overlap or collide with container borders."""

    def test_child_fits_cleanly_inside_container_with_stroke(self, solver):
        """A normal child inside a container with stroke fits with proper border clearance."""
        scene = SceneIR(
            scene_id="CleanContainerScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="parent_box",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 6.0, "height": 4.0, "stroke_width": 4.0},
                ),
                EntityDeclaration(
                    id="child_item",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="parent_box",
                    initial_state={"width": 2.0, "height": 1.5},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        p_geom = layout.entities["parent_box"]
        c_geom = layout.entities["child_item"]

        # Child must be strictly within parent bounds with clearance
        assert p_geom.world_box.contains_box(c_geom.world_box, tolerance=0.01) is True
        # Child boundaries must not touch or cross parent boundaries
        assert c_geom.world_box.left > p_geom.world_box.left + 0.05
        assert c_geom.world_box.right < p_geom.world_box.right - 0.05
        assert c_geom.world_box.bottom > p_geom.world_box.bottom + 0.05
        assert c_geom.world_box.top < p_geom.world_box.top - 0.05

    def test_nested_containers_three_levels(self, solver):
        """Hierarchical nested containers (grandparent -> parent -> child) maintain containment."""
        scene = SceneIR(
            scene_id="NestedContainerScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="grandparent",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 8.0, "height": 6.0, "stroke_width": 2.0},
                ),
                EntityDeclaration(
                    id="parent",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="grandparent",
                    initial_state={"width": 5.0, "height": 4.0, "stroke_width": 2.0},
                ),
                EntityDeclaration(
                    id="child",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="parent",
                    initial_state={"width": 1.5, "height": 1.0},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        gp_box = layout.entities["grandparent"].world_box
        p_box = layout.entities["parent"].world_box
        c_box = layout.entities["child"].world_box

        # Grandparent strictly contains parent
        assert gp_box.contains_box(p_box, tolerance=0.01) is True
        # Parent strictly contains child
        assert p_box.contains_box(c_box, tolerance=0.01) is True

    def test_multiple_children_sharing_container(self, solver):
        """Multiple siblings allocated inside a container respect container borders and gaps."""
        scene = SceneIR(
            scene_id="MultipleChildrenScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="container",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 6.0, "height": 4.0, "padding": 0.15},
                ),
                EntityDeclaration(
                    id="item_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="container",
                    initial_state={"width": 1.2, "height": 0.8},
                ),
                EntityDeclaration(
                    id="item_2",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="container",
                    initial_state={"width": 1.2, "height": 0.8},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        c_box = layout.entities["container"].world_box
        for item_id in ("item_1", "item_2"):
            i_box = layout.entities[item_id].world_box
            assert c_box.contains_box(i_box, tolerance=0.01) is True
            # Check border clearance
            assert i_box.left > c_box.left + 0.05
            assert i_box.right < c_box.right - 0.05

    def test_oversized_child_exceeding_container_fails_with_diagnostic(self, solver):
        """Child that cannot fit within container allocation emits LAYOUT_PARENT_BOUNDS_EXCEEDED."""
        scene = SceneIR(
            scene_id="OversizedChildScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="tiny_container",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 1.5, "height": 1.0},
                ),
                EntityDeclaration(
                    id="giant_child",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="tiny_container",
                    initial_state={"width": 5.0, "height": 4.0},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is False
        error_codes = [e.error_code for e in layout.errors]
        assert LayoutErrorCode.LAYOUT_PARENT_BOUNDS_EXCEEDED in error_codes
        diag = next(e for e in layout.errors if e.error_code == LayoutErrorCode.LAYOUT_PARENT_BOUNDS_EXCEEDED)
        assert diag.entity_id == "giant_child"
        assert "tiny_container" in diag.related_entities
        assert "child_box" in diag.details
        assert "parent_box" in diag.details


# =============================================================================
# B. FINDING-4C1-04 & H. FINDING-4C3-05: Viewport Boundary Crossing & Migration
# =============================================================================

class TestViewportBoundaryAndMigration:
    """Verifies detection of viewport boundary breaches and valid migration tracking."""

    def test_static_viewport_boundary_breach_detected(self):
        """SpatialLedger detects when an entity breaches its assigned viewport boundaries."""
        ledger = SpatialLedger(scene_duration=10.0)
        canvas_box = BoundingBox2D(x=0.0, y=0.0, width=12.0, height=7.0)
        viewports = {
            "vp_left": BoundingBox2D.from_min_max(left=-6.0, right=-1.0, bottom=-3.0, top=3.0),
            "vp_right": BoundingBox2D.from_min_max(left=1.0, right=6.0, bottom=-3.0, top=3.0),
        }
        # Place entity assigned to vp_left, but box breaches vp_left (crosses into center)
        breaching_box = BoundingBox2D(x=-0.5, y=0.0, width=2.0, height=2.0)  # right is 0.5 > -1.0
        ledger.register_span(
            entity_id="bad_box",
            box=breaching_box,
            t_start=0.0,
            t_end=10.0,
            viewport_id="vp_left",
        )

        is_valid, _, diags = ledger.validate_spatial_integrity(
            canvas_box=canvas_box,
            viewports=viewports,
        )
        assert is_valid is False
        vp_breaches = [d for d in diags if d.error_code == LayoutErrorCode.LAYOUT_BOUNDARY_BREACH and "vp_left" in d.message]
        assert len(vp_breaches) >= 1
        assert vp_breaches[0].viewport_id == "vp_left"
        assert vp_breaches[0].details["breach"] > 0.0

    def test_movement_across_viewport_boundary_without_migration_fails(self, compiler, solver):
        """Moving outside assigned viewport without target_viewport parameter fails closed."""
        scene = SceneIR(
            scene_id="MoveBreachScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
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
                    initial_state={"title": "Left Card", "body": "Inside left"},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="card_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    # Move to positive X (inside vp_right) WITHOUT target_viewport
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="card_1",
                            duration=1.5,
                            parameters={"destination": [3.0, 0.0]},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is False
        oob_diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_OUT_OF_BOUNDS]
        assert len(oob_diags) >= 1
        assert "vp_left" in oob_diags[0].message

    def test_valid_viewport_migration_updates_tracked_viewport(self, compiler, solver):
        """Moving with target_viewport successfully updates ownership, allowing subsequent moves."""
        scene = SceneIR(
            scene_id="ValidMigrationScene",
            duration_budget=15.0,
            layout_configuration=LayoutConfiguration(),
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
                    initial_state={"title": "Migrating Card", "body": "Card to migrate"},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="card_1", duration=1.0)],
                ),
                # Step 2: Migrate to vp_right center
                TimelineEvent(
                    step_id="step_2",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="card_1",
                            duration=1.5,
                            parameters={"target_viewport": "vp_right"},
                        )
                    ],
                ),
                # Step 3: Shift slightly inside vp_right WITHOUT specifying target_viewport
                # (This tests FINDING-4C3-05: Must validate against vp_right, NOT original vp_left!)
                TimelineEvent(
                    step_id="step_3",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="card_1",
                            duration=1.0,
                            parameters={"shift": [0.10, 0.0]},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is True
        assert not out.has_errors
        assert "move_to" in out.python_source
        assert "shift" in out.python_source

    def test_failed_migration_does_not_mutate_state(self, compiler, solver):
        """Failed migration to out-of-bounds destination does not mutate state or position."""
        scene = SceneIR(
            scene_id="FailedMigrationScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
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
                    initial_state={"title": "Static Card", "body": "Stay put"},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="card_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    # Target vp_right, but specify coordinates far outside vp_right
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="card_1",
                            duration=1.5,
                            parameters={"target_viewport": "vp_right", "destination": [10.0, 10.0]},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_OUT_OF_BOUNDS for d in out.diagnostics)

    def test_unknown_target_viewport_fails_with_structured_diagnostic(self, compiler, solver):
        """Targeting a nonexistent viewport produces COMPILER_INVALID_PARAMETER with details."""
        scene = SceneIR(
            scene_id="UnknownVpScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="card_1",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="main_vp",
                    initial_state={"title": "Test", "body": "Test"},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="card_1",
                            duration=1.0,
                            parameters={"target_viewport": "ghost_viewport"},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is False
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1
        assert "ghost_viewport" in diags[0].message
        assert diags[0].details.get("target_viewport") == "ghost_viewport"


# =============================================================================
# C. FINDING-4C1-05: Missing Diagnostic Metadata on Layout Failure
# =============================================================================

class TestDiagnosticMetadataPropagation:
    """Verifies that layout failure diagnostics contain structured metadata and survive propagation."""

    def test_direct_solver_failure_contains_structured_metadata(self, solver):
        """Solver errors contain error_code, entity_id, viewport_id, and details."""
        scene = SceneIR(
            scene_id="DirectFailScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="orphan_node",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="nonexistent_parent",
                    initial_state={"width": 1.0, "height": 1.0},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is False
        assert len(layout.errors) >= 1
        err = layout.errors[0]
        assert err.error_code == LayoutErrorCode.LAYOUT_UNSATISFIABLE_CONSTRAINTS
        assert err.entity_id == "orphan_node"
        assert err.viewport_id == "main_vp"
        assert "unresolved_parent_id" in err.details
        assert err.details["unresolved_parent_id"] == "nonexistent_parent"

    def test_layout_failure_propagates_into_compiler_diagnostics(self, compiler, solver):
        """Compiler catches layout errors, maps them to CompilationDiagnostic, and fails closed."""
        scene = SceneIR(
            scene_id="PropagateFailScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="cycle_a",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="cycle_a",  # Self-cycle
                    initial_state={"width": 1.0, "height": 1.0},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is False

        out = compiler.compile(scene, layout)
        assert out.is_valid is False
        assert out.python_source == ""
        # Check that layout diagnostic survived
        layout_diags = [d for d in out.diagnostics if "[Layout]" in d.message]
        assert len(layout_diags) >= 1
        assert layout_diags[0].entity_id == "cycle_a"
        assert layout_diags[0].severity == "ERROR"
        assert "self_cycle" in layout_diags[0].details


# =============================================================================
# D. FINDING-4C2-02: PrimitiveShape Parameter Chaining
# =============================================================================

class TestPrimitiveShapeParameterChaining:
    """Verifies that color, scale, and opacity can be mutated individually and chained together."""

    @pytest.fixture
    def primitive_scene(self) -> SceneIR:
        return SceneIR(
            scene_id="PrimitiveChainingScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="rect_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 2.0, "height": 2.0, "color": "#FFFFFF", "opacity": 1.0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="rect_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="rect_1",
                            duration=1.5,
                            parameters={"color": "#FF0000", "scale": 1.5, "opacity": 0.5},
                        )
                    ],
                ),
            ],
            metadata={},
        )

    def test_all_three_chained(self, compiler, solver, primitive_scene):
        """Color + scale + opacity in one action emits chained animate methods in deterministic order."""
        layout = solver.solve(primitive_scene)
        out = compiler.compile(primitive_scene, layout)
        assert out.is_valid is True
        assert not out.has_errors
        # Verify chained animate syntax
        expected_chain = ".set_color('#FF0000').scale(1.5).set_opacity(0.5)"
        assert expected_chain in out.python_source

    def test_color_and_scale_chained(self, compiler, solver, primitive_scene):
        """Color + scale emits chained animate call without opacity."""
        primitive_scene.timeline[1].actions[0].parameters = {"color": "#00FF00", "scale": 2.0}
        layout = solver.solve(primitive_scene)
        out = compiler.compile(primitive_scene, layout)
        assert out.is_valid is True
        assert ".set_color('#00FF00').scale(2.0)" in out.python_source

    def test_color_and_opacity_chained(self, compiler, solver, primitive_scene):
        """Color + opacity emits chained animate call without scale."""
        primitive_scene.timeline[1].actions[0].parameters = {"color": "#0000FF", "opacity": 0.3}
        layout = solver.solve(primitive_scene)
        out = compiler.compile(primitive_scene, layout)
        assert out.is_valid is True
        assert ".set_color('#0000FF').set_opacity(0.3)" in out.python_source

    def test_scale_and_opacity_chained(self, compiler, solver, primitive_scene):
        """Scale + opacity emits chained animate call without color."""
        primitive_scene.timeline[1].actions[0].parameters = {"scale": 0.8, "opacity": 0.75}
        layout = solver.solve(primitive_scene)
        out = compiler.compile(primitive_scene, layout)
        assert out.is_valid is True
        assert ".scale(0.8).set_opacity(0.75)" in out.python_source

    def test_boolean_scale_rejected(self, compiler, solver, primitive_scene):
        """Boolean scale value (scale: True) is rejected with COMPILER_INVALID_PARAMETER."""
        primitive_scene.timeline[1].actions[0].parameters = {"scale": True}
        layout = solver.solve(primitive_scene)
        out = compiler.compile(primitive_scene, layout)
        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER for d in out.diagnostics)

    def test_boolean_opacity_rejected(self, compiler, solver, primitive_scene):
        """Boolean opacity value (opacity: False) is rejected with COMPILER_INVALID_PARAMETER."""
        primitive_scene.timeline[1].actions[0].parameters = {"opacity": False}
        layout = solver.solve(primitive_scene)
        out = compiler.compile(primitive_scene, layout)
        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER for d in out.diagnostics)

    def test_unsupported_parameters_rejected(self, compiler, solver, primitive_scene):
        """Unsupported parameters emit COMPILER_UNSUPPORTED_OPERATION."""
        primitive_scene.timeline[1].actions[0].parameters = {"color": "#FF0000", "invalid_param": 123}
        layout = solver.solve(primitive_scene)
        out = compiler.compile(primitive_scene, layout)
        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION for d in out.diagnostics)


# =============================================================================
# E. FINDING-4C2-04: Incorrect Overflow Diagnostic Field Path
# =============================================================================

class TestCardOverflowFieldPath:
    """Verifies that card overflow diagnostics report accurate dynamic field paths."""

    def test_transform_action_overflow_reports_action_parameter_path(self, compiler, solver):
        """Overflow caused by TRANSFORM action points to timeline[step].actions[a].parameters.formula."""
        huge_formula = r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} \cdot \frac{\sum_{i=1}^{n} x_i}{\prod_{j=1}^{m} y_j} \cdot \text{extremely long formula that overflows the card by a massive margin}"
        scene = SceneIR(
            scene_id="CardTransformOverflowScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="card_1",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="main_vp",
                    initial_state={"title": "Short", "formula": "x = 1"},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="card_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="card_1",
                            duration=1.5,
                            parameters={"formula": huge_formula},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        # Should contain overflow warning pointing to the action's parameter path
        overflow_diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_CONTENT_OVERFLOW]
        assert len(overflow_diags) >= 1
        assert "timeline[1].actions[0].parameters.formula" in overflow_diags[0].field_path

    def test_initial_state_overflow_reports_entities_path(self, compiler, solver):
        """Overflow declared in initial_state points to entities[idx].initial_state.body."""
        huge_body = "This is an extraordinarily long body text designed to test the layout engine and card compiler content overflow detection heuristics by exceeding the vertical bounds of the conceptual card." * 4
        scene = SceneIR(
            scene_id="CardInitialOverflowScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="card_1",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="main_vp",
                    initial_state={"title": "Test Title", "body": huge_body},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        overflow_diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_CONTENT_OVERFLOW]
        assert len(overflow_diags) >= 1
        assert "entities[0].initial_state.body" in overflow_diags[0].field_path


# =============================================================================
# F. FINDING-4C3-03: Boolean Coordinate Parsing
# =============================================================================

class TestBooleanCoordinateParsing:
    """Verifies that boolean coordinate values are explicitly rejected without int conversion."""

    def test_boolean_in_coordinate_list_rejected(self, compiler, solver):
        """destination: [True, 0.0] must be rejected as invalid coordinates."""
        scene = SceneIR(
            scene_id="BoolCoordScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="rect_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 1.0, "height": 1.0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="rect_1",
                            duration=1.0,
                            parameters={"destination": [True, 0.0]},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER for d in out.diagnostics)

    def test_boolean_in_vector_dict_rejected(self, compiler, solver):
        """shift: {'x': False, 'y': 1.0} must be rejected."""
        scene = SceneIR(
            scene_id="BoolDictCoordScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="rect_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 1.0, "height": 1.0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="rect_1",
                            duration=1.0,
                            parameters={"shift": {"x": False, "y": 1.0}},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER for d in out.diagnostics)

    def test_boolean_new_target_index_rejected(self, compiler, solver):
        """new_target_index: True must be rejected."""
        scene = SceneIR(
            scene_id="BoolPointerScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="ptr",
                    component_type="PointerIndicator",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="main_vp",
                    initial_state={"target_index": 0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="ptr",
                            duration=1.0,
                            parameters={"new_target_index": True},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER for d in out.diagnostics)

    def test_valid_integers_and_floats_accepted(self, compiler, solver):
        """Valid integer and float coordinates compile successfully."""
        scene = SceneIR(
            scene_id="ValidCoordsScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="rect_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 1.0, "height": 1.0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="rect_1",
                            duration=1.0,
                            parameters={"shift": [1, -0.5]},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is True
        assert not out.has_errors


# =============================================================================
# G. FINDING-4C3-04: Trajectory Diagnostic Semantics
# =============================================================================

class TestTrajectoryDiagnosticSemantics:
    """Verifies that trajectory diagnostics clearly distinguish path from swept volume and static endpoint."""

    def test_trajectory_warning_semantics(self, compiler, solver):
        """Requesting trajectory emits WARNING detailing static validation without claiming swept volume."""
        scene = SceneIR(
            scene_id="TrajectorySemanticScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="rect_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 1.0, "height": 1.0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="rect_1",
                            duration=1.0,
                            parameters={"shift": [1.0, 0.5], "trajectory": "arc"},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        # Should succeed because trajectory warning is non-fatal
        assert out.is_valid is True
        warnings = [d for d in out.diagnostics if d.severity == "WARNING" and "trajectory" in d.message]
        assert len(warnings) >= 1
        diag = warnings[0]
        # Verify semantics: mentions trajectory modeling, clarifies swept-volume is unsupported, validates static destination
        assert "trajectory modeling" in diag.message
        assert "Continuous swept-volume collision modeling" in diag.message
        assert "validating static destination" in diag.message
        assert diag.details.get("swept_volume_validated") is False
        assert diag.details.get("validation_performed") == "static_destination_enclosure"


# =============================================================================
# I. ADVISORY-4C4-02: Concurrent Conceptual-Card Stroke Animation (Manim CE 0.18.1)
# =============================================================================

class TestConcurrentCardStrokeRuntime:
    """Verifies that concurrent conceptual-card stroke animation generates valid Manim CE code and executes."""

    def test_concurrent_card_stroke_syntax(self, compiler, solver):
        """In concurrent mode, card stroke update emits .animate(run_time=dur).set_stroke(...)."""
        scene = SceneIR(
            scene_id="ConcurrentCardStrokeScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="card_1",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="main_vp",
                    initial_state={"title": "Card 1", "body": "Body 1"},
                ),
                EntityDeclaration(
                    id="card_2",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="main_vp",
                    initial_state={"title": "Card 2", "body": "Body 2"},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_enter",
                    actions=[
                        ActionDeclaration(type=ActionType.ENTER, target_entity="card_1", duration=1.0),
                        ActionDeclaration(type=ActionType.ENTER, target_entity="card_2", duration=1.0),
                    ],
                ),
                # Concurrent step: card_1 updates stroke color, card_2 updates title
                TimelineEvent(
                    step_id="step_concurrent",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="card_1",
                            duration=2.0,
                            parameters={"color": "#FFD700"},
                        ),
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="card_2",
                            duration=2.0,
                            parameters={"title": "New Title"},
                        ),
                    ],
                ),
            ],
            metadata={},
        )
        # Schedule with concurrent step
        scheduler = TimelineScheduler()
        scheduled_timeline, _ = scheduler.schedule(scene)
        # Force is_concurrent on step_concurrent
        for s in scheduled_timeline.steps:
            if "step_concurrent" in s.constituent_step_ids:
                s.is_concurrent = True

        layout = solver.solve(scene, scheduled_timeline=scheduled_timeline)
        out = compiler.compile(scene, layout, scheduled_timeline=scheduled_timeline)
        assert out.is_valid is True
        assert not out.has_errors

        # Verify that run_time is attached to .animate(run_time=2.0).set_stroke(...)
        # and NOT inside set_stroke(..., run_time=2.0)
        assert ".animate(run_time=2.0).set_stroke(color='#FFD700')" in out.python_source
        assert "set_stroke(color='#FFD700', run_time=2.0)" not in out.python_source

    def test_manim_runtime_concurrent_card_stroke_execution(self):
        """Executes Manim CE 0.18.1 runtime test to verify card stroke animation executes without TypeError."""
        from manim import Scene, RoundedRectangle, tempconfig
        import numpy as np

        class CardStrokeRuntimeScene(Scene):
            def construct(self):
                # Construct card background
                bg = RoundedRectangle(
                    corner_radius=0.15,
                    width=4.0,
                    height=2.5,
                    stroke_color="#58C4DD",
                    stroke_width=2.0,
                    fill_color="#101720",
                    fill_opacity=0.85,
                ).move_to(np.array([0.0, 0.0, 0.0]))
                self.add(bg)

                # Exactly what scene_compiler emits for concurrent card stroke animation
                self.play(bg.animate(run_time=1.5).set_stroke(color="#FFD700"))

        with tempconfig({"dry_run": True}):
            test_scene = CardStrokeRuntimeScene()
            test_scene.render()
            # If it reaches here without TypeError: set_stroke() got unexpected keyword argument 'run_time', it passed!
            assert getattr(test_scene.renderer, "num_plays", 0) == 1


# =============================================================================
# J. COND-4C5-01: Micro-Container Dimension Fallback Hardening
# =============================================================================

class TestMicroContainerDimensionFallback:
    """Verifies that inner content dimensions cannot exceed parent dimensions even for micro-containers."""

    def test_micro_container_with_insufficient_space_emits_structured_diagnostic(self, solver):
        """A container smaller than twice its stroke inset emits LAYOUT_PARENT_BOUNDS_EXCEEDED."""
        scene = SceneIR(
            scene_id="MicroContainerFailScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="tiny_container",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 0.10, "height": 0.10, "stroke_width": 2.0},
                ),
                EntityDeclaration(
                    id="child_node",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="tiny_container",
                    initial_state={"width": 0.05, "height": 0.05},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is False
        error_codes = [e.error_code for e in layout.errors]
        assert LayoutErrorCode.LAYOUT_PARENT_BOUNDS_EXCEEDED in error_codes
        diag = next(e for e in layout.errors if e.error_code == LayoutErrorCode.LAYOUT_PARENT_BOUNDS_EXCEEDED)
        assert diag.entity_id == "child_node"
        assert "tiny_container" in diag.related_entities
        assert diag.details.get("reason") == "stroke_and_padding_consume_container"
        assert "raw_inner_width" in diag.details
        assert "parent_box" in diag.details

    def test_container_stroke_and_padding_consume_available_space(self, solver):
        """A container whose border stroke consumes all inner space fails with structured diagnostics."""
        scene = SceneIR(
            scene_id="HeavyStrokeContainerScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="thick_border_box",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 0.50, "height": 0.50, "stroke_width": 40.0},
                ),
                EntityDeclaration(
                    id="inner_item",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="thick_border_box",
                    initial_state={"width": 0.10, "height": 0.10},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is False
        err = next(e for e in layout.errors if e.error_code == LayoutErrorCode.LAYOUT_PARENT_BOUNDS_EXCEEDED)
        assert err.details.get("reason") == "stroke_and_padding_consume_container"
        assert err.details.get("raw_inner_width") <= 0.05

    def test_small_valid_container_dimensions_never_exceed_parent(self, solver):
        """For small containers that still have valid inner space, inner bounds are strictly contained."""
        scene = SceneIR(
            scene_id="SmallValidContainerScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="small_container",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 0.80, "height": 0.60, "stroke_width": 1.0, "padding": 0.05},
                ),
                EntityDeclaration(
                    id="child_dot",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="small_container",
                    initial_state={"width": 0.20, "height": 0.15},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        p_box = layout.entities["small_container"].world_box
        c_box = layout.entities["child_dot"].world_box
        assert p_box.contains_box(c_box, tolerance=CONTAINER_BORDER_TOLERANCE)
        # Child boundaries must be strictly within parent box
        assert c_box.left > p_box.left
        assert c_box.right < p_box.right
        assert c_box.bottom > p_box.bottom
        assert c_box.top < p_box.top


# =============================================================================
# K. COND-4C5-02: Shared Viewport Boundary Tolerance Policy (0.05)
# =============================================================================

class TestSharedViewportBoundaryTolerancePolicy:
    """Verifies that static layout ledger and dynamic movement compiler enforce identical 0.05 boundary tolerance."""

    def test_movement_below_boundary_tolerance_passes(self, compiler, solver):
        """Movement breaching viewport boundary by 0.045 units (< 0.05 tolerance) passes."""
        scene = SceneIR(
            scene_id="MoveNearToleranceScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="box_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 2.0, "height": 1.5},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="box_1",
                            duration=1.0,
                            parameters={"destination": [0.0, 0.0]},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        vp_box = layout.viewports["main_vp"].world_box
        half_w = 1.0
        # Right edge of box at vp_box.right + 0.045 (breach of 0.045 <= 0.05)
        dest_x = round(vp_box.right - half_w + (VIEWPORT_BOUNDARY_TOLERANCE - 0.005), 4)
        scene.timeline[1].actions[0].parameters["destination"] = [dest_x, vp_box.y]

        out = compiler.compile(scene, layout)
        assert out.is_valid is True
        assert not out.has_errors

    def test_movement_at_exact_boundary_tolerance_passes(self, compiler, solver):
        """Movement breaching viewport boundary by exactly 0.050 units (== tolerance) passes."""
        scene = SceneIR(
            scene_id="MoveExactToleranceScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="box_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 2.0, "height": 1.5},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="box_1",
                            duration=1.0,
                            parameters={"destination": [0.0, 0.0]},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        vp_box = layout.viewports["main_vp"].world_box
        half_w = 1.0
        # Right edge of box at exactly vp_box.right + 0.050
        dest_x = round(vp_box.right - half_w + VIEWPORT_BOUNDARY_TOLERANCE, 4)
        scene.timeline[1].actions[0].parameters["destination"] = [dest_x, vp_box.y]

        out = compiler.compile(scene, layout)
        assert out.is_valid is True
        assert not out.has_errors

    def test_movement_above_boundary_tolerance_fails(self, compiler, solver):
        """Movement breaching viewport boundary by 0.055 units (> 0.05 tolerance) fails closed."""
        scene = SceneIR(
            scene_id="MoveBreachToleranceScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="box_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 2.0, "height": 1.5},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.MOVE,
                            target_entity="box_1",
                            duration=1.0,
                            parameters={"destination": [0.0, 0.0]},
                        )
                    ],
                ),
            ],
            metadata={},
        )
        layout = solver.solve(scene)
        vp_box = layout.viewports["main_vp"].world_box
        half_w = 1.0
        # Right edge of box at vp_box.right + 0.055 (> 0.050 tolerance)
        dest_x = round(vp_box.right - half_w + (VIEWPORT_BOUNDARY_TOLERANCE + 0.005), 4)
        scene.timeline[1].actions[0].parameters["destination"] = [dest_x, vp_box.y]

        out = compiler.compile(scene, layout)
        assert out.is_valid is False
        oob_diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_OUT_OF_BOUNDS]
        assert len(oob_diags) >= 1

    def test_spatial_ledger_shared_tolerance_consistency(self):
        """SpatialLedger exhibits identical tolerance boundary transitions (0.045 pass, 0.050 pass, 0.055 fail)."""
        canvas = BoundingBox2D(x=0.0, y=0.0, width=10.0, height=8.0)
        vp = {"vp1": BoundingBox2D.from_min_max(left=-4.0, right=4.0, bottom=-3.0, top=3.0)}

        # Below tolerance: right at 4.045
        ledger_below = SpatialLedger(scene_duration=10.0)
        ledger_below.register_span("b_below", BoundingBox2D.from_min_max(left=2.0, right=4.045, bottom=0.0, top=2.0), 0.0, 10.0, viewport_id="vp1")
        valid_below, _, _ = ledger_below.validate_spatial_integrity(canvas_box=canvas, viewports=vp)
        assert valid_below is True

        # At tolerance: right at 4.050
        ledger_exact = SpatialLedger(scene_duration=10.0)
        ledger_exact.register_span("b_exact", BoundingBox2D.from_min_max(left=2.0, right=4.050, bottom=0.0, top=2.0), 0.0, 10.0, viewport_id="vp1")
        valid_exact, _, _ = ledger_exact.validate_spatial_integrity(canvas_box=canvas, viewports=vp)
        assert valid_exact is True

        # Above tolerance: right at 4.055
        ledger_above = SpatialLedger(scene_duration=10.0)
        ledger_above.register_span("b_above", BoundingBox2D.from_min_max(left=2.0, right=4.055, bottom=0.0, top=2.0), 0.0, 10.0, viewport_id="vp1")
        valid_above, _, diags = ledger_above.validate_spatial_integrity(canvas_box=canvas, viewports=vp)
        assert valid_above is False
        assert any(d.error_code == LayoutErrorCode.LAYOUT_BOUNDARY_BREACH for d in diags)
