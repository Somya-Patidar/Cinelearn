"""
CineLearn — Milestone 4A: Hierarchical Box Model Layout Solver Tests
===================================================================
Comprehensive test suite verifying deterministic spatial layout solving,
hierarchical containment, flexbox box-model allocation, intrinsic dimension
estimation, time-aware collision detection, and 8-step overcrowding mitigation.
"""

import copy
import pytest

from src.compiler import (
    BoundingBox2D,
    CollisionCategory,
    ComputedEntityGeometry,
    ComputedViewportGeometry,
    CoordinateSpace,
    HierarchicalBoxModelLayoutSolver,
    IntrinsicDimensionEstimator,
    LayoutDiagnostic,
    LayoutErrorCode,
    LayoutFailureReport,
    SceneSpatialLayout,
    SpatialLedger,
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


# =====================================================================
# 1. BoundingBox2D and Geometry Tests
# =====================================================================
class TestBoundingBoxAndGeometry:
    def test_bounding_box_creation_and_properties(self):
        box = BoundingBox2D.from_min_max(left=-2.0, right=4.0, bottom=-1.0, top=3.0)
        assert box.width == 6.0
        assert box.height == 4.0
        assert box.x == 1.0
        assert box.y == 1.0
        assert box.left == -2.0
        assert box.right == 4.0
        assert box.bottom == -1.0
        assert box.top == 3.0
        assert box.area == 24.0
        assert box.aspect_ratio == 1.5
        assert box.center == (1.0, 1.0)

    def test_invalid_bounds_raise_value_error(self):
        with pytest.raises(ValueError, match="left"):
            BoundingBox2D.from_min_max(left=5.0, right=2.0, bottom=-1.0, top=1.0)
        with pytest.raises(ValueError, match="bottom"):
            BoundingBox2D.from_min_max(left=-1.0, right=1.0, bottom=3.0, top=1.0)

    def test_bounding_box_containment(self):
        parent = BoundingBox2D.from_min_max(left=-3.0, right=3.0, bottom=-2.0, top=2.0)
        child_inside = BoundingBox2D.from_min_max(left=-1.0, right=1.0, bottom=-1.0, top=1.0)
        child_outside = BoundingBox2D.from_min_max(left=2.0, right=4.0, bottom=0.0, top=1.0)

        assert parent.contains_point(0.0, 0.0) is True
        assert parent.contains_point(5.0, 5.0) is False
        assert parent.contains_box(child_inside) is True
        assert parent.contains_box(child_outside) is False

    def test_bounding_box_intersections(self):
        b1 = BoundingBox2D(x=0.0, y=0.0, width=4.0, height=4.0)  # [-2, 2, -2, 2], area=16
        b2 = BoundingBox2D(x=1.0, y=1.0, width=4.0, height=4.0)  # [-1, 3, -1, 3], area=16
        b_disjoint = BoundingBox2D(x=10.0, y=10.0, width=2.0, height=2.0)

        assert b1.intersects(b2) is True
        assert b1.intersects(b_disjoint) is False

        ibox = b1.intersection_box(b2)
        assert ibox is not None
        assert ibox.left == -1.0
        assert ibox.right == 2.0
        assert ibox.bottom == -1.0
        assert ibox.top == 2.0
        assert ibox.width == 3.0
        assert ibox.height == 3.0
        assert ibox.area == 9.0

        assert b1.intersection_area(b2) == 9.0
        assert b1.overlap_ratio(b2) == 9.0 / 16.0
        assert b1.intersection_box(b_disjoint) is None
        assert b1.overlap_ratio(b_disjoint) == 0.0

    def test_bounding_box_scale_and_shift(self):
        box = BoundingBox2D(x=2.0, y=1.0, width=4.0, height=2.0)
        scaled = box.scale(0.5)
        assert scaled.x == 2.0 and scaled.y == 1.0
        assert scaled.width == 2.0 and scaled.height == 1.0

        shifted = box.shift(1.5, -0.5)
        assert shifted.x == 3.5 and shifted.y == 0.5
        assert shifted.width == 4.0 and shifted.height == 2.0


# =====================================================================
# 2. Dynamic Canvas Bounds Computation Tests
# =====================================================================
class TestDynamicCanvasComputation:
    def test_default_canvas_computation(self):
        solver = HierarchicalBoxModelLayoutSolver()
        cfg = LayoutConfiguration(
            has_title_banner=True,
            has_caption_column=True,
            has_subtitles=True,
        )
        canvas = solver.compute_canvas_box(cfg)
        assert canvas.left == pytest.approx(-2.15)
        assert canvas.right == pytest.approx(6.60)
        assert canvas.bottom == pytest.approx(-2.65)
        assert canvas.top == pytest.approx(2.55)
        assert canvas.width == pytest.approx(8.75)
        assert canvas.height == pytest.approx(5.20)
        assert canvas.x == pytest.approx(2.225)
        assert canvas.y == pytest.approx(-0.05)

    def test_canvas_without_caption_column(self):
        solver = HierarchicalBoxModelLayoutSolver()
        cfg = LayoutConfiguration(
            has_title_banner=True,
            has_caption_column=False,
            has_subtitles=True,
        )
        canvas = solver.compute_canvas_box(cfg)
        assert canvas.left == -6.60
        assert canvas.right == 6.60
        assert canvas.width == pytest.approx(13.20)

    def test_canvas_without_title_or_subtitles(self):
        solver = HierarchicalBoxModelLayoutSolver()
        cfg = LayoutConfiguration(
            has_title_banner=False,
            has_caption_column=True,
            has_subtitles=False,
        )
        canvas = solver.compute_canvas_box(cfg)
        assert canvas.top == 3.60
        assert canvas.bottom == -3.60
        assert canvas.height == pytest.approx(7.20)


# =====================================================================
# 3. Intrinsic Dimension Estimator Tests
# =====================================================================
class TestIntrinsicDimensionEstimation:
    def test_explicit_dimensions_priority(self):
        estimator = IntrinsicDimensionEstimator()
        ent = EntityDeclaration(
            id="custom_rect",
            component_type="PrimitiveShape",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="main",
            initial_state={"width": 4.5, "height": 2.2},
        )
        dim = estimator.estimate(ent)
        assert dim.width == 4.5
        assert dim.height == 2.2
        assert dim.source == "explicit_width_height"
        assert dim.is_uncertain is False

    def test_registered_component_manifest_dimensions(self):
        estimator = IntrinsicDimensionEstimator()
        ent = EntityDeclaration(
            id="chem_curve",
            component_type="ReactionEnergyDiagram",
            visual_tier=VisualTier.TIER1_REGISTERED,
            viewport="main",
            initial_state={},
        )
        dim = estimator.estimate(ent)
        assert dim.width == 6.0
        assert dim.height == 3.6
        assert dim.aspect_ratio == pytest.approx(6.0 / 3.6)
        assert dim.source in ("registered_component_manifest", "registered_component_aspect")

    def test_conceptual_card_text_heuristic(self):
        estimator = IntrinsicDimensionEstimator()
        ent = EntityDeclaration(
            id="formula_card",
            component_type="ConceptualCard",
            visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
            viewport="inspector",
            initial_state={
                "title": "Conservation of Energy",
                "formula": "E_total = const",
                "badge": "LAW",
            },
        )
        dim = estimator.estimate(ent)
        assert dim.width >= 3.2
        assert dim.height >= 1.4
        assert dim.source == "text_heuristic_with_safety_margin"
        assert dim.safety_margin_applied >= 0.15

    def test_latex_uncertainty_flag(self):
        estimator = IntrinsicDimensionEstimator()
        ent = EntityDeclaration(
            id="complex_math",
            component_type="ConceptualCard",
            visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
            viewport="inspector",
            initial_state={
                "title": "Fourier Transform",
                "formula": "\\int_{-\\infty}^{\\infty} f(t) e^{-i\\omega t} dt",
            },
        )
        dim = estimator.estimate(ent)
        assert dim.is_uncertain is True
        assert dim.safety_margin_applied == 0.25

    def test_primitive_tree_heuristics(self):
        estimator = IntrinsicDimensionEstimator()
        ent_graph = EntityDeclaration(
            id="graph",
            component_type="CompositeNodeGraph",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="main",
            initial_state={"nodes": ["N1", "N2", "N3", "N4", "N5"]},
        )
        dim_graph = estimator.estimate(ent_graph)
        assert dim_graph.width == pytest.approx(4.5)
        assert dim_graph.height == 3.2


# =====================================================================
# 4. Viewport Planning and Flexbox Allocation Tests
# =====================================================================
class TestSingleAndSiblingAllocation:
    def test_single_entity_centered_in_viewport(self):
        solver = HierarchicalBoxModelLayoutSolver()
        scene = SceneIR(
            scene_id="SingleScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={
                "main_vp": ViewportDeclaration(
                    type=ViewportType.MAIN,
                    flow=FlowDirection.SINGLE,
                    alignment=Alignment.CENTER,
                )
            },
            entities=[
                EntityDeclaration(
                    id="single_box",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 4.0, "height": 2.0},
                )
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        assert "single_box" in layout.entities
        geom = layout.entities["single_box"]
        vp_box = layout.viewports["main_vp"].world_box
        assert geom.world_box.x == pytest.approx(vp_box.x)
        assert geom.world_box.y == pytest.approx(vp_box.y)
        assert geom.world_box.width == 4.0
        assert geom.world_box.height == 2.0

    def test_multiple_siblings_column_allocation(self):
        solver = HierarchicalBoxModelLayoutSolver()
        scene = SceneIR(
            scene_id="ColumnScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={
                "main_vp": ViewportDeclaration(
                    type=ViewportType.MAIN,
                    flow=FlowDirection.COLUMN,
                    alignment=Alignment.CENTER,
                    padding=0.10,
                )
            },
            entities=[
                EntityDeclaration(
                    id="item_top",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 3.0, "height": 1.2},
                ),
                EntityDeclaration(
                    id="item_bottom",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 3.0, "height": 1.2},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        top_geom = layout.entities["item_top"]
        bottom_geom = layout.entities["item_bottom"]

        assert top_geom.world_box.y > bottom_geom.world_box.y
        # Zero spatial overlap
        assert top_geom.world_box.intersects(bottom_geom.world_box) is False

    def test_multiple_siblings_row_allocation(self):
        solver = HierarchicalBoxModelLayoutSolver()
        scene = SceneIR(
            scene_id="RowScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={
                "main_vp": ViewportDeclaration(
                    type=ViewportType.MAIN,
                    flow=FlowDirection.ROW,
                    alignment=Alignment.CENTER,
                    padding=0.10,
                )
            },
            entities=[
                EntityDeclaration(
                    id="item_left",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 2.5, "height": 2.0},
                ),
                EntityDeclaration(
                    id="item_right",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 2.5, "height": 2.0},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        left_geom = layout.entities["item_left"]
        right_geom = layout.entities["item_right"]

        assert left_geom.world_box.x < right_geom.world_box.x
        assert left_geom.world_box.intersects(right_geom.world_box) is False

    def test_sub_slots_partitioning_and_allocation(self):
        solver = HierarchicalBoxModelLayoutSolver()
        scene = SceneIR(
            scene_id="SubSlotsScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={
                "inspector_vp": ViewportDeclaration(
                    type=ViewportType.INSPECTOR,
                    flow=FlowDirection.SINGLE,
                    alignment=Alignment.CENTER,
                    sub_slots=["slot_left", "slot_right"],
                )
            },
            entities=[
                EntityDeclaration(
                    id="card_left",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="inspector_vp",
                    slot="slot_left",
                    initial_state={"width": 2.0, "height": 0.8},
                ),
                EntityDeclaration(
                    id="card_right",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="inspector_vp",
                    slot="slot_right",
                    initial_state={"width": 2.0, "height": 0.8},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        vp = layout.viewports["inspector_vp"]
        assert "slot_left" in vp.sub_slots
        assert "slot_right" in vp.sub_slots

        g_left = layout.entities["card_left"]
        g_right = layout.entities["card_right"]
        assert g_left.world_box.x < g_right.world_box.x
        assert g_left.world_box.intersects(g_right.world_box) is False


# =====================================================================
# 5. Hierarchical Containment & Coordinate Transformation Tests
# =====================================================================
class TestHierarchicalContainmentAndCoordinates:
    def test_nested_child_containment_and_coordinates(self):
        solver = HierarchicalBoxModelLayoutSolver()
        scene = SceneIR(
            scene_id="ParentChildScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={
                "main_vp": ViewportDeclaration(
                    type=ViewportType.MAIN,
                    flow=FlowDirection.SINGLE,
                    alignment=Alignment.CENTER,
                )
            },
            entities=[
                EntityDeclaration(
                    id="container_parent",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 6.0, "height": 4.0},
                ),
                EntityDeclaration(
                    id="item_child",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="container_parent",
                    initial_state={"width": 1.5, "height": 1.0},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        p_geom = layout.entities["container_parent"]
        c_geom = layout.entities["item_child"]

        assert c_geom.parent_id == "container_parent"
        assert p_geom.world_box.contains_box(c_geom.world_box, tolerance=0.15) is True

    def test_parent_bounds_exceeded_diagnostic(self):
        solver = HierarchicalBoxModelLayoutSolver()
        scene = SceneIR(
            scene_id="ChildExceedsScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={
                "main_vp": ViewportDeclaration(type=ViewportType.MAIN)
            },
            entities=[
                EntityDeclaration(
                    id="small_parent",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 2.0, "height": 1.0},
                ),
                EntityDeclaration(
                    id="giant_child",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="small_parent",
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

    def test_cyclic_hierarchy_detection(self):
        solver = HierarchicalBoxModelLayoutSolver()
        scene = SceneIR(
            scene_id="CyclicScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main_vp": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="node_a",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="node_b",
                    initial_state={"width": 2.0, "height": 2.0},
                ),
                EntityDeclaration(
                    id="node_b",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    parent_id="node_a",
                    initial_state={"width": 2.0, "height": 2.0},
                ),
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is False
        error_codes = [e.error_code for e in layout.errors]
        assert LayoutErrorCode.LAYOUT_CYCLIC_HIERARCHY in error_codes


# =====================================================================
# 6. Spatial Ledger and Collision Category Tests
# =====================================================================
class TestSpatialLedgerAndCollisionCategories:
    def test_sequential_replacement_valid(self):
        ledger = SpatialLedger(scene_duration=20.0)
        box = BoundingBox2D(x=1.0, y=0.0, width=4.0, height=3.0)
        ledger.register_span("ent_a", box, t_start=0.0, t_end=6.0)
        ledger.register_span("ent_b", box, t_start=6.0, t_end=12.0)

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.SEQUENTIAL_REPLACEMENT
        assert colls[0].is_fatal is False

    def test_intentional_layering_valid(self):
        ledger = SpatialLedger(scene_duration=20.0)
        card_box = BoundingBox2D(x=1.0, y=0.0, width=4.0, height=3.0)
        badge_box = BoundingBox2D(x=1.0, y=1.0, width=1.5, height=0.6)

        ledger.register_span("card", card_box, t_start=0.0, t_end=10.0, z_index=10)
        ledger.register_span("badge", badge_box, t_start=0.0, t_end=10.0, z_index=20, layer_role="badge")

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.INTENTIONAL_LAYERING
        assert colls[0].is_fatal is False

    def test_intentional_containment_valid(self):
        ledger = SpatialLedger(scene_duration=20.0)
        pbox = BoundingBox2D(x=0.0, y=0.0, width=5.0, height=4.0)
        cbox = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=1.5)

        ledger.register_span("parent", pbox, t_start=0.0, t_end=10.0)
        ledger.register_span("child", cbox, t_start=0.0, t_end=10.0, parent_id="parent")

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.INTENTIONAL_CONTAINMENT
        assert colls[0].is_fatal is False

    def test_temporary_transition_valid(self):
        ledger = SpatialLedger(scene_duration=20.0)
        box1 = BoundingBox2D(x=0.0, y=0.0, width=3.0, height=3.0)
        box2 = BoundingBox2D(x=1.0, y=0.0, width=3.0, height=3.0)

        ledger.register_span("stationary", box1, t_start=0.0, t_end=10.0)
        ledger.register_span("moving", box2, t_start=2.0, t_end=2.6, move_duration=0.60)

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.TEMPORARY_TRANSITION
        assert colls[0].is_fatal is False

    def test_forbidden_overlap_fatal(self):
        ledger = SpatialLedger(scene_duration=20.0)
        box = BoundingBox2D(x=0.0, y=0.0, width=4.0, height=3.0)

        ledger.register_span("clash_1", box, t_start=0.0, t_end=10.0, z_index=10)
        ledger.register_span("clash_2", box, t_start=2.0, t_end=8.0, z_index=10)

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.FORBIDDEN_OVERLAP
        assert colls[0].is_fatal is True


# =====================================================================
# 7. Overcrowding Mitigation Hierarchy Tests
# =====================================================================
class TestOvercrowdingMitigationHierarchy:
    def test_mitigation_proportional_scaling_floored(self):
        solver = HierarchicalBoxModelLayoutSolver()
        # Viewport height is ~5.20. 3 entities each requiring 2.0 height -> total ~6.40 > 5.20
        # Scale needed ~5.20 / 6.40 = ~0.81 (which is >= 0.70 floor) -> should succeed via Step 7 scaling!
        scene = SceneIR(
            scene_id="ScalingScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={
                "main_vp": ViewportDeclaration(type=ViewportType.MAIN, flow=FlowDirection.COLUMN)
            },
            entities=[
                EntityDeclaration(
                    id=f"card_{i}",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 4.0, "height": 2.0},
                )
                for i in range(3)
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        # Check scale factor was reduced
        for eid, geom in layout.entities.items():
            assert geom.scale < 1.0
            assert geom.scale >= 0.70
        assert any("Step 7: proportional_scaling_floored" in m for m in layout.mitigations_applied)

    def test_mitigation_failure_emits_structured_report(self):
        solver = HierarchicalBoxModelLayoutSolver()
        # Viewport height is ~5.20. 8 entities each requiring 2.0 height -> total ~17.0 > 5.20
        # Required scale ~0.30 (< 0.70 floor) -> Step 7 fails, triggers Step 8 LayoutFailureReport!
        scene = SceneIR(
            scene_id="ImpossibleOvercrowdingScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={
                "main_vp": ViewportDeclaration(type=ViewportType.MAIN, flow=FlowDirection.COLUMN)
            },
            entities=[
                EntityDeclaration(
                    id=f"huge_card_{i}",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 4.0, "height": 2.0},
                )
                for i in range(8)
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is False
        assert layout.failure_report is not None
        assert layout.failure_report.status == "LAYOUT_OVERCROWDING_ERROR"
        assert layout.failure_report.spatial_deficit > 0
        assert layout.failure_report.recommended_action == "SPLIT_INTO_SEQUENTIAL_BEATS"


# =====================================================================
# 8. Edge Cases & Determinism Tests
# =====================================================================
class TestEdgeCasesAndDeterminism:
    def test_empty_scene_solves_cleanly(self):
        solver = HierarchicalBoxModelLayoutSolver()
        scene = SceneIR(
            scene_id="EmptyScene",
            duration_budget=5.0,
            layout_configuration=LayoutConfiguration(),
            viewports={},
            entities=[],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        assert len(layout.entities) == 0
        assert len(layout.errors) == 0

    def test_empty_viewport_handled_without_crash(self):
        solver = HierarchicalBoxModelLayoutSolver()
        scene = SceneIR(
            scene_id="UnusedViewportScene",
            duration_budget=5.0,
            layout_configuration=LayoutConfiguration(),
            viewports={
                "main_vp": ViewportDeclaration(type=ViewportType.MAIN),
                "empty_inspector": ViewportDeclaration(type=ViewportType.INSPECTOR),
            },
            entities=[
                EntityDeclaration(
                    id="main_box",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main_vp",
                    initial_state={"width": 3.0, "height": 2.0},
                )
            ],
            timeline=[],
            metadata={},
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        assert "empty_inspector" in layout.viewports
        assert len(layout.viewports["empty_inspector"].assigned_entities) == 0

    def test_deterministic_repeatability(self):
        solver = HierarchicalBoxModelLayoutSolver()
        bs = get_binary_search_fixture()

        layout1 = solver.solve(bs)
        layout2 = solver.solve(bs)

        for eid in layout1.entities:
            b1 = layout1.entities[eid].world_box
            b2 = layout2.entities[eid].world_box
            assert b1.x == b2.x
            assert b1.y == b2.y
            assert b1.width == b2.width
            assert b1.height == b2.height


# =====================================================================
# 9. Canonical Domain Fixtures & OS CPU Scheduling Fixture Tests
# =====================================================================
class TestCanonicalAndOSDomainFixtures:
    def test_binary_search_canonical_fixture(self):
        solver = HierarchicalBoxModelLayoutSolver()
        bs = get_binary_search_fixture()
        layout = solver.solve(bs)

        assert layout.is_valid is True
        assert len(layout.errors) == 0
        assert "search_array" in layout.entities
        assert "formula_card" in layout.entities
        assert "pointer_low" in layout.entities
        assert "pointer_high" in layout.entities

        # Parent-child containment
        search_box = layout.entities["search_array"].world_box
        p_low_box = layout.entities["pointer_low"].world_box
        p_high_box = layout.entities["pointer_high"].world_box
        assert search_box.contains_box(p_low_box, tolerance=0.15)
        assert search_box.contains_box(p_high_box, tolerance=0.15)

    def test_chemical_kinetics_canonical_fixture(self):
        solver = HierarchicalBoxModelLayoutSolver()
        ck = get_chemical_kinetics_fixture()
        layout = solver.solve(ck)

        assert layout.is_valid is True
        assert len(layout.errors) == 0
        assert "energy_diagram" in layout.entities
        assert "arrhenius_equation" in layout.entities

    def test_pythagorean_theorem_canonical_fixture(self):
        solver = HierarchicalBoxModelLayoutSolver()
        pt = get_pythagorean_theorem_fixture()
        layout = solver.solve(pt)

        assert layout.is_valid is True
        assert len(layout.errors) == 0
        assert "square_a_plus_b_left" in layout.entities
        assert "square_a_plus_b_right" in layout.entities
        assert "algebraic_derivation" in layout.entities

        # Disjoint split viewports
        left_box = layout.entities["square_a_plus_b_left"].world_box
        right_box = layout.entities["square_a_plus_b_right"].world_box
        assert left_box.intersects(right_box) is False

    def test_os_cpu_scheduling_scene_fixture(self):
        """
        Tests an OS CPU scheduling scene (ReadyQueue, CPUExecution, GanttChart, ProcessCard)
        verifying that the generic solver handles an OS composition without hardcoded OS rules.
        """
        scene = SceneIR(
            scene_id="FCFSSchedulingScene",
            title="FCFS Process Execution Demonstration",
            duration_budget=20.0,
            layout_configuration=LayoutConfiguration(
                has_title_banner=True,
                has_caption_column=True,
                has_subtitles=True,
            ),
            viewports={
                "queue_vp": ViewportDeclaration(
                    type=ViewportType.SPLIT_LEFT,
                    flow=FlowDirection.COLUMN,
                    alignment=Alignment.CENTER,
                ),
                "cpu_vp": ViewportDeclaration(
                    type=ViewportType.SPLIT_RIGHT,
                    flow=FlowDirection.SINGLE,
                    alignment=Alignment.CENTER,
                ),
                "gantt_vp": ViewportDeclaration(
                    type=ViewportType.INSPECTOR,
                    flow=FlowDirection.SINGLE,
                    alignment=Alignment.CENTER,
                ),
            },
            entities=[
                EntityDeclaration(
                    id="ready_queue",
                    component_type="ReadyQueueVisual",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="queue_vp",
                    initial_state={"capacity": 4, "width": 4.0, "height": 1.6},
                ),
                EntityDeclaration(
                    id="cpu_core",
                    component_type="CPUExecutionVisual",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="cpu_vp",
                    initial_state={"width": 3.8, "height": 2.2},
                ),
                EntityDeclaration(
                    id="gantt_chart",
                    component_type="GanttChartVisual",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="gantt_vp",
                    initial_state={"total_time": 10, "width": 8.0, "height": 1.0},
                ),
                EntityDeclaration(
                    id="process_p1",
                    component_type="ProcessCardVisual",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="queue_vp",
                    parent_id="ready_queue",
                    initial_state={"pid": "P1", "burst_time": 4, "arrival_time": 0, "width": 1.2, "height": 1.0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 5.0)),
                    actions=[
                        ActionDeclaration(type=ActionType.ENTER, target_entity="ready_queue", duration=1.0),
                        ActionDeclaration(type=ActionType.ENTER, target_entity="cpu_core", duration=1.0),
                        ActionDeclaration(type=ActionType.ENTER, target_entity="gantt_chart", duration=1.0),
                        ActionDeclaration(type=ActionType.ENTER, target_entity="process_p1", duration=0.8),
                    ],
                )
            ],
            metadata={},
        )

        solver = HierarchicalBoxModelLayoutSolver()
        layout = solver.solve(scene)

        assert layout.is_valid is True
        assert len(layout.errors) == 0
        assert len(layout.entities) == 4

        # Ready queue in split-left
        q_box = layout.entities["ready_queue"].world_box
        assert q_box.right <= 2.10

        # CPU core in split-right
        cpu_box = layout.entities["cpu_core"].world_box
        assert cpu_box.left >= 2.35

        # Gantt chart in bottom inspector
        gantt_box = layout.entities["gantt_chart"].world_box
        assert gantt_box.bottom >= -2.65
        assert gantt_box.top <= -1.60

        # Process card inside ready queue
        p1_box = layout.entities["process_p1"].world_box
        assert q_box.contains_box(p1_box, tolerance=0.15)


# =====================================================================
# 11. Milestone 4A Remediation Regression Tests
# =====================================================================
class TestMilestone4ARemediationRegressions:
    """
    Focused regression tests reproducing and verifying fixes for all 5 audit findings:
    1. Multi-level child allocation topological resolution (Remediation 1).
    2. Move-window scoped temporary transition collision classification (Remediation 2).
    3. Multi-lifecycle entity lifespan handling (Remediation 3).
    4. Configurable canvas geometry with custom aspect ratios & padding (Remediation 4).
    5. Robust dimension estimation for non-numeric attributes (Robustness).
    """

    def test_multi_level_child_out_of_order_declaration(self):
        """
        Verifies that declaring a grandchild before its parent does not cause it to be silently dropped.
        Hierarchy: grandparent (top-level) -> parent (child of grandparent) -> grandchild (child of parent).
        Declaration order in entities: grandchild, grandparent, parent.
        """
        scene = SceneIR(
            scene_id="MultiLevelOrderScene",
            duration_budget=15.0,
            layout_configuration=LayoutConfiguration(),
            viewports={
                "main": ViewportDeclaration(type=ViewportType.MAIN),
            },
            entities=[
                # Grandchild declared FIRST
                EntityDeclaration(
                    id="grandchild",
                    component_type="card",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main",
                    parent_id="child",
                    initial_state={"width": 1.0, "height": 0.8},
                ),
                # Grandparent declared SECOND
                EntityDeclaration(
                    id="grandparent",
                    component_type="container",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main",
                    initial_state={"width": 6.0, "height": 4.0},
                ),
                # Child declared THIRD
                EntityDeclaration(
                    id="child",
                    component_type="container",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main",
                    parent_id="grandparent",
                    initial_state={"width": 3.0, "height": 2.0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="evt1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 5.0)),
                    actions=[
                        ActionDeclaration(type=ActionType.ENTER, target_entity="grandparent", duration=1.0),
                        ActionDeclaration(type=ActionType.ENTER, target_entity="child", duration=1.0),
                        ActionDeclaration(type=ActionType.ENTER, target_entity="grandchild", duration=1.0),
                    ],
                )
            ],
            metadata={},
        )
        solver = HierarchicalBoxModelLayoutSolver()
        layout = solver.solve(scene)

        # All 3 entities must be resolved in layout
        assert "grandparent" in layout.entities
        assert "child" in layout.entities
        assert "grandchild" in layout.entities
        assert len(layout.entities) == 3

        # Containment verification
        gp_box = layout.entities["grandparent"].world_box
        c_box = layout.entities["child"].world_box
        gc_box = layout.entities["grandchild"].world_box

        assert gp_box.contains_box(c_box, tolerance=0.15)
        assert c_box.contains_box(gc_box, tolerance=0.15)

    def test_unresolved_missing_parent_emits_diagnostic(self):
        """
        Verifies that an entity referencing a non-existent parent_id produces a
        LAYOUT_UNSATISFIABLE_CONSTRAINTS diagnostic instead of being silently dropped.
        """
        scene = SceneIR(
            scene_id="OrphanChildScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="orphan",
                    component_type="card",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="main",
                    parent_id="ghost_parent",  # Does not exist
                    initial_state={"width": 1.0, "height": 1.0},
                )
            ],
            timeline=[],
            metadata={},
        )
        solver = HierarchicalBoxModelLayoutSolver()
        layout = solver.solve(scene)
        assert layout.is_valid is False
        assert any(e.error_code == LayoutErrorCode.LAYOUT_UNSATISFIABLE_CONSTRAINTS for e in layout.errors)

    def test_move_window_scoping_prevents_false_positive_temporary_transition(self):
        """
        Verifies that a 10s static overlap between two entities is NOT classified as
        TEMPORARY_TRANSITION just because one entity has a 0.5s move action somewhere.
        """
        ledger = SpatialLedger(scene_duration=20.0)
        box = BoundingBox2D(x=0.0, y=0.0, width=3.0, height=3.0)

        # Both entities overlap identically for 10 seconds
        ledger.register_span("ent_a", box, t_start=0.0, t_end=10.0)
        # ent_b is active for 10 seconds, but only has a move action from 2.0 to 2.5
        ledger.register_span(
            "ent_b",
            box,
            t_start=0.0,
            t_end=10.0,
            move_duration=0.5,
        )

        colls = ledger.check_collisions()
        # Must be fatal FORBIDDEN_OVERLAP because of the static 10s overlap
        assert any(c.category == CollisionCategory.FORBIDDEN_OVERLAP and c.is_fatal for c in colls)

    def test_move_window_scoping_allows_true_temporary_transition(self):
        """
        Verifies that an intersection occurring strictly during a short move action (<= 0.8s)
        is correctly classified as non-fatal TEMPORARY_TRANSITION.
        """
        ledger = SpatialLedger(scene_duration=20.0)
        box1 = BoundingBox2D(x=0.0, y=0.0, width=3.0, height=3.0)
        box2 = BoundingBox2D(x=0.5, y=0.0, width=3.0, height=3.0)

        # Stationary entity at location
        ledger.register_span("stationary", box1, t_start=0.0, t_end=10.0)
        # Moving entity passes through the location during a 0.6s move window
        ledger.register_span(
            "moving",
            box2,
            t_start=2.0,
            t_end=2.6,
            move_duration=0.60,
        )

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.TEMPORARY_TRANSITION
        assert colls[0].is_fatal is False

    def test_multi_lifecycle_disjoint_lifespans(self):
        """
        Verifies that an entity exiting and re-entering does not cause collisions
        during its interim exit window when another entity occupies the same space.
        """
        ledger = SpatialLedger(scene_duration=20.0)
        box = BoundingBox2D(x=0.0, y=0.0, width=3.0, height=3.0)

        # Entity A: active [0, 4] and [10, 14]
        ledger.register_span(
            "entity_a",
            box,
            t_start=0.0,
            t_end=14.0,
            intervals=[(0.0, 4.0), (10.0, 14.0)],
        )
        # Entity B: active [5, 9] (strictly inside A's absence gap)
        ledger.register_span(
            "entity_b",
            box,
            t_start=5.0,
            t_end=9.0,
            intervals=[(5.0, 9.0)],
        )

        colls = ledger.check_collisions()
        # Should be SEQUENTIAL_REPLACEMENT (non-fatal), NOT FORBIDDEN_OVERLAP
        assert all(not c.is_fatal for c in colls)

    def test_dynamic_aspect_ratio_and_safety_padding_canvas(self):
        """
        Verifies that compute_canvas_box respects non-16:9 aspect ratios (e.g. 4:3, 9:16)
        and custom safety_padding.
        """
        solver = HierarchicalBoxModelLayoutSolver()

        # 4:3 aspect ratio
        cfg_43 = LayoutConfiguration(
            aspect_ratio="4:3",
            has_title_banner=False,
            has_caption_column=False,
            has_subtitles=False,
            safety_padding=0.20,
        )
        canvas_43 = solver.compute_canvas_box(cfg_43)
        assert canvas_43.width < 12.0
        assert canvas_43.height == pytest.approx(7.20)

        # Custom safety padding
        cfg_pad = LayoutConfiguration(
            aspect_ratio="16:9",
            has_title_banner=False,
            has_caption_column=False,
            has_subtitles=False,
            safety_padding=0.35,
        )
        canvas_pad = solver.compute_canvas_box(cfg_pad)
        assert canvas_pad.height == pytest.approx(6.60)
        assert canvas_pad.top == pytest.approx(3.30)
        assert canvas_pad.bottom == pytest.approx(-3.30)

    def test_dimension_estimator_non_numeric_fallback(self):
        """
        Verifies that string/non-numeric values in initial_state dimensions (e.g. 'auto')
        do not raise ValueError, but gracefully fall back.
        """
        estimator = IntrinsicDimensionEstimator()
        entity = EntityDeclaration(
            id="robust_entity",
            component_type="card",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="main",
            initial_state={"width": "auto", "height": "100%"},
        )
        dim = estimator.estimate(entity)
        assert dim.width > 0
        assert dim.height > 0

    def test_category4_composite_grouping_valid(self):
        """
        Direct assertion test for Collision Category 4: COMPOSITE_GROUPING.
        Verifies that overlapping elements sharing a declared group_id are classified as non-fatal.
        """
        ledger = SpatialLedger(scene_duration=20.0)
        box1 = BoundingBox2D(x=0.0, y=0.0, width=3.0, height=3.0)
        box2 = BoundingBox2D(x=0.5, y=0.0, width=3.0, height=3.0)

        ledger.register_span("comp_node_1", box1, t_start=0.0, t_end=10.0, group_id="composite_graph_1")
        ledger.register_span("comp_node_2", box2, t_start=0.0, t_end=10.0, group_id="composite_graph_1")

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.COMPOSITE_GROUPING
        assert colls[0].is_fatal is False

    def test_category7_shared_sub_slot_valid(self):
        """
        Direct assertion test for Collision Category 7: SHARED_SUB_SLOT.
        Verifies that elements assigned to distinct partitioned sub-slots are classified as non-fatal.
        """
        ledger = SpatialLedger(scene_duration=20.0)
        box1 = BoundingBox2D(x=0.0, y=0.0, width=3.0, height=3.0)
        box2 = BoundingBox2D(x=0.5, y=0.0, width=3.0, height=3.0)

        ledger.register_span("slot_ent_1", box1, t_start=0.0, t_end=10.0, viewport_id="main", slot="sub_slot_left")
        ledger.register_span("slot_ent_2", box2, t_start=0.0, t_end=10.0, viewport_id="main", slot="sub_slot_right")

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.SHARED_SUB_SLOT
        assert colls[0].is_fatal is False


