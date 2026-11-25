"""
CineLearn — Milestone 4C.1: Layout Engine Hardening Tests
=========================================================
Verifies:
1. Multi-lifecycle interval handling in HierarchicalBoxModelLayoutSolver._check_sequential_lifespans().
2. Simultaneous movement-window union in SpatialLedger.check_collisions().
3. Regression preservation across layout solving and collision classification.
"""

import pytest
from src.compiler.layout_engine import HierarchicalBoxModelLayoutSolver
from src.compiler.spatial_ledger import SpatialLedger
from src.compiler.layout_models import BoundingBox2D, CollisionCategory
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
# 1. Fix A: Multi-Lifecycle Interval Tests (_check_sequential_lifespans)
# =====================================================================
class TestMultiLifecycleSequentialLifespans:
    @pytest.fixture
    def solver(self):
        return HierarchicalBoxModelLayoutSolver()

    def test_single_lifecycle_interval(self, solver):
        """Entity A active [0, 5], Entity B active [5, 10] are sequential."""
        e1 = EntityDeclaration(
            id="A",
            component_type="PrimitiveShape",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={"width": 4.0, "height": 2.0},
        )
        e2 = EntityDeclaration(
            id="B",
            component_type="PrimitiveShape",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={"width": 4.0, "height": 2.0},
        )
        timeline = [
            TimelineEvent(
                step_id="step_1",
                narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 5.0)),
                actions=[
                    ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5),
                    ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5),
                ],
            ),
            TimelineEvent(
                step_id="step_2",
                narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(5.0, 10.0)),
                actions=[
                    ActionDeclaration(type=ActionType.ENTER, target_entity="B", duration=0.5),
                    ActionDeclaration(type=ActionType.EXIT, target_entity="B", duration=0.5),
                ],
            ),
        ]
        assert solver._check_sequential_lifespans([e1, e2], timeline) is True

    def test_two_disjoint_intervals_reentry(self, solver):
        """Entity A active [0, 4] and [10, 14], Entity B active [4, 10] are sequential."""
        e1 = EntityDeclaration(
            id="A",
            component_type="PrimitiveShape",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={"width": 4.0, "height": 2.0},
        )
        e2 = EntityDeclaration(
            id="B",
            component_type="PrimitiveShape",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={"width": 4.0, "height": 2.0},
        )
        timeline = [
            TimelineEvent(
                step_id="step_1",
                narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 4.0)),
                actions=[
                    ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5),
                    ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5),
                ],
            ),
            TimelineEvent(
                step_id="step_2",
                narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(4.0, 10.0)),
                actions=[
                    ActionDeclaration(type=ActionType.ENTER, target_entity="B", duration=0.5),
                    ActionDeclaration(type=ActionType.EXIT, target_entity="B", duration=0.5),
                ],
            ),
            TimelineEvent(
                step_id="step_3",
                narration_anchor=NarrationAnchor(lecture_line_index=2, audio_timestamp_range=(10.0, 14.0)),
                actions=[
                    ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5),
                    ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5),
                ],
            ),
        ]
        # Previous code collapsed A into [0, 14] and failed; new code recognizes disjoint spans
        assert solver._check_sequential_lifespans([e1, e2], timeline) is True

    def test_three_reentries(self, solver):
        """Entity A active [1, 3], [5, 7], [9, 11]; Entity B active [3, 5], [7, 9]."""
        e1 = EntityDeclaration(id="A", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")
        e2 = EntityDeclaration(id="B", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")

        timeline = [
            TimelineEvent(step_id="s1", narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(1.0, 3.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5)]),
            TimelineEvent(step_id="s2", narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(3.0, 5.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="B", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="B", duration=0.5)]),
            TimelineEvent(step_id="s3", narration_anchor=NarrationAnchor(lecture_line_index=2, audio_timestamp_range=(5.0, 7.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5)]),
            TimelineEvent(step_id="s4", narration_anchor=NarrationAnchor(lecture_line_index=3, audio_timestamp_range=(7.0, 9.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="B", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="B", duration=0.5)]),
            TimelineEvent(step_id="s5", narration_anchor=NarrationAnchor(lecture_line_index=4, audio_timestamp_range=(9.0, 11.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5)]),
        ]
        assert solver._check_sequential_lifespans([e1, e2], timeline) is True

    def test_genuine_overlap_returns_false(self, solver):
        """Entity A active [0, 6], Entity B active [4, 10] overlap during [4, 6]."""
        e1 = EntityDeclaration(id="A", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")
        e2 = EntityDeclaration(id="B", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")

        timeline = [
            TimelineEvent(step_id="s1", narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 6.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5)]),
            TimelineEvent(step_id="s2", narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(4.0, 10.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="B", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="B", duration=0.5)]),
        ]
        assert solver._check_sequential_lifespans([e1, e2], timeline) is False

    def test_touching_knife_edge_boundaries(self, solver):
        """Entity A ends at 4.0, Entity B begins at 4.0 -> touching is not overlapping."""
        e1 = EntityDeclaration(id="A", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")
        e2 = EntityDeclaration(id="B", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")

        timeline = [
            TimelineEvent(step_id="s1", narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 4.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5)]),
            TimelineEvent(step_id="s2", narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(4.0, 8.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="B", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="B", duration=0.5)]),
        ]
        assert solver._check_sequential_lifespans([e1, e2], timeline) is True

    def test_unmatched_enter_remains_active(self, solver):
        """Entity A has ENTER at 0.0 with NO exit -> active through end of scene; overlaps with B at 5.0."""
        e1 = EntityDeclaration(id="A", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")
        e2 = EntityDeclaration(id="B", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")

        timeline = [
            TimelineEvent(step_id="s1", narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 5.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5)]),
            TimelineEvent(step_id="s2", narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(5.0, 10.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="B", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="B", duration=0.5)]),
        ]
        # A never exited, so B overlaps with A
        assert solver._check_sequential_lifespans([e1, e2], timeline) is False

    def test_unmatched_initial_exit(self, solver):
        """Entity A starts present and exits at 3.0 (no enter); B enters at 4.0 and exits at 7.0 -> sequential."""
        e1 = EntityDeclaration(id="A", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")
        e2 = EntityDeclaration(id="B", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")

        timeline = [
            TimelineEvent(step_id="s1", narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 3.0)),
                          actions=[ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5)]),
            TimelineEvent(step_id="s2", narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(4.0, 7.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="B", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="B", duration=0.5)]),
        ]
        assert solver._check_sequential_lifespans([e1, e2], timeline) is True

    def test_deterministic_repeated_evaluation(self, solver):
        """Repeated calls produce identical boolean classifications."""
        e1 = EntityDeclaration(id="A", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")
        e2 = EntityDeclaration(id="B", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")

        timeline = [
            TimelineEvent(step_id="s1", narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 4.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5)]),
            TimelineEvent(step_id="s2", narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(4.0, 8.0)),
                          actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="B", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="B", duration=0.5)]),
        ]
        res1 = solver._check_sequential_lifespans([e1, e2], timeline)
        res2 = solver._check_sequential_lifespans([e1, e2], timeline)
        assert res1 == res2 is True

    def test_single_flow_layout_sequential_replacement_integration(self, solver):
        """End-to-end solver test: alternating entities in FlowDirection.SINGLE allocate sequential replacement."""
        scene = SceneIR(
            scene_id="SequentialReplaceScene",
            duration_budget=14.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN, flow=FlowDirection.SINGLE)},
            entities=[
                EntityDeclaration(id="A", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main", initial_state={"width": 4.0, "height": 2.0}),
                EntityDeclaration(id="B", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main", initial_state={"width": 4.0, "height": 2.0}),
            ],
            timeline=[
                TimelineEvent(step_id="s1", narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 4.0)),
                              actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5)]),
                TimelineEvent(step_id="s2", narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(4.0, 10.0)),
                              actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="B", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="B", duration=0.5)]),
                TimelineEvent(step_id="s3", narration_anchor=NarrationAnchor(lecture_line_index=2, audio_timestamp_range=(10.0, 14.0)),
                              actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="A", duration=0.5), ActionDeclaration(type=ActionType.EXIT, target_entity="A", duration=0.5)]),
            ],
        )
        layout = solver.solve(scene)
        assert layout.is_valid is True
        # Both entities must be allocated at full scale (scale=1.0) via sequential replacement
        assert layout.entities["A"].scale == pytest.approx(1.0)
        assert layout.entities["B"].scale == pytest.approx(1.0)
        # Verify mitigation list recorded Step 6
        assert any("Step 6: sequential_visual_replacement" in m for m in layout.mitigations_applied)


# =====================================================================
# 2. Fix B: Movement-Window Union Tests (SpatialLedger.check_collisions)
# =====================================================================
class TestMovementWindowUnionCollisions:
    def test_original_m4a_staggered_dual_move_resolved(self):
        """
        Original M4A Scenario 6:
        Entity A moves during [2.0, 2.5]
        Entity B moves during [2.3, 2.8]
        Overlap interval is [2.2, 2.6] (0.4s <= 0.8s).
        Union [2.0, 2.8] continuously covers [2.2, 2.6].
        Must be classified as non-fatal TEMPORARY_TRANSITION.
        """
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)
        ledger = SpatialLedger(scene_duration=20.0)
        ledger.register_span("A", box, intervals=[(2.2, 2.6)], move_windows=[(2.0, 2.5)])
        ledger.register_span("B", box, intervals=[(2.2, 2.6)], move_windows=[(2.3, 2.8)])

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.TEMPORARY_TRANSITION
        assert colls[0].is_fatal is False
        assert "Temporary transition trajectory intersection" in colls[0].description

    def test_two_overlapping_movement_windows_single_entity(self):
        """Entity A has two contiguous movement windows [1.0, 1.4] and [1.35, 1.75]; overlap [1.1, 1.6]."""
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)
        ledger = SpatialLedger(scene_duration=20.0)
        ledger.register_span("A", box, intervals=[(1.1, 1.6)], move_windows=[(1.0, 1.4), (1.35, 1.75)])
        ledger.register_span("B", box, intervals=[(1.1, 1.6)])

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.TEMPORARY_TRANSITION
        assert colls[0].is_fatal is False

    def test_three_movement_windows_union_covers_overlap(self):
        """A moves [2.0, 2.3], B moves [2.25, 2.55], A moves [2.5, 2.8]; overlap [2.1, 2.7] (0.6s <= 0.8s)."""
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)
        ledger = SpatialLedger(scene_duration=20.0)
        ledger.register_span("A", box, intervals=[(2.1, 2.7)], move_windows=[(2.0, 2.3), (2.5, 2.8)])
        ledger.register_span("B", box, intervals=[(2.1, 2.7)], move_windows=[(2.25, 2.55)])

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.TEMPORARY_TRANSITION
        assert colls[0].is_fatal is False

    def test_uncovered_gap_between_windows_is_forbidden(self):
        """A moves [2.0, 2.3], B moves [2.5, 2.8]; gap [2.3, 2.5] is uncovered; overlap [2.1, 2.7]."""
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)
        ledger = SpatialLedger(scene_duration=20.0)
        ledger.register_span("A", box, intervals=[(2.1, 2.7)], move_windows=[(2.0, 2.3)])
        ledger.register_span("B", box, intervals=[(2.1, 2.7)], move_windows=[(2.5, 2.8)])

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.FORBIDDEN_OVERLAP
        assert colls[0].is_fatal is True

    def test_partial_coverage_is_forbidden(self):
        """Move window [2.0, 2.5] covers only part of overlap [1.0, 2.5] -> FORBIDDEN_OVERLAP."""
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)
        ledger = SpatialLedger(scene_duration=20.0)
        ledger.register_span("A", box, intervals=[(1.0, 2.5)], move_windows=[(2.0, 2.5)])
        ledger.register_span("B", box, intervals=[(1.0, 2.5)])

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.FORBIDDEN_OVERLAP
        assert colls[0].is_fatal is True

    def test_unrelated_entity_move_windows_not_used(self):
        """Entities A and B overlap statically [0, 10]; unrelated Entity C moves [2.0, 2.5]. Collision between A and B is fatal."""
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)
        ledger = SpatialLedger(scene_duration=20.0)
        ledger.register_span("A", box, intervals=[(0.0, 10.0)])
        ledger.register_span("B", box, intervals=[(0.0, 10.0)])
        ledger.register_span("C", box, intervals=[(2.0, 2.5)], move_windows=[(2.0, 2.5)])

        colls = [c for c in ledger.check_collisions() if {c.entity_a, c.entity_b} == {"A", "B"}]
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.FORBIDDEN_OVERLAP
        assert colls[0].is_fatal is True

    def test_transition_duration_at_exact_boundary(self):
        """Overlap duration exactly 0.80s is accepted as TEMPORARY_TRANSITION."""
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)
        ledger = SpatialLedger(scene_duration=20.0)
        ledger.register_span("A", box, intervals=[(2.0, 2.8)], move_windows=[(2.0, 2.8)])
        ledger.register_span("B", box, intervals=[(2.0, 2.8)])

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.TEMPORARY_TRANSITION
        assert colls[0].is_fatal is False

    def test_transition_duration_above_boundary_rejected(self):
        """Overlap duration 0.82s (>0.80s) is rejected as FORBIDDEN_OVERLAP."""
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)
        ledger = SpatialLedger(scene_duration=20.0)
        ledger.register_span("A", box, intervals=[(2.0, 2.82)], move_windows=[(2.0, 2.82)])
        ledger.register_span("B", box, intervals=[(2.0, 2.82)])

        colls = ledger.check_collisions()
        assert len(colls) == 1
        assert colls[0].category == CollisionCategory.FORBIDDEN_OVERLAP
        assert colls[0].is_fatal is True

    def test_deterministic_reverse_input_ordering(self):
        """Move windows registered in reverse chronological order yield identical result."""
        box = BoundingBox2D(x=0.0, y=0.0, width=2.0, height=2.0)

        ledger1 = SpatialLedger(scene_duration=20.0)
        ledger1.register_span("A", box, intervals=[(2.2, 2.6)], move_windows=[(2.0, 2.5)])
        ledger1.register_span("B", box, intervals=[(2.2, 2.6)], move_windows=[(2.3, 2.8)])

        ledger2 = SpatialLedger(scene_duration=20.0)
        ledger2.register_span("A", box, intervals=[(2.2, 2.6)], move_windows=[(2.3, 2.8)])
        ledger2.register_span("B", box, intervals=[(2.2, 2.6)], move_windows=[(2.0, 2.5)])

        colls1 = ledger1.check_collisions()
        colls2 = ledger2.check_collisions()

        assert colls1[0].category == colls2[0].category == CollisionCategory.TEMPORARY_TRANSITION
        assert colls1[0].is_fatal == colls2[0].is_fatal is False
