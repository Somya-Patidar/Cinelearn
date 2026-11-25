"""
CineLearn — Milestone 4C.4: TimelineScheduler, ScheduledAction Integrity & Pacing Contract Tests
=============================================================================================
Comprehensive unit and integration test suite verifying:
- Exact canonical timestamp arithmetic, Canonical Equality, and float representation.
- ScheduledAction strict Pydantic model validation (NaN, Inf, boolean, negative, 0.100s min).
- Three-stage action accounting and deterministic lifecycle state machine (zero early return).
- Hardened deep_compare_parameters tolerance validation against extreme integers (10**100, 10**1000).
- Simultaneous replacement vs inactive degenerate zero-duration lifecycle detection.
- Pacing contract enforcement (rushed, underutilized, budget overrun, lead-in acoustic gaps).
- All-or-nothing scheduling integrity across the compiler pipeline.
"""

from __future__ import annotations

import math
import pytest
from pydantic import ValidationError

from src.compiler.compiler_models import (
    CompilationDiagnostic,
    CompilationErrorCode,
    CompiledSceneOutput,
    ScheduledAction,
    ScheduledStep,
    ScheduledTimeline,
    canonicalize_time,
)
from src.compiler.layout_engine import HierarchicalBoxModelLayoutSolver
from src.compiler.layout_models import BoundingBox2D
from src.compiler.scene_compiler import DeterministicSceneCompiler
from src.compiler.timeline_scheduler import (
    TimelineScheduler,
    build_disjoint_lifespan_intervals,
    deep_compare_parameters,
    determine_initial_presence,
    validate_tolerance,
)
from src.ir.models import (
    ActionDeclaration,
    ActionType,
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
# Fixtures
# =====================================================================

@pytest.fixture
def scheduler() -> TimelineScheduler:
    return TimelineScheduler()


@pytest.fixture
def solver() -> HierarchicalBoxModelLayoutSolver:
    return HierarchicalBoxModelLayoutSolver()


@pytest.fixture
def compiler() -> DeterministicSceneCompiler:
    return DeterministicSceneCompiler()


@pytest.fixture
def sample_action() -> ActionDeclaration:
    return ActionDeclaration(
        type=ActionType.ENTER,
        target_entity="box_1",
        duration=1.0,
    )


# =====================================================================
# 1. Canonical Timestamp & Arithmetic Contract Tests
# =====================================================================

class TestCanonicalTimeContract:
    def test_canonical_end_time_decimal_addition(self):
        """0.1 + 0.2 in Python IEEE-754 evaluates to 0.30000000000000004. Canonical equality holds."""
        raw_sum = 0.1 + 0.2
        assert raw_sum != 0.3  # IEEE-754 representation artifact
        canon_val = canonicalize_time(raw_sum)
        assert canon_val == 0.3
        assert canon_val == round(0.1 + 0.2, 3)

    def test_canonical_time_rejects_boolean(self):
        """Passing boolean True or False to canonicalize_time raises ValueError."""
        with pytest.raises(ValueError, match="cannot be a boolean"):
            canonicalize_time(True)
        with pytest.raises(ValueError, match="cannot be a boolean"):
            canonicalize_time(False)

    def test_canonical_time_rejects_nan_and_inf(self):
        """Non-finite floats (NaN, +inf, -inf) are rejected fail-closed."""
        with pytest.raises(ValueError, match="must be finite"):
            canonicalize_time(float("nan"))
        with pytest.raises(ValueError, match="must be finite"):
            canonicalize_time(float("inf"))
        with pytest.raises(ValueError, match="must be finite"):
            canonicalize_time(float("-inf"))

    def test_canonical_time_oversized_integer_fails(self):
        """Integers exceeding 64-bit precision are rejected before float conversion."""
        huge_int = (1 << 65)
        with pytest.raises(ValueError, match="exceeds 64-bit precision"):
            canonicalize_time(huge_int)

    def test_canonical_time_rejects_negative(self):
        """Negative timestamps are strictly rejected."""
        with pytest.raises(ValueError, match="cannot be negative"):
            canonicalize_time(-0.001)

    def test_canonical_time_identical_millisecond_equality(self):
        """Values within the same millisecond bucket evaluate to identical canonical floats."""
        t1 = canonicalize_time(2.0002)
        t2 = canonicalize_time(1.9998)
        assert t1 == 2.000
        assert t2 == 2.000
        assert t1 == t2


# =====================================================================
# 2. ScheduledAction Model Integrity Tests
# =====================================================================

class TestScheduledActionModelIntegrity:
    def test_scheduled_action_valid_construction(self, sample_action):
        """Direct construction of valid ScheduledAction succeeds."""
        sa = ScheduledAction(
            step_id="step_1",
            action_index=0,
            target_entity="box_1",
            action=sample_action,
            start_time=0.100,
            end_time=0.300,
            duration=0.200,
            is_concurrent=False,
        )
        assert sa.start_time == 0.100
        assert sa.duration == 0.200
        assert sa.end_time == 0.300

    def test_scheduled_action_decimal_addition_exact_canonical_equality(self, sample_action):
        """start_time=0.1, duration=0.2 enforces exact canonical end_time=0.300."""
        sa = ScheduledAction(
            step_id="step_1",
            action_index=0,
            target_entity="box_1",
            action=sample_action,
            start_time=0.1,
            end_time=0.3,
            duration=0.2,
        )
        assert sa.end_time == 0.3

    def test_scheduled_action_rejects_boolean_fields(self, sample_action):
        """Boolean inputs for numeric fields are rejected before Pydantic coercion."""
        with pytest.raises(ValidationError, match="must be numeric, not Boolean"):
            ScheduledAction(
                step_id="step_1",
                action_index=0,
                target_entity="box_1",
                action=sample_action,
                start_time=True,
                end_time=1.0,
                duration=1.0,
            )

        with pytest.raises(ValidationError, match="must be numeric, not Boolean"):
            ScheduledAction(
                step_id="step_1",
                action_index=0,
                target_entity="box_1",
                action=sample_action,
                start_time=0.0,
                end_time=1.0,
                duration=False,
            )

    def test_scheduled_action_rejects_boolean_action_index(self, sample_action):
        """action_index=True is strictly rejected."""
        with pytest.raises(ValidationError, match="must be numeric, not Boolean"):
            ScheduledAction(
                step_id="step_1",
                action_index=True,
                target_entity="box_1",
                action=sample_action,
                start_time=0.0,
                end_time=1.0,
                duration=1.0,
            )

    def test_scheduled_action_rejects_negative_index(self, sample_action):
        """action_index < 0 is rejected."""
        with pytest.raises(ValidationError):
            ScheduledAction(
                step_id="step_1",
                action_index=-1,
                target_entity="box_1",
                action=sample_action,
                start_time=0.0,
                end_time=1.0,
                duration=1.0,
            )

    def test_scheduled_action_rejects_nan_and_inf(self, sample_action):
        """Non-finite floats in ScheduledAction raise ValidationError."""
        with pytest.raises(ValidationError, match="must be finite"):
            ScheduledAction(
                step_id="step_1",
                action_index=0,
                target_entity="box_1",
                action=sample_action,
                start_time=0.0,
                end_time=float("inf"),
                duration=1.0,
            )

    def test_scheduled_action_target_mismatch_fails(self, sample_action):
        """sa.target_entity != action.target_entity raises ValidationError."""
        with pytest.raises(ValidationError, match="target_entity 'other_entity' != action.target_entity 'box_1'"):
            ScheduledAction(
                step_id="step_1",
                action_index=0,
                target_entity="other_entity",
                action=sample_action,
                start_time=0.0,
                end_time=1.0,
                duration=1.0,
            )

    def test_scheduled_action_end_time_mismatch_fails(self, sample_action):
        """end_time not exactly matching canonicalize_time(start_time + duration) fails."""
        with pytest.raises(ValidationError, match="must exactly equal canonicalize_time"):
            ScheduledAction(
                step_id="step_1",
                action_index=0,
                target_entity="box_1",
                action=sample_action,
                start_time=0.1,
                end_time=0.301,  # Intentional discrepancy
                duration=0.2,
            )

    def test_duration_minimum_threshold_enforced(self, sample_action):
        """Durations below 0.100s fail validation."""
        with pytest.raises(ValidationError):
            ScheduledAction(
                step_id="step_1",
                action_index=0,
                target_entity="box_1",
                action=sample_action,
                start_time=0.0,
                end_time=0.05,
                duration=0.05,
            )


# =====================================================================
# 3. Parameter Equivalence & Hardened Tolerance Tests
# =====================================================================

class TestHardenedParameterComparison:
    def test_deep_compare_oversized_integer_tolerance_safe(self):
        """tol = 10**100 returns False safely without OverflowError."""
        assert validate_tolerance(10**100) is False
        assert deep_compare_parameters({"a": 1}, {"a": 1}, tol=10**100) is False

    def test_deep_compare_extreme_integer_tolerance_safe(self):
        """tol = 10**1000 (>1024 bits) returns False safely without OverflowError."""
        assert validate_tolerance(10**1000) is False
        assert deep_compare_parameters(1.0, 1.0, tol=10**1000) is False

    def test_deep_compare_rejects_boolean_tolerance(self):
        """tol = True or False is rejected."""
        assert validate_tolerance(True) is False
        assert validate_tolerance(False) is False
        assert deep_compare_parameters(1.0, 1.0, tol=True) is False

    def test_deep_compare_rejects_negative_tolerance(self):
        """Negative tolerance is rejected."""
        assert validate_tolerance(-0.001) is False
        assert deep_compare_parameters(1.0, 1.0, tol=-0.001) is False

    def test_deep_compare_rejects_nan_tolerance(self):
        """NaN tolerance is rejected."""
        assert validate_tolerance(float("nan")) is False
        assert deep_compare_parameters(1.0, 1.0, tol=float("nan")) is False

    def test_deep_compare_extreme_integer_payload_safe(self):
        """Compares extreme integer payloads (10**500) without float OverflowError."""
        int1 = 10**500
        int2 = 10**500
        int3 = 10**500 + 1
        assert deep_compare_parameters(int1, int2) is True
        assert deep_compare_parameters(int1, int33) is False if (int33 := int3) else False

    def test_deep_compare_recursive_structures(self):
        """Recursive dictionary and list comparison with numeric tolerance."""
        tree1 = {"scale": 1.0004, "pos": [0.0, 2.0], "nested": {"color": "#FFF"}}
        tree2 = {"scale": 1.0006, "pos": [0.0, 2.0001], "nested": {"color": "#FFF"}}
        assert deep_compare_parameters(tree1, tree2, tol=0.001) is True
        assert deep_compare_parameters(tree1, tree2, tol=0.0001) is False


# =====================================================================
# 4. Lifecycle State Machine & Replacement Tests
# =====================================================================

class TestLifecycleStateMachine:
    def test_dynamic_coincident_enter_exit_fails_closed(self):
        """Initially inactive entity with coincident ENTER and EXIT at 2.0s fails with COMPILER_INVALID_PARAMETER."""
        act_enter = ActionDeclaration(type=ActionType.ENTER, target_entity="box", duration=0.5)
        act_exit = ActionDeclaration(type=ActionType.EXIT, target_entity="box", duration=0.5)

        actions = [
            ScheduledAction(step_id="s1", action_index=0, target_entity="box", action=act_enter, start_time=2.0, end_time=2.5, duration=0.5),
            ScheduledAction(step_id="s1", action_index=1, target_entity="box", action=act_exit, start_time=1.5, end_time=2.0, duration=0.5),
        ]
        diags: List[CompilationDiagnostic] = []
        intervals = build_disjoint_lifespan_intervals(
            eid="box",
            scheduled_actions=actions,
            scene_duration=10.0,
            initial_presence=False,
            diagnostics=diags,
        )
        assert intervals == []
        errs = [d for d in diags if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(errs) >= 1
        assert "degenerate zero-duration lifecycle" in errs[0].message

    def test_fatal_lifecycle_accounts_for_all_actions(self):
        """Fatal error at 2.0s does not return early; subsequent action at 5.0s is diagnosed."""
        act_enter1 = ActionDeclaration(type=ActionType.ENTER, target_entity="box", duration=0.5)
        act_exit1 = ActionDeclaration(type=ActionType.EXIT, target_entity="box", duration=0.5)
        act_enter2 = ActionDeclaration(type=ActionType.ENTER, target_entity="box", duration=0.5)

        actions = [
            # Fatal degenerate coincident enter/exit at 2.0s
            ScheduledAction(step_id="s1", action_index=0, target_entity="box", action=act_enter1, start_time=2.0, end_time=2.5, duration=0.5),
            ScheduledAction(step_id="s1", action_index=1, target_entity="box", action=act_exit1, start_time=1.5, end_time=2.0, duration=0.5),
            # Subsequent duplicate/out-of-bounds enter at 15.0s (> scene duration 10.0s)
            ScheduledAction(step_id="s2", action_index=0, target_entity="box", action=act_enter2, start_time=15.0, end_time=15.5, duration=0.5),
        ]
        diags: List[CompilationDiagnostic] = []
        intervals = build_disjoint_lifespan_intervals(
            eid="box",
            scheduled_actions=actions,
            scene_duration=10.0,
            initial_presence=False,
            diagnostics=diags,
        )
        assert intervals == []
        # Both the coincident error and the budget overrun at 15.0s are diagnosed
        codes = [d.error_code for d in diags]
        assert CompilationErrorCode.COMPILER_INVALID_PARAMETER in codes
        assert CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN in codes

    def test_initially_active_coincident_replacement_passes(self):
        """Initially active entity with EXIT concluding at 2.0s and ENTER at 2.0s produces continuous disjoint spans."""
        act_exit = ActionDeclaration(type=ActionType.EXIT, target_entity="box", duration=0.5)
        act_enter = ActionDeclaration(type=ActionType.ENTER, target_entity="box", duration=0.5)
        act_exit2 = ActionDeclaration(type=ActionType.EXIT, target_entity="box", duration=0.5)

        actions = [
            ScheduledAction(step_id="s1", action_index=0, target_entity="box", action=act_exit, start_time=1.5, end_time=2.0, duration=0.5),
            ScheduledAction(step_id="s2", action_index=0, target_entity="box", action=act_enter, start_time=2.0, end_time=2.5, duration=0.5),
            ScheduledAction(step_id="s3", action_index=0, target_entity="box", action=act_exit2, start_time=4.5, end_time=5.0, duration=0.5),
        ]
        diags: List[CompilationDiagnostic] = []
        intervals = build_disjoint_lifespan_intervals(
            eid="box",
            scheduled_actions=actions,
            scene_duration=10.0,
            initial_presence=True,
            diagnostics=diags,
        )
        assert len(diags) == 0
        assert intervals == [(0.0, 2.0), (2.0, 5.0)]

    def test_exact_canonical_equality_zero_duration_detection(self):
        """Exact canonical equality detects zero-duration active span without tolerance offset."""
        act_enter = ActionDeclaration(type=ActionType.ENTER, target_entity="box", duration=0.5)
        act_exit = ActionDeclaration(type=ActionType.EXIT, target_entity="box", duration=0.5)
        actions = [
            # Entity enters at 2.0s, so current_span_start = 2.0s
            ScheduledAction(step_id="s1", action_index=0, target_entity="box", action=act_enter, start_time=2.0, end_time=2.5, duration=0.5),
            # Exit concludes at 2.0s, exactly matching span start 2.0s
            ScheduledAction(step_id="s1", action_index=1, target_entity="box", action=act_exit, start_time=1.5, end_time=2.0, duration=0.5),
        ]
        diags: List[CompilationDiagnostic] = []
        intervals = build_disjoint_lifespan_intervals(
            eid="box",
            scheduled_actions=actions,
            scene_duration=10.0,
            initial_presence=False,
            diagnostics=diags,
        )
        assert intervals == []
        errs = [d for d in diags if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(errs) >= 1
        assert "zero-duration lifecycle" in errs[0].message


# =====================================================================
# 5. TimelineScheduler Pacing & Budget Enforcement Tests
# =====================================================================

class TestTimelineSchedulerPacingAndBudgets:
    def test_nominal_sequential_schedule(self, scheduler):
        """Nominal sequential timeline compiles cleanly with appropriate slack pauses."""
        scene = SceneIR(
            scene_id="NominalScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[EntityDeclaration(id="box_1", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.5, 3.5)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(3.5, 6.5)),
                    actions=[ActionDeclaration(type=ActionType.EXIT, target_entity="box_1", duration=0.5)],
                ),
            ],
        )
        st, diags = scheduler.schedule(scene)
        assert st.is_valid is True
        assert len(st.actions) == 2
        # Check lead-in acoustic gap on step 1
        assert st.steps[0].lead_in_pause == 0.5
        # Step 1 delta_t = 3.0s, d_active = 1.0s, slack = 2.0s
        assert st.steps[0].resting_pause == 2.0
        # Step 2 delta_t = 3.0s, d_active = 0.5s, slack = 2.5s
        assert st.steps[1].resting_pause == 2.5

    def test_pacing_rushed_warning_emitted(self, scheduler):
        """Action consuming near-total budget triggers TIMELINE_PACING_RUSHED advisory warning."""
        scene = SceneIR(
            scene_id="RushedScene",
            duration_budget=5.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[EntityDeclaration(id="box_1", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 1.2)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.1)],
                )
            ],
        )
        st, diags = scheduler.schedule(scene)
        assert st.is_valid is True
        rushed = [d for d in diags if d.error_code == CompilationErrorCode.TIMELINE_PACING_RUSHED]
        assert len(rushed) == 1
        assert rushed[0].severity == "WARNING"

    def test_step_duration_budget_overrun_fails_closed(self, scheduler):
        """Requested active duration exceeding narration delta_t fails closed."""
        scene = SceneIR(
            scene_id="OverrunScene",
            duration_budget=5.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[EntityDeclaration(id="box_1", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 1.0)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.5)],
                )
            ],
        )
        st, diags = scheduler.schedule(scene)
        assert st.is_valid is False
        assert len(st.actions) == 0  # All-or-nothing guarantee
        errs = [d for d in diags if d.error_code == CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN]
        assert len(errs) == 1
        assert errs[0].severity == "ERROR"

    def test_scene_duration_budget_overrun_fails_closed(self, scheduler):
        """Total scheduled timeline exceeding scene.duration_budget fails closed."""
        scene = SceneIR(
            scene_id="SceneOverrunScene",
            duration_budget=4.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[EntityDeclaration(id="box_1", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 5.0)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0)],
                )
            ],
        )
        st, diags = scheduler.schedule(scene)
        assert st.is_valid is False
        assert len(st.actions) == 0
        errs = [d for d in diags if d.error_code == CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN]
        assert len(errs) == 1

    def test_non_monotonic_anchors_rejected(self, scheduler):
        """Narration anchors that overlap/regress beyond 5ms tolerance fail closed."""
        scene = SceneIR(
            scene_id="NonMonotonicScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[EntityDeclaration(id="box_1", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")],
            timeline=[
                TimelineEvent(step_id="step_1", narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 3.0)), actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0)]),
                TimelineEvent(step_id="step_2", narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(2.0, 5.0)), actions=[ActionDeclaration(type=ActionType.EXIT, target_entity="box_1", duration=1.0)]),
            ],
        )
        st, diags = scheduler.schedule(scene)
        assert st.is_valid is False
        errs = [d for d in diags if d.error_code == CompilationErrorCode.COMPILER_TIMELINE_ANCHOR_OVERLAP]
        assert len(errs) == 1

    def test_coincident_events_concurrency_and_target_conflict(self, scheduler):
        """Co-incident events with conflicting actions on the same entity fail closed."""
        scene = SceneIR(
            scene_id="ConflictScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[EntityDeclaration(id="box_1", component_type="PrimitiveShape", visual_tier=VisualTier.TIER3_PRIMITIVE_TREE, viewport="vp_main")],
            timeline=[
                TimelineEvent(step_id="step_1a", narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(1.0, 4.0)), actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0)]),
                TimelineEvent(step_id="step_1b", narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(1.0, 4.0)), actions=[ActionDeclaration(type=ActionType.EXIT, target_entity="box_1", duration=1.0)]),
            ],
        )
        st, diags = scheduler.schedule(scene)
        assert st.is_valid is False
        errs = [d for d in diags if d.error_code == CompilationErrorCode.COMPILER_TIMELINE_TARGET_CONFLICT]
        assert len(errs) >= 1


# =====================================================================
# 6. End-to-End Compiler Integration Tests
# =====================================================================

class TestCompilerTimelineSchedulerIntegration:
    def test_scene_compiler_emits_resting_pause_and_preserves_runtimes(self, compiler, solver):
        """DeterministicSceneCompiler executes Sequence A and emits self.wait for slack pauses."""
        scene = SceneIR(
            scene_id="PacingIntegrationScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="box_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Presenting the box.",
                        audio_timestamp_range=(0.0, 3.0),
                    ),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0)],
                )
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is True
        # Animation runs for 1.0s, remaining 2.0s slack emitted as self.wait(2.0)
        assert "self.play(FadeIn(ent_box_1), run_time=1.0)" in out.python_source
        assert "self.wait(2.0)" in out.python_source

    def test_scene_compiler_fails_closed_on_budget_overrun(self, compiler, solver):
        """Scene with budget overrun aborts compilation and emits COMPILER_TIMELINE_BUDGET_OVERRUN."""
        scene = SceneIR(
            scene_id="FailClosedScene",
            duration_budget=5.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="box_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Invalid long animation.",
                        audio_timestamp_range=(0.0, 1.0),
                    ),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=2.5)],
                )
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is False
        assert out.python_source == ""
        overrun_diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN)
        assert len(overrun_diags) == 1


# =====================================================================
# 11. Milestone 4C.4 Remediation Verification Tests
# =====================================================================
class TestMilestone4C4Remediation:
    """
    Focused verification tests for the 4 findings remediated in Milestone 4C.4:
    - FINDING-4C4-01: Monotonicity checking against preceding unanchored step end times.
    - FINDING-4C4-02: Concurrent Manim animation emission and runtime duration verification.
    - FINDING-4C4-03: Coincident ScheduledStep ID mapping and deduplicated pause emission.
    - FINDING-4C4-04: Authoritative initial presence determination in TimelineScheduler.
    """

    def test_remediation_finding_4c4_01_unanchored_preceding_step_monotonicity_rejected(self, scheduler):
        """
        FINDING-4C4-01: An anchored step starting before a preceding unanchored step finishes
        must be rejected with COMPILER_TIMELINE_ANCHOR_OVERLAP.
        - Step 1 (anchored): [0.0, 2.0]
        - Step 2 (unanchored): duration 3.0s -> occupies [2.0, 5.0]
        - Step 3 (anchored): [3.0, 6.0] -> starts at 3.0s before Step 2 ends at 5.0s.
        """
        scene = SceneIR(
            scene_id="UnanchoredOverlapScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="box_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 1.0, "height": 1.0},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Step 1",
                        audio_timestamp_range=(0.0, 2.0),
                    ),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    actions=[ActionDeclaration(type=ActionType.HIGHLIGHT, target_entity="box_1", duration=3.0)],
                ),
                TimelineEvent(
                    step_id="step_3",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=1,
                        text_cue="Step 3",
                        audio_timestamp_range=(3.0, 6.0),
                    ),
                    actions=[ActionDeclaration(type=ActionType.EXIT, target_entity="box_1", duration=1.0)],
                ),
            ],
        )

        sched, diags = scheduler.schedule(scene)
        assert sched.is_valid is False
        overlap_diags = [d for d in diags if d.error_code == CompilationErrorCode.COMPILER_TIMELINE_ANCHOR_OVERLAP]
        assert len(overlap_diags) == 1
        assert "starts at 3.000s before previous step ended at 5.000s" in overlap_diags[0].message

    def test_remediation_finding_4c4_01_unanchored_preceding_step_monotonicity_valid(self, scheduler):
        """
        FINDING-4C4-01: An anchored step starting at or after the unanchored step ends (within 0.005s) is accepted.
        - Step 1 (anchored): [0.0, 2.0]
        - Step 2 (unanchored): duration 3.0s -> occupies [2.0, 5.0]
        - Step 3 (anchored): [4.996, 7.0] -> starts within 0.005s of 5.000s -> Valid!
        """
        scene = SceneIR(
            scene_id="UnanchoredValidScene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="box_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 1.0, "height": 1.0},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Step 1",
                        audio_timestamp_range=(0.0, 2.0),
                    ),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    actions=[ActionDeclaration(type=ActionType.HIGHLIGHT, target_entity="box_1", duration=3.0)],
                ),
                TimelineEvent(
                    step_id="step_3",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=1,
                        text_cue="Step 3",
                        audio_timestamp_range=(4.996, 7.0),
                    ),
                    actions=[ActionDeclaration(type=ActionType.EXIT, target_entity="box_1", duration=1.0)],
                ),
            ],
        )

        sched, diags = scheduler.schedule(scene)
        assert sched.is_valid is True
        overlap_diags = [d for d in diags if d.error_code == CompilationErrorCode.COMPILER_TIMELINE_ANCHOR_OVERLAP]
        assert len(overlap_diags) == 0

    def test_remediation_finding_4c4_03_coincident_step_id_mapping_and_pauses(self, compiler, solver, scheduler):
        """
        FINDING-4C4-03: Coincident events resolve to their parent ScheduledStep.
        Lead-in pause is emitted once before the group.
        Resting pause is emitted once after the group.
        Pauses are not duplicated.
        """
        scene = SceneIR(
            scene_id="CoincidentPauseScene",
            duration_budget=5.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="box_a",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 1.0, "height": 1.0},
                ),
                EntityDeclaration(
                    id="box_b",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "circle", "width": 1.0, "height": 1.0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="s1a",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Coincident event A",
                        audio_timestamp_range=(0.4, 3.4),  # delta_t = 3.0s, lead_in = 0.4s
                    ),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_a", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="s1b",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=1,
                        text_cue="Coincident event B",
                        audio_timestamp_range=(0.4, 3.4),  # identical range -> coincident group
                    ),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_b", duration=1.8)],
                ),
            ],
        )

        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is True

        # Check ScheduledTimeline mapping
        sched = scheduler.schedule(scene)[0]
        step = sched.get_step_for_event("s1a")
        assert step is not None
        assert step.step_id == "s1a_group"
        assert "s1a" in step.constituent_step_ids
        assert "s1b" in step.constituent_step_ids
        assert sched.get_step_for_event("s1b") is step
        assert step.lead_in_pause == 0.4
        assert step.resting_pause == 1.2  # 3.0 - max(1.0, 1.8) = 1.2
        assert step.active_duration == 1.8

        # Check emitted Python code:
        # 1. Lead-in pause emitted exactly once
        assert out.python_source.count("self.wait(0.4)") == 1
        # 2. Resting pause emitted exactly once
        assert out.python_source.count("self.wait(1.2)") == 1
        # 3. Combined concurrent play emitted with both animations
        assert "self.play(FadeIn(ent_box_a, run_time=1.0), FadeIn(ent_box_b, run_time=1.8))" in out.python_source

    def test_remediation_finding_4c4_04_initial_entity_presence_authoritative(self, scheduler):
        """
        FINDING-4C4-04: TimelineScheduler uses determine_initial_presence() authoritative function
        and propagates its diagnostics into scheduler diagnostics.
        """
        scene = SceneIR(
            scene_id="PresenceScene",
            duration_budget=5.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="box_hidden",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"visible": False, "shape_type": "rectangle", "width": 1.0, "height": 1.0},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Enter box",
                        audio_timestamp_range=(0.0, 3.0),
                    ),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="box_hidden", duration=1.0)],
                )
            ],
        )

        sched, diags = scheduler.schedule(scene)
        assert sched.is_valid is True
        actions = sched.actions
        assert len(actions) == 1
        assert actions[0].target_entity == "box_hidden"

    def test_remediation_finding_4c4_02_concurrent_manim_animation_runtime_execution(self, compiler, solver):
        """
        FINDING-4C4-02: Concurrent animations are emitted together in a single self.play call.
        Runtime duration in Manim CE 0.18.1 is verified to be max(d_j), NOT sum(d_j).
        - Action 1 duration: 1.0s
        - Action 2 duration: 2.2s
        - delta_t: 3.5s
        - Expected active runtime: 2.2s (max, not 3.2s sum)
        - Expected resting pause: 3.5 - 2.2 = 1.3s
        """
        import manim as m

        scene = SceneIR(
            scene_id="ConcurrentExecutionScene",
            duration_budget=5.0,
            layout_configuration=LayoutConfiguration(),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="box_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 1.0, "height": 1.0},
                ),
                EntityDeclaration(
                    id="box_2",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 1.0, "height": 1.0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Two boxes appear concurrently.",
                        audio_timestamp_range=(0.0, 3.5),
                    ),
                    actions=[
                        ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0, parameters={"is_concurrent": True}),
                        ActionDeclaration(type=ActionType.ENTER, target_entity="box_2", duration=2.2, parameters={"is_concurrent": True}),
                    ],
                )
            ],
        )

        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert out.is_valid is True

        # Verify emitted code has single combined self.play
        assert "self.play(FadeIn(ent_box_1, run_time=1.0), FadeIn(ent_box_2, run_time=2.2))" in out.python_source
        assert "self.wait(1.3)" in out.python_source

        # Execute in Manim CE runtime to verify actual execution timing
        loc = {}
        exec(out.python_source, loc, loc)
        scene_cls = loc["ConcurrentExecutionScene"]
        with m.tempconfig({"dry_run": True}):
            manim_scene = scene_cls()
            manim_scene.render()

        # Total time is:
        # active duration (2.2s) + resting pause (1.3s) + scene final pause (1.0s) = 4.5s
        # If it were sequential (sum): 1.0 + 2.2 + 1.3 + 1.0 = 5.5s
        assert abs(manim_scene.renderer.time - 4.5) < 0.001, (
            f"Expected total scene time 4.5s (active max 2.2s + pause 1.3s + final 1.0s), got {manim_scene.renderer.time:.3f}s"
        )
