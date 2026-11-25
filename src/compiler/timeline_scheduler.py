"""
CineLearn — Milestone 4C.4: TimelineScheduler & Pacing Contract
==============================================================
Deterministic timeline scheduler, canonical timestamp management,
lifecycle state-machine reconstruction, and pacing contract enforcement.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from src.compiler.compiler_models import (
    CompilationDiagnostic,
    CompilationErrorCode,
    ScheduledAction,
    ScheduledStep,
    ScheduledTimeline,
    canonicalize_time,
)
from src.ir.models import (
    ActionDeclaration,
    ActionType,
    EntityDeclaration,
    SceneIR,
    TimelineEvent,
)


def validate_tolerance(tol: Any) -> bool:
    """
    Validate that tol is a valid, finite, non-negative numerical tolerance.
    Guarantees zero OverflowError on arbitrarily large integers.
    """
    # 1. Strict Boolean Guard: In Python, isinstance(True, int) is True!
    if isinstance(tol, bool):
        return False

    # 2. Strict Numeric Type Guard
    if not isinstance(tol, (int, float)):
        return False

    # 3. Integer Tolerance Guard: Check bit_length BEFORE float conversion or math calls!
    if isinstance(tol, int):
        if tol.bit_length() > 64:
            return False
        if tol < 0:
            return False
        return True

    # 4. Float Tolerance Guard: Safe because tol is already float
    if isinstance(tol, float):
        if math.isnan(tol) or math.isinf(tol) or tol < 0.0:
            return False
        return True

    return False


def deep_compare_parameters(p1: Any, p2: Any, tol: float = 0.001) -> bool:
    """
    Recursively compares action parameter trees with strict type separation,
    numerical floating-point tolerance, and fail-closed NaN/Inf rejection.
    Safely handles arbitrarily large integers without OverflowError.
    """
    # 1. Validate tolerance parameter
    if not validate_tolerance(tol):
        return False

    # 2. Identity / None match
    if p1 is None or p2 is None:
        return p1 is None and p2 is None

    # 3. Strict Boolean Guard: Must be checked before int/float!
    # In Python, isinstance(True, int) is True; this prevents True == 1.0
    if isinstance(p1, bool) or isinstance(p2, bool):
        if not (isinstance(p1, bool) and isinstance(p2, bool)):
            return False
        return p1 == p2

    # 4. Integer vs Integer: Exact comparison without float conversion!
    # Protects against OverflowError on arbitrarily large Python integers
    if isinstance(p1, int) and isinstance(p2, int):
        return p1 == p2

    # 5. Mixed Float / Integer or Float vs Float Numerical Comparison
    if isinstance(p1, (int, float)) and isinstance(p2, (int, float)):
        if isinstance(p1, int) and p1.bit_length() > 1024:
            return False
        if isinstance(p2, int) and p2.bit_length() > 1024:
            return False

        try:
            f1 = float(p1)
            f2 = float(p2)
        except OverflowError:
            return False

        # Fail closed on non-finite values
        if math.isnan(f1) or math.isnan(f2) or math.isinf(f1) or math.isinf(f2):
            return False

        return abs(f1 - f2) <= tol

    # 6. Strict Type Equality for Non-Numeric Types
    if type(p1) != type(p2):
        return False

    # 7. String Identity
    if isinstance(p1, str):
        return p1 == p2

    # 8. Dictionary Comparison
    if isinstance(p1, dict):
        if set(p1.keys()) != set(p2.keys()):
            return False
        return all(deep_compare_parameters(p1[k], p2[k], tol) for k in p1)

    # 9. Sequence Comparison (list, tuple)
    if isinstance(p1, (list, tuple)):
        if len(p1) != len(p2):
            return False
        return all(deep_compare_parameters(a, b, tol) for a, b in zip(p1, p2))

    # 10. Fail-Closed Fallback for Unsupported Types
    return False


def determine_initial_presence(
    entity: EntityDeclaration,
    scene: SceneIR,
    scheduled_actions: Optional[List[ScheduledAction]] = None,
) -> Tuple[bool, List[CompilationDiagnostic]]:
    """
    Authoritatively determines an entity's initial presence (visible or off-screen at t=0)
    by reconciling entity styling, initial_state metadata, inherited status, and timeline actions.
    Covers all 8 canonical conflict scenarios with deterministic diagnostics.
    """
    diags: List[CompilationDiagnostic] = []
    eid = entity.id

    # 1. Extract metadata from styling and initial_state
    styling_vis = getattr(entity.styling, "visible", None) if entity.styling else None
    state_vis = None
    state_visibility = None
    if entity.initial_state and isinstance(entity.initial_state, dict):
        state_vis = entity.initial_state.get("visible")
        state_visibility = entity.initial_state.get("visibility")

    # Case 7 & 8: Check conflicts or redundancy between visible and visibility
    if state_vis is not None and state_visibility is not None:
        if state_vis == state_visibility:
            # Case 7: Redundant metadata declaration
            diags.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_REDUNDANT_DECLARATION,
                    message=f"Entity '{eid}' declares both 'visible' and 'visibility' with identical value {state_vis}.",
                    entity_id=eid,
                    severity="WARNING",
                )
            )
        else:
            # Case 8: Irreconcilable conflict
            diags.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                    message=f"Entity '{eid}' declares conflicting 'visible' ({state_vis}) and 'visibility' ({state_visibility}).",
                    entity_id=eid,
                    severity="ERROR",
                )
            )
            return False, diags

    declared_vis = state_vis if state_vis is not None else (
        state_visibility if state_visibility is not None else styling_vis
    )

    # Case 6: Inherited entity checks
    is_inherited = any(inh.entity_id == eid for inh in getattr(scene, "inherited_entities", []))
    if is_inherited:
        if declared_vis is False:
            diags.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                    message=f"Inherited entity '{eid}' cannot declare visibility=False across scene boundary.",
                    entity_id=eid,
                    severity="ERROR",
                )
            )
            return False, diags
        return True, diags

    # Default initial presence when no explicit visibility metadata is provided
    if declared_vis is not None:
        initial_presence = bool(declared_vis)
    else:
        # Check if entity has an explicit ENTER action in scheduled_actions or timeline
        has_enter = False
        if scheduled_actions is not None:
            has_enter = any(sa.target_entity == eid and sa.action.type == ActionType.ENTER for sa in scheduled_actions)
        else:
            for event in getattr(scene, "timeline", []):
                for action in getattr(event, "actions", []):
                    if getattr(action, "target_entity", None) == eid and getattr(action, "type", None) == ActionType.ENTER:
                        has_enter = True
                        break
                if has_enter:
                    break
        initial_presence = not has_enter

    # Inspect timeline actions targeting this entity
    entity_enters: List[Tuple[int, ActionDeclaration]] = []
    entity_exits: List[Tuple[int, ActionDeclaration]] = []
    if scheduled_actions is not None:
        for a_idx, sa in enumerate(scheduled_actions):
            if sa.target_entity == eid:
                if sa.action.type == ActionType.ENTER:
                    entity_enters.append((a_idx, sa.action))
                elif sa.action.type == ActionType.EXIT:
                    entity_exits.append((a_idx, sa.action))
    else:
        for step_idx, event in enumerate(getattr(scene, "timeline", [])):
            for a_idx, action in enumerate(getattr(event, "actions", [])):
                if getattr(action, "target_entity", None) == eid:
                    if getattr(action, "type", None) == ActionType.ENTER:
                        entity_enters.append((step_idx, action))
                    elif getattr(action, "type", None) == ActionType.EXIT:
                        entity_exits.append((step_idx, action))

    # Case 1 & 3: Check visible=True interactions with ENTER actions
    if initial_presence is True and entity_enters:
        first_enter_step, _ = entity_enters[0]
        if not entity_exits or first_enter_step <= entity_exits[0][0]:
            if first_enter_step == 0:
                # Case 1: Redundant ENTER at start of scene
                diags.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.TIMELINE_LIFECYCLE_REDUNDANT_ENTER,
                        message=f"Entity '{eid}' is initially active (visible=True); ENTER action at scene start is redundant.",
                        entity_id=eid,
                        severity="WARNING",
                    )
                )
            else:
                # Case 3: Action conflict - cannot enter an already visible entity at t > 0 without prior EXIT
                diags.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_TIMELINE_TARGET_CONFLICT,
                        message=f"Entity '{eid}' is initially active (visible=True) and cannot declare ENTER at step {first_enter_step} without prior EXIT.",
                        entity_id=eid,
                        severity="ERROR",
                    )
                )

    # Case 5: Check visible=False interactions with EXIT actions
    if initial_presence is False and entity_exits:
        first_exit_step, _ = entity_exits[0]
        if not entity_enters or first_exit_step < entity_enters[0][0]:
            diags.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                    message=f"Entity '{eid}' is off-screen (visible=False) and cannot declare EXIT at step {first_exit_step} before entering.",
                    entity_id=eid,
                    severity="ERROR",
                )
            )

    return initial_presence, diags


def build_disjoint_lifespan_intervals(
    eid: str,
    scheduled_actions: List[ScheduledAction],
    scene_duration: float,
    initial_presence: bool,
    diagnostics: List[CompilationDiagnostic],
) -> List[Tuple[float, float]]:
    """
    Deterministic ACTIVE/INACTIVE state machine with complete action accounting (zero early return).
    Uses exact canonical float equality (NO downstream round() and NO downstream tolerance).

    Guarantees:
    - 100% of visibility actions targeting the entity are evaluated.
    - Active replacement vs degenerate zero-duration lifespans are cleanly resolved.
    - Zero partial invalid intervals escape: returns [] on any fatal error.
    - All returned intervals strictly satisfy: 0.0 <= start < end <= scene_duration.
    """
    target_actions = [
        sa for sa in scheduled_actions
        if sa.target_entity == eid and sa.action.type in (ActionType.ENTER, ActionType.EXIT)
    ]

    timestamp_map: Dict[float, Dict[str, List[ScheduledAction]]] = {}
    for sa in target_actions:
        t = sa.start_time if sa.action.type == ActionType.ENTER else sa.end_time
        if t not in timestamp_map:
            timestamp_map[t] = {"ENTER": [], "EXIT": []}
        act_key = "ENTER" if sa.action.type == ActionType.ENTER else "EXIT"
        timestamp_map[t][act_key].append(sa)

    sorted_timestamps = sorted(timestamp_map.keys())

    state = "ACTIVE" if initial_presence else "INACTIVE"
    current_span_start: Optional[float] = 0.0 if initial_presence else None
    intervals: List[Tuple[float, float]] = []
    is_lifecycle_valid = True

    for t in sorted_timestamps:
        enters = timestamp_map[t]["ENTER"]
        exits = timestamp_map[t]["EXIT"]

        # -------------------------------------------------------------
        # STATE: INACTIVE
        # -------------------------------------------------------------
        if state == "INACTIVE":
            # Coincident ENTER & EXIT while INACTIVE: Fatal degenerate zero-duration lifecycle
            if enters and exits:
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                        message=f"Entity '{eid}' has degenerate zero-duration lifecycle: ENTER and EXIT both scheduled at {t:.3f}s while inactive.",
                        entity_id=eid,
                        severity="ERROR",
                    )
                )
                is_lifecycle_valid = False

                # Exhaustive Duplicate Accounting
                if len(enters) > 1:
                    for _ in enters[1:]:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.TIMELINE_LIFECYCLE_DUPLICATE_ENTER,
                                message=f"Entity '{eid}' declared duplicate ENTER at {t:.3f}s during degenerate conflict.",
                                entity_id=eid,
                                severity="WARNING",
                            )
                        )
                if len(exits) > 1:
                    for _ in exits[1:]:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.TIMELINE_LIFECYCLE_DUPLICATE_EXIT,
                                message=f"Entity '{eid}' declared duplicate EXIT at {t:.3f}s during degenerate conflict.",
                                entity_id=eid,
                                severity="WARNING",
                            )
                        )
                # Continue loop without returning early

            elif exits:
                for _ in exits:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.TIMELINE_LIFECYCLE_UNMATCHED_EXIT,
                            message=f"Entity '{eid}' declared EXIT concluding at {t:.3f}s while inactive.",
                            entity_id=eid,
                            severity="WARNING",
                        )
                    )

            elif enters:
                if t > scene_duration:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN,
                            message=f"Entity '{eid}' has ENTER at {t:.3f}s exceeding scene duration {scene_duration:.3f}s.",
                            entity_id=eid,
                            severity="ERROR",
                        )
                    )
                    is_lifecycle_valid = False
                else:
                    state = "ACTIVE"
                    current_span_start = t

                if len(enters) > 1:
                    for _ in enters[1:]:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.TIMELINE_LIFECYCLE_DUPLICATE_ENTER,
                                message=f"Entity '{eid}' declared duplicate ENTER at {t:.3f}s.",
                                entity_id=eid,
                                severity="WARNING",
                            )
                        )

        # -------------------------------------------------------------
        # STATE: ACTIVE
        # -------------------------------------------------------------
        else:
            if exits:
                # Exact Canonical Equality: zero-duration lifecycle detection
                if current_span_start is not None and t == current_span_start:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Entity '{eid}' exits at {t:.3f}s matching span start; zero-duration lifecycle.",
                            entity_id=eid,
                            severity="ERROR",
                        )
                    )
                    is_lifecycle_valid = False

                span_end = min(t, scene_duration)
                if current_span_start is not None and current_span_start < span_end:
                    intervals.append((current_span_start, span_end))

                if len(exits) > 1:
                    for _ in exits[1:]:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.TIMELINE_LIFECYCLE_DUPLICATE_EXIT,
                                message=f"Entity '{eid}' declared duplicate EXIT concluding at {t:.3f}s.",
                                entity_id=eid,
                                severity="WARNING",
                            )
                        )

                # Active Replacement: ENTER at same timestamp t while active
                if enters:
                    state = "ACTIVE"
                    current_span_start = t
                    if len(enters) > 1:
                        for _ in enters[1:]:
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.TIMELINE_LIFECYCLE_DUPLICATE_ENTER,
                                    message=f"Entity '{eid}' declared duplicate ENTER at {t:.3f}s during replacement.",
                                    entity_id=eid,
                                    severity="WARNING",
                                )
                            )
                else:
                    state = "INACTIVE"
                    current_span_start = None

            elif enters:
                for _ in enters:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.TIMELINE_LIFECYCLE_REDUNDANT_ENTER,
                            message=f"Entity '{eid}' declared redundant ENTER at {t:.3f}s while already active.",
                            entity_id=eid,
                            severity="WARNING",
                        )
                    )

    # Scene Termination Handling
    if state == "ACTIVE" and current_span_start is not None:
        if current_span_start < scene_duration:
            intervals.append((current_span_start, scene_duration))
        elif current_span_start == scene_duration:
            diagnostics.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                    message=f"Entity '{eid}' entered at exact scene duration {scene_duration:.3f}s with zero duration.",
                    entity_id=eid,
                    severity="ERROR",
                )
            )
            is_lifecycle_valid = False

    # All-or-Nothing Output: Never return partial intervals if lifecycle had fatal errors
    if not is_lifecycle_valid:
        return []

    # Final Strict Invariant Verification
    for start, end in intervals:
        assert 0.0 <= start < end <= scene_duration, f"Strict invariant 0 <= start < end <= T violated: ({start}, {end})"
    for i in range(len(intervals) - 1):
        assert intervals[i][1] <= intervals[i + 1][0], f"Disjoint invariant prev_end <= next_start violated: {intervals[i]} and {intervals[i + 1]}"

    return intervals


class TimelineScheduler:
    """
    Sole authoritative scheduling engine for CineLearn scenes.
    Transforms SceneIR actions and narration anchors into canonical ScheduledTimeline,
    enforces duration budgets, pacing contracts, and all-or-nothing scheduling integrity.
    """

    def __init__(self, target_pause_normal: float = 0.60, target_pause_short: float = 0.20):
        self.target_pause_normal = target_pause_normal
        self.target_pause_short = target_pause_short

    def schedule(
        self,
        scene: SceneIR,
        auto_retime: bool = False,
        inter_cue_pause: float = 0.50,
        scene_end_pause: float = 1.00,
    ) -> Tuple[ScheduledTimeline, List[CompilationDiagnostic]]:
        """
        Schedules a validated SceneIR into a ScheduledTimeline.
        Returns (ScheduledTimeline, diagnostics).
        If any fatal error occurs, returns ScheduledTimeline(is_valid=False, actions=[]).
        When auto_retime is True, delegates to audio-authoritative retiming.
        """
        if auto_retime:
            retimed_scene, timeline, diagnostics = self.retime_scene_timeline(
                scene=scene,
                inter_cue_pause=inter_cue_pause,
                scene_end_pause=scene_end_pause,
            )
            return timeline, diagnostics

        diagnostics: List[CompilationDiagnostic] = []
        duration_budget = float(scene.duration_budget or 30.0)
        is_schedule_valid = True

        # Check for duplicate entity IDs
        seen_entities: Set[str] = set()
        for idx, ent in enumerate(scene.entities):
            if ent.id in seen_entities:
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                        message=f"Duplicate entity id '{ent.id}' declared in scene.",
                        entity_id=ent.id,
                        field_path=f"entities[{idx}].id",
                        severity="ERROR",
                    )
                )
                is_schedule_valid = False
            seen_entities.add(ent.id)

        # Check initial presence and metadata consistency for all entities
        for ent in scene.entities:
            _, init_diags = determine_initial_presence(ent, scene)
            diagnostics.extend(init_diags)
            if any(d.severity == "ERROR" for d in init_diags):
                is_schedule_valid = False

        if not scene.timeline:
            return ScheduledTimeline(
                scene_id=scene.scene_id,
                total_duration=0.0,
                is_valid=is_schedule_valid,
                steps=[],
                actions=[],
                diagnostics=diagnostics,
            ), diagnostics

        # Group co-incident events sharing identical audio anchor intervals
        # An event without anchor is treated as standalone
        raw_events: List[TimelineEvent] = list(scene.timeline)
        event_groups: List[List[Tuple[int, TimelineEvent]]] = []
        i = 0
        while i < len(raw_events):
            ev = raw_events[i]
            anchor = ev.narration_anchor
            if anchor and anchor.audio_timestamp_range:
                t0_anchor, t1_anchor = anchor.audio_timestamp_range
                group = [(i, ev)]
                j = i + 1
                while j < len(raw_events):
                    next_ev = raw_events[j]
                    next_anchor = next_ev.narration_anchor
                    if next_anchor and next_anchor.audio_timestamp_range:
                        nt0, nt1 = next_anchor.audio_timestamp_range
                        if nt0 == t0_anchor and nt1 == t1_anchor:
                            group.append((j, next_ev))
                            j += 1
                            continue
                    break
                event_groups.append(group)
                i = j
            else:
                event_groups.append([(i, ev)])
                i += 1

        all_scheduled_actions: List[ScheduledAction] = []
        scheduled_steps: List[ScheduledStep] = []
        prev_step_end: Optional[float] = None
        prev_anchor_t0: Optional[float] = None
        prev_anchor_t1: Optional[float] = None
        unanchored_clock: float = 0.0

        for group_idx, group in enumerate(event_groups):
            is_group_coincident = len(group) > 1
            first_idx, first_ev = group[0]
            step_id = first_ev.step_id if not is_group_coincident else f"{first_ev.step_id}_group"

            # ---------------------------------------------------------
            # 1. Narration Anchor Validation & Monotonicity
            # ---------------------------------------------------------
            has_anchor = False
            t0: float = 0.0
            t1: float = 0.0

            if first_ev.narration_anchor and first_ev.narration_anchor.audio_timestamp_range:
                raw_t0, raw_t1 = first_ev.narration_anchor.audio_timestamp_range
                has_anchor = True

                # Strict type guards
                if isinstance(raw_t0, bool) or isinstance(raw_t1, bool) or not isinstance(raw_t0, (int, float)) or not isinstance(raw_t1, (int, float)):
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Step '{first_ev.step_id}' audio timestamp range must be numeric, got ({type(raw_t0).__name__}, {type(raw_t1).__name__}).",
                            field_path=f"timeline[{first_idx}].narration_anchor.audio_timestamp_range",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                elif math.isnan(raw_t0) or math.isnan(raw_t1) or math.isinf(raw_t0) or math.isinf(raw_t1):
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Step '{first_ev.step_id}' audio timestamp range must be finite, got ({raw_t0}, {raw_t1}).",
                            field_path=f"timeline[{first_idx}].narration_anchor.audio_timestamp_range",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                elif raw_t0 < 0.0 or raw_t1 < 0.0:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Step '{first_ev.step_id}' audio timestamp range cannot be negative: ({raw_t0}, {raw_t1}).",
                            field_path=f"timeline[{first_idx}].narration_anchor.audio_timestamp_range",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                else:
                    t0 = canonicalize_time(raw_t0)
                    t1 = canonicalize_time(raw_t1)

                    if t1 < t0:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                                message=f"Step '{first_ev.step_id}' invalid anchor range [{t0:.3f}s, {t1:.3f}s]: end precedes start.",
                                field_path=f"timeline[{first_idx}].narration_anchor.audio_timestamp_range",
                                severity="ERROR",
                            )
                        )
                        is_schedule_valid = False

                    # Check monotonicity against previous step's end
                    if prev_step_end is not None:
                        # ε_overlap = 0.005s tolerance
                        if t0 < prev_step_end - 0.005:
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_TIMELINE_ANCHOR_OVERLAP,
                                    message=f"Step '{first_ev.step_id}' starts at {t0:.3f}s before previous step ended at {prev_step_end:.3f}s (overlap exceeds 0.005s).",
                                    field_path=f"timeline[{first_idx}].narration_anchor.audio_timestamp_range",
                                    severity="ERROR",
                                )
                            )
                            is_schedule_valid = False

                    prev_anchor_t0 = t0
                    prev_anchor_t1 = t1

            # ---------------------------------------------------------
            # 2. Co-Incident Event Conflict & Duplicate Detection
            # ---------------------------------------------------------
            if is_group_coincident:
                # Pairwise conflict detection across events in this co-incident group
                for g_i in range(len(group)):
                    for g_j in range(g_i + 1, len(group)):
                        idx_a, ev_a = group[g_i]
                        idx_b, ev_b = group[g_j]
                        targets_a = {a.target_entity for a in ev_a.actions}
                        targets_b = {a.target_entity for a in ev_b.actions}
                        common_targets = targets_a.intersection(targets_b)

                        for c_tgt in common_targets:
                            acts_a = [a for a in ev_a.actions if a.target_entity == c_tgt]
                            acts_b = [a for a in ev_b.actions if a.target_entity == c_tgt]
                            for act_a in acts_a:
                                for act_b in acts_b:
                                    if act_a.type == act_b.type and deep_compare_parameters(act_a.parameters, act_b.parameters):
                                        # Identical duplicate action
                                        code = (
                                            CompilationErrorCode.TIMELINE_LIFECYCLE_DUPLICATE_ENTER
                                            if act_a.type == ActionType.ENTER
                                            else CompilationErrorCode.TIMELINE_LIFECYCLE_DUPLICATE_ACTION
                                        )
                                        diagnostics.append(
                                            CompilationDiagnostic(
                                                error_code=code,
                                                message=f"Duplicate action '{act_a.type.value}' on entity '{c_tgt}' across co-incident events '{ev_a.step_id}' and '{ev_b.step_id}'.",
                                                entity_id=c_tgt,
                                                severity="WARNING",
                                            )
                                        )
                                    else:
                                        # Conflicting mutation
                                        diagnostics.append(
                                            CompilationDiagnostic(
                                                error_code=CompilationErrorCode.COMPILER_TIMELINE_TARGET_CONFLICT,
                                                message=(
                                                    f"Entity '{c_tgt}' is targeted by concurrent conflicting actions across "
                                                    f"co-incident events '{ev_a.step_id}' and '{ev_b.step_id}' at [{t0:.3f}s, {t1:.3f}s]."
                                                ),
                                                entity_id=c_tgt,
                                                severity="ERROR",
                                            )
                                        )
                                        is_schedule_valid = False

            # ---------------------------------------------------------
            # 3. Action Duration Validation & Step Concurrency
            # ---------------------------------------------------------
            # Determine if this step / group executes concurrently
            step_is_concurrent = is_group_coincident or any(
                getattr(ev, "is_concurrent", False) or any(
                    getattr(a, "parameters", {}).get("is_concurrent", False) for a in ev.actions
                )
                for _, ev in group
            )

            # Flatten actions in group while keeping provenance
            group_actions: List[Tuple[str, int, ActionDeclaration]] = []
            for ev_idx, ev in group:
                for a_idx, action in enumerate(ev.actions):
                    group_actions.append((ev.step_id, a_idx, action))

            # Validate each action's duration
            for s_id, a_idx, action in group_actions:
                raw_dur = action.duration
                if isinstance(raw_dur, bool) or not isinstance(raw_dur, (int, float)):
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Action duration must be numeric, got {type(raw_dur).__name__}.",
                            entity_id=action.target_entity,
                            field_path=f"step('{s_id}').actions[{a_idx}].duration",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                elif math.isnan(raw_dur) or math.isinf(raw_dur) or raw_dur < 0.0:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Action duration must be finite and non-negative, got {raw_dur}.",
                            entity_id=action.target_entity,
                            field_path=f"step('{s_id}').actions[{a_idx}].duration",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                elif raw_dur < 0.100:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Action duration {raw_dur:.4f}s is less than minimum effective duration 0.100s.",
                            entity_id=action.target_entity,
                            field_path=f"step('{s_id}').actions[{a_idx}].duration",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False

            # Compute active animation duration D
            if step_is_concurrent:
                # Concurrent actions: D = max(action durations)
                d_active = max((canonicalize_time(a.duration) for _, _, a in group_actions if isinstance(a.duration, (int, float)) and not isinstance(a.duration, bool) and a.duration >= 0.100), default=0.0)
            else:
                # Sequential actions: D = sum(action durations)
                d_active = sum((canonicalize_time(a.duration) for _, _, a in group_actions if isinstance(a.duration, (int, float)) and not isinstance(a.duration, bool) and a.duration >= 0.100))
                d_active = round(d_active, 3)

            # ---------------------------------------------------------
            # 4. Budget & Pacing Evaluation
            # ---------------------------------------------------------
            step_t_start: float
            step_t_end: float
            delta_t: float
            lead_in: float = 0.0
            slack: float = 0.0

            if has_anchor:
                step_t_start = t0
                step_t_end = t1
                delta_t = round(t1 - t0, 3)

                # Compute lead-in pause from previous step
                if prev_step_end is not None:
                    if t0 > prev_step_end:
                        lead_in = round(t0 - prev_step_end, 3)
                elif t0 > 0.0:
                    lead_in = t0

                # Check budget overrun against delta_t
                if d_active > delta_t + 0.001:
                    overrun = round(d_active - delta_t, 3)
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN,
                            message=f"Step '{step_id}' required duration {d_active:.3f}s exceeds available budget {delta_t:.3f}s by {overrun:.3f}s.",
                            field_path=f"step('{step_id}').actions",
                            severity="ERROR",
                            details={"required_duration": d_active, "available_budget": delta_t, "overrun": overrun},
                        )
                    )
                    is_schedule_valid = False
                    slack = 0.0
                else:
                    slack = round(delta_t - d_active, 3)
                    target_pause = self.target_pause_normal if delta_t >= 1.50 else self.target_pause_short
                    if slack < target_pause:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.TIMELINE_PACING_RUSHED,
                                message=f"Step '{step_id}' resting pause {slack:.3f}s is less than target pause {target_pause:.3f}s.",
                                field_path=f"step('{step_id}')",
                                severity="WARNING",
                                details={"slack": slack, "target_pause": target_pause},
                            )
                        )
                    elif slack > 3.0:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.TIMELINE_PACING_UNDERUTILIZED,
                                message=f"Step '{step_id}' has excessive slack {slack:.3f}s (> 3.0s).",
                                field_path=f"step('{step_id}')",
                                severity="WARNING",
                                details={"slack": slack},
                            )
                        )

                prev_step_end = t1
                unanchored_clock = t1
            else:
                # No audio anchor: sequential execution driven by animation durations
                step_t_start = canonicalize_time(unanchored_clock)
                step_t_end = canonicalize_time(unanchored_clock + d_active)
                delta_t = d_active
                lead_in = 0.0
                slack = 0.0
                unanchored_clock = step_t_end
                prev_step_end = step_t_end

            # ---------------------------------------------------------
            # 5. Build ScheduledAction Records
            # ---------------------------------------------------------
            step_actions: List[ScheduledAction] = []
            curr_act_time = step_t_start

            for s_id, a_idx, action in group_actions:
                if not (isinstance(action.duration, (int, float)) and not isinstance(action.duration, bool) and action.duration >= 0.100):
                    continue

                a_dur = canonicalize_time(action.duration)
                if step_is_concurrent:
                    act_start = step_t_start
                else:
                    act_start = curr_act_time

                act_end = canonicalize_time(act_start + a_dur)
                if not step_is_concurrent:
                    curr_act_time = act_end

                if is_schedule_valid:
                    sa = ScheduledAction(
                        step_id=s_id,
                        action_index=a_idx,
                        target_entity=action.target_entity,
                        action=action,
                        start_time=act_start,
                        end_time=act_end,
                        duration=a_dur,
                        is_concurrent=step_is_concurrent,
                    )
                    step_actions.append(sa)
                    all_scheduled_actions.append(sa)

            # ---------------------------------------------------------
            # 6. Build ScheduledStep
            # ---------------------------------------------------------
            s_step = ScheduledStep(
                step_id=step_id,
                step_index=group_idx,
                t_start=step_t_start,
                t_end=step_t_end,
                delta_t=delta_t,
                is_concurrent=step_is_concurrent,
                actions=step_actions,
                resting_pause=max(0.0, slack),
                lead_in_pause=max(0.0, lead_in),
                constituent_step_ids=[ev.step_id for _, ev in group],
            )
            scheduled_steps.append(s_step)

        # -------------------------------------------------------------
        # 7. Total Scene Duration Budget Enforcement
        # -------------------------------------------------------------
        total_sched_dur = scheduled_steps[-1].t_end if scheduled_steps else 0.0
        # ε_budget = 0.10s macro headroom
        if total_sched_dur > duration_budget + 0.10:
            overrun = round(total_sched_dur - duration_budget, 3)
            diagnostics.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN,
                    message=f"Total scheduled scene duration {total_sched_dur:.3f}s exceeds duration budget {duration_budget:.3f}s by {overrun:.3f}s.",
                    severity="ERROR",
                    details={"total_duration": total_sched_dur, "duration_budget": duration_budget, "overrun": overrun},
                )
            )
            is_schedule_valid = False

        # -------------------------------------------------------------
        # 8. Lifecycle State Machine Validation for All Entities
        # -------------------------------------------------------------
        for ent in scene.entities:
            # Reconstruct disjoint intervals
            ent_diags: List[CompilationDiagnostic] = []
            init_presence, presence_diags = determine_initial_presence(
                entity=ent,
                scene=scene,
                scheduled_actions=all_scheduled_actions,
            )
            ent_diags.extend(presence_diags)

            intervals = build_disjoint_lifespan_intervals(
                eid=ent.id,
                scheduled_actions=all_scheduled_actions,
                scene_duration=total_sched_dur or duration_budget,
                initial_presence=init_presence,
                diagnostics=ent_diags,
            )
            diagnostics.extend(ent_diags)
            if any(d.severity == "ERROR" for d in ent_diags):
                is_schedule_valid = False

        # -------------------------------------------------------------
        # 9. All-or-Nothing Final Schedule Emission
        # -------------------------------------------------------------
        if not is_schedule_valid or any(d.severity == "ERROR" for d in diagnostics):
            return ScheduledTimeline(
                scene_id=scene.scene_id,
                total_duration=total_sched_dur,
                is_valid=False,
                steps=[],
                actions=[],
                diagnostics=diagnostics,
            ), diagnostics

        return ScheduledTimeline(
            scene_id=scene.scene_id,
            total_duration=total_sched_dur,
            is_valid=True,
            steps=scheduled_steps,
            actions=all_scheduled_actions,
            diagnostics=diagnostics,
        ), diagnostics

    def retime_scene_timeline(
        self,
        scene: SceneIR,
        inter_cue_pause: float = 0.50,
        scene_end_pause: float = 1.00,
    ) -> Tuple[SceneIR, ScheduledTimeline, List[CompilationDiagnostic]]:
        """
        Audio-authoritative timeline retiming for SceneIR documents.

        Reconciles spoken narration durations and visual animation requirements:
        1. If D_visual > D_audio: Visual step extends to D_visual, downstream steps shift forward
           preserving audio integrity and strictly monotonic event progression.
        2. If D_audio >= D_visual: Step duration takes D_audio, visuals complete within speech,
           and resting pause holds the final visual frame until narration ends.
        3. Inter-cue pause: Enforces configurable breathing pause between sequential narration cues.
        4. Scene end pause: Holds the terminal scene frame before transition.
        5. Total duration budget: Fails closed with COMPILER_TIMELINE_BUDGET_OVERRUN if the
           retimed scene exceeds scene.duration_budget.
        6. Updates retimed_scene.audio_reference.alignment_status = "retimed" on successful schedule.
        """
        diagnostics: List[CompilationDiagnostic] = []
        is_schedule_valid = True

        if not isinstance(scene, SceneIR):
            raise TypeError(f"Expected SceneIR, got {type(scene).__name__}")

        retimed_scene = scene.model_copy(deep=True)
        duration_budget = float(scene.duration_budget or 30.0)

        # Check for duplicate entity IDs
        seen_entities: Set[str] = set()
        for idx, ent in enumerate(retimed_scene.entities):
            if ent.id in seen_entities:
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                        message=f"Duplicate entity id '{ent.id}' declared in scene.",
                        entity_id=ent.id,
                        field_path=f"entities[{idx}].id",
                        severity="ERROR",
                    )
                )
                is_schedule_valid = False
            seen_entities.add(ent.id)

        # Check initial presence and metadata consistency for all entities
        for ent in retimed_scene.entities:
            _, init_diags = determine_initial_presence(ent, retimed_scene)
            diagnostics.extend(init_diags)
            if any(d.severity == "ERROR" for d in init_diags):
                is_schedule_valid = False

        if not retimed_scene.timeline:
            return retimed_scene, ScheduledTimeline(
                scene_id=retimed_scene.scene_id,
                total_duration=0.0,
                is_valid=is_schedule_valid,
                steps=[],
                actions=[],
                diagnostics=diagnostics,
            ), diagnostics

        # Group co-incident events sharing identical audio anchor intervals
        raw_events: List[TimelineEvent] = list(retimed_scene.timeline)
        event_groups: List[List[Tuple[int, TimelineEvent]]] = []
        i = 0
        while i < len(raw_events):
            ev = raw_events[i]
            anchor = ev.narration_anchor
            if anchor and anchor.audio_timestamp_range:
                t0_anchor, t1_anchor = anchor.audio_timestamp_range
                group = [(i, ev)]
                j = i + 1
                while j < len(raw_events):
                    next_ev = raw_events[j]
                    next_anchor = next_ev.narration_anchor
                    if next_anchor and next_anchor.audio_timestamp_range:
                        nt0, nt1 = next_anchor.audio_timestamp_range
                        if nt0 == t0_anchor and nt1 == t1_anchor:
                            group.append((j, next_ev))
                            j += 1
                            continue
                    break
                event_groups.append(group)
                i = j
            else:
                event_groups.append([(i, ev)])
                i += 1

        all_scheduled_actions: List[ScheduledAction] = []
        scheduled_steps: List[ScheduledStep] = []
        prev_step_end: Optional[float] = None
        prev_had_anchor: bool = False

        for group_idx, group in enumerate(event_groups):
            is_group_coincident = len(group) > 1
            first_idx, first_ev = group[0]
            step_id = first_ev.step_id if not is_group_coincident else f"{first_ev.step_id}_group"

            # 1. Narration Anchor Validation
            has_anchor = False
            d_audio = 0.0

            if first_ev.narration_anchor and first_ev.narration_anchor.audio_timestamp_range:
                raw_t0, raw_t1 = first_ev.narration_anchor.audio_timestamp_range
                if isinstance(raw_t0, bool) or isinstance(raw_t1, bool) or not isinstance(raw_t0, (int, float)) or not isinstance(raw_t1, (int, float)):
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Step '{first_ev.step_id}' audio timestamp range must be numeric, got ({type(raw_t0).__name__}, {type(raw_t1).__name__}).",
                            field_path=f"timeline[{first_idx}].narration_anchor.audio_timestamp_range",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                elif math.isnan(raw_t0) or math.isnan(raw_t1) or math.isinf(raw_t0) or math.isinf(raw_t1):
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Step '{first_ev.step_id}' audio timestamp range must be finite, got ({raw_t0}, {raw_t1}).",
                            field_path=f"timeline[{first_idx}].narration_anchor.audio_timestamp_range",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                elif raw_t0 < 0.0 or raw_t1 < 0.0:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Step '{first_ev.step_id}' audio timestamp range cannot be negative: ({raw_t0}, {raw_t1}).",
                            field_path=f"timeline[{first_idx}].narration_anchor.audio_timestamp_range",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                elif raw_t1 < raw_t0:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Step '{first_ev.step_id}' invalid anchor range [{raw_t0:.3f}s, {raw_t1:.3f}s]: end precedes start.",
                            field_path=f"timeline[{first_idx}].narration_anchor.audio_timestamp_range",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                else:
                    has_anchor = True
                    d_audio = canonicalize_time(raw_t1 - raw_t0)

            # Check coincident conflicts (if coincident)
            if is_group_coincident:
                for g_i in range(len(group)):
                    for g_j in range(g_i + 1, len(group)):
                        idx_a, ev_a = group[g_i]
                        idx_b, ev_b = group[g_j]
                        targets_a = {a.target_entity for a in ev_a.actions}
                        targets_b = {a.target_entity for a in ev_b.actions}
                        common_targets = targets_a.intersection(targets_b)

                        for c_tgt in common_targets:
                            acts_a = [a for a in ev_a.actions if a.target_entity == c_tgt]
                            acts_b = [a for a in ev_b.actions if a.target_entity == c_tgt]
                            for act_a in acts_a:
                                for act_b in acts_b:
                                    if act_a.type == act_b.type and deep_compare_parameters(act_a.parameters, act_b.parameters):
                                        code = (
                                            CompilationErrorCode.TIMELINE_LIFECYCLE_DUPLICATE_ENTER
                                            if act_a.type == ActionType.ENTER
                                            else CompilationErrorCode.TIMELINE_LIFECYCLE_DUPLICATE_ACTION
                                        )
                                        diagnostics.append(
                                            CompilationDiagnostic(
                                                error_code=code,
                                                message=f"Duplicate action '{act_a.type.value}' on entity '{c_tgt}' across co-incident events '{ev_a.step_id}' and '{ev_b.step_id}'.",
                                                entity_id=c_tgt,
                                                severity="WARNING",
                                            )
                                        )
                                    else:
                                        diagnostics.append(
                                            CompilationDiagnostic(
                                                error_code=CompilationErrorCode.COMPILER_TIMELINE_TARGET_CONFLICT,
                                                message=f"Entity '{c_tgt}' is targeted by concurrent conflicting actions across co-incident events '{ev_a.step_id}' and '{ev_b.step_id}'.",
                                                entity_id=c_tgt,
                                                severity="ERROR",
                                            )
                                        )
                                        is_schedule_valid = False

            # 2. Action Duration Validation & Concurrency
            step_is_concurrent = is_group_coincident or any(
                getattr(ev, "is_concurrent", False) or any(
                    getattr(a, "parameters", {}).get("is_concurrent", False) for a in ev.actions
                )
                for _, ev in group
            )

            group_actions: List[Tuple[str, int, ActionDeclaration]] = []
            for ev_idx, ev in group:
                for a_idx, action in enumerate(ev.actions):
                    group_actions.append((ev.step_id, a_idx, action))

            for s_id, a_idx, action in group_actions:
                raw_dur = action.duration
                if isinstance(raw_dur, bool) or not isinstance(raw_dur, (int, float)):
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Action duration must be numeric, got {type(raw_dur).__name__}.",
                            entity_id=action.target_entity,
                            field_path=f"step('{s_id}').actions[{a_idx}].duration",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                elif math.isnan(raw_dur) or math.isinf(raw_dur) or raw_dur < 0.0:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Action duration must be finite and non-negative, got {raw_dur}.",
                            entity_id=action.target_entity,
                            field_path=f"step('{s_id}').actions[{a_idx}].duration",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False
                elif raw_dur < 0.100:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Action duration {raw_dur:.4f}s is less than minimum effective duration 0.100s.",
                            entity_id=action.target_entity,
                            field_path=f"step('{s_id}').actions[{a_idx}].duration",
                            severity="ERROR",
                        )
                    )
                    is_schedule_valid = False

            # Compute active animation duration D_visual
            if step_is_concurrent:
                d_visual = max(
                    (canonicalize_time(a.duration) for _, _, a in group_actions
                     if isinstance(a.duration, (int, float)) and not isinstance(a.duration, bool) and a.duration >= 0.100),
                    default=0.0
                )
            else:
                d_visual = sum(
                    (canonicalize_time(a.duration) for _, _, a in group_actions
                     if isinstance(a.duration, (int, float)) and not isinstance(a.duration, bool) and a.duration >= 0.100)
                )
                d_visual = round(d_visual, 3)

            # 3. Audio-Authoritative Retiming & Step Positioning
            if has_anchor:
                delta_t = max(d_visual, d_audio)
            else:
                delta_t = d_visual

            if group_idx == 0:
                step_t_start = 0.0
                lead_in = 0.0
            else:
                assert prev_step_end is not None
                if has_anchor or prev_had_anchor:
                    step_t_start = canonicalize_time(prev_step_end + inter_cue_pause)
                    lead_in = canonicalize_time(inter_cue_pause)
                else:
                    step_t_start = canonicalize_time(prev_step_end)
                    lead_in = 0.0

            step_t_end = canonicalize_time(step_t_start + delta_t)

            # Resting pause: held after visual activity completes
            slack = max(0.0, canonicalize_time(delta_t - d_visual))

            # Update narration anchor timestamps in retimed SceneIR
            if has_anchor:
                retimed_audio_range = (step_t_start, canonicalize_time(step_t_start + d_audio))
                for _, ev in group:
                    if ev.narration_anchor:
                        ev.narration_anchor.audio_timestamp_range = retimed_audio_range

            # 4. Build ScheduledActions
            step_actions: List[ScheduledAction] = []
            curr_act_time = step_t_start
            for s_id, a_idx, action in group_actions:
                if not (isinstance(action.duration, (int, float)) and not isinstance(action.duration, bool) and action.duration >= 0.100):
                    continue
                a_dur = canonicalize_time(action.duration)
                act_start = step_t_start if step_is_concurrent else curr_act_time
                act_end = canonicalize_time(act_start + a_dur)
                if not step_is_concurrent:
                    curr_act_time = act_end

                if is_schedule_valid:
                    sa = ScheduledAction(
                        step_id=s_id,
                        action_index=a_idx,
                        target_entity=action.target_entity,
                        action=action,
                        start_time=act_start,
                        end_time=act_end,
                        duration=a_dur,
                        is_concurrent=step_is_concurrent,
                    )
                    step_actions.append(sa)
                    all_scheduled_actions.append(sa)

            # 5. Build ScheduledStep
            s_step = ScheduledStep(
                step_id=step_id,
                step_index=group_idx,
                t_start=step_t_start,
                t_end=step_t_end,
                delta_t=round(step_t_end - step_t_start, 3),
                is_concurrent=step_is_concurrent,
                actions=step_actions,
                resting_pause=slack,
                lead_in_pause=lead_in,
                constituent_step_ids=[ev.step_id for _, ev in group],
            )
            scheduled_steps.append(s_step)

            prev_step_end = step_t_end
            prev_had_anchor = has_anchor

        # 6. Total Scene Duration & Scene End Pause
        if scheduled_steps:
            total_sched_dur = canonicalize_time(scheduled_steps[-1].t_end + scene_end_pause)
        else:
            total_sched_dur = 0.0

        # 7. Total Scene Duration Budget Enforcement
        if total_sched_dur > duration_budget + 0.10:
            overrun = round(total_sched_dur - duration_budget, 3)
            diagnostics.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN,
                    message=f"Total scheduled scene duration {total_sched_dur:.3f}s exceeds duration budget {duration_budget:.3f}s by {overrun:.3f}s.",
                    severity="ERROR",
                    details={"total_duration": total_sched_dur, "duration_budget": duration_budget, "overrun": overrun},
                )
            )
            is_schedule_valid = False

        # 8. Lifecycle State Machine Validation for All Entities
        for ent in retimed_scene.entities:
            ent_diags: List[CompilationDiagnostic] = []
            init_presence, presence_diags = determine_initial_presence(
                entity=ent,
                scene=retimed_scene,
                scheduled_actions=all_scheduled_actions,
            )
            ent_diags.extend(presence_diags)

            intervals = build_disjoint_lifespan_intervals(
                eid=ent.id,
                scheduled_actions=all_scheduled_actions,
                scene_duration=total_sched_dur or duration_budget,
                initial_presence=init_presence,
                diagnostics=ent_diags,
            )
            diagnostics.extend(ent_diags)
            if any(d.severity == "ERROR" for d in ent_diags):
                is_schedule_valid = False

        # 9. Update alignment status on retimed_scene
        if is_schedule_valid and not any(d.severity == "ERROR" for d in diagnostics):
            if retimed_scene.audio_reference is not None:
                retimed_scene.audio_reference.alignment_status = "retimed"

        # 10. All-or-Nothing Final Schedule Emission
        if not is_schedule_valid or any(d.severity == "ERROR" for d in diagnostics):
            return retimed_scene, ScheduledTimeline(
                scene_id=retimed_scene.scene_id,
                total_duration=total_sched_dur,
                is_valid=False,
                steps=[],
                actions=[],
                diagnostics=diagnostics,
            ), diagnostics

        return retimed_scene, ScheduledTimeline(
            scene_id=retimed_scene.scene_id,
            total_duration=total_sched_dur,
            is_valid=True,
            steps=scheduled_steps,
            actions=all_scheduled_actions,
            diagnostics=diagnostics,
        ), diagnostics
