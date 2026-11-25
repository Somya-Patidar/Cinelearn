"""
Milestone 6.2: Event-Driven Temporal Keyframe Sampler.

Derives deterministic, event-associated video sampling requests from authoritative
timeline events (ScheduledTimeline, ScheduledStep, ScheduledAction, TimelineEvent)
rather than arbitrary fixed intervals.

Features:
- Samples: pre-action, mid-transition, settled, resting, and scene terminal phases.
- Deduplicates timestamps within configurable tolerance while preserving all event references.
- Distinguishes requested vs actual decoded timestamps with structured tolerance diagnostics.
- Strict numeric validation rejecting NaN, Inf, negative, and boolean values.
- Zero live video rendering or LLM dependencies.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.compiler.compiler_models import ScheduledAction, ScheduledStep, ScheduledTimeline
from src.ir.models import ActionDeclaration, ActionType, SceneIR, TimelineEvent
from src.qc.models import (
    QCDiagnostic,
    QCDiagnosticCode,
    QCDiagnosticLevel,
    QCLayer,
)


# ============================================================================
# Enums and Typed Models for Keyframe Sampling
# ============================================================================

class SamplePhase(str, Enum):
    """Lifecycle phase representing the visual animation state at sampling time."""
    PRE_ACTION = "PRE_ACTION"            # Instant before or at start of action
    MID_TRANSITION = "MID_TRANSITION"    # Intermediate state during dynamic motion / transition
    SETTLED = "SETTLED"                  # State immediately upon action completion
    RESTING = "RESTING"                  # Static holding state during pause or narration interval
    SCENE_TERMINAL = "SCENE_TERMINAL"    # Final resting frame before scene conclusion


class SamplingEventRef(BaseModel):
    """
    Structured provenance reference associating a keyframe sample with its source
    timeline step, action, entity, and phase.
    """
    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=1)
    step_id: Optional[str] = None
    action_index: Optional[int] = Field(default=None, ge=0)
    target_entity: Optional[str] = None
    action_type: Optional[str] = None
    phase: SamplePhase
    event_start_time: Optional[float] = Field(default=None, ge=0.0)
    event_end_time: Optional[float] = Field(default=None, ge=0.0)


class KeyframeSamplerConfig(BaseModel):
    """
    Configurable policies and tolerances for event-driven keyframe sampling.
    """
    model_config = ConfigDict(extra="forbid")

    sample_pre_action: bool = True
    sample_mid_transition: bool = True
    sample_settled: bool = True
    sample_resting: bool = True
    sample_scene_terminal: bool = True

    min_motion_duration: float = Field(
        default=0.30,
        ge=0.0,
        description="Minimum action duration required to sample an intermediate mid-transition frame.",
    )
    mid_point_ratio: float = Field(
        default=0.50,
        gt=0.0,
        lt=1.0,
        description="Relative progress point [0, 1] for intermediate motion sampling.",
    )
    resting_offset_seconds: float = Field(
        default=0.10,
        ge=0.0,
        description="Offset before step end to sample resting state during a pause.",
    )
    scene_terminal_offset_seconds: float = Field(
        default=0.05,
        ge=0.0,
        description="Offset before scene total duration to sample terminal state.",
    )
    dedup_tolerance_seconds: float = Field(
        default=0.033,
        ge=0.0,
        description="Temporal clustering window for merging adjacent sample requests (e.g. ~1/30s).",
    )
    max_seek_deviation_seconds: float = Field(
        default=0.100,
        ge=0.0,
        description="Maximum permitted deviation between requested and actual decoded video timestamps.",
    )


class KeyframeSampleRequest(BaseModel):
    """
    A single deterministic keyframe extraction request at a specific video timestamp,
    carrying all associated timeline event references.
    """
    model_config = ConfigDict(extra="forbid")

    requested_timestamp: float = Field(ge=0.0, description="Canonical target timestamp (seconds)")
    scene_id: str = Field(min_length=1)
    event_refs: List[SamplingEventRef] = Field(default_factory=list)
    associated_step_ids: List[str] = Field(default_factory=list)
    associated_entity_ids: List[str] = Field(default_factory=list)
    sample_phases: List[SamplePhase] = Field(default_factory=list)

    @field_validator("requested_timestamp", mode="before")
    @classmethod
    def validate_timestamp_finite(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("requested_timestamp must be a numeric value, not boolean.")
        if not isinstance(v, (int, float)):
            raise ValueError(f"requested_timestamp must be numeric, got {type(v).__name__}.")
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            raise ValueError("requested_timestamp must be finite.")
        if f < 0.0:
            raise ValueError("requested_timestamp cannot be negative.")
        return round(f, 3)

    @property
    def is_terminal(self) -> bool:
        return SamplePhase.SCENE_TERMINAL in self.sample_phases

    @property
    def primary_step_id(self) -> Optional[str]:
        return self.associated_step_ids[0] if self.associated_step_ids else None


class TimestampEvidence(BaseModel):
    """
    Verification evidence comparing requested versus actual decoded video timestamp.
    """
    model_config = ConfigDict(extra="forbid")

    requested_timestamp: float = Field(ge=0.0)
    actual_timestamp: float = Field(ge=0.0)
    deviation_seconds: float = Field(ge=0.0)
    tolerance_seconds: float = Field(ge=0.0)
    is_within_tolerance: bool


class KeyframeSamplingPlan(BaseModel):
    """
    Complete deterministic sampling plan for a scene timeline.
    """
    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=1)
    total_duration: float = Field(ge=0.0)
    sample_requests: List[KeyframeSampleRequest] = Field(default_factory=list)
    diagnostics: List[QCDiagnostic] = Field(default_factory=list)

    @property
    def timestamps(self) -> List[float]:
        """Returns sorted requested timestamps."""
        return [r.requested_timestamp for r in self.sample_requests]

    @property
    def count(self) -> int:
        return len(self.sample_requests)

    def get_requests_for_step(self, step_id: str) -> List[KeyframeSampleRequest]:
        """Returns all sample requests associated with a specific step ID."""
        return [r for r in self.sample_requests if step_id in r.associated_step_ids]

    def get_requests_for_entity(self, entity_id: str) -> List[KeyframeSampleRequest]:
        """Returns all sample requests associated with a target entity ID."""
        return [r for r in self.sample_requests if entity_id in r.associated_entity_ids]


class KeyframeSampleResult(BaseModel):
    """
    Result of keyframe extraction containing requested vs actual timestamp evidence
    and any decoding diagnostics.
    """
    model_config = ConfigDict(extra="forbid")

    request: KeyframeSampleRequest
    timestamp_evidence: TimestampEvidence
    frame_extracted: bool
    frame_shape: Optional[Tuple[int, int, int]] = None
    diagnostics: List[QCDiagnostic] = Field(default_factory=list)


# ============================================================================
# Keyframe Sampler Engine
# ============================================================================

def _validate_numeric_time(val: Any, name: str = "timestamp") -> float:
    """Strictly validates a time parameter rejecting booleans, non-numbers, NaN, Inf, and negatives."""
    if isinstance(val, bool):
        raise ValueError(f"{name} cannot be a boolean value.")
    if not isinstance(val, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(val).__name__}.")
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{name} must be finite, got {val}.")
    if f < 0.0:
        raise ValueError(f"{name} cannot be negative, got {val}.")
    return round(f, 3)


class KeyframeSampler:
    """
    Deterministic event-driven keyframe sampler for CineLearn QC.
    Translates timeline events into deduplicated keyframe sample requests.
    """

    def __init__(self, config: Optional[KeyframeSamplerConfig] = None):
        self.config = config or KeyframeSamplerConfig()

    def sample_timeline(
        self,
        timeline: ScheduledTimeline,
        scene_duration: Optional[float] = None,
    ) -> KeyframeSamplingPlan:
        """
        Derives an event-driven keyframe sampling plan from an authoritative ScheduledTimeline.
        """
        scene_id = timeline.scene_id
        duration = float(scene_duration if scene_duration is not None else timeline.total_duration)
        duration = _validate_numeric_time(duration, name="scene_duration")

        diagnostics: List[QCDiagnostic] = []
        raw_candidates: List[Tuple[float, SamplingEventRef]] = []

        # 1. Traverse scheduled steps and their constituent scheduled actions
        for step in timeline.steps:
            step_id = step.step_id
            t_start = _validate_numeric_time(step.t_start, f"step[{step_id}].t_start")
            t_end = _validate_numeric_time(step.t_end, f"step[{step_id}].t_end")

            # Check for negative or inverted step intervals
            if t_end < t_start:
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_TIMELINE_BUDGET_OVERRUN,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_B,
                    message=f"Step '{step_id}' has inverted time range: start={t_start}s > end={t_end}s.",
                    scene_id=scene_id,
                    step_id=step_id,
                    timestamp=t_start,
                )
                diagnostics.append(diag)
                continue

            # Process actions within the step
            if step.actions:
                for act in step.actions:
                    a_start = _validate_numeric_time(act.start_time, f"action[{act.step_id}].start_time")
                    a_end = _validate_numeric_time(act.end_time, f"action[{act.step_id}].end_time")
                    a_dur = _validate_numeric_time(act.duration, f"action[{act.step_id}].duration")
                    eid = act.target_entity
                    act_type_str = act.action.type.value if hasattr(act.action.type, "value") else str(act.action.type)

                    # A. Pre-action sample
                    if self.config.sample_pre_action:
                        ref = SamplingEventRef(
                            scene_id=scene_id,
                            step_id=step_id,
                            action_index=act.action_index,
                            target_entity=eid,
                            action_type=act_type_str,
                            phase=SamplePhase.PRE_ACTION,
                            event_start_time=a_start,
                            event_end_time=a_end,
                        )
                        raw_candidates.append((a_start, ref))

                    # B. Mid-transition sample (only for actions with meaningful duration)
                    if self.config.sample_mid_transition and a_dur >= self.config.min_motion_duration:
                        t_mid = round(a_start + self.config.mid_point_ratio * a_dur, 3)
                        ref = SamplingEventRef(
                            scene_id=scene_id,
                            step_id=step_id,
                            action_index=act.action_index,
                            target_entity=eid,
                            action_type=act_type_str,
                            phase=SamplePhase.MID_TRANSITION,
                            event_start_time=a_start,
                            event_end_time=a_end,
                        )
                        raw_candidates.append((t_mid, ref))

                    # C. Settled state sample
                    if self.config.sample_settled:
                        ref = SamplingEventRef(
                            scene_id=scene_id,
                            step_id=step_id,
                            action_index=act.action_index,
                            target_entity=eid,
                            action_type=act_type_str,
                            phase=SamplePhase.SETTLED,
                            event_start_time=a_start,
                            event_end_time=a_end,
                        )
                        raw_candidates.append((a_end, ref))

                # D. Resting state sample during step holding pause
                if self.config.sample_resting:
                    # If step has a resting pause or finishes after its actions
                    max_action_end = max((a.end_time for a in step.actions), default=t_start)
                    if step.resting_pause >= self.config.resting_offset_seconds or t_end > max_action_end + 0.05:
                        t_rest = max(max_action_end, round(t_end - self.config.resting_offset_seconds, 3))
                        ref = SamplingEventRef(
                            scene_id=scene_id,
                            step_id=step_id,
                            phase=SamplePhase.RESTING,
                            event_start_time=t_start,
                            event_end_time=t_end,
                        )
                        raw_candidates.append((t_rest, ref))

            else:
                # Step without actions (e.g. pause, narrator holding interval)
                if self.config.sample_pre_action:
                    raw_candidates.append((
                        t_start,
                        SamplingEventRef(
                            scene_id=scene_id,
                            step_id=step_id,
                            phase=SamplePhase.PRE_ACTION,
                            event_start_time=t_start,
                            event_end_time=t_end,
                        ),
                    ))
                if self.config.sample_settled and t_end > t_start:
                    raw_candidates.append((
                        t_end,
                        SamplingEventRef(
                            scene_id=scene_id,
                            step_id=step_id,
                            phase=SamplePhase.SETTLED,
                            event_start_time=t_start,
                            event_end_time=t_end,
                        ),
                    ))

        # 2. Scene terminal state sample
        if self.config.sample_scene_terminal and duration > 0.0:
            t_terminal = max(0.0, round(duration - self.config.scene_terminal_offset_seconds, 3))
            raw_candidates.append((
                t_terminal,
                SamplingEventRef(
                    scene_id=scene_id,
                    phase=SamplePhase.SCENE_TERMINAL,
                    event_start_time=0.0,
                    event_end_time=duration,
                ),
            ))

        # 3. Deduplicate and cluster timestamps while preserving all event references
        deduped_requests, cluster_diags = self._deduplicate_candidates(
            raw_candidates=raw_candidates,
            scene_id=scene_id,
            scene_duration=duration,
        )
        diagnostics.extend(cluster_diags)

        return KeyframeSamplingPlan(
            scene_id=scene_id,
            total_duration=duration,
            sample_requests=deduped_requests,
            diagnostics=diagnostics,
        )

    def sample_scene_ir(
        self,
        scene: SceneIR,
    ) -> KeyframeSamplingPlan:
        """
        Derives a keyframe sampling plan directly from a validated SceneIR document.
        Uses TimelineScheduler to compute authoritative scheduled timings.
        """
        from src.compiler.timeline_scheduler import TimelineScheduler

        scheduler = TimelineScheduler()
        scheduled_timeline, _ = scheduler.schedule(scene)
        scene_duration = float(scene.duration_budget) if scene.duration_budget else scheduled_timeline.total_duration
        return self.sample_timeline(scheduled_timeline, scene_duration=scene_duration)

    def sample_raw_events(
        self,
        events: Sequence[TimelineEvent],
        scene_duration: float,
        scene_id: str = "scene",
    ) -> KeyframeSamplingPlan:
        """
        Convenience sampler for raw TimelineEvent sequences without full SceneIR.
        Constructs synthetic scheduled steps for testing.
        """
        duration = _validate_numeric_time(scene_duration, name="scene_duration")
        steps: List[ScheduledStep] = []
        clock = 0.0

        for idx, ev in enumerate(events):
            step_id = ev.step_id or f"step_{idx}"
            actions_list: List[ScheduledAction] = []
            a_clock = clock

            if ev.narration_anchor and ev.narration_anchor.audio_timestamp_range:
                t0, t1 = ev.narration_anchor.audio_timestamp_range
                step_start = t0
                step_end = t1
                a_clock = step_start
            else:
                step_start = a_clock
                step_dur = sum((a.duration for a in ev.actions)) if ev.actions else 1.0
                step_end = round(step_start + step_dur, 3)

            for a_idx, act in enumerate(ev.actions):
                dur = act.duration if act.duration > 0.0 else 0.100
                sa = ScheduledAction(
                    step_id=step_id,
                    action_index=a_idx,
                    target_entity=act.target_entity,
                    action=act,
                    start_time=round(a_clock, 3),
                    end_time=round(a_clock + dur, 3),
                    duration=round(dur, 3),
                )
                actions_list.append(sa)
                a_clock = round(a_clock + dur, 3)

            step = ScheduledStep(
                step_id=step_id,
                step_index=idx,
                t_start=step_start,
                t_end=max(step_end, a_clock),
                delta_t=round(max(step_end, a_clock) - step_start, 3),
                actions=actions_list,
            )
            steps.append(step)
            clock = step.t_end

        timeline = ScheduledTimeline(
            scene_id=scene_id,
            total_duration=max(duration, clock),
            steps=steps,
        )
        return self.sample_timeline(timeline, scene_duration=duration)

    def _deduplicate_candidates(
        self,
        raw_candidates: List[Tuple[float, SamplingEventRef]],
        scene_id: str,
        scene_duration: float,
    ) -> Tuple[List[KeyframeSampleRequest], List[QCDiagnostic]]:
        """
        Deduplicates candidate timestamps within dedup_tolerance_seconds while merging
        all event references, preserving provenance, and filtering out-of-bounds timestamps.
        """
        diagnostics: List[QCDiagnostic] = []
        if not raw_candidates:
            return [], diagnostics

        # Filter and validate candidate timestamps
        valid_candidates: List[Tuple[float, SamplingEventRef]] = []
        tol = self.config.dedup_tolerance_seconds

        for raw_t, ref in raw_candidates:
            # Timestamp bounds checking
            if raw_t < 0.0:
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_TIMELINE_BUDGET_OVERRUN,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_B,
                    message=f"Negative sampling timestamp generated ({raw_t:.3f}s < 0.0s).",
                    scene_id=scene_id,
                    step_id=ref.step_id,
                    entity_id=ref.target_entity,
                    timestamp=raw_t,
                )
                diagnostics.append(diag)
                continue

            if raw_t > scene_duration + 0.001:
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_TIMELINE_BUDGET_OVERRUN,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_B,
                    message=f"Sampling timestamp {raw_t:.3f}s exceeds scene duration {scene_duration:.3f}s.",
                    scene_id=scene_id,
                    step_id=ref.step_id,
                    entity_id=ref.target_entity,
                    timestamp=raw_t,
                )
                diagnostics.append(diag)
                # Clamp to scene duration rather than dropping entirely
                clamped_t = round(scene_duration, 3)
                valid_candidates.append((clamped_t, ref))
                continue

            valid_candidates.append((round(raw_t, 3), ref))

        if not valid_candidates:
            return [], diagnostics

        # Sort candidates deterministically by timestamp, then step_id, then phase
        valid_candidates.sort(key=lambda item: (item[0], item[1].step_id or "", item[1].phase.value))

        # Cluster adjacent timestamps within tolerance
        clusters: List[List[Tuple[float, SamplingEventRef]]] = []
        curr_cluster: List[Tuple[float, SamplingEventRef]] = [valid_candidates[0]]

        for cand in valid_candidates[1:]:
            cand_t = cand[0]
            cluster_base_t = curr_cluster[0][0]
            if cand_t - cluster_base_t <= tol:
                curr_cluster.append(cand)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [cand]
        clusters.append(curr_cluster)

        # Build KeyframeSampleRequest per cluster
        requests: List[KeyframeSampleRequest] = []
        for cluster in clusters:
            canonical_ts = cluster[0][0]  # Authoritative base timestamp
            merged_refs: List[SamplingEventRef] = [c[1] for c in cluster]

            # Collect unique step IDs, entity IDs, and phases
            step_ids: Set[str] = set()
            entity_ids: Set[str] = set()
            phases: Set[SamplePhase] = set()

            for r in merged_refs:
                if r.step_id:
                    step_ids.add(r.step_id)
                if r.target_entity:
                    entity_ids.add(r.target_entity)
                phases.add(r.phase)

            req = KeyframeSampleRequest(
                requested_timestamp=canonical_ts,
                scene_id=scene_id,
                event_refs=merged_refs,
                associated_step_ids=sorted(step_ids),
                associated_entity_ids=sorted(entity_ids),
                sample_phases=sorted(phases, key=lambda p: p.value),
            )
            requests.append(req)

        return requests, diagnostics

    def evaluate_timestamp_accuracy(
        self,
        requested_ts: float,
        actual_ts: float,
    ) -> Tuple[TimestampEvidence, Optional[QCDiagnostic]]:
        """
        Compares requested versus actual decoded video timestamp.
        Emits a structured diagnostic if deviation exceeds max_seek_deviation_seconds.
        """
        r_ts = _validate_numeric_time(requested_ts, "requested_ts")
        a_ts = _validate_numeric_time(actual_ts, "actual_ts")
        deviation = round(abs(a_ts - r_ts), 4)
        tol = self.config.max_seek_deviation_seconds
        is_ok = deviation <= tol

        evidence = TimestampEvidence(
            requested_timestamp=r_ts,
            actual_timestamp=a_ts,
            deviation_seconds=deviation,
            tolerance_seconds=tol,
            is_within_tolerance=is_ok,
        )

        diagnostic = None
        if not is_ok:
            diagnostic = QCDiagnostic(
                code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                level=QCDiagnosticLevel.WARNING,
                layer=QCLayer.LAYER_C,
                message=(
                    f"Excessive timestamp seek deviation: requested {r_ts:.3f}s, "
                    f"decoded {a_ts:.3f}s (delta={deviation:.3f}s > {tol:.3f}s)."
                ),
                timestamp=r_ts,
                evidence=evidence.model_dump(),
            )

        return evidence, diagnostic

    def extract_keyframes(
        self,
        video_path: str,
        plan: KeyframeSamplingPlan,
    ) -> List[KeyframeSampleResult]:
        """
        Extracts keyframes from video artifact for all requests in the sampling plan.
        Validates timestamps at the sampler boundary before invoking OpenCV reader.
        """
        from src.qc.raster_analyzer import read_keyframe_at_timestamp

        results: List[KeyframeSampleResult] = []

        for req in plan.sample_requests:
            ts = req.requested_timestamp
            diags: List[QCDiagnostic] = []

            # Guard against invalid or non-finite timestamp before OpenCV
            try:
                _validate_numeric_time(ts, "requested_timestamp")
            except ValueError as e:
                safe_ts = max(0.0, float(ts)) if isinstance(ts, (int, float)) and not (math.isnan(ts) or math.isinf(ts)) else 0.0
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_C,
                    message=f"Invalid timestamp in sample request: {e}",
                    scene_id=req.scene_id,
                    timestamp=safe_ts,
                )
                evidence = TimestampEvidence(
                    requested_timestamp=safe_ts,
                    actual_timestamp=0.0,
                    deviation_seconds=safe_ts,
                    tolerance_seconds=self.config.max_seek_deviation_seconds,
                    is_within_tolerance=False,
                )
                results.append(
                    KeyframeSampleResult(
                        request=req,
                        timestamp_evidence=evidence,
                        frame_extracted=False,
                        frame_shape=None,
                        diagnostics=[diag],
                    )
                )
                continue

            try:
                frame, meta = read_keyframe_at_timestamp(video_path, ts)
                if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim != 3:
                    raise ValueError(f"Decoded invalid or empty frame array at timestamp {ts:.3f}s")

                # M6.2 acceptance condition: strict metadata validation (fail-closed)
                if not isinstance(meta, dict) or "actual_timestamp_sec" not in meta:
                    raise ValueError(f"Decoder metadata missing mandatory 'actual_timestamp_sec' field at {ts:.3f}s")
                raw_act_ts = meta["actual_timestamp_sec"]
                if raw_act_ts is None or isinstance(raw_act_ts, bool) or not isinstance(raw_act_ts, (int, float)):
                    raise ValueError(f"Decoder returned non-numeric actual_timestamp_sec '{raw_act_ts}' at {ts:.3f}s")
                actual_ts_val = float(raw_act_ts)
                if math.isnan(actual_ts_val) or math.isinf(actual_ts_val):
                    raise ValueError(f"Decoder returned non-finite actual_timestamp_sec '{actual_ts_val}' at {ts:.3f}s")
                actual_ts = max(0.0, actual_ts_val)

                evidence, seek_diag = self.evaluate_timestamp_accuracy(ts, actual_ts)
                if seek_diag:
                    diags.append(seek_diag)

                results.append(
                    KeyframeSampleResult(
                        request=req,
                        timestamp_evidence=evidence,
                        frame_extracted=True,
                        frame_shape=frame.shape,
                        diagnostics=diags,
                    )
                )
            except Exception as e:
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_C,
                    message=f"Keyframe extraction failed at timestamp {ts:.3f}s: {e}",
                    scene_id=req.scene_id,
                    timestamp=ts,
                )
                evidence = TimestampEvidence(
                    requested_timestamp=ts,
                    actual_timestamp=0.0,
                    deviation_seconds=ts,
                    tolerance_seconds=self.config.max_seek_deviation_seconds,
                    is_within_tolerance=False,
                )
                results.append(
                    KeyframeSampleResult(
                        request=req,
                        timestamp_evidence=evidence,
                        frame_extracted=False,
                        frame_shape=None,
                        diagnostics=[diag],
                    )
                )

        return results
