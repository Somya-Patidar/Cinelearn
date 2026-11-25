"""
Milestone 6.2: Unit and Property Tests for Event-Driven Keyframe Sampler.

Verifies:
- Single instantaneous event sampling.
- Event with defined duration and motion thresholding.
- Movement events generating intermediate mid-transition samples.
- Settled state and resting pause state extraction.
- Adjacent, simultaneous, concurrent, and overlapping event handling.
- Pause intervals and zero-duration event handling.
- Scene boundary adherence (no samples outside [0, T_scene]).
- Deterministic deduplication with multi-event provenance preservation.
- Strict numeric validation rejecting NaN, Inf, negative, and boolean timestamps.
- Requested versus actual timestamp mismatch and excessive seek deviation diagnostics.
- Deterministic repeated execution and full JSON serialization.
"""

import math
from pathlib import Path
import pytest
from typing import List

from src.compiler.compiler_models import ScheduledAction, ScheduledStep, ScheduledTimeline
from src.ir.models import ActionDeclaration, ActionType, NarrationAnchor, TimelineEvent
from src.qc.keyframe_sampler import (
    KeyframeSampler,
    KeyframeSamplerConfig,
    KeyframeSampleRequest,
    KeyframeSampleResult,
    KeyframeSamplingPlan,
    SamplePhase,
    SamplingEventRef,
    TimestampEvidence,
)
from src.qc.models import QCDiagnosticCode, QCDiagnosticLevel


class TestKeyframeSamplerBasics:
    """Verifies core event sampling behaviors on synthetic timeline fixtures."""

    def test_single_instantaneous_or_minimal_duration_event(self):
        """Zero or sub-motion duration event samples settled state without mid-motion sample."""
        config = KeyframeSamplerConfig(min_motion_duration=0.30)
        sampler = KeyframeSampler(config=config)

        # Single step with a 0.1s action (below min_motion_duration)
        action = ActionDeclaration(
            type=ActionType.ENTER,
            target_entity="box1",
            duration=0.10,
        )
        step = ScheduledStep(
            step_id="step_intro",
            step_index=0,
            t_start=0.0,
            t_end=0.10,
            delta_t=0.10,
            actions=[
                ScheduledAction(
                    step_id="step_intro",
                    action_index=0,
                    target_entity="box1",
                    action=action,
                    start_time=0.0,
                    end_time=0.10,
                    duration=0.10,
                )
            ],
        )
        timeline = ScheduledTimeline(
            scene_id="scene_single",
            total_duration=1.0,
            steps=[step],
        )

        plan = sampler.sample_timeline(timeline)
        assert plan.scene_id == "scene_single"
        assert plan.count >= 2

        # Verify no MID_TRANSITION phase generated because duration < 0.3s
        all_phases = [p for req in plan.sample_requests for p in req.sample_phases]
        assert SamplePhase.MID_TRANSITION not in all_phases
        assert SamplePhase.PRE_ACTION in all_phases
        assert SamplePhase.SETTLED in all_phases

    def test_event_with_defined_duration_and_mid_transition(self):
        """Action with duration >= min_motion_duration generates mid-transition point."""
        config = KeyframeSamplerConfig(
            min_motion_duration=0.30,
            mid_point_ratio=0.50,
            sample_pre_action=True,
            sample_mid_transition=True,
            sample_settled=True,
            sample_resting=False,
            sample_scene_terminal=False,
            dedup_tolerance_seconds=0.01,
        )
        sampler = KeyframeSampler(config=config)

        action = ActionDeclaration(
            type=ActionType.MOVE,
            target_entity="ball",
            duration=1.0,
        )
        step = ScheduledStep(
            step_id="step_move",
            step_index=0,
            t_start=1.0,
            t_end=2.0,
            delta_t=1.0,
            actions=[
                ScheduledAction(
                    step_id="step_move",
                    action_index=0,
                    target_entity="ball",
                    action=action,
                    start_time=1.0,
                    end_time=2.0,
                    duration=1.0,
                )
            ],
        )
        timeline = ScheduledTimeline(
            scene_id="scene_move",
            total_duration=5.0,
            steps=[step],
        )

        plan = sampler.sample_timeline(timeline)
        # We expect: 1.0 (PRE), 1.5 (MID), 2.0 (SETTLED)
        timestamps = plan.timestamps
        assert 1.0 in timestamps
        assert 1.5 in timestamps
        assert 2.0 in timestamps

        req_mid = next(r for r in plan.sample_requests if r.requested_timestamp == 1.5)
        assert SamplePhase.MID_TRANSITION in req_mid.sample_phases
        assert "ball" in req_mid.associated_entity_ids
        assert "step_move" in req_mid.associated_step_ids

    def test_settled_and_resting_pause_states(self):
        """Step with resting pause samples both settled action state and holding resting state."""
        config = KeyframeSamplerConfig(
            sample_pre_action=True,
            sample_mid_transition=False,
            sample_settled=True,
            sample_resting=True,
            resting_offset_seconds=0.10,
            sample_scene_terminal=False,
            dedup_tolerance_seconds=0.01,
        )
        sampler = KeyframeSampler(config=config)

        # Action finishes at 1.0s, step holds static until 3.0s (resting pause 2.0s)
        action = ActionDeclaration(
            type=ActionType.TRANSFORM,
            target_entity="graph",
            duration=1.0,
        )
        step = ScheduledStep(
            step_id="step_graph",
            step_index=0,
            t_start=0.0,
            t_end=3.0,
            delta_t=1.0,
            resting_pause=2.0,
            actions=[
                ScheduledAction(
                    step_id="step_graph",
                    action_index=0,
                    target_entity="graph",
                    action=action,
                    start_time=0.0,
                    end_time=1.0,
                    duration=1.0,
                )
            ],
        )
        timeline = ScheduledTimeline(
            scene_id="scene_pause",
            total_duration=3.0,
            steps=[step],
        )

        plan = sampler.sample_timeline(timeline)
        timestamps = plan.timestamps

        # Action settled at 1.0s
        assert 1.0 in timestamps
        # Resting state sampled at 3.0 - 0.10 = 2.90s
        assert 2.90 in timestamps

        req_rest = next(r for r in plan.sample_requests if r.requested_timestamp == 2.90)
        assert SamplePhase.RESTING in req_rest.sample_phases


class TestKeyframeSamplerComplexSequences:
    """Verifies adjacent, simultaneous, overlapping, and pause intervals."""

    def test_adjacent_events_without_gap(self):
        """Adjacent events where step1 end equals step2 start."""
        sampler = KeyframeSampler()

        a1 = ActionDeclaration(type=ActionType.ENTER, target_entity="e1", duration=1.0)
        a2 = ActionDeclaration(type=ActionType.ENTER, target_entity="e2", duration=1.0)

        s1 = ScheduledStep(
            step_id="s1",
            step_index=0,
            t_start=0.0,
            t_end=1.0,
            delta_t=1.0,
            actions=[
                ScheduledAction(
                    step_id="s1", action_index=0, target_entity="e1", action=a1,
                    start_time=0.0, end_time=1.0, duration=1.0
                )
            ],
        )
        s2 = ScheduledStep(
            step_id="s2",
            step_index=1,
            t_start=1.0,
            t_end=2.0,
            delta_t=1.0,
            actions=[
                ScheduledAction(
                    step_id="s2", action_index=0, target_entity="e2", action=a2,
                    start_time=1.0, end_time=2.0, duration=1.0
                )
            ],
        )
        timeline = ScheduledTimeline(scene_id="scene_adj", total_duration=2.5, steps=[s1, s2])

        plan = sampler.sample_timeline(timeline)
        # At boundary t=1.0: s1 SETTLED and s2 PRE_ACTION coalesce
        boundary_req = next(r for r in plan.sample_requests if abs(r.requested_timestamp - 1.0) <= 0.033)
        assert "s1" in boundary_req.associated_step_ids
        assert "s2" in boundary_req.associated_step_ids
        assert SamplePhase.SETTLED in boundary_req.sample_phases
        assert SamplePhase.PRE_ACTION in boundary_req.sample_phases

    def test_simultaneous_concurrent_events(self):
        """Concurrent actions sharing the same start time."""
        sampler = KeyframeSampler()

        a1 = ActionDeclaration(type=ActionType.MOVE, target_entity="node_a", duration=1.0)
        a2 = ActionDeclaration(type=ActionType.MOVE, target_entity="node_b", duration=1.0)

        step = ScheduledStep(
            step_id="step_concurrent",
            step_index=0,
            t_start=0.5,
            t_end=1.5,
            delta_t=1.0,
            is_concurrent=True,
            actions=[
                ScheduledAction(
                    step_id="step_concurrent", action_index=0, target_entity="node_a", action=a1,
                    start_time=0.5, end_time=1.5, duration=1.0, is_concurrent=True
                ),
                ScheduledAction(
                    step_id="step_concurrent", action_index=1, target_entity="node_b", action=a2,
                    start_time=0.5, end_time=1.5, duration=1.0, is_concurrent=True
                ),
            ],
        )
        timeline = ScheduledTimeline(scene_id="scene_conc", total_duration=2.0, steps=[step])

        plan = sampler.sample_timeline(timeline)
        req_start = next(r for r in plan.sample_requests if abs(r.requested_timestamp - 0.5) <= 0.033)
        assert "node_a" in req_start.associated_entity_ids
        assert "node_b" in req_start.associated_entity_ids
        assert len(req_start.event_refs) >= 2

    def test_pure_pause_interval_without_actions(self):
        """Step containing zero actions (pure pause) samples start and end."""
        sampler = KeyframeSampler()

        pause_step = ScheduledStep(
            step_id="step_pause",
            step_index=0,
            t_start=2.0,
            t_end=4.0,
            delta_t=2.0,
            actions=[],
        )
        timeline = ScheduledTimeline(scene_id="scene_pause", total_duration=5.0, steps=[pause_step])

        plan = sampler.sample_timeline(timeline)
        ts = plan.timestamps
        assert 2.0 in ts
        assert 4.0 in ts

    def test_scene_boundary_enforcement(self):
        """No sampled timestamp exceeds total scene duration or is negative."""
        sampler = KeyframeSampler()

        a1 = ActionDeclaration(type=ActionType.ENTER, target_entity="box", duration=1.0)
        step = ScheduledStep(
            step_id="s1",
            step_index=0,
            t_start=0.0,
            t_end=1.0,
            delta_t=1.0,
            actions=[
                ScheduledAction(
                    step_id="s1", action_index=0, target_entity="box", action=a1,
                    start_time=0.0, end_time=1.0, duration=1.0
                )
            ],
        )
        timeline = ScheduledTimeline(scene_id="scene_bounds", total_duration=10.0, steps=[step])

        plan = sampler.sample_timeline(timeline)
        for t in plan.timestamps:
            assert 0.0 <= t <= 10.0

        # Scene terminal sample should be near 10.0s (e.g. 9.95s)
        assert any(abs(t - 9.95) < 0.05 for t in plan.timestamps)


class TestKeyframeSamplerDeduplicationAndProvenance:
    """Verifies that clustering and deduplication correctly preserve references."""

    def test_deduplication_clusters_nearby_timestamps(self):
        """Adjacent candidate timestamps within 33ms are merged into one canonical request."""
        config = KeyframeSamplerConfig(dedup_tolerance_seconds=0.040)
        sampler = KeyframeSampler(config=config)

        # Two actions finishing 20ms apart
        a1 = ActionDeclaration(type=ActionType.ENTER, target_entity="e1", duration=1.0)
        a2 = ActionDeclaration(type=ActionType.ENTER, target_entity="e2", duration=1.020)

        step = ScheduledStep(
            step_id="s_clustered",
            step_index=0,
            t_start=0.0,
            t_end=1.020,
            delta_t=1.020,
            actions=[
                ScheduledAction(
                    step_id="s_clustered", action_index=0, target_entity="e1", action=a1,
                    start_time=0.0, end_time=1.0, duration=1.0
                ),
                ScheduledAction(
                    step_id="s_clustered", action_index=1, target_entity="e2", action=a2,
                    start_time=0.0, end_time=1.020, duration=1.020
                ),
            ],
        )
        timeline = ScheduledTimeline(scene_id="scene_cluster", total_duration=3.0, steps=[step])

        plan = sampler.sample_timeline(timeline)

        # There should only be ONE request for the settled states around ~1.0s, not two separate ones
        settled_reqs = [r for r in plan.sample_requests if 0.98 <= r.requested_timestamp <= 1.05]
        assert len(settled_reqs) == 1

        req = settled_reqs[0]
        # Both e1 and e2 must be preserved in associated entities
        assert "e1" in req.associated_entity_ids
        assert "e2" in req.associated_entity_ids
        assert len(req.event_refs) >= 2


class TestKeyframeSamplerValidationAndFailures:
    """Verifies rejection of non-numeric, NaN, Inf, negative, and invalid values."""

    def test_rejects_nan_and_infinite_timestamps_in_request(self):
        """KeyframeSampleRequest strictly rejects NaN and infinite timestamps."""
        with pytest.raises(ValueError, match="requested_timestamp must be finite"):
            KeyframeSampleRequest(requested_timestamp=float("nan"), scene_id="s1")

        with pytest.raises(ValueError, match="requested_timestamp must be finite"):
            KeyframeSampleRequest(requested_timestamp=float("inf"), scene_id="s1")

    def test_rejects_negative_timestamp_in_request(self):
        """KeyframeSampleRequest strictly rejects negative timestamps."""
        with pytest.raises(ValueError, match="requested_timestamp cannot be negative"):
            KeyframeSampleRequest(requested_timestamp=-1.5, scene_id="s1")

    def test_rejects_boolean_timestamps(self):
        """KeyframeSampleRequest strictly rejects boolean values."""
        with pytest.raises(ValueError, match="numeric value, not boolean"):
            KeyframeSampleRequest(requested_timestamp=True, scene_id="s1")  # type: ignore[arg-type]

    def test_rejects_invalid_scene_duration_in_sampler(self):
        """KeyframeSampler rejects negative or non-finite scene duration."""
        sampler = KeyframeSampler()
        timeline = ScheduledTimeline(scene_id="s1", total_duration=5.0)

        with pytest.raises(ValueError, match="cannot be negative"):
            sampler.sample_timeline(timeline, scene_duration=-5.0)

        with pytest.raises(ValueError, match="must be finite"):
            sampler.sample_timeline(timeline, scene_duration=float("nan"))

    def test_excessive_seek_deviation_diagnostic(self):
        """evaluate_timestamp_accuracy produces structured diagnostic when delta exceeds threshold."""
        config = KeyframeSamplerConfig(max_seek_deviation_seconds=0.050)
        sampler = KeyframeSampler(config=config)

        # Requested 5.0s, actual decoded 5.120s (deviation = 120ms > 50ms)
        evidence, diag = sampler.evaluate_timestamp_accuracy(requested_ts=5.0, actual_ts=5.120)

        assert not evidence.is_within_tolerance
        assert evidence.deviation_seconds == 0.120
        assert diag is not None
        assert diag.code == QCDiagnosticCode.QC_FRAME_CORRUPTION
        assert diag.level == QCDiagnosticLevel.WARNING
        assert "Excessive timestamp seek deviation" in diag.message

    def test_normal_seek_accuracy_passes(self):
        """Normal seek deviation within tolerance produces zero diagnostic."""
        config = KeyframeSamplerConfig(max_seek_deviation_seconds=0.100)
        sampler = KeyframeSampler(config=config)

        # Requested 2.0s, actual decoded 2.033s (delta = 33ms <= 100ms)
        evidence, diag = sampler.evaluate_timestamp_accuracy(requested_ts=2.0, actual_ts=2.033)

        assert evidence.is_within_tolerance
        assert diag is None


class TestKeyframeSamplerReproducibilityAndSerialization:
    """Verifies deterministic output and JSON serialization."""

    def test_deterministic_repeated_execution(self):
        """Repeated runs on identical timeline produce bit-for-bit identical sampling plans."""
        sampler = KeyframeSampler()

        a1 = ActionDeclaration(type=ActionType.MOVE, target_entity="e1", duration=0.8)
        a2 = ActionDeclaration(type=ActionType.HIGHLIGHT, target_entity="e2", duration=0.4)

        step = ScheduledStep(
            step_id="step_det",
            step_index=0,
            t_start=1.0,
            t_end=2.2,
            delta_t=1.2,
            actions=[
                ScheduledAction(
                    step_id="step_det", action_index=0, target_entity="e1", action=a1,
                    start_time=1.0, end_time=1.8, duration=0.8
                ),
                ScheduledAction(
                    step_id="step_det", action_index=1, target_entity="e2", action=a2,
                    start_time=1.8, end_time=2.2, duration=0.4
                ),
            ],
        )
        timeline = ScheduledTimeline(scene_id="scene_det", total_duration=5.0, steps=[step])

        plan1 = sampler.sample_timeline(timeline)
        plan2 = sampler.sample_timeline(timeline)

        assert plan1.model_dump() == plan2.model_dump()
        assert plan1.timestamps == plan2.timestamps

    def test_full_json_roundtrip(self):
        """KeyframeSamplingPlan serializes to JSON and round-trips cleanly."""
        sampler = KeyframeSampler()
        a = ActionDeclaration(type=ActionType.ENTER, target_entity="card", duration=0.5)
        step = ScheduledStep(
            step_id="step_json",
            step_index=0,
            t_start=0.0,
            t_end=1.0,
            delta_t=0.5,
            resting_pause=0.5,
            actions=[
                ScheduledAction(
                    step_id="step_json", action_index=0, target_entity="card", action=a,
                    start_time=0.0, end_time=0.5, duration=0.5
                )
            ],
        )
        timeline = ScheduledTimeline(scene_id="scene_json", total_duration=2.0, steps=[step])

        plan = sampler.sample_timeline(timeline)
        json_str = plan.model_dump_json()

        restored = KeyframeSamplingPlan.model_validate_json(json_str)
        assert restored.scene_id == plan.scene_id
        assert restored.timestamps == plan.timestamps
        assert len(restored.sample_requests) == len(plan.sample_requests)


class TestKeyframeSamplerRawEvents:
    """Verifies KeyframeSampler.sample_raw_events() handling of raw TimelineEvent inputs."""

    def test_sample_raw_events_single_event_with_actions(self):
        """Events with explicit actions correctly sum durations without syntax/runtime errors."""
        sampler = KeyframeSampler()
        a1 = ActionDeclaration(type=ActionType.ENTER, target_entity="box", duration=0.8)
        a2 = ActionDeclaration(type=ActionType.HIGHLIGHT, target_entity="box", duration=0.4)
        ev = TimelineEvent(step_id="step_raw_act", actions=[a1, a2])

        plan = sampler.sample_raw_events([ev], scene_duration=5.0, scene_id="scene_raw")

        assert plan.scene_id == "scene_raw"
        assert plan.count >= 3
        assert 0.0 in plan.timestamps
        # Settled state should be scheduled around 1.2s (0.8 + 0.4)
        assert any(abs(t - 1.2) < 0.05 for t in plan.timestamps)

    def test_sample_raw_events_empty_actions_fallback_duration(self):
        """TimelineEvent with empty action sequence defaults step duration to 1.0s."""
        sampler = KeyframeSampler()
        ev = TimelineEvent(step_id="step_empty_actions", actions=[])

        plan = sampler.sample_raw_events([ev], scene_duration=5.0, scene_id="scene_raw")

        assert plan.count >= 2
        assert 0.0 in plan.timestamps
        # 1.0s fallback duration
        assert any(abs(t - 1.0) < 0.05 for t in plan.timestamps)

    def test_sample_raw_events_with_narration_anchor(self):
        """TimelineEvent with narration anchor audio range uses audio range boundaries."""
        sampler = KeyframeSampler()
        anchor = NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(1.5, 4.0))
        ev = TimelineEvent(step_id="step_narrated", narration_anchor=anchor)

        plan = sampler.sample_raw_events([ev], scene_duration=6.0)

        assert any(abs(t - 1.5) < 0.05 for t in plan.timestamps)
        assert any(abs(t - 4.0) < 0.05 for t in plan.timestamps)

    def test_sample_raw_events_multi_event_sequence_monotonicity(self):
        """Sequential mixture of actioned, empty, and anchored events schedules monotonically."""
        sampler = KeyframeSampler()
        ev1 = TimelineEvent(
            step_id="step1",
            actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="e1", duration=1.0)],
        )
        ev2 = TimelineEvent(step_id="step2", actions=[])  # 1.0s default duration -> clock at 2.0s
        ev3 = TimelineEvent(
            step_id="step3",
            narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(2.5, 4.0)),
        )

        plan = sampler.sample_raw_events([ev1, ev2, ev3], scene_duration=5.0)

        assert plan.timestamps == sorted(plan.timestamps)
        assert len(plan.timestamps) == len(set(plan.timestamps))  # deduplicated
        assert plan.timestamps[0] == 0.0
        assert any(abs(t - 1.0) < 0.05 for t in plan.timestamps)
        assert any(abs(t - 2.0) < 0.05 for t in plan.timestamps)
        assert any(abs(t - 4.0) < 0.05 for t in plan.timestamps)

    def test_sample_raw_events_reproducibility(self):
        """Repeated sample_raw_events calls produce identical sampling plans."""
        sampler = KeyframeSampler()
        ev = TimelineEvent(
            step_id="step_rep",
            actions=[ActionDeclaration(type=ActionType.MOVE, target_entity="node", duration=0.7)],
        )

        plan1 = sampler.sample_raw_events([ev], scene_duration=4.0)
        plan2 = sampler.sample_raw_events([ev], scene_duration=4.0)

        assert plan1.model_dump() == plan2.model_dump()
        assert plan1.timestamps == plan2.timestamps


class TestKeyframeSamplerKeyframeExtraction:
    """Verifies KeyframeSampler.extract_keyframes() against video artifacts and error conditions."""

    @pytest.fixture
    def binary_search_video_path(self) -> Path:
        base_dir = Path(__file__).resolve().parent.parent
        p = base_dir / "CASES" / "artifacts" / "m5_3_4_2_binary_search_muxed.mp4"
        if not p.exists():
            pytest.skip(f"Binary search video artifact not found at {p}")
        return p

    def test_extract_keyframes_real_artifact_success(self, binary_search_video_path: Path):
        """Extracts valid video keyframes from real MP4 artifact without tuple unpacking error."""
        sampler = KeyframeSampler()
        reqs = [
            KeyframeSampleRequest(requested_timestamp=0.5, scene_id="binary_search"),
            KeyframeSampleRequest(requested_timestamp=1.0, scene_id="binary_search"),
            KeyframeSampleRequest(requested_timestamp=2.0, scene_id="binary_search"),
        ]
        plan = KeyframeSamplingPlan(
            scene_id="binary_search",
            total_duration=10.267,
            sample_requests=reqs,
        )

        results = sampler.extract_keyframes(str(binary_search_video_path), plan)

        assert len(results) == 3
        for res in results:
            assert res.frame_extracted is True
            assert res.frame_shape == (480, 854, 3)
            assert res.timestamp_evidence.is_within_tolerance is True
            assert res.timestamp_evidence.deviation_seconds <= 0.100
            assert len(res.diagnostics) == 0

    def test_extract_keyframes_excessive_seek_deviation_diagnostic(self, binary_search_video_path: Path):
        """Decoded timestamp deviating beyond strict tolerance emits WARNING diagnostic."""
        strict_config = KeyframeSamplerConfig(max_seek_deviation_seconds=0.001)
        sampler = KeyframeSampler(config=strict_config)
        # 1.25s at 15fps will decode to ~1.267s, creating ~0.017s deviation > 0.001s
        req = KeyframeSampleRequest(requested_timestamp=1.25, scene_id="binary_search")
        plan = KeyframeSamplingPlan(
            scene_id="binary_search",
            total_duration=10.267,
            sample_requests=[req],
        )

        results = sampler.extract_keyframes(str(binary_search_video_path), plan)

        assert len(results) == 1
        res = results[0]
        assert res.frame_extracted is True
        assert res.frame_shape == (480, 854, 3)
        assert res.timestamp_evidence.is_within_tolerance is False
        assert len(res.diagnostics) == 1
        diag = res.diagnostics[0]
        assert diag.code == QCDiagnosticCode.QC_FRAME_CORRUPTION
        assert diag.level == QCDiagnosticLevel.WARNING
        assert "Excessive timestamp seek deviation" in diag.message

    def test_extract_keyframes_nonexistent_video(self):
        """Extraction from non-existent video file emits structured ERROR diagnostic."""
        sampler = KeyframeSampler()
        req = KeyframeSampleRequest(requested_timestamp=1.0, scene_id="scene_fail")
        plan = KeyframeSamplingPlan(
            scene_id="scene_fail",
            total_duration=5.0,
            sample_requests=[req],
        )

        results = sampler.extract_keyframes("/nonexistent/path/missing_video.mp4", plan)

        assert len(results) == 1
        res = results[0]
        assert res.frame_extracted is False
        assert res.frame_shape is None
        assert res.timestamp_evidence.is_within_tolerance is False
        assert len(res.diagnostics) == 1
        diag = res.diagnostics[0]
        assert diag.code == QCDiagnosticCode.QC_FRAME_CORRUPTION
        assert diag.level == QCDiagnosticLevel.ERROR
        assert "Video file does not exist" in diag.message

    def test_extract_keyframes_empty_plan(self):
        """Empty KeyframeSamplingPlan produces an empty result list without errors."""
        sampler = KeyframeSampler()
        plan = KeyframeSamplingPlan(scene_id="scene_empty", total_duration=5.0, sample_requests=[])

        results = sampler.extract_keyframes("dummy.mp4", plan)
        assert results == []

    def test_extract_keyframes_corrupted_empty_file(self, tmp_path: Path):
        """Extraction from empty 0-byte video file records error diagnostic gracefully."""
        sampler = KeyframeSampler()
        empty_file = tmp_path / "empty_video.mp4"
        empty_file.write_bytes(b"")

        req = KeyframeSampleRequest(requested_timestamp=1.0, scene_id="scene_empty_file")
        plan = KeyframeSamplingPlan(
            scene_id="scene_empty_file",
            total_duration=5.0,
            sample_requests=[req],
        )

        results = sampler.extract_keyframes(str(empty_file), plan)

        assert len(results) == 1
        res = results[0]
        assert res.frame_extracted is False
        assert res.frame_shape is None
        assert len(res.diagnostics) == 1
        assert res.diagnostics[0].code == QCDiagnosticCode.QC_FRAME_CORRUPTION
        assert res.diagnostics[0].level == QCDiagnosticLevel.ERROR
        assert "empty (0 bytes)" in res.diagnostics[0].message
