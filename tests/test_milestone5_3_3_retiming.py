"""
CineLearn — Milestone 5.3.3 Test Suite
======================================
Audio-Authoritative Timeline Retiming, Pacing Hardening,
Subtitle Artifact Atomicity, and Alignment Status Verification.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest
from pydub import AudioSegment

from src.audio_manager import AudioManager, NativeSceneAudioSynthesizer
from src.compiler.compiler_models import CompilationErrorCode
from src.compiler.timeline_scheduler import TimelineScheduler
from src.ir.models import (
    ActionDeclaration,
    ActionType,
    AudioReference,
    EntityDeclaration,
    NarrationAnchor,
    SceneIR,
    TimelineEvent,
    VisualTier,
)
from src.timing_contract import AlignmentStatus, SceneTimingContract


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

def make_mock_tts_provider(durations_map: Optional[Dict[str, float]] = None, default_duration: float = 2.0):
    """Deterministic mock TTS provider generating silent MP3 files of exact duration."""
    call_records: List[Dict[str, Any]] = []

    def mock_provider(text: str, output_path: str, voice: str = "en-US-GuyNeural", rate: Optional[str] = "-4%") -> float:
        call_records.append({"text": text, "output_path": output_path, "voice": voice, "rate": rate})
        dur = default_duration
        if durations_map and text in durations_map:
            dur = durations_map[text]
        elif len(text.split()) > 0:
            dur = max(1.0, round(len(text.split()) * 0.4, 2))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        silence = AudioSegment.silent(duration=int(dur * 1000))
        silence.export(output_path, format="mp3")
        return dur

    mock_provider.call_records = call_records
    mock_provider.provider_name = "mock-tts"
    return mock_provider


def make_scene(
    scene_id: str = "TestScene",
    step_specs: Optional[List[Dict[str, Any]]] = None,
    duration_budget: float = 30.0,
    has_audio_reference: bool = True,
) -> SceneIR:
    """
    Constructs a native SceneIR with specified step definitions.
    step_specs format:
    [
        {
            "step_id": "step_1",
            "cue": "First concept narration cue.",
            "anchor_range": (0.0, 2.5),
            "action_durations": [1.5],
            "is_concurrent": False,
        },
        ...
    ]
    """
    if step_specs is None:
        step_specs = [
            {
                "step_id": "step_1",
                "cue": "First concept narration cue.",
                "anchor_range": (0.0, 3.0),
                "action_durations": [1.0],
                "is_concurrent": False,
            },
            {
                "step_id": "step_2",
                "cue": "Second concept narration cue.",
                "anchor_range": (3.5, 6.0),
                "action_durations": [2.0],
                "is_concurrent": False,
            },
        ]

    entities = [
        EntityDeclaration(
            id="card_1",
            component_type="ProcessCardVisual",
            viewport="dock_main",
            visual_tier=VisualTier.TIER1_REGISTERED,
        ),
        EntityDeclaration(
            id="card_2",
            component_type="ProcessCardVisual",
            viewport="dock_main",
            visual_tier=VisualTier.TIER1_REGISTERED,
        ),
    ]

    timeline: List[TimelineEvent] = []
    for idx, spec in enumerate(step_specs):
        s_id = spec.get("step_id", f"step_{idx+1}")
        cue = spec.get("cue")
        anchor_range = spec.get("anchor_range")
        action_durs = spec.get("action_durations", [1.0])
        is_concurrent = spec.get("is_concurrent", False)

        actions: List[ActionDeclaration] = []
        for a_idx, dur in enumerate(action_durs):
            params: Dict[str, Any] = {"label": f"act_{idx+1}_{a_idx+1}"}
            if is_concurrent:
                params["is_concurrent"] = True
            actions.append(
                ActionDeclaration(
                    type=ActionType.ENTER,
                    target_entity="card_1" if a_idx % 2 == 0 else "card_2",
                    duration=dur,
                    parameters=params,
                )
            )

        anchor = None
        if cue is not None and anchor_range is not None:
            anchor = NarrationAnchor(
                lecture_line_index=idx,
                text_cue=cue,
                audio_timestamp_range=anchor_range,
            )

        timeline.append(
            TimelineEvent(
                step_id=s_id,
                actions=actions,
                narration_anchor=anchor,
            )
        )

    audio_ref = None
    if has_audio_reference:
        audio_ref = AudioReference(
            audio_path="/tmp/mock_scene_audio.mp3",
            timestamps_path="/tmp/mock_scene_timestamps.json",
            total_duration=10.0,
            alignment_status="unaligned",
        )

    return SceneIR(
        scene_id=scene_id,
        duration_budget=duration_budget,
        entities=entities,
        timeline=timeline,
        audio_reference=audio_ref,
    )


# ============================================================================
# 1. Audio-Authoritative Retiming Core Tests
# ============================================================================

class TestAudioAuthoritativeRetiming:
    """Verifies invariant adherence for audio-authoritative timeline scheduling."""

    def test_audio_dominated_retiming_step_takes_audio_duration_with_resting_pause(self):
        """
        When D_audio >= D_visual:
        The step duration equals D_audio, visuals finish at t_start + D_visual,
        and resting_pause holds the scene for (D_audio - D_visual) seconds.
        """
        scheduler = TimelineScheduler()
        specs = [
            {
                "step_id": "step_1",
                "cue": "Long narration line.",
                "anchor_range": (0.0, 3.5),  # D_audio = 3.5s
                "action_durations": [1.5],   # D_visual = 1.5s
            }
        ]
        scene = make_scene(step_specs=specs, duration_budget=10.0)

        retimed_scene, timeline, diags = scheduler.retime_scene_timeline(
            scene=scene, inter_cue_pause=0.50, scene_end_pause=1.00
        )

        assert timeline.is_valid is True
        assert len(timeline.steps) == 1
        s0 = timeline.steps[0]
        assert s0.t_start == 0.0
        assert s0.t_end == 3.5
        assert s0.delta_t == 3.5
        assert s0.resting_pause == pytest.approx(2.0, abs=1e-3)
        assert len(s0.actions) == 1
        assert s0.actions[0].start_time == 0.0
        assert s0.actions[0].end_time == 1.5

        # Scene end pause (1.00s) added to total scheduled duration
        assert timeline.total_duration == pytest.approx(4.5, abs=1e-3)
        assert retimed_scene.audio_reference.alignment_status == "retimed"

    def test_visual_dominated_retiming_step_extends_to_visual_duration_downstream_shifts(self):
        """
        When D_visual > D_audio:
        The visual step extends to D_visual, speech is NOT cut off, resting pause is 0.0,
        and the downstream step shifts forward preserving sequential monotonicity.
        """
        scheduler = TimelineScheduler()
        specs = [
            {
                "step_id": "step_1",
                "cue": "Short narration cue.",
                "anchor_range": (0.0, 2.0),  # D_audio = 2.0s
                "action_durations": [2.0, 1.5],  # D_visual = 3.5s (sequential)
            },
            {
                "step_id": "step_2",
                "cue": "Second narration cue.",
                "anchor_range": (2.5, 4.5),  # D_audio = 2.0s
                "action_durations": [1.0],   # D_visual = 1.0s
            },
        ]
        scene = make_scene(step_specs=specs, duration_budget=15.0)

        retimed_scene, timeline, diags = scheduler.retime_scene_timeline(
            scene=scene, inter_cue_pause=0.50, scene_end_pause=1.00
        )

        assert timeline.is_valid is True
        assert len(timeline.steps) == 2

        s0 = timeline.steps[0]
        assert s0.t_start == 0.0
        # Step 0 extended to visual duration 3.5s
        assert s0.t_end == 3.5
        assert s0.delta_t == 3.5
        assert s0.resting_pause == 0.0
        # Audio speech timestamp range preserved at (0.0, 2.0)
        assert retimed_scene.timeline[0].narration_anchor.audio_timestamp_range == (0.0, 2.0)

        # Step 1 shifts to t_end(s0) + inter_cue_pause(0.50) = 4.0s
        s1 = timeline.steps[1]
        assert s1.t_start == 4.0
        # D_audio = 2.0s >= D_visual = 1.0s -> step takes 2.0s -> ends at 6.0s
        assert s1.t_end == 6.0
        assert s1.delta_t == 2.0
        assert s1.resting_pause == pytest.approx(1.0, abs=1e-3)
        assert s1.lead_in_pause == 0.50
        # Audio speech timestamp range retimed to (4.0, 6.0)
        assert retimed_scene.timeline[1].narration_anchor.audio_timestamp_range == (4.0, 6.0)

        # Total scheduled duration = 6.0 + 1.00 (scene_end_pause) = 7.0s
        assert timeline.total_duration == pytest.approx(7.0, abs=1e-3)
        assert retimed_scene.audio_reference.alignment_status == "retimed"

    def test_inter_cue_pause_enforced_between_sequential_cues(self):
        """Custom inter_cue_pause is strictly respected between sequential steps."""
        scheduler = TimelineScheduler()
        specs = [
            {"step_id": "s1", "cue": "Cue A", "anchor_range": (0.0, 2.0), "action_durations": [1.0]},
            {"step_id": "s2", "cue": "Cue B", "anchor_range": (2.5, 4.5), "action_durations": [1.0]},
        ]
        scene = make_scene(step_specs=specs, duration_budget=15.0)

        # Test with custom 0.75s pause
        _, timeline, _ = scheduler.retime_scene_timeline(scene=scene, inter_cue_pause=0.75, scene_end_pause=1.00)
        assert timeline.is_valid is True
        assert timeline.steps[0].t_end == 2.0
        assert timeline.steps[1].t_start == pytest.approx(2.75, abs=1e-3)
        assert timeline.steps[1].lead_in_pause == pytest.approx(0.75, abs=1e-3)

    def test_scene_end_pause_enforced_after_final_step(self):
        """Scene end pause holds terminal frame before transition."""
        scheduler = TimelineScheduler()
        specs = [
            {"step_id": "s1", "cue": "Solo Cue", "anchor_range": (0.0, 3.0), "action_durations": [1.0]},
        ]
        scene = make_scene(step_specs=specs, duration_budget=10.0)

        # Test with custom 2.50s scene end pause
        _, timeline, _ = scheduler.retime_scene_timeline(scene=scene, inter_cue_pause=0.50, scene_end_pause=2.50)
        assert timeline.is_valid is True
        assert timeline.steps[0].t_end == 3.0
        assert timeline.total_duration == pytest.approx(5.50, abs=1e-3)

    def test_strictly_monotonic_timestamps_across_all_steps(self):
        """Verifies strictly monotonic timeline ordering across multi-step scenes."""
        scheduler = TimelineScheduler()
        specs = [
            {"step_id": "s1", "cue": "Cue 1", "anchor_range": (0.0, 2.0), "action_durations": [3.0]}, # Visual dominates
            {"step_id": "s2", "cue": "Cue 2", "anchor_range": (2.0, 5.0), "action_durations": [1.0]}, # Audio dominates
            {"step_id": "s3", "cue": "Cue 3", "anchor_range": (5.0, 6.0), "action_durations": [2.5]}, # Visual dominates
            {"step_id": "s4", "cue": "Cue 4", "anchor_range": (6.0, 8.0), "action_durations": [1.0]}, # Audio dominates
        ]
        scene = make_scene(step_specs=specs, duration_budget=30.0)

        retimed_scene, timeline, _ = scheduler.retime_scene_timeline(scene=scene)
        assert timeline.is_valid is True
        assert len(timeline.steps) == 4

        for i in range(len(timeline.steps) - 1):
            curr_step = timeline.steps[i]
            next_step = timeline.steps[i + 1]
            assert next_step.t_start >= curr_step.t_end, f"Monotonicity violated between step {i} and {i+1}"
            assert next_step.t_start > curr_step.t_start

        # Check narration anchors on retimed_scene
        for i in range(len(retimed_scene.timeline) - 1):
            anchor_curr = retimed_scene.timeline[i].narration_anchor
            anchor_next = retimed_scene.timeline[i + 1].narration_anchor
            assert anchor_next.audio_timestamp_range[0] >= anchor_curr.audio_timestamp_range[1]

    def test_coincident_events_retiming(self):
        """Coincident events sharing anchor timestamps are scheduled concurrently and retimed cleanly."""
        scheduler = TimelineScheduler()
        entities = [
            EntityDeclaration(id="e1", component_type="Box", viewport="dock_main", visual_tier=VisualTier.TIER1_REGISTERED),
            EntityDeclaration(id="e2", component_type="Box", viewport="dock_main", visual_tier=VisualTier.TIER1_REGISTERED),
        ]
        timeline_events = [
            TimelineEvent(
                step_id="coinc_1",
                actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="e1", duration=2.0)],
                narration_anchor=NarrationAnchor(lecture_line_index=0, text_cue="Coincident cue", audio_timestamp_range=(0.0, 3.0)),
            ),
            TimelineEvent(
                step_id="coinc_2",
                actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="e2", duration=2.5)],
                narration_anchor=NarrationAnchor(lecture_line_index=0, text_cue="Coincident cue", audio_timestamp_range=(0.0, 3.0)),
            ),
        ]
        scene = SceneIR(scene_id="CoincidentScene", duration_budget=15.0, entities=entities, timeline=timeline_events)

        retimed_scene, timeline, diags = scheduler.retime_scene_timeline(scene)
        assert timeline.is_valid is True
        assert len(timeline.steps) == 1
        step = timeline.steps[0]
        assert step.is_concurrent is True
        # D_visual for concurrent = max(2.0, 2.5) = 2.5s. D_audio = 3.0s. delta_t = 3.0s.
        assert step.delta_t == 3.0
        assert step.resting_pause == pytest.approx(0.5, abs=1e-3)
        assert len(step.actions) == 2


# ============================================================================
# 2. Edge Cases and Unanchored Steps
# ============================================================================

class TestEdgeCasesAndSpecialPayloads:
    """Verifies unanchored steps, zero narration, and schedule delegation."""

    def test_zero_narration_empty_cues_fallback_to_visual_duration(self):
        """Scene with no narration anchors falls back to visual durations without errors."""
        scheduler = TimelineScheduler()
        specs = [
            {"step_id": "v1", "cue": None, "anchor_range": None, "action_durations": [1.5]},
            {"step_id": "v2", "cue": None, "anchor_range": None, "action_durations": [2.0]},
        ]
        scene = make_scene(step_specs=specs, duration_budget=10.0, has_audio_reference=False)

        retimed_scene, timeline, diags = scheduler.retime_scene_timeline(scene)
        assert timeline.is_valid is True
        assert len(timeline.steps) == 2
        assert timeline.steps[0].t_start == 0.0
        assert timeline.steps[0].t_end == 1.5
        assert timeline.steps[1].t_start == 1.5
        assert timeline.steps[1].t_end == 3.5
        assert timeline.total_duration == pytest.approx(4.5, abs=1e-3)

    def test_mixed_anchored_and_unanchored_events(self):
        """Unanchored intermediate visual action preserves its duration and shifts downstream anchored step."""
        scheduler = TimelineScheduler()
        specs = [
            {"step_id": "s1", "cue": "Narration 1", "anchor_range": (0.0, 2.0), "action_durations": [1.0]},
            {"step_id": "s2_visual", "cue": None, "anchor_range": None, "action_durations": [2.5]},
            {"step_id": "s3", "cue": "Narration 2", "anchor_range": (2.0, 4.0), "action_durations": [1.0]},
        ]
        scene = make_scene(step_specs=specs, duration_budget=20.0)

        retimed_scene, timeline, diags = scheduler.retime_scene_timeline(scene, inter_cue_pause=0.50)
        assert timeline.is_valid is True
        assert len(timeline.steps) == 3

        # Step 1: 0.0 to 2.0 (D_audio = 2.0)
        assert timeline.steps[0].t_end == 2.0
        # Step 2: unanchored, starts after s1 (with inter-cue pause since s1 had anchor)
        assert timeline.steps[1].t_start == 2.5
        assert timeline.steps[1].t_end == pytest.approx(5.0, abs=1e-3)
        # Step 3: anchored, starts at 5.0 + 0.50 = 5.5s
        assert timeline.steps[2].t_start == pytest.approx(5.5, abs=1e-3)
        assert timeline.steps[2].t_end == pytest.approx(7.5, abs=1e-3)

    def test_schedule_with_auto_retime_flag_delegates_to_retiming(self):
        """TimelineScheduler.schedule(scene, auto_retime=True) delegates cleanly to retiming."""
        scheduler = TimelineScheduler()
        specs = [
            {
                "step_id": "s1",
                "cue": "Short audio cue",
                "anchor_range": (0.0, 1.0),   # 1.0s audio
                "action_durations": [3.0],    # 3.0s visual (would overrun if not retimed)
            }
        ]
        scene = make_scene(step_specs=specs, duration_budget=10.0)

        # With auto_retime=True, step is retimed to 3.0s without budget overrun error
        timeline, diags = scheduler.schedule(scene, auto_retime=True)
        assert timeline.is_valid is True
        assert len(timeline.steps) == 1
        assert timeline.steps[0].t_end == 3.0

    def test_schedule_without_auto_retime_retains_strict_fail_closed_budget_overrun(self):
        """TimelineScheduler.schedule(scene, auto_retime=False) fails closed on step overrun."""
        scheduler = TimelineScheduler()
        specs = [
            {
                "step_id": "s1",
                "cue": "Short audio cue",
                "anchor_range": (0.0, 1.0),   # 1.0s audio budget
                "action_durations": [3.0],    # 3.0s visual duration exceeds available budget
            }
        ]
        scene = make_scene(step_specs=specs, duration_budget=10.0)

        # Without auto_retime (default False), strict M4C budget enforcement fails closed
        timeline, diags = scheduler.schedule(scene, auto_retime=False)
        assert timeline.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN for d in diags)


# ============================================================================
# 3. Budget and Fail-Closed Hardening Tests
# ============================================================================

class TestBudgetAndFailClosedHardening:
    """Verifies fail-closed behavior on macro budget overrun and invalid parameters."""

    def test_macro_duration_budget_overrun_fails_closed(self):
        """When retimed scene duration exceeds scene.duration_budget, fails closed with diagnostic."""
        scheduler = TimelineScheduler()
        specs = [
            {"step_id": "s1", "cue": "Cue 1", "anchor_range": (0.0, 6.0), "action_durations": [2.0]},
            {"step_id": "s2", "cue": "Cue 2", "anchor_range": (6.0, 12.0), "action_durations": [2.0]},
        ]
        # Scene budget is 10.0s, but retimed duration = 12.0s + 0.50s (pause) + 1.00s (end) = 13.5s
        scene = make_scene(step_specs=specs, duration_budget=10.0)

        retimed_scene, timeline, diags = scheduler.retime_scene_timeline(scene)
        assert timeline.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN for d in diags)
        assert len(timeline.steps) == 0

    def test_invalid_action_duration_fails_closed_without_retiming(self):
        """Action duration below 0.100s or non-numeric fails closed."""
        scheduler = TimelineScheduler()
        specs = [
            {"step_id": "s1", "cue": "Cue", "anchor_range": (0.0, 3.0), "action_durations": [0.05]}, # < 0.100s
        ]
        scene = make_scene(step_specs=specs, duration_budget=10.0)

        retimed_scene, timeline, diags = scheduler.retime_scene_timeline(scene)
        assert timeline.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER for d in diags)
        # Alignment status must NOT be updated to retimed on failure
        assert retimed_scene.audio_reference.alignment_status == "unaligned"

    def test_invalid_anchor_range_fails_closed(self):
        """Pydantic model and scheduler fail closed on negative or inverted anchor timestamp ranges."""
        with pytest.raises(Exception):
            NarrationAnchor(lecture_line_index=0, text_cue="Invalid cue", audio_timestamp_range=(4.0, 2.0))
        with pytest.raises(Exception):
            NarrationAnchor(lecture_line_index=0, text_cue="Invalid cue", audio_timestamp_range=(-1.0, 2.0))

    def test_duplicate_entity_id_fails_closed(self):
        """Duplicate entity declaration fails closed."""
        scheduler = TimelineScheduler()
        entities = [
            EntityDeclaration(id="dupe", component_type="Card", viewport="dock_main", visual_tier=VisualTier.TIER1_REGISTERED),
            EntityDeclaration(id="dupe", component_type="Card", viewport="dock_main", visual_tier=VisualTier.TIER1_REGISTERED),
        ]
        scene = SceneIR(scene_id="DupeScene", duration_budget=10.0, entities=entities, timeline=[])

        _, timeline_res, diags = scheduler.retime_scene_timeline(scene)
        assert timeline_res.is_valid is False
        assert any(d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER for d in diags)


# ============================================================================
# 4. Alignment Status Semantics Tests
# ============================================================================

class TestAlignmentStatusSemantics:
    """Verifies alignment_status lifecycle states and contract roundtrip."""

    def test_audio_reference_alignment_status_field_and_validation(self):
        """AudioReference accepts valid alignment status values and rejects invalid ones."""
        ref = AudioReference(
            audio_path="/tmp/test.mp3",
            timestamps_path="/tmp/test.json",
            total_duration=5.0,
            alignment_status="unaligned",
        )
        assert ref.alignment_status == "unaligned"

        ref_retimed = AudioReference(
            audio_path="/tmp/test.mp3",
            timestamps_path="/tmp/test.json",
            total_duration=5.0,
            alignment_status="retimed",
        )
        assert ref_retimed.alignment_status == "retimed"

        ref_muxed = AudioReference(
            audio_path="/tmp/test.mp3",
            timestamps_path="/tmp/test.json",
            total_duration=5.0,
            alignment_status="muxed",
        )
        assert ref_muxed.alignment_status == "muxed"

        with pytest.raises(ValueError):
            AudioReference(
                audio_path="/tmp/test.mp3",
                timestamps_path="/tmp/test.json",
                total_duration=5.0,
                alignment_status="invalid_state",
            )

    def test_scene_timing_contract_alignment_status_serialization_roundtrip(self):
        """SceneTimingContract serializes and deserializes alignment_status losslessly."""
        contract = SceneTimingContract(
            section_id="sec_1",
            alignment_status=AlignmentStatus.RETIMED,
            total_audio_duration=4.5,
        )
        data = contract.to_dict()
        assert data["alignment_status"] == "retimed"

        restored = SceneTimingContract.from_dict(data)
        assert restored.alignment_status == AlignmentStatus.RETIMED

    def test_from_scene_ir_propagates_alignment_status(self):
        """SceneTimingContract.from_scene_ir accurately copies alignment_status."""
        specs = [{"step_id": "s1", "cue": "Beat", "anchor_range": (0.0, 2.0), "action_durations": [1.0]}]
        scene = make_scene(step_specs=specs, duration_budget=10.0)

        contract_unaligned = SceneTimingContract.from_scene_ir(scene)
        assert contract_unaligned.alignment_status == AlignmentStatus.UNALIGNED

        # Retime scene
        scheduler = TimelineScheduler()
        retimed_scene, timeline, _ = scheduler.retime_scene_timeline(scene)
        assert retimed_scene.audio_reference.alignment_status == "retimed"

        contract_retimed = SceneTimingContract.from_scene_ir(retimed_scene)
        assert contract_retimed.alignment_status == AlignmentStatus.RETIMED

    def test_alignment_status_remains_unaligned_on_retiming_failure(self):
        """When retiming fails, audio_reference.alignment_status remains unaligned."""
        scheduler = TimelineScheduler()
        specs = [{"step_id": "s1", "cue": "Cue", "anchor_range": (0.0, 10.0), "action_durations": [1.0]}]
        scene = make_scene(step_specs=specs, duration_budget=5.0) # Overrun

        retimed_scene, timeline, diags = scheduler.retime_scene_timeline(scene)
        assert timeline.is_valid is False
        assert retimed_scene.audio_reference.alignment_status == "unaligned"


# ============================================================================
# 5. Subtitle Artifact Hardening and Cache Tests
# ============================================================================

class TestSubtitleArtifactHardeningAndCache:
    """Verifies atomic SRT generation, cache validation, and corrupt artifact purge."""

    def test_atomic_srt_export_creates_file_without_temporary_artifacts(self, tmp_path):
        """AudioManager.export_srt writes atomically via PID-staged tmp file."""
        srt_file = tmp_path / "scene_output.srt"
        lines = [
            {"index": 1, "start": 0.0, "end": 2.5, "caption": "Hello world", "raw_text": "Hello world"},
            {"index": 2, "start": 3.0, "end": 5.0, "caption": "Second subtitle line", "raw_text": "Second subtitle line"},
        ]

        AudioManager.export_srt(lines, str(srt_file))
        assert srt_file.exists()
        assert srt_file.stat().st_size > 0

        content = srt_file.read_text(encoding="utf-8")
        assert "Hello world" in content
        assert "Second subtitle line" in content

        # Check no temporary files were left behind
        tmp_files = list(tmp_path.glob("*.tmp.*"))
        assert len(tmp_files) == 0

    def test_atomic_srt_export_cleans_up_on_failure(self, tmp_path):
        """If an error occurs during export, the temporary file is purged."""
        srt_file = tmp_path / "broken.srt"
        invalid_lines = [{"start": "not_a_float", "end": None}]

        with pytest.raises(Exception):
            AudioManager.export_srt(invalid_lines, str(srt_file))

        assert not srt_file.exists()
        tmp_files = list(tmp_path.glob("*.tmp.*"))
        assert len(tmp_files) == 0

    def test_cache_invalidation_when_srt_file_is_missing(self, tmp_path):
        """NativeSceneAudioSynthesizer treats missing .srt artifact as cache miss and purges stale files."""
        out_dir = tmp_path / "audio_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        provider = make_mock_tts_provider(default_duration=2.0)
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=provider)

        specs = [{"step_id": "s1", "cue": "Unique test narration line for cache.", "anchor_range": (0.0, 0.0), "action_durations": [1.0]}]
        scene = make_scene(scene_id="CacheSceneMissingSRT", step_specs=specs, has_audio_reference=False)

        # 1. First run generates artifacts (.mp3, .json, .srt)
        synthesizer.synthesize_scene_ir(scene, output_dir=out_dir)
        srt_path = out_dir / "CacheSceneMissingSRT.srt"
        mp3_path = out_dir / "CacheSceneMissingSRT_audio.mp3"
        json_path = out_dir / "CacheSceneMissingSRT_timestamps.json"
        assert srt_path.exists()
        assert mp3_path.exists()
        assert json_path.exists()

        # 2. Simulate cache hit
        scene2 = make_scene(scene_id="CacheSceneMissingSRT", step_specs=specs, has_audio_reference=False)
        _, meta_hit = synthesizer.synthesize_scene_ir(scene2, output_dir=out_dir)
        assert meta_hit["cache_hit"] is True

        # 3. Delete the SRT artifact
        srt_path.unlink()
        assert not srt_path.exists()

        # 4. Synthesizer must detect cache miss and regenerate all artifacts including SRT
        scene3 = make_scene(scene_id="CacheSceneMissingSRT", step_specs=specs, has_audio_reference=False)
        _, meta_regen = synthesizer.synthesize_scene_ir(scene3, output_dir=out_dir)
        assert meta_regen["cache_hit"] is False
        assert srt_path.exists()
        assert srt_path.stat().st_size > 0

    def test_cache_invalidation_when_srt_file_is_empty_or_corrupt(self, tmp_path):
        """NativeSceneAudioSynthesizer treats zero-byte .srt as cache miss."""
        out_dir = tmp_path / "audio_cache_corrupt"
        out_dir.mkdir(parents=True, exist_ok=True)
        provider = make_mock_tts_provider(default_duration=2.0)
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=provider)

        specs = [{"step_id": "s1", "cue": "Another unique test line for corruption.", "anchor_range": (0.0, 0.0), "action_durations": [1.0]}]
        scene = make_scene(scene_id="CacheSceneCorruptSRT", step_specs=specs, has_audio_reference=False)

        # 1. Synthesize initially
        synthesizer.synthesize_scene_ir(scene, output_dir=out_dir)
        srt_path = out_dir / "CacheSceneCorruptSRT.srt"
        assert srt_path.exists()

        # 2. Corrupt SRT by truncating to 0 bytes
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("")
        assert srt_path.stat().st_size == 0

        # 3. Next call must detect invalid SRT and trigger cache miss
        scene2 = make_scene(scene_id="CacheSceneCorruptSRT", step_specs=specs, has_audio_reference=False)
        _, meta_miss = synthesizer.synthesize_scene_ir(scene2, output_dir=out_dir)
        assert meta_miss["cache_hit"] is False
        assert srt_path.exists()
        assert srt_path.stat().st_size > 0
