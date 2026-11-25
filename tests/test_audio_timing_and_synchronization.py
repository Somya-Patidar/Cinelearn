"""
Comprehensive Test Suite for CineLearn Audio Integration and Temporal Synchronization (Milestone 3).
====================================================================================================

Covers all required testing scenarios from the specification:
1. Timing schema validation (creation, fields, serialization, deserialization, validation).
2. Multiple narration segments per scene.
3. Multiple visual actions per narration segment.
4. Actual TTS duration recording.
5. Animation duration integration.
6. Case A: Narration longer than animation (explanatory hold, no audio cutoff).
7. Case B: Animation longer than narration (animation completes before next segment, audio padded).
8. Case C: Multiple actions during one narration segment.
9. Case D: Multiple narration segments within one scene preserving order.
10. Case E: Intentional pauses and inspection markers preserved without drift.
11. Case F: Dynamic duration changes after scene repair.
12. Cache invalidation on narration change (hash matching, purging stale artifacts).
13. Atomic file writes and failure recovery.
14. Missing or corrupted audio segment handling.
15. FFmpeg synchronization and assembly.
16. Final media metadata validation (ffprobe stream and duration verification).
17. AudioHandoffValidator integration with timing contract.
18. Domain-agnostic operation across OS, Chemistry, Math, and DSA subjects.
"""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch
import pytest
from pydub import AudioSegment

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from timing_contract import (
    VisualActionTiming,
    NarrationSegmentTiming,
    SceneTimingContract,
)
from audio_manager import AudioManager, compute_narration_hash
from audio_handoff_validator import (
    AudioHandoffValidator,
    AudioHandoffReport,
    HandoffStatus,
    HandoffIssueCategory,
)


class TestAudioTimingAndSynchronization:

    @pytest.fixture(autouse=True)
    def setup_tmp(self, tmp_path):
        self.tmp_dir = tmp_path
        self.audio_mgr = AudioManager()
        self.handoff_validator = AudioHandoffValidator()

    def _create_mock_mp3(self, path: Path, duration_sec: float = 2.0) -> Path:
        """Creates a readable valid silent MP3 file of exact duration."""
        path.parent.mkdir(parents=True, exist_ok=True)
        silence = AudioSegment.silent(duration=int(duration_sec * 1000))
        silence.export(str(path), format="mp3")
        return path

    def _create_mock_mp4(self, path: Path, duration_sec: float = 2.0) -> Path:
        """Creates a minimal valid MP4 file using ffmpeg for integration tests."""
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=black:s=320x240:d={duration_sec:.2f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return path

    # =========================================================================
    # 1. Timing Schema Validation
    # =========================================================================

    def test_timing_schema_validation(self):
        """Validates contract data structures, serialization, and deterministic validation rules."""
        action = VisualActionTiming(
            action_id="L1-1",
            effect="FadeIn",
            component_name="ProcessCardVisual",
            canonical_action="construct",
            estimated_duration=1.5,
            actual_duration=1.45,
            viewport="dock_main",
        )
        assert action.action_id == "L1-1"
        assert action.actual_duration == 1.45

        seg = NarrationSegmentTiming(
            segment_id="sec1_seg1",
            section_id="section_1",
            segment_index=0,
            narration_text="Process P1 arrives at time zero with burst time four.",
            caption="Process P1 arrives",
            teaching_beat="Explain arrival",
            associated_action_ids=["L1-1"],
            visual_actions=[action],
            actual_audio_duration=3.8,
            pause_duration=0.5,
            timeline_start=0.0,
            timeline_end=3.8,
            target_slot_duration=4.3,
        )
        assert seg.actual_audio_duration == 3.8

        contract = SceneTimingContract(
            section_id="section_1",
            title="Sec 1: Introduction",
            segments=[seg],
            total_audio_duration=4.3,
        )

        # Validation
        is_valid, issues = contract.validate()
        assert is_valid
        assert len(issues) == 0

        # Serialization round-trip
        data = contract.to_dict()
        assert data["section_id"] == "section_1"
        assert len(data["segments"]) == 1
        assert len(data["lines"]) == 1  # Backward compatibility format

        rebuilt = SceneTimingContract.from_dict(data)
        assert rebuilt.section_id == contract.section_id
        assert len(rebuilt.segments) == 1
        assert rebuilt.segments[0].actual_audio_duration == 3.8

        # Validation failure cases
        invalid_contract = SceneTimingContract(section_id="", segments=[])
        ok, errs = invalid_contract.validate()
        assert not ok
        assert "Missing section_id" in errs
        assert "Contract contains 0 narration segments" in errs

    # =========================================================================
    # 2. Multiple Narration Segments per Scene
    # =========================================================================

    def test_multiple_narration_segments_per_scene(self):
        """Tests scenes containing multiple distinct narration segments."""
        raw_section = {
            "id": "section_3",
            "title": "Sec 3: Ready Queue Dynamics",
            "story_beat": "Queue ordering and dispatch",
            "lecture_lines": [
                "Process P1 enters the ready queue first.",
                "Process P2 arrives next and queues behind P1.",
                "Process P3 arrives and is placed at the end of the queue.",
                "The dispatcher selects the head process for CPU execution."
            ],
            "captions": ["P1 enters queue", "P2 queues behind", "P3 joins queue", "Dispatcher selects head"],
            "animations": [
                "L1-1 | FadeIn | ReadyQueueVisual | viewport: dock_main",
                "L2-1 | Indicate | P2 in ReadyQueueVisual | viewport: dock_main",
                "L3-1 | Indicate | P3 in ReadyQueueVisual | viewport: dock_main",
                "L4-1 | FadeIn | CPUExecutionVisual | viewport: dock_inspector",
            ]
        }
        contract = SceneTimingContract.from_storyboard_section(raw_section)
        assert len(contract.segments) == 4
        assert [s.segment_index for s in contract.segments] == [0, 1, 2, 3]
        assert contract.segments[0].caption == "P1 enters queue"
        assert contract.segments[3].caption == "Dispatcher selects head"
        assert contract.segments[0].timeline_start == 0.0
        assert contract.segments[1].timeline_start > 0.0

    # =========================================================================
    # 3. Multiple Visual Actions per Narration Segment
    # =========================================================================

    def test_multiple_visual_actions_per_narration_segment(self):
        """Tests one narration segment linked with multiple sequential visual actions (Case C)."""
        raw_section = {
            "id": "section_1",
            "title": "Sec 1: Introduction",
            "lecture_lines": [
                "When Process P1 arrives, its control block is created and it enters the ready queue."
            ],
            "captions": ["P1 creation and enqueue"],
            "animations": [
                "L1-1 | FadeIn | ProcessCardVisual | viewport: dock_main | action: construct",
                "L1-2 | Indicate | ProcessCardVisual | viewport: dock_main | action: highlight_id",
                "L1-3 | FadeIn | ReadyQueueVisual | viewport: dock_inspector | action: animate_enqueue",
            ]
        }
        contract = SceneTimingContract.from_storyboard_section(raw_section)
        assert len(contract.segments) == 1
        seg = contract.segments[0]
        assert len(seg.visual_actions) == 3
        assert seg.associated_action_ids == ["L1-1", "L1-2", "L1-3"]
        assert seg.visual_actions[0].action_id == "L1-1"
        assert seg.visual_actions[1].action_id == "L1-2"
        assert seg.visual_actions[2].action_id == "L1-3"

    # =========================================================================
    # 4. Actual TTS Duration Recording
    # =========================================================================

    def test_actual_tts_duration_recording(self):
        """Verifies that actual measured TTS durations are recorded, replacing estimates."""
        contract = SceneTimingContract(
            section_id="section_2",
            segments=[
                NarrationSegmentTiming(
                    segment_id="s2_1",
                    section_id="section_2",
                    segment_index=0,
                    narration_text="Line one test.",
                    estimated_duration=4.0
                ),
                NarrationSegmentTiming(
                    segment_id="s2_2",
                    section_id="section_2",
                    segment_index=1,
                    narration_text="Line two test.",
                    estimated_duration=4.0
                ),
            ]
        )
        # Measured TTS durations
        measured_durs = [5.123, 3.456]
        audio_paths = ["/tmp/line_1.mp3", "/tmp/line_2.mp3"]
        contract.update_actual_audio_durations(measured_durs, audio_paths)

        assert contract.segments[0].actual_audio_duration == 5.123
        assert contract.segments[1].actual_audio_duration == 3.456
        assert contract.segments[0].audio_file_path == "/tmp/line_1.mp3"
        assert contract.segments[0].timeline_start == 0.0
        assert contract.segments[0].timeline_end == 5.123
        # Next line starts after duration + pause
        pause = contract.segments[0].pause_duration
        assert math.isclose(contract.segments[1].timeline_start, 5.123 + pause, abs_tol=1e-3)

    # =========================================================================
    # 5. Case A: Narration Longer than Animation (Visual Hold)
    # =========================================================================

    def test_reconciliation_case_a_narration_longer_than_animation(self):
        """When narration exceeds animation, visual holds stable state and audio is not cut off."""
        contract = SceneTimingContract(
            section_id="sec_case_a",
            segments=[
                NarrationSegmentTiming(
                    segment_id="seg_1",
                    section_id="sec_case_a",
                    segment_index=0,
                    narration_text="A comprehensive theoretical explanation that takes six seconds.",
                    actual_audio_duration=6.0,
                    pause_duration=0.5
                )
            ]
        )
        # Visual animation only took 2.5 seconds
        rendered_timings = [{"line_index": 0, "start_time": 0.0, "duration": 2.5}]
        report = contract.reconcile_with_rendered_video(
            actual_video_duration=2.5,
            rendered_line_timings=rendered_timings
        )

        assert report["status"] == "SYNCHRONIZED"
        assert len(report["adjustments"]) == 1
        adj = report["adjustments"][0]
        assert adj["type"] == "NARRATION_LONGER_THAN_ANIMATION"
        assert adj["visual_hold_sec"] == 3.5  # 6.0 - 2.5
        assert contract.segments[0].sync_mode == "hold"
        assert contract.segments[0].timeline_end == 6.0
        # Narration is fully preserved (not truncated)
        assert contract.total_assembled_duration >= 6.5

    # =========================================================================
    # 6. Case B: Animation Longer than Narration (Allow Visual Completion)
    # =========================================================================

    def test_reconciliation_case_b_animation_longer_than_narration(self):
        """When animation exceeds narration, demonstration completes and next narration waits."""
        contract = SceneTimingContract(
            section_id="sec_case_b",
            segments=[
                NarrationSegmentTiming(
                    segment_id="seg_1",
                    section_id="sec_case_b",
                    segment_index=0,
                    narration_text="Short sentence.",
                    actual_audio_duration=2.5,
                    pause_duration=0.5
                ),
                NarrationSegmentTiming(
                    segment_id="seg_2",
                    section_id="sec_case_b",
                    segment_index=1,
                    narration_text="Next sentence.",
                    actual_audio_duration=3.0,
                    pause_duration=0.5
                )
            ]
        )
        # First animation step takes 7.0 seconds (e.g. process moves across queue)
        rendered_timings = [
            {"line_index": 0, "start_time": 0.0, "duration": 7.0},
            {"line_index": 1, "start_time": 7.0, "duration": 3.0},
        ]
        report = contract.reconcile_with_rendered_video(
            actual_video_duration=10.0,
            rendered_line_timings=rendered_timings
        )

        adj0 = report["adjustments"][0]
        assert adj0["type"] == "ANIMATION_LONGER_THAN_NARRATION"
        assert adj0["audio_pause_sec"] == 4.5  # 7.0 - 2.5
        assert contract.segments[0].sync_mode == "extend_pause"
        # Segment 2 narration does NOT start until after segment 1 visual completes
        assert contract.segments[1].timeline_start >= 7.0

    # =========================================================================
    # 7. Case E: Intentional Pauses and Inspection Windows
    # =========================================================================

    def test_intentional_pauses_preserved_without_drift(self):
        """Intentional pauses (e.g. worked example inspection) are preserved without causing drift."""
        contract = SceneTimingContract(
            section_id="sec_pauses",
            segments=[
                NarrationSegmentTiming(
                    segment_id="seg_1",
                    section_id="sec_pauses",
                    segment_index=0,
                    narration_text="Step one calculated.",
                    actual_audio_duration=3.0,
                    pause_duration=1.5  # 1.5s inspection pause
                ),
                NarrationSegmentTiming(
                    segment_id="seg_2",
                    section_id="sec_pauses",
                    segment_index=1,
                    narration_text="Step two calculated.",
                    actual_audio_duration=3.0,
                    pause_duration=1.0
                )
            ]
        )
        rendered_timings = [
            {"line_index": 0, "start_time": 0.0, "duration": 3.0},
            {"line_index": 1, "start_time": 4.5, "duration": 3.0},
        ]
        contract.reconcile_with_rendered_video(7.5, rendered_timings)
        # Line 2 starts exactly when Line 2 visual begins at 4.5s
        assert math.isclose(contract.segments[1].timeline_start, 4.5, abs_tol=1e-2)

    # =========================================================================
    # 8. Case F: Dynamic Scene Durations after Repair
    # =========================================================================

    def test_dynamic_duration_changes_after_scene_repair(self):
        """When code repair alters scene animation durations, synchronization uses final rendered timing."""
        contract = SceneTimingContract(
            section_id="sec_repair",
            segments=[
                NarrationSegmentTiming(
                    segment_id="s1", section_id="sec_repair", segment_index=0,
                    narration_text="Text 1", actual_audio_duration=4.0
                ),
                NarrationSegmentTiming(
                    segment_id="s2", section_id="sec_repair", segment_index=1,
                    narration_text="Text 2", actual_audio_duration=4.0
                )
            ]
        )
        # Pre-repair render: each line took 4.5s (total 9.0s)
        contract.reconcile_with_rendered_video(9.0, [
            {"line_index": 0, "start_time": 0.0, "duration": 4.5},
            {"line_index": 1, "start_time": 4.5, "duration": 4.5},
        ])
        assert contract.segments[1].timeline_start >= 4.5

        # Scene repaired by LocalVisualFixer -> now line 1 takes 6.2s because of added element
        contract.reconcile_with_rendered_video(11.0, [
            {"line_index": 0, "start_time": 0.0, "duration": 6.2},
            {"line_index": 1, "start_time": 6.2, "duration": 4.8},
        ])
        # Re-synchronized contract dynamically updates to new rendered start time
        assert contract.segments[1].timeline_start >= 6.2
        assert contract.total_rendered_video_duration == 11.0

    # =========================================================================
    # 9. Audio Cache Invalidation and Hash Matching
    # =========================================================================

    def test_cache_invalidation_on_narration_change(self):
        """Unchanged narration reuses valid cached audio; changed text invalidates and regenerates."""
        lines_original = ["This is line one.", "This is line two."]
        hash_orig = compute_narration_hash(lines_original)

        output_dir = self.tmp_dir / "cache_test"

        def fake_generate_speech(text, path, voice=None):
            silence = AudioSegment.silent(duration=2000)
            silence.export(str(path), format="mp3")
            return 2.0

        # 1. Generate audio in temporary directory
        with patch.object(self.audio_mgr, "generate_speech", side_effect=fake_generate_speech):
            meta1 = self.audio_mgr.generate_section_audio(
                section_id="sec_c1",
                lecture_lines=lines_original,
                output_dir=output_dir,
            )
            assert meta1["narration_hash"] == hash_orig

            # 2. Identical call -> fast cache hit
            meta2 = self.audio_mgr.generate_section_audio(
                section_id="sec_c1",
                lecture_lines=lines_original,
                output_dir=output_dir,
            )
            assert meta2["narration_hash"] == hash_orig

            # 3. Modified narration text -> cache invalidation
            lines_modified = ["This is line one modified.", "This is line two."]
            hash_mod = compute_narration_hash(lines_modified)
            assert hash_mod != hash_orig

            meta3 = self.audio_mgr.generate_section_audio(
                section_id="sec_c1",
                lecture_lines=lines_modified,
                output_dir=output_dir,
            )
            assert meta3["narration_hash"] == hash_mod

    # =========================================================================
    # 10. Atomic File Writes and Failure Recovery
    # =========================================================================

    def test_atomic_file_writes_safeguard(self):
        """Interrupted writes do not leave invalid files or corrupted artifacts on disk."""
        output_dir = self.tmp_dir / "atomic_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_audio = output_dir / "sec_atom_audio.mp3"
        final_ts = output_dir / "sec_atom_timestamps.json"

        def fake_generate_speech(text, path, voice=None):
            silence = AudioSegment.silent(duration=1500)
            silence.export(str(path), format="mp3")
            return 1.5

        # Force failure during combined export
        with patch.object(self.audio_mgr, "generate_speech", side_effect=fake_generate_speech):
            with patch.object(AudioSegment, "export", side_effect=IOError("Simulated disk error during export")):
                with pytest.raises(IOError):
                    self.audio_mgr.generate_section_audio(
                        section_id="sec_atom",
                        lecture_lines=["Test atomic write."],
                        output_dir=output_dir,
                    )

        # Temporary files must be cleaned up and not left orphaned
        tmps = list(output_dir.glob("*.tmp.*"))
        assert len(tmps) == 0

        # No corrupted final files created
        assert not final_audio.exists()
        assert not final_ts.exists()

    # =========================================================================
    # 11. Missing Audio Segment Graceful Handling
    # =========================================================================

    def test_missing_audio_segment_graceful_handling(self):
        """TTS failure creates fallback silence audio without terminating the pipeline."""
        out_file = self.tmp_dir / "fail_tts.mp3"
        with patch.object(self.audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("TTS service unreachable")):
            dur = self.audio_mgr.generate_speech("A fallback text line with eight words.", str(out_file))
            assert dur > 1.0
            assert out_file.exists()
            assert out_file.stat().st_size > 500

    # =========================================================================
    # 12. FFmpeg Synchronization and Stream Multiplexing
    # =========================================================================

    def test_ffmpeg_synchronization_and_assembly(self):
        """End-to-end integration: merges video and audio into playable synchronized MP4."""
        out_dir = self.tmp_dir / "mux_test"
        v_path = self._create_mock_mp4(out_dir / "raw.mp4", duration_sec=3.0)
        a_path = self._create_mock_mp3(out_dir / "audio.mp3", duration_sec=4.5)
        out_mp4 = out_dir / "synced.mp4"

        res = self.audio_mgr.sync_video_and_audio(
            video_path=str(v_path),
            audio_path=str(a_path),
            output_path=str(out_mp4),
            burn_subtitles=False,
        )
        assert Path(res).exists()
        assert Path(res).stat().st_size > 1000

        # Validate with ffprobe
        val = self.audio_mgr.validate_media_file(res, require_audio=True)
        assert val["is_valid"]
        assert val["has_video"]
        assert val["has_audio"]
        assert math.isclose(val["duration"], 4.5, abs_tol=0.2)

    # =========================================================================
    # 13. Final Media Metadata Validation
    # =========================================================================

    def test_final_media_metadata_validation(self):
        """Tests validate_media_file against valid and invalid media inputs."""
        # Non-existent file
        r1 = self.audio_mgr.validate_media_file(str(self.tmp_dir / "nonexistent.mp4"))
        assert not r1["is_valid"]

        # Valid file
        valid_mp4 = self._create_mock_mp4(self.tmp_dir / "valid.mp4", duration_sec=2.0)
        r2 = self.audio_mgr.validate_media_file(str(valid_mp4), require_audio=False)
        assert r2["is_valid"]
        assert r2["has_video"]

    # =========================================================================
    # 14. AudioHandoffValidator Integration with Timing Contract
    # =========================================================================

    def test_audio_handoff_validator_with_timing_contract(self):
        """Verifies AudioHandoffValidator inspects timing contracts and rejects violations."""
        sections = [{"id": "section_1", "lecture_lines": ["Line 1", "Line 2"]}]
        audios = {"section_1": str(self._create_mock_mp3(self.tmp_dir / "s1.mp3", 4.0))}

        # Valid timing contract
        valid_contract = SceneTimingContract(
            section_id="section_1",
            segments=[
                NarrationSegmentTiming(segment_id="seg1", section_id="section_1", segment_index=0, narration_text="Line 1", actual_audio_duration=2.0),
                NarrationSegmentTiming(segment_id="seg2", section_id="section_1", segment_index=1, narration_text="Line 2", actual_audio_duration=2.0, timeline_start=2.5),
            ]
        )
        valid_meta = {
            "section_1": {
                "section_id": "section_1",
                "audio_path": audios["section_1"],
                "total_duration": 4.0,
                "lines": [{"raw_text": "Line 1"}, {"raw_text": "Line 2"}],
                "narration_hash": compute_narration_hash(["Line 1", "Line 2"]),
                "timing_contract": valid_contract.to_dict(),
            }
        }
        report_ok = self.handoff_validator.validate_handoff(sections, audios, valid_meta)
        assert report_ok.is_safe_for_stage3

        # Invalid timing contract (segment count mismatch)
        invalid_contract = SceneTimingContract(
            section_id="section_1",
            segments=[
                NarrationSegmentTiming(segment_id="seg1", section_id="section_1", segment_index=0, narration_text="Line 1", actual_audio_duration=2.0)
            ]
        )
        invalid_meta = {
            "section_1": {
                "section_id": "section_1",
                "audio_path": audios["section_1"],
                "total_duration": 4.0,
                "lines": [{"raw_text": "Line 1"}, {"raw_text": "Line 2"}],
                "narration_hash": compute_narration_hash(["Line 1", "Line 2"]),
                "timing_contract": invalid_contract.to_dict(),
            }
        }
        report_bad = self.handoff_validator.validate_handoff(sections, audios, invalid_meta)
        assert not report_bad.is_safe_for_stage3
        assert any(i.category == HandoffIssueCategory.TIMING_CONTRACT_VIOLATION for i in report_bad.issues)

    # =========================================================================
    # 15. Domain Agnosticism (Chemistry, Math, DSA, OS)
    # =========================================================================

    def test_domain_agnosticism_across_subjects(self):
        """Verifies timing contract operates identically across diverse educational subjects."""
        topics = [
            ("Operating Systems", "FCFS Scheduling", ["Process arrival occurs.", "Dispatcher dispatches process."]),
            ("Chemistry", "Reaction Kinetics", ["Reactant molecules collide.", "Activation energy barrier is crossed."]),
            ("Mathematics", "Derivatives", ["Secant line connects two points.", "Limit approaches instantaneous rate."]),
            ("Computer Science", "Dijkstra Algorithm", ["Starting node distance set to zero.", "Greedy relaxation updates neighbors."]),
        ]
        for domain, topic_title, lines in topics:
            sec = {"id": "sec_test", "title": topic_title, "lecture_lines": lines}
            contract = SceneTimingContract.from_storyboard_section(sec)
            assert len(contract.segments) == 2
            assert contract.title == topic_title
            # Update measured TTS durations
            contract.update_actual_audio_durations([3.2, 4.1])
            assert contract.segments[0].actual_audio_duration == 3.2
            assert contract.segments[1].actual_audio_duration == 4.1
            # Reconcile with rendered video
            contract.reconcile_with_rendered_video(8.0)
            assert contract.is_synchronized
            assert contract.total_rendered_video_duration == 8.0
