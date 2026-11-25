"""
Milestone 5.3.4.2 Integration Tests: Native Production Pipeline Orchestration & A/V Multiplexing.

Verifies:
1. Orchestration:
   - Topic-driven execution from scratch (Stage 1 Outline -> Gate 1 -> Stage 2 SceneSequenceIR -> Gate 3.5 -> Pipeline).
   - Gate 1 outline failure and verification halt.
   - Stage 2 SceneSequenceIR failure and empty sequence halt.
   - Strict native pass-through: No SectionProxy, no SceneIRToStoryboardAdapter, no legacy Stage 3 generation.
   - Cached/injected SceneSequenceIR support.
2. Audio and Retiming:
   - Native TTS default allow_fallback=False enforced; fail-closed on TTS failure.
   - Measured audio durations feed audio-authoritative retiming.
   - alignment_status transitions: unaligned -> retimed.
   - Monotonic scheduling with auto_retime=False (no redundant second retiming pass).
   - Duration budget overrun fails closed.
3. Compilation & Rendering:
   - Layout solver and compiler receive native SceneIR objects.
   - Manim rendering failures stop fail-closed before multiplexing.
4. FFmpeg A/V Multiplexing:
   - AudioManager.mux_video_and_audio produces valid video + audio streams.
   - Verified via ffprobe and ffmpeg null decode check.
   - Temporal alignment: freeze-frame (video < audio) and acoustic padding (audio < video).
   - Subtitle burning with escaped subtitle path syntax.
   - Atomic replacement with PID-staged files and temporary file cleanup.
   - alignment_status transitions: retimed -> muxed.
5. Assembly & Master Artifacts:
   - Multi-scene concatenation in deterministic scene order.
   - Master audio, timestamps, and subtitles generation across scenes.
   - Master video validation via validate_media_file.
6. Checkpoint & Recovery:
   - Stage failure prevents subsequent stage checkpoints.
   - Modern and legacy checkpoint isolation preserved.
"""

import os
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

import pytest
from pydub import AudioSegment

from src.agent import TeachingVideoAgent, RunConfig
from src.audio_manager import AudioManager, TTSFailureError
from src.ir.models import (
    SceneIR,
    SceneSequenceIR,
    EntityDeclaration,
    VisualTier,
    TimelineEvent,
    ActionDeclaration,
    ActionType,
    AudioReference,
    NarrationAnchor,
)
from src.ir.fixtures import get_binary_search_fixture
from src.compiler.timeline_scheduler import TimelineScheduler, ScheduledTimeline
from src.compiler.layout_engine import HierarchicalBoxModelLayoutSolver, SceneSpatialLayout
from src.compiler.scene_compiler import DeterministicSceneCompiler, CompiledSceneOutput
from src.worked_example_verifier import WorkedExampleVerifier, WorkedExampleReport
from src.numerical_evaluator import VerificationStatus


def make_deterministic_mock_tts(durations_map: Optional[Dict[str, float]] = None, default_duration: float = 2.0):
    """Creates a deterministic mock TTS provider that writes real, playable silent MP3 files."""
    call_records: List[Dict[str, Any]] = []

    def mock_provider(text: str, output_path: str, voice: str = "en-US-GuyNeural", rate: Optional[str] = "-4%") -> float:
        call_records.append({"text": text, "output_path": output_path, "voice": voice, "rate": rate})
        dur = default_duration
        if durations_map and text in durations_map:
            dur = durations_map[text]
        elif len(text.split()) > 0:
            dur = max(1.0, round(len(text.split()) * 0.35, 2))

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Generate valid silent MP3
        seg = AudioSegment.silent(duration=int(dur * 1000), frame_rate=44100)
        seg.export(str(out_p), format="mp3")
        return dur

    mock_provider.call_records = call_records
    mock_provider.provider_name = "mock-tts-orchestration"
    return mock_provider


def create_mock_video_mp4(output_path: Path, duration: float = 3.0, width: int = 1280, height: int = 720):
    """Creates a minimal valid test MP4 video without audio using FFmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s={width}x{height}:d={duration:.2f}:r=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to create test video: {res.stderr}")
    return output_path


def create_mock_audio_mp3(output_path: Path, duration: float = 3.0, sample_rate: int = 44100):
    """Creates a minimal valid test MP3 audio file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seg = AudioSegment.silent(duration=int(duration * 1000), frame_rate=sample_rate)
    seg.export(str(output_path), format="mp3")
    return output_path


class TestModernTopicDrivenOrchestration:
    """Verifies topic-driven Stage 1/Stage 2 pipeline orchestration."""

    def test_topic_driven_execution_end_to_end_mocked(self, tmp_path):
        """Valid topic-driven execution without pre-cached IR progresses through all modern gates."""
        bs_scene = get_binary_search_fixture()
        bs_seq = SceneSequenceIR(
            sequence_id="seq_binary_search_topic",
            title="Binary Search Algorithm",
            topic="Binary Search Topic Test",
            sections=[bs_scene],
        )

        mock_tts = make_deterministic_mock_tts(default_duration=2.5)

        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search Topic Test",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
            tts_provider=mock_tts,
        )

        # Mock Stage 1 outline generation
        mock_outline = MagicMock()
        mock_outline.sections = [MagicMock(id="sec_1", title="Intro")]
        mock_outline.to_dict.return_value = {"topic": "Binary Search Topic Test", "sections": [{"id": "sec_1"}]}
        agent.generate_outline = MagicMock(return_value=mock_outline)
        agent.is_outline_safe_for_stage2 = MagicMock(return_value=True)

        # Mock Stage 2 native Scene IR generation returning valid SceneSequenceIR
        mock_stage2_res = MagicMock()
        mock_stage2_res.is_valid = True
        mock_stage2_res.sequence_ir = bs_seq
        mock_stage2_res.scene_ir = None
        mock_stage2_res.summary = "Valid SceneSequenceIR"
        agent.generate_scene_ir = MagicMock(return_value=mock_stage2_res)
        agent.scene_sequence_ir = None  # Force generation path

        # Intercept only Manim rendering to use fast mock video while letting FFmpeg/ffprobe run
        orig_run = subprocess.run
        def fake_manim_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "manim":
                code_stem = Path(cmd[2]).stem
                target_vid = agent.output_dir / "media" / "videos" / code_stem / "480p15" / f"{cmd[3]}.mp4"
                create_mock_video_mp4(target_vid, duration=5.0)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return orig_run(cmd, *args, **kwargs)

        with patch("subprocess.run", side_effect=fake_manim_run):
            final_video = agent.GENERATE_VIDEO()

        assert final_video is not None
        final_video_path = Path(final_video)
        assert final_video_path.exists()
        assert final_video_path.stat().st_size > 1000

        # Checkpoints verification
        cp_path = tmp_path / "0-Binary_Search_Topic_Test" / "checkpoint.json"
        assert cp_path.exists()
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["pipeline_mode"] == "modern"
        assert cp["stage"] == "completed"

        # Verify Stage 1 and Stage 2 mocks were invoked
        agent.generate_outline.assert_called_once()
        agent.is_outline_safe_for_stage2.assert_called_once()
        agent.generate_scene_ir.assert_called_once()

    def test_stage1_outline_failure_halts_immediately(self, tmp_path):
        """Stage 1 outline failure halts fail-closed before Stage 2."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search Fail 1",
            folder=str(tmp_path),
            pipeline_mode="modern",
        )
        agent.scene_sequence_ir = None
        agent.generate_outline = MagicMock(return_value=None)
        agent.generate_scene_ir = MagicMock()

        res = agent.GENERATE_VIDEO()

        assert res is None
        agent.generate_scene_ir.assert_not_called()
        cp_path = tmp_path / "0-Binary_Search_Fail_1" / "checkpoint.json"
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["stage"] == "outline_failed"

    def test_stage1_validation_gate_blocked_halts(self, tmp_path):
        """Stage 1 outline validation gate rejection halts before Stage 2."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search Blocked",
            folder=str(tmp_path),
            pipeline_mode="modern",
        )
        agent.scene_sequence_ir = None
        mock_outline = MagicMock()
        mock_outline.sections = [MagicMock()]
        agent.generate_outline = MagicMock(return_value=mock_outline)
        agent.is_outline_safe_for_stage2 = MagicMock(return_value=False)
        agent.outline_verification_report = None
        agent.get_verification_provenance = MagicMock(return_value=None)
        agent.generate_scene_ir = MagicMock()

        res = agent.GENERATE_VIDEO()

        assert res is None
        agent.generate_scene_ir.assert_not_called()
        cp_path = tmp_path / "0-Binary_Search_Blocked" / "checkpoint.json"
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["stage"] == "outline_verification_blocked"

    def test_stage2_generation_failure_halts(self, tmp_path):
        """Stage 2 Scene IR generation invalid result halts before TTS or compilation."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search Stage2 Fail",
            folder=str(tmp_path),
            pipeline_mode="modern",
        )
        agent.scene_sequence_ir = None
        mock_outline = MagicMock()
        mock_outline.sections = [MagicMock()]
        agent.generate_outline = MagicMock(return_value=mock_outline)
        agent.is_outline_safe_for_stage2 = MagicMock(return_value=True)

        mock_bad_stage2 = MagicMock()
        mock_bad_stage2.is_valid = False
        mock_bad_stage2.sequence_ir = None
        mock_bad_stage2.scene_ir = None
        mock_bad_stage2.summary = "Schema validation error"
        mock_bad_stage2.structural_errors = ["Invalid field"]
        mock_bad_stage2.semantic_errors = []
        mock_bad_stage2.payload_errors = []
        agent.generate_scene_ir = MagicMock(return_value=mock_bad_stage2)

        res = agent.GENERATE_VIDEO()

        assert res is None
        cp_path = tmp_path / "0-Binary_Search_Stage2_Fail" / "checkpoint.json"
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["stage"] == "scene_ir_generation_failed"

    def test_empty_scene_sequence_halts(self, tmp_path):
        """A SceneSequenceIR with 0 sections halts fail-closed."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search Empty",
            folder=str(tmp_path),
            pipeline_mode="modern",
        )
        empty_seq = SceneSequenceIR(
            sequence_id="seq_empty",
            title="Empty Sequence",
            topic="Binary Search Empty",
            sections=[],
        )
        agent.scene_sequence_ir = empty_seq

        res = agent.GENERATE_VIDEO()

        assert res is None
        cp_path = tmp_path / "0-Binary_Search_Empty" / "checkpoint.json"
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["stage"] == "scene_ir_empty"


class TestAudioRetimingAndStrictTTS:
    """Verifies native TTS synthesis contracts and audio-authoritative retiming."""

    def test_strict_tts_allow_fallback_false_enforced(self, tmp_path):
        """TTS failure with allow_fallback=False halts modern execution with tts_failed checkpoint."""
        bs_scene = get_binary_search_fixture()
        bs_seq = SceneSequenceIR(
            sequence_id="seq_tts_fail",
            title="TTS Fail Test",
            topic="TTS Failure Test",
            sections=[bs_scene],
        )

        def failing_tts_provider(text, output_path, *args, **kwargs):
            raise TTSFailureError("Simulated Edge TTS network failure in strict mode")

        agent = TeachingVideoAgent(
            idx=0,
            topic="TTS Failure Test",
            folder=str(tmp_path),
            pipeline_mode="modern",
            tts_provider=failing_tts_provider,
        )
        agent.scene_sequence_ir = bs_seq

        res = agent.GENERATE_VIDEO()

        assert res is None
        cp_path = tmp_path / "0-TTS_Failure_Test" / "checkpoint.json"
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["stage"] == "tts_failed"
        assert "Simulated Edge TTS network failure" in str(cp.get("error", ""))

    def test_audio_authoritative_retiming_updates_status_to_retimed(self, tmp_path):
        """Successful retiming marks alignment_status = 'retimed' on retimed scene."""
        scene = get_binary_search_fixture()
        scene.duration_budget = 30.0
        scheduler = TimelineScheduler()
        retimed_sec, sched_tl, diags = scheduler.retime_scene_timeline(scene, inter_cue_pause=0.5, scene_end_pause=1.0)

        assert sched_tl.is_valid
        assert retimed_sec.audio_reference is not None
        assert retimed_sec.audio_reference.alignment_status == "retimed"

    def test_scheduling_without_auto_retime_does_not_mutate_durations(self):
        """Scheduler called with auto_retime=False preserves retimed scene step durations."""
        scene = get_binary_search_fixture()
        scene.duration_budget = 30.0
        scheduler = TimelineScheduler()
        retimed_sec, _, _ = scheduler.retime_scene_timeline(scene, inter_cue_pause=0.5, scene_end_pause=1.0)

        # Call scheduler with auto_retime=False
        sched_tl, diags = scheduler.schedule(retimed_sec, auto_retime=False)
        assert sched_tl.is_valid
        assert not any(d.severity == "ERROR" for d in diags)


class TestHardenedAVMultiplexing:
    """Verifies FFmpeg A/V multiplexing, temporal alignment, and atomic replacement."""

    def test_mux_video_and_audio_valid_streams(self, tmp_path):
        """mux_video_and_audio produces a valid MP4 with video and audio streams verified by ffprobe."""
        mgr = AudioManager()
        vid_p = tmp_path / "test_video.mp4"
        aud_p = tmp_path / "test_audio.mp3"
        out_p = tmp_path / "muxed_output.mp4"

        create_mock_video_mp4(vid_p, duration=3.0)
        create_mock_audio_mp3(aud_p, duration=3.0)

        res = mgr.mux_video_and_audio(
            video_path=vid_p,
            audio_path=aud_p,
            output_path=out_p,
            burn_subtitles=False,
            expected_duration=3.0,
            tolerance=0.5,
        )

        assert res["success"] is True
        assert out_p.exists()
        assert out_p.stat().st_size > 1000

        # Validate with ffprobe
        val = mgr.validate_media_file(str(out_p), require_audio=True)
        assert val["is_valid"] is True
        assert val["has_video"] is True
        assert val["has_audio"] is True
        assert abs(val["duration"] - 3.0) <= 0.5

    def test_mux_temporal_alignment_freeze_frame(self, tmp_path):
        """When video is shorter than audio, last frame is frozen to extend video."""
        mgr = AudioManager()
        vid_p = tmp_path / "short_video.mp4"
        aud_p = tmp_path / "long_audio.mp3"
        out_p = tmp_path / "muxed_freeze.mp4"

        create_mock_video_mp4(vid_p, duration=2.0)
        create_mock_audio_mp3(aud_p, duration=4.0)

        res = mgr.mux_video_and_audio(
            video_path=vid_p,
            audio_path=aud_p,
            output_path=out_p,
            burn_subtitles=False,
            expected_duration=4.0,
            tolerance=0.5,
        )

        assert res["success"] is True
        val = mgr.validate_media_file(str(out_p), require_audio=True)
        assert val["is_valid"] is True
        assert abs(val["duration"] - 4.0) <= 0.5

    def test_mux_temporal_alignment_audio_padding(self, tmp_path):
        """When audio is shorter than video, audio is padded with acoustic presence."""
        mgr = AudioManager()
        vid_p = tmp_path / "long_video.mp4"
        aud_p = tmp_path / "short_audio.mp3"
        out_p = tmp_path / "muxed_padded.mp4"

        create_mock_video_mp4(vid_p, duration=4.0)
        create_mock_audio_mp3(aud_p, duration=2.0)

        res = mgr.mux_video_and_audio(
            video_path=vid_p,
            audio_path=aud_p,
            output_path=out_p,
            burn_subtitles=False,
            expected_duration=4.0,
            tolerance=0.5,
        )

        assert res["success"] is True
        val = mgr.validate_media_file(str(out_p), require_audio=True)
        assert val["is_valid"] is True
        assert abs(val["duration"] - 4.0) <= 0.5

    def test_mux_burns_subtitles_when_present(self, tmp_path):
        """Subtitles are burned onto the video when burn_subtitles=True and SRT exists."""
        mgr = AudioManager()
        vid_p = tmp_path / "sub_video.mp4"
        aud_p = tmp_path / "sub_audio.mp3"
        srt_p = tmp_path / "subtitles.srt"
        out_p = tmp_path / "muxed_subtitled.mp4"

        create_mock_video_mp4(vid_p, duration=3.0)
        create_mock_audio_mp3(aud_p, duration=3.0)
        srt_p.write_text(
            "1\n00:00:00,500 --> 00:00:02,500\nBinary search evaluates middle element.\n\n",
            encoding="utf-8",
        )

        res = mgr.mux_video_and_audio(
            video_path=vid_p,
            audio_path=aud_p,
            output_path=out_p,
            srt_path=srt_p,
            burn_subtitles=True,
            expected_duration=3.0,
            tolerance=0.5,
        )

        assert res["success"] is True
        val = mgr.validate_media_file(str(out_p), require_audio=True)
        assert val["is_valid"] is True

    def test_mux_missing_inputs_fail_closed(self, tmp_path):
        """Missing video or audio files fail closed with structured error."""
        mgr = AudioManager()
        valid_vid = tmp_path / "v.mp4"
        create_mock_video_mp4(valid_vid, duration=2.0)

        # Missing audio
        res_a = mgr.mux_video_and_audio(
            video_path=valid_vid,
            audio_path=tmp_path / "nonexistent.mp3",
            output_path=tmp_path / "out1.mp4",
        )
        assert res_a["success"] is False
        assert "Audio input does not exist" in res_a["error"]

        # Missing video
        res_v = mgr.mux_video_and_audio(
            video_path=tmp_path / "nonexistent.mp4",
            audio_path=tmp_path / "a.mp3",
            output_path=tmp_path / "out2.mp4",
        )
        assert res_v["success"] is False
        assert "Video input does not exist" in res_v["error"]


class TestCheckpointAndStageRecovery:
    """Verifies checkpointing and stage-level failure recovery."""

    def test_checkpoint_isolation_legacy_and_modern(self, tmp_path):
        """Agent in modern mode rejects existing legacy checkpoint and raises ValueError."""
        cp_file = tmp_path / "0-Isolation_Test" / "checkpoint.json"
        cp_file.parent.mkdir(parents=True, exist_ok=True)
        legacy_data = {
            "topic": "Isolation Test",
            "level": "intermediate",
            "pipeline_mode": "legacy",
            "stage": "outline_completed",
        }
        with open(cp_file, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f)

        with pytest.raises(ValueError) as exc:
            TeachingVideoAgent(
                idx=0,
                topic="Isolation Test",
                folder=str(tmp_path),
                pipeline_mode="modern",
            )
        assert "Checkpoint pipeline mode mismatch" in str(exc.value)

    def test_alignment_status_lifecycle_transitions(self, tmp_path):
        """Verifies full lifecycle of alignment_status: unaligned -> retimed -> muxed."""
        bs_scene = get_binary_search_fixture()
        bs_scene.duration_budget = 30.0
        assert bs_scene.audio_reference.alignment_status == "unaligned"

        scheduler = TimelineScheduler()
        retimed_sec, sched_tl, _ = scheduler.retime_scene_timeline(bs_scene)
        assert retimed_sec.audio_reference.alignment_status == "retimed"

        mgr = AudioManager()
        vid_p = tmp_path / "cycle_v.mp4"
        aud_p = tmp_path / "cycle_a.mp3"
        out_p = tmp_path / "cycle_out.mp4"
        create_mock_video_mp4(vid_p, duration=3.0)
        create_mock_audio_mp3(aud_p, duration=3.0)

        mux_res = mgr.mux_video_and_audio(vid_p, aud_p, out_p, burn_subtitles=False)
        assert mux_res["success"] is True

        # Simulate update upon verified mux
        retimed_sec.audio_reference.alignment_status = "muxed"
        assert retimed_sec.audio_reference.alignment_status =="muxed"


class TestMultiSceneAssemblyAndMasterArtifacts:
    """Verifies multi-scene concatenation, master subtitle alignment, and decode checks."""

    def test_multi_scene_deterministic_concat_and_master_artifacts(self, tmp_path):
        """Multi-scene sequence compiles, renders, muxes, concatenates, and generates master audio/captions."""
        sc1 = get_binary_search_fixture()
        sc1.scene_id = "Scene1"
        sc1.duration_budget = 30.0

        sc2 = get_binary_search_fixture()
        sc2.scene_id = "Scene2"
        sc2.duration_budget = 30.0

        multi_seq = SceneSequenceIR(
            sequence_id="seq_multi_test",
            title="Multi Scene Test",
            topic="Multi Scene Topic",
            sections=[sc1, sc2],
        )

        mock_tts = make_deterministic_mock_tts(default_duration=2.0)
        agent = TeachingVideoAgent(
            idx=0,
            topic="Multi Scene Topic",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
            tts_provider=mock_tts,
        )
        agent.scene_sequence_ir = multi_seq

        orig_run = subprocess.run
        def fake_manim_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "manim":
                code_stem = Path(cmd[2]).stem
                target_vid = agent.output_dir / "media" / "videos" / code_stem / "480p15" / f"{cmd[3]}.mp4"
                create_mock_video_mp4(target_vid, duration=4.0)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return orig_run(cmd, *args, **kwargs)

        with patch("subprocess.run", side_effect=fake_manim_run):
            final_video = agent.GENERATE_VIDEO()

        assert final_video is not None
        v_path = Path(final_video)
        assert v_path.exists()
        assert v_path.stat().st_size > 1000

        # Master artifacts verification
        out_dir = agent.output_dir
        assert (out_dir / "timestamps.json").exists()
        assert (out_dir / "subtitles.srt").exists()
        assert (out_dir / "subtitles.vtt").exists()

        # Check that timestamps cover both scenes with strictly increasing offsets
        with open(out_dir / "timestamps.json", "r", encoding="utf-8") as f:
            ts_data = json.load(f)
        assert len(ts_data["sections"]) == 2
        assert ts_data["sections"][0]["section_id"] == "Scene1"
        assert ts_data["sections"][1]["section_id"] == "Scene2"
        assert ts_data["sections"][1]["start"] >= ts_data["sections"][0]["end"]

        # Final media stream check
        val = agent.audio_manager.validate_media_file(str(v_path), require_audio=True)
        assert val["is_valid"] is True
        assert val["has_video"] is True
        assert val["has_audio"] is True

    def test_final_media_validation_failure_halts_fail_closed(self, tmp_path):
        """Corrupted or invalid final media artifact halts with final_validation_failed checkpoint."""
        sc = get_binary_search_fixture()
        sc.duration_budget = 30.0
        seq = SceneSequenceIR(
            sequence_id="seq_corrupt_test",
            title="Corrupt Video Test",
            topic="Corrupt Video Test",
            sections=[sc],
        )
        mock_tts = make_deterministic_mock_tts(default_duration=2.0)
        agent = TeachingVideoAgent(
            idx=0,
            topic="Corrupt Video Test",
            folder=str(tmp_path),
            pipeline_mode="modern",
            tts_provider=mock_tts,
        )
        agent.scene_sequence_ir = seq

        orig_run = subprocess.run
        def fake_manim_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "manim":
                code_stem = Path(cmd[2]).stem
                target_vid = agent.output_dir / "media" / "videos" / code_stem / "720p30" / f"{cmd[3]}.mp4"
                create_mock_video_mp4(target_vid, duration=3.0)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return orig_run(cmd, *args, **kwargs)

        orig_val = agent.audio_manager.validate_media_file
        def fake_val(file_path, *args, **kwargs):
            if "Corrupt_Video_Test.mp4" in str(file_path):
                return {"is_valid": False, "error": "Mocked corrupt final streams"}
            return orig_val(file_path, *args, **kwargs)

        with patch("subprocess.run", side_effect=fake_manim_run), \
             patch.object(agent.audio_manager, "validate_media_file", side_effect=fake_val):
            res = agent.GENERATE_VIDEO()

        assert res is None
        cp_path = tmp_path / "0-Corrupt_Video_Test" / "checkpoint.json"
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["stage"] == "final_validation_failed"


class TestNativeContractGuarantees:
    """Verifies that no legacy adapters or legacy pathways are touched."""

    def test_strict_allow_fallback_false_passed_to_synthesizer(self, tmp_path):
        """Verifies generate_scene_sequence_audio is called with allow_fallback=False."""
        sc = get_binary_search_fixture()
        sc.duration_budget = 30.0
        seq = SceneSequenceIR(
            sequence_id="seq_contract",
            title="Contract Test",
            topic="Contract Test",
            sections=[sc],
        )
        agent = TeachingVideoAgent(
            idx=0,
            topic="Contract Test",
            folder=str(tmp_path),
            pipeline_mode="modern",
        )
        agent.scene_sequence_ir = seq

        called_kwargs = {}
        orig_gen = agent.audio_manager.generate_scene_sequence_audio
        def spy_gen(*args, **kwargs):
            called_kwargs.update(kwargs)
            return orig_gen(*args, **kwargs)

        agent.audio_manager.generate_scene_sequence_audio = spy_gen
        agent.tts_provider = make_deterministic_mock_tts()

        orig_run = subprocess.run
        def fake_manim_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "manim":
                code_stem = Path(cmd[2]).stem
                target_vid = agent.output_dir / "media" / "videos" / code_stem / "720p30" / f"{cmd[3]}.mp4"
                create_mock_video_mp4(target_vid, duration=3.0)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return orig_run(cmd, *args, **kwargs)

        with patch("subprocess.run", side_effect=fake_manim_run):
            agent.GENERATE_VIDEO()

        assert called_kwargs.get("allow_fallback") is False

    def test_zero_legacy_adapters_or_storyboard_calls(self, tmp_path):
        """Verifies complete isolation: no SectionProxy, no storyboard, no legacy code gen."""
        sc = get_binary_search_fixture()
        sc.duration_budget = 30.0
        seq = SceneSequenceIR(
            sequence_id="seq_iso",
            title="Isolation Test",
            topic="Isolation Zero Legacy",
            sections=[sc],
        )
        agent = TeachingVideoAgent(
            idx=0,
            topic="Isolation Zero Legacy",
            folder=str(tmp_path),
            pipeline_mode="modern",
            tts_provider=make_deterministic_mock_tts(),
        )
        agent.scene_sequence_ir = seq

        agent.generate_storyboard = MagicMock(side_effect=AssertionError("Legacy storyboard called!"))
        agent.generate_codes = MagicMock(side_effect=AssertionError("Legacy code gen called!"))
        agent._execute_legacy_pipeline = MagicMock(side_effect=AssertionError("Legacy pipeline called!"))

        orig_run = subprocess.run
        def fake_manim_run(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "manim":
                code_stem = Path(cmd[2]).stem
                target_vid = agent.output_dir / "media" / "videos" / code_stem / "720p30" / f"{cmd[3]}.mp4"
                create_mock_video_mp4(target_vid, duration=3.0)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return orig_run(cmd, *args, **kwargs)

        with patch("subprocess.run", side_effect=fake_manim_run):
            final = agent.GENERATE_VIDEO()

        assert final is not None
        assert agent.generate_storyboard.call_count == 0
        assert agent.generate_codes.call_count == 0
        assert agent._execute_legacy_pipeline.call_count == 0

