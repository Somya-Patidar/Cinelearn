"""
Unit and integration tests for CineLearn Milestone 5.3.4.1:
Native TTS Fail-Closed Handling & Rejection of Silent Room-Tone Fallback.

Verifies resolution of condition COND-5.3.2-01:
1. Native synthesis path fails closed with TTSFailureError when speech synthesis fails.
2. Silent room-tone fallback is NEVER treated as synthesized speech by the native pipeline.
3. No false AudioReference is attached to SceneIR on failure.
4. No false narration timestamps are assigned to NarrationAnchor.audio_timestamp_range.
5. No misleading SRT captions are generated spanning silent intervals.
6. Silent room-tone artifacts are never cached as valid speech.
7. Scoped artifact cleanup purges partial cue and scene files on failure.
8. Subsequent synthesis succeeds cleanly after an earlier failure.
9. Cache entries generated under fallback policy are rejected under strict policy.
10. Existing injected mock providers returning float continue to work with zero regressions.
11. Legacy storyboard fallback behavior (allow_fallback=True) remains 100% functional.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from pydub import AudioSegment

from src.ir.models import (
    SceneIR,
    SceneSequenceIR,
    TimelineEvent,
    NarrationAnchor,
    ActionDeclaration,
    ActionType,
    EntityDeclaration,
    VisualTier,
    AudioReference,
)
from src.audio_manager import (
    AudioManager,
    NativeSceneAudioSynthesizer,
    TTSFailureError,
    SpeechSynthesisResult,
)


def make_test_scene_ir(
    scene_id: str = "FailClosedScene",
    cues: tuple = ("First explanation beat.", "Second concept explanation.", "Third concluding step."),
    budget: float = 25.0,
    indices: list = None,
    initial_timestamp_range: tuple = (0.0, 0.0),
) -> SceneIR:
    """Helper to construct a valid native SceneIR with timeline narration anchors."""
    if indices is None:
        indices = list(range(len(cues)))

    timeline = []
    for i, (cue, l_idx) in enumerate(zip(cues, indices)):
        anchor = None
        if cue is not None:
            anchor = NarrationAnchor(
                lecture_line_index=l_idx,
                text_cue=cue,
                audio_timestamp_range=initial_timestamp_range,
            )
        timeline.append(
            TimelineEvent(
                step_id=f"step_{i+1}",
                narration_anchor=anchor,
                actions=[
                    ActionDeclaration(
                        type=ActionType.ENTER,
                        target_entity="card_1",
                        duration=1.0,
                    )
                ],
            )
        )

    return SceneIR(
        scene_id=scene_id,
        title="Fail-Closed Test Scene",
        duration_budget=budget,
        entities=[
            EntityDeclaration(
                id="card_1",
                component_type="ProcessCardVisual",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="dock_main",
            )
        ],
        timeline=timeline,
    )


def make_mock_tts_provider(default_duration: float = 1.5):
    """Creates a deterministic mock TTS provider that writes valid silent MP3 files."""
    call_records = []

    def mock_provider(text: str, output_path: str, voice: str = "en-US-GuyNeural", rate: str = "-4%") -> float:
        call_records.append({"text": text, "output_path": output_path, "voice": voice, "rate": rate})
        dur = default_duration
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        silence = AudioSegment.silent(duration=int(dur * 1000))
        silence.export(output_path, format="mp3")
        return dur

    mock_provider.call_records = call_records
    mock_provider.provider_name = "mock-tts"
    return mock_provider


class TestSpeechSynthesisResultContract:
    """Tests for the typed SpeechSynthesisResult contract."""

    def test_result_subclasses_float_for_backward_compatibility(self):
        """SpeechSynthesisResult must be a float subclass with arithmetic and comparison support."""
        res = SpeechSynthesisResult(
            duration=3.14159,
            is_speech=True,
            is_fallback=False,
            output_path="/tmp/audio.mp3",
            provider="edge-tts",
        )
        assert isinstance(res, float)
        assert res > 3.0
        assert res < 3.5
        assert res == 3.14159
        assert round(res, 2) == 3.14
        assert res + 1.0 == 4.14159

        # Metadata attributes
        assert res.is_speech is True
        assert res.is_fallback is False
        assert res.output_path == "/tmp/audio.mp3"
        assert res.provider == "edge-tts"
        assert res.error is None

        # Dict-like access
        assert res["is_speech"] is True
        assert res.get("is_fallback") is False
        assert res.to_dict()["provider"] == "edge-tts"

    def test_fallback_result_metadata(self):
        """Fallback result records is_speech=False, is_fallback=True, and provider."""
        res = SpeechSynthesisResult(
            duration=2.5,
            is_speech=False,
            is_fallback=True,
            output_path="/tmp/room_tone.mp3",
            provider="room-tone-fallback",
            error="Connection timed out",
            cue="Sample cue text",
        )
        assert res.is_speech is False
        assert res.is_fallback is True
        assert res.provider == "room-tone-fallback"
        assert res.error == "Connection timed out"
        assert res.cue == "Sample cue text"
        assert res == 2.5


class TestNativeTTSFailClosed:
    """Core tests verifying native TTS fail-closed handling and room-tone rejection."""

    def test_native_synthesizer_fails_closed_when_edge_tts_fails(self, tmp_path):
        """When Edge-TTS fails and default target_provider is None, synthesizer raises TTSFailureError."""
        scene = make_test_scene_ir("SceneTTSFail")
        audio_mgr = AudioManager()

        # Patch Edge-TTS async call to simulate network or service failure
        with patch.object(audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("Edge-TTS connection refused")):
            synthesizer = NativeSceneAudioSynthesizer(audio_manager=audio_mgr, allow_fallback=False)
            with pytest.raises(TTSFailureError) as exc_info:
                synthesizer.synthesize_scene_ir(scene, tmp_path)

            assert "Edge-TTS connection refused" in str(exc_info.value)
            assert exc_info.value.provider == "edge-tts"

        # Fail-closed invariants: SceneIR has NO AudioReference and anchors have NO ranges
        assert scene.audio_reference is None
        for event in scene.timeline:
            assert event.narration_anchor.audio_timestamp_range is None

    def test_room_tone_fallback_rejected_by_native_path(self, tmp_path):
        """If a provider returns a room-tone fallback result, native path rejects it fail-closed."""
        scene = make_test_scene_ir("SceneFallbackRejection")

        def room_tone_provider(text, output_path, voice, rate):
            # Writes silence and returns SpeechSynthesisResult marked as fallback
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            silence = AudioSegment.silent(duration=2000)
            silence.export(output_path, format="mp3")
            return SpeechSynthesisResult(
                duration=2.0,
                is_speech=False,
                is_fallback=True,
                output_path=output_path,
                provider="simulated-room-tone",
            )

        synthesizer = NativeSceneAudioSynthesizer(tts_provider=room_tone_provider, allow_fallback=False)
        with pytest.raises(TTSFailureError, match="rejected in fail-closed mode"):
            synthesizer.synthesize_scene_ir(scene, tmp_path)

        assert scene.audio_reference is None
        for event in scene.timeline:
            assert event.narration_anchor.audio_timestamp_range is None

    def test_no_false_timestamps_or_srt_captions_on_failure(self, tmp_path):
        """On synthesis failure, no SRT captions, JSON timestamps, or combined audio artifacts exist on disk."""
        scene = make_test_scene_ir("SceneNoArtifacts")
        audio_mgr = AudioManager()

        with patch.object(audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("Service unavailable")):
            synthesizer = NativeSceneAudioSynthesizer(audio_manager=audio_mgr, allow_fallback=False)
            with pytest.raises(TTSFailureError):
                synthesizer.synthesize_scene_ir(scene, tmp_path)

        # Confirm zero false artifacts remain on disk
        srt_file = tmp_path / "SceneNoArtifacts.srt"
        ts_file = tmp_path / "SceneNoArtifacts_timestamps.json"
        audio_file = tmp_path / "SceneNoArtifacts_audio.mp3"

        assert not srt_file.exists()
        assert not ts_file.exists()
        assert not audio_file.exists()

    def test_failed_synthesis_not_cached_as_speech(self, tmp_path):
        """A failed synthesis attempt writes no cache entry; subsequent attempt does not hit cache."""
        scene = make_test_scene_ir("SceneNoCachePollution")
        audio_mgr = AudioManager()

        with patch.object(audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("Network drop")):
            synthesizer = NativeSceneAudioSynthesizer(audio_manager=audio_mgr, allow_fallback=False)
            with pytest.raises(TTSFailureError):
                synthesizer.synthesize_scene_ir(scene, tmp_path)

        ts_file = tmp_path / "SceneNoCachePollution_timestamps.json"
        assert not ts_file.exists()

        # Check that subsequent call cannot find any cache entry
        mock_prov = make_mock_tts_provider()
        synthesizer2 = NativeSceneAudioSynthesizer(tts_provider=mock_prov, allow_fallback=False)
        _, meta2 = synthesizer2.synthesize_scene_ir(scene, tmp_path)

        assert meta2["cache_hit"] is False
        assert len(mock_prov.call_records) == 3

    def test_scoped_artifact_cleanup_on_multi_cue_failure(self, tmp_path):
        """When cue 1 succeeds but cue 2 fails, committed cue 1 files and temp files are cleanly unlinked."""
        scene = make_test_scene_ir("SceneMultiCueCleanup", cues=("First cue.", "Second failing cue.", "Third cue."))
        call_count = 0

        def failing_on_cue_2(text, output_path, voice, rate):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Cue 1 succeeds
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                silence = AudioSegment.silent(duration=1500)
                silence.export(output_path, format="mp3")
                return 1.5
            else:
                # Cue 2 fails
                raise RuntimeError("Synthesis engine crashed on cue 2")

        synthesizer = NativeSceneAudioSynthesizer(tts_provider=failing_on_cue_2, allow_fallback=False)
        with pytest.raises(RuntimeError, match="crashed on cue 2"):
            synthesizer.synthesize_scene_ir(scene, tmp_path)

        audio_dir = tmp_path / "audio_lines"
        # Cue 1's committed file must have been unlinked during exception cleanup
        cue_1_file = audio_dir / "SceneMultiCueCleanup_cue_1.mp3"
        assert not cue_1_file.exists()

        # No temp files remaining
        remaining_files = list(tmp_path.rglob("*.tmp.*"))
        assert len(remaining_files) == 0

    def test_subsequent_synthesis_succeeds_after_earlier_failure(self, tmp_path):
        """After a failed synthesis attempt is cleaned up, a subsequent attempt succeeds normally."""
        scene = make_test_scene_ir("SceneRetryRecovery")
        audio_mgr = AudioManager()

        # 1. First attempt fails
        with patch.object(audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("Transient 503 error")):
            synthesizer = NativeSceneAudioSynthesizer(audio_manager=audio_mgr, allow_fallback=False)
            with pytest.raises(TTSFailureError):
                synthesizer.synthesize_scene_ir(scene, tmp_path)

        assert scene.audio_reference is None

        # 2. Second attempt succeeds (using working mock provider)
        mock_prov = make_mock_tts_provider(default_duration=2.0)
        synthesizer_ok = NativeSceneAudioSynthesizer(tts_provider=mock_prov, allow_fallback=False)
        updated_scene, meta = synthesizer_ok.synthesize_scene_ir(scene, tmp_path)

        assert updated_scene.audio_reference is not None
        assert updated_scene.audio_reference.total_duration > 0
        assert Path(meta["audio_path"]).exists()
        assert Path(meta["timestamps_path"]).exists()
        assert Path(meta["srt_path"]).exists()
        assert meta["is_speech"] is True
        assert meta["is_fallback"] is False

    def test_cache_entries_under_different_fallback_policies_not_reused(self, tmp_path):
        """A cache artifact created with allow_fallback=True is rejected when queried with allow_fallback=False."""
        scene = make_test_scene_ir("ScenePolicyIsolation")
        audio_mgr = AudioManager()

        # 1. Synthesize with allow_fallback=True under failing Edge-TTS
        with patch.object(audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("Network offline")):
            synthesizer_fallback = NativeSceneAudioSynthesizer(audio_manager=audio_mgr, allow_fallback=True)
            _, meta_fb = synthesizer_fallback.synthesize_scene_ir(scene, tmp_path)

        assert meta_fb["is_fallback"] is True
        assert meta_fb["is_speech"] is False
        assert Path(meta_fb["audio_path"]).exists()

        # 2. Query cache with allow_fallback=False: MUST reject the fallback cache
        mock_prov = make_mock_tts_provider()
        synthesizer_strict = NativeSceneAudioSynthesizer(tts_provider=mock_prov, allow_fallback=False)
        _, meta_strict = synthesizer_strict.synthesize_scene_ir(scene, tmp_path)

        # Must not be a cache hit; must have executed mock provider
        assert meta_strict["cache_hit"] is False
        assert meta_strict["is_fallback"] is False
        assert meta_strict["is_speech"] is True
        assert len(mock_prov.call_records) == 3

    def test_existing_injected_mock_providers_returning_float_continue_to_work(self, tmp_path):
        """Injected test mock providers returning plain float durations continue to work seamlessly."""
        scene = make_test_scene_ir("SceneFloatMock")
        mock_prov = make_mock_tts_provider(default_duration=1.75)
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_prov, allow_fallback=False)

        updated_scene, meta = synthesizer.synthesize_scene_ir(scene, tmp_path)

        assert updated_scene.audio_reference is not None
        assert meta["cache_hit"] is False
        assert meta["is_speech"] is True
        assert meta["is_fallback"] is False
        assert len(mock_prov.call_records) == 3


class TestLegacyAudioManagerCompatibility:
    """Verifies that legacy AudioManager APIs preserve fallback when requested."""

    def test_legacy_generate_speech_defaults_to_allow_fallback(self, tmp_path):
        """Default AudioManager.generate_speech continues to fall back to calibrated silence."""
        audio_mgr = AudioManager()
        out_file = tmp_path / "legacy_fallback.mp3"

        with patch.object(audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("Edge-TTS 500")):
            res = audio_mgr.generate_speech("A legacy test line with seven words.", str(out_file))

        assert isinstance(res, float)
        assert res > 1.0
        assert res.is_fallback is True
        assert res.is_speech is False
        assert out_file.exists()
        assert out_file.stat().st_size > 500

    def test_legacy_generate_speech_strict_mode_raises(self, tmp_path):
        """AudioManager.generate_speech(allow_fallback=False) raises TTSFailureError."""
        audio_mgr = AudioManager()
        out_file = tmp_path / "strict_failure.mp3"

        with patch.object(audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("Network down")):
            with pytest.raises(TTSFailureError, match="Network down"):
                audio_mgr.generate_speech("Strict mode test line.", str(out_file), allow_fallback=False)

        assert not out_file.exists()

    def test_generate_speech_structured_defaults_to_strict(self, tmp_path):
        """AudioManager.generate_speech_structured defaults to allow_fallback=False."""
        audio_mgr = AudioManager()
        out_file = tmp_path / "structured_strict.mp3"

        with patch.object(audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("DNS failure")):
            with pytest.raises(TTSFailureError, match="DNS failure"):
                audio_mgr.generate_speech_structured("Structured mode test line.", str(out_file))

        assert not out_file.exists()

    def test_audio_manager_generate_scene_ir_audio_defaults_to_fail_closed(self, tmp_path):
        """AudioManager.generate_scene_ir_audio defaults to fail closed on Edge-TTS failure."""
        audio_mgr = AudioManager()
        scene = make_test_scene_ir("SceneMgrStrict")

        with patch.object(audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("Edge-TTS unreachable")):
            with pytest.raises(TTSFailureError):
                audio_mgr.generate_scene_ir_audio(scene, tmp_path)

        assert scene.audio_reference is None

    def test_audio_manager_generate_scene_sequence_audio_defaults_to_fail_closed(self, tmp_path):
        """AudioManager.generate_scene_sequence_audio defaults to fail closed on Edge-TTS failure."""
        audio_mgr = AudioManager()
        scene1 = make_test_scene_ir("SeqScene1", cues=("Cue 1",))
        scene2 = make_test_scene_ir("SeqScene2", cues=("Cue 2",))
        seq_ir = SceneSequenceIR(sequence_id="TestSeq", sections=[scene1, scene2])

        with patch.object(audio_mgr, "_generate_speech_edge_tts", side_effect=RuntimeError("Edge-TTS unreachable")):
            with pytest.raises(TTSFailureError):
                audio_mgr.generate_scene_sequence_audio(seq_ir, tmp_path)

        assert scene1.audio_reference is None
        assert scene2.audio_reference is None
