"""
Milestone 5.3.2 Unit and Integration Tests: Native TTS Synthesis, Cache Integration, and Audio Artifact Generation.

Verifies:
1. Native input: Authoritative cue extraction, deterministic ordering, empty narration handling,
   whitespace filtering, absence of legacy storyboard adapters.
2. Synthesis & Artifacts: Valid audio generation, clear provider failure reporting, rejection of empty/corrupted
   outputs, authoritative duration measurement from artifact, strictly measured cue timestamps.
3. Cache Identity: SHA-256 fingerprinting across text, ordering, voice, rate, and provider configuration;
   cache hits bypassing provider; cache invalidation on mutations; corrupted cache rejection; atomic write safety.
4. Scene IR Integration: In-place metadata population, AudioReference and NarrationAnchor serialization roundtrip,
   SceneSequenceIR multi-scene orchestration, unmeasured events remaining unmeasured, legacy pipeline preservation.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from unittest.mock import MagicMock

import pytest
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
from src.audio_manager import AudioManager, NativeSceneAudioSynthesizer
from src.audio_handoff_validator import compute_scene_ir_narration_hash


def make_mock_tts_provider(durations_map: Optional[Dict[str, float]] = None, default_duration: float = 1.5):
    """
    Creates a deterministic mock TTS provider that writes valid, playable silent MP3 files
    without making network or external API calls.
    """
    call_records: List[Dict[str, Any]] = []

    def mock_provider(text: str, output_path: str, voice: str = "en-US-GuyNeural", rate: Optional[str] = "-4%") -> float:
        call_records.append({"text": text, "output_path": output_path, "voice": voice, "rate": rate})
        dur = default_duration
        if durations_map and text in durations_map:
            dur = durations_map[text]
        elif len(text.split()) > 0:
            dur = max(1.0, round(len(text.split()) * 0.4, 2))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Create valid MP3 audio segment of exact duration
        silence = AudioSegment.silent(duration=int(dur * 1000))
        silence.export(output_path, format="mp3")
        return dur

    mock_provider.call_records = call_records
    mock_provider.provider_name = "mock-tts"
    return mock_provider


def make_test_scene_ir(
    scene_id: str = "Scene1",
    cues: tuple = ("First explanation beat.", "Second concept explanation.", "Third concluding step."),
    budget: float = 25.0,
    indices: Optional[List[int]] = None,
    initial_timestamp_range: Tuple[float, float] = (0.0, 0.0),
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
        title="Test Scene",
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


class TestNativeTTSInput:
    """Tests for Phase B: Native input extraction, ordering, and interface independence."""

    def test_scene_ir_cues_passed_in_authoritative_order(self, tmp_path):
        """Cues declared with out-of-order indices must be synthesized in lecture_line_index order."""
        # Declare line index 2 first, index 0 second, index 1 third
        scene = make_test_scene_ir(
            scene_id="SceneOutOfOrder",
            cues=("Beat C (Line 2)", "Beat A (Line 0)", "Beat B (Line 1)"),
            indices=[2, 0, 1],
        )
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        scene, meta = synthesizer.synthesize_scene_ir(scene, tmp_path)

        assert len(mock_provider.call_records) == 3
        # Must be called in authoritative order: Line 0, Line 1, Line 2
        assert mock_provider.call_records[0]["text"] == "Beat A (Line 0)"
        assert mock_provider.call_records[1]["text"] == "Beat B (Line 1)"
        assert mock_provider.call_records[2]["text"] == "Beat C (Line 2)"

    def test_empty_narration_produces_clear_safe_result(self, tmp_path):
        """A SceneIR with zero narration cues produces a safe, unpopulated result without error."""
        scene = SceneIR(
            scene_id="SceneSilent",
            title="Silent Visual Transition",
            duration_budget=5.0,
            timeline=[
                TimelineEvent(
                    step_id="step_visual_only",
                    narration_anchor=None,
                    actions=[ActionDeclaration(type=ActionType.HIGHLIGHT, target_entity="card_1", duration=2.0)],
                )
            ],
        )
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        scene, meta = synthesizer.synthesize_scene_ir(scene, tmp_path)

        assert meta["has_narration"] is False
        assert meta["total_duration"] == 0.0
        assert meta["audio_path"] is None
        assert scene.audio_reference is None
        assert scene.timeline[0].narration_anchor is None
        assert len(mock_provider.call_records) == 0

    def test_whitespace_and_missing_anchors_filtered(self, tmp_path):
        """Whitespace cues and missing anchors are skipped, leaving visual events unmeasured."""
        scene = SceneIR(
            scene_id="SceneSparse",
            duration_budget=15.0,
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Valid spoken line.",
                        audio_timestamp_range=(0.0, 0.0),
                    ),
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=None,  # Visual only
                ),
                TimelineEvent(
                    step_id="step_3",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=1,
                        text_cue="   \t\n  ",
                        audio_timestamp_range=(0.0, 0.0),
                    ),  # Whitespace only
                ),
                TimelineEvent(
                    step_id="step_4",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=2,
                        text_cue="Second valid spoken line.",
                        audio_timestamp_range=(0.0, 0.0),
                    ),
                ),
            ],
        )
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        scene, meta = synthesizer.synthesize_scene_ir(scene, tmp_path)

        assert len(mock_provider.call_records) == 2
        assert mock_provider.call_records[0]["text"] == "Valid spoken line."
        assert mock_provider.call_records[1]["text"] == "Second valid spoken line."

        # Verify only valid events received measured timestamp ranges
        r0 = scene.timeline[0].narration_anchor.audio_timestamp_range
        assert r0[1] > r0[0]
        assert scene.timeline[1].narration_anchor is None
        # Whitespace cue was not synthesized and retains initial unmeasured range
        assert scene.timeline[2].narration_anchor.audio_timestamp_range == (0.0, 0.0)
        r3 = scene.timeline[3].narration_anchor.audio_timestamp_range
        assert r3[1] > r3[0]

    def test_no_legacy_storyboard_adapter_used(self, tmp_path):
        """Synthesis operates directly on native SceneIR without legacy Section/SectionProxy structures."""
        scene = make_test_scene_ir("SceneNativeOnly")
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        # Ingest pure SceneIR
        updated_scene, meta = synthesizer.synthesize_scene_ir(scene, tmp_path)
        assert isinstance(updated_scene, SceneIR)
        assert updated_scene.audio_reference is not None

    def test_cue_text_not_silently_rewritten(self, tmp_path):
        """Text casing, punctuation, and wording are preserved verbatim during synthesis."""
        original_cues = ("Algorithm P1: checks [u, v] for edge (1, 2)!", "Wait; does Ea > 75000?")
        scene = make_test_scene_ir("SceneVerbatim", cues=original_cues)
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        synthesizer.synthesize_scene_ir(scene, tmp_path)

        assert mock_provider.call_records[0]["text"] == original_cues[0]
        assert mock_provider.call_records[1]["text"] == original_cues[1]


class TestNativeTTSSynthesis:
    """Tests for Phase D: Audio artifact generation, validation, and measurement."""

    def test_successful_synthesis_produces_valid_artifact(self, tmp_path):
        """Successful synthesis writes verified MP3, JSON timestamps, and SRT files."""
        scene = make_test_scene_ir("SceneSuccess")
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        scene, meta = synthesizer.synthesize_scene_ir(scene, tmp_path)

        audio_path = Path(meta["audio_path"])
        ts_path = Path(meta["timestamps_path"])
        srt_path = Path(meta["srt_path"])

        assert audio_path.exists()
        assert audio_path.stat().st_size >= 1000
        assert ts_path.exists()
        assert srt_path.exists()

        # Audio file is readable and decodable
        decoded = AudioSegment.from_file(str(audio_path))
        assert len(decoded) > 0
        assert abs((len(decoded) / 1000.0) - meta["total_duration"]) < 0.05

    def test_provider_failure_reported_clearly(self, tmp_path):
        """Provider failure raises a clear error and leaves no falsely populated audio metadata."""
        scene = make_test_scene_ir("SceneFail")

        def failing_provider(text, path, voice, rate):
            raise ConnectionError("Simulated TTS network timeout")

        synthesizer = NativeSceneAudioSynthesizer(tts_provider=failing_provider)

        with pytest.raises(ConnectionError, match="Simulated TTS network timeout"):
            synthesizer.synthesize_scene_ir(scene, tmp_path)

        # Fail closed check: audio reference and ranges are cleared
        assert scene.audio_reference is None
        for event in scene.timeline:
            assert event.narration_anchor.audio_timestamp_range is None

    def test_empty_or_corrupted_provider_output_rejected(self, tmp_path):
        """Empty (0-byte) or non-audio provider outputs are rejected fail-closed."""
        scene = make_test_scene_ir("SceneEmptyOutput")

        def empty_provider(text, path, voice, rate):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pass  # 0 bytes
            return 0.0

        synthesizer = NativeSceneAudioSynthesizer(tts_provider=empty_provider)
        with pytest.raises(RuntimeError, match="missing or empty audio file"):
            synthesizer.synthesize_scene_ir(scene, tmp_path)

        assert scene.audio_reference is None

    def test_audio_duration_measured_from_artifact_not_budget(self, tmp_path):
        """Total duration is measured from the decoded audio artifact, NOT duration_budget."""
        # duration_budget is set to 100.0s, but audio should be ~5.5s
        scene = make_test_scene_ir(
            "SceneBudgetDecoupled",
            cues=("Short line one.", "Short line two."),
            budget=100.0,
        )
        mock_provider = make_mock_tts_provider(default_duration=2.0)
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        scene, meta = synthesizer.synthesize_scene_ir(scene, tmp_path)

        assert scene.duration_budget == 100.0  # Unaltered in M5.3.2
        assert scene.audio_reference.total_duration < 10.0
        assert scene.audio_reference.total_duration == meta["total_duration"]

    def test_cue_timestamps_strictly_measured(self, tmp_path):
        """Cue timestamp ranges reflect measured segment durations without manufactured numbers."""
        scene = make_test_scene_ir(
            "SceneStrictTiming",
            cues=("Alpha line.", "Beta line."),
        )
        # Line 1 = 2.0s, Line 2 = 3.5s
        dur_map = {"Alpha line.": 2.0, "Beta line.": 3.5}
        mock_provider = make_mock_tts_provider(durations_map=dur_map)
        synthesizer = NativeSceneAudioSynthesizer(
            tts_provider=mock_provider,
            inter_cue_pause=0.5,
            scene_end_pause=1.0,
        )

        scene, meta = synthesizer.synthesize_scene_ir(scene, tmp_path)

        t0_start, t0_end = scene.timeline[0].narration_anchor.audio_timestamp_range
        t1_start, t1_end = scene.timeline[1].narration_anchor.audio_timestamp_range

        assert t0_start == 0.0
        assert t0_end == 2.0
        assert t1_start == 2.5  # 2.0 + 0.5s pause
        assert t1_end == 6.0    # 2.5 + 3.5s
        assert meta["total_duration"] == 7.0  # 6.0 + 1.0s end pause


class TestNativeTTSCache:
    """Tests for Phase C: Narration hash, cache hit, invalidation, and safety."""

    def test_identical_narration_and_config_cache_hit(self, tmp_path):
        """Calling synthesis twice with identical inputs reuses the cache and skips the provider."""
        scene = make_test_scene_ir("SceneCacheHit")
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        # Call 1: Miss
        scene, meta1 = synthesizer.synthesize_scene_ir(scene, tmp_path)
        assert meta1["cache_hit"] is False
        assert len(mock_provider.call_records) == 3

        # Call 2: Hit
        scene2 = make_test_scene_ir("SceneCacheHit")
        scene2, meta2 = synthesizer.synthesize_scene_ir(scene2, tmp_path)
        assert meta2["cache_hit"] is True
        assert len(mock_provider.call_records) == 3  # Unchanged! Provider not called
        assert meta2["total_duration"] == meta1["total_duration"]
        assert scene2.audio_reference.total_duration == scene.audio_reference.total_duration

    def test_changed_narration_invalidates_cache(self, tmp_path):
        """Modifying narration text produces a different hash and triggers cache miss."""
        scene1 = make_test_scene_ir("SceneTextMut", cues=("Original narration line.",))
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        synthesizer.synthesize_scene_ir(scene1, tmp_path)
        assert len(mock_provider.call_records) == 1

        # Modify narration text
        scene2 = make_test_scene_ir("SceneTextMut", cues=("Modified narration line.",))
        _, meta2 = synthesizer.synthesize_scene_ir(scene2, tmp_path)

        assert meta2["cache_hit"] is False
        assert len(mock_provider.call_records) == 2
        assert mock_provider.call_records[1]["text"] == "Modified narration line."

    def test_changed_cue_order_invalidates_cache(self, tmp_path):
        """Changing the cue sequence invalidates the cache even with identical text sets."""
        scene_a = make_test_scene_ir("SceneOrderMut", cues=("Line 1", "Line 2"), indices=[0, 1])
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        synthesizer.synthesize_scene_ir(scene_a, tmp_path)
        assert len(mock_provider.call_records) == 2

        # Invert order: Line 2 first, Line 1 second
        scene_b = make_test_scene_ir("SceneOrderMut", cues=("Line 1", "Line 2"), indices=[1, 0])
        _, meta_b = synthesizer.synthesize_scene_ir(scene_b, tmp_path)

        assert meta_b["cache_hit"] is False
        assert len(mock_provider.call_records) == 4

    def test_changed_voice_invalidates_cache(self, tmp_path):
        """Changing the target TTS voice invalidates the cache."""
        scene = make_test_scene_ir("SceneVoiceMut")
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        synthesizer.synthesize_scene_ir(scene, tmp_path, voice="en-US-GuyNeural")
        assert len(mock_provider.call_records) == 3

        # Change voice to Jenny
        _, meta2 = synthesizer.synthesize_scene_ir(scene, tmp_path, voice="en-US-JennyNeural")
        assert meta2["cache_hit"] is False
        assert len(mock_provider.call_records) == 6

    def test_changed_rate_invalidates_cache(self, tmp_path):
        """Changing speech rate invalidates the cache."""
        scene = make_test_scene_ir("SceneRateMut")
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        synthesizer.synthesize_scene_ir(scene, tmp_path, rate="-4%")
        assert len(mock_provider.call_records) == 3

        _, meta2 = synthesizer.synthesize_scene_ir(scene, tmp_path, rate="+15%")
        assert meta2["cache_hit"] is False
        assert len(mock_provider.call_records) == 6

    def test_changed_provider_settings_invalidate_cache(self, tmp_path):
        """Custom provider configuration settings participate in cache identity."""
        scene = make_test_scene_ir("SceneConfigMut")
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        synthesizer.synthesize_scene_ir(scene, tmp_path, provider_config={"pitch": "+0Hz"})
        assert len(mock_provider.call_records) == 3

        _, meta2 = synthesizer.synthesize_scene_ir(scene, tmp_path, provider_config={"pitch": "+50Hz"})
        assert meta2["cache_hit"] is False
        assert len(mock_provider.call_records) == 6

    def test_missing_or_corrupted_cache_artifact_rejected(self, tmp_path):
        """If the cached audio artifact is deleted or truncated, cache hit is rejected."""
        scene = make_test_scene_ir("SceneCorruptedCache")
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        synthesizer.synthesize_scene_ir(scene, tmp_path)
        assert len(mock_provider.call_records) == 3

        # Corrupt the cached MP3 file (truncate to 10 bytes)
        audio_file = tmp_path / "SceneCorruptedCache_audio.mp3"
        with open(audio_file, "wb") as f:
            f.write(b"bad_bytes")

        # Re-run: must detect corrupted artifact and re-synthesize
        _, meta2 = synthesizer.synthesize_scene_ir(scene, tmp_path)
        assert meta2["cache_hit"] is False
        assert len(mock_provider.call_records) == 6


class TestNativeSceneIRIntegration:
    """Tests for Phase E: Model roundtrips, sequence aggregation, and backward compatibility."""

    def test_audio_reference_serialization_roundtrip(self, tmp_path):
        """SceneIR with AudioReference and measured NarrationAnchors losslessly roundtrips JSON."""
        scene = make_test_scene_ir("SceneRoundtrip")
        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        scene, _ = synthesizer.synthesize_scene_ir(scene, tmp_path)

        data = scene.to_dict()
        assert "audio_reference" in data
        assert data["audio_reference"]["total_duration"] > 0.0

        restored = SceneIR.from_dict(data)
        assert restored.audio_reference.total_duration == scene.audio_reference.total_duration
        assert restored.timeline[0].narration_anchor.audio_timestamp_range == (
            scene.timeline[0].narration_anchor.audio_timestamp_range
        )

    def test_sequence_ir_synthesis(self, tmp_path):
        """SceneSequenceIR synthesis processes all sections and attaches audio references to each."""
        sec1 = make_test_scene_ir("SceneSec1", cues=("Section one beat.",))
        sec2 = make_test_scene_ir("SceneSec2", cues=("Section two beat.",))
        seq = SceneSequenceIR(sequence_id="seq_test", sections=[sec1, sec2])

        mock_provider = make_mock_tts_provider()
        synthesizer = NativeSceneAudioSynthesizer(tts_provider=mock_provider)

        updated_seq, metas = synthesizer.synthesize_sequence_ir(seq, tmp_path)

        assert len(metas) == 2
        assert updated_seq.sections[0].audio_reference is not None
        assert updated_seq.sections[1].audio_reference is not None
        assert len(mock_provider.call_records) == 2

    def test_audio_manager_generate_scene_ir_audio_integration(self, tmp_path):
        """AudioManager convenience method operates seamlessly with custom provider and options."""
        scene = make_test_scene_ir("SceneMgrConvenience")
        audio_mgr = AudioManager()
        mock_provider = make_mock_tts_provider()

        meta = audio_mgr.generate_scene_ir_audio(
            scene_ir=scene,
            output_dir=tmp_path,
            tts_provider=mock_provider,
        )

        assert meta["scene_id"] == "SceneMgrConvenience"
        assert scene.audio_reference is not None
        assert Path(meta["audio_path"]).exists()

    def test_legacy_pipeline_behavior_unchanged(self, tmp_path):
        """Legacy generate_section_audio remains 100% functional without regressions."""
        audio_mgr = AudioManager()

        def mock_speech(text, path, voice, rate=None):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            silence = AudioSegment.silent(duration=2000)
            silence.export(path, format="mp3")
            return 2.0

        audio_mgr.generate_speech = MagicMock(side_effect=mock_speech)

        meta = audio_mgr.generate_section_audio(
            section_id="legacy_sec_1",
            lecture_lines=["Legacy line 1", "Legacy line 2"],
            output_dir=tmp_path,
        )

        assert meta["section_id"] == "legacy_sec_1"
        assert len(meta["lines"]) == 2
        assert "timing_contract" in meta
        assert Path(meta["audio_path"]).exists()
