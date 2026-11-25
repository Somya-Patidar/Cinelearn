"""
Milestone 5.3.1 Unit Tests: Native Audio Contract and Scene IR Timing Foundation.

Verifies:
1. Narration extraction from native SceneIR and SceneSequenceIR.
2. Deterministic narration ordering (by lecture_line_index and event position).
3. Duplicate narration indices resolved stably by timeline position.
4. Missing narration anchors (skipped gracefully without crashing).
5. Empty / whitespace-only narration cues (filtered out).
6. Native SceneTimingContract.from_scene_ir() construction without storyboard adapters.
7. Preservation of scene and event identities across segments and actions.
8. Distinguishing missing audio duration (actual_audio_duration is None) from measured audio.
9. Strict validation of AudioReference and NarrationAnchor (rejection of bools, NaNs, non-positives).
10. Lossless dictionary serialization and deserialization roundtrip.
11. Deterministic narration hash generation via compute_narration_hash and compute_scene_ir_narration_hash.
12. Cache identity variation on synthesis configuration changes (voice, rate, etc.).
"""

import math
import pytest
from pydantic import ValidationError

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
from src.timing_contract import (
    SceneTimingContract,
    NarrationSegmentTiming,
    VisualActionTiming,
)
from src.audio_handoff_validator import (
    compute_narration_hash,
    compute_scene_ir_narration_hash,
)


class TestNativeNarrationExtraction:
    """Tests for Scope A: Deterministic narration extraction from native Scene IR."""

    def test_basic_narration_extraction(self):
        """Extracts valid, non-empty narration cues in deterministic order."""
        scene = SceneIR(
            scene_id="SceneBasic",
            duration_budget=10.0,
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="First concept explanation.",
                        audio_timestamp_range=(0.0, 3.0),
                    ),
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=1,
                        text_cue="Second concept explanation.",
                        audio_timestamp_range=(3.0, 6.0),
                    ),
                ),
            ],
        )

        cues = scene.get_ordered_narration_cues()
        assert cues == ["First concept explanation.", "Second concept explanation."]
        assert scene.narration_cues == cues

    def test_deterministic_ordering_out_of_order_indices(self):
        """Orders cues by lecture_line_index regardless of timeline declaration order."""
        scene = SceneIR(
            scene_id="SceneOutOfOrder",
            duration_budget=10.0,
            timeline=[
                TimelineEvent(
                    step_id="step_b",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=2,
                        text_cue="Line 2 (Third in lecture)",
                        audio_timestamp_range=(5.0, 8.0),
                    ),
                ),
                TimelineEvent(
                    step_id="step_a",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Line 0 (First in lecture)",
                        audio_timestamp_range=(0.0, 2.5),
                    ),
                ),
                TimelineEvent(
                    step_id="step_c",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=1,
                        text_cue="Line 1 (Second in lecture)",
                        audio_timestamp_range=(2.5, 5.0),
                    ),
                ),
            ],
        )

        cues = scene.get_ordered_narration_cues()
        assert cues == [
            "Line 0 (First in lecture)",
            "Line 1 (Second in lecture)",
            "Line 2 (Third in lecture)",
        ]

    def test_duplicate_narration_indices_stable_tie_break(self):
        """Duplicate lecture_line_index values are resolved stably by timeline event position."""
        scene = SceneIR(
            scene_id="SceneDuplicateIndices",
            duration_budget=10.0,
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Introduction part A",
                        audio_timestamp_range=(0.0, 2.0),
                    ),
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Introduction part B",
                        audio_timestamp_range=(2.0, 4.0),
                    ),
                ),
                TimelineEvent(
                    step_id="step_3",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=1,
                        text_cue="Follow-up explanation",
                        audio_timestamp_range=(4.0, 6.0),
                    ),
                ),
            ],
        )

        cues = scene.get_ordered_narration_cues()
        assert cues == [
            "Introduction part A",
            "Introduction part B",
            "Follow-up explanation",
        ]

    def test_missing_and_empty_narration_anchors_filtered(self):
        """Omits steps with missing anchors or empty/whitespace-only text cues."""
        scene = SceneIR(
            scene_id="SceneFilteredAnchors",
            duration_budget=12.0,
            timeline=[
                TimelineEvent(step_id="silent_enter_step"),
                TimelineEvent(
                    step_id="valid_step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Spoken narration line 1",
                        audio_timestamp_range=(1.0, 4.0),
                    ),
                ),
                TimelineEvent(
                    step_id="whitespace_step",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=1,
                        text_cue="    \t \n ",
                        audio_timestamp_range=(4.0, 5.0),
                    ),
                ),
                TimelineEvent(
                    step_id="empty_step",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=2,
                        text_cue="",
                        audio_timestamp_range=(5.0, 6.0),
                    ),
                ),
                TimelineEvent(
                    step_id="valid_step_2",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=3,
                        text_cue="Spoken narration line 2",
                        audio_timestamp_range=(6.0, 9.0),
                    ),
                ),
            ],
        )

        cues = scene.get_ordered_narration_cues()
        assert cues == ["Spoken narration line 1", "Spoken narration line 2"]

    def test_sequence_level_narration_aggregation(self):
        """SceneSequenceIR aggregates cues across multiple scenes in presentation order."""
        s1 = SceneIR(
            scene_id="Scene1",
            duration_budget=5.0,
            timeline=[
                TimelineEvent(
                    step_id="s1_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Section 1 narration",
                        audio_timestamp_range=(0.0, 3.0),
                    ),
                )
            ],
        )
        s2 = SceneIR(
            scene_id="Scene2",
            duration_budget=5.0,
            timeline=[
                TimelineEvent(
                    step_id="s2_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Section 2 narration",
                        audio_timestamp_range=(0.0, 3.0),
                    ),
                )
            ],
        )
        seq = SceneSequenceIR(
            sequence_id="Seq_Multi",
            title="Presentation",
            sections=[s1, s2],
        )

        assert seq.get_ordered_narration_cues() == [
            "Section 1 narration",
            "Section 2 narration",
        ]
        assert seq.narration_cues == [
            "Section 1 narration",
            "Section 2 narration",
        ]


class TestNativeSceneTimingContract:
    """Tests for Scope B: Native SceneTimingContract construction from SceneIR."""

    def test_from_scene_ir_unmeasured_audio(self):
        """Constructs contract without audio_reference: actual_audio_duration is None."""
        scene = SceneIR(
            scene_id="BinarySearchIntro",
            title="Introduction to Binary Search",
            duration_budget=15.0,
            entities=[
                EntityDeclaration(
                    id="arr",
                    component_type="ArrayVisual",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="viewport_main",
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_init",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Binary search operates on sorted collections.",
                        audio_timestamp_range=(0.0, 4.5),
                    ),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.ENTER,
                            target_entity="arr",
                            duration=1.2,
                        )
                    ],
                ),
                TimelineEvent(
                    step_id="step_silent_transition",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.COMPONENT_OP,
                            target_entity="arr",
                            operation="highlight_cell",
                            parameters={"index": 2},
                            duration=0.8,
                        )
                    ],
                ),
            ],
        )

        contract = SceneTimingContract.from_scene_ir(scene)
        assert contract.section_id == "BinarySearchIntro"
        assert contract.title == "Introduction to Binary Search"
        assert contract.total_assembled_duration == 15.0
        assert contract.total_audio_duration == 0.0
        assert not contract.is_synchronized
        assert contract.audio_path is None

        # Verify segments
        assert len(contract.segments) == 2
        seg0 = contract.segments[0]
        assert seg0.segment_id == "BinarySearchIntro_step_init"
        assert seg0.narration_text == "Binary search operates on sorted collections."
        assert seg0.actual_audio_duration is None  # Never invent measured audio!
        assert seg0.estimated_duration == 4.5
        assert len(seg0.visual_actions) == 1
        assert seg0.visual_actions[0].component_name == "ArrayVisual"
        assert seg0.visual_actions[0].viewport == "viewport_main"

        seg1 = contract.segments[1]
        assert seg1.segment_id == "BinarySearchIntro_step_silent_transition"
        assert seg1.narration_text == ""  # Silent step
        assert seg1.actual_audio_duration is None
        assert seg1.estimated_duration == 0.8
        assert len(seg1.visual_actions) == 1
        assert seg1.visual_actions[0].effect == "highlight_cell"

    def test_from_scene_ir_with_measured_audio(self):
        """Constructs contract with populated AudioReference: marks contract synchronized."""
        scene = SceneIR(
            scene_id="SJFScene",
            title="Shortest Job First",
            duration_budget=20.0,
            audio_reference=AudioReference(
                audio_path="/path/to/SJFScene_audio.mp3",
                timestamps_path="/path/to/SJFScene_timestamps.json",
                total_duration=18.5,
            ),
            entities=[
                EntityDeclaration(
                    id="gantt",
                    component_type="GanttChartVisual",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="viewport_split_right",
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_p1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Process P1 runs first.",
                        audio_timestamp_range=(0.0, 5.0),
                    ),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.COMPONENT_OP,
                            target_entity="gantt",
                            operation="animate_add_block",
                            parameters={"pid": "P1"},
                            duration=0.8,
                        )
                    ],
                )
            ],
        )

        contract = SceneTimingContract.from_scene_ir(scene)
        assert contract.is_synchronized
        assert contract.total_audio_duration == 18.5
        assert contract.audio_path == "/path/to/SJFScene_audio.mp3"
        assert contract.segments[0].actual_audio_duration == 5.0

    def test_contract_serialization_roundtrip(self):
        """Verifies lossless roundtrip through to_dict and from_dict."""
        scene = SceneIR(
            scene_id="RoundtripScene",
            title="Roundtrip Test",
            duration_budget=10.0,
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Hello roundtrip",
                        audio_timestamp_range=(0.0, 3.0),
                    ),
                )
            ],
        )

        contract = SceneTimingContract.from_scene_ir(scene)
        data = contract.to_dict()
        reconstructed = SceneTimingContract.from_dict(data)

        assert reconstructed.section_id == contract.section_id
        assert reconstructed.title == contract.title
        assert len(reconstructed.segments) == len(contract.segments)
        assert reconstructed.segments[0].narration_text == contract.segments[0].narration_text
        assert reconstructed.segments[0].timeline_start == contract.segments[0].timeline_start
        assert reconstructed.segments[0].timeline_end == contract.segments[0].timeline_end

    def test_from_scene_ir_rejects_invalid_type(self):
        """Raises TypeError if input is neither SceneIR nor dict."""
        with pytest.raises(TypeError, match="Expected SceneIR or dict"):
            SceneTimingContract.from_scene_ir(12345)


class TestAudioValidationContract:
    """Tests for Scope C: Strict Pydantic validation on audio data structures."""

    def test_audio_reference_valid(self):
        """Accepts valid paths and positive finite durations."""
        ref = AudioReference(
            audio_path="media/audio.mp3",
            timestamps_path="media/timestamps.json",
            total_duration=12.345,
        )
        assert ref.total_duration == 12.345
        assert ref.audio_path == "media/audio.mp3"

    def test_audio_reference_rejects_empty_and_whitespace_paths(self):
        """Rejects empty string or whitespace-only paths."""
        with pytest.raises(ValidationError):
            AudioReference(
                audio_path="",
                timestamps_path="valid.json",
                total_duration=5.0,
            )
        with pytest.raises(ValidationError):
            AudioReference(
                audio_path="   ",
                timestamps_path="valid.json",
                total_duration=5.0,
            )

    def test_audio_reference_rejects_non_positive_and_bool_durations(self):
        """Rejects zero, negative, boolean, and non-finite durations."""
        with pytest.raises(ValidationError):
            AudioReference(audio_path="a.mp3", timestamps_path="a.json", total_duration=0.0)
        with pytest.raises(ValidationError):
            AudioReference(audio_path="a.mp3", timestamps_path="a.json", total_duration=-1.5)
        with pytest.raises(ValidationError):
            AudioReference(audio_path="a.mp3", timestamps_path="a.json", total_duration=True)
        with pytest.raises(ValidationError):
            AudioReference(audio_path="a.mp3", timestamps_path="a.json", total_duration=float("nan"))
        with pytest.raises(ValidationError):
            AudioReference(audio_path="a.mp3", timestamps_path="a.json", total_duration=float("inf"))

    def test_narration_anchor_rejects_booleans_and_invalids(self):
        """Rejects boolean values in lecture_line_index and timestamp ranges."""
        with pytest.raises(ValidationError):
            NarrationAnchor(lecture_line_index=True, audio_timestamp_range=(0.0, 1.0))
        with pytest.raises(ValidationError):
            NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(True, 2.0))
        with pytest.raises(ValidationError):
            NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, False))
        with pytest.raises(ValidationError):
            NarrationAnchor(lecture_line_index=-1, audio_timestamp_range=(0.0, 1.0))
        with pytest.raises(ValidationError):
            NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(5.0, 2.0))  # Inverted

    def test_scene_ir_duration_budget_strict_validation(self):
        """Rejects boolean, non-positive, and non-finite duration_budget values."""
        with pytest.raises(ValidationError):
            SceneIR(scene_id="S", duration_budget=True)
        with pytest.raises(ValidationError):
            SceneIR(scene_id="S", duration_budget=0.0)
        with pytest.raises(ValidationError):
            SceneIR(scene_id="S", duration_budget=-5.0)
        with pytest.raises(ValidationError):
            SceneIR(scene_id="S", duration_budget=float("nan"))


class TestNarrationCacheIdentity:
    """Tests for Scope D: Deterministic narration caching and identity evaluation."""

    def test_hash_identical_for_identical_text(self):
        """Produces identical hash for identical lines."""
        h1 = compute_narration_hash(["First line", "Second line"])
        h2 = compute_narration_hash(["First line", "Second line"])
        assert h1 == h2
        assert len(h1) == 16

    def test_hash_changes_on_text_modification(self):
        """Altering narration text alters the hash."""
        h1 = compute_narration_hash(["First line", "Second line"])
        h2 = compute_narration_hash(["First line", "Modified line"])
        assert h1 != h2

    def test_hash_changes_on_order_modification(self):
        """Reordering lines alters the hash."""
        h1 = compute_narration_hash(["First line", "Second line"])
        h2 = compute_narration_hash(["Second line", "First line"])
        assert h1 != h2

    def test_hash_changes_on_synthesis_config_modification(self):
        """Synthesis configuration change (e.g. voice, rate) alters the hash."""
        lines = ["Explain dynamic programming."]
        h_default = compute_narration_hash(lines)
        h_voice1 = compute_narration_hash(lines, audio_config={"voice": "en-US-JennyNeural"})
        h_voice2 = compute_narration_hash(lines, audio_config={"voice": "en-US-GuyNeural"})
        h_rate = compute_narration_hash(lines, audio_config={"voice": "en-US-JennyNeural", "rate": "+10%"})

        assert h_default != h_voice1
        assert h_voice1 != h_voice2
        assert h_voice1 != h_rate

    def test_compute_scene_ir_narration_hash(self):
        """Extracts cues directly from SceneIR and hashes deterministically."""
        scene = SceneIR(
            scene_id="SceneHashTest",
            duration_budget=10.0,
            timeline=[
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=1,
                        text_cue="Step two narration",
                        audio_timestamp_range=(2.0, 4.0),
                    ),
                ),
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Step one narration",
                        audio_timestamp_range=(0.0, 2.0),
                    ),
                ),
            ],
        )

        expected_hash = compute_narration_hash(["Step one narration", "Step two narration"])
        actual_hash = compute_scene_ir_narration_hash(scene)
        assert actual_hash == expected_hash

        # Voice config change changes SceneIR hash
        voiced_hash = compute_scene_ir_narration_hash(scene, audio_config={"voice": "en-GB-SoniaNeural"})
        assert voiced_hash != actual_hash
