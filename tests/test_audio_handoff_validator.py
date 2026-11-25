"""
Integration tests for CineLearn Audio and Render Handoff Validation (Step 2B.4C).

Covers the 12 required scenarios:
1. Validated storyboard fingerprint remains consistent through handoff.
2. Storyboard mutation after validation is detected.
3. Stale storyboard/audio combinations are rejected.
4. Missing audio section blocks progression.
5. Empty or invalid audio files are rejected.
6. Narration changes invalidate associated audio.
7. Correct section ordering is preserved.
8. Stage 3 cannot run when the audio gate fails.
9. Rendering cannot run when required handoff validation fails.
10. Checkpoint restoration revalidates the required artifacts.
11. Existing valid checkpoint workflows still work.
12. Existing numerical fidelity gate remains enforced.
"""

import json
from pathlib import Path
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

from agent import TeachingVideoAgent, TeachingOutline, Section, RunConfig
from audio_handoff_validator import (
    AudioHandoffValidator,
    AudioHandoffReport,
    HandoffStatus,
    HandoffIssueCategory,
    compute_storyboard_fingerprint,
    compute_narration_hash,
)
from numerical_evaluator import VerificationStatus
from storyboard_fidelity_checker import (
    StoryboardFidelityChecker,
    StoryboardFidelityReport,
    FidelityCategory,
)
from audio_manager import AudioManager


class TestAudioHandoffValidator:
    @pytest.fixture(autouse=True)
    def setup_tmp(self, tmp_path):
        self.tmp_dir = tmp_path
        self.agent_dir = tmp_path / "CASES" / "Arrhenius_Equation"
        self.agent_dir.mkdir(parents=True, exist_ok=True)

    def _create_agent(self, topic="Arrhenius Equation", use_audio=True):
        cfg = RunConfig(level="intermediate", audio=use_audio)
        agent = TeachingVideoAgent(
            idx=1,
            knowledge_point=topic,
            folder=str(self.tmp_dir / "CASES"),
            cfg=cfg,
        )
        return agent

    def _setup_verified_agent(self, agent, sections_count=2):
        """Helper to configure an agent with a verified Stage 1 outline and Stage 2 sections."""
        agent.outline = TeachingOutline.from_dict({
            "topic": "Arrhenius Equation",
            "learning_level": "intermediate",
            "sections": [{"id": f"section_{i+1}", "title": f"Title {i+1}"} for i in range(sections_count)],
            "worked_example_spec": {
                "inputs": {"Ea": 75000, "R": 8.314, "T": 300, "A": 7.5e10},
                "steps": [
                    {
                        "step": "Calculate exponent: -Ea / (R * T)",
                        "result": "-75000 / (8.314 * 300) = -30.07",
                    },
                    {
                        "step": "Compute k1 = A * exp(-30.07)",
                        "result": "7.5e10 * exp(-30.07) = 0.006545 s^-1",
                    },
                ],
                "final_result": "k1 = 0.006545 s^-1",
            },
        })
        agent._ensure_outline_verified()

        sections = []
        for i in range(sections_count):
            sid = f"section_{i+1}"
            lines = [f"This is section {i+1} discussing reaction rate.", "Notice the temperature dependence."]
            if i == 0:
                vvals = {"k1": "0.006545 s^-1"}
            else:
                vvals = {}
            sec = Section(
                id=sid,
                title=f"Section {i+1}",
                lecture_lines=lines,
                animations=[f"Show diagram {i+1}"],
                story_beat="Concept",
                captions=[f"Caption {i+1}"],
                layout_plan="standard",
                verified_values=vvals,
            )
            sections.append(sec)

        agent.sections = sections
        # Run fidelity check to establish verified state
        agent._ensure_storyboard_verified()
        agent.validated_storyboard_fingerprint = compute_storyboard_fingerprint(agent.sections)
        return agent

    def _create_mock_audio_file(self, file_path: Path, duration_ms: int = 1000):
        """Creates a valid decodable mp3 file using pydub silent segment."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        seg = AudioSegment.silent(duration=duration_ms)
        seg.export(str(file_path), format="mp3")

    def _create_section_audio_artifacts(self, agent, sid: str, lines: list):
        """Creates on-disk audio mp3 and timestamps json with narration hash."""
        audio_path = agent.output_dir / f"{sid}_audio.mp3"
        self._create_mock_audio_file(audio_path, duration_ms=1200)

        narr_hash = compute_narration_hash(lines)
        ts_data = {
            "section_id": sid,
            "total_duration": 1.2,
            "narration_hash": narr_hash,
            "lines": [
                {"text": l, "raw_text": l, "start": 0.0, "end": 0.6} for l in lines
            ],
            "audio_path": str(audio_path),
        }
        ts_path = agent.output_dir / f"{sid}_timestamps.json"
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump(ts_data, f, indent=2)

        agent.section_audios[sid] = str(audio_path)
        agent.section_timestamps[sid] = ts_data

    # -------------------------------------------------------------------------
    # Test 1: Validated storyboard fingerprint remains consistent through handoff
    # -------------------------------------------------------------------------
    def test_1_validated_storyboard_fingerprint_remains_consistent_through_handoff(self):
        """1. Validated storyboard fingerprint remains consistent through handoff."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        initial_fp = agent.validated_storyboard_fingerprint
        assert initial_fp is not None
        assert initial_fp.startswith("sb_")

        # Set up valid audio artifacts for both sections
        for s in agent.sections:
            self._create_section_audio_artifacts(agent, s.id, s.lecture_lines)

        report = agent._ensure_audio_verified()
        assert report is not None
        assert report.overall_status == HandoffStatus.VALIDATED
        assert report.is_safe_for_stage3 is True
        assert report.storyboard_fingerprint == initial_fp
        assert agent.is_audio_safe_for_stage3() is True

    # -------------------------------------------------------------------------
    # Test 2: Storyboard mutation after validation is detected
    # -------------------------------------------------------------------------
    def test_2_storyboard_mutation_after_validation_is_detected(self):
        """2. Storyboard mutation after validation is detected."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        # Set up audio artifacts for original sections
        for s in agent.sections:
            self._create_section_audio_artifacts(agent, s.id, s.lecture_lines)

        # Mutate the storyboard after validation was established
        agent.sections[0].lecture_lines.append("An unauthorized mutated narration line!")

        # Gate must detect mutation and report failure
        assert agent.is_storyboard_safe_for_audio() is False
        assert agent.is_audio_safe_for_stage3() is False

        report = agent._ensure_audio_verified(force=True)
        assert report.overall_status == HandoffStatus.FAILED
        assert any(i.category == HandoffIssueCategory.MUTATED_STORYBOARD for i in report.issues)

    # -------------------------------------------------------------------------
    # Test 3: Stale storyboard/audio combinations are rejected
    # -------------------------------------------------------------------------
    def test_3_stale_storyboard_audio_combinations_are_rejected(self):
        """3. Stale storyboard/audio combinations are rejected."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        # Set up audio with old narration
        old_lines = ["Old narration line 1", "Old narration line 2"]
        self._create_section_audio_artifacts(agent, "section_1", old_lines)
        self._create_section_audio_artifacts(agent, "section_2", agent.sections[1].lecture_lines)

        # Re-establish storyboard fingerprint with current lines (so fingerprint itself matches)
        agent.validated_storyboard_fingerprint = compute_storyboard_fingerprint(agent.sections)

        # Section 1 audio timestamp has old narration hash
        report = agent._ensure_audio_verified(force=True)
        assert report.overall_status == HandoffStatus.FAILED
        assert report.is_safe_for_stage3 is False
        assert any(i.category == HandoffIssueCategory.STALE_AUDIO and i.section_id == "section_1" for i in report.issues)

    # -------------------------------------------------------------------------
    # Test 4: Missing audio section blocks progression
    # -------------------------------------------------------------------------
    def test_4_missing_audio_section_blocks_progression(self):
        """4. Missing audio section blocks progression."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        # Only configure section_1; section_2 is missing
        self._create_section_audio_artifacts(agent, "section_1", agent.sections[0].lecture_lines)

        assert agent.is_audio_safe_for_stage3() is False
        report = agent._ensure_audio_verified(force=True)
        assert report.overall_status == HandoffStatus.FAILED
        assert any(i.category == HandoffIssueCategory.MISSING_AUDIO_SECTION and i.section_id == "section_2" for i in report.issues)

    # -------------------------------------------------------------------------
    # Test 5: Empty or invalid audio files are rejected
    # -------------------------------------------------------------------------
    def test_5_empty_or_invalid_audio_files_are_rejected(self):
        """5. Empty or invalid audio files are rejected."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        # Section 1: Valid
        self._create_section_audio_artifacts(agent, "section_1", agent.sections[0].lecture_lines)

        # Section 2: Empty (0 bytes) file
        empty_audio = agent.output_dir / "section_2_audio.mp3"
        empty_audio.write_bytes(b"")
        ts_data = {
            "section_id": "section_2",
            "total_duration": 1.0,
            "narration_hash": compute_narration_hash(agent.sections[1].lecture_lines),
            "lines": [{"text": l, "raw_text": l} for l in agent.sections[1].lecture_lines],
            "audio_path": str(empty_audio),
        }
        with open(agent.output_dir / "section_2_timestamps.json", "w", encoding="utf-8") as f:
            json.dump(ts_data, f)
        agent.section_audios["section_2"] = str(empty_audio)
        agent.section_timestamps["section_2"] = ts_data

        assert agent.is_audio_safe_for_stage3() is False
        report = agent._ensure_audio_verified(force=True)
        assert report.overall_status == HandoffStatus.FAILED
        assert any(i.category == HandoffIssueCategory.INVALID_AUDIO_FILE and i.section_id == "section_2" for i in report.issues)

    # -------------------------------------------------------------------------
    # Test 6: Narration changes invalidate associated audio
    # -------------------------------------------------------------------------
    def test_6_narration_changes_invalidate_associated_audio(self):
        """6. Narration changes invalidate associated audio in audio_manager."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=1)

        sec = agent.sections[0]
        initial_lines = list(sec.lecture_lines)
        self._create_section_audio_artifacts(agent, sec.id, initial_lines)

        # Verify initial cache is accepted by audio_manager
        cached = agent.audio_manager.generate_section_audio(
            section_id=sec.id,
            lecture_lines=initial_lines,
            animations=sec.animations,
            output_dir=agent.output_dir,
        )
        assert cached.get("narration_hash") == compute_narration_hash(initial_lines)

        # Now change narration lines
        new_lines = ["Completely new narration line about catalysis."]
        new_hash = compute_narration_hash(new_lines)

        # Mock the TTS engine call so audio_manager actually regenerates instead of calling edge_tts
        def mock_speech(text, output_path, voice=None):
            self._create_mock_audio_file(Path(output_path), 1000)
            return 1.0

        with patch.object(agent.audio_manager, "generate_speech", side_effect=mock_speech):
            regenerated = agent.audio_manager.generate_section_audio(
                section_id=sec.id,
                lecture_lines=new_lines,
                animations=sec.animations,
                output_dir=agent.output_dir,
            )

        assert regenerated["narration_hash"] == new_hash
        assert regenerated["narration_hash"] != compute_narration_hash(initial_lines)

    # -------------------------------------------------------------------------
    # Test 7: Correct section ordering is preserved
    # -------------------------------------------------------------------------
    def test_7_correct_section_ordering_is_preserved(self):
        """7. Correct section ordering is preserved and identifier mismatch is flagged."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        # Section 1 has section 2's metadata
        self._create_section_audio_artifacts(agent, "section_1", agent.sections[0].lecture_lines)
        self._create_section_audio_artifacts(agent, "section_2", agent.sections[1].lecture_lines)

        # Invert metadata IDs
        agent.section_timestamps["section_1"]["section_id"] = "section_2"

        report = agent._ensure_audio_verified(force=True)
        assert report.overall_status == HandoffStatus.FAILED
        assert any(i.category == HandoffIssueCategory.SECTION_ORDER_MISMATCH for i in report.issues)

    # -------------------------------------------------------------------------
    # Test 8: Stage 3 cannot run when the audio gate fails
    # -------------------------------------------------------------------------
    def test_8_stage3_cannot_run_when_audio_gate_fails(self):
        """8. Stage 3 cannot run when the audio gate fails."""
        agent = self._create_agent(use_audio=True)
        self._setup_verified_agent(agent, sections_count=2)

        # Missing audio for sections
        assert agent.is_audio_safe_for_stage3() is False

        with pytest.raises(ValueError, match="Cannot generate code"):
            agent.generate_codes()

        with pytest.raises(ValueError, match="Cannot generate code"):
            agent.generate_section_code(agent.sections[0])

    # -------------------------------------------------------------------------
    # Test 9: Rendering cannot run when required handoff validation fails
    # -------------------------------------------------------------------------
    def test_9_rendering_cannot_run_when_required_handoff_validation_fails(self):
        """9. Rendering cannot run when required handoff validation fails."""
        agent = self._create_agent(use_audio=True)
        self._setup_verified_agent(agent, sections_count=2)

        # Audio not generated / gate failing
        with pytest.raises(ValueError, match="Cannot render section"):
            agent.render_section(agent.sections[0])

        with pytest.raises(ValueError, match="Cannot render sections"):
            agent.render_all_sections()

        with pytest.raises(ValueError, match="Cannot merge videos"):
            agent.merge_videos()

    # Test 10: Checkpoint restoration revalidates the required artifacts
    # -------------------------------------------------------------------------
    def test_10_checkpoint_restoration_revalidates_required_artifacts(self):
        """10. Checkpoint restoration revalidates the required artifacts."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        for s in agent.sections:
            self._create_section_audio_artifacts(agent, s.id, s.lecture_lines)

        agent.save_checkpoint("audio_completed")

        # Now corrupt one of the audio files on disk
        corrupt_audio = agent.output_dir / "section_1_audio.mp3"
        corrupt_audio.write_bytes(b"corrupt junk data that is not an mp3")

        # Restore new agent instance from the checkpoint
        new_agent = self._create_agent()
        new_agent._load_existing_checkpoint_and_files()

        assert len(new_agent.sections) == 2
        # is_audio_safe_for_stage3 must detect the corrupted audio on disk upon revalidation
        assert new_agent.is_audio_safe_for_stage3() is False

    # -------------------------------------------------------------------------
    # Test 11: Existing valid checkpoint workflows still work
    # -------------------------------------------------------------------------
    def test_11_existing_valid_checkpoint_workflows_still_work(self):
        """11. Existing valid checkpoint workflows still work."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        for s in agent.sections:
            self._create_section_audio_artifacts(agent, s.id, s.lecture_lines)

        agent.save_checkpoint("audio_completed")

        # Restore new agent instance from the checkpoint
        new_agent = self._create_agent()
        new_agent._load_existing_checkpoint_and_files()

        assert len(new_agent.sections) == 2
        assert new_agent.is_storyboard_safe_for_audio() is True
        assert new_agent.is_audio_safe_for_stage3() is True

    # -------------------------------------------------------------------------
    # Test 12: Existing numerical fidelity gate remains enforced
    # -------------------------------------------------------------------------
    def test_12_existing_numerical_fidelity_gate_remains_enforced(self):
        """12. Existing numerical fidelity gate remains enforced in handoff validation."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        # Set up valid audio
        for s in agent.sections:
            self._create_section_audio_artifacts(agent, s.id, s.lecture_lines)

        # Create a mock failed fidelity report
        failed_fidelity_rep = StoryboardFidelityReport(
            overall_status=VerificationStatus.FAILED,
            total_checks=2,
            verified_claims=0,
            summary="Numerical fidelity failed: wrong calculation value.",
            is_safe_for_audio=False,
        )
        agent.storyboard_fidelity_report = failed_fidelity_rep

        report = agent.audio_handoff_validator.validate_handoff(
            sections=agent.sections,
            section_audios=agent.section_audios,
            section_timestamps=agent.section_timestamps,
            expected_fingerprint=agent.validated_storyboard_fingerprint,
            fidelity_report=failed_fidelity_rep,
        )

        assert report.overall_status == HandoffStatus.FAILED
        assert report.is_safe_for_stage3 is False
        assert any(i.category == HandoffIssueCategory.MISSING_STORYBOARD_GATE for i in report.issues)

    # -------------------------------------------------------------------------
    # Test 13: Animation changes with unchanged narration changes fingerprint
    # -------------------------------------------------------------------------
    def test_13_animation_changes_with_unchanged_narration_changes_fingerprint(self):
        """13. Modifying animation instructions while keeping narration identical alters fingerprint."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        initial_fp = agent.validated_storyboard_fingerprint
        assert initial_fp is not None

        # Change only animation instructions, narration remains identical
        agent.sections[0].animations = ["A completely different 3D transformation animation."]
        new_fp = compute_storyboard_fingerprint(agent.sections)

        assert new_fp != initial_fp
        assert agent.is_storyboard_safe_for_audio() is False

    # -------------------------------------------------------------------------
    # Test 14: Section reordering changes fingerprint
    # -------------------------------------------------------------------------
    def test_14_section_reordering_changes_fingerprint(self):
        """14. Reordering storyboard sections alters fingerprint and trips mutation gate."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        initial_fp = agent.validated_storyboard_fingerprint

        # Swap section order
        agent.sections = [agent.sections[1], agent.sections[0]]
        new_fp = compute_storyboard_fingerprint(agent.sections)

        assert new_fp != initial_fp
        assert agent.is_storyboard_safe_for_audio() is False

    # -------------------------------------------------------------------------
    # Test 15: Layout-plan changes alters fingerprint
    # -------------------------------------------------------------------------
    def test_15_layout_plan_changes_changes_fingerprint(self):
        """15. Modifying layout_plan changes fingerprint and blocks unverified downstream execution."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        initial_fp = agent.validated_storyboard_fingerprint
        agent.sections[0].layout_plan = "dock_split_left"
        new_fp = compute_storyboard_fingerprint(agent.sections)

        assert new_fp != initial_fp
        assert agent.is_storyboard_safe_for_audio() is False

    # -------------------------------------------------------------------------
    # Test 16: Story-beat changes alters fingerprint
    # -------------------------------------------------------------------------
    def test_16_story_beat_changes_changes_fingerprint(self):
        """16. Modifying story_beat changes fingerprint and blocks unverified downstream execution."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        initial_fp = agent.validated_storyboard_fingerprint
        agent.sections[0].story_beat = "DramaticClimax"
        new_fp = compute_storyboard_fingerprint(agent.sections)

        assert new_fp != initial_fp
        assert agent.is_storyboard_safe_for_audio() is False

    # -------------------------------------------------------------------------
    # Test 17: Duration consistency checks and discrepancy rejection
    # -------------------------------------------------------------------------
    def test_17_duration_consistency_checks_and_discrepancy_rejection(self):
        """17. Decoded audio duration is reconciled with metadata duration within tolerance."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=1)

        sec = agent.sections[0]
        # Audio is 1.2s
        self._create_section_audio_artifacts(agent, sec.id, sec.lecture_lines)

        # Case A: Exact match (1.2s audio, 1.2s metadata) -> Validated
        rep = agent.audio_handoff_validator.validate_handoff(
            sections=agent.sections,
            section_audios=agent.section_audios,
            section_timestamps=agent.section_timestamps,
            expected_fingerprint=agent.validated_storyboard_fingerprint,
        )
        assert rep.overall_status == HandoffStatus.VALIDATED

        # Case B: Small acceptable MP3 rounding difference (1.2s audio, 1.25s metadata; diff = 0.05s <= 0.5s) -> Validated
        agent.section_timestamps[sec.id]["total_duration"] = 1.25
        rep_rounding = agent.audio_handoff_validator.validate_handoff(
            sections=agent.sections,
            section_audios=agent.section_audios,
            section_timestamps=agent.section_timestamps,
            expected_fingerprint=agent.validated_storyboard_fingerprint,
        )
        assert rep_rounding.overall_status == HandoffStatus.VALIDATED

        # Case C: Substantial duration mismatch (1.2s audio, 3.5s metadata; diff = 2.3s > 0.5s) -> Failed
        agent.section_timestamps[sec.id]["total_duration"] = 3.5
        rep_mismatch = agent.audio_handoff_validator.validate_handoff(
            sections=agent.sections,
            section_audios=agent.section_audios,
            section_timestamps=agent.section_timestamps,
            expected_fingerprint=agent.validated_storyboard_fingerprint,
        )
        assert rep_mismatch.overall_status == HandoffStatus.FAILED
        assert any(
            i.category == HandoffIssueCategory.INVALID_AUDIO_FILE and "disagrees with metadata" in i.message
            for i in rep_mismatch.issues
        )

    # -------------------------------------------------------------------------
    # Test 18: Corrupt duration metadata handling
    # -------------------------------------------------------------------------
    def test_18_corrupt_duration_metadata_handling(self):
        """18. None, non-numeric, NaN, inf, and negative durations fail closed with explicit issues."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=1)
        sec = agent.sections[0]
        self._create_section_audio_artifacts(agent, sec.id, sec.lecture_lines)

        # None duration
        agent.section_timestamps[sec.id]["total_duration"] = None
        rep_none = agent.audio_handoff_validator.validate_handoff(
            sections=agent.sections,
            section_audios=agent.section_audios,
            section_timestamps=agent.section_timestamps,
            expected_fingerprint=agent.validated_storyboard_fingerprint,
        )
        assert rep_none.overall_status == HandoffStatus.FAILED
        assert any(i.category == HandoffIssueCategory.UNREADABLE_METADATA for i in rep_none.issues)

        # Non-numeric duration
        agent.section_timestamps[sec.id]["total_duration"] = "not_a_number"
        rep_str = agent.audio_handoff_validator.validate_handoff(
            sections=agent.sections,
            section_audios=agent.section_audios,
            section_timestamps=agent.section_timestamps,
            expected_fingerprint=agent.validated_storyboard_fingerprint,
        )
        assert rep_str.overall_status == HandoffStatus.FAILED
        assert any(i.category == HandoffIssueCategory.UNREADABLE_METADATA for i in rep_str.issues)

        # NaN duration
        agent.section_timestamps[sec.id]["total_duration"] = float("nan")
        rep_nan = agent.audio_handoff_validator.validate_handoff(
            sections=agent.sections,
            section_audios=agent.section_audios,
            section_timestamps=agent.section_timestamps,
            expected_fingerprint=agent.validated_storyboard_fingerprint,
        )
        assert rep_nan.overall_status == HandoffStatus.FAILED
        assert any(i.category == HandoffIssueCategory.INVALID_AUDIO_FILE for i in rep_nan.issues)

        # Inf duration
        agent.section_timestamps[sec.id]["total_duration"] = float("inf")
        rep_inf = agent.audio_handoff_validator.validate_handoff(
            sections=agent.sections,
            section_audios=agent.section_audios,
            section_timestamps=agent.section_timestamps,
            expected_fingerprint=agent.validated_storyboard_fingerprint,
        )
        assert rep_inf.overall_status == HandoffStatus.FAILED
        assert any(i.category == HandoffIssueCategory.INVALID_AUDIO_FILE for i in rep_inf.issues)

        # Negative duration
        agent.section_timestamps[sec.id]["total_duration"] = -5.0
        rep_neg = agent.audio_handoff_validator.validate_handoff(
            sections=agent.sections,
            section_audios=agent.section_audios,
            section_timestamps=agent.section_timestamps,
            expected_fingerprint=agent.validated_storyboard_fingerprint,
        )
        assert rep_neg.overall_status == HandoffStatus.FAILED
        assert any(i.category == HandoffIssueCategory.INVALID_AUDIO_FILE for i in rep_neg.issues)

    # -------------------------------------------------------------------------
    # Test 19: Stale combined audio after interrupted regeneration
    # -------------------------------------------------------------------------
    def test_19_stale_combined_audio_after_interrupted_regeneration(self):
        """19. If synthesis fails midway during regeneration, old combined audio is purged and not mistaken for valid output."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=1)
        sec = agent.sections[0]

        # Initial valid audio for narration A
        initial_lines = ["Initial narration line one.", "Initial narration line two."]
        self._create_section_audio_artifacts(agent, sec.id, initial_lines)

        old_combined_audio = agent.output_dir / f"{sec.id}_audio.mp3"
        old_ts = agent.output_dir / f"{sec.id}_timestamps.json"
        assert old_combined_audio.exists()
        assert old_ts.exists()

        # New narration B
        new_lines = ["Brand new narration line 1.", "Brand new narration line 2."]

        # Mock speech generation: line 1 succeeds, line 2 throws an exception (simulating mid-generation crash)
        call_count = 0
        def failing_speech(text, out_path, voice=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                self._create_mock_audio_file(Path(out_path), 500)
                return 0.5
            raise RuntimeError("Simulated TTS synthesis crash on line 2")

        with patch.object(agent.audio_manager, "generate_speech", side_effect=failing_speech):
            with pytest.raises(RuntimeError, match="Simulated TTS synthesis crash"):
                agent.audio_manager.generate_section_audio(
                    section_id=sec.id,
                    lecture_lines=new_lines,
                    output_dir=agent.output_dir,
                )

        # Assert old combined audio and timestamps were cleared so stale output cannot be used
        assert not old_ts.exists()
        assert not old_combined_audio.exists()

        # Update in-memory sections to new lines and verify handoff validator flags MISSING_AUDIO_SECTION
        sec.lecture_lines = new_lines
        agent.validated_storyboard_fingerprint = compute_storyboard_fingerprint(agent.sections)
        handoff_rep = agent.audio_handoff_validator.validate_handoff(
            sections=agent.sections,
            section_audios=agent.section_audios,
            section_timestamps=agent.section_timestamps,
            expected_fingerprint=agent.validated_storyboard_fingerprint,
        )
        assert handoff_rep.is_safe_for_stage3 is False

    # -------------------------------------------------------------------------
    # Test 20: Section-scoped cleanup does not affect other sections
    # -------------------------------------------------------------------------
    def test_20_section_scoped_cleanup_does_not_affect_other_sections(self):
        """20. Purging line audio for section_1 strictly preserves section_10 files."""
        agent = self._create_agent()
        output_dir = agent.output_dir
        audio_dir = output_dir / "audio_lines"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Create line files for section_1
        sec1_l1 = audio_dir / "section_1_line_1.mp3"
        sec1_l2 = audio_dir / "section_1_line_2.mp3"
        sec1_l1.write_bytes(b"dummy1")
        sec1_l2.write_bytes(b"dummy2")

        # Create line files for section_10
        sec10_l1 = audio_dir / "section_10_line_1.mp3"
        sec10_l2 = audio_dir / "section_10_line_2.mp3"
        sec10_l1.write_bytes(b"dummy10_1")
        sec10_l2.write_bytes(b"dummy10_2")

        # Trigger regeneration / stale cleanup on section_1
        def mock_speech(text, out_path, voice=None):
            self._create_mock_audio_file(Path(out_path), 500)
            return 0.5

        with patch.object(agent.audio_manager, "generate_speech", side_effect=mock_speech):
            agent.audio_manager.generate_section_audio(
                section_id="section_1",
                lecture_lines=["New line for section 1"],
                output_dir=output_dir,
            )

        # section_10 line files must remain untouched
        assert sec10_l1.exists()
        assert sec10_l2.exists()
        assert sec10_l1.read_bytes() == b"dummy10_1"
        assert sec10_l2.read_bytes() == b"dummy10_2"

    # -------------------------------------------------------------------------
    # Test 21: Recovery utility entry-gate rejects unsafe storyboard
    # -------------------------------------------------------------------------
    def test_21_synchronize_case_audio_entry_gate_rejects_unsafe_storyboard(self):
        """21. synchronize_case_audio() checks storyboard fidelity at entry before generating audio."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=2)

        # Create an on-disk storyboard with a wrong numerical value
        sb_path = agent.output_dir / "storyboard.json"
        sb_data = {
            "sections": [
                {
                    "id": "section_1",
                    "title": "Rate Calculation",
                    "lecture_lines": ["Line 1"],
                    "animations": ["Show MathTex"],
                    "verified_values": {"k1": "99999.0 s^-1"},  # Hallucinated / wrong value!
                },
                {
                    "id": "section_2",
                    "title": "Summary",
                    "lecture_lines": ["Line 2"],
                    "animations": ["Show Summary"],
                    "verified_values": {},
                },
            ]
        }
        with open(sb_path, "w", encoding="utf-8") as f:
            json.dump(sb_data, f, indent=2)

        # Reset in-memory sections to simulate standalone recovery run
        agent.sections = []
        agent.storyboard_fidelity_report = None

        with pytest.raises(ValueError, match="Cannot synchronize audio"):
            agent.synchronize_case_audio()

    # -------------------------------------------------------------------------
    # Test 22: use_audio=False still enforces storyboard/numerical fidelity gates
    # -------------------------------------------------------------------------
    def test_22_use_audio_false_still_enforces_numerical_fidelity_gates(self):
        """22. Disabling audio does not bypass numerical fidelity or mutation gates."""
        agent = self._create_agent(use_audio=False)
        self._setup_verified_agent(agent, sections_count=2)

        # Sections are currently valid
        assert agent.is_storyboard_safe_for_audio() is True
        assert agent.is_audio_safe_for_stage3() is True

        # Now introduce a numerical discrepancy into section_1
        agent.sections[0].verified_values = {"k1": "0.9999 s^-1"}  # Contradicts Stage 1 0.006545
        agent.storyboard_fidelity_report = None

        # Even with use_audio=False, fidelity gate must fail
        assert agent.is_storyboard_safe_for_audio() is False

        # Stage 3 and rendering must be blocked
        with pytest.raises(ValueError, match="Cannot generate code"):
            agent.generate_codes()

        with pytest.raises(ValueError, match="Cannot generate code"):
            agent.generate_section_code(agent.sections[0])

        with pytest.raises(ValueError, match="Cannot render section"):
            agent.render_section(agent.sections[0])

        with pytest.raises(ValueError, match="Cannot render sections"):
            agent.render_all_sections()

        with pytest.raises(ValueError, match="Cannot merge videos"):
            agent.merge_videos()

    # -------------------------------------------------------------------------
    # Test 23: Relative audio metadata path resolution (and symlink support)
    # -------------------------------------------------------------------------
    def test_23_audio_metadata_path_resolution_relative_and_absolute(self):
        """23. Validates relative paths against audio_file_path.parent and supports symlinks."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=1)

        self._create_section_audio_artifacts(agent, "section_1", agent.sections[0].lecture_lines)
        audio_path = Path(agent.section_audios["section_1"])
        ts_path = agent.output_dir / "section_1_timestamps.json"

        # Case A: Exact absolute path
        agent.audio_handoff_report = None
        report = agent._ensure_audio_verified()
        assert report.overall_status == HandoffStatus.VALIDATED

        # Case B: Simple relative path "section_1_audio.mp3"
        agent.section_timestamps["section_1"]["audio_path"] = "section_1_audio.mp3"
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump(agent.section_timestamps["section_1"], f)
        agent.audio_handoff_report = None
        report = agent._ensure_audio_verified()
        assert report.overall_status == HandoffStatus.VALIDATED

        # Case C: Explicit ./ relative path
        agent.section_timestamps["section_1"]["audio_path"] = "./section_1_audio.mp3"
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump(agent.section_timestamps["section_1"], f)
        agent.audio_handoff_report = None
        report = agent._ensure_audio_verified()
        assert report.overall_status == HandoffStatus.VALIDATED

        # Case D: Relative path with ..
        agent.section_timestamps["section_1"]["audio_path"] = f"../{agent.output_dir.name}/section_1_audio.mp3"
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump(agent.section_timestamps["section_1"], f)
        agent.audio_handoff_report = None
        report = agent._ensure_audio_verified()
        assert report.overall_status == HandoffStatus.VALIDATED

        # Case E: Symlink pointing to the audio file
        symlink_path = agent.output_dir / "section_1_symlink.mp3"
        if symlink_path.exists():
            symlink_path.unlink()
        try:
            symlink_path.symlink_to(audio_path.name)
            agent.section_timestamps["section_1"]["audio_path"] = str(symlink_path)
            with open(ts_path, "w", encoding="utf-8") as f:
                json.dump(agent.section_timestamps["section_1"], f)
            agent.audio_handoff_report = None
            report = agent._ensure_audio_verified()
            assert report.overall_status == HandoffStatus.VALIDATED
        finally:
            if symlink_path.exists():
                symlink_path.unlink()

        # Case F: Non-matching path fails with INVALID_AUDIO_FILE
        agent.section_timestamps["section_1"]["audio_path"] = "completely_different_audio.mp3"
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump(agent.section_timestamps["section_1"], f)
        agent.audio_handoff_report = None
        report = agent._ensure_audio_verified(force=True)
        assert report.overall_status == HandoffStatus.FAILED
        assert any(i.category == HandoffIssueCategory.INVALID_AUDIO_FILE for i in report.issues)

    # -------------------------------------------------------------------------
    # Test 24: Audio metadata path resolution failure fails closed
    # -------------------------------------------------------------------------
    def test_24_audio_metadata_path_resolution_failure_fails_closed(self):
        """24. Unresolvable or corrupted audio paths fail closed and record INVALID_AUDIO_FILE."""
        agent = self._create_agent()
        self._setup_verified_agent(agent, sections_count=1)

        self._create_section_audio_artifacts(agent, "section_1", agent.sections[0].lecture_lines)
        ts_path = agent.output_dir / "section_1_timestamps.json"

        # Embedded null byte raises ValueError in Path
        agent.section_timestamps["section_1"]["audio_path"] = "section_1\0_bad_path.mp3"
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump(agent.section_timestamps["section_1"], f)

        agent.audio_handoff_report = None
        report = agent._ensure_audio_verified(force=True)
        assert report.overall_status == HandoffStatus.FAILED
        invalid_path_issues = [
            i for i in report.issues
            if i.category == HandoffIssueCategory.INVALID_AUDIO_FILE and "failed path resolution" in i.message
        ]
        assert len(invalid_path_issues) == 1
        assert "section_1" in invalid_path_issues[0].section_id

    # -------------------------------------------------------------------------
    # Test 25: Safe temporary audio file cleanup on failure
    # -------------------------------------------------------------------------
    def test_25_temporary_audio_file_cleanup_on_failure(self):
        """25. Temporary .tmp.{pid} files are purged on failure, leaving committed files intact."""
        output_dir = self.agent_dir / "output_test25"
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_mgr = AudioManager()

        section_id = "section_1"

        def mock_gen_speech(text, output_path, voice=None):
            AudioSegment.silent(duration=500).export(str(output_path), format="mp3")
            return 0.5

        # 1. Test generate_section_audio cleanup on JSON dump failure
        with patch.object(audio_mgr, "generate_speech", side_effect=mock_gen_speech):
            with patch("json.dump", side_effect=IOError("Simulated disk error during json.dump")):
                with pytest.raises(IOError, match="Simulated disk error"):
                    audio_mgr.generate_section_audio(
                        section_id=section_id,
                        lecture_lines=["Line 1", "Line 2"],
                        output_dir=output_dir,
                    )

        # Confirm no temporary files leaked
        tmp_files = list(output_dir.glob(f"{section_id}_*.tmp.*"))
        assert len(tmp_files) == 0, f"Found lingering temporary files: {tmp_files}"

        # 2. Test handle_external_audio cleanup on JSON dump failure preserves committed files
        committed_audio = output_dir / f"{section_id}_audio.mp3"
        committed_ts = output_dir / f"{section_id}_timestamps.json"
        silence = AudioSegment.silent(duration=1000)
        silence.export(str(committed_audio), format="mp3")
        committed_ts.write_text('{"committed": true}', encoding="utf-8")
        orig_audio_bytes = committed_audio.read_bytes()
        orig_ts_text = committed_ts.read_text(encoding="utf-8")

        dummy_external = output_dir / "dummy_external.mp3"
        silence.export(str(dummy_external), format="mp3")

        with patch("json.dump", side_effect=IOError("Simulated disk error during external json.dump")):
            with pytest.raises(IOError, match="Simulated disk error"):
                audio_mgr.handle_external_audio(
                    section_id=section_id,
                    audio_file_path=dummy_external,
                    lecture_lines=["Line 1", "Line 2"],
                    output_dir=output_dir,
                )

        # Confirm no temporary files leaked
        tmp_files = list(output_dir.glob(f"{section_id}_*.tmp.*"))
        assert len(tmp_files) == 0, f"Found lingering temporary files: {tmp_files}"
        # Confirm committed files remain intact
        assert committed_audio.exists()
        assert committed_audio.read_bytes() == orig_audio_bytes
        assert committed_ts.exists()
        assert committed_ts.read_text(encoding="utf-8") == orig_ts_text

    # -------------------------------------------------------------------------
    # Test 26: Missing/None story-beat handling in fingerprint
    # -------------------------------------------------------------------------
    def test_26_story_beat_none_and_empty_behavior_consistent_with_legacy_fallback(self):
        """26. Fingerprint handles None/empty story_beat without producing 'None' and falls back to story_step."""
        # Case 1: story_beat is None, story_step is present
        sec_dict_none_with_step = [
            {
                "id": "sec_1",
                "title": "Intro",
                "lecture_lines": ["Hello world"],
                "story_beat": None,
                "story_step": "Step 1",
            }
        ]
        # Case 2: story_beat is explicitly "Step 1"
        sec_dict_explicit_beat = [
            {
                "id": "sec_1",
                "title": "Intro",
                "lecture_lines": ["Hello world"],
                "story_beat": "Step 1",
            }
        ]
        fp1 = compute_storyboard_fingerprint(sec_dict_none_with_step)
        fp2 = compute_storyboard_fingerprint(sec_dict_explicit_beat)
        assert fp1 == fp2, "story_beat=None must fall back to story_step ('Step 1')"

        # Case 3: Both story_beat and story_step are None or missing
        sec_dict_both_none = [
            {
                "id": "sec_1",
                "title": "Intro",
                "lecture_lines": ["Hello world"],
                "story_beat": None,
                "story_step": None,
            }
        ]
        sec_dict_omitted = [
            {
                "id": "sec_1",
                "title": "Intro",
                "lecture_lines": ["Hello world"],
            }
        ]
        sec_dict_empty = [
            {
                "id": "sec_1",
                "title": "Intro",
                "lecture_lines": ["Hello world"],
                "story_beat": "",
                "story_step": "",
            }
        ]
        fp3 = compute_storyboard_fingerprint(sec_dict_both_none)
        fp4 = compute_storyboard_fingerprint(sec_dict_omitted)
        fp5 = compute_storyboard_fingerprint(sec_dict_empty)
        assert fp3 == fp4 == fp5, "None, missing, and empty must produce identical empty-string fingerprint entries"

        # Case 4: Literal string "None" is different from None
        sec_dict_literal_none = [
            {
                "id": "sec_1",
                "title": "Intro",
                "lecture_lines": ["Hello world"],
                "story_beat": "None",
            }
        ]
        fp_lit = compute_storyboard_fingerprint(sec_dict_literal_none)
        assert fp_lit != fp3, "Literal string 'None' must not equal None value"

        # Case 5: Section object with None story_beat
        sec_obj = Section(
            id="sec_1",
            title="Intro",
            lecture_lines=["Hello world"],
            animations=[],
            story_beat=None,
        )
        fp_obj = compute_storyboard_fingerprint([sec_obj])
        assert fp_obj == fp3, "Section object with story_beat=None must match dict with story_beat=None"
