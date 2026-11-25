"""
Unit and integration tests for CineLearn Stage 2 Storyboard Numerical Fidelity Checker (Milestone 4).

Covers:
1. Correct storyboard values pass.
2. Incorrect numerical equation is detected.
3. Incorrect displayed value is detected.
4. Incorrect narration value is detected.
5. Wrong units are detected.
6. T1/T2 or variable-context mix-ups are detected.
7. Missing required worked-example result is detected.
8. Harmless section numbers and animation timings do not trigger false positives.
9. Unsupported Stage 2 formulas are not trusted.
10. Ambiguous numerical claims return NEEDS_REVIEW.
11. A corrected storyboard passes after retry.
12. Retry exhaustion blocks audio generation.
13. Original Stage 1 values remain unchanged.
14. Existing non-numerical storyboards continue to work.
"""

import copy
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent import TeachingVideoAgent, TeachingOutline, RunConfig, Section
from numerical_evaluator import VerificationStatus
from storyboard_fidelity_checker import (
    StoryboardFidelityChecker,
    StoryboardFidelityReport,
    FidelityCategory,
    FidelityDiscrepancy,
)


# ============================================================================
# Test Fixtures & Sample Outlines / Storyboards
# ============================================================================

SAMPLE_ARRHENIUS_OUTLINE = {
    "topic": "Arrhenius Equation",
    "learning_level": "intermediate",
    "sections": [
        {"id": "section_1", "title": "Sec 1: Overview"},
        {"id": "section_2", "title": "Sec 2: Worked Example: Rate at 300K"},
    ],
    "governing_principles": ["Arrhenius Equation", "k = A * exp(-Ea / (R * T))"],
    "worked_example_spec": {
        "inputs": {"Ea": 75000, "R": 8.314, "T1": 300, "T2": 350, "A": 7.5e10},
        "input_units": {"Ea": "J/mol", "R": "J/(mol*K)", "T1": "K", "T2": "K"},
        "steps": [
            {
                "step": "Calculate exponent at T1: -Ea / (R * T1)",
                "result": "-75000 / (8.314 * 300) = -30.07",
                "original_result": "-75000 / (8.314 * 300) = -29.74",
                "structured_calc": {
                    "output_variable": "exp_T1",
                    "computed_value": -30.069761847486167,
                    "corrected_value": "-30.07",
                    "unit": None,
                    "original_claimed": "-29.74",
                },
            },
            {
                "step": "Compute k1 at T1: A * exp(-30.07)",
                "result": "7.5e10 * exp(-30.07) = 0.006545 s^-1",
                "original_result": "7.5e10 * exp(-29.74) = 0.0001087 s^-1",
                "structured_calc": {
                    "output_variable": "k1",
                    "computed_value": 0.00654504,
                    "corrected_value": "0.006545",
                    "unit": "s^-1",
                    "original_claimed": "0.0001087 s^-1",
                },
            },
        ],
        "final_result": "k1 = 0.006545 s^-1",
        "original_final_result": "k1 = 0.0001087 s^-1",
        "verification_metadata": {"overall_status": "VERIFIED", "is_safe_for_stage2": True},
    },
}

SAMPLE_VALID_STORYBOARD = {
    "visual_conventions": {},
    "sections": [
        {
            "id": "section_1",
            "title": "Sec 1: Overview",
            "lecture_lines": ["We examine chemical reaction rates and activation energy."],
            "captions": ["Reaction rates overview."],
            "animations": ["step 1: Text('Reaction Kinetics') | write | (0, 0) | (0, 0) | None | None"],
        },
        {
            "id": "section_2",
            "title": "Sec 2: Worked Example: Rate at 300K",
            "lecture_lines": [
                "At 300 Kelvin, the exponent evaluates to negative 30.07.",
                "Multiplying by the pre-exponential factor yields a rate constant k1 of 0.006545 per second.",
            ],
            "captions": [
                "Exponent is -30.07.",
                "Rate constant k1 is 0.006545 s^-1.",
            ],
            "verified_values": {
                "exp_T1": "-30.07",
                "k1": "0.006545 s^-1",
                "Ea": "75000 J/mol",
            },
            "animations": [
                'step 1: MathTex(r"-\\frac{E_a}{R T_1} = -30.07") | write | (0, 0) | (0, 0) | None | None',
                'step 2: MathTex(r"k_1 = 7.5 \\times 10^{10} \\times \\exp(-30.07) = 0.006545\\ \\text{s}^{-1}") | write | (0, 0) | (0, 0) | None | None',
            ],
        },
    ],
}


class TestStoryboardFidelityChecker:
    @pytest.fixture(autouse=True)
    def setup_checker(self, tmp_path):
        self.checker = StoryboardFidelityChecker()
        self.tmp_dir = tmp_path

    # 1. Correct storyboard values pass
    def test_1_correct_storyboard_values_pass(self):
        """1. Correct storyboard values pass verification with zero discrepancies."""
        report = self.checker.check_storyboard_fidelity(SAMPLE_ARRHENIUS_OUTLINE, SAMPLE_VALID_STORYBOARD)
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_audio is True
        assert len(report.discrepancies) == 0
        assert report.verified_claims > 0

    # 2. Incorrect numerical equation is detected
    def test_2_incorrect_numerical_equation_is_detected(self):
        """2. Incorrect numerical equation in MathTex is detected."""
        sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        # Relapse to rejected hallucination in equation: -29.74 instead of -30.07
        sb["sections"][1]["animations"][0] = 'step 1: MathTex(r"-\\frac{75000}{8.314 \\times 300} = -29.74") | write | (0, 0) | (0, 0) | None | None'

        report = self.checker.check_storyboard_fidelity(SAMPLE_ARRHENIUS_OUTLINE, sb)
        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_audio is False
        assert any(
            d.category in (FidelityCategory.VALUE_MISMATCH, FidelityCategory.HALLUCINATION_RELAPSE)
            for d in report.discrepancies
        )

    # 3. Incorrect displayed value is detected
    def test_3_incorrect_displayed_value_is_detected(self):
        """3. Incorrect displayed value in verified_values is detected."""
        sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        # Erroneous rate constant in verified_values
        sb["sections"][1]["verified_values"]["k1"] = "0.0001087 s^-1"

        report = self.checker.check_storyboard_fidelity(SAMPLE_ARRHENIUS_OUTLINE, sb)
        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_audio is False
        assert any("k1" in d.location for d in report.discrepancies)
        assert any(
            d.category in (FidelityCategory.VALUE_MISMATCH, FidelityCategory.HALLUCINATION_RELAPSE)
            for d in report.discrepancies
        )

    # 4. Incorrect narration value is detected
    def test_4_incorrect_narration_value_is_detected(self):
        """4. Incorrect numerical value spoken in narration is detected."""
        sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        # Narration uses hallucinated number 0.0001087
        sb["sections"][1]["lecture_lines"][1] = "Multiplying by the factor yields a rate constant of 0.0001087 per second."

        report = self.checker.check_storyboard_fidelity(SAMPLE_ARRHENIUS_OUTLINE, sb)
        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_audio is False
        assert any("lecture_lines" in d.location for d in report.discrepancies)

    # 5. Wrong units are detected
    def test_5_wrong_units_are_detected(self):
        """5. Incompatible or conflicting units are detected."""
        sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        # Ea declared as 75000 kJ/mol (off by 1000x from 75000 J/mol)
        sb["sections"][1]["verified_values"]["Ea"] = "75000 kJ/mol"

        report = self.checker.check_storyboard_fidelity(SAMPLE_ARRHENIUS_OUTLINE, sb)
        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_audio is False
        assert any(d.category == FidelityCategory.UNIT_MISMATCH for d in report.discrepancies)

    # 6. T1/T2 or variable-context mix-ups are detected
    def test_6_t1_t2_variable_context_mixup_detected(self):
        """6. Variable-context mix-up (e.g. associating T1 with T2 values) is detected."""
        sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        # T1 condition given an exponent belonging to T2
        sb["sections"][1]["verified_values"]["exp_T1"] = "-25.77"  # Value for T2=350K, not T1=300K

        report = self.checker.check_storyboard_fidelity(SAMPLE_ARRHENIUS_OUTLINE, sb)
        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_audio is False
        assert any(
            d.category in (FidelityCategory.VALUE_MISMATCH, FidelityCategory.CONTEXT_MISMATCH)
            for d in report.discrepancies
        )

    # 7. Missing required worked-example result is detected
    def test_7_missing_required_worked_example_result_detected(self):
        """7. Omission of the required final worked-example calculation result is detected."""
        sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        # Strip all mentions of the final rate constant k1 = 0.006545
        sb["sections"][1]["verified_values"] = {"Ea": "75000 J/mol"}
        sb["sections"][1]["lecture_lines"] = ["We calculate only the preliminary terms.", "The discussion ends here."]
        sb["sections"][1]["animations"] = ["step 1: Text('Setup') | write | (0, 0) | (0, 0) | None | None"]

        report = self.checker.check_storyboard_fidelity(SAMPLE_ARRHENIUS_OUTLINE, sb)
        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_audio is False
        assert any(d.category == FidelityCategory.MISSING_REQUIRED_RESULT for d in report.discrepancies)

    # 8. Harmless section numbers and animation timings do not trigger false positives
    def test_8_harmless_section_numbers_and_animation_timings_do_not_trigger_false_positives(self):
        """8. Section numbers, 3-component counts, coordinates, and animation durations are ignored."""
        sb = {
            "sections": [
                {
                    "id": "section_3",
                    "title": "Sec 3: 3 Key Factors in Rate Equations",
                    "lecture_lines": [
                        "In Section 3, we observe 2 distinct reactions across 4 steps.",
                        "At 300 Kelvin, the exponent evaluates to negative 30.07.",
                        "Multiplying by the pre-exponential factor yields a rate constant k1 of 0.006545 per second.",
                    ],
                    "captions": ["Line 1", "Line 2", "Line 3"],
                    "verified_values": {"exp_T1": "-30.07", "k1": "0.006545 s^-1"},
                    "animations": [
                        "step 1: Circle() | write | (0, 0) | (2.0, 3.5) | run_time=2.0 | None",
                        'step 2: MathTex(r"k_1 = 0.006545\\ \\text{s}^{-1}") | write | (0, 0) | (0, 0) | run_time=1.5 | None',
                    ],
                }
            ]
        }
        report = self.checker.check_storyboard_fidelity(SAMPLE_ARRHENIUS_OUTLINE, sb)
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_audio is True
        assert len(report.discrepancies) == 0

    # 9. Unsupported Stage 2 formulas are not trusted
    def test_9_unsupported_stage2_formulas_are_not_trusted(self):
        """9. Unverified formulas introduced in Stage 2 without registry support are flagged for review."""
        sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        # Stage 2 introduces a fabricated formula: k = Ea / (R * T) without exponential
        sb["sections"][1]["animations"][0] = 'step 1: MathTex(r"k = \\frac{E_a}{R \\cdot T} = 30.07") | write | (0, 0) | (0, 0) | None | None'

        report = self.checker.check_storyboard_fidelity(SAMPLE_ARRHENIUS_OUTLINE, sb)
        assert report.is_safe_for_audio is False
        assert any(d.category == FidelityCategory.UNSUPPORTED_FORMULA for d in report.discrepancies)

    # 10. Ambiguous numerical claims return NEEDS_REVIEW
    def test_10_ambiguous_numerical_claims_return_needs_review(self):
        """10. Unparseable or ambiguous numerical claims return NEEDS_REVIEW."""
        sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        sb["sections"][1]["verified_values"]["k1"] = "k1 = approximately ????"

        report = self.checker.check_storyboard_fidelity(SAMPLE_ARRHENIUS_OUTLINE, sb)
        assert report.overall_status == VerificationStatus.NEEDS_REVIEW
        assert report.is_safe_for_audio is False
        assert any(d.category == FidelityCategory.AMBIGUOUS_CLAIM for d in report.discrepancies)

    # 11. A corrected storyboard passes after retry
    def test_11_corrected_storyboard_passes_after_retry(self):
        """11. A flawed storyboard corrected on retry passes the gate."""
        cfg = RunConfig(level="intermediate", max_fidelity_retries=2)
        agent = TeachingVideoAgent(
            idx=1,
            knowledge_point="Arrhenius Equation",
            folder=str(self.tmp_dir / "CASES"),
            cfg=cfg,
        )
        agent.outline = TeachingOutline.from_dict(SAMPLE_ARRHENIUS_OUTLINE)

        # Flawed attempt 1 (hallucinated 0.0001087)
        flawed_sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        flawed_sb["sections"][1]["verified_values"]["k1"] = "0.0001087 s^-1"
        flawed_sb["sections"][1]["lecture_lines"][1] = "We get 0.0001087 per second."

        # Corrected attempt 2
        corrected_sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)

        call_count = 0
        def mock_request(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps(flawed_sb)
            return json.dumps(corrected_sb)

        with patch.object(agent, "_request_api_and_track_tokens", side_effect=mock_request):
            sections = agent.generate_storyboard()

        assert sections is not None
        assert len(sections) == 2
        assert agent.is_storyboard_safe_for_audio() is True
        assert agent.storyboard_fidelity_report.overall_status == VerificationStatus.VERIFIED

    # 12. Retry exhaustion blocks audio generation
    def test_12_retry_exhaustion_blocks_audio_generation(self):
        """12. Retry exhaustion halts pipeline before audio generation."""
        cfg = RunConfig(level="intermediate", max_fidelity_retries=2)
        agent = TeachingVideoAgent(
            idx=1,
            knowledge_point="Arrhenius Equation",
            folder=str(self.tmp_dir / "CASES"),
            cfg=cfg,
        )

        # Flawed storyboard that constantly returns discrepancies
        flawed_sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        flawed_sb["sections"][1]["verified_values"]["k1"] = "0.0001087 s^-1"
        flawed_sb["sections"][1]["lecture_lines"][1] = "We get 0.0001087 per second."

        def mock_request(prompt, **kwargs):
            stage = kwargs.get("stage")
            if stage == "outline":
                return json.dumps(SAMPLE_ARRHENIUS_OUTLINE)
            return json.dumps(flawed_sb)

        with patch.object(agent, "_request_api_and_track_tokens", side_effect=mock_request):
            result = agent.GENERATE_VIDEO()

        # Pipeline halts before audio generation
        assert result is None
        cp_file = agent.output_dir / "checkpoint.json"
        assert cp_file.exists()
        with open(cp_file, "r", encoding="utf-8") as f:
            cp_data = json.load(f)
        assert cp_data["stage"] == "storyboard_fidelity_blocked"

    # 13. Original Stage 1 values remain unchanged
    def test_13_original_stage1_values_remain_unchanged(self):
        """13. Storyboard fidelity checking does not mutate or overwrite Stage 1 outline."""
        outline_copy = copy.deepcopy(SAMPLE_ARRHENIUS_OUTLINE)
        flawed_sb = copy.deepcopy(SAMPLE_VALID_STORYBOARD)
        flawed_sb["sections"][1]["verified_values"]["k1"] = "0.0001087 s^-1"

        report = self.checker.check_storyboard_fidelity(outline_copy, flawed_sb)
        assert report.is_safe_for_audio is False

        # Verify Stage 1 outline was not mutated
        step_calc = outline_copy["worked_example_spec"]["steps"][1]["structured_calc"]
        assert step_calc["computed_value"] == 0.00654504
        assert step_calc["corrected_value"] == "0.006545"

    # 14. Existing non-numerical storyboards continue to work
    def test_14_existing_non_numerical_storyboards_continue_to_work(self):
        """14. Existing non-numerical storyboards pass immediately with NOT_APPLICABLE status."""
        non_num_outline = {
            "topic": "Graph Algorithms",
            "learning_level": "intermediate",
            "sections": [{"id": "section_1", "title": "Sec 1: Introduction"}],
        }
        non_num_sb = {
            "sections": [
                {
                    "id": "section_1",
                    "title": "Sec 1: Introduction",
                    "lecture_lines": ["Graphs consist of vertices and edges."],
                    "captions": ["Vertices and edges."],
                    "animations": ["step 1: Circle() | write | (0, 0) | (0, 0) | None | None"],
                }
            ]
        }
        report = self.checker.check_storyboard_fidelity(non_num_outline, non_num_sb)
        assert report.overall_status == VerificationStatus.NOT_APPLICABLE
        assert report.is_safe_for_audio is True
        assert len(report.discrepancies) == 0
