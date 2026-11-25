"""
Integration tests for CineLearn Stage 1 WorkedExampleVerifier Gate (Milestone 3).

Covers:
1. A correct numerical worked example proceeds to Stage 2.
2. An incorrect but safely correctable example is corrected and proceeds.
3. An unresolved numerical example is blocked.
4. A non-correctable FAILED example is blocked.
5. An outline without a numerical worked example proceeds normally.
6. A cached legacy outline without verification metadata is verified.
7. A cached corrected outline preserves its original claims after restoration.
8. The saved outline.json and in-memory outline contain identical corrected values and verification metadata.
9. Stage 2 is never invoked when the validation gate blocks the outline.
10. Existing outline serialization and checkpoint tests remain valid.
"""

import json
import math
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

from agent import TeachingVideoAgent, TeachingOutline, RunConfig
from numerical_evaluator import VerificationStatus


class TestStage1WorkedExampleGate:
    @pytest.fixture(autouse=True)
    def setup_tmp(self, tmp_path):
        self.tmp_dir = tmp_path
        self.agent_dir = tmp_path / "CASES" / "test_run"
        self.agent_dir.mkdir(parents=True, exist_ok=True)

    def _create_agent(self, topic="Arrhenius Equation", level="intermediate"):
        cfg = RunConfig(level=level)
        agent = TeachingVideoAgent(
            idx=1,
            knowledge_point=topic,
            folder=str(self.tmp_dir / "CASES"),
            cfg=cfg,
        )
        return agent

    def test_1_correct_numerical_worked_example_proceeds_to_stage2(self):
        """1. A correct numerical worked example proceeds to Stage 2."""
        agent = self._create_agent()
        correct_outline_json = {
            "topic": "Arrhenius Equation",
            "learning_level": "intermediate",
            "sections": [
                {
                    "id": "section_1",
                    "title": "Rate Calculation",
                    "lecture_lines": ["Calculate k1 at 300K"],
                    "animations": ["Show MathTex"],
                }
            ],
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
        }

        with patch.object(agent, "_request_api_and_track_tokens") as mock_api:
            mock_api.return_value = json.dumps(correct_outline_json)
            outline = agent.generate_outline()

        assert outline is not None
        assert agent.is_outline_safe_for_stage2() is True
        assert agent.outline_verification_report.overall_status == VerificationStatus.VERIFIED

        # Calling generate_storyboard should not raise validation gate error
        with patch.object(agent, "_request_api_and_track_tokens") as mock_sb_api:
            mock_sb_api.return_value = json.dumps({
                "sections": [
                    {
                        "id": "section_1",
                        "title": "Rate Calculation",
                        "lecture_lines": ["Line 1"],
                        "animations": ["Anim 1"],
                        "verified_values": {"k1": "0.006545 s^-1"},
                    }
                ]
            })
            sections = agent.generate_storyboard()
            assert len(sections) == 1

    def test_2_incorrect_but_safely_correctable_example_is_corrected_and_proceeds(self):
        """2. An incorrect but safely correctable example is corrected and proceeds."""
        agent = self._create_agent()
        hallucinated_outline_json = {
            "topic": "Arrhenius Equation",
            "learning_level": "intermediate",
            "sections": [{"id": "section_1", "title": "Sec 1"}],
            "worked_example_spec": {
                "inputs": {"Ea": 75000, "R": 8.314, "T": 300, "A": 7.5e10},
                "steps": [
                    {
                        "step": "Calculate exponent",
                        "result": "-75000 / (8.314 * 300) = -29.74",  # Hallucinated
                    },
                    {
                        "step": "Compute k1",
                        "result": "7.5e10 * exp(-29.74) = 0.0001087 s^-1",  # Hallucinated
                    },
                ],
                "final_result": "k1 = 0.0001087 s^-1",
            },
        }

        with patch.object(agent, "_request_api_and_track_tokens") as mock_api:
            mock_api.return_value = json.dumps(hallucinated_outline_json)
            outline = agent.generate_outline()

        # Gate passes with safe auto-corrections
        assert agent.is_outline_safe_for_stage2() is True
        assert agent.outline_verification_report.corrected_steps == 2
        assert agent.outline_verification_report.overall_status == VerificationStatus.VERIFIED

        # Check values are corrected and originals preserved
        steps = outline.worked_example_spec["steps"]
        assert steps[0]["original_result"] == "-75000 / (8.314 * 300) = -29.74"
        assert steps[0]["result"] == "-75000 / (8.314 * 300) = -30.07"
        assert steps[1]["original_result"] == "7.5e10 * exp(-29.74) = 0.0001087 s^-1"
        assert "0.006545" in steps[1]["result"]

        assert outline.worked_example_spec["original_final_result"] == "k1 = 0.0001087 s^-1"
        assert "k1 = 0.006545 s^-1" in outline.worked_example_spec["final_result"]

    def test_3_unresolved_numerical_example_is_blocked(self):
        """3. An unresolved numerical example is blocked from Stage 2."""
        agent = self._create_agent()
        untrusted_outline_json = {
            "topic": "Arrhenius Equation",
            "learning_level": "intermediate",
            "sections": [{"id": "section_1", "title": "Sec 1"}],
            "worked_example_spec": {
                "inputs": {"Ea": 75000, "R": 8.314, "T": 300, "A": 7.5e10},
                "steps": [
                    {
                        "step": "Inverted Arrhenius formula (Untrusted physics)",
                        "result": "(8.314 * 300) / 75000 = 0.033",
                    }
                ],
                "final_result": "0.033",
            },
        }

        with patch.object(agent, "_request_api_and_track_tokens") as mock_api:
            mock_api.return_value = json.dumps(untrusted_outline_json)
            outline = agent.generate_outline()

        # Gate blocks Stage 2
        assert agent.is_outline_safe_for_stage2() is False
        assert agent.outline_verification_report.overall_status == VerificationStatus.NEEDS_REVIEW

        # Calling generate_storyboard raises explicit gate error
        with pytest.raises(ValueError, match="Stage 2 is blocked by worked example validation gate"):
            agent.generate_storyboard()

        # Checkpoint recorded block status
        cp_file = agent.output_dir / "checkpoint.json"
        assert cp_file.exists()
        with open(cp_file, "r", encoding="utf-8") as f:
            cp_data = json.load(f)
        assert cp_data["stage"] == "outline_verification_blocked"

    def test_4_non_correctable_failed_example_is_blocked(self):
        """4. A non-correctable FAILED example is blocked from Stage 2."""
        agent = self._create_agent()
        failed_outline_json = {
            "topic": "Basic Math",
            "learning_level": "intermediate",
            "sections": [{"id": "section_1", "title": "Sec 1"}],
            "worked_example_spec": {
                "inputs": {"x": 10.0, "y": 20.0},
                "steps": [
                    {
                        "step": "Sum: x + y",
                        "result": "10.0 + 20.0 = 999.0",
                        "allow_auto_correction": False,  # Disallows auto-correction
                    }
                ],
                "final_result": "999.0",
            },
        }

        with patch.object(agent, "_request_api_and_track_tokens") as mock_api:
            mock_api.return_value = json.dumps(failed_outline_json)
            agent.generate_outline()

        assert agent.is_outline_safe_for_stage2() is False
        assert agent.outline_verification_report.overall_status == VerificationStatus.FAILED
        assert agent.outline_verification_report.failed_steps == 1

        with pytest.raises(ValueError, match="FAILED"):
            agent.generate_storyboard()

    def test_5_outline_without_numerical_worked_example_proceeds_normally(self):
        """5. An outline without a numerical worked example proceeds normally."""
        agent = self._create_agent("Graph Algorithms")
        prose_outline_json = {
            "topic": "Graph Algorithms",
            "learning_level": "intermediate",
            "sections": [{"id": "section_1", "title": "Introduction"}],
        }

        with patch.object(agent, "_request_api_and_track_tokens") as mock_api:
            mock_api.return_value = json.dumps(prose_outline_json)
            outline = agent.generate_outline()

        assert outline is not None
        assert agent.is_outline_safe_for_stage2() is True
        assert agent.outline_verification_report.overall_status == VerificationStatus.NOT_APPLICABLE

        # Storyboard proceeds without error
        with patch.object(agent, "_request_api_and_track_tokens") as mock_sb_api:
            mock_sb_api.return_value = json.dumps({
                "sections": [{"id": "section_1", "title": "Intro", "lecture_lines": ["Hi"], "animations": ["Anim"]}]
            })
            sections = agent.generate_storyboard()
            assert len(sections) == 1

    def test_6_cached_legacy_outline_without_verification_metadata_is_verified(self):
        """6. A cached legacy outline without verification metadata is verified upon load."""
        agent = self._create_agent()
        legacy_outline_file = agent.output_dir / "outline.json"
        legacy_outline_data = {
            "topic": "Arrhenius Equation",
            "learning_level": "intermediate",
            "sections": [{"id": "section_1", "title": "Sec 1"}],
            "worked_example_spec": {
                "inputs": {"Ea": 75000, "R": 8.314, "T": 300},
                "steps": [
                    {
                        "step": "Calculate exponent",
                        "result": "-75000 / (8.314 * 300) = -29.74",  # Hallucinated, no verification metadata
                    }
                ],
                "final_result": "-29.74",
            },
        }
        with open(legacy_outline_file, "w", encoding="utf-8") as f:
            json.dump(legacy_outline_data, f, indent=2)

        # generate_outline loads disk file without calling LLM
        outline = agent.generate_outline()

        # Must have been verified and auto-corrected automatically
        assert agent.is_outline_safe_for_stage2() is True
        assert outline.worked_example_spec["steps"][0]["result"] == "-75000 / (8.314 * 300) = -30.07"
        assert outline.worked_example_spec["steps"][0]["original_result"] == "-75000 / (8.314 * 300) = -29.74"
        assert "verification_metadata" in outline.worked_example_spec

    def test_7_cached_corrected_outline_preserves_original_claims_after_restoration(self):
        """7. A cached corrected outline preserves its original claims after restoration."""
        agent1 = self._create_agent()
        raw_outline = {
            "topic": "Arrhenius Equation",
            "learning_level": "intermediate",
            "sections": [{"id": "section_1", "title": "Sec 1"}],
            "worked_example_spec": {
                "inputs": {"Ea": 75000, "R": 8.314, "T": 300},
                "steps": [
                    {
                        "step": "Calculate exponent",
                        "result": "-75000 / (8.314 * 300) = -29.74",
                    }
                ],
                "final_result": "-29.74",
            },
        }
        with patch.object(agent1, "_request_api_and_track_tokens") as mock_api:
            mock_api.return_value = json.dumps(raw_outline)
            agent1.generate_outline()

        # Agent 2 starts fresh in the same output directory
        agent2 = self._create_agent()
        restored_outline = agent2.generate_outline()

        # Must preserve original claims losslessly across reload and re-verification
        step = restored_outline.worked_example_spec["steps"][0]
        assert step["original_result"] == "-75000 / (8.314 * 300) = -29.74"
        assert step["result"] == "-75000 / (8.314 * 300) = -30.07"
        assert restored_outline.worked_example_spec["original_final_result"] == "-29.74"
        assert restored_outline.worked_example_spec["final_result"] == "-30.07"

    def test_8_saved_outline_json_and_in_memory_outline_are_identical(self):
        """8. The saved outline.json and in-memory outline contain identical corrected values and metadata."""
        agent = self._create_agent()
        outline_json_data = {
            "topic": "Arrhenius Equation",
            "learning_level": "intermediate",
            "sections": [{"id": "section_1", "title": "Sec 1"}],
            "worked_example_spec": {
                "inputs": {"Ea": 75000, "R": 8.314, "T": 300},
                "steps": [
                    {
                        "step": "Calculate exponent",
                        "result": "-75000 / (8.314 * 300) = -29.74",
                    }
                ],
                "final_result": "-29.74",
            },
        }

        with patch.object(agent, "_request_api_and_track_tokens") as mock_api:
            mock_api.return_value = json.dumps(outline_json_data)
            agent.generate_outline()

        outline_file = agent.output_dir / "outline.json"
        with open(outline_file, "r", encoding="utf-8") as f:
            disk_data = json.load(f)

        in_mem_data = agent.outline.to_dict()

        assert disk_data["worked_example_spec"]["steps"][0]["result"] == in_mem_data["worked_example_spec"]["steps"][0]["result"]
        assert disk_data["worked_example_spec"]["steps"][0]["original_result"] == in_mem_data["worked_example_spec"]["steps"][0]["original_result"]
        assert disk_data["worked_example_spec"]["verification_metadata"] == in_mem_data["worked_example_spec"]["verification_metadata"]

    def test_9_stage2_is_never_invoked_when_validation_gate_blocks(self):
        """9. Stage 2 is never invoked when the validation gate blocks the outline in GENERATE_VIDEO."""
        agent = self._create_agent()
        untrusted_outline_json = {
            "topic": "Arrhenius Equation",
            "learning_level": "intermediate",
            "sections": [{"id": "section_1", "title": "Sec 1"}],
            "worked_example_spec": {
                "inputs": {"Ea": 75000, "R": 8.314, "T": 300, "A": 7.5e10},
                "steps": [
                    {
                        "step": "Inverted Arrhenius formula",
                        "result": "(8.314 * 300) / 75000 = 0.033",
                    }
                ],
                "final_result": "0.033",
            },
        }

        with patch.object(agent, "_request_api_and_track_tokens") as mock_api:
            mock_api.return_value = json.dumps(untrusted_outline_json)
            result = agent.GENERATE_VIDEO()

        # Pipeline halted before Stage 2
        assert result is None

        # Verify that mock_api was NEVER called for storyboard stage
        for call_args in mock_api.call_args_list:
            kwargs = call_args[1] if len(call_args) > 1 else {}
            assert kwargs.get("stage") != "storyboard"

        # Checkpoint recorded block
        cp_file = agent.output_dir / "checkpoint.json"
        assert cp_file.exists()
        with open(cp_file, "r", encoding="utf-8") as f:
            cp_data = json.load(f)
        assert cp_data["stage"] == "outline_verification_blocked"

    def test_10_existing_outline_serialization_and_checkpoint_tests_remain_valid(self):
        """10. Verify that TeachingOutline serialization preserves all metadata through from_dict and to_dict."""
        spec_data = {
            "inputs": {"Ea": 75000, "R": 8.314, "T": 300},
            "steps": [
                {
                    "step": "Calculate exponent",
                    "result": "-75000 / (8.314 * 300) = -30.07",
                    "original_result": "-75000 / (8.314 * 300) = -29.74",
                    "structured_calc": {
                        "status": "VERIFIED",
                        "diagnostic": "ROUNDING_MATCH",
                    },
                }
            ],
            "verification_metadata": {"overall_status": "VERIFIED", "is_safe_for_stage2": True},
        }

        outline = TeachingOutline(
            topic="Arrhenius",
            worked_example_spec=spec_data,
        )

        d = outline.to_dict()
        assert d["worked_example_spec"]["verification_metadata"]["overall_status"] == "VERIFIED"

        reconstructed = TeachingOutline.from_dict(d)
        assert reconstructed.worked_example_spec["steps"][0]["original_result"] == "-75000 / (8.314 * 300) = -29.74"
