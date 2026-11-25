"""
Tests for configuration propagation across all stages of CineLearn.
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent import TeachingVideoAgent, RunConfig, Section
from prompts.stage1 import get_prompt1_outline
from prompts.stage2 import get_prompt2_storyboard
from prompts.stage3 import get_prompt3_code
from prompts.base_class import base_class


class TestConfigPropagation(unittest.TestCase):
    def test_run_config_defaults_and_overrides(self):
        cfg = RunConfig(
            level="beginner",
            analogy="rubber sheet",
            soft_target_duration=180.0,
            hard_max_duration=300.0,
            speaking_wpm=130.0,
        )
        self.assertEqual(cfg.level, "beginner")
        self.assertEqual(cfg.analogy, "rubber sheet")
        self.assertEqual(cfg.soft_target_duration, 180.0)
        self.assertEqual(cfg.hard_max_duration, 300.0)
        self.assertEqual(cfg.speaking_wpm, 130.0)

    def test_agent_initialization_propagation(self):
        cfg = RunConfig(level="advanced", analogy="vector flow")
        agent = TeachingVideoAgent(
            idx=1,
            knowledge_point="Vector Fields",
            folder="TEST_CASES",
            cfg=cfg,
            soft_target_duration=360.0,
            hard_max_duration=600.0,
        )
        self.assertEqual(agent.topic, "Vector Fields")
        self.assertEqual(agent.learning_topic, "Vector Fields")
        self.assertEqual(agent.level, "advanced")
        self.assertEqual(agent.analogy, "vector flow")
        self.assertEqual(agent.duration_manager.level, "advanced")
        self.assertEqual(agent.duration_manager.soft_target_sec, 360.0)
        self.assertEqual(agent.duration_manager.hard_max_sec, 600.0)
        self.assertIsNotNone(agent.math_validator)

    def test_outline_prompt_propagation(self):
        # Beginner with analogy
        prompt_beg = get_prompt1_outline(
            knowledge_point="Matrices",
            duration=3.5,
            level="beginner",
            analogy="cookie cutters"
        )
        self.assertIn("Target Learner Level: BEGINNER", prompt_beg)
        self.assertIn("cookie cutters", prompt_beg)
        self.assertIn("3.5 minutes", prompt_beg)
        self.assertIn("Analogy Suitability Assessment", prompt_beg)
        self.assertIn("Accuracy Over Personalization", prompt_beg)
        self.assertIn("Analogy Breakdown & Boundaries", prompt_beg)

        # Advanced with analogy
        prompt_adv = get_prompt1_outline(
            knowledge_point="Matrices",
            duration=6.0,
            level="advanced",
            analogy="space-time transformations"
        )
        self.assertIn("Target Learner Level: ADVANCED", prompt_adv)
        self.assertIn("space-time transformations", prompt_adv)
        self.assertIn("6.0 minutes", prompt_adv)

    def test_storyboard_prompt_propagation(self):
        prompt_sb = get_prompt2_storyboard(
            outline="{}",
            level="intermediate",
            analogy="shadow projections"
        )
        self.assertIn("Level-Specific Storyboard Guidance (INTERMEDIATE)", prompt_sb)
        self.assertIn("shadow projections", prompt_sb)
        self.assertIn("WHERE THE ANALOGY BREAKS DOWN", prompt_sb)
        self.assertIn("NEVER sacrifice mathematical correctness for the analogy", prompt_sb)

    def test_code_prompt_propagation(self):
        sec = Section(
            id="section_1",
            title="Sec 1: Introduction",
            lecture_lines=["Line 1", "Line 2"],
            animations=["Animation 1", "Animation 2"],
        )
        audio_meta = {
            "total_duration": 10.0,
            "lines": [
                {"line_index": 0, "duration": 4.5},
                {"line_index": 1, "duration": 5.5},
            ]
        }
        prompt_code = get_prompt3_code(
            regenerate_note="",
            section=sec,
            base_class=base_class,
            audio_meta=audio_meta,
            level="advanced",
            analogy="fluid flow"
        )
        self.assertIn("LEVEL-SPECIFIC VISUAL STYLING (ADVANCED)", prompt_code)
        self.assertIn("fluid flow", prompt_code)
        self.assertIn("PACING & AUDIO SYNCHRONIZATION", prompt_code)
        self.assertIn("Lecture Line 1 (index 0): target duration = 4.5s", prompt_code)
        self.assertIn("Lecture Line 2 (index 1): target duration = 5.5s", prompt_code)


if __name__ == "__main__":
    unittest.main()
