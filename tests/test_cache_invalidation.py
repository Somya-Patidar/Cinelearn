"""
Tests for cache safety, output directory generation, and checkpoint invalidation in CineLearn.
"""

import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils import get_output_dir, topic_to_safe_name
from agent import TeachingVideoAgent, RunConfig


class TestCacheSafety(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_output_dir_backward_compatibility(self):
        # Default intermediate without analogy
        path_default = get_output_dir(0, "Linear Transformations", self.temp_dir)
        self.assertEqual(path_default.name, "0-Linear_Transformations")

        path_explicit_inter = get_output_dir(0, "Linear Transformations", self.temp_dir, level="intermediate", analogy="")
        self.assertEqual(path_explicit_inter.name, "0-Linear_Transformations")

    def test_get_output_dir_with_levels_and_analogies(self):
        # Beginner level
        path_beg = get_output_dir(0, "Linear Transformations", self.temp_dir, level="beginner")
        self.assertEqual(path_beg.name, "0-Linear_Transformations_beginner")

        # Advanced level
        path_adv = get_output_dir(0, "Linear Transformations", self.temp_dir, level="advanced")
        self.assertEqual(path_adv.name, "0-Linear_Transformations_advanced")

        # Beginner with analogy
        path_beg_analogy = get_output_dir(0, "Linear Transformations", self.temp_dir, level="beginner", analogy="rubber sheet")
        self.assertEqual(path_beg_analogy.name, "0-Linear_Transformations_beginner_rubber_sheet")

        # Intermediate with analogy
        path_int_analogy = get_output_dir(0, "Linear Transformations", self.temp_dir, level="intermediate", analogy="shadow puppets")
        self.assertEqual(path_int_analogy.name, "0-Linear_Transformations_shadow_puppets")

    def test_config_signatures_differ(self):
        cfg = RunConfig()
        agent1 = TeachingVideoAgent(0, "Calculus", folder=self.temp_dir, cfg=cfg, level="beginner")
        agent2 = TeachingVideoAgent(0, "Calculus", folder=self.temp_dir, cfg=cfg, level="advanced")
        agent3 = TeachingVideoAgent(0, "Calculus", folder=self.temp_dir, cfg=cfg, level="beginner", analogy="speedometer")

        sig1 = agent1.get_config_signature()
        sig2 = agent2.get_config_signature()
        sig3 = agent3.get_config_signature()

        self.assertNotEqual(sig1, sig2)
        self.assertNotEqual(sig1, sig3)
        self.assertNotEqual(sig2, sig3)

    def test_checkpoint_level_mismatch_prevents_incorrect_reuse(self):
        case_dir = Path(self.temp_dir) / "0-Calculus"
        case_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = case_dir / "checkpoint.json"

        # Write existing checkpoint for 'beginner' level
        cp_data = {
            "topic": "Calculus",
            "level": "beginner",
            "analogy": "",
            "stage": "completed",
            "outline": {"topic": "Calculus", "target_audience": "Beginners", "sections": [{"id": "sec_1"}]},
            "sections": [{"id": "sec_1", "title": "Sec 1", "lecture_lines": [], "animations": []}],
            "section_codes": {"sec_1": "class Sec1Scene(TeachingScene):\n    def construct(self):\n        pass"},
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(cp_data, f)

        # Initialize agent for 'advanced' level targeting the same folder
        agent_adv = TeachingVideoAgent(0, "Calculus", folder=self.temp_dir, level="advanced")
        # In a separate directory path by default, but let's test pointed at the same directory
        agent_adv.output_dir = case_dir
        agent_adv._load_existing_checkpoint_and_files()

        # Outline and sections must NOT be loaded because level mismatched!
        self.assertIsNone(agent_adv.outline)
        self.assertEqual(len(agent_adv.sections), 0)
        self.assertEqual(len(agent_adv.section_codes), 0)

    def test_checkpoint_analogy_mismatch_prevents_incorrect_reuse(self):
        case_dir = Path(self.temp_dir) / "0-Calculus"
        case_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = case_dir / "checkpoint.json"

        # Write existing checkpoint for analogy 'water flow'
        cp_data = {
            "topic": "Calculus",
            "level": "intermediate",
            "analogy": "water flow",
            "stage": "completed",
            "outline": {"topic": "Calculus", "target_audience": "AP Students", "sections": []},
            "sections": [],
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(cp_data, f)

        # Initialize agent with different analogy 'speedometer'
        agent_new = TeachingVideoAgent(0, "Calculus", folder=self.temp_dir, level="intermediate", analogy="speedometer")
        agent_new.output_dir = case_dir
        agent_new._load_existing_checkpoint_and_files()

        self.assertIsNone(agent_new.outline)

    def test_matching_checkpoint_is_reused(self):
        case_dir = get_output_dir(0, "Calculus", self.temp_dir, level="advanced", analogy="physics engine")
        case_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = case_dir / "checkpoint.json"

        cp_data = {
            "topic": "Calculus",
            "level": "advanced",
            "analogy": "physics engine",
            "stage": "outline_completed",
            "outline": {"topic": "Calculus", "target_audience": "Advanced", "sections": [{"id": "sec_1"}]},
            "sections": [{"id": "sec_1", "title": "Sec 1", "lecture_lines": [], "animations": []}],
        }
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(cp_data, f)

        # Initialize agent matching exact topic, level, and analogy
        agent_match = TeachingVideoAgent(0, "Calculus", folder=self.temp_dir, level="advanced", analogy="physics engine")
        self.assertIsNotNone(agent_match.outline)
        self.assertEqual(agent_match.outline.topic, "Calculus")
        self.assertEqual(len(agent_match.sections), 1)


if __name__ == "__main__":
    unittest.main()
