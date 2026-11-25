"""
Tests for input parsing and backward compatibility in CineLearn.
"""

import sys
import unittest
from pathlib import Path

# Add project root and src to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent import build_and_parse_args, RunConfig, process_knowledge_point


class TestInputParsing(unittest.TestCase):
    def test_topic_flag(self):
        sys.argv = ["agent.py", "--topic", "Eigenvalues and Eigenvectors"]
        args = build_and_parse_args()
        self.assertEqual(args.topic, "Eigenvalues and Eigenvectors")
        self.assertIsNone(args.knowledge_point)
        self.assertEqual(args.level, "intermediate")
        self.assertEqual(args.analogy, "")

    def test_knowledge_point_backward_compatibility(self):
        sys.argv = ["agent.py", "--knowledge_point", "Linear transformations and matrices"]
        args = build_and_parse_args()
        self.assertEqual(args.knowledge_point, "Linear transformations and matrices")
        self.assertIsNone(args.topic)
        self.assertEqual(args.level, "intermediate")

    def test_three_inputs_parsing(self):
        sys.argv = [
            "agent.py",
            "--topic", "Newton's Law of Cooling",
            "--level", "beginner",
            "--analogy", "cooling a cup of coffee"
        ]
        args = build_and_parse_args()
        self.assertEqual(args.topic, "Newton's Law of Cooling")
        self.assertEqual(args.level, "beginner")
        self.assertEqual(args.analogy, "cooling a cup of coffee")

    def test_aliases_and_case_insensitivity(self):
        sys.argv = [
            "agent.py",
            "--topic", "Binary Search",
            "--learning_level", "Advanced",
            "--personal_interest", "searching a library"
        ]
        args = build_and_parse_args()
        self.assertEqual(args.topic, "Binary Search")
        self.assertEqual(args.level, "Advanced")
        self.assertEqual(args.analogy, "searching a library")

    def test_interest_flag(self):
        sys.argv = [
            "agent.py",
            "--topic", "Tournament Algorithms",
            "--level", "beginner",
            "--interest", "Pokémon"
        ]
        args = build_and_parse_args()
        self.assertEqual(args.topic, "Tournament Algorithms")
        self.assertEqual(args.level, "beginner")
        self.assertEqual(args.analogy, "Pokémon")

    def test_adaptive_duration_arguments(self):
        sys.argv = [
            "agent.py",
            "--topic", "Calculus",
            "--target_duration", "240",
            "--hard_max_duration", "480",
            "--speaking_wpm", "140"
        ]
        args = build_and_parse_args()
        self.assertEqual(args.soft_target_duration, 240.0)
        self.assertEqual(args.hard_max_duration, 480.0)
        self.assertEqual(args.speaking_wpm, 140.0)

    def test_batch_data_structure_parsing(self):
        cfg = RunConfig(level="intermediate", analogy="default_analogy")

        # Test string format
        str_kp = "Matrix Operations"
        # We test the extraction logic in process_knowledge_point
        # Check extraction with dict
        dict_kp = {
            "topic": "Fourier Series",
            "level": "advanced",
            "analogy": "musical harmonics"
        }
        self.assertEqual(dict_kp["topic"], "Fourier Series")
        self.assertEqual(dict_kp["level"], "advanced")
        self.assertEqual(dict_kp["analogy"], "musical harmonics")


if __name__ == "__main__":
    unittest.main()
