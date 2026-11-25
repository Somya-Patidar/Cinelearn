"""
Tests for adaptive duration estimation, compression, reorganization, and timing validation.
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

from adaptive_duration import (
    AdaptiveDurationManager,
    LineDurationEstimate,
    SectionDurationEstimate,
    LessonDurationEstimate,
)


class TestAdaptiveDuration(unittest.TestCase):
    def setUp(self):
        self.mgr_beg = AdaptiveDurationManager(level="beginner", soft_target_duration=180.0, hard_max_duration=300.0)
        self.mgr_int = AdaptiveDurationManager(level="intermediate", soft_target_duration=240.0, hard_max_duration=400.0)
        self.mgr_adv = AdaptiveDurationManager(level="advanced", soft_target_duration=360.0, hard_max_duration=600.0)

    def test_level_based_defaults(self):
        mgr_default_beg = AdaptiveDurationManager(level="beginner")
        mgr_default_adv = AdaptiveDurationManager(level="advanced")
        self.assertEqual(mgr_default_beg.speaking_wpm, 135.0)
        self.assertEqual(mgr_default_adv.speaking_wpm, 155.0)
        self.assertEqual(mgr_default_beg.soft_target_sec, 210.0)
        self.assertEqual(mgr_default_adv.soft_target_sec, 360.0)

    def test_line_estimation_with_equation_and_example(self):
        # Plain line
        plain_line = "In this section we observe the basic geometric properties of the vector."
        est_plain = self.mgr_int.estimate_line(plain_line)
        self.assertFalse(est_plain.has_equation)
        self.assertFalse(est_plain.has_worked_example)

        # Line with equation
        eq_line = "We calculate the determinant using det(A) = ad - bc."
        est_eq = self.mgr_int.estimate_line(eq_line)
        self.assertTrue(est_eq.has_equation)
        self.assertGreater(est_eq.visual_overhead, est_plain.visual_overhead)

        # Line with worked example calculation
        ex_line = "For example, let us calculate step 1 by evaluating 2 * 3 = 6."
        est_ex = self.mgr_int.estimate_line(ex_line)
        self.assertTrue(est_ex.has_worked_example)
        self.assertGreater(est_ex.visual_overhead, est_plain.visual_overhead)

    def test_optimal_duration_decision(self):
        sb = {
            "sections": [
                {
                    "id": "sec_1",
                    "title": "Sec 1: Introduction",
                    "lecture_lines": ["Line 1 short", "Line 2 short"],
                    "animations": ["Fade in title", "Fade in point"],
                }
            ]
        }
        adapted_sb, est, report = self.mgr_int.adapt_storyboard_if_needed(sb)
        self.assertEqual(est.status, "OPTIMAL")
        self.assertEqual(report["final_status"], "OPTIMAL")

    def test_soft_target_exceeded_decision_and_compression(self):
        # Create storyboard with repetitive text that exceeds soft target
        mgr_tight = AdaptiveDurationManager(level="intermediate", soft_target_duration=25.0, hard_max_duration=120.0)
        sb = {
            "sections": [
                {
                    "id": "sec_1",
                    "title": "Sec 1: Core Principles",
                    "lecture_lines": [
                        "As we have already observed earlier, the matrix transformation maps vectors across space.",
                        "To reiterate what was said earlier, the origin remains strictly fixed at zero.",
                        "As mentioned in the previous section, the determinant det(A) measures the area scaling factor.",
                        "It goes without saying that the linear grid lines remain parallel and evenly spaced throughout.",
                        "Furthermore, the unit vectors i-hat and j-hat land directly at their new transformed coordinates.",
                    ],
                    "animations": ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"],
                }
            ]
        }
        initial_est = mgr_tight.estimate_lesson(sb)
        self.assertEqual(initial_est.status, "SOFT_TARGET_EXCEEDED")

        adapted_sb, final_est, report = mgr_tight.adapt_storyboard_if_needed(sb)
        self.assertIn("Streamlined repetitive narrative filler", report["actions_taken"][0])

        # Verify repetitive phrases were removed from lines
        for sec in adapted_sb["sections"]:
            for line in sec["lecture_lines"]:
                self.assertNotIn("As we have already observed earlier", line)
                self.assertNotIn("To reiterate what was said earlier", line)
                # Verify core definitions and equations are preserved
                if "determinant" in line.lower():
                    self.assertIn("det(A)", line)

    def test_hard_max_exceeded_multi_part_reorganization(self):
        # Create a large lesson that exceeds hard max duration
        mgr_small_max = AdaptiveDurationManager(level="beginner", soft_target_duration=30.0, hard_max_duration=60.0)
        sb = {
            "sections": [
                {
                    "id": f"section_{i}",
                    "title": f"Sec {i}: Complex Step {i}",
                    "lecture_lines": [
                        f"Detailed explanation line A for concept {i} discussing properties and structures.",
                        f"Detailed calculation step B for concept {i} evaluating equations x + y = {i}.",
                        f"Detailed worked example C for concept {i} showing concrete substitution.",
                        f"Detailed summary line D for concept {i} consolidating geometric intuition.",
                    ],
                    "animations": ["Anim 1", "Anim 2", "Anim 3", "Anim 4"],
                }
                for i in range(1, 6)
            ]
        }
        adapted_sb, final_est, report = mgr_small_max.adapt_storyboard_if_needed(sb)
        self.assertEqual(final_est.status, "HARD_MAX_EXCEEDED")
        self.assertIsNotNone(final_est.multi_part_plan)
        self.assertTrue(final_est.multi_part_plan["is_multi_part"])
        self.assertEqual(len(final_est.multi_part_plan["parts"]), 2)
        # Part 1 and Part 2 are non-empty and encompass all sections
        part1 = final_est.multi_part_plan["parts"][0]
        part2 = final_est.multi_part_plan["parts"][1]
        self.assertGreater(part1["section_count"], 0)
        self.assertGreater(part2["section_count"], 0)
        self.assertEqual(part1["section_count"] + part2["section_count"], 5)

    def test_line_timing_validation(self):
        audio_meta = {
            "section_id": "sec_test",
            "total_duration": 25.0,
            "lines": [
                {
                    "line_index": 0,
                    "raw_text": "Good normal length line with clear speech pacing.",
                    "duration": 4.2,
                },
                {
                    "line_index": 1,
                    "raw_text": "Too fast",
                    "duration": 0.4,  # too short (< 1.0s)
                },
                {
                    "line_index": 2,
                    "raw_text": " ".join(["word"] * 40),
                    "duration": 18.5,  # too long (> 16.0s)
                },
            ],
        }
        res = self.mgr_int.validate_line_timings(audio_meta)
        self.assertFalse(res["is_valid"])
        self.assertGreater(len(res["warnings"]), 0)

        # Check line status tags
        checks = res["line_checks"]
        self.assertEqual(checks[0]["status"], "PASS")
        self.assertEqual(checks[1]["status"], "WARN_TOO_FAST")
        self.assertEqual(checks[2]["status"], "WARN_TOO_LONG")


if __name__ == "__main__":
    unittest.main()
