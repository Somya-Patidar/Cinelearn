"""
Tests for mathematical validation using SymPy and NumPy.
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

from math_validator import MathValidator, MathClaim
from agent import Section


class TestMathValidator(unittest.TestCase):
    def setUp(self):
        self.validator = MathValidator()

    def test_arithmetic_equality_verification(self):
        # True arithmetic
        status, claimed, computed, details = self.validator.verify_equality("1*4 - 2*3", "-2")
        self.assertEqual(status, "VERIFIED_TRUE")
        self.assertEqual(float(computed), -2.0)

        # LaTeX fraction arithmetic
        status, claimed, computed, details = self.validator.verify_equality(r"\frac{10}{2} + 3", "8")
        self.assertEqual(status, "VERIFIED_TRUE")

        # False arithmetic
        status, claimed, computed, details = self.validator.verify_equality("2 + 2", "5")
        self.assertEqual(status, "VERIFIED_FALSE")
        self.assertEqual(float(computed), 4.0)

    def test_algebraic_identity_verification(self):
        # Symbolic identity
        status, claimed, computed, details = self.validator.verify_equality("(x + y)**2", "x**2 + 2*x*y + y**2")
        self.assertEqual(status, "VERIFIED_TRUE")

        # False symbolic equation
        status, claimed, computed, details = self.validator.verify_equality("x + 1", "x + 2")
        self.assertEqual(status, "VERIFIED_FALSE")

    def test_matrix_determinant_verification(self):
        # 2x2 matrix determinant calculation
        mat = [[1.0, 2.0], [3.0, 4.0]]
        status, claimed, computed, details = self.validator.verify_matrix_determinant(mat, -2.0)
        self.assertEqual(status, "VERIFIED_TRUE")
        self.assertAlmostEqual(computed, -2.0, places=3)

        # Incorrect determinant claim
        status, claimed, computed, details = self.validator.verify_matrix_determinant(mat, 4.0)
        self.assertEqual(status, "VERIFIED_FALSE")

    def test_extract_and_verify_claims_from_text(self):
        text = "Notice that 3 * 4 = 12 and the matrix [[1, 0], [0, 1]] has determinant 1."
        claims = self.validator.extract_and_verify_claims_from_text(text, source="lecture_line")
        self.assertGreater(len(claims), 0)
        true_claims = [c for c in claims if c.status == "VERIFIED_TRUE"]
        self.assertGreater(len(true_claims), 0)

    def test_cross_check_consistency_narration_and_mathtex(self):
        lecture_lines = [
            "We calculate the area scaling factor, which is the determinant.",
            "The determinant is -2 for this transformation matrix.",
        ]

        # Case 1: Consistent code MathTex
        code_consistent = """
        class SecScene(TeachingScene):
            def construct(self):
                eq = MathTex(r"\\det(A) = -2")
                self.play(Write(eq))
        """
        inconsistencies = self.validator.cross_check_narration_and_code(lecture_lines, code_consistent)
        self.assertEqual(len(inconsistencies), 0)

        # Case 2: Inconsistent code MathTex (+2 instead of -2)
        code_inconsistent = """
        class SecScene(TeachingScene):
            def construct(self):
                eq = MathTex(r"\\det(A) = 2")
                self.play(Write(eq))
        """
        inconsistencies_bad = self.validator.cross_check_narration_and_code(lecture_lines, code_inconsistent)
        self.assertGreater(len(inconsistencies_bad), 0)
        self.assertIn("Consistency contradiction", inconsistencies_bad[0])

    def test_qualitative_claims_flagged_for_review(self):
        text = "The eigenvectors correspond to the invariant directions of the transformation."
        claims = self.validator.extract_and_verify_claims_from_text(text, source="lecture_line")
        review_claims = [c for c in claims if c.status == "NEEDS_REVIEW"]
        self.assertGreater(len(review_claims), 0)
        self.assertEqual(review_claims[0].claim_type, "qualitative_claim")

    def test_section_validation(self):
        sec = Section(
            id="sec_1",
            title="Sec 1: Determinants",
            lecture_lines=[
                "First, we calculate 2 * 5 = 10.",
                "Next, det(A) = ad - bc.",
            ],
            animations=[
                "Display MathTex formula 2 * 5 = 10",
            ],
        )
        report = self.validator.validate_section(sec)
        self.assertTrue(report.is_valid)
        self.assertGreater(report.verified_true_count, 0)
        self.assertEqual(report.verified_false_count, 0)


if __name__ == "__main__":
    unittest.main()
