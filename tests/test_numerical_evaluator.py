"""
Unit tests for CineLearn Numerical Evaluator (Milestone 1).

Covers:
- Safe AST evaluation & operator precedence
- Complexity bounds, recursion depth, and magnitude limits
- Security whitelisting and code-injection rejection
- Displayed precision extraction & half-ULP computation
- Rounding agreement vs. rounding contradiction
- Scale-aware relative tolerance & approximate values
- Integer precision policy
- Conservative formula trust matching & literal-to-input mapping
- Rejection of inverted, sign-reversed, and ambiguous formulas
- Arrhenius regression cases
"""

import math
import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from numerical_evaluator import (
    SafeASTEvaluator,
    PrecisionExtractor,
    ToleranceEvaluator,
    TrustedFormulaRegistry,
    FormulaNormalizer,
    NumericalStepVerifier,
    VerificationStatus,
    FormulaTrustStatus,
    DiagnosticCode,
)


# ============================================================================
# 1. Safe AST Evaluation Tests
# ============================================================================

class TestSafeASTEvaluator:
    def setup_method(self):
        self.evaluator = SafeASTEvaluator()

    def test_basic_arithmetic_and_precedence(self):
        # Precedence: 2 + 3 * 4 == 14, not 20
        assert self.evaluator.evaluate("2 + 3 * 4") == 14.0
        # Parentheses
        assert self.evaluator.evaluate("(2 + 3) * 4") == 20.0
        # Division and subtraction
        assert self.evaluator.evaluate("100 / 4 - 5") == 20.0
        # Unary negation
        assert self.evaluator.evaluate("-5 + 10") == 5.0
        assert self.evaluator.evaluate("-(5 + 10)") == -15.0

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            self.evaluator.evaluate("10 / 0")

        res = self.evaluator.safe_evaluate("10 / 0")
        assert res.status == VerificationStatus.NEEDS_REVIEW
        assert res.diagnostic == DiagnosticCode.DIVISION_BY_ZERO

    def test_negative_values_and_scientific_notation(self):
        # Python e-notation
        assert self.evaluator.evaluate("7.5e10") == 7.5e10
        assert self.evaluator.evaluate("-1.087e-4") == -1.087e-4
        assert self.evaluator.evaluate("2.5e3 * 4") == 10000.0

    def test_large_and_small_numbers(self):
        # Astronomical scale
        val_large = self.evaluator.evaluate("7.5e10 * 1e10")
        assert val_large == 7.5e20

        # Microscopic scale
        val_small = self.evaluator.evaluate("1e-15 * 5")
        assert math.isclose(val_small, 5e-15)

    def test_exponent_limits(self):
        # Exponent within limit (150)
        assert self.evaluator.evaluate("2 ** 10") == 1024.0

        # Exponent exceeding limit (>150)
        with pytest.raises(OverflowError, match="Exponent exceeds magnitude limit"):
            self.evaluator.evaluate("2 ** 200")

    def test_expression_length_and_complexity_limits(self):
        # Length exceeding 300
        long_expr = "1 + " * 100 + "1"
        with pytest.raises(ValueError, match="Expression length"):
            self.evaluator.evaluate(long_expr)

        # Depth limit exceeding 10
        deep_expr = "1 + " * 15 + "1"
        with pytest.raises(ValueError, match="AST recursion depth exceeded"):
            self.evaluator.evaluate(deep_expr)

    def test_unsupported_ast_constructs_and_code_injection(self):
        # Arbitrary attribute access
        with pytest.raises(TypeError, match="Disallowed"):
            self.evaluator.evaluate("math.sin(1)")

        # Code injection attempts
        with pytest.raises((TypeError, SyntaxError, NameError)):
            self.evaluator.evaluate("__import__('os').system('ls')")

        # Subscripts
        with pytest.raises(TypeError, match="Disallowed AST construct: Subscript"):
            self.evaluator.evaluate("a[0]", {"a": [1, 2]})

        # Lambda / function definitions
        with pytest.raises((TypeError, SyntaxError)):
            self.evaluator.evaluate("lambda x: x + 1")

        # Disallowed function calls
        with pytest.raises(NameError, match="Disallowed function call"):
            self.evaluator.evaluate("custom_func(5)")

    def test_whitelisted_math_functions(self):
        # exp
        assert math.isclose(self.evaluator.evaluate("exp(0)"), 1.0)
        assert math.isclose(self.evaluator.evaluate("exp(-1)"), math.exp(-1))

        # sqrt
        assert self.evaluator.evaluate("sqrt(16)") == 4.0

        # ln / log
        assert math.isclose(self.evaluator.evaluate("ln(10)"), math.log(10))

        # exp overflow guard (>100.0)
        with pytest.raises(OverflowError, match="exceeds exponent safety ceiling"):
            self.evaluator.evaluate("exp(150)")

        # sqrt negative domain guard
        with pytest.raises(ValueError, match="sqrt argument must be non-negative"):
            self.evaluator.evaluate("sqrt(-4)")


# ============================================================================
# 2. Precision Extraction & Tolerance Tests
# ============================================================================

class TestPrecisionAndTolerance:
    def test_standard_decimal_half_ulp(self):
        # -30.07 has 2 decimal places -> half_ulp = 0.005
        p1 = PrecisionExtractor.extract_precision("-30.07")
        assert p1.claimed_float == -30.07
        assert p1.decimal_places == 2
        assert p1.half_ulp == 0.005
        assert not p1.is_scientific
        assert not p1.is_integer

        # 0.0001087 has 7 decimal places -> half_ulp = 5e-8
        p2 = PrecisionExtractor.extract_precision("0.0001087")
        assert p2.decimal_places == 7
        assert math.isclose(p2.half_ulp, 5e-8)

    def test_scientific_notation_half_ulp(self):
        # 1.087e-4: significand has 3 decimal places, exponent = -4
        # LSD = 10^(-4 - 3) = 10^-7, half_ulp = 0.5 * 10^-7 = 5e-8
        p1 = PrecisionExtractor.extract_precision("1.087e-4")
        assert p1.is_scientific
        assert math.isclose(p1.claimed_float, 1.087e-4)
        assert math.isclose(p1.half_ulp, 5e-8)

        # 7.5e10: significand has 1 decimal place, exponent = 10
        # LSD = 10^(10 - 1) = 10^9, half_ulp = 0.5 * 10^9 = 5e8
        p2 = PrecisionExtractor.extract_precision("7.5e10")
        assert p2.is_scientific
        assert math.isclose(p2.claimed_float, 7.5e10)
        assert math.isclose(p2.half_ulp, 5e8)

    def test_unit_and_approximation_stripping(self):
        p1 = PrecisionExtractor.extract_precision(r"0.0001087 \, \text{s}^{-1}")
        assert math.isclose(p1.claimed_float, 0.0001087)
        assert p1.unit == "s^-1"
        assert not p1.is_approximate

        p2 = PrecisionExtractor.extract_precision(r"\approx -30.07 J/mol")
        assert p2.claimed_float == -30.07
        assert p2.is_approximate
        assert p2.unit == "J/mol"

    def test_rounding_match_vs_rounding_contradiction(self):
        # True value: -30.0697618...
        computed = -30.06976184748617

        # Case 1: Claimed -30.07 (correct 2-decimal rounding)
        # diff = |-30.07 - (-30.06976)| = 0.000238 <= 0.005 -> MATCH
        p_good = PrecisionExtractor.extract_precision("-30.07")
        match, diag, _ = ToleranceEvaluator.evaluate(p_good, computed)
        assert match is True
        assert diag == DiagnosticCode.ROUNDING_MATCH

        # Case 2: Claimed -29.74 (hallucinated value from Arrhenius validation)
        # diff = |-29.74 - (-30.06976)| = 0.32976 > 0.005 -> CONTRADICTION
        p_bad = PrecisionExtractor.extract_precision("-29.74")
        match_bad, diag_bad, _ = ToleranceEvaluator.evaluate(p_bad, computed)
        assert match_bad is False
        assert diag_bad == DiagnosticCode.ROUNDING_CONTRADICTION

    def test_relative_tolerance_cannot_override_rounding_contradiction(self):
        # Strict user contract test:
        # True value V = 10.000
        # Claimed C = 10.01 (2 decimal places, half_ulp = 0.005)
        # Relative difference is |10.01 - 10.00| / 10.00 = 0.1% (<= rtol 0.1%)
        # However, 10.01 contradicts its displayed precision (diff 0.010 > half_ulp 0.005)!
        # Policy requirement: Must NOT accept 10.01 under exact decimal mode!
        p = PrecisionExtractor.extract_precision("10.01")
        match, diag, _ = ToleranceEvaluator.evaluate(p, 10.000, rtol=1e-3)
        assert match is False
        assert diag == DiagnosticCode.ROUNDING_CONTRADICTION

    def test_approximate_marker_permits_relative_tolerance(self):
        # When explicitly marked approximate (~ 10.01 or \approx 10.01):
        p_approx = PrecisionExtractor.extract_precision("~ 10.01")
        assert p_approx.is_approximate
        match, diag, _ = ToleranceEvaluator.evaluate(p_approx, 10.000, rtol=1.5e-3)
        assert match is True
        assert diag == DiagnosticCode.RELATIVE_MATCH

    def test_integer_precision_policy(self):
        # Default exact policy: requires exact integer match (< 1e-12)
        p_int_exact = PrecisionExtractor.extract_precision("42", integer_policy="exact")
        assert p_int_exact.is_integer
        assert p_int_exact.half_ulp == 1e-12

        match, _, _ = ToleranceEvaluator.evaluate(p_int_exact, 42.0)
        assert match is True

        # Claiming integer 42 for 42.1 fails under exact policy
        match_fail, _, _ = ToleranceEvaluator.evaluate(p_int_exact, 42.1)
        assert match_fail is False

        # Unit rounding policy: integer 42 represents [41.5, 42.5]
        p_int_unit = PrecisionExtractor.extract_precision("42", integer_policy="unit_rounding")
        assert p_int_unit.half_ulp == 0.5
        match_unit, diag_unit, _ = ToleranceEvaluator.evaluate(p_int_unit, 42.1)
        assert match_unit is True
        assert diag_unit == DiagnosticCode.ROUNDING_MATCH


# ============================================================================
# 3. Formula Normalization & Registry Tests
# ============================================================================

class TestFormulaRegistry:
    def test_arrhenius_exponent_formula_matching(self):
        inputs = {"Ea": 75000, "R": 8.314, "T": 300}

        # 1. Direct symbolic formula
        status1, tmpl1, _ = TrustedFormulaRegistry.check_formula_trust(
            "-Ea / (R * T)", inputs
        )
        assert status1 == FormulaTrustStatus.TRUSTED
        assert tmpl1 == "chemistry.arrhenius_exponent"

        # 2. Commutative variation (T * R instead of R * T)
        status2, tmpl2, _ = TrustedFormulaRegistry.check_formula_trust(
            "-Ea / (T * R)", inputs
        )
        assert status2 == FormulaTrustStatus.TRUSTED
        assert tmpl2 == "chemistry.arrhenius_exponent"

        # 3. Literal-to-input mapping: "-75000 / (8.314 * 300)"
        status3, tmpl3, _ = TrustedFormulaRegistry.check_formula_trust(
            "-75000 / (8.314 * 300)", inputs
        )
        assert status3 == FormulaTrustStatus.TRUSTED
        assert tmpl3 == "chemistry.arrhenius_exponent"

        # 4. LaTeX fraction format: r"-\frac{75000}{8.314 \times 300}"
        status4, tmpl4, _ = TrustedFormulaRegistry.check_formula_trust(
            r"-\frac{75000}{8.314 \times 300}", inputs
        )
        assert status4 == FormulaTrustStatus.TRUSTED
        assert tmpl4 == "chemistry.arrhenius_exponent"

    def test_rejection_of_inverted_and_sign_reversed_formulas(self):
        inputs = {"Ea": 75000, "R": 8.314, "T": 300}

        # Inverted Arrhenius: (R * T) / Ea
        status1, _, _ = TrustedFormulaRegistry.check_formula_trust(
            "(8.314 * 300) / 75000", inputs
        )
        assert status1 == FormulaTrustStatus.UNTRUSTED

        # Missing negative sign: Ea / (R * T)
        status2, _, _ = TrustedFormulaRegistry.check_formula_trust(
            "75000 / (8.314 * 300)", inputs
        )
        assert status2 == FormulaTrustStatus.UNTRUSTED

        # Spurious constant: -75000 / (8.314 * 300 * 2)
        status3, _, _ = TrustedFormulaRegistry.check_formula_trust(
            "-75000 / (8.314 * 300 * 2)", inputs
        )
        assert status3 == FormulaTrustStatus.UNTRUSTED

    def test_duplicate_and_ambiguous_literal_mappings(self):
        # If inputs contain two distinct variables with the EXACT same value:
        # e.g. T1 = 300 and T2 = 300
        # Then substituting literal 300 is ambiguous!
        ambiguous_inputs = {"Ea": 75000, "R": 8.314, "T1": 300, "T2": 300}
        status, _, reason = TrustedFormulaRegistry.check_formula_trust(
            "-75000 / (8.314 * 300)", ambiguous_inputs
        )
        assert status == FormulaTrustStatus.AMBIGUOUS
        assert "ambiguous" in reason.lower()

    def test_missing_variables_and_domain_constraints(self):
        # Missing 'T'
        incomplete_inputs = {"Ea": 75000, "R": 8.314}
        status, _, reason = TrustedFormulaRegistry.check_formula_trust(
            "-Ea / (R * T)", incomplete_inputs
        )
        assert status == FormulaTrustStatus.UNTRUSTED
        assert "Missing required input" in reason

        # Negative temperature violates domain constraint (T > 0 K)
        unphysical_inputs = {"Ea": 75000, "R": 8.314, "T": -300}
        status_unphys, _, reason_unphys = TrustedFormulaRegistry.check_formula_trust(
            "-Ea / (R * T)", unphysical_inputs
        )
        assert status_unphys == FormulaTrustStatus.UNTRUSTED
        assert "violates domain constraint" in reason_unphys

    def test_governing_principles_matching(self):
        inputs = {"m": 10, "a": 5}
        gps = ["Newton second law: F = m * a"]

        status, tmpl, _ = TrustedFormulaRegistry.check_formula_trust(
            "m * a", inputs, governing_principles=gps
        )
        assert status == FormulaTrustStatus.TRUSTED
        assert tmpl == "governing_principles.explicit_equation"


# ============================================================================
# 4. Integrated Step Verifier Tests
# ============================================================================

class TestNumericalStepVerifier:
    def setup_method(self):
        self.verifier = NumericalStepVerifier()

    def test_arrhenius_step1_verification_and_safe_correction_flag(self):
        inputs = {"Ea": 75000, "R": 8.314, "T": 300}

        # Step 1 with wrong LLM hallucination: -29.74
        raw_step = "-75000 / (8.314 * 300) = -29.74"
        res = self.verifier.verify_step(raw_step, inputs)

        # Formula trust is trusted, arithmetic is failed
        assert res.status == VerificationStatus.FAILED
        assert res.formula_trust == FormulaTrustStatus.TRUSTED
        assert res.diagnostic == DiagnosticCode.ROUNDING_CONTRADICTION
        assert math.isclose(res.computed_value, -30.06976184748617)
        # Because formula is trusted, it can be safely corrected
        assert res.can_safely_correct is True

    def test_arrhenius_step1_verified_with_correct_rounding(self):
        inputs = {"Ea": 75000, "R": 8.314, "T": 300}

        # Step 1 with correct rounding: -30.07
        raw_step = "-75000 / (8.314 * 300) = -30.07"
        res = self.verifier.verify_step(raw_step, inputs)

        assert res.status == VerificationStatus.VERIFIED
        assert res.formula_trust == FormulaTrustStatus.TRUSTED
        assert res.diagnostic == DiagnosticCode.ROUNDING_MATCH
        assert res.can_safely_correct is False

    def test_untrusted_formula_never_auto_corrected(self):
        inputs = {"Ea": 75000, "R": 8.314, "T": 300}

        # Inverted Arrhenius formula: (8.314 * 300) / 75000 = 10
        raw_step = "(8.314 * 300) / 75000 = 10"
        res = self.verifier.verify_step(raw_step, inputs)

        # Marked NEEDS_REVIEW because formula is untrusted
        assert res.status == VerificationStatus.NEEDS_REVIEW
        assert res.formula_trust == FormulaTrustStatus.UNTRUSTED
        # Crucial safety requirement: can_safely_correct MUST be False!
        assert res.can_safely_correct is False

    def test_non_evaluable_step_handling(self):
        # Algorithmic state update without '='
        res = self.verifier.verify_step("Initialize priority queue Q with all nodes", {})
        assert res.status == VerificationStatus.NOT_APPLICABLE
        assert res.diagnostic == DiagnosticCode.NON_EVALUABLE_STEP

    def test_latex_sqrt_and_logarithms_normalization(self):
        evaluator = SafeASTEvaluator()
        # Square root
        s1 = FormulaNormalizer.sanitize_latex(r"\sqrt{16}")
        assert s1 == "sqrt(16)"
        assert evaluator.evaluate(s1) == 4.0

        # N-th root
        s2 = FormulaNormalizer.sanitize_latex(r"\sqrt[3]{27}")
        assert evaluator.evaluate(s2) == 3.0

        # Natural logarithm
        s3 = FormulaNormalizer.sanitize_latex(r"\ln(1)")
        assert s3 == "ln(1)"
        assert evaluator.evaluate(s3) == 0.0

        s4 = FormulaNormalizer.sanitize_latex(r"\ln{2.718281828459}")
        assert math.isclose(evaluator.evaluate(s4), 1.0, rel_tol=1e-5)

        # Malformed root safely handled
        s_malformed = FormulaNormalizer.sanitize_latex(r"\sqrt{")
        res_malformed = evaluator.safe_evaluate(s_malformed)
        assert res_malformed.value is None
        assert res_malformed.details != ""
