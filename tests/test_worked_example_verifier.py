"""
Unit tests for CineLearn WorkedExampleVerifier (Milestone 2).

Covers:
- Dual-fidelity adapter (legacy strings and structured calculation dicts)
- Preservation of original values and representations
- Multi-step dependency tracking and calculation environment
- Safe auto-correction of trusted formulas with arithmetic errors
- Complete multi-step Arrhenius calculation across T1 (300 K) and T2 (350 K)
- Downstream recalculation using corrected intermediate results
- Downstream blocking when a dependency is untrusted or unresolved
- Rejection of untrusted formula propagation
- Ambiguous literal mapping and missing variables
- Mixed numerical, prose, and algorithmic steps
- Missing or empty worked_example_spec
- Repeated output variable names
- Non-destructive history preservation and serialization compatibility
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
    DiagnosticCode,
    FormulaTrustStatus,
    VerificationStatus,
)
from worked_example_verifier import (
    AdaptedStep,
    CalculationEnvironment,
    CalculationStepAdapter,
    VariableTrust,
    WorkedExampleReport,
    WorkedExampleVerifier,
    format_corrected_value,
)


# ============================================================================
# 1. CalculationStepAdapter Tests
# ============================================================================

class TestCalculationStepAdapter:
    def test_legacy_string_adaptation(self):
        legacy_step = {
            "step": "Calculate the exponent term for T1: -Ea / (R * T1)",
            "result": "-75000 / (8.314 * 300) = -29.74",
        }
        adapted = CalculationStepAdapter.adapt_step(legacy_step, 0)

        assert adapted.is_evaluable is True
        assert adapted.expression == "-75000 / (8.314 * 300)"
        assert adapted.claimed_value == "-29.74"
        assert adapted.output_variable == "exp_T1"
        assert adapted.raw_result == "-75000 / (8.314 * 300) = -29.74"
        assert adapted.original_dict == legacy_step

    def test_structured_calculation_adaptation(self):
        structured_step = {
            "step": "Calculate rate constant k1",
            "expression": "A * exp(exp_T1)",
            "claimed_value": "0.006545",
            "output_variable": "k1",
            "unit": "s^-1",
            "is_approximate": False,
        }
        adapted = CalculationStepAdapter.adapt_step(structured_step, 1)

        assert adapted.is_evaluable is True
        assert adapted.expression == "A * exp(exp_T1)"
        assert adapted.claimed_value == "0.006545"
        assert adapted.output_variable == "k1"
        assert adapted.unit == "s^-1"

    def test_algorithmic_state_update_not_evaluable(self):
        # Dijkstra queue or dist update
        dijkstra_step = {
            "step": "Initialize all distances as infinity except source A, set dist[A] = 0.",
            "result": "dist[A] = 0, dist[B] = ∞, dist[C] = ∞",
        }
        adapted = CalculationStepAdapter.adapt_step(dijkstra_step, 0)
        assert adapted.is_evaluable is False
        assert adapted.raw_result == "dist[A] = 0, dist[B] = ∞, dist[C] = ∞"

    def test_infer_output_variable_precedence(self):
        # 1. Explicit tag
        assert CalculationStepAdapter.infer_output_variable("compute", "1+1", 0, "my_var") == "my_var"
        # 2. Description keyword k2
        assert CalculationStepAdapter.infer_output_variable("Compute k2 rate constant", "7.5e10 * exp(-25)", 1) == "k2"
        # 3. Fallback to step index
        assert CalculationStepAdapter.infer_output_variable("General calculation", "10 / 2", 3) == "step_4_val"


# ============================================================================
# 2. Multi-Step Dependency Environment Tests
# ============================================================================

class TestCalculationEnvironment:
    def test_environment_initialization_with_inputs(self):
        inputs = {"Ea": 75000, "R": 8.314, "T1": 300}
        env = CalculationEnvironment(inputs)

        assert env.is_trusted("Ea") is True
        assert env.is_trusted("R") is True
        assert env.is_trusted("T1") is True
        assert env.get("Ea").value == 75000.0
        assert env.get("Ea").trust == VariableTrust.TRUSTED_INPUT

    def test_step_result_registration_and_trust(self):
        env = CalculationEnvironment({"a": 10})
        # Register verified step
        env.set_step_result(
            step_index=0,
            var_name="b",
            value=20.0,
            trust=VariableTrust.VERIFIED_INTERMEDIATE,
            diagnostic="EXACT_MATCH",
        )
        assert env.is_trusted("b") is True
        assert env.get("b").value == 20.0

        # Register untrusted step
        env.set_unresolved(
            step_index=1,
            var_name="c",
            diagnostic="UNTRUSTED_FORMULA",
        )
        assert env.is_trusted("c") is False
        assert env.get("c").value is None

    def test_repeated_output_variable_names(self):
        env = CalculationEnvironment()
        # Step 0 assigns dist = 5
        env.set_step_result(0, "dist", 5.0, VariableTrust.VERIFIED_INTERMEDIATE)
        assert env.get("dist").value == 5.0

        # Step 1 re-assigns dist = 3 (relaxing edge in graph algorithm)
        env.set_step_result(1, "dist", 3.0, VariableTrust.VERIFIED_INTERMEDIATE)
        assert env.get("dist").value == 3.0
        # History preserves both assignment records
        assert len(env.history) == 2
        assert env.history[0].value == 5.0
        assert env.history[1].value == 3.0

    def test_substitute_corrected_literal_dependencies(self):
        env = CalculationEnvironment()
        # Suppose step 0 claimed -29.74, but was corrected to -30.069762
        env.set_step_result(
            step_index=0,
            var_name="exp_T1",
            value=-30.06976184748617,
            trust=VariableTrust.CORRECTED_INTERMEDIATE,
            claimed_float=-29.74,
        )

        # Subsequent step literally wrote '-29.74': '7.5e10 * exp(-29.74)'
        expr = "7.5e10 * exp(-29.74)"
        sub_expr, modified, reasons = env.substitute_dependencies(expr)
        assert modified is True
        assert "-30.06976" in sub_expr
        assert len(reasons) > 0


# ============================================================================
# 3. Full Arrhenius Kinetics Verification Tests
# ============================================================================

class TestArrheniusVerification:
    def setup_method(self):
        self.verifier = WorkedExampleVerifier()
        self.canonical_inputs = {
            "Ea": 75000,
            "R": 8.314,
            "T1": 300,
            "T2": 350,
            "A": 75000000000.0,
        }

    def test_arrhenius_full_cascade_with_hallucinations(self):
        """
        The crucial CineLearn regression test:
        Stage 1 generated outline with hallucinated arithmetic:
        - T1 exponent: claimed -29.74 (true: -30.07)
        - T2 exponent: claimed -24.49 (true: -25.77)
        - k1 rate: claimed 0.0001087 (true: 0.006545)
        - k2 rate: claimed 0.00284 (true: 0.4803)
        - final_result: "k1 = 0.0001087 s^-1, k2 = 0.00284 s^-1"
        Verify that all 4 steps are safely auto-corrected and final_result is updated.
        """
        raw_spec = {
            "setup": "Calculate rate constant k at T1=300K and T2=350K",
            "inputs": dict(self.canonical_inputs),
            "steps": [
                {
                    "step": "Calculate the exponent term for T1: -Ea / (R * T1)",
                    "result": "-75000 / (8.314 * 300) = -29.74",
                },
                {
                    "step": "Calculate the exponent term for T2: -Ea / (R * T2)",
                    "result": "-75000 / (8.314 * 350) = -24.49",
                },
                {
                    "step": "Compute k1 = A * exp(-Ea / (R * T1))",
                    "result": "7.5e10 * exp(-29.74) = 0.0001087 s^-1",
                },
                {
                    "step": "Compute k2 = A * exp(-Ea / (R * T2))",
                    "result": "7.5e10 * exp(-24.49) = 0.00284 s^-1",
                },
            ],
            "final_result": "k1 = 0.0001087 s^-1, k2 = 0.00284 s^-1",
        }

        updated_spec, report = self.verifier.verify_worked_example_spec(raw_spec)

        # 1. Report verification
        assert report.total_steps == 4
        assert report.corrected_steps == 4
        assert report.review_steps == 0
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_stage2 is True

        # 2. Step 1 (T1 Exponent) Verification & Correction
        s1 = updated_spec["steps"][0]
        assert s1["original_result"] == "-75000 / (8.314 * 300) = -29.74"
        assert s1["result"] == "-75000 / (8.314 * 300) = -30.07"
        assert s1["structured_calc"]["corrected_value"] == "-30.07"
        assert math.isclose(s1["structured_calc"]["computed_value"], -30.06976184748617)

        # 3. Step 2 (T2 Exponent) Verification & Correction
        s2 = updated_spec["steps"][1]
        assert s2["original_result"] == "-75000 / (8.314 * 350) = -24.49"
        assert s2["result"] == "-75000 / (8.314 * 350) = -25.77"
        assert s2["structured_calc"]["corrected_value"] == "-25.77"
        assert math.isclose(s2["structured_calc"]["computed_value"], -25.77408158356199)

        # 4. Step 3 (k1 Rate) Recalculated with Corrected Exponent
        s3 = updated_spec["steps"][2]
        assert s3["original_result"] == "7.5e10 * exp(-29.74) = 0.0001087 s^-1"
        assert "0.006545" in s3["result"]
        assert s3["structured_calc"]["unit"] == "s^-1"
        assert math.isclose(s3["structured_calc"]["computed_value"], 0.006545300963146607, rel_tol=1e-4)

        # 5. Step 4 (k2 Rate) Recalculated with Corrected Exponent
        s4 = updated_spec["steps"][3]
        assert s4["original_result"] == "7.5e10 * exp(-24.49) = 0.00284 s^-1"
        assert "0.4803" in s4["result"]
        assert s4["structured_calc"]["unit"] == "s^-1"
        assert math.isclose(s4["structured_calc"]["computed_value"], 0.4803080447021703, rel_tol=1e-4)

        # 6. Final Result String Reconciled
        assert updated_spec["original_final_result"] == "k1 = 0.0001087 s^-1, k2 = 0.00284 s^-1"
        assert "k1 = 0.006545 s^-1" in updated_spec["final_result"]
        assert "k2 = 0.4803" in updated_spec["final_result"]

    def test_arrhenius_already_verified_correct(self):
        """When the outline already contains correct figures, no corrections are made."""
        correct_spec = {
            "setup": "Arrhenius calculation",
            "inputs": dict(self.canonical_inputs),
            "steps": [
                {
                    "step": "Calculate exponent T1",
                    "result": "-75000 / (8.314 * 300) = -30.07",
                },
                {
                    "step": "Compute k1",
                    "result": "7.5e10 * exp(-30.07) = 0.006545 s^-1",
                },
            ],
            "final_result": "k1 = 0.006545 s^-1",
        }
        updated_spec, report = self.verifier.verify_worked_example_spec(correct_spec)

        assert report.verified_steps == 2
        assert report.corrected_steps == 0
        assert report.overall_status == VerificationStatus.VERIFIED
        assert "original_result" not in updated_spec["steps"][0]


# ============================================================================
# 4. Dependency Blocking & Safety Policies
# ============================================================================

class TestSafetyAndBlocking:
    def setup_method(self):
        self.verifier = WorkedExampleVerifier()

    def test_downstream_blocking_when_dependency_untrusted(self):
        """
        If Step 1 uses an inverted untrusted formula, Step 1 is marked NEEDS_REVIEW.
        Step 2 depends on Step 1, so Step 2 must be BLOCKED (NEEDS_REVIEW) rather
        than computing on untrusted data.
        """
        spec = {
            "inputs": {"Ea": 75000, "R": 8.314, "T": 300, "A": 7.5e10},
            "steps": [
                {
                    "step": "Calculate exp_T1 with inverted formula (Untrusted)",
                    "result": "exp_T1 = (8.314 * 300) / 75000 = 0.033",
                },
                {
                    "step": "Compute k1 using previous step",
                    "result": "A * exp(exp_T1) = 7.5e10",
                },
            ],
        }

        updated_spec, report = self.verifier.verify_worked_example_spec(spec)

        # Step 1: Untrusted formula -> NEEDS_REVIEW
        s1 = updated_spec["steps"][0]
        assert s1["structured_calc"]["status"] == VerificationStatus.NEEDS_REVIEW.value
        assert s1["structured_calc"]["diagnostic"] == DiagnosticCode.UNTRUSTED_FORMULA.value
        assert "original_result" not in s1  # Untrusted formulas are NEVER auto-corrected

        # Step 2: Blocked because dependency exp_T1 is unresolved
        s2 = updated_spec["steps"][1]
        assert s2["structured_calc"]["status"] == VerificationStatus.NEEDS_REVIEW.value
        assert s2["structured_calc"]["diagnostic"] == DiagnosticCode.MISSING_VARIABLE.value
        assert "unresolved" in s2["structured_calc"]["details"].lower()
        assert report.overall_status == VerificationStatus.NEEDS_REVIEW
        assert report.is_safe_for_stage2 is False

    def test_ambiguous_literal_mapping_blocks_correction(self):
        # Two temperatures have identical values 300 K
        ambiguous_spec = {
            "inputs": {"Ea": 75000, "R": 8.314, "T1": 300, "T2": 300},
            "steps": [
                {
                    "step": "Calculate exponent",
                    "result": "-75000 / (8.314 * 300) = -29.74",
                }
            ],
        }
        updated_spec, report = self.verifier.verify_worked_example_spec(ambiguous_spec)
        s = updated_spec["steps"][0]

        assert s["structured_calc"]["status"] == VerificationStatus.NEEDS_REVIEW.value
        assert s["structured_calc"]["can_safely_correct"] is False
        assert report.overall_status == VerificationStatus.NEEDS_REVIEW

    def test_missing_variables_marked_needs_review(self):
        missing_var_spec = {
            "inputs": {"Ea": 75000},  # Missing R and T
            "steps": [
                {
                    "step": "Calculate exponent",
                    "result": "-Ea / (R * T) = -30.07",
                }
            ],
        }
        updated_spec, report = self.verifier.verify_worked_example_spec(missing_var_spec)
        s = updated_spec["steps"][0]
        assert s["structured_calc"]["status"] == VerificationStatus.NEEDS_REVIEW.value


# ============================================================================
# 5. Outline Verification Entry Point & Edge Cases
# ============================================================================

class TestOutlineVerificationEntryPoint:
    def setup_method(self):
        self.verifier = WorkedExampleVerifier()

    def test_verify_outline_with_missing_spec(self):
        # Outline with no worked_example_spec
        outline = {"topic": "Calculus", "sections": []}
        updated, report = self.verifier.verify_outline(outline)

        assert updated == outline
        assert report.overall_status == VerificationStatus.NOT_APPLICABLE
        assert report.total_steps == 0
        assert report.is_safe_for_stage2 is True

    def test_verify_outline_with_mixed_prose_and_math(self):
        # Dijkstra with non-evaluable prose setup followed by arithmetic step
        outline = {
            "topic": "Graph Algorithm",
            "worked_example_spec": {
                "inputs": {"edge_weight": 5, "current_dist": 10},
                "steps": [
                    {
                        "step": "Initialize dist[A] = 0, dist[B] = ∞",
                        "result": "dist[A] = 0, dist[B] = ∞",
                    },
                    {
                        "step": "Relax edge: current_dist + edge_weight",
                        "result": "10 + 5 = 15",
                    },
                ],
                "final_result": "dist = 15",
            },
        }

        updated, report = self.verifier.verify_outline(outline)
        steps = updated["worked_example_spec"]["steps"]

        # Step 0: Algorithmic state update -> NOT_APPLICABLE
        assert steps[0]["structured_calc"]["status"] == VerificationStatus.NOT_APPLICABLE.value
        # Step 1: Elementary arithmetic -> VERIFIED
        assert steps[1]["structured_calc"]["status"] == VerificationStatus.VERIFIED.value

        assert report.verified_steps == 1
        assert report.not_applicable_steps == 1
        assert report.overall_status == VerificationStatus.VERIFIED

    def test_serialization_compatibility(self):
        # Verify that output dictionary is pure JSON-serializable (no custom objects)
        import json

        outline = {
            "topic": "Kinetics",
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

        updated, report = self.verifier.verify_outline(outline)

        # Must serialize cleanly with json.dumps without TypeError
        serialized = json.dumps(updated)
        deserialized = json.loads(serialized)
        assert "verification_metadata" in deserialized["worked_example_spec"]
        assert deserialized["worked_example_spec"]["steps"][0]["structured_calc"]["can_safely_correct"] is True


# ============================================================================
# 6. Targeted Hardening Regression Tests (Step 2B.4B.2B)
# ============================================================================

class TestTargetedHardening:
    def setup_method(self):
        self.verifier = WorkedExampleVerifier()

    def test_contextual_t2_mapping_regression(self):
        """
        Reproduce the exact audit failure scenario:
        Step 4 calculates k2, but its description and result text do NOT contain
        'T2' or '350'. It only says 'Compute rate constant k2'.
        Verify that k2 is correctly evaluated using T2 (350 K) -> 0.4803 s^-1,
        and is NOT evaluated using T1 (300 K) -> 0.006545 s^-1.
        """
        spec = {
            "inputs": {
                "Ea": 75000,
                "R": 8.314,
                "T1": 300,
                "T2": 350,
                "A": 7.5e10,
            },
            "steps": [
                {
                    "step": "Calculate exponent for T1",
                    "result": "-75000 / (8.314 * 300) = -29.74",
                },
                {
                    "step": "Compute k1",
                    "result": "7.5e10 * exp(-29.74) = 0.0001087 s^-1",
                },
                {
                    "step": "Calculate exponent for T2",
                    "result": "-75000 / (8.314 * 350) = -24.49",
                },
                {
                    # Neither 'T2' nor '350' in description/result, only 'k2'
                    "step": "Compute rate constant k2",
                    "result": "7.5e10 * exp(-24.49) = 0.00284 s^-1",
                },
            ],
            "final_result": "k1 = 0.0001087 s^-1, k2 = 0.00284 s^-1",
        }

        updated, report = self.verifier.verify_worked_example_spec(spec)
        s4 = updated["steps"][3]

        # Crucial check: k2 MUST evaluate near 0.4803 (T2=350K), NOT near 0.006545 (T1=300K)
        computed_k2 = s4["structured_calc"]["computed_value"]
        assert math.isclose(computed_k2, 0.4803080447021703, rel_tol=1e-4)
        assert "0.4803" in s4["result"]
        assert s4["structured_calc"]["resolution"] == "AUTO_CORRECTED"
        assert report.overall_status == VerificationStatus.VERIFIED

    def test_safe_final_result_reconciliation_token_boundary(self):
        """
        Token-aware matching: correcting '3' -> '4' must NOT mutate '300 K' or '13.5'.
        """
        env = CalculationEnvironment()
        env.claimed_to_corrected[3.0] = 4.0

        raw_final = "At 300 K, value is 13.5 with n of 3 mol and factor of 3."
        reconciled, status = self.verifier.verify_and_reconcile_final_result(raw_final, env)

        assert "300 K" in reconciled
        assert "13.5" in reconciled
        assert "3 mol" not in reconciled
        assert "4 mol" in reconciled
        assert "factor of 4." in reconciled
        assert status == VerificationStatus.VERIFIED

        # Also test variable assignment with token boundaries
        env.set_step_result(step_index=0, var_name="factor", value=4.0, trust=VariableTrust.CORRECTED_INTERMEDIATE, claimed_sample="3", claimed_float=3.0)
        reconciled_var, status_var = self.verifier.verify_and_reconcile_final_result("factor = 3", env)
        assert reconciled_var == "factor = 4"
        assert status_var == VerificationStatus.VERIFIED

    def test_final_result_verification_status_detection(self):
        """
        Final result status must reflect trustworthiness:
        - Missing or untrusted variable -> NEEDS_REVIEW
        - Unreconciled arbitrary numbers -> NEEDS_REVIEW
        - Reconciled or verified numbers -> VERIFIED
        - Conceptual text without numbers -> NOT_APPLICABLE
        """
        env = CalculationEnvironment({"a": 10.0})

        # Case A: Untrusted variable in final result
        env.set_unresolved(step_index=0, var_name="bad_var", diagnostic="Untrusted")
        res_a, status_a = self.verifier.verify_and_reconcile_final_result("bad_var = 5.0", env)
        assert status_a == VerificationStatus.NEEDS_REVIEW

        # Case B: Arbitrary unreconciled numbers
        res_b, status_b = self.verifier.verify_and_reconcile_final_result("unknown = 99999.0", env)
        assert status_b == VerificationStatus.NEEDS_REVIEW

        # Case C: Conceptual text without numbers
        res_c, status_c = self.verifier.verify_and_reconcile_final_result("Reaction reaches equilibrium.", env)
        assert status_c == VerificationStatus.NOT_APPLICABLE

        # Case D: Verified variable
        env.set_step_result(step_index=1, var_name="good_var", value=42.0, trust=VariableTrust.VERIFIED_INTERMEDIATE)
        res_d, status_d = self.verifier.verify_and_reconcile_final_result("good_var = 42.0", env)
        assert status_d == VerificationStatus.VERIFIED

    def test_idempotent_repeated_verification_preserves_original_claims(self):
        """
        Verifying an already-corrected outline must NOT overwrite original claims
        with corrected values.
        """
        raw_spec = {
            "inputs": {"Ea": 75000, "R": 8.314, "T": 300},
            "steps": [
                {
                    "step": "Calculate exponent",
                    "result": "-75000 / (8.314 * 300) = -29.74",
                }
            ],
            "final_result": "-29.74",
        }

        # Pass 1: Auto-correction
        pass1_spec, rep1 = self.verifier.verify_worked_example_spec(raw_spec)
        assert rep1.corrected_steps == 1
        assert pass1_spec["steps"][0]["original_result"] == "-75000 / (8.314 * 300) = -29.74"
        assert pass1_spec["steps"][0]["result"] == "-75000 / (8.314 * 300) = -30.07"
        assert pass1_spec["original_final_result"] == "-29.74"
        assert pass1_spec["final_result"] == "-30.07"

        # Pass 2: Re-verification on already-corrected spec
        pass2_spec, rep2 = self.verifier.verify_worked_example_spec(pass1_spec)
        assert rep2.corrected_steps == 0
        assert rep2.verified_steps == 1
        assert rep2.overall_status == VerificationStatus.VERIFIED

        # Idempotency: Original claims are still preserved!
        assert pass2_spec["steps"][0]["original_result"] == "-75000 / (8.314 * 300) = -29.74"
        assert pass2_spec["steps"][0]["result"] == "-75000 / (8.314 * 300) = -30.07"
        assert pass2_spec["original_final_result"] == "-29.74"
        assert pass2_spec["final_result"] == "-30.07"

    def test_failed_count_distinct_from_review_steps(self):
        """
        A non-correctable failed step increments failed_steps (not review_steps)
        and sets overall_status to FAILED.
        """
        # Step arithmetic fails tolerance with trusted elementary arithmetic,
        # but claimed value is non-numeric/catastrophic mismatch where can_safely_correct is False
        spec = {
            "inputs": {"x": 10.0, "y": 20.0},
            "steps": [
                {
                    "step": "Sum: x + y",
                    "result": "10.0 + 20.0 = 999.0",
                    "allow_auto_correction": False,
                }
            ],
            "final_result": "30.0",
        }

        updated, report = self.verifier.verify_worked_example_spec(spec)
        assert report.total_steps == 1
        # Elementary arithmetic mismatch without auto-correction is FAILED
        assert report.failed_steps == 1
        assert report.review_steps == 0
        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
