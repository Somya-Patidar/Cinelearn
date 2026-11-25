"""
Comprehensive Test Suite for General-Purpose Worked Example Verification (Milestone 2B.6C).

Validates:
1. Binary Search regression test using actual outline data (proves fix for issue).
2. Binary Search negative tests: arithmetic contradiction and sequence element mismatch.
3. Structured modern step schema with state_before, state_after, operations.
4. State continuity mismatch detection (fails closed when state_before contradicts env).
5. Unfamiliar / unverified topic handling: UNVERIFIED coverage proceeds safely to Stage 2.
6. Untrusted physics formula in registered domain continues to block (NEEDS_REVIEW).
7. Specialized verifier registry dispatch (Dijkstra) and non-interference.
8. Lossless JSON serialization of verification metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from numerical_evaluator import (
    DiagnosticCode,
    VerificationStatus,
)
from worked_example_verifier import (
    CalculationEnvironment,
    CalculationStepAdapter,
    SpecializedVerifierRegistry,
    StepType,
    VariableTrust,
    WorkedExampleReport,
    WorkedExampleVerifier,
)
from dijkstra_verifier import DijkstraTraceVerifier


# ============================================================================
# Test Data: Canonical Binary Search Outline
# ============================================================================

BINARY_SEARCH_SPEC = {
    "setup": "Perform binary search for target = 7 in a 0-indexed sorted array arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19].",
    "inputs": {
        "arr": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        "target": 7,
    },
    "steps": [
        {
            "step": "Initialize low = 0, high = 9.",
            "result": "low = 0, high = 9",
        },
        {
            "step": "Calculate mid = (low + high) // 2 = (0 + 9) // 2 = 4.",
            "result": "mid = 4",
        },
        {
            "step": "Compare arr[mid] = arr[4] = 9 with target = 7. Since 9 > 7, update high = mid - 1 = 4 - 1 = 3.",
            "result": "high = 3",
        },
        {
            "step": "Calculate mid = (low + high) // 2 = (0 + 3) // 2 = 1.",
            "result": "mid = 1",
        },
        {
            "step": "Compare arr[mid] = arr[1] = 3 with target = 7. Since 3 < 7, update low = mid + 1 = 1 + 1 = 2.",
            "result": "low = 2",
        },
        {
            "step": "Calculate mid = (low + high) // 2 = (2 + 3) // 2 = 2.",
            "result": "mid = 2",
        },
        {
            "step": "Compare arr[mid] = arr[2] = 5 with target = 7. Since 5 < 7, update low = mid + 1 = 2 + 1 = 3.",
            "result": "low = 3",
        },
        {
            "step": "Calculate mid = (low + high) // 2 = (3 + 3) // 2 = 3.",
            "result": "mid = 3",
        },
        {
            "step": "Compare arr[mid] = arr[3] = 7 with target = 7. Since arr[mid] == target, return mid = 3.",
            "result": "mid = 3",
        },
    ],
    "final_result": "mid = 3",
}


# ============================================================================
# 1. Binary Search Regression & Gate Clearance Tests
# ============================================================================

class TestBinarySearchVerification:
    """Verifies that the generic verification orchestrator correctly handles Binary Search."""

    def setup_method(self):
        self.verifier = WorkedExampleVerifier()

    def test_binary_search_passes_and_is_safe_for_stage2(self):
        """The canonical Binary Search worked example must pass verification and clear the gate."""
        outline = {
            "topic": "Binary Search",
            "worked_example_spec": BINARY_SEARCH_SPEC,
            "governing_principles": [
                "The array must be sorted for binary search to work correctly.",
                "The time complexity of binary search is O(log n).",
            ],
        }

        updated, report = self.verifier.verify_outline(outline)

        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_stage2 is True
        assert report.failed_steps == 0
        assert report.review_steps == 0
        assert report.verified_steps == 9
        assert report.final_result_status == VerificationStatus.VERIFIED

        # Check that environment preserved the array input
        env_inputs = updated["worked_example_spec"]["inputs"]
        assert isinstance(env_inputs["arr"], list)
        assert len(env_inputs["arr"]) == 10

    def test_actual_saved_failure_case_outline_proceeds(self):
        """Tests the actual outline file from the failed run on disk."""
        outline_file = Path(_PROJECT_ROOT) / "CASES" / "TEST-single_Ollama" / "0-Binary_Search" / "outline.json"
        if not outline_file.exists():
            pytest.skip("Test outline file not present on disk")

        with open(outline_file, "r", encoding="utf-8") as f:
            outline_data = json.load(f)

        updated, report = self.verifier.verify_outline(outline_data)

        assert report.is_safe_for_stage2 is True
        assert report.failed_steps == 0
        assert report.review_steps == 0
        assert report.overall_status in (VerificationStatus.VERIFIED, VerificationStatus.UNVERIFIED)

    def test_binary_search_arithmetic_contradiction_is_blocked(self):
        """If mid calculation contains an arithmetic error, the step must fail and block Stage 2."""
        flawed_spec = json.loads(json.dumps(BINARY_SEARCH_SPEC))
        # Flawed calculation: (0 + 9) // 2 is claimed to be 7
        flawed_spec["steps"][1] = {
            "step": "Calculate mid = (low + high) // 2 = (0 + 9) // 2 = 7.",
            "result": "mid = 7",
        }

        outline = {
            "topic": "Binary Search",
            "worked_example_spec": flawed_spec,
        }

        updated, report = self.verifier.verify_outline(outline)

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert report.failed_steps >= 1

        step1 = updated["worked_example_spec"]["steps"][1]
        assert step1["structured_calc"]["status"] == VerificationStatus.FAILED.value
        assert step1["structured_calc"]["diagnostic"] == DiagnosticCode.ARITHMETIC_MISMATCH.value

    def test_binary_search_sequence_contradiction_is_blocked(self):
        """If array access claims a value that contradicts arr[idx], it must fail and block."""
        flawed_spec = json.loads(json.dumps(BINARY_SEARCH_SPEC))
        # Contradiction: claimed arr[4] = 99 when arr[4] is 9
        flawed_spec["steps"][2] = {
            "step": "Compare arr[mid] = arr[4] = 99 with target = 7.",
            "result": "high = 3",
        }

        outline = {
            "topic": "Binary Search",
            "worked_example_spec": flawed_spec,
        }

        updated, report = self.verifier.verify_outline(outline)

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert report.failed_steps >= 1

        step2 = updated["worked_example_spec"]["steps"][2]
        assert step2["structured_calc"]["status"] == VerificationStatus.FAILED.value
        assert step2["structured_calc"]["diagnostic"] == DiagnosticCode.SEQUENCE_MISMATCH.value


# ============================================================================
# 2. Structured Modern Schema & State Continuity Tests
# ============================================================================

class TestStructuredStepVerification:
    """Verifies structured steps with state_before, state_after, operations."""

    def setup_method(self):
        self.verifier = WorkedExampleVerifier()

    def test_structured_state_transition_positive(self):
        """Structured state transitions with verified state continuity pass."""
        spec = {
            "inputs": {"counter": 0, "threshold": 10},
            "steps": [
                {
                    "step": "Increment counter",
                    "step_type": "state_transition",
                    "operation": "assign",
                    "state_before": {"counter": 0},
                    "state_after": {"counter": 1},
                    "result": "counter = 1",
                },
                {
                    "step": "Increment counter again",
                    "step_type": "state_transition",
                    "operation": "assign",
                    "state_before": {"counter": 1},
                    "state_after": {"counter": 2},
                    "result": "counter = 2",
                },
            ],
            "final_result": "counter = 2",
        }

        outline = {"topic": "State Counter", "worked_example_spec": spec}
        updated, report = self.verifier.verify_outline(outline)

        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_stage2 is True
        assert report.verified_steps == 2
        assert report.failed_steps == 0

    def test_structured_state_continuity_mismatch_blocks_gate(self):
        """When state_before contradicts the environment state, step fails and blocks Stage 2."""
        spec = {
            "inputs": {"counter": 0},
            "steps": [
                {
                    "step": "Increment counter",
                    "step_type": "state_transition",
                    "operation": "assign",
                    "state_before": {"counter": 0},
                    "state_after": {"counter": 1},
                    "result": "counter = 1",
                },
                {
                    "step": "Invalid transition assuming counter was 99",
                    "step_type": "state_transition",
                    "operation": "assign",
                    "state_before": {"counter": 99},  # Contradicts env (counter is 1)
                    "state_after": {"counter": 100},
                    "result": "counter = 100",
                },
            ],
        }

        outline = {"topic": "State Counter", "worked_example_spec": spec}
        updated, report = self.verifier.verify_outline(outline)

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert report.failed_steps >= 1

        step1 = updated["worked_example_spec"]["steps"][1]
        assert step1["structured_calc"]["status"] == VerificationStatus.FAILED.value
        assert step1["structured_calc"]["diagnostic"] == DiagnosticCode.STATE_CONTINUITY_MISMATCH.value


# ============================================================================
# 3. Unverified / Unfamiliar Topic Gate Semantics
# ============================================================================

class TestUnfamiliarTopicGateSemantics:
    """Verifies that lack of verification coverage produces UNVERIFIED and does NOT block."""

    def setup_method(self):
        self.verifier = WorkedExampleVerifier()

    def test_unregistered_algorithm_proceeds_as_unverified(self):
        """An arbitrary algorithm without a registered specialized verifier receives UNVERIFIED and clears gate."""
        spec = {
            "inputs": {"nodes": ["P1", "P2", "P3"], "leader": "P1"},
            "steps": [
                {
                    "step": "Node P1 crashes; election starts",
                    "step_type": "conceptual_explanation",
                    "result": "P1 offline",
                },
                {
                    "step": "P2 requests votes from cluster",
                    "step_type": "state_transition",
                    "result": "votes = 2",
                },
                {
                    "step": "P2 becomes new leader",
                    "step_type": "state_transition",
                    "result": "leader = P2",
                },
            ],
            "final_result": "leader = P2",
        }

        outline = {
            "topic": "Raft Leader Election",
            "worked_example_spec": spec,
        }

        updated, report = self.verifier.verify_outline(outline)

        # Must NOT be blocked! Lack of verification coverage != incorrectness
        assert report.is_safe_for_stage2 is True
        assert report.failed_steps == 0
        assert report.overall_status in (VerificationStatus.UNVERIFIED, VerificationStatus.VERIFIED)

    def test_registered_domain_inverted_formula_remains_blocked(self):
        """In registered physics/chemistry domains, untrusted formulas remain blocked (NEEDS_REVIEW)."""
        spec = {
            "inputs": {"Ea": 75000, "R": 8.314, "T": 300, "A": 7.5e10},
            "steps": [
                {
                    "step": "Calculate exp with inverted formula",
                    "result": "exp_T1 = (8.314 * 300) / 75000 = 0.033",
                },
            ],
        }

        outline = {
            "topic": "Arrhenius Kinetics",
            "worked_example_spec": spec,
        }

        updated, report = self.verifier.verify_outline(outline)

        assert report.overall_status == VerificationStatus.NEEDS_REVIEW
        assert report.is_safe_for_stage2 is False
        assert report.review_steps >= 1


# ============================================================================
# 4. Specialized Verifier Registry & Routing
# ============================================================================

class TestSpecializedVerifierRegistry:
    """Verifies registry lookup, dispatch, and clean decoupling."""

    def test_dijkstra_registered_in_registry(self):
        """DijkstraTraceVerifier is registered and correctly discovered."""
        verifier = SpecializedVerifierRegistry.find_verifier(
            spec={"setup": "A graph with 5 nodes: A-B (1). Source: A"},
            topic="Dijkstra's Shortest Path",
        )
        assert verifier is not None
        assert verifier == DijkstraTraceVerifier

    def test_non_dijkstra_not_intercepted(self):
        """Binary search and other topics are not intercepted by Dijkstra verifier."""
        verifier = SpecializedVerifierRegistry.find_verifier(
            spec=BINARY_SEARCH_SPEC,
            topic="Binary Search",
        )
        assert verifier is None

    def test_registry_registration_and_clearing(self):
        """Custom verifiers can be registered and cleared."""
        class MockCustomVerifier:
            @classmethod
            def can_verify(cls, spec, topic=""):
                return "mock_topic" in topic.lower()

            @classmethod
            def verify_worked_example_spec(cls, spec, topic=""):
                rep = WorkedExampleReport(
                    overall_status=VerificationStatus.VERIFIED,
                    total_steps=1,
                    verified_steps=1,
                    corrected_steps=0,
                    failed_steps=0,
                    review_steps=0,
                    not_applicable_steps=0,
                    is_safe_for_stage2=True,
                    verification_method="mock_custom",
                )
                return spec, rep

        SpecializedVerifierRegistry.register(MockCustomVerifier)
        found = SpecializedVerifierRegistry.find_verifier({}, topic="Mock_Topic_Test")
        assert found == MockCustomVerifier

        orchestrator = WorkedExampleVerifier()
        _, rep = orchestrator.verify_worked_example_spec({"steps": [{"step": "a", "result": "b"}]}, topic="Mock_Topic_Test")
        assert rep.verification_method == "mock_custom"

        # Clean up
        SpecializedVerifierRegistry._verifiers.remove(MockCustomVerifier)


# ============================================================================
# 5. Serialization & Round-Trip Preservation
# ============================================================================

class TestVerificationMetadataSerialization:
    """Verifies that verification metadata serializes and restores cleanly."""

    def test_json_roundtrip_preserves_all_metadata(self):
        orchestrator = WorkedExampleVerifier()
        outline = {
            "topic": "Binary Search",
            "worked_example_spec": BINARY_SEARCH_SPEC,
        }

        updated, report = orchestrator.verify_outline(outline)
        dumped = json.dumps(updated)
        restored = json.loads(dumped)

        meta = restored["verification_metadata"]
        assert meta["overall_status"] == report.overall_status.value
        assert meta["is_safe_for_stage2"] is True
        assert meta["total_steps"] == 9
        assert meta["verified_steps"] == 9
        assert meta["verification_method"] == "generic_orchestrator"
        assert len(meta["step_diagnostics"]) == 9
