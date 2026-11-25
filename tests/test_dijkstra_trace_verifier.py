"""
Tests for Deterministic Dijkstra Trace Verifier (Milestone 2B.6A).

Validates:
1. Canonical pilot graph regression trace (A, B, C, D, E).
2. Negative test: Incorrect C = 4 claim (the pilot regression).
3. Negative test: Incorrect extract-min order.
4. Negative test: Missing relaxation operations.
5. Negative test: Incomplete trace / missing algorithmic steps.
6. Negative test: Incorrect settled-node state.
7. Negative test: Incorrect final distances.
8. Negative test: Negative edge weights rejected.
9. Disconnected vertices handling (inf distance preserved).
10. Equal-distance tie-breaking rule (lexicographical order).
11. Stale priority-queue entry handling.
12. Malformed graph or trace input handling.
13. Stage 1 WorkedExampleVerifier and validation gate integration.
14. Serialization and checkpoint restoration preservation.
15. Non-regression of scalar numerical verification.
"""

import json
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

from dijkstra_verifier import (
    DijkstraStepState,
    DijkstraTraceReport,
    DijkstraTraceVerifier,
    RelaxationDetail,
)
from numerical_evaluator import (
    DiagnosticCode,
    VerificationStatus,
)
from worked_example_verifier import (
    WorkedExampleReport,
    WorkedExampleVerifier,
)


# ============================================================================
# Canonical Pilot Graph Test Data
# ============================================================================

PILOT_VERTICES = ["A", "B", "C", "D", "E"]
PILOT_EDGES = [
    ("A", "B", 1.0),
    ("A", "C", 4.0),
    ("B", "C", 2.0),
    ("B", "D", 5.0),
    ("C", "D", 1.0),
    ("D", "E", 3.0),
]
PILOT_SOURCE = "A"

PILOT_CORRECT_STEPS = [
    {
        "step": "Initialize dist[A] = 0, dist[B] = ∞, dist[C] = ∞, dist[D] = ∞, dist[E] = ∞, Q = {A, B, C, D, E}",
        "result": "dist = {A: 0, B: ∞, C: ∞, D: ∞, E: ∞}, Q = {A, B, C, D, E}",
    },
    {
        "step": "Extract A from Q, update dist for neighbors B and C",
        "result": "dist = {A: 0, B: 1, C: 4, D: ∞, E: ∞}, Q = {B, C, D, E}",
    },
    {
        "step": "Extract B from Q, update dist for neighbors C and D",
        "result": "dist = {A: 0, B: 1, C: 3, D: 6, E: ∞}, Q = {C, D, E}",
    },
    {
        "step": "Extract C from Q, update dist for neighbor D",
        "result": "dist = {A: 0, B: 1, C: 3, D: 4, E: ∞}, Q = {D, E}",
    },
    {
        "step": "Extract D from Q, update dist for neighbor E",
        "result": "dist = {A: 0, B: 1, C: 3, D: 4, E: 7}, Q = {E}",
    },
    {
        "step": "Extract E from Q, no updates needed",
        "result": "dist = {A: 0, B: 1, C: 3, D: 4, E: 7}, Q = {}",
    },
]

PILOT_FINAL_RESULT = "dist = {A: 0, B: 1, C: 3, D: 4, E: 7}"


# ============================================================================
# 1. Canonical Simulation & Ground-Truth Verification
# ============================================================================

class TestDijkstraSimulation:
    def test_canonical_pilot_simulation(self):
        trace, distances, settled, predecessors, shortest_paths = DijkstraTraceVerifier.simulate(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            directed=False,
        )

        assert len(trace) == 6
        assert distances == {"A": 0.0, "B": 1.0, "C": 3.0, "D": 4.0, "E": 7.0}
        assert settled == ["A", "B", "C", "D", "E"]
        assert predecessors == {"A": None, "B": "A", "C": "B", "D": "C", "E": "D"}
        assert shortest_paths["A"] == ["A"]
        assert shortest_paths["B"] == ["A", "B"]
        assert shortest_paths["C"] == ["A", "B", "C"]
        assert shortest_paths["D"] == ["A", "B", "C", "D"]
        assert shortest_paths["E"] == ["A", "B", "C", "D", "E"]

        # Step 0: Init
        assert trace[0].step_type == "initialization"
        assert trace[0].extracted_node is None
        assert trace[0].tentative_distances["A"] == 0.0
        assert math.isinf(trace[0].tentative_distances["B"])

        # Step 1: Extract A
        assert trace[1].extracted_node == "A"
        assert trace[1].extracted_dist == 0.0
        assert trace[1].tentative_distances["B"] == 1.0
        assert trace[1].tentative_distances["C"] == 4.0

        # Step 2: Extract B
        assert trace[2].extracted_node == "B"
        assert trace[2].extracted_dist == 1.0
        assert trace[2].tentative_distances["C"] == 3.0  # 1 + 2 = 3 < 4
        assert trace[2].tentative_distances["D"] == 6.0  # 1 + 5 = 6

        # Step 3: Extract C
        assert trace[3].extracted_node == "C"
        assert trace[3].extracted_dist == 3.0
        assert trace[3].tentative_distances["D"] == 4.0  # 3 + 1 = 4 < 6

        # Step 4: Extract D
        assert trace[4].extracted_node == "D"
        assert trace[4].extracted_dist == 4.0
        assert trace[4].tentative_distances["E"] == 7.0  # 4 + 3 = 7

        # Step 5: Extract E
        assert trace[5].extracted_node == "E"
        assert trace[5].extracted_dist == 7.0

    def test_pilot_trace_verification_positive(self):
        report = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=PILOT_CORRECT_STEPS,
            directed=False,
            claimed_final=PILOT_FINAL_RESULT,
        )

        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.total_steps == 6
        assert report.verified_steps == 6
        assert report.failed_steps == 0
        assert report.not_applicable_steps == 0
        assert report.is_safe_for_stage2 is True
        assert report.final_result_status == VerificationStatus.VERIFIED
        assert len(report.discrepancies) == 0


# ============================================================================
# 2. Negative Regression Tests
# ============================================================================

class TestDijkstraNegativeCases:
    def test_negative_incorrect_c_equals_4_claim(self):
        """Tests rejection of the exact pilot error: claiming C = 4 instead of C = 3."""
        flawed_steps = [dict(s) for s in PILOT_CORRECT_STEPS]
        # At step 2, claim C remains 4
        flawed_steps[2] = {
            "step": "Extract B from Q, update dist for neighbors",
            "result": "dist = {A: 0, B: 1, C: 4, D: 6, E: ∞}, Q = {C, D, E}",
        }

        report = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=flawed_steps,
            directed=False,
            claimed_final=PILOT_FINAL_RESULT,
        )

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert report.step_verifications[2].status == VerificationStatus.FAILED
        assert any("mismatch for vertex 'C'" in d for d in report.discrepancies)

    def test_negative_incorrect_extract_min_order(self):
        """Extracting C (dist 4) before B (dist 1) must be rejected."""
        flawed_steps = [dict(s) for s in PILOT_CORRECT_STEPS]
        flawed_steps[2] = {
            "step": "Extract C from Q, update dist for neighbor D",
            "result": "dist = {A: 0, B: 1, C: 4, D: 5, E: ∞}, Q = {B, D, E}",
        }

        report = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=flawed_steps,
            directed=False,
            claimed_final=PILOT_FINAL_RESULT,
        )

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert report.step_verifications[2].diagnostic == DiagnosticCode.INVALID_EXTRACT_MIN
        assert any("Invalid extract-min node" in d for d in report.discrepancies)

    def test_negative_missing_relaxation_operation(self):
        """Extracting C but omitting the relaxation of edge C-D (leaving D = 6) must fail."""
        flawed_steps = [dict(s) for s in PILOT_CORRECT_STEPS]
        flawed_steps[3] = {
            "step": "Extract C from Q, no updates needed",
            "result": "dist = {A: 0, B: 1, C: 3, D: 6, E: ∞}, Q = {D, E}",
        }

        report = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=flawed_steps,
            directed=False,
            claimed_final=PILOT_FINAL_RESULT,
        )

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert any("mismatch for vertex 'D'" in d for d in report.discrepancies)

    def test_negative_incomplete_trace_missing_steps(self):
        """Providing only 3 steps when 6 steps are needed must be flagged as incomplete."""
        truncated_steps = PILOT_CORRECT_STEPS[:3]

        report = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=truncated_steps,
            directed=False,
            claimed_final=PILOT_FINAL_RESULT,
        )

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert any("incomplete" in d.lower() for d in report.discrepancies)

    def test_negative_incorrect_settled_node_state(self):
        """Claiming settled node set containing unvisited nodes must fail."""
        flawed_steps = [dict(s) for s in PILOT_CORRECT_STEPS]
        flawed_steps[1] = {
            "step": "Extract A from Q",
            "result": "dist = {A: 0, B: 1, C: 4, D: ∞, E: ∞}, settled = {A, E}",
        }

        report = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=flawed_steps,
            directed=False,
            claimed_final=PILOT_FINAL_RESULT,
        )

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert any("Settled set mismatch" in d for d in report.discrepancies)

    def test_negative_incorrect_final_distances(self):
        """Steps correct, but claimed final result has wrong distance for E."""
        flawed_final = "dist = {A: 0, B: 1, C: 3, D: 4, E: 9}"

        report = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=PILOT_CORRECT_STEPS,
            directed=False,
            claimed_final=flawed_final,
        )

        assert report.overall_status == VerificationStatus.FAILED
        assert report.final_result_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert any("Final distance mismatch for vertex 'E'" in d for d in report.discrepancies)

    def test_negative_edge_weights_rejected(self):
        """Negative edge weight must be rejected deterministically."""
        bad_edges = [
            ("A", "B", 1.0),
            ("A", "C", -4.0),  # Negative weight!
            ("B", "C", 2.0),
        ]

        report = DijkstraTraceVerifier.verify_trace(
            vertices=["A", "B", "C"],
            edges=bad_edges,
            source="A",
            claimed_steps=[],
            directed=False,
        )

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert any("negative" in d.lower() for d in report.discrepancies)

    def test_disconnected_vertices_handling(self):
        """Graph with disconnected vertex F: F distance must remain infinity."""
        vertices_with_f = ["A", "B", "C", "D", "E", "F"]
        trace, distances, settled, predecessors, shortest_paths = DijkstraTraceVerifier.simulate(
            vertices=vertices_with_f,
            edges=PILOT_EDGES,
            source="A",
            directed=False,
        )

        assert math.isinf(distances["F"])
        assert "F" not in settled
        assert predecessors["F"] is None
        assert shortest_paths["F"] == []

        # Verify positive trace with F remaining infinity
        steps_with_f = [
            {
                "step": "Initialize dist[A] = 0, all others ∞",
                "result": "dist = {A: 0, B: ∞, C: ∞, D: ∞, E: ∞, F: ∞}",
            },
            {
                "step": "Extract A",
                "result": "dist = {A: 0, B: 1, C: 4, D: ∞, E: ∞, F: ∞}",
            },
            {
                "step": "Extract B",
                "result": "dist = {A: 0, B: 1, C: 3, D: 6, E: ∞, F: ∞}",
            },
            {
                "step": "Extract C",
                "result": "dist = {A: 0, B: 1, C: 3, D: 4, E: ∞, F: ∞}",
            },
            {
                "step": "Extract D",
                "result": "dist = {A: 0, B: 1, C: 3, D: 4, E: 7, F: ∞}",
            },
            {
                "step": "Extract E",
                "result": "dist = {A: 0, B: 1, C: 3, D: 4, E: 7, F: ∞}",
            },
        ]
        report = DijkstraTraceVerifier.verify_trace(
            vertices=vertices_with_f,
            edges=PILOT_EDGES,
            source="A",
            claimed_steps=steps_with_f,
            claimed_final="dist = {A: 0, B: 1, C: 3, D: 4, E: 7, F: ∞}",
        )
        assert report.overall_status == VerificationStatus.VERIFIED

        # If trace claims F = 10, it must be rejected
        bad_f_final = "dist = {A: 0, B: 1, C: 3, D: 4, E: 7, F: 10}"
        bad_report = DijkstraTraceVerifier.verify_trace(
            vertices=vertices_with_f,
            edges=PILOT_EDGES,
            source="A",
            claimed_steps=steps_with_f,
            claimed_final=bad_f_final,
        )
        assert bad_report.overall_status == VerificationStatus.FAILED
        assert bad_report.final_result_status == VerificationStatus.FAILED

    def test_equal_distance_tie_breaking_rule(self):
        """Tie-breaking rule: Lexicographical order by vertex label."""
        vertices = ["A", "B", "C"]
        edges = [("A", "B", 2.0), ("A", "C", 2.0)]
        source = "A"

        trace, distances, settled, predecessors, shortest_paths = DijkstraTraceVerifier.simulate(
            vertices=vertices,
            edges=edges,
            source=source,
            directed=False,
        )

        # B and C have equal tentative distance (2.0). Lexicographical: B < C, so B extracted first.
        assert settled == ["A", "B", "C"]
        assert trace[1].extracted_node == "A"
        assert trace[2].extracted_node == "B"
        assert trace[3].extracted_node == "C"

        # Trace claiming C extracted before B must be flagged
        inverted_steps = [
            {"step": "Initialize dist[A]=0", "result": "dist = {A: 0, B: ∞, C: ∞}"},
            {"step": "Extract A", "result": "dist = {A: 0, B: 2, C: 2}"},
            {"step": "Extract C", "result": "dist = {A: 0, B: 2, C: 2}"},
            {"step": "Extract B", "result": "dist = {A: 0, B: 2, C: 2}"},
        ]
        rep = DijkstraTraceVerifier.verify_trace(
            vertices=vertices,
            edges=edges,
            source=source,
            claimed_steps=inverted_steps,
        )
        assert rep.overall_status == VerificationStatus.FAILED
        assert rep.step_verifications[2].diagnostic == DiagnosticCode.INVALID_EXTRACT_MIN

    def test_stale_priority_queue_entries_handling(self):
        """Stale priority-queue entry: popping already-settled node is recognized as safe discard."""
        claimed_steps_with_stale = [
            {"step": "Initialize dist[A] = 0", "result": "dist = {A: 0, B: ∞, C: ∞}"},
            {"step": "Extract A", "result": "dist = {A: 0, B: 1, C: 4}"},
            {"step": "Extract B", "result": "dist = {A: 0, B: 1, C: 3}"},
            {"step": "Extract C", "result": "dist = {A: 0, B: 1, C: 3}"},
            # Explicit stale entry skip step:
            {
                "step": "Pop stale entry for node C (dist 4) from heap. Node C already settled, discard.",
                "result": "dist = {A: 0, B: 1, C: 3}, Q = {}",
                "is_stale_discard": True,
                "extracted_node": "C",
            },
        ]
        rep = DijkstraTraceVerifier.verify_trace(
            vertices=["A", "B", "C"],
            edges=[("A", "B", 1.0), ("A", "C", 4.0), ("B", "C", 2.0)],
            source="A",
            claimed_steps=claimed_steps_with_stale,
        )
        assert rep.overall_status == VerificationStatus.VERIFIED
        assert rep.step_verifications[4].status == VerificationStatus.VERIFIED

    def test_malformed_graph_inputs(self):
        """Malformed vertices, unknown nodes in edges, or invalid source must fail gracefully."""
        # 1. Unknown vertex in edge
        bad_edge_graph = {
            "setup": "Graph with nodes (A, B) and edge weights: A-Z (5). Starting node is A.",
            "inputs": {"s": "A"},
            "steps": [{"step": "Init", "result": "dist={A: 0}"}],
        }
        verifier = WorkedExampleVerifier()
        updated, rep = verifier.verify_worked_example_spec(bad_edge_graph, topic="Dijkstra")
        assert rep.overall_status == VerificationStatus.FAILED
        assert rep.is_safe_for_stage2 is False

        # 2. Invalid source not in vertices
        bad_source_graph = {
            "setup": "Graph with nodes (A, B) and edge weights: A-B (5). Starting node is Z.",
            "inputs": {"s": "Z"},
            "steps": [{"step": "Init", "result": "dist={Z: 0}"}],
        }
        updated2, rep2 = verifier.verify_worked_example_spec(bad_source_graph, topic="Dijkstra")
        assert rep2.overall_status == VerificationStatus.FAILED
        assert rep2.is_safe_for_stage2 is False


# ============================================================================
# 3. Stage 1 WorkedExampleVerifier & Gate Integration Tests
# ============================================================================

class TestDijkstraStage1GateIntegration:
    def setup_method(self):
        self.verifier = WorkedExampleVerifier()

    def test_pilot_outline_full_verification_gate(self):
        """
        Loads the actual pilot outline structure and verifies that the new verifier:
        1. Classifies all 6 steps as VERIFIED (not NOT_APPLICABLE).
        2. Exports verified algorithmic states in verification_metadata.
        3. Grants safe gate clearance (is_safe_for_stage2=True).
        """
        outline = {
            "topic": "Dijkstra's Shortest Path Algorithm",
            "worked_example_spec": {
                "setup": (
                    "A graph with 5 nodes (A, B, C, D, E) and the following edge weights: "
                    "A-B (1), A-C (4), B-C (2), B-D (5), C-D (1), D-E (3). Starting node is A."
                ),
                "inputs": {
                    "G": "A graph with nodes A, B, C, D, E",
                    "s": "A",
                },
                "steps": [dict(s) for s in PILOT_CORRECT_STEPS],
                "final_result": PILOT_FINAL_RESULT,
            },
        }

        updated_outline, report = self.verifier.verify_outline(outline)

        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.total_steps == 6
        assert report.verified_steps == 6
        assert report.failed_steps == 0
        assert report.not_applicable_steps == 0  # Blind spot eliminated!
        assert report.is_safe_for_stage2 is True

        # Verify structured_calc on steps
        steps = updated_outline["worked_example_spec"]["steps"]
        assert len(steps) == 6
        for i, s in enumerate(steps):
            sc = s["structured_calc"]
            assert sc["status"] == VerificationStatus.VERIFIED.value
            assert sc["is_evaluable"] is True
            assert "tentative_distances" in sc

        # Verify algorithmic metadata
        meta = updated_outline["verification_metadata"]
        assert meta["overall_status"] == "VERIFIED"
        assert "verified_algorithmic_state" in meta
        algo_state = meta["verified_algorithmic_state"]
        assert algo_state["algorithm"] == "dijkstra"
        assert algo_state["source"] == "A"
        assert algo_state["final_distances"] == {"A": 0.0, "B": 1.0, "C": 3.0, "D": 4.0, "E": 7.0}
        assert algo_state["settled_order"] == ["A", "B", "C", "D", "E"]

    def test_pilot_outline_flawed_blocked_at_gate(self):
        """Flawed Dijkstra outline (e.g. C=4) MUST be blocked at the gate (is_safe_for_stage2=False)."""
        flawed_steps = [dict(s) for s in PILOT_CORRECT_STEPS]
        flawed_steps[2]["result"] = "dist = {A: 0, B: 1, C: 4, D: 6, E: ∞}"

        outline = {
            "topic": "Dijkstra's Shortest Path Algorithm",
            "worked_example_spec": {
                "setup": (
                    "A graph with 5 nodes (A, B, C, D, E) and the following edge weights: "
                    "A-B (1), A-C (4), B-C (2), B-D (5), C-D (1), D-E (3). Starting node is A."
                ),
                "inputs": {"s": "A"},
                "steps": flawed_steps,
                "final_result": "dist = {A: 0, B: 1, C: 4, D: 4, E: 7}",
            },
        }

        updated_outline, report = self.verifier.verify_outline(outline)

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert updated_outline["verification_metadata"]["is_safe_for_stage2"] is False

    def test_serialization_and_restoration_lossless(self):
        """Verified outline dictionary must serialize to JSON and restore losslessly."""
        outline = {
            "topic": "Dijkstra's Shortest Path Algorithm",
            "worked_example_spec": {
                "setup": "A graph with 5 nodes (A, B, C, D, E): A-B (1), A-C (4), B-C (2), B-D (5), C-D (1), D-E (3). Source: A",
                "inputs": {"s": "A"},
                "steps": [dict(s) for s in PILOT_CORRECT_STEPS],
                "final_result": PILOT_FINAL_RESULT,
            },
        }
        updated_outline, report = self.verifier.verify_outline(outline)

        serialized = json.dumps(updated_outline)
        restored = json.loads(serialized)

        assert restored["verification_metadata"]["overall_status"] == "VERIFIED"
        assert restored["verification_metadata"]["is_safe_for_stage2"] is True
        assert restored["verification_metadata"]["verified_algorithmic_state"]["final_distances"]["E"] == 7.0

    def test_non_regression_scalar_numerical_verification(self):
        """Scalar Arrhenius numerical verification remains 100% unaffected."""
        arrhenius_spec = {
            "inputs": {"Ea": 75000, "R": 8.314, "T": 300, "A": 7.5e10},
            "steps": [
                {
                    "step": "Calculate exponent: -Ea / (R * T)",
                    "result": "-75000 / (8.314 * 300) = -29.74",
                },
                {
                    "step": "Calculate rate constant k",
                    "result": "7.5e10 * exp(-29.74) = 0.0001087",
                },
            ],
            "final_result": "k = 0.0001087 s^-1",
        }
        outline = {
            "topic": "Arrhenius Reaction Kinetics",
            "worked_example_spec": arrhenius_spec,
        }

        updated, report = self.verifier.verify_outline(outline)
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.corrected_steps == 2
        assert report.is_safe_for_stage2 is True
        assert "verified_algorithmic_state" not in updated["verification_metadata"]


# ============================================================================
# 4. Hardening Tests (Step 2B.6A.2)
# ============================================================================

class TestDijkstraHardening:
    def setup_method(self):
        self.verifier = WorkedExampleVerifier()

    def test_negative_empty_and_vague_iteration_claims(self):
        """Empty or vague iteration claims must fail closed and never pass as VERIFIED."""
        # 1. Prose claim without extracted node or distance state
        vague_steps = [dict(s) for s in PILOT_CORRECT_STEPS]
        vague_steps[1] = {
            "step": "Examine adjacent nodes in the neighborhood",
            "result": "Continue algorithm traversal",
        }

        report = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=vague_steps,
            directed=False,
            claimed_final=PILOT_FINAL_RESULT,
        )

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert report.step_verifications[1].status == VerificationStatus.FAILED
        assert report.step_verifications[1].diagnostic == DiagnosticCode.INVALID_EXTRACT_MIN
        assert any("Unparseable or missing extract-min node" in d for d in report.discrepancies)

        # 2. Blank step result
        blank_steps = [dict(s) for s in PILOT_CORRECT_STEPS]
        blank_steps[2] = {"step": "Step 2", "result": ""}

        report2 = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=blank_steps,
            directed=False,
            claimed_final=PILOT_FINAL_RESULT,
        )

        assert report2.overall_status == VerificationStatus.FAILED
        assert report2.is_safe_for_stage2 is False
        assert report2.step_verifications[2].status == VerificationStatus.FAILED

    def test_negative_missing_vertices_in_distance_snapshot(self):
        """Incomplete distance snapshots that omit vertices must fail closed."""
        flawed_steps = [dict(s) for s in PILOT_CORRECT_STEPS]
        # At step 1, omit vertex C from the distances table
        flawed_steps[1] = {
            "step": "Extract A from Q, update dist for neighbor B",
            "result": "dist = {A: 0, B: 1, D: ∞, E: ∞}, Q = {B, C, D, E}",
        }

        report = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=flawed_steps,
            directed=False,
            claimed_final=PILOT_FINAL_RESULT,
        )

        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert report.step_verifications[1].status == VerificationStatus.FAILED
        assert any("missing vertices ['C']" in d for d in report.discrepancies)

    def test_negative_incorrect_infinity_values(self):
        """Incorrect infinity values (e.g. finite for unreached or inf for reached) must be caught."""
        # 1. Claiming finite distance for unreached vertex D at step 1
        flawed_steps_1 = [dict(s) for s in PILOT_CORRECT_STEPS]
        flawed_steps_1[1] = {
            "step": "Extract A from Q",
            "result": "dist = {A: 0, B: 1, C: 4, D: 5, E: ∞}, Q = {B, C, D, E}",
        }

        rep1 = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=flawed_steps_1,
        )
        assert rep1.overall_status == VerificationStatus.FAILED
        assert any("mismatch for vertex 'D': claimed 5, expected ∞" in d for d in rep1.discrepancies)

        # 2. Claiming infinity for relaxed vertex B at step 1
        flawed_steps_2 = [dict(s) for s in PILOT_CORRECT_STEPS]
        flawed_steps_2[1] = {
            "step": "Extract A from Q",
            "result": "dist = {A: 0, B: ∞, C: 4, D: ∞, E: ∞}, Q = {B, C, D, E}",
        }

        rep2 = DijkstraTraceVerifier.verify_trace(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            claimed_steps=flawed_steps_2,
        )
        assert rep2.overall_status == VerificationStatus.FAILED
        assert any("mismatch for vertex 'B': claimed ∞, expected 1" in d for d in rep2.discrepancies)

    def test_predecessors_and_shortest_paths_reconstruction(self):
        """Predecessors and shortest paths must be deterministically generated from simulation."""
        trace, distances, settled, predecessors, shortest_paths = DijkstraTraceVerifier.simulate(
            vertices=PILOT_VERTICES,
            edges=PILOT_EDGES,
            source=PILOT_SOURCE,
            directed=False,
        )

        assert predecessors == {
            "A": None,
            "B": "A",
            "C": "B",  # Improved from A (4) to B (1+2=3)
            "D": "C",  # Improved from B (1+5=6) to C (3+1=4)
            "E": "D",  # 4 + 3 = 7
        }
        assert shortest_paths["A"] == ["A"]
        assert shortest_paths["B"] == ["A", "B"]
        assert shortest_paths["C"] == ["A", "B", "C"]
        assert shortest_paths["D"] == ["A", "B", "C", "D"]
        assert shortest_paths["E"] == ["A", "B", "C", "D", "E"]

        # Also verify predecessors are recorded in canonical trace steps
        assert trace[0].predecessors == {"A": None, "B": None, "C": None, "D": None, "E": None}
        assert trace[1].predecessors["B"] == "A"
        assert trace[1].predecessors["C"] == "A"
        assert trace[2].predecessors["C"] == "B"  # Updated at Step 2!

    def test_directed_vs_undirected_graph_metadata(self):
        """Tests that directed and undirected graphs export complete, distinct metadata."""
        # Directed graph: A -> B (3), B -> C (2), A -> C (6)
        dir_vertices = ["A", "B", "C"]
        dir_edges = [("A", "B", 3.0), ("B", "C", 2.0), ("A", "C", 6.0)]
        dir_source = "A"

        trace, distances, settled, predecessors, shortest_paths = DijkstraTraceVerifier.simulate(
            vertices=dir_vertices,
            edges=dir_edges,
            source=dir_source,
            directed=True,
        )

        assert distances == {"A": 0.0, "B": 3.0, "C": 5.0}
        assert predecessors == {"A": None, "B": "A", "C": "B"}
        assert shortest_paths["C"] == ["A", "B", "C"]

        # In directed graph, C cannot reach A
        trace_c, dist_c, _, _, paths_c = DijkstraTraceVerifier.simulate(
            vertices=dir_vertices,
            edges=dir_edges,
            source="C",
            directed=True,
        )
        assert math.isinf(dist_c["A"])
        assert paths_c["A"] == []

    def test_complete_algorithmic_metadata_export_and_serialization(self):
        """Tests that verified_algorithmic_state exports all fields and round-trips cleanly."""
        outline = {
            "topic": "Dijkstra's Shortest Path Algorithm",
            "worked_example_spec": {
                "setup": (
                    "A graph with 5 nodes (A, B, C, D, E) and the following edge weights: "
                    "A-B (1), A-C (4), B-C (2), B-D (5), C-D (1), D-E (3). Starting node is A."
                ),
                "inputs": {"s": "A"},
                "steps": [dict(s) for s in PILOT_CORRECT_STEPS],
                "final_result": PILOT_FINAL_RESULT,
            },
        }

        updated, report = self.verifier.verify_outline(outline)
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_stage2 is True

        meta = updated["verification_metadata"]
        assert "verified_algorithmic_state" in meta
        algo = meta["verified_algorithmic_state"]

        # Check all required fields
        assert algo["algorithm"] == "dijkstra"
        assert algo["source"] == "A"
        assert algo["vertices"] == ["A", "B", "C", "D", "E"]
        assert algo["directed"] is False
        assert len(algo["edges"]) == 6
        assert {"u": "A", "v": "B", "weight": 1.0} in algo["edges"]
        assert algo["predecessors"] == {"A": None, "B": "A", "C": "B", "D": "C", "E": "D"}
        assert algo["shortest_paths"]["E"] == ["A", "B", "C", "D", "E"]
        assert algo["final_distances"] == {"A": 0.0, "B": 1.0, "C": 3.0, "D": 4.0, "E": 7.0}
        assert algo["settled_order"] == ["A", "B", "C", "D", "E"]
        assert len(algo["canonical_trace"]) == 6

        # Check structured_calc on steps
        for step in updated["worked_example_spec"]["steps"]:
            sc = step["structured_calc"]
            assert "predecessors" in sc
            assert "relaxations" in sc

        # Test lossless JSON serialization and restoration
        dumped = json.dumps(updated)
        restored = json.loads(dumped)
        algo_restored = restored["verification_metadata"]["verified_algorithmic_state"]
        assert algo_restored["predecessors"]["E"] == "D"
        assert algo_restored["shortest_paths"]["E"] == ["A", "B", "C", "D", "E"]
        assert algo_restored["edges"][0]["u"] == "A"
        assert algo_restored["directed"] is False
