"""
Unit and integration tests for Milestone 2.1:
Generic Temporal Trace Contract and Snapshot-Aware Fidelity Checking.
"""

import pytest
import copy
from typing import Dict, Any

from grounding.trace_engine import StateSnapshot, ComputationTrace
from worked_example_verifier import CalculationEnvironment, WorkedExampleVerifier
from storyboard_fidelity_checker import (
    StoryboardFidelityChecker,
    AuthoritativeOutlineModel,
    VerificationStatus,
    FidelityCategory,
)


class TestGenericTemporalTraceContract:
    """Tests for StateSnapshot and ComputationTrace domain-independent models."""

    def test_state_snapshot_deep_copy_isolation(self):
        """StateSnapshot must isolate mutations to the input state dict."""
        state = {"low": 0, "high": 9, "mid": 4, "arr": [1, 2, 3]}
        snapshot = StateSnapshot(
            step_index=0,
            state=state,
            description="Initialized bounds",
            step_type="INITIALIZATION",
        )
        # Mutate original dictionary and array
        state["low"] = 99
        state["arr"].append(4)

        assert snapshot.state["low"] == 0
        assert snapshot.state["arr"] == [1, 2, 3]
        assert snapshot["low"] == 0
        assert "high" in snapshot
        assert len(snapshot) == 4

    def test_state_snapshot_serialization_roundtrip(self):
        """StateSnapshot to_dict and from_dict roundtrip."""
        snap = StateSnapshot(
            step_index=2,
            state={"dist": {"A": 0, "B": 4}, "settled": ["A"]},
            description="Relaxed edge (A, B)",
            step_type="STATE_TRANSITION",
            metadata={"edge": ("A", "B"), "weight": 4},
        )
        data = snap.to_dict()
        restored = StateSnapshot.from_dict(data)

        assert restored.step_index == 2
        assert restored.state == {"dist": {"A": 0, "B": 4}, "settled": ["A"]}
        assert restored.step_type == "STATE_TRANSITION"
        assert restored.metadata["weight"] == 4

    def test_computation_trace_chronological_ordering(self):
        """ComputationTrace maintains strictly ordered chronological snapshots."""
        trace = ComputationTrace(
            algorithm_or_domain="binary_search",
            initial_state={"low": 0, "high": 9, "arr": [2, 4, 6, 8, 10]},
        )
        snap0 = StateSnapshot(step_index=0, state={"low": 0, "high": 9, "mid": 4})
        snap1 = StateSnapshot(step_index=1, state={"low": 5, "high": 9, "mid": 4})
        snap2 = StateSnapshot(step_index=2, state={"low": 5, "high": 9, "mid": 7})

        trace.add_snapshot(snap0)
        trace.add_snapshot(snap1)
        trace.add_snapshot(snap2)

        assert len(trace.snapshots) == 3
        assert trace.has_step(0)
        assert trace.has_step(1)
        assert trace.has_step(2)
        assert not trace.has_step(3)

        assert trace.get_state_at_step(-1)["high"] == 9
        assert trace.get_state_at_step(0)["mid"] == 4
        assert trace.get_state_at_step(1)["low"] == 5
        assert trace.get_state_at_step(2)["mid"] == 7

    def test_computation_trace_out_of_order_addition_rejected(self):
        """Snapshots cannot be added with non-chronological or duplicate step indices."""
        trace = ComputationTrace(algorithm_or_domain="test")
        trace.add_snapshot(StateSnapshot(step_index=1, state={"x": 10}))
        with pytest.raises(ValueError, match="chronological ordering"):
            trace.add_snapshot(StateSnapshot(step_index=0, state={"x": 5}))

    def test_computation_trace_out_of_range_handling(self):
        """Out of range step queries raise IndexError; non-integer raises TypeError."""
        trace = ComputationTrace(
            algorithm_or_domain="test",
            initial_state={"init_val": 1},
            snapshots=[StateSnapshot(step_index=0, state={"x": 10})],
        )
        assert trace.get_state_at_step(-1) == {"init_val": 1}
        assert trace.get_state_at_step(0) == {"x": 10}

        with pytest.raises(IndexError):
            trace.get_state_at_step(1)

        with pytest.raises(IndexError):
            trace.get_state_at_step(-2)

        with pytest.raises(TypeError):
            trace.get_state_at_step("0")  # type: ignore


class TestCalculationEnvironmentTemporalSnapshots:
    """Tests for CalculationEnvironment state snapshot recording and retrieval."""

    def test_initial_snapshot_behavior(self):
        """Initial state is available via get_state_at_step(-1) and get_initial_state()."""
        env = CalculationEnvironment(
            inputs={"target": 7, "low": 0, "high": 9},
            units={"low": "index", "high": "index"},
        )
        init_state = env.get_state_at_step(-1)
        assert init_state == {"target": 7, "low": 0, "high": 9}
        assert env.get_initial_state() == {"target": 7, "low": 0, "high": 9}

    def test_state_updates_across_multiple_steps(self):
        """Variables evolve across steps, and each step preserves its complete state."""
        env = CalculationEnvironment(inputs={"low": 0, "high": 9})
        # Step 0: mid = (low + high) // 2 = 4
        env.record_step_snapshot(
            step_index=0,
            updates={"mid": 4},
            description="Compute mid for step 0",
            step_type="CALCULATION",
        )
        # Step 1: target > arr[4], so low = mid + 1 = 5
        env.record_step_snapshot(
            step_index=1,
            updates={"low": 5},
            description="Update lower bound",
            step_type="STATE_TRANSITION",
        )
        # Step 2: mid = (5 + 9) // 2 = 7
        env.record_step_snapshot(
            step_index=2,
            updates={"mid": 7},
            description="Recompute mid for step 2",
            step_type="CALCULATION",
        )

        # Verify Step 0 state
        s0 = env.get_state_at_step(0)
        assert s0["low"] == 0
        assert s0["high"] == 9
        assert s0["mid"] == 4

        # Verify Step 1 state
        s1 = env.get_state_at_step(1)
        assert s1["low"] == 5
        assert s1["high"] == 9
        assert s1["mid"] == 4  # Inherited from step 0

        # Verify Step 2 state
        s2 = env.get_state_at_step(2)
        assert s2["low"] == 5
        assert s2["high"] == 9
        assert s2["mid"] == 7  # Evolved to 7

    def test_multiple_variable_updates_within_single_step(self):
        """A single step updating multiple variables captures all of them in the snapshot."""
        env = CalculationEnvironment(inputs={})
        env.record_step_snapshot(
            step_index=0,
            updates={"low": 0, "high": 9, "found": False},
            description="Initialize multi-variable search window",
        )
        s0 = env.get_state_at_step(0)
        assert s0["low"] == 0
        assert s0["high"] == 9
        assert s0["found"] is False

    def test_complete_state_retrieval_with_complex_types(self):
        """Captures structured collections (lists, dicts, sets) and isolates them from mutation."""
        env = CalculationEnvironment(inputs={"items": [10, 20], "visited": {"A"}})
        env.record_step_snapshot(
            step_index=0,
            updates={"visited": {"A", "B"}, "distances": {"A": 0, "B": 5}},
        )

        s0 = env.get_state_at_step(0)
        assert s0["visited"] == {"A", "B"}
        assert s0["distances"] == {"A": 0, "B": 5}

        # Later step modifies distances
        env.record_step_snapshot(
            step_index=1,
            updates={"distances": {"A": 0, "B": 5, "C": 12}},
        )

        # Earlier snapshot must remain unaffected
        s0_again = env.get_state_at_step(0)
        assert "C" not in s0_again["distances"]

    def test_invalid_step_indices_raise_expected_errors(self):
        """Negative steps < -1 or steps exceeding max step index raise IndexError."""
        env = CalculationEnvironment(inputs={"a": 1})
        env.record_step_snapshot(0, {"b": 2})

        with pytest.raises(IndexError):
            env.get_state_at_step(-2)

        with pytest.raises(IndexError):
            env.get_state_at_step(5)

        with pytest.raises(TypeError):
            env.get_state_at_step("0")  # type: ignore

    def test_export_to_computation_trace(self):
        """CalculationEnvironment can export its complete execution into a ComputationTrace."""
        env = CalculationEnvironment(inputs={"x": 10})
        env.record_step_snapshot(0, {"x": 20}, description="Step 0")
        env.record_step_snapshot(1, {"x": 30}, description="Step 1")

        trace = env.get_trace("test_domain")
        assert trace.algorithm_or_domain == "test_domain"
        assert trace.initial_state == {"x": 10}
        assert len(trace.snapshots) == 2
        assert trace.get_state_at_step(0)["x"] == 20
        assert trace.get_state_at_step(1)["x"] == 30


class TestSnapshotAwareStoryboardFidelity:
    """Tests for step-aware fidelity checking against authoritative temporal snapshots."""

    @pytest.fixture
    def binary_search_outline(self) -> Dict[str, Any]:
        """Outline representing a multi-step binary search where 'mid' evolves over time."""
        return {
            "topic": "Binary Search Algorithm",
            "worked_example_spec": {
                "inputs": {"target": 7, "low": 0, "high": 9},
                "steps": [
                    {
                        "step": "Step 1: Calculate midpoint for range [0, 9]",
                        "step_index": 0,
                        "step_number": 1,
                        "result": "mid = 4",
                        "structured_calc": {
                            "output_variable": "mid",
                            "computed_value": 4.0,
                            "corrected_value": "4",
                            "state_snapshot": {"low": 0, "high": 9, "mid": 4},
                        },
                    },
                    {
                        "step": "Step 2: Compare target with arr[mid] and update lower bound",
                        "step_index": 1,
                        "step_number": 2,
                        "result": "low = 5",
                        "structured_calc": {
                            "output_variable": "low",
                            "computed_value": 5.0,
                            "corrected_value": "5",
                            "state_snapshot": {"low": 5, "high": 9, "mid": 4},
                        },
                    },
                    {
                        "step": "Step 3: Recalculate midpoint for range [5, 9]",
                        "step_index": 2,
                        "step_number": 3,
                        "result": "mid = 6",
                        "structured_calc": {
                            "output_variable": "mid",
                            "computed_value": 6.0,
                            "corrected_value": "6",
                            "state_snapshot": {"low": 5, "high": 9, "mid": 6},
                        },
                    },
                ],
                "final_result": "found at index 6",
            },
        }

    def test_binary_search_regression_earlier_value_not_overwritten(
        self, binary_search_outline
    ):
        """
        Required regression scenario:
        Step 1: low = 0, mid = 4
        Step 2: variables change (low = 5)
        Step 3: mid = 6

        A storyboard claiming 'mid = 4' at Step 1 and 'mid = 6' at Step 3 must PASS.
        The later value mid = 6 must not overwrite Step 1's authoritative target.
        """
        checker = StoryboardFidelityChecker()

        storyboard = {
            "sections": [
                {
                    "id": "sec_step_1",
                    "step_number": 1,
                    "title": "Step 1: Initial Midpoint Calculation",
                    "verified_values": {"mid": "4"},
                    "animations": [
                        'MathTex(r"mid = 4") | write | 1.0s',
                    ],
                    "lecture_lines": [
                        "In step 1, our middle pointer mid is calculated as 4.",
                    ],
                },
                {
                    "id": "sec_step_2",
                    "step_number": 2,
                    "title": "Step 2: Updating Bounds",
                    "verified_values": {"low": "5"},
                    "animations": [
                        'MathTex(r"low = 5") | write | 1.0s',
                    ],
                    "lecture_lines": [
                        "Since 7 is greater than the middle element, low becomes 5.",
                    ],
                },
                {
                    "id": "sec_step_3",
                    "step_number": 3,
                    "title": "Step 3: New Midpoint Calculation",
                    "verified_values": {"mid": "6"},
                    "animations": [
                        'MathTex(r"mid = 6") | write | 1.0s',
                    ],
                    "lecture_lines": [
                        "In step 3, our new middle index mid is 6, matching our target found at index 6.",
                    ],
                },
            ]
        }

        report = checker.check_storyboard_fidelity(binary_search_outline, storyboard)
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_audio is True
        assert len(report.discrepancies) == 0

    def test_genuinely_incorrect_claim_at_given_step_still_fails(
        self, binary_search_outline
    ):
        """
        A claim of 'mid = 6' at Step 1 (where mid should be 4) MUST fail with VALUE_MISMATCH.
        """
        checker = StoryboardFidelityChecker()

        bad_storyboard = {
            "sections": [
                {
                    "id": "sec_step_1",
                    "step_number": 1,
                    "title": "Step 1: Midpoint Calculation",
                    # Claiming mid = 6 at Step 1 is genuinely incorrect
                    "verified_values": {"mid": "6"},
                    "animations": [
                        'MathTex(r"mid = 6") | write | 1.0s',
                    ],
                    "lecture_lines": [
                        "In step 1, mid is 6.",
                    ],
                },
            ]
        }

        report = checker.check_storyboard_fidelity(binary_search_outline, bad_storyboard)
        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_audio is False

        # Find the value mismatch discrepancies
        mismatches = [
            d for d in report.discrepancies if d.category == FidelityCategory.VALUE_MISMATCH
        ]
        assert len(mismatches) >= 1
        assert any("4" in str(d.expected_value) for d in mismatches)

    def test_wrong_step_in_narration_line_fails(self, binary_search_outline):
        """
        A spoken narration at Step 1 stating mid is 6 must fail with VALUE_MISMATCH.
        """
        checker = StoryboardFidelityChecker()

        storyboard = {
            "sections": [
                {
                    "id": "sec_step_1",
                    "step_number": 1,
                    "title": "Step 1",
                    "verified_values": {"mid": "4"},
                    "animations": ['MathTex(r"mid = 4")'],
                    # Spoken claim hallucinates mid = 8 at step 1
                    "lecture_lines": ["At this stage, mid is 8."],
                }
            ]
        }

        report = checker.check_storyboard_fidelity(binary_search_outline, storyboard)
        assert report.overall_status == VerificationStatus.FAILED
        narr_mismatches = [
            d for d in report.discrepancies if "lecture_lines" in d.location
        ]
        assert len(narr_mismatches) == 1
        assert narr_mismatches[0].category == FidelityCategory.VALUE_MISMATCH

    def test_out_of_range_step_reference_produces_needs_review(
        self, binary_search_outline
    ):
        """
        Referencing an out-of-range step (e.g. step 99) in a section or claim
        must be handled honestly and flagged as NEEDS_REVIEW (not silently verified).
        """
        checker = StoryboardFidelityChecker()

        storyboard = {
            "sections": [
                {
                    "id": "sec_step_99",
                    "step_number": 99,
                    "title": "Step 99: Out of Bounds",
                    "verified_values": {"mid": "4"},
                    "animations": ['MathTex(r"mid = 4")'],
                    "lecture_lines": ["In step 99, we do something else."],
                },
                {
                    "id": "sec_step_3",
                    "step_number": 3,
                    "title": "Conclusion",
                    "verified_values": {"mid": "6"},
                    "animations": ['MathTex(r"mid = 6")'],
                    "lecture_lines": ["In step 3, the target was found with mid equal to 6."],
                },
            ]
        }

        report = checker.check_storyboard_fidelity(binary_search_outline, storyboard)
        assert report.overall_status == VerificationStatus.NEEDS_REVIEW
        assert report.is_safe_for_audio is False

        ambiguous = [
            d for d in report.discrepancies if d.category == FidelityCategory.AMBIGUOUS_CLAIM
        ]
        assert len(ambiguous) >= 1
        assert any("99" in d.reason for d in ambiguous)

    def test_non_stepped_existing_outlines_remain_fully_compatible(self):
        """
        Standard single-step / scientific outlines without step indexes (e.g. Arrhenius)
        must continue to verify correctly without any regressions.
        """
        checker = StoryboardFidelityChecker()

        outline = {
            "topic": "Arrhenius Equation",
            "governing_principles": ["Arrhenius Equation: k = A * exp(-Ea / (R * T))"],
            "worked_example_spec": {
                "inputs": {
                    "A": "7.5e10",
                    "Ea": "85000",
                    "R": "8.314",
                    "T1": "300",
                },
                "steps": [
                    {
                        "step": "Calculate rate constant k1 at T1 = 300 K",
                        "result": "k1 = 0.001225 s^-1",
                        "structured_calc": {
                            "output_variable": "k1",
                            "computed_value": 0.001225,
                            "unit": "s^-1",
                        },
                    }
                ],
                "final_result": "k1 = 0.001225 s^-1",
            },
        }

        storyboard = {
            "sections": [
                {
                    "id": "worked_example",
                    "verified_values": {"k1": "0.001225 s^-1"},
                    "animations": [
                        'MathTex(r"k_1 = 0.001225\\ s^{-1}") | write | 1.0s'
                    ],
                    "lecture_lines": [
                        "Substituting our values, we find the rate constant k1 is 0.001225 s^-1."
                    ],
                }
            ]
        }

        report = checker.check_storyboard_fidelity(outline, storyboard)
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_audio is True
        assert len(report.discrepancies) == 0


class TestMilestone2CrossDomainGrounding:
    """
    Milestone-level cross-domain grounding verification tests (Milestone 2.3).
    Validates that Binary Search, FCFS, 0/1 Knapsack, and Dijkstra all operate
    authoritatively on the unified temporal trace architecture without cross-domain collisions,
    and that non-computational topics receive NOT_APPLICABLE without numerical errors.
    """

    def test_knapsack_recurrence_and_boundary_correctness(self):
        """0/1 Knapsack complete recurrence evaluation with capacity boundary conditions."""
        from grounding.verifiers.dp_verifier import KnapsackSpec, KnapsackSimulator
        spec = KnapsackSpec(capacity=6, weights=(2, 3, 4), values=(3, 4, 5))
        res = KnapsackSimulator.run(spec)

        # Base case validations
        for w in range(7):
            assert res.table[0][w] == 0
        for i in range(4):
            assert res.table[i][0] == 0

        # Optimum: Item 1 + Item 3 (wt 2+4=6, val 3+5=8) vs Item 1 + Item 2 (wt 5, val 7)
        assert res.optimal_value == 8
        assert res.selected_items == [0, 2]
        assert res.total_weight == 6
        assert res.total_value == 8

    def test_knapsack_excess_capacity_handling(self):
        """Excess-capacity branch copies predecessor directly without evaluating inclusion."""
        from grounding.verifiers.dp_verifier import KnapsackSpec, KnapsackSimulator
        spec = KnapsackSpec(capacity=3, weights=(5,), values=(100,))
        res = KnapsackSimulator.run(spec)

        for w in range(1, 4):
            c_res = res.cell_results[(1, w)]
            assert c_res.selected_branch == "excess_capacity"
            assert c_res.include_candidate is None
            assert c_res.final_value == 0
            assert c_res.selected_predecessor == (0, w)

    def test_knapsack_chronological_snapshots_and_backtracking(self):
        """Snapshots evolve chronologically with uncomputed cells as None and valid backtracking."""
        from grounding.verifiers.dp_verifier import KnapsackSpec, KnapsackSimulator
        # At capacity 2, only one item can fit: Item 2 (wt 2, val 7) vs Item 1 (wt 2, val 3)
        spec = KnapsackSpec(capacity=2, weights=(2, 2), values=(3, 7))
        res = KnapsackSimulator.run(spec)
        trace = res.computation_trace

        assert len(trace.snapshots) > 0
        # Snapshot 0 has None for uncomputed cells
        snap_0 = trace.snapshots[0]
        assert snap_0.state["table"][1][1] is None
        assert snap_0.state["computation_phase"] == "initialization"

        # Backtrack steps reconstructed Item 2 (wt 2, val 7) over Item 1 (wt 2, val 3)
        assert res.selected_items == [1]
        assert res.optimal_value == 7

    def test_dijkstra_authoritative_step_snapshots_and_registry_selection(self):
        """Dijkstra verifier outputs authoritative step snapshots via SpecializedVerifierRegistry."""
        from worked_example_verifier import SpecializedVerifierRegistry
        from grounding.verifiers.graph.dijkstra_verifier import DijkstraTraceVerifier

        spec = {
            "setup": "A directed graph with 3 vertices: A, B, C. Source: A.",
            "inputs": {
                "vertices": ["A", "B", "C"],
                "edges": [["A", "B", 4], ["A", "C", 2], ["C", "B", 1]],
                "source": "A",
            },
            "steps": [
                {
                    "step": "Initialize dist[A] = 0, dist[B] = ∞, dist[C] = ∞, Q = {A, B, C}",
                    "result": "dist = {A: 0, B: ∞, C: ∞}, Q = {A, B, C}",
                },
                {
                    "step": "Extract A from Q, update dist for neighbors B and C",
                    "result": "dist = {A: 0, B: 4, C: 2}, Q = {B, C}",
                },
                {
                    "step": "Extract C from Q, update dist for neighbor B",
                    "result": "dist = {A: 0, B: 3, C: 2}, Q = {B}",
                },
                {
                    "step": "Extract B from Q, no updates needed",
                    "result": "dist = {A: 0, B: 3, C: 2}, Q = {}",
                },
            ],
            "final_result": "dist = {A: 0, B: 3, C: 2}",
        }

        # Registry discovery
        verifier = SpecializedVerifierRegistry.find_verifier(spec, topic="Dijkstra's Shortest Path")
        assert verifier is not None
        assert verifier == DijkstraTraceVerifier

        out_spec, report = verifier.verify_worked_example_spec(spec, topic="Dijkstra's Shortest Path")
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.verification_method == "dijkstra_trace_verifier"
        assert len(report.state_snapshots) >= 4

        # Check step-aware snapshot values
        snap_2 = report.state_snapshots[2]  # Extraction of C
        assert snap_2["extracted_node"] == "C"
        assert snap_2["dist_C"] == 2
        assert snap_2["dist_B"] == 3

    def test_fcfs_compatibility_with_shared_trace_architecture(self):
        """FCFS scheduler runs seamlessly on the shared trace architecture via registry."""
        from worked_example_verifier import SpecializedVerifierRegistry, WorkedExampleVerifier
        from grounding.verifiers.os.fcfs_verifier import FCFSTraceVerifier

        spec = {
            "inputs": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                    {"pid": "P2", "arrival_time": 1, "burst_time": 2},
                ]
            },
            "steps": [
                {"step_number": 1, "current_time": 0, "running_pid": "P1"},
            ],
        }

        verifier = SpecializedVerifierRegistry.find_verifier(spec, topic="FCFS CPU Scheduling")
        assert verifier is not None
        assert verifier == FCFSTraceVerifier

        we_verifier = WorkedExampleVerifier()
        out_spec, report = we_verifier.verify_worked_example_spec(spec, topic="FCFS CPU Scheduling")
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.verification_method == "fcfs_trace_verifier"
        assert len(report.state_snapshots) >= 3

    def test_non_computational_topics_handled_as_not_applicable(self):
        """Qualitative topics (e.g. Photosynthesis) gracefully bypass calculations without AST errors."""
        from worked_example_verifier import WorkedExampleVerifier, VerificationStatus
        outline = {
            "topic": "Photosynthesis: Light Reactions and Calvin Cycle",
            "teaching_beats": [
                {
                    "beat_title": "Light Reactions",
                    "section_type": "conceptual_mechanism",
                    "narration": "Chlorophyll absorbs photon energy to split water molecules.",
                }
            ],
        }
        verifier = WorkedExampleVerifier()
        _, report = verifier.verify_outline(outline)
        assert report.overall_status == VerificationStatus.NOT_APPLICABLE
        assert report.is_safe_for_stage2 is True
        assert report.failed_steps == 0
