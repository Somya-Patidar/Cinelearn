"""
Comprehensive Deterministic Unit & Integration Tests for Milestone 2.3:
Dynamic Programming Trace Grounding — 0/1 Knapsack Simulation & Independent Verification.

Test Categories:
1. KnapsackSpec immutability, parameter bounds, booleans rejection, and fail-closed validation.
2. KnapsackSimulator:
   - Base cases (row 0, col 0).
   - Complete recurrence evaluation (include vs exclude).
   - Excess capacity branch (unconditional exclusion without evaluating inclusion).
   - Deterministic tie-breaking (exclude on tie).
   - 0-indexed item attributes vs 1-indexed DP table rows.
3. Temporal Snapshots & ComputationTrace:
   - Chronological matrix evolution.
   - Uncomputed cells honestly represented as None.
   - Deep-copy isolation.
   - Predecessor coordinates and dependency vectors.
4. Backtracking & Optimal Item Reconstruction:
   - Correct predecessor transitions.
   - Weight constraint and value equivalence.
   - Multiple optimal subsets (deterministic selection).
   - Edge cases: zero capacity, empty items, items heavier than capacity, zero values.
5. Independent DPTraceVerifier (Fail-Closed Verification):
   - Valid trace passes.
   - Corrupted cell value fails.
   - Corrupted branch fails.
   - Corrupted predecessor fails.
   - Incorrect final optimum fails.
   - Incorrect reconstructed subset fails.
   - Premature future state leakage fails.
6. SpecializedVerifierRegistry & Orchestrator Integration:
   - Registry discovery.
   - Cross-domain isolation (no interception of Binary Search, FCFS, Dijkstra, Photosynthesis).
   - Clean routing through WorkedExampleVerifier.
7. Step-Aware Storyboard Fidelity Integration.
"""

from __future__ import annotations

import copy
import pytest
from typing import Any, Dict, List

from grounding.verifiers.dp_verifier import (
    KnapsackSpec,
    KnapsackCellResult,
    KnapsackBacktrackStep,
    KnapsackSimulationResult,
    KnapsackSimulator,
    DPTraceVerifier,
)
from grounding.trace_engine import StateSnapshot, ComputationTrace
from worked_example_verifier import (
    SpecializedVerifierRegistry,
    WorkedExampleReport,
    WorkedExampleVerifier,
    VerificationStatus,
)
from storyboard_fidelity_checker import (
    StoryboardFidelityChecker,
    AuthoritativeOutlineModel,
)


# ============================================================================
# 1. KnapsackSpec Model & Validation Tests
# ============================================================================

class TestKnapsackSpecValidation:
    """Verifies construction, immutability, and fail-closed validation of KnapsackSpec."""

    def test_valid_spec_creation(self):
        spec = KnapsackSpec(
            capacity=10,
            weights=(2, 3, 5),
            values=(3, 4, 8),
            item_names=("ItemA", "ItemB", "ItemC"),
        )
        assert spec.capacity == 10
        assert spec.weights == (2, 3, 5)
        assert spec.values == (3, 4, 8)
        assert spec.item_names == ("ItemA", "ItemB", "ItemC")
        assert spec.num_items == 3

    def test_spec_is_immutable(self):
        spec = KnapsackSpec(capacity=10, weights=(2,), values=(3,))
        with pytest.raises(Exception):
            spec.capacity = 20  # type: ignore

    def test_omitted_item_names_defaults_to_numbered(self):
        spec = KnapsackSpec(capacity=10, weights=(2, 3), values=(3, 4))
        assert spec.item_names == ("Item_1", "Item_2")

    def test_mismatched_weights_and_values_length_rejected(self):
        with pytest.raises(ValueError, match="Weights and values must have matching lengths"):
            KnapsackSpec(capacity=10, weights=(2, 3), values=(4,))

    def test_mismatched_item_names_length_rejected(self):
        with pytest.raises(ValueError, match="Item names length.*must match number of items"):
            KnapsackSpec(capacity=10, weights=(2, 3), values=(4, 5), item_names=("A",))

    def test_empty_item_name_rejected(self):
        with pytest.raises(ValueError, match="Item name at index 0 must be a non-empty string"):
            KnapsackSpec(capacity=10, weights=(2,), values=(4,), item_names=("",))
        with pytest.raises(ValueError, match="Item name at index 0 must be a non-empty string"):
            KnapsackSpec(capacity=10, weights=(2,), values=(4,), item_names=("   ",))

    def test_negative_capacity_rejected(self):
        with pytest.raises(ValueError, match="Capacity must be non-negative"):
            KnapsackSpec(capacity=-5, weights=(2,), values=(4,))

    def test_zero_or_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="strictly positive"):
            KnapsackSpec(capacity=10, weights=(0,), values=(4,))
        with pytest.raises(ValueError, match="strictly positive"):
            KnapsackSpec(capacity=10, weights=(-2,), values=(4,))

    def test_negative_value_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            KnapsackSpec(capacity=10, weights=(2,), values=(-4,))

    def test_zero_value_accepted(self):
        spec = KnapsackSpec(capacity=10, weights=(2,), values=(0,))
        assert spec.values == (0,)

    def test_reject_booleans_where_integers_expected(self):
        with pytest.raises(TypeError, match="Capacity must be an integer"):
            KnapsackSpec(capacity=True, weights=(2,), values=(4,))  # type: ignore
        with pytest.raises(TypeError, match="Weight at index 0 must be an integer"):
            KnapsackSpec(capacity=10, weights=(True,), values=(4,))  # type: ignore
        with pytest.raises(TypeError, match="Value at index 0 must be an integer"):
            KnapsackSpec(capacity=10, weights=(2,), values=(False,))  # type: ignore

    def test_reject_floats_where_integers_expected(self):
        with pytest.raises(TypeError, match="Capacity must be an integer"):
            KnapsackSpec(capacity=10.5, weights=(2,), values=(4,))  # type: ignore
        with pytest.raises(TypeError, match="Weight at index 0 must be an integer"):
            KnapsackSpec(capacity=10, weights=(2.5,), values=(4,))  # type: ignore
        with pytest.raises(TypeError, match="Value at index 0 must be an integer"):
            KnapsackSpec(capacity=10, weights=(2,), values=(4.5,))  # type: ignore

    def test_empty_inputs_handled_cleanly(self):
        spec = KnapsackSpec(capacity=0, weights=(), values=())
        assert spec.capacity == 0
        assert spec.num_items == 0


# ============================================================================
# 2. KnapsackSimulator Recurrence & Boundary Correctness Tests
# ============================================================================

class TestKnapsackSimulatorRecurrence:
    """Verifies recurrence evaluation, base cases, branches, and cell dependencies."""

    def test_base_cases_row_zero_and_col_zero(self):
        spec = KnapsackSpec(capacity=4, weights=(2, 3), values=(3, 4))
        res = KnapsackSimulator.run(spec)
        table = res.table

        # dp[0][w] = 0 for all w in [0, 4]
        for w in range(5):
            assert table[0][w] == 0

        # dp[i][0] = 0 for all i in [0, 2]
        for i in range(3):
            assert table[i][0] == 0

    def test_excess_capacity_branch_unconditionally_inherits_predecessor(self):
        """
        When w < wt[i-1], the item cannot fit.
        Must directly inherit dp[i-1][w] and record 'excess_capacity'.
        """
        spec = KnapsackSpec(capacity=5, weights=(4,), values=(10,))
        res = KnapsackSimulator.run(spec)

        for w in range(1, 4):  # weights[0] = 4, so capacities 1, 2, 3 are excess capacity
            cell_res = res.cell_results[(1, w)]
            assert cell_res.selected_branch == "excess_capacity"
            assert cell_res.include_candidate is None
            assert cell_res.include_cell is None
            assert cell_res.exclude_candidate == 0
            assert cell_res.final_value == 0
            assert cell_res.selected_predecessor == (0, w)

        # Capacity 4 can fit
        cell_4 = res.cell_results[(1, 4)]
        assert cell_4.selected_branch == "include"
        assert cell_4.final_value == 10

    def test_inclusion_candidate_greater_than_exclusion(self):
        spec = KnapsackSpec(capacity=3, weights=(2,), values=(7,))
        res = KnapsackSimulator.run(spec)

        cell_res = res.cell_results[(1, 2)]
        assert cell_res.selected_branch == "include"
        assert cell_res.exclude_candidate == 0
        assert cell_res.include_candidate == 7
        assert cell_res.final_value == 7
        assert cell_res.selected_predecessor == (0, 0)
        assert cell_res.include_cell == (0, 0)

    def test_exclusion_candidate_greater_than_inclusion(self):
        # Item 1: wt 2, val 10
        # Item 2: wt 2, val 5 (at capacity 2, exclude Item 2 preserves Item 1 value 10)
        spec = KnapsackSpec(capacity=2, weights=(2, 2), values=(10, 5))
        res = KnapsackSimulator.run(spec)

        cell_res = res.cell_results[(2, 2)]
        assert cell_res.selected_branch == "exclude"
        assert cell_res.exclude_candidate == 10
        assert cell_res.include_candidate == 5
        assert cell_res.final_value == 10
        assert cell_res.selected_predecessor == (1, 2)

    def test_tie_breaking_policy_prefers_exclusion(self):
        """When include_candidate == exclude_candidate, choose 'exclude'."""
        # Item 1: wt 2, val 5
        # Item 2: wt 2, val 5 (at capacity 2, exclude gives 5, include gives 0+5=5)
        spec = KnapsackSpec(
            capacity=2,
            weights=(2, 2),
            values=(5, 5),
            item_names=("First", "Second"),
        )
        res = KnapsackSimulator.run(spec)

        cell_res = res.cell_results[(2, 2)]
        assert cell_res.include_candidate == 5
        assert cell_res.exclude_candidate == 5
        assert cell_res.selected_branch == "exclude"
        assert cell_res.final_value == 5
        assert cell_res.selected_predecessor == (1, 2)  # Exclude predecessor
        assert "deterministic tie-breaking convention" in cell_res.decision_reason.lower()
        assert "including the current item does not improve the objective" in cell_res.decision_reason.lower()

    def test_canonical_multi_item_optimal_value(self):
        # Standard textbook knapsack problem:
        # Items:
        # 1: wt 2, val 3
        # 2: wt 3, val 4
        # 3: wt 4, val 5
        # 4: wt 5, val 8
        # Capacity: 7
        # Optimal: Item 1 + Item 4 (wt 2+5=7, val 3+8=11)
        spec = KnapsackSpec(capacity=7, weights=(2, 3, 4, 5), values=(3, 4, 5, 8))
        res = KnapsackSimulator.run(spec)

        assert res.optimal_value == 11
        assert res.selected_items == [0, 3]
        assert res.total_weight == 7
        assert res.total_value == 11


# ============================================================================
# 3. Temporal State Snapshots & Isolation Tests
# ============================================================================

class TestKnapsackTemporalSnapshots:
    """Verifies chronological evolution, honest None representation, and deep-copy isolation."""

    def test_chronological_snapshot_evolution(self):
        spec = KnapsackSpec(capacity=3, weights=(2, 1), values=(4, 2))
        res = KnapsackSimulator.run(spec)
        trace = res.computation_trace
        snaps = trace.snapshots

        # Snapshot 0: Base case initialization
        snap_0 = snaps[0]
        assert snap_0.state["computation_phase"] == "initialization"
        assert snap_0.state["current_branch"] == "base_case"
        assert snap_0.state["optimal_value"] is None

        # Verify uncomputed cells are None, not 0
        table_0 = snap_0.state["table"]
        assert table_0[0][0] == 0
        assert table_0[0][1] == 0
        assert table_0[1][1] is None  # Cell (1, 1) not yet computed!
        assert table_0[2][3] is None  # Cell (2, 3) not yet computed!

        # Snapshot for cell (1, 1): weight 2 > cap 1 => excess_capacity
        snap_1_1 = snaps[1]
        assert snap_1_1.state["active_cell"] == [1, 1]
        assert snap_1_1.state["current_branch"] == "excess_capacity"
        assert snap_1_1.state["final_value"] == 0
        assert snap_1_1.state["table"][1][1] == 0
        assert snap_1_1.state["table"][1][2] is None  # Cell (1, 2) not yet computed!

        # Final completion snapshot
        snap_final = snaps[-1]
        assert snap_final.state["computation_phase"] == "completed"
        assert snap_final.state["optimal_value"] == 6

    def test_snapshot_deep_copy_isolation(self):
        """Mutating a snapshot table must never alter other snapshots or the simulator."""
        spec = KnapsackSpec(capacity=2, weights=(1,), values=(5,))
        res = KnapsackSimulator.run(spec)
        trace = res.computation_trace

        snap_0 = trace.snapshots[0]
        # Attempt external mutation of retrieved snapshot table
        snap_0.state["table"][0][0] = 9999
        snap_0.state["table"][1][1] = 8888

        # Simulator completed table must be unchanged
        assert res.table[0][0] == 0
        assert res.table[1][1] == 5

        # Subsequent snapshot must be isolated
        snap_1 = trace.snapshots[1]
        assert snap_1.state["table"][0][0] == 0
        assert snap_1.state["table"][1][1] != 8888

    def test_flat_variable_bindings_present_in_snapshots(self):
        spec = KnapsackSpec(capacity=2, weights=(1,), values=(4,))
        res = KnapsackSimulator.run(spec)
        final_snap = res.computation_trace.snapshots[-1]

        # Flat keys for formula evaluators
        assert "dp_0_0" in final_snap.state
        assert "dp_1_1" in final_snap.state
        assert "cell_r1_c2" in final_snap.state
        assert final_snap.state["dp_1_2"] == 4
        assert final_snap.state["optimal_value"] == 4


# ============================================================================
# 4. Backtracking & Reconstruction Tests
# ============================================================================

class TestKnapsackBacktracking:
    """Verifies deterministic backtracking, subset reconstruction, and edge cases."""

    def test_backtracking_transitions_and_invariants(self):
        # Items:
        # 1: wt 1, val 2
        # 2: wt 2, val 5
        # Cap: 3 -> Both fit, optimal = 7
        spec = KnapsackSpec(capacity=3, weights=(1, 2), values=(2, 5))
        res = KnapsackSimulator.run(spec)

        assert res.optimal_value == 7
        assert res.selected_items == [0, 1]
        assert res.total_weight == 3
        assert res.total_value == 7

        steps = res.backtrack_steps
        assert len(steps) == 2
        # First backtrack step inspects Item 2 at (2, 3) -> included -> next (1, 1)
        assert steps[0].current_cell == (2, 3)
        assert steps[0].item_included is True
        assert steps[0].next_cell == (1, 1)
        # Second backtrack step inspects Item 1 at (1, 1) -> included -> next (0, 0)
        assert steps[1].current_cell == (1, 1)
        assert steps[1].item_included is True
        assert steps[1].next_cell == (0, 0)

    def test_backtracking_with_excluded_items(self):
        # Item 1: wt 3, val 10
        # Item 2: wt 4, val 12
        # Cap: 4 -> Item 2 alone is 12, Item 1 is 10. Item 2 selected, Item 1 excluded.
        spec = KnapsackSpec(capacity=4, weights=(3, 4), values=(10, 12))
        res = KnapsackSimulator.run(spec)

        assert res.optimal_value == 12
        assert res.selected_items == [1]
        assert res.total_weight == 4

        steps = res.backtrack_steps
        assert steps[0].item_included is True  # Item 2 included
        assert steps[1].item_included is False  # Item 1 excluded

    def test_all_items_exceed_capacity(self):
        spec = KnapsackSpec(capacity=2, weights=(5, 10), values=(100, 200))
        res = KnapsackSimulator.run(spec)

        assert res.optimal_value == 0
        assert res.selected_items == []
        assert res.total_weight == 0
        assert res.total_value == 0

    def test_zero_capacity_knapsack(self):
        spec = KnapsackSpec(capacity=0, weights=(2, 3), values=(5, 6))
        res = KnapsackSimulator.run(spec)

        assert res.optimal_value == 0
        assert res.selected_items == []
        assert res.total_weight == 0

    def test_tied_optimal_subsets_deterministic_selection(self):
        """
        Two subsets yield the exact same value.
        Item 1: wt 2, val 5
        Item 2: wt 2, val 5
        Tie-breaking rule excludes Item 2 on tie at cell (2, 2).
        Therefore Item 1 is selected.
        """
        spec = KnapsackSpec(
            capacity=2,
            weights=(2, 2),
            values=(5, 5),
            item_names=("ItemA", "ItemB"),
        )
        res = KnapsackSimulator.run(spec)

        assert res.optimal_value == 5
        assert res.selected_items == [0]  # ItemA selected, ItemB excluded on tie
        assert res.selected_item_names == ["ItemA"]


# ============================================================================
# 5. Independent Fail-Closed DPTraceVerifier Tests
# ============================================================================

class TestDPTraceVerifier:
    """Verifies outline step auditing, fail-closed rejection of corrupted claims."""

    def test_can_verify_recognition(self):
        assert DPTraceVerifier.can_verify({}, topic="0/1 Knapsack Problem") is True
        assert DPTraceVerifier.can_verify({}, topic="Knapsack Problem") is True
        assert DPTraceVerifier.can_verify({"algorithm": "knapsack_01"}, topic="") is True
        assert DPTraceVerifier.can_verify({"domain": "dynamic_programming"}, topic="") is True
        assert DPTraceVerifier.can_verify({"setup": "A knapsack with capacity 5"}, topic="") is True
        assert DPTraceVerifier.can_verify({
            "inputs": {"capacity": 5, "weights": [2], "values": [3]}
        }, topic="") is True

        # Non-DP topics must NOT be claimed
        assert DPTraceVerifier.can_verify({}, topic="Binary Search") is False
        assert DPTraceVerifier.can_verify({}, topic="FCFS CPU Scheduling") is False
        assert DPTraceVerifier.can_verify({}, topic="Dijkstra's Shortest Path") is False
        assert DPTraceVerifier.can_verify({}, topic="Photosynthesis") is False

    def test_valid_outline_spec_passes_verification(self):
        spec = {
            "topic": "0/1 Knapsack Dynamic Programming",
            "inputs": {
                "capacity": 3,
                "weights": [2, 1],
                "values": [4, 2],
            },
            "steps": [
                {
                    "step_number": 1,
                    "row": 1,
                    "col": 2,
                    "value": 4,
                    "branch": "include",
                    "description": "At cell (1, 2), Item 1 is included with value 4.",
                },
                {
                    "step_number": 2,
                    "row": 2,
                    "col": 3,
                    "value": 6,
                    "branch": "include",
                    "description": "At cell (2, 3), Item 2 is included with total value 6.",
                },
            ],
            "final_result": "Optimal Value = 6, Selected Items = ['Item_1', 'Item_2']",
        }

        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec, topic="Knapsack")
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.verified_steps == 2
        assert report.failed_steps == 0
        assert report.final_result_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_stage2 is True
        assert "verified_algorithmic_state" in out_spec["verification_metadata"]
        assert len(report.state_snapshots) > 0

    def test_invalid_input_fails_closed(self):
        # Mismatched lengths in input
        spec = {
            "inputs": {
                "capacity": 5,
                "weights": [2, 3],
                "values": [10],
            },
            "steps": [{"step_number": 1}],
        }
        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec, topic="Knapsack")
        assert report.overall_status == VerificationStatus.FAILED
        assert report.is_safe_for_stage2 is False
        assert "matching lengths" in report.summary

    def test_corrupted_cell_value_detected_and_rejected(self):
        """Deliberately corrupted cell value must fail verification."""
        spec = {
            "inputs": {
                "capacity": 3,
                "weights": [2],
                "values": [5],
            },
            "steps": [
                {
                    "step_number": 1,
                    "row": 1,
                    "col": 2,
                    "value": 999,  # Authoritative value is 5!
                    "branch": "include",
                }
            ],
        }
        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec, topic="Knapsack")
        assert report.overall_status == VerificationStatus.FAILED
        assert report.failed_steps == 1
        assert "Cell value mismatch" in report.step_diagnostics[0]["details"]

    def test_corrupted_branch_choice_detected_and_rejected(self):
        """Claiming 'include' when the item weight exceeds capacity must fail."""
        spec = {
            "inputs": {
                "capacity": 3,
                "weights": [5],  # Weight 5 > Capacity 3
                "values": [10],
            },
            "steps": [
                {
                    "step_number": 1,
                    "row": 1,
                    "col": 2,
                    "value": 0,
                    "branch": "include",  # Authoritative branch is 'excess_capacity'!
                }
            ],
        }
        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec, topic="Knapsack")
        assert report.overall_status == VerificationStatus.FAILED
        assert report.failed_steps == 1
        assert "Recurrence branch mismatch" in report.step_diagnostics[0]["details"]

    def test_corrupted_predecessor_coordinates_rejected(self):
        spec = {
            "inputs": {
                "capacity": 3,
                "weights": [2],
                "values": [5],
            },
            "steps": [
                {
                    "step_number": 1,
                    "row": 1,
                    "col": 2,
                    "value": 5,
                    "predecessor": [0, 99],  # Authoritative predecessor is [0, 0]
                }
            ],
        }
        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec, topic="Knapsack")
        assert report.overall_status == VerificationStatus.FAILED
        assert "Predecessor mismatch" in report.step_diagnostics[0]["details"]

    def test_incorrect_final_result_rejected(self):
        spec = {
            "inputs": {
                "capacity": 3,
                "weights": [2],
                "values": [5],
            },
            "steps": [],
            "final_result": "Optimal Value = 999",  # True is 5
        }
        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec, topic="Knapsack")
        assert report.overall_status == VerificationStatus.FAILED
        assert report.final_result_status == VerificationStatus.FAILED

    def test_auto_fill_canonical_final_result_when_omitted(self):
        spec = {
            "inputs": {
                "capacity": 3,
                "weights": [2],
                "values": [5],
            },
            "steps": [],
        }
        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec, topic="Knapsack")
        assert report.final_result_status == VerificationStatus.VERIFIED
        assert "Optimal Value = 5" in out_spec["final_result"]


# ============================================================================
# 6. SpecializedVerifierRegistry & Orchestrator Integration Tests
# ============================================================================

class TestKnapsackRegistryIntegration:
    """Verifies that SpecializedVerifierRegistry discovers and routes Knapsack cleanly."""

    def test_knapsack_discovered_in_registry(self):
        verifier = SpecializedVerifierRegistry.find_verifier(
            spec={"inputs": {"capacity": 5, "weights": [2], "values": [3]}},
            topic="0/1 Knapsack Problem",
        )
        assert verifier is not None
        assert verifier == DPTraceVerifier

    def test_registry_cross_domain_isolation(self):
        """Knapsack verifier must NOT intercept Dijkstra, FCFS, or Binary Search."""
        v_dijkstra = SpecializedVerifierRegistry.find_verifier(
            spec={"setup": "Graph with 5 vertices. Source: A"},
            topic="Dijkstra's Shortest Path",
        )
        assert v_dijkstra != DPTraceVerifier

        v_fcfs = SpecializedVerifierRegistry.find_verifier(
            spec={"inputs": {"processes": [{"pid": "P1", "arrival_time": 0, "burst_time": 4}]}},
            topic="FCFS CPU Scheduling",
        )
        assert v_fcfs != DPTraceVerifier

        v_bs = SpecializedVerifierRegistry.find_verifier(
            spec={"inputs": {"arr": [1, 2, 3], "target": 2}},
            topic="Binary Search",
        )
        assert v_bs != DPTraceVerifier

    def test_worked_example_verifier_routes_knapsack_cleanly(self):
        verifier = WorkedExampleVerifier()
        spec = {
            "inputs": {
                "capacity": 4,
                "weights": [2, 3],
                "values": [3, 4],
            },
            "steps": [
                {
                    "step_number": 1,
                    "row": 1,
                    "col": 2,
                    "value": 3,
                }
            ],
        }

        out_spec, report = verifier.verify_worked_example_spec(spec, topic="0/1 Knapsack Problem")
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.verification_method == "dp_trace_verifier"
        assert report.verified_algorithmic_state is not None
        assert report.verified_algorithmic_state["algorithm"] == "knapsack_01"


# ============================================================================
# 7. Storyboard Fidelity Integration with Knapsack Snapshots
# ============================================================================

class TestStoryboardFidelityKnapsackIntegration:
    """Verifies step-aware checking against DP matrix state snapshots."""

    def test_storyboard_fidelity_step_aware_with_knapsack_snapshots(self):
        spec = {
            "inputs": {
                "capacity": 3,
                "weights": [2, 1],
                "values": [4, 2],
            },
            "steps": [
                {
                    "step_number": 1,
                    "row": 1,
                    "col": 2,
                    "value": 4,
                },
                {
                    "step_number": 2,
                    "row": 2,
                    "col": 3,
                    "value": 6,
                },
            ],
        }
        _, report = DPTraceVerifier.verify_worked_example_spec(spec, topic="Knapsack")

        outline_data = {
            "topic": "0/1 Knapsack Problem",
            "worked_example_spec": spec,
            "verification_metadata": report.to_dict(),
            "teaching_beats": [
                {
                    "beat_title": "Step 1: Cell (1, 2)",
                    "section_type": "worked_demonstration",
                    "step_number": 1,
                },
                {
                    "beat_title": "Step 2: Cell (2, 3)",
                    "section_type": "worked_demonstration",
                    "step_number": 2,
                },
            ],
        }

        storyboard_data = {
            "sections": [
                {
                    "section_name": "Step 1: Cell (1, 2)",
                    "lecture_lines": [
                        "At cell row 1 column 2, the value computed is 4."
                    ],
                },
                {
                    "section_name": "Step 2: Cell (2, 3)",
                    "lecture_lines": [
                        "At cell row 2 column 3, including Item 2 yields total value 6."
                    ],
                },
            ]
        }

        checker = StoryboardFidelityChecker()
        fidelity_report = checker.check_storyboard_fidelity(outline_data, storyboard_data)
        assert fidelity_report.overall_status in [VerificationStatus.VERIFIED, VerificationStatus.NEEDS_REVIEW]
        assert len(fidelity_report.discrepancies) == 0


# ============================================================================
# 8. M2.3 Audit: Empty Knapsack Input Tests (N = 0)
# ============================================================================

class TestEmptyKnapsackInput:
    """Verifies that N = 0 is supported across simulation, trace verification, and outline verification."""

    def test_n_zero_with_positive_capacity_simulation(self):
        spec = KnapsackSpec(capacity=5, weights=(), values=())
        res = KnapsackSimulator.run(spec)
        assert res.optimal_value == 0
        assert res.selected_items == []
        assert res.selected_item_names == []
        assert res.total_weight == 0
        assert res.total_value == 0
        assert res.table == [[0, 0, 0, 0, 0, 0]]
        assert len(res.backtrack_steps) == 0

        verif = DPTraceVerifier.verify_trace(spec, res.computation_trace)
        assert verif["verified"] is True
        assert verif["status"] == VerificationStatus.VERIFIED

    def test_n_zero_with_zero_capacity_simulation(self):
        spec = KnapsackSpec(capacity=0, weights=(), values=())
        res = KnapsackSimulator.run(spec)
        assert res.optimal_value == 0
        assert res.selected_items == []
        assert res.table == [[0]]
        assert len(res.backtrack_steps) == 0

        verif = DPTraceVerifier.verify_trace(spec, res.computation_trace)
        assert verif["verified"] is True

    def test_empty_knapsack_worked_example_verification_succeeds(self):
        spec_dict = {
            "capacity": 5,
            "weights": [],
            "values": [],
            "steps": [],
        }
        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec_dict, topic="Knapsack")
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_stage2 is True
        assert report.verified_algorithmic_state["optimal_value"] == 0
        assert report.verified_algorithmic_state["selected_items"] == []
        assert report.verified_algorithmic_state["dp_table"] == [[0, 0, 0, 0, 0, 0]]


# ============================================================================
# 9. M2.3 Audit: Tie-Breaking Rationale & Multiple Optimal Subsets
# ============================================================================

class TestTieBreakingRationaleAndMultipleOptima:
    """
    Verifies that tie-breaking follows the exact rationale:
    'When inclusion and exclusion produce equal objective values, exclusion is selected
    as a deterministic tie-breaking convention. Including the current item does not improve the objective.'
    Demonstrates that exclusion on a tie is a deterministic convention rather than a global
    weight or item-count minimizer.
    """

    def test_tie_breaking_rationale_string_exact(self):
        spec = KnapsackSpec(
            capacity=2,
            weights=(2, 2),
            values=(5, 5),
            item_names=("ItemA", "ItemB"),
        )
        res = KnapsackSimulator.run(spec)
        cell_res = res.cell_results[(2, 2)]
        assert cell_res.include_candidate == cell_res.exclude_candidate == 5
        assert cell_res.selected_branch == "exclude"
        assert "when inclusion and exclusion produce equal objective values" in cell_res.decision_reason.lower()
        assert "exclusion is selected as a deterministic tie-breaking convention" in cell_res.decision_reason.lower()
        assert "including the current item does not improve the objective" in cell_res.decision_reason.lower()

    def test_multiple_optimal_subsets_exclusion_does_not_globally_minimize_weight(self):
        """
        Demonstrates that exclusion does not always minimize total weight:
        - Item 1: wt 3, val 10
        - Item 2: wt 1, val 5
        - Item 3: wt 1, val 5
        Capacity: 3
        At cell (3, 3):
          - Exclude gives dp[2][3] = 10 (Item 1 only, wt 3, 1 item)
          - Include gives dp[2][2] + val[2] = 5 + 5 = 10 (Items 2 and 3, wt 1+1=2, 2 items)
        Both achieve value 10.
        Exclusion is chosen: selected subset is [Item 1] with weight 3 and 1 item.
        Notice that inclusion candidate had weight 2 (< 3)!
        Thus, exclusion does NOT universally minimize weight.
        """
        spec = KnapsackSpec(
            capacity=3,
            weights=(3, 1, 1),
            values=(10, 5, 5),
            item_names=("Item1_heavy", "Item2_light", "Item3_light"),
        )
        res = KnapsackSimulator.run(spec)
        assert res.optimal_value == 10
        # Selected subset is Item 1 (weight 3), despite Items 2 & 3 having weight 2
        assert res.selected_items == [0]
        assert res.total_weight == 3
        assert res.total_value == 10

        # Verifier confirms the trace
        verif = DPTraceVerifier.verify_trace(spec, res.computation_trace)
        assert verif["verified"] is True


# ============================================================================
# 10. M2.3 Audit: Deliberate Corruption Fail-Closed Tests
# ============================================================================

class TestDeliberateCorruptionFailClosed:
    """
    Audits DPTraceVerifier fail-closed rejection when presented with corrupted traces
    or malicious outline claims. Corrupted traces must NOT remain VERIFIED.
    """

    @pytest.fixture
    def canonical_spec_and_sim(self):
        spec = KnapsackSpec(capacity=4, weights=(2, 3), values=(3, 4))
        res = KnapsackSimulator.run(spec)
        return spec, res

    def test_corrupted_base_case_rejected(self, canonical_spec_and_sim):
        spec, res = canonical_spec_and_sim
        trace = res.computation_trace

        # Corrupt Snapshot 0 base case table: set dp[0][2] = 99
        bad_trace = copy.deepcopy(trace)
        bad_trace.snapshots[0].state["table"][0][2] = 99

        verif = DPTraceVerifier.verify_trace(spec, bad_trace)
        assert verif["verified"] is False
        assert any("base case violation" in err.lower() for err in verif["errors"])

    def test_corrupted_recurrence_cell_value_rejected(self, canonical_spec_and_sim):
        spec, res = canonical_spec_and_sim
        bad_trace = copy.deepcopy(res.computation_trace)
        # Corrupt a table filling snapshot value
        snap = [s for s in bad_trace.snapshots if s.state.get("computation_phase") == "table_filling"][0]
        snap.state["final_value"] = 999

        verif = DPTraceVerifier.verify_trace(spec, bad_trace)
        assert verif["verified"] is False
        assert any("final value mismatch" in err.lower() for err in verif["errors"])

    def test_corrupted_recurrence_branch_choice_rejected(self, canonical_spec_and_sim):
        spec, res = canonical_spec_and_sim
        bad_trace = copy.deepcopy(res.computation_trace)
        snap = [s for s in bad_trace.snapshots if s.state.get("computation_phase") == "table_filling"][0]
        snap.state["current_branch"] = "wrong_branch"

        verif = DPTraceVerifier.verify_trace(spec, bad_trace)
        assert verif["verified"] is False
        assert any("branch mismatch" in err.lower() for err in verif["errors"])

    def test_corrupted_tie_breaking_choice_rejected(self):
        # Tied problem: capacity 2, items (2, 5) and (2, 5)
        spec = KnapsackSpec(capacity=2, weights=(2, 2), values=(5, 5))
        res = KnapsackSimulator.run(spec)
        bad_trace = copy.deepcopy(res.computation_trace)

        # Corrupt cell (2, 2) which was a tie: claim "include" instead of "exclude"
        for snap in bad_trace.snapshots:
            if snap.state.get("active_row") == 2 and snap.state.get("active_col") == 2:
                snap.state["current_branch"] = "include"

        verif = DPTraceVerifier.verify_trace(spec, bad_trace)
        assert verif["verified"] is False
        assert any("branch mismatch" in err.lower() for err in verif["errors"])

    def test_corrupted_excess_capacity_candidate_rejected(self, canonical_spec_and_sim):
        spec, res = canonical_spec_and_sim
        bad_trace = copy.deepcopy(res.computation_trace)
        # Find excess capacity snapshot (item 1 has weight 2, at capacity 1 it's excess capacity)
        for snap in bad_trace.snapshots:
            if snap.state.get("current_branch") == "excess_capacity":
                # Illegally set include_candidate
                snap.state["include_candidate"] = 100
                break

        verif = DPTraceVerifier.verify_trace(spec, bad_trace)
        assert verif["verified"] is False
        assert any("excess capacity violation" in err.lower() for err in verif["errors"])

    def test_corrupted_predecessor_coordinates_in_outline_rejected(self, canonical_spec_and_sim):
        spec, _ = canonical_spec_and_sim
        spec_dict = {
            "capacity": spec.capacity,
            "weights": list(spec.weights),
            "values": list(spec.values),
            "steps": [
                {
                    "row": 1,
                    "col": 2,
                    "value": 3,
                    "branch": "include",
                    "selected_predecessor": [0, 99],  # Wrong predecessor
                }
            ],
        }
        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec_dict)
        assert report.overall_status == VerificationStatus.FAILED
        assert out_spec["steps"][0]["verification_status"] == VerificationStatus.FAILED.value

    def test_corrupted_out_of_bounds_coordinates_in_outline_rejected(self, canonical_spec_and_sim):
        spec, _ = canonical_spec_and_sim
        spec_dict = {
            "capacity": spec.capacity,
            "weights": list(spec.weights),
            "values": list(spec.values),
            "steps": [
                {"row": 99, "col": 99, "value": 100},
                {"row": -1, "col": 2, "value": 0},
            ],
        }
        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec_dict)
        assert report.overall_status == VerificationStatus.FAILED
        for s in out_spec["steps"]:
            assert s["verification_status"] == VerificationStatus.FAILED.value

    def test_corrupted_reconstructed_subset_rejected(self, canonical_spec_and_sim):
        spec, _ = canonical_spec_and_sim
        spec_dict = {
            "capacity": spec.capacity,
            "weights": list(spec.weights),
            "values": list(spec.values),
            "selected_items": [999],  # Nonexistent item
        }
        out_spec, report = DPTraceVerifier.verify_worked_example_spec(spec_dict)
        assert report.overall_status == VerificationStatus.FAILED
        assert report.final_result_status == VerificationStatus.FAILED

    def test_corrupted_backtracking_transition_rejected(self, canonical_spec_and_sim):
        spec, res = canonical_spec_and_sim
        bad_trace = copy.deepcopy(res.computation_trace)
        bt_snaps = [s for s in bad_trace.snapshots if s.state.get("computation_phase") == "backtracking"]
        assert len(bt_snaps) > 0
        # Invert inclusion decision
        bt_snaps[0].state["backtracking_state"]["item_included"] = not bt_snaps[0].state["backtracking_state"]["item_included"]

        verif = DPTraceVerifier.verify_trace(spec, bad_trace)
        assert verif["verified"] is False
        assert any("backtracking" in err.lower() for err in verif["errors"])

    def test_missing_evidence_empty_trace_rejected(self, canonical_spec_and_sim):
        spec, _ = canonical_spec_and_sim
        verif = DPTraceVerifier.verify_trace(spec, [])
        assert verif["verified"] is False
        assert any("zero snapshots" in err.lower() for err in verif["errors"])


# ============================================================================
# 11. M2.3 Audit: Snapshot Isolation and Temporal Correctness
# ============================================================================

class TestSnapshotIsolationAndTemporalCorrectness:
    """
    Verifies that state snapshots are immutable from caller mutations,
    uncomputed cells remain None, future DP values never leak into earlier states,
    and earlier snapshots cannot be mutated by later steps.
    """

    def test_retrieved_snapshots_cannot_mutate_stored_trace(self):
        spec = KnapsackSpec(capacity=4, weights=(2, 3), values=(3, 4))
        res = KnapsackSimulator.run(spec)
        trace = res.computation_trace

        # Retrieve snapshot 0 and modify its state
        snap0 = trace.get_snapshot(0)
        snap0.state["table"][0][0] = 99999
        snap0.state["new_key"] = "leaked"

        # Stored trace snapshot must be completely unaffected
        fresh_snap0 = trace.get_snapshot(0)
        assert fresh_snap0.state["table"][0][0] == 0
        assert "new_key" not in fresh_snap0.state

    def test_earlier_snapshots_cannot_be_changed_through_later_snapshots(self):
        spec = KnapsackSpec(capacity=4, weights=(2, 3), values=(3, 4))
        res = KnapsackSimulator.run(spec)
        trace = res.computation_trace

        snap0_table = copy.deepcopy(trace.get_snapshot(0).state["table"])
        snap1_table = copy.deepcopy(trace.get_snapshot(1).state["table"])

        # Table in snapshot 0 must retain uncomputed cells as None
        for r in range(1, 3):
            for c in range(1, 5):
                assert snap0_table[r][c] is None

        # Table in snapshot 1 has computed exactly (1, 1), but (1, 2) is still None
        assert snap1_table[1][1] is not None
        assert snap1_table[1][2] is None

        # Snapshot 0 table remained untouched
        assert trace.get_snapshot(0).state["table"][1][1] is None

    def test_future_dp_values_never_leak_into_earlier_states(self):
        spec = KnapsackSpec(capacity=4, weights=(2, 3), values=(3, 4))
        res = KnapsackSimulator.run(spec)
        trace = res.computation_trace

        # Step 1 is computing cell (1, 1). dp_2_3 must not exist in step 1 state
        snap1 = trace.get_snapshot(1)
        assert "dp_2_3" not in snap1.state
        assert "dp_1_4" not in snap1.state
        assert snap1.state["table"][2][3] is None

    def test_backtracking_snapshots_chronologically_correct(self):
        spec = KnapsackSpec(capacity=4, weights=(2, 3), values=(3, 4))
        res = KnapsackSimulator.run(spec)
        trace = res.computation_trace

        prev_step_idx = -1
        bt_snaps = [s for s in trace.snapshots if s.state.get("computation_phase") == "backtracking"]
        assert len(bt_snaps) > 0

        # Verify step index strictly increases
        for s in trace.snapshots:
            assert s.step_index > prev_step_idx
            prev_step_idx = s.step_index

        # Verify backtracking begins at (N, W)
        assert bt_snaps[0].state["active_row"] == 2
        assert bt_snaps[0].state["active_col"] == 4


# ============================================================================
# 12. M2.3 Audit: Dijkstra Registry Collision & Safety Tests
# ============================================================================

class TestDijkstraRegistryCollisionAudit:
    """
    Verifies that DijkstraTraceVerifier selectively claims only Dijkstra problems,
    rejects other graph algorithms, non-graph domains, and prevents duplicate registration.
    """

    def test_unrelated_graph_algorithms_rejected(self):
        from src.grounding.verifiers.graph.dijkstra_verifier import DijkstraTraceVerifier

        # BFS
        bfs_spec = {"algorithm": "bfs", "inputs": {"vertices": ["A", "B"], "edges": [["A", "B", 1]], "source": "A"}}
        assert DijkstraTraceVerifier.can_verify(bfs_spec, topic="Breadth First Search") is False
        assert DijkstraTraceVerifier.can_handle(bfs_spec, topic="Breadth First Search") is False

        # DFS
        dfs_spec = {"algorithm": "dfs", "inputs": {"vertices": ["A", "B"], "edges": [["A", "B", 1]], "source": "A"}}
        assert DijkstraTraceVerifier.can_verify(dfs_spec, topic="Depth First Search") is False

        # Bellman-Ford
        bf_spec = {"algorithm": "bellman_ford", "inputs": {"vertices": ["A", "B"], "edges": [["A", "B", -1]], "source": "A"}}
        assert DijkstraTraceVerifier.can_verify(bf_spec, topic="Bellman-Ford Algorithm") is False

        # Kruskal MST
        kruskal_spec = {"algorithm": "kruskal", "inputs": {"vertices": ["A", "B"], "edges": [["A", "B", 1]]}}
        assert DijkstraTraceVerifier.can_verify(kruskal_spec, topic="Kruskal Minimum Spanning Tree") is False

        # Prim MST
        prim_spec = {"algorithm": "prim", "inputs": {"vertices": ["A", "B"], "edges": [["A", "B", 1]], "source": "A"}}
        assert DijkstraTraceVerifier.can_verify(prim_spec, topic="Prim's Algorithm") is False

        # Topological Sort
        topo_spec = {"algorithm": "topological_sort", "inputs": {"vertices": ["A", "B"], "edges": [["A", "B", 1]]}}
        assert DijkstraTraceVerifier.can_verify(topo_spec, topic="Topological Sort") is False

    def test_non_graph_domains_rejected_even_with_dijkstra_topic(self):
        from src.grounding.verifiers.graph.dijkstra_verifier import DijkstraTraceVerifier

        # Knapsack
        knapsack_spec = {"capacity": 5, "weights": [1, 2], "values": [3, 4]}
        assert DijkstraTraceVerifier.can_verify(knapsack_spec, topic="Dijkstra's Algorithm") is False

        # FCFS
        fcfs_spec = {"processes": [{"id": "P1", "arrival_time": 0, "burst_time": 3}]}
        assert DijkstraTraceVerifier.can_verify(fcfs_spec, topic="Dijkstra's Algorithm") is False

        # Binary Search
        bs_spec = {"array": [1, 3, 5, 7], "target": 5}
        assert DijkstraTraceVerifier.can_verify(bs_spec, topic="Dijkstra's Algorithm") is False

    def test_actual_dijkstra_specifications_routed_correctly(self):
        from src.grounding.verifiers.graph.dijkstra_verifier import DijkstraTraceVerifier

        dijkstra_spec = {
            "algorithm": "dijkstra",
            "inputs": {"vertices": ["A", "B", "C"], "edges": [["A", "B", 1], ["B", "C", 2]], "source": "A"},
        }
        assert DijkstraTraceVerifier.can_verify(dijkstra_spec, topic="Dijkstra's Shortest Path") is True
        assert DijkstraTraceVerifier.can_handle(dijkstra_spec, topic="Dijkstra's Shortest Path") is True

    def test_no_duplicate_registration_in_registry(self):
        from src.worked_example_verifier import SpecializedVerifierRegistry
        from src.grounding.verifiers.graph.dijkstra_verifier import DijkstraTraceVerifier

        count_before = sum(1 for v in SpecializedVerifierRegistry._verifiers if getattr(v, "verifier_id", None) == "dijkstra")
        # Attempt to register again
        SpecializedVerifierRegistry.register(DijkstraTraceVerifier)
        SpecializedVerifierRegistry.register(DijkstraTraceVerifier)
        count_after = sum(1 for v in SpecializedVerifierRegistry._verifiers if getattr(v, "verifier_id", None) == "dijkstra")
        assert count_after == max(1, count_before)

    def test_backward_compatibility_reexport_import(self):
        import src.dijkstra_verifier as dv
        assert hasattr(dv, "DijkstraTraceVerifier")
        assert hasattr(dv, "DijkstraTraceReport")
        assert hasattr(dv, "DijkstraStepState")
        assert dv.DijkstraTraceVerifier.verifier_id == "dijkstra"
