"""
Deterministic 0/1 Knapsack Dynamic Programming Simulator & Trace Verifier.

Part of CineLearn Milestone 2.3: Dynamic Programming & Graph Trace Grounding.
Implements deterministic recurrence evaluation, boundary validation, cell indexing,
excess-capacity handling, dependency tracking, step-by-step table snapshots,
deterministic backtracking, and independent fail-closed trace verification.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    from grounding.trace_engine import StateSnapshot, ComputationTrace
except ImportError:
    from src.grounding.trace_engine import StateSnapshot, ComputationTrace

from numerical_evaluator import (
    DiagnosticCode,
    VerificationStatus,
)
from worked_example_verifier import (
    SpecializedVerifier,
    SpecializedVerifierRegistry,
    WorkedExampleReport,
)


# ============================================================================
# Core Data Models
# ============================================================================

@dataclass(frozen=True)
class KnapsackSpec:
    """
    Canonical specification for a 0/1 Knapsack problem.
    Immutable, validated strictly upon construction.
    """
    capacity: int
    weights: Tuple[int, ...]
    values: Tuple[int, ...]
    item_names: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        # 1. Capacity validation: non-negative integer, reject booleans
        if not isinstance(self.capacity, int) or isinstance(self.capacity, bool):
            raise TypeError(
                f"Capacity must be an integer, got {type(self.capacity).__name__}"
            )
        if self.capacity < 0:
            raise ValueError(
                f"Capacity must be non-negative, got {self.capacity}"
            )

        # 2. Weights and Values type and length validation
        if not isinstance(self.weights, (tuple, list)):
            raise TypeError(
                f"Weights must be a sequence of integers, got {type(self.weights).__name__}"
            )
        if not isinstance(self.values, (tuple, list)):
            raise TypeError(
                f"Values must be a sequence of integers, got {type(self.values).__name__}"
            )

        if len(self.weights) != len(self.values):
            raise ValueError(
                f"Weights and values must have matching lengths: "
                f"got {len(self.weights)} weights and {len(self.values)} values"
            )

        # Force tuples to guarantee immutability
        cleaned_weights: List[int] = []
        for idx, w in enumerate(self.weights):
            if not isinstance(w, int) or isinstance(w, bool):
                raise TypeError(
                    f"Weight at index {idx} must be an integer, got {type(w).__name__}: {w}"
                )
            if w <= 0:
                raise ValueError(
                    f"Weight at index {idx} must be strictly positive (> 0), got {w}"
                )
            cleaned_weights.append(w)

        cleaned_values: List[int] = []
        for idx, v in enumerate(self.values):
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(
                    f"Value at index {idx} must be an integer, got {type(v).__name__}: {v}"
                )
            if v < 0:
                raise ValueError(
                    f"Value at index {idx} must be non-negative (>= 0), got {v}"
                )
            cleaned_values.append(v)

        # 3. Item names validation & contract for omitted names
        cleaned_names: List[str] = []
        if self.item_names:
            if not isinstance(self.item_names, (tuple, list)):
                raise TypeError(
                    f"Item names must be a sequence of strings, got {type(self.item_names).__name__}"
                )
            if len(self.item_names) != len(cleaned_weights):
                raise ValueError(
                    f"Item names length ({len(self.item_names)}) must match "
                    f"number of items ({len(cleaned_weights)})"
                )
            for idx, name in enumerate(self.item_names):
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"Item name at index {idx} must be a non-empty string"
                    )
                cleaned_names.append(name.strip())
        else:
            cleaned_names = [f"Item_{i + 1}" for i in range(len(cleaned_weights))]

        # Re-assign frozen fields
        object.__setattr__(self, "weights", tuple(cleaned_weights))
        object.__setattr__(self, "values", tuple(cleaned_values))
        object.__setattr__(self, "item_names", tuple(cleaned_names))

    @property
    def num_items(self) -> int:
        return len(self.weights)


@dataclass(frozen=True)
class KnapsackCellResult:
    """Structured computation outcome and dependencies for a single DP cell."""
    item_index: int       # 1-indexed item number i in [1, N]
    capacity: int         # Column capacity w in [0, W]
    item_name: str
    item_weight: int      # wt[i-1]
    item_value: int       # val[i-1]
    exclude_candidate: int
    include_candidate: Optional[int]  # None if w < wt[i-1]
    selected_branch: str  # "excess_capacity" | "exclude" | "include"
    final_value: int
    exclude_cell: Tuple[int, int]
    include_cell: Optional[Tuple[int, int]]
    selected_predecessor: Tuple[int, int]
    decision_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_index": self.item_index,
            "capacity": self.capacity,
            "item_name": self.item_name,
            "item_weight": self.item_weight,
            "item_value": self.item_value,
            "exclude_candidate": self.exclude_candidate,
            "include_candidate": self.include_candidate,
            "selected_branch": self.selected_branch,
            "final_value": self.final_value,
            "exclude_cell": list(self.exclude_cell),
            "include_cell": list(self.include_cell) if self.include_cell else None,
            "selected_predecessor": list(self.selected_predecessor),
            "decision_reason": self.decision_reason,
        }


@dataclass(frozen=True)
class KnapsackBacktrackStep:
    """Authoritative transition step during optimal item subset reconstruction."""
    step_index: int
    item_index: int       # 1-indexed item number i in [1, N]
    item_name: str
    item_weight: int
    item_value: int
    capacity_before: int
    capacity_after: int
    current_cell: Tuple[int, int]
    next_cell: Tuple[int, int]
    item_included: bool
    decision_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "item_index": self.item_index,
            "item_name": self.item_name,
            "item_weight": self.item_weight,
            "item_value": self.item_value,
            "capacity_before": self.capacity_before,
            "capacity_after": self.capacity_after,
            "current_cell": list(self.current_cell),
            "next_cell": list(self.next_cell),
            "item_included": self.item_included,
            "decision_reason": self.decision_reason,
        }


@dataclass
class KnapsackSimulationResult:
    """Authoritative result of deterministic 0/1 Knapsack simulation."""
    spec: KnapsackSpec
    table: List[List[int]]
    cell_results: Dict[Tuple[int, int], KnapsackCellResult]
    optimal_value: int
    selected_items: List[int]       # 0-indexed item indices
    selected_item_names: List[str]
    total_weight: int
    total_value: int
    backtrack_steps: List[KnapsackBacktrackStep]
    computation_trace: ComputationTrace


# ============================================================================
# Deterministic Knapsack Simulator
# ============================================================================

class KnapsackSimulator:
    """
    Deterministic 0/1 Knapsack Simulator.

    Guarantees:
    - Base case validation: dp[0][w] = 0 for all w in [0, W], and dp[i][0] = 0 for all i in [0, N].
    - Valid cell indexing: strictly enforces 0 <= i <= N and 0 <= w <= W.
    - Excess capacity handling: when w < wt[i-1], directly inherits dp[i-1][w]
      without evaluating the include branch or accessing invalid indices.
    - Deterministic tie-breaking: when include == exclude, exclusion is selected as a deterministic
      tie-breaking convention. Including the current item does not improve the objective.
    - Dependency tracking: provides exact predecessor cell coordinates and branch reasons.
    - Step-by-step temporal state snapshots: uncomputed cells are honestly represented
      as None (never misleadingly as 0).
    - Deterministic backtracking: reconstructs optimal subset and records transitions.
    - Emits standard ComputationTrace with deep-copy isolated StateSnapshots.
    """

    @classmethod
    def run(cls, spec: KnapsackSpec) -> KnapsackSimulationResult:
        if not isinstance(spec, KnapsackSpec):
            raise TypeError(f"Expected KnapsackSpec, got {type(spec).__name__}")

        N = spec.num_items
        W = spec.capacity

        # 1. Initialize table with base cases (0) and uncomputed cells (None)
        # Dimensions: (N + 1) rows x (W + 1) columns
        table: List[List[Optional[int]]] = [
            [None for _ in range(W + 1)] for _ in range(N + 1)
        ]

        # Base case row 0: dp[0][w] = 0
        for w in range(W + 1):
            table[0][w] = 0

        # Base case column 0: dp[i][0] = 0
        for i in range(N + 1):
            table[i][0] = 0

        trace = ComputationTrace(
            algorithm_or_domain="01_knapsack_dp",
            metadata={
                "capacity": W,
                "num_items": N,
                "weights": list(spec.weights),
                "values": list(spec.values),
                "item_names": list(spec.item_names),
                "tie_breaking_rule": "exclude_on_tie",
            },
        )

        step_counter = 0

        def _make_snapshot(
            active_row: Optional[int],
            active_col: Optional[int],
            branch: str,
            phase: str,
            description: str,
            exclude_cand: Optional[int] = None,
            include_cand: Optional[int] = None,
            final_val: Optional[int] = None,
            exclude_cell: Optional[Tuple[int, int]] = None,
            include_cell: Optional[Tuple[int, int]] = None,
            selected_pred: Optional[Tuple[int, int]] = None,
            decision_reason: str = "",
            backtrack_info: Optional[Dict[str, Any]] = None,
            selected_items: Optional[List[int]] = None,
            opt_val: Optional[int] = None,
        ) -> None:
            nonlocal step_counter
            # Build isolated state dictionary
            curr_item_idx = (active_row - 1) if (active_row is not None and active_row > 0) else None
            state_dict: Dict[str, Any] = {
                "table": copy.deepcopy(table),
                "num_items": N,
                "capacity": W,
                "active_row": active_row,
                "active_col": active_col,
                "active_cell": [active_row, active_col] if (active_row is not None and active_col is not None) else None,
                "current_item_index": curr_item_idx,
                "current_item_name": spec.item_names[curr_item_idx] if curr_item_idx is not None else None,
                "current_item_weight": spec.weights[curr_item_idx] if curr_item_idx is not None else None,
                "current_item_value": spec.values[curr_item_idx] if curr_item_idx is not None else None,
                "current_branch": branch,
                "exclude_candidate": exclude_cand,
                "include_candidate": include_cand,
                "final_value": final_val,
                "exclude_cell": list(exclude_cell) if exclude_cell else None,
                "include_cell": list(include_cell) if include_cell else None,
                "selected_predecessor": list(selected_pred) if selected_pred else None,
                "contributing_cells": [
                    list(c) for c in ([exclude_cell] if exclude_cell else []) + ([include_cell] if include_cell else [])
                ],
                "decision_reason": decision_reason,
                "computation_phase": phase,
                "backtracking_state": copy.deepcopy(backtrack_info) if backtrack_info else None,
                "selected_items": list(selected_items) if selected_items is not None else [],
                "optimal_value": opt_val,
            }

            # Flat variable bindings for expression evaluation
            for r in range(N + 1):
                for c in range(W + 1):
                    val = table[r][c]
                    if val is not None:
                        state_dict[f"dp_{r}_{c}"] = val
                        state_dict[f"cell_r{r}_c{c}"] = val

            if opt_val is not None:
                state_dict["optimal_value"] = opt_val

            snap = StateSnapshot(
                step_index=step_counter,
                state=state_dict,
                description=description,
                step_type=f"DP_{phase.upper()}",
                metadata={
                    "active_variables": [
                        "table", "active_row", "active_col", "current_branch", "final_value"
                    ],
                },
            )
            trace.add_snapshot(snap)
            step_counter += 1

        # Snapshot 0: Base-case initialization
        _make_snapshot(
            active_row=0,
            active_col=0,
            branch="base_case",
            phase="initialization",
            description=f"Base case initialization: dp[0][w] = 0 for all w in [0, {W}], and dp[i][0] = 0 for all i in [0, {N}].",
            final_val=0,
            decision_reason="Base case invariant: 0 items or 0 capacity yields 0 value.",
        )

        cell_results: Dict[Tuple[int, int], KnapsackCellResult] = {}

        # 2. Main DP recurrence loop: row-major order
        # i ranges from 1 to N (representing the first i items)
        for i in range(1, N + 1):
            item_idx = i - 1  # 0-indexed item attribute
            wt_i = spec.weights[item_idx]
            val_i = spec.values[item_idx]
            item_name = spec.item_names[item_idx]

            # w ranges from 1 to W (column 0 was already initialized to 0)
            for w in range(1, W + 1):
                exclude_cand = table[i - 1][w]
                assert exclude_cand is not None, f"Predecessor ({i-1}, {w}) must already be computed"
                exclude_cell = (i - 1, w)

                if w < wt_i:
                    # Excess capacity: item weight exceeds remaining capacity
                    final_val = exclude_cand
                    branch = "excess_capacity"
                    include_cand = None
                    include_cell = None
                    selected_pred = exclude_cell
                    reason = (
                        f"Item '{item_name}' weight ({wt_i}) exceeds capacity ({w}). "
                        f"Excess capacity: directly inherit value ({final_val}) from predecessor cell ({i-1}, {w})."
                    )
                else:
                    # Item can fit: evaluate both include and exclude branches
                    pred_w = w - wt_i
                    pred_val = table[i - 1][pred_w]
                    assert pred_val is not None, f"Predecessor ({i-1}, {pred_w}) must already be computed"

                    include_cand = pred_val + val_i
                    include_cell = (i - 1, pred_w)

                    if include_cand > exclude_cand:
                        final_val = include_cand
                        branch = "include"
                        selected_pred = include_cell
                        reason = (
                            f"Include candidate ({include_cand} = dp[{i-1}][{pred_w}] + {val_i}) > "
                            f"exclude candidate ({exclude_cand} = dp[{i-1}][{w}]). Selected include branch."
                        )
                    else:
                        # Tie-breaking rule: exclude_on_tie
                        final_val = exclude_cand
                        branch = "exclude"
                        selected_pred = exclude_cell
                        if include_cand == exclude_cand:
                            reason = (
                                f"Include candidate ({include_cand}) == exclude candidate ({exclude_cand}). "
                                f"When inclusion and exclusion produce equal objective values, exclusion is "
                                f"selected as a deterministic tie-breaking convention. Including the current "
                                f"item does not improve the objective."
                            )
                        else:
                            reason = (
                                f"Exclude candidate ({exclude_cand}) > include candidate ({include_cand}). "
                                f"Selected exclude branch."
                            )

                table[i][w] = final_val

                cell_res = KnapsackCellResult(
                    item_index=i,
                    capacity=w,
                    item_name=item_name,
                    item_weight=wt_i,
                    item_value=val_i,
                    exclude_candidate=exclude_cand,
                    include_candidate=include_cand,
                    selected_branch=branch,
                    final_value=final_val,
                    exclude_cell=exclude_cell,
                    include_cell=include_cell,
                    selected_predecessor=selected_pred,
                    decision_reason=reason,
                )
                cell_results[(i, w)] = cell_res

                _make_snapshot(
                    active_row=i,
                    active_col=w,
                    branch=branch,
                    phase="table_filling",
                    description=f"Computed cell ({i}, {w}) for '{item_name}': value = {final_val} via {branch}.",
                    exclude_cand=exclude_cand,
                    include_cand=include_cand,
                    final_val=final_val,
                    exclude_cell=exclude_cell,
                    include_cell=include_cell,
                    selected_pred=selected_pred,
                    decision_reason=reason,
                )

        optimal_value = int(table[N][W] if table[N][W] is not None else 0)

        # Snapshot: Optimal value reached
        _make_snapshot(
            active_row=N,
            active_col=W,
            branch="optimal_reached",
            phase="optimal_reached",
            description=f"Table filling complete. Authoritative optimal value at dp[{N}][{W}] = {optimal_value}.",
            final_val=optimal_value,
            decision_reason="All matrix cells computed. Beginning optimal subset reconstruction via backtracking.",
            opt_val=optimal_value,
        )

        # 3. Deterministic Backtracking & Optimal Subset Reconstruction
        # Starts at (N, W) and steps backward to (0, 0)
        curr_i = N
        curr_w = W
        selected_item_indices: List[int] = []
        backtrack_steps: List[KnapsackBacktrackStep] = []
        bt_step_idx = 0

        while curr_i > 0 and curr_w >= 0:
            item_idx = curr_i - 1
            wt_i = spec.weights[item_idx]
            val_i = spec.values[item_idx]
            item_name = spec.item_names[item_idx]
            current_cell = (curr_i, curr_w)

            # Check if item was included
            if curr_w < wt_i:
                # Could not have fit
                item_included = False
                next_cell = (curr_i - 1, curr_w)
                next_w = curr_w
                bt_reason = f"At cell ({curr_i}, {curr_w}): remaining capacity {curr_w} < weight {wt_i}. Item '{item_name}' was excluded."
            else:
                # Tied or unequal: tie-break rule preferred exclusion when equal
                val_exclude = table[curr_i - 1][curr_w]
                if table[curr_i][curr_w] == val_exclude:
                    # Item was excluded (or tied and excluded)
                    item_included = False
                    next_cell = (curr_i - 1, curr_w)
                    next_w = curr_w
                    bt_reason = (
                        f"At cell ({curr_i}, {curr_w}): dp[{curr_i}][{curr_w}] == dp[{curr_i-1}][{curr_w}] ({val_exclude}). "
                        f"Item '{item_name}' was excluded (deterministic tie-breaking convention: including does not improve objective)."
                    )
                else:
                    # Item was included
                    item_included = True
                    next_w = curr_w - wt_i
                    next_cell = (curr_i - 1, next_w)
                    selected_item_indices.append(item_idx)
                    bt_reason = f"At cell ({curr_i}, {curr_w}): value reflects inclusion of item '{item_name}' (weight={wt_i}, value={val_i}). Next capacity = {next_w}."

            step_obj = KnapsackBacktrackStep(
                step_index=bt_step_idx,
                item_index=curr_i,
                item_name=item_name,
                item_weight=wt_i,
                item_value=val_i,
                capacity_before=curr_w,
                capacity_after=next_w,
                current_cell=current_cell,
                next_cell=next_cell,
                item_included=item_included,
                decision_reason=bt_reason,
            )
            backtrack_steps.append(step_obj)
            bt_step_idx += 1

            _make_snapshot(
                active_row=curr_i,
                active_col=curr_w,
                branch="backtrack",
                phase="backtracking",
                description=bt_reason,
                final_val=table[curr_i][curr_w],
                selected_pred=next_cell,
                decision_reason=bt_reason,
                backtrack_info=step_obj.to_dict(),
                selected_items=list(reversed(selected_item_indices)),
                opt_val=optimal_value,
            )

            curr_i = next_cell[0]
            curr_w = next_cell[1]

        # Selected items are collected in reverse order (from N down to 1), restore chronological order
        selected_item_indices.reverse()
        selected_names = [spec.item_names[idx] for idx in selected_item_indices]
        tot_weight = sum(spec.weights[idx] for idx in selected_item_indices)
        tot_value = sum(spec.values[idx] for idx in selected_item_indices)

        # Invariant validations on reconstruction
        if tot_weight > W:
            raise RuntimeError(
                f"Reconstructed subset weight ({tot_weight}) exceeds knapsack capacity ({W})"
            )
        if tot_value != optimal_value:
            raise RuntimeError(
                f"Reconstructed subset value ({tot_value}) does not match optimal DP value ({optimal_value})"
            )

        # Final completion snapshot
        _make_snapshot(
            active_row=None,
            active_col=None,
            branch="completed",
            phase="completed",
            description=(
                f"Knapsack computation and backtracking complete. "
                f"Optimal value: {optimal_value}, Total weight: {tot_weight}/{W}, "
                f"Selected items: {selected_names}."
            ),
            final_val=optimal_value,
            decision_reason="Reconstruction verified against capacity and optimal value invariants.",
            selected_items=selected_item_indices,
            opt_val=optimal_value,
        )

        completed_table = [[int(cell if cell is not None else 0) for cell in row] for row in table]

        return KnapsackSimulationResult(
            spec=spec,
            table=completed_table,
            cell_results=cell_results,
            optimal_value=optimal_value,
            selected_items=selected_item_indices,
            selected_item_names=selected_names,
            total_weight=tot_weight,
            total_value=tot_value,
            backtrack_steps=backtrack_steps,
            computation_trace=trace,
        )


# ============================================================================
# Specialized Verifier Registry Integration
# ============================================================================

class DPTraceVerifier(SpecializedVerifier):
    """
    Independent Specialized Verifier for 0/1 Knapsack Dynamic Programming outlines.
    Registers cleanly with SpecializedVerifierRegistry.
    """

    verifier_id: str = "dp_01_knapsack"

    @classmethod
    def can_handle(cls, spec: Any, topic: str = "") -> bool:
        """Alias for can_verify to ensure full contract compatibility."""
        return cls.can_verify(spec, topic)

    @classmethod
    def can_verify(cls, spec: Dict[str, Any], topic: str = "") -> bool:
        if not isinstance(spec, dict):
            return False

        # Reject explicit conflicting algorithms
        algo = str(spec.get("algorithm", "")).strip().lower()
        if algo in ["fcfs", "dijkstra", "binary_search", "bfs", "dfs", "bellman_ford"]:
            return False

        topic_clean = topic.strip().lower()
        if any(kw in topic_clean for kw in [
            "knapsack",
            "0/1 knapsack",
            "knapsack problem",
            "0-1 knapsack",
        ]):
            return True

        if algo in ["knapsack", "knapsack_01", "01_knapsack", "dp_knapsack"]:
            return True

        domain = str(spec.get("domain", "")).strip().lower()
        if domain in ["dynamic_programming", "dp_knapsack"]:
            return True

        setup = str(spec.get("setup", "")).lower()
        if "knapsack" in setup:
            return True

        inputs = spec.get("inputs")
        if isinstance(inputs, dict):
            inp_algo = str(inputs.get("algorithm", "")).strip().lower()
            if inp_algo in ["knapsack", "knapsack_01", "01_knapsack", "dp_knapsack"]:
                return True
            if "weights" in inputs and "values" in inputs and "capacity" in inputs:
                return True

        return False

    @classmethod
    def parse_spec(cls, spec: Dict[str, Any]) -> KnapsackSpec:
        """Extracts and constructs validated KnapsackSpec from outline spec."""
        inputs = spec.get("inputs")
        src: Dict[str, Any] = inputs if isinstance(inputs, dict) else spec

        cap = src.get("capacity")
        if cap is None and "W" in src:
            cap = src.get("W")
        if cap is None:
            raise ValueError("Knapsack specification missing required 'capacity' field")

        wts = src.get("weights")
        if wts is None and "wt" in src:
            wts = src.get("wt")
        if wts is None:
            raise ValueError("Knapsack specification missing required 'weights' field")

        vals = src.get("values")
        if vals is None and "val" in src:
            vals = src.get("val")
        if vals is None:
            raise ValueError("Knapsack specification missing required 'values' field")

        names = src.get("item_names") or src.get("names") or src.get("items") or ()

        return KnapsackSpec(
            capacity=cap,
            weights=tuple(wts),
            values=tuple(vals),
            item_names=tuple(names) if names else (),
        )

    @classmethod
    def verify_trace(
        cls,
        spec: KnapsackSpec,
        trace_or_snapshots: Union[ComputationTrace, Sequence[Any]],
    ) -> Dict[str, Any]:
        """
        Independently audits an algorithmic trace against 0/1 Knapsack recurrence invariants,
        monotonicity constraints, capacity bounds, and backtracking transitions.
        Fails closed on missing evidence, corrupted values, branch mismatches, or invariant violations.
        """
        if not isinstance(spec, KnapsackSpec):
            raise TypeError(f"Expected KnapsackSpec, got {type(spec).__name__}")

        if isinstance(trace_or_snapshots, ComputationTrace):
            snapshots = trace_or_snapshots.snapshots
        elif isinstance(trace_or_snapshots, (list, tuple)):
            snapshots = list(trace_or_snapshots)
        else:
            raise TypeError("Expected ComputationTrace or sequence of snapshots")

        errors: List[str] = []
        N = spec.num_items
        W = spec.capacity

        if not snapshots:
            return {
                "verified": False,
                "status": VerificationStatus.FAILED,
                "errors": ["Trace contains zero snapshots; missing evidence."],
            }

        # 1. Verify snapshot chronology
        prev_idx = -1
        for s_idx, snap in enumerate(snapshots):
            idx = snap.step_index if isinstance(snap, StateSnapshot) else snap.get("step_index")
            if idx is None or not isinstance(idx, int) or idx <= prev_idx:
                errors.append(f"Chronology violation at snapshot {s_idx}: step_index {idx} must be strictly > {prev_idx}")
            prev_idx = idx if idx is not None else s_idx

        # 2. Verify initialization snapshot (Snapshot 0)
        snap0 = snapshots[0]
        st0 = snap0.state if isinstance(snap0, StateSnapshot) else snap0.get("state", {})
        tbl0 = st0.get("table")
        if not isinstance(tbl0, list) or len(tbl0) != N + 1:
            errors.append(f"Initialization snapshot missing valid (N+1) table: expected {N+1} rows")
        else:
            # Check row 0 is all 0
            if any(cell != 0 for cell in tbl0[0]):
                errors.append("Base case violation in snapshot 0: row 0 must contain only zeros")
            # Check column 0 is all 0
            for r in range(N + 1):
                if tbl0[r][0] != 0:
                    errors.append(f"Base case violation in snapshot 0: cell ({r}, 0) must be 0")
            # Check uncomputed cells are None
            for r in range(1, N + 1):
                for c in range(1, W + 1):
                    if tbl0[r][c] is not None:
                        errors.append(f"Future state leak in snapshot 0: cell ({r}, {c}) is {tbl0[r][c]}, must be None")

        # 3. Independent recurrence verification across table
        verif_table: List[List[Optional[int]]] = [[None for _ in range(W + 1)] for _ in range(N + 1)]
        for w in range(W + 1):
            verif_table[0][w] = 0
        for i in range(N + 1):
            verif_table[i][0] = 0

        # Scan table filling snapshots
        table_snaps = [
            s for s in snapshots
            if (s.state if isinstance(s, StateSnapshot) else s.get("state", {})).get("computation_phase") == "table_filling"
        ]
        for snap in table_snaps:
            st = snap.state if isinstance(snap, StateSnapshot) else snap.get("state", {})
            r = st.get("active_row")
            c = st.get("active_col")
            if r is None or c is None:
                errors.append("Table filling snapshot missing active_row or active_col")
                continue
            if r < 1 or r > N or c < 1 or c > W:
                errors.append(f"Snapshot active cell ({r}, {c}) out of bounds")
                continue

            wt_i = spec.weights[r - 1]
            val_i = spec.values[r - 1]

            # Predecessor verification
            exp_exc = verif_table[r - 1][c]
            if exp_exc is None:
                errors.append(f"Dependency violation: predecessor cell ({r-1}, {c}) not yet computed when computing ({r}, {c})")
                exp_exc = 0

            claimed_exc = st.get("exclude_candidate")
            if claimed_exc is not None and claimed_exc != exp_exc:
                errors.append(f"Cell ({r}, {c}) exclude candidate mismatch: claimed {claimed_exc}, expected {exp_exc}")

            if c < wt_i:
                # Excess capacity
                exp_branch = "excess_capacity"
                exp_val = exp_exc
                exp_pred = [r - 1, c]
                if st.get("include_candidate") is not None:
                    errors.append(f"Cell ({r}, {c}) excess capacity violation: include_candidate must be None")
            else:
                pred_c = c - wt_i
                pred_val = verif_table[r - 1][pred_c]
                if pred_val is None:
                    errors.append(f"Dependency violation: predecessor cell ({r-1}, {pred_c}) not yet computed")
                    pred_val = 0
                exp_inc = pred_val + val_i
                claimed_inc = st.get("include_candidate")
                if claimed_inc is not None and claimed_inc != exp_inc:
                    errors.append(f"Cell ({r}, {c}) include candidate mismatch: claimed {claimed_inc}, expected {exp_inc}")

                if exp_inc > exp_exc:
                    exp_branch = "include"
                    exp_val = exp_inc
                    exp_pred = [r - 1, pred_c]
                else:
                    exp_branch = "exclude"
                    exp_val = exp_exc
                    exp_pred = [r - 1, c]

            claimed_branch = st.get("current_branch")
            if claimed_branch and str(claimed_branch).strip().lower() != exp_branch:
                errors.append(f"Cell ({r}, {c}) branch mismatch: claimed '{claimed_branch}', expected '{exp_branch}'")

            claimed_val = st.get("final_value")
            if claimed_val is not None and claimed_val != exp_val:
                errors.append(f"Cell ({r}, {c}) final value mismatch: claimed {claimed_val}, expected {exp_val}")

            # Check snapshot table cell
            snap_table = st.get("table")
            if isinstance(snap_table, list) and len(snap_table) > r and len(snap_table[r]) > c:
                if snap_table[r][c] != exp_val:
                    errors.append(f"Snapshot table cell ({r}, {c}) has {snap_table[r][c]}, expected {exp_val}")

            # Monotonicity check
            if exp_val < exp_exc:
                errors.append(f"Monotonicity violation: cell ({r}, {c}) value {exp_val} < cell ({r-1}, {c}) value {exp_exc}")
            if c > 1 and verif_table[r][c - 1] is not None and exp_val < verif_table[r][c - 1]:
                errors.append(f"Monotonicity violation: cell ({r}, {c}) value {exp_val} < cell ({r}, {c-1}) value {verif_table[r][c-1]}")

            verif_table[r][c] = exp_val

        # 4. Final optimum verification
        authoritative_opt = verif_table[N][W] if (N >= 0 and W >= 0 and verif_table[N][W] is not None) else 0

        # 5. Backtracking verification
        bt_snaps = [
            s for s in snapshots
            if (s.state if isinstance(s, StateSnapshot) else s.get("state", {})).get("computation_phase") == "backtracking"
        ]
        curr_i = N
        curr_w = W
        reconstructed_items: List[int] = []

        for bt_snap in bt_snaps:
            st = bt_snap.state if isinstance(bt_snap, StateSnapshot) else bt_snap.get("state", {})
            r = st.get("active_row")
            c = st.get("active_col")
            bt_info = st.get("backtracking_state") or {}
            if r != curr_i or c != curr_w:
                errors.append(f"Backtracking chronology error: expected cell ({curr_i}, {curr_w}), got ({r}, {c})")

            if curr_i > 0:
                wt_i = spec.weights[curr_i - 1]
                val_i = spec.values[curr_i - 1]
                if curr_w < wt_i:
                    exp_inc = False
                    next_i = curr_i - 1
                    next_w = curr_w
                else:
                    if verif_table[curr_i][curr_w] == verif_table[curr_i - 1][curr_w]:
                        exp_inc = False
                        next_i = curr_i - 1
                        next_w = curr_w
                    else:
                        exp_inc = True
                        next_i = curr_i - 1
                        next_w = curr_w - wt_i
                        reconstructed_items.append(curr_i - 1)

                claimed_inc = bt_info.get("item_included")
                if claimed_inc is not None and bool(claimed_inc) != exp_inc:
                    errors.append(f"Backtracking at ({curr_i}, {curr_w}) decision mismatch: claimed included={claimed_inc}, expected {exp_inc}")

                curr_i = next_i
                curr_w = next_w

        # 6. Reconstructed subset constraints
        reconstructed_items.reverse()
        tot_wt = sum(spec.weights[idx] for idx in reconstructed_items)
        tot_val = sum(spec.values[idx] for idx in reconstructed_items)

        if tot_wt > W:
            errors.append(f"Capacity constraint violation: total weight {tot_wt} exceeds capacity {W}")
        if tot_val != authoritative_opt:
            errors.append(f"Optimality constraint violation: total value {tot_val} != optimal DP value {authoritative_opt}")

        is_verified = (len(errors) == 0)
        return {
            "verified": is_verified,
            "status": VerificationStatus.VERIFIED if is_verified else VerificationStatus.FAILED,
            "errors": errors,
            "optimal_value": authoritative_opt,
            "selected_items": reconstructed_items,
            "total_weight": tot_wt,
            "total_value": tot_val,
        }

    @classmethod
    def verify_worked_example_spec(
        cls,
        spec: Dict[str, Any],
        topic: str = "",
    ) -> Tuple[Dict[str, Any], WorkedExampleReport]:
        """
        Public verifier entry point called by WorkedExampleVerifier.
        Executes deterministic DP simulation, audits claimed outline steps,
        cell values, branch choices, and reconstruction, and constructs WorkedExampleReport.
        """
        spec_copy = dict(spec)
        raw_steps = spec_copy.get("steps", [])

        # 1. Parse and validate input specification (fail-closed)
        try:
            knapsack_spec = cls.parse_spec(spec_copy)
        except Exception as e:
            err_rep = WorkedExampleReport(
                overall_status=VerificationStatus.FAILED,
                total_steps=len(raw_steps),
                verified_steps=0,
                corrected_steps=0,
                failed_steps=max(1, len(raw_steps)),
                review_steps=0,
                not_applicable_steps=0,
                final_result_status=VerificationStatus.FAILED,
                summary=f"0/1 Knapsack input specification validation failed: {e}",
                is_safe_for_stage2=False,
                verification_method="dp_trace_verifier",
                step_diagnostics=[{
                    "step_index": 0,
                    "status": VerificationStatus.FAILED.value,
                    "diagnostic": DiagnosticCode.INVALID_INPUT_VALUES.value,
                    "details": str(e),
                }],
            )
            spec_copy["verification_metadata"] = err_rep.to_dict()
            return spec_copy, err_rep

        # 2. Run authoritative deterministic simulation
        sim_result = KnapsackSimulator.run(knapsack_spec)
        trace = sim_result.computation_trace
        snapshots = trace.snapshots

        # Build snapshot index map for fast step lookup
        cell_table_snap_idx: Dict[Tuple[int, int], int] = {}
        cell_backtrack_snap_idx: Dict[Tuple[int, int], int] = {}
        for idx, snap in enumerate(snapshots):
            ac = snap.state.get("active_cell")
            phase = snap.state.get("computation_phase")
            if ac and isinstance(ac, (list, tuple)) and len(ac) == 2:
                coord = (ac[0], ac[1])
                if phase == "table_filling":
                    cell_table_snap_idx[coord] = idx
                elif phase == "backtracking":
                    cell_backtrack_snap_idx[coord] = idx

        # 3. Audit outline steps against authoritative snapshots
        verified_steps_output: List[Dict[str, Any]] = []
        step_diagnostics: List[Dict[str, Any]] = []
        verified_count = 0
        failed_count = 0

        for i, step_data in enumerate(raw_steps):
            step_out = dict(step_data) if isinstance(step_data, dict) else {"step": str(step_data)}

            # Determine claimed coordinates and attributes
            claimed_row = step_out.get("row")
            if claimed_row is None:
                claimed_row = step_out.get("item_index")
            if claimed_row is None:
                claimed_row = step_out.get("i")

            claimed_col = step_out.get("col")
            if claimed_col is None:
                claimed_col = step_out.get("capacity")
            if claimed_col is None:
                claimed_col = step_out.get("w")

            claimed_val = step_out.get("value")
            if claimed_val is None:
                claimed_val = step_out.get("claimed_value")

            claimed_branch = step_out.get("branch")
            if claimed_branch is None:
                claimed_branch = step_out.get("selected_branch")

            claimed_phase = str(step_out.get("phase") or step_out.get("computation_phase", "")).strip().lower()

            step_failed = False
            fail_reasons: List[str] = []

            # Coordinate bounds check
            if claimed_row is not None:
                try:
                    r_int = int(claimed_row)
                    if r_int < 0 or r_int > knapsack_spec.num_items:
                        step_failed = True
                        fail_reasons.append(
                            f"Row index out of bounds: {r_int} (valid range: 0 to {knapsack_spec.num_items})"
                        )
                except (ValueError, TypeError):
                    step_failed = True
                    fail_reasons.append(f"Invalid row index: {claimed_row}")

            if claimed_col is not None:
                try:
                    c_int = int(claimed_col)
                    if c_int < 0 or c_int > knapsack_spec.capacity:
                        step_failed = True
                        fail_reasons.append(
                            f"Capacity column out of bounds: {c_int} (valid range: 0 to {knapsack_spec.capacity})"
                        )
                except (ValueError, TypeError):
                    step_failed = True
                    fail_reasons.append(f"Invalid column index: {claimed_col}")

            # Base case verification: row 0 or col 0 must have value 0
            if claimed_row is not None and int(claimed_row) == 0:
                if claimed_val is not None and int(claimed_val) != 0:
                    step_failed = True
                    fail_reasons.append(f"Base case violation: row 0 must have value 0, got {claimed_val}")
                if claimed_branch is not None and str(claimed_branch).strip().lower() == "include":
                    step_failed = True
                    fail_reasons.append("Base case violation: row 0 cannot select 'include' branch")

            if claimed_col is not None and int(claimed_col) == 0:
                if claimed_val is not None and int(claimed_val) != 0:
                    step_failed = True
                    fail_reasons.append(f"Base case violation: column 0 must have value 0, got {claimed_val}")
                if claimed_branch is not None and str(claimed_branch).strip().lower() == "include":
                    step_failed = True
                    fail_reasons.append("Base case violation: column 0 cannot select 'include' branch")

            snap: Optional[StateSnapshot] = None
            if claimed_row is not None and claimed_col is not None:
                try:
                    coord = (int(claimed_row), int(claimed_col))
                    if claimed_phase == "backtracking" or (claimed_branch and str(claimed_branch).strip().lower() == "backtrack"):
                        snap = snapshots[cell_backtrack_snap_idx[coord]] if coord in cell_backtrack_snap_idx else None
                    elif coord in cell_table_snap_idx:
                        snap = snapshots[cell_table_snap_idx[coord]]
                    elif coord in cell_backtrack_snap_idx:
                        snap = snapshots[cell_backtrack_snap_idx[coord]]
                    elif int(claimed_row) == 0 or int(claimed_col) == 0:
                        snap = snapshots[0]
                except (ValueError, TypeError):
                    pass
            elif i < len(snapshots):
                snap = snapshots[i]

            if snap is None and not step_failed:
                step_failed = True
                fail_reasons.append(
                    f"No authoritative state snapshot found matching step coordinates ({claimed_row}, {claimed_col}) "
                    f"or step index {i}"
                )

            if snap is not None:
                exp_row = snap.state.get("active_row")
                exp_col = snap.state.get("active_col")
                exp_val = snap.state.get("final_value")
                exp_branch = snap.state.get("current_branch")

                # Check cell value claim
                if claimed_val is not None:
                    try:
                        if int(claimed_val) != exp_val:
                            step_failed = True
                            fail_reasons.append(
                                f"Cell value mismatch: claimed {claimed_val}, expected {exp_val}"
                            )
                    except (ValueError, TypeError):
                        step_failed = True
                        fail_reasons.append(f"Invalid claimed value: {claimed_val}")

                # Check branch claim
                if claimed_branch is not None:
                    clean_branch = str(claimed_branch).strip().lower()
                    if clean_branch != str(exp_branch).strip().lower():
                        step_failed = True
                        fail_reasons.append(
                            f"Recurrence branch mismatch: claimed '{claimed_branch}', expected '{exp_branch}'"
                        )

                # Check candidate values if claimed
                claimed_inc = step_out.get("include_candidate") or step_out.get("include_cand")
                if claimed_inc is not None:
                    exp_inc = snap.state.get("include_candidate")
                    if exp_inc is None:
                        step_failed = True
                        fail_reasons.append(
                            f"Excess capacity violation: claimed include candidate {claimed_inc}, "
                            f"but item does not fit remaining capacity (expected None)"
                        )
                    elif int(claimed_inc) != exp_inc:
                        step_failed = True
                        fail_reasons.append(
                            f"Include candidate mismatch: claimed {claimed_inc}, expected {exp_inc}"
                        )

                claimed_exc = step_out.get("exclude_candidate") or step_out.get("exclude_cand")
                if claimed_exc is not None:
                    exp_exc = snap.state.get("exclude_candidate")
                    if exp_exc is not None and int(claimed_exc) != exp_exc:
                        step_failed = True
                        fail_reasons.append(
                            f"Exclude candidate mismatch: claimed {claimed_exc}, expected {exp_exc}"
                        )

                # Check cell coordinate claims
                if claimed_row is not None and exp_row is not None and int(claimed_row) != exp_row:
                    step_failed = True
                    fail_reasons.append(
                        f"Row index mismatch: claimed {claimed_row}, expected {exp_row}"
                    )
                if claimed_col is not None and exp_col is not None and int(claimed_col) != exp_col:
                    step_failed = True
                    fail_reasons.append(
                        f"Capacity column mismatch: claimed {claimed_col}, expected {exp_col}"
                    )

                # Check predecessor coordinate claim if provided
                claimed_pred = step_out.get("selected_predecessor") or step_out.get("predecessor")
                if claimed_pred is not None and isinstance(claimed_pred, (list, tuple)) and len(claimed_pred) == 2:
                    exp_pred = snap.state.get("selected_predecessor")
                    if exp_pred and list(claimed_pred) != exp_pred:
                        step_failed = True
                        fail_reasons.append(
                            f"Predecessor mismatch: claimed {list(claimed_pred)}, expected {exp_pred}"
                        )

                # Backtracking specific claims
                claimed_item_inc = step_out.get("item_included")
                if claimed_item_inc is not None:
                    bt_state = snap.state.get("backtracking_state")
                    if bt_state and "item_included" in bt_state:
                        if bool(claimed_item_inc) != bool(bt_state["item_included"]):
                            step_failed = True
                            fail_reasons.append(
                                f"Backtracking decision mismatch: claimed included={claimed_item_inc}, "
                                f"expected {bt_state['item_included']}"
                            )

                claimed_next_w = step_out.get("capacity_after") or step_out.get("next_w")
                if claimed_next_w is not None:
                    bt_state = snap.state.get("backtracking_state")
                    if bt_state and "capacity_after" in bt_state:
                        if int(claimed_next_w) != bt_state["capacity_after"]:
                            step_failed = True
                            fail_reasons.append(
                                f"Backtracking capacity mismatch: claimed next_w={claimed_next_w}, "
                                f"expected {bt_state['capacity_after']}"
                            )

            if step_failed:
                failed_count += 1
                diag = {
                    "step_index": i,
                    "status": VerificationStatus.FAILED.value,
                    "diagnostic": DiagnosticCode.ALGORITHMIC_STATE_MISMATCH.value,
                    "details": "; ".join(fail_reasons),
                }
                step_out["verification_status"] = VerificationStatus.FAILED.value
                step_out["diagnostics"] = diag
            else:
                verified_count += 1
                diag = {
                    "step_index": i,
                    "status": VerificationStatus.VERIFIED.value,
                    "diagnostic": DiagnosticCode.ALGORITHMIC_VERIFIED.value,
                    "details": "Step claims match authoritative 0/1 Knapsack DP state",
                }
                step_out["verification_status"] = VerificationStatus.VERIFIED.value
                step_out["diagnostics"] = diag

            if snap is not None:
                step_out["authoritative_state"] = copy.deepcopy(snap.state)

            verified_steps_output.append(step_out)
            step_diagnostics.append(diag)

        spec_copy["steps"] = verified_steps_output

        # 4. Final Result & Subset Invariant Audit
        claimed_final = spec_copy.get("final_result")
        canonical_final_str = (
            f"Optimal Value = {sim_result.optimal_value}, "
            f"Selected Items = {sim_result.selected_item_names}"
        )

        final_result_status = VerificationStatus.VERIFIED
        if claimed_final:
            nums = [int(x) for x in re.findall(r"\b\d+\b", str(claimed_final))]
            if nums:
                if sim_result.optimal_value not in nums:
                    final_result_status = VerificationStatus.FAILED
            else:
                final_result_status = VerificationStatus.FAILED
        else:
            spec_copy["final_result"] = canonical_final_str

        # Check spec-level claimed selected items if provided
        claimed_items = spec_copy.get("selected_items")
        if claimed_items is not None:
            if list(claimed_items) != sim_result.selected_items:
                failed_count += 1
                final_result_status = VerificationStatus.FAILED
                step_diagnostics.append({
                    "step_index": len(raw_steps),
                    "status": VerificationStatus.FAILED.value,
                    "diagnostic": DiagnosticCode.ALGORITHMIC_STATE_MISMATCH.value,
                    "details": f"Claimed selected items {claimed_items} != expected {sim_result.selected_items}",
                })

        claimed_item_names = spec_copy.get("selected_item_names")
        if claimed_item_names is not None:
            if list(claimed_item_names) != sim_result.selected_item_names:
                failed_count += 1
                final_result_status = VerificationStatus.FAILED
                step_diagnostics.append({
                    "step_index": len(raw_steps),
                    "status": VerificationStatus.FAILED.value,
                    "diagnostic": DiagnosticCode.ALGORITHMIC_STATE_MISMATCH.value,
                    "details": f"Claimed selected item names {claimed_item_names} != expected {sim_result.selected_item_names}",
                })

        claimed_tot_wt = spec_copy.get("total_weight")
        if claimed_tot_wt is not None:
            if int(claimed_tot_wt) != sim_result.total_weight:
                failed_count += 1
                final_result_status = VerificationStatus.FAILED

        # 5. Algorithmic State Payload
        algo_state = {
            "algorithm": "knapsack_01",
            "capacity": knapsack_spec.capacity,
            "num_items": knapsack_spec.num_items,
            "weights": list(knapsack_spec.weights),
            "values": list(knapsack_spec.values),
            "item_names": list(knapsack_spec.item_names),
            "dp_table": sim_result.table,
            "optimal_value": sim_result.optimal_value,
            "selected_items": sim_result.selected_items,
            "selected_item_names": sim_result.selected_item_names,
            "total_weight": sim_result.total_weight,
            "total_value": sim_result.total_value,
            "backtrack_steps": [s.to_dict() for s in sim_result.backtrack_steps],
            "tie_breaking_rule": "exclude_on_tie",
        }

        overall_status = (
            VerificationStatus.FAILED
            if (failed_count > 0 or final_result_status == VerificationStatus.FAILED)
            else VerificationStatus.VERIFIED
        )

        report = WorkedExampleReport(
            overall_status=overall_status,
            total_steps=len(raw_steps),
            verified_steps=verified_count,
            corrected_steps=0,
            failed_steps=failed_count,
            review_steps=0,
            not_applicable_steps=0,
            step_diagnostics=step_diagnostics,
            final_result_status=final_result_status,
            summary=(
                f"0/1 Knapsack DP verification {'PASSED' if overall_status == VerificationStatus.VERIFIED else 'FAILED'}: "
                f"{verified_count}/{len(raw_steps)} steps verified, Optimal Value={sim_result.optimal_value}, "
                f"Selected Items={sim_result.selected_item_names}"
            ),
            is_safe_for_stage2=(overall_status == VerificationStatus.VERIFIED),
            verified_algorithmic_state=algo_state,
            verification_method="dp_trace_verifier",
            state_snapshots={s.step_index: copy.deepcopy(s.state) for s in snapshots},
        )

        spec_copy["verification_metadata"] = report.to_dict()
        return spec_copy, report

        spec_copy["verification_metadata"] = report.to_dict()
        return spec_copy, report


# ============================================================================
# Automatic Module-Level Registration
# ============================================================================
SpecializedVerifierRegistry.register(DPTraceVerifier)
