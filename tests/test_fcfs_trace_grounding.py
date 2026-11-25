"""
Comprehensive Deterministic Unit & Integration Tests for Milestone 2.2:
Operating Systems Deep Specialization — FCFS CPU Scheduling Simulation & Verification.

Tests:
1. ProcessSpec immutability, validation, and fail-closed edge case rejection.
2. FCFSSimulator discrete-event loop:
   - Sequential execution.
   - Staggered arrivals.
   - Simultaneous arrivals with stable input-order tie-breaking.
   - CPU idle periods (initial and interleaved).
   - Process arrivals during active execution.
3. Honest metric accounting (None until termination, averages None until final snapshot).
4. Snapshot isolation, monotonicity, and schema compliance.
5. FCFSTraceVerifier outline verification, premature metric detection, and error diagnostics.
6. SpecializedVerifierRegistry non-invasive integration without monolithic branching.
7. Storyboard fidelity step-aware checking with FCFS state snapshots.
"""

from __future__ import annotations

import copy
import pytest
from typing import Any, Dict, List

from fcfs_verifier import (
    ProcessSpec,
    GanttBlock,
    ProcessMetrics,
    ProcessState,
    FCFSSimulator,
    FCFSSimulationResult,
    FCFSTraceVerifier,
    validate_process_list,
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
    FidelityCategory,
)


# ============================================================================
# 1. ProcessSpec and Model Validation Tests
# ============================================================================

class TestFCFSDataModels:
    """Verifies construction, immutability, and parameter bounds of FCFS data models."""

    def test_valid_process_spec_creation(self):
        p = ProcessSpec(pid="P1", arrival_time=0, burst_time=5)
        assert p.pid == "P1"
        assert p.arrival_time == 0
        assert p.burst_time == 5

    def test_process_spec_is_immutable(self):
        p = ProcessSpec(pid="P1", arrival_time=0, burst_time=5)
        with pytest.raises(Exception):
            p.pid = "P2"  # type: ignore

    def test_empty_or_whitespace_pid_rejected(self):
        with pytest.raises(ValueError, match="PID must be a non-empty string"):
            ProcessSpec(pid="", arrival_time=0, burst_time=5)
        with pytest.raises(ValueError, match="PID must be a non-empty string"):
            ProcessSpec(pid="   ", arrival_time=0, burst_time=5)

    def test_negative_arrival_time_rejected(self):
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            ProcessSpec(pid="P1", arrival_time=-1, burst_time=5)

    def test_zero_or_negative_burst_time_rejected(self):
        with pytest.raises(ValueError, match="must be a positive integer"):
            ProcessSpec(pid="P1", arrival_time=0, burst_time=0)
        with pytest.raises(ValueError, match="must be a positive integer"):
            ProcessSpec(pid="P1", arrival_time=0, burst_time=-4)

    def test_non_integer_times_rejected(self):
        with pytest.raises(TypeError, match="must be an integer"):
            ProcessSpec(pid="P1", arrival_time=1.5, burst_time=4)  # type: ignore
        with pytest.raises(TypeError, match="must be an integer"):
            ProcessSpec(pid="P1", arrival_time=0, burst_time=3.2)  # type: ignore
        with pytest.raises(TypeError, match="must be an integer"):
            ProcessSpec(pid="P1", arrival_time=True, burst_time=4)  # type: ignore

    def test_gantt_block_properties(self):
        b_busy = GanttBlock(pid="P1", start_time=2, end_time=7, is_idle=False)
        assert b_busy.duration == 5
        assert b_busy.to_dict() == {
            "pid": "P1",
            "start_time": 2,
            "end_time": 7,
            "duration": 5,
            "is_idle": False,
        }

        b_idle = GanttBlock(pid=None, start_time=0, end_time=3, is_idle=True)
        assert b_idle.duration == 3
        assert b_idle.is_idle is True

    def test_process_metrics_to_dict(self):
        m = ProcessMetrics(
            pid="P1",
            arrival_time=1,
            burst_time=4,
            completion_time=5,
            turnaround_time=4,
            waiting_time=0,
            response_time=0,
        )
        assert m.to_dict() == {
            "pid": "P1",
            "arrival_time": 1,
            "burst_time": 4,
            "completion_time": 5,
            "turnaround_time": 4,
            "waiting_time": 0,
            "response_time": 0,
        }


class TestFCFSInputValidation:
    """Verifies fail-closed validation of process lists."""

    def test_empty_process_list_rejected(self):
        with pytest.raises(ValueError, match="Process list cannot be empty"):
            validate_process_list([])
        with pytest.raises(ValueError, match="Process list cannot be empty"):
            FCFSSimulator.run([])

    def test_duplicate_pid_rejected(self):
        procs = [
            ProcessSpec(pid="P1", arrival_time=0, burst_time=3),
            ProcessSpec(pid="P1", arrival_time=2, burst_time=4),
        ]
        with pytest.raises(ValueError, match="Duplicate process ID detected: 'P1'"):
            validate_process_list(procs)
        with pytest.raises(ValueError, match="Duplicate process ID detected: 'P1'"):
            FCFSSimulator.run(procs)

    def test_non_process_spec_rejected(self):
        with pytest.raises(TypeError, match="Expected ProcessSpec instance"):
            validate_process_list([{"pid": "P1", "arrival_time": 0, "burst_time": 3}])  # type: ignore


# ============================================================================
# 2. FCFSSimulator Discrete Event Execution Tests
# ============================================================================

class TestFCFSSimulatorExecution:
    """Tests simulation execution across different arrival and workload scenarios."""

    def test_single_process_execution(self):
        procs = [ProcessSpec(pid="P1", arrival_time=0, burst_time=4)]
        res = FCFSSimulator.run(procs)

        assert res.total_simulation_time == 4
        assert res.total_idle_time == 0
        assert len(res.gantt_chart) == 1
        assert res.gantt_chart[0] == GanttBlock(pid="P1", start_time=0, end_time=4, is_idle=False)

        m = res.process_metrics["P1"]
        assert m.completion_time == 4
        assert m.turnaround_time == 4
        assert m.waiting_time == 0
        assert res.average_turnaround_time == 4.0
        assert res.average_waiting_time == 0.0

    def test_sequential_staggered_execution_no_idle(self):
        # P1: arr 0, bt 4 -> runs [0, 4], CT=4, TAT=4, WT=0
        # P2: arr 1, bt 3 -> runs [4, 7], CT=7, TAT=6, WT=3
        # P3: arr 2, bt 1 -> runs [7, 8], CT=8, TAT=6, WT=5
        procs = [
            ProcessSpec(pid="P1", arrival_time=0, burst_time=4),
            ProcessSpec(pid="P2", arrival_time=1, burst_time=3),
            ProcessSpec(pid="P3", arrival_time=2, burst_time=1),
        ]
        res = FCFSSimulator.run(procs)

        assert res.total_simulation_time == 8
        assert res.total_idle_time == 0
        assert [(b.pid, b.start_time, b.end_time) for b in res.gantt_chart] == [
            ("P1", 0, 4),
            ("P2", 4, 7),
            ("P3", 7, 8),
        ]

        p1_m = res.process_metrics["P1"]
        assert p1_m.completion_time == 4
        assert p1_m.turnaround_time == 4
        assert p1_m.waiting_time == 0

        p2_m = res.process_metrics["P2"]
        assert p2_m.completion_time == 7
        assert p2_m.turnaround_time == 6
        assert p2_m.waiting_time == 3

        p3_m = res.process_metrics["P3"]
        assert p3_m.completion_time == 8
        assert p3_m.turnaround_time == 6
        assert p3_m.waiting_time == 5

        # Avg TAT = (4 + 6 + 6) / 3 = 16/3
        assert abs(res.average_turnaround_time - (16 / 3)) < 1e-6
        # Avg WT = (0 + 3 + 5) / 3 = 8/3
        assert abs(res.average_waiting_time - (8 / 3)) < 1e-6

    def test_simultaneous_arrivals_at_zero_stable_input_order(self):
        """Simultaneous arrivals at t=0 must execute in stable input order."""
        # PB declared before PA in input sequence
        procs = [
            ProcessSpec(pid="P_B", arrival_time=0, burst_time=3),
            ProcessSpec(pid="P_A", arrival_time=0, burst_time=2),
        ]
        res = FCFSSimulator.run(procs)

        assert [(b.pid, b.start_time, b.end_time) for b in res.gantt_chart] == [
            ("P_B", 0, 3),
            ("P_A", 3, 5),
        ]
        assert res.process_metrics["P_B"].waiting_time == 0
        assert res.process_metrics["P_A"].waiting_time == 3

    def test_simultaneous_arrivals_mid_simulation_stable_input_order(self):
        """Processes arriving simultaneously while another is running break ties by input order."""
        procs = [
            ProcessSpec(pid="P1", arrival_time=0, burst_time=5),
            ProcessSpec(pid="P_Z", arrival_time=2, burst_time=2),
            ProcessSpec(pid="P_M", arrival_time=2, burst_time=1),
        ]
        res = FCFSSimulator.run(procs)

        assert [(b.pid, b.start_time, b.end_time) for b in res.gantt_chart] == [
            ("P1", 0, 5),
            ("P_Z", 5, 7),
            ("P_M", 7, 8),
        ]
        assert res.process_metrics["P_Z"].waiting_time == 3  # arrived at 2, started at 5
        assert res.process_metrics["P_M"].waiting_time == 5  # arrived at 2, started at 7

    def test_initial_cpu_idle_period(self):
        """CPU idle from t=0 until first arrival at t=3."""
        procs = [ProcessSpec(pid="P1", arrival_time=3, burst_time=4)]
        res = FCFSSimulator.run(procs)

        assert res.total_simulation_time == 7
        assert res.total_idle_time == 3
        assert len(res.gantt_chart) == 2

        idle_block = res.gantt_chart[0]
        assert idle_block.is_idle is True
        assert idle_block.pid is None
        assert idle_block.start_time == 0
        assert idle_block.end_time == 3

        exec_block = res.gantt_chart[1]
        assert exec_block.is_idle is False
        assert exec_block.pid == "P1"
        assert exec_block.start_time == 3
        assert exec_block.end_time == 7

        m = res.process_metrics["P1"]
        assert m.completion_time == 7
        assert m.turnaround_time == 4
        assert m.waiting_time == 0

    def test_interleaved_cpu_idle_period(self):
        """CPU completes P1 at t=2, waits idle until P2 arrives at t=5."""
        procs = [
            ProcessSpec(pid="P1", arrival_time=0, burst_time=2),
            ProcessSpec(pid="P2", arrival_time=5, burst_time=3),
        ]
        res = FCFSSimulator.run(procs)

        assert res.total_simulation_time == 8
        assert res.total_idle_time == 3
        assert [(b.pid, b.start_time, b.end_time, b.is_idle) for b in res.gantt_chart] == [
            ("P1", 0, 2, False),
            (None, 2, 5, True),
            ("P2", 5, 8, False),
        ]

        assert res.process_metrics["P1"].completion_time == 2
        assert res.process_metrics["P2"].completion_time == 8
        assert res.process_metrics["P2"].waiting_time == 0

    def test_canonical_event_ordering_same_tick_completion_and_arrival(self):
        """
        When P1 completes at t=4 and P2 arrives at t=4:
        Completions first (P1 terminates, freeing CPU) ->
        Arrivals second (P2 enters ready queue) ->
        Dispatch third (P2 immediately dispatched at t=4 without idle gap).
        """
        procs = [
            ProcessSpec(pid="P1", arrival_time=0, burst_time=4),
            ProcessSpec(pid="P2", arrival_time=4, burst_time=2),
        ]
        res = FCFSSimulator.run(procs)

        assert res.total_idle_time == 0
        assert [(b.pid, b.start_time, b.end_time) for b in res.gantt_chart] == [
            ("P1", 0, 4),
            ("P2", 4, 6),
        ]
        assert res.process_metrics["P2"].waiting_time == 0


# ============================================================================
# 3. Honest Metric Accounting & Snapshot Semantics Tests
# ============================================================================

class TestFCFSSnapshotSemantics:
    """Verifies that snapshots are complete, isolated, and strictly honest about metrics."""

    def test_metrics_are_none_until_process_terminates(self):
        procs = [
            ProcessSpec(pid="P1", arrival_time=0, burst_time=4),
            ProcessSpec(pid="P2", arrival_time=2, burst_time=3),
        ]
        res = FCFSSimulator.run(procs)
        trace = res.computation_trace
        snaps = trace.snapshots

        # Snapshot 0 (t=0): P1 starts running, P2 not arrived
        snap_0 = snaps[0]
        assert snap_0.state["current_time"] == 0
        assert snap_0.state["running_pid"] == "P1"
        assert snap_0.state["process_metrics"]["P1"]["completion_time"] is None
        assert snap_0.state["process_metrics"]["P2"]["completion_time"] is None
        assert snap_0.state["average_waiting_time"] is None
        assert snap_0.state["average_turnaround_time"] is None

        # Snapshot 1 (t=2): P2 arrives mid-burst, enters ready queue
        snap_1 = snaps[1]
        assert snap_1.state["current_time"] == 2
        assert snap_1.state["running_pid"] == "P1"
        assert snap_1.state["ready_queue"] == ["P2"]
        assert snap_1.state["process_metrics"]["P1"]["completion_time"] is None
        assert snap_1.state["process_metrics"]["P2"]["completion_time"] is None
        assert snap_1.state["average_waiting_time"] is None

        # Snapshot 2 (t=4): P1 completes, P2 dispatched
        snap_2 = snaps[2]
        assert snap_2.state["current_time"] == 4
        assert snap_2.state["running_pid"] == "P2"
        assert snap_2.state["completed_pids"] == ["P1"]
        # P1 metrics are now populated!
        assert snap_2.state["process_metrics"]["P1"]["completion_time"] == 4
        assert snap_2.state["process_metrics"]["P1"]["turnaround_time"] == 4
        assert snap_2.state["process_metrics"]["P1"]["waiting_time"] == 0
        # P2 metrics are still None!
        assert snap_2.state["process_metrics"]["P2"]["completion_time"] is None
        assert snap_2.state["average_waiting_time"] is None

        # Final Snapshot 3 (t=7): P2 completes, simulation done
        final_snap = snaps[3]
        assert final_snap.state["current_time"] == 7
        assert final_snap.state["cpu_state"] == "IDLE"
        assert final_snap.state["running_pid"] is None
        assert final_snap.state["completed_pids"] == ["P1", "P2"]
        # P2 metrics now populated!
        assert final_snap.state["process_metrics"]["P2"]["completion_time"] == 7
        assert final_snap.state["process_metrics"]["P2"]["turnaround_time"] == 5  # 7 - 2
        assert final_snap.state["process_metrics"]["P2"]["waiting_time"] == 2     # 5 - 3
        # Final averages now populated!
        assert final_snap.state["average_turnaround_time"] == 4.5
        assert final_snap.state["average_waiting_time"] == 1.0

    def test_snapshot_deep_copy_isolation(self):
        procs = [ProcessSpec(pid="P1", arrival_time=0, burst_time=2)]
        res = FCFSSimulator.run(procs)
        trace = res.computation_trace

        snap_0 = trace.snapshots[0]
        # Mutate snap_0 state externally
        snap_0.state["current_time"] = 999
        snap_0.state["ready_queue"].append("MUTATED")

        # Snapshot 1 must not be affected
        snap_1 = trace.snapshots[1]
        assert snap_1.state["current_time"] == 2
        assert "MUTATED" not in snap_1.state["ready_queue"]

    def test_computation_trace_monotonically_ordered(self):
        procs = [
            ProcessSpec(pid="P1", arrival_time=0, burst_time=3),
            ProcessSpec(pid="P2", arrival_time=1, burst_time=2),
        ]
        res = FCFSSimulator.run(procs)
        trace = res.computation_trace

        times = [s.state["current_time"] for s in trace.snapshots]
        assert times == sorted(times)
        step_indices = [s.step_index for s in trace.snapshots]
        assert step_indices == list(range(len(trace.snapshots)))


# ============================================================================
# 4. FCFSTraceVerifier Outline Integration Tests
# ============================================================================

class TestFCFSTraceVerifier:
    """Tests FCFSTraceVerifier auditing and outline step verification."""

    def test_can_verify_recognition(self):
        assert FCFSTraceVerifier.can_verify({}, topic="FCFS CPU Scheduling") is True
        assert FCFSTraceVerifier.can_verify({}, topic="First Come First Serve") is True
        assert FCFSTraceVerifier.can_verify({}, topic="First-Come First-Served CPU Scheduling") is True
        assert FCFSTraceVerifier.can_verify({"algorithm": "fcfs"}, topic="") is True
        assert FCFSTraceVerifier.can_verify({"domain": "os_scheduling"}, topic="") is True
        assert FCFSTraceVerifier.can_verify({"setup": "FCFS scheduling with 3 processes"}, topic="") is True

        # Must not claim non-OS topics
        assert FCFSTraceVerifier.can_verify({}, topic="Binary Search") is False
        assert FCFSTraceVerifier.can_verify({}, topic="Dijkstra's Shortest Path") is False
        assert FCFSTraceVerifier.can_verify({}, topic="0/1 Knapsack Problem") is False
        assert FCFSTraceVerifier.can_verify({}, topic="Photosynthesis") is False

    def test_valid_outline_spec_verified_successfully(self):
        spec = {
            "topic": "First-Come, First-Served CPU Scheduling",
            "inputs": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                    {"pid": "P2", "arrival_time": 2, "burst_time": 2},
                ]
            },
            "steps": [
                {
                    "step_number": 1,
                    "description": "At time 0, P1 arrives and is dispatched",
                    "current_time": 0,
                    "running_pid": "P1",
                },
                {
                    "step_number": 2,
                    "description": "At time 2, P2 arrives and waits in queue",
                    "current_time": 2,
                    "running_pid": "P1",
                },
            ],
            "final_result": "Average Waiting Time = 0.50, Average Turnaround Time = 3.00",
        }

        out_spec, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS")
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.verified_steps == 2
        assert report.failed_steps == 0
        assert report.final_result_status == VerificationStatus.VERIFIED
        assert report.is_safe_for_stage2 is True
        assert "verified_algorithmic_state" in out_spec["verification_metadata"]
        assert len(report.state_snapshots) >= 2

    def test_invalid_input_fails_closed_with_diagnostics(self):
        spec = {
            "inputs": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                    {"pid": "P1", "arrival_time": 1, "burst_time": 2},  # Duplicate PID
                ]
            },
            "steps": [{"description": "Initial step"}],
        }

        out_spec, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS")
        assert report.overall_status == VerificationStatus.FAILED
        assert report.failed_steps >= 1
        assert report.is_safe_for_stage2 is False
        assert "Duplicate process ID" in report.summary

    def test_incorrect_time_claim_detected(self):
        spec = {
            "inputs": {
                "processes": [{"pid": "P1", "arrival_time": 0, "burst_time": 3}]
            },
            "steps": [
                {
                    "step_number": 1,
                    "description": "Wrong time claim",
                    "current_time": 99,  # Authoritative is 0
                    "running_pid": "P1",
                }
            ],
        }

        out_spec, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS")
        assert report.overall_status == VerificationStatus.FAILED
        assert report.failed_steps == 1
        assert "Time mismatch" in report.step_diagnostics[0]["details"]

    def test_premature_metric_claim_detected_and_rejected(self):
        """Claiming a process's completion time before it terminates must fail."""
        spec = {
            "inputs": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 4},
                    {"pid": "P2", "arrival_time": 1, "burst_time": 3},
                ]
            },
            "steps": [
                {
                    "step_number": 1,
                    "description": "Premature metric claim for P2",
                    "current_time": 0,
                    "running_pid": "P1",
                    "claimed_metrics": {
                        "P2": {"completion_time": 7}  # P2 has not terminated at step 0!
                    },
                }
            ],
        }

        out_spec, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS")
        assert report.overall_status == VerificationStatus.FAILED
        assert report.failed_steps == 1
        assert "Premature metric claim" in report.step_diagnostics[0]["details"]

    def test_incorrect_final_result_averages_rejected(self):
        spec = {
            "inputs": {
                "processes": [{"pid": "P1", "arrival_time": 0, "burst_time": 4}]
            },
            "steps": [],
            "final_result": "Average Waiting Time = 99.0, Average Turnaround Time = 99.0",
        }

        out_spec, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS")
        assert report.final_result_status == VerificationStatus.FAILED
        assert report.overall_status == VerificationStatus.FAILED

    def test_canonical_final_result_auto_filled_when_omitted(self):
        spec = {
            "inputs": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 2},
                    {"pid": "P2", "arrival_time": 1, "burst_time": 2},
                ]
            },
            "steps": [],
        }

        out_spec, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS")
        assert report.final_result_status == VerificationStatus.VERIFIED
        assert "Average Waiting Time = 0.50, Average Turnaround Time = 2.50" in out_spec["final_result"]


# ============================================================================
# 5. SpecializedVerifierRegistry & WorkedExampleVerifier Integration Tests
# ============================================================================

class TestRegistryAndOrchestratorIntegration:
    """Verifies that SpecializedVerifierRegistry routes cleanly without monolithic branching."""

    def test_fcfs_is_registered_in_registry(self):
        verifier = SpecializedVerifierRegistry.find_verifier(
            spec={"inputs": {"processes": [{"pid": "P1", "arrival_time": 0, "burst_time": 4}]}},
            topic="FCFS CPU Scheduling",
        )
        assert verifier is not None
        assert verifier == FCFSTraceVerifier

    def test_non_fcfs_topics_not_intercepted_by_fcfs(self):
        verifier = SpecializedVerifierRegistry.find_verifier(
            spec={"inputs": {"arr": [1, 3, 5], "target": 3}},
            topic="Binary Search",
        )
        # Should not be FCFS verifier
        assert verifier != FCFSTraceVerifier

    def test_worked_example_verifier_routes_fcfs_cleanly(self):
        verifier = WorkedExampleVerifier()
        spec = {
            "inputs": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                    {"pid": "P2", "arrival_time": 1, "burst_time": 2},
                ]
            },
            "steps": [
                {
                    "step_number": 1,
                    "current_time": 0,
                    "running_pid": "P1",
                }
            ],
        }

        out_spec, report = verifier.verify_worked_example_spec(spec, topic="FCFS CPU Scheduling")
        assert report.overall_status == VerificationStatus.VERIFIED
        assert report.verification_method == "fcfs_trace_verifier"
        assert report.verified_algorithmic_state is not None
        assert report.verified_algorithmic_state["algorithm"] == "fcfs"


# ============================================================================
# 6. Storyboard Fidelity Integration with FCFS Snapshots
# ============================================================================

class TestStoryboardFidelityFCFSIntegration:
    """Verifies that StoryboardFidelityChecker checks storyboard scenes against FCFS snapshots."""

    def test_storyboard_fidelity_step_aware_with_fcfs_snapshots(self):
        spec = {
            "inputs": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                    {"pid": "P2", "arrival_time": 1, "burst_time": 2},
                ]
            },
            "steps": [
                {
                    "step_number": 1,
                    "description": "P1 starts running at time 0",
                    "current_time": 0,
                    "running_pid": "P1",
                },
                {
                    "step_number": 2,
                    "description": "P2 arrives at time 1 and waits in ready queue",
                    "current_time": 1,
                    "running_pid": "P1",
                },
            ],
        }
        _, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS")

        outline_data = {
            "topic": "FCFS CPU Scheduling",
            "worked_example_spec": spec,
            "verification_metadata": report.to_dict(),
            "teaching_beats": [
                {
                    "beat_title": "Step 1: P1 Execution",
                    "section_type": "worked_demonstration",
                    "step_number": 1,
                },
                {
                    "beat_title": "Step 2: P2 Arrival",
                    "section_type": "worked_demonstration",
                    "step_number": 2,
                },
            ],
        }

        storyboard_data = {
            "sections": [
                {
                    "section_name": "Step 1: P1 Execution",
                    "lecture_lines": [
                        "At time 0, process P1 is dispatched to the CPU."
                    ],
                },
                {
                    "section_name": "Step 2: P2 Arrival",
                    "lecture_lines": [
                        "At time 1, process P2 arrives while P1 continues execution."
                    ],
                },
            ]
        }

        checker = StoryboardFidelityChecker()
        fidelity_report = checker.check_storyboard_fidelity(outline_data, storyboard_data)
        assert fidelity_report.overall_status in [VerificationStatus.VERIFIED, VerificationStatus.NEEDS_REVIEW]
        assert len(fidelity_report.discrepancies) == 0
