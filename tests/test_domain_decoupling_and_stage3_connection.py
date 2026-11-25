"""
Test suite for Generic Domain Decoupling, Input Normalization, Response Time Metrics,
and Structured Stage 3 Action Resolution Integration.
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List

from src.domains import (
    VisualRegistry,
    DomainDescriptor,
    DomainTraceGuidanceRegistry,
    DomainFidelityVerifier,
    DomainFidelityRegistry,
    ResolvedVisualAction,
    StoryboardActionResolver,
)
from src.domains.cs.os_scheduling import (
    ProcessCardVisual,
    ReadyQueueVisual,
    CPUExecutionVisual,
    GanttChartVisual,
    MetricTableVisual,
    OSTraceVisualAdapter,
)
from src.grounding.verifiers.os.fcfs_verifier import (
    ProcessSpec,
    FCFSSimulator,
    FCFSTraceVerifier,
)
from src.agent import Section
from prompts.stage2 import get_prompt2_storyboard, extract_verified_algorithmic_state
from prompts.stage3 import get_prompt3_code, _format_resolved_actions
from prompts.base_class import TeachingScene


# ============================================================================
# 1. Worked-Example Input Normalization Tests
# ============================================================================

class TestFCFSInputNormalization:
    """Tests robust parsing and normalization of worked example specs."""

    def test_dict_of_dicts_format_with_aliases(self):
        """Dict-of-dicts format with field aliases (AT/BT) parses into canonical ProcessSpec list."""
        spec = {
            "inputs": {
                "P1": {"AT": 0, "BT": 3},
                "P2": {"AT": 1, "BT": 4},
                "P3": {"AT": 2, "BT": 2},
            }
        }
        procs = FCFSTraceVerifier.parse_processes_from_spec(spec)
        assert len(procs) == 3
        assert procs[0] == ProcessSpec("P1", 0, 3)
        assert procs[1] == ProcessSpec("P2", 1, 4)
        assert procs[2] == ProcessSpec("P3", 2, 2)
        # Order is preserved
        assert [p.pid for p in procs] == ["P1", "P2", "P3"]

    def test_list_of_dicts_format_with_lowercase_aliases(self):
        """List-of-dicts with lowercase keys (arrival, burst) parses canonically."""
        spec = {
            "inputs": [
                {"pid": "ProcA", "arrival": 0, "burst": 5},
                {"name": "ProcB", "at": 3, "bt": 2},
            ]
        }
        procs = FCFSTraceVerifier.parse_processes_from_spec(spec)
        assert len(procs) == 2
        assert procs[0].pid == "ProcA" and procs[0].arrival_time == 0 and procs[0].burst_time == 5
        assert procs[1].pid == "ProcB" and procs[1].arrival_time == 3 and procs[1].burst_time == 2

    def test_reject_duplicate_pids(self):
        """Duplicate PIDs fail-closed with clear error."""
        spec = {
            "inputs": [
                {"pid": "P1", "AT": 0, "BT": 3},
                {"pid": "P1", "AT": 1, "BT": 2},
            ]
        }
        with pytest.raises(ValueError, match="Duplicate process ID"):
            FCFSTraceVerifier.parse_processes_from_spec(spec)

    def test_reject_negative_arrival_time(self):
        """Negative arrival time is rejected fail-closed."""
        spec = {
            "inputs": {
                "P1": {"AT": -1, "BT": 3}
            }
        }
        with pytest.raises(ValueError, match="non-negative integer"):
            FCFSTraceVerifier.parse_processes_from_spec(spec)

    def test_reject_zero_or_negative_burst_time(self):
        """Zero or negative burst time is rejected fail-closed."""
        spec = {
            "inputs": {
                "P1": {"AT": 0, "BT": 0}
            }
        }
        with pytest.raises(ValueError, match="positive integer"):
            FCFSTraceVerifier.parse_processes_from_spec(spec)

    def test_reject_missing_fields(self):
        """Missing arrival or burst time raises error without silent truncation."""
        spec = {
            "inputs": {
                "P1": {"AT": 0}
            }
        }
        with pytest.raises(ValueError, match="missing 'burst_time'"):
            FCFSTraceVerifier.parse_processes_from_spec(spec)

    def test_can_verify_recognizes_dict_of_dicts(self):
        """can_verify accurately detects dict-of-dicts worked example specs."""
        spec = {
            "inputs": {
                "P1": {"AT": 0, "BT": 3},
                "P2": {"AT": 1, "BT": 4},
            }
        }
        assert FCFSTraceVerifier.can_verify(spec, topic="Operating Systems FCFS") is True


# ============================================================================
# 2. Response Time (RT) Metrics Tests
# ============================================================================

class TestResponseTimeMetrics:
    """Verifies Response Time (RT = first_cpu_start_time - arrival_time) semantics."""

    def test_response_time_calculation_and_non_premature_leakage(self):
        """RT becomes known upon first CPU dispatch, while CT/TAT/WT remain None until termination."""
        procs = [
            ProcessSpec("P1", arrival_time=0, burst_time=3),
            ProcessSpec("P2", arrival_time=1, burst_time=4),
            ProcessSpec("P3", arrival_time=2, burst_time=2),
        ]
        result = FCFSSimulator.run(procs)
        trace = result.computation_trace

        # Step 0: t=0, P1 dispatched (start=0, AT=0 -> RT=0). P2, P3 NOT ARRIVED.
        st0 = trace.snapshots[0].state
        assert st0["process_metrics"]["P1"]["response_time"] == 0
        assert st0["process_metrics"]["P1"]["completion_time"] is None
        assert st0["process_metrics"]["P1"]["turnaround_time"] is None
        assert st0["process_metrics"]["P1"]["waiting_time"] is None

        # Step 2: t=2, P1 still running. P2 in queue (RT is still None!).
        st2 = trace.snapshots[2].state
        assert st2["process_metrics"]["P2"]["response_time"] is None

        # Step 3: t=3, P1 terminates (CT=3, TAT=3, WT=0, RT=0).
        # P2 dispatched at t=3 (start=3, AT=1 -> RT = 3 - 1 = 2).
        st3 = trace.snapshots[3].state
        assert st3["process_metrics"]["P1"]["completion_time"] == 3
        assert st3["process_metrics"]["P2"]["response_time"] == 2
        assert st3["process_metrics"]["P2"]["completion_time"] is None

        # Final snapshot: all metrics known and average response time computed
        final_st = trace.snapshots[-1].state
        assert final_st["process_metrics"]["P3"]["response_time"] == 5  # start=7, AT=2 -> RT=5
        assert round(final_st["average_response_time"], 2) == 2.33  # (0 + 2 + 5) / 3 = 7 / 3 ≈ 2.33
        assert round(result.average_response_time, 2) == 2.33

    def test_metric_table_response_time_updates(self):
        """MetricTableVisual properly exposes RT handle and updates on dispatch and final averages."""
        specs = [{"pid": "P1", "arrival_time": 0, "burst_time": 3}]
        table = MetricTableVisual(processes=specs)

        # Before dispatch: RT is "-"
        assert table.get_semantic_handle("cell_P1_RT").text == "-"

        # Upon first dispatch: animate_update_response_time reveals RT
        table.animate_update_response_time(pid="P1", rt=0)
        assert table.get_semantic_handle("cell_P1_RT").text == "0"
        # CT, TAT, WT remain unrevealed
        assert table.get_semantic_handle("cell_P1_CT").text == "-"

        # Show averages updates RT average handle
        table.animate_show_averages(avg_tat=3.0, avg_wt=0.0, avg_rt=0.0)
        assert table.get_semantic_handle("cell_avg_rt").text == "0.00"


# ============================================================================
# 3. Domain Decoupling & Guidance Registry Tests
# ============================================================================

class TestDomainDecoupling:
    """Verifies DomainTraceGuidanceRegistry, DomainFidelityRegistry, and DomainDescriptor discovery."""

    def test_guidance_registry_formatting(self):
        """DomainTraceGuidanceRegistry formats FCFS guidance without hardcoding in stage2.py."""
        spec = {
            "inputs": {
                "P1": {"AT": 0, "BT": 3},
                "P2": {"AT": 1, "BT": 4},
            }
        }
        _, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS")
        algo_state = report.verified_algorithmic_state

        guidance = DomainTraceGuidanceRegistry.format_guidance(algo_state)
        assert "Authoritative FCFS CPU Scheduling Simulation & Trace Facts" in guidance
        assert "P1: Arrival Time = 0, Burst Time = 3" in guidance
        assert "P2: Arrival Time = 1, Burst Time = 4" in guidance
        assert "Average Response Time" in guidance

    def test_domain_fidelity_registry_registration(self):
        """DomainFidelityRegistry holds verifiers for registered domains."""
        fcfs_verifier = DomainFidelityRegistry.get("fcfs")
        assert fcfs_verifier is not None
        assert isinstance(fcfs_verifier, DomainFidelityVerifier)

        dijkstra_verifier = DomainFidelityRegistry.get("dijkstra")
        assert dijkstra_verifier is not None
        assert isinstance(dijkstra_verifier, DomainFidelityVerifier)

    def test_extensible_domain_discovery_via_descriptor(self):
        """VisualRegistry.get_domain_for_topic uses descriptors rather than hardcoded string logic."""
        # Standard keywords
        assert VisualRegistry.get_domain_for_topic("Operating Systems FCFS") == "cs"
        assert VisualRegistry.get_domain_for_topic("Reaction Kinetics Arrhenius Equation") == "chemistry"
        assert VisualRegistry.get_domain_for_topic("Calculus Derivative Power Rule") == "math"

        # Register a new custom domain dynamically
        custom_domain = DomainDescriptor(
            name="robotics",
            display_name="Autonomous Robotics",
            keywords=["kinematics", "lidar", "slam", "odometry", "inverse kinematics"],
            visual_modes=["coordinate_geometry", "process_flowchart"],
        )
        VisualRegistry.register_domain(custom_domain)

        assert VisualRegistry.get_domain_for_topic("Forward Kinematics Transformation") == "robotics"
        assert VisualRegistry.get_domain_for_topic("Lidar SLAM Mapping") == "robotics"


# ============================================================================
# 4. Action Resolver Component Ownership & Stage 3 Connection Tests
# ============================================================================

class TestActionResolverAndStage3Connection:
    """Verifies component ownership disambiguation and Stage 3 prompt connection."""

    def test_component_ownership_disambiguation(self):
        """When ProcessCardVisual specifies animate_enqueue, ReadyQueueVisual is resolved as operating component."""
        step = "L1-2 | FadeIn | ProcessCardVisual(pid='P1', arrival_time=0) | viewport: dock_split_left | action: animate_enqueue() | on screen after: [queue, p1]"
        is_valid, resolved, err = StoryboardActionResolver.resolve_visual_action(step)

        assert is_valid is True
        assert err is None
        assert resolved is not None
        assert resolved.component_name == "ProcessCardVisual"
        # Ownership disambiguation: ReadyQueueVisual actually executes animate_enqueue!
        assert resolved.operating_component == "ReadyQueueVisual"
        assert resolved.operating_class is ReadyQueueVisual
        assert resolved.canonical_action == "animate_enqueue"
        assert "card_p1" in resolved.intended_call or "ready_queue.animate_enqueue" in resolved.intended_call
        assert resolved.target_instance_id == "ready_queue"
        assert "enters the Ready Queue" in resolved.expected_visual_result

    def test_gantt_and_table_resolution_details(self):
        """Action resolver produces explicit instance IDs and calls for Gantt and MetricTable."""
        gantt_step = "L2-1 | FadeIn | GanttChartVisual() | viewport: dock_inspector | action: animate_add_block(pid='P1', start_time=0, end_time=3) | on screen after: [gantt]"
        _, res_gantt, _ = StoryboardActionResolver.resolve_visual_action(gantt_step)
        assert res_gantt.operating_component == "GanttChartVisual"
        assert res_gantt.target_instance_id == "gantt_chart"
        assert "gantt_chart.animate_add_block" in res_gantt.intended_call
        assert "pid='P1'" in res_gantt.intended_call

        table_step = "L2-2 | Transform | MetricTableVisual() | viewport: dock_inspector | action: animate_update_response_time(pid='P1', response_time=0) | on screen after: [table]"
        _, res_table, _ = StoryboardActionResolver.resolve_visual_action(table_step)
        assert res_table.operating_component == "MetricTableVisual"
        assert res_table.target_instance_id == "metric_table"
        assert "metric_table.animate_update_response_time" in res_table.intended_call
        assert "Response Time" in res_table.expected_visual_result

    def test_stage3_prompt_receives_structured_resolved_actions(self):
        """Stage 3 prompt distinguishes teaching context, visual intent, and validated visual operations."""
        resolved_actions = [
            {
                "step_tag": "L1-1",
                "effect": "FadeIn",
                "component_name": "ProcessCardVisual",
                "operating_component": "ReadyQueueVisual",
                "target_instance_id": "ready_queue",
                "canonical_action": "animate_enqueue",
                "intended_call": "ready_queue.animate_enqueue(card_p1)",
                "viewport": "dock_split_left",
                "subject_entity": "ProcessCardVisual(pid='P1')",
                "expected_visual_result": "Process P1 enters Ready Queue tail slot",
                "trace_ref": "arrival of P1 at t=0",
                "kwargs": {"pid": "P1"},
            }
        ]

        section = Section(
            id="section_2",
            title="Sec 2: Process Arrival and Queueing",
            lecture_lines=[
                "Process P1 arrives at time zero and enters the ready queue.",
                "The ready queue holds all processes waiting for the CPU in strict arrival order.",
            ],
            animations=[
                "L1-1 | FadeIn | ProcessCardVisual(pid='P1') | viewport: dock_split_left | action: animate_enqueue() | on screen after: [queue, p1]"
            ],
            resolved_actions=resolved_actions,
        )

        prompt = get_prompt3_code(
            regenerate_note="",
            section=section,
            base_class=TeachingScene,
        )

        # Distinguishes Teaching Context, Visual Intent, and Validated Visual Operations
        assert "TEACHING AND NARRATION CONTEXT" in prompt
        assert "Visual Intent (Storyboard Animation Steps):" in prompt
        assert "Validated Visual Actions & Operations (STAGE 3 AUTHORITATIVE CONTRACT)" in prompt
        assert "Operating Component: `ReadyQueueVisual`" in prompt
        assert "Target Instance: `ready_queue`" in prompt
        assert "Concrete Code Call: `ready_queue.animate_enqueue(card_p1)`" in prompt
        assert "Authoritative Trace Reference: arrival of P1 at t=0" in prompt
        assert "Authoritative Execution Data & Verified Values:" in prompt

    def test_action_resolver_handle_highlight_and_parameter_aliases(self):
        """Verifies that handle highlights and parameter aliases (pid, tick_count) resolve cleanly."""
        resolver = StoryboardActionResolver()

        # 1. CPUExecutionVisual animate_dispatch with pid
        valid, resolved, err = resolver.resolve_visual_action(
            visual_rep="CPUExecutionVisual",
            action_str="animate_dispatch(pid='P1')",
            viewport="dock_split_right",
        )
        assert valid is True
        assert err is None
        assert resolved.canonical_action == "animate_dispatch"
        assert resolved.kwargs.get("pid") == "P1"

        # 2. CPUExecutionVisual animate_execute_tick with tick_count
        valid2, resolved2, err2 = resolver.resolve_visual_action(
            visual_rep="CPUExecutionVisual",
            action_str="animate_execute_tick(tick_count=3)",
            viewport="dock_split_right",
        )
        assert valid2 is True
        assert err2 is None
        assert resolved2.canonical_action == "animate_execute_tick"
        assert resolved2.kwargs.get("tick_count") == 3

        # 3. MetricTableVisual handle highlight
        valid3, resolved3, err3 = resolver.resolve_visual_action(
            visual_rep="MetricTableVisual",
            action_str="highlight_cell_avg_wt()",
            viewport="dock_inspector",
        )
        assert valid3 is True
        assert err3 is None

    def test_fcfs_fidelity_multi_process_narration_no_leak(self):
        """Verifies that multi-process narration does not falsely attribute one process's metrics to another."""
        from src.storyboard_fidelity_checker import StoryboardFidelityChecker

        model_dict = {
            "topic": "FCFS CPU Scheduling",
            "worked_example_spec": {
                "inputs": {"P1": {"AT": 0, "BT": 3}, "P2": {"AT": 1, "BT": 4}, "P3": {"AT": 2, "BT": 2}},
                "steps": [
                    {"step": "P1 completes at 3", "operation": "complete", "state_after": {"ready_queue": []}}
                ],
                "authoritative_state": {
                    "algorithm": "fcfs",
                    "processes": [
                        {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                        {"pid": "P2", "arrival_time": 1, "burst_time": 4},
                        {"pid": "P3", "arrival_time": 2, "burst_time": 2},
                    ],
                    "process_metrics": {
                        "P1": {"completion_time": 3, "turnaround_time": 3, "waiting_time": 0, "response_time": 0},
                        "P2": {"completion_time": 7, "turnaround_time": 6, "waiting_time": 2, "response_time": 2},
                        "P3": {"completion_time": 9, "turnaround_time": 7, "waiting_time": 5, "response_time": 5},
                    },
                    "average_turnaround_time": 5.33,
                    "average_waiting_time": 2.33,
                    "average_response_time": 2.33,
                    "gantt_chart": [
                        {"pid": "P1", "start_time": 0, "end_time": 3},
                        {"pid": "P2", "start_time": 3, "end_time": 7},
                        {"pid": "P3", "start_time": 7, "end_time": 9},
                    ],
                }
            }
        }

        # Multi-process narration in section 4
        sections = [
            {
                "id": "section_3",
                "animations": [
                    "L1-1 | FadeIn | CPUExecutionVisual | viewport: dock_split_right | action: animate_dispatch(pid='P1') | on screen after: [cpu]",
                    "L1-2 | Create | GanttChartVisual | viewport: dock_inspector | action: animate_add_block(pid='P1', start_time=0, end_time=3) | on screen after: [cpu, gantt]",
                    "L1-3 | FadeIn | CPUExecutionVisual | viewport: dock_split_right | action: animate_dispatch(pid='P2') | on screen after: [cpu, gantt]",
                    "L1-4 | Create | GanttChartVisual | viewport: dock_inspector | action: animate_add_block(pid='P2', start_time=3, end_time=7) | on screen after: [cpu, gantt]",
                    "L1-5 | FadeIn | CPUExecutionVisual | viewport: dock_split_right | action: animate_dispatch(pid='P3') | on screen after: [cpu, gantt]",
                    "L1-6 | Create | GanttChartVisual | viewport: dock_inspector | action: animate_add_block(pid='P3', start_time=7, end_time=9) | on screen after: [cpu, gantt]",
                ],
                "lecture_lines": ["All processes complete their execution."],
            },
            {
                "id": "section_4",
                "animations": [
                    "L2-1 | FadeIn | MetricTableVisual | viewport: dock_inspector | action: animate_show_averages() | on screen after: [table]",
                    "L2-2 | Indicate | MetricTableVisual | viewport: dock_inspector | action: highlight_cell_avg_wt() | on screen after: [table]",
                ],
                "lecture_lines": [
                    "Process P1 completes at 3. Turnaround time for P1 is 3. Waiting time for P1 is 0. Response time for P1 is 0.",
                    "Meanwhile, P2 has completion time 7, turnaround time 6, and waiting time 2. Response time for P2 is 2.",
                    "Finally, P3 has completion time 9, turnaround time 7, and waiting time 5. Response time for P3 is 5.",
                    "The average waiting time is 2.33, and the average turnaround time is 5.33.",
                ],
            }
        ]

        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(model_dict, sections)
        assert report.is_safe_for_audio is True
        assert report.overall_status.value == "VERIFIED"
        assert len(report.discrepancies) == 0
