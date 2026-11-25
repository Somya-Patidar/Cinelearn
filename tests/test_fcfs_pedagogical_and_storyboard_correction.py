"""
Unit & Integration Tests for CineLearn Phase 1:
FCFS Pedagogical and Storyboard Correction.

Covers:
1. Input specification introduced before execution vs. execution starting without input specification.
2. Input data consistency between input specification, Gantt intervals, and metrics.
3. Detection of duplicate consecutive state transitions.
4. Detection of invalid repeated dispatch.
5. Explicit execution coverage for P1, P2, and P3.
6. Detection of unsupported component actions (highlight_pid_at, highlight_scheduler, etc.).
7. Detection of excessive generic highlighting.
8. Detection of conflicting primary-visual viewport assignments (MetricTableVisual in dock_inspector).
9. Decoupling and regression: non-FCFS topics (e.g. Dijkstra, general outline) are completely unaffected.
"""

from __future__ import annotations

import copy
import pytest
from typing import Any, Dict, List

from fcfs_verifier import (
    ProcessSpec,
    FCFSSimulator,
    FCFSTraceVerifier,
)
from worked_example_verifier import WorkedExampleVerifier, VerificationStatus
from storyboard_fidelity_checker import (
    StoryboardFidelityChecker,
    FidelityCategory,
)


@pytest.fixture
def canonical_fcfs_outline() -> Dict[str, Any]:
    """Returns an authoritative FCFS outline verified by FCFSTraceVerifier."""
    spec = {
        "inputs": {
            "processes": [
                {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                {"pid": "P2", "arrival_time": 1, "burst_time": 4},
                {"pid": "P3", "arrival_time": 2, "burst_time": 2},
            ]
        },
        "steps": [
            {"step_number": 1, "current_time": 0, "running_pid": "P1"},
            {"step_number": 2, "current_time": 3, "running_pid": "P2"},
            {"step_number": 3, "current_time": 7, "running_pid": "P3"},
        ],
    }
    out_spec, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS CPU Scheduling")
    return {
        "topic": "FCFS CPU Scheduling",
        "worked_example_spec": out_spec,
        "verification_metadata": report.to_dict(),
        "sections": [
            {"id": "sec_1", "title": "Motivation & Overview"},
            {"id": "sec_2", "title": "Process State & Lifecycle"},
            {"id": "sec_3", "title": "Process Input Specification: P1, P2, P3 Arrival and Burst Times"},
            {"id": "sec_4", "title": "Ready Queue Setup"},
            {"id": "sec_5", "title": "Scheduler & Dispatcher"},
            {"id": "sec_6", "title": "First-Come First-Served Principle"},
            {"id": "sec_7", "title": "Non-Preemptive Execution Rule"},
            {"id": "sec_8", "title": "Chronological Execution Trace"},
            {"id": "sec_9", "title": "Gantt Chart Construction"},
            {"id": "sec_10", "title": "Metric Derivation: CT, TAT, WT, RT"},
            {"id": "sec_11", "title": "Summary Averages Calculation"},
            {"id": "sec_12", "title": "FCFS Limitations & The Convoy Effect"},
        ],
    }


@pytest.fixture
def valid_fcfs_storyboard() -> Dict[str, Any]:
    """Returns a fully valid, grounded 12-beat FCFS storyboard using standard 6-field syntax."""
    return {
        "sections": [
            {
                "id": "sec_1",
                "title": "Motivation & Overview",
                "lecture_lines": ["Why CPU scheduling is needed to maximize CPU utilization."],
                "animations": ["L1-1 | Write | Text('CPU Scheduling') | viewport: dock_main | action: setup | on screen after: [title]"],
            },
            {
                "id": "sec_2",
                "title": "Process State & Lifecycle",
                "lecture_lines": ["A process moves through NEW, READY, RUNNING, and TERMINATED states."],
                "animations": ["L2-1 | Transform | ProcessCardVisual(pid='P1', arrival_time=0, burst_time=3) | viewport: dock_main | action: animate_set_state(new_state='NEW') | on screen after: [p1_card]"],
            },
            {
                "id": "sec_3",
                "title": "Process Input Specification",
                "lecture_lines": [
                    "We consider three processes: P1 with arrival time 0 and burst time 3, "
                    "P2 with arrival time 1 and burst time 4, and P3 with arrival time 2 and burst time 2."
                ],
                "animations": [
                    "L3-1 | FadeIn | ProcessCardVisual(pid='P1', arrival_time=0, burst_time=3) | viewport: dock_split_left | action: setup | on screen after: [p1_card]",
                    "L3-2 | FadeIn | ProcessCardVisual(pid='P2', arrival_time=1, burst_time=4) | viewport: dock_main | action: setup | on screen after: [p1_card, p2_card]",
                    "L3-3 | FadeIn | ProcessCardVisual(pid='P3', arrival_time=2, burst_time=2) | viewport: dock_split_right | action: setup | on screen after: [p1_card, p2_card, p3_card]",
                ],
            },
            {
                "id": "sec_4",
                "title": "Ready Queue Setup",
                "lecture_lines": ["At time 0, P1 arrives and enters the Ready Queue."],
                "animations": ["L4-1 | Transform | ReadyQueueVisual() | viewport: dock_main | action: animate_enqueue(item='P1') | on screen after: [queue]"],
            },
            {
                "id": "sec_5",
                "title": "Scheduler & Dispatcher",
                "lecture_lines": ["The short-term scheduler selects P1 and the dispatcher allocates CPU to P1."],
                "animations": [
                    "L5-1 | Transform | ReadyQueueVisual() | viewport: dock_main | action: animate_dequeue() | on screen after: [queue]",
                    "L5-2 | Transform | CPUExecutionVisual() | viewport: dock_main | action: animate_dispatch(item='P1') | on screen after: [cpu]",
                ],
            },
            {
                "id": "sec_6",
                "title": "First-Come First-Served Principle",
                "lecture_lines": ["FCFS selects the process with the earliest arrival time."],
                "animations": ["L6-1 | Indicate | ProcessCardVisual(pid='P1', arrival_time=0, burst_time=3) | viewport: dock_main | action: animate_highlight() | on screen after: [p1_card]"],
            },
            {
                "id": "sec_7",
                "title": "Non-Preemptive Execution Rule",
                "lecture_lines": ["FCFS is strictly non-preemptive. Once P1 gets CPU, it runs until burst completion."],
                "animations": ["L7-1 | Transform | CPUExecutionVisual() | viewport: dock_main | action: animate_execute_tick(ticks=1) | on screen after: [cpu]"],
            },
            {
                "id": "sec_8",
                "title": "Chronological Execution Trace",
                "lecture_lines": [
                    "P1 completes at time 3. Next, P2 runs from 3 to 7. Finally, P3 runs from 7 to 9."
                ],
                "animations": [
                    "L8-1 | Transform | CPUExecutionVisual() | viewport: dock_main | action: animate_complete(pid='P1') | on screen after: [cpu]",
                    "L8-2 | Transform | CPUExecutionVisual() | viewport: dock_main | action: animate_dispatch(item='P2') | on screen after: [cpu]",
                    "L8-3 | Transform | CPUExecutionVisual() | viewport: dock_main | action: animate_complete(pid='P2') | on screen after: [cpu]",
                    "L8-4 | Transform | CPUExecutionVisual() | viewport: dock_main | action: animate_dispatch(item='P3') | on screen after: [cpu]",
                    "L8-5 | Transform | CPUExecutionVisual() | viewport: dock_main | action: animate_complete(pid='P3') | on screen after: [cpu]",
                ],
            },
            {
                "id": "sec_9",
                "title": "Gantt Chart Construction",
                "lecture_lines": [
                    "The Gantt chart shows P1 from 0 to 3, P2 from 3 to 7, and P3 from 7 to 9."
                ],
                "animations": [
                    "L9-1 | Transform | GanttChartVisual() | viewport: dock_main | action: animate_add_block(pid='P1', start_time=0, end_time=3) | on screen after: [gantt]",
                    "L9-2 | Transform | GanttChartVisual() | viewport: dock_main | action: animate_add_block(pid='P2', start_time=3, end_time=7) | on screen after: [gantt]",
                    "L9-3 | Transform | GanttChartVisual() | viewport: dock_main | action: animate_add_block(pid='P3', start_time=7, end_time=9) | on screen after: [gantt]",
                ],
            },
            {
                "id": "sec_10",
                "title": "Metric Derivation",
                "lecture_lines": [
                    "P1 has CT 3, TAT 3, WT 0. P2 has CT 7, TAT 6, WT 2. P3 has CT 9, TAT 7, WT 5."
                ],
                "animations": [
                    "L10-1 | Transform | MetricTableVisual() | viewport: dock_main | action: animate_highlight_row(row=0) | on screen after: [table]",
                ],
            },
            {
                "id": "sec_11",
                "title": "Summary Averages Calculation",
                "lecture_lines": [
                    "Average Turnaround Time = 5.33, and Average Waiting Time = 2.33."
                ],
                "animations": [
                    "L11-1 | Transform | MetricTableVisual() | viewport: dock_main | action: animate_show_averages() | on screen after: [table]",
                ],
            },
            {
                "id": "sec_12",
                "title": "FCFS Limitations & The Convoy Effect",
                "lecture_lines": ["Short processes stuck behind long CPU bursts experience the Convoy Effect."],
                "animations": ["L12-1 | Write | Text('Convoy Effect') | viewport: dock_main | action: setup | on screen after: [conclusion]"],
            },
        ]
    }


class TestFCFSPedagogicalAndStoryboardCorrection:
    """Deterministic validation test suite for Phase 1 FCFS corrections."""

    def test_1a_input_specification_before_execution_passes(self, canonical_fcfs_outline, valid_fcfs_storyboard):
        """Test 1A: Introducing input specification before execution passes without MISSING_INPUT_SPECIFICATION."""
        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(canonical_fcfs_outline, valid_fcfs_storyboard)
        missing_input_discs = [d for d in report.discrepancies if d.category == FidelityCategory.MISSING_INPUT_SPECIFICATION]
        assert len(missing_input_discs) == 0
        assert report.is_safe_for_audio is True

    def test_1b_execution_starting_without_input_specification_fails(self, canonical_fcfs_outline, valid_fcfs_storyboard):
        """Test 1B: Execution starting before input specification is flagged with MISSING_INPUT_SPECIFICATION."""
        flawed_sb = copy.deepcopy(valid_fcfs_storyboard)
        # Remove input specification from sec_3 and jump straight into dispatching in sec_2
        flawed_sb["sections"][1]["animations"] = [
            "step 1: cpu.animate_dispatch(item='P1') | write | (0, 0) | (0, 0) | None | None"
        ]
        flawed_sb["sections"][2]["title"] = "General Discussion"
        flawed_sb["sections"][2]["lecture_lines"] = ["Processes run on the CPU."]
        flawed_sb["sections"][2]["animations"] = []

        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(canonical_fcfs_outline, flawed_sb)
        missing_input_discs = [d for d in report.discrepancies if d.category == FidelityCategory.MISSING_INPUT_SPECIFICATION]
        assert len(missing_input_discs) > 0
        assert report.is_safe_for_audio is False
        assert any("without prior input specification" in d.actual_claim for d in missing_input_discs)

    def test_2_consistency_between_inputs_and_gantt_intervals(self, canonical_fcfs_outline, valid_fcfs_storyboard):
        """Test 2: Gantt interval inconsistent with authoritative simulation trace is flagged as GANTT_TIMELINE_MISMATCH."""
        flawed_sb = copy.deepcopy(valid_fcfs_storyboard)
        # Tamper P1 Gantt block: should be [0, 3], but claimed [0, 5]
        flawed_sb["sections"][8]["animations"] = [
            "step 1: gantt.animate_add_block(pid='P1', start_time=0, end_time=5) | write | (0, 0) | (0, 0) | None | None",
            "step 2: gantt.animate_add_block(pid='P2', start_time=5, end_time=9) | write | (0, 0) | (0, 0) | None | None",
            "step 3: gantt.animate_add_block(pid='P3', start_time=9, end_time=11) | write | (0, 0) | (0, 0) | None | None",
        ]
        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(canonical_fcfs_outline, flawed_sb)
        mismatch_discs = [d for d in report.discrepancies if d.category == FidelityCategory.GANTT_TIMELINE_MISMATCH]
        assert len(mismatch_discs) > 0
        assert report.is_safe_for_audio is False
        assert any("P1" in d.actual_claim for d in mismatch_discs)

    def test_3_detection_of_duplicate_consecutive_state_transitions(self, canonical_fcfs_outline, valid_fcfs_storyboard):
        """Test 3: Calling set_state with identical state consecutively is flagged as INVALID_STATE_TRANSITION."""
        flawed_sb = copy.deepcopy(valid_fcfs_storyboard)
        # Repeat RUNNING consecutively on P1
        flawed_sb["sections"][4]["animations"] = [
            "step 1: p1_card.set_state('RUNNING') | write | (0, 0) | (0, 0) | None | None",
            "step 2: p1_card.set_state('RUNNING') | write | (0, 0) | (0, 0) | None | None",
        ]
        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(canonical_fcfs_outline, flawed_sb)
        state_discs = [d for d in report.discrepancies if d.category == FidelityCategory.INVALID_STATE_TRANSITION]
        assert len(state_discs) > 0
        assert report.is_safe_for_audio is False
        assert any("set_state('RUNNING') called while already in 'RUNNING'" in d.actual_claim for d in state_discs)

    def test_4a_detection_of_invalid_repeated_dispatch_running_process(self, canonical_fcfs_outline, valid_fcfs_storyboard):
        """Test 4A: Repeatedly dispatching an already-running process is flagged as INVALID_REPEATED_DISPATCH."""
        flawed_sb = copy.deepcopy(valid_fcfs_storyboard)
        # Dispatch P1 twice without completing
        flawed_sb["sections"][4]["animations"] = [
            "step 1: cpu.animate_dispatch(item='P1') | write | (0, 0) | (0, 0) | None | None",
            "step 2: cpu.animate_dispatch(item='P1') | write | (0, 0) | (0, 0) | None | None",
        ]
        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(canonical_fcfs_outline, flawed_sb)
        disp_discs = [d for d in report.discrepancies if d.category == FidelityCategory.INVALID_REPEATED_DISPATCH]
        assert len(disp_discs) > 0
        assert report.is_safe_for_audio is False
        assert any("Repeated dispatch of running process 'P1'" in d.actual_claim for d in disp_discs)

    def test_4b_detection_of_invalid_repeated_dispatch_terminated_process(self, canonical_fcfs_outline, valid_fcfs_storyboard):
        """Test 4B: Dispatching an already-terminated process is flagged as INVALID_REPEATED_DISPATCH."""
        flawed_sb = copy.deepcopy(valid_fcfs_storyboard)
        # P1 completes in sec_8, then dispatch P1 again in sec_8
        flawed_sb["sections"][7]["animations"] = [
            "step 1: cpu.animate_complete(pid='P1') | write | (0, 0) | (0, 0) | None | None",
            "step 2: cpu.animate_dispatch(item='P1') | write | (0, 0) | (0, 0) | None | None",
        ]
        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(canonical_fcfs_outline, flawed_sb)
        disp_discs = [d for d in report.discrepancies if d.category == FidelityCategory.INVALID_REPEATED_DISPATCH]
        assert len(disp_discs) > 0
        assert report.is_safe_for_audio is False
        assert any("Repeated dispatch of terminated process 'P1'" in d.actual_claim for d in disp_discs)

    def test_5_explicit_execution_coverage_for_all_processes(self, canonical_fcfs_outline, valid_fcfs_storyboard):
        """Test 5: Omitting P2 or P3 from execution is flagged as INCOMPLETE_PROCESS_COVERAGE."""
        flawed_sb = copy.deepcopy(valid_fcfs_storyboard)
        # Only execute P1 in sec_8, omitting P2 and P3 entirely
        flawed_sb["sections"][7]["animations"] = [
            "step 1: cpu.animate_complete(pid='P1') | write | (0, 0) | (0, 0) | None | None",
        ]
        # Only add P1 to Gantt chart in sec_9
        flawed_sb["sections"][8]["animations"] = [
            "step 1: gantt.animate_add_block(pid='P1', start_time=0, end_time=3) | write | (0, 0) | (0, 0) | None | None",
        ]
        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(canonical_fcfs_outline, flawed_sb)
        cov_discs = [d for d in report.discrepancies if d.category == FidelityCategory.INCOMPLETE_PROCESS_COVERAGE]
        assert len(cov_discs) > 0
        assert report.is_safe_for_audio is False
        assert any("P2" in d.actual_claim and "P3" in d.actual_claim for d in cov_discs)

    def test_6_detection_of_unsupported_component_actions(self, canonical_fcfs_outline, valid_fcfs_storyboard):
        """Test 6: Storyboard referencing unsupported methods is flagged as UNSUPPORTED_OPERATION."""
        flawed_sb = copy.deepcopy(valid_fcfs_storyboard)
        flawed_sb["sections"][2]["animations"].append(
            "step 4: card.highlight_pid_at(0) | write | (0, 0) | (0, 0) | None | None"
        )
        flawed_sb["sections"][4]["animations"].append(
            "step 3: cpu.highlight_scheduler() | write | (0, 0) | (0, 0) | None | None"
        )
        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(canonical_fcfs_outline, flawed_sb)
        unsupp_discs = [d for d in report.discrepancies if d.category == FidelityCategory.UNSUPPORTED_OPERATION]
        assert len(unsupp_discs) >= 2
        assert report.is_safe_for_audio is False
        reasons = [d.reason for d in unsupp_discs]
        assert any("highlight_pid_at" in r for r in reasons)
        assert any("highlight_scheduler" in r for r in reasons)

    def test_7_detection_of_excessive_generic_highlighting(self, canonical_fcfs_outline, valid_fcfs_storyboard):
        """Test 7: >= 3 consecutive generic Indicate/highlight actions flagged as EXCESSIVE_HIGHLIGHTING."""
        flawed_sb = copy.deepcopy(valid_fcfs_storyboard)
        flawed_sb["sections"][5]["animations"] = [
            "step 1: card.animate_highlight() | write | (0, 0) | (0, 0) | None | None",
            "step 2: Indicate(card) | write | (0, 0) | (0, 0) | None | None",
            "step 3: Indicate(cpu) | write | (0, 0) | (0, 0) | None | None",
        ]
        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(canonical_fcfs_outline, flawed_sb)
        hl_discs = [d for d in report.discrepancies if d.category == FidelityCategory.EXCESSIVE_HIGHLIGHTING]
        assert len(hl_discs) > 0
        assert report.is_safe_for_audio is False
        assert any("generic highlight" in d.actual_claim for d in hl_discs)

    def test_8_detection_of_conflicting_viewport_assignments(self, canonical_fcfs_outline, valid_fcfs_storyboard):
        """Test 8: Docking both MetricTableVisual and GanttChartVisual to dock_inspector is flagged as VIEWPORT_CONFLICT."""
        flawed_sb = copy.deepcopy(valid_fcfs_storyboard)
        flawed_sb["sections"][9]["animations"] = [
            "L10-1 | FadeIn | MetricTableVisual() | viewport: dock_inspector | action: setup | on screen after: [table]",
            "L10-2 | FadeIn | GanttChartVisual() | viewport: dock_inspector | action: setup | on screen after: [table, gantt]",
        ]
        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(canonical_fcfs_outline, flawed_sb)
        vp_discs = [d for d in report.discrepancies if d.category == FidelityCategory.VIEWPORT_CONFLICT]
        assert len(vp_discs) > 0
        assert report.overall_status == VerificationStatus.NEEDS_REVIEW
        assert any("dock_inspector" in d.actual_claim for d in vp_discs)

    def test_9_non_fcfs_topics_unaffected_by_fcfs_rules(self):
        """Test 9: Verifies that non-FCFS domains (e.g. Dijkstra, general math) are unaffected."""
        dijkstra_outline = {
            "topic": "Dijkstra Shortest Path",
            "learning_level": "intermediate",
            "sections": [
                {"id": "sec_1", "title": "Sec 1: Introduction to Dijkstra"},
                {"id": "sec_2", "title": "Sec 2: Relaxing Edges"},
            ],
            "worked_example_spec": {
                "inputs": {
                    "graph": {"A": [("B", 1), ("C", 4)], "B": [("C", 2)], "C": []},
                    "start_node": "A",
                },
                "steps": [
                    {"step_number": 1, "description": "Visit A, distance 0", "current_node": "A"},
                ],
                "verification_metadata": {"overall_status": "VERIFIED"},
            },
        }
        dijkstra_sb = {
            "sections": [
                {
                    "id": "sec_1",
                    "title": "Sec 1: Introduction to Dijkstra",
                    "lecture_lines": ["Dijkstra's algorithm finds shortest paths from a single source."],
                    "animations": ["L1-1 | Create | GraphVisual() | viewport: dock_main | action: setup | on screen after: [graph]"],
                },
                {
                    "id": "sec_2",
                    "title": "Sec 2: Relaxing Edges",
                    "lecture_lines": ["We relax edge (A, B) with weight 1."],
                    "animations": ["L2-1 | Transform | GraphVisual() | viewport: dock_main | action: animate_relax_edge(u='A', v='B') | on screen after: [graph]"],
                },
            ]
        }
        checker = StoryboardFidelityChecker()
        report = checker.check_storyboard_fidelity(dijkstra_outline, dijkstra_sb)
        # Should not raise any FCFS-specific discrepancies
        fcfs_cats = {
            FidelityCategory.MISSING_INPUT_SPECIFICATION,
            FidelityCategory.INVALID_STATE_TRANSITION,
            FidelityCategory.INVALID_REPEATED_DISPATCH,
            FidelityCategory.INCOMPLETE_PROCESS_COVERAGE,
            FidelityCategory.EXCESSIVE_HIGHLIGHTING,
            FidelityCategory.VIEWPORT_CONFLICT,
            FidelityCategory.UNAUTHORIZED_PREEMPTION,
            FidelityCategory.GANTT_TIMELINE_MISMATCH,
        }
        assert not any(d.category in fcfs_cats for d in report.discrepancies)
        assert report.is_safe_for_audio is True
