"""
Comprehensive integration tests for CineLearn's generic LLM-driven outline
and storyboard generation, capability discovery, action resolution, and
domain verification (validated against FCFS CPU Scheduling and cross-domain topics).
"""

import json
import pytest
from pathlib import Path

from src.domains import VisualRegistry, VisualComponent
from src.domains.action_resolver import StoryboardActionResolver, ResolvedVisualAction
from src.agent import TeachingOutline, Section, TeachingVideoAgent
from src.storyboard_fidelity_checker import (
    StoryboardFidelityChecker,
    VerificationStatus,
    FidelityCategory,
)
from prompts.base_class import TeachingScene
from prompts.stage1 import get_prompt1_outline
from prompts.stage2 import get_prompt2_storyboard
from src.grounding.verifiers.os.fcfs_verifier import FCFSTraceVerifier, FCFSSimulator


# ============================================================================
# 1. Generic Pipeline Tests
# ============================================================================

class TestGenericPipelineOutlineAndStoryboard:
    """Tests verifying the common, domain-agnostic outline and storyboard pipeline."""

    def test_component_manifest_discovery_generic(self):
        """VisualRegistry dynamically provides introspected component manifests
        containing parameter types, defaults, operations, and viewport recommendations
        without hardcoded domain logic."""
        VisualRegistry.discover_domains("src.domains")
        manifest = VisualRegistry.get_schema_manifest()

        assert "cs" in manifest
        cs_comps = {c["component_id"]: c for c in manifest["cs"]}

        # Check that the 5 OS components and general CS components are registered
        expected_comps = [
            "ProcessCardVisual",
            "ReadyQueueVisual",
            "CPUExecutionVisual",
            "GanttChartVisual",
            "MetricTableVisual",
            "GraphVisual",
            "DPTableMatrix",
        ]
        for comp_name in expected_comps:
            assert comp_name in cs_comps, f"Missing registered component: {comp_name}"
            comp_info = cs_comps[comp_name]
            assert "init_params" in comp_info
            assert "operations" in comp_info
            assert "supported_handles" in comp_info
            assert "suggested_viewport" in comp_info

        # Check parameter schema inspection on ProcessCardVisual
        pc_params = cs_comps["ProcessCardVisual"]["init_params"]
        assert "pid" in pc_params
        assert pc_params["pid"]["required"] is True
        assert "arrival_time" in pc_params
        assert str(pc_params["arrival_time"]["default"]) == "0"

    def test_stage1_prompt_includes_dynamic_capabilities(self):
        """Stage 1 prompt dynamically exposes registered domain capabilities
        without imposing hardcoded teaching beats or domain templates."""
        prompt = get_prompt1_outline(
            knowledge_point="Operating Systems FCFS CPU Scheduling",
            duration=90,
            level="undergraduate",
        )
        assert "Available Domain Visual Capabilities & Components" in prompt
        assert "ProcessCardVisual" in prompt
        assert "ReadyQueueVisual" in prompt
        assert "CPUExecutionVisual" in prompt
        assert "GanttChartVisual" in prompt
        assert "MetricTableVisual" in prompt
        # Crucial: Prompt does NOT enforce a fixed number of scenes or hardcoded FCFS beats
        assert "MANDATORY SCENES" not in prompt
        assert "FIXED_BEATS" not in prompt

    def test_stage2_prompt_includes_dynamic_manifest_and_trace(self):
        """Stage 2 prompt presents verified simulation facts and component manifests
        while giving the LLM pedagogical freedom to organize scenes and beats."""
        spec = {
            "inputs": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 4},
                    {"pid": "P2", "arrival_time": 2, "burst_time": 3},
                ]
            },
            "steps": [
                {"step_number": 1, "description": "P1 arrives and executes", "current_time": 0, "running_pid": "P1"},
            ]
        }
        _, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS")

        outline = TeachingOutline(
            topic="FCFS CPU Scheduling",
            formal_definition="First-Come First-Served scheduling executes processes in order of arrival.",
            estimated_duration=90,
            sections=[
                {"id": "sec_1", "goal": "Introduce FIFO Queue"},
                {"id": "sec_2", "goal": "Demonstrate CPU Dispatch"},
            ],
            beats=[
                {
                    "beat_id": "b1",
                    "beat_title": "Introduction",
                    "section_type": "conceptual_introduction",
                    "pedagogical_objective": "Understand FIFO queueing",
                    "concept_addressed": "FIFO Queue",
                    "visual_intent": "Display queue container",
                },
                {
                    "beat_id": "b2",
                    "beat_title": "Step 1: Simulation",
                    "section_type": "worked_demonstration",
                    "pedagogical_objective": "Dispatch process to CPU",
                    "concept_addressed": "CPU Dispatch",
                    "visual_intent": "Animate dispatch to core",
                    "step_number": 1,
                },
            ],
            worked_example_spec=spec,
            verification_metadata=report.to_dict(),
        )

        prompt = get_prompt2_storyboard(outline=outline, level="undergraduate")
        # Manifest is included
        assert "Available Visual Architecture & Registry Components" in prompt
        assert "ReadyQueueVisual" in prompt
        assert "CPUExecutionVisual" in prompt
        # Authoritative trace facts are provided
        assert "Authoritative FCFS CPU Scheduling Simulation" in prompt
        assert "P1" in prompt and "P2" in prompt
        assert "Pedagogical Organization" in prompt

    def test_outline_validation_success_and_failures(self):
        """Outline validation validates required fields, learning objective alignment,
        concept DAG acyclicity, and step bounds fail-closed."""
        # 1. Valid outline
        valid_outline = TeachingOutline(
            topic="FCFS CPU Scheduling",
            formal_definition="Non-preemptive scheduling policy.",
            estimated_duration=120,
            sections=[
                {"id": "sec_1", "goal": "Explain FIFO queue order", "pedagogical_role": "Concept intro"},
                {"id": "sec_2", "goal": "Trace process execution", "pedagogical_role": "Worked demo"},
            ],
            beats=[
                {
                    "beat_id": "b1",
                    "beat_title": "Concept A Intro",
                    "section_type": "conceptual_introduction",
                    "pedagogical_objective": "Explain FIFO",
                    "concept_addressed": "FIFO",
                    "visual_intent": "Diagram",
                },
                {
                    "beat_id": "b2",
                    "beat_title": "Concept B Demo",
                    "section_type": "worked_demonstration",
                    "pedagogical_objective": "Trace demo",
                    "concept_addressed": "CPU",
                    "visual_intent": "Diagram",
                },
            ],
        )
        is_valid, errors = valid_outline.validate_outline()
        assert is_valid is True
        assert len(errors) == 0

        # 2. Missing topic
        bad_topic_outline = TeachingOutline(
            topic="",
            formal_definition="Non-preemptive scheduling policy.",
            sections=[{"id": "sec_1", "goal": "Some goal"}],
        )
        is_valid, errors = bad_topic_outline.validate_outline()
        assert is_valid is False
        assert any("mandatory 'topic'" in e for e in errors)

        # 3. Empty sections
        bad_sections_outline = TeachingOutline(
            topic="FCFS CPU Scheduling",
            formal_definition="Non-preemptive scheduling policy.",
            sections=[],
        )
        is_valid, errors = bad_sections_outline.validate_outline()
        assert is_valid is False
        assert any("at least one section" in e for e in errors)

        # 4. Out-of-bounds worked example step reference
        bad_step_outline = TeachingOutline(
            topic="FCFS CPU Scheduling",
            formal_definition="Non-preemptive scheduling policy.",
            worked_example_spec={"steps": [{"step_number": 1}]},
            sections=[
                {"id": "sec_1", "goal": "Step 5 Demo", "worked_example_step": 5}
            ],
        )
        is_valid, errors = bad_step_outline.validate_outline()
        assert is_valid is False
        assert any("references nonexistent worked example step 5" in e for e in errors)

    def test_storyboard_action_resolver_valid_and_invalid(self):
        """StoryboardActionResolver resolves valid operations and rejects unsupported
        operations, invalid parameters, and unknown viewports fail-closed."""
        # 1. Valid operation resolution
        step_str = "L1-1 | FadeIn | ReadyQueueVisual(capacity=5) | viewport: dock_main | action: animate_enqueue(item='P1', run_time=0.5) | on screen after: [ready_queue]"
        is_valid, resolved, err = StoryboardActionResolver.resolve_visual_action(step_str)
        assert is_valid is True
        assert resolved is not None
        assert resolved.component_name == "ReadyQueueVisual"
        assert resolved.canonical_action == "animate_enqueue"
        assert resolved.kwargs.get("item") == "P1"

        # 1b. Shorthand operation resolution (enqueue -> animate_enqueue)
        step_shorthand = "L1-2 | Transform | ReadyQueueVisual() | viewport: dock_main | action: enqueue(item='P2') | on screen after: [ready_queue]"
        is_valid_s, resolved_s, err_s = StoryboardActionResolver.resolve_visual_action(step_shorthand)
        assert is_valid_s is True
        assert resolved_s is not None
        assert resolved_s.canonical_action == "animate_enqueue"

        # 2. Unsupported operation on registered component
        bad_op_step = "L1-3 | Transform | ReadyQueueVisual() | viewport: dock_main | action: unsupported_magic_jump(height=10) | on screen after: []"
        is_valid_bad, _, err_bad = StoryboardActionResolver.resolve_visual_action(bad_op_step)
        assert is_valid_bad is False
        assert "does not support operation 'unsupported_magic_jump'" in err_bad

        # 3. Unsupported parameter on registered operation
        bad_param_step = "L1-4 | Indicate | ReadyQueueVisual() | viewport: dock_main | action: animate_peek(nonexistent_param=999) | on screen after: []"
        is_valid_p, _, err_p = StoryboardActionResolver.resolve_visual_action(bad_param_step)
        assert is_valid_p is False
        assert "unsupported parameter 'nonexistent_param'" in err_p.lower()

        # 4. Non-existent component method on ProceduralComponent
        procedural_step = "L1-5 | Transform | ProceduralComponent(Diagram) | viewport: dock_main | action: calculate_shortest_path('A', 'B') | on screen after: []"
        is_valid_proc, _, err_proc = StoryboardActionResolver.resolve_visual_action(procedural_step)
        assert is_valid_proc is False
        assert "ProceduralComponent is a raw Manim wrapper" in err_proc

        # 5. Invalid viewport
        bad_dock_step = "L1-6 | FadeIn | CPUExecutionVisual() | viewport: dock_galaxy_outer | action: animate_set_idle() | on screen after: []"
        is_valid_d, _, err_d = StoryboardActionResolver.resolve_visual_action(bad_dock_step)
        assert is_valid_d is False
        assert "Unknown target viewport" in err_d


# ============================================================================
# 2. OS FCFS Storyboarding and Verification Tests
# ============================================================================

class TestFCFSSchedulingStoryboardingAndFidelity:
    """Tests verifying FCFS CPU scheduling trace grounding, storyboard fidelity,
    and visual component rendering without hardcoded beats."""

    @pytest.fixture
    def fcfs_outline_and_spec(self):
        spec = {
            "inputs": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                    {"pid": "P2", "arrival_time": 1, "burst_time": 2},
                    {"pid": "P3", "arrival_time": 5, "burst_time": 1},
                ]
            },
            "steps": [
                {"step_number": 1, "description": "P1 starts execution at 0", "current_time": 0, "running_pid": "P1"},
                {"step_number": 2, "description": "P2 arrives at 1, queues", "current_time": 1, "running_pid": "P1"},
                {"step_number": 3, "description": "P1 finishes at 3, P2 dispatches", "current_time": 3, "running_pid": "P2"},
                {"step_number": 4, "description": "P2 finishes at 5, CPU idle until P3 arrives", "current_time": 5, "running_pid": "P3"},
            ]
        }
        _, report = FCFSTraceVerifier.verify_worked_example_spec(spec, topic="FCFS")

        outline = {
            "topic": "FCFS CPU Scheduling",
            "formal_definition": "FCFS scheduling serves processes in arrival sequence.",
            "worked_example_spec": spec,
            "verification_metadata": report.to_dict(),
            "teaching_beats": [
                {"beat_title": "Overview", "section_type": "conceptual_introduction"},
                {"beat_title": "Simulation Part 1", "section_type": "worked_demonstration", "step_number": 1},
                {"beat_title": "Simulation Part 2", "section_type": "worked_demonstration", "step_number": 3},
            ]
        }
        return outline, spec, report

    def test_valid_fcfs_multi_scene_storyboard_passes_fidelity(self, fcfs_outline_and_spec):
        """A realistic 3-scene FCFS storyboard organized dynamically by the LLM
        passes fidelity verification, correctly referencing components and trace."""
        outline, spec, report = fcfs_outline_and_spec

        # Dynamic 3-scene storyboard chosen by LLM (not hardcoded template)
        storyboard = {
            "sections": [
                {
                    "id": "scene_1_intro",
                    "section_name": "Initial Dispatch and Queueing",
                    "lecture_lines": [
                        "At time 0, Process P1 arrives with burst time 3 and immediately takes the CPU.",
                        "At time 1, Process P2 arrives with burst time 2 and enters the ready queue.",
                    ],
                    "animations": [
                        "L1-1 | FadeIn | CPUExecutionVisual() | viewport: dock_split_right | action: animate_dispatch(item='P1', run_time=0.5) | on screen after: [cpu]",
                        "L2-1 | FadeIn | ReadyQueueVisual(capacity=3) | viewport: dock_split_left | action: animate_enqueue(item='P2', run_time=0.5) | on screen after: [cpu, queue]",
                        "L2-2 | FadeIn | GanttChartVisual(total_time=6) | viewport: dock_inspector | action: animate_add_block(pid='P1', start_time=0, end_time=3) | on screen after: [cpu, queue, gantt]",
                    ],
                },
                {
                    "id": "scene_2_transition",
                    "section_name": "P1 Completion and P2 Execution",
                    "lecture_lines": [
                        "At time 3, P1 finishes execution. The CPU dequeues P2 from the ready queue.",
                        "P2 executes non-preemptively until time 5.",
                    ],
                    "animations": [
                        "L1-1 | Transform | CPUExecutionVisual() | viewport: dock_split_right | action: animate_complete(pid='P1', run_time=0.5) | on screen after: [cpu, queue, gantt]",
                        "L1-2 | Transform | ReadyQueueVisual(capacity=3) | viewport: dock_split_left | action: animate_dequeue() | on screen after: [cpu, queue, gantt]",
                        "L2-1 | Transform | CPUExecutionVisual() | viewport: dock_split_right | action: animate_dispatch(item='P2', run_time=0.5) | on screen after: [cpu, queue, gantt]",
                        "L2-2 | Transform | GanttChartVisual(total_time=6) | viewport: dock_inspector | action: animate_add_block(pid='P2', start_time=3, end_time=5) | on screen after: [cpu, queue, gantt]",
                    ],
                },
                {
                    "id": "scene_3_metrics",
                    "section_name": "Completion and Accounting",
                    "lecture_lines": [
                        "At time 5, P2 completes. Its completion time is 5, turnaround time is 4, and waiting time is 2.",
                        "Finally, P3 executes from 5 to 6, completing our FCFS schedule.",
                    ],
                    "animations": [
                        "L1-1 | Transform | CPUExecutionVisual() | viewport: dock_split_right | action: animate_complete(pid='P2', run_time=0.5) | on screen after: [cpu, queue, gantt]",
                        "L1-2 | FadeIn | MetricTableVisual() | viewport: dock_inspector | action: animate_update_process_metrics(pid='P2', completion_time=5, turnaround_time=4, waiting_time=2) | on screen after: [cpu, queue, table]",
                        "L2-1 | Transform | GanttChartVisual(total_time=6) | viewport: dock_inspector | action: animate_add_block(pid='P3', start_time=5, end_time=6) | on screen after: [cpu, queue, table, gantt]",
                        "L2-2 | Transform | MetricTableVisual() | viewport: dock_inspector | action: animate_show_averages(avg_tat=3.0, avg_wt=1.0) | on screen after: [cpu, queue, table, gantt]",
                    ],
                },
            ]
        }

        checker = StoryboardFidelityChecker()
        fidelity_report = checker.check_storyboard_fidelity(outline, storyboard)

        assert fidelity_report.overall_status in [VerificationStatus.VERIFIED, VerificationStatus.NEEDS_REVIEW]
        assert fidelity_report.is_safe_for_audio is True
        assert len([d for d in fidelity_report.discrepancies if d.severity == VerificationStatus.FAILED]) == 0

    def test_fcfs_fidelity_rejects_premature_metric_leakage(self, fcfs_outline_and_spec):
        """Prematurely claiming turnaround time or waiting time before a process terminates
        fails fidelity verification fail-closed."""
        outline, spec, report = fcfs_outline_and_spec

        # Scene 1 claims P2's completion time at t=1 before it even executed!
        bad_storyboard = {
            "sections": [
                {
                    "id": "scene_1_leak",
                    "section_name": "Premature Metric Claim",
                    "lecture_lines": [
                        "At time 1, process P2 arrives. We immediately know its completion time is 5 and waiting time is 2.",
                    ],
                    "animations": [
                        "L1-1 | FadeIn | ReadyQueueVisual(capacity=3) | viewport: dock_split_left | action: animate_enqueue(item='P2') | on screen after: [queue]",
                        "L1-2 | FadeIn | MetricTableVisual() | viewport: dock_inspector | action: animate_update_process_metrics(pid='P2', completion_time=5, turnaround_time=4, waiting_time=2) | on screen after: [queue, table]",
                    ],
                }
            ]
        }

        checker = StoryboardFidelityChecker()
        fidelity_report = checker.check_storyboard_fidelity(outline, bad_storyboard)

        assert fidelity_report.overall_status == VerificationStatus.FAILED
        assert fidelity_report.is_safe_for_audio is False
        assert any(
            d.category == FidelityCategory.PREMATURE_METRIC_LEAK or "Premature" in d.actual_claim
            for d in fidelity_report.discrepancies
        )

    def test_fcfs_fidelity_rejects_fifo_order_inversion(self, fcfs_outline_and_spec):
        """Dispatching a later-arriving process before an earlier-arriving one
        violates FIFO scheduling and fails fidelity verification."""
        outline, spec, report = fcfs_outline_and_spec

        bad_order_storyboard = {
            "sections": [
                {
                    "id": "scene_1_inversion",
                    "section_name": "FIFO Inversion",
                    "lecture_lines": [
                        "At time 0, instead of P1, the CPU chooses P2 to dispatch first.",
                    ],
                    "animations": [
                        "L1-1 | FadeIn | CPUExecutionVisual() | viewport: dock_split_right | action: animate_dispatch(item='P2') | on screen after: [cpu]",
                    ],
                }
            ]
        }

        checker = StoryboardFidelityChecker()
        fidelity_report = checker.check_storyboard_fidelity(outline, bad_order_storyboard)

        assert fidelity_report.overall_status == VerificationStatus.FAILED
        assert any(
            d.category == FidelityCategory.INVALID_DISPATCH_ORDER or "FIFO" in d.actual_claim
            for d in fidelity_report.discrepancies
        )

    def test_fcfs_fidelity_rejects_preemption(self, fcfs_outline_and_spec):
        """Preempting a running process before its burst time completes violates
        FCFS non-preemptive semantics and fails verification."""
        outline, spec, report = fcfs_outline_and_spec

        preempting_storyboard = {
            "sections": [
                {
                    "id": "scene_1_preemption",
                    "section_name": "Unwarranted Preemption",
                    "lecture_lines": [
                        "At time 1, P1 is running, but the CPU preempts P1 and switches to P2.",
                    ],
                    "animations": [
                        "L1-1 | FadeIn | CPUExecutionVisual() | viewport: dock_split_right | action: animate_dispatch(item='P1') | on screen after: [cpu]",
                        "L1-2 | Transform | CPUExecutionVisual() | viewport: dock_split_right | action: animate_dispatch(item='P2') | on screen after: [cpu]",
                    ],
                }
            ]
        }

        checker = StoryboardFidelityChecker()
        fidelity_report = checker.check_storyboard_fidelity(outline, preempting_storyboard)

        assert fidelity_report.overall_status == VerificationStatus.FAILED
        assert any(
            d.category == FidelityCategory.UNAUTHORIZED_PREEMPTION or "preempt" in d.actual_claim.lower()
            for d in fidelity_report.discrepancies
        )

    def test_fcfs_fidelity_rejects_gantt_interval_mismatch(self, fcfs_outline_and_spec):
        """Gantt timeline intervals that contradict the authoritative simulation trace
        fail fidelity verification fail-closed."""
        outline, spec, report = fcfs_outline_and_spec

        # Authoritative: P1 runs [0, 3]. Bad storyboard claims [0, 6].
        bad_gantt_storyboard = {
            "sections": [
                {
                    "id": "scene_1_bad_gantt",
                    "section_name": "Corrupted Gantt Interval",
                    "lecture_lines": [
                        "Process P1 starts at 0 and runs until time 6.",
                    ],
                    "animations": [
                        "L1-1 | FadeIn | GanttChartVisual(total_time=6) | viewport: dock_inspector | action: animate_add_block(pid='P1', start_time=0, end_time=6) | on screen after: [gantt]",
                    ],
                }
            ]
        }

        checker = StoryboardFidelityChecker()
        fidelity_report = checker.check_storyboard_fidelity(outline, bad_gantt_storyboard)

        assert fidelity_report.overall_status == VerificationStatus.FAILED
        assert any(
            d.category == FidelityCategory.GANTT_TIMELINE_MISMATCH or "Gantt" in d.actual_claim
            for d in fidelity_report.discrepancies
        )

    def test_fcfs_manim_scene_execution_and_rendering(self, tmp_path):
        """Constructs and executes a Manim scene containing all five registered OS
        visual components driven by OSTraceVisualAdapter, ensuring full rendering sanity."""
        from manim import tempconfig, FadeIn
        from src.domains.cs.os_scheduling import (
            ProcessCardVisual,
            ReadyQueueVisual,
            CPUExecutionVisual,
            GanttChartVisual,
            MetricTableVisual,
            OSTraceVisualAdapter,
        )

        processes = [
            {"pid": "P1", "arrival_time": 0, "burst_time": 2},
            {"pid": "P2", "arrival_time": 1, "burst_time": 1},
        ]
        result = FCFSSimulator.run(processes)
        adapter = OSTraceVisualAdapter(result.computation_trace)

        class IntegratedFCFSTestScene(TeachingScene):
            def construct(self):
                self.setup_layout(
                    "FCFS Execution Demonstration",
                    [
                        "Process P1 arrives at t=0 and enters the Ready Queue",
                        "Dispatcher allocates CPU to P1 for non-preemptive burst",
                    ],
                )

                queue = ReadyQueueVisual(capacity=3, slot_width=1.0, slot_height=0.8)
                cpu = CPUExecutionVisual(width=2.2, height=1.7)
                gantt = GanttChartVisual(total_time=adapter.get_total_simulation_time(), width=5.6, height=0.8)
                table = MetricTableVisual(processes=adapter.get_process_specs(), width=5.6, height=1.4)

                self.dock_split_left(queue)
                self.dock_split_right(cpu)
                self.dock_inspector(gantt)

                p1_card = ProcessCardVisual(pid="P1", arrival_time=0, burst_time=2, state="READY")

                self.play(FadeIn(queue), FadeIn(cpu), FadeIn(gantt))
                self.play(self.activate_caption(0))
                self.play(queue.animate_enqueue(p1_card))
                self.play(queue.animate_peek())

                self.play(self.activate_caption(1))
                self.play(
                    queue.animate_dequeue(),
                    cpu.animate_dispatch(p1_card),
                )
                self.play(
                    cpu.animate_execute_tick(remaining_time=1),
                    gantt.animate_add_block("P1", 0, 2),
                )
                self.play(cpu.animate_complete())

        with tempconfig({
            "quality": "low_quality",
            "preview": False,
            "format": "mp4",
            "media_dir": str(tmp_path),
            "disable_caching": True,
        }):
            scene = IntegratedFCFSTestScene()
            scene.render()


# ============================================================================
# 3. Cross-Domain Regression Tests
# ============================================================================

class TestCrossDomainRegressionAndQualitativeTopics:
    """Verifies that non-OS algorithmic topics (Dijkstra, Knapsack, Binary Search)
    and qualitative non-computational topics remain completely supported."""

    def test_dijkstra_regression(self):
        """Existing Dijkstra trace grounding and storyboard fidelity remain fully functional."""
        from tests.test_dijkstra_storyboard_fidelity import (
            make_verified_pilot_outline,
            make_valid_pilot_storyboard,
        )

        outline = make_verified_pilot_outline()
        sb = make_valid_pilot_storyboard()
        checker = StoryboardFidelityChecker()

        rep = checker.check_storyboard_fidelity(outline, sb)
        assert rep is not None
        assert rep.overall_status == VerificationStatus.VERIFIED
        assert rep.is_safe_for_audio is True
        assert len(rep.discrepancies) == 0

    def test_qualitative_topic_without_simulator(self):
        """Qualitative topics (e.g. Photosynthesis or Deadlock Conditions) that do not
        have deterministic simulators still validate outline and storyboard cleanly."""
        qualitative_outline = TeachingOutline(
            topic="Deadlock Conditions in Operating Systems",
            formal_definition="A deadlock is a state where a set of processes are blocked because each is holding a resource and waiting for another.",
            estimated_duration=90,
            sections=[
                {"id": "sec_1", "goal": "Define the four Coffman deadlock conditions", "pedagogical_role": "Concept intro"},
                {"id": "sec_2", "goal": "Analyze Mutual Exclusion and Hold & Wait", "pedagogical_role": "Detailed explanation"},
            ],
            beats=[
                {
                    "beat_id": "b1",
                    "beat_title": "Introduction to Deadlocks",
                    "section_type": "conceptual_introduction",
                    "pedagogical_objective": "Understand resource deadlock",
                    "concept_addressed": "Deadlock Definition",
                    "visual_intent": "Diagram",
                },
                {
                    "beat_id": "b2",
                    "beat_title": "The Four Conditions",
                    "section_type": "conceptual_introduction",
                    "pedagogical_objective": "Learn Coffman conditions",
                    "concept_addressed": "Coffman Conditions",
                    "visual_intent": "Diagram",
                },
            ],
            # No worked_example_spec!
            worked_example_spec={},
        )

        is_valid, errors = qualitative_outline.validate_outline()
        assert is_valid is True
        assert len(errors) == 0

        # Storyboard fidelity for qualitative topic without simulator
        qualitative_storyboard = {
            "sections": [
                {
                    "id": "intro",
                    "section_name": "Introduction to Deadlocks",
                    "lecture_lines": [
                        "A deadlock is a situation where processes cannot proceed because each holds a resource needed by another."
                    ],
                    "animations": [
                        "L1-1 | FadeIn | ProceduralComponent(SetupDiagram) | viewport: dock_main | action: initialize_structure() | on screen after: [primary visual]",
                    ],
                }
            ]
        }

        checker = StoryboardFidelityChecker()
        rep = checker.check_storyboard_fidelity(qualitative_outline.to_dict(), qualitative_storyboard)
        assert rep is not None
        assert rep.overall_status in [VerificationStatus.VERIFIED, VerificationStatus.NEEDS_REVIEW, VerificationStatus.NOT_APPLICABLE]
        assert rep.is_safe_for_audio is True
