"""
Comprehensive Unit & Integration Test Suite for OS CPU Scheduling Visual Components
===================================================================================
Tests:
- ProcessCardVisual (PCB card, state transitions, remaining time updates)
- ReadyQueueVisual (FIFO queue, enqueue, dequeue, peek, slot anchors)
- CPUExecutionVisual (CPU core, dispatch, execute tick, complete, idle)
- GanttChartVisual (Timeline intervals, proportional widths, idle gaps, time marker)
- MetricTableVisual (Accounting matrix, honest non-leakage, metric updates, averages)
- OSTraceVisualAdapter (Trace-to-component mapping, step deltas, batch sync)
- VisualRegistry discovery and manifest schema generation
- Viewport layout docking and boundary contracts
- Standalone Manim scene rendering to MP4
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure CineLearn root is in sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
import numpy as np
from manim import Scene, tempconfig, FadeIn, Animation, AnimationGroup

from src.domains import VisualRegistry
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
    FCFSSimulationResult,
)
from prompts import base_class

# Dynamically extract TeachingScene from base_class template
_scope = {}
exec(base_class.base_class if hasattr(base_class, "base_class") else base_class, _scope)
TeachingScene = _scope["TeachingScene"]


@pytest.fixture(autouse=True)
def ensure_cs_discovered():
    VisualRegistry.discover_domains("src.domains")


# ============================================================================
# 1. ProcessCardVisual Tests
# ============================================================================

class TestProcessCardVisual:
    def test_registration(self):
        cls = VisualRegistry.get("cs", "ProcessCardVisual")
        assert cls is ProcessCardVisual
        manifest = ProcessCardVisual.get_manifest()
        assert manifest["component_id"] == "ProcessCardVisual"
        assert manifest["domain"] == "cs"
        assert "animate_set_state" in manifest["semantic_actions"]
        assert "animate_update_remaining_time" in manifest["semantic_actions"]
        assert "animate_highlight" in manifest["semantic_actions"]

    def test_initialization_defaults_and_states(self):
        for state in ["NEW", "READY", "RUNNING", "TERMINATED", "NOT_ARRIVED"]:
            card = ProcessCardVisual(
                pid="P1",
                arrival_time=0,
                burst_time=4,
                state=state,
            )
            assert card.pid == "P1"
            assert card.arrival_time == 0
            assert card.burst_time == 4
            assert card.remaining_time == 4  # Defaulted to burst_time
            assert card.current_state == state

    def test_semantic_handles(self):
        card = ProcessCardVisual(pid="P2", arrival_time=1, burst_time=5, remaining_time=3, state="READY")
        assert card.get_semantic_handle("card_body") is not None
        assert card.get_semantic_handle("pid_label") is not None
        assert card.get_semantic_handle("state_badge") is not None
        assert card.get_semantic_handle("state_badge_box") is not None
        assert card.get_semantic_handle("state_badge_text") is not None
        assert card.get_semantic_handle("remaining_time_label") is not None
        assert card.get_semantic_handle("at_label") is not None
        assert card.get_semantic_handle("bt_label") is not None

    def test_animate_set_state(self):
        card = ProcessCardVisual(pid="P1", state="NEW")
        anim_group = card.animate_set_state("RUNNING")
        assert isinstance(anim_group, AnimationGroup)
        assert len(anim_group.animations) >= 3
        assert card.current_state == "RUNNING"

    def test_animate_update_remaining_time(self):
        card = ProcessCardVisual(pid="P1", burst_time=4, remaining_time=4)
        anim_group = card.animate_update_remaining_time(3)
        assert isinstance(anim_group, AnimationGroup)
        assert card.remaining_time == 3

    def test_animate_highlight(self):
        card = ProcessCardVisual(pid="P1")
        anim = card.animate_highlight()
        assert isinstance(anim, Animation)

    def test_instant_updates(self):
        card = ProcessCardVisual(pid="P1", state="NEW", burst_time=4)
        card.set_state_instant("READY")
        assert card.current_state == "READY"
        card.set_remaining_time_instant(2)
        assert card.remaining_time == 2


# ============================================================================
# 2. ReadyQueueVisual Tests
# ============================================================================

class TestReadyQueueVisual:
    def test_registration(self):
        cls = VisualRegistry.get("cs", "ReadyQueueVisual")
        assert cls is ReadyQueueVisual
        manifest = ReadyQueueVisual.get_manifest()
        assert manifest["component_id"] == "ReadyQueueVisual"
        assert "animate_enqueue" in manifest["semantic_actions"]
        assert "animate_dequeue" in manifest["semantic_actions"]
        assert "animate_peek" in manifest["semantic_actions"]

    def test_initialization_and_handles(self):
        queue = ReadyQueueVisual(capacity=4)
        assert queue.capacity == 4
        assert queue.get_semantic_handle("container") is not None
        assert queue.get_semantic_handle("title") is not None
        assert queue.get_semantic_handle("head") is not None
        assert queue.get_semantic_handle("tail") is not None
        for i in range(4):
            assert queue.get_semantic_handle(f"slot_{i}") is not None

    def test_animate_enqueue_and_dequeue(self):
        queue = ReadyQueueVisual(capacity=3)
        card1 = ProcessCardVisual(pid="P1", state="READY")
        card2 = ProcessCardVisual(pid="P2", state="READY")

        anim_enq1 = queue.animate_enqueue(card1)
        assert isinstance(anim_enq1, AnimationGroup)
        assert len(queue.queue_items) == 1
        assert queue.get_head_item() is card1

        anim_enq2 = queue.animate_enqueue(card2)
        assert isinstance(anim_enq2, AnimationGroup)
        assert len(queue.queue_items) == 2

        # Dequeue card1
        anim_deq = queue.animate_dequeue()
        assert isinstance(anim_deq, AnimationGroup)
        assert len(queue.queue_items) == 1
        assert queue.get_head_item() is card2

    def test_animate_peek(self):
        queue = ReadyQueueVisual(capacity=3)
        anim_empty = queue.animate_peek()
        assert isinstance(anim_empty, Animation)

        card = ProcessCardVisual(pid="P1", state="READY")
        queue.animate_enqueue(card)
        anim_peek = queue.animate_peek()
        assert isinstance(anim_peek, Animation)

    def test_set_queue_pids(self):
        queue = ReadyQueueVisual(capacity=4)
        queue.set_queue_pids(["P1", "P2", "P3"])
        assert len(queue.queue_items) == 3
        assert queue.queue_items[0].pid == "P1"
        assert queue.queue_items[1].pid == "P2"
        assert queue.queue_items[2].pid == "P3"


# ============================================================================
# 3. CPUExecutionVisual Tests
# ============================================================================

class TestCPUExecutionVisual:
    def test_registration(self):
        cls = VisualRegistry.get("cs", "CPUExecutionVisual")
        assert cls is CPUExecutionVisual
        manifest = CPUExecutionVisual.get_manifest()
        assert manifest["component_id"] == "CPUExecutionVisual"
        assert "animate_dispatch" in manifest["semantic_actions"]
        assert "animate_execute_tick" in manifest["semantic_actions"]
        assert "animate_complete" in manifest["semantic_actions"]
        assert "animate_set_idle" in manifest["semantic_actions"]

    def test_initialization_and_handles(self):
        cpu = CPUExecutionVisual()
        assert cpu.cpu_state == "IDLE"
        assert cpu.current_process is None
        assert cpu.get_semantic_handle("core_box") is not None
        assert cpu.get_semantic_handle("title") is not None
        assert cpu.get_semantic_handle("status_badge") is not None
        assert cpu.get_semantic_handle("slot") is not None

    def test_animate_dispatch(self):
        cpu = CPUExecutionVisual()
        card = ProcessCardVisual(pid="P1", burst_time=3, state="READY")
        anim_disp = cpu.animate_dispatch(card)
        assert isinstance(anim_disp, AnimationGroup)
        assert cpu.cpu_state == "RUNNING"
        assert cpu.current_process is card

    def test_animate_execute_tick(self):
        cpu = CPUExecutionVisual()
        card = ProcessCardVisual(pid="P1", burst_time=3, remaining_time=3, state="READY")
        cpu.animate_dispatch(card)
        anim_tick = cpu.animate_execute_tick(remaining_time=2)
        assert isinstance(anim_tick, AnimationGroup)
        assert card.remaining_time == 2

    def test_animate_complete_and_idle(self):
        cpu = CPUExecutionVisual()
        card = ProcessCardVisual(pid="P1", burst_time=3, state="READY")
        cpu.animate_dispatch(card)

        anim_comp = cpu.animate_complete()
        assert isinstance(anim_comp, AnimationGroup)
        assert card.current_state == "TERMINATED"

        anim_idle = cpu.animate_set_idle()
        assert isinstance(anim_idle, AnimationGroup)
        assert cpu.cpu_state == "IDLE"
        assert cpu.current_process is None

    def test_set_state_instant(self):
        cpu = CPUExecutionVisual()
        cpu.set_state_instant("RUNNING", "P2")
        assert cpu.cpu_state == "RUNNING"
        cpu.set_state_instant("IDLE")
        assert cpu.cpu_state == "IDLE"

    def test_consecutive_dispatch_lifecycle_cleans_up_cards(self):
        """Regression test for Work Item 1: CPU card lifecycle and stacked card prevention."""
        cpu = CPUExecutionVisual()
        # Dispatch P1
        cpu.animate_dispatch(pid="P1")
        cards_1 = [m for m in cpu.submobjects if isinstance(m, ProcessCardVisual)]
        assert len(cards_1) == 1
        assert cards_1[0].pid == "P1"
        assert cpu.active_card is cards_1[0]

        # Complete P1 using string PID argument (the pattern generated in section 9)
        cpu.animate_complete(pid="P1")
        assert cpu.cpu_state == "COMPLETED"
        assert isinstance(cpu.current_process, ProcessCardVisual)
        assert cpu.active_card is cards_1[0]
        assert cards_1[0].current_state == "TERMINATED"

        # Dispatch P2
        cpu.animate_dispatch(pid="P2")
        cards_2 = [m for m in cpu.submobjects if isinstance(m, ProcessCardVisual)]
        assert len(cards_2) == 1, f"Expected exactly 1 active process card in CPU, found {len(cards_2)}"
        assert cards_2[0].pid == "P2"
        assert cpu.active_card is cards_2[0]

        # Complete P2
        cpu.animate_complete(pid="P2")
        # Dispatch P3
        cpu.animate_dispatch(pid="P3")
        cards_3 = [m for m in cpu.submobjects if isinstance(m, ProcessCardVisual)]
        assert len(cards_3) == 1
        assert cards_3[0].pid == "P3"

        # Idle clears slot completely
        cpu.animate_set_idle()
        cards_idle = [m for m in cpu.submobjects if isinstance(m, ProcessCardVisual)]
        assert len(cards_idle) == 0
        assert cpu.current_process is None
        assert cpu.active_card is None


# ============================================================================
# 4. GanttChartVisual Tests
# ============================================================================

class TestGanttChartVisual:
    def test_registration(self):
        cls = VisualRegistry.get("cs", "GanttChartVisual")
        assert cls is GanttChartVisual
        manifest = GanttChartVisual.get_manifest()
        assert manifest["component_id"] == "GanttChartVisual"
        assert "animate_add_block" in manifest["semantic_actions"]
        assert "animate_update_time_marker" in manifest["semantic_actions"]

    def test_initialization_and_handles(self):
        gantt = GanttChartVisual(total_time=10)
        assert gantt.total_time == 10
        assert gantt.get_semantic_handle("axis") is not None
        assert gantt.get_semantic_handle("cursor") is not None
        assert gantt.get_semantic_handle("title") is not None
        assert gantt.get_semantic_handle("blocks_group") is not None

    def test_animate_add_block_process_and_idle(self):
        gantt = GanttChartVisual(total_time=10)
        anim_idle = gantt.animate_add_block(pid=None, start_time=0, end_time=2)
        assert isinstance(anim_idle, AnimationGroup)
        assert gantt.get_semantic_handle("block_IDLE_0") is not None
        assert gantt.current_marker_time == 2.0

        anim_p1 = gantt.animate_add_block(pid="P1", start_time=2, end_time=5)
        assert isinstance(anim_p1, AnimationGroup)
        assert gantt.get_semantic_handle("block_P1_2") is not None
        assert gantt.current_marker_time == 5.0

    def test_animate_update_time_marker(self):
        gantt = GanttChartVisual(total_time=10)
        anim = gantt.animate_update_time_marker(4.0)
        assert isinstance(anim, Animation)
        assert gantt.current_marker_time == 4.0

    def test_set_gantt_blocks(self):
        gantt = GanttChartVisual(total_time=9)
        blocks = [
            {"pid": "P1", "start_time": 0, "end_time": 3},
            {"pid": "P2", "start_time": 3, "end_time": 7},
            {"pid": "P3", "start_time": 7, "end_time": 9},
        ]
        gantt.set_gantt_blocks(blocks)
        assert len(gantt.blocks_group) == 3
        assert gantt.get_semantic_handle("block_P1_0") is not None
        assert gantt.get_semantic_handle("block_P2_3") is not None
        assert gantt.get_semantic_handle("block_P3_7") is not None
        assert gantt.current_marker_time == 9.0


# ============================================================================
# 5. MetricTableVisual Tests
# ============================================================================

class TestMetricTableVisual:
    def test_registration(self):
        cls = VisualRegistry.get("cs", "MetricTableVisual")
        assert cls is MetricTableVisual
        manifest = MetricTableVisual.get_manifest()
        assert manifest["component_id"] == "MetricTableVisual"
        assert "animate_update_process_metrics" in manifest["semantic_actions"]
        assert "animate_show_averages" in manifest["semantic_actions"]

    def test_initialization_honest_accounting(self):
        specs = [
            {"pid": "P1", "arrival_time": 0, "burst_time": 3},
            {"pid": "P2", "arrival_time": 1, "burst_time": 4},
        ]
        table = MetricTableVisual(processes=specs)
        assert table.num_rows == 2

        # Check handles
        assert table.get_semantic_handle("header") is not None
        assert table.get_semantic_handle("row_P1") is not None
        assert table.get_semantic_handle("row_P2") is not None
        assert table.get_semantic_handle("cell_P1_CT") is not None
        assert table.get_semantic_handle("cell_P1_TAT") is not None
        assert table.get_semantic_handle("cell_P1_WT") is not None
        assert table.get_semantic_handle("cell_P1_RT") is not None
        assert table.get_semantic_handle("summary_row") is not None
        assert table.get_semantic_handle("cell_avg_tat") is not None
        assert table.get_semantic_handle("cell_avg_wt") is not None
        assert table.get_semantic_handle("cell_avg_rt") is not None

        # Verify no premature leakage: CT, TAT, WT, RT are "-"
        assert table.get_semantic_handle("cell_P1_CT").text == "-"
        assert table.get_semantic_handle("cell_P1_TAT").text == "-"
        assert table.get_semantic_handle("cell_P1_WT").text == "-"
        assert table.get_semantic_handle("cell_P1_RT").text == "-"
        assert table.get_semantic_handle("cell_avg_tat").text == "-"
        assert table.get_semantic_handle("cell_avg_wt").text == "-"
        assert table.get_semantic_handle("cell_avg_rt").text == "-"

    def test_animate_update_process_metrics(self):
        specs = [{"pid": "P1", "arrival_time": 0, "burst_time": 3}]
        table = MetricTableVisual(processes=specs)

        anim_group = table.animate_update_process_metrics(pid="P1", ct=3, tat=3, wt=0, rt=0)
        assert isinstance(anim_group, AnimationGroup)
        assert table.get_semantic_handle("cell_P1_CT").text == "3"
        assert table.get_semantic_handle("cell_P1_TAT").text == "3"
        assert table.get_semantic_handle("cell_P1_WT").text == "0"
        assert table.get_semantic_handle("cell_P1_RT").text == "0"

    def test_animate_update_response_time(self):
        specs = [{"pid": "P1", "arrival_time": 0, "burst_time": 3}]
        table = MetricTableVisual(processes=specs)

        anim_group = table.animate_update_response_time(pid="P1", rt=0)
        assert isinstance(anim_group, AnimationGroup)
        assert table.get_semantic_handle("cell_P1_RT").text == "0"
        # CT, TAT, WT must remain unrevealed
        assert table.get_semantic_handle("cell_P1_CT").text == "-"
        assert table.get_semantic_handle("cell_P1_TAT").text == "-"
        assert table.get_semantic_handle("cell_P1_WT").text == "-"

    def test_animate_show_averages(self):
        specs = [{"pid": "P1", "arrival_time": 0, "burst_time": 3}]
        table = MetricTableVisual(processes=specs)

        anim_group = table.animate_show_averages(avg_tat=5.33, avg_wt=2.33, avg_rt=2.33)
        assert isinstance(anim_group, AnimationGroup)
        assert table.get_semantic_handle("cell_avg_tat").text == "5.33"
        assert table.get_semantic_handle("cell_avg_wt").text == "2.33"
        assert table.get_semantic_handle("cell_avg_rt").text == "2.33"

    def test_set_metrics_from_state(self):
        specs = [
            {"pid": "P1", "arrival_time": 0, "burst_time": 3},
            {"pid": "P2", "arrival_time": 1, "burst_time": 4},
        ]
        table = MetricTableVisual(processes=specs)
        metrics = {
            "P1": {"completion_time": 3, "turnaround_time": 3, "waiting_time": 0, "response_time": 0},
            "P2": {"completion_time": 7, "turnaround_time": 6, "waiting_time": 2, "response_time": 2},
        }
        table.set_metrics_from_state(metrics, avg_tat=4.5, avg_wt=1.0, avg_rt=1.0)
        assert table.get_semantic_handle("cell_P1_CT").text == "3"
        assert table.get_semantic_handle("cell_P2_CT").text == "7"
        assert table.get_semantic_handle("cell_P1_RT").text == "0"
        assert table.get_semantic_handle("cell_P2_RT").text == "2"
        assert table.get_semantic_handle("cell_avg_tat").text == "4.50"
        assert table.get_semantic_handle("cell_avg_rt").text == "1.00"


# ============================================================================
# 6. OSTraceVisualAdapter Tests
# ============================================================================

class TestOSTraceVisualAdapter:
    @pytest.fixture
    def canonical_fcfs_trace(self):
        processes = [
            ProcessSpec(pid="P1", arrival_time=0, burst_time=3),
            ProcessSpec(pid="P2", arrival_time=1, burst_time=4),
            ProcessSpec(pid="P3", arrival_time=2, burst_time=2),
        ]
        result: FCFSSimulationResult = FCFSSimulator.run(processes)
        return result.computation_trace

    def test_adapter_initialization(self, canonical_fcfs_trace):
        adapter = OSTraceVisualAdapter(canonical_fcfs_trace)
        assert adapter.total_steps == 6
        assert len(adapter.get_process_specs()) == 3
        assert adapter.get_total_simulation_time() == 9

    def test_visual_delta_step_by_step(self, canonical_fcfs_trace):
        adapter = OSTraceVisualAdapter(canonical_fcfs_trace)

        # Delta 0 -> 1: t=0 (P1 dispatched) -> t=1 (P2 arrives)
        delta_0_1 = adapter.get_visual_delta(0, 1)
        assert delta_0_1["time_start"] == 0
        assert delta_0_1["time_end"] == 1
        assert "P2" in delta_0_1["arrivals"]
        assert delta_0_1["completions"] == []
        assert delta_0_1["ready_queue"] == ["P2"]
        assert delta_0_1["running_pid"] == "P1"
        assert delta_0_1["response_time_updates"] == {}

        # Delta 2 -> 3: t=2 (P3 arrives) -> t=3 (P1 completes, P2 dispatched)
        delta_2_3 = adapter.get_visual_delta(2, 3)
        assert delta_2_3["time_start"] == 2
        assert delta_2_3["time_end"] == 3
        assert "P1" in delta_2_3["completions"]
        assert delta_2_3["dispatched_pid"] == "P2"
        assert "P1" in delta_2_3["metric_updates"]
        assert delta_2_3["metric_updates"]["P1"] == {"ct": 3, "tat": 3, "wt": 0, "rt": 0}
        assert delta_2_3["response_time_updates"] == {"P2": 2}
        assert delta_2_3["ready_queue"] == ["P3"]

        # Final delta 4 -> 5: t=7 (P2 completes) -> t=9 (P3 completes, simulation ends)
        delta_4_5 = adapter.get_visual_delta(4, 5)
        assert "P3" in delta_4_5["completions"]
        assert delta_4_5["averages_revealed"] is not None
        avg_tat, avg_wt, avg_rt = delta_4_5["averages_revealed"]
        assert round(avg_tat, 2) == 5.33
        assert round(avg_wt, 2) == 2.33
        assert round(avg_rt, 2) == 2.33

    def test_sync_all(self, canonical_fcfs_trace):
        adapter = OSTraceVisualAdapter(canonical_fcfs_trace)
        specs = adapter.get_process_specs()

        queue = ReadyQueueVisual(capacity=4)
        cpu = CPUExecutionVisual()
        gantt = GanttChartVisual(total_time=9)
        table = MetricTableVisual(processes=specs)

        components = {"queue": queue, "cpu": cpu, "gantt": gantt, "table": table}

        # Sync to step 3 (t=3): P1 completed, P2 running, P3 in queue
        adapter.sync_all(components, step_index=3)

        assert queue.queue_items[0].pid == "P3"
        assert cpu.cpu_state == "RUNNING"
        assert table.get_semantic_handle("cell_P1_CT").text == "3"
        assert table.get_semantic_handle("cell_P2_CT").text == "-"


# ============================================================================
# 7. Viewport Layout & Docking Integration Tests
# ============================================================================

class TestViewportLayoutAndDocking:
    def test_docking_and_viewport_boundaries(self):
        scene = TeachingScene()
        scene.setup_layout(
            "FCFS CPU Scheduling Viewport Verification",
            ["FIFO Ready Queue enqueues arriving processes", "CPU Core executes non-preemptively"],
        )

        queue = ReadyQueueVisual(capacity=4, slot_width=1.0, slot_height=0.8)
        cpu = CPUExecutionVisual(width=2.2, height=1.7)
        gantt = GanttChartVisual(total_time=10, width=5.8, height=0.9)
        table = MetricTableVisual(
            processes=[
                {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                {"pid": "P2", "arrival_time": 1, "burst_time": 4},
            ],
            width=5.8,
            height=1.6,
        )

        # Dock components
        scene.dock_split_left(queue)
        scene.dock_split_right(cpu)
        scene.dock_inspector(gantt)

        bb_queue = queue.get_bounding_box()
        bb_cpu = cpu.get_bounding_box()
        bb_gantt = gantt.get_bounding_box()

        # Split left and right horizontal separation
        assert bb_queue["right"] < bb_cpu["left"] or not scene.layout.check_overlap(queue, cpu)[0]

        # Inspector vertical separation: gantt must be below split containers
        assert bb_queue["bottom"] >= bb_gantt["top"] - 0.2

        # Subtitle clearance: Y >= -2.65
        assert bb_gantt["bottom"] >= -2.65, f"Gantt bottom ({bb_gantt['bottom']}) encroaches on subtitle area"

        # Caption safe zone clearance: X >= -2.15
        assert bb_queue["left"] >= -2.15
        assert bb_gantt["left"] >= -2.15

        # Right safe zone clearance: X <= 6.65
        assert bb_cpu["right"] <= 6.65
        assert bb_gantt["right"] <= 6.65


# ============================================================================
# 8. Standalone Manim Scene Rendering Test
# ============================================================================

class TestManimSceneRendering:
    def test_render_scene_with_os_components(self, tmp_path):
        """
        Renders a minimal Manim scene exercising all OS scheduling visual components.
        Verifies clean execution, zero crashes, and valid video generation.
        """
        class SampleFCFSScene(TeachingScene):
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
                gantt = GanttChartVisual(total_time=6, width=5.6, height=0.8)
                table = MetricTableVisual(
                    processes=[
                        {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                    ],
                    width=5.6,
                    height=1.4,
                )

                self.dock_split_left(queue)
                self.dock_split_right(cpu)
                self.dock_inspector(gantt)

                p1_card = ProcessCardVisual(pid="P1", arrival_time=0, burst_time=3, state="READY")

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
                    cpu.animate_execute_tick(remaining_time=2),
                    gantt.animate_add_block("P1", 0, 3),
                )
                self.play(cpu.animate_complete())

        with tempconfig({
            "quality": "low_quality",
            "preview": False,
            "format": "mp4",
            "media_dir": str(tmp_path),
            "disable_caching": True,
        }):
            scene = SampleFCFSScene()
            scene.render()

        rendered_files = list(tmp_path.glob("**/*.mp4"))
        assert len(rendered_files) > 0, "Expected at least one rendered MP4 file"
        assert rendered_files[0].stat().st_size > 1000, "Rendered video file is too small or corrupt"
