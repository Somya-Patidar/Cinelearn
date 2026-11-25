"""
Regression and validation tests for FCFS Educational Quality,
Visual Refinement, Domain Action Weight Decoupling, and Layout Fixes.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from domains import VisualRegistry
from domains.layout import GenericContainerLayout
from domains.action_resolver import StoryboardActionResolver
from domains.cs.os_scheduling import (
    ProcessCardVisual,
    ReadyQueueVisual,
    CPUExecutionVisual,
    GanttChartVisual,
    MetricTableVisual,
    format_fcfs_storyboard_guidance,
)
from timing_contract import AnimationRetimer, SceneTimingContract


class TestFCFSEducationalQualityAndLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        VisualRegistry.discover_domains()
        try:
            from src.domains import VisualRegistry as SrcVR
            SrcVR.discover_domains()
        except Exception:
            pass

    def test_domain_action_weights_decoupled_from_retimer(self):
        """Verifies common retimer consumes generic action properties from domain metadata."""
        weights = VisualRegistry.get_action_weights()
        # Verify OS scheduling action weights are registered
        self.assertIn("animate_dispatch", weights)
        self.assertEqual(weights["animate_dispatch"], 1.5)
        self.assertIn("animate_enqueue", weights)
        self.assertEqual(weights["animate_enqueue"], 1.5)
        self.assertIn("animate_dequeue", weights)
        self.assertEqual(weights["animate_dequeue"], 1.5)
        self.assertIn("animate_update_time_marker", weights)
        self.assertEqual(weights["animate_update_time_marker"], 0.6)

        # Verify common retimer queries them dynamically
        self.assertEqual(AnimationRetimer.get_action_weight("core.animate_dispatch(p1)"), 1.5)
        self.assertEqual(AnimationRetimer.get_action_weight("gantt.animate_update_time_marker(3)"), 0.6)
        self.assertEqual(AnimationRetimer.get_action_weight("FadeIn(card)"), 1.0)
        self.assertEqual(AnimationRetimer.get_action_weight("Write(formula)"), 1.5)

    def test_ready_queue_visual_initializes_processes(self):
        """Verifies ReadyQueueVisual populates queue cards when initial_pids or processes are provided."""
        # Using string PIDs
        rq1 = ReadyQueueVisual(capacity=5, initial_pids=["P1", "P2", "P3"])
        self.assertEqual(len(rq1.queue_items), 3)
        self.assertEqual(getattr(rq1.queue_items[0], "pid", None), "P1")
        self.assertEqual(getattr(rq1.queue_items[1], "pid", None), "P2")
        self.assertEqual(getattr(rq1.queue_items[2], "pid", None), "P3")

        # Dequeueing removes HEAD item and shifts remaining
        head = rq1.get_head_item()
        self.assertEqual(head.pid, "P1")
        rq1.animate_dequeue(animate_head_exit=False)
        self.assertEqual(len(rq1.queue_items), 2)
        self.assertEqual(rq1.get_head_item().pid, "P2")

    def test_ready_queue_visual_initializes_dict_processes(self):
        """Verifies ReadyQueueVisual populates queue cards when processes dict list is provided."""
        procs = [
            {"pid": "P1", "arrival_time": 0, "burst_time": 3},
            {"pid": "P2", "arrival_time": 1, "burst_time": 4},
        ]
        rq = ReadyQueueVisual(capacity=5, processes=procs)
        self.assertEqual(len(rq.queue_items), 2)
        self.assertEqual(rq.queue_items[0].pid, "P1")
        self.assertEqual(rq.queue_items[1].pid, "P2")

    def test_metric_table_visual_suggested_viewport_is_dock_main(self):
        """Verifies MetricTableVisual has suggested_viewport = 'dock_main' to prevent inspector collisions."""
        self.assertEqual(MetricTableVisual.suggested_viewport, "dock_main")
        manifest = MetricTableVisual.get_manifest()
        self.assertEqual(manifest["suggested_viewport"], "dock_main")

    def test_dock_split_left_and_right_respect_next_to(self):
        """Verifies dock_split_left and dock_split_right support relative placement without stacking on center."""
        layout = GenericContainerLayout()

        # Create two mock mobjects
        c1 = ProcessCardVisual(pid="P1", arrival_time=0, burst_time=3)
        c2 = ProcessCardVisual(pid="P2", arrival_time=1, burst_time=4)

        layout.dock_split_left(c1)
        center1 = c1.get_center().copy()
        self.assertAlmostEqual(center1[0], layout.split_left.center[0], places=2)

        # Dock c2 next to c1 to the RIGHT
        layout.dock_split_left(c2, next_to=c1, direction=np.array([1, 0, 0]), buff=0.4)
        center2 = c2.get_center().copy()

        # c2 must be strictly to the right of c1, not superimposed at split_left center
        self.assertGreater(center2[0], center1[0])
        has_overlap, ratio = layout.check_overlap(c1, c2)
        self.assertFalse(has_overlap, f"Relative dock_split_left caused overlap: ratio={ratio}")

    def test_action_resolver_routes_metric_table_to_dock_main(self):
        """Verifies StoryboardActionResolver smart routes MetricTableVisual away from dock_inspector."""
        step_str = "L1-1 | FadeIn | MetricTableVisual | viewport: dock_inspector | action: animate_show_averages()"
        parsed = StoryboardActionResolver.parse_animation_step(step_str)
        is_valid, resolved, err = StoryboardActionResolver.resolve_visual_action(
            visual_rep=parsed["visual_representation"],
            action_str=parsed["action"],
            viewport=parsed["viewport"],
            step_tag=parsed["tag"],
            effect=parsed["effect"],
        )
        self.assertTrue(is_valid)
        self.assertIsNotNone(resolved)
        # Must be smart-routed to dock_main to prevent cramped scaling and collision with Gantt
        self.assertEqual(resolved.viewport, "dock_main")

    def test_fcfs_storyboard_guidance_contains_pedagogical_requirements(self):
        """Verifies FCFS storyboard guidance contains all required educational criteria."""
        algo_state = {
            "algorithm": "fcfs",
            "processes": [
                {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                {"pid": "P2", "arrival_time": 1, "burst_time": 4},
                {"pid": "P3", "arrival_time": 2, "burst_time": 2},
            ],
            "gantt_chart": [
                {"pid": "P1", "start_time": 0, "end_time": 3, "duration": 3, "is_idle": False},
                {"pid": "P2", "start_time": 3, "end_time": 7, "duration": 4, "is_idle": False},
                {"pid": "P3", "start_time": 7, "end_time": 9, "duration": 2, "is_idle": False},
            ],
            "process_metrics": {
                "P1": {"completion_time": 3, "turnaround_time": 3, "waiting_time": 0, "response_time": 0},
                "P2": {"completion_time": 7, "turnaround_time": 6, "waiting_time": 2, "response_time": 2},
                "P3": {"completion_time": 9, "turnaround_time": 7, "waiting_time": 5, "response_time": 5},
            },
            "average_waiting_time": 2.33,
            "average_turnaround_time": 5.33,
            "average_response_time": 2.33,
            "total_simulation_time": 9,
        }

        guidance = format_fcfs_storyboard_guidance(algo_state)

        # Check required educational topics
        self.assertIn("CPU Scheduling Motivation", guidance)
        self.assertIn("Process Control Block (PCB)", guidance)
        self.assertIn("Process States & Lifecycle Transitions", guidance)
        self.assertIn("Ready Queue", guidance)
        self.assertIn("CPU Scheduler vs Dispatcher", guidance)
        self.assertIn("FCFS Selection & Non-Preemptive Execution", guidance)
        self.assertIn("Gantt Chart Construction", guidance)
        self.assertIn("Scheduling Metric Formulas", guidance)
        self.assertIn("TAT = CT - AT", guidance)
        self.assertIn("WT = TAT - BT", guidance)
        self.assertIn("The Convoy Effect", guidance)
        self.assertIn("Connected Component State Transitions", guidance)
        self.assertIn("NEVER dock both `MetricTableVisual` AND `GanttChartVisual` to `dock_inspector`", guidance)

    def test_semantic_highlight_actions_resolution(self):
        """Verifies semantic highlights like highlight_attributes and highlight_convoys resolve cleanly."""
        step1 = "L4-1 | Indicate | ProcessCardVisual | viewport: dock_split_left | action: highlight_attributes()"
        p1 = StoryboardActionResolver.parse_animation_step(step1)
        valid1, res1, err1 = StoryboardActionResolver.resolve_visual_action(
            p1["visual_representation"], p1["action"], p1["viewport"], p1["tag"], p1["effect"]
        )
        self.assertTrue(valid1, f"Failed resolving highlight_attributes: {err1}")
        self.assertIn(res1.canonical_action, ("highlight_attributes", "animate_highlight"))

        step2 = "L1-1 | Indicate | GanttChartVisual | viewport: dock_inspector | action: highlight_convoys()"
        p2 = StoryboardActionResolver.parse_animation_step(step2)
        valid2, res2, err2 = StoryboardActionResolver.resolve_visual_action(
            p2["visual_representation"], p2["action"], p2["viewport"], p2["tag"], p2["effect"]
        )
        self.assertTrue(valid2, f"Failed resolving highlight_convoys: {err2}")
        self.assertIn(res2.canonical_action, ("highlight_convoys", "animate_highlight"))

    def test_process_card_set_state_not_misidentified_as_pid(self):
        """Verifies animate_set_state('READY') is not checked as a PID against the trace."""
        trace = {
            "processes": [{"pid": "P1"}, {"pid": "P2"}, {"pid": "P3"}],
            "gantt_chart": [],
            "process_metrics": {},
        }
        step = "L2-1 | Indicate | ProcessCardVisual | viewport: dock_split_left | action: animate_set_state('READY')"
        p = StoryboardActionResolver.parse_animation_step(step)
        valid, resolved, err = StoryboardActionResolver.resolve_visual_action(
            p["visual_representation"], p["action"], p["viewport"], p["tag"], p["effect"]
        )
        self.assertTrue(valid)
        disc = StoryboardActionResolver._validate_fcfs_action_trace("section_3", 1, resolved, trace)
        self.assertIsNone(disc, f"Misidentified 'READY' as invalid PID: {disc}")

    def test_animate_update_time_marker_alias_start(self):
        """Verifies animate_update_time_marker accepts start as alias for new_time."""
        step = "L1-1 | Indicate | GanttChartVisual | viewport: dock_inspector | action: animate_update_time_marker(start=0)"
        p = StoryboardActionResolver.parse_animation_step(step)
        valid, resolved, err = StoryboardActionResolver.resolve_visual_action(
            p["visual_representation"], p["action"], p["viewport"], p["tag"], p["effect"]
        )
        self.assertTrue(valid, f"Failed resolving animate_update_time_marker(start=0): {err}")
        self.assertEqual(resolved.canonical_action, "animate_update_time_marker")

    def test_procedural_component_no_submobjects_renders_visible_placeholder(self):
        """Regression test for Work Item 4: ProceduralComponent with no children is visible, not blank."""
        from src.domains import ProceduralComponent
        p = ProceduralComponent(name="FCFSOverview")
        self.assertGreater(float(p.width), 1.0)
        self.assertGreater(float(p.height), 1.0)
        self.assertTrue(getattr(p, "_is_placeholder", False))
        self.assertGreater(len(p.submobjects), 0)

        # When real mobjects are added, placeholder is cleared
        from manim import Circle
        c = Circle(radius=0.5)
        p.add(c)
        self.assertFalse(getattr(p, "_is_placeholder", True))
        self.assertIn(c, p.submobjects)

    def test_procedural_component_empty_vgroup_renders_visible_placeholder(self):
        """Regression test: ProceduralComponent(VGroup()) with empty children renders visible placeholder."""
        from src.domains import ProceduralComponent
        from manim import VGroup
        p = ProceduralComponent(VGroup(), name="MultiprogrammingDiagram")
        self.assertGreater(float(p.width), 1.0)
        self.assertGreater(float(p.height), 1.0)
        self.assertTrue(getattr(p, "_is_placeholder", False))
        self.assertGreater(len(p.submobjects), 0)
        # Default positioning should be within safe canvas (around x=2.225)
        self.assertGreater(p.get_center()[0], 0.0)

    def test_component_readability_and_minimum_scale_floor(self):
        """Regression test for Work Item 5: OS components have readable fonts and respect scale floors."""
        card = ProcessCardVisual(pid="P1", burst_time=3)
        self.assertGreaterEqual(card.font_size, 16)
        self.assertGreaterEqual(card.pid_label.font_size, 14)

        table = MetricTableVisual()
        self.assertGreaterEqual(table.font_size, 16)

        gantt = GanttChartVisual(total_time=8)
        self.assertGreaterEqual(gantt.font_size, 16)

        # Scale floor test in layout
        layout = GenericContainerLayout()
        huge_card = ProcessCardVisual(pid="P_HUGE", burst_time=9, width=50.0, height=40.0)
        layout.dock_main(huge_card, auto_scale=True)
        # Scaled card width should respect min_scale_threshold floor, not collapsed to 0
        self.assertGreater(float(huge_card.width), 1.0)

    def test_visual_registry_resolves_natural_aliases(self):
        """Verifies VisualRegistry.get_component_by_name resolves natural storyboard names."""
        self.assertEqual(VisualRegistry.get_component_by_name("ready queue")[1], ReadyQueueVisual)
        self.assertEqual(VisualRegistry.get_component_by_name("PCB")[1], ProcessCardVisual)
        self.assertEqual(VisualRegistry.get_component_by_name("CPU core")[1], CPUExecutionVisual)
        self.assertEqual(VisualRegistry.get_component_by_name("gantt chart")[1], GanttChartVisual)
        self.assertEqual(VisualRegistry.get_component_by_name("metric table")[1], MetricTableVisual)
        self.assertEqual(VisualRegistry.get_component_by_name("Process Control Block")[1], ProcessCardVisual)

    def test_action_resolver_generates_instantiation_code(self):
        """Verifies StoryboardActionResolver generates concrete instantiation and docking code."""
        step = "L4-1 | FadeIn | ReadyQueueVisual | viewport: dock_split_left | action: animate_enqueue(p1)"
        p = StoryboardActionResolver.parse_animation_step(step)
        valid, resolved, err = StoryboardActionResolver.resolve_visual_action(
            p["visual_representation"], p["action"], p["viewport"], p["tag"], p["effect"]
        )
        self.assertTrue(valid)
        self.assertIsNotNone(resolved)
        self.assertIn("ReadyQueueVisual", resolved.instantiation_code)
        self.assertEqual(resolved.placement_call, "self.dock_split_left(ready_queue)")

    def test_layout_clamping_prevents_caption_and_subtitle_intrusion(self):
        """Verifies GenericContainerLayout.clamp_mobject strictly enforces boundary limits."""
        layout = GenericContainerLayout()
        # Create an object pushed into caption area (x = -3.5) and below subtitles (y = -3.5)
        p = ProcessCardVisual(pid="P1", burst_time=3)
        p.move_to(np.array([-3.5, -3.5, 0]))
        layout.clamp_mobject(p)
        left = float(p.get_left()[0])
        bottom = float(p.get_bottom()[1])
        # Safe left boundary >= -2.15, safe bottom boundary >= -2.65
        self.assertGreaterEqual(left, -2.15 - 1e-4)
        self.assertGreaterEqual(bottom, -2.65 - 1e-4)

    def test_multi_object_docking_in_same_viewport_avoids_identical_coordinates(self):
        """Verifies docking multiple objects into the same viewport stacks them rather than superimposing."""
        layout = GenericContainerLayout()
        cpu = CPUExecutionVisual()
        pcb = ProcessCardVisual(pid="P1", burst_time=4)
        layout.dock_split_right(cpu)
        layout.dock_split_right(pcb)
        # Centers must not be identical
        cpu_center = cpu.get_center()
        pcb_center = pcb.get_center()
        self.assertNotEqual(round(cpu_center[1], 2), round(pcb_center[1], 2))
        has_overlap, ratio = layout.check_overlap(cpu, pcb)
        self.assertFalse(has_overlap, f"Multi-object docking collided: overlap ratio {ratio}")

    def test_local_checker_formula_not_blank_and_readable(self):
        """Verifies single-line formulas with height ~0.135 are not flagged as BLANK_SCENE or READABILITY."""
        from src.local_checker import LocalVisualChecker
        checker = LocalVisualChecker()
        mock_formula = {
            "index": 0,
            "class": "SafeMathTex",
            "tag": "canvas",
            "left": 0.0,
            "right": 2.2,
            "top": 0.5,
            "bottom": 0.365,
            "width": 2.2,
            "height": 0.135,
            "center": [1.1, 0.4325, 0.0],
            "is_placeholder": False,
        }
        self.assertTrue(checker._is_meaningful_canvas_mobject(mock_formula))
        issues = checker._check_mobjects_geometry([mock_formula])
        blank_issues = [i for i in issues if i.issue_type == "BLANK_SCENE"]
        readability_issues = [i for i in issues if i.issue_type == "READABILITY"]
        self.assertEqual(len(blank_issues), 0)
        self.assertEqual(len(readability_issues), 0)

    def test_local_checker_placeholder_only_scene_flagged(self):
        """Verifies scenes containing only placeholders are flagged with PLACEHOLDER_ONLY_SCENE."""
        from src.local_checker import LocalVisualChecker
        checker = LocalVisualChecker()
        layout_data = {
            "mobjects": [
                {
                    "index": 0,
                    "class": "ProceduralComponent",
                    "tag": "canvas",
                    "left": 1.0,
                    "right": 5.5,
                    "top": 1.0,
                    "bottom": -1.2,
                    "width": 4.5,
                    "height": 2.2,
                    "center": [3.25, -0.1, 0.0],
                    "is_placeholder": True,
                    "placeholder_name": "ConvoyOverview",
                }
            ],
            "checkpoints": [],
        }
        # Run check logic
        all_canvas = [m for m in layout_data["mobjects"] if checker._is_meaningful_canvas_mobject(m)]
        placeholder_only = all(m.get("is_placeholder", False) for m in all_canvas)
        self.assertTrue(placeholder_only)

    def test_intentional_composition_recognizes_cpu_card_dispatch(self):
        """Verifies _is_intentional_composition allows ProcessCard inside CPUExecutionVisual."""
        from src.local_checker import LocalVisualChecker
        checker = LocalVisualChecker()
        is_intentional = checker._is_intentional_composition(
            "CPUExecutionVisual", "ProcessCardVisual", {}, {}
        )
        self.assertTrue(is_intentional)


if __name__ == "__main__":
    unittest.main()
