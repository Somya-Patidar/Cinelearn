"""
Unit and regression tests for Stage 5.1:
Multi-Frame and Synchronization Deterministic Inspector.
"""

import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add project root and src to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from local_checker import LocalVisualChecker, VisualIssue, VisualCheckResult


class TestMultiFrameLayoutAndSyncInspector(unittest.TestCase):
    def setUp(self):
        self.checker = LocalVisualChecker(safety_margin=0.35)

    # -------------------------------------------------------------------------
    # 1. Context-Aware Visual Complexity Tests (Requirement 2)
    # -------------------------------------------------------------------------
    def test_composite_components_counted_as_single_focal_units(self):
        """
        Verify that composite domain components (GraphVisual, DPTableMatrix, MoleculeVisual)
        are counted as 1 focal unit, absorbing internal vertices, edges, or table cells.
        """
        mobjects = [
            # Composite GraphVisual
            {
                "class": "GraphVisual",
                "tag": "canvas",
                "left": 0.0,
                "right": 4.0,
                "top": 2.0,
                "bottom": -1.0,
                "width": 4.0,
                "height": 3.0,
                "center": [2.0, 0.5, 0.0],
            },
            # Internal node labels contained inside the graph bounding box
            {
                "class": "Text",
                "tag": "canvas",
                "left": 0.5,
                "right": 1.0,
                "top": 1.8,
                "bottom": 1.4,
                "width": 0.5,
                "height": 0.4,
                "center": [0.75, 1.6, 0.0],
            },
            {
                "class": "Text",
                "tag": "canvas",
                "left": 2.5,
                "right": 3.0,
                "top": 1.8,
                "bottom": 1.4,
                "width": 0.5,
                "height": 0.4,
                "center": [2.75, 1.6, 0.0],
            },
            # Connecting arrow
            {
                "class": "Arrow",
                "tag": "canvas",
                "left": 1.0,
                "right": 2.5,
                "top": 1.6,
                "bottom": 1.6,
                "width": 1.5,
                "height": 0.1,
                "center": [1.75, 1.6, 0.0],
            },
            # Secondary status table
            {
                "class": "DPTableMatrix",
                "tag": "canvas",
                "left": -1.5,
                "right": 5.0,
                "top": -1.2,
                "bottom": -2.4,
                "width": 6.5,
                "height": 1.2,
                "center": [1.75, -1.8, 0.0],
            },
        ]
        issues = self.checker.check_visual_complexity(mobjects, checkpoint_name="step_1")
        # 1 GraphVisual + 1 DPTableMatrix = 2 focal units (under limit of 4), no complexity overload
        self.assertEqual(len(issues), 0)

    def test_excessive_independent_focal_units_triggers_complexity_overload(self):
        """
        Verify that 5+ distinct, unrelated focal mobjects trigger COMPLEXITY_OVERLOAD with evidence.
        """
        mobjects = [
            {"class": "Circle", "tag": "canvas", "left": -1.0, "right": -0.2, "top": 2.0, "bottom": 1.2, "width": 0.8, "height": 0.8, "center": [-0.6, 1.6, 0.0]},
            {"class": "Square", "tag": "canvas", "left": 1.0, "right": 1.8, "top": 2.0, "bottom": 1.2, "width": 0.8, "height": 0.8, "center": [1.4, 1.6, 0.0]},
            {"class": "MathTex", "tag": "canvas", "left": 3.0, "right": 5.0, "top": 2.0, "bottom": 1.5, "width": 2.0, "height": 0.5, "center": [4.0, 1.75, 0.0]},
            {"class": "FunctionCurve", "tag": "canvas", "left": -1.5, "right": 1.5, "top": 0.5, "bottom": -1.5, "width": 3.0, "height": 2.0, "center": [0.0, -0.5, 0.0]},
            {"class": "DPTableMatrix", "tag": "canvas", "left": 2.0, "right": 6.0, "top": 0.5, "bottom": -1.5, "width": 4.0, "height": 2.0, "center": [4.0, -0.5, 0.0]},
        ]
        issues = self.checker.check_visual_complexity(mobjects, checkpoint_name="step_crowded", timestamp=3.5, max_focal_units=4)
        self.assertEqual(len(issues), 1)
        issue = issues[0]
        self.assertEqual(issue.issue_type, "COMPLEXITY_OVERLOAD")
        self.assertEqual(issue.severity, "warning")
        self.assertIn("step_crowded", issue.description)
        # Verify evidence structure (Requirement 1)
        self.assertEqual(issue.evidence.get("metric"), "focal_units")
        self.assertEqual(issue.evidence.get("actual"), 5)
        self.assertEqual(issue.evidence.get("expected"), "<= 4")
        self.assertEqual(issue.evidence.get("timestamp"), 3.5)

    # -------------------------------------------------------------------------
    # 2. Dynamic Trajectory Collision Tests
    # -------------------------------------------------------------------------
    def test_trajectory_collision_detected_when_mobject_moves_into_occupied_space(self):
        """
        Verify that an object moving into another stationary object's bounding box
        across checkpoints is flagged as a dynamic trajectory collision.
        """
        checkpoints = [
            {
                "checkpoint": "step_1",
                "time": 1.0,
                "mobjects": [
                    {"class": "Square", "tag": "canvas", "left": 0.0, "right": 1.5, "top": 1.0, "bottom": -0.5, "width": 1.5, "height": 1.5, "center": [0.75, 0.25, 0.0]},
                    {"class": "Circle", "tag": "canvas", "left": 4.0, "right": 5.5, "top": 1.0, "bottom": -0.5, "width": 1.5, "height": 1.5, "center": [4.75, 0.25, 0.0]},
                ]
            },
            {
                "checkpoint": "step_2",
                "time": 2.5,
                "mobjects": [
                    # Square stays stationary
                    {"class": "Square", "tag": "canvas", "left": 0.0, "right": 1.5, "top": 1.0, "bottom": -0.5, "width": 1.5, "height": 1.5, "center": [0.75, 0.25, 0.0]},
                    # Circle moves from x=4.75 to x=0.85, colliding with Square
                    {"class": "Circle", "tag": "canvas", "left": 0.1, "right": 1.6, "top": 1.0, "bottom": -0.5, "width": 1.5, "height": 1.5, "center": [0.85, 0.25, 0.0]},
                ]
            }
        ]
        issues = self.checker.check_trajectory_collisions(checkpoints)
        self.assertTrue(len(issues) >= 1)
        traj_issue = issues[0]
        self.assertEqual(traj_issue.issue_type, "OVERLAP")
        self.assertIn("trajectory collision", traj_issue.description.lower())
        self.assertIn("step_2", traj_issue.description)
        # Verify evidence
        self.assertEqual(traj_issue.evidence.get("metric"), "trajectory_overlap")
        self.assertEqual(traj_issue.evidence.get("timestamp"), 2.5)
        self.assertTrue(traj_issue.evidence.get("overlap_ratio", 0) > 0.5)

    def test_trajectory_ignores_stationary_non_colliding_objects(self):
        checkpoints = [
            {
                "checkpoint": "step_1",
                "time": 1.0,
                "mobjects": [
                    {"class": "Square", "tag": "canvas", "left": 0.0, "right": 1.5, "top": 1.0, "bottom": -0.5, "width": 1.5, "height": 1.5, "center": [0.75, 0.25, 0.0]},
                    {"class": "Circle", "tag": "canvas", "left": 3.0, "right": 4.5, "top": 1.0, "bottom": -0.5, "width": 1.5, "height": 1.5, "center": [3.75, 0.25, 0.0]},
                ]
            },
            {
                "checkpoint": "step_2",
                "time": 2.0,
                "mobjects": [
                    {"class": "Square", "tag": "canvas", "left": 0.0, "right": 1.5, "top": 1.0, "bottom": -0.5, "width": 1.5, "height": 1.5, "center": [0.75, 0.25, 0.0]},
                    # Circle shifts slightly to the right, remaining well clear
                    {"class": "Circle", "tag": "canvas", "left": 4.0, "right": 5.5, "top": 1.0, "bottom": -0.5, "width": 1.5, "height": 1.5, "center": [4.75, 0.25, 0.0]},
                ]
            }
        ]
        issues = self.checker.check_trajectory_collisions(checkpoints)
        self.assertEqual(len(issues), 0)

    # -------------------------------------------------------------------------
    # 3. Viewport Boundary & Subtitle Clearance
    # -------------------------------------------------------------------------
    def test_subtitle_margin_intrusion_in_checkpoints(self):
        """
        Verify that intermediate checkpoints breaching SAFE_BOTTOM (-2.75 / -2.85) are flagged.
        """
        checkpoints = [
            {
                "checkpoint": "step_intro",
                "time": 0.5,
                "mobjects": [
                    {"class": "Text", "tag": "canvas", "left": 0.0, "right": 2.0, "top": -2.5, "bottom": -3.2, "width": 2.0, "height": 0.7, "center": [1.0, -2.85, 0.0]}
                ]
            }
        ]
        issues = self.checker.check_multiframe_checkpoints(checkpoints)
        self.assertTrue(any(i.issue_type == "OUT_OF_SCREEN" for i in issues))
        subtitle_issue = next(i for i in issues if i.issue_type == "OUT_OF_SCREEN")
        self.assertIn("subtitle", subtitle_issue.description.lower())
        self.assertIn("step_intro", subtitle_issue.description)
        self.assertEqual(subtitle_issue.evidence.get("boundary"), "bottom_subtitle")

    # -------------------------------------------------------------------------
    # 4. Narration Synchronization Tests (Requirement 3)
    # -------------------------------------------------------------------------
    def test_sync_detects_dead_line_without_animations(self):
        """
        Verify detection when a lecture line has speech (>1.5s) but zero animations (dead screen).
        """
        code = """
class TestScene(TeachingScene):
    def construct(self):
        # === Animation for Lecture Line 1 ===
        self.activate_caption(0)
        self.wait(3.0)

        # === Animation for Lecture Line 2 ===
        self.activate_caption(1)
        self.play(FadeIn(circle), run_time=1.5)
        self.wait(1.5)
"""
        ts_data = {
            "lines": [
                {"line_index": 0, "start": 0.0, "end": 3.0, "duration": 3.0, "total_budget": 3.5},
                {"line_index": 1, "start": 3.5, "end": 6.5, "duration": 3.0, "total_budget": 3.5},
            ]
        }
        issues = self.checker.check_narration_synchronization(code, ts_data)
        self.assertTrue(any(i.issue_type == "SYNC_MISSING_VISUAL" for i in issues))
        dead_line_issue = next(i for i in issues if i.issue_type == "SYNC_MISSING_VISUAL")
        self.assertIn("Lecture line 1", dead_line_issue.description)
        self.assertEqual(dead_line_issue.evidence.get("line_index"), 0)
        self.assertEqual(dead_line_issue.evidence.get("animations_found"), 0)

    def test_sync_detects_caption_activation_mismatch(self):
        """
        Verify detection when line 1 activates caption index 2 instead of 0.
        """
        code = """
class TestScene(TeachingScene):
    def construct(self):
        # === Animation for Lecture Line 1 ===
        self.activate_caption(2)
        self.play(Create(line), run_time=1.0)
        self.wait(1.0)
"""
        ts_data = {
            "lines": [
                {"line_index": 0, "start": 0.0, "end": 2.0, "duration": 2.0, "total_budget": 2.5},
            ]
        }
        issues = self.checker.check_narration_synchronization(code, ts_data)
        self.assertTrue(any(i.issue_type == "SYNC_CAPTION_MISMATCH" for i in issues))
        mismatch_issue = next(i for i in issues if i.issue_type == "SYNC_CAPTION_MISMATCH")
        self.assertEqual(mismatch_issue.evidence.get("expected_caption"), 0)
        self.assertEqual(mismatch_issue.evidence.get("actual_caption"), 2)

    def test_sync_detects_duration_budget_overrun(self):
        """
        Verify detection when total animation duration significantly exceeds narration budget.
        """
        code = """
class TestScene(TeachingScene):
    def construct(self):
        # === Animation for Lecture Line 1 ===
        self.activate_caption(0)
        self.play(FadeIn(graph), run_time=3.0)
        self.play(graph.animate_relax_edge("A", "B"), run_time=2.5)
        self.wait(2.0)
"""
        # Budget is 3.5s, but animations sum to 3.0 + 2.5 + 2.0 = 7.5s (overrun by 4.0s)
        ts_data = {
            "lines": [
                {"line_index": 0, "start": 0.0, "end": 3.0, "duration": 3.0, "total_budget": 3.5},
            ]
        }
        issues = self.checker.check_narration_synchronization(code, ts_data)
        self.assertTrue(any(i.issue_type == "SYNC_DURATION_OVERRUN" for i in issues))
        overrun_issue = next(i for i in issues if i.issue_type == "SYNC_DURATION_OVERRUN")
        self.assertEqual(overrun_issue.severity, "error")  # delta > 2.5s is error
        self.assertEqual(overrun_issue.evidence.get("expected"), 3.5)
        self.assertEqual(overrun_issue.evidence.get("actual"), 7.5)
        self.assertEqual(overrun_issue.evidence.get("delta"), "+4.0s")

    def test_sync_detects_premature_animation_end(self):
        """
        Verify detection when animations end prematurely leaving substantial unillustrated speech.
        """
        code = """
class TestScene(TeachingScene):
    def construct(self):
        # === Animation for Lecture Line 1 ===
        self.activate_caption(0)
        self.play(FadeIn(dot), run_time=0.5)
        self.wait(0.5)
"""
        # Spoken duration is 6.0s, but animation ends at 1.0s (leaving 5.0s unillustrated)
        ts_data = {
            "lines": [
                {"line_index": 0, "start": 0.0, "end": 6.0, "duration": 6.0, "total_budget": 6.5},
            ]
        }
        issues = self.checker.check_narration_synchronization(code, ts_data)
        self.assertTrue(any(i.issue_type == "SYNC_DURATION_UNDERRUN" for i in issues))
        underrun_issue = next(i for i in issues if i.issue_type == "SYNC_DURATION_UNDERRUN")
        self.assertEqual(underrun_issue.evidence.get("actual"), 1.0)
        self.assertEqual(underrun_issue.evidence.get("expected"), 6.0)

    def test_sync_detects_visual_event_onset_lag(self):
        """
        Verify detection when primary visual action begins after spoken narration ended.
        """
        code = """
class TestScene(TeachingScene):
    def construct(self):
        # === Animation for Lecture Line 1 ===
        self.activate_caption(0)
        self.wait(3.5)
        self.play(Create(graph), run_time=1.5)
"""
        # Spoken narration ends at 2.5s, but Create(graph) triggers at 3.5s
        ts_data = {
            "lines": [
                {"line_index": 0, "start": 0.0, "end": 2.5, "duration": 2.5, "total_budget": 5.5},
            ]
        }
        issues = self.checker.check_narration_synchronization(code, ts_data)
        self.assertTrue(any(i.issue_type == "SYNC_EVENT_LAG" for i in issues))
        lag_issue = next(i for i in issues if i.issue_type == "SYNC_EVENT_LAG")
        self.assertEqual(lag_issue.evidence.get("event_onset"), 3.5)
        self.assertEqual(lag_issue.evidence.get("narration_end"), 2.5)

    def test_sync_clean_when_events_and_timing_properly_aligned(self):
        """
        Verify that a properly aligned section with synchronous caption and animation returns zero issues.
        """
        code = """
class TestScene(TeachingScene):
    def construct(self):
        # === Animation for Lecture Line 1 ===
        self.activate_caption(0)
        self.play(FadeIn(graph), run_time=1.5)
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.activate_caption(1)
        self.play(graph.animate_visit_node("A"), run_time=1.0)
        self.wait(1.5)
"""
        ts_data = {
            "lines": [
                {"line_index": 0, "start": 0.0, "end": 2.8, "duration": 2.8, "total_budget": 3.2},
                {"line_index": 1, "start": 3.2, "end": 5.5, "duration": 2.3, "total_budget": 2.7},
            ]
        }
        issues = self.checker.check_narration_synchronization(code, ts_data)
        self.assertEqual(len(issues), 0)

    # -------------------------------------------------------------------------
    # 5. Integrated check_section Test with Multi-Frame & Timestamps
    # -------------------------------------------------------------------------
    def test_check_section_integrated_multiframe_and_sync(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            sec_id = "section_1"

            # 1. Write layout metadata with checkpoints
            layout_meta = {
                "scene": "Section1Scene",
                "frame_width": 14.222,
                "frame_height": 8.0,
                "mobjects": [
                    {"class": "GraphVisual", "tag": "canvas", "left": 0.0, "right": 3.5, "top": 1.5, "bottom": -0.5, "width": 3.5, "height": 2.0, "center": [1.75, 0.5, 0.0]}
                ],
                "checkpoints": [
                    {
                        "checkpoint": "step_1",
                        "time": 1.0,
                        "mobjects": [
                            {"class": "GraphVisual", "tag": "canvas", "left": 0.0, "right": 3.5, "top": 1.5, "bottom": -0.5, "width": 3.5, "height": 2.0, "center": [1.75, 0.5, 0.0]}
                        ]
                    }
                ]
            }
            with open(out_path / "layout_Section1Scene.json", "w") as f:
                json.dump(layout_meta, f)

            # 2. Write timestamps metadata
            ts_meta = {
                "lines": [
                    {"line_index": 0, "start": 0.0, "end": 3.0, "duration": 3.0, "total_budget": 3.5}
                ]
            }
            with open(out_path / f"{sec_id}_timestamps.json", "w") as f:
                json.dump(ts_meta, f)

            code = """
class Section1Scene(TeachingScene):
    def construct(self):
        # === Animation for Lecture Line 1 ===
        self.activate_caption(0)
        self.play(FadeIn(graph), run_time=1.5)
        self.wait(1.5)
"""
            result = self.checker.check_section(sec_id, code, out_path)
            self.assertTrue(result.is_clean)
            self.assertEqual(result.status, "CLEAN")

    def test_internal_composite_overlap_detected(self):
        """Regression test for Work Item 3: Detect internal overlapping cards inside composite component."""
        mobjects = [
            {
                "class": "CPUExecutionVisual",
                "tag": "canvas",
                "left": 0.0,
                "right": 4.0,
                "top": 2.0,
                "bottom": -1.0,
                "width": 4.0,
                "height": 3.0,
                "center": [2.0, 0.5, 0.0],
                "children": [
                    # Two distinct process cards stacked in the same slot
                    {
                        "child_index": 0,
                        "class": "ProcessCardVisual",
                        "left": 1.0,
                        "right": 3.0,
                        "top": 1.0,
                        "bottom": 0.0,
                        "width": 2.0,
                        "height": 1.0,
                        "center": [2.0, 0.5, 0.0],
                    },
                    {
                        "child_index": 1,
                        "class": "ProcessCardVisual",
                        "left": 1.1,
                        "right": 3.1,
                        "top": 1.05,
                        "bottom": 0.05,
                        "width": 2.0,
                        "height": 1.0,
                        "center": [2.1, 0.55, 0.0],
                    },
                ],
            }
        ]
        issues = self.checker._check_mobjects_geometry(mobjects)
        overlap_issues = [i for i in issues if i.issue_type == "OVERLAP"]
        self.assertTrue(len(overlap_issues) > 0, "Expected internal overlap issue to be flagged")
        self.assertTrue(any("ProcessCardVisual" in i.description for i in overlap_issues))

    def test_blank_scene_detected(self):
        """Regression test for Work Item 3 & 4: Blank scenes without visible mobjects are flagged."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir)
            sec_id = "section_blank"
            # Write layout metadata with 0 visible canvas mobjects (e.g. empty procedural component)
            layout_meta = {
                "scene": "SectionBlankScene",
                "mobjects": [
                    {"class": "ProceduralComponent", "tag": "canvas", "width": 0.0, "height": 0.0, "center": [0, 0, 0]}
                ],
                "checkpoints": [
                    {"checkpoint": "step_1", "mobjects": [{"class": "ProceduralComponent", "tag": "canvas", "width": 0.0, "height": 0.0, "center": [0, 0, 0]}]}
                ]
            }
            with open(out_path / "layout_SectionBlankScene.json", "w") as f:
                json.dump(layout_meta, f)

            code = "class SectionBlankScene(TeachingScene):\n    def construct(self):\n        pass\n"
            result = self.checker.check_section(sec_id, code, out_path)
            self.assertFalse(result.is_clean)
            blank_issues = [i for i in result.issues if i.issue_type == "BLANK_SCENE"]
            self.assertTrue(len(blank_issues) > 0, "Expected BLANK_SCENE issue to be flagged")


if __name__ == "__main__":
    unittest.main()
