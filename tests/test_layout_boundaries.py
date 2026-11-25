"""
Tests for Manim layout boundaries, collision detection, and deterministic local fixes.
Protects clean layout generation without external API calls.
"""

import sys
import unittest
from pathlib import Path

# Add project root and src to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from local_checker import LocalVisualChecker, LocalVisualFixer, VisualIssue


class TestLayoutBoundariesAndCollision(unittest.TestCase):
    def setUp(self):
        self.checker = LocalVisualChecker(safety_margin=0.35)
        self.fixer = LocalVisualFixer()

    def test_out_of_screen_detection_from_geometry(self):
        # Mobject extending beyond right frame boundary
        mobjects = [
            {
                "class": "MathTex",
                "tag": "canvas",
                "left": 5.0,
                "right": 7.5,  # Exceeds right_limit (6.71)
                "top": 1.0,
                "bottom": 0.0,
                "width": 2.5,
                "height": 1.0,
            }
        ]
        issues = self.checker._check_mobjects_geometry(mobjects)
        self.assertTrue(any(i.issue_type == "OUT_OF_SCREEN" for i in issues))
        self.assertTrue(any(i.details.get("boundary") == "right" for i in issues))

    def test_top_title_and_bottom_subtitle_intrusions(self):
        mobjects = [
            {
                "class": "Text",
                "tag": "canvas",
                "left": 0.0,
                "right": 3.0,
                "top": 3.2,  # Intruding into title buffer (y > 2.65)
                "bottom": 2.8,
                "width": 3.0,
                "height": 0.4,
            },
            {
                "class": "Text",
                "tag": "canvas",
                "left": 0.0,
                "right": 3.0,
                "top": -3.0,
                "bottom": -3.4,  # Intruding into subtitle buffer (y < -3.05)
                "width": 3.0,
                "height": 0.4,
            },
        ]
        issues = self.checker._check_mobjects_geometry(mobjects)
        title_issues = [i for i in issues if i.details.get("boundary") == "top_title"]
        sub_issues = [i for i in issues if i.details.get("boundary") == "bottom_subtitle"]
        self.assertGreater(len(title_issues), 0)
        self.assertGreater(len(sub_issues), 0)

    def test_obvious_overlap_detection(self):
        # Two distinct canvas squares heavily overlapping at the same coordinates
        mobjects = [
            {
                "class": "Square",
                "tag": "canvas",
                "left": 1.0,
                "right": 3.0,
                "top": 1.0,
                "bottom": -1.0,
                "width": 2.0,
                "height": 2.0,
            },
            {
                "class": "Circle",
                "tag": "canvas",
                "left": 1.2,
                "right": 2.8,
                "top": 0.8,
                "bottom": -0.8,
                "width": 1.6,
                "height": 1.6,
            },
        ]
        issues = self.checker._check_mobjects_geometry(mobjects)
        overlap_issues = [i for i in issues if i.issue_type == "OVERLAP"]
        self.assertGreater(len(overlap_issues), 0)
        self.assertIn("Obvious overlap", overlap_issues[0].description)

    def test_intentional_composition_not_flagged_as_overlap(self):
        # Arrow pointing at a Circle or a SurroundingRectangle around Text
        mobjects = [
            {
                "class": "Arrow",
                "tag": "canvas",
                "left": 1.0,
                "right": 2.5,
                "top": 1.0,
                "bottom": 0.0,
                "width": 1.5,
                "height": 1.0,
            },
            {
                "class": "Circle",
                "tag": "canvas",
                "left": 2.2,
                "right": 3.5,
                "top": 0.5,
                "bottom": -0.5,
                "width": 1.3,
                "height": 1.0,
            },
        ]
        issues = self.checker._check_mobjects_geometry(mobjects)
        overlap_issues = [i for i in issues if i.issue_type == "OVERLAP"]
        self.assertEqual(len(overlap_issues), 0)

    def test_duplicate_grid_cell_static_detection(self):
        code = '''
class SecScene(TeachingScene):
    def construct(self):
        self.setup_layout("Title", ["Line 1", "Line 2"])
        box1 = Square()
        self.place_at_grid(box1, 'C3')
        self.play(FadeIn(box1))
        box2 = Circle()
        self.place_at_grid(box2, 'C3')
        self.play(FadeIn(box2))
'''
        issues = self.checker._check_code_grid_placements(code)
        dup_issues = [i for i in issues if i.issue_type == "OVERLAP" and "C3" in i.description]
        self.assertGreater(len(dup_issues), 0)

    def test_unpositioned_object_detection(self):
        code = '''
class SecScene(TeachingScene):
    def construct(self):
        self.setup_layout("Title", ["Line 1"])
        orphaned_circle = Circle()
        self.play(Create(orphaned_circle))
'''
        issues = self.checker._check_code_grid_placements(code)
        bad_pos = [i for i in issues if i.issue_type == "BAD_PLACEMENT" and "orphaned_circle" in i.description]
        self.assertGreater(len(bad_pos), 0)

    def test_long_equation_without_scaling_detection(self):
        code = '''
class SecScene(TeachingScene):
    def construct(self):
        self.setup_layout("Title", ["Line 1"])
        long_formula = MathTex(r"\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi} \\quad \\text{and} \\quad \\sum_{k=1}^n k^2 = \\frac{n(n+1)(2n+1)}{6}")
        self.place_at_grid(long_formula, 'B3')
        self.play(Write(long_formula))
'''
        issues = self.checker._check_code_grid_placements(code)
        wide_issues = [i for i in issues if i.issue_type == "OUT_OF_SCREEN" and "long_formula" in i.description]
        self.assertGreater(len(wide_issues), 0)

    def test_local_fixer_resolves_row_a_intrusion(self):
        code = '''
        self.place_at_grid(obj, 'A3', scale_factor=1.2)
'''
        issues = [
            VisualIssue(
                issue_type="BAD_PLACEMENT",
                severity="warning",
                description="Row A placement with large scale",
                can_fix_locally=True,
                suggested_fix={"action": "move_to_row_b_or_scale_down", "cell": "A3"}
            )
        ]
        fixed_code, was_modified = self.fixer.apply_fixes(code, issues)
        self.assertTrue(was_modified)
        self.assertIn("'B3'", fixed_code)
        self.assertNotIn("'A3'", fixed_code)

    def test_local_fixer_resolves_col_6_overflow(self):
        code = '''
        self.place_at_grid(obj, 'C6', scale_factor=1.1)
'''
        issues = [
            VisualIssue(
                issue_type="OUT_OF_SCREEN",
                severity="warning",
                description="Col 6 placement with large scale",
                can_fix_locally=True,
                suggested_fix={"action": "move_to_col_5_or_scale_down", "cell": "C6"}
            )
        ]
        fixed_code, was_modified = self.fixer.apply_fixes(code, issues)
        self.assertTrue(was_modified)
        self.assertIn("'C5'", fixed_code)
        self.assertNotIn("'C6'", fixed_code)

    def test_local_fixer_adds_grid_placement_for_unpositioned_object(self):
        code = '''
        circ = Circle()
        self.play(Create(circ))
'''
        issues = [
            VisualIssue(
                issue_type="BAD_PLACEMENT",
                severity="warning",
                description="Unpositioned object circ",
                details={"object": "circ", "line": 2},
                can_fix_locally=True,
                suggested_fix={"action": "add_grid_placement", "object": "circ"}
            )
        ]
        fixed_code, was_modified = self.fixer.apply_fixes(code, issues)
        self.assertTrue(was_modified)
        self.assertIn("self.place_at_grid(circ, 'C3'", fixed_code)

    def test_to_edge_left_detection_and_fix(self):
        code = '''class TestScene(TeachingScene):
    def construct(self):
        self.setup_layout("Title", ["L1", "L2"])
        diag = VGroup()
        diag.to_edge(LEFT)
        self.play(Create(diag))
'''
        issues = self.checker._check_code_grid_placements(code)
        self.assertTrue(any(i.suggested_fix and i.suggested_fix.get("action") == "fix_to_edge_left" for i in issues))
        fixed_code, was_modified = self.fixer.apply_fixes(code, issues)
        self.assertTrue(was_modified)
        self.assertNotIn("diag.to_edge(LEFT)", fixed_code)
        self.assertIn("self.place_at_grid(diag, 'C3'", fixed_code)

    def test_next_to_lecture_detection_and_fix(self):
        code = '''class TestScene(TeachingScene):
    def construct(self):
        self.setup_layout("Title", ["L1", "L2"])
        diag = Rectangle()
        diag.next_to(self.lecture[0], DOWN)
        self.play(Create(diag))
'''
        issues = self.checker._check_code_grid_placements(code)
        self.assertTrue(any(i.suggested_fix and i.suggested_fix.get("action") == "fix_next_to_lecture" for i in issues))
        fixed_code, was_modified = self.fixer.apply_fixes(code, issues)
        self.assertTrue(was_modified)
        self.assertNotIn("diag.next_to(self.lecture[0], DOWN)", fixed_code)
        self.assertIn("self.place_at_grid(diag, 'C3'", fixed_code)

    def test_duplicate_title_detection_and_neutralization(self):
        code = '''class TestScene(TeachingScene):
    def construct(self):
        self.setup_layout("Title", ["L1", "L2"])
        title = Text("Sec 1: Redundant Title").to_edge(UP)
        self.play(FadeIn(title))
'''
        issues = self.checker._check_code_grid_placements(code)
        self.assertTrue(any(i.suggested_fix and i.suggested_fix.get("action") == "fix_duplicate_title" for i in issues))
        fixed_code, was_modified = self.fixer.apply_fixes(code, issues)
        self.assertTrue(was_modified)
        self.assertIn("[AUTO-FIX] Neutralized duplicate title", fixed_code)

    def test_circle_and_text_not_flagged_as_overlap(self):
        # A node circle containing a centered text label
        mobjects = [
            {
                "class": "Circle",
                "tag": "canvas",
                "left": 1.0,
                "right": 3.0,
                "top": 1.0,
                "bottom": -1.0,
                "width": 2.0,
                "height": 2.0,
                "center": [2.0, 0.0, 0.0],
            },
            {
                "class": "Text",
                "tag": "canvas",
                "left": 1.7,
                "right": 2.3,
                "top": 0.4,
                "bottom": -0.4,
                "width": 0.6,
                "height": 0.8,
                "center": [2.0, 0.0, 0.0],
            },
        ]
        issues = self.checker._check_mobjects_geometry(mobjects)
        overlap_issues = [i for i in issues if i.issue_type == "OVERLAP"]
        self.assertEqual(len(overlap_issues), 0, "Text centered inside a circle must not be flagged as overlap")

    def test_square_and_mathtex_not_flagged_as_overlap(self):
        # A container box with MathTex inside
        mobjects = [
            {
                "class": "Square",
                "tag": "canvas",
                "left": 0.0,
                "right": 3.0,
                "top": 1.5,
                "bottom": -1.5,
                "width": 3.0,
                "height": 3.0,
                "center": [1.5, 0.0, 0.0],
            },
            {
                "class": "MathTex",
                "tag": "canvas",
                "left": 0.5,
                "right": 2.5,
                "top": 0.6,
                "bottom": -0.6,
                "width": 2.0,
                "height": 1.2,
                "center": [1.5, 0.0, 0.0],
            },
        ]
        issues = self.checker._check_mobjects_geometry(mobjects)
        overlap_issues = [i for i in issues if i.issue_type == "OVERLAP"]
        self.assertEqual(len(overlap_issues), 0, "MathTex formula inside a container square must not be flagged as overlap")

    def test_dot_inside_circle_not_flagged_as_overlap(self):
        # Concentric marker dot inside a node circle
        mobjects = [
            {
                "class": "Circle",
                "tag": "canvas",
                "left": 1.0,
                "right": 3.0,
                "top": 1.0,
                "bottom": -1.0,
                "width": 2.0,
                "height": 2.0,
                "center": [2.0, 0.0, 0.0],
            },
            {
                "class": "Dot",
                "tag": "canvas",
                "left": 1.9,
                "right": 2.1,
                "top": 0.1,
                "bottom": -0.1,
                "width": 0.2,
                "height": 0.2,
                "center": [2.0, 0.0, 0.0],
            },
        ]
        issues = self.checker._check_mobjects_geometry(mobjects)
        overlap_issues = [i for i in issues if i.issue_type == "OVERLAP"]
        self.assertEqual(len(overlap_issues), 0, "Dot marker inside a circle must not be flagged as overlap")

    def test_shape_and_label_at_same_grid_cell_not_separated(self):
        # Placing a circle and its label at 'C3' is an intentional layout
        code = '''class TestScene(TeachingScene):
    def construct(self):
        self.setup_layout("Title", ["L1", "L2"])
        node_circ = Circle()
        node_lbl = Text("Node A")
        self.place_at_grid(node_circ, 'C3')
        self.place_at_grid(node_lbl, 'C3')
        self.play(Create(node_circ), Write(node_lbl))
'''
        issues = self.checker._check_code_grid_placements(code)
        cell_c3_overlaps = [i for i in issues if i.issue_type == "OVERLAP" and "C3" in i.description]
        self.assertEqual(len(cell_c3_overlaps), 0, "Placing a shape and its label at the same cell must not trigger duplicate cell collision")

    def test_intermediate_checkpoints_detection(self):
        # Test that checkpoints in layout_data are inspected for temporary mid-animation collisions
        import tempfile, json
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            layout_data = {
                "scene": "TestScene",
                "frame_width": 14.222,
                "frame_height": 8.0,
                "safe_bounds": {"left": -2.15, "right": 6.65, "top": 2.65, "bottom": -2.85},
                "mobjects": [],  # Clean final state
                "checkpoints": [
                    {
                        "checkpoint": "step_2",
                        "time": 3.5,
                        "mobjects": [
                            {
                                "class": "Square",
                                "tag": "canvas",
                                "left": 6.0,
                                "right": 7.8,  # Temporary boundary overflow at step 2
                                "top": 1.0,
                                "bottom": 0.0,
                                "width": 1.8,
                                "height": 1.0
                            }
                        ]
                    }
                ]
            }
            with open(tmp_path / "layout_TestScene.json", "w") as f:
                json.dump(layout_data, f)

            code = "class TestScene(TeachingScene):\n    pass\n"
            result = self.checker.check_section("test", code, tmp_path)
            self.assertFalse(result.is_clean)
            self.assertTrue(any("step_2" in i.description and i.issue_type == "OUT_OF_SCREEN" for i in result.issues))

    def test_unpositioned_multiline_object_fix_preserves_syntax(self):
        code = '''
class SectionScene(TeachingScene):
    def construct(self):
        edge_labels = VGroup(
            Text("1"),
            Text("4")
        )
        self.play(FadeIn(edge_labels))
'''
        issue = VisualIssue(
            issue_type="BAD_PLACEMENT",
            severity="warning",
            description="Unpositioned object edge_labels",
            details={"object": "edge_labels", "line": 4},
            can_fix_locally=True,
            suggested_fix={"action": "add_grid_placement", "object": "edge_labels"}
        )
        fixed_code, modified = self.fixer.apply_fixes(code, [issue])
        self.assertTrue(modified)
        import ast
        # Fixed code must be valid syntax (not broken by inserting inside VGroup)
        tree = ast.parse(fixed_code)
        self.assertIsNotNone(tree)
        self.assertIn("self.place_at_grid(edge_labels", fixed_code)

    def test_internally_positioned_vgroup_not_flagged(self):
        code = '''
class SectionScene(TeachingScene):
    def construct(self):
        edges = VGroup(
            Line(ORIGIN, UP),
            Line(ORIGIN, RIGHT)
        )
        edge_labels = VGroup(
            Text("1").next_to(edges[0], UP),
            Text("4").next_to(edges[1], UP)
        )
        self.play(FadeIn(edges), FadeIn(edge_labels))
'''
        issues = self.checker._check_code_grid_placements(code)
        unpositioned = [i for i in issues if i.issue_type == "BAD_PLACEMENT"]
        self.assertEqual(len(unpositioned), 0)

    def test_vgroup_mutual_overlap_detection(self):
        mobjects = [
            {
                "class": "VGroup",
                "tag": "canvas",
                "left": 1.0,
                "right": 3.0,
                "top": 2.0,
                "bottom": 0.0,
                "width": 2.0,
                "height": 2.0,
            },
            {
                "class": "VGroup",
                "tag": "canvas",
                "left": 2.0,
                "right": 4.0,
                "top": 1.5,
                "bottom": -0.5,
                "width": 2.0,
                "height": 2.0,
            },
        ]
        issues = self.checker._check_mobjects_geometry(mobjects)
        overlap_issues = [i for i in issues if i.issue_type == "OVERLAP"]
        self.assertEqual(len(overlap_issues), 1, "Two overlapping VGroups must be flagged as overlap")

    def test_coordinate_system_not_flagged_as_overlap(self):
        mobjects = [
            {
                "class": "NumberPlane",
                "tag": "canvas",
                "left": -3.0,
                "right": 3.0,
                "top": 3.0,
                "bottom": -3.0,
                "width": 6.0,
                "height": 6.0,
            },
            {
                "class": "SafeMathTex",
                "tag": "canvas",
                "left": 0.0,
                "right": 1.0,
                "top": 1.0,
                "bottom": 0.5,
                "width": 1.0,
                "height": 0.5,
            },
        ]
        issues = self.checker._check_mobjects_geometry(mobjects)
        overlap_issues = [i for i in issues if i.issue_type == "OVERLAP"]
        self.assertEqual(len(overlap_issues), 0, "NumberPlane background must not collide with plots/labels")


if __name__ == "__main__":
    unittest.main()
