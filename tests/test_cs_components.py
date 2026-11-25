"""
Tests for Computer Science Visual Components
===========================================
Validates GraphVisual and DPTableMatrix functionality, semantic handles,
animation methods, container docking, zero overlap, and subtitle clearance.
"""

import sys
from pathlib import Path

# Ensure CineLearn root is in sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
import numpy as np
from manim import Scene, tempconfig, FadeIn

from src.domains import VisualRegistry
from src.domains.cs import GraphVisual, DPTableMatrix
from prompts import base_class

# Dynamically extract TeachingScene from base_class template
_scope = {}
exec(base_class.base_class if hasattr(base_class, "base_class") else base_class, _scope)
TeachingScene = _scope["TeachingScene"]


@pytest.fixture(autouse=True)
def ensure_cs_discovered():
    VisualRegistry.discover_domains("src.domains")


class TestGraphVisual:
    def test_registration(self):
        cls = VisualRegistry.get("cs", "GraphVisual")
        assert cls is GraphVisual
        manifest = GraphVisual.get_manifest()
        assert manifest["component_id"] == "GraphVisual"
        assert manifest["domain"] == "cs"
        assert "animate_highlight_node" in manifest["semantic_actions"]
        assert "animate_relax_edge" in manifest["semantic_actions"]

    def test_circular_layout(self):
        vertices = ["A", "B", "C", "D"]
        edges = [("A", "B", 3), ("B", "C", 5), ("C", "D", 2), ("D", "A", 1)]
        graph = GraphVisual(vertices=vertices, edges=edges, layout="circular")

        # Verify semantic handles
        for v in vertices:
            assert graph.get_semantic_handle(f"node_{v}") is not None
            assert graph.get_semantic_handle(f"circle_{v}") is not None
            assert graph.get_semantic_handle(f"label_{v}") is not None

        assert graph.get_semantic_handle("edge_A_B") is not None
        assert graph.get_semantic_handle("edge_B_A") is not None  # Undirected alias
        assert graph.get_semantic_handle("weight_A_B") is not None

    def test_spring_layout(self):
        vertices = [1, 2, 3, 4, 5]
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
        graph = GraphVisual(vertices=vertices, edges=edges, layout="spring", directed=True)

        assert graph.get_semantic_handle("node_1") is not None
        assert graph.get_semantic_handle("edge_1_2") is not None
        # Directed: reverse edge handle should NOT be registered
        assert graph.get_semantic_handle("edge_2_1") is None

    def test_tree_layout(self):
        vertices = ["root", "L", "R", "LL", "LR"]
        edges = [("root", "L"), ("root", "R"), ("L", "LL"), ("L", "LR")]
        graph = GraphVisual(vertices=vertices, edges=edges, layout="tree", root_vertex="root")

        pos_root = graph.positions["root"]
        pos_l = graph.positions["L"]
        assert pos_root[1] > pos_l[1], "Root must be vertically higher than children in tree layout"

    def test_explicit_layout(self):
        custom_pos = {"A": [0, 0, 0], "B": [2, 1, 0]}
        graph = GraphVisual(vertices=["A", "B"], edges=[("A", "B", 10)], layout=custom_pos)
        assert np.allclose(graph.positions["A"], [0, 0, 0])
        assert np.allclose(graph.positions["B"], [2, 1, 0])

    def test_semantic_animations(self):
        graph = GraphVisual(vertices=["A", "B"], edges=[("A", "B", 4)])
        anim_node = graph.animate_highlight_node("A")
        assert anim_node is not None

        anim_visit = graph.animate_visit_node("A")
        assert anim_visit is not None

        anim_edge = graph.animate_highlight_edge("A", "B")
        assert anim_edge is not None

        anim_traverse = graph.animate_traverse_edge("A", "B")
        assert anim_traverse is not None

        anim_update_lbl = graph.animate_update_node_label("B", "dist: 4")
        assert anim_update_lbl is not None

        anim_relax = graph.animate_relax_edge("A", "B", new_v_label="4")
        assert anim_relax is not None


class TestDPTableMatrix:
    def test_registration(self):
        cls = VisualRegistry.get("cs", "DPTableMatrix")
        assert cls is DPTableMatrix
        manifest = DPTableMatrix.get_manifest()
        assert manifest["component_id"] == "DPTableMatrix"
        assert manifest["domain"] == "cs"
        assert "animate_update_cell" in manifest["semantic_actions"]
        assert "animate_compute_dependency" in manifest["semantic_actions"]

    def test_grid_and_headers(self):
        values = [[0, 1, 2], [3, 4, 5]]
        row_lbls = ["r0", "r1"]
        col_lbls = ["c0", "c1", "c2"]
        table = DPTableMatrix(
            matrix_values=values,
            row_labels=row_lbls,
            col_labels=col_lbls,
            table_title="Memoization Table",
        )

        assert table.num_rows == 2
        assert table.num_cols == 3

        for r in range(2):
            for c in range(3):
                assert table.get_semantic_handle(f"cell_{r}_{c}") is not None
                assert table.get_semantic_handle(f"box_{r}_{c}") is not None
                assert table.get_semantic_handle(f"val_{r}_{c}") is not None

        assert table.get_semantic_handle("row_0") is not None
        assert table.get_semantic_handle("col_1") is not None
        assert table.get_semantic_handle("row_header_0") is not None
        assert table.get_semantic_handle("col_header_2") is not None
        assert table.get_semantic_handle("table_title") is not None

    def test_semantic_animations(self):
        values = [[0, 0, 0], [0, 0, 0]]
        table = DPTableMatrix(matrix_values=values)

        anim_upd = table.animate_update_cell(0, 1, 42)
        assert anim_upd is not None

        anim_highlight = table.animate_highlight_cell(1, 2)
        assert anim_highlight is not None

        anim_row = table.animate_highlight_row(0)
        assert anim_row is not None

        anim_col = table.animate_highlight_col(1)
        assert anim_col is not None

        anim_dep = table.animate_compute_dependency(
            target_rc=(1, 2),
            dependency_rcs=[(0, 2), (1, 1)],
        )
        assert anim_dep is not None

        anim_fill = table.animate_fill_row(0, [10, 20, 30])
        assert anim_fill is not None


class TestCSComponentsDockingAndRendering:
    def test_graph_and_table_composition_docking(self):
        """
        Tests composing a GraphVisual in dock_main and DPTableMatrix in dock_inspector
        in a full TeachingScene, verifying zero overlap and subtitle clearance.
        """
        scene = TeachingScene()
        scene.setup_layout(
            "Dijkstra Shortest Path: Worked Example",
            [
                "Initialize tentative distances to infinity",
                "Extract minimum unvisited node A with distance 0",
                "Relax outgoing edges to neighbors B and C",
            ],
        )

        # 1. Main visual: GraphVisual
        vertices = ["A", "B", "C", "D", "E"]
        edges = [("A", "B", 4), ("A", "C", 2), ("B", "C", 1), ("B", "D", 5), ("C", "D", 8), ("C", "E", 10), ("D", "E", 2)]
        graph = GraphVisual(vertices=vertices, edges=edges, layout="circular")
        scene.dock_main(graph)

        # 2. Inspector visual: DPTableMatrix
        table_vals = [
            ["0", "inf", "inf", "inf", "inf"],
            ["0", "4", "2", "inf", "inf"],
        ]
        table = DPTableMatrix(
            matrix_values=table_vals,
            row_labels=["Init", "Step 1"],
            col_labels=["A", "B", "C", "D", "E"],
            table_title="Distance Table",
        )
        scene.dock_inspector(table)

        # Verify zero overlap between graph and table
        bb_graph = graph.get_bounding_box()
        bb_table = table.get_bounding_box()

        has_overlap, ratio = scene.layout.check_overlap(graph, table)
        assert not has_overlap, f"Graph and table overlap detected! Ratio: {ratio}"
        assert ratio == 0.0

        # Verify vertical separation: graph must be completely above table
        assert bb_graph["bottom"] > bb_table["top"], (
            f"Graph bottom ({bb_graph['bottom']}) must be above table top ({bb_table['top']})"
        )

        # Verify subtitle clearance: bottom of table must be >= -2.65
        assert bb_table["bottom"] >= -2.65, (
            f"Table bottom ({bb_table['bottom']}) encroaches on subtitle area (below -2.65)"
        )

        # Verify lecture caption clearance: x_left must be >= -2.15
        assert bb_graph["left"] >= -2.15
        assert bb_table["left"] >= -2.15

        # Verify right boundary: x_right <= 6.65
        assert bb_graph["right"] <= 6.65
        assert bb_table["right"] <= 6.65

    def test_render_scene_with_cs_components(self, tmp_path):
        """
        Renders a Manim scene exercising GraphVisual and DPTableMatrix animations.
        """
        class SampleDijkstraScene(TeachingScene):
            def construct(self):
                self.setup_layout(
                    "Dijkstra Algorithm Demonstration",
                    [
                        "Start at node A with tentative distance 0",
                        "Evaluate neighbors B and C, updating shortest estimates",
                    ],
                )

                # Primary graph visual
                graph = GraphVisual(
                    vertices=["A", "B", "C"],
                    edges=[("A", "B", 4), ("A", "C", 2), ("B", "C", 1)],
                    layout="circular",
                )
                self.dock_main(graph)

                # Secondary distance table
                table = DPTableMatrix(
                    matrix_values=[["0", "inf", "inf"]],
                    row_labels=["Dist"],
                    col_labels=["A", "B", "C"],
                    table_title="Current Distances",
                )
                self.dock_inspector(table)

                self.play(FadeIn(graph), FadeIn(table))
                self.play(self.activate_caption(0))
                self.play(graph.animate_visit_node("A"))
                self.play(self.activate_caption(1))
                self.play(
                    graph.animate_relax_edge("A", "C", new_v_label="C: 2"),
                    table.animate_update_cell(0, 2, "2"),
                )

        with tempconfig({
            "quality": "low_quality",
            "preview": False,
            "format": "mp4",
            "media_dir": str(tmp_path),
            "disable_caching": True,
        }):
            scene = SampleDijkstraScene()
            scene.render()

        rendered_files = list(tmp_path.glob("**/*.mp4"))
        assert len(rendered_files) > 0, "Expected at least one rendered MP4 file"
        assert rendered_files[0].stat().st_size > 1000, "Rendered video file is too small or corrupt"
