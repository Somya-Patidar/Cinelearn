"""
Unit Tests for CineLearn Phase 2: VisualComponent Protocol and VisualRegistry
=============================================================================
Validates universal component protocol, strict boundary adherence in auto_fit_container,
semantic handle registration, Tier 3 ProceduralComponent fallback, and domain-independent
component registry functionality.
"""

import pytest
import numpy as np
from manim import Circle, Square, Text, VGroup

from src.domains import VisualComponent, ProceduralComponent, VisualRegistry


class TestVisualComponentProtocol:
    """Test suite for the VisualComponent base protocol."""

    def test_visual_component_is_vgroup(self):
        c1 = Circle(radius=0.5)
        c2 = Square(side_length=1.0)
        comp = VisualComponent(c1, c2)
        assert isinstance(comp, VGroup)
        assert len(comp.submobjects) == 2

    def test_auto_fit_container_strictly_respects_boundaries(self):
        """
        CRITICAL TEST FOR USER CONDITION:
        Components must NEVER exceed container boundaries.
        When scaled down heavily below min_scale_threshold, is_compressed is flagged,
        but the scale is NOT clamped upward to breach the boundary.
        """
        # Huge component: 100 x 50
        huge_box = Square(side_length=100.0)
        comp = VisualComponent(huge_box)

        # Target container: 10.0 width x 5.0 height with padding 0.04
        max_w, max_h = 10.0, 5.0
        scale_factor, is_compressed = comp.auto_fit_container(max_w, max_h, padding=0.04)

        # Check strict containment: width and height MUST be within container
        assert comp.width <= max_w
        assert comp.height <= max_h
        # Must be flagged as compressed because scale_factor < 0.55
        assert is_compressed is True
        assert scale_factor < 0.55
        # Crucially: comp.width must NOT have expanded to min_scale_threshold * 100 = 55.0!
        assert comp.width <= 10.0

    def test_auto_fit_container_already_fitting_object(self):
        # Small component (1.0 x 1.0) inside large container (10.0 x 10.0)
        small_box = Square(side_length=1.0)
        comp = VisualComponent(small_box)

        scale_factor, is_compressed = comp.auto_fit_container(10.0, 10.0, padding=0.04)
        assert scale_factor == 1.0
        assert is_compressed is False
        assert comp.width == pytest.approx(1.0, rel=1e-2)

    def test_semantic_handles(self):
        atom_o = Circle(radius=0.4, color="#FF0000")
        atom_h1 = Circle(radius=0.2, color="#FFFFFF")
        atom_h2 = Circle(radius=0.2, color="#FFFFFF")

        water_mol = VisualComponent(atom_o, atom_h1, atom_h2)
        water_mol.register_semantic_handle("oxygen", atom_o)
        water_mol.register_semantic_handle("hydrogen_1", atom_h1)
        water_mol.register_semantic_handle("hydrogen_2", atom_h2)

        assert water_mol.get_semantic_handle("oxygen") is atom_o
        assert water_mol.get_semantic_handle("hydrogen_1") is atom_h1
        assert water_mol.get_semantic_handle("hydrogen_2") is atom_h2
        assert water_mol.get_semantic_handle("non_existent") is None

    def test_bounding_box_extraction(self):
        sq = Square(side_length=2.0).move_to(np.array([1.0, 2.0, 0.0]))
        comp = VisualComponent(sq)
        bbox = comp.get_bounding_box()

        assert bbox["left"] == pytest.approx(0.0, abs=1e-2)
        assert bbox["right"] == pytest.approx(2.0, abs=1e-2)
        assert bbox["bottom"] == pytest.approx(1.0, abs=1e-2)
        assert bbox["top"] == pytest.approx(3.0, abs=1e-2)
        assert bbox["width"] == pytest.approx(2.0, abs=1e-2)
        assert bbox["height"] == pytest.approx(2.0, abs=1e-2)

    def test_get_manifest(self):
        class SampleComponent(VisualComponent):
            """A test component for Dijkstra graph visualization."""
            component_id = "SampleGraph"
            domain = "cs"
            preferred_aspect = 1.75

            def __init__(self, node_count: int = 5, directed: bool = False, **kwargs):
                super().__init__(**kwargs)
                self.node_count = node_count

            def animate_highlight_shortest_path(self, path):
                pass

        manifest = SampleComponent.get_manifest()
        assert manifest["component_id"] == "SampleGraph"
        assert manifest["domain"] == "cs"
        assert "Dijkstra" in manifest["description"]
        assert manifest["preferred_aspect"] == 1.75
        assert "node_count" in manifest["init_params"]
        assert "animate_highlight_shortest_path" in manifest["semantic_actions"]


class TestProceduralComponent:
    """Test suite for Tier 3 ProceduralComponent fallback."""

    def test_procedural_component_wrapping(self):
        shapes = VGroup(Circle(), Square())
        proc = ProceduralComponent(shapes, name="ReactionEnergyDiagram")

        assert isinstance(proc, VisualComponent)
        assert proc.procedural_name == "ReactionEnergyDiagram"
        assert proc.component_id == "ProceduralComponent"
        assert proc.domain == "generic"

    def test_procedural_component_scales_and_docks(self):
        from src.domains.layout import GenericContainerLayout
        layout = GenericContainerLayout()

        proc = ProceduralComponent(Square(side_length=15.0), name="OversizedCustomDiagram")
        layout.dock_main(proc)

        # Must be scaled down to main_canvas
        assert proc.width <= layout.main_canvas.width
        assert proc.height <= layout.main_canvas.height
        assert proc.get_top()[1] <= layout.main_canvas.top + 0.05
        assert proc.get_bottom()[1] >= layout.main_canvas.bottom - 0.05


class TestVisualRegistry:
    """Test suite for domain-independent VisualRegistry."""

    def setup_method(self):
        VisualRegistry.clear()

    def teardown_method(self):
        VisualRegistry.clear()

    def test_register_and_retrieve_component(self):
        @VisualRegistry.register("cs", "GraphVisual")
        class MockGraph(VisualComponent):
            pass

        retrieved = VisualRegistry.get("cs", "GraphVisual")
        assert retrieved is MockGraph

        # Case-insensitive retrieval
        retrieved_upper = VisualRegistry.get("CS", "GRAPHVISUAL")
        assert retrieved_upper is MockGraph

    def test_register_multiple_domains(self):
        @VisualRegistry.register("cs", "GraphVisual")
        class MockGraph(VisualComponent):
            pass

        @VisualRegistry.register("chemistry", "MoleculeVisual")
        class MockMolecule(VisualComponent):
            pass

        @VisualRegistry.register("math", "FunctionPlotVisual")
        class MockPlot(VisualComponent):
            pass

        domains = VisualRegistry.list_domains()
        assert "chemistry" in domains
        assert "cs" in domains
        assert "math" in domains

        cs_comps = VisualRegistry.list_components("cs")
        assert len(cs_comps) == 1
        assert cs_comps[0]["component_id"] == "GraphVisual"

        chem_comps = VisualRegistry.list_components("chemistry")
        assert len(chem_comps) == 1
        assert chem_comps[0]["component_id"] == "MoleculeVisual"

    def test_schema_manifest_json_serializable(self):
        @VisualRegistry.register("cs", "GraphVisual")
        class MockGraph(VisualComponent):
            """A directed or undirected weighted graph visual."""
            def __init__(self, vertices=None, **kwargs):
                super().__init__(**kwargs)

        manifest = VisualRegistry.get_schema_manifest()
        assert "cs" in manifest
        assert isinstance(manifest["cs"], list)
        assert manifest["cs"][0]["component_id"] == "GraphVisual"

    def test_get_unregistered_component_returns_none(self):
        assert VisualRegistry.get("cs", "NonExistent") is None
        assert VisualRegistry.get("biology", "CellOrganelle") is None

    def test_discover_domains_structure(self):
        # Verifies discover_domains scans package without error
        discovered = VisualRegistry.discover_domains("src.domains")
        assert isinstance(discovered, list)

    def test_recursive_discover_domains(self, tmp_path, monkeypatch):
        """Verifies recursive discovery walks through nested subpackages and registers components."""
        import sys
        # Create a mock nested package structure: mock_pkg/subdomain/nested_mod.py
        pkg_dir = tmp_path / "mock_pkg"
        sub_dir = pkg_dir / "subdomain"
        sub_dir.mkdir(parents=True)
        (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
        (sub_dir / "__init__.py").write_text("", encoding="utf-8")

        mod_code = (
            "from src.domains import VisualComponent, VisualRegistry\n"
            "@VisualRegistry.register('test_nested', 'RecursiveMockComponent')\n"
            "class RecursiveMockComponent(VisualComponent):\n"
            "    pass\n"
        )
        (sub_dir / "nested_mod.py").write_text(mod_code, encoding="utf-8")

        monkeypatch.syspath_prepend(str(tmp_path))

        discovered = VisualRegistry.discover_domains("mock_pkg")
        assert any("nested_mod" in d for d in discovered)
        retrieved = VisualRegistry.get("test_nested", "RecursiveMockComponent")
        assert retrieved is not None
        assert retrieved.component_id == "RecursiveMockComponent"
