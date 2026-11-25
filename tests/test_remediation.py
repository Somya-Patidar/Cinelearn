"""
Regression and Integration Tests for Phase 4 Remediation Fixes:
1. Subprocess Registry Discovery (fresh Python subshell)
2. Direct Component Availability (all 6 domain components directly accessible)
3. MathTex Accidental Double-Brace Sanitization & SafeMathTex Fallback
4. Docking API Clarification & Callable DockingHandler
5. Component API Validation (DPTableMatrix vs ProceduralComponent, GraphVisual vs native Graph)
6. Standalone Subprocess Manim Render Validation
"""

import sys
import os
import subprocess
from pathlib import Path
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

PYTHON_EXEC = sys.executable


# ============================================================================
# Test 1: Subprocess Registry Discovery
# ============================================================================
def test_subprocess_registry_discovery():
    """Verify that a fresh Python subprocess discovers all domains and components."""
    cmd = [
        PYTHON_EXEC,
        "-c",
        """
import sys
sys.path.insert(0, '.')
from src.domains import VisualRegistry

domains = VisualRegistry.list_domains()
assert 'cs' in domains, f"cs missing from {domains}"
assert 'chemistry' in domains, f"chemistry missing from {domains}"
assert 'math' in domains, f"math missing from {domains}"

expected_components = [
    ('cs', 'GraphVisual'),
    ('cs', 'DPTableMatrix'),
    ('chemistry', 'MoleculeVisual'),
    ('chemistry', 'ReactionEnergyDiagram'),
    ('math', 'FunctionCurve'),
    ('math', 'NumberLineVisual'),
]
for domain, comp_name in expected_components:
    cls = VisualRegistry.get(domain, comp_name)
    assert cls is not None, f"Registry failed to return {domain}:{comp_name}"
    assert cls.component_id == comp_name
print("SUBPROCESS_DISCOVERY_SUCCESS")
""",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert res.returncode == 0, f"Subprocess failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "SUBPROCESS_DISCOVERY_SUCCESS" in res.stdout


# ============================================================================
# Test 2: Direct Component Availability in Generated Script Namespace
# ============================================================================
def test_direct_component_availability_in_namespace():
    """Verify that all 6 domain components are directly available in scene namespace without extra imports."""
    cmd = [
        PYTHON_EXEC,
        "-c",
        """
import sys
sys.path.insert(0, '.')
from prompts.base_class import *

# All 6 components should be in the global namespace
assert GraphVisual is not None
assert DPTableMatrix is not None
assert MoleculeVisual is not None
assert ReactionEnergyDiagram is not None
assert FunctionCurve is not None
assert NumberLineVisual is not None

# Quick instantiation check
gv = GraphVisual(vertices=['A', 'B'], edges=[('A', 'B', 5)])
assert len(gv.vertices) == 2

dp = DPTableMatrix(matrix_values=[['1', '2'], ['3', '4']])
assert dp.shape == (2, 2)

mol = MoleculeVisual(atoms=[{'symbol': 'H', 'position': [0, 0, 0]}])
assert len(mol.atoms) == 1

diag = ReactionEnergyDiagram(e_reactants=1.0, e_transition_state=3.0, e_products=0.5)
assert diag.ea == 2.0

fc = FunctionCurve(function='quadratic')
assert callable(fc.func)

nl = NumberLineVisual(x_range=[-5, 5, 1])
assert list(nl.x_range) == [-5, 5, 1]

print("DIRECT_AVAILABILITY_SUCCESS")
""",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert res.returncode == 0, f"Namespace test failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "DIRECT_AVAILABILITY_SUCCESS" in res.stdout


# ============================================================================
# Test 3: MathTex Accidental Double-Brace Sanitization
# ============================================================================
def test_mathtex_double_brace_sanitization():
    """Verify pre-render sanitizer collapses accidental double braces while preserving Python dicts/sets."""
    from src.agent import TeachingVideoAgent

    agent = TeachingVideoAgent.__new__(TeachingVideoAgent)

    code_with_double_braces = '''
from manim import *
class MathScene(Scene):
    def construct(self):
        title = Text("Derivative Definition")
        # Accidental double braces in MathTex
        formula = MathTex(r"f'(x) = \\lim_{{h \\to 0}} \\frac{{f(x+h) - f(x)}}{{h}}", color=WHITE)
        # Accidental double braces in safe_math
        opt = self.safe_math(r"\\Delta G = \\Delta H - T\\Delta S", r"{{kJ/mol}}")
        # Regular Python dictionary and set that MUST NOT be corrupted
        params = {"learning_rate": 0.01, "batch_size": 32}
        unique_nodes = {1, 2, 3}
'''
    sanitized_code, modified = agent.sanitize_code_components_and_docking(code_with_double_braces)
    assert modified is True
    assert "{{h \\to 0}}" not in sanitized_code
    assert "{h \\to 0}" in sanitized_code
    assert "\\frac{f(x+h) - f(x)}{h}" in sanitized_code
    assert "{{kJ/mol}}" not in sanitized_code
    assert "{kJ/mol}" in sanitized_code

    # Python dict and set preserved
    assert 'params = {"learning_rate": 0.01, "batch_size": 32}' in sanitized_code
    assert "unique_nodes = {1, 2, 3}" in sanitized_code

    # AST syntax check on sanitized code
    import ast
    ast.parse(sanitized_code)


def test_safe_mathtex_runtime_tolerance():
    """Verify SafeMathTex drop-in handles double braces and LaTeX compilation errors without crashing."""
    cmd = [
        PYTHON_EXEC,
        "-c",
        r"""
import sys
sys.path.insert(0, '.')
from prompts.base_class import *

# 1. Double braces inside MathTex (which crashes native Manim MathTex)
m1 = MathTex(r"f'(x) = \lim_{{h \to 0}} \frac{{f(x+h) - f(x)}}{{h}}")
assert len(m1.submobjects) > 0

# 2. Broken TeX macro fallback to Text
m2 = MathTex(r"\unknownmacro{broken\syntax")
assert len(m2.submobjects) > 0

# 3. safe_math helper
scene = TeachingScene()
m3 = scene.safe_math(r"\int_{{0}}^{{\infty}} e^{-x^2} dx = \frac{{\sqrt{\pi}}}{{2}}")
assert len(m3.submobjects) > 0

print("SAFE_MATHTEX_SUCCESS")
""",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert res.returncode == 0, f"SafeMathTex test failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "SAFE_MATHTEX_SUCCESS" in res.stdout


# ============================================================================
# Test 4: Docking API Clarification & DockingHandler
# ============================================================================
def test_docking_api_callable_and_properties():
    """Verify that self.dock_* are callable and also support .get_center() and .center property access."""
    from prompts.base_class import TeachingScene, Circle, Text

    scene = TeachingScene()
    scene.setup_layout("Docking Test", ["Line 1", "Line 2"])

    # Method call works
    c1 = Circle()
    docked = scene.dock_main(c1)
    assert docked is c1

    # Inspector method call works
    t1 = Text("Status Table")
    docked_insp = scene.dock_inspector(t1)
    assert docked_insp is t1

    # get_center() on method works without AttributeError
    center1 = scene.dock_inspector.get_center()
    assert isinstance(center1, np.ndarray)
    assert len(center1) == 3
    assert np.allclose(center1, scene.layout.viewports["inspector"].center)

    # .center property on method works
    center2 = scene.dock_inspector.center
    assert isinstance(center2, np.ndarray)
    assert np.allclose(center2, scene.layout.viewports["inspector"].center)

    # dock_main center
    main_center = scene.dock_main.get_center()
    assert np.allclose(main_center, scene.layout.viewports["main"].center)


def test_docking_pre_render_sanitizer():
    """Verify pre-render sanitizer rewrites self.dock_*.get_center() in code."""
    from src.agent import TeachingVideoAgent

    agent = TeachingVideoAgent.__new__(TeachingVideoAgent)

    code = '''
table.move_to(self.dock_inspector.get_center())
main_card.move_to(self.dock_main.center)
left_card.move_to(self.dock_split_left.get_center())
'''
    sanitized, modified = agent.sanitize_code_components_and_docking(code)
    assert modified is True
    assert 'self.dock_inspector.get_center()' not in sanitized
    assert 'self.layout.viewports["inspector"].center' in sanitized
    assert 'self.dock_main.center' not in sanitized
    assert 'self.layout.viewports["main"].center' in sanitized
    assert 'self.layout.viewports["split_left"].center' in sanitized


# ============================================================================
# Test 5: Component API Validation
# ============================================================================
def test_procedural_component_get_cell_tolerance():
    """Verify ProceduralComponent provides a safe get_cell method to avoid Manim getter crash."""
    from src.domains import ProceduralComponent
    from manim import Square, Circle, Text

    proc = ProceduralComponent(Square(), Circle(), Text("Item"))
    # Calling get_cell should not raise TypeError: getter() takes 1 positional argument
    cell = proc.get_cell(0, 0)
    assert cell is not None

    cell_by_name = proc.get_cell("A")
    assert cell_by_name is not None


def test_safe_graph_wrapper_weighted_edges():
    """Verify SafeGraphWrapper routes weighted 3-tuple edges to GraphVisual seamlessly."""
    from prompts.base_class import Graph, GraphVisual

    # 2-tuple edges -> native Graph
    g1 = Graph(["A", "B"], [("A", "B")])
    assert g1 is not None

    # 3-tuple weighted edges -> automatically routes to GraphVisual without crashing
    g2 = Graph(["A", "B", "C"], [("A", "B", 4), ("B", "C", 2)])
    assert isinstance(g2, GraphVisual)
    assert len(g2.vertices) == 3


def test_graph_pre_render_sanitizer():
    """Verify pre-render sanitizer rewrites Graph(...) to GraphVisual(...) when 3-tuples are present."""
    from src.agent import TeachingVideoAgent

    agent = TeachingVideoAgent.__new__(TeachingVideoAgent)

    code = '''
vertices = ['A', 'B', 'C']
edges = [('A', 'B', 4), ('B', 'C', 2)]
g = Graph(vertices, edges, layout="spring")
'''
    sanitized, modified = agent.sanitize_code_components_and_docking(code)
    assert modified is True
    assert "GraphVisual(vertices, edges" in sanitized


# ============================================================================
# Test 6: Standalone Subprocess Manim Render Validation
# ============================================================================
def test_standalone_manim_render_remediation(tmp_path):
    """Render a standalone scene exercising the remediated features through the actual Manim CLI."""
    from prompts import base_class
    from src.utils import replace_base_class

    scene_script = tmp_path / "test_remediation_scene.py"
    scene_code = r'''
class RemediationTestScene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Remediation Integration Test",
            [
                "Testing GraphVisual and weighted edges",
                "Testing DPTableMatrix and safe docking",
                "Testing SafeMathTex double brace tolerance",
            ],
            ["Graph", "Table", "Formula"]
        )

        # 1. Graph with 3-tuple weighted edges
        graph = GraphVisual(
            vertices=["A", "B", "C", "D"],
            edges=[("A", "B", 3), ("B", "C", 5), ("C", "D", 2)],
            layout={"A": [-1.5, 0.5, 0], "B": [0, 1.0, 0], "C": [1.5, 0.5, 0], "D": [0, -0.5, 0]}
        )
        self.dock_main(graph)
        self.play(FadeIn(graph), run_time=0.4)

        # 2. DPTableMatrix docked in inspector
        dp = DPTableMatrix(
            matrix_values=[["0", "3", "8", "10"]],
            col_labels=["A", "B", "C", "D"],
            row_labels=["Dist"]
        )
        self.dock_inspector(dp)
        self.play(FadeIn(dp), run_time=0.4)

        # Test dock_inspector.get_center() compatibility
        c_insp = self.dock_inspector.get_center()
        assert len(c_insp) == 3

        # 3. SafeMathTex with double braces
        formula = MathTex(r"f'(x) = \lim_{{h \to 0}} \frac{{f(x+h) - f(x)}}{{h}}")
        formula.scale_to_fit_width(4.0)
        self.play(Transform(dp, formula), run_time=0.4)
        self.wait(0.2)
'''
    full_code = replace_base_class(scene_code, base_class)
    scene_script.write_text(full_code, encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(_PROJECT_ROOT)
    cmd = [
        "manim",
        "-ql",
        "-s",
        str(scene_script.name),
        "RemediationTestScene",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path, env=env)
    assert res.returncode == 0, f"Manim render failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    print("STANDALONE_RENDER_SUCCESS")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
