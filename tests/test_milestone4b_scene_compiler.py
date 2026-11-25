"""
CineLearn — Milestone 4B: Deterministic Scene Compiler Tests
============================================================
Comprehensive test suite verifying deterministic compilation of validated
Scene IR documents and computed spatial geometry into safe, syntactically valid
Manim Python source code subclassing TeachingScene.

Covers:
1. Input and Contract Validation
2. Tier 3 Declarative Visual Primitives
3. Tier 4 Conceptual Cards
4. Determinism & Output Stability
5. Python Code Injection & Parameter Safety
6. Strict Geometry Consumption
7. Canonical Domain Fixtures & Integration
"""

import ast
import copy
import pytest

from src.compiler import (
    BoundingBox2D,
    CompilationDiagnostic,
    CompilationErrorCode,
    CompiledSceneOutput,
    ConceptualCardCompiler,
    DeterministicSceneCompiler,
    HierarchicalBoxModelLayoutSolver,
    PrimitiveCompiler,
    SceneSpatialLayout,
    sanitize_latex,
)
from src.ir.fixtures import (
    get_binary_search_fixture,
    get_chemical_kinetics_fixture,
    get_pythagorean_theorem_fixture,
)
from src.ir.models import (
    ActionDeclaration,
    ActionType,
    Alignment,
    EntityDeclaration,
    EntityStyling,
    FlowDirection,
    InheritedEntityReference,
    LayoutConfiguration,
    NarrationAnchor,
    SceneIR,
    TimelineEvent,
    ViewportDeclaration,
    ViewportType,
    VisualTier,
)


@pytest.fixture
def compiler() -> DeterministicSceneCompiler:
    return DeterministicSceneCompiler()


@pytest.fixture
def solver() -> HierarchicalBoxModelLayoutSolver:
    return HierarchicalBoxModelLayoutSolver()


@pytest.fixture
def minimal_scene() -> SceneIR:
    """Creates a minimal valid SceneIR for compiler unit tests."""
    return SceneIR(
        scene_id="MinimalTestScene",
        title="Minimal Test Scene",
        duration_budget=10.0,
        layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
        viewports={
            "vp_main": ViewportDeclaration(type=ViewportType.MAIN),
        },
        entities=[
            EntityDeclaration(
                id="box_1",
                component_type="PrimitiveShape",
                visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                viewport="vp_main",
                initial_state={"shape_type": "rectangle", "width": 4.0, "height": 2.0, "label": "Box 1"},
            )
        ],
        timeline=[
            TimelineEvent(
                step_id="step_1",
                narration_anchor=NarrationAnchor(
                    lecture_line_index=0,
                    text_cue="Showing box one.",
                    audio_timestamp_range=(0.0, 5.0),
                ),
                actions=[
                    ActionDeclaration(type=ActionType.ENTER, target_entity="box_1", duration=1.0),
                ],
            )
        ],
    )


# =====================================================================
# 1. Input and Contract Validation Tests
# =====================================================================
class TestInputValidation:
    def test_valid_scene_and_layout_compiles_successfully(self, compiler, solver, minimal_scene):
        layout = solver.solve(minimal_scene)
        out = compiler.compile(minimal_scene, layout)

        assert isinstance(out, CompiledSceneOutput)
        assert out.is_valid is True
        assert out.scene_id == "MinimalTestScene"
        assert not out.has_errors
        assert "class MinimalTestScene(TeachingScene):" in out.python_source
        assert "ent_box_1" in out.entity_variable_map["box_1"]

    def test_missing_geometry_emits_structured_diagnostic(self, compiler, solver, minimal_scene):
        layout = solver.solve(minimal_scene)
        # Deliberately remove geometry for box_1
        del layout.entities["box_1"]

        out = compiler.compile(minimal_scene, layout)
        assert out.is_valid is False
        assert out.has_errors is True

        diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_MISSING_GEOMETRY)
        assert len(diags) == 1
        assert diags[0].entity_id == "box_1"
        assert diags[0].severity == "ERROR"

    def test_unknown_entity_in_timeline_action_emits_diagnostic(self, compiler, solver, minimal_scene):
        # Add an action referencing an undeclared entity
        minimal_scene.timeline[0].actions.append(
            ActionDeclaration(type=ActionType.HIGHLIGHT, target_entity="ghost_entity", duration=0.5)
        )
        layout = solver.solve(minimal_scene)
        out = compiler.compile(minimal_scene, layout)

        assert out.is_valid is False
        diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_UNKNOWN_ENTITY)
        assert len(diags) == 1
        assert diags[0].entity_id == "ghost_entity"

    def test_unsupported_composite_group_id_rejected(self, compiler, solver, minimal_scene):
        # Inject group_id into entity initial_state
        minimal_scene.entities[0].initial_state["group_id"] = "group_alpha"
        layout = solver.solve(minimal_scene)
        out = compiler.compile(minimal_scene, layout)

        assert out.is_valid is False
        diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_UNSUPPORTED_FEATURE)
        assert len(diags) == 1
        assert "group_id" in diags[0].message


# =====================================================================
# 2. Tier 3 Declarative Visual Primitives Tests
# =====================================================================
class TestTier3PrimitiveCompilation:
    def test_primitive_shape_variants(self, compiler, solver):
        shapes = ["rectangle", "square", "circle", "rounded_rectangle", "ellipse"]
        entities = []
        for idx, s in enumerate(shapes):
            entities.append(
                EntityDeclaration(
                    id=f"shape_{s}",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": s, "width": 2.0, "height": 1.5, "label": f"{s.title()}"},
                    styling=EntityStyling(color="#FFD700", fill_opacity=0.5, stroke_width=3.0),
                )
            )

        scene = SceneIR(
            scene_id="ShapesScene",
            title="Geometric Shapes",
            duration_budget=10.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN, flow=FlowDirection.ROW)},
            entities=entities,
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        src = out.python_source

        # Verify each Manim primitive is emitted
        assert "Rectangle(" in src
        assert "Square(" in src
        assert "Circle(" in src
        assert "RoundedRectangle(" in src
        assert "Ellipse(" in src
        assert "stroke_width=3.0" in src
        assert "fill_opacity=0.5" in src

    def test_pointer_indicator_directions(self, compiler, solver):
        directions = ["up", "down", "left", "right"]
        entities = []
        for d in directions:
            entities.append(
                EntityDeclaration(
                    id=f"ptr_{d}",
                    component_type="PointerIndicator",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"direction": d, "label": f"Ptr {d}"},
                )
            )

        scene = SceneIR(
            scene_id="PointersScene",
            title="Pointers",
            duration_budget=8.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN, flow=FlowDirection.ROW)},
            entities=entities,
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        assert out.python_source.count("Arrow(") == 4

    def test_composite_node_graph_array_cells(self, compiler, solver):
        entity = EntityDeclaration(
            id="array_nodes",
            component_type="CompositeNodeGraph",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={
                "elements": [
                    {"index": 0, "val": "10", "state": "active"},
                    {"index": 1, "val": "20", "state": "midpoint"},
                    {"index": 2, "val": "30", "state": "inactive"},
                ]
            },
        )
        scene = SceneIR(
            scene_id="ArrayScene",
            title="Array Elements",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        src = out.python_source
        assert "_cells_ent_array_nodes" in src
        assert "Array Cell 0: 10 (active)" in src
        assert "Array Cell 1: 20 (midpoint)" in src
        assert "Array Cell 2: 30 (inactive)" in src

    def test_composite_dissection_square(self, compiler, solver):
        entity = EntityDeclaration(
            id="diss_sq",
            component_type="CompositeDissectionSquare",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={"side_length": 4.0, "labels": ["a^2", "ab", "ab", "b^2"]},
        )
        scene = SceneIR(
            scene_id="SquareScene",
            title="Dissection Square",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        assert "Composite Dissection Square:" in out.python_source
        assert "_outer_ent_diss_sq = Square(" in out.python_source

    def test_boundary_container(self, compiler, solver):
        entity = EntityDeclaration(
            id="main_boundary",
            component_type="container",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={"title": "Execution Boundary"},
        )
        scene = SceneIR(
            scene_id="ContainerScene",
            title="Container",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        assert "_boundary_ent_main_boundary" in out.python_source
        assert "Execution Boundary" in out.python_source


# =====================================================================
# 3. Tier 4 Conceptual Cards Tests
# =====================================================================
class TestTier4ConceptualCardsCompilation:
    def test_formula_card_variant(self, compiler, solver):
        card = EntityDeclaration(
            id="formula_1",
            component_type="ConceptualCard",
            visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
            viewport="vp_main",
            initial_state={
                "title": "Kinetic Energy",
                "formula": "E_k = \\frac{1}{2} m v^2",
                "badge": "PHYSICS",
            },
        )
        scene = SceneIR(
            scene_id="FormulaScene",
            title="Formula",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[card],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        src = out.python_source
        assert "MathTex(" in src
        assert "Kinetic Energy" in src
        assert "PHYSICS" in src

    def test_definition_card_variant(self, compiler, solver):
        card = EntityDeclaration(
            id="def_card",
            component_type="ConceptualCard",
            visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
            viewport="vp_main",
            initial_state={
                "title": "Critical Section",
                "body": "A segment of code where shared resources are accessed and mutual exclusion is required.",
            },
        )
        scene = SceneIR(
            scene_id="DefinitionScene",
            title="Definition",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[card],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        src = out.python_source
        assert "Critical Section" in src
        assert "mutual exclusion" in src

    def test_boundary_invariant_card_variant(self, compiler, solver):
        card = EntityDeclaration(
            id="invariant_card",
            component_type="ConceptualCard",
            visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
            viewport="vp_main",
            initial_state={
                "title": "Loop Invariant",
                "formula": "low \\le target \\le high",
                "badge": "INVARIANT",
                "subtext": "Holds at the start of every iteration",
            },
        )
        scene = SceneIR(
            scene_id="InvariantScene",
            title="Invariant",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[card],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        src = out.python_source
        assert "INVARIANT" in src
        assert "Loop Invariant" in src
        assert "Holds at the start of every iteration" in src

    def test_latex_sanitizer_removes_double_braces(self):
        raw_tex = r"{{x}} + {{y}} = {{z}}"
        sanitized = sanitize_latex(raw_tex)
        assert sanitized == r"{x} + {y} = {z}"

    def test_content_overflow_diagnostic_emitted(self, compiler, solver):
        # Create a card with massive text crammed into a small dimension
        card = EntityDeclaration(
            id="overflow_card",
            component_type="ConceptualCard",
            visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
            viewport="vp_main",
            initial_state={
                "title": "This is an extraordinarily long title that cannot possibly fit comfortably inside a tiny card",
                "formula": "\\sum_{i=1}^{\\infty} \\frac{1}{n^s} = \\prod_{p \\text{ prime}} \\frac{1}{1 - p^{-s}} = \\zeta(s)",
            },
        )
        scene = SceneIR(
            scene_id="OverflowScene",
            title="Overflow",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[card],
            timeline=[],
        )
        layout = solver.solve(scene)
        # Artificially constrain geometry to small box
        layout.entities["overflow_card"].world_box = BoundingBox2D(x=0.0, y=0.0, width=1.5, height=1.0)

        out = compiler.compile(scene, layout)
        diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_CONTENT_OVERFLOW)
        assert len(diags) >= 1
        assert diags[0].severity == "WARNING"
        assert "exceeds card bounds" in diags[0].message


# =====================================================================
# 4. Determinism & Stability Tests
# =====================================================================
class TestDeterminismAndStability:
    def test_bit_identical_repeatability(self, compiler, solver):
        scene = get_binary_search_fixture()
        layout = solver.solve(scene)

        out1 = compiler.compile(scene, layout)
        out2 = compiler.compile(scene, layout)

        assert out1.python_source == out2.python_source
        assert out1.entity_variable_map == out2.entity_variable_map
        assert len(out1.diagnostics) == len(out2.diagnostics)

    def test_entity_variable_name_stability(self, compiler):
        # Different naming cases
        v1 = compiler._generate_var_name("search_array")
        v2 = compiler._generate_var_name("123_invalid_prefix")
        v3 = compiler._generate_var_name("entity-with-hyphens")

        assert v1 == "ent_search_array"
        assert v2.isidentifier()
        assert v3.isidentifier()
        assert "-" not in v3


# =====================================================================
# 5. Safety & Injection Prevention Tests
# =====================================================================
class TestSafetyAndCodeInjection:
    def test_code_injection_in_title_and_label_prevented(self, compiler, solver):
        malicious_string = 'Box 1"); import os; os.system("echo hacked"); ("'
        entity = EntityDeclaration(
            id="exploit_test",
            component_type="PrimitiveShape",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={"shape_type": "rectangle", "label": malicious_string},
        )
        scene = SceneIR(
            scene_id="ExploitScene",
            title=malicious_string,
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        # AST parse must succeed cleanly without executing injection
        parsed = ast.parse(out.python_source)
        assert parsed is not None

        # Verify that os.system is NOT an AST call node
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "system":
                    pytest.fail("Malicious os.system was parsed as an executable function call!")

    def test_code_injection_in_latex_formula_prevented(self, compiler, solver):
        malicious_latex = r'k = \frac{a}{b} \'); import sys; sys.exit(1); (\''
        card = EntityDeclaration(
            id="latex_exploit",
            component_type="ConceptualCard",
            visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
            viewport="vp_main",
            initial_state={"formula": malicious_latex},
        )
        scene = SceneIR(
            scene_id="LatexExploitScene",
            title="Latex Safe",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[card],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        parsed = ast.parse(out.python_source)
        assert parsed is not None


# =====================================================================
# 6. Geometry Consumption Tests
# =====================================================================
class TestGeometryConsumption:
    def test_exact_world_box_used_for_position_and_scale(self, compiler, solver):
        entity = EntityDeclaration(
            id="geo_ent",
            component_type="PrimitiveShape",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={"shape_type": "rectangle", "width": 4.5, "height": 2.5},
        )
        scene = SceneIR(
            scene_id="GeoScene",
            title="Geometry Check",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[],
        )
        layout = solver.solve(scene)
        geom = layout.entities["geo_ent"]

        out = compiler.compile(scene, layout)
        src = out.python_source

        # Verify exact world_box coordinates appear in generated code
        cx = round(geom.world_box.x, 4)
        cy = round(geom.world_box.y, 4)
        w = round(geom.world_box.width, 4)
        h = round(geom.world_box.height, 4)
        expected_pos = f"np.array([{cx}, {cy}, 0.0])"
        assert expected_pos in src
        assert f"width={w}" in src
        assert f"height={h}" in src
        assert f"set_z_index({geom.z_index})" in src


# =====================================================================
# 7. Canonical Domain Fixtures Compilation Tests
# =====================================================================
class TestCanonicalDomainFixtures:
    def test_canonical_binary_search_compilation(self, compiler, solver):
        scene = get_binary_search_fixture()
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        assert out.error_count == 0
        assert out.entity_variable_map.get("search_array") == "ent_search_array"
        assert out.entity_variable_map.get("pointer_low") == "ent_pointer_low"
        assert out.entity_variable_map.get("pointer_high") == "ent_pointer_high"
        assert out.entity_variable_map.get("formula_card") == "ent_formula_card"

        # Verify timeline step plays
        assert "self.play(FadeIn(ent_search_array)" in out.python_source
        assert "self.play(FadeIn(ent_formula_card)" in out.python_source

    def test_canonical_chemical_kinetics_compilation(self, compiler, solver):
        scene = get_chemical_kinetics_fixture()
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        assert out.error_count == 0
        assert out.entity_variable_map.get("energy_diagram") == "ent_energy_diagram"
        assert out.entity_variable_map.get("arrhenius_equation") == "ent_arrhenius_equation"
        assert "ReactionEnergyDiagram(" in out.python_source

    def test_canonical_pythagorean_theorem_compilation(self, compiler, solver):
        scene = get_pythagorean_theorem_fixture()
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        assert out.error_count == 0
        assert out.entity_variable_map.get("square_a_plus_b_left") == "ent_square_a_plus_b_left"
        assert out.entity_variable_map.get("square_a_plus_b_right") == "ent_square_a_plus_b_right"
        assert "Composite Dissection Square:" in out.python_source


# =====================================================================
# 8. Milestone 4B Remediation A: Tier 1 Component Security Tests
# =====================================================================
class TestRemediationTier1ComponentSecurity:
    def test_malicious_component_type_injection_rejected(self, compiler, solver):
        exploit_ctype = "int; import os; os.system('echo PWNED'); #"
        entity = EntityDeclaration(
            id="exp1",
            component_type=exploit_ctype,
            visual_tier=VisualTier.TIER1_REGISTERED,
            viewport="vp_main",
            initial_state={},
        )
        scene = SceneIR(
            scene_id="ExploitScene",
            title="Exploit Test",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is False
        diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_UNSUPPORTED_COMPONENT)
        assert len(diags) == 1
        assert diags[0].severity == "ERROR"
        assert "not registered in VisualRegistry" in diags[0].message

        # Verify static AST inspection: no malicious function calls exist
        parsed = ast.parse(out.python_source)
        for node in ast.walk(parsed):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr == "system":
                    pytest.fail("Malicious os.system was parsed in AST!")

    def test_unregistered_arbitrary_identifier_rejected(self, compiler, solver):
        # Even if ctype is a valid identifier like 'eval' or 'MyUnregisteredClass', it must be rejected
        for bad_ctype in ["eval", "exec", "os", "sys", "ArbitraryCustomClass"]:
            entity = EntityDeclaration(
                id="unreg_ent",
                component_type=bad_ctype,
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="vp_main",
                initial_state={},
            )
            scene = SceneIR(
                scene_id="UnregScene",
                title="Unregistered",
                duration_budget=5.0,
                viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
                entities=[entity],
                timeline=[],
            )
            layout = solver.solve(scene)
            out = compiler.compile(scene, layout)

            assert out.is_valid is False
            diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_UNSUPPORTED_COMPONENT)
            assert len(diags) == 1

    def test_valid_registered_component_compiles_with_trusted_symbol(self, compiler, solver):
        entity = EntityDeclaration(
            id="valid_diag",
            component_type="ReactionEnergyDiagram",
            visual_tier=VisualTier.TIER1_REGISTERED,
            viewport="vp_main",
            initial_state={"reactant_energy": 1.5, "product_energy": 0.5},
        )
        scene = SceneIR(
            scene_id="ValidScene",
            title="Valid Diagram",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        assert out.error_count == 0
        assert "ReactionEnergyDiagram(" in out.python_source

    def test_registered_component_alias_resolves_to_canonical_symbol(self, compiler, solver):
        entity = EntityDeclaration(
            id="alias_diag",
            component_type="reaction energy diagram",  # Alias in lowercase
            visual_tier=VisualTier.TIER1_REGISTERED,
            viewport="vp_main",
            initial_state={},
        )
        scene = SceneIR(
            scene_id="AliasScene",
            title="Alias Diagram",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        # Must resolve to canonical Python class name ReactionEnergyDiagram
        assert "ReactionEnergyDiagram(" in out.python_source


# =====================================================================
# 9. Milestone 4B Remediation B: Component Parameter Key Validation Tests
# =====================================================================
class TestRemediationParameterKeyValidation:
    def test_malicious_parameter_key_in_initial_state_rejected(self, compiler, solver):
        exploit_key = "x=1); import os; os.system('echo HACK'); ("
        entity = EntityDeclaration(
            id="bad_key_ent",
            component_type="ReactionEnergyDiagram",
            visual_tier=VisualTier.TIER1_REGISTERED,
            viewport="vp_main",
            initial_state={exploit_key: "value"},
        )
        scene = SceneIR(
            scene_id="BadKeyScene",
            title="Bad Key",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is False
        diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_INVALID_PARAMETER)
        assert len(diags) >= 1
        assert "is invalid" in diags[0].message
        assert exploit_key not in out.python_source

    def test_reserved_keywords_as_parameter_keys_rejected(self, compiler, solver):
        for reserved in ["class", "def", "import", "return", "for", "while", "lambda", "pass"]:
            entity = EntityDeclaration(
                id=f"res_{reserved}",
                component_type="ReactionEnergyDiagram",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="vp_main",
                initial_state={reserved: 42},
            )
            scene = SceneIR(
                scene_id="ReservedScene",
                title="Reserved",
                duration_budget=5.0,
                viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
                entities=[entity],
                timeline=[],
            )
            layout = solver.solve(scene)
            out = compiler.compile(scene, layout)

            assert out.is_valid is False
            diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_INVALID_PARAMETER)
            assert len(diags) >= 1

    def test_malicious_parameter_key_in_component_op_rejected(self, compiler, solver):
        entity = EntityDeclaration(
            id="diag_op",
            component_type="ReactionEnergyDiagram",
            visual_tier=VisualTier.TIER1_REGISTERED,
            viewport="vp_main",
            initial_state={},
        )
        scene = SceneIR(
            scene_id="OpKeyScene",
            title="Op Key",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.COMPONENT_OP,
                            target_entity="diag_op",
                            operation="show_barrier",
                            parameters={"__import__('os').system('hack')": 1},
                        )
                    ],
                )
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is False
        diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_INVALID_PARAMETER)
        assert len(diags) >= 1

    def test_malicious_operation_name_rejected(self, compiler, solver):
        entity = EntityDeclaration(
            id="diag_op2",
            component_type="ReactionEnergyDiagram",
            visual_tier=VisualTier.TIER1_REGISTERED,
            viewport="vp_main",
            initial_state={},
        )
        scene = SceneIR(
            scene_id="OpNameScene",
            title="Op Name",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[entity],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.COMPONENT_OP,
                            target_entity="diag_op2",
                            operation="op(); import os; os.system('hack'); #",
                            parameters={},
                        )
                    ],
                )
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is False
        diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_INVALID_PARAMETER)
        assert len(diags) >= 1


# =====================================================================
# 10. Milestone 4B Remediation C: Collision-Free Variable Naming Tests
# =====================================================================
class TestRemediationCollisionFreeVariableNaming:
    def test_normalization_collision_box_hyphen_and_underscore(self, compiler, solver):
        # Declare two entities that normalize to the same string: 'box-1' and 'box_1'
        e1 = EntityDeclaration(
            id="box-1",
            component_type="PrimitiveShape",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0},
        )
        e2 = EntityDeclaration(
            id="box_1",
            component_type="PrimitiveShape",
            visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
            viewport="vp_main",
            initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0},
        )
        scene = SceneIR(
            scene_id="CollisionScene",
            title="Collision Test",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN, flow=FlowDirection.ROW)},
            entities=[e1, e2],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[
                        ActionDeclaration(type=ActionType.ENTER, target_entity="box-1", duration=0.5),
                        ActionDeclaration(type=ActionType.HIGHLIGHT, target_entity="box_1", duration=0.5),
                    ],
                )
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        v1 = out.entity_variable_map["box-1"]
        v2 = out.entity_variable_map["box_1"]

        # Variables must be distinct identifiers
        assert v1 != v2
        assert v1 == "ent_box_1"
        assert v2 == "ent_box_1_2"

        # Verify timeline actions target the correct distinct variables
        assert f"FadeIn({v1})" in out.python_source
        assert f"Indicate({v2}" in out.python_source

    def test_multiple_identical_normalized_ids(self, compiler, solver):
        # 4 entities with different hyphens/underscores that all normalize to 'ent_item_a'
        ids = ["item-a", "item_a", "item--a", "item__a"]
        entities = [
            EntityDeclaration(
                id=eid,
                component_type="PrimitiveShape",
                visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                viewport="vp_main",
                initial_state={"shape_type": "circle", "radius": 0.5},
            )
            for eid in ids
        ]
        scene = SceneIR(
            scene_id="MultiCollisionScene",
            title="Multi Collision",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN, flow=FlowDirection.ROW)},
            entities=entities,
            timeline=[],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is True
        allocated_vars = list(out.entity_variable_map.values())
        assert len(allocated_vars) == 4
        assert len(set(allocated_vars)) == 4  # All 4 must be completely unique
        assert out.entity_variable_map["item-a"] == "ent_item_a"
        assert out.entity_variable_map["item_a"] == "ent_item_a_2"
        assert out.entity_variable_map["item--a"] == "ent_item_a_3"
        assert out.entity_variable_map["item__a"] == "ent_item_a_4"

    def test_repeated_compilation_produces_bit_identical_mappings(self, compiler, solver):
        ids = ["box-1", "box_1", "class", "123"]
        entities = [
            EntityDeclaration(
                id=eid,
                component_type="PrimitiveShape",
                visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                viewport="vp_main",
                initial_state={"shape_type": "rectangle", "width": 1.0, "height": 1.0},
            )
            for eid in ids
        ]
        scene = SceneIR(
            scene_id="RepeatScene",
            title="Repeat Test",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN, flow=FlowDirection.ROW)},
            entities=entities,
            timeline=[],
        )
        layout = solver.solve(scene)

        out1 = compiler.compile(scene, layout)
        out2 = compiler.compile(scene, layout)

        assert out1.python_source == out2.python_source
        assert out1.entity_variable_map == out2.entity_variable_map


# =====================================================================
# 11. Milestone 4B Remediation D: Inherited Entity Reference Safety Tests
# =====================================================================
class TestRemediationInheritedEntityReferenceSafety:
    def test_action_targeting_inherited_entity_rejected(self, compiler, solver):
        # Entity 'inherited_counter' is in inherited_entities, NOT in scene.entities
        scene = SceneIR(
            scene_id="InheritedActionScene",
            title="Inherited Test",
            duration_budget=5.0,
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            inherited_entities=[
                InheritedEntityReference(source_section="sec_1", entity_id="inherited_counter")
            ],
            entities=[],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    actions=[
                        ActionDeclaration(type=ActionType.HIGHLIGHT, target_entity="inherited_counter", duration=0.5)
                    ],
                )
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.is_valid is False
        diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION)
        assert len(diags) == 1
        assert "inherited entity 'inherited_counter'" in diags[0].message
        assert diags[0].severity == "ERROR"

        # Verify that construct() does NOT contain an undeclared 'ent_inherited_counter'
        assert "ent_inherited_counter" not in out.python_source

    def test_scene_with_unreferenced_inherited_entities_compiles_successfully(self, compiler, solver, minimal_scene):
        # A scene containing inherited entities that are NOT referenced in timeline actions is valid
        minimal_scene.inherited_entities.append(
            InheritedEntityReference(source_section="sec_1", entity_id="prior_table")
        )
        layout = solver.solve(minimal_scene)
        out = compiler.compile(minimal_scene, layout)

        assert out.is_valid is True
        assert out.error_count == 0
        assert "ent_box_1" in out.entity_variable_map["box_1"]

    def test_action_targeting_unknown_non_inherited_entity_rejected(self, compiler, solver, minimal_scene):
        minimal_scene.timeline[0].actions.append(
            ActionDeclaration(type=ActionType.ENTER, target_entity="totally_random_id", duration=0.5)
        )
        layout = solver.solve(minimal_scene)
        out = compiler.compile(minimal_scene, layout)

        assert out.is_valid is False
        diags = out.get_diagnostics_by_code(CompilationErrorCode.COMPILER_UNKNOWN_ENTITY)
        assert len(diags) == 1
        assert diags[0].entity_id == "totally_random_id"

