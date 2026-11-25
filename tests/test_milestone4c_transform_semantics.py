"""
CineLearn — Milestone 4C.2: Visual Transform Semantics Test Suite
=================================================================
Exhaustive verification of deterministic, semantically correct visual
transformations for ActionType.TRANSFORM across:
1. Token matching vs replacement logic (LaTeX tokenization, matching vs replacement).
2. Conceptual card element updates (title, body, formula, badge, subtext, color).
3. Identity and geometry preservation.
4. AST safety, sanitization, and code validity (injection defense, ast.parse).
5. Error handling and diagnostics (fail-closed behavior, structured diagnostics).
6. Determinism and output stability.
7. Architectural regression preservation.
"""

import ast
import copy
import pytest
from typing import Dict, Any

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
    is_safe_keyword_identifier,
    should_use_matching_tex,
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
    LayoutConfiguration,
    NarrationAnchor,
    SceneIR,
    TimelineEvent,
    ViewportDeclaration,
    ViewportType,
    VisualTier,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def compiler() -> DeterministicSceneCompiler:
    return DeterministicSceneCompiler()


@pytest.fixture
def solver() -> HierarchicalBoxModelLayoutSolver:
    return HierarchicalBoxModelLayoutSolver()


@pytest.fixture
def sample_card_scene() -> SceneIR:
    """Creates a valid SceneIR containing a ConceptualCard entity."""
    return SceneIR(
        scene_id="TransformCardScene",
        title="Transform Card Scene",
        duration_budget=12.0,
        layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
        viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
        entities=[
            EntityDeclaration(
                id="theory_card",
                component_type="ConceptualCard",
                visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                viewport="vp_main",
                initial_state={
                    "card_type": "formula",
                    "title": "Energy-Mass Equivalence",
                    "formula": r"E = mc^2",
                    "badge": "LAW",
                    "subtext": "Einstein (1905)",
                },
            )
        ],
        timeline=[
            TimelineEvent(
                step_id="step_1",
                narration_anchor=NarrationAnchor(
                    lecture_line_index=0,
                    text_cue="Introducing mass-energy equivalence.",
                    audio_timestamp_range=(0.0, 4.0),
                ),
                actions=[
                    ActionDeclaration(type=ActionType.ENTER, target_entity="theory_card", duration=1.0),
                ],
            ),
            TimelineEvent(
                step_id="step_2",
                narration_anchor=NarrationAnchor(
                    lecture_line_index=1,
                    text_cue="Transforming equation to solve for mass.",
                    audio_timestamp_range=(4.0, 8.0),
                ),
                actions=[
                    ActionDeclaration(
                        type=ActionType.TRANSFORM,
                        target_entity="theory_card",
                        duration=1.5,
                        parameters={"formula": r"m = \frac{E}{c^2}"},
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def sample_body_card_scene() -> SceneIR:
    """Creates a valid SceneIR containing a body-text ConceptualCard entity."""
    return SceneIR(
        scene_id="TransformBodyCardScene",
        title="Transform Body Card Scene",
        duration_budget=12.0,
        layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
        viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
        entities=[
            EntityDeclaration(
                id="concept_card",
                component_type="ConceptualCard",
                visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                viewport="vp_main",
                initial_state={
                    "card_type": "concept",
                    "title": "Initial Definition",
                    "body": "A process is an instance of a program in execution.",
                    "badge": "CONCEPT",
                },
            )
        ],
        timeline=[
            TimelineEvent(
                step_id="step_1",
                narration_anchor=NarrationAnchor(
                    lecture_line_index=0,
                    text_cue="Showing concept card.",
                    audio_timestamp_range=(0.0, 4.0),
                ),
                actions=[
                    ActionDeclaration(type=ActionType.ENTER, target_entity="concept_card", duration=1.0),
                ],
            ),
            TimelineEvent(
                step_id="step_2",
                narration_anchor=NarrationAnchor(
                    lecture_line_index=1,
                    text_cue="Updating concept definition.",
                    audio_timestamp_range=(4.0, 8.0),
                ),
                actions=[
                    ActionDeclaration(
                        type=ActionType.TRANSFORM,
                        target_entity="concept_card",
                        duration=1.2,
                        parameters={
                            "title": "Refined Definition",
                            "body": "A process has its own address space and PCB.",
                            "badge": "INVARIANT",
                        },
                    ),
                ],
            ),
        ],
    )


# =====================================================================
# 1. LaTeX Token Matching vs Replacement Logic
# =====================================================================

class TestLaTeXTokenMatchingLogic:
    def test_shared_tokens_yields_matching_tex(self):
        """Formulas with overlapping mathematical tokens select TransformMatchingTex."""
        src = r"E = mc^2"
        tgt = r"m = \frac{E}{c^2}"
        assert should_use_matching_tex(src, tgt) is True

    def test_algebraic_derivation_yields_matching_tex(self):
        """Algebraic step with common variables and exponents matches."""
        src = r"x^2 + 2x + 1 = 0"
        tgt = r"(x + 1)^2 = 0"
        assert should_use_matching_tex(src, tgt) is True

    def test_disjoint_tokens_yields_replacement_transform(self):
        """Completely disjoint mathematical expressions fall back to ReplacementTransform."""
        src = r"A = \pi r^2"
        tgt = r"F = m a"
        assert should_use_matching_tex(src, tgt) is False

    def test_explicit_replacement_mode_override(self):
        """Setting mode='replacement' forces ReplacementTransform even if tokens overlap."""
        src = r"E = mc^2"
        tgt = r"m = \frac{E}{c^2}"
        assert should_use_matching_tex(src, tgt, mode="replacement") is False

    def test_explicit_matching_mode_with_shared_tokens(self):
        """Setting mode='matching' with shared tokens returns True."""
        src = r"a^2 + b^2 = c^2"
        tgt = r"c = \sqrt{a^2 + b^2}"
        assert should_use_matching_tex(src, tgt, mode="matching") is True

    def test_none_or_empty_formula_gracefully_falls_back(self):
        """None, empty, or whitespace formulas return False without exception."""
        assert should_use_matching_tex(None, r"E = mc^2") is False
        assert should_use_matching_tex(r"E = mc^2", None) is False
        assert should_use_matching_tex("", "") is False
        assert should_use_matching_tex("   ", "   ") is False

    def test_malformed_unclosed_latex_does_not_crash(self):
        """Malformed LaTeX expressions tokenize without raising exceptions."""
        malformed_src = r"\frac{a}{b"
        malformed_tgt = r"\sqrt[3]{x"
        result = should_use_matching_tex(malformed_src, malformed_tgt)
        assert isinstance(result, bool)

    def test_complex_latex_commands(self):
        """Expressions with LaTeX macros tokenize correctly."""
        src = r"\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}"
        tgt = r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}"
        assert should_use_matching_tex(src, tgt) is True

    # -----------------------------------------------------------------
    # M4C.7 Remediation: FINDING-4C2-01 (LaTeX Mode Matching Precedence)
    # -----------------------------------------------------------------

    def test_precedence_empty_or_none_src_with_explicit_matching_mode(self):
        """FINDING-4C2-01: Explicit mode='matching_tex' must return False when src_formula is None/empty."""
        assert should_use_matching_tex(None, r"E = mc^2", mode="matching_tex") is False
        assert should_use_matching_tex("", r"E = mc^2", mode="matching_tex") is False
        assert should_use_matching_tex("   ", r"E = mc^2", mode="matching_tex") is False
        assert should_use_matching_tex(None, r"E = mc^2", mode="matching") is False

    def test_precedence_empty_or_none_tgt_with_explicit_matching_mode(self):
        """FINDING-4C2-01: Explicit mode='matching_tex' must return False when tgt_formula is None/empty."""
        assert should_use_matching_tex(r"E = mc^2", None, mode="matching_tex") is False
        assert should_use_matching_tex(r"E = mc^2", "", mode="matching_tex") is False
        assert should_use_matching_tex(r"E = mc^2", "   ", mode="matching_tex") is False
        assert should_use_matching_tex(r"E = mc^2", None, mode="matching") is False

    def test_precedence_explicit_matching_mode_with_valid_inputs(self):
        """Explicit mode='matching_tex' or 'matching' forces True when both formulas are present."""
        assert should_use_matching_tex(r"a + b", r"c + d", mode="matching_tex") is True
        assert should_use_matching_tex(r"a + b", r"c + d", mode="matching") is True

    # -----------------------------------------------------------------
    # M4C.7 Remediation: FINDING-4C2-03 (False-Positive Single-Token Matching)
    # -----------------------------------------------------------------

    def test_single_digit_overlap_rejects_matching_tex(self):
        """FINDING-4C2-03: Solitary digit overlap must not trigger TransformMatchingTex."""
        assert should_use_matching_tex(r"x = 1", r"y = 1") is False
        assert should_use_matching_tex(r"f(x) = 0", r"g(y) = 0") is False
        assert should_use_matching_tex(r"a = 2", r"b = 2") is False

    def test_single_variable_overlap_rejects_matching_tex(self):
        """FINDING-4C2-03: Solitary single variable without structural overlap must not match."""
        assert should_use_matching_tex(r"x = 1", r"x = 2") is False
        assert should_use_matching_tex(r"f(x) = 1", r"g(x) = 2") is False
        assert should_use_matching_tex(r"a = 3", r"b = a") is False

    def test_disjoint_variables_with_only_number_overlap_rejects_matching_tex(self):
        """Formulas with disjoint variables sharing only numeric literals must not match."""
        assert should_use_matching_tex(r"x = 10", r"y = 10") is False
        assert should_use_matching_tex(r"a + 1 = 2", r"b + 1 = 2") is False

    def test_single_macro_token_allows_matching_tex(self):
        """Solitary LaTeX macro represents structural/operator overlap and enables matching."""
        assert should_use_matching_tex(r"\sqrt{a}", r"\sqrt{b}") is True
        assert should_use_matching_tex(r"\frac{a}{b}", r"\frac{c}{d}") is True
        assert should_use_matching_tex(r"\alpha = 1", r"\alpha = 2") is True
        assert should_use_matching_tex(r"\int f", r"\int g") is True

    def test_identical_expressions_match(self):
        """Identical expressions always match."""
        assert should_use_matching_tex(r"E = mc^2", r"E = mc^2") is True
        assert should_use_matching_tex(r"x + 1", r"x + 1") is True


# =====================================================================
# 2. Conceptual Card Element Updates
# =====================================================================

class TestConceptualCardTransformUpdates:
    def test_card_formula_matching_transform_compilation(self, compiler, solver, sample_card_scene):
        """Verify that transforming a card formula produces TransformMatchingTex and valid AST."""
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert not out.has_errors, [d.message for d in out.diagnostics]
        code = out.python_source

        # AST must be valid
        tree = ast.parse(code)
        assert tree is not None

        # Must NOT contain Indicate placeholder
        assert "Indicate" not in code

        # Must emit TransformMatchingTex
        assert "TransformMatchingTex" in code
        assert "_new_formula_ent_theory_card" in code
        assert "ent_theory_card.remove(" in code
        assert "ent_theory_card.remove(_formula_ent_theory_card).add(_new_formula_ent_theory_card" in code

    def test_card_formula_replacement_transform_compilation(self, compiler, solver, sample_card_scene):
        """Disjoint formula transform produces ReplacementTransform."""
        sample_card_scene.timeline[1].actions[0].parameters = {"formula": r"\lambda + \theta = 1"}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert "ReplacementTransform(_formula_ent_theory_card, _new_formula_ent_theory_card" in code
        assert "TransformMatchingTex" not in code
        assert ast.parse(code) is not None

    def test_card_formula_explicit_replacement_mode(self, compiler, solver, sample_card_scene):
        """Explicit transform_mode='replacement' forces ReplacementTransform even with shared tokens."""
        sample_card_scene.timeline[1].actions[0].parameters = {
            "formula": r"m = \frac{E}{c^2}",
            "transform_mode": "replacement",
        }
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert "ReplacementTransform(_formula_ent_theory_card, _new_formula_ent_theory_card" in code
        assert "TransformMatchingTex" not in code
        assert ast.parse(code) is not None

    def test_card_title_isolated_update(self, compiler, solver, sample_card_scene):
        """Transforming only the title updates title and preserves formula."""
        sample_card_scene.timeline[1].actions[0].parameters = {"title": "Updated Theory Title"}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert "ReplacementTransform(_title_ent_theory_card, _new_title_ent_theory_card" in code
        assert "Updated Theory Title" in code
        assert "ent_theory_card.remove(_title_ent_theory_card).add(_new_title_ent_theory_card" in code
        # Formula should not have been transformed
        assert "_new_formula_ent_theory_card" not in code
        assert ast.parse(code) is not None

    def test_card_body_isolated_update(self, compiler, solver, sample_body_card_scene):
        """Transforming body text updates body submobject and preserves title."""
        sample_body_card_scene.timeline[1].actions[0].parameters = {"body": "New definition of process."}
        layout = solver.solve(sample_body_card_scene)
        out = compiler.compile(sample_body_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert "ReplacementTransform(_body_ent_concept_card, _new_body_ent_concept_card" in code
        assert "New definition of process." in code
        assert "ent_concept_card.remove(_body_ent_concept_card).add(_new_body_ent_concept_card" in code
        assert ast.parse(code) is not None

    def test_card_badge_isolated_update(self, compiler, solver, sample_card_scene):
        """Transforming badge updates badge text and rounded rectangle."""
        sample_card_scene.timeline[1].actions[0].parameters = {"badge": "THEOREM"}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert "ReplacementTransform(_badge_grp_ent_theory_card, _new_badge_grp_ent_theory_card" in code
        assert "'THEOREM'" in code
        assert "ent_theory_card.remove(_badge_grp_ent_theory_card).add(_new_badge_grp_ent_theory_card" in code
        assert ast.parse(code) is not None

    def test_card_subtext_isolated_update(self, compiler, solver, sample_card_scene):
        """Transforming subtext updates subtext element."""
        sample_card_scene.timeline[1].actions[0].parameters = {"subtext": "Updated Citation (2026)"}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert "ReplacementTransform(_subtext_ent_theory_card, _new_subtext_ent_theory_card" in code
        assert "Updated Citation (2026)" in code
        assert ast.parse(code) is not None

    def test_card_stroke_color_update(self, compiler, solver, sample_card_scene):
        """Transforming card color emits animate.set_stroke on background rectangle."""
        sample_card_scene.timeline[1].actions[0].parameters = {"color": "#FF8800"}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert "_bg_ent_theory_card.animate.set_stroke(color='#FF8800')" in code
        assert ast.parse(code) is not None

    def test_card_multi_field_simultaneous_update(self, compiler, solver, sample_body_card_scene):
        """Updating title, body, and badge simultaneously combines into a single self.play call."""
        layout = solver.solve(sample_body_card_scene)
        out = compiler.compile(sample_body_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        # All 3 transforms should be in the self.play call
        assert "self.play(" in code
        assert "ReplacementTransform(_badge_grp_ent_concept_card" in code
        assert "ReplacementTransform(_title_ent_concept_card" in code
        assert "ReplacementTransform(_body_ent_concept_card" in code
        assert ast.parse(code) is not None

    def test_repeated_successive_transforms(self, compiler, solver, sample_card_scene):
        """Repeated transforms across 3 steps maintain state tracking and valid AST."""
        # Add a 3rd step transforming title
        sample_card_scene.timeline.append(
            TimelineEvent(
                step_id="step_3",
                narration_anchor=NarrationAnchor(
                    lecture_line_index=2,
                    text_cue="Finalizing theorem name.",
                    audio_timestamp_range=(8.0, 11.0),
                ),
                actions=[
                    ActionDeclaration(
                        type=ActionType.TRANSFORM,
                        target_entity="theory_card",
                        duration=1.0,
                        parameters={"title": "Relativistic Mass Formula"},
                    ),
                ],
            )
        )
        sample_card_scene.duration_budget = 14.0
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        # Step 2: formula transform
        assert "_new_formula_ent_theory_card_1_0" in code
        # Step 3: title transform
        assert "_new_title_ent_theory_card_2_0" in code
        assert ast.parse(code) is not None

    def test_body_to_formula_variant_transition(self, compiler, solver, sample_body_card_scene):
        """Transforming a body card with a new formula replaces _body with _formula."""
        sample_body_card_scene.timeline[1].actions[0].parameters = {"formula": r"\sigma = \frac{F}{A}"}
        layout = solver.solve(sample_body_card_scene)
        out = compiler.compile(sample_body_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert "ReplacementTransform(_body_ent_concept_card, _new_formula_ent_concept_card" in code
        assert "ent_concept_card.remove(_body_ent_concept_card).add(_new_formula_ent_concept_card" in code
        assert ast.parse(code) is not None


# =====================================================================
# 3. Structural Entity Replacement & Primitive Shapes
# =====================================================================

class TestStructuralEntityAndPrimitiveTransform:
    def test_entity_to_entity_replacement_same_tier(self, compiler, solver):
        """Transforming entity A into replacement_entity B of the same tier emits ReplacementTransform."""
        scene = SceneIR(
            scene_id="EntityReplScene",
            title="Entity Replacement Scene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="shape_a",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0},
                ),
                EntityDeclaration(
                    id="shape_b",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "circle", "radius": 1.0},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 4.0)),
                    actions=[
                        ActionDeclaration(type=ActionType.ENTER, target_entity="shape_a", duration=1.0),
                    ],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(4.0, 8.0)),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="shape_a",
                            duration=1.5,
                            parameters={"replacement_entity": "shape_b"},
                        ),
                    ],
                ),
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors
        code = out.python_source
        assert "ReplacementTransform(ent_shape_a, ent_shape_b)" in code
        assert ast.parse(code) is not None

    def test_entity_to_entity_replacement_tier_mismatch_fallback_crossfade(self, compiler, solver):
        """Cross-tier replacement emits a safe cross-fade (FadeOut + FadeIn) and a WARNING diagnostic."""
        scene = SceneIR(
            scene_id="CrossTierScene",
            title="Cross Tier Scene",
            duration_budget=10.0,
            layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="shape_a",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0},
                ),
                EntityDeclaration(
                    id="card_b",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                    initial_state={"card_type": "concept", "title": "Card B", "body": "Card B text"},
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 4.0)),
                    actions=[
                        ActionDeclaration(type=ActionType.ENTER, target_entity="shape_a", duration=1.0),
                    ],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(4.0, 8.0)),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="shape_a",
                            duration=2.0,
                            parameters={"replacement_entity": "card_b"},
                        ),
                    ],
                ),
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors
        # Must have generated a WARNING diagnostic
        warnings = [d for d in out.diagnostics if d.severity == "WARNING"]
        assert len(warnings) >= 1
        assert "Falling back to safe cross-fade sequence" in warnings[0].message

        code = out.python_source
        assert "FadeOut(ent_shape_a, run_time=" in code
        assert "FadeIn(ent_card_b, run_time=" in code
        assert ast.parse(code) is not None

    def test_primitive_shape_color_mutation(self, compiler, solver):
        """Transform on PrimitiveShape with color emits animate.set_color."""
        scene = SceneIR(
            scene_id="PrimColorScene",
            title="Primitive Color Scene",
            duration_budget=8.0,
            layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="rect_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0, "fill_color": "#333333"},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 3.0)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="rect_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(3.0, 6.0)),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="rect_1",
                            duration=1.0,
                            parameters={"color": "#FF0000"},
                        )
                    ],
                ),
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors
        assert "rect_1.animate.set_color('#FF0000')" in out.python_source
        assert ast.parse(out.python_source) is not None

    def test_primitive_shape_scale_mutation(self, compiler, solver):
        """Transform on PrimitiveShape with scale emits animate.scale."""
        scene = SceneIR(
            scene_id="PrimScaleScene",
            title="Primitive Scale Scene",
            duration_budget=8.0,
            layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="circle_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "circle", "radius": 1.0},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 3.0)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="circle_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(3.0, 6.0)),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="circle_1",
                            duration=1.0,
                            parameters={"scale": 1.5},
                        )
                    ],
                ),
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors
        assert "circle_1.animate.scale(1.5)" in out.python_source
        assert ast.parse(out.python_source) is not None

    def test_primitive_shape_opacity_mutation(self, compiler, solver):
        """Transform on PrimitiveShape with opacity emits animate.set_opacity."""
        scene = SceneIR(
            scene_id="PrimOpacityScene",
            title="Primitive Opacity Scene",
            duration_budget=8.0,
            layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="rect_op",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 3.0)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="rect_op", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(3.0, 6.0)),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="rect_op",
                            duration=1.0,
                            parameters={"opacity": 0.4},
                        )
                    ],
                ),
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert not out.has_errors
        assert "rect_op.animate.set_opacity(0.4)" in out.python_source
        assert ast.parse(out.python_source) is not None


# =====================================================================
# 4. AST Safety, Code Validity, and Injection Defense
# =====================================================================

class TestASTSafetyAndInjectionDefense:
    def test_is_safe_keyword_identifier(self):
        """Verify keyword identifier validator rejects dangerous/invalid names."""
        assert is_safe_keyword_identifier("formula") is True
        assert is_safe_keyword_identifier("target_formula") is True
        assert is_safe_keyword_identifier("var_1") is True
        assert is_safe_keyword_identifier("__import__") is False
        assert is_safe_keyword_identifier("import os") is False
        assert is_safe_keyword_identifier("foo.bar") is False
        assert is_safe_keyword_identifier("a-b") is False
        assert is_safe_keyword_identifier("123abc") is False
        assert is_safe_keyword_identifier("") is False
        assert is_safe_keyword_identifier("class") is False
        assert is_safe_keyword_identifier("def") is False

    def test_code_injection_in_card_title_sanitized(self, compiler, solver, sample_card_scene):
        """Malicious code injection attempt in card title is safely escaped via repr()."""
        injection = '"); import os; os.system("echo hacked"); ("'
        sample_card_scene.timeline[1].actions[0].parameters = {"title": injection}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        # Code must parse into valid AST without syntax error
        tree = ast.parse(code)
        assert tree is not None
        # Verify no rogue import statement exists inside construct() method
        construct_func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "construct")
        for node in ast.walk(construct_func):
            assert not isinstance(node, (ast.Import, ast.ImportFrom))

    def test_code_injection_in_card_formula_sanitized(self, compiler, solver, sample_card_scene):
        """Malicious code in formula string is safely sanitized and escaped."""
        injection = r"E = mc^2 \"); __import__('os').system('rm -rf /'); (\""
        sample_card_scene.timeline[1].actions[0].parameters = {"formula": injection}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert not out.has_errors
        code = out.python_source
        tree = ast.parse(code)
        assert tree is not None
        construct_func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "construct")
        for node in ast.walk(construct_func):
            assert not isinstance(node, (ast.Import, ast.ImportFrom))

    def test_invalid_parameter_key_rejected_fail_closed(self, compiler, solver, sample_card_scene):
        """Malicious or invalid parameter key causes fail-closed compilation diagnostic."""
        bad_key = "__import__('os').system('echo hacked')"
        sample_card_scene.timeline[1].actions[0].parameters = {bad_key: "value"}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1
        assert "invalid parameter key" in diags[0].message

    def test_non_scalar_parameter_value_rejected(self, compiler, solver, sample_card_scene):
        """Passing non-scalar dictionary or list as formula/title produces diagnostic."""
        sample_card_scene.timeline[1].actions[0].parameters = {"title": {"nested": "payload"}}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1
        assert "non-scalar value" in diags[0].message


# =====================================================================
# 5. Error Handling and Diagnostics
# =====================================================================

class TestErrorHandlingAndDiagnostics:
    def test_target_entity_not_found(self, compiler, solver, sample_card_scene):
        """TRANSFORM action referencing nonexistent entity produces COMPILER_UNKNOWN_ENTITY."""
        sample_card_scene.timeline[1].actions[0].target_entity = "ghost_entity"
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_UNKNOWN_ENTITY]
        assert len(diags) >= 1
        assert "ghost_entity" in diags[0].message

    def test_replacement_entity_not_found(self, compiler, solver, sample_card_scene):
        """replacement_entity referencing nonexistent entity produces COMPILER_UNKNOWN_ENTITY."""
        sample_card_scene.timeline[1].actions[0].parameters = {"replacement_entity": "nonexistent_target"}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_UNKNOWN_ENTITY]
        assert len(diags) >= 1
        assert "nonexistent_target" in diags[0].message

    def test_card_transform_with_empty_parameters(self, compiler, solver, sample_card_scene):
        """Card transform with empty parameters produces COMPILER_INVALID_PARAMETER."""
        sample_card_scene.timeline[1].actions[0].parameters = {}
        layout = solver.solve(sample_card_scene)
        out = compiler.compile(sample_card_scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1
        assert "no recognized transformation parameters" in diags[0].message

    def test_primitive_shape_invalid_scale(self, compiler, solver):
        """Non-numeric scale parameter produces COMPILER_INVALID_PARAMETER."""
        scene = SceneIR(
            scene_id="BadScaleScene",
            title="Bad Scale Scene",
            duration_budget=8.0,
            layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="rect_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 3.0)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="rect_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(3.0, 6.0)),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="rect_1",
                            duration=1.0,
                            parameters={"scale": "not_a_number"},
                        )
                    ],
                ),
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_INVALID_PARAMETER]
        assert len(diags) >= 1
        assert "non-numeric scale" in diags[0].message

    def test_primitive_shape_unsupported_operation(self, compiler, solver):
        """Unsupported parameter on PrimitiveShape produces COMPILER_UNSUPPORTED_OPERATION."""
        scene = SceneIR(
            scene_id="BadParamScene",
            title="Bad Param Scene",
            duration_budget=8.0,
            layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="rect_1",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                    initial_state={"shape_type": "rectangle", "width": 2.0, "height": 1.0},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 3.0)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="rect_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(3.0, 6.0)),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="rect_1",
                            duration=1.0,
                            parameters={"unsupported_field": "xyz"},
                        )
                    ],
                ),
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION]
        assert len(diags) >= 1
        assert "unsupported parameters" in diags[0].message

    def test_tier1_registered_component_without_replacement_fails(self, compiler, solver):
        """ActionType.TRANSFORM on Tier 1 registered component without replacement produces diagnostic."""
        scene = SceneIR(
            scene_id="Tier1TransformScene",
            title="Tier 1 Transform Scene",
            duration_budget=8.0,
            layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
            viewports={"vp_main": ViewportDeclaration(type=ViewportType.MAIN)},
            entities=[
                EntityDeclaration(
                    id="gantt_1",
                    component_type="GanttChart",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="vp_main",
                    initial_state={},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_1",
                    narration_anchor=NarrationAnchor(lecture_line_index=0, audio_timestamp_range=(0.0, 3.0)),
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="gantt_1", duration=1.0)],
                ),
                TimelineEvent(
                    step_id="step_2",
                    narration_anchor=NarrationAnchor(lecture_line_index=1, audio_timestamp_range=(3.0, 6.0)),
                    actions=[
                        ActionDeclaration(
                            type=ActionType.TRANSFORM,
                            target_entity="gantt_1",
                            duration=1.0,
                            parameters={"some_param": 123},
                        )
                    ],
                ),
            ],
        )
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)

        assert out.has_errors
        diags = [d for d in out.diagnostics if d.error_code == CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION]
        assert len(diags) >= 1
        assert "Tier 1 registered component" in diags[0].message


# =====================================================================
# 6. Determinism & Output Stability
# =====================================================================

class TestDeterminismAndOutputStability:
    def test_bit_identical_compilation(self, compiler, solver, sample_card_scene):
        """Repeated compilation of identical SceneIR yields bit-identical Python source."""
        layout1 = solver.solve(sample_card_scene)
        out1 = compiler.compile(sample_card_scene, layout1)

        layout2 = solver.solve(sample_card_scene)
        out2 = compiler.compile(sample_card_scene, layout2)

        assert out1.python_source == out2.python_source
        assert len(out1.diagnostics) == len(out2.diagnostics)


# =====================================================================
# 7. Canonical Domain Fixtures Regression Preservation
# =====================================================================

class TestCanonicalDomainFixturesPreservation:
    def test_binary_search_fixture_compilation(self, compiler, solver):
        """Binary search fixture compiles cleanly without error."""
        scene = get_binary_search_fixture()
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert not out.has_errors, [d.message for d in out.diagnostics]
        assert ast.parse(out.python_source) is not None

    def test_chemical_kinetics_fixture_compilation(self, compiler, solver):
        """Chemical kinetics fixture compiles cleanly without error."""
        scene = get_chemical_kinetics_fixture()
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert not out.has_errors, [d.message for d in out.diagnostics]
        assert ast.parse(out.python_source) is not None

    def test_pythagorean_theorem_fixture_compilation(self, compiler, solver):
        """Pythagorean theorem fixture compiles cleanly without error."""
        scene = get_pythagorean_theorem_fixture()
        layout = solver.solve(scene)
        out = compiler.compile(scene, layout)
        assert not out.has_errors, [d.message for d in out.diagnostics]
        assert ast.parse(out.python_source) is not None
