"""
Unit & Integration Tests for CineLearn Phase 3: Storyboard Planner & Prompt Refactor
===================================================================================
Validates domain-independent visual reasoning, schema manifest injection,
container docking instructions, 4-tier fallback hierarchy, and prompt generation
across three representative academic topics: Dijkstra, Dynamic Programming, and Chemistry.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import re
import pytest

from prompts.stage2 import get_prompt2_storyboard
from prompts.stage3 import get_prompt3_code
from prompts.base_class import base_class
from src.agent import Section
from src.domains import VisualRegistry, VisualComponent


class TestPhase3PromptArchitecture:
    """Test suite for Phase 3 prompt refactoring and manifest injection."""

    def test_stage2_prompt_injects_empty_manifest_fallback(self):
        """When registry has no domain components, prompt explains 4-tier fallback."""
        VisualRegistry.clear()
        mock_outline = {
            "topic": "General Topic",
            "sections": [{"id": "section_1", "title": "Sec 1: Overview", "visual_idea": "Diagram"}]
        }
        prompt = get_prompt2_storyboard(mock_outline, level="intermediate")

        assert "Available Visual Architecture & Registry Components" in prompt
        assert "4-Tier Fallback Hierarchy" in prompt
        assert "Generic Container Layout & Viewport Contract" in prompt
        assert "dock_main" in prompt
        assert "dock_inspector" in prompt
        assert "dock_split_left" in prompt
        assert "Domain-Independent Visual Reasoning" in prompt

    def test_stage2_prompt_injects_registered_component_manifest(self):
        """When components are registered, manifest schema is dynamically injected into prompt."""
        VisualRegistry.clear()

        @VisualRegistry.register("cs", "MockGraphVisual")
        class MockGraphVisual(VisualComponent):
            """Directed or undirected graph component."""
            preferred_aspect = 1.6

            def __init__(self, node_count: int = 5, **kwargs):
                super().__init__(**kwargs)

            def animate_highlight_node(self, node_id):
                pass

        mock_outline = {"topic": "Graph Theory", "sections": []}
        prompt = get_prompt2_storyboard(mock_outline, level="intermediate")

        assert "MockGraphVisual" in prompt
        assert "Domain: `cs`" in prompt
        assert "animate_highlight_node" in prompt
        assert "node_count" in prompt

        VisualRegistry.clear()
        VisualRegistry.discover_domains("src.domains")

    def test_dijkstra_topic_storyboard_and_code_prompts(self):
        """Validates prompt generation and sample storyboard for Dijkstra Algorithm (CS/Graph)."""
        dijkstra_outline = {
            "topic": "Dijkstra's Algorithm",
            "formal_definition": "Finds shortest paths from a single source vertex to all other vertices in a weighted graph with non-negative edge weights.",
            "sections": [
                {
                    "id": "section_1",
                    "title": "Sec 1: Graph Representation & Problem Setup",
                    "visual_idea": "Weighted graph in dock_main with tentative distance table in dock_inspector."
                },
                {
                    "id": "section_2",
                    "title": "Sec 2: Greedy Node Selection & Edge Relaxation",
                    "visual_idea": "Highlight minimum distance node in dock_main and update tentative table row in dock_inspector."
                }
            ]
        }
        prompt_sb = get_prompt2_storyboard(dijkstra_outline, level="intermediate", analogy="Travelling")
        assert "Dijkstra" in prompt_sb
        assert "dock_main" in prompt_sb
        assert "dock_inspector" in prompt_sb
        assert "anti-text-card" in prompt_sb.lower() or "do not use bare text cards" in prompt_sb.lower()

        # Representative structured storyboard section for Dijkstra
        sec_dijkstra = Section(
            id="section_1",
            title="Sec 1: Graph Representation & Problem Setup",
            lecture_lines=[
                "Consider a road network where cities are vertices and road travel times are non-negative edge weights.",
                "To find the shortest route, we initialize all tentative distances from the source to infinity, except the source itself.",
            ],
            captions=["Road network setup", "Initialize distances"],
            animations=[
                "L1-1 | FadeIn | ProceduralComponent(WeightedGraph) | viewport: dock_main | action: render_network() | on screen after: [graph]",
                "L2-1 | FadeIn | Table(DistanceTable) | viewport: dock_inspector | action: init_tentative_costs() | on screen after: [graph, distance table]",
            ],
            story_beat="Trainer models cities as graph vertices with distance table."
        )

        prompt_code = get_prompt3_code(
            regenerate_note="",
            section=sec_dijkstra,
            base_class=base_class,
            level="intermediate",
            analogy="Travelling"
        )
        assert "dock_main" in prompt_code
        assert "dock_inspector" in prompt_code
        assert "ProceduralComponent" in prompt_code

    def test_dynamic_programming_topic_storyboard_and_code_prompts(self):
        """Validates prompt generation and sample storyboard for Dynamic Programming (CS/Matrix/Recurrence)."""
        dp_outline = {
            "topic": "Dynamic Programming",
            "formal_definition": "Solves complex optimization problems by breaking them into overlapping subproblems with optimal substructure.",
            "sections": [
                {
                    "id": "section_1",
                    "title": "Sec 1: Overlapping Subproblems and Memoization",
                    "visual_idea": "Subproblem recursion DAG in dock_main and 2D memoization table in dock_inspector."
                }
            ]
        }
        prompt_sb = get_prompt2_storyboard(dp_outline, level="intermediate")
        assert "Dynamic Programming" in prompt_sb
        assert "dock_main" in prompt_sb
        assert "dock_inspector" in prompt_sb

        sec_dp = Section(
            id="section_1",
            title="Sec 1: Overlapping Subproblems and Memoization",
            lecture_lines=[
                "Naive recursive evaluation recomputes the same overlapping subproblems exponentially many times.",
                "Dynamic programming eliminates redundant work by memoizing computed states into a structured table.",
            ],
            captions=["Overlapping subproblems", "Memoization table"],
            animations=[
                "L1-1 | FadeIn | ProceduralComponent(RecursionDAG) | viewport: dock_main | action: show_redundant_branches() | on screen after: [dag]",
                "L2-1 | FadeIn | Table(DPTableMatrix) | viewport: dock_inspector | action: cache_solution() | on screen after: [dag, dp_table]",
            ],
            story_beat="Memoizing subproblem states into a 2D matrix."
        )

        prompt_code = get_prompt3_code(
            regenerate_note="",
            section=sec_dp,
            base_class=base_class,
            level="intermediate"
        )
        assert "dock_main" in prompt_code
        assert "dock_inspector" in prompt_code
        assert "create_table_matrix" in prompt_code

    def test_chemistry_topic_storyboard_and_code_prompts(self):
        """Validates prompt generation and sample storyboard for Chemistry (Reaction Kinetics / Non-CS)."""
        chem_outline = {
            "topic": "First-Order Chemical Kinetics",
            "formal_definition": "A reaction where the rate depends linearly on the concentration of a single reactant: Rate = -d[A]/dt = k[A].",
            "sections": [
                {
                    "id": "section_1",
                    "title": "Sec 1: Rate Law and Molecular Decay",
                    "visual_idea": "Molecular reactant flow in dock_main with governing rate equation in dock_inspector."
                },
                {
                    "id": "section_2",
                    "title": "Sec 2: Concentration Exponential Decay Curve",
                    "visual_idea": "Concentration decay plot curve in dock_main with half-life values in dock_inspector."
                }
            ]
        }
        prompt_sb = get_prompt2_storyboard(chem_outline, level="intermediate")
        assert "Chemical & Molecular" in prompt_sb
        assert "dock_main" in prompt_sb
        assert "dock_inspector" in prompt_sb

        sec_chem = Section(
            id="section_1",
            title="Sec 1: Rate Law and Molecular Decay",
            lecture_lines=[
                "In a first-order chemical reaction, reactant molecules decompose at a rate directly proportional to their instantaneous concentration.",
                "Integrating this differential relationship yields the characteristic exponential concentration decay over time.",
            ],
            captions=["First-order reaction rate", "Exponential decay law"],
            animations=[
                "L1-1 | FadeIn | ProceduralComponent(MolecularDecomposition) | viewport: dock_main | action: show_reactant_molecules() | on screen after: [molecules]",
                "L2-1 | FadeIn | ConceptualCard(IntegratedRateLaw) | viewport: dock_inspector | action: display_equation() | on screen after: [molecules, equation card]",
            ],
            story_beat="Molecular decay and integrated rate law."
        )

        prompt_code = get_prompt3_code(
            regenerate_note="",
            section=sec_chem,
            base_class=base_class,
            level="intermediate"
        )
        assert "dock_main" in prompt_code
        assert "dock_inspector" in prompt_code
        assert "MathTex" in prompt_code

    def test_stage2_prompt_mathtex_strict_latex_formatting(self):
        """P1: Verifies Stage 2 prompt strictly mandates raw LaTeX strings for MathTex and bans comma-tokenization."""
        mock_outline = {
            "topic": "Arrhenius Equation",
            "sections": [{"id": "section_1", "title": "Sec 1: Overview", "visual_idea": "Formula"}]
        }
        prompt = get_prompt2_storyboard(mock_outline, level="intermediate")

        # Must mandate raw LaTeX string literal
        assert "STRICT MATHTEX RULE: MathTex MUST ALWAYS contain a valid raw LaTeX string literal" in prompt
        assert 'MathTex(r"k = A e^{-\\frac{E_a}{RT}}")' in prompt

        # Must explicitly forbid comma-separated tokens
        assert "NEVER split mathematical notation into comma-separated English tokens" in prompt
        assert "NEVER write MathTex(k, equals, A, times, exp" in prompt

        # Conceptually separate Python variables and LaTeX expressions
        assert "keep Python variables and LaTeX strings conceptually distinct" in prompt or \
               "keep Python variables and LaTeX expressions conceptually distinct" in prompt

        # Mathematical preservation of outline values
        assert "Preservation of Outline Mathematical Values" in prompt
        assert "Take every definition, formula, intermediate calculation, and numeric value from the outline" in prompt

    def test_stage2_prompt_anti_text_card_and_meaningful_visuals(self):
        """P2: Verifies anti-text-card rules discourage text dumping while allowing concise labels and formulas."""
        mock_outline = {
            "topic": "Dijkstra's Algorithm",
            "sections": [{"id": "section_1", "title": "Sec 1: Graph", "visual_idea": "Graph visual"}]
        }
        prompt = get_prompt2_storyboard(mock_outline, level="intermediate")

        # Explicitly forbids replacing processes with collections of explanatory Text objects
        assert "Do NOT replace algorithmic, mathematical, or scientific processes with collections of explanatory Text objects" in prompt
        assert "NEVER emit Text(Select Smallest Distance)" in prompt

        # Permitted text usage (no blanket ban)
        assert "Text is permitted ONLY for concise labels" in prompt
        assert "Do NOT place numerous explanatory text objects in dock_main" in prompt
        assert "Make the visual structure (graphs, plots, matrices, geometric shapes) carry the explanation" in prompt

    def test_stage2_prompt_transition_syntax_and_extra_pipe_regression(self):
        """P3: Verifies exact 6-field transition grammar and includes regression test for malformed extra-pipe case."""
        mock_outline = {
            "topic": "Derivatives",
            "sections": [{"id": "section_1", "title": "Sec 1: Tangent", "visual_idea": "Tangent slope"}]
        }
        prompt = get_prompt2_storyboard(mock_outline, level="beginner")

        # Prompt instructions
        assert "EXACT 6-FIELD DELIMITER GRAMMAR" in prompt
        assert "EXACTLY 5 pipe '|' delimiters separating 6 fields" in prompt
        assert "FadeOut <old units>; then <effect>" in prompt
        assert "NEVER insert extra pipe '|' characters inside the transition or effect field" in prompt
        assert "FadeOut | old_unit | then FadeIn" in prompt  # Mentioned as INCORRECT

        # Downstream 6-field parser grammar simulation
        def parse_animation_step(step_str: str):
            parts = [p.strip() for p in step_str.split("|")]
            if len(parts) != 6:
                raise ValueError(f"Expected exactly 6 fields (5 '|' delimiters), got {len(parts)} fields: {parts}")
            tag, effect, visual_rep, viewport_str, action_str, on_screen_str = parts
            assert re.match(r"^L\d+-\d+$", tag), f"Invalid tag format: {tag}"
            assert viewport_str.startswith("viewport:"), f"Invalid viewport field: {viewport_str}"
            assert action_str.startswith("action:"), f"Invalid action field: {action_str}"
            assert on_screen_str.startswith("on screen after:"), f"Invalid on-screen field: {on_screen_str}"
            return {
                "tag": tag,
                "effect": effect,
                "visual_representation": visual_rep,
                "viewport": viewport_str,
                "action": action_str,
                "on_screen_after": on_screen_str,
            }

        # Valid step with transition syntax in Field 2 (effect)
        valid_step = (
            "L4-1 | FadeOut Definition; then FadeIn | Text(Velocity) | "
            "viewport: dock_inspector | action: show_text() | on screen after: [Car Position Function, Tangent Line, Velocity]"
        )
        parsed = parse_animation_step(valid_step)
        assert parsed["tag"] == "L4-1"
        assert parsed["effect"] == "FadeOut Definition; then FadeIn"
        assert parsed["visual_representation"] == "Text(Velocity)"

        # Malformed regression step from Scenario D (extra pipe delimiters inside transition)
        malformed_scenario_d_step = (
            "L4-1 | FadeOut | Definition | then FadeIn | Text(Velocity) | "
            "viewport: dock_inspector | action: show_text() | on screen after: [Car Position Function, Tangent Line, Velocity]"
        )
        with pytest.raises(ValueError, match="Expected exactly 6 fields"):
            parse_animation_step(malformed_scenario_d_step)

    def test_stage2_prompt_meaningful_visual_progression(self):
        """P4: Verifies prompt enforces distinct visual development and forbids repeating identical setups."""
        mock_outline = {
            "topic": "Graph Algorithms",
            "sections": [{"id": "section_1", "title": "Sec 1", "visual_idea": "Graph"}]
        }
        prompt = get_prompt2_storyboard(mock_outline, level="intermediate")

        assert "Meaningful Visual Progression Across Sections (CRITICAL)" in prompt
        assert "A distinct teaching objective and a clear visual purpose" in prompt
        assert "the state within that structure MUST EVOLVE" in prompt
        assert "Repeating identical initialization and highlighting sequences across consecutive sections is strictly prohibited" in prompt

    def test_stage2_prompt_formal_first_and_analogy_routing_preserved(self):
        """P5: Verifies formal-first ordering and analogy routing remain completely intact."""
        mock_outline = {"topic": "Dijkstra", "sections": []}

        # Direct mode (analogy=None)
        prompt_direct = get_prompt2_storyboard(mock_outline, level="intermediate", analogy=None)
        assert "Direct Technical Narration & Mental Model Guidance" in prompt_direct
        assert "Formal-First Order" in prompt_direct
        assert "Grounding via AI Mental Model" in prompt_direct
        assert "Do NOT automatically turn a mental model into fictional role-playing" in prompt_direct

        # Analogy mode (analogy="road navigation")
        prompt_analogy = get_prompt2_storyboard(mock_outline, level="intermediate", analogy="road navigation")
        assert "Story-Driven Storyboard & Narration Requirements" in prompt_analogy
        assert "Requested analogy / personal interest: road navigation" in prompt_analogy
        assert "Order inside each section (Formal-First)" in prompt_analogy
        assert "Continuous Narrative Progression Across ALL Sections" in prompt_analogy
        assert "Do not confuse it with an independently selected AI mental model" in prompt_analogy
