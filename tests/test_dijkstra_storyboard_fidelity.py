"""
Deterministic tests for Dijkstra Stage 2 Integration & Storyboard Fidelity Checker (Milestone 2B.6B).

Validates:
1. Canonical verified trace is passed into Stage 2 prompt construction.
2. Variable number of vertices and extraction steps.
3. Correct extraction order.
4. Negative test: Missing extraction.
5. Negative test: Incorrect extraction order.
6. Negative test: Missing relaxation.
7. Negative test: Incorrect updated distance (Pilot regression C = 4).
8. Negative test: Incorrect unsuccessful-relaxation representation.
9. Negative test: Correct final distances but incomplete intermediate trace.
10. Negative test: Narration-only description without corresponding visual transition.
11. Negative test: Graph/source mismatch.
12. Positive test: Valid complete Dijkstra storyboard passes with VERIFIED.
13. Regression test: Non-Dijkstra storyboard continues using scalar numerical validation.
14. Pipeline test: Retry exhaustion blocks audio pre-generation and Stage 3 Manim code generation.
15. Pipeline test: Checkpoint saving and restoration preserves verified algorithmic state.
"""

import copy
import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from agent import TeachingVideoAgent, TeachingOutline, RunConfig, Section
from numerical_evaluator import VerificationStatus
from prompts.stage2 import (
    get_prompt2_storyboard,
    get_prompt2_storyboard_fidelity_repair,
    extract_verified_algorithmic_state,
)
from storyboard_fidelity_checker import (
    StoryboardFidelityChecker,
    StoryboardFidelityReport,
    FidelityCategory,
    FidelityDiscrepancy,
)
from dijkstra_verifier import DijkstraTraceVerifier
from worked_example_verifier import WorkedExampleVerifier


# ============================================================================
# Test Fixtures: Pilot Graph (5 Vertices) and Small Graph (3 Vertices)
# ============================================================================

PILOT_SPEC = {
    "algorithm": "dijkstra",
    "topic": "Dijkstra's Shortest Path Algorithm",
    "inputs": {
        "vertices": ["A", "B", "C", "D", "E"],
        "source": "A",
        "edges": [
            {"u": "A", "v": "B", "weight": 1.0},
            {"u": "A", "v": "C", "weight": 4.0},
            {"u": "B", "v": "C", "weight": 2.0},
            {"u": "B", "v": "D", "weight": 5.0},
            {"u": "C", "v": "D", "weight": 1.0},
            {"u": "D", "v": "E", "weight": 3.0},
        ],
        "directed": False,
    },
    "steps": [
        {"step": "Initialize dist[A]=0, others=∞", "result": "dist = {A: 0, B: ∞, C: ∞, D: ∞, E: ∞}"},
        {"step": "Extract A, relax neighbors B and C", "result": "dist = {A: 0, B: 1, C: 4, D: ∞, E: ∞}"},
        {"step": "Extract B, relax neighbors C and D", "result": "dist = {A: 0, B: 1, C: 3, D: 6, E: ∞}"},
        {"step": "Extract C, relax neighbor D", "result": "dist = {A: 0, B: 1, C: 3, D: 4, E: ∞}"},
        {"step": "Extract D, relax neighbor E", "result": "dist = {A: 0, B: 1, C: 3, D: 4, E: 7}"},
        {"step": "Extract E, complete trace", "result": "dist = {A: 0, B: 1, C: 3, D: 4, E: 7}"},
    ],
    "final_result": "dist = {A: 0, B: 1, C: 3, D: 4, E: 7}",
}

SMALL_SPEC = {
    "algorithm": "dijkstra",
    "topic": "Dijkstra's Algorithm on 3 Vertices",
    "inputs": {
        "vertices": ["X", "Y", "Z"],
        "source": "X",
        "edges": [
            {"u": "X", "v": "Y", "weight": 2.0},
            {"u": "X", "v": "Z", "weight": 7.0},
            {"u": "Y", "v": "Z", "weight": 3.0},
        ],
        "directed": False,
    },
    "steps": [
        {"step": "Init dist[X]=0, Y=∞, Z=∞", "result": "dist = {X: 0, Y: ∞, Z: ∞}"},
        {"step": "Extract X, relax Y and Z", "result": "dist = {X: 0, Y: 2, Z: 7}"},
        {"step": "Extract Y, relax Z", "result": "dist = {X: 0, Y: 2, Z: 5}"},
        {"step": "Extract Z, done", "result": "dist = {X: 0, Y: 2, Z: 5}"},
    ],
    "final_result": "dist = {X: 0, Y: 2, Z: 5}",
}


def make_verified_pilot_outline():
    """Generates an outline with verified_algorithmic_state attached for the pilot graph."""
    updated_spec, rep = DijkstraTraceVerifier.verify_worked_example_spec(PILOT_SPEC, topic="Dijkstra's Shortest Path Algorithm")
    return {
        "topic": "Dijkstra's Shortest Path Algorithm",
        "learning_level": "intermediate",
        "sections": [
            {"id": "section_1", "title": "Sec 1: Problem Definition and Setup"},
            {"id": "section_2", "title": "Sec 2: Initial Extractions and Relaxations"},
            {"id": "section_3", "title": "Sec 3: Completing Shortest Paths"},
        ],
        "worked_example_spec": updated_spec,
        "verification_metadata": rep.to_dict(),
    }


def make_valid_pilot_storyboard():
    """Constructs a fully valid, step-by-step storyboard demonstrating canonical Dijkstra execution."""
    return {
        "visual_conventions": {
            "palette": {"primary": "#58C4DD", "highlight": "#FFD700", "positive": "#83C167"},
        },
        "sections": [
            {
                "id": "section_1",
                "title": "Sec 1: Graph Setup & First Extraction",
                "lecture_lines": [
                    "We consider a graph with five vertices and start at source node A.",
                    "We extract node A with minimum distance 0 and relax its neighbors B and C.",
                ],
                "captions": ["Graph setup", "Extract A and relax"],
                "animations": [
                    "L1-1 | FadeIn | GraphVisual(vertices=['A', 'B', 'C', 'D', 'E'], source='A') | viewport: dock_main | action: initialize_graph(vertices=['A', 'B', 'C', 'D', 'E'], source='A') | on screen after: [graph]",
                    "L1-2 | FadeIn | Table(DistanceTable) | viewport: dock_inspector | action: init_table() | on screen after: [graph, table]",
                    "L2-1 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('A', 0) | on screen after: [graph, table]",
                    "L2-2 | Write | MathTex(r\"dist(B) = \\min(\\infty, 0 + 1) = 1\") | viewport: dock_inspector | action: relax_edge(u='A', v='B', candidate=1, updated=True, new_dist=1) | on screen after: [graph, table, formula]",
                    "L2-3 | Write | MathTex(r\"dist(C) = \\min(\\infty, 0 + 4) = 4\") | viewport: dock_inspector | action: relax_edge(u='A', v='C', candidate=4, updated=True, new_dist=4) | on screen after: [graph, table, formula]",
                ],
                "verified_values": ["extract_min(A) = 0", "dist(B) = 1", "dist(C) = 4"],
            },
            {
                "id": "section_2",
                "title": "Sec 2: Extracting Nodes B and C",
                "lecture_lines": [
                    "Next, node B has the smallest tentative distance of 1 and is extracted.",
                    "Relaxing edge B to C improves distance from 4 down to 3, and edge B to D gives 6.",
                    "Then node C is extracted with distance 3, improving D from 6 down to 4.",
                ],
                "captions": ["Extract B", "Relax B's edges", "Extract C and relax D"],
                "animations": [
                    "L1-1 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('B', 1) | on screen after: [graph, table]",
                    "L2-1 | Write | MathTex(r\"dist(C) = \\min(4, 1 + 2) = 3\") | viewport: dock_inspector | action: relax_edge(u='B', v='C', candidate=3, updated=True, new_dist=3) | on screen after: [graph, table]",
                    "L2-2 | Write | MathTex(r\"dist(D) = \\min(\\infty, 1 + 5) = 6\") | viewport: dock_inspector | action: relax_edge(u='B', v='D', candidate=6, updated=True, new_dist=6) | on screen after: [graph, table]",
                    "L3-1 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('C', 3) | on screen after: [graph, table]",
                    "L3-2 | Write | MathTex(r\"dist(D) = \\min(6, 3 + 1) = 4\") | viewport: dock_inspector | action: relax_edge(u='C', v='D', candidate=4, updated=True, new_dist=4) | on screen after: [graph, table]",
                ],
                "verified_values": ["extract_min(B) = 1", "dist(C) = 3", "dist(D) = 6", "extract_min(C) = 3", "dist(D) = 4"],
            },
            {
                "id": "section_3",
                "title": "Sec 3: Extracting D, E and Final Paths",
                "lecture_lines": [
                    "Node D is extracted with distance 4 and relaxes edge to E, setting distance 7.",
                    "Finally, node E is extracted with distance 7, completing the shortest path tree.",
                ],
                "captions": ["Extract D and relax E", "Extract E and finalize"],
                "animations": [
                    "L1-1 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('D', 4) | on screen after: [graph, table]",
                    "L1-2 | Write | MathTex(r\"dist(E) = \\min(\\infty, 4 + 3) = 7\") | viewport: dock_inspector | action: relax_edge(u='D', v='E', candidate=7, updated=True, new_dist=7) | on screen after: [graph, table]",
                    "L2-1 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('E', 7) | on screen after: [graph, table]",
                    "L2-2 | Write | Table(FinalDistances) | viewport: dock_inspector | action: show_final_table() | on screen after: [graph, final_table]",
                ],
                "verified_values": [
                    "extract_min(D) = 4", "dist(E) = 7", "extract_min(E) = 7",
                    "dist(A) = 0", "dist(B) = 1", "dist(C) = 3", "dist(D) = 4", "dist(E) = 7"
                ],
            },
        ],
    }


# ============================================================================
# 1. Stage 2 Prompt Construction Tests
# ============================================================================

def test_stage2_prompt_construction_receives_verified_trace():
    """Verify that get_prompt2_storyboard injects the authoritative canonical Dijkstra trace."""
    outline = make_verified_pilot_outline()
    prompt = get_prompt2_storyboard(outline, level="intermediate")

    # Authoritative facts must be present
    assert "Authoritative Dijkstra Worked Example Execution" in prompt
    assert "Extraction Order: ['A', 'B', 'C', 'D', 'E']" in prompt
    assert "dist(B) = 1" in prompt or "candidate=1" in prompt
    assert "candidate=3" in prompt  # Relaxation of B->C improving C to 3
    assert "candidate=7" in prompt  # Final relaxation D->E to 7
    assert "dock_main" in prompt
    assert "dock_inspector" in prompt

    # Required teaching sequence and visual progression
    assert "Required Teaching Sequence" in prompt
    assert "Visual Progression & Action Requirements" in prompt
    assert "Visible State Transitions - NOT Just Narration" in prompt


def test_extract_verified_algorithmic_state_helper():
    """Test safe extraction from dict, string, and outline objects."""
    outline_dict = make_verified_pilot_outline()
    state1 = extract_verified_algorithmic_state(outline_dict)
    assert state1 is not None
    assert state1["algorithm"] == "dijkstra"
    assert state1["source"] == "A"

    # From JSON string
    state2 = extract_verified_algorithmic_state(json.dumps(outline_dict))
    assert state2 is not None
    assert state2["source"] == "A"

    # From unverified raw spec (dynamic fallback)
    raw_outline = {"topic": "Dijkstra's Shortest Path Algorithm", "worked_example_spec": PILOT_SPEC}
    state3 = extract_verified_algorithmic_state(raw_outline)
    assert state3 is not None
    assert state3["source"] == "A"


# ============================================================================
# 2. Variable Number of Vertices and Trace Length
# ============================================================================

def test_variable_vertices_and_trace_length():
    """Verify fidelity checker dynamically adapts to different graph topologies and vertex counts."""
    spec_3v, rep_3v = DijkstraTraceVerifier.verify_worked_example_spec(SMALL_SPEC, topic="Dijkstra's Algorithm on 3 Vertices")
    outline_3v = {
        "topic": "Dijkstra's Algorithm on 3 Vertices",
        "sections": [{"id": "sec_1", "title": "3-Vertex Dijkstra"}],
        "worked_example_spec": spec_3v,
        "verification_metadata": rep_3v.to_dict(),
    }

    # Complete 3-vertex storyboard
    sb_3v = {
        "sections": [
            {
                "id": "sec_1",
                "title": "3-Vertex Dijkstra",
                "lecture_lines": [
                    "We start at X and relax neighbors Y and Z.",
                    "Then we extract Y and relax Z, finishing at Z.",
                ],
                "captions": ["Extract X and relax", "Extract Y and Z"],
                "animations": [
                    "L1-1 | FadeIn | GraphVisual(vertices=['X', 'Y', 'Z'], source='X') | viewport: dock_main | action: initialize_graph(vertices=['X', 'Y', 'Z'], source='X') | on screen after: [graph]",
                    "L1-2 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('X', 0) | on screen after: [graph]",
                    "L1-3 | Write | MathTex(r\"dist(Y) = 2\") | viewport: dock_inspector | action: relax_edge(u='X', v='Y', candidate=2, updated=True, new_dist=2) | on screen after: [graph]",
                    "L1-4 | Write | MathTex(r\"dist(Z) = 7\") | viewport: dock_inspector | action: relax_edge(u='X', v='Z', candidate=7, updated=True, new_dist=7) | on screen after: [graph]",
                    "L2-1 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('Y', 2) | on screen after: [graph]",
                    "L2-2 | Write | MathTex(r\"dist(Z) = 5\") | viewport: dock_inspector | action: relax_edge(u='Y', v='Z', candidate=5, updated=True, new_dist=5) | on screen after: [graph]",
                    "L2-3 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('Z', 5) | on screen after: [graph]",
                ],
                "verified_values": ["dist(X) = 0", "dist(Y) = 2", "dist(Z) = 5"],
            }
        ]
    }

    checker = StoryboardFidelityChecker()
    report = checker.check_storyboard_fidelity(outline_3v, sb_3v)
    assert report.overall_status == VerificationStatus.VERIFIED
    assert report.is_safe_for_audio is True
    assert len(report.discrepancies) == 0


# ============================================================================
# 3. Positive Test: Valid Complete Storyboard
# ============================================================================

def test_valid_complete_dijkstra_storyboard():
    """Verify that a valid, complete Dijkstra storyboard passes with VERIFIED and is_safe_for_audio=True."""
    outline = make_verified_pilot_outline()
    sb = make_valid_pilot_storyboard()
    checker = StoryboardFidelityChecker()

    report = checker.check_storyboard_fidelity(outline, sb)
    assert report.overall_status == VerificationStatus.VERIFIED
    assert report.is_safe_for_audio is True
    assert len(report.discrepancies) == 0
    assert "Fully verified coverage" in report.summary


# ============================================================================
# 4. Negative Test: Missing Extraction
# ============================================================================

def test_negative_missing_extraction():
    """Verify that omitting an extraction beat (e.g. node E) fails validation."""
    outline = make_verified_pilot_outline()
    sb = make_valid_pilot_storyboard()

    # Delete extraction of node E from section 3
    sec3_anims = sb["sections"][2]["animations"]
    sb["sections"][2]["animations"] = [a for a in sec3_anims if "extract_min('E'" not in a]
    sb["sections"][2]["verified_values"] = [v for v in sb["sections"][2]["verified_values"] if "extract_min(E)" not in v]

    checker = StoryboardFidelityChecker()
    report = checker.check_storyboard_fidelity(outline, sb)

    assert report.overall_status == VerificationStatus.FAILED
    assert report.is_safe_for_audio is False
    categories = [d.category for d in report.discrepancies]
    assert FidelityCategory.MISSING_ALGORITHMIC_STEP in categories
    assert any("node 'e' extraction omitted" in d.actual_claim.lower() for d in report.discrepancies)


# ============================================================================
# 5. Negative Test: Incorrect Extraction Order
# ============================================================================

def test_negative_incorrect_extraction_order():
    """Verify that extracting nodes out of canonical order (e.g. C before B) fails validation."""
    outline = make_verified_pilot_outline()
    sb = make_valid_pilot_storyboard()

    # Invert order of extractions in Section 2: extract C before B
    sb["sections"][1]["animations"][0] = "L1-1 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('C', 3) | on screen after: [graph, table]"
    sb["sections"][1]["animations"][3] = "L3-1 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('B', 1) | on screen after: [graph, table]"

    checker = StoryboardFidelityChecker()
    report = checker.check_storyboard_fidelity(outline, sb)

    assert report.overall_status == VerificationStatus.FAILED
    assert report.is_safe_for_audio is False
    categories = [d.category for d in report.discrepancies]
    assert FidelityCategory.ALGORITHMIC_ORDER_MISMATCH in categories


# ============================================================================
# 6. Negative Test: Missing Relaxation
# ============================================================================

def test_negative_missing_relaxation():
    """Verify that omitting a required outgoing edge relaxation fails validation."""
    outline = make_verified_pilot_outline()
    sb = make_valid_pilot_storyboard()

    # Remove relaxation of edge (B, C) from section 2
    sec2_anims = sb["sections"][1]["animations"]
    sb["sections"][1]["animations"] = [a for a in sec2_anims if "v='C'" not in a]

    checker = StoryboardFidelityChecker()
    report = checker.check_storyboard_fidelity(outline, sb)

    assert report.overall_status == VerificationStatus.FAILED
    assert report.is_safe_for_audio is False
    categories = [d.category for d in report.discrepancies]
    assert FidelityCategory.MISSING_ALGORITHMIC_STEP in categories
    assert any("relaxation of edge (b, c) missing" in d.actual_claim.lower() for d in report.discrepancies)


# ============================================================================
# 7. Negative Test: Incorrect Updated Distance (Pilot Regression C = 4)
# ============================================================================

def test_negative_incorrect_updated_distance_pilot_regression():
    """Verify that claiming an incorrect intermediate distance (C = 4 instead of 3) fails validation."""
    outline = make_verified_pilot_outline()
    sb = make_valid_pilot_storyboard()

    # The actual pilot regression: claiming C stayed 4 after relaxing edge (B, C)
    sb["sections"][1]["animations"][1] = "L2-1 | Write | MathTex(r\"dist(C) = 4\") | viewport: dock_inspector | action: relax_edge(u='B', v='C', candidate=4, updated=True, new_dist=4) | on screen after: [graph, table]"

    checker = StoryboardFidelityChecker()
    report = checker.check_storyboard_fidelity(outline, sb)

    assert report.overall_status == VerificationStatus.FAILED
    assert report.is_safe_for_audio is False
    categories = [d.category for d in report.discrepancies]
    assert FidelityCategory.ALGORITHMIC_STATE_MISMATCH in categories
    assert any("dist(C) = 4" in d.actual_claim for d in report.discrepancies)


# ============================================================================
# 8. Negative Test: Incorrect Unsuccessful-Relaxation Representation
# ============================================================================

def test_negative_incorrect_unsuccessful_relaxation_representation():
    """Verify that falsely claiming an unsuccessful relaxation updated the distance fails validation."""
    unsuccessful_spec = {
        "algorithm": "dijkstra",
        "topic": "Dijkstra with Unsuccessful Relaxation",
        "inputs": {
            "vertices": ["A", "B", "C"],
            "source": "A",
            "edges": [
                {"u": "A", "v": "B", "weight": 1.0},
                {"u": "A", "v": "C", "weight": 4.0},
                {"u": "B", "v": "C", "weight": 10.0},
            ],
            "directed": True,
        },
        "steps": [
            {"step": "Init dist[A]=0, others=∞", "result": "dist = {A: 0, B: 1, C: 4}"},
            {"step": "Extract A, relax B and C", "result": "dist = {A: 0, B: 1, C: 4}"},
            {"step": "Extract B, examine B->C (no update)", "result": "dist = {A: 0, B: 1, C: 4}"},
            {"step": "Extract C, trace complete", "result": "dist = {A: 0, B: 1, C: 4}"},
        ],
        "final_result": "dist = {A: 0, B: 1, C: 4}",
    }
    spec_out, rep = DijkstraTraceVerifier.verify_worked_example_spec(unsuccessful_spec, topic="Dijkstra with Unsuccessful Relaxation")
    outline = {
        "topic": "Dijkstra with Unsuccessful Relaxation",
        "sections": [{"id": "sec_1", "title": "Unsuccessful Relaxation"}],
        "worked_example_spec": spec_out,
        "verification_metadata": rep.to_dict(),
    }

    # Storyboard falsely claims that candidate 11 updated node C
    sb = {
        "sections": [
            {
                "id": "sec_1",
                "title": "Unsuccessful Relaxation",
                "lecture_lines": [
                    "We start at A and relax B and C.",
                    "Then we extract B and falsely update C to 11.",
                    "Finally we extract C.",
                ],
                "captions": ["Init", "Falsely update C", "Done"],
                "animations": [
                    "L1-1 | FadeIn | GraphVisual(vertices=['A', 'B', 'C'], source='A') | viewport: dock_main | action: initialize_graph(vertices=['A', 'B', 'C'], source='A') | on screen after: [graph]",
                    "L1-2 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('A', 0) | on screen after: [graph]",
                    "L1-3 | Write | MathTex(r\"dist(B) = 1\") | viewport: dock_inspector | action: relax_edge(u='A', v='B', candidate=1, updated=True, new_dist=1) | on screen after: [graph]",
                    "L1-4 | Write | MathTex(r\"dist(C) = 4\") | viewport: dock_inspector | action: relax_edge(u='A', v='C', candidate=4, updated=True, new_dist=4) | on screen after: [graph]",
                    "L2-1 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('B', 1) | on screen after: [graph]",
                    "L2-2 | Write | MathTex(r\"dist(C) = 11\") | viewport: dock_inspector | action: relax_edge(u='B', v='C', candidate=11, updated=True, new_dist=11) | on screen after: [graph]",
                    "L3-1 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('C', 4) | on screen after: [graph]",
                ],
                "verified_values": ["dist(A) = 0", "dist(B) = 1", "dist(C) = 4"],
            }
        ]
    }

    checker = StoryboardFidelityChecker()
    report = checker.check_storyboard_fidelity(outline, sb)

    assert report.overall_status == VerificationStatus.FAILED
    assert report.is_safe_for_audio is False
    categories = [d.category for d in report.discrepancies]
    assert FidelityCategory.ALGORITHMIC_STATE_MISMATCH in categories
    assert any("unsuccessful relaxation of edge (b, c)" in d.reason.lower() for d in report.discrepancies)


# ============================================================================
# 9. Negative Test: Correct Final Distances but Incomplete Intermediate Trace
# ============================================================================

def test_negative_correct_final_distances_but_incomplete_intermediate_trace():
    """Verify that showing final distances while skipping intermediate extractions fails validation."""
    outline = make_verified_pilot_outline()

    # Storyboard displays initial graph and final table, but skips intermediate extractions
    sb_skipped = {
        "sections": [
            {
                "id": "section_1",
                "title": "Sec 1: Overview and Final Result",
                "lecture_lines": [
                    "We start at node A and find all shortest paths.",
                    "Here are the final shortest distances.",
                ],
                "captions": ["Overview", "Final result"],
                "animations": [
                    "L1-1 | FadeIn | GraphVisual(vertices=['A', 'B', 'C', 'D', 'E'], source='A') | viewport: dock_main | action: initialize_graph(vertices=['A', 'B', 'C', 'D', 'E'], source='A') | on screen after: [graph]",
                    "L1-2 | Indicate | GraphVisual | viewport: dock_main | action: extract_min('A', 0) | on screen after: [graph]",
                    "L2-1 | Write | Table(FinalDistances) | viewport: dock_inspector | action: show_final_table() | on screen after: [graph, final_table]",
                ],
                "verified_values": ["dist(A) = 0", "dist(B) = 1", "dist(C) = 3", "dist(D) = 4", "dist(E) = 7"],
            }
        ]
    }

    checker = StoryboardFidelityChecker()
    report = checker.check_storyboard_fidelity(outline, sb_skipped)

    assert report.overall_status == VerificationStatus.FAILED
    assert report.is_safe_for_audio is False
    categories = [d.category for d in report.discrepancies]
    assert FidelityCategory.MISSING_ALGORITHMIC_STEP in categories
    # Nodes B, C, D, E extractions are missing
    assert any("node 'b' extraction omitted" in d.actual_claim.lower() for d in report.discrepancies)


# ============================================================================
# 10. Negative Test: Narration-Only Without Visual Transitions
# ============================================================================

def test_negative_narration_only_without_visual_transitions():
    """Verify that describing operations in narration without visual transitions fails validation."""
    outline = make_verified_pilot_outline()

    # Storyboard describes everything in spoken narration, but animations are static Text cards
    sb_narration_only = {
        "sections": [
            {
                "id": "section_1",
                "title": "Sec 1: Spoken Dijkstra",
                "lecture_lines": [
                    "We extract node A and relax its neighbors B and C.",
                    "We then extract node B and update tentative distances.",
                    "We continue this greedy selection until all shortest paths are found.",
                ],
                "captions": ["Extract A", "Extract B", "Finish trace"],
                "animations": [
                    "L1-1 | FadeIn | Text('Dijkstra Algorithm Overview') | viewport: dock_main | action: show_title() | on screen after: [title]",
                    "L2-1 | Write | Text('Greedy Edge Relaxation') | viewport: dock_main | action: show_concept() | on screen after: [title, concept]",
                    "L3-1 | Write | Text('Shortest Path Complete') | viewport: dock_main | action: show_summary() | on screen after: [title, summary]",
                ],
                "verified_values": [],
            }
        ]
    }

    checker = StoryboardFidelityChecker()
    report = checker.check_storyboard_fidelity(outline, sb_narration_only)

    assert report.overall_status == VerificationStatus.FAILED
    assert report.is_safe_for_audio is False
    categories = [d.category for d in report.discrepancies]
    assert FidelityCategory.INSUFFICIENT_VISUAL_COVERAGE in categories


# ============================================================================
# 11. Negative Test: Graph / Source Mismatch
# ============================================================================

def test_negative_graph_source_mismatch():
    """Verify that specifying an incorrect source node or graph vertices fails validation."""
    outline = make_verified_pilot_outline()
    sb = make_valid_pilot_storyboard()

    # Claim starting source is B instead of A
    sb["sections"][0]["animations"][0] = "L1-1 | FadeIn | GraphVisual(vertices=['A', 'B', 'C', 'D', 'E'], source='B') | viewport: dock_main | action: initialize_graph(vertices=['A', 'B', 'C', 'D', 'E'], source='B') | on screen after: [graph]"

    checker = StoryboardFidelityChecker()
    report = checker.check_storyboard_fidelity(outline, sb)

    assert report.overall_status == VerificationStatus.FAILED
    assert report.is_safe_for_audio is False
    categories = [d.category for d in report.discrepancies]
    assert FidelityCategory.GRAPH_SPEC_MISMATCH in categories


# ============================================================================
# 12. Regression Test: Non-Dijkstra Storyboards
# ============================================================================

def test_non_dijkstra_storyboard_regression():
    """Verify that non-Dijkstra worked examples (e.g. Arrhenius equation) continue using scalar numerical checking."""
    arrhenius_outline = {
        "topic": "Arrhenius Equation",
        "learning_level": "intermediate",
        "sections": [
            {"id": "section_1", "title": "Sec 1: Overview"},
            {"id": "section_2", "title": "Sec 2: Worked Example: Rate at 300K"},
        ],
        "governing_principles": ["Arrhenius Equation", "k = A * exp(-Ea / (R * T))"],
        "worked_example_spec": {
            "inputs": {"Ea": 75000, "R": 8.314, "T1": 300, "A": 7.5e10},
            "steps": [
                {
                    "step": "Compute k1",
                    "result": "k1 = 0.006545 s^-1",
                    "structured_calc": {
                        "output_variable": "k1",
                        "computed_value": 0.006545,
                        "corrected_value": "0.006545",
                        "unit": "s^-1",
                    },
                }
            ],
            "final_result": "k1 = 0.006545 s^-1",
        },
    }

    valid_arrhenius_sb = {
        "sections": [
            {
                "id": "section_2",
                "title": "Sec 2: Worked Example: Rate at 300K",
                "lecture_lines": ["The rate constant k1 evaluates to 0.006545 per second."],
                "captions": ["Calculate k1"],
                "animations": [
                    "L1-1 | Write | MathTex(r\"k_1 = 0.006545\\text{ s}^{-1}\") | viewport: dock_inspector | action: show_calc() | on screen after: [formula]"
                ],
                "verified_values": {"k1": "0.006545 s^-1"},
            }
        ]
    }

    checker = StoryboardFidelityChecker()
    report = checker.check_storyboard_fidelity(arrhenius_outline, valid_arrhenius_sb)
    assert report.overall_status == VerificationStatus.VERIFIED
    assert report.is_safe_for_audio is True


# ============================================================================
# 13. Pipeline Test: Retry Exhaustion Blocks Audio and Stage 3
# ============================================================================

def test_retry_exhaustion_blocks_audio_and_stage3(tmp_path):
    """Verify that when storyboard fidelity validation fails after max retries, audio and Stage 3 are blocked."""
    outline_data = make_verified_pilot_outline()
    cfg = RunConfig(max_fidelity_retries=1)

    agent = TeachingVideoAgent(
        idx=1,
        knowledge_point="Dijkstra's Shortest Path Algorithm",
        folder=str(tmp_path / "CASES"),
        cfg=cfg,
    )
    agent.outline = TeachingOutline.from_dict(outline_data)

    # Invalid storyboard with missing extractions and pilot regression
    invalid_sb = {
        "sections": [
            {
                "id": "section_1",
                "title": "Sec 1: Incomplete Dijkstra",
                "lecture_lines": ["A brief outline."],
                "captions": ["Brief"],
                "animations": ["L1-1 | FadeIn | Text('Dijkstra') | viewport: dock_main | action: show_text() | on screen after: [text]"],
                "verified_values": [],
            }
        ]
    }
    agent.enhanced_storyboard = invalid_sb

    # Ensure fidelity check fails
    report = agent._ensure_storyboard_verified(force=True)
    assert report.is_safe_for_audio is False

    # pregenerate_all_section_audios must raise ValueError
    with pytest.raises(ValueError, match="Cannot pre-generate audio: Storyboard failed numerical fidelity gate"):
        agent.pregenerate_all_section_audios()

    # generate_codes must raise ValueError
    sec = Section(id="section_1", title="Sec 1", lecture_lines=["Line"], animations=["Anim"])
    agent.sections = [sec]
    with pytest.raises(ValueError, match="Cannot generate codes: Storyboard failed numerical fidelity gate"):
        agent.generate_codes()


# ============================================================================
# 14. Pipeline Test: Checkpoint Restoration Preserves Algorithmic State
# ============================================================================

def test_checkpoint_preserves_verified_algorithmic_state(tmp_path):
    """Verify that saving and loading checkpoints preserves verified_algorithmic_state."""
    outline_data = make_verified_pilot_outline()
    cases_dir = tmp_path / "CASES"

    agent = TeachingVideoAgent(
        idx=1,
        knowledge_point="Dijkstra's Shortest Path Algorithm",
        folder=str(cases_dir),
    )
    agent.outline = TeachingOutline.from_dict(outline_data)

    # Save checkpoint
    agent.save_checkpoint("storyboard_generated")

    checkpoint_file = agent.output_dir / "checkpoint.json"
    assert checkpoint_file.exists()

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        cp = json.load(f)

    # Verify outline in checkpoint has verified_algorithmic_state
    cp_outline = cp.get("outline", {})
    vm = cp_outline.get("verification_metadata", {})
    assert "verified_algorithmic_state" in vm
    algo = vm["verified_algorithmic_state"]
    assert algo["algorithm"] == "dijkstra"
    assert algo["source"] == "A"
    assert algo["settled_order"] == ["A", "B", "C", "D", "E"]

    # Re-instantiate agent and load checkpoint
    agent2 = TeachingVideoAgent(
        idx=1,
        knowledge_point="Dijkstra's Shortest Path Algorithm",
        folder=str(cases_dir),
    )
    agent2._load_existing_checkpoint_and_files()

    assert agent2.outline is not None
    assert agent2.outline.topic == "Dijkstra's Shortest Path Algorithm"
    vm2 = agent2.outline.to_dict().get("verification_metadata", {})
    assert "verified_algorithmic_state" in vm2
    assert vm2["verified_algorithmic_state"]["source"] == "A"


# ============================================================================
# 15. Repair Prompt Generation Test
# ============================================================================

def test_repair_prompt_generation_includes_algorithmic_instructions():
    """Verify that get_prompt2_storyboard_fidelity_repair includes algorithmic instructions for Dijkstra."""
    outline = make_verified_pilot_outline()
    discrepancies = [
        FidelityDiscrepancy(
            section_id="section_2",
            location="animations[1]",
            category=FidelityCategory.ALGORITHMIC_STATE_MISMATCH,
            actual_claim="dist(C) = 4",
            expected_value="dist(C) = 3",
            reason="Incorrect updated distance",
            suggested_correction="Update distance of C to 3",
            severity=VerificationStatus.FAILED,
        )
    ]
    repair_prompt = get_prompt2_storyboard_fidelity_repair(
        outline=outline,
        storyboard="{}",
        discrepancies=discrepancies,
    )

    assert "Issue #1 [ALGORITHMIC_STATE_MISMATCH]" in repair_prompt
    assert "Algorithmic Worked Example Fidelity (MANDATORY)" in repair_prompt
    assert "['A', 'B', 'C', 'D', 'E']" in repair_prompt
    assert "relax_edge" in repair_prompt
