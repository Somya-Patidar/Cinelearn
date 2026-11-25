"""
Milestone 1.5 Test Suite: Generic Prompt Decoupling & Dynamic Profile Routing.

Covers:
A. Generic topic isolation across diverse canonical domains:
   - Binary Search (CS / Algorithms)
   - Chemical reaction kinetics / Arrhenius (Natural Science / Chemistry)
   - Pythagorean Theorem (Mathematics / Proof)
   - Photosynthesis (Natural Science / Biology)
   Verify absence of hardcoded FCFS CPU scheduling and Dijkstra instructions.
B. Dynamic FCFS CPU scheduling guidance:
   - Verified presence of OS scheduling concepts (PCB, 4-state lifecycle, FIFO ready queue,
     non-preemptive invariant, execution-connected metrics, averages, Convoy effect)
     when topic/profile is FCFS CPU scheduling.
   - Verified ABSENCE of these concepts on unrelated topics.
C. Dynamic Dijkstra / Graph guidance:
   - Verified presence of greedy finalization invariant and non-negative edge weights
     when topic/profile is Dijkstra.
   - Verified ABSENCE of these concepts on unrelated topics.
D. Missing/partial profile handling:
   - Safe fallback when topic_profile is None, partial dict, or UNCLASSIFIED.
   - Graceful generation without inventing fabricated specialized strategies.
E. Downstream compatibility & Schema preservation:
   - get_prompt1_outline and get_prompt2_storyboard interfaces remain backward-compatible.
   - Outline and storyboard JSON schemas remain unchanged.
F. Prompt inspection with bidirectional assertions:
   - Verify both positive presence of relevant guidance and negative absence of leakage.
"""

import json
from pathlib import Path
import sys
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from prompts.stage1 import get_prompt1_outline, _resolve_topic_profile, _format_stage1_profile_guidance
from prompts.stage2 import get_prompt2_storyboard, extract_topic_profile, _format_stage2_profile_guidance
from src.pedagogy import (
    TopicProfile,
    TopicProfiler,
    StateEvolution,
    FormalismLevel,
    ComputationRequirement,
    PrimaryVisualMode,
    ProfileStatus,
)
from src.agent import TeachingOutline


# Key instructional strings that must NOT leak into unrelated topics
OS_SCHEDULING_LEAKAGE_TERMS = [
    "In operating systems and CPU scheduling",
    "P1: AT=0, BT=3",
    "Ready Queue FIFO dynamics",
    "NEW -> READY -> RUNNING -> TERMINATED",
    "non-preemptive execution invariant",
    "Turnaround Time (TAT = CT - AT)",
    "Waiting Time (WT = TAT - BT)",
    "Convoy Effect",
    "Teaching Sequence Guidance",
    "Implementation-Oriented Teaching Quality",
    "Authoritative FCFS CPU Scheduling Simulation",
]

DIJKSTRA_LEAKAGE_TERMS = [
    "greedy finalization invariant",
    "min-heap guarantees true shortest distance",
    "extracting vertex u with minimal tentative distance",
    "Authoritative Dijkstra Worked Example Execution",
]


# ============================================================================
# A. Generic Topic Isolation Across Diverse Canonical Domains
# ============================================================================

class TestGenericTopicIsolation:
    """Verifies that generic Stage 1 and Stage 2 prompts do NOT inject irrelevant
    FCFS CPU scheduling or Dijkstra instructions into unrelated topics."""

    @pytest.mark.parametrize("topic", [
        "Binary Search",
        "Chemical reaction kinetics",
        "Arrhenius Activation Energy",
        "The Pythagorean Theorem",
        "Photosynthesis: Light Reactions and Calvin Cycle",
        "Calculus Derivatives and Rates of Change",
    ])
    def test_stage1_prompt_free_of_fcfs_and_dijkstra_leakage(self, topic):
        prompt = get_prompt1_outline(topic, level="intermediate")

        # Must not contain hardcoded FCFS CPU scheduling instructions
        for term in OS_SCHEDULING_LEAKAGE_TERMS:
            assert term not in prompt, f"Leakage detected: '{term}' found in Stage 1 prompt for '{topic}'"

        # Must not contain hardcoded Dijkstra instructions
        for term in DIJKSTRA_LEAKAGE_TERMS:
            assert term not in prompt, f"Leakage detected: '{term}' found in Stage 1 prompt for '{topic}'"

    @pytest.mark.parametrize("topic", [
        "Binary Search",
        "Chemical reaction kinetics",
        "The Pythagorean Theorem",
        "Photosynthesis",
    ])
    def test_stage2_prompt_free_of_fcfs_and_dijkstra_leakage(self, topic):
        mock_outline = {
            "topic": topic,
            "sections": [
                {"id": "section_1", "title": "Sec 1: Introduction", "visual_idea": "Diagram"},
                {"id": "section_2", "title": "Sec 2: Mechanism", "visual_idea": "Process"},
            ]
        }
        prompt = get_prompt2_storyboard(mock_outline, level="intermediate")

        # Must not contain hardcoded FCFS CPU scheduling instructions
        for term in OS_SCHEDULING_LEAKAGE_TERMS:
            assert term not in prompt, f"Leakage detected: '{term}' found in Stage 2 prompt for '{topic}'"

        # Must not contain hardcoded Dijkstra instructions
        for term in DIJKSTRA_LEAKAGE_TERMS:
            assert term not in prompt, f"Leakage detected: '{term}' found in Stage 2 prompt for '{topic}'"


# ============================================================================
# B. FCFS Dynamic Guidance & Isolation
# ============================================================================

class TestFCFSDynamicGuidance:
    """Verifies that FCFS CPU Scheduling receives complete, appropriate pedagogical
    guidance when supported by topic profile and context, while remaining isolated."""

    def test_fcfs_receives_complete_scheduling_guidance_stage1(self):
        prompt = get_prompt1_outline("FCFS CPU Scheduling", level="intermediate")

        # Positive presence of core OS CPU scheduling concepts
        assert "operating systems and cpu scheduling" in prompt.lower()
        assert "Process Control Block (PCB)" in prompt
        assert "PID, arrival time, CPU burst time, current process state" in prompt
        assert "NEW -> READY -> RUNNING -> TERMINATED" in prompt
        assert "Ready Queue FIFO dynamics" in prompt
        assert "non-preemptive execution invariant" in prompt
        assert "Completion Time (CT)" in prompt
        assert "Turnaround Time (TAT = CT - AT)" in prompt
        assert "Waiting Time (WT = TAT - BT)" in prompt
        assert "Response Time (RT)" in prompt
        assert "Average CT = (3 + 7 + 9) / 3 = 19/3 = 6.33" in prompt
        assert "Average TAT = (3 + 6 + 7) / 3 = 16/3 = 5.33" in prompt
        assert "Average WT = (0 + 2 + 5) / 3 = 7/3 = 2.33" in prompt
        assert "Convoy Effect" in prompt

        # Negative check: Dijkstra guidance must NOT leak into FCFS
        for term in DIJKSTRA_LEAKAGE_TERMS:
            assert term not in prompt, f"Dijkstra term '{term}' leaked into FCFS prompt"

    def test_fcfs_with_explicit_profile_injects_guidance(self):
        fcfs_profile = TopicProfile(
            state_evolution=StateEvolution.DISCRETE_STATE_MACHINE,
            formalism_level=FormalismLevel.SYSTEM_STRUCTURAL,
            computation_requirement=ComputationRequirement.DETERMINISTIC_TRACE,
            primary_visual_mode=PrimaryVisualMode.STATE_TABLE_MATRIX,
            rationale_and_evidence="FCFS CPU Scheduling is a discrete state machine managing process queues.",
            profile_status=ProfileStatus.CLASSIFIED,
            confidence=0.95,
            secondary_visual_modes=["process_flowchart"],
            secondary_characteristics=["cpu_scheduling", "queue_discipline"],
        )
        prompt = get_prompt1_outline("Custom CPU Scheduler", topic_profile=fcfs_profile)

        assert "operating systems and cpu scheduling" in prompt.lower()
        assert "Process Control Block (PCB)" in prompt
        assert "Ready Queue FIFO dynamics" in prompt
        assert "Turnaround Time (TAT = CT - AT)" in prompt

    def test_fcfs_stage2_receives_trace_guidance_when_outline_has_fcfs_trace(self):
        fcfs_outline = {
            "topic": "FCFS CPU Scheduling",
            "verification_metadata": {
                "verified_algorithmic_state": {
                    "algorithm": "fcfs",
                    "processes": [{"pid": "P1", "arrival_time": 0, "burst_time": 3}],
                    "gantt_chart": [{"pid": "P1", "start_time": 0, "end_time": 3, "duration": 3}],
                    "process_metrics": {"P1": {"completion_time": 3, "turnaround_time": 3, "waiting_time": 0, "response_time": 0}},
                    "average_waiting_time": 0.0,
                    "average_turnaround_time": 3.0,
                    "average_response_time": 0.0,
                    "total_simulation_time": 3,
                }
            },
            "sections": [{"id": "sec_1", "title": "Execution", "visual_idea": "Gantt"}]
        }
        prompt = get_prompt2_storyboard(fcfs_outline)
        assert "Authoritative FCFS CPU Scheduling Simulation" in prompt
        assert "GanttChartVisual" in prompt
        assert "ReadyQueueVisual" in prompt
        for term in DIJKSTRA_LEAKAGE_TERMS:
            assert term not in prompt


# ============================================================================
# C. Dijkstra Dynamic Guidance & Isolation
# ============================================================================

class TestDijkstraDynamicGuidance:
    """Verifies that Dijkstra Shortest Path receives appropriate greedy graph
    guidance when supported by topic profile and context, while remaining isolated."""

    def test_dijkstra_receives_greedy_invariant_stage1(self):
        prompt = get_prompt1_outline("Dijkstra's Shortest Path Algorithm", level="intermediate")

        # Positive presence of Dijkstra guidance
        assert "greedy finalization invariant" in prompt
        assert "non-negative edge weights" in prompt
        assert "min-heap" in prompt

        # Negative check: OS scheduling guidance must NOT leak into Dijkstra
        for term in OS_SCHEDULING_LEAKAGE_TERMS:
            assert term not in prompt, f"OS scheduling term '{term}' leaked into Dijkstra prompt"

    def test_dijkstra_with_explicit_profile_injects_guidance(self):
        dijkstra_profile = TopicProfile(
            state_evolution=StateEvolution.DISCRETE_STATE_MACHINE,
            formalism_level=FormalismLevel.RIGOROUS_MATHEMATICAL,
            computation_requirement=ComputationRequirement.DETERMINISTIC_TRACE,
            primary_visual_mode=PrimaryVisualMode.NETWORK_GRAPH,
            rationale_and_evidence="Graph shortest path algorithm with priority queue relaxation.",
            profile_status=ProfileStatus.CLASSIFIED,
            confidence=0.95,
            secondary_characteristics=["greedy_relaxation", "priority_queue"],
        )
        prompt = get_prompt1_outline("Shortest Path Search", topic_profile=dijkstra_profile)

        assert "greedy finalization invariant" in prompt
        assert "non-negative edge weights" in prompt

    def test_dijkstra_stage2_receives_trace_guidance_when_outline_has_dijkstra_trace(self):
        dijkstra_outline = {
            "topic": "Dijkstra's Shortest Path Algorithm",
            "verification_metadata": {
                "verified_algorithmic_state": {
                    "algorithm": "dijkstra",
                    "source": "A",
                    "vertices": ["A", "B"],
                    "edges": [{"u": "A", "v": "B", "weight": 2}],
                    "directed": False,
                    "settled_order": ["A", "B"],
                    "final_distances": {"A": 0, "B": 2},
                    "shortest_paths": {"A": ["A"], "B": ["A", "B"]},
                    "canonical_trace": [],
                }
            },
            "sections": [{"id": "sec_1", "title": "Trace", "visual_idea": "Graph"}]
        }
        prompt = get_prompt2_storyboard(dijkstra_outline)
        assert "Authoritative Dijkstra Worked Example Execution" in prompt
        assert "settled_order" in prompt or "Extraction Order" in prompt
        for term in OS_SCHEDULING_LEAKAGE_TERMS:
            assert term not in prompt
        assert "settled_order" in prompt or "Extraction Order" in prompt
        for term in OS_SCHEDULING_LEAKAGE_TERMS:
            assert term not in prompt


# ============================================================================
# D. Missing and Partial Profile Behavior
# ============================================================================

class TestMissingAndPartialProfileBehavior:
    """Verifies safe, deterministic fallback behavior when topic_profile is
    missing, partial, or UNCLASSIFIED."""

    def test_none_profile_uses_deterministic_fallback(self):
        prompt = get_prompt1_outline("Unrecognized Novel Concept", topic_profile=None)
        assert "Dynamic Pedagogical Strategy" in prompt
        assert "Profile Status: UNCLASSIFIED (Confidence: 0.00)" in prompt
        # Must still contain core educational planning requirements
        assert "Technical Accuracy & Correctness" in prompt
        assert "Numerical & Worked-Example Verification" in prompt
        # Must not fabricate specialized strategies
        for term in OS_SCHEDULING_LEAKAGE_TERMS + DIJKSTRA_LEAKAGE_TERMS:
            assert term not in prompt

    def test_empty_dict_profile_falls_back_safely(self):
        prompt = get_prompt1_outline("Generic Logic Topic", topic_profile={})
        assert "Dynamic Pedagogical Strategy" in prompt
        assert "Profile Status: UNCLASSIFIED" in prompt
        assert "Pedagogical Progression (FORMAL FIRST)" in prompt

    def test_partial_dict_profile_deserializes_and_falls_back(self):
        partial_data = {
            "state_evolution": "discrete_state_machine",
            # missing formalism_level, computation_requirement, primary_visual_mode, rationale
        }
        prompt = get_prompt1_outline("Partial Specification Topic", topic_profile=partial_data)
        assert "Dynamic Pedagogical Strategy" in prompt
        # Schema validation catches missing fields and falls back to UNCLASSIFIED
        assert "Profile Status: UNCLASSIFIED" in prompt

    def test_stage2_with_missing_profile_uses_safe_visual_mode(self):
        outline = {
            "topic": "Abstract Pure Mathematics",
            "sections": [{"id": "s1", "title": "Overview", "visual_idea": "Shapes"}]
        }
        prompt = get_prompt2_storyboard(outline)
        assert "Topic Profile Visual Strategy" in prompt
        assert "Domain-Independent Visual Reasoning" in prompt
        for term in OS_SCHEDULING_LEAKAGE_TERMS + DIJKSTRA_LEAKAGE_TERMS:
            assert term not in prompt


# ============================================================================
# E. Downstream Compatibility & Schema Preservation
# ============================================================================

class TestCompatibilityAndSchemaPreservation:
    """Verifies that public signatures, outline schemas, and storyboard contracts
    remain 100% backward-compatible."""

    def test_get_prompt1_outline_signature_is_backward_compatible(self):
        # Can be called with 1 to 5 positional arguments exactly as before
        p1 = get_prompt1_outline("Binary Search")
        assert len(p1) > 0
        p2 = get_prompt1_outline("Binary Search", 5.0)
        assert len(p2) > 0
        p3 = get_prompt1_outline("Binary Search", 5.0, None)
        assert len(p3) > 0
        p4 = get_prompt1_outline("Binary Search", 5.0, None, "advanced")
        assert len(p4) > 0
        p5 = get_prompt1_outline("Binary Search", 5.0, None, "advanced", "Book indexing")
        assert len(p5) > 0
        # Accepts new optional topic_profile
        p6 = get_prompt1_outline("Binary Search", 5.0, None, "advanced", "Book indexing", topic_profile=None)
        assert len(p6) > 0

    def test_get_prompt2_storyboard_signature_is_backward_compatible(self):
        outline = {"topic": "Binary Search", "sections": []}
        # Positional arguments 1 to 5 work exactly as before
        p1 = get_prompt2_storyboard(outline)
        assert len(p1) > 0
        p2 = get_prompt2_storyboard(outline, None)
        assert len(p2) > 0
        p3 = get_prompt2_storyboard(outline, None, "advanced")
        assert len(p3) > 0
        p4 = get_prompt2_storyboard(outline, None, "advanced", "Analogy")
        assert len(p4) > 0
        p5 = get_prompt2_storyboard(outline, None, "advanced", "Analogy", {})
        assert len(p5) > 0
        # Accepts new optional topic_profile
        p6 = get_prompt2_storyboard(outline, topic_profile=None)
        assert len(p6) > 0

    def test_outline_schema_output_remains_json_parsable(self):
        prompt = get_prompt1_outline("Binary Search")
        # Extract schema JSON object from the prompt text
        match = prompt.split("Output ONLY one valid JSON object (no markdown fences, no comments, no trailing commas) with exactly this structure:\n")
        assert len(match) == 2
        schema_json_str = match[1].split("\nThe 'sections' array")[0].strip()
        schema_data = json.loads(schema_json_str)

        # Mandatory top-level outline fields
        assert "topic" in schema_data
        assert "topic_profile" in schema_data
        assert "formal_definition" in schema_data
        assert "worked_example_spec" in schema_data
        assert "sections" in schema_data
        assert "pedagogical_conclusion" in schema_data


# ============================================================================
# F. Bidirectional Content Assertions on Representative Topics
# ============================================================================

class TestBidirectionalContentAssertions:
    """Verifies that each domain receives its expected pedagogical strategy
    and excludes other domains' strategies."""

    def test_binary_search_has_search_guidance_and_no_leakage(self):
        prompt = get_prompt1_outline("Binary Search", level="intermediate")
        # Expected search guidance
        assert "Search Algorithms & Pointer Convergence" in prompt
        assert "pointer convergence (low, high, mid)" in prompt
        assert "discrete_state_machine" in prompt
        assert "deterministic_trace" in prompt
        # Disallowed leakage
        assert "In operating systems and CPU scheduling" not in prompt
        assert "P1: AT=0, BT=3" not in prompt
        assert "Ready Queue FIFO dynamics" not in prompt
        assert "greedy finalization invariant" not in prompt
        assert "Maxwell-Boltzmann" not in prompt
        assert "catalyst provides an alternative" not in prompt

    def test_chemical_kinetics_has_chemistry_guidance_and_no_leakage(self):
        prompt = get_prompt1_outline("Chemical reaction kinetics", level="intermediate")
        # Expected chemistry guidance
        assert "Chemical Kinetics & Reaction Coordinates" in prompt
        assert "catalyst provides an alternative lower-activation-energy reaction pathway" in prompt
        assert "Maxwell-Boltzmann distribution" in prompt
        assert "Arrhenius pre-exponential factor A" in prompt
        assert "continuous_dynamic" in prompt
        # Disallowed leakage
        assert "In operating systems and CPU scheduling" not in prompt
        assert "P1: AT=0, BT=3" not in prompt
        assert "Ready Queue FIFO dynamics" not in prompt
        assert "greedy finalization invariant" not in prompt

    def test_pythagorean_theorem_has_proof_guidance_and_no_leakage(self):
        prompt = get_prompt1_outline("The Pythagorean Theorem", level="intermediate")
        # Expected proof guidance
        assert "Geometric Proof & Invariant Dissection" in prompt
        assert "geometric dissection and area equivalence" in prompt
        assert "static_structural" in prompt
        assert "symbolic_identity" in prompt
        # Disallowed leakage
        assert "In operating systems and CPU scheduling" not in prompt
        assert "P1: AT=0, BT=3" not in prompt
        assert "Ready Queue FIFO dynamics" not in prompt
        assert "greedy finalization invariant" not in prompt
        assert "catalyst" not in prompt

    def test_photosynthesis_has_biochemical_guidance_and_no_leakage(self):
        prompt = get_prompt1_outline("Photosynthesis: Light Reactions and Calvin Cycle", level="intermediate")
        # Expected biochemical guidance
        assert "Biochemical Pathways & Multi-Stage Transformations" in prompt
        assert "multi-stage biochemical pathways" in prompt
        assert "multi_stage_process" in prompt
        assert "none_not_applicable" in prompt
        # Disallowed leakage
        assert "In operating systems and CPU scheduling" not in prompt
        assert "P1: AT=0, BT=3" not in prompt
        assert "Ready Queue FIFO dynamics" not in prompt
        assert "greedy finalization invariant" not in prompt
        assert "derivative of position" not in prompt
