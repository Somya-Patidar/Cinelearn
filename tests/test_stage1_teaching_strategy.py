"""Unit and regression tests for CineLearn Step 2B.1 Stage 1 Teaching Strategy.
Validates formal-first pedagogy, autonomous intuition selection, flexible section progression,
visual progression types, technical accuracy safeguards, and schema compatibility.
"""

import inspect
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pytest
from prompts.stage1 import (
    get_prompt1_outline,
    _norm_level,
    _clean_text,
    _narration_budget,
    _outline_schema,
)


def test_public_prompt_builder_signature_unchanged():
    """Verify that get_prompt1_outline retains its exact public signature and defaults,
    with backward-compatible support for topic_profile."""
    sig = inspect.signature(get_prompt1_outline)
    params = list(sig.parameters.keys())
    assert params == ["knowledge_point", "duration", "reference_image_path", "level", "analogy", "topic_profile"]
    assert sig.parameters["duration"].default is None
    assert sig.parameters["reference_image_path"].default is None
    assert sig.parameters["level"].default == "intermediate"
    assert sig.parameters["analogy"].default is None
    assert sig.parameters["topic_profile"].default is None


def test_no_analogy_encourages_autonomous_intuition_without_fictional_stories():
    """Verify when no analogy is requested, Stage 1 encourages autonomous intuition
    without commanding the model to fill fields with 'None' or forcing fictional stories."""
    prompt = get_prompt1_outline("Dijkstra's Algorithm", analogy=None)

    # Must encourage autonomous mental model / intuition
    assert "Autonomous Intuition & Mental Model Guidance" in prompt
    assert "Independently consider an appropriate mental model" in prompt
    assert "Do NOT invent a fictional story, character, or role-playing quest" in prompt
    assert '"mental_model"' in prompt
    assert '"breakdown_boundary"' in prompt

    # The obsolete command forbidding intuition must NOT be present
    assert 'Fill every story field in the JSON with the literal string "None"' not in prompt


def test_user_analogy_requires_suitability_evaluation():
    """Verify that when a user analogy is provided, Stage 1 mandates a suitability evaluation."""
    prompt = get_prompt1_outline("Dijkstra's Algorithm", analogy="Island courier")

    assert "Story-Driven Personalization & Analogy Guidance" in prompt
    assert "Analogy Suitability Assessment" in prompt
    assert "Accuracy Over Personalization" in prompt
    assert "Analogy Breakdown & Boundaries" in prompt
    assert "never a replacement for the formal definition" in prompt
    assert '"analogy_evaluation"' in prompt
    assert '"is_suitable"' in prompt
    assert '"adopted_analogy"' in prompt


def test_formal_definition_required_before_intuition():
    """Verify that Section 1 is explicitly required to introduce the formal definition before intuition."""
    prompt = get_prompt1_outline("Calculus Limits")

    assert "Pedagogical Progression (FORMAL FIRST)" in prompt
    assert "Section 1 MUST establish the textbook-accurate formal definition" in prompt
    assert "BEFORE introducing analogies or metaphors" in prompt
    assert "Ground the learner in technical reality first" in prompt


def test_fixed_six_section_instructions_removed_and_flexible_planning_supported():
    """Verify fixed-section constraints are removed and flexible section planning is supported."""
    prompt = get_prompt1_outline("Binary Search", duration=4.0)

    # Obsolete rigid constraint removed
    assert "Produce EXACTLY" not in prompt

    # Flexible planning instructions present
    assert "3 to 6 logical pedagogical sections" in prompt
    assert "Do NOT force every topic into a rigid six-section template" in prompt
    assert "The exact number, titles, and roles of sections must emerge naturally" in prompt


def test_visual_progression_instructions_included():
    """Verify visual progression types are defined and required in section design."""
    prompt = get_prompt1_outline("Reaction Kinetics")

    assert "Visual Progression Across Sections" in prompt
    assert "representation_introduction" in prompt
    assert "state_transformation" in prompt
    assert "relational_linkage" in prompt
    assert "case_comparison" in prompt
    assert "boundary_exposure" in prompt
    assert '"visual_progression_type"' in prompt
    assert "Avoid static visual repetition" in prompt


def test_technical_accuracy_and_safeguards_included():
    """Verify that critical technical and domain-specific safeguards are present in prompt instructions."""
    prompt = get_prompt1_outline("Dijkstra and Derivatives")

    # Invariants & conditions
    assert "Distinguish necessary conditions from sufficient conditions" in prompt
    assert "continuity is necessary but not sufficient" in prompt
    assert "greedy finalization invariant (extracting vertex u with minimal tentative distance" in prompt

    # Physical / chemistry distinctions on chemistry topic
    chem_prompt = get_prompt1_outline("Chemical Kinetics")
    assert "catalyst provides an alternative lower-activation-energy reaction pathway" in chem_prompt
    assert "NOT quantum mechanical tunnelling" in chem_prompt
    assert "Arrhenius pre-exponential factor A accounts for both collision frequency Z and steric orientation factor p" in chem_prompt
    assert "Maxwell-Boltzmann distribution" in chem_prompt

    # Worked example precomputation and verification
    assert "Numerical & Worked-Example Verification" in prompt
    assert "independently recompute" in prompt


def test_json_reliability_schema_and_prompt_instructions():
    """Verify that schema and prompt strictly mandate quoted JSON strings and forbid raw arithmetic."""
    prompt = get_prompt1_outline("Chemical Kinetics")

    # Prompt requirements
    assert "JSON Reliability: every 'worked_example_spec.steps[].result' and 'final_result' MUST be a quoted JSON string" in prompt
    assert "NEVER emit raw arithmetic expressions or unquoted operators as JSON values" in prompt
    assert "The entire output must remain 100% valid JSON" in prompt

    # Schema definition
    schema_str = _outline_schema("intermediate", None, _narration_budget(4.5, "intermediate"))
    schema = json.loads(schema_str)
    step_spec = schema["worked_example_spec"]["steps"][0]
    assert "ALWAYS a JSON string in quotes" in step_spec["result"]
    assert "NEVER emit raw unquoted arithmetic" in step_spec["result"]
    assert "ALWAYS a JSON string in quotes" in schema["worked_example_spec"]["final_result"]


def test_numerical_recomputation_and_units_consistency():
    """Verify prompt instructs independent recomputation of arithmetic, limits, and explicit units."""
    prompt = get_prompt1_outline("Reaction Kinetics", duration=4.5)

    assert "Independent Recomputation: independently recompute intermediate arithmetic" in prompt
    assert "exponential, logarithmic, or limit calculations" in prompt
    assert "Do NOT simply repeat a result stated earlier in your reasoning without recomputing it" in prompt
    assert "Units & Dimensional Consistency: state explicit units" in prompt
    assert "If a numerical result cannot be reliably verified with exact precision, avoid presenting it as an exact verified result" in prompt


def test_calculus_terminology_and_local_differentiability():
    """Verify calculus rules: differentiability implies continuity, continuity is not sufficient,
    differentiability is local, and counterexample f(x) = |x| at x = 0 is available without being forced."""
    prompt = get_prompt1_outline("Derivatives in Calculus")

    assert "differentiability at a point implies continuity at that point" in prompt
    assert "continuity at a point does NOT guarantee differentiability" in prompt
    assert "continuity is necessary but not sufficient" in prompt
    assert "differentiability is a local property" in prompt
    assert "A function need not be globally continuous to be differentiable at a particular point" in prompt
    assert "f(x) = |x| at x = 0" in prompt
    assert "do NOT force this counterexample into every calculus lesson" in prompt


def test_rate_of_change_velocity_vs_speed_analogy_guidance():
    """Verify moving-car analogy distinguishes signed velocity from non-negative speed,
    maps position derivative to velocity, and is not forced into all derivative lessons."""
    prompt = get_prompt1_outline("Derivatives in Calculus", analogy="moving car")

    assert "distinguish signed velocity from non-negative speed" in prompt
    assert "Map the derivative of position with respect to time strictly to instantaneous velocity" in prompt
    assert "speed is the magnitude of velocity: speed = |velocity|" in prompt
    assert "Do NOT hardcode this analogy into all derivative lessons" in prompt


def test_visual_progression_pedagogical_selection_and_reuse_permitted():
    """Verify visual progression types are grounded in pedagogical intent and reuse is explicitly allowed."""
    prompt = get_prompt1_outline("Graph Algorithms")

    assert "Select visual progression types based on what each individual section actually teaches" in prompt
    assert "representation_introduction: Introduce a representation when establishing a new visual model" in prompt
    assert "state_transformation: Transform existing visual state when demonstrating a mechanism" in prompt
    assert "relational_linkage: Link representations when connecting equations, graphs, tables" in prompt
    assert "case_comparison: Compare cases when contrasting alternative situations" in prompt
    assert "boundary_exposure: Expose boundaries when demonstrating limitations" in prompt
    assert "Reusing a progression type across consecutive sections" in prompt
    assert "fully acceptable when justified by the teaching objective" in prompt
    assert "Avoid merely assigning labels mechanically; do NOT force every section to use a different type" in prompt


def test_schema_contains_required_conditional_and_backward_compatible_fields():
    """Verify schema contains all Step 2A.1 fields while remaining compatible with TeachingOutline."""
    schema_str = _outline_schema("intermediate", "water flow", _narration_budget(6.5, "intermediate"))
    schema = json.loads(schema_str)

    # Core required
    assert "topic" in schema
    assert "topic_type" in schema
    assert "problem_statement" in schema
    assert "formal_definition" in schema
    assert "notation" in schema
    assert "narration_budget" in schema
    assert "sections" in schema

    # Conditional / pedagogical reasoning
    assert "governing_principles" in schema
    assert "assumptions_and_bounds" in schema
    assert "mental_model" in schema
    assert "analogy_evaluation" in schema
    assert "worked_example_spec" in schema
    assert "common_misconceptions" in schema
    assert "pedagogical_conclusion" in schema

    # Backward compatibility with existing TeachingOutline fields
    assert "analogy_used" in schema
    assert "analogy_breakdown" in schema
    assert "story_framework" in schema

    # Per-section attributes
    sec = schema["sections"][0]
    assert "id" in sec
    assert "title" in sec
    assert "pedagogical_role" in sec
    assert "goal" in sec
    assert "visual_idea" in sec
    assert "visual_progression_type" in sec
    assert "worked_example_step" in sec
    assert "misconception_addressed" in sec


def test_supported_input_combinations():
    """Verify get_prompt1_outline executes cleanly across all valid argument combinations."""
    levels = ["beginner", "intermediate", "advanced", None]
    durations = [None, 3.0, 6.5, 10.0]
    analogies = [None, "water pipes", ""]
    ref_images = [None, "/path/to/diagram.png"]

    # Test representative permutations
    for lvl in levels:
        p = get_prompt1_outline("Test Topic", level=lvl)
        assert isinstance(p, str)
        assert len(p) > 500

    for dur in durations:
        p = get_prompt1_outline("Test Topic", duration=dur)
        assert isinstance(p, str)
        assert len(p) > 500

    for a in analogies:
        p = get_prompt1_outline("Test Topic", analogy=a)
        assert isinstance(p, str)
        assert len(p) > 500

    p_img = get_prompt1_outline("Test Topic", reference_image_path="/path/to/diagram.png")
    assert "Reference Image Available" in p_img


def test_os_cpu_scheduling_stage1_guidance():
    """Verify that Stage 1 prompt includes comprehensive OS CPU scheduling guidance,
    covering process representation (PCB), 4-state lifecycle, FIFO ready queue,
    dispatcher distinctions, non-preemptive execution invariants, execution-connected
    metrics, and implementation-oriented teaching quality rules."""
    prompt = get_prompt1_outline("FCFS CPU Scheduling")

    # 1. OS Scheduling & Process Representation (PCB)
    assert "operating systems and CPU scheduling" in prompt
    assert "Process Control Block (PCB)" in prompt
    assert "PID, arrival time, CPU burst time, current process state" in prompt
    assert "distinguish illustrative teaching representations from implementation-specific kernel structures" in prompt

    # 2. 4-State Lifecycle
    assert "NEW -> READY -> RUNNING -> TERMINATED" in prompt
    assert "exact trigger conditions that cause each state transition" in prompt

    # 3. Ready Queue FIFO & Dispatcher
    assert "Ready Queue FIFO dynamics" in prompt
    assert "enqueueing at the queue tail" in prompt
    assert "scheduler selection of the earliest eligible process from the queue head" in prompt
    assert "Distinguish scheduling (selecting which process runs next) from dispatching" in prompt

    # 4. Non-preemptive Execution Invariant
    assert "non-preemptive execution invariant" in prompt
    assert "NEVER interrupt or preempt the executing process" in prompt

    # 5. Scheduling Metrics Connected to Execution
    assert "Completion Time (CT)" in prompt
    assert "Turnaround Time (TAT = CT - AT)" in prompt
    assert "Waiting Time (WT = TAT - BT)" in prompt
    assert "Response Time (RT)" in prompt

    # 6. Implementation-Oriented Teaching Quality
    assert "Prioritize implementation-oriented execution over raw definitions and formulas" in prompt
    assert "What data the OS maintains" in prompt
    assert "What operation happens" in prompt
    assert "Which component performs it" in prompt
    assert "How the process state changes" in prompt
    assert "What the resulting execution state looks like" in prompt

    # 7. Staggered Arrival Worked Example & Simulator Truth
    assert "P1: AT=0, BT=3; P2: AT=1, BT=4; P3: AT=2, BT=2" in prompt
    assert "deterministic simulator and its trace remain the authoritative source of truth" in prompt

    # 8. Non-OS Topics Remain Unaffected
    non_os_prompt = get_prompt1_outline("Dijkstra's Shortest Path Algorithm")
    assert "greedy finalization invariant" in non_os_prompt
    assert "non-negative edge weights" in non_os_prompt

