import json
from typing import Any, Dict, List, Optional, Union

try:
    from src.pedagogy import (
        TopicProfile,
        TopicProfiler,
        StateEvolution,
        FormalismLevel,
        ComputationRequirement,
        PrimaryVisualMode,
        ProfileStatus,
    )
except ImportError:
    try:
        from pedagogy import (
            TopicProfile,
            TopicProfiler,
            StateEvolution,
            FormalismLevel,
            ComputationRequirement,
            PrimaryVisualMode,
            ProfileStatus,
        )
    except ImportError:
        TopicProfile = None
        TopicProfiler = None

# Average TTS narration pace including natural pauses. Used to turn the requested
# duration into a concrete word / line budget so that stage 2 does not have to guess.
SPEECH_WORDS_PER_MINUTE = 135


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _norm_level(level):
    norm = (level or "intermediate").lower()
    return norm if norm in ("beginner", "intermediate", "advanced") else "intermediate"


def _clean_text(value):
    """Single-line, stripped string without double quotes.

    User-supplied text (topic / analogy) is later pasted into prompts AND into generated
    Python string literals, so stray newlines or double quotes would break the code.
    """
    if value is None:
        return ""
    return " ".join(str(value).split()).replace('"', "'")


def _narration_budget(duration, norm_level):
    """Translate the requested duration into sections / lines / words."""
    try:
        dur_val = float(duration) if duration else 0.0
    except (TypeError, ValueError):
        dur_val = 0.0

    user_specified = dur_val > 0
    if user_specified:
        target = dur_val
    else:
        target = 6.5 if norm_level in ("intermediate", "advanced") else 4.5

    sections = max(4, min(7, round(target / 1.1)))
    total_words = int(target * SPEECH_WORDS_PER_MINUTE)
    per_section = total_words / sections
    lines = max(4, min(8, round(per_section / 28)))
    words_per_line = per_section / lines
    lo = max(15, int(round(words_per_line * 0.8)))
    hi = min(45, int(round(words_per_line * 1.2)))

    return {
        "target_minutes": target,
        "user_specified": user_specified,
        "sections": sections,
        "lines_per_section": lines,
        "words_per_line": [lo, hi],
        "total_words": total_words,
    }


# ----------------------------------------------------------------------------
# Level guidance
# ----------------------------------------------------------------------------
LEVEL_GUIDANCE = {
    "beginner": r"""
## Target Learner Level: BEGINNER
- Audience: introductory learners, high-school students, or newcomers with minimal background.
- Prerequisites: introduce every foundational prerequisite explicitly in Section 1 before core concepts.
- Explanation depth: prioritize intuitive understanding and visual meaning over jargon. Scaffold ideas gradually.
- Mathematical rigor: keep notation accessible and define every symbol. Ground concepts in pictures and simple arithmetic BEFORE formulas.
- Worked example: tangible, concrete, step-by-step, with small whole numbers.
- Visual complexity & pacing: clean, uncluttered visuals, simple shapes, high-contrast labels, unhurried pacing.
""",
    "intermediate": r"""
## Target Learner Level: INTERMEDIATE
- Audience: undergraduate students, AP students, or technical practitioners.
- Prerequisites: state key prerequisites briefly in Section 1, then move to the formal formulation.
- Explanation depth: balance intuition and formalism; connect each picture to its algebraic / formal counterpart.
- Mathematical rigor: standard formal definitions, explicit formulas, the standard notation of the field, exact calculations.
- Worked example: a canonical problem solved step by step with full numerical or algebraic detail.
- Visual complexity & pacing: structured diagrams and standard coordinate frames; steady, engaging pacing.
""",
    "advanced": r"""
## Target Learner Level: ADVANCED
- Audience: advanced undergraduate / graduate students, researchers, or domain specialists.
- Prerequisites: assume solid foundations; summarize assumptions in one or two sentences and move quickly to subtleties.
- Explanation depth: structural properties, invariants, general cases, edge cases, complexity or analytical implications.
- Mathematical rigor: precise definitions, short derivations or proof sketches, explicit boundary conditions.
- Worked example: a non-trivial case, an edge case, or a counterexample that exposes a subtlety.
- Visual complexity & pacing: information-dense but still readable; multi-step transformations; brisk pacing.
""",
}


# ----------------------------------------------------------------------------
# Analogy / direct explanation guidance (topic-agnostic: no hard-coded subject)
# ----------------------------------------------------------------------------
def _analogy_guidance(knowledge_point, analogy_text, norm_level):
    return rf"""
## Story-Driven Personalization & Analogy Guidance
Requested analogy / personal interest: {analogy_text}

### Build a Coherent Story Around the Interest
Weave the theme of '{analogy_text}' naturally throughout the section progression:
- Phase 1 (Introduction): Formal definition and purpose first; bridge with the theme.
- Phase 2 (Concept Introduction): Ground key entities using the analogy.
- Phase 3 (Step-by-Step Explanation): Trace the mechanism step by step.
- Phase 4 (Worked Example): Execute a concrete worked example with exact numbers.
- Phase 5 (Deeper Explanation & Boundaries): Explain deeper properties and formal boundaries.
- Phase 6 (Conclusion / Resolution): Meaningful conclusion and real-world implications.

MANDATORY RULES:
1. Analogy Suitability Assessment (do this FIRST): decide whether '{analogy_text}' can faithfully carry the core ideas of '{knowledge_point}'.
   - If it can: adopt it, possibly narrowed to the part that maps well.
   - If it only fits part of the topic: adopt it for that part only and use direct explanation for the rest.
   - If it cannot: set "is_suitable": false, choose the closest FAITHFUL alternative (or "None - Direct Explanation"), and explain why in "rationale". Never keep a misleading analogy just to please the request.
2. Accuracy Over Personalization. The analogy is a teaching aid, never a replacement for the formal definition. Every mapping must be conceptually sound.
3. Build explicit entity mappings (story entity -> technical concept / symbol), one-to-one. Each mapping must also state where it is imprecise.
4. Order of ideas: Section 1 gives the formal definition and purpose FIRST before any analogy; the analogy is then introduced as an intuitive bridge.
5. Story progression across the section arc: premise -> structure/mappings -> mechanism -> worked example with exact numbers -> formal rules plus WHERE THE ANALOGY BREAKS DOWN -> meaningful resolution.
6. Analogy Breakdown & Boundaries: 'analogy_breakdown' must name at least one concrete place where the metaphor would give a WRONG answer and what the formal rule says instead.
7. Adapt depth to the {norm_level} level: keep definitions rigorous and numbers exact. Never invent fictional characters or narrative quests.
"""


def _direct_guidance(knowledge_point):
    return rf"""
## Autonomous Intuition & Mental Model Guidance (No Analogy Requested)
No analogy was requested by the user. As an expert instructional designer:
1. Formal Understanding First: The lesson begins with the standard formal definition, problem context, and notation in Section 1 before introducing intuition.
2. Autonomous Intuition Selection: Independently consider an appropriate mental model or intuitive aid:
   - Physical metaphor (e.g. uniform expanding wavefront, potential energy hill, spring-mass damper).
   - Geometric picture (e.g. local linearity / zooming in, rotating vectors, area accumulation).
   - Concrete real-world scenario (e.g. network packet delivery, buffer queues, vehicle motion with signed velocity vs speed).
3. Select an Intuitive Model ONLY When It Genuinely Helps:
   - If a mental model clarifies the concept, describe it in 'mental_model' with clear entity mappings to formal symbols.
   - If no analogy or metaphor is suitable (e.g. for purely axiomatic, syntactic, or self-evident concepts), choose a direct visual and mathematical explanation. Do NOT invent a fictional story, character, or role-playing quest merely to satisfy a prompt requirement.
4. State Explicit Boundaries: Identify where the mental model diverges from exact technical reality in 'mental_model.breakdown_boundary'.
"""


def _arc_guidance(knowledge_point, n):
    return rf"""
## Flexible Pedagogical Progression
Design a coherent, scaffolded lesson with 3 to 6 logical pedagogical sections (recommended ~{n} sections for this duration). The exact number, titles, and roles of sections must emerge naturally from the conceptual milestones of '{knowledge_point}'. Do NOT force a rigid six-section template.

### Internal Reasoning Order vs Learner-Facing Order
1. Internal Reasoning Order (How you plan the lesson):
   - First establish the formal problem, standard definition, governing principles/laws, and notation symbols.
   - Identify prerequisite knowledge, boundary constraints, and common student misconceptions.
   - Precompute a concrete worked example with verified numbers or symbolic steps.
   - Select or audit the intuitive mental model and locate its breakdown boundary.
   - Assemble the learner-facing sections based on dependencies.

2. Learner-Facing Order (What the student sees on screen):
   - Section 1 MUST introduce the textbook-accurate formal definition, purpose, and problem being solved BEFORE introducing analogies or metaphors. Ground the learner in technical reality first.
   - Subsequent sections progressively unfold:
     * Intuitive interpretation and visual mental model (grounding the formal definition).
     * Governing principles, formal mechanism, or structural transformations.
     * Concrete worked example or numerical demonstration (step-by-step trace).
     * Relevant boundaries, edge cases, or misconception corrections (where naive intuition breaks down).
     * Synthesis and meaningful conclusion (complexity trade-offs, fundamental theorem, or practical consequence).

### Visual Progression Across Sections
Select visual progression types based on what each individual section actually teaches:
- representation_introduction: Introduce a representation when establishing a new visual model, spatial layout, or coordinate framework (e.g. Cartesian axes, initial graph network, molecule).
- state_transformation: Transform existing visual state when demonstrating a mechanism, tracing an algorithm, or updating elements (e.g. recoloring a visited node, updating a table row, advancing a reaction coordinate). Retains visual continuity without redrawing static diagrams.
- relational_linkage: Link representations when connecting equations, graphs, tables, or dual viewports (e.g. graph in dock_main and priority queue in dock_inspector).
- case_comparison: Compare cases when contrasting alternative situations (e.g. catalyzed vs uncatalyzed, valid vs negative edge failure).
- boundary_exposure: Expose boundaries when demonstrating limitations, failure cases, corner points, or counterexamples (e.g. f(x) = |x| at x = 0).
Note: Reusing a progression type across consecutive sections (e.g. multiple state transformations during a multi-step algorithmic trace or worked example) is fully acceptable when justified by the teaching objective. Avoid merely assigning labels mechanically; do NOT force every section to use a different type.
"""


# ----------------------------------------------------------------------------
# JSON schema shown to the model (built with json.dumps: no brace-escaping bugs)
# ----------------------------------------------------------------------------
def _outline_schema(norm_level, analogy_text, budget):
    has_analogy = bool(analogy_text)
    lo, hi = budget["words_per_line"]
    target_sec = budget["sections"]

    schema = {
        "topic": "Topic name",
        "topic_type": "computational | mathematical | natural_science | engineered_system | conceptual_logic",
        "topic_profile": {
            "state_evolution": "discrete_state_machine | continuous_dynamic | multi_stage_process | static_structural | conceptual_axiomatic | uncertain",
            "formalism_level": "rigorous_mathematical | mechanistic_empirical | system_structural | qualitative_intuitive | uncertain",
            "computation_requirement": "deterministic_trace | arithmetic_calculation | symbolic_identity | none_not_applicable | uncertain",
            "primary_visual_mode": "network_graph | coordinate_geometry | state_table_matrix | molecular_spatial | process_flowchart | structured_concept_card | uncertain",
            "rationale_and_evidence": "Concrete technical topic characteristics justifying this profile",
            "profile_status": "CLASSIFIED",
            "confidence": 0.95,
            "secondary_visual_modes": ["optional_secondary_visual_mode"],
        },
        "target_audience": f"Audience matching the {norm_level} level",
        "learning_level": norm_level,
        "estimated_duration_minutes": round(budget["target_minutes"], 1),
        "narration_budget": {
            "target_minutes": round(budget["target_minutes"], 1),
            "sections": target_sec,
            "recommended_sections": target_sec,
            "lines_per_section": budget["lines_per_section"],
            "words_per_line": [lo, hi],
            "total_words": budget["total_words"],
        },
        "problem_statement": "The concrete problem, question, or physical phenomenon being addressed",
        "formal_definition": "Textbook-accurate, standard formal definition in 1-3 sentences",
        "governing_principles": ["Core invariant, law, theorem, or governing equation (domain-appropriate)"],
        "prerequisites": ["Prerequisite concept 1", "Prerequisite concept 2"],
        "assumptions_and_bounds": ["Domain assumption, boundary condition, or validity constraint"],
        "notation": {"symbol": "meaning - exact symbols locked across all later stages"},
        "mental_model": {
            "model_type": "physical_metaphor | geometric_model | concrete_scenario | direct_visual",
            "description": "Clear conceptual grounding without fictional characters or narrative fluff",
            "entity_mappings": {"intuitive_feature": "formal_symbol"},
            "breakdown_boundary": "Where this intuition stops and exact formalism begins",
        },
        "analogy_evaluation": {
            "requested_analogy": analogy_text if has_analogy else "None",
            "is_suitable": "true | false" if has_analogy else None,
            "adopted_analogy": "Description of adopted analogy or mental model",
            "rationale": "Why accepted, narrowed, or replaced (or rationale for autonomous model)",
            "limitations": "Where the metaphor fails to represent the technical reality",
        },
        "analogy_used": "Short label of adopted analogy or mental model",
        "analogy_breakdown": "Concrete case where the metaphor would give a wrong answer",
        "story_framework": {
            "premise": "Context, setting or situation built from the interest/analogy (or 'None')",
            "protagonist_goal": "Objective or problem to solve in the story (or 'None')",
            "entity_mappings": {"story_entity_1": "technical concept / symbol"},
            "story_progression": "How understanding advances across sections from problem to resolution",
        },
        "common_misconceptions": [
            {"misconception": "Common wrong belief", "correction": "Accurate formal statement"}
        ],
        "worked_example_spec": {
            "setup": "Concrete problem instance with small clean numbers",
            "inputs": {"name": "value"},
            "steps": [
                {
                    "step": "Action taken or step description",
                    "step_type": "Optional: calculation | state_transition | sequence_operation | logical_proof | conceptual_explanation",
                    "operation": "Optional operation performed (e.g. evaluate, assign, compare, swap, update)",
                    "expression": "Optional mathematical or logical expression (e.g. '(low + high) // 2' or '-Ea / (R * T)')",
                    "state_before": "Optional dictionary of relevant state variables before this step",
                    "result": "Exact intermediate state/value or assignment as a string (ALWAYS a JSON string in quotes, e.g. 'y = 4' or 'mid = 4' or 'k = 0.237 s^-1'; NEVER emit raw unquoted arithmetic)",
                    "state_after": "Optional dictionary of relevant state variables after this step",
                    "explanation": "Optional concise rationale for this step",
                }
            ],
            "final_result": "Exact final answer as a string (ALWAYS a JSON string in quotes, e.g. 'dist = 7' or 'k = 0.237 s^-1')",
            "verification": "Independent check (second computation, alternative method, or invariant check)",
        },
        "conceptual_breakdown": {
            "mechanism_stages": [
                {
                    "stage_name": "Stage name (e.g. 'Photolysis & Photosystem II')",
                    "action": "Biochemical, mechanistic, or deductive action description",
                    "entities": ["P680", "H2O", "O2", "electron_transport_chain"],
                    "output_state": "Proton gradient across thylakoid membrane",
                }
            ]
        },
        "proof_spec": {
            "hypotheses": ["Given premise or geometric construct"],
            "deductive_steps": ["Step 1", "Step 2"],
            "conclusion_invariant": "Proven invariant identity",
        },
        "pedagogical_conclusion": {
            "conclusion_type": "tradeoff_analysis | fundamental_theorem | practical_application | boundary_synthesis",
            "core_takeaway": "The lasting conceptual takeaway",
            "implications": "Real-world significance, trade-offs, or forward connections",
        },
        "sections": [
            {
                "id": "section_1",
                "title": "Clear, informative section title (not raw beat codes)",
                "pedagogical_role": "formal_definition_and_purpose | intuitive_interpretation | core_mechanism | worked_demonstration | boundary_and_misconception | synthesis_and_conclusion",
                "goal": "After this section the learner can ...",
                "story_step": "Contextual event or bridge illustrating the concept (or 'None')",
                "content": "What this section explains and in what order",
                "key_points": ["precise technical takeaway 1", "precise technical takeaway 2"],
                "worked_example_step": "Which step of worked_example_spec is active (or 'None')",
                "misconception_addressed": "A misconception from the list above (or 'None')",
                "visual_idea": "Concrete visual representation (shapes, labels, coordinates, equations) - must fit ~4 visual units in 5.8 x 3.6 area",
                "visual_progression_type": "representation_introduction | state_transformation | relational_linkage | case_comparison | boundary_exposure",
                "beats": [
                    {
                        "beat_id": "section_1_beat_1",
                        "pedagogical_objective": "Concrete measurable learning objective for this beat",
                        "teaching_move": "intuition_bridge | formal_definition | mechanistic_trace | counterexample_exposure | boundary_clarification | case_comparison | relational_synthesis | conceptual_breakdown | proof_derivation | worked_demonstration",
                        "concept_addressed": "Specific sub-concept or invariant being taught",
                        "visual_intent": "Intended visual focus and layout arrangement",
                        "required_state_changes": [
                            {"entity": "object_id", "operation": "highlight | update | move | transform", "detail": "transition description"}
                        ],
                        "verification_references": ["eq_1", "step_1", "theorem_invariant"],
                    }
                ],
            }
        ],
    }
    return json.dumps(schema, indent=4, ensure_ascii=False)


def _format_stage1_capabilities(manifest):
    if not manifest:
        return ""
    lines = ["## Available Domain Visual Capabilities & Components"]
    lines.append("The CineLearn pipeline provides registered visual components that you may reference in 'visual_idea' and beat 'visual_intent':")
    for domain, comps in manifest.items():
        comp_strs = []
        for c in comps:
            cid = c.get("component_id")
            desc = c.get("description", "").split("\n")[0].strip()
            if cid:
                comp_strs.append(f"{cid} ({desc})")
        if comp_strs:
            lines.append(f"- **Domain `{domain}`**:\n  * " + "\n  * ".join(comp_strs))
    lines.append(
        "\nPedagogical Flexibility: You have full instructional design freedom to decide the teaching progression, "
        "learning objectives, and section structure based on the topic. Do NOT assume a fixed order of scenes or rigid beat sequence. "
        "Select the visual representations that best support each section's pedagogical goal."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Dynamic Profile Resolution & Strategy Formatting
# ----------------------------------------------------------------------------
def _resolve_topic_profile(knowledge_point, topic_profile=None):
    """Resolves a typed TopicProfile using explicit input or TopicProfiler."""
    kp = _clean_text(knowledge_point)
    if topic_profile is not None:
        if isinstance(topic_profile, dict):
            if TopicProfile and TopicProfiler:
                return TopicProfiler.profile_from_data(topic_profile, context={"topic": kp})
            return topic_profile
        return topic_profile

    if TopicProfiler:
        prof = TopicProfiler.profile_heuristic_fallback(kp)
        if prof.is_unclassified():
            kp_low = kp.lower()
            if any(term in kp_low for term in ["fcfs", "cpu scheduling", "scheduling", "process management"]):
                prof = TopicProfiler.profile_heuristic_fallback(kp, description="operating system")
            elif any(term in kp_low for term in ["chemical kinetics", "kinetics", "chemical reaction"]):
                prof = TopicProfiler.profile_heuristic_fallback(kp, description="reaction kinetics")
        return prof
    return None


def _format_stage1_profile_guidance(profile, kp):
    """Dynamically formats pedagogical strategy and domain guidance from TopicProfile."""
    if not profile:
        return ""

    if hasattr(profile, "to_dict"):
        p_dict = profile.to_dict()
    elif isinstance(profile, dict):
        p_dict = profile
    else:
        return ""

    state_evo = str(p_dict.get("state_evolution", "")).lower()
    form_level = str(p_dict.get("formalism_level", "")).lower()
    comp_req = str(p_dict.get("computation_requirement", "")).lower()
    vis_mode = str(p_dict.get("primary_visual_mode", "")).lower()
    sec_chars = set(p_dict.get("secondary_characteristics", []))
    rationale = str(p_dict.get("rationale_and_evidence", "")).strip()
    status = str(p_dict.get("profile_status", "CLASSIFIED")).upper()
    try:
        conf = float(p_dict.get("confidence", 1.0))
    except (ValueError, TypeError):
        conf = 0.0
    kp_low = kp.lower()

    lines = ["## Dynamic Pedagogical Strategy (Topic Profile Guidance)"]
    lines.append(f"Profile Status: {status} (Confidence: {conf:.2f})")
    if rationale:
        lines.append(f"- Topic Characteristics & Evidence: {rationale}")

    # 1. State Evolution Guidance
    if state_evo == "discrete_state_machine":
        lines.append(
            "- State Evolution (Discrete State Machine): Structure instructional progression around discrete state machine transitions. "
            "Ensure every transition explicitly identifies the pre-state, the triggering condition or event, and the resulting post-state."
        )
    elif state_evo == "continuous_dynamic":
        lines.append(
            "- State Evolution (Continuous Dynamic): Structure instructional progression around continuous dynamic behavior. "
            "Emphasize continuous rates of change, parameter variation, and smooth curve transitions."
        )
    elif state_evo == "multi_stage_process":
        lines.append(
            "- State Evolution (Multi-Stage Process): Structure instructional progression around multi-stage causal processes and pathways. "
            "Trace input/output transformations across stages without forcing artificial arithmetic."
        )
    elif state_evo in ("static_structural", "conceptual_axiomatic"):
        lines.append(
            "- State Evolution (Static Structural / Axiomatic): Structure instructional progression around structural decomposition, "
            "geometric constructions, or deductive proof transformations."
        )
    else:
        lines.append(
            "- State Evolution (General Progression): Structure instructional progression with clear scaffolding from formal definition "
            "to intuitive mental model to worked demonstration to boundary conditions."
        )

    # 2. Formalism Level Guidance
    if form_level == "rigorous_mathematical":
        lines.append(
            "- Formalism Level (Rigorous Mathematical): Maintain rigorous mathematical formalism. Use textbook-accurate definitions, "
            "standard symbolic notation, and exact formulas. Ground every claim in formal mathematical identity."
        )
    elif form_level == "mechanistic_empirical":
        lines.append(
            "- Formalism Level (Mechanistic Empirical): Ground explanations in mechanistic empirical reality. Connect observations to "
            "underlying mechanisms and governing physical/chemical laws."
        )
    elif form_level == "system_structural":
        lines.append(
            "- Formalism Level (System Structural): Focus on system-level structural architecture. Emphasize component roles, "
            "state-holding entities, and execution interactions."
        )
    elif form_level == "qualitative_intuitive":
        lines.append(
            "- Formalism Level (Qualitative Intuitive): Prioritize conceptual clarity and intuitive meaning while preserving technical accuracy."
        )
    else:
        lines.append(
            "- Formalism Level (Balanced Scaffolding): Balance accessible conceptual scaffolding with textbook-accurate formal terminology."
        )

    # 3. Computation & Grounding Guidance
    if comp_req == "deterministic_trace":
        lines.append(
            "- Computation Requirement (Deterministic Trace): Provide a complete, step-by-step 'worked_example_spec' tracing intermediate state variables. "
            "Every step must quote intermediate and final values as JSON strings. Ensure chronological fidelity without skipping intermediate states."
        )
    elif comp_req == "arithmetic_calculation":
        lines.append(
            "- Computation Requirement (Arithmetic Calculation): Provide verified numerical calculations in 'worked_example_spec' with explicit units, "
            "small clean numbers, and dimensional consistency. Recompute all results independently."
        )
    elif comp_req == "symbolic_identity":
        lines.append(
            "- Computation Requirement (Symbolic Identity / Proof): Provide a structured deductive derivation or proof in 'proof_spec' "
            "(hypotheses, deductive steps, invariant conclusion) or symbolic 'worked_example_spec'."
        )
    elif comp_req == "none_not_applicable":
        lines.append(
            "- Computation Requirement (None - Qualitative / Mechanistic): This topic is non-computational or mechanistic/qualitative. "
            "Do NOT force arithmetic calculations, numerical formulas, or fabricated variables into the worked example. Provide a 'conceptual_breakdown' instead."
        )
    else:
        lines.append(
            "- Computation Requirement (Concrete Example Grounding): Ground the lesson in a concrete, verified worked example or tangible demonstration appropriate to the subject."
        )

    # 4. Primary Visual Mode Guidance
    if vis_mode == "network_graph":
        lines.append(
            "- Recommended Primary Visual Mode: Network Graph. Use nodes, edges, traversals, and path highlights."
        )
    elif vis_mode == "coordinate_geometry":
        lines.append(
            "- Recommended Primary Visual Mode: Coordinate Geometry / Spatial Diagrams. Use Cartesian axes, curves, geometric figures, or spatial pointer intervals."
        )
    elif vis_mode == "state_table_matrix":
        lines.append(
            "- Recommended Primary Visual Mode: State Table / Matrix. Use structured tabular or matrix layouts to inspect multi-variable states or metric records."
        )
    elif vis_mode == "molecular_spatial":
        lines.append(
            "- Recommended Primary Visual Mode: Molecular / Spatial Representation. Use 2D molecular structures, particle arrangements, or spatial configurations."
        )
    elif vis_mode == "process_flowchart":
        lines.append(
            "- Recommended Primary Visual Mode: Process Flowchart / Pipeline. Use directed block diagrams, pipeline stages, or queue flow representations."
        )
    elif vis_mode == "structured_concept_card":
        lines.append(
            "- Recommended Primary Visual Mode: Structured Concept Cards. Organize information into clear visual cards, panels, or formula containers."
        )
    else:
        lines.append(
            "- Recommended Visual Mode: Adaptive Native Visuals. Select clear native Manim diagrams, shapes, and structured cards best suited to the topic."
        )

    # 5. Dynamic Domain-Specific Guidance (injected ONLY when indicated by profile or topic context)
    is_os_scheduling = (
        any(k in sec_chars for k in ["cpu_scheduling", "interleaving_threads", "mutual_exclusion"])
        or any(k in kp_low for k in ["fcfs", "cpu scheduling", "process management", "operating system"])
    )
    is_graph_algo = (
        any(k in sec_chars for k in ["greedy_relaxation", "priority_queue"])
        or any(k in kp_low for k in ["dijkstra", "shortest path"])
    )
    is_calculus = (
        any(k in sec_chars for k in ["limiting_process", "local_tangent"])
        or any(k in kp_low for k in ["derivative", "calculus", "rate of change"])
    )
    is_chem_kinetics = (
        any(k in sec_chars for k in ["exponential_scaling", "potential_energy_surface"])
        or any(k in kp_low for k in ["arrhenius", "activation energy", "reaction kinetics"])
    )
    is_search_algo = (
        any(k in sec_chars for k in ["pointer_convergence", "logarithmic_reduction"])
        or "binary search" in kp_low
    )
    is_geometric_proof = (
        any(k in sec_chars for k in ["geometric_dissection", "algebraic_invariance"])
        or any(k in kp_low for k in ["pythagorean", "theorem", "proof"])
    )
    is_biochemical = (
        any(k in sec_chars for k in ["cyclic_pathway", "membrane_transport"])
        or any(k in kp_low for k in ["photosynthesis", "calvin cycle", "cellular respiration"])
    )

    if is_os_scheduling:
        lines.append(
            """### Domain-Specific Guidance: Operating Systems & CPU Scheduling
- In operating systems and CPU scheduling (e.g. FCFS, CPU scheduling, process management):
  * Teaching Sequence Guidance (Logical, Scaffolding Progression):
    Ensure the generated outline establishes these 12 core concepts in natural pedagogical order:
  1. Why CPU scheduling is needed (multiprogramming context, maximizing CPU utilization across CPU and I/O bursts).
  2. What a process and PCB represent: Process Control Block (PCB) with PID, arrival time, CPU burst time, current process state; distinguish illustrative teaching representations from implementation-specific kernel structures.
  3. What process arrival and burst times mean: CRITICAL REQUIREMENT - an explicit section/beat MUST introduce the problem input specification table before any CPU dispatch or timeline execution occurs:
     * P1: AT=0, BT=3; P2: AT=1, BT=4; P3: AT=2, BT=2
     This input specification is the authoritative source of truth for every subsequent execution interval and metric, where the deterministic simulator and its trace remain the authoritative source of truth.
  4. How the Ready Queue works: 4-State Lifecycle NEW -> READY -> RUNNING -> TERMINATED and exact trigger conditions that cause each state transition; Ready Queue FIFO dynamics, enqueueing at the queue tail, and scheduler selection of the earliest eligible process from the queue head.
  5. What the scheduler and dispatcher do: Distinguish scheduling (selecting which process runs next) from dispatching (allocating CPU, context switch).
  6. How FCFS chooses the next process (First-Come, First-Served: process with earliest arrival time runs first; tie-breaking via stable arrival order).
  7. Why execution is non-preemptive: non-preemptive execution invariant: once dispatched, NEVER interrupt or preempt the executing process until burst finishes.
  8. How each process executes in chronological order (P1 executes from t=0 to 3, then P2 executes from t=3 to 7, then P3 executes from t=7 to 9).
  9. How the Gantt chart represents the execution timeline (horizontal timeline tracking start time, elapsed duration, and completion time for each process block).
  10. How CT, TAT, WT, and RT are derived with explicit mathematical formulas:
      * Completion Time (CT): simulation time when burst finishes (P1=3, P2=7, P3=9).
      * Turnaround Time (TAT = CT - AT): P1: 3 - 0 = 3, P2: 7 - 1 = 6, P3: 9 - 2 = 7.
      * Waiting Time (WT = TAT - BT): P1: 3 - 3 = 0, P2: 6 - 4 = 2, P3: 7 - 2 = 5.
      * Response Time (RT): First Run Time - AT (in non-preemptive FCFS, RT == WT: P1=0, P2=2, P3=5).
  11. How summary averages are calculated with exact arithmetic:
      * Average CT = (3 + 7 + 9) / 3 = 19/3 = 6.33
      * Average TAT = (3 + 6 + 7) / 3 = 16/3 = 5.33
      * Average WT = (0 + 2 + 5) / 3 = 7/3 = 2.33
      * Average RT = (0 + 2 + 5) / 3 = 7/3 = 2.33
  12. FCFS limitations and the Convoy Effect: a long CPU-bound process arriving first blocks short interactive tasks, inflating average waiting time and stalling I/O; contrasting with preemptive scheduling (e.g. Round Robin).
- Implementation-Oriented Teaching Quality:
  Prioritize implementation-oriented execution over raw definitions and formulas. At each step show:
  * What data the OS maintains
  * What operation happens
  * Which component performs it
  * How the process state changes
  * What the resulting execution state looks like
- In 'worked_example_spec', strictly ground the simulation on the canonical processes:
  `inputs`: {"processes": [{"pid": "P1", "arrival_time": 0, "burst_time": 3}, {"pid": "P2", "arrival_time": 1, "burst_time": 4}, {"pid": "P3", "arrival_time": 2, "burst_time": 2}]}
  Trace all discrete simulation ticks and state transitions deterministically. Do not independently fabricate alternative values."""
        )

    if is_graph_algo:
        lines.append(
            """### Domain-Specific Guidance: Graph Algorithms & Greedy Invariants
- In algorithms: non-negative edge weights are necessary for Dijkstra's greedy finalization invariant (extracting vertex u with minimal tentative distance from min-heap guarantees true shortest distance if and only if all edge weights are non-negative).
- Visually and pedagogically trace vertex extraction, tentative distance comparison, and relaxation updates."""
        )

    if is_calculus:
        lines.append(
            """### Domain-Specific Guidance: Calculus & Local Linearity
- In calculus and analysis: differentiability at a point implies continuity at that point, but continuity at a point does NOT guarantee differentiability (continuity is necessary but not sufficient). A function need not be globally continuous to be differentiable at a particular point (differentiability is a local property). When pedagogically useful for illustrating boundaries, cusps, or misconceptions, introduce a suitable counterexample such as f(x) = |x| at x = 0 (continuous at 0, but non-differentiable because left and right difference quotients disagree); do NOT force this counterexample into every calculus lesson.
- Rate-of-Change & Vehicle Metaphors: when using a moving-car analogy for derivatives, distinguish signed velocity from non-negative speed. Map the derivative of position with respect to time strictly to instantaneous velocity (which can be positive, negative, or zero indicating direction of motion), and explain that speed is the magnitude of velocity: speed = |velocity|. Preserve the usefulness of the analogy without claiming it covers every mathematical function. Do NOT hardcode this analogy into all derivative lessons."""
        )

    if is_chem_kinetics:
        lines.append(
            """### Domain-Specific Guidance: Chemical Kinetics & Reaction Coordinates
- In physical sciences and chemistry, distinguish approximate physical metaphors from exact governing laws (e.g. a catalyst provides an alternative lower-activation-energy reaction pathway with intermediate complexes, NOT quantum mechanical tunnelling; temperature increases the fraction of molecules with E >= Ea according to the Maxwell-Boltzmann distribution rather than lowering the energy barrier; the Arrhenius pre-exponential factor A accounts for both collision frequency Z and steric orientation factor p: A = p * Z)."""
        )

    if is_search_algo:
        lines.append(
            """### Domain-Specific Guidance: Search Algorithms & Pointer Convergence
- Trace pointer convergence (low, high, mid) across discrete halving steps.
- In 'worked_example_spec', trace sorted array intervals, midpoint calculations, element comparisons, and boundary updates until target is found or search terminates."""
        )

    if is_geometric_proof:
        lines.append(
            """### Domain-Specific Guidance: Geometric Proof & Invariant Dissection
- Ground the proof in geometric dissection and area equivalence (a^2 + b^2 = c^2).
- Provide 'proof_spec' detailing hypotheses, geometric construction, algebraic transformations, and conclusion invariant."""
        )

    if is_biochemical:
        lines.append(
            """### Domain-Specific Guidance: Biochemical Pathways & Multi-Stage Transformations
- Detail multi-stage biochemical pathways (e.g. light reactions and Calvin cycle).
- Provide 'conceptual_breakdown' detailing stage names, molecular entities, and membrane transport states without forcing artificial arithmetic."""
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------
def get_prompt1_outline(
    knowledge_point, duration=None, reference_image_path=None, level="intermediate", analogy=None, topic_profile=None
):
    norm_level = _norm_level(level)
    kp = _clean_text(knowledge_point)
    analogy_text = _clean_text(analogy)
    budget = _narration_budget(duration, norm_level)
    n = budget["sections"]
    lo, hi = budget["words_per_line"]

    analogy_guidance = (
        _analogy_guidance(kp, analogy_text, norm_level) if analogy_text else _direct_guidance(kp)
    )

    if budget["user_specified"]:
        duration_line = (
            f"The user requested about {duration} minutes ({budget['target_minutes']:.1f} minutes). Match it within +/- 1 minute. "
            f"At ~{SPEECH_WORDS_PER_MINUTE} spoken words per minute this is about {budget['total_words']} narration words in total."
        )
    else:
        duration_line = (
            f"No duration given: design a full-depth video of about {budget['target_minutes']:.1f} minutes for a "
            f"{norm_level} learner (about {budget['total_words']} narration words at ~{SPEECH_WORDS_PER_MINUTE} words/minute). "
            f"Do NOT produce a shallow 1-2 minute overview."
        )

    try:
        from src.domains import VisualRegistry
        component_manifest = VisualRegistry.get_schema_manifest()
    except Exception:
        component_manifest = {}
    capabilities_block = _format_stage1_capabilities(component_manifest)

    resolved_profile = _resolve_topic_profile(kp, topic_profile)
    profile_guidance = _format_stage1_profile_guidance(resolved_profile, kp)

    prompt = f"""
As an outstanding instructional design expert, design a logically clear, step-by-step, example-driven teaching outline.

Knowledge Point / Topic: {kp}
Learning Level: {norm_level.capitalize()}
{LEVEL_GUIDANCE[norm_level]}
{analogy_guidance}
{_arc_guidance(kp, n)}

{capabilities_block}

{profile_guidance}
"""

    if reference_image_path:
        prompt += """
## Reference Image Available
A reference image related to this knowledge point has been provided.
- Identify which concepts, diagrams and relationships the image emphasizes.
- Design the sections that can best use those visuals, and recreate them later with native Manim shapes (the image itself is never shown in the video).
- Where the image and standard textbook facts disagree, follow the textbook facts.
"""

    prompt += f"""
## Requirements
1. Duration & depth: {duration_line}
   Narration budget: aim for 3 to 6 logical pedagogical sections based on topic structure (recommended ~{n} sections for this duration), {budget['lines_per_section']} lecture lines per section, {lo}-{hi} words per line.
2. Pedagogical Progression (FORMAL FIRST):
   - Section 1 MUST establish the textbook-accurate formal definition, purpose, and problem being solved BEFORE introducing analogies or metaphors. Ground the learner in technical reality first.
   - Subsequent sections progressively unfold intuition, mechanism, concrete worked demonstration, boundaries/misconceptions, and a meaningful conclusion.
   - Do NOT force every topic into a rigid six-section template; choose the natural section count ($N \\in [3, 6]$) that best teaches the concept.
3. Technical Accuracy & Correctness (HIGHEST PRIORITY):
   - Use only standard, textbook-accurate definitions, formulas, and claims. Never invent unsupported claims or fictional facts.
   - Distinguish necessary conditions from sufficient conditions where applicable to the domain.
   - State assumptions, domain bounds, and governing constraints explicitly (e.g. non-negative edge weights, continuous/differentiable domain, standard temperature and pressure).
   - In physical sciences and chemistry, distinguish approximate physical metaphors from exact governing laws.
   - Follow the specific pedagogical guidance in 'Dynamic Pedagogical Strategy' above for domain invariants, technical distinctions, and worked example requirements.
4. Numerical & Worked-Example Verification (STRICT JSON & RECOMPUTATION):
   - 'worked_example_spec' must use clean, concrete numbers (mostly small integers) that can be verified and clearly displayed on screen.
   - Sequential Beat Structure: partition multi-step derivations, calculations, and metric accounting into distinct sequential steps rather than condensing multiple calculations into one step. This ensures downstream storyboard narration and visual animations maintain 1:1 pedagogical alignment without long static intervals.
   - JSON Reliability: every 'worked_example_spec.steps[].result' and 'final_result' MUST be a quoted JSON string (e.g. "result": "75000 / (8.314 * 300) = 29.74", "result": "2x + h"). Intermediate mathematical expressions, calculations, and exact numerical values must always be quoted strings where they are part of the explanatory trace. NEVER emit raw arithmetic expressions or unquoted operators as JSON values. The entire output must remain 100% valid JSON.
   - Independent Recomputation: independently recompute intermediate arithmetic, final numerical results, units and dimensional consistency, and any exponential, logarithmic, or limit calculations. Do NOT simply repeat a result stated earlier in your reasoning without recomputing it.
   - Units & Dimensional Consistency: state explicit units (e.g. s^-1, m/s, J/mol) at each step where applicable.
   - In 'verification', describe the independent verification method (e.g. second computation via alternative formula, dimension check, or invariant check). If a numerical result cannot be reliably verified with exact precision, avoid presenting it as an exact verified result.
   - Narration and visual ideas must show identical values consistent with 'worked_example_spec'.
5. Intuition & Mental Model Rules:
   - An analogy or mental model is an explanatory cognitive bridge, NOT a fictional role-playing story. Never invent fictional characters, quests, or narrative fluff.
   - If an analogy was requested, evaluate its suitability: adopt it if sound, narrow it if only partially applicable, or reject it with clear technical rationale if misleading.
   - If no analogy was requested, independently synthesize an appropriate physical metaphor, geometric picture, or concrete real-world scenario that clarifies the mechanism. If no metaphor helps, use direct visual/mathematical modeling.
   - Always state where the mental model breaks down in 'mental_model.breakdown_boundary'.
6. Visual Progression & Anti-Repetition:
   - Every section must be drawable with native Manim shapes, text, and MathTex in a right-side workspace of about 5.8 x 3.6 units with at most 4 visual units on screen at once.
   - Select 'visual_progression_type' ('representation_introduction', 'state_transformation', 'relational_linkage', 'case_comparison', 'boundary_exposure') based genuinely on what each section actually teaches (introducing models, transforming state, linking representations, comparing cases, or exposing boundaries).
   - Reusing a progression type across sections is completely acceptable when justified by the teaching objective; avoid merely assigning labels mechanically and do NOT force every section to use a different type.
   - Avoid static visual repetition: if the same base diagram is retained across sections, explicitly declare what state transformation or new relationship is highlighted.
7. Multidimensional Topic Profiling:
   - Evaluate 'topic_profile' along four orthogonal dimensions:
     * 'state_evolution': discrete_state_machine (algorithms, DP, searches) | continuous_dynamic (kinetics, calculus) | multi_stage_process (biochemical pathways, photosynthesis) | static_structural (geometric proofs) | conceptual_axiomatic | uncertain.
     * 'formalism_level': rigorous_mathematical | mechanistic_empirical | system_structural | qualitative_intuitive | uncertain.
     * 'computation_requirement': deterministic_trace (step snapshots) | arithmetic_calculation | symbolic_identity (proofs) | none_not_applicable | uncertain.
     * 'primary_visual_mode': network_graph | coordinate_geometry | state_table_matrix | molecular_spatial | process_flowchart | structured_concept_card | uncertain.
   - Set 'rationale_and_evidence' providing concrete technical evidence from the topic.
   - Where topics have overlapping characteristics, declare secondary modes in 'secondary_visual_modes' as an array of 0 to 2 valid individual enum strings from 'primary_visual_mode' (e.g. ["molecular_spatial"] or []). NEVER emit pipe-separated strings inside the array.
8. Heterogeneous Teaching Beats:
   - In each section, break down instructional delivery into 2 to 4 granular 'beats'.
   - Each beat defines:
     * 'beat_id': unique beat identifier (e.g. section_1_beat_1).
     * 'pedagogical_objective': concrete measurable learning outcome for this beat.
     * 'teaching_move': select the strategy suited to this beat: intuition_bridge | formal_definition | mechanistic_trace | counterexample_exposure | boundary_clarification | case_comparison | relational_synthesis | conceptual_breakdown | proof_derivation | worked_demonstration.
     * 'concept_addressed': the invariant or sub-concept being taught.
     * 'visual_intent': layout focus and visual targets.
     * 'required_state_changes': visual/computational transitions.
     * 'verification_references': citations to equations, steps, or invariants.
   - Combine diverse moves across the lesson (e.g. an algorithm lesson combines intuitive overview, formal invariant proof, discrete simulation trace, and asymptotic analysis). Do NOT force a single strategy or beat sequence across all beats.
9. Conditional Grounding & Domain Adaptation:
   - Ground the lesson in accordance with the topic's multidimensional profile:
     * For computational topics with state machines or calculations (e.g. algorithms, DP, scheduling): declare 'state_evolution' as 'discrete_state_machine'; provide complete 'worked_example_spec' with quoted JSON strings.
     * For continuous dynamic systems (e.g. chemical kinetics, calculus): declare 'state_evolution' as 'continuous_dynamic'; provide verified numerical calculations with explicit units and dimensional consistency.
     * For multi-stage biological pathways and non-computational processes: declare 'state_evolution' as 'multi_stage_process'; provide 'conceptual_breakdown' detailing stages and transformations without forcing artificial arithmetic.
     * For deductive mathematical proofs and geometric theorems: declare 'state_evolution' as 'static_structural' or 'conceptual_axiomatic'; provide 'proof_spec' with hypotheses, deductive transformations, and conclusion invariants.
   - Follow the specific pedagogical guidance in 'Dynamic Pedagogical Strategy' above for topic-specific constraints and representations.

## Output format
Output ONLY one valid JSON object (no markdown fences, no comments, no trailing commas) with exactly this structure:
{_outline_schema(norm_level, analogy_text, budget)}

The 'sections' array should contain 3 to 6 objects (recommended ~{n} for this duration), with ids section_1 ... section_N.
"""
    return prompt


# ----------------------------------------------------------------------------
# Milestone 5.1.2: Bounded Stage 1 Outline Diagnostic Repair Prompt
# ----------------------------------------------------------------------------
def get_prompt1_outline_repair(
    knowledge_point: str,
    failed_output: str,
    error_message: str,
    line: Optional[int] = None,
    column: Optional[int] = None,
) -> str:
    """
    Constructs a targeted, single-turn repair prompt for Stage 1 Outline JSON parsing failures.
    Includes the exact parsing error and line/column information.
    Enforces a strict bounded repair policy: exactly 1 repair retry.
    """
    loc_str = ""
    if line is not None and column is not None:
        loc_str = f" at line {line}, column {column}"
    elif line is not None:
        loc_str = f" at line {line}"

    prompt = f"""You are an expert educational course designer. Your previous Stage 1 Teaching Outline for '{knowledge_point}' failed JSON parsing with the following syntax error:

## JSON Parsing Error
{error_message}{loc_str}

## Previous Output Excerpt (Failed JSON)
```json
{failed_output}
```

## Bounded Repair Instructions
1. Correct the JSON syntax error identified above{loc_str}.
2. Ensure all property names and string values are enclosed in valid double quotes. Do NOT mix escaped quotes (like `\\\"`) inside raw unquoted JSON objects.
3. Ensure all brackets, braces, and commas are properly balanced.
4. Output ONLY the repaired raw JSON object conforming strictly to the TeachingOutline schema.
5. Do NOT include markdown fences, comments, or conversational text. Output only raw JSON.
"""
    return prompt
