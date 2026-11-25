import json
import math
import re
from typing import Any, Dict, List, Optional, Union

try:
    from src.pedagogy import TopicProfile, TopicProfiler
except ImportError:
    try:
        from pedagogy import TopicProfile, TopicProfiler
    except ImportError:
        TopicProfile = None
        TopicProfiler = None

try:
    from src.ir.serialization import export_scene_ir_json_schema
    from src.ir.models import SceneSequenceIR
except ImportError:
    try:
        from ir.serialization import export_scene_ir_json_schema
        from ir.models import SceneSequenceIR
    except ImportError:
        export_scene_ir_json_schema = None
        SceneSequenceIR = None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _norm_level(level):
    norm = (level or "intermediate").lower()
    return norm if norm in ("beginner", "intermediate", "advanced") else "intermediate"


def _clean_text(value):
    """Single-line string without double quotes (safe to paste into prompts and generated code)."""
    if value is None:
        return ""
    return " ".join(str(value).split()).replace('"', "'")


def _outline_text(outline):
    if isinstance(outline, str):
        return outline
    if hasattr(outline, "to_dict"):
        try:
            return json.dumps(outline.to_dict(), indent=2, ensure_ascii=False)
        except Exception:
            pass
    try:
        return json.dumps(outline, indent=2, ensure_ascii=False)
    except TypeError:
        return str(outline)


# ----------------------------------------------------------------------------
# Level guidance (topic-agnostic: no linear-algebra / cache bias)
# ----------------------------------------------------------------------------
LEVEL_GUIDANCE = {
    "beginner": r"""
### Level-Specific Storyboard Guidance (BEGINNER)
- Narration: plain, clear language; define every symbol the first time it is spoken or shown.
- Animation: intuitive geometric visualizations and visual analogies, built up one small step at a time. No crowded diagrams.
- Mathematical rigor: simple, transparent arithmetic shown step by step; formulas appear only after the picture that motivates them.
""",
    "intermediate": r"""
### Level-Specific Storyboard Guidance (INTERMEDIATE)
- Narration: balanced, substantive explanations that connect intuition directly to the formal definition and the mechanics.
- Animation: structured diagrams, coordinate frames, tables or MathTex formulas in lockstep with what is being said.
- Mathematical rigor: standard formal definitions and exact calculations; every number is correct.
""",
    "advanced": r"""
### Level-Specific Storyboard Guidance (ADVANCED)
- Narration: formal precision and technical vocabulary; deep mechanistic explanation without rushing.
- Animation: information-dense but readable visualizations (transformations, invariants, parametric views, edge cases).
- Mathematical rigor: formal definitions, derivation or proof sketches, edge cases and assumptions stated explicitly; every claim exact.
""",
}


# ----------------------------------------------------------------------------
# Analogy / direct guidance
# ----------------------------------------------------------------------------
def _analogy_guidance(analogy_text):
    return rf"""
### Story-Driven Storyboard & Narration Requirements
Requested analogy / personal interest: {analogy_text}
Follow 'story_framework', 'analogy_evaluation' and 'notation' from the outline:
- Analogy Grounding: Continue using the evaluated or adopted user analogy from 'analogy_evaluation.adopted_analogy'. Preserve its explicit entity mappings and limitations. Do not confuse it with an independently selected AI mental model.
- Order inside each section (Formal-First): Section 1 introduces the formal definition and purpose first; subsequent sections bridge with story events and close with formal rules. Never leave the learner with only the metaphor.

Continuous Narrative Progression Across ALL Sections:
- DO NOT introduce '{analogy_text}' in Section 1 and then revert to abstract generic visuals without explanation.
- Worked Example Section: execute a concrete trace using story-grounded entities and verified numeric values from 'worked_example_spec'.
- Deeper / Boundary Section: identify where the metaphor breaks down ('analogy_breakdown' and 'analogy_evaluation.limitations') and contrast with the exact formal rule.
- Conclusion Section: resolve the story narrative and summarize real-world significance.

RULES:
1. Order inside each section: formal statement first, story event as the bridge, formal statement again to close (never leave the learner with only the metaphor).
2. Keep the story entities visible and consistent: the same entity keeps the same shape, color and home area in every section (record them in 'visual_conventions').
3. Script concrete story visuals in 'animations' with NATIVE Manim objects only:
   - an active agent/traveler token = a small colored circle or rounded rectangle that physically moves between locations (no emoji, no images);
   - locations carry real names (e.g. "City A"), not bare letters, whenever the analogy uses place names;
   - quantities carry units (e.g. "12 km", "30 min") instead of bare digits.
4. Before a summary table or formula appears, the story diagram must be faded out (say so in the step).
5. Show WHERE THE ANALOGY BREAKS DOWN in the section the outline designates, using clear visual contrast (different color or dashed outline) and the exact formal statement. NEVER sacrifice mathematical correctness for the analogy.
6. The examples above are illustrative only: adapt every example to THIS topic and THIS analogy; never copy them literally.
"""


def _direct_guidance():
    return r"""
### Direct Technical Narration & Mental Model Guidance (no analogy requested)
No user analogy was requested. Ground the lesson directly in standard technical formalism, native diagrams, and mathematical models:
- Do NOT add fictional stories, roleplaying quests, or characters. Explain with the formal definition, diagrams, examples and equations of the topic.
- Formal-First Order: Section 1 establishes the standard formal definition, purpose, and notation before introducing intuitive mental models.
- Grounding via AI Mental Model:
  * If the outline provides a 'mental_model' (with 'description' and 'entity_mappings'), use it to ground appropriate visual representations (e.g. coordinate frames, expanding wavefronts, potential energy barriers, or state machines) when demonstrating mechanisms.
  * Respect 'mental_model.breakdown_boundary': show where the intuitive picture ends and exact technical rules apply.
  * Do NOT automatically turn a mental model into fictional role-playing. Do NOT introduce traveler tokens, fictional characters, or a continuous fictional narrative unless the pedagogical plan genuinely requires them.
  * If the mental model is missing or unsuitable, use direct technical explanation, concrete worked examples, and suitable native Manim visual representations.
- Follow the section arc of the outline exactly: definition and motivation, structure and notation, mechanism, worked example, formal results and trade-offs, applications and summary (merged or split as the outline does). Reusing visual progression types across sections is fully acceptable when pedagogically justified.
- The worked-example section must reuse the values from the outline's 'worked_example_spec' exactly, shown step by step.
"""


# ----------------------------------------------------------------------------
# Visual Architecture & Layout Contract
# ----------------------------------------------------------------------------
def _format_component_manifest(manifest):
    if not manifest:
        return r"""
## Available Visual Architecture & Registry Components
No domain-specific components are currently registered in VisualRegistry.
Follow the 4-Tier Fallback Hierarchy below:
- Tier 1: Registered Domain Component (none registered currently)
- Tier 2: Multi-Component Composition across disjoint viewports
- Tier 3: ProceduralComponent (wrap custom native Manim groups: Circle, Rectangle, Arrow, Axes, MathTex)
- Tier 4: Structured Conceptual Card (formal definition, equation card, or invariant; NEVER verbatim narration dumping)
"""
    lines = ["## Available Visual Architecture & Registry Components"]
    lines.append("The following domain visual components are registered in VisualRegistry and available for use:\n")
    for domain, comps in manifest.items():
        lines.append(f"### Domain: {domain.upper()}")
        for comp in comps:
            cid = comp.get("component_id", "Component")
            desc = comp.get("description", "").strip()
            params = comp.get("init_params", {})
            actions = comp.get("semantic_actions", [])
            handles = comp.get("supported_handles", [])
            suggested_vp = comp.get("suggested_viewport", "dock_main")

            param_items = []
            for k, v in params.items():
                req = "required" if v.get("required") else f"default={v.get('default')}"
                ptype = v.get("type", "Any")
                param_items.append(f"{k}={'required' if v.get('required') else v.get('default')} ({ptype})")
            param_str = ", ".join(param_items) if param_items else "none"

            action_str = ", ".join(actions) if actions else "standard transforms (FadeIn, FadeOut, MoveTo, Recolor)"
            lines.append(f"- **{cid}** (Domain: `{domain}`): {desc}")
            lines.append(f"  * Suggested Viewport: `{suggested_vp}`")
            lines.append(f"  * Init Parameters: {param_str}")
            if handles:
                lines.append(f"  * Semantic Handles: {', '.join(handles)}")
            lines.append(f"  * Semantic Actions: {action_str}")
    lines.append("\n### 4-Tier Fallback Hierarchy:")
    lines.append("- Tier 1 (Registered): If a registered component above matches the concept, specify it directly.")
    lines.append("- Tier 2 (Composition): Compose multiple components across disjoint viewports (e.g. primary in dock_main + status in dock_inspector).")
    lines.append("- Tier 3 (ProceduralComponent): Wrap custom native Manim shapes/diagrams in ProceduralComponent.")
    lines.append("- Tier 4 (Structured Conceptual Card): Use structured cards for formal definitions, theorems, or equations. NEVER dump spoken narration into a text card.")
    return "\n".join(lines)


LAYOUT_CONTRACT = r"""
### Generic Container Layout & Viewport Contract
CineLearn provides bounded, non-overlapping viewports on the 16:9 canvas (X in [-7.11, 7.11], Y in [-4.00, 4.00]):
- Reserved zones:
  * LEFT column: Lecture captions (managed by TeachingScene, X < -2.15).
  * TOP strip: Section Title reserved zone (Y in [2.70, 3.80]).
  * BOTTOM strip: Subtitle / Captions reserved zone (Y < -2.75).
- Standard Bounded Viewports:
  * dock_main: Primary visual canvas (Y in [-1.55, 2.55], X in [-2.15, 6.60]).
  * dock_inspector: Secondary status table / equation / metric inspector (Y in [-2.65, -1.65], X in [-2.15, 6.60]). Guaranteed to never collide with subtitles.
  * dock_split_left (X in [-2.15, 2.10]) & dock_split_right (X in [2.35, 6.60]): Comparative side-by-side viewports (disjoint X gap >= 0.25).
  * dock_full: Full-canvas immersive diagram or simulation (Y in [-2.65, 2.55], X in [-2.15, 6.60]).
- Legacy 6x6 Grid Compatibility:
  * Discrete grid areas (e.g. B2-D4, C3, E5) remain supported for fine-grained sub-element layout.
  * Rows A-F, Columns 1-6. Row F is strictly clamped to Y >= -2.65.
- Capacity:
  * At most 4 visual units visible at once.
  * Simultaneous objects must be placed in disjoint viewports (e.g. primary in dock_main + status table in dock_inspector, or split left vs split right).
"""

ANIMATION_FORMAT = r"""
### Animation Step Format (MANDATORY - EXACT 6-FIELD DELIMITER GRAMMAR)
Write every step as ONE string with EXACTLY 5 pipe '|' delimiters separating 6 fields:
  "L<line>-<k> | <effect> | <visual_representation> | viewport: <dock_main|dock_inspector|dock_split_left|dock_split_right|dock_full|area: B2-D4> | action: <semantic_action or description> | on screen after: [<units>]"

- Field 1 (Tag): "L<line>-<k>" where <line> is the lecture line number (1-based) and <k> is the step order within that line. Every lecture line has 1 to 2 steps.
- Field 2 (Effect): One of: FadeIn, FadeIn(shift), GrowFromCenter, Create, Write, Indicate, MoveTo, Recolor, FadeOut.
  * Content replacement transitions MUST place the entire transition in this single field separated by a semicolon:
    "FadeOut <old units>; then <effect>"
    CORRECT: "FadeOut old_unit; then FadeIn"
    STRICTLY FORBIDDEN: NEVER insert extra pipe '|' characters inside the transition or effect field!
    INCORRECT: "FadeOut | old_unit | then FadeIn" (this injects illegal pipe delimiters and corrupts the 6-field parser).
- Field 3 (Visual Representation): Names the visual structure following the 4-Tier Fallback:
  * Tier 1 (Registered): e.g. "GraphVisual" or "MoleculeVisual"
  * Tier 2 (Composed): e.g. "GraphVisual + Table" or "MoleculeVisual + EnergyCurve"
  * Tier 3 (Procedural): e.g. "ProceduralComponent(ReactionEnergyProfile)" or "ProceduralComponent(StateMatrix)"
  * Tier 4 (Conceptual Card / MathTex): e.g. "ConceptualCard(FirstOrderRateLaw)" or 'MathTex(r"k = A e^{-\frac{E_a}{RT}}")'
    STRICT MATHTEX RULE: MathTex MUST ALWAYS contain a valid raw LaTeX string literal enclosed in quotes (e.g. MathTex(r"...")).
    NEVER split mathematical notation into comma-separated English tokens (e.g. NEVER write MathTex(k, equals, A, times, exp)).
- Field 4 (Viewport): Specifies docking viewport ("dock_main", "dock_inspector", "dock_split_left", "dock_split_right", "dock_full") or backward-compatible grid area (e.g. "area: B2-D4").
- Field 5 (Action): Specifies semantic action or transform (e.g. "highlight_node('A')", "update_cell(row=1, col=2)", "break_bond('O-H')", "show_formula()").
- Field 6 (On Screen After): Lists visual units still visible after the step (max 4). Anything not listed is considered gone.
- Colors: use only light, high-contrast colors from 'visual_conventions.palette'.
"""


# ----------------------------------------------------------------------------
# Storyboard schema (built with json.dumps)
# ----------------------------------------------------------------------------
def _storyboard_schema(analogy_text):
    section = {
        "id": "section_1",
        "title": "Sec 1: Short Section Title",
        "story_beat": "Pedagogical focus of this section (one sentence)",
        "lecture_lines": [
            "<Line 1: Full spoken sentence introducing the core concept or initial setup.>",
            "<Line 2: Next spoken sentence detailing the mechanism, state, or mathematical relation.>",
            "<Line 3: Third spoken sentence explaining the computation step, transition, or decision.>",
            "<Line 4: Concluding spoken sentence summarizing the result, invariant, or next takeaway.>",
        ],
        "captions": [
            "<=7 words for line 1",
            "<=7 words for line 2",
            "<=7 words for line 3",
            "<=7 words for line 4",
        ],
        "animations": [
            "L1-1 | FadeIn | ProceduralComponent(SetupDiagram) | viewport: dock_main | action: initialize_structure() | on screen after: [primary visual]",
            "L2-1 | FadeIn | Table(StateInspector) | viewport: dock_inspector | action: show_initial_values() | on screen after: [primary visual, status table]",
            "L3-1 | Indicate | active element in primary visual | viewport: dock_main | action: highlight_step(1) | on screen after: [primary visual, status table]",
            "L4-1 | FadeOut status table; then Write | MathTex(r\"k = A e^{-\\frac{E_a}{RT}}\") | viewport: dock_inspector | action: update_final_result() | on screen after: [primary visual, formula card]",
        ],
        "layout_plan": "Which unit sits in which viewport, e.g. 'primary visual in dock_main, table in dock_inspector'",
        "verified_values": ["every number/formula displayed or spoken, with its check, e.g. '7 + 5 = 12'"],
    }
    schema = {
        "visual_conventions": {
            "palette": {
                "primary": "#58C4DD",
                "highlight": "#FFD700",
                "positive": "#83C167",
                "accent": "#FFA07A",
                "warning": "#FF6B6B",
            },
            "entities": [
                {
                    "name": "entity or concept that appears in several sections",
                    "shape": "native Manim shape",
                    "color": "#hex from the palette",
                    "home_area": "dock_main or dock_inspector or grid cell",
                }
            ],
        },
        "sections": [section],
    }
    return json.dumps(schema, indent=4, ensure_ascii=False)


def extract_topic_profile(outline):
    """Safely extracts or reconstructs TopicProfile from outline string, dict, or object."""
    outline_data = None
    if isinstance(outline, str):
        try:
            outline_data = json.loads(outline)
        except Exception:
            outline_data = None
    elif isinstance(outline, dict):
        outline_data = outline
    elif hasattr(outline, "to_dict"):
        outline_data = outline.to_dict()
    elif hasattr(outline, "__dict__"):
        outline_data = outline.__dict__

    if isinstance(outline_data, dict):
        prof_data = outline_data.get("topic_profile")
        if prof_data and isinstance(prof_data, dict):
            if TopicProfile and TopicProfiler:
                return TopicProfiler.profile_from_data(prof_data, context={"topic": outline_data.get("topic", "")})
            return prof_data
        elif hasattr(outline, "get_topic_profile") and callable(outline.get_topic_profile):
            p = outline.get_topic_profile()
            if p:
                return p
        topic = str(outline_data.get("topic", ""))
        if topic and TopicProfiler:
            prof = TopicProfiler.profile_heuristic_fallback(topic)
            if prof.is_unclassified():
                t_low = topic.lower()
                if any(term in t_low for term in ["fcfs", "cpu scheduling", "scheduling", "process management"]):
                    prof = TopicProfiler.profile_heuristic_fallback(topic, description="operating system")
                elif any(term in t_low for term in ["chemical kinetics", "kinetics", "chemical reaction"]):
                    prof = TopicProfiler.profile_heuristic_fallback(topic, description="reaction kinetics")
            return prof
    return None


def _format_stage2_profile_guidance(profile):
    """Formats dynamic visual strategy guidance from TopicProfile for Stage 2 storyboard."""
    if not profile:
        return ""

    if hasattr(profile, "to_dict"):
        p_dict = profile.to_dict()
    elif isinstance(profile, dict):
        p_dict = profile
    else:
        return ""

    vis_mode = str(p_dict.get("primary_visual_mode", "")).lower()
    status = str(p_dict.get("profile_status", "CLASSIFIED")).upper()
    try:
        conf = float(p_dict.get("confidence", 1.0))
    except (ValueError, TypeError):
        conf = 0.0

    lines = ["### Topic Profile Visual Strategy"]
    lines.append(f"Profile Status: {status} (Confidence: {conf:.2f})")
    if vis_mode == "network_graph":
        lines.append(
            "- Primary Visual Mode: Network Graph. Emphasize node creation, edge connections, traversal highlights, and state transitions on nodes and edges."
        )
    elif vis_mode == "coordinate_geometry":
        lines.append(
            "- Primary Visual Mode: Coordinate Geometry. Structure visual elements along continuous coordinates, spatial intervals, or mathematical axes."
        )
    elif vis_mode == "state_table_matrix":
        lines.append(
            "- Primary Visual Mode: State Table / Matrix. Display multi-step data or state evolution using structured grid/table layouts."
        )
    elif vis_mode == "process_flowchart":
        lines.append(
            "- Primary Visual Mode: Process Flowchart / Pipeline. Use distinct stage blocks, queues, and directed connectors showing items moving through pipeline stages."
        )
    elif vis_mode == "molecular_spatial":
        lines.append(
            "- Primary Visual Mode: Molecular / Spatial Structure. Use structural 2D representations, reaction coordinates, and bond transformations."
        )
    elif vis_mode == "structured_concept_card":
        lines.append(
            "- Primary Visual Mode: Structured Concept Cards. Organize formal definitions, theorems, or governing equations in dedicated cards in dock_inspector or dock_main."
        )
    else:
        lines.append(
            "- Visual Mode: Adaptive Multi-Tier Composition. Select native Manim shapes and structured containers matched to each section's instructional goals."
        )
    return "\n".join(lines)


def extract_verified_algorithmic_state(outline):
    """Safely extracts verified algorithmic state (e.g. Dijkstra) from outline string, dict, or object."""
    outline_data = None
    if isinstance(outline, str):
        try:
            outline_data = json.loads(outline)
        except Exception:
            return None
    elif isinstance(outline, dict):
        outline_data = outline
    elif hasattr(outline, "to_dict"):
        outline_data = outline.to_dict()
    elif hasattr(outline, "__dict__"):
        outline_data = outline.__dict__

    if not isinstance(outline_data, dict):
        return None

    # Check outline-level verification_metadata
    vm = outline_data.get("verification_metadata")
    if isinstance(vm, dict) and vm.get("verified_algorithmic_state"):
        return vm["verified_algorithmic_state"]

    # Check worked_example_spec verification_metadata
    spec = outline_data.get("worked_example_spec")
    if isinstance(spec, dict):
        spec_vm = spec.get("verification_metadata")
        if isinstance(spec_vm, dict) and spec_vm.get("verified_algorithmic_state"):
            return spec_vm["verified_algorithmic_state"]

    if outline_data.get("verified_algorithmic_state"):
        return outline_data["verified_algorithmic_state"]

    # Dynamic fallback if spec exists and topic indicates Dijkstra or FCFS
    spec = outline_data.get("worked_example_spec", {})
    topic = str(outline_data.get("topic", ""))
    if isinstance(spec, dict):
        try:
            from worked_example_verifier import WorkedExampleVerifier
            if WorkedExampleVerifier.is_dijkstra_spec(spec, topic):
                from dijkstra_verifier import DijkstraTraceVerifier
                _, rep = DijkstraTraceVerifier.verify_worked_example_spec(spec, topic=topic)
                return rep.verified_algorithmic_state
        except Exception:
            pass
        try:
            from fcfs_verifier import FCFSTraceVerifier
            if FCFSTraceVerifier.can_verify(spec, topic):
                _, rep = FCFSTraceVerifier.verify_worked_example_spec(spec, topic=topic)
                return rep.verified_algorithmic_state
        except Exception:
            pass

    return None


def _format_dijkstra_storyboard_guidance(algo_state):
    source = algo_state.get("source", "A")
    vertices = algo_state.get("vertices", [])
    edges = algo_state.get("edges", [])
    directed = algo_state.get("directed", False)
    settled_order = algo_state.get("settled_order", [])
    final_distances = algo_state.get("final_distances", {})
    shortest_paths = algo_state.get("shortest_paths", {})
    canonical_trace = algo_state.get("canonical_trace", [])

    edge_strs = [f"({e.get('u')}, {e.get('v')}, w={e.get('weight')})" for e in edges]
    edges_formatted = ", ".join(edge_strs)

    trace_lines = []
    for step in canonical_trace:
        stype = step.get("step_type", "iteration")
        sidx = step.get("step_index", 0)
        if stype == "initialization":
            dists = ", ".join(f"{k}: {v}" for k, v in step.get("tentative_distances", {}).items())
            queue = ", ".join(step.get("remaining_queue", []))
            trace_lines.append(
                f"  * Beat 1 (Setup): Source node '{source}', Initial Tentative Distances = {{{dists}}}, Unsettled Queue = [{queue}]"
            )
        else:
            ext_node = step.get("extracted_node")
            ext_dist = step.get("extracted_dist")
            relax_strs = []
            for r in step.get("relaxations", []):
                u, v, w = r.get("u"), r.get("v"), r.get("weight")
                prev_d = r.get("previous_dist")
                cand_d = r.get("candidate_dist")
                upd = r.get("is_updated")
                prev_str = "∞" if prev_d == "inf" or (isinstance(prev_d, float) and math.isinf(prev_d)) else prev_d
                status_str = f"UPDATED to {cand_d}" if upd else f"NO UPDATE (candidate {cand_d} >= current {prev_str})"
                relax_strs.append(f"examine ({u}, {v}, w={w}): previous={prev_str}, candidate={cand_d} -> {status_str}")
            relax_formatted = "; ".join(relax_strs) if relax_strs else "no unvisited outgoing edges"
            dists = ", ".join(f"{k}: {v}" for k, v in step.get("tentative_distances", {}).items())
            trace_lines.append(
                f"  * Extraction Beat (Node '{ext_node}', dist={ext_dist}):\n"
                f"      - Outgoing Relaxations: {relax_formatted}\n"
                f"      - Updated Distances: {{{dists}}}\n"
                f"      - Settled Nodes: {step.get('settled_nodes', [])}"
            )
    trace_str = "\n".join(trace_lines)

    final_dists_str = ", ".join(f"{k}: {v}" for k, v in final_distances.items())
    paths_str = ", ".join(f"{k}: {' -> '.join(p)}" for k, p in shortest_paths.items())

    return f"""
### Authoritative Dijkstra Worked Example Execution (MANDATORY VERIFIED TRACE)
The outline contains a deterministically verified Dijkstra simulation on graph G. Stage 2 MUST faithfully visualize this exact execution without omitting steps, altering numbers, or skipping intermediate states.

#### 1. Immutable Verified Facts (MUST PRESERVE EXACTLY)
- Graph Topology: Vertices V = {vertices}, Source s = '{source}', Directed = {directed}
- Edge Weights: [{edges_formatted}]
- Extraction Order: {settled_order}
- Canonical Trace Steps ({len(canonical_trace)} steps):
{trace_str}
- Authoritative Final Distances: {{{final_dists_str}}}
- Shortest Paths: {{{paths_str}}}

#### 2. Required Teaching Sequence (Beats to Cover in Worked Example)
1. Initial graph, source node '{source}', initial distances (dist[{source}]=0, others=∞), and initial unsettled set.
2. First minimum-distance extraction (extract node '{settled_order[0]}' with distance 0).
3. Examination of outgoing edges and relaxation decisions.
4. Distance updates, explicitly showing previous distance, candidate distance, and whether an update occurred.
5. Updated graph/table/queue state following relaxation.
6. Subsequent extractions and relaxations ({', '.join(settled_order[1:])}), repeated until all reachable nodes are settled.
7. Final shortest distances ({{{final_dists_str}}}) and predecessor/shortest-path interpretation.

#### 3. Visual Progression & Action Requirements (Visible State Transitions - NOT Just Narration)
The storyboard MUST describe visible state transitions in 'animations', NOT merely spoken narration about them:
- Highlight Extracted Node: 'action: extract_min(\'{source}\')' or 'action: select_node(\'{source}\')' / 'action: highlight_node(\'{source}\')'.
- Highlight Examined Edge: 'action: highlight_edge(\'u\', \'v\')' or 'action: traverse_edge(\'u\', \'v\')'.
- Visible Distance Update (when relaxation improves distance):
  * 'action: relax_edge(u=\'u\', v=\'v\', candidate=D, updated=True, new_dist=D)' OR 'action: update_node_label(\'v\', D)' / 'action: update_distance(\'v\', D)'.
  * Accompany with MathTex calculation: MathTex(r"dist(v) = \\min(prev, dist(u) + w) = D").
- Visible Unsuccessful Relaxation (candidate >= current distance):
  * 'action: relax_edge(u=\'u\', v=\'v\', candidate=C, updated=False)' OR 'action: no_update(\'v\')' / 'action: keep_distance(\'v\')'.
  * Accompany with MathTex: MathTex(r"dist(u) + w \\ge current \\implies \\text{{no update}}").
- Mark Settled Nodes: 'action: visit_node(\'u\')' or 'action: mark_settled(\'u\')'.
- Update Distance Table & Queue: update table entries visibly in 'dock_inspector'.
- Final Shortest Path Tree: 'action: highlight_path(...)'.

#### 4. Pedagogical Choices (Stage 2 MAY DECIDE)
- Section Partitioning: You may organize the worked example across 1 to 3 storyboard sections, provided NO canonical operation is omitted.
- Intuition & Narration: Explain WHY each node is selected (minimum tentative distance) and WHY non-negative edge weights guarantee optimality.
- Visual Layout: Use 'dock_main' for GraphVisual and 'dock_inspector' or 'dock_split_right' for Table/DPTableMatrix.

#### 5. FORBIDDEN Errors (Validation Gate will REJECT storyboard if violated)
- FORBIDDEN: Showing correct final distances while skipping intermediate extraction/relaxation steps will FAIL validation.
- FORBIDDEN: Narration mentioning operations without corresponding visual animation actions will FAIL validation.
- FORBIDDEN: Incorrect intermediate distance claims (e.g. claiming an incorrect distance like 4 instead of 2).
- FORBIDDEN: Presenting an unsuccessful relaxation as having updated the distance.
"""


def _format_fcfs_storyboard_guidance(algo_state):
    processes = algo_state.get("processes", [])
    gantt = algo_state.get("gantt_chart", [])
    metrics = algo_state.get("process_metrics", {})
    avg_tat = algo_state.get("average_turnaround_time", 0.0)
    avg_wt = algo_state.get("average_waiting_time", 0.0)
    avg_rt = algo_state.get("average_response_time", 0.0)
    total_time = algo_state.get("total_simulation_time", 0)

    proc_strs = [
        f"{p.get('pid')}: Arrival Time = {p.get('arrival_time')}, Burst Time = {p.get('burst_time')}"
        for p in processes
    ]
    procs_formatted = "\n".join(f"  * {s}" for s in proc_strs)

    gantt_strs = []
    for b in gantt:
        if b.get("is_idle"):
            gantt_strs.append(f"  * Time [{b.get('start_time')}, {b.get('end_time')}]: CPU IDLE")
        else:
            gantt_strs.append(
                f"  * Time [{b.get('start_time')}, {b.get('end_time')}]: Process {b.get('pid')} executing (duration {b.get('duration')})"
            )
    gantt_formatted = "\n".join(gantt_strs)

    metric_strs = []
    for pid, m in metrics.items():
        rt_part = f", Response Time (RT) = {m.get('response_time')}" if m.get("response_time") is not None else ""
        metric_strs.append(
            f"  * {pid}: Completion Time (CT) = {m.get('completion_time')}, "
            f"Turnaround Time (TAT = CT - AT) = {m.get('turnaround_time')}, "
            f"Waiting Time (WT = TAT - BT) = {m.get('waiting_time')}{rt_part}"
        )
    metrics_formatted = "\n".join(metric_strs)

    return f"""
### Authoritative FCFS CPU Scheduling Simulation & Trace Facts (MANDATORY VERIFIED TRACE)
The outline contains a deterministically verified FCFS CPU scheduling simulation. Stage 2 MUST faithfully visualize this exact execution without altering arrival times, changing queue order, preempting running processes, or fabricating numbers.

#### 1. Immutable Verified Facts (MUST PRESERVE EXACTLY)
- Processes & Problem Input Specifications (Authoritative Ground Truth):
{procs_formatted}
- Authoritative Gantt Chart Timeline (Total Time = {total_time}):
{gantt_formatted}
- Authoritative Per-Process Metrics:
{metrics_formatted}
- Summary Averages: Average Waiting Time = {avg_wt:.2f}, Average Turnaround Time = {avg_tat:.2f}, Average Response Time = {avg_rt:.2f}

#### 2. Pedagogical Organization & 12 Core Educational Beats (MANDATORY TEACHING PROGRESSION)
A genuinely good educational video MUST teach these 12 concepts in natural pedagogical progression:
1. **CPU Scheduling Motivation**: In multiprogramming, multiple processes in memory compete for a single CPU core. Illustrate this with a visual diagram showing a CPU core with tasks waiting and switching to maximize CPU utilization across CPU and I/O bursts.
2. **Process Control Block (PCB)**: An OS kernel data structure containing PID, Arrival Time (AT), Burst Time (BT), and current state. Use `ProcessCardVisual(pid="P1", arrival_time=0, burst_time=3, state="NEW")` in `dock_split_left` or `dock_main`.
3. **Process States & Lifecycle Transitions**: Explicitly trace the 4-state lifecycle (NEW -> READY -> RUNNING -> TERMINATED) and explain trigger conditions. Never duplicate consecutive state transitions.
4. **Process Input Specification (CRITICAL BEAT BEFORE EXECUTION)**:
   - MUST introduce the problem input data table/specification BEFORE any CPU dispatch or execution occurs:
     * P1: Arrival Time = 0, Burst Time = 3
     * P2: Arrival Time = 1, Burst Time = 4
     * P3: Arrival Time = 2, Burst Time = 2
   - Note: There is no dedicated 3-column table component in the registry. Display this input specification using `ProcessCardVisual` cards for P1, P2, P3 showing AT and BT, or procedural presentation. Do not assume `MetricTableVisual` has an input-only mode.
5. **Ready Queue**: A FIFO queue holding processes in READY state waiting for CPU. Use `ReadyQueueVisual(capacity=4)` in `dock_split_left`. Visibly distinguish HEAD (left, towards CPU) and TAIL (right, entrance).
6. **CPU Scheduler vs Dispatcher**: Decision vs execution flow:
   - Scheduler: Selects which process from the Ready Queue runs next based on scheduling policy.
   - Dispatcher: Performs the actual context switch and allocates the CPU core to the selected process.
7. **FCFS Selection & Non-Preemptive Execution**:
   - FCFS policy: The process that requests the CPU first gets the CPU first (earliest arrival time). Tie-breaking is stable arrival order.
   - Non-preemption rule: Once a process starts running on the CPU, it holds the CPU until completion. Arriving processes enter the Ready Queue; they NEVER interrupt or preempt the executing process.
   - Use `ReadyQueueVisual` in `dock_split_left` and `CPUExecutionVisual` in `dock_split_right`.
8. **Chronological Process Execution (COMPLETE P1, P2, P3 COVERAGE)**:
   - Every process (P1, P2, AND P3) MUST be explicitly dispatched, executed, and completed in animations:
     * Step 1: P1 arrives at t=0, dispatched to CPU, runs from t=0 to 3.
     * Step 2: P2 arrives at t=1 (waits in Ready Queue), P3 arrives at t=2 (waits in Ready Queue).
     * Step 3: P1 completes at t=3. P2 is dispatched from queue head, runs from t=3 to 7.
     * Step 4: P2 completes at t=7. P3 is dispatched from queue head, runs from t=7 to 9.
     * Step 5: P3 completes at t=9. System idle/terminated.
   - DO NOT stop after P1! P2 and P3 must both be visibly dispatched and executed to completion.
9. **Gantt Chart Construction**: Build the horizontal timeline bar step-by-step using `GanttChartVisual` as a primary visual, marking start, duration, and completion times for [0, 3], [3, 7], and [7, 9].
10. **Scheduling Metric Formulas (CRITICAL - EXPLAIN FORMULAS EXPLICITLY)**:
    - Display and teach the mathematical formulas clearly in narration and visuals:
      * Completion Time (CT): Time when process finishes execution (P1=3, P2=7, P3=9).
      * Turnaround Time (TAT = CT - AT): Total elapsed time from arrival to completion (P1: 3-0=3, P2: 7-1=6, P3: 9-2=7).
      * Waiting Time (WT = TAT - BT): Time spent waiting in the ready queue (P1: 3-3=0, P2: 6-4=2, P3: 7-2=5).
      * Response Time (RT = First Run Time - AT): In non-preemptive FCFS, RT == WT (P1=0, P2=2, P3=5).
11. **Summary Averages Calculation**:
    - Use `MetricTableVisual` in `dock_main` to display all process metrics. Call `animate_show_averages()` to reveal averages with exact arithmetic:
      * Average CT = (3 + 7 + 9) / 3 = 19/3 = 6.33
      * Average TAT = (3 + 6 + 7) / 3 = 16/3 = 5.33
      * Average WT = (0 + 2 + 5) / 3 = 7/3 = 2.33
      * Average RT = (0 + 2 + 5) / 3 = 7/3 = 2.33
12. **FCFS Limitations & The Convoy Effect**:
    - The Convoy Effect: If a CPU-bound process with a large burst time arrives first, all subsequent short processes get stuck waiting behind it in the Ready Queue, drastically inflating average waiting time and underutilizing I/O. Contrast with preemptive scheduling (e.g. Round Robin).

#### 3. Connected Component State Transitions (One Coherent Scheduling Event)
When a process is dispatched, your animations MUST communicate one synchronized scheduling transition:
- Ready Queue: `ready_queue.animate_dequeue()` (process leaves queue head, subsequent processes shift forward).
- CPU Core: `cpu_exec.animate_dispatch(pid=...)` (process enters execution dock, badge updates to EXECUTING: {pid}).
- Gantt Chart: `gantt_chart.animate_add_block(pid=..., start=..., end=...)` (timeline adds block and advances time marker).
- CPU Execution: `cpu_exec.animate_execute_tick()` for burst duration, then `cpu_exec.animate_complete(pid=...)`.
- Metric Update: When a process completes, update its row in the metric table: `metric_table.animate_update_process_metrics(pid=..., ct=..., tat=..., wt=...)`.

#### 4. Strict Component Method Contracts & Action Semantics
Specify ONLY valid, registered component methods. Do NOT invent unsupported methods:
- **`ProcessCardVisual`**:
  * Supported methods: `animate_set_state(new_state)` (valid states: "NEW", "READY", "RUNNING", "TERMINATED"), `animate_update_remaining_time(rem)`, `animate_highlight()`.
  * FORBIDDEN (DO NOT CALL): `highlight_pid_at()`, `highlight_bt()`, `highlight_state()`, `initialize_structure()`.
  * State transitions must follow: NEW -> READY -> RUNNING -> TERMINATED.
  * NEVER duplicate consecutive state transitions (e.g. do not call `set_state("RUNNING")` twice in a row).
- **`ReadyQueueVisual`**:
  * Supported methods: `animate_enqueue(item=...)`, `animate_dequeue()`.
  * Enqueue and dequeue must follow chronological order.
- **`CPUExecutionVisual`**:
  * Supported methods: `animate_dispatch(pid=...)`, `animate_execute_tick(remaining_time=...)`, `animate_complete(pid=...)`, `animate_idle()`.
  * FORBIDDEN (DO NOT CALL): `highlight_scheduler()`, `highlight_dispatcher()`, `highlight_context_switch()`.
  * NEVER dispatch the same process twice while it is running or after it completed.
- **`GanttChartVisual`**:
  * Supported methods: `animate_add_block(pid=..., start_time=..., end_time=...)`, `animate_advance_time(new_time=...)`.
- **`MetricTableVisual`**:
  * Supported methods: `animate_update_process_metrics(pid=..., ct=..., tat=..., wt=...)`, `animate_show_averages()`, `highlight_cell_avg_wt()`, `highlight_cell_avg_tat()`, `highlight_cell_avg_rt()`.

#### 4. Highlighting Discipline (NO UNNECESSARY PULSING)
- Animate with purpose: Use animations to show state changes, queue movements, and timeline progress.
- Highlight ONLY when directing attention to a specific concept being explained.
- DO NOT force an `Indicate` or `highlight` action onto every single narration line.
- Avoid consecutive repetitive highlights on the same component.

#### 5. Viewport Allocation & Layout Discipline
- **`MetricTableVisual`**: Belongs in `dock_main`. NEVER dock it to `dock_inspector` (which is only 1.0 unit high and causes extreme downscaling).
- **`GanttChartVisual`**: When constructing or explaining the timeline as the primary visual, place it in `dock_main` or ensure adequate viewport space.
- - **CRITICAL LAYOUT INVARIANT**: NEVER dock both `MetricTableVisual` AND `GanttChartVisual` to `dock_inspector` at the same time! `dock_inspector` is only 1.0 unit high; `MetricTableVisual` belongs in `dock_main`.
- Space out mathematical formula cards so they do not accumulate in a single cramped dock.
"""


# ----------------------------------------------------------------------------
# Public entry point (signature unchanged)
# ----------------------------------------------------------------------------
def get_prompt2_storyboard(outline, reference_image_path=None, level="intermediate", analogy=None, component_manifest=None, topic_profile=None):
    norm_level = _norm_level(level)
    analogy_text = _clean_text(analogy)
    analogy_guidance = _analogy_guidance(analogy_text) if analogy_text else _direct_guidance()

    if topic_profile is not None:
        if isinstance(topic_profile, dict) and TopicProfile and TopicProfiler:
            profile = TopicProfiler.profile_from_data(topic_profile)
        else:
            profile = topic_profile
    else:
        profile = extract_topic_profile(outline)
    profile_guidance = _format_stage2_profile_guidance(profile)

    algo_state = extract_verified_algorithmic_state(outline)
    trace_guidance = ""
    if algo_state and isinstance(algo_state, dict):
        try:
            from src.domains import DomainTraceGuidanceRegistry
            trace_guidance = DomainTraceGuidanceRegistry.format_guidance(algo_state)
        except Exception:
            pass
        if not trace_guidance:
            algo = algo_state.get("algorithm")
            if algo == "dijkstra":
                trace_guidance = _format_dijkstra_storyboard_guidance(algo_state)
            elif algo == "fcfs":
                trace_guidance = _format_fcfs_storyboard_guidance(algo_state)

    if component_manifest is None:
        try:
            from src.domains import VisualRegistry
            component_manifest = VisualRegistry.get_schema_manifest()
        except Exception:
            component_manifest = {}

    manifest_section = _format_component_manifest(component_manifest)

    prompt = f"""
You are a professional educator and Manim animator. Convert the teaching outline below into a precise, step-by-step storyboard for the Manim animation system, adapted for a {norm_level.capitalize()} learner.

## Teaching Outline
{_outline_text(outline)}
"""

    if reference_image_path:
        prompt += """
## Reference Image Available
A reference image has been provided to guide the animation design.
- Study its diagrams, layout and representations and recreate the concepts with native Manim objects (never reference the image itself).
- Put the richest animation in the sections that benefit most from the image.
- Where the image and the outline disagree on facts, follow the outline.
"""

    prompt += f"""
## Storyboard Requirements
{LEVEL_GUIDANCE[norm_level]}
{analogy_guidance}

{profile_guidance}

{trace_guidance}

{manifest_section}

### Content Structure
- One storyboard section per outline section, same order, same ids. Titles: 4 to 7 words, format 'Sec N: ...'.
- Narration size: follow the outline's 'narration_budget' (lecture lines per section and words per line). If it is missing, use 5 to 7 lines of 22 to 35 words.
- 'lecture_lines' are SPOKEN by a text-to-speech voice:
  * one pedagogical beat per line, complete sentences, no bullet fragments;
  * avoid packing multiple distinct calculation steps, metric outcomes, or discrete algorithm operations into a single long line (> 20 words). If an explanation covers multiple results or metrics, separate them into consecutive lines so each has its own visual beat;
  * do NOT split sentences arbitrarily or damage conceptual continuity merely to shorten lines;
  * if an explanation legitimately requires a prolonged visual hold while the learner absorbs a complex displayed structure (e.g. inspecting a completed diagram, table, or chart), keep the line intact and tag the animation action with 'action: inspect' or 'pacing: inspection_emphasis' (e.g. 'L4-1 | FadeIn | ProceduralComponent(SummaryView) | action: inspect') so the pacing engine treats the static interval as an intentional pedagogical absorption window;
  * write mathematics in words ("x squared plus one", "the sum of the weights"), no LaTeX, no raw symbols, no emoji;
  * define a symbol or term the first time it is used; reuse the outline's 'notation' symbols consistently;
  * the lines of a section form a continuous explanation that ends by leading into the next section.
- 'captions' are SHOWN on the left side: exactly one per lecture line, at most 7 words, noun-phrase style, no formulas. They are what the viewer reads while listening.
- Do not apply any animation to lecture lines or captions; only their color changes when the matching visual appears.
{LAYOUT_CONTRACT}
{ANIMATION_FORMAT}

### Domain-Independent Visual Reasoning (CRITICAL)
Do NOT assume the topic is an algorithm or computer science. Analyze the educational domain and choose the representation that best teaches the concept:
- Discrete / Computational: State transition diagrams, network graphs, 2D memoization matrices, arrays, queues.
- Mathematical (e.g. Linear Algebra, Calculus, Statistics): Coordinate axes, plotted functions, vector transformations, geometric constructions, symbolic derivations.
- Chemical & Molecular (e.g. Kinetics, Equilibrium, Bonding): Molecular 2D structures, bond transformation flows, reaction energy profile curves, ICE concentration tables.
- Physical & Natural (e.g. Mechanics, Thermodynamics, Circuits): Free-body force diagrams, field lines, particle flows, circuit schematics, trajectory plots.
- Systems & Comparative (e.g. Concurrency vs Parallelism, Architectures): Side-by-side split execution timelines (dock_split_left vs dock_split_right), pipeline queues, resource flowcharts.

### Visual Realization & Anti-Text-Card Rules (CRITICAL)
- Each section in the outline provides a 'visual_idea'. The storyboard animations MUST implement and visually depict this 'visual_idea'.
- Meaningful Visual Representations Over Text-Card Dumping:
  * Do NOT replace algorithmic, mathematical, or scientific processes with collections of explanatory Text objects (e.g. NEVER emit Text(Select Smallest Distance), Text(Update Adjacent Distances), Text(Vertices=Intersections), or Text(Difference Quotient)).
  * Prefer semantic visual actions on registered domain components (e.g. GraphVisual.highlight_node('A'), Table.update_cell(row=1, col=2)) or native Manim primitives (Circle, Rectangle, Arrow, Dot, Line, Axes, FunctionGraph).
  * Text is permitted ONLY for concise labels (e.g. node names "City A", vertex "u", coordinates), annotations, formal definitions, and formula values.
  * Do NOT place numerous explanatory text objects in dock_main. Make the visual structure (graphs, plots, matrices, geometric shapes) carry the explanation rather than merely decorating narration with text cards.
- FORBIDDEN: Spoken narration must NEVER be duplicated verbatim into a text card. Text cards are ONLY permitted for formal definitions, theorems, or governing equations in dock_inspector (Tier 4).
- When explaining two concepts or comparing side-by-side (e.g. Concurrency vs Parallelism, Single Core vs Multi Core, Task 1 vs Task 2), place them in separate non-overlapping zones:
  * Left vs Right: Entity A in dock_split_left, Entity B in dock_split_right
  * Top vs Bottom: Entity A in dock_main, Entity B in dock_inspector
- If an entity replaces another in the same zone, explicitly declare: "FadeOut <old>; then FadeIn <new>".
- Keep 'layout_plan' explicit: specify exact non-overlapping viewport allocations for all units visible together.

### Meaningful Visual Progression Across Sections (CRITICAL)
Every section must have:
- A distinct teaching objective and a clear visual purpose.
- A meaningful visual change, development, or state transition that advances learner understanding.
- Visual continuity: Reusing the same underlying structure (e.g. a coordinate frame, network, or structural layout) across sections is encouraged, BUT the state within that structure MUST EVOLVE (e.g. initial setup -> active element highlighted -> state updated -> finalized structure).
- FORBIDDEN: Repeating identical initialization and highlighting sequences across consecutive sections is strictly prohibited. (e.g. repeatedly generating initialize_graph(...) and highlighting the same node 'A' in Section 1, Section 2, and Section 3 without visual progression is forbidden).

### Scene Transitions & Cleanliness (CRITICAL)
- Every step that replaces content must fade out the previous temporary units explicitly. At the start of each new lecture line, the screen contains only the units that line still needs.
- Do not place equations on top of diagrams; fade the diagram out first or put the equation in its own cell area that is free.
- Colors: background is #000000; use only light, high-contrast colors from 'visual_conventions.palette'. Never use dark colors on the black background.
- 'visual_conventions' lists every entity that appears in more than one section (shape, color, home area). Keep them identical across sections.

### Animation Effects
- FadeIn / FadeOut: introductions, definitions, clearing.  FadeIn(shift): sequential or directional flow.
- GrowFromCenter: core entities and focal shapes.  Indicate: key formulas or symbols.  Create / Write: lines, arrows, equations.

### Mathematical Correctness & Strict MathTex Rules (HIGHEST PRIORITY)
- Strict MathTex Formatting:
  * Every MathTex element MUST contain a valid raw LaTeX string literal enclosed in quotes, e.g. MathTex(r"k = A e^{{-\\frac{{E_a}}{{RT}}}}"), MathTex(r"f'(x) = \\lim_{{h \\to 0}} \\frac{{f(x+h) - f(x)}}{{h}}"), or MathTex(r"E_a = 75\\text{{ kJ/mol}}").
  * FORBIDDEN: NEVER split mathematical notation into comma-separated English tokens or Python pseudo-code (e.g. NEVER write MathTex(k, equals, A, times, exp, left_parenthesis, negative, E_a, ...)).
  * MathTex is for LaTeX expressions; keep Python variables and LaTeX strings conceptually distinct.
- Preservation of Outline Mathematical Values:
  * Take every definition, formula, intermediate calculation, and numeric value from the outline's 'worked_example_spec'. Do not invent new formulas, values or fabricated claims.
  * Any number or formula that appears in narration OR animation must be listed in the section's 'verified_values' together with its check (the computation, or the outline step it comes from). Narration and animation must show identical values.
  * Worked-example sections must reproduce 'worked_example_spec' step by step, with the same intermediate values and final result.
- If a requested visual would be mathematically misleading (for example a picture that implies a false property), choose a different visual.
- Do not claim more than the outline supports: keep the assumptions and domain of validity that the outline states.

## Output format
Output ONLY one valid JSON object (no markdown fences, no comments) with this structure. Section objects repeat for every outline section; 'lecture_lines', 'captions' and 'animations' contain as many entries as needed:
{_storyboard_schema(analogy_text)}

Hard checks before you answer:
1. len(captions) == len(lecture_lines) in every section.
2. Every section MUST contain 4 to 5 lecture lines (matching outline narration_budget); do NOT collapse to 2 lines.
3. Every lecture line number appears in at least one animation step tag (L<line>-<k>).
4. No step lists more than 4 units in 'on screen after', and nothing is placed outside B1..F6 core/edge rules.
5. Every number in lecture_lines or animations appears in 'verified_values'.
6. Every worked example section MUST explicitly state input data, state transitions, and step-by-step calculations.
7. Every section's animations actually implement the section's 'visual_idea' from the outline; NO bare text cards for concepts.
8. Simultaneous objects have distinct, non-overlapping grid cells or areas.
9. Strict MathTex Syntax: MathTex MUST ALWAYS contain valid raw LaTeX strings (e.g. MathTex(r"...")); NEVER split into comma-separated English tokens (e.g. MathTex(k, equals, A, ...)).
10. Strict 6-Field Animation Syntax: Every animation step MUST contain EXACTLY 5 pipe '|' delimiters separating 6 fields. Content replacements MUST use 'FadeOut <old>; then <effect>' entirely in the effect field with NO extra pipes.
11. Meaningful Visual Progression: Every section MUST contribute a distinct visual development. Repeating identical initialization and highlight sequences across consecutive sections is strictly forbidden.
12. Anti-Text-Card Rule: Do NOT dump spoken narration into collections of bare Text(...) objects in dock_main. Use semantic actions on domain components or native Manim primitives; reserve text for concise labels, annotations, or formulas in dock_inspector.
"""
    return prompt


# ----------------------------------------------------------------------------
# OPTIONAL: critic pass. Run it once on the storyboard (text-only, cheap) and use the corrected JSON.
# ----------------------------------------------------------------------------
def get_prompt2_verify_storyboard(outline, storyboard):
    result_schema = json.dumps(
        {
            "issues_found": [
                {
                    "section_id": "section_1",
                    "type": "concept | numeric | narration | sync | layout | consistency",
                    "detail": "what was wrong",
                    "fix": "what you changed",
                }
            ],
            "storyboard": "<the full corrected storyboard JSON object, same structure as the input>",
        },
        indent=4,
    )
    return f"""
You are a meticulous subject-matter reviewer and Manim storyboard editor.

Review the storyboard against the outline. Fix problems; do not rewrite things that are already correct.

## Outline
{_outline_text(outline)}

## Storyboard
{_outline_text(storyboard)}

## Checks (in priority order)
1. Conceptual correctness: wrong or misleading definitions, formulas, claims, units, or over-generalizations. Missing assumptions the outline requires.
2. Numerical correctness: recompute every number in 'verified_values', narration and animations. Narration and animation must match each other and the outline's worked example.
3. Narration quality: each lecture line is speakable (math in words), one idea, complete sentence, within the budget, and the lines flow as one explanation.
4. Sync & Visual Realization: every lecture line has at least one tagged animation step; the visuals actively realize the section's 'visual_idea' from the outline with shapes/diagrams/workflows rather than bare text cards.
5. Layout feasibility: steps stay in the right-side grid, respect the core zone / edge ring and capacity limits (max 4 units, labels <= 3 words, tables <= 4x4, graphs <= 6 nodes); simultaneous objects have distinct non-overlapping cell allocations; 'on screen after' is consistent from step to step and previous units are explicitly faded out.
6. Consistency: entity colors / shapes / home areas follow 'visual_conventions'; 'captions' match 'lecture_lines' one to one.
7. Syntax & Component Validity: MathTex must use valid raw LaTeX strings (e.g. MathTex(r"...")); transitions must use 'FadeOut <old>; then <effect>' with exactly 5 pipe delimiters (no extra pipes); no bare Text cards cluttering dock_main; every section must advance the visual representation.

## Output format
Output ONLY one valid JSON object:
{result_schema}
If nothing is wrong, return an empty 'issues_found' list and the storyboard unchanged.
"""


# ----------------------------------------------------------------------------
# Storyboard Numerical Fidelity Repair Prompt (Milestone 4)
# ----------------------------------------------------------------------------
def get_prompt2_storyboard_fidelity_repair(outline, storyboard, discrepancies):
    """
    Constructs a targeted prompt requesting Stage 2 storyboard correction
    to resolve specific numerical fidelity discrepancies with the Stage 1 outline.
    """
    disc_text_lines = []
    for i, d in enumerate(discrepancies, 1):
        if hasattr(d, "to_dict"):
            d_dict = d.to_dict()
        elif isinstance(d, dict):
            d_dict = d
        else:
            d_dict = {"reason": str(d)}

        sec = d_dict.get("section_id", "unknown")
        loc = d_dict.get("location", "")
        cat = d_dict.get("category", "")
        claim = d_dict.get("actual_claim", "")
        expected = d_dict.get("expected_value", "")
        unit = d_dict.get("expected_unit", "")
        reason = d_dict.get("reason", "")
        fix = d_dict.get("suggested_correction", "")

        disc_text_lines.append(
            f"Issue #{i} [{cat}] in {sec} ({loc}):\n"
            f"  - Actual Storyboard Claim: {claim}\n"
            f"  - Verified Expected Value: {expected} {unit or ''}\n"
            f"  - Problem: {reason}\n"
            f"  - Required Correction: {fix}\n"
        )

    discrepancies_str = "\n".join(disc_text_lines)

    algo_state = extract_verified_algorithmic_state(outline)
    extra_algo_instruction = ""
    if algo_state and isinstance(algo_state, dict):
        algo = algo_state.get("algorithm")
        if algo == "dijkstra":
            settled = algo_state.get("settled_order", [])
            final_d = algo_state.get("final_distances", {})
            extra_algo_instruction = f"""
6. Algorithmic Worked Example Fidelity (MANDATORY):
   - You MUST visually represent every canonical extraction beat in exact order: {settled}.
   - You MUST visually represent every outgoing edge relaxation and distance update using visible animation actions (e.g. relax_edge, update_node_label, update_distance) rather than mere narration.
   - Do NOT skip intermediate extractions or relaxations even if final distances ({final_d}) are shown.
   - For unsuccessful relaxations, show that candidate >= current distance and that NO distance update occurs.
"""
        elif algo == "fcfs":
            procs = [p.get("pid") for p in algo_state.get("processes", [])]
            gantt = [(b.get("pid"), b.get("start_time"), b.get("end_time")) for b in algo_state.get("gantt_chart", [])]
            avg_wt = algo_state.get("average_waiting_time", 0.0)
            avg_tat = algo_state.get("average_turnaround_time", 0.0)
            extra_algo_instruction = f"""
6. FCFS Scheduling Worked Example Fidelity (MANDATORY):
   - Process Specifications: Faithfully use exact PIDs ({procs}) and arrival/burst times from the trace.
   - Non-Preemptive Execution: Running processes must execute to completion; do NOT preempt running processes upon arrivals.
   - FIFO Queue Order: Enqueue and dispatch processes strictly according to arrival order.
   - Authoritative Gantt Blocks: Use exact timeline intervals: {gantt}.
   - Honest Metric Accounting: Do NOT reveal CT, TAT, or WT for any process before it finishes execution.
   - Summary Averages: Average Waiting Time = {avg_wt:.2f}, Average Turnaround Time = {avg_tat:.2f}.
   - Valid Component Operations: Use valid component actions (e.g. animate_enqueue, animate_dispatch, animate_add_block).
"""

    return f"""You are a meticulous educational script editor and Manim storyboard reviewer.

Your previous storyboard generation contained the following deterministic numerical fidelity discrepancies when compared against the authoritative Stage 1 outline:

{discrepancies_str}

## Authoritative Stage 1 Outline
{_outline_text(outline)}

## Current Storyboard to Fix
{_outline_text(storyboard)}

## Strict Correction Instructions
1. Correct all flagged numerical discrepancies across 'verified_values', 'animations' (MathTex and Text objects), and 'lecture_lines' (spoken narration).
2. Use the exact verified numerical values and units from the outline's 'worked_example_spec'. DO NOT revert to any unverified or hallucinated claims.
3. Ensure that spoken narration and displayed animations state identical numbers and units.
4. Keep the outline's section structure, layout plans, and pedagogical arc intact.
5. Output ONLY one valid JSON object containing the complete corrected storyboard with the same structure as the input storyboard:
{extra_algo_instruction}
{{
  "visual_conventions": {{ ... }},
  "sections": [ ... ]
}}
Do NOT wrap in markdown fences or include explanations. Output only raw JSON.
"""


def _detect_topic_domain(outline: Any) -> str:
    """Detects topic domain from outline to customize worked demonstration contracts."""
    raw = ""
    if isinstance(outline, str):
        raw = outline.lower()
    elif isinstance(outline, dict):
        raw = (
            str(outline.get("topic", ""))
            + " "
            + str(outline.get("title", ""))
            + " "
            + str(outline.get("slug", ""))
            + " "
            + str(outline.get("prompt", ""))
        ).lower()
    elif hasattr(outline, "__dict__"):
        d = getattr(outline, "__dict__", {})
        raw = (
            str(d.get("topic", ""))
            + " "
            + str(d.get("title", ""))
            + " "
            + str(d.get("slug", ""))
            + " "
            + str(d.get("prompt", ""))
        ).lower()
    else:
        raw = str(outline).lower()

    if any(k in raw for k in ("binary search", "binary_search", "sorted array", "low/high/mid")):
        return "binary_search"
    if any(k in raw for k in ("shortest job first", "sjf", "non-preemptive selection")):
        return "sjf"
    if any(k in raw for k in ("first come first served", "fcfs", "arrival-order")):
        return "fcfs"
    return "general"


# ----------------------------------------------------------------------------
# Milestone 3B: Typed Scene IR v1.0 Generation Prompt
# ----------------------------------------------------------------------------
def get_prompt2_scene_ir(
    outline,
    level="intermediate",
    analogy=None,
    component_manifest=None,
    topic_profile=None,
    format_as_sequence=True,
):
    """
    Constructs a dedicated, schema-grounded Stage 2 Scene IR generation prompt.
    Directly injects the authoritative JSON Schema from Scene IR models and
    real component capability manifests from VisualRegistry.
    """
    norm_level = _norm_level(level)
    analogy_text = _clean_text(analogy)
    analogy_guidance = _analogy_guidance(analogy_text) if analogy_text else ""

    # 1. Topic Profile Resolution
    if topic_profile is not None:
        if isinstance(topic_profile, dict) and TopicProfile and TopicProfiler:
            profile = TopicProfiler.profile_from_data(topic_profile)
        else:
            profile = topic_profile
    else:
        profile = extract_topic_profile(outline)
    profile_guidance = _format_stage2_profile_guidance(profile)

    # 2. Visual Registry Component Manifest
    if component_manifest is None:
        try:
            from src.domains import VisualRegistry
            component_manifest = VisualRegistry.get_schema_manifest()
        except Exception:
            component_manifest = {}

    comp_lines = ["## Available Visual Capabilities & Registered Components"]
    comp_lines.append("The following visual components are registered in VisualRegistry and available for instantiation:\n")
    if component_manifest:
        for domain, comps in component_manifest.items():
            comp_lines.append(f"### Domain: {domain.upper()}")
            for comp in comps:
                cid = comp.get("component_id", "Component")
                desc = comp.get("description", "").strip()
                params = comp.get("init_params", {})
                ops = comp.get("operations", {})
                vp = comp.get("suggested_viewport", "dock_main")

                param_items = []
                for k, v in params.items():
                    req_str = "REQUIRED" if v.get("required") else f"default={v.get('default')}"
                    param_items.append(f"{k} ({req_str}, {v.get('type', 'Any')})")
                param_str = ", ".join(param_items) if param_items else "none"

                comp_lines.append(f"- **{cid}** (Domain: `{domain}`): {desc}")
                comp_lines.append(f"  * Suggested Viewport: `{vp}`")
                comp_lines.append(f"  * initial_state parameters: {param_str}")
                if ops:
                    comp_lines.append("  * Supported component_op operations:")
                    for op_name, op_info in ops.items():
                        op_p = op_info.get("parameters", {})
                        p_strs = [f"{pk}={'REQUIRED' if pv.get('required') else pv.get('default')}" for pk, pv in op_p.items()]
                        comp_lines.append(f"    - `{op_name}({', '.join(p_strs)})`")
    else:
        comp_lines.append("- No specialized domain components currently registered.")

    comp_lines.append("\n### 4-Tier Visual Capability Hierarchy:")
    comp_lines.append("- Tier 1 (Registered): Use exact registered component ID with valid initial_state parameters.")
    comp_lines.append("- Tier 2 (Composition): Compose multiple entities across disjoint viewports.")
    comp_lines.append("- Tier 3 (Primitive Trees): Use declarative primitives: CompositeNodeGraph, PointerIndicator, CompositeDissectionSquare, PrimitiveShape, ProceduralComponent.")
    comp_lines.append("- Tier 4 (Conceptual Cards): Use ConceptualCard for definitions, governing equations, and invariants (properties: title, formula, badge, body).")
    comp_lines.append("- STRICT INVARIANT: Do NOT invent unregistered component names or unlisted component operations.")
    comp_lines.append("- MULTI-SECTION CONTRACT: Each section is compiled independently as its own standalone Manim scene. Every section MUST declare all visual entities it uses directly in its local 'entities' field. Do NOT leave 'entities' empty, and do NOT use 'inherited_entities' because cross-section entity reconstitution is not currently supported.")
    manifest_section = "\n".join(comp_lines)

    # 3. Exported JSON Schema Contract
    if format_as_sequence and SceneSequenceIR is not None:
        schema_dict = SceneSequenceIR.model_json_schema()
        schema_name = "SceneSequenceIR (Multi-Section Sequence)"
    elif export_scene_ir_json_schema is not None:
        schema_dict = export_scene_ir_json_schema()
        schema_name = "SceneIR (Section Scene)"
    else:
        schema_dict = {"title": "SceneIR", "version": "1.0"}
        schema_name = "SceneIR v1.0"

    schema_json = json.dumps(schema_dict, indent=2, ensure_ascii=False)

    topic_domain = _detect_topic_domain(outline)
    if topic_domain == "binary_search":
        trace_milestones_guidance = """      * Explicit Binary Search Trace Milestones: A binary search demonstration on a sorted array (e.g. [2, 5, 8, 12, 16, 23, 38] searching for 23) must explicitly communicate the complete ordered sequence:
        1) Initial search boundaries (low = 0, high = 6, interval [0, 6]);
        2) First midpoint evaluation (mid = 3);
        3) Reading/inspecting element at midpoint (array[3] = 12);
        4) Comparison (12 < 23, concluding target must be in the right half);
        5) Updating boundaries (low = 4, high = 6, active interval [4, 6]);
        6) Second midpoint evaluation (mid = 5);
        7) Reading/inspecting element at new midpoint (array[5] = 23);
        8) Target declaration (target found at index 5)."""
    elif topic_domain == "fcfs":
        trace_milestones_guidance = """      * Explicit FCFS CPU Scheduling Trace Milestones: An FCFS scheduling demonstration on processes [P1(AT=0, BT=4), P2(AT=1, BT=3), P3(AT=2, BT=1), P4(AT=3, BT=2)] must explicitly communicate the complete ordered sequence:
        1) Initial process queue and arrival order: P1 arrives at t=0 with BT=4;
        2) Execution of P1 from t=0 to t=4 without preemption, recording completion time CT=4, turnaround time TAT=4, waiting time WT=0;
        3) Dispatch and execution of P2 from t=4 to t=7, recording CT=7, TAT=6, WT=3;
        4) Dispatch and execution of P3 from t=7 to t=8, recording CT=8, TAT=6, WT=5;
        5) Dispatch and execution of P4 from t=8 to t=10, recording CT=10, TAT=7, WT=5;
        6) Completion of all processes and calculation of summary metrics: Average TAT = 5.75, Average WT = 3.25."""
    elif topic_domain == "sjf":
        trace_milestones_guidance = """      * Explicit Non-Preemptive SJF Scheduling Trace Milestones: An SJF scheduling demonstration on processes [P1(AT=0, BT=7), P2(AT=2, BT=4), P3(AT=4, BT=1), P4(AT=5, BT=4)] must explicitly communicate the complete ordered sequence and intermediate decisions:
        1) Non-preemptive execution of P1 from t=0 to t=7 (P1 runs to completion without preemption despite P2 arriving at t=2 and P3 at t=4), recording CT=7, TAT=7, WT=0;
        2) Ready queue evaluation at t=7: ready queue contains {P2(BT=4), P3(BT=1), P4(BT=4)}; P3 is explicitly selected because it has the shortest burst time (BT=1);
        3) Execution of P3 from t=7 to t=8, recording CT=8, TAT=4, WT=3;
        4) Ready queue tie-breaking evaluation at t=8: ready queue contains {P2(BT=4), P4(BT=4)}; since burst times tie, tie is resolved by earlier arrival time (AT(P2)=2 < AT(P4)=5), explicitly selecting P2;
        5) Execution of P2 from t=8 to t=12, recording CT=12, TAT=10, WT=6;
        6) Execution of P4 from t=12 to t=16, recording CT=16, TAT=11, WT=7;
        7) Completion of all processes and calculation of summary metrics: Average TAT = 8.0, Average WT = 4.0."""
    else:
        trace_milestones_guidance = """      * Algorithmic Trace Milestones: When demonstrating an algorithm with a worked example, you must translate all ordered algorithmic steps, state transitions, intermediate calculations, and final outcomes from 'worked_example_spec' into sequential timeline actions and substantive narration cues."""

    scheduling_layout_guidance = ""
    scheduling_preservation_guidance = ""
    array_preservation_guidance = ""
    if topic_domain in ("fcfs", "sjf"):
        scheduling_layout_guidance = """
       - For CPU Scheduling: Both `MetricTableVisual` (width ~3.8-6.8, height ~2.0) and `GanttChartVisual` (width ~3.7-6.8, height ~1.0) are large components. Do NOT vertically stack both `MetricTableVisual` and `GanttChartVisual` in a single column viewport. Instead, use separate viewports (e.g. `table_viewport` with `viewport_split_left` and `gantt_viewport` with `viewport_split_right`), and place decision/formula cards in `viewport_inspector`. NEVER dock both `MetricTableVisual` and `GanttChartVisual` into `dock_inspector` simultaneously."""
        scheduling_preservation_guidance = """
       - For CPU scheduling: Every process specified in 'worked_example_spec' (PID, Arrival Time, and Burst Time) must be declared and preserved in the initial state of `MetricTableVisual` (e.g., `initial_state.processes = [{{'pid': 'P1', 'arrival_time': 0, 'burst_time': 4}}, ...]`) or process table visual. Omitting process entries, arrival times, or burst times violates input preservation."""
        if topic_domain == "sjf":
            scheduling_preservation_guidance += """
       - For SJF Tie-Breaking Narration: At t=8, both P2 and P4 are in the ready queue with identical burst time BT=4. You MUST explicitly explain the tie and the earlier-arrival tie-break rule in the narration text_cue: e.g. "At time 8, P2 and P4 both have equal burst time 4; the tie is broken by earlier arrival time, selecting P2 because P2 arrived at t=2 while P4 arrived at t=5." Omitting the explicit mention of the tie or earlier arrival time violates SJF tie-breaking evaluation."""
    elif topic_domain == "binary_search":
        array_preservation_guidance = """
       - For sequence and array search: Every value in 'worked_example_spec.inputs.array' MUST be explicitly preserved and represented in the visual state and narration. Inactive, boundary, and non-target elements (e.g. the final element 38) must NOT be omitted. All values must be explicitly declared in the visual component state (e.g. as labeled nodes in CompositeNodeGraph, or explicit tick labels/pointers on NumberLineVisual, or within a data card) AND explicitly referenced in narration text cues so that all input values are preserved.
         1. Visual Array Presentation: Declare an explanatory array card (e.g. ConceptualCard with title "Input Array", formula "A = [2, 5, 8, 12, 16, 23, 38]", badge "Target: 23") in a secondary viewport (such as viewport_side or viewport_card) OR declare all 7 array values in CompositeNodeGraph nodes. Displaying tick marks 0 to 6 on a number line alone does NOT preserve array values.
         2. Narration Recitation: In the first worked demonstration beat (initial bounds), the narration text_cue MUST explicitly recite the complete input array: "Given sorted array [2, 5, 8, 12, 16, 23, 38] and target 23, initial search bounds are low = 0 and high = 6." All 7 array elements including boundary value 38 must be spoken and visually declared.
         3. Duration Budget Calculation: For worked demonstration sections with multiple detailed algorithmic beats (e.g. 5 steps spanning 75 seconds), calculate and set 'duration_budget' to at least 90.0 seconds (e.g. 90.0 to 100.0s). The duration_budget MUST strictly exceed the final beat's audio_timestamp_range end time. Never use 60.0s when 5 beats require 75s."""

    prompt = f"""You are an expert pedagogical designer and visual scene architect. Convert the Stage 1 teaching outline below into a typed, declarative Scene Intermediate Representation (Scene IR v1.0) conforming strictly to the supplied JSON Schema and component manifest.

## Teaching Outline
{_outline_text(outline)}

## Pedagogical & Visual Guidance
{LEVEL_GUIDANCE[norm_level]}
{analogy_guidance}
{profile_guidance}

{manifest_section}

## Authoritative JSON Schema Contract ({schema_name})
The generated output MUST conform strictly to the following JSON Schema:
```json
{schema_json}
```

## Two-Layer Validation Requirements
Your output will be verified by two independent deterministic validation layers:
1. Layer 1 (Structural JSON Schema Validation):
   - Strict `additionalProperties: false` is enforced on all structural objects.
   - Do NOT emit hallucinated or unexpected structural keys.
   - Enforce non-negative timing values (duration >= 0, audio timestamps >= 0).
   - DURATION BUDGET CONTRACT: 'duration_budget' must be explicitly specified in SECONDS (e.g., 45.0 to 90.0 seconds per section, never minutes like 1.0). The duration_budget for each section MUST be strictly greater than or equal to the total scheduled duration of all timeline actions and narration anchors in that section.
   - Exact Section Mapping: The generated `sections` array in `SceneSequenceIR` MUST contain EXACTLY one scene per outline section, matching the outline section IDs and titles in the exact same sequence order (e.g. if the outline has 3 or 4 sections, generate exactly 3 or 4 sections; do NOT split individual steps into separate scene sections or invent extra sections like scene_5, scene_6).
   - Concise Declarative Representation: Keep each section concise and focused. Do not repeat redundant entity definitions or declare excessive duplicate cards. Each worked demonstration section should use 1-2 primary visual entities (e.g. MetricTableVisual and GanttChartVisual for CPU scheduling) and 4-6 timeline steps. This ensures the IR is well-structured, within generation budgets, and avoids output truncation.
   - Version must be "1.0".
2. Layer 2 (Semantic Gate 3.5 & Payload Validation):
   - Local Entity Declaration & Section Independence: Each section is compiled independently into its own Manim scene. Therefore, EVERY section must declare all visual entities it uses directly in its local `entities` field. Do NOT leave `entities` empty.
   - Prohibited Entity Inheritance: `inherited_entities` must NOT be used because cross-section entity reconstitution is not currently supported in the compilation engine. Even if an entity visually carries over from a previous section, declare it locally in that section's `entities` array.
   - Scope Identifiers: All entity IDs and timeline step IDs must be strictly unique within the scene.
   - Viewport References: Every entity's `viewport` MUST reference a viewport declared in `viewports`.
   - Sub-slots: If an entity declares `slot`, the slot must exist in the referenced viewport's `sub_slots`.
   - Parent-Child DAG: `parent_id` must reference an existing entity; no entity may parent itself; the containment hierarchy MUST form a Directed Acyclic Graph (DAG) with ZERO circular containment cycles.
   - Action Target Integrity: Every action in the timeline MUST target a locally declared entity in the current section's `entities` list.
   - Payload Validation: `initial_state` and action `parameters` must use ONLY the supported parameters declared in the component manifest.
   - Anti-Text-Card Rule: Do NOT dump spoken narration into text cards in main viewports. Use visual representations (graphs, arrays, curves, shapes) to teach concepts.
   - Visual Progression: Every section must contribute distinct visual development. Repeating identical initializations across sections is prohibited.
   - Algorithmic Worked Demonstration Contract:
     * When translating an algorithmic outline with a 'worked_example_spec' (such as Binary Search, sorting, or scheduling), you MUST translate the complete step-by-step trace into explicit, ordered Scene IR timeline actions and substantive narration cues.
     * Numerical & Computational Fidelity: Preserve all numerical inputs, boundary indices, intermediate calculations, element lookups, comparisons, and final outcomes from 'worked_example_spec'. Never replace concrete numerical steps with vague placeholders (e.g. do NOT write generic text cues like "Calculate midpoint", "Update pointers", or "Find target"). The text_cue must explicitly explain the specific calculation (e.g., "Calculate midpoint: mid = floor((0 + 6) / 2) = 3; read array[3] = 12", "Compare array[3]=12 with target 23; since 12 < 23, update low to 4", "Recalculate mid = floor((4 + 6) / 2) = 5; read array[5] = 23", "array[5]=23 matches target 23; target found at index 5").
     * Meaningful Visual Operations (Anti-Generic-Pulse Rule): Do NOT apply identical generic highlights (e.g. Indicate on the entire canvas/line) across all beats. Use registered semantic animation operations on your entities:
       - For NumberLineVisual: use `animate_show_interval(start, end)` to show active intervals and interval halving; use `animate_add_pointer(name, val)` and `animate_move_pointer(name, new_val)` to show pointer movements (e.g. low, high, mid); use `animate_highlight_number(val)` to highlight evaluated indices/values.
       - Alternatively, compose with ConceptualCard / FormulaCard (title, formula, badge) or PointerIndicator to display mathematical formulas, state badges, and pointer indicators.
{trace_milestones_guidance}
     * Do NOT hardcode fixture-specific entity IDs (e.g. you may name your entities array_visual, pointer_low, search_line, formula_display, etc.), but you MUST use actual registered components and operations from the manifest.
   - Worked Example Spatial Layout & Composition Contract:
     * Prefer Built-in Component Operations: When using `NumberLineVisual` (or other registered domain components), prefer its supported built-in pointer and interval operations (`animate_add_pointer(name, val)`, `animate_move_pointer(name, new_val)`, `animate_show_interval(start, end)`, `animate_highlight_number(val)`) when they can represent the required algorithmic state. These operations operate directly on the visual component and do not consume additional viewport layout budget.
     * Avoid Multiple Independent Root Entities: Avoid creating multiple independent root-level entities for elements that can be represented within one visual component. For example, do NOT declare separate standalone `PointerIndicator` entities when `NumberLineVisual` provides native pointer operations.
     * Anti-Vertical-Overcrowding Rule: Do NOT vertically stack a full-width number line, multiple pointer objects, and explanatory cards in one constrained viewport under `flow: column`. In a 16:9 viewport with vertical padding (available height ~5.2 units), vertically stacking 5 or more root entities creates an unresolvable spatial deficit and will cause the layout solver to fail closed with `LAYOUT_OVERCROWDING_ERROR`.
     * Supported Layout Structures: If separate entities are necessary (e.g., an array visual and a separate calculation card, or a process table and a Gantt chart), arrange them using ONLY layout structures supported by the current schema and compiler:
       - For array/interval search: assign entities to separate viewports (e.g. `viewport_main` for the primary visual and a secondary viewport such as `viewport_side` or `viewport_card` for cards).{scheduling_layout_guidance}
       - Or attach secondary entities to a parent using `parent_id` (e.g., `parent_id: "<base_visual_id>"`).
     * Sizing & Boundary Constraints: Ensure the declared layout fits the actual viewport dimensions without spatial deficits or boundary collisions.
     * Information Completeness Preserved: Do NOT remove or omit required algorithmic information merely to make the layout fit. Communicate explanatory calculations and comparisons through substantive narration cues (`text_cue`) and sequential timeline actions rather than creating excessive persistent cards.
   - Input & Discrete Value Preservation Contract:
     * When a worked example contains an input array, list, sequence, or table in 'worked_example_spec.inputs':
       - For array/search topics: Every supplied array element must be represented in the relevant visual component's initial state or equivalent rendered visual representation. Displaying indices alone (e.g., only indices 0, 1, 2... on tick marks) is strictly INSUFFICIENT. Displaying only the target value or midpoint value is strictly INSUFFICIENT. Merely mentioning array values in narration text cues or metadata does NOT satisfy visual input preservation. The generated visual representation must retain the exact original order and complete values of all supplied input elements (e.g., via `CompositeNodeGraph` elements `[{{"val": "..."}}, ...]`, or `NumberLineVisual` ticks/labels/pointers, or data cards/tables containing the complete sequence).{array_preservation_guidance}{scheduling_preservation_guidance}
   - No Python/Manim Code: Scene IR is a declarative intermediate representation; do NOT write executable Python, Manim commands, or LaTeX double-braces.

## Output Format
Output ONLY one valid JSON object conforming to the {schema_name} schema.
Do NOT include markdown fences, comments, or conversational text. Output only raw JSON.
"""
    return prompt


# ----------------------------------------------------------------------------
# Milestone 5.2.2: Bounded Scene IR Diagnostic Repair Prompt
# ----------------------------------------------------------------------------
def get_prompt2_scene_ir_repair(
    outline: Any,
    failed_output: Union[str, Dict[str, Any]],
    processing_result: Any,
    level: str = "intermediate",
    analogy: Optional[str] = None,
    topic_profile: Optional[Any] = None,
) -> Optional[str]:
    """
    Constructs a targeted, bounded repair prompt for Stage 2 Scene IR generation.
    Instructs the LLM to fix specific structural, semantic, or payload discrepancies
    identified by deterministic validation layers.
    Enforces a strict bounded repair policy:
    - Never used for raw JSON syntax errors (fails fast).
    - Maximum 2 repair retries.
    - Explicitly lists errors and field paths.
    """
    if processing_result is None or getattr(processing_result, "is_valid", False):
        return None

    # Malformed JSON or input type failures fail fast; repair prompts are not sent for syntax breakdown
    structural_errors = getattr(processing_result, "structural_errors", [])
    for err in structural_errors:
        if err.get("error_code") in ("JSON_PARSE_ERROR", "INVALID_INPUT_TYPE"):
            return None

    all_errors = getattr(processing_result, "all_errors", [])
    if not all_errors:
        return None

    error_lines = []
    for idx, err in enumerate(all_errors, 1):
        ecode = err.get("error_code", "ERROR")
        msg = err.get("message", "")
        fpath = err.get("field_path", "")
        eid = err.get("entity_id")
        param = err.get("parameter")
        sec_id = err.get("section_id")
        sec_str = f" [section: `{sec_id}`]" if sec_id else ""
        loc_str = f" at `{fpath}`" if fpath else ""
        ent_str = f" [entity: `{eid}`]" if eid else ""
        param_str = f" [param: `{param}`]" if param else ""
        error_lines.append(f"{idx}. **[{ecode}]**{sec_str}{loc_str}{ent_str}{param_str}: {msg}")

    error_block = "\n".join(error_lines)

    if isinstance(failed_output, dict):
        failed_json_str = json.dumps(failed_output, indent=2, ensure_ascii=False)
    else:
        failed_json_str = str(failed_output)

    norm_level = _norm_level(level)

    prompt = f"""You are an expert pedagogical designer and visual scene architect. Your previous Stage 2 Scene IR generation failed deterministic validation with the following specific errors:

## Validation Diagnostics ({len(all_errors)} Error(s) Found)
{error_block}

## Previous Generated Scene IR Output
```json
{failed_json_str}
```

## Bounded Repair Instructions
1. Address and resolve EVERY flagged diagnostic error above.
2. Correct the exact field paths and parameters indicated in the diagnostics:
   - For `UNSUPPORTED_CAPABILITY` or empty entities: each section is compiled independently into its own Manim scene. Every section MUST declare all visual entities it uses directly in its local `entities` list. Do NOT leave `entities` empty, and do NOT use `inherited_entities`.
   - For `UNSUPPORTED_COMPONENT_OPERATION`: use ONLY operations explicitly listed as supported for the targeted component in the diagnostic message.
   - For `INVALID_PAYLOAD_PARAM_TYPE`: ensure values match the expected manifest types (e.g., float, int, str). Note that booleans are NOT acceptable where numeric types are expected.
   - For `UNRECOGNIZED_PAYLOAD_PARAM` or `MISSING_REQUIRED_PAYLOAD_PARAM`: verify initial_state and action parameters against declared component contracts.
   - For `UNDEFINED_VIEWPORT_REFERENCE` or `UNDEFINED_SLOT_REFERENCE`: ensure viewports are declared in `viewports` before referencing.
   - For `UNDEFINED_ANIMATION_TARGET`: actions must target declared entities in the current section's `entities` list.
   - For duration or timing budget errors: ensure 'duration_budget' is specified strictly in SECONDS (e.g. 45.0 to 100.0s) and is strictly greater than or equal to the total scheduled timeline duration.
   - For algorithmic worked examples: ensure all numerical calculations, comparisons, interval updates, tie-breaking explanations (such as equal burst time broken by earlier arrival time for SJF at t=8), and final results from 'worked_example_spec' are explicitly executed in timeline actions and substantive narration cues, avoiding generic full-visual highlights.
   - For layout and overcrowding: prefer built-in component operations (e.g. on NumberLineVisual) over multiple independent root entities; never vertically stack a number line, multiple pointer entities, and cards in a single constrained column viewport; for CPU scheduling, separate MetricTableVisual and GanttChartVisual into distinct viewports (e.g. table_viewport and gantt_viewport).
   - For input preservation: ensure all input values from 'worked_example_spec.inputs' (every element of inputs.array for sequence search; all process PIDs, Arrival Times, and Burst Times for CPU scheduling) are visually declared and represented in component state and text cues. Inactive, boundary, and non-target elements must NOT be omitted.
3. Do NOT invent new unregistered component names or operations.
4. Preserve all valid, error-free viewports, entities, and timeline steps.
5. Do NOT include markdown fences, conversational text, or explanations. Output ONLY the repaired raw JSON object conforming to the Scene IR contract.
"""
    return prompt


