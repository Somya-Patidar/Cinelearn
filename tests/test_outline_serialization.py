"""Unit and regression tests for TeachingOutline deserialization, serialization,
and data transfer between Stage 1 and Stage 2 in CineLearn.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pytest
from src.agent import TeachingOutline, TeachingVideoAgent, RunConfig
from prompts.stage2 import get_prompt2_storyboard


SAMPLE_STAGE1_BLUEPRINT = {
    "topic": "Dijkstra's Algorithm",
    "topic_type": "algorithm_or_process",
    "target_audience": "Undergraduate computer science students",
    "learning_level": "intermediate",
    "estimated_duration_minutes": 6.5,
    "narration_budget": {
        "sections": 6,
        "lines_per_section": 5,
        "words_per_line": [22, 35],
    },
    "formal_definition": (
        "Dijkstra's algorithm is a greedy single-source shortest path algorithm for "
        "directed or undirected graphs with non-negative edge weights."
    ),
    "prerequisites": [
        "Graph representation with vertices V and edges E",
        "Weighted adjacency list or matrix",
        "Priority queue / Min-heap operations",
    ],
    "notation": {
        "G": "Graph G = (V, E)",
        "V": "Set of vertices",
        "E": "Set of weighted edges",
        "dist[u]": "Tentative shortest distance from source s to vertex u",
        "Q": "Min-priority queue of unvisited vertices",
        "w(u, v)": "Non-negative weight of edge from u to v",
    },
    "common_misconceptions": [
        {
            "misconception": "Dijkstra's algorithm works correctly with negative edge weights.",
            "correction": (
                "Negative edge weights can invalidate greedy optimality because a visited "
                "vertex's distance might later be decreased. Use Bellman-Ford instead."
            ),
        }
    ],
    "worked_example_spec": {
        "setup": "4-node directed graph: A (source), B, C, D",
        "inputs": {
            "source": "A",
            "edges": [
                {"u": "A", "v": "B", "w": 4},
                {"u": "A", "v": "C", "w": 2},
                {"u": "C", "v": "B", "w": 1},
                {"u": "B", "v": "D", "w": 5},
                {"u": "C", "v": "D", "w": 8},
            ],
        },
        "steps": [
            {
                "step": "Initialize dist: dist[A]=0, dist[B]=inf, dist[C]=inf, dist[D]=inf",
                "result": "dist={A:0, B:inf, C:inf, D:inf}",
            },
            {
                "step": "Extract min A (dist 0), relax neighbors C (w=2) and B (w=4)",
                "result": "dist={A:0, B:4, C:2, D:inf}",
            },
            {
                "step": "Extract min C (dist 2), relax neighbors B (2+1=3 < 4) and D (2+8=10)",
                "result": "dist={A:0, B:3, C:2, D:10}",
            },
            {
                "step": "Extract min B (dist 3), relax neighbor D (3+5=8 < 10)",
                "result": "dist={A:0, B:3, C:2, D:8}",
            },
            {
                "step": "Extract min D (dist 8), no outgoing edges",
                "result": "Final shortest paths: dist[A]=0, dist[B]=3, dist[C]=2, dist[D]=8",
            },
        ],
        "final_result": "Shortest distances from A: B=3, C=2, D=8",
        "verification": "Independent path enumeration confirms A->C->B=3, A->C->B->D=8 are minimal.",
    },
    "story_framework": {
        "premise": "A courier navigating an archipelago with ferry transit times between islands.",
        "protagonist_goal": "Deliver urgent medical cargo from Island A to Island D in minimum total transit time.",
        "entity_mappings": {
            "Islands": "Vertices V",
            "Ferry routes": "Weighted edges E with transit time w",
            "Port logbook": "dist[] array tracking minimum known transit times",
            "Next departure board": "Priority queue Q",
        },
        "story_progression": "Courier departs Island A, evaluates nearest ferry connections, discovers faster detour via Island C, and reaches Island D.",
    },
    "analogy_evaluation": {
        "requested_analogy": "Island courier with ferry routes",
        "is_suitable": True,
        "adopted_analogy": "Island courier navigating weighted ferry routes",
        "rationale": "Directly models non-negative travel times and greedy exploration without misleading abstractions.",
        "limitations": "Ferry schedules do not involve negative travel times, which naturally matches Dijkstra's non-negative weight constraint.",
    },
    "analogy_used": "Island courier with ferry routes",
    "analogy_breakdown": "If a ferry had a negative travel time (time travel), the courier's greedy choices would fail.",
    "problem_statement": "Find the shortest paths from a single source vertex to all other vertices in a weighted graph.",
    "assumptions": ["All edge weights are non-negative: w(u, v) >= 0"],
    "invariants": ["For any settled vertex u, dist[u] is the true shortest distance from source."],
    "mental_model": "Expanding wave or ripple front moving outwards from the source.",
    "sections": [
        {
            "id": "section_1",
            "title": "Sec 1: The Shortest Path Problem",
            "goal": "Understand why greedy shortest path exploration is required.",
            "story_step": "Courier examines archipelago map with ferry transit times.",
            "content": "Formal definition of single-source shortest path problem on graphs.",
            "key_points": ["Graph representation G=(V,E)", "Non-negative edge weights"],
            "example": "Source vertex A with neighbors B and C",
            "misconception_addressed": "None",
            "visual_idea": "Graph diagram with 4 nodes and weighted edges",
        },
        {
            "id": "section_2",
            "title": "Sec 2: Priority Queue and Relaxation",
            "goal": "Master the edge relaxation condition.",
            "story_step": "Courier updates port logbook when finding a faster connection.",
            "content": "Relaxation formula: dist[v] = min(dist[v], dist[u] + w(u,v)).",
            "key_points": ["Min-heap priority queue", "Relaxation updates tentative distances"],
            "example": "Relaxing edge C->B improves dist[B] from 4 to 3",
            "misconception_addressed": "None",
            "visual_idea": "Priority queue table updating side-by-side with graph",
        },
    ],
}


def test_full_stage1_blueprint_deserialization():
    """Verify that all Stage 1 pedagogical blueprint fields survive deserialization into TeachingOutline."""
    outline = TeachingOutline.from_dict(SAMPLE_STAGE1_BLUEPRINT)

    assert outline.topic == "Dijkstra's Algorithm"
    assert outline.topic_type == "algorithm_or_process"
    assert outline.target_audience == "Undergraduate computer science students"
    assert outline.learning_level == "intermediate"
    assert outline.estimated_duration_minutes == 6.5
    assert outline.formal_definition.startswith("Dijkstra's algorithm is a greedy")
    assert len(outline.prerequisites) == 3
    assert "Graph representation with vertices V and edges E" in outline.prerequisites
    assert outline.notation["dist[u]"] == "Tentative shortest distance from source s to vertex u"
    assert len(outline.common_misconceptions) == 1
    assert "negative edge weights" in outline.common_misconceptions[0]["misconception"].lower()
    assert outline.worked_example_spec["setup"] == "4-node directed graph: A (source), B, C, D"
    assert len(outline.worked_example_spec["steps"]) == 5
    assert outline.analogy_evaluation["is_suitable"] is True
    assert outline.analogy_used == "Island courier with ferry routes"
    assert outline.problem_statement.startswith("Find the shortest paths")
    assert outline.assumptions == ["All edge weights are non-negative: w(u, v) >= 0"]
    assert len(outline.invariants) == 1
    assert outline.mental_model == "Expanding wave or ripple front moving outwards from the source."
    assert len(outline.sections) == 2


def test_nested_notation_and_worked_example_intact():
    """Verify that deeply nested structures inside notation and worked_example_spec remain exact."""
    outline = TeachingOutline.from_dict(SAMPLE_STAGE1_BLUEPRINT)

    # Notation
    assert isinstance(outline.notation, dict)
    assert outline.notation["G"] == "Graph G = (V, E)"
    assert outline.notation["w(u, v)"] == "Non-negative weight of edge from u to v"

    # Worked example spec
    spec = outline.worked_example_spec
    assert isinstance(spec, dict)
    assert spec["inputs"]["source"] == "A"
    assert len(spec["inputs"]["edges"]) == 5
    assert spec["inputs"]["edges"][2] == {"u": "C", "v": "B", "w": 1}
    assert spec["steps"][1]["step"].startswith("Extract min A")
    assert spec["steps"][2]["result"] == "dist={A:0, B:3, C:2, D:10}"
    assert spec["final_result"] == "Shortest distances from A: B=3, C=2, D=8"
    assert "Independent path enumeration" in spec["verification"]


def test_story_framework_and_analogy_evaluation_preserved():
    """Verify story_framework, entity mappings, and analogy evaluation are preserved."""
    outline = TeachingOutline.from_dict(SAMPLE_STAGE1_BLUEPRINT)

    sf = outline.story_framework
    assert isinstance(sf, dict)
    assert sf["protagonist_goal"] == "Deliver urgent medical cargo from Island A to Island D in minimum total transit time."
    assert sf["entity_mappings"]["Port logbook"] == "dist[] array tracking minimum known transit times"
    assert sf["entity_mappings"]["Next departure board"] == "Priority queue Q"

    ae = outline.analogy_evaluation
    assert isinstance(ae, dict)
    assert ae["is_suitable"] is True
    assert "negative travel times" in ae["limitations"]
    assert outline.analogy_breakdown.startswith("If a ferry had a negative travel time")


def test_optional_missing_fields_safe_handling():
    """Verify that missing optional fields do not crash and supply safe defaults."""
    minimal = {"topic": "Binary Search"}
    outline = TeachingOutline.from_dict(minimal)

    assert outline.topic == "Binary Search"
    assert outline.target_audience == ""
    assert outline.sections == []
    assert outline.learning_level == "intermediate"
    assert outline.formal_definition == ""
    assert outline.notation == {}
    assert outline.prerequisites == []
    assert outline.common_misconceptions == []
    assert outline.worked_example_spec == {}
    assert outline.story_framework is None
    assert outline.analogy_evaluation == {}
    assert outline.problem_statement == ""
    assert outline.assumptions == []
    assert outline.invariants == []
    assert outline.mental_model == ""

    # Empty payload
    empty_outline = TeachingOutline.from_dict({})
    assert empty_outline.topic == ""
    assert empty_outline.sections == []

    # Non-dict payload
    invalid_outline = TeachingOutline.from_dict(None)
    assert invalid_outline.topic == ""


def test_legacy_backward_compatibility():
    """Verify existing legacy instantiation patterns continue to work without errors."""
    # Positional legacy instantiation
    legacy_pos = TeachingOutline(
        "Calculus Limits",
        "High School Students",
        [{"id": "section_1", "title": "Definition"}],
    )
    assert legacy_pos.topic == "Calculus Limits"
    assert legacy_pos.target_audience == "High School Students"
    assert len(legacy_pos.sections) == 1
    assert legacy_pos.learning_level == "intermediate"

    # Keyword legacy instantiation (matching test_model_agnostic.py)
    legacy_kw = TeachingOutline(
        topic="Calculus Limits",
        target_audience="College Students",
        sections=[{"id": "section_1", "title": "Definition"}],
        learning_level="intermediate",
    )
    assert legacy_kw.topic == "Calculus Limits"
    assert legacy_kw.learning_level == "intermediate"

    # Legacy with story_framework (matching test_story_personalization.py)
    legacy_story = TeachingOutline(
        topic="Graph BFS",
        target_audience="Engineers",
        sections=[{"id": "s1"}],
        learning_level="advanced",
        story_framework={"protagonist_goal": "Find shortest path"},
    )
    assert legacy_story.story_framework["protagonist_goal"] == "Find shortest path"


def test_serialization_for_stage2_contains_complete_blueprint():
    """Verify that to_dict() and json serialization output contains all blueprint fields."""
    outline = TeachingOutline.from_dict(SAMPLE_STAGE1_BLUEPRINT)
    serialized_dict = outline.to_dict()

    # Core identification
    assert serialized_dict["topic"] == "Dijkstra's Algorithm"
    assert serialized_dict["topic_type"] == "algorithm_or_process"
    assert serialized_dict["learning_level"] == "intermediate"
    assert serialized_dict["estimated_duration_minutes"] == 6.5

    # Blueprint components required by Stage 2 prompt
    assert serialized_dict["formal_definition"] == outline.formal_definition
    assert serialized_dict["notation"] == outline.notation
    assert serialized_dict["prerequisites"] == outline.prerequisites
    assert serialized_dict["common_misconceptions"] == outline.common_misconceptions
    assert serialized_dict["worked_example_spec"] == outline.worked_example_spec
    assert serialized_dict["story_framework"] == outline.story_framework
    assert serialized_dict["analogy_evaluation"] == outline.analogy_evaluation
    assert serialized_dict["analogy_used"] == outline.analogy_used
    assert serialized_dict["analogy_breakdown"] == outline.analogy_breakdown

    # Internal tracking fields should not leak into serialized dict
    assert "extra_fields" not in serialized_dict

    # Verify JSON serializability
    json_str = json.dumps(serialized_dict, ensure_ascii=False, indent=2)
    reloaded = json.loads(json_str)
    assert reloaded["formal_definition"] == outline.formal_definition
    assert reloaded["worked_example_spec"]["inputs"]["source"] == "A"


def test_arbitrary_extra_fields_preserved():
    """Verify arbitrary custom fields survive deserialization and appear in to_dict and attribute access."""
    payload = dict(SAMPLE_STAGE1_BLUEPRINT)
    payload["custom_pedagogical_rule"] = "Always state theorem before proof"
    payload["domain_taxonomy"] = {"domain": "computer_science", "subdomain": "graph_theory"}

    outline = TeachingOutline.from_dict(payload)
    assert outline.custom_pedagogical_rule == "Always state theorem before proof"
    assert outline.domain_taxonomy["domain"] == "computer_science"

    serialized = outline.to_dict()
    assert serialized["custom_pedagogical_rule"] == "Always state theorem before proof"
    assert serialized["domain_taxonomy"]["subdomain"] == "graph_theory"


def test_duration_harmonization():
    """Verify estimated_duration and estimated_duration_minutes harmonize across legacy and blueprint formats."""
    # Only estimated_duration_minutes provided
    o1 = TeachingOutline.from_dict({"topic": "Sorting", "estimated_duration_minutes": 4.5})
    assert o1.estimated_duration_minutes == 4.5
    assert o1.estimated_duration == 4.5

    # Only estimated_duration provided
    o2 = TeachingOutline.from_dict({"topic": "Sorting", "estimated_duration": 5.0})
    assert o2.estimated_duration_minutes == 5.0
    assert o2.estimated_duration == 5.0


def test_stage2_prompt_receives_complete_blueprint():
    """Verify that Stage 2 prompt (get_prompt2_storyboard) receives the full blueprint fields."""
    outline = TeachingOutline.from_dict(SAMPLE_STAGE1_BLUEPRINT)
    serialized_outline = json.dumps(outline.to_dict(), ensure_ascii=False, indent=2)

    prompt = get_prompt2_storyboard(
        outline=serialized_outline,
        level="intermediate",
        analogy="Island courier",
    )

    # Verify that the Stage 2 prompt text embeds all critical blueprint sections
    assert "Dijkstra's algorithm is a greedy single-source shortest path algorithm" in prompt
    assert "dist[u]" in prompt
    assert "worked_example_spec" in prompt
    assert "courier navigating an archipelago" in prompt
    assert "All edge weights are non-negative" in prompt
    assert "Negative edge weights can invalidate greedy optimality" in prompt


def test_dict_like_access_and_membership():
    """Verify dictionary-style access (outline['key'], outline.get('key'), 'key' in outline)."""
    outline = TeachingOutline.from_dict(SAMPLE_STAGE1_BLUEPRINT)

    assert outline["topic"] == "Dijkstra's Algorithm"
    assert outline.get("formal_definition").startswith("Dijkstra's algorithm")
    assert outline.get("non_existent_field", "default_val") == "default_val"
    assert "formal_definition" in outline
    assert "notation" in outline
    assert "non_existent_key" not in outline


def test_agent_checkpoint_preserves_full_blueprint(tmp_path):
    """Verify that TeachingVideoAgent save_checkpoint and _load_existing_checkpoint_and_files
    retains 100% of the blueprint fields without degradation."""
    agent = TeachingVideoAgent(
        idx=999,
        knowledge_point="Dijkstra's Algorithm",
        folder=str(tmp_path),
        cfg=RunConfig(level="intermediate"),
    )
    agent.outline = TeachingOutline.from_dict(SAMPLE_STAGE1_BLUEPRINT)
    agent.save_checkpoint("outline_completed")

    # Inspect checkpoint.json on disk in agent.output_dir
    cp_file = agent.output_dir / "checkpoint.json"
    assert cp_file.exists()
    with open(cp_file, "r", encoding="utf-8") as f:
        cp_data = json.load(f)
    assert cp_data["outline"]["formal_definition"] == agent.outline.formal_definition
    assert cp_data["outline"]["notation"]["G"] == "Graph G = (V, E)"
    assert cp_data["outline"]["worked_example_spec"]["inputs"]["source"] == "A"

    # Create fresh agent in same directory to test reload
    agent2 = TeachingVideoAgent(
        idx=999,
        knowledge_point="Dijkstra's Algorithm",
        folder=str(tmp_path),
        cfg=RunConfig(level="intermediate"),
    )
    assert agent2.outline is not None
    assert agent2.outline.formal_definition == agent.outline.formal_definition
    assert agent2.outline.notation["G"] == "Graph G = (V, E)"
    assert agent2.outline.worked_example_spec["inputs"]["source"] == "A"


def test_agent_disk_outline_fallback_loading(tmp_path):
    """Verify that when loading an older checkpoint that had truncated fields,
    if outline.json exists on disk with full fields, they are safely merged and recovered."""
    # Compute output_dir
    dummy = TeachingVideoAgent(
        idx=998,
        knowledge_point="Dijkstra's Algorithm",
        folder=str(tmp_path),
        cfg=RunConfig(level="intermediate"),
    )
    target_dir = dummy.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # Write full outline.json to target_dir
    outline_json_path = target_dir / "outline.json"
    with open(outline_json_path, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_STAGE1_BLUEPRINT, f, ensure_ascii=False, indent=2)

    # Write truncated checkpoint (old format)
    old_truncated_cp = {
        "topic": "Dijkstra's Algorithm",
        "level": "intermediate",
        "analogy": "Island courier",
        "idx": 998,
        "stage": "outline_completed",
        "timestamp": 123456.0,
        "outline": {
            "topic": "Dijkstra's Algorithm",
            "target_audience": "Undergraduate",
            "sections": [],
            "learning_level": "intermediate",
            "analogy_used": "Island courier",
            "analogy_breakdown": "",
            "story_framework": None,
        },
    }
    cp_path = target_dir / "checkpoint.json"
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(old_truncated_cp, f, ensure_ascii=False, indent=2)

    # Loading agent should recover blueprint fields from disk outline.json
    agent = TeachingVideoAgent(
        idx=998,
        knowledge_point="Dijkstra's Algorithm",
        folder=str(tmp_path),
        cfg=RunConfig(level="intermediate"),
    )
    assert agent.outline is not None
    assert agent.outline.formal_definition.startswith("Dijkstra's algorithm is a greedy")
    assert agent.outline.notation["G"] == "Graph G = (V, E)"
    assert agent.outline.worked_example_spec["inputs"]["source"] == "A"


# ----------------------------------------------------------------------------
# Step 1.2: Defensive Serialization Hardening Tests
# ----------------------------------------------------------------------------

def test_extra_field_named_to_dict_does_not_shadow_method():
    """Verify an extra field named 'to_dict' does not overwrite the to_dict method,
    yet preserves its data inside extra_fields and serialization."""
    payload = dict(SAMPLE_STAGE1_BLUEPRINT)
    payload["to_dict"] = "unexpected_to_dict_string"

    outline = TeachingOutline.from_dict(payload)

    # 1. to_dict remains a callable method, NOT shadowed by string
    assert callable(outline.to_dict)
    assert not isinstance(outline.__dict__.get("to_dict"), str)

    # 2. to_dict() executes cleanly and serializes the full blueprint including the colliding key
    serialized = outline.to_dict()
    assert isinstance(serialized, dict)
    assert serialized["topic"] == "Dijkstra's Algorithm"
    assert serialized["formal_definition"].startswith("Dijkstra's algorithm")
    assert serialized["to_dict"] == "unexpected_to_dict_string"


def test_extra_fields_named_from_dict_and_get_do_not_shadow_methods():
    """Verify extra fields named 'from_dict' and 'get' do not overwrite class methods."""
    payload = {
        "topic": "Graph Theory",
        "from_dict": "data_for_from_dict",
        "get": "data_for_get",
    }
    outline = TeachingOutline.from_dict(payload)

    # Methods remain callable on class and instance
    assert callable(outline.get)
    assert callable(TeachingOutline.from_dict)
    assert outline.__dict__.get("get") is None
    assert outline.__dict__.get("from_dict") is None

    # Stored values are accessible via dict lookup
    assert outline["get"] == "data_for_get"
    assert outline.get("get") == "data_for_get"
    assert outline["from_dict"] == "data_for_from_dict"
    assert outline.get("from_dict") == "data_for_from_dict"

    # Serialization preserves them
    ser = outline.to_dict()
    assert ser["get"] == "data_for_get"
    assert ser["from_dict"] == "data_for_from_dict"


def test_dunder_and_private_keys_preserved_safely():
    """Verify dunder and private-looking keys are stored in extra_fields without corrupting instance attributes."""
    payload = {
        "topic": "Neural Networks",
        "__secret_weight": 0.42,
        "_internal_tag": "experimental",
    }
    outline = TeachingOutline.from_dict(payload)

    # Should NOT be dynamic attributes on the instance
    assert "__secret_weight" not in outline.__dict__
    assert "_internal_tag" not in outline.__dict__

    # Must be in extra_fields
    assert outline.extra_fields["__secret_weight"] == 0.42
    assert outline.extra_fields["_internal_tag"] == "experimental"

    # Must be accessible via get and __getitem__
    assert outline["__secret_weight"] == 0.42
    assert outline.get("_internal_tag") == "experimental"

    # Must be preserved in to_dict()
    ser = outline.to_dict()
    assert ser["__secret_weight"] == 0.42
    assert ser["_internal_tag"] == "experimental"


def test_invalid_identifiers_preserved_in_extra_fields():
    """Verify non-identifier keys (hyphenated, space, numeric) are preserved in extra_fields without syntax errors."""
    payload = {
        "topic": "Sorting",
        "space key": "val1",
        "k-e-y": "val2",
        "123numeric": "val3",
    }
    outline = TeachingOutline.from_dict(payload)

    assert outline["space key"] == "val1"
    assert outline["k-e-y"] == "val2"
    assert outline["123numeric"] == "val3"
    assert outline.get("space key") == "val1"

    ser = outline.to_dict()
    assert ser["space key"] == "val1"
    assert ser["k-e-y"] == "val2"
    assert ser["123numeric"] == "val3"


def test_declared_fields_retain_original_values_under_collisions():
    """Verify declared dataclass fields are never overwritten by extra fields."""
    payload = {
        "topic": "Real Analysis",
        "formal_definition": "Study of real numbers and limits",
        "learning_level": "advanced",
    }
    outline = TeachingOutline.from_dict(payload)
    assert outline.topic == "Real Analysis"
    assert outline.formal_definition == "Study of real numbers and limits"
    assert outline.learning_level == "advanced"
    assert "topic" not in outline.extra_fields
    assert "formal_definition" not in outline.extra_fields


def test_missing_key_behavior_and_no_method_leakage():
    """Verify get() and __getitem__() do not return class methods when keys are missing."""
    outline = TeachingOutline(topic="Calculus")

    # Methods must not leak via dictionary access
    with pytest.raises(KeyError):
        _ = outline["to_dict"]

    with pytest.raises(KeyError):
        _ = outline["from_dict"]

    with pytest.raises(KeyError):
        _ = outline["get"]

    with pytest.raises(KeyError):
        _ = outline["non_existent_field"]

    # get() returns default, not class methods
    assert outline.get("to_dict", "default_val") == "default_val"
    assert outline.get("from_dict", "default_val") == "default_val"
    assert outline.get("get", "default_val") == "default_val"
    assert outline.get("missing_key", None) is None

    # Membership checks
    assert "to_dict" not in outline
    assert "from_dict" not in outline
    assert "get" not in outline
    assert "topic" in outline
    assert "formal_definition" in outline


def test_callable_in_data_field_is_preserved():
    """Verify legitimate callable values in data fields are preserved and retrieved correctly."""
    scorer = lambda x: x * 2
    outline = TeachingOutline.from_dict({
        "topic": "Optimization",
        "scoring_fn": scorer,
    })

    assert outline["scoring_fn"] is scorer
    assert outline.get("scoring_fn") is scorer
    assert outline["scoring_fn"](5) == 10


def test_stage2_serialization_with_colliding_keys():
    """Verify the full Stage 2 prompt pipeline works seamlessly even if colliding keys exist."""
    payload = dict(SAMPLE_STAGE1_BLUEPRINT)
    payload["to_dict"] = "collision_test_val"
    payload["get"] = "getter_collision_test_val"

    outline = TeachingOutline.from_dict(payload)
    serialized_outline = json.dumps(outline.to_dict(), ensure_ascii=False, indent=2)

    prompt = get_prompt2_storyboard(
        outline=serialized_outline,
        level="intermediate",
        analogy="Island courier",
    )

    # Core blueprint must still be present
    assert "Dijkstra's algorithm is a greedy single-source shortest path algorithm" in prompt
    assert "dist[u]" in prompt
    assert "worked_example_spec" in prompt
    assert "collision_test_val" in prompt
    assert "getter_collision_test_val" in prompt


# ----------------------------------------------------------------------------
# Step 2B.2B: Mental Model Integration Tests
# ----------------------------------------------------------------------------

def test_no_user_analogy_leaves_self_analogy_empty_with_ai_mental_model(tmp_path):
    """Verify that when no user analogy is supplied, an AI-generated mental model
    does NOT overwrite self.analogy, leaving self.analogy empty for downstream stages."""
    agent = TeachingVideoAgent(
        idx=101,
        knowledge_point="Reaction Kinetics",
        folder=str(tmp_path),
        cfg=RunConfig(level="intermediate"),
    )
    assert agent.requested_analogy == ""
    assert agent.analogy == ""

    # Mock Stage 1 output containing an AI mental model
    mock_stage1_output = {
        "topic": "Reaction Kinetics",
        "learning_level": "intermediate",
        "mental_model": {
            "model_type": "physical_metaphor",
            "description": "Particles colliding over a potential energy barrier hill.",
            "entity_mappings": {"particles": "molecules", "hill": "activation energy Ea"},
            "breakdown_boundary": "Does not model quantum tunneling or steric factors.",
        },
        "analogy_evaluation": {
            "requested_analogy": "None",
            "is_suitable": None,
            "adopted_analogy": "Molecular collision model",
            "rationale": "Autonomous mental model selected",
            "limitations": "Does not model quantum tunneling",
        },
        "analogy_used": "Molecular collision model",
        "sections": [{"id": "section_1", "title": "Sec 1"}],
    }

    # Simulate generate_outline post-processing logic
    with patch.object(agent, "_request_api_and_track_tokens") as mock_api:
        mock_api.return_value = json.dumps(mock_stage1_output)
        agent.generate_outline()

    # self.analogy MUST remain empty because user requested no analogy
    assert agent.requested_analogy == ""
    assert agent.analogy == ""
    assert isinstance(agent.outline.mental_model, dict)
    assert agent.outline.mental_model["model_type"] == "physical_metaphor"
    assert agent.outline.analogy_evaluation["adopted_analogy"] == "Molecular collision model"


def test_user_analogy_preserves_requested_and_updates_adopted_analogy(tmp_path):
    """Verify that when user supplies an analogy, requested_analogy is preserved,
    and self.analogy is updated with the adopted/refined analogy."""
    agent = TeachingVideoAgent(
        idx=102,
        knowledge_point="Dijkstra's Algorithm",
        folder=str(tmp_path),
        cfg=RunConfig(level="intermediate", analogy="Island courier"),
    )
    assert agent.requested_analogy == "Island courier"
    assert agent.analogy == "Island courier"

    mock_stage1_output = {
        "topic": "Dijkstra's Algorithm",
        "learning_level": "intermediate",
        "mental_model": {
            "model_type": "physical_metaphor",
            "description": "Ferry network connecting islands with route travel times.",
            "entity_mappings": {"islands": "vertices", "ferries": "edges"},
            "breakdown_boundary": "Fails with negative ferry subsidies.",
        },
        "analogy_evaluation": {
            "requested_analogy": "Island courier",
            "is_suitable": True,
            "adopted_analogy": "Archipelago ferry network",
            "rationale": "Faithful mapping of shortest paths",
            "limitations": "No negative weights",
        },
        "analogy_used": "Archipelago ferry network",
        "sections": [{"id": "section_1", "title": "Sec 1"}],
    }

    with patch.object(agent, "_request_api_and_track_tokens") as mock_api:
        mock_api.return_value = json.dumps(mock_stage1_output)
        agent.generate_outline()

    assert agent.requested_analogy == "Island courier"
    assert agent.analogy == "Archipelago ferry network"
    assert agent.outline.mental_model["model_type"] == "physical_metaphor"


def test_structured_dictionary_mental_model_survives_serialization_and_checkpoint(tmp_path):
    """Verify structured dictionary mental model survives to_dict(), from_dict(),
    and checkpoint saving/reloading."""
    structured_mm = {
        "model_type": "geometric_model",
        "description": "Secant lines rotating into the tangent line as interval h approaches zero.",
        "entity_mappings": {"tangent line": "f'(x)", "curve": "f(x)"},
        "breakdown_boundary": "Fails at sharp corners or step discontinuities.",
    }

    payload = dict(SAMPLE_STAGE1_BLUEPRINT)
    payload["mental_model"] = structured_mm

    outline = TeachingOutline.from_dict(payload)
    assert isinstance(outline.mental_model, dict)
    assert outline.mental_model["model_type"] == "geometric_model"
    assert outline.mental_model["description"].startswith("Secant lines")

    # to_dict roundtrip
    d = outline.to_dict()
    assert d["mental_model"] == structured_mm

    # JSON serialization
    json_str = json.dumps(d)
    reloaded = TeachingOutline.from_dict(json.loads(json_str))
    assert reloaded.mental_model == structured_mm

    # Checkpoint save/load
    agent = TeachingVideoAgent(
        idx=103,
        knowledge_point="Derivatives",
        folder=str(tmp_path),
        cfg=RunConfig(level="beginner"),
    )
    agent.outline = outline
    agent.save_checkpoint("outline_completed")

    agent2 = TeachingVideoAgent(
        idx=103,
        knowledge_point="Derivatives",
        folder=str(tmp_path),
        cfg=RunConfig(level="beginner"),
    )
    assert agent2.outline is not None
    assert agent2.outline.mental_model == structured_mm


def test_legacy_string_mental_model_and_missing_field_backward_compatibility():
    """Verify legacy string-valued mental models and completely missing mental_model keys
    continue to deserialize and serialize cleanly."""
    # Legacy string
    o_str = TeachingOutline.from_dict({
        "topic": "Graph Search",
        "mental_model": "Wavefront expanding from source node",
    })
    assert o_str.mental_model == "Wavefront expanding from source node"
    assert o_str.to_dict()["mental_model"] == "Wavefront expanding from source node"

    # Missing field
    o_none = TeachingOutline.from_dict({
        "topic": "Graph Search",
    })
    assert o_none.mental_model == ""
    assert o_none.to_dict()["mental_model"] == ""


def test_stage2_prompt_guidance_selection_and_mental_model_embedding():
    """Verify Stage 2 receives mental model in outline text, selects direct guidance
    when analogy is empty, and selects analogy guidance when user analogy is present."""
    structured_mm = {
        "model_type": "physical_metaphor",
        "description": "Particles colliding over an activation hill.",
        "entity_mappings": {"particles": "molecules"},
        "breakdown_boundary": "Does not model quantum tunneling.",
    }
    payload = dict(SAMPLE_STAGE1_BLUEPRINT)
    payload["mental_model"] = structured_mm
    outline = TeachingOutline.from_dict(payload)
    serialized_outline = json.dumps(outline.to_dict(), ensure_ascii=False, indent=2)

    # 1. No user analogy -> triggers direct guidance with mental model instructions
    prompt_direct = get_prompt2_storyboard(
        outline=serialized_outline,
        level="intermediate",
        analogy="",
    )
    assert "Direct Technical Narration & Mental Model Guidance (no analogy requested)" in prompt_direct
    assert "Do NOT add fictional stories, roleplaying quests, or characters" in prompt_direct
    assert "Grounding via AI Mental Model:" in prompt_direct
    assert "Do NOT automatically turn a mental model into fictional role-playing" in prompt_direct
    assert "Do NOT introduce traveler tokens" in prompt_direct
    # Mental model must be present in outline text block
    assert "Particles colliding over an activation hill" in prompt_direct
    # Must NOT contain analogy traveler token commands in direct mode
    assert "an active agent/traveler token" not in prompt_direct

    # 2. Genuine user analogy -> triggers analogy guidance
    prompt_analogy = get_prompt2_storyboard(
        outline=serialized_outline,
        level="intermediate",
        analogy="Island courier",
    )
    assert "Story-Driven Storyboard & Narration Requirements" in prompt_analogy
    assert "Requested analogy / personal interest: Island courier" in prompt_analogy
    assert "Continuous Narrative Progression Across ALL Sections" in prompt_analogy
    assert "an active agent/traveler token" in prompt_analogy
    assert "Do not confuse it with an independently selected AI mental model" in prompt_analogy


def test_unsuitable_analogy_routing_and_fallbacks(tmp_path):
    """Verify all rejected and fallback analogy routing cases:
    1. Boolean False with a non-empty AI replacement clears self.analogy.
    2. String 'false' with a non-empty AI replacement clears self.analogy.
    3. Boolean False with 'None - Direct Explanation' clears self.analogy.
    4. Missing suitability fields fall back to original requested analogy.
    In all cases, requested_analogy remains preserved."""

    # 1. Boolean False with non-empty AI replacement
    agent1 = TeachingVideoAgent(
        idx=201,
        knowledge_point="Dijkstra's Algorithm",
        folder=str(tmp_path),
        cfg=RunConfig(level="intermediate", analogy="Quantum teleportation"),
    )
    with patch.object(agent1, "_request_api_and_track_tokens") as mock_api:
        mock_api.return_value = json.dumps({
            "topic": "Dijkstra's Algorithm",
            "analogy_evaluation": {
                "requested_analogy": "Quantum teleportation",
                "is_suitable": False,
                "adopted_analogy": "Wavefront expansion",
                "rationale": "Quantum phenomena do not map to shortest paths",
            },
            "analogy_used": "Wavefront expansion",
            "mental_model": {"description": "Wavefront expanding outward"},
            "sections": [{"id": "section_1", "title": "Sec 1"}],
        })
        agent1.generate_outline()

    assert agent1.requested_analogy == "Quantum teleportation"
    assert agent1.analogy == ""

    # 2. String 'false' with non-empty AI replacement
    agent2 = TeachingVideoAgent(
        idx=202,
        knowledge_point="Dijkstra's Algorithm",
        folder=str(tmp_path),
        cfg=RunConfig(level="intermediate", analogy="Fantasy quest"),
    )
    with patch.object(agent2, "_request_api_and_track_tokens") as mock_api:
        mock_api.return_value = json.dumps({
            "topic": "Dijkstra's Algorithm",
            "analogy_evaluation": {
                "requested_analogy": "Fantasy quest",
                "is_suitable": "false",
                "adopted_analogy": "Wavefront expansion",
                "rationale": "Fantasy quests are unfaithful to graph weights",
            },
            "analogy_used": "Wavefront expansion",
            "mental_model": {"description": "Wavefront expanding outward"},
            "sections": [{"id": "section_1", "title": "Sec 1"}],
        })
        agent2.generate_outline()

    assert agent2.requested_analogy == "Fantasy quest"
    assert agent2.analogy == ""

    # 3. Boolean False with 'None - Direct Explanation'
    agent3 = TeachingVideoAgent(
        idx=203,
        knowledge_point="Calculus Limits",
        folder=str(tmp_path),
        cfg=RunConfig(level="beginner", analogy="Magic potion brewing"),
    )
    with patch.object(agent3, "_request_api_and_track_tokens") as mock_api:
        mock_api.return_value = json.dumps({
            "topic": "Calculus Limits",
            "analogy_evaluation": {
                "requested_analogy": "Magic potion brewing",
                "is_suitable": False,
                "adopted_analogy": "None - Direct Explanation",
                "rationale": "Not faithful to epsilon-delta definitions",
            },
            "analogy_used": "None - Direct Explanation",
            "sections": [{"id": "section_1", "title": "Sec 1"}],
        })
        agent3.generate_outline()

    assert agent3.requested_analogy == "Magic potion brewing"
    assert agent3.analogy == ""

    # 4. Missing suitability fields with original-analogy fallback
    agent4 = TeachingVideoAgent(
        idx=204,
        knowledge_point="Fluid Dynamics",
        folder=str(tmp_path),
        cfg=RunConfig(level="intermediate", analogy="Water pipes"),
    )
    with patch.object(agent4, "_request_api_and_track_tokens") as mock_api:
        mock_api.return_value = json.dumps({
            "topic": "Fluid Dynamics",
            "analogy_evaluation": {},
            "sections": [{"id": "section_1", "title": "Sec 1"}],
        })
        agent4.generate_outline()

    assert agent4.requested_analogy == "Water pipes"
    assert agent4.analogy == "Water pipes"


