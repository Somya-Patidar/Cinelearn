"""
Milestone 5.2.2-A: Native Stage 2 Scene IR Generation Contract Tests.
=====================================================================

Validates:
1. Native Stage 2 prompt generation from Stage 1 outline and TopicProfile.
2. Propagation of Stage 1 metadata and pedagogical guidance without domain bias.
3. Schema and VisualRegistry manifest inclusion in the prompt.
4. Domain-independent prompt behavior (zero hardcoded topic assumptions).
5. Valid native Scene IR and SceneSequenceIR deserialization.
6. Detection and fail-fast rejection of invalid top-level structures.
7. Strict validation of nested collection values (e.g. Sequence[str]).
8. Strict validation of Union types (e.g. Union[ProcessCardVisual, str]).
9. Detection of invalid component parameter types and boolean isolation.
10. Fail-closed rejection of unknown components and unsupported operations.
11. Clean handling of malformed or invalid JSON input without unbounded loops.
12. Structured processing results and diagnostic classification.
13. Native Stage 2 generation outputs SceneSequenceIR without legacy adapter projection.
14. Preservation of legacy pipeline execution behavior.
15. Fail-closed protection of modern pipeline downstream execution.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prompts.stage2 import (
    get_prompt2_scene_ir,
    get_prompt2_scene_ir_repair,
    get_prompt2_storyboard,
)
from src.agent import RunConfig, TeachingOutline, TeachingVideoAgent
from src.domains import VisualRegistry
from src.ir.models import (
    ActionDeclaration,
    ActionType,
    EntityDeclaration,
    SceneIR,
    SceneSequenceIR,
    ViewportDeclaration,
    ViewportType,
    VisualTier,
)
from src.ir.payload_validator import PayloadErrorCode, SceneIRPayloadValidator
from src.ir.processor import SceneIRProcessingResult, process_stage2_scene_ir_output
from src.pedagogy import TopicProfile, TopicProfiler
from tests.test_milestone3a_scene_ir import (
    get_binary_search_fixture,
    get_chemical_kinetics_fixture,
    get_pythagorean_theorem_fixture,
)


def _make_minimal_scene(entities=None, timeline=None, **kwargs) -> SceneIR:
    defaults = {
        "scene_id": "NativeStage2Test",
        "section_index": 1,
        "title": "Minimal Test Scene",
        "duration_budget": 10.0,
        "viewports": {"dock_main": ViewportDeclaration(type=ViewportType.MAIN)},
        "entities": entities or [],
        "timeline": timeline or [],
        "metadata": {"lecture_lines": ["Testing native Stage 2 generation."]},
    }
    defaults.update(kwargs)
    return SceneIR(**defaults)


# =============================================================================
# 1. Native Stage 2 Prompt Generation
# =============================================================================

def test_native_stage2_prompt_generation():
    """Prompt must include authoritative schema, manifest, and intermediate level guidance."""
    outline = {
        "topic": "Binary Search Algorithm",
        "target_audience": "undergraduate",
        "sections": [{"id": "sec_1", "title": "Array Bisection", "lecture_lines": ["Bisect the search space."]}],
    }
    prompt = get_prompt2_scene_ir(outline=outline, level="intermediate", analogy="Looking up words in a dictionary")
    assert "Scene Intermediate Representation (Scene IR v1.0)" in prompt
    assert "Available Visual Capabilities & Registered Components" in prompt
    assert "Authoritative JSON Schema Contract" in prompt
    assert "Two-Layer Validation Requirements" in prompt
    assert "do NOT write executable Python, Manim commands" in prompt
    assert "Looking up words in a dictionary" in prompt


# =============================================================================
# 2. Stage 1 Outline and TopicProfile Propagation
# =============================================================================

def test_stage1_outline_and_topic_profile_propagation():
    """Stage 1 TeachingOutline and TopicProfile must cleanly format into prompt guidance."""
    profile_data = {
        "primary_domain": "computer_science",
        "subdomain": "algorithms",
        "formalism_level": "moderate",
        "computation_level": "algorithmic",
        "primary_visual_mode": "state_evolution",
        "secondary_visual_mode": "table",
        "recommended_visual_tiers": [1, 2, 4],
        "rationale": "Dynamic state changes across arrays.",
    }
    profile = TopicProfile.from_dict(profile_data)
    outline = TeachingOutline(
        topic="Dynamic Programming Knapsack",
        target_audience="intermediate",
        topic_profile=profile_data,
        sections=[],
    )
    prompt = get_prompt2_scene_ir(outline=outline, topic_profile=profile)
    assert "Dynamic Programming Knapsack" in prompt
    assert "Pedagogical & Visual Guidance" in prompt
    assert "state_evolution" in prompt or "COMPUTER_SCIENCE" in prompt
    assert "Authoritative JSON Schema Contract" in prompt


# =============================================================================
# 3. Schema and Manifest Inclusion in Prompt
# =============================================================================

def test_schema_and_manifest_inclusion_in_prompt():
    """Registered domain components from VisualRegistry must appear with viewports and parameter specs."""
    manifest = VisualRegistry.get_schema_manifest()
    prompt = get_prompt2_scene_ir(outline={"topic": "General Topic"}, component_manifest=manifest)

    # Check key registered components
    for comp in ("ReactionEnergyDiagram", "NumberLineVisual", "FunctionCurve", "ProcessCardVisual"):
        assert comp in prompt, f"Expected registered component '{comp}' in prompt"

    # Check that initial_state parameters and operations are documented
    assert "initial_state parameters:" in prompt
    assert "Supported component_op operations:" in prompt
    assert "4-Tier Visual Capability Hierarchy:" in prompt


# =============================================================================
# 4. Domain-Independent Prompt Behavior
# =============================================================================

def test_domain_independent_prompt_behavior():
    """Prompt must not contain hardcoded domain biases (e.g., FCFS or Dijkstra specifics)."""
    for topic in ("Photosynthesis in C3 Plants", "Pythagorean Theorem", "Electromagnetic Induction"):
        prompt = get_prompt2_scene_ir(outline={"topic": topic})
        assert "fcfs" not in prompt.lower()
        assert "dijkstra" not in prompt.lower()
        assert topic in prompt


# =============================================================================
# 5. Valid Native Scene IR Deserialization
# =============================================================================

def test_valid_native_scene_ir_deserialization():
    """Canonical domain fixtures must pass deserialization and deterministic validation."""
    for fixture in (get_binary_search_fixture(), get_chemical_kinetics_fixture(), get_pythagorean_theorem_fixture()):
        fixture_dict = fixture.to_dict()
        res = process_stage2_scene_ir_output(fixture_dict)
        assert res.is_valid, f"Validation failed for {fixture.scene_id}: {res.all_errors}"
        assert res.scene_ir is not None
        assert res.scene_ir.scene_id == fixture.scene_id
        assert not res.has_structural_errors
        assert not res.has_semantic_errors
        assert not res.has_payload_errors


# =============================================================================
# 6. Invalid Top-Level Scene IR Structure
# =============================================================================

def test_invalid_top_level_scene_ir_structure():
    """Invalid schema fields (negative duration, missing scene_id, forbidden keys) fail fast."""
    bad_scene = {
        "version": "1.0",
        "scene_id": "BadScene",
        "duration_budget": -5.0,  # Negative duration forbidden
        "unknown_top_level_key": "forbidden",
        "viewports": {},
    }
    res = process_stage2_scene_ir_output(bad_scene)
    assert not res.is_valid
    assert res.has_structural_errors
    assert any("duration_budget" in str(e) or "extra" in str(e).lower() for e in res.structural_errors)


# =============================================================================
# 7. Invalid Nested Collection Values
# =============================================================================

def test_invalid_nested_collection_values():
    """Nested collection type mismatches (e.g. ints inside Sequence[str]) must be caught."""
    validator = SceneIRPayloadValidator()
    # CompositeDissectionSquare expects labels: Sequence[str]
    scene = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="sq1",
                component_type="CompositeDissectionSquare",
                visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                viewport="dock_main",
                initial_state={"side_length": 4.0, "labels": [101, 102]},  # Should be strings!
            )
        ]
    )
    res = validator.validate(scene)
    assert not res.is_valid
    type_errs = [e for e in res.errors if e.error_code == PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE]
    assert len(type_errs) >= 1
    assert type_errs[0].parameter == "labels"


# =============================================================================
# 8. Invalid Union Values
# =============================================================================

def test_invalid_union_values():
    """Union type parameters must reject incompatible scalar types (e.g. int for Union[ProcessCardVisual, str])."""
    validator = SceneIRPayloadValidator()
    scene = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="cpu_vis",
                component_type="CPUExecutionVisual",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="dock_main",
                initial_state={},
            )
        ],
        timeline=[
            {
                "step_id": "s1",
                "actions": [
                    {
                        "type": ActionType.COMPONENT_OP,
                        "target_entity": "cpu_vis",
                        "operation": "animate_dispatch",
                        # dispatch expects item: Optional[Union[ProcessCardVisual, str]]
                        "parameters": {"item": 99999},  # Int is invalid for Union[ProcessCardVisual, str]
                    }
                ],
            }
        ],
    )
    res = validator.validate(scene)
    assert not res.is_valid
    type_errs = [e for e in res.errors if e.error_code == PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE]
    assert len(type_errs) >= 1
    assert type_errs[0].parameter == "item"


# =============================================================================
# 9. Invalid Component Parameter Types
# =============================================================================

def test_invalid_component_parameter_types_and_boolean_isolation():
    """Booleans must never be accepted as numeric types (int or float)."""
    validator = SceneIRPayloadValidator()
    # NumberLineVisual expects length: float
    scene = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="nl1",
                component_type="NumberLineVisual",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="dock_main",
                initial_state={"length": True},  # True is bool, not float
            )
        ]
    )
    res = validator.validate(scene)
    assert not res.is_valid
    type_errs = [e for e in res.errors if e.error_code == PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE]
    assert len(type_errs) >= 1
    assert type_errs[0].parameter == "length"


# =============================================================================
# 10. Unknown or Unsupported Visual Capabilities Fail-Closed
# =============================================================================

def test_unknown_or_unsupported_visual_capabilities():
    """Unregistered component types and unsupported operations must fail closed."""
    # 1. Unregistered component
    scene_unreg = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="c1",
                component_type="QuantumSuperpositionVisual",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="dock_main",
            )
        ]
    )
    res_unreg = process_stage2_scene_ir_output(scene_unreg.to_dict())
    assert not res_unreg.is_valid
    assert res_unreg.has_unsupported_capabilities
    assert any(e["error_code"] == "UNSUPPORTED_COMPONENT_TYPE" for e in res_unreg.payload_errors)

    # 2. Unsupported operation
    scene_bad_op = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="p1",
                component_type="ProcessCardVisual",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="dock_main",
                initial_state={"pid": "P1", "arrival_time": 0, "burst_time": 2},
            )
        ],
        timeline=[
            {
                "step_id": "s1",
                "actions": [
                    {
                        "type": ActionType.COMPONENT_OP,
                        "target_entity": "p1",
                        "operation": "quantum_teleport",  # Unsupported operation
                    }
                ],
            }
        ],
    )
    res_bad_op = process_stage2_scene_ir_output(scene_bad_op.to_dict())
    assert not res_bad_op.is_valid
    assert res_bad_op.has_unsupported_capabilities
    assert any(e["error_code"] == "UNSUPPORTED_COMPONENT_OPERATION" for e in res_bad_op.payload_errors)


# =============================================================================
# 11. Malformed JSON Handling
# =============================================================================

def test_malformed_json_handling():
    """Malformed JSON fails fast without prompting open-ended repairs."""
    res_syntax = process_stage2_scene_ir_output("{broken_json: missing_quotes,")
    assert not res_syntax.is_valid
    assert res_syntax.has_structural_errors
    assert not res_syntax.has_recoverable_diagnostics

    repair_prompt = get_prompt2_scene_ir_repair(
        outline={"topic": "T"},
        failed_output="{broken_json",
        processing_result=res_syntax,
    )
    # Syntax errors must fail fast; repair prompt returns None
    assert repair_prompt is None


# =============================================================================
# 12. Structured Processing Results and Diagnostics
# =============================================================================

def test_structured_processing_results_and_diagnostics():
    """SceneIRProcessingResult provides comprehensive, structured diagnostic metadata."""
    scene = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="card1",
                component_type="ConceptualCard",
                visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                viewport="dock_main",
                initial_state={"title": "Good Title", "formula": 12345},  # formula must be str
            )
        ]
    )
    res = process_stage2_scene_ir_output(scene.to_dict())
    assert not res.is_valid
    assert res.has_payload_errors
    assert not res.has_structural_errors
    assert not res.has_unsupported_capabilities
    assert res.has_recoverable_diagnostics

    d = res.to_dict()
    assert d["is_valid"] is False
    assert d["has_payload_errors"] is True
    assert d["has_recoverable_diagnostics"] is True
    assert len(d["payload_errors"]) == 1
    assert d["payload_errors"][0]["error_code"] == "INVALID_PAYLOAD_PARAM_TYPE"


# =============================================================================
# 13. Native Stage 2 Does Not Invoke Legacy Storyboard Adapter
# =============================================================================

def test_native_stage2_does_not_invoke_legacy_storyboard_adapter():
    """generate_scene_ir must persist SceneSequenceIR/SceneIR without converting through SectionProxy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search Algorithm",
            folder=tmpdir,
            pipeline_mode="modern",
        )
        agent.outline = TeachingOutline(
            topic="Binary Search Algorithm",
            target_audience="intermediate",
            sections=[],
        )

        valid_fixture = get_binary_search_fixture()
        mock_response = valid_fixture.to_json()

        with patch.object(agent, "_request_api_and_track_tokens", return_value=mock_response):
            with patch("src.agent.scene_ir_to_storyboard_section_obj") as mock_adapter:
                res = agent.generate_scene_ir()

                assert res.is_valid
                assert agent.scene_ir is not None
                assert agent.scene_ir.scene_id == valid_fixture.scene_id
                # Adapter must NOT have been called as primary output
                mock_adapter.assert_not_called()

                # Verify file on disk
                saved_file = Path(tmpdir) / "0-Binary_Search_Algorithm" / "scene_ir.json"
                assert saved_file.exists()


# =============================================================================
# 14. Existing Legacy Pipeline Remains Unchanged
# =============================================================================

def test_existing_legacy_pipeline_remains_unchanged():
    """Legacy pipeline configuration and storyboard prompt paths remain functional."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = TeachingVideoAgent(
            idx=0,
            topic="Sorting Algorithms",
            folder=tmpdir,
            pipeline_mode="legacy",
        )
        assert agent.pipeline_mode == "legacy"

        # Verify legacy prompt generator continues to produce storyboard prompt
        legacy_prompt = get_prompt2_storyboard(
            outline={"topic": "Sorting Algorithms", "sections": []},
            level="intermediate",
        )
        assert "Storyboard" in legacy_prompt
        assert "visual_conventions" in legacy_prompt


# =============================================================================
# 15. Modern Mode Downstream Execution Fails Closed
# =============================================================================

def test_modern_mode_still_fails_closed_downstream():
    """Modern mode GENERATE_VIDEO fails closed until downstream compilation stages are integrated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = TeachingVideoAgent(
            idx=0,
            topic="Modern Pipeline Guard",
            folder=tmpdir,
            pipeline_mode="modern",
        )
        with pytest.raises(NotImplementedError) as exc_info:
            agent.GENERATE_VIDEO()
        assert "Modern Scene IR pipeline execution is not yet implemented" in str(exc_info.value)


# =============================================================================
# 16. Condition 1 (FINDING-5.2.2A-01): Visual Tier 2 Composition Validation
# =============================================================================

def test_tier2_composition_unrecognized_component_rejected_with_unsupported_type():
    """Tier 2 entity with unrecognized/unregistered component fails closed with UNSUPPORTED_COMPONENT_TYPE."""
    validator = SceneIRPayloadValidator()

    # 1. Arbitrary / fake component name specifying tier2_composition
    entity_fake = EntityDeclaration(
        id="comp_fake",
        component_type="FakeNonexistentComponent",
        visual_tier=VisualTier.TIER2_COMPOSITION,
        viewport="dock_main",
        initial_state={"junk_param": 12345},
    )
    errors = validator.validate_entity_payload(entity_fake)
    assert len(errors) == 1
    assert errors[0].error_code == PayloadErrorCode.UNSUPPORTED_COMPONENT_TYPE
    assert "unsupported Tier 2 composite component" in errors[0].message
    assert errors[0].component_type == "FakeNonexistentComponent"

    # 2. Tier 1 registered component without composite contract declared as Tier 2 fails closed
    entity_tier1_misdeclared = EntityDeclaration(
        id="comp_pcard",
        component_type="ProcessCardVisual",
        visual_tier=VisualTier.TIER2_COMPOSITION,
        viewport="dock_main",
        initial_state={"pid": "P1", "arrival_time": 0, "burst_time": 2},
    )
    errors_pcard = validator.validate_entity_payload(entity_tier1_misdeclared)
    assert len(errors_pcard) == 1
    assert errors_pcard[0].error_code == PayloadErrorCode.UNSUPPORTED_COMPONENT_TYPE


def test_tier2_composition_recognized_contract_validation():
    """A recognized Tier 2 component is validated against its actual supported contract."""
    # Define a manifest specifying a recognized Tier 2 composite component
    composite_manifest = {
        "composite": [
            {
                "component_id": "CompositeSplitContainer",
                "tier": 2,
                "is_composite": True,
                "domain": "composite",
                "init_params": {
                    "sub_components": {"type": "Sequence[str]", "required": True},
                    "layout": {"type": "str", "required": False},
                    "padding": {"type": "float", "required": False},
                },
                "operations": {
                    "animate_highlight_subview": {
                        "name": "animate_highlight_subview",
                        "parameters": {
                            "index": {"type": "int", "required": True},
                            "color": {"type": "str", "required": False},
                        },
                    }
                },
            }
        ]
    }
    validator = SceneIRPayloadValidator(manifest=composite_manifest)

    # Valid entity conforming to the contract
    valid_entity = EntityDeclaration(
        id="split_container_1",
        component_type="CompositeSplitContainer",
        visual_tier=VisualTier.TIER2_COMPOSITION,
        viewport="dock_main",
        initial_state={"sub_components": ["view_left", "view_right"], "layout": "split_horizontal", "padding": 0.15},
    )
    errors = validator.validate_entity_payload(valid_entity)
    assert len(errors) == 0

    # Valid operation on recognized Tier 2 entity
    valid_action = ActionDeclaration(
        type=ActionType.COMPONENT_OP,
        target_entity="split_container_1",
        operation="animate_highlight_subview",
        parameters={"index": 0, "color": "#FFD700"},
    )
    action_errors = validator.validate_action_payload(valid_action, valid_entity)
    assert len(action_errors) == 0


def test_tier2_composition_invalid_initial_state_parameters_rejected():
    """Invalid Tier 2 initial-state parameters are rejected with structured diagnostics."""
    composite_manifest = {
        "composite": [
            {
                "component_id": "DualViewComposition",
                "is_composite": True,
                "init_params": {
                    "views": {"type": "Sequence[str]", "required": True},
                    "split_ratio": {"type": "float", "required": True},
                    "border_color": {"type": "str", "required": False},
                },
                "operations": {
                    "animate_swap_views": {
                        "name": "animate_swap_views",
                        "parameters": {
                            "duration": {"type": "float", "required": False},
                        },
                    }
                },
            }
        ]
    }
    validator = SceneIRPayloadValidator(manifest=composite_manifest)

    # 1. Missing required parameter ("split_ratio" missing)
    entity_missing = EntityDeclaration(
        id="dual_1",
        component_type="DualViewComposition",
        visual_tier=VisualTier.TIER2_COMPOSITION,
        viewport="dock_main",
        initial_state={"views": ["v1", "v2"]},
    )
    errs_missing = validator.validate_entity_payload(entity_missing)
    assert len(errs_missing) == 1
    assert errs_missing[0].error_code == PayloadErrorCode.MISSING_REQUIRED_PAYLOAD_PARAM
    assert errs_missing[0].parameter == "split_ratio"

    # 2. Unrecognized parameter ("extra_unrecognized" key)
    entity_unrecognized = EntityDeclaration(
        id="dual_2",
        component_type="DualViewComposition",
        visual_tier=VisualTier.TIER2_COMPOSITION,
        viewport="dock_main",
        initial_state={"views": ["v1", "v2"], "split_ratio": 0.5, "extra_unrecognized": 999},
    )
    errs_unrecognized = validator.validate_entity_payload(entity_unrecognized)
    assert len(errs_unrecognized) == 1
    assert errs_unrecognized[0].error_code == PayloadErrorCode.UNRECOGNIZED_PAYLOAD_PARAM
    assert errs_unrecognized[0].parameter == "extra_unrecognized"

    # 3. Invalid parameter type: "views" is int instead of Sequence[str], "split_ratio" is bool instead of float
    entity_bad_types = EntityDeclaration(
        id="dual_3",
        component_type="DualViewComposition",
        visual_tier=VisualTier.TIER2_COMPOSITION,
        viewport="dock_main",
        initial_state={"views": 12345, "split_ratio": True},
    )
    errs_types = validator.validate_entity_payload(entity_bad_types)
    assert len(errs_types) == 2
    assert all(e.error_code == PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE for e in errs_types)
    bad_params = {e.parameter for e in errs_types}
    assert bad_params == {"views", "split_ratio"}

    # 4. Invalid operation on Tier 2 entity
    entity_valid_decl = EntityDeclaration(
        id="dual_4",
        component_type="DualViewComposition",
        visual_tier=VisualTier.TIER2_COMPOSITION,
        viewport="dock_main",
        initial_state={"views": ["v1", "v2"], "split_ratio": 0.5},
    )
    bad_op_action = ActionDeclaration(
        type=ActionType.COMPONENT_OP,
        target_entity="dual_4",
        operation="nonexistent_tier2_operation",
        parameters={},
    )
    errs_op = validator.validate_action_payload(bad_op_action, entity_valid_decl)
    assert len(errs_op) == 1
    assert errs_op[0].error_code == PayloadErrorCode.UNSUPPORTED_COMPONENT_OPERATION


def test_tier2_validation_preserves_tier1_tier3_tier4():
    """Existing Tier 1, Tier 3, and Tier 4 validation behavior remains unaffected."""
    validator = SceneIRPayloadValidator()

    # Tier 1 valid
    e_t1 = EntityDeclaration(
        id="pcard",
        component_type="ProcessCardVisual",
        visual_tier=VisualTier.TIER1_REGISTERED,
        viewport="dock_main",
        initial_state={"pid": "P1", "arrival_time": 0, "burst_time": 4},
    )
    assert len(validator.validate_entity_payload(e_t1)) == 0

    # Tier 3 valid
    e_t3 = EntityDeclaration(
        id="shape1",
        component_type="PrimitiveShape",
        visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
        viewport="dock_main",
        initial_state={"shape_type": "rectangle", "width": 4.0, "height": 2.0},
    )
    assert len(validator.validate_entity_payload(e_t3)) == 0

    # Tier 4 valid
    e_t4 = EntityDeclaration(
        id="card1",
        component_type="ConceptualCard",
        visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
        viewport="dock_main",
        initial_state={"title": "Pythagorean Theorem", "formula": "a^2 + b^2 = c^2"},
    )
    assert len(validator.validate_entity_payload(e_t4)) == 0


# =============================================================================
# 17. Condition 2 (FINDING-5.2.2A-02): Repair Loop Candidate Synchronization
# =============================================================================

def test_bounded_repair_candidate_synchronization_end_to_end():
    """Deterministic integration test for repair candidate synchronization across retry attempts.

    Test sequence:
    1. Initial generation returns an invalid Scene IR payload (formula is an int).
    2. First repair returns a different invalid payload (formula is fixed, but title is an int).
    3. Second repair prompt receives the first repair's output (NOT the original output) and
       receives diagnostics matching the first repair's output.
    4. Second repair returns valid Scene IR.
    5. Final result is validated successfully.
    6. Valid Scene IR is persisted and checkpointed.
    7. Number of repair requests never exceeds two.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = TeachingVideoAgent(
            idx=0,
            topic="Repair Candidate Synchronization Test",
            folder=tmpdir,
            pipeline_mode="modern",
        )
        agent.outline = TeachingOutline(
            topic="Repair Candidate Synchronization Test",
            target_audience="undergraduate",
            sections=[],
        )

        # Payload 1 (Initial Generation): Invalid formula (int instead of str)
        payload_turn0 = json.dumps({
            "version": "1.0",
            "scene_id": "RepairSyncScene",
            "section_index": 1,
            "title": "Repair Sync Test Scene",
            "duration_budget": 10.0,
            "viewports": {"dock_main": {"type": "viewport_main", "flow": "single", "alignment": "center", "padding": 0.2, "sub_slots": []}},
            "entities": [
                {
                    "id": "card_turn0",
                    "component_type": "ConceptualCard",
                    "visual_tier": "tier4_conceptual_card",
                    "viewport": "dock_main",
                    "initial_state": {"title": "Valid Title", "formula": 11111},  # Invalid type: int for formula
                }
            ],
            "timeline": [
                {
                    "step_id": "step_turn0_1",
                    "narration_anchor": {"lecture_line_index": 0, "text_cue": "Valid Title", "audio_timestamp_range": [0.0, 3.0]},
                    "actions": [{"type": "enter", "target_entity": "card_turn0", "duration": 1.0}],
                }
            ],
            "metadata": {"lecture_lines": ["Turn 0 line"]},
        })

        # Payload 2 (Repair Attempt 1): Different invalid payload (formula fixed, but title is int, id changed to card_turn1)
        payload_turn1 = json.dumps({
            "version": "1.0",
            "scene_id": "RepairSyncScene",
            "section_index": 1,
            "title": "Repair Sync Test Scene",
            "duration_budget": 10.0,
            "viewports": {"dock_main": {"type": "viewport_main", "flow": "single", "alignment": "center", "padding": 0.2, "sub_slots": []}},
            "entities": [
                {
                    "id": "card_turn1",
                    "component_type": "ConceptualCard",
                    "visual_tier": "tier4_conceptual_card",
                    "viewport": "dock_main",
                    "initial_state": {"title": 22222, "formula": "a^2 + b^2 = c^2"},  # Invalid type: int for title
                }
            ],
            "timeline": [
                {
                    "step_id": "step_turn1_1",
                    "narration_anchor": {"lecture_line_index": 0, "text_cue": "Turn 1", "audio_timestamp_range": [0.0, 3.0]},
                    "actions": [{"type": "enter", "target_entity": "card_turn1", "duration": 1.0}],
                }
            ],
            "metadata": {"lecture_lines": ["Turn 1 line"]},
        })

        # Payload 3 (Repair Attempt 2): Completely valid Scene IR
        payload_turn2 = json.dumps({
            "version": "1.0",
            "scene_id": "RepairSyncScene",
            "section_index": 1,
            "title": "Repair Sync Test Scene",
            "duration_budget": 10.0,
            "viewports": {"dock_main": {"type": "viewport_main", "flow": "single", "alignment": "center", "padding": 0.2, "sub_slots": []}},
            "entities": [
                {
                    "id": "card_turn2",
                    "component_type": "ConceptualCard",
                    "visual_tier": "tier4_conceptual_card",
                    "viewport": "dock_main",
                    "initial_state": {"title": "Repaired Title", "formula": "a^2 + b^2 = c^2"},
                }
            ],
            "timeline": [
                {
                    "step_id": "step_turn2_1",
                    "narration_anchor": {"lecture_line_index": 0, "text_cue": "Turn 2", "audio_timestamp_range": [0.0, 3.0]},
                    "actions": [{"type": "enter", "target_entity": "card_turn2", "duration": 1.0}],
                }
            ],
            "metadata": {"lecture_lines": ["Turn 2 line"]},
        })

        # Track all calls to _request_api_and_track_tokens
        prompts_sent = []
        stages_called = []

        def mock_request(prompt, max_tokens=None, stage=None):
            prompts_sent.append(prompt)
            stages_called.append(stage)
            if stage == "scene_ir":
                return payload_turn0
            elif stage == "scene_ir_repair":
                repair_count = stages_called.count("scene_ir_repair")
                if repair_count == 1:
                    return payload_turn1
                elif repair_count == 2:
                    return payload_turn2
                else:
                    raise AssertionError(f"Repair requests exceeded 2 (count={repair_count})")
            raise ValueError(f"Unexpected stage {stage}")

        with patch.object(agent, "_request_api_and_track_tokens", side_effect=mock_request):
            res = agent.generate_scene_ir()

        # Step 7: Number of repair requests never exceeds two
        repair_calls = [s for s in stages_called if s == "scene_ir_repair"]
        assert len(repair_calls) == 2, f"Expected exactly 2 repair calls, got {len(repair_calls)}"

        # Step 3: Verify prompts sent for repair attempt 1 and attempt 2
        assert len(prompts_sent) == 3  # 1 initial + 2 repairs

        # Prompt for Repair 1 must receive Turn 0 payload and Turn 0 diagnostics
        prompt_repair1 = prompts_sent[1]
        assert "card_turn0" in prompt_repair1
        assert "11111" in prompt_repair1
        assert "formula" in prompt_repair1

        # Prompt for Repair 2 must receive Turn 1 payload (NOT Turn 0 payload!)
        prompt_repair2 = prompts_sent[2]
        assert "card_turn1" in prompt_repair2, "Second repair prompt must contain candidate from first repair response"
        assert "22222" in prompt_repair2, "Second repair prompt must contain candidate from first repair response"
        assert "card_turn0" not in prompt_repair2, "Second repair prompt must NOT contain stale turn 0 candidate"
        assert "11111" not in prompt_repair2, "Second repair prompt must NOT contain stale turn 0 candidate"

        # Verify diagnostics supplied to second repair correspond to the first repair response
        assert "title" in prompt_repair2, "Diagnostics in second repair must flag title error from first repair"
        assert "[param: `title`]" in prompt_repair2 or "title" in prompt_repair2

        # Step 5: Final result is validated successfully
        assert res.is_valid is True
        assert agent.scene_ir is not None
        assert agent.scene_ir.scene_id == "RepairSyncScene"
        assert agent.scene_ir.entities[0].id == "card_turn2"

        # Step 6: Valid Scene IR is persisted and checkpointed
        saved_file = Path(tmpdir) / "0-Repair_Candidate_Synchronization_Test" / "scene_ir.json"
        assert saved_file.exists(), f"Expected saved file at {saved_file}"
        with open(saved_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data["entities"][0]["id"] == "card_turn2"

        checkpoint_file = Path(tmpdir) / "0-Repair_Candidate_Synchronization_Test" / "checkpoint.json"
        assert checkpoint_file.exists()
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            cp_data = json.load(f)
        assert cp_data["stage"] == "scene_ir_generated"
        assert cp_data.get("has_scene_ir") is True or cp_data.get("extra", {}).get("has_scene_ir") is True


def test_bounded_repair_exhaustion_does_not_persist_invalid_output():
    """When all repair attempts fail, retry count is capped at 2 and invalid output is never persisted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = TeachingVideoAgent(
            idx=0,
            topic="Repair Exhaustion Safety Test",
            folder=tmpdir,
            pipeline_mode="modern",
        )
        agent.outline = TeachingOutline(
            topic="Repair Exhaustion Safety Test",
            target_audience="undergraduate",
            sections=[],
        )

        invalid_turn0 = json.dumps({
            "version": "1.0",
            "scene_id": "ExhaustionScene",
            "section_index": 1,
            "title": "Exhaustion Test Scene",
            "duration_budget": 10.0,
            "viewports": {"dock_main": {"type": "viewport_main", "flow": "single", "alignment": "center", "padding": 0.2, "sub_slots": []}},
            "entities": [
                {
                    "id": "e_turn0",
                    "component_type": "ConceptualCard",
                    "visual_tier": "tier4_conceptual_card",
                    "viewport": "dock_main",
                    "initial_state": {"title": "Title 0", "formula": 100},
                }
            ],
            "timeline": [],
        })
        invalid_turn1 = json.dumps({
            "version": "1.0",
            "scene_id": "ExhaustionScene",
            "section_index": 1,
            "title": "Exhaustion Test Scene",
            "duration_budget": 10.0,
            "viewports": {"dock_main": {"type": "viewport_main", "flow": "single", "alignment": "center", "padding": 0.2, "sub_slots": []}},
            "entities": [
                {
                    "id": "e_turn1",
                    "component_type": "ConceptualCard",
                    "visual_tier": "tier4_conceptual_card",
                    "viewport": "dock_main",
                    "initial_state": {"title": "Title 1", "formula": 200},
                }
            ],
            "timeline": [],
        })
        invalid_turn2 = json.dumps({
            "version": "1.0",
            "scene_id": "ExhaustionScene",
            "section_index": 1,
            "title": "Exhaustion Test Scene",
            "duration_budget": 10.0,
            "viewports": {"dock_main": {"type": "viewport_main", "flow": "single", "alignment": "center", "padding": 0.2, "sub_slots": []}},
            "entities": [
                {
                    "id": "e_turn2",
                    "component_type": "ConceptualCard",
                    "visual_tier": "tier4_conceptual_card",
                    "viewport": "dock_main",
                    "initial_state": {"title": "Title 2", "formula": 300},
                }
            ],
            "timeline": [],
        })

        stages_called = []
        prompts_sent = []

        def mock_failing_requests(prompt, max_tokens=None, stage=None):
            prompts_sent.append(prompt)
            stages_called.append(stage)
            if stage == "scene_ir":
                return invalid_turn0
            elif stage == "scene_ir_repair":
                count = stages_called.count("scene_ir_repair")
                if count == 1:
                    return invalid_turn1
                elif count == 2:
                    return invalid_turn2
                else:
                    raise AssertionError("Repair requests exceeded 2")
            return None

        with patch.object(agent, "_request_api_and_track_tokens", side_effect=mock_failing_requests):
            res = agent.generate_scene_ir()

        # Repair requests capped strictly at 2
        assert stages_called.count("scene_ir_repair") == 2
        assert res.is_valid is False
        assert res.has_payload_errors is True

        # Candidate synchronization: Prompt for repair 2 received turn 1 candidate
        prompt_repair2 = prompts_sent[2]
        assert "e_turn1" in prompt_repair2
        assert "200" in prompt_repair2
        assert "e_turn0" not in prompt_repair2

        # Invalid scene IR MUST NOT be persisted to scene_ir.json or scene_sequence_ir.json
        case_dir = Path(tmpdir) / "0-Repair_Exhaustion_Safety_Test"
        assert not (case_dir / "scene_ir.json").exists()
        assert not (case_dir / "scene_sequence_ir.json").exists()

        # Checkpoint MUST NOT be recorded as successful
        checkpoint_file = case_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                cp = json.load(f)
            assert cp.get("stage") != "scene_ir_generated"

        # Error diagnostics file MUST be saved for auditability
        err_file = case_dir / "scene_ir_validation_errors.json"
        assert err_file.exists()
        with open(err_file, "r", encoding="utf-8") as f:
            err_data = json.load(f)
        assert err_data["is_valid"] is False
        assert len(err_data["payload_errors"]) > 0
