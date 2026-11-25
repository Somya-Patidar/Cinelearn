"""
CineLearn — Milestone 3B: Stage 2 Scene IR Generation Contract & Prompt Integration Tests.
========================================================================================

Verifies:
1. Generation contract and prompt construction (schema injection, registry metadata, domain independence).
2. Output processing and boundary validation (JSON extraction, Pydantic parsing, Gate 3.5, payload validation).
3. Visual capability and payload constraints (unknown components, operations, extra/missing parameters).
4. Stage 3 compatibility adapter (deterministic mapping to legacy storyboard, 6-field animation format).
5. Explicit fail-closed behavior for unmappable or incomplete Scene IR documents.
6. Verification of repaired Milestone 3A diagnostic error codes (timestamp, inherited entities, empty containers).
"""

import json
import pytest

from pydantic import ValidationError as PydanticValidationError

from src.ir.models import (
    ActionDeclaration,
    ActionType,
    Alignment,
    AudioReference,
    EntityDeclaration,
    EntityStyling,
    FlowDirection,
    InheritedEntityReference,
    LayoutConfiguration,
    NarrationAnchor,
    SceneIR,
    SceneSequenceIR,
    TimelineEvent,
    ViewportDeclaration,
    ViewportType,
    VisualTier,
)
from src.ir.validator import (
    SceneIRValidator,
    ValidationError,
    ValidationErrorCode,
    ValidationResult,
)
from src.ir.payload_validator import (
    PayloadErrorCode,
    PayloadValidationError,
    PayloadValidationResult,
    SceneIRPayloadValidator,
)
from src.ir.adapter import (
    SceneIRAdapterError,
    scene_ir_to_storyboard_section,
    sequence_ir_to_storyboard,
    storyboard_section_to_scene_ir,
    storyboard_to_sequence_ir,
)
from src.ir.processor import (
    SceneIRProcessingResult,
    process_stage2_scene_ir_output,
)
from src.ir.fixtures import (
    get_binary_search_fixture,
    get_binary_search_fixture_dict,
    get_chemical_kinetics_fixture,
    get_chemical_kinetics_fixture_dict,
    get_pythagorean_theorem_fixture,
    get_pythagorean_theorem_fixture_dict,
)
from prompts.stage2 import get_prompt2_scene_ir, get_prompt2_storyboard
from prompts.stage3 import get_prompt3_code
from types import SimpleNamespace


# ============================================================================
# Helpers
# ============================================================================

def make_test_scene(**kwargs) -> SceneIR:
    """Helper to build a valid test SceneIR model."""
    defaults = {
        "version": "1.0",
        "scene_id": "TestScene",
        "section_index": 1,
        "title": "Minimal Test Scene",
        "duration_budget": 10.0,
        "layout_configuration": LayoutConfiguration(),
        "viewports": {
            "dock_main": ViewportDeclaration(type=ViewportType.MAIN, sub_slots=["slot_1"]),
            "dock_inspector": ViewportDeclaration(type=ViewportType.INSPECTOR),
        },
        "entities": [
            EntityDeclaration(
                id="test_card",
                component_type="ConceptualCard",
                visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                viewport="dock_main",
                initial_state={"title": "Test Title", "formula": "E = mc^2"},
            )
        ],
        "timeline": [
            TimelineEvent(
                step_id="step_1",
                narration_anchor=NarrationAnchor(
                    lecture_line_index=0,
                    text_cue="We begin by presenting the central equation.",
                    audio_timestamp_range=(0.0, 3.5),
                ),
                actions=[
                    ActionDeclaration(
                        type=ActionType.ENTER,
                        target_entity="test_card",
                        duration=1.0,
                    )
                ],
            )
        ],
        "metadata": {
            "story_beat": "Introduce the equation card",
            "lecture_lines": ["We begin by presenting the central equation."],
            "captions": ["Central equation introduction"],
            "verified_values": ["E = mc^2"],
        },
    }
    defaults.update(kwargs)
    return SceneIR(**defaults)


# ============================================================================
# 1. TestSceneIRGenerationPrompt
# ============================================================================

class TestSceneIRGenerationPrompt:
    """Tests the dedicated Stage 2 Scene IR generation prompt."""

    def test_prompt_contains_exported_json_schema(self):
        outline = {
            "topic": "Binary Search",
            "sections": [{"id": "sec1", "title": "Interval Halving"}],
        }
        prompt = get_prompt2_scene_ir(outline)
        assert "Authoritative JSON Schema Contract" in prompt
        assert "SceneSequenceIR" in prompt
        assert "additionalProperties" in prompt
        assert "version" in prompt

    def test_prompt_contains_visual_registry_manifest(self):
        outline = {"topic": "OS Scheduling"}
        prompt = get_prompt2_scene_ir(outline)
        assert "Available Visual Capabilities" in prompt
        # Check actual registered components appear
        assert "ProcessCardVisual" in prompt
        assert "ReadyQueueVisual" in prompt
        assert "CPUExecutionVisual" in prompt
        assert "GanttChartVisual" in prompt
        assert "MetricTableVisual" in prompt

    def test_prompt_domain_independence(self):
        outline = {"topic": "Photosynthesis Light Reactions"}
        prompt = get_prompt2_scene_ir(outline)
        # Should not force CPU scheduling or Dijkstra text as universal requirements
        assert "Photosynthesis" in prompt
        assert "additionalProperties: false" in prompt
        assert "Output ONLY one valid JSON object" in prompt

    def test_prompt_supports_single_scene_mode(self):
        outline = {"topic": "Calculus Derivatives"}
        prompt = get_prompt2_scene_ir(outline, format_as_sequence=False)
        assert "SceneIR (Section Scene)" in prompt
        assert "SceneIR" in prompt


# ============================================================================
# 2. TestPayloadValidation
# ============================================================================

class TestPayloadValidation:
    """Tests SceneIRPayloadValidator against real VisualRegistry manifests."""

    def setup_method(self):
        self.validator = SceneIRPayloadValidator()

    def test_canonical_fixtures_pass_payload_validation(self):
        bs = get_binary_search_fixture()
        chem = get_chemical_kinetics_fixture()
        pyth = get_pythagorean_theorem_fixture()

        for name, fix in [("BS", bs), ("Chem", chem), ("Pyth", pyth)]:
            res = self.validator.validate(fix)
            assert res.is_valid, f"{name} failed payload validation: {[e.message for e in res.errors]}"

    def test_unsupported_tier1_component_rejected(self):
        scene = make_test_scene(
            entities=[
                EntityDeclaration(
                    id="ghost_comp",
                    component_type="HallucinatedQuantumVisual",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="dock_main",
                )
            ]
        )
        res = self.validator.validate(scene)
        assert not res.is_valid
        assert any(e.error_code == PayloadErrorCode.UNSUPPORTED_COMPONENT_TYPE for e in res.errors)

    def test_unsupported_operation_rejected(self):
        scene = make_test_scene(
            entities=[
                EntityDeclaration(
                    id="pcb_1",
                    component_type="ProcessCardVisual",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="dock_main",
                    initial_state={"pid": "P1", "arrival_time": 0, "burst_time": 4},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_invalid_op",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.COMPONENT_OP,
                            target_entity="pcb_1",
                            operation="non_existent_exploding_method",
                            parameters={},
                        )
                    ],
                )
            ],
        )
        res = self.validator.validate(scene)
        assert not res.is_valid
        assert any(e.error_code == PayloadErrorCode.UNSUPPORTED_COMPONENT_OPERATION for e in res.errors)

    def test_unrecognized_initial_state_param_rejected(self):
        scene = make_test_scene(
            entities=[
                EntityDeclaration(
                    id="pcb_1",
                    component_type="ProcessCardVisual",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="dock_main",
                    initial_state={
                        "pid": "P1",
                        "arrival_time": 0,
                        "burst_time": 4,
                        "fabricated_cpu_voltage": 1.2,
                    },
                )
            ]
        )
        res = self.validator.validate(scene)
        assert not res.is_valid
        assert any(e.error_code == PayloadErrorCode.UNRECOGNIZED_PAYLOAD_PARAM for e in res.errors)

    def test_unrecognized_action_param_rejected(self):
        scene = make_test_scene(
            entities=[
                EntityDeclaration(
                    id="pcb_1",
                    component_type="ProcessCardVisual",
                    visual_tier=VisualTier.TIER1_REGISTERED,
                    viewport="dock_main",
                    initial_state={"pid": "P1", "arrival_time": 0, "burst_time": 4},
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_bad_param",
                    actions=[
                        ActionDeclaration(
                            type=ActionType.COMPONENT_OP,
                            target_entity="pcb_1",
                            operation="animate_set_state",
                            parameters={"new_state": "RUNNING", "hallucinated_flag": True},
                        )
                    ],
                )
            ],
        )
        res = self.validator.validate(scene)
        assert not res.is_valid
        assert any(e.error_code == PayloadErrorCode.UNRECOGNIZED_PAYLOAD_PARAM for e in res.errors)


# ============================================================================
# 3. TestOutputProcessor
# ============================================================================

class TestOutputProcessor:
    """Tests process_stage2_scene_ir_output boundary processor."""

    def test_valid_json_string_processing(self):
        bs_dict = get_binary_search_fixture_dict()
        res = process_stage2_scene_ir_output(json.dumps(bs_dict))
        assert res.is_valid
        assert res.scene_ir is not None
        assert res.scene_ir.scene_id == "Section3Scene"
        assert len(res.all_errors) == 0

    def test_markdown_fence_extraction(self):
        bs_dict = get_binary_search_fixture_dict()
        fenced = f"Here is the generated Scene IR:\n```json\n{json.dumps(bs_dict)}\n```\nExplanation."
        res = process_stage2_scene_ir_output(fenced)
        assert res.is_valid
        assert res.scene_ir is not None

    def test_malformed_json_returns_structural_error(self):
        res = process_stage2_scene_ir_output("{bad_json: missing_quotes}")
        assert not res.is_valid
        assert any(e["error_code"] == "JSON_PARSE_ERROR" for e in res.structural_errors)

    def test_payload_failure_returns_diagnostics(self):
        bad_dict = get_binary_search_fixture_dict()
        bad_dict["entities"].append({
            "id": "bad_ent",
            "component_type": "ProcessCardVisual",
            "visual_tier": "tier1_registered",
            "viewport": "array_viewport",
            "initial_state": {"pid": "P1", "invalid_param_x": 123},
        })
        res = process_stage2_scene_ir_output(bad_dict)
        assert not res.is_valid
        assert len(res.payload_errors) > 0


# ============================================================================
# 4. TestCompatibilityAdapter
# ============================================================================

class TestCompatibilityAdapter:
    """Tests Scene IR <-> Legacy Storyboard compatibility adapter."""

    def test_scene_ir_to_storyboard_section(self):
        bs = get_binary_search_fixture()
        section_dict = scene_ir_to_storyboard_section(bs)

        assert section_dict["id"] == "Section3Scene"
        assert "lecture_lines" in section_dict
        assert len(section_dict["lecture_lines"]) > 0
        assert "captions" in section_dict
        assert len(section_dict["captions"]) == len(section_dict["lecture_lines"])
        assert "animations" in section_dict
        assert len(section_dict["animations"]) == 7

        # Check all animation strings conform to 6-field grammar
        for anim in section_dict["animations"]:
            parts = anim.split("|")
            assert len(parts) == 6, f"Animation string lacks 6 fields: '{anim}'"

    def test_adapter_output_consumed_by_stage3(self):
        bs = get_binary_search_fixture()
        sec_dict = scene_ir_to_storyboard_section(bs)
        sec_obj = SimpleNamespace(**sec_dict)

        prompt3 = get_prompt3_code(
            regenerate_note="",
            section=sec_obj,
            base_class="class TeachingScene: pass",
        )
        assert "Section3Scene" in prompt3
        assert "CompositeNodeGraph" in prompt3

    def test_sequence_ir_to_full_storyboard(self):
        s1 = make_test_scene(scene_id="Scene1", section_index=1)
        s2 = make_test_scene(scene_id="Scene2", section_index=2)
        seq = SceneSequenceIR(
            sequence_id="seq_test",
            title="Module Title",
            sections=[s1, s2],
        )
        storyboard = sequence_ir_to_storyboard(seq)
        assert "visual_conventions" in storyboard
        assert len(storyboard["sections"]) == 2
        assert storyboard["sections"][0]["id"] == "Scene1"
        assert storyboard["sections"][1]["id"] == "Scene2"

    def test_missing_narration_lines_raises_explicit_error(self):
        scene = make_test_scene(
            metadata={},  # No lecture_lines
            timeline=[
                TimelineEvent(
                    step_id="step_no_cue",
                    narration_anchor=None,  # No text cue
                    actions=[ActionDeclaration(type=ActionType.ENTER, target_entity="test_card")],
                )
            ],
        )
        with pytest.raises(SceneIRAdapterError) as exc_info:
            scene_ir_to_storyboard_section(scene)
        assert "missing essential 'lecture_lines'" in str(exc_info.value)

    def test_storyboard_to_sequence_ir_reverse_adapter(self):
        storyboard = {
            "sections": [
                {
                    "id": "section_1",
                    "title": "Sec 1: Introduction",
                    "story_beat": "Introductory beat",
                    "lecture_lines": ["Line 1", "Line 2"],
                    "captions": ["Cap 1", "Cap 2"],
                    "animations": [
                        "L1-1 | FadeIn | ConceptualCard | viewport: dock_main | action: show() | on screen after: [card]",
                        "L2-1 | Indicate | ConceptualCard | viewport: dock_main | action: highlight() | on screen after: [card]",
                    ],
                }
            ]
        }
        seq_ir = storyboard_to_sequence_ir(storyboard)
        assert isinstance(seq_ir, SceneSequenceIR)
        assert len(seq_ir.sections) == 1
        assert seq_ir.sections[0].scene_id == "section_1"
        assert len(seq_ir.sections[0].entities) == 1
        assert len(seq_ir.sections[0].timeline) == 2


# ============================================================================
# 5. TestValidatorDiagnosticsRepairs (Milestone 3A Audit Findings)
# ============================================================================

class TestValidatorDiagnosticsRepairs:
    """Verifies that shadowed error mappings from Milestone 3A are properly resolved."""

    def setup_method(self):
        self.validator = SceneIRValidator()

    def test_invalid_timestamp_range_mapping_from_dict(self):
        raw = {
            "scene_id": "s1",
            "duration_budget": 10.0,
            "viewports": {"vp": {"type": "viewport_main"}},
            "entities": [{"id": "e1", "component_type": "Card", "visual_tier": "tier4_conceptual_card", "viewport": "vp"}],
            "timeline": [{
                "step_id": "st1",
                "narration_anchor": {
                    "lecture_line_index": 0,
                    "text_cue": "cue",
                    "audio_timestamp_range": [5.0, 2.0],  # inverted: start > end
                },
                "actions": [{"type": "enter", "target_entity": "e1"}],
            }],
        }
        res = self.validator.validate(raw)
        assert not res.is_valid
        assert any(e.error_code == ValidationErrorCode.INVALID_TIMESTAMP_RANGE for e in res.errors)

    def test_invalid_inherited_entity_mapping_from_dict(self):
        raw = {
            "scene_id": "s1",
            "duration_budget": 10.0,
            "viewports": {"vp": {"type": "viewport_main"}},
            "entities": [{"id": "e1", "component_type": "Card", "visual_tier": "tier4_conceptual_card", "viewport": "vp"}],
            "inherited_entities": [{"source_section": "", "entity_id": "e2"}],  # empty source_section
            "timeline": [{"step_id": "st1", "actions": [{"type": "enter", "target_entity": "e1"}]}],
        }
        res = self.validator.validate(raw)
        assert not res.is_valid
        assert any(e.error_code == ValidationErrorCode.INVALID_INHERITED_ENTITY for e in res.errors)

    def test_empty_container_diagnostic_assertion(self):
        scene = make_test_scene(entities=[])
        res = self.validator.validate(scene)
        assert not res.is_valid
        assert any(e.error_code == ValidationErrorCode.EMPTY_CONTAINER for e in res.errors)

    def test_missing_required_field_assertion(self):
        raw = {"title": "Only Title - Missing scene_id and duration_budget"}
        res = self.validator.validate(raw)
        assert not res.is_valid
        assert any(e.error_code == ValidationErrorCode.MISSING_REQUIRED_FIELD for e in res.errors)

    def test_invalid_timing_value_assertion(self):
        raw = {
            "scene_id": "s1",
            "duration_budget": -5.0,  # Negative duration
            "viewports": {"vp": {"type": "viewport_main"}},
            "entities": [{"id": "e1", "component_type": "Card", "visual_tier": "tier4_conceptual_card", "viewport": "vp"}],
            "timeline": [{"step_id": "st1", "actions": [{"type": "enter", "target_entity": "e1"}]}],
        }
        res = self.validator.validate(raw)
        assert not res.is_valid
        assert any(e.error_code == ValidationErrorCode.INVALID_TIMING_VALUE for e in res.errors)

    def test_structural_schema_error_assertion(self):
        raw = {
            "scene_id": "s1",
            "duration_budget": 10.0,
            "unexpected_extra_field": "disallowed",  # extra=forbid
            "viewports": {"vp": {"type": "viewport_main"}},
            "entities": [{"id": "e1", "component_type": "Card", "visual_tier": "tier4_conceptual_card", "viewport": "vp"}],
            "timeline": [{"step_id": "st1", "actions": [{"type": "enter", "target_entity": "e1"}]}],
        }
        res = self.validator.validate(raw)
        assert not res.is_valid
        assert any(e.error_code == ValidationErrorCode.STRUCTURAL_SCHEMA_ERROR for e in res.errors)


# ============================================================================
# 6. TestLegacyStage2Preservation
# ============================================================================

class TestLegacyStage2Preservation:
    """Verifies existing Stage 2 prompt generation continues to work identically."""

    def test_legacy_get_prompt2_storyboard_unchanged(self):
        outline = {
            "topic": "Binary Search",
            "sections": [
                {
                    "id": "section_1",
                    "title": "Sec 1: Introduction",
                    "lecture_lines": ["Line 1", "Line 2", "Line 3", "Line 4"],
                    "captions": ["C1", "C2", "C3", "C4"],
                }
            ],
        }
        prompt = get_prompt2_storyboard(outline)
        assert "You are a professional educator and Manim animator." in prompt
        assert "Animation Step Format (MANDATORY - EXACT 6-FIELD DELIMITER GRAMMAR)" in prompt
        assert "dock_main" in prompt
