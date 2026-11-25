"""Unit and integration tests for Milestone 3A: Typed Scene IR Foundation & Model Hierarchy.

Verifies:
1. Model integrity and Pydantic validation (enums, defaults, extra=forbid).
2. Spatial and structural reference integrity (duplicate IDs, viewports, circular parentage).
3. Animation and timeline validation (action targets, audio anchors, non-negative durations).
4. Serialization, schema export, and round-trip determinism.
5. Domain-independent canonical fixtures (Binary Search, Chemical Kinetics, Pythagorean Theorem).
6. Multi-scene sequence validation.
7. Backward compatibility with existing pipeline imports.
"""

import json
import pytest
from pydantic import ValidationError as PydanticValidationError

from src.ir.models import (
    ActionDeclaration,
    ActionType,
    Alignment,
    AudioReference,
    CompositeGraphContent,
    ConceptualCardContent,
    EasingType,
    EntityDeclaration,
    EntityStyling,
    FlowDirection,
    InheritedEntityReference,
    LayoutConfiguration,
    NarrationAnchor,
    PrimitiveShapeContent,
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
from src.ir.serialization import (
    deserialize_scene_ir,
    deserialize_sequence_ir,
    export_scene_ir_json_schema,
    serialize_scene_ir,
    serialize_sequence_ir,
    verify_round_trip,
)
from src.ir.fixtures import (
    get_binary_search_fixture,
    get_binary_search_fixture_dict,
    get_chemical_kinetics_fixture,
    get_chemical_kinetics_fixture_dict,
    get_pythagorean_theorem_fixture,
    get_pythagorean_theorem_fixture_dict,
)


# ============================================================================
# Helpers
# ============================================================================

def make_minimal_valid_scene(**kwargs) -> SceneIR:
    """Construct a minimal valid SceneIR for testing."""
    defaults = {
        "version": "1.0",
        "scene_id": "TestScene",
        "section_index": 1,
        "title": "Minimal Test Scene",
        "duration_budget": 10.0,
        "audio_reference": AudioReference(
            audio_path="audio/test.mp3",
            timestamps_path="audio/test_timestamps.json",
            total_duration=10.0,
        ),
        "layout_configuration": LayoutConfiguration(
            aspect_ratio="16:9",
            has_title_banner=True,
            has_caption_column=True,
            has_subtitles=True,
            safety_padding=0.20,
        ),
        "viewports": {
            "vp_main": ViewportDeclaration(
                type=ViewportType.MAIN,
                flow=FlowDirection.COLUMN,
                alignment=Alignment.CENTER,
                padding=0.20,
                sub_slots=["left", "right"],
            )
        },
        "entities": [
            EntityDeclaration(
                id="card_1",
                component_type="ConceptualCard",
                visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                viewport="vp_main",
                styling=EntityStyling(color="#FFFFFF"),
            )
        ],
        "timeline": [
            TimelineEvent(
                step_id="step_1",
                narration_anchor=NarrationAnchor(
                    lecture_line_index=0,
                    text_cue="Test card appears",
                    audio_timestamp_range=(0.0, 2.0),
                ),
                actions=[
                    ActionDeclaration(
                        type=ActionType.ENTER,
                        target_entity="card_1",
                        duration=1.0,
                    )
                ],
            )
        ],
    }
    defaults.update(kwargs)
    return SceneIR(**defaults)


# ============================================================================
# 1. TestModelIntegrity
# ============================================================================

class TestModelIntegrity:
    """Tests Pydantic validation, enum adherence, and structural constraints."""

    def test_valid_minimal_scene_construction(self):
        scene = make_minimal_valid_scene()
        assert scene.scene_id == "TestScene"
        assert scene.version == "1.0"
        assert scene.duration_budget == 10.0
        assert len(scene.viewports) == 1
        assert len(scene.entities) == 1
        assert len(scene.timeline) == 1

    def test_missing_required_fields_raises_pydantic_error(self):
        with pytest.raises(PydanticValidationError) as exc_info:
            SceneIR(
                # Missing required duration_budget, scene_id
                title="Only Title",
            )
        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "scene_id" in error_fields
        assert "duration_budget" in error_fields

    def test_invalid_viewport_type_rejected(self):
        with pytest.raises(PydanticValidationError):
            ViewportDeclaration(
                type="non_existent_viewport_type",  # invalid enum
            )

    def test_invalid_action_type_rejected(self):
        with pytest.raises(PydanticValidationError):
            ActionDeclaration(
                target_entity="ent_1",
                type="EXPLODE_INTO_PIECES",  # invalid enum
                duration=1.0,
            )

    def test_duration_budget_must_be_positive(self):
        with pytest.raises(PydanticValidationError):
            make_minimal_valid_scene(duration_budget=-5.0)

        with pytest.raises(PydanticValidationError):
            make_minimal_valid_scene(duration_budget=0.0)

    def test_extra_forbidden_fields(self):
        with pytest.raises(PydanticValidationError):
            make_minimal_valid_scene(unexpected_field="disallowed")

    def test_model_dict_and_json_methods(self):
        scene = make_minimal_valid_scene()
        data = scene.to_dict()
        assert isinstance(data, dict)
        assert data["scene_id"] == "TestScene"

        json_str = scene.to_json()
        assert isinstance(json_str, str)
        assert '"scene_id": "TestScene"' in json_str

        restored = SceneIR.from_json(json_str)
        assert restored.scene_id == scene.scene_id


# ============================================================================
# 2. TestReferenceIntegrity
# ============================================================================

class TestReferenceIntegrity:
    """Tests spatial, viewport, and hierarchical parentage validation."""

    def setup_method(self):
        self.validator = SceneIRValidator()

    def test_valid_scene_passes_validation(self):
        scene = make_minimal_valid_scene()
        result = self.validator.validate(scene)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_duplicate_entity_id_rejected(self):
        scene = make_minimal_valid_scene(
            entities=[
                EntityDeclaration(
                    id="duplicate_id",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                ),
                EntityDeclaration(
                    id="duplicate_id",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                ),
            ]
        )
        result = self.validator.validate(scene)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.DUPLICATE_ENTITY_ID in codes

    def test_undeclared_viewport_reference_rejected(self):
        scene = make_minimal_valid_scene(
            entities=[
                EntityDeclaration(
                    id="card_1",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="non_existent_vp",
                )
            ]
        )
        result = self.validator.validate(scene)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.UNDEFINED_VIEWPORT_REFERENCE in codes

    def test_undeclared_slot_reference_rejected(self):
        scene = make_minimal_valid_scene(
            entities=[
                EntityDeclaration(
                    id="card_1",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                    slot="invalid_sub_slot",
                )
            ]
        )
        result = self.validator.validate(scene)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.UNDEFINED_SLOT_REFERENCE in codes

    def test_undeclared_parent_entity_rejected(self):
        scene = make_minimal_valid_scene(
            entities=[
                EntityDeclaration(
                    id="child_ent",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                    parent_id="ghost_parent",
                )
            ]
        )
        result = self.validator.validate(scene)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.UNDEFINED_PARENT_REFERENCE in codes

    def test_self_parenting_rejected(self):
        scene = make_minimal_valid_scene(
            entities=[
                EntityDeclaration(
                    id="loop_ent",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                    parent_id="loop_ent",
                )
            ]
        )
        result = self.validator.validate(scene)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.SELF_PARENT_REFERENCE in codes

    def test_direct_circular_parenting_rejected(self):
        # A -> B -> A
        scene = make_minimal_valid_scene(
            entities=[
                EntityDeclaration(
                    id="ent_a",
                    parent_id="ent_b",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                ),
                EntityDeclaration(
                    id="ent_b",
                    parent_id="ent_a",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                ),
            ]
        )
        result = self.validator.validate(scene)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.CIRCULAR_CONTAINMENT in codes

    def test_indirect_circular_parenting_rejected(self):
        # A -> B -> C -> A
        scene = make_minimal_valid_scene(
            entities=[
                EntityDeclaration(
                    id="ent_a",
                    parent_id="ent_b",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                ),
                EntityDeclaration(
                    id="ent_b",
                    parent_id="ent_c",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                ),
                EntityDeclaration(
                    id="ent_c",
                    parent_id="ent_a",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                ),
            ]
        )
        result = self.validator.validate(scene)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.CIRCULAR_CONTAINMENT in codes

    def test_valid_parent_child_hierarchy_passes(self):
        # Parent -> Child -> Grandchild (DAG without cycle)
        scene = make_minimal_valid_scene(
            entities=[
                EntityDeclaration(
                    id="root_card",
                    component_type="ConceptualCard",
                    visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                    viewport="vp_main",
                ),
                EntityDeclaration(
                    id="child_shape",
                    parent_id="root_card",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                ),
                EntityDeclaration(
                    id="grandchild_shape",
                    parent_id="child_shape",
                    component_type="PrimitiveShape",
                    visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                    viewport="vp_main",
                ),
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_dag",
                    actions=[ActionDeclaration(target_entity="root_card", type=ActionType.ENTER)],
                )
            ],
        )
        result = self.validator.validate(scene)
        assert result.is_valid


# ============================================================================
# 3. TestAnimationAndTimelineValidation
# ============================================================================

class TestAnimationAndTimelineValidation:
    """Tests timeline events, action target validation, and duration sanity."""

    def setup_method(self):
        self.validator = SceneIRValidator()

    def test_undeclared_target_entity_in_action_rejected(self):
        scene = make_minimal_valid_scene(
            timeline=[
                TimelineEvent(
                    step_id="step_ghost",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Ghost action",
                        audio_timestamp_range=(0.0, 1.0),
                    ),
                    actions=[
                        ActionDeclaration(
                            target_entity="non_existent_entity",
                            type=ActionType.ENTER,
                            duration=1.0,
                        )
                    ],
                )
            ]
        )
        result = self.validator.validate(scene)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.UNDEFINED_ANIMATION_TARGET in codes

    def test_duplicate_step_id_rejected(self):
        scene = make_minimal_valid_scene(
            timeline=[
                TimelineEvent(
                    step_id="dup_step",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Phrase 1",
                        audio_timestamp_range=(0.0, 1.0),
                    ),
                    actions=[ActionDeclaration(target_entity="card_1", type=ActionType.ENTER)],
                ),
                TimelineEvent(
                    step_id="dup_step",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=1,
                        text_cue="Phrase 2",
                        audio_timestamp_range=(1.0, 2.0),
                    ),
                    actions=[ActionDeclaration(target_entity="card_1", type=ActionType.HIGHLIGHT)],
                ),
            ]
        )
        result = self.validator.validate(scene)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.DUPLICATE_STEP_ID in codes

    def test_negative_action_duration_rejected(self):
        with pytest.raises(PydanticValidationError):
            ActionDeclaration(
                target_entity="card_1",
                type=ActionType.ENTER,
                duration=-1.5,
            )

    def test_invalid_audio_anchor_timestamp_range(self):
        with pytest.raises(PydanticValidationError):
            NarrationAnchor(
                lecture_line_index=0,
                text_cue="Inverted range",
                audio_timestamp_range=(5.0, 2.0),  # start > end
            )

    def test_negative_audio_anchor_timestamp(self):
        with pytest.raises(PydanticValidationError):
            NarrationAnchor(
                lecture_line_index=0,
                text_cue="Negative start",
                audio_timestamp_range=(-1.0, 2.0),
            )

    def test_inherited_entity_target_in_action_is_valid(self):
        scene = make_minimal_valid_scene(
            inherited_entities=[
                InheritedEntityReference(
                    entity_id="inherited_diagram",
                    source_section="Section1Scene",
                    carry_over_state="terminal_state",
                )
            ],
            timeline=[
                TimelineEvent(
                    step_id="step_inherited",
                    narration_anchor=NarrationAnchor(
                        lecture_line_index=0,
                        text_cue="Transform inherited diagram",
                        audio_timestamp_range=(0.0, 1.2),
                    ),
                    actions=[
                        ActionDeclaration(
                            target_entity="inherited_diagram",
                            type=ActionType.TRANSFORM,
                            duration=1.2,
                        )
                    ],
                )
            ],
        )
        result = self.validator.validate(scene)
        assert result.is_valid


# ============================================================================
# 4. TestSerializationAndRoundTrip
# ============================================================================

class TestSerializationAndRoundTrip:
    """Tests JSON serialization, deserialization, and lossless round-tripping."""

    def test_serialize_and_deserialize_scene_ir(self):
        scene = make_minimal_valid_scene()
        json_output = serialize_scene_ir(scene, indent=2)
        assert isinstance(json_output, str)
        assert "TestScene" in json_output

        restored = deserialize_scene_ir(json_output)
        assert restored.scene_id == scene.scene_id
        assert restored.version == scene.version
        assert restored.duration_budget == scene.duration_budget
        assert len(restored.entities) == len(scene.entities)

    def test_lossless_round_trip_verification(self):
        scene = make_minimal_valid_scene()
        success, diff = verify_round_trip(scene)
        assert success, f"Round-trip failed: {diff}"

    def test_json_schema_export(self):
        schema = export_scene_ir_json_schema()
        assert isinstance(schema, dict)
        assert "$defs" in schema or "properties" in schema
        properties = schema.get("properties", {})
        assert "scene_id" in properties
        assert "version" in properties
        assert "viewports" in properties
        assert "entities" in properties
        assert "timeline" in properties

    def test_malformed_json_raises_error(self):
        with pytest.raises(Exception):
            deserialize_scene_ir("{broken_json: not_quoted}")

    def test_unsupported_version_detected_by_validator(self):
        raw_dict = make_minimal_valid_scene().to_dict()
        raw_dict["version"] = "99.9"
        validator = SceneIRValidator()
        result = validator.validate(raw_dict)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.INVALID_SCHEMA_VERSION in codes


# ============================================================================
# 5. TestDomainIndependence
# ============================================================================

class TestDomainIndependence:
    """Verifies that canonical fixtures across three distinct domains all adhere to Scene IR."""

    def setup_method(self):
        self.validator = SceneIRValidator()

    def test_binary_search_fixture_validates_and_round_trips(self):
        fixture = get_binary_search_fixture()
        assert fixture.scene_id == "Section3Scene"
        assert fixture.version == "1.0"
        result = self.validator.validate(fixture)
        assert result.is_valid, f"Binary Search validation failed: {result.errors}"
        success, diff = verify_round_trip(fixture)
        assert success, f"Binary Search round-trip failed: {diff}"

    def test_chemical_kinetics_fixture_validates_and_round_trips(self):
        fixture = get_chemical_kinetics_fixture()
        assert fixture.scene_id == "Section2Scene"
        assert fixture.version == "1.0"
        result = self.validator.validate(fixture)
        assert result.is_valid, f"Chemical Kinetics validation failed: {result.errors}"
        success, diff = verify_round_trip(fixture)
        assert success, f"Chemical Kinetics round-trip failed: {diff}"

    def test_pythagorean_theorem_fixture_validates_and_round_trips(self):
        fixture = get_pythagorean_theorem_fixture()
        assert fixture.scene_id == "Section4Scene"
        assert fixture.version == "1.0"
        result = self.validator.validate(fixture)
        assert result.is_valid, f"Pythagorean Theorem validation failed: {result.errors}"
        success, diff = verify_round_trip(fixture)
        assert success, f"Pythagorean Theorem round-trip failed: {diff}"

    def test_fixtures_dict_constructors(self):
        # Ensure raw dict constructors also produce valid dictionaries
        bs_dict = get_binary_search_fixture_dict()
        chem_dict = get_chemical_kinetics_fixture_dict()
        pyth_dict = get_pythagorean_theorem_fixture_dict()

        assert bs_dict["scene_id"] == "Section3Scene"
        assert chem_dict["scene_id"] == "Section2Scene"
        assert pyth_dict["scene_id"] == "Section4Scene"

        for d in [bs_dict, chem_dict, pyth_dict]:
            res = self.validator.validate(d)
            assert res.is_valid, f"Raw dict validation failed: {res.errors}"

    def test_zero_domain_specific_coupling_in_schema(self):
        """Verifies schema contains no domain-specific algorithmic hardcoding."""
        schema = export_scene_ir_json_schema()
        schema_text = json.dumps(schema).lower()

        forbidden_domain_terms = [
            "fcfs",
            "dijkstra",
            "cpu_scheduling",
            "arrhenius",
            "chemical_kinetics",
            "pythagorean",
            "burst_time",
            "turnaround_time",
        ]
        for term in forbidden_domain_terms:
            assert term not in schema_text, f"Schema contains domain-specific term: '{term}'"


# ============================================================================
# 6. TestSequenceIR
# ============================================================================

class TestSequenceIR:
    """Tests multi-scene sequence container and cross-scene validation."""

    def setup_method(self):
        self.validator = SceneIRValidator()

    def test_valid_sequence_validates(self):
        s1 = make_minimal_valid_scene(scene_id="Scene1", section_index=1)
        s2 = make_minimal_valid_scene(scene_id="Scene2", section_index=2)
        seq = SceneSequenceIR(
            sequence_id="seq_full_module",
            title="Full Educational Module",
            sections=[s1, s2],
        )
        assert seq.total_scenes == 2
        assert seq.total_duration == 20.0

        result = self.validator.validate_sequence(seq)
        assert result.is_valid

    def test_duplicate_scene_id_in_sequence_rejected(self):
        s1 = make_minimal_valid_scene(scene_id="DuplicateScene", section_index=1)
        s2 = make_minimal_valid_scene(scene_id="DuplicateScene", section_index=2)
        seq = SceneSequenceIR(
            sequence_id="seq_dup",
            title="Duplicate Scene Sequence",
            sections=[s1, s2],
        )
        result = self.validator.validate_sequence(seq)
        assert not result.is_valid
        codes = [e.error_code for e in result.errors]
        assert ValidationErrorCode.DUPLICATE_SECTION_ID in codes

    def test_sequence_serialization_and_deserialization(self):
        s1 = make_minimal_valid_scene(scene_id="Scene1", section_index=1)
        s2 = make_minimal_valid_scene(scene_id="Scene2", section_index=2)
        seq = SceneSequenceIR(
            sequence_id="seq_test",
            title="Test Sequence",
            sections=[s1, s2],
        )
        serialized = serialize_sequence_ir(seq)
        restored = deserialize_sequence_ir(serialized)
        assert restored.sequence_id == seq.sequence_id
        assert restored.total_scenes == 2


# ============================================================================
# 7. TestBackwardCompatibility
# ============================================================================

class TestBackwardCompatibility:
    """Verifies existing pipeline imports and modules are undisturbed by Scene IR."""

    def test_existing_pipeline_imports(self):
        import prompts.stage1 as stage1
        import prompts.stage2 as stage2
        import prompts.stage3 as stage3
        import src.agent as agent
        import src.storyboard_fidelity_checker as sfc

        assert hasattr(stage1, "get_prompt1_outline")
        assert hasattr(stage2, "get_prompt2_storyboard")
        assert hasattr(stage3, "get_prompt3_code")
        assert hasattr(agent, "process_knowledge_point")
        assert hasattr(sfc, "StoryboardFidelityChecker")
