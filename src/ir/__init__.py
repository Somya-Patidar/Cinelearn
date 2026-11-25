"""Typed Scene Intermediate Representation (Scene IR v1.0) package.

This package defines the domain-independent Scene IR data model, schema validation,
serialization, and canonical fixtures per the CineLearn visual architecture redesign.
"""

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
    get_binary_search_target23_fixture,
    get_binary_search_target23_fixture_dict,
    get_chemical_kinetics_fixture,
    get_chemical_kinetics_fixture_dict,
    get_fcfs_scheduling_fixture,
    get_fcfs_scheduling_fixture_dict,
    get_pythagorean_theorem_fixture,
    get_pythagorean_theorem_fixture_dict,
    get_sjf_scheduling_fixture,
    get_sjf_scheduling_fixture_dict,
)
from src.ir.payload_validator import (
    PayloadErrorCode,
    PayloadValidationError,
    PayloadValidationResult,
    SceneIRPayloadValidator,
)
from src.ir.adapter import (
    SceneIRAdapterError,
    SectionProxy,
    scene_ir_to_storyboard_section,
    scene_ir_to_storyboard_section_obj,
    sequence_ir_to_storyboard,
    storyboard_section_to_scene_ir,
    storyboard_to_sequence_ir,
)
from src.ir.processor import (
    SceneIRProcessingResult,
    process_stage2_scene_ir_output,
)

__all__ = [
    # Models & Enums
    "ActionDeclaration",
    "ActionType",
    "Alignment",
    "AudioReference",
    "CompositeGraphContent",
    "ConceptualCardContent",
    "EasingType",
    "EntityDeclaration",
    "EntityStyling",
    "FlowDirection",
    "InheritedEntityReference",
    "LayoutConfiguration",
    "NarrationAnchor",
    "PrimitiveShapeContent",
    "SceneIR",
    "SceneSequenceIR",
    "TimelineEvent",
    "ViewportDeclaration",
    "ViewportType",
    "VisualTier",
    # Validator
    "SceneIRValidator",
    "ValidationError",
    "ValidationErrorCode",
    "ValidationResult",
    # Serialization
    "deserialize_scene_ir",
    "deserialize_sequence_ir",
    "export_scene_ir_json_schema",
    "serialize_scene_ir",
    "serialize_sequence_ir",
    "verify_round_trip",
    # Fixtures
    "get_binary_search_fixture",
    "get_binary_search_fixture_dict",
    "get_binary_search_target23_fixture",
    "get_binary_search_target23_fixture_dict",
    "get_chemical_kinetics_fixture",
    "get_chemical_kinetics_fixture_dict",
    "get_fcfs_scheduling_fixture",
    "get_fcfs_scheduling_fixture_dict",
    "get_pythagorean_theorem_fixture",
    "get_pythagorean_theorem_fixture_dict",
    "get_sjf_scheduling_fixture",
    "get_sjf_scheduling_fixture_dict",
    # Payload Validator (Milestone 3B)
    "PayloadErrorCode",
    "PayloadValidationError",
    "PayloadValidationResult",
    "SceneIRPayloadValidator",
    # Adapter (Milestone 3B & 5.1)
    "SceneIRAdapterError",
    "SectionProxy",
    "scene_ir_to_storyboard_section",
    "scene_ir_to_storyboard_section_obj",
    "sequence_ir_to_storyboard",
    "storyboard_section_to_scene_ir",
    "storyboard_to_sequence_ir",
    # Processor (Milestone 3B)
    "SceneIRProcessingResult",
    "process_stage2_scene_ir_output",
]
