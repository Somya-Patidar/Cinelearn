"""
CineLearn — Deterministic Pre-Layout Scene IR Validator (Gate 3.5).
==================================================================

Implements deterministic, model-independent validation for Scene IR schemas (v1.0)
conforming to docs/architecture/scene_ir_and_spatial_composition_specification.md.

Validates:
- Schema version and mandatory fields
- Scope identifier uniqueness (entities, steps, viewports)
- Viewport and sub-slot reference integrity
- Parent-child hierarchy and circular containment cycle detection
- Timeline event and animation action target existence
- Temporal ordering, non-negative durations, and audio timestamp ranges
- Cross-section inherited entity reference integrity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import ValidationError as PydanticValidationError

from .models import (
    ActionDeclaration,
    ActionType,
    EntityDeclaration,
    SceneIR,
    SceneSequenceIR,
    SUPPORTED_SCENE_IR_VERSIONS,
    TimelineEvent,
    ViewportDeclaration,
)


class ValidationErrorCode(str, Enum):
    """Machine-readable diagnostic error codes for Scene IR Gate 3.5."""
    INVALID_SCHEMA_VERSION = "INVALID_SCHEMA_VERSION"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    EMPTY_CONTAINER = "EMPTY_CONTAINER"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    DUPLICATE_ENTITY_ID = "DUPLICATE_ENTITY_ID"
    DUPLICATE_STEP_ID = "DUPLICATE_STEP_ID"
    DUPLICATE_SECTION_ID = "DUPLICATE_SECTION_ID"
    UNDEFINED_VIEWPORT_REFERENCE = "UNDEFINED_VIEWPORT_REFERENCE"
    UNDEFINED_SLOT_REFERENCE = "UNDEFINED_SLOT_REFERENCE"
    UNDEFINED_PARENT_REFERENCE = "UNDEFINED_PARENT_REFERENCE"
    SELF_PARENT_REFERENCE = "SELF_PARENT_REFERENCE"
    CIRCULAR_CONTAINMENT = "CIRCULAR_CONTAINMENT"
    UNDEFINED_ANIMATION_TARGET = "UNDEFINED_ANIMATION_TARGET"
    INVALID_TIMING_VALUE = "INVALID_TIMING_VALUE"
    INVALID_TIMESTAMP_RANGE = "INVALID_TIMESTAMP_RANGE"
    INVALID_INHERITED_ENTITY = "INVALID_INHERITED_ENTITY"
    STRUCTURAL_SCHEMA_ERROR = "STRUCTURAL_SCHEMA_ERROR"


@dataclass
class ValidationError:
    """Structured, machine-readable validation error."""
    error_code: ValidationErrorCode
    message: str
    field_path: str = ""
    entity_id: Optional[str] = None
    section_id: Optional[str] = None
    severity: str = "ERROR"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "field_path": self.field_path,
            "entity_id": self.entity_id,
            "section_id": self.section_id,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """Aggregate result of a Scene IR validation pass."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    scene_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "scene_id": self.scene_id,
        }


class SceneIRValidator:
    """
    Deterministic validator enforcing Gate 3.5 rules on Scene IR data structures.
    Operates without LLM or rendering dependencies.
    """

    @classmethod
    def validate(cls, scene: Union[SceneIR, Dict[str, Any]]) -> ValidationResult:
        """
        Validates a single Section Scene IR instance or dictionary.
        Returns a structured ValidationResult.
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        # 1. Structural parse & Pydantic schema validation
        scene_obj: Optional[SceneIR] = None
        if isinstance(scene, SceneIR):
            scene_obj = scene
        elif isinstance(scene, dict):
            # Check version explicitly before Pydantic to yield specific error code
            ver = str(scene.get("version", "")).strip().lower()
            if ver and ver not in SUPPORTED_SCENE_IR_VERSIONS:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.INVALID_SCHEMA_VERSION,
                        message=f"Unsupported Scene IR version '{scene.get('version')}'. Supported: {sorted(SUPPORTED_SCENE_IR_VERSIONS)}",
                        field_path="version",
                        section_id=str(scene.get("scene_id")),
                    )
                )

            try:
                scene_obj = SceneIR.from_dict(scene)
            except PydanticValidationError as pe:
                for err in pe.errors():
                    loc_tuple = tuple(str(x) for x in err.get("loc", []))
                    loc = ".".join(loc_tuple)
                    msg = err.get("msg", "Invalid value")
                    err_type = str(err.get("type", ""))

                    if "audio_timestamp_range" in loc_tuple:
                        code = ValidationErrorCode.INVALID_TIMESTAMP_RANGE
                    elif loc_tuple and loc_tuple[0] == "inherited_entities":
                        code = ValidationErrorCode.INVALID_INHERITED_ENTITY
                    elif "missing" in err_type:
                        code = ValidationErrorCode.MISSING_REQUIRED_FIELD
                    elif any(k in loc_tuple for k in ("duration", "duration_budget", "post_action_pause")):
                        code = ValidationErrorCode.INVALID_TIMING_VALUE
                    elif "version" in loc_tuple:
                        code = ValidationErrorCode.INVALID_SCHEMA_VERSION
                    else:
                        code = ValidationErrorCode.STRUCTURAL_SCHEMA_ERROR

                    errors.append(
                        ValidationError(
                            error_code=code,
                            message=f"{loc}: {msg}",
                            field_path=loc,
                            section_id=str(scene.get("scene_id")) if isinstance(scene, dict) else None,
                        )
                    )
            except Exception as e:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.STRUCTURAL_SCHEMA_ERROR,
                        message=f"Failed to parse Scene IR: {e}",
                        field_path="root",
                        section_id=str(scene.get("scene_id")) if isinstance(scene, dict) else None,
                    )
                )

        if scene_obj is None:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                scene_id=str(scene.get("scene_id")) if isinstance(scene, dict) else None,
            )

        scene_id = scene_obj.scene_id

        # 2. Schema version check
        if scene_obj.version not in SUPPORTED_SCENE_IR_VERSIONS:
            errors.append(
                ValidationError(
                    error_code=ValidationErrorCode.INVALID_SCHEMA_VERSION,
                    message=f"Unsupported Scene IR version '{scene_obj.version}'.",
                    field_path="version",
                    section_id=scene_id,
                )
            )

        # 3. Non-empty containers
        if not scene_obj.viewports:
            errors.append(
                ValidationError(
                    error_code=ValidationErrorCode.EMPTY_CONTAINER,
                    message="Scene specifies no viewports. At least one viewport must be declared.",
                    field_path="viewports",
                    section_id=scene_id,
                )
            )

        if not scene_obj.entities:
            if scene_obj.inherited_entities:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.UNSUPPORTED_CAPABILITY,
                        message=(
                            "Scene specifies no local visual entities and relies on 'inherited_entities'. "
                            "Cross-section entity inheritance is not supported in the compiler; every section "
                            "must declare all visual entities it uses directly in its local 'entities' list."
                        ),
                        field_path="entities",
                        section_id=scene_id,
                    )
                )
            else:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.EMPTY_CONTAINER,
                        message="Scene specifies no visual entities.",
                        field_path="entities",
                        section_id=scene_id,
                    )
                )

        if not scene_obj.timeline:
            errors.append(
                ValidationError(
                    error_code=ValidationErrorCode.EMPTY_CONTAINER,
                    message="Scene specifies no timeline events.",
                    field_path="timeline",
                    section_id=scene_id,
                )
            )

        # 4. Viewports & slots reference index
        declared_viewports = set(scene_obj.viewports.keys())
        viewport_slots: Dict[str, Set[str]] = {
            vp_id: set(vp_decl.sub_slots) for vp_id, vp_decl in scene_obj.viewports.items()
        }

        # 5. Entity ID uniqueness & reference integrity
        seen_entity_ids: Set[str] = set()
        parent_map: Dict[str, str] = {}  # child_id -> parent_id

        for idx, entity in enumerate(scene_obj.entities):
            eid = entity.id
            path = f"entities[{idx}]"

            # Check duplicate entity ID
            if eid in seen_entity_ids:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.DUPLICATE_ENTITY_ID,
                        message=f"Duplicate entity ID '{eid}' detected in scene.",
                        field_path=f"{path}.id",
                        entity_id=eid,
                        section_id=scene_id,
                    )
                )
            else:
                seen_entity_ids.add(eid)

            # Check viewport reference
            if entity.viewport not in declared_viewports:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.UNDEFINED_VIEWPORT_REFERENCE,
                        message=f"Entity '{eid}' references undeclared viewport '{entity.viewport}'. Declared viewports: {sorted(declared_viewports)}",
                        field_path=f"{path}.viewport",
                        entity_id=eid,
                        section_id=scene_id,
                    )
                )

            # Check sub-slot reference
            if entity.slot:
                valid_slots = viewport_slots.get(entity.viewport, set())
                if entity.slot not in valid_slots:
                    errors.append(
                        ValidationError(
                            error_code=ValidationErrorCode.UNDEFINED_SLOT_REFERENCE,
                            message=f"Entity '{eid}' references slot '{entity.slot}' which is not in viewport '{entity.viewport}' sub_slots {sorted(valid_slots)}.",
                            field_path=f"{path}.slot",
                            entity_id=eid,
                            section_id=scene_id,
                        )
                    )

            # Register parent relationship
            if entity.parent_id:
                if entity.parent_id == eid:
                    errors.append(
                        ValidationError(
                            error_code=ValidationErrorCode.SELF_PARENT_REFERENCE,
                            message=f"Entity '{eid}' cannot reference itself as parent.",
                            field_path=f"{path}.parent_id",
                            entity_id=eid,
                            section_id=scene_id,
                        )
                    )
                else:
                    parent_map[eid] = entity.parent_id

        # 6. Validate parent existence and circular containment
        for child_id, parent_id in parent_map.items():
            if parent_id not in seen_entity_ids:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.UNDEFINED_PARENT_REFERENCE,
                        message=f"Entity '{child_id}' references undefined parent entity '{parent_id}'.",
                        field_path="entities.parent_id",
                        entity_id=child_id,
                        section_id=scene_id,
                    )
                )

        # Detect containment cycles via cycle trace
        cls._detect_containment_cycles(parent_map, scene_id, errors)

        # 7. Timeline event validation
        seen_step_ids: Set[str] = set()
        for s_idx, event in enumerate(scene_obj.timeline):
            sid = event.step_id
            spath = f"timeline[{s_idx}]"

            if sid in seen_step_ids:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.DUPLICATE_STEP_ID,
                        message=f"Duplicate step ID '{sid}' in timeline.",
                        field_path=f"{spath}.step_id",
                        section_id=scene_id,
                    )
                )
            else:
                seen_step_ids.add(sid)

            # Validate narration anchor timestamps
            if event.narration_anchor:
                t_start, t_end = event.narration_anchor.audio_timestamp_range
                if t_start < 0 or t_end < 0 or t_start > t_end:
                    errors.append(
                        ValidationError(
                            error_code=ValidationErrorCode.INVALID_TIMESTAMP_RANGE,
                            message=f"Step '{sid}' has invalid audio timestamp range [{t_start}, {t_end}].",
                            field_path=f"{spath}.narration_anchor.audio_timestamp_range",
                            section_id=scene_id,
                        )
                    )

            # Validate actions and target entity references
            inherited_entity_ids = {inh.entity_id for inh in scene_obj.inherited_entities if inh.entity_id}
            all_valid_targets = seen_entity_ids | inherited_entity_ids

            for a_idx, action in enumerate(event.actions):
                apath = f"{spath}.actions[{a_idx}]"
                target = action.target_entity

                if target not in all_valid_targets:
                    errors.append(
                        ValidationError(
                            error_code=ValidationErrorCode.UNDEFINED_ANIMATION_TARGET,
                            message=f"Action '{action.type.value}' in step '{sid}' targets undefined entity '{target}'.",
                            field_path=f"{apath}.target_entity",
                            entity_id=target,
                            section_id=scene_id,
                        )
                    )

                if action.duration < 0.0 or action.post_action_pause < 0.0:
                    errors.append(
                        ValidationError(
                            error_code=ValidationErrorCode.INVALID_TIMING_VALUE,
                            message=f"Action in step '{sid}' has negative duration or pause: duration={action.duration}, pause={action.post_action_pause}.",
                            field_path=f"{apath}.duration",
                            section_id=scene_id,
                        )
                    )

        # 8. Inherited entities validation
        for i_idx, inh in enumerate(scene_obj.inherited_entities):
            ipath = f"inherited_entities[{i_idx}]"
            if not inh.source_section or not inh.entity_id:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.INVALID_INHERITED_ENTITY,
                        message=f"Inherited entity at index {i_idx} must specify both source_section and entity_id.",
                        field_path=ipath,
                        section_id=scene_id,
                    )
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            scene_id=scene_id,
        )

    @classmethod
    def validate_sequence(cls, sequence: Union[SceneSequenceIR, Dict[str, Any]]) -> ValidationResult:
        """
        Validates an entire multi-section sequence document.
        Verifies individual sections and sequence-level invariants (e.g. unique scene_ids).
        """
        errors: List[ValidationError] = []
        warnings: List[ValidationError] = []

        seq_obj: Optional[SceneSequenceIR] = None
        if isinstance(sequence, SceneSequenceIR):
            seq_obj = sequence
        elif isinstance(sequence, dict):
            try:
                seq_obj = SceneSequenceIR.from_dict(sequence)
            except Exception as e:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.STRUCTURAL_SCHEMA_ERROR,
                        message=f"Failed to parse SceneSequenceIR: {e}",
                        field_path="root",
                    )
                )
                return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        if seq_obj is None:
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        seen_scene_ids: Set[str] = set()
        for idx, section in enumerate(seq_obj.sections):
            sid = section.scene_id
            if sid in seen_scene_ids:
                errors.append(
                    ValidationError(
                        error_code=ValidationErrorCode.DUPLICATE_SECTION_ID,
                        message=f"Duplicate scene_id '{sid}' in sequence at section {idx}.",
                        field_path=f"sections[{idx}].scene_id",
                        section_id=sid,
                    )
                )
            else:
                seen_scene_ids.add(sid)

            sec_res = cls.validate(section)
            errors.extend(sec_res.errors)
            warnings.extend(sec_res.warnings)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            scene_id=seq_obj.sequence_id,
        )

    @classmethod
    def _detect_containment_cycles(
        cls,
        parent_map: Dict[str, str],
        scene_id: str,
        errors: List[ValidationError],
    ) -> None:
        """Detects cyclic parent-child chains using depth-first cycle search."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            parent = parent_map.get(node)
            if parent and parent in parent_map:
                if parent in rec_stack:
                    # Cycle found!
                    cycle_start = path.index(parent)
                    cycle_str = " -> ".join(path[cycle_start:] + [parent])
                    errors.append(
                        ValidationError(
                            error_code=ValidationErrorCode.CIRCULAR_CONTAINMENT,
                            message=f"Circular parent-child containment cycle detected: {cycle_str}",
                            field_path="entities.parent_id",
                            entity_id=node,
                            section_id=scene_id,
                        )
                    )
                elif parent not in visited:
                    dfs(parent, path)

            path.pop()
            rec_stack.remove(node)

        for entity_id in list(parent_map.keys()):
            if entity_id not in visited:
                dfs(entity_id, [])
