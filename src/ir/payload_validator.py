"""
CineLearn — Visual Capability & Payload Constraints Validator.
==============================================================

Enforces strict component-level and operation-level payload constraints on Scene IR
data structures based on authoritative VisualRegistry metadata manifests.

Validates:
- Component type registration against VisualRegistry.
- EntityDeclaration.initial_state parameters against component constructor (init_params).
- ActionDeclaration.operation against component semantic methods (operations).
- ActionDeclaration.parameters against operation method parameters.
- Tier 3/4 content structures (ConceptualCard, PrimitiveShape, CompositeNodeGraph).
- Fail-closed rejection of unknown components, operations, and unconstrained dictionary keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .models import (
    ActionDeclaration,
    ActionType,
    EntityDeclaration,
    SceneIR,
    SceneSequenceIR,
    VisualTier,
)


class PayloadErrorCode(str, Enum):
    """Machine-readable error codes for visual capability & payload validation."""
    UNSUPPORTED_COMPONENT_TYPE = "UNSUPPORTED_COMPONENT_TYPE"
    UNSUPPORTED_COMPONENT_OPERATION = "UNSUPPORTED_COMPONENT_OPERATION"
    MISSING_REQUIRED_PAYLOAD_PARAM = "MISSING_REQUIRED_PAYLOAD_PARAM"
    UNRECOGNIZED_PAYLOAD_PARAM = "UNRECOGNIZED_PAYLOAD_PARAM"
    INVALID_PAYLOAD_PARAM_TYPE = "INVALID_PAYLOAD_PARAM_TYPE"
    UNSUPPORTED_TIER_CONFIGURATION = "UNSUPPORTED_TIER_CONFIGURATION"
    INVALID_TIMING_VALUE = "INVALID_TIMING_VALUE"


@dataclass
class PayloadValidationError:
    """Structured, machine-readable payload validation error."""
    error_code: PayloadErrorCode
    message: str
    field_path: str = ""
    entity_id: Optional[str] = None
    component_type: Optional[str] = None
    operation: Optional[str] = None
    parameter: Optional[str] = None
    section_id: Optional[str] = None
    severity: str = "ERROR"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "field_path": self.field_path,
            "entity_id": self.entity_id,
            "component_type": self.component_type,
            "operation": self.operation,
            "parameter": self.parameter,
            "section_id": self.section_id,
            "severity": self.severity,
        }


@dataclass
class PayloadValidationResult:
    """Aggregate result of a Scene IR visual capability validation pass."""
    is_valid: bool
    errors: List[PayloadValidationError] = field(default_factory=list)
    warnings: List[PayloadValidationError] = field(default_factory=list)
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


# Standard allowed parameters for generic actions
STANDARD_ACTION_PARAMS: Dict[ActionType, Set[str]] = {
    ActionType.ENTER: {"shift", "scale", "color", "opacity", "duration"},
    ActionType.EXIT: {"shift", "scale", "opacity", "duration"},
    ActionType.MOVE: {"target_position", "shift", "relative_vector", "new_target_index", "direction", "duration"},
    ActionType.HIGHLIGHT: {"subterm", "color", "scale", "fill_opacity", "duration", "label"},
    ActionType.TRANSFORM: {"formula", "title", "badge", "body", "subtext", "color", "duration", "new_state"},
}

STANDARD_ACTION_PARAM_TYPES: Dict[str, Tuple[type, ...]] = {
    "duration": (int, float),
    "opacity": (int, float),
    "fill_opacity": (int, float),
    "scale": (int, float),
    "color": (str,),
    "label": (str,),
    "subterm": (str,),
    "formula": (str,),
    "title": (str,),
    "badge": (str,),
    "body": (str,),
    "subtext": (str,),
    "new_target_index": (int,),
    "shift": (list, tuple, str),
    "target_position": (list, tuple, str),
    "relative_vector": (list, tuple),
    "direction": (str,),
    "new_state": (dict,),
}

TIER3_PARAM_TYPES: Dict[str, Dict[str, str]] = {
    "PrimitiveShape": {
        "shape_type": "str",
        "width": "float",
        "height": "float",
        "radius": "float",
        "label": "str",
        "color": "str",
        "fill_opacity": "float",
    },
    "PointerIndicator": {
        "target_index": "int",
        "label": "str",
        "color": "str",
        "direction": "str",
        "size": "float",
    },
    "CompositeDissectionSquare": {
        "side_length": "float",
        "configuration": "str",
        "labels": "Union[Dict, Sequence[str]]",
        "color": "str",
    },
    "CompositeNodeGraph": {
        "elements": "Sequence",
        "nodes": "Sequence",
        "edges": "Sequence",
        "layout": "str",
        "directed": "bool",
        "labels": "Dict",
    },
}


def _split_type_parameters(type_str: str) -> List[str]:
    """Splits comma-separated type parameters while respecting nested brackets."""
    parts = []
    current = []
    depth = 0
    for char in type_str:
        if char == "[":
            depth += 1
            current.append(char)
        elif char == "]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts

# Known parameter and operation aliases for Tier 1 registered components
COMPONENT_PARAM_ALIASES: Dict[str, Dict[str, str]] = {
    "ReactionEnergyDiagram": {
        "reactant_energy": "e_reactants",
        "product_energy": "e_products",
        "barrier_peak": "e_transition_state",
        "reaction_label": "reactants_label",
    },
    "ProcessCardVisual": {
        "arrival": "arrival_time",
        "burst": "burst_time",
    },
}

COMPONENT_OP_ALIASES: Dict[str, Dict[str, str]] = {
    "ReactionEnergyDiagram": {
        "animate_highlight_activation_barrier": "animate_show_activation_barrier",
        "highlight_activation_barrier": "animate_show_activation_barrier",
    },
    "GanttChartVisual": {
        "add_block": "animate_add_block",
    },
    "ReadyQueueVisual": {
        "enqueue": "animate_enqueue",
        "dequeue": "animate_dequeue",
    },
    "CPUExecutionVisual": {
        "dispatch": "animate_dispatch",
        "complete": "animate_complete",
    },
    "MetricTableVisual": {
        "show_averages": "animate_show_averages",
        "update_process_metrics": "animate_update_process_metrics",
    },
}

# Recognized Tier 3 primitive types and their allowed initial_state keys
RECOGNIZED_TIER3_PRIMITIVES: Dict[str, Set[str]] = {
    "CompositeNodeGraph": {"elements", "nodes", "edges", "layout", "directed", "labels"},
    "PointerIndicator": {"target_index", "label", "color", "direction", "size"},
    "CompositeDissectionSquare": {"side_length", "configuration", "labels", "color"},
    "PrimitiveShape": {"shape_type", "width", "height", "radius", "label", "color", "fill_opacity"},
    "ProceduralComponent": {"name", "elements", "diagram_type", "labels", "submobjects"},
}

# Recognized Tier 4 conceptual card initial_state keys
RECOGNIZED_TIER4_KEYS: Set[str] = {
    "title", "formula", "body", "badge", "subtext", "math_str", "text", "label"
}

# Recognized Tier 2 composite component specifications (if registered outside manifest)
RECOGNIZED_TIER2_COMPONENTS: Dict[str, Dict[str, Any]] = {}


class SceneIRPayloadValidator:
    """
    Validates Scene IR entity payloads and action parameters against real
    component manifests from VisualRegistry.
    """

    def __init__(self, manifest: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        if manifest is None:
            try:
                from src.domains import VisualRegistry
                manifest = VisualRegistry.get_schema_manifest()
            except Exception:
                manifest = {}
        self.manifest = manifest or {}
        self._component_index: Dict[str, Dict[str, Any]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Flattens the domain-grouped manifest into a fast component lookup index."""
        for domain, comps in self.manifest.items():
            for comp in comps:
                cid = comp.get("component_id")
                if cid:
                    self._component_index[cid] = comp
                for alias in comp.get("aliases", []):
                    self._component_index[alias] = comp

    @staticmethod
    def _is_type_compatible(val: Any, type_str: Optional[str]) -> bool:
        """Determines if a payload parameter value matches the expected manifest type."""
        if not type_str or type_str in ("Any", "Optional[Any]", "Optional", "Union"):
            return True
        if val is None:
            return type_str.startswith("Optional") or "Optional" in type_str or type_str in ("Any", "None")

        clean = type_str.strip()

        # Handle Optional[...] wrapping
        if clean.startswith("Optional[") and clean.endswith("]"):
            clean = clean[len("Optional["):-1].strip()

        # Handle Union[...]
        if clean.startswith("Union[") and clean.endswith("]"):
            inner_u = clean[len("Union["):-1].strip()
            variants = _split_type_parameters(inner_u)
            return any(SceneIRPayloadValidator._is_type_compatible(val, v) for v in variants)

        if clean in ("int", "Optional[int]"):
            return isinstance(val, int) and not isinstance(val, bool)
        if clean in ("float", "Optional[float]"):
            return isinstance(val, (int, float)) and not isinstance(val, bool)
        if clean in ("str", "Optional[str]"):
            return isinstance(val, str)
        if clean in ("bool", "Optional[bool]"):
            return isinstance(val, bool)
        if clean in ("Dict", "dict") or clean.startswith("Dict["):
            return isinstance(val, dict)

        # Handle Sequence / List / Tuple
        if clean in ("Sequence", "Tuple", "list", "tuple"):
            return isinstance(val, (list, tuple))
        if clean.startswith("Sequence[") or clean.startswith("List[") or clean.startswith("Tuple["):
            if not isinstance(val, (list, tuple)):
                return False
            inner = clean[clean.index("[") + 1 : -1].strip()
            if inner and inner != "Any":
                return all(SceneIRPayloadValidator._is_type_compatible(elem, inner) for elem in val)
            return True

        # Component reference or class name (e.g. ProcessCardVisual)
        if clean and clean[0].isupper() and clean.isalnum():
            return isinstance(val, (str, dict)) or hasattr(val, "component_id")

        return True

    def validate(self, scene: SceneIR) -> PayloadValidationResult:
        """Validates all entities and timeline actions in a SceneIR document."""
        errors: List[PayloadValidationError] = []
        warnings: List[PayloadValidationError] = []

        # Index declared entities by ID for action resolution
        entity_map: Dict[str, EntityDeclaration] = {e.id: e for e in scene.entities}

        # 1. Validate entity initial_state payloads
        for idx, entity in enumerate(scene.entities):
            epath = f"entities[{idx}]"
            e_errors = self.validate_entity_payload(entity, path=epath)
            errors.extend(e_errors)

        # 2. Validate action parameters and operations
        for s_idx, event in enumerate(scene.timeline):
            for a_idx, action in enumerate(event.actions):
                apath = f"timeline[{s_idx}].actions[{a_idx}]"
                target_entity = entity_map.get(action.target_entity)
                a_errors = self.validate_action_payload(action, target_entity, path=apath)
                errors.extend(a_errors)

        # 3. Validate timeline duration against scene duration_budget
        if scene.timeline and scene.duration_budget:
            max_timeline_time = 0.0
            for event in scene.timeline:
                if event.narration_anchor and event.narration_anchor.audio_timestamp_range:
                    rng = event.narration_anchor.audio_timestamp_range
                    if len(rng) == 2 and isinstance(rng[1], (int, float)):
                        max_timeline_time = max(max_timeline_time, float(rng[1]))
            if max_timeline_time > float(scene.duration_budget) + 0.10:
                errors.append(
                    PayloadValidationError(
                        error_code=PayloadErrorCode.INVALID_TIMING_VALUE,
                        message=(
                            f"Section '{scene.scene_id}' duration_budget ({scene.duration_budget:.1f}s) is smaller than "
                            f"the scheduled timeline duration ({max_timeline_time:.1f}s). Increase duration_budget to at least "
                            f"{max_timeline_time + 10.0:.1f}s."
                        ),
                        field_path="duration_budget",
                        section_id=scene.scene_id,
                    )
                )

        sec_id = getattr(scene, "scene_id", None)
        if sec_id:
            for err in errors:
                if not err.section_id:
                    err.section_id = sec_id
            for warn in warnings:
                if not warn.section_id:
                    warn.section_id = sec_id

        return PayloadValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            scene_id=scene.scene_id,
        )

    def validate_sequence(self, sequence: SceneSequenceIR) -> PayloadValidationResult:
        """Validates all sections in a SceneSequenceIR document."""
        errors: List[PayloadValidationError] = []
        warnings: List[PayloadValidationError] = []

        for section in sequence.sections:
            res = self.validate(section)
            errors.extend(res.errors)
            warnings.extend(res.warnings)

        return PayloadValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            scene_id=sequence.sequence_id,
        )

    def validate_entity_payload(
        self, entity: EntityDeclaration, path: str = ""
    ) -> List[PayloadValidationError]:
        """Validates an entity's component_type and initial_state parameters."""
        errors: List[PayloadValidationError] = []
        ctype = entity.component_type
        tier = entity.visual_tier

        # Tier 1: Registered Domain Component
        if tier == VisualTier.TIER1_REGISTERED:
            if ctype not in self._component_index:
                errors.append(
                    PayloadValidationError(
                        error_code=PayloadErrorCode.UNSUPPORTED_COMPONENT_TYPE,
                        message=f"Entity '{entity.id}' references unregistered Tier 1 component '{ctype}'. Registered: {sorted(self._component_index.keys())}",
                        field_path=f"{path}.component_type",
                        entity_id=entity.id,
                        component_type=ctype,
                    )
                )
                return errors

            comp_spec = self._component_index[ctype]
            init_params = comp_spec.get("init_params", {})
            param_aliases = COMPONENT_PARAM_ALIASES.get(ctype, {})

            # Check required parameters (accounting for aliases)
            for pname, pinfo in init_params.items():
                if pinfo.get("required"):
                    has_param = pname in entity.initial_state or any(
                        alias in entity.initial_state for alias, target in param_aliases.items() if target == pname
                    )
                    if not has_param:
                        errors.append(
                            PayloadValidationError(
                                error_code=PayloadErrorCode.MISSING_REQUIRED_PAYLOAD_PARAM,
                                message=f"Entity '{entity.id}' ({ctype}) missing required initial_state parameter '{pname}'.",
                                field_path=f"{path}.initial_state",
                                entity_id=entity.id,
                                component_type=ctype,
                                parameter=pname,
                            )
                        )

            # Check unrecognized parameters
            for k in entity.initial_state.keys():
                canonical_k = param_aliases.get(k, k)
                if canonical_k not in init_params:
                    errors.append(
                        PayloadValidationError(
                            error_code=PayloadErrorCode.UNRECOGNIZED_PAYLOAD_PARAM,
                            message=f"Entity '{entity.id}' ({ctype}) specifies unrecognized initial_state parameter '{k}'. Allowed: {sorted(init_params.keys())}",
                            field_path=f"{path}.initial_state.{k}",
                            entity_id=entity.id,
                            component_type=ctype,
                            parameter=k,
                        )
                    )

            # Check parameter types
            for k, val in entity.initial_state.items():
                canonical_k = param_aliases.get(k, k)
                if canonical_k in init_params:
                    pinfo = init_params[canonical_k]
                    type_str = pinfo.get("type")
                    if not self._is_type_compatible(val, type_str):
                        errors.append(
                            PayloadValidationError(
                                error_code=PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE,
                                message=(
                                    f"Entity '{entity.id}' ({ctype}) parameter '{k}' has invalid type "
                                    f"'{type(val).__name__}' (value: {val!r}). Expected type '{type_str}'."
                                ),
                                field_path=f"{path}.initial_state.{k}",
                                entity_id=entity.id,
                                component_type=ctype,
                                parameter=k,
                            )
                        )

        # Tier 4: Conceptual Card
        elif tier == VisualTier.TIER4_CONCEPTUAL_CARD:
            if ctype not in ("ConceptualCard", "MathTex", "TextCard"):
                errors.append(
                    PayloadValidationError(
                        error_code=PayloadErrorCode.UNSUPPORTED_COMPONENT_TYPE,
                        message=f"Tier 4 entity '{entity.id}' specifies unsupported component '{ctype}'. Expected 'ConceptualCard', 'MathTex', or 'TextCard'.",
                        field_path=f"{path}.component_type",
                        entity_id=entity.id,
                        component_type=ctype,
                    )
                )
            for k, val in entity.initial_state.items():
                if k not in RECOGNIZED_TIER4_KEYS:
                    errors.append(
                        PayloadValidationError(
                            error_code=PayloadErrorCode.UNRECOGNIZED_PAYLOAD_PARAM,
                            message=f"ConceptualCard '{entity.id}' specifies unrecognized parameter '{k}'. Allowed: {sorted(RECOGNIZED_TIER4_KEYS)}",
                            field_path=f"{path}.initial_state.{k}",
                            entity_id=entity.id,
                            component_type=ctype,
                            parameter=k,
                        )
                    )
                elif val is not None and not isinstance(val, str):
                    errors.append(
                        PayloadValidationError(
                            error_code=PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE,
                            message=f"ConceptualCard '{entity.id}' parameter '{k}' has invalid type '{type(val).__name__}' (value: {val!r}). Expected 'str'.",
                            field_path=f"{path}.initial_state.{k}",
                            entity_id=entity.id,
                            component_type=ctype,
                            parameter=k,
                        )
                    )

        # Tier 3: Primitive Tree
        elif tier == VisualTier.TIER3_PRIMITIVE_TREE:
            if ctype not in RECOGNIZED_TIER3_PRIMITIVES:
                errors.append(
                    PayloadValidationError(
                        error_code=PayloadErrorCode.UNSUPPORTED_COMPONENT_TYPE,
                        message=f"Tier 3 primitive tree '{entity.id}' specifies unrecognized primitive '{ctype}'. Recognized: {sorted(RECOGNIZED_TIER3_PRIMITIVES.keys())}",
                        field_path=f"{path}.component_type",
                        entity_id=entity.id,
                        component_type=ctype,
                    )
                )
            else:
                allowed_keys = RECOGNIZED_TIER3_PRIMITIVES[ctype]
                expected_types = TIER3_PARAM_TYPES.get(ctype, {})
                for k, val in entity.initial_state.items():
                    if k not in allowed_keys:
                        errors.append(
                            PayloadValidationError(
                                error_code=PayloadErrorCode.UNRECOGNIZED_PAYLOAD_PARAM,
                                message=f"Primitive '{ctype}' ('{entity.id}') specifies unrecognized parameter '{k}'. Allowed: {sorted(allowed_keys)}",
                                field_path=f"{path}.initial_state.{k}",
                                entity_id=entity.id,
                                component_type=ctype,
                                parameter=k,
                            )
                        )
                    elif k in expected_types and val is not None:
                        t_str = expected_types[k]
                        if not self._is_type_compatible(val, t_str):
                            errors.append(
                                PayloadValidationError(
                                    error_code=PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE,
                                    message=(
                                        f"Primitive '{ctype}' ('{entity.id}') parameter '{k}' has invalid type "
                                        f"'{type(val).__name__}' (value: {val!r}). Expected type '{t_str}'."
                                    ),
                                    field_path=f"{path}.initial_state.{k}",
                                    entity_id=entity.id,
                                    component_type=ctype,
                                    parameter=k,
                                )
                            )

        # Tier 2: Multi-Component Composition
        elif tier == VisualTier.TIER2_COMPOSITION:
            comp_spec = self._component_index.get(ctype)
            is_recognized_tier2 = False
            if comp_spec:
                is_recognized_tier2 = bool(
                    comp_spec.get("is_composite")
                    or comp_spec.get("is_container")
                    or comp_spec.get("tier") in (2, "tier2", "tier2_composition", VisualTier.TIER2_COMPOSITION)
                    or comp_spec.get("domain") in ("composite", "composition")
                )
            elif ctype in RECOGNIZED_TIER2_COMPONENTS:
                comp_spec = RECOGNIZED_TIER2_COMPONENTS[ctype]
                is_recognized_tier2 = True

            if not is_recognized_tier2:
                errors.append(
                    PayloadValidationError(
                        error_code=PayloadErrorCode.UNSUPPORTED_COMPONENT_TYPE,
                        message=(
                            f"Entity '{entity.id}' references unsupported Tier 2 composite component '{ctype}'. "
                            f"Tier 2 compositions require a registered composite or container contract."
                        ),
                        field_path=f"{path}.component_type",
                        entity_id=entity.id,
                        component_type=ctype,
                    )
                )
                return errors

            init_params = comp_spec.get("init_params", {})
            param_aliases = COMPONENT_PARAM_ALIASES.get(ctype, {})

            # Check required parameters
            for pname, pinfo in init_params.items():
                if pinfo.get("required"):
                    has_param = pname in entity.initial_state or any(
                        alias in entity.initial_state for alias, target in param_aliases.items() if target == pname
                    )
                    if not has_param:
                        errors.append(
                            PayloadValidationError(
                                error_code=PayloadErrorCode.MISSING_REQUIRED_PAYLOAD_PARAM,
                                message=f"Tier 2 composite entity '{entity.id}' ({ctype}) missing required initial_state parameter '{pname}'.",
                                field_path=f"{path}.initial_state",
                                entity_id=entity.id,
                                component_type=ctype,
                                parameter=pname,
                            )
                        )

            # Check unrecognized parameters
            for k in entity.initial_state.keys():
                canonical_k = param_aliases.get(k, k)
                if canonical_k not in init_params:
                    errors.append(
                        PayloadValidationError(
                            error_code=PayloadErrorCode.UNRECOGNIZED_PAYLOAD_PARAM,
                            message=f"Tier 2 composite entity '{entity.id}' ({ctype}) specifies unrecognized initial_state parameter '{k}'. Allowed: {sorted(init_params.keys())}",
                            field_path=f"{path}.initial_state.{k}",
                            entity_id=entity.id,
                            component_type=ctype,
                            parameter=k,
                        )
                    )

            # Check parameter types
            for k, val in entity.initial_state.items():
                canonical_k = param_aliases.get(k, k)
                if canonical_k in init_params:
                    pinfo = init_params[canonical_k]
                    type_str = pinfo.get("type")
                    if not self._is_type_compatible(val, type_str):
                        errors.append(
                            PayloadValidationError(
                                error_code=PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE,
                                message=(
                                    f"Tier 2 composite entity '{entity.id}' ({ctype}) parameter '{k}' has invalid type "
                                    f"'{type(val).__name__}' (value: {val!r}). Expected type '{type_str}'."
                                ),
                                field_path=f"{path}.initial_state.{k}",
                                entity_id=entity.id,
                                component_type=ctype,
                                parameter=k,
                            )
                        )

        else:
            errors.append(
                PayloadValidationError(
                    error_code=PayloadErrorCode.UNSUPPORTED_TIER_CONFIGURATION,
                    message=f"Entity '{entity.id}' specifies unsupported visual_tier '{tier}'.",
                    field_path=f"{path}.visual_tier",
                    entity_id=entity.id,
                )
            )

        return errors

    def validate_action_payload(
        self,
        action: ActionDeclaration,
        target_entity: Optional[EntityDeclaration],
        path: str = "",
    ) -> List[PayloadValidationError]:
        """Validates an action's operation name and parameter dictionary."""
        errors: List[PayloadValidationError] = []
        atype = action.type
        target_id = action.target_entity

        # If target entity is not declared in scene, Gate 3.5 already flags UNDEFINED_ANIMATION_TARGET
        if target_entity is None:
            return errors

        ctype = target_entity.component_type

        # Case 1: component_op on a Tier 1 or Tier 2 registered component
        if atype == ActionType.COMPONENT_OP:
            if target_entity.visual_tier in (VisualTier.TIER1_REGISTERED, VisualTier.TIER2_COMPOSITION):
                comp_spec = self._component_index.get(ctype) or RECOGNIZED_TIER2_COMPONENTS.get(ctype)
                if not comp_spec:
                    # Already flagged in entity check
                    return errors

                operations = comp_spec.get("operations", {})
                op_name = action.operation
                if not op_name:
                    errors.append(
                        PayloadValidationError(
                            error_code=PayloadErrorCode.UNSUPPORTED_COMPONENT_OPERATION,
                            message=f"Action in '{path}' targeting '{target_id}' has type 'component_op' but specifies no operation. Supported: {sorted(operations.keys())}",
                            field_path=f"{path}.operation",
                            entity_id=target_id,
                            component_type=ctype,
                        )
                    )
                    return errors
                op_aliases = COMPONENT_OP_ALIASES.get(ctype, {})
                canonical_op = op_aliases.get(op_name, op_name)

                if canonical_op not in operations:
                    errors.append(
                        PayloadValidationError(
                            error_code=PayloadErrorCode.UNSUPPORTED_COMPONENT_OPERATION,
                            message=f"Component '{ctype}' does not support operation '{op_name}'. Supported: {sorted(operations.keys())}",
                            field_path=f"{path}.operation",
                            entity_id=target_id,
                            component_type=ctype,
                            operation=op_name,
                        )
                    )
                    return errors

                op_spec = operations[canonical_op]
                op_params = op_spec.get("parameters", {})
                allowed_op_params = set(op_params.keys()) | {"run_time", "label", "color"}

                # Check required parameters for operation
                for pname, pinfo in op_params.items():
                    if pinfo.get("required") and pname not in action.parameters:
                        errors.append(
                            PayloadValidationError(
                                error_code=PayloadErrorCode.MISSING_REQUIRED_PAYLOAD_PARAM,
                                message=f"Operation '{op_name}' on '{target_id}' ({ctype}) missing required parameter '{pname}'.",
                                field_path=f"{path}.parameters",
                                entity_id=target_id,
                                component_type=ctype,
                                operation=op_name,
                                parameter=pname,
                            )
                        )

                # Check unrecognized parameters
                for k in action.parameters.keys():
                    if k not in allowed_op_params:
                        errors.append(
                            PayloadValidationError(
                                error_code=PayloadErrorCode.UNRECOGNIZED_PAYLOAD_PARAM,
                                message=f"Operation '{op_name}' on '{target_id}' ({ctype}) specifies unrecognized parameter '{k}'. Allowed: {sorted(allowed_op_params)}",
                                field_path=f"{path}.parameters.{k}",
                                entity_id=target_id,
                                component_type=ctype,
                                operation=op_name,
                                parameter=k,
                            )
                        )

                # Check operation parameter types
                for k, val in action.parameters.items():
                    if k in op_params:
                        pinfo = op_params[k]
                        type_str = pinfo.get("type")
                        if not self._is_type_compatible(val, type_str):
                            errors.append(
                                PayloadValidationError(
                                    error_code=PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE,
                                    message=(
                                        f"Operation '{op_name}' on '{target_id}' ({ctype}) parameter '{k}' "
                                        f"has invalid type '{type(val).__name__}' (value: {val!r}). Expected type '{type_str}'."
                                    ),
                                    field_path=f"{path}.parameters.{k}",
                                    entity_id=target_id,
                                    component_type=ctype,
                                    operation=op_name,
                                    parameter=k,
                                )
                            )

            elif target_entity.visual_tier == VisualTier.TIER3_PRIMITIVE_TREE:
                # Common primitive operations: dim_subrange, dim_triangles, highlight_subrange, etc.
                op_name = action.operation
                if not op_name:
                    errors.append(
                        PayloadValidationError(
                            error_code=PayloadErrorCode.UNSUPPORTED_COMPONENT_OPERATION,
                            message=f"Action in '{path}' targeting primitive '{target_id}' has type 'component_op' but specifies no operation.",
                            field_path=f"{path}.operation",
                            entity_id=target_id,
                            component_type=ctype,
                        )
                    )

        # Case 2: Standard animation types (ENTER, EXIT, MOVE, HIGHLIGHT, TRANSFORM)
        else:
            allowed_params = STANDARD_ACTION_PARAMS.get(atype, set())
            for k, val in action.parameters.items():
                if k not in allowed_params:
                    errors.append(
                        PayloadValidationError(
                            error_code=PayloadErrorCode.UNRECOGNIZED_PAYLOAD_PARAM,
                            message=f"Action '{atype.value}' targeting '{target_id}' specifies unrecognized parameter '{k}'. Allowed: {sorted(allowed_params)}",
                            field_path=f"{path}.parameters.{k}",
                            entity_id=target_id,
                            component_type=ctype,
                            parameter=k,
                        )
                    )
                elif k in STANDARD_ACTION_PARAM_TYPES:
                    expected_types = STANDARD_ACTION_PARAM_TYPES[k]
                    is_valid = isinstance(val, expected_types)
                    if (int in expected_types or float in expected_types) and isinstance(val, bool):
                        is_valid = False
                    if not is_valid:
                        errors.append(
                            PayloadValidationError(
                                error_code=PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE,
                                message=(
                                    f"Action '{atype.value}' targeting '{target_id}' parameter '{k}' "
                                    f"has invalid type '{type(val).__name__}' (value: {val!r}). "
                                    f"Expected {[t.__name__ for t in expected_types]}."
                                ),
                                field_path=f"{path}.parameters.{k}",
                                entity_id=target_id,
                                component_type=ctype,
                                parameter=k,
                            )
                        )

        return errors
