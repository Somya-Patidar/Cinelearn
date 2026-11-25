"""
CineLearn — Stage 2 Scene IR Deterministic Output Processor.
=============================================================

Implements the deterministic processing boundary for Stage 2 Scene IR generation:
1. Receives raw LLM response string or dictionary.
2. Extracts and parses clean JSON without markdown fences.
3. Deserializes through typed Pydantic v2 Scene IR models.
4. Enforces Pre-Layout Gate 3.5 semantic validation (SceneIRValidator).
5. Enforces VisualRegistry capability and payload constraints (SceneIRPayloadValidator).
6. Returns typed, verified Scene IR models or structured diagnostic error reports.

Guarantees:
- Zero silent field deletion or arbitrary automatic repair.
- Zero fabricated references or unconstrained dictionary acceptance.
- Returns comprehensive diagnostic codes and field paths for bounded downstream correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional, Union

from .models import SceneIR, SceneSequenceIR
from .validator import SceneIRValidator, ValidationResult
from .payload_validator import SceneIRPayloadValidator, PayloadValidationResult
from .serialization import deserialize_scene_ir, deserialize_sequence_ir


def _extract_json_block(text: str) -> str:
    """Extracts raw JSON content from markdown code fences or unformatted text."""
    text_clean = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text_clean, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text_clean


@dataclass
class SceneIRProcessingResult:
    """Structured result of processing and validating a Stage 2 Scene IR output."""
    is_valid: bool
    scene_ir: Optional[SceneIR] = None
    sequence_ir: Optional[SceneSequenceIR] = None
    structural_errors: List[Dict[str, Any]] = field(default_factory=list)
    semantic_errors: List[Dict[str, Any]] = field(default_factory=list)
    payload_errors: List[Dict[str, Any]] = field(default_factory=list)
    raw_json: Optional[Dict[str, Any]] = None
    summary: str = ""

    @property
    def all_errors(self) -> List[Dict[str, Any]]:
        """Returns all aggregated errors across structural, semantic, and payload passes."""
        return self.structural_errors + self.semantic_errors + self.payload_errors

    @property
    def has_structural_errors(self) -> bool:
        """True if raw JSON parsing, input types, or Pydantic structural validation failed."""
        return len(self.structural_errors) > 0

    @property
    def has_semantic_errors(self) -> bool:
        """True if Gate 3.5 identifier, reference, or hierarchy validation failed."""
        return len(self.semantic_errors) > 0

    @property
    def has_payload_errors(self) -> bool:
        """True if component payload, manifest, parameter, or type validation failed."""
        return len(self.payload_errors) > 0

    @property
    def has_unsupported_capabilities(self) -> bool:
        """True if errors include unregistered component types or unsupported operations."""
        return any(
            err.get("error_code") in ("UNSUPPORTED_COMPONENT_TYPE", "UNSUPPORTED_COMPONENT_OPERATION")
            for err in self.payload_errors
        )

    @property
    def has_unsupported_component_types(self) -> bool:
        """True if errors include genuinely unregistered component types."""
        return any(
            err.get("error_code") == "UNSUPPORTED_COMPONENT_TYPE"
            for err in self.payload_errors
        )

    @property
    def has_recoverable_diagnostics(self) -> bool:
        """True if errors are recoverable via bounded repair (not fatal syntax or unsupported component types)."""
        if self.is_valid:
            return False
        for err in self.structural_errors:
            if err.get("error_code") in ("JSON_PARSE_ERROR", "INVALID_INPUT_TYPE"):
                return False
        if self.has_unsupported_component_types:
            return False
        return len(self.all_errors) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "has_scene_ir": self.scene_ir is not None,
            "has_sequence_ir": self.sequence_ir is not None,
            "total_errors": len(self.all_errors),
            "has_structural_errors": self.has_structural_errors,
            "has_semantic_errors": self.has_semantic_errors,
            "has_payload_errors": self.has_payload_errors,
            "has_unsupported_capabilities": self.has_unsupported_capabilities,
            "has_unsupported_component_types": self.has_unsupported_component_types,
            "has_recoverable_diagnostics": self.has_recoverable_diagnostics,
            "structural_errors": self.structural_errors,
            "semantic_errors": self.semantic_errors,
            "payload_errors": self.payload_errors,
            "summary": self.summary,
        }


def process_stage2_scene_ir_output(
    response: Union[str, Dict[str, Any]],
    component_manifest: Optional[Dict[str, Any]] = None,
) -> SceneIRProcessingResult:
    """
    Deterministically processes and validates Stage 2 Scene IR output.

    Parameters:
    - response: Raw string output from LLM (potentially containing markdown fences) or dictionary.
    - component_manifest: Optional pre-fetched VisualRegistry manifest.

    Returns:
    - SceneIRProcessingResult containing typed models and detailed diagnostic logs.
    """
    raw_dict: Optional[Dict[str, Any]] = None

    # Step 1: JSON Parsing
    if isinstance(response, str):
        json_str = _extract_json_block(response)
        try:
            raw_dict = json.loads(json_str)
        except Exception as e:
            return SceneIRProcessingResult(
                is_valid=False,
                structural_errors=[{
                    "error_code": "JSON_PARSE_ERROR",
                    "message": f"Failed to parse JSON response: {e}",
                    "field_path": "root",
                }],
                summary=f"JSON parsing failed: {e}",
            )
    elif isinstance(response, dict):
        raw_dict = response
    else:
        return SceneIRProcessingResult(
            is_valid=False,
            structural_errors=[{
                "error_code": "INVALID_INPUT_TYPE",
                "message": f"Expected str or dict, got {type(response).__name__}",
                "field_path": "root",
            }],
            summary=f"Invalid response type: {type(response).__name__}",
        )

    # Step 2: Determine document type (Sequence vs Single Scene)
    is_sequence = "sections" in raw_dict or "sequence_id" in raw_dict

    scene_obj: Optional[SceneIR] = None
    sequence_obj: Optional[SceneSequenceIR] = None
    structural_errors: List[Dict[str, Any]] = []

    # Step 3: Deserialization and Structural Schema Validation
    if is_sequence:
        try:
            sequence_obj = deserialize_sequence_ir(raw_dict)
        except Exception as e:
            structural_errors.append({
                "error_code": "STRUCTURAL_SCHEMA_ERROR",
                "message": f"SceneSequenceIR deserialization error: {e}",
                "field_path": "root",
            })
    else:
        try:
            scene_obj = deserialize_scene_ir(raw_dict)
        except Exception as e:
            structural_errors.append({
                "error_code": "STRUCTURAL_SCHEMA_ERROR",
                "message": f"SceneIR deserialization error: {e}",
                "field_path": "root",
            })

    # If structural deserialization failed, also run SceneIRValidator on raw dict to extract specific codes
    semantic_errors: List[Dict[str, Any]] = []
    if is_sequence:
        val_res = SceneIRValidator.validate_sequence(sequence_obj or raw_dict)
    else:
        val_res = SceneIRValidator.validate(scene_obj or raw_dict)

    if not val_res.is_valid:
        semantic_errors = [err.to_dict() for err in val_res.errors]

    # Step 4: Visual Capability and Payload Validation
    payload_validator = SceneIRPayloadValidator(manifest=component_manifest)
    payload_errors: List[Dict[str, Any]] = []

    if is_sequence and sequence_obj:
        p_res = payload_validator.validate_sequence(sequence_obj)
        if not p_res.is_valid:
            payload_errors = [err.to_dict() for err in p_res.errors]
    elif not is_sequence and scene_obj:
        p_res = payload_validator.validate(scene_obj)
        if not p_res.is_valid:
            payload_errors = [err.to_dict() for err in p_res.errors]

    total_err_count = len(structural_errors) + len(semantic_errors) + len(payload_errors)
    is_valid = (total_err_count == 0) and (scene_obj is not None or sequence_obj is not None)

    summary_parts = []
    if structural_errors:
        summary_parts.append(f"{len(structural_errors)} structural error(s)")
    if semantic_errors:
        summary_parts.append(f"{len(semantic_errors)} semantic error(s)")
    if payload_errors:
        summary_parts.append(f"{len(payload_errors)} payload error(s)")

    summary = (
        "Scene IR verification PASSED."
        if is_valid
        else f"Scene IR verification FAILED: {', '.join(summary_parts)}."
    )

    return SceneIRProcessingResult(
        is_valid=is_valid,
        scene_ir=scene_obj,
        sequence_ir=sequence_obj,
        structural_errors=structural_errors,
        semantic_errors=semantic_errors,
        payload_errors=payload_errors,
        raw_json=raw_dict,
        summary=summary,
    )
