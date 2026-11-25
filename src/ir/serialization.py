"""
CineLearn — Scene IR Serialization, Deserialization & JSON Schema Export.
========================================================================

Provides deterministic JSON round-trip utilities and official JSON Schema export
for Scene IR conforming to docs/architecture/scene_ir_and_spatial_composition_specification.md.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple, Union

from .models import SceneIR, SceneSequenceIR


def serialize_scene_ir(scene: SceneIR, indent: int = 2) -> str:
    """
    Serializes a SceneIR object to a deterministic, formatted JSON string.
    Ensures stable key ordering and proper alias preservation ($schema).
    """
    if not isinstance(scene, SceneIR):
        raise TypeError(f"Expected SceneIR instance, got {type(scene).__name__}")
    return scene.to_json(indent=indent)


def deserialize_scene_ir(data: Union[str, Dict[str, Any]]) -> SceneIR:
    """
    Deserializes raw JSON string or dictionary into a validated SceneIR model.
    """
    if isinstance(data, str):
        return SceneIR.from_json(data)
    elif isinstance(data, dict):
        return SceneIR.from_dict(data)
    raise TypeError(f"Expected str or dict, got {type(data).__name__}")


def serialize_sequence_ir(sequence: SceneSequenceIR, indent: int = 2) -> str:
    """
    Serializes a multi-section SceneSequenceIR to a deterministic JSON string.
    """
    if not isinstance(sequence, SceneSequenceIR):
        raise TypeError(f"Expected SceneSequenceIR instance, got {type(sequence).__name__}")
    return sequence.to_json(indent=indent)


def deserialize_sequence_ir(data: Union[str, Dict[str, Any]]) -> SceneSequenceIR:
    """
    Deserializes raw JSON string or dictionary into a validated SceneSequenceIR model.
    """
    if isinstance(data, str):
        return SceneSequenceIR.from_json(data)
    elif isinstance(data, dict):
        return SceneSequenceIR.from_dict(data)
    raise TypeError(f"Expected str or dict, got {type(data).__name__}")


def export_scene_ir_json_schema() -> Dict[str, Any]:
    """
    Exports the official JSON Schema specification for SceneIR.
    Useful for external tooling, IDE validation, and Stage 2 prompt integration.
    """
    return SceneIR.model_json_schema()


def verify_round_trip(scene: SceneIR) -> Tuple[bool, Optional[str]]:
    """
    Verifies that a SceneIR instance survives deterministic round-trip serialization:
    SceneIR -> JSON -> Dict -> SceneIR.
    Returns (success, discrepancy_reason).
    """
    try:
        json_str_1 = serialize_scene_ir(scene)
        reconstructed = deserialize_scene_ir(json_str_1)
        json_str_2 = serialize_scene_ir(reconstructed)

        if json_str_1 != json_str_2:
            return False, "Serialized JSON strings differ after round-trip deserialization."

        dict_1 = json.loads(json_str_1)
        dict_2 = json.loads(json_str_2)
        if dict_1 != dict_2:
            return False, "Parsed JSON dictionaries differ after round trip."

        return True, None
    except Exception as e:
        return False, f"Round-trip failed with exception: {e}"
