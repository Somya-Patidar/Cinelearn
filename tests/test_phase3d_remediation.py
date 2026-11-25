"""
Unit tests for Phase 3D — Minimal Verified Remediation.
Verifies the three approved corrections:
  P1: Gate 3.5 repair classification for UNSUPPORTED_COMPONENT_OPERATION.
  P2: Bounded 1-turn Stage 1 JSON repair with line/column diagnostics.
  P3: Strengthened domain-aware input value preservation contract in Stage 2.
"""

import json
from pathlib import Path
import pytest

from src.ir.processor import process_stage2_scene_ir_output, SceneIRProcessingResult
from src.ir.models import SceneIR, SceneSequenceIR, ActionType, EntityDeclaration, VisualTier
from prompts.stage1 import get_prompt1_outline_repair
from prompts.stage2 import get_prompt2_scene_ir, get_prompt2_scene_ir_repair
from tests.test_milestone5_stage2_native import _make_minimal_scene


# =============================================================================
# P1: Gate 3.5 Repair Classification Tests
# =============================================================================

def test_unsupported_operation_is_recoverable():
    """
    P1: An unsupported operation on a registered component (e.g. animate_process_execution on GanttChartVisual)
    must be treated as a recoverable diagnostic, NOT as an unrecoverable system capability failure.
    """
    scene_bad_op = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="gantt_chart",
                component_type="GanttChartVisual",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="dock_main",
                initial_state={"total_time": 16, "width": 6.2, "height": 1.0},
            )
        ],
        timeline=[
            {
                "step_id": "s1",
                "actions": [
                    {
                        "type": ActionType.COMPONENT_OP,
                        "target_entity": "gantt_chart",
                        "operation": "animate_process_execution",  # Hallucinated op!
                        "parameters": {"pid": "P1", "start": 0, "end": 7},
                    }
                ],
            }
        ],
    )

    res = process_stage2_scene_ir_output(scene_bad_op.to_dict())
    assert not res.is_valid, "Payload with unsupported operation must fail validation"
    assert any(e["error_code"] == "UNSUPPORTED_COMPONENT_OPERATION" for e in res.payload_errors)

    # Legacy contract preserved
    assert res.has_unsupported_capabilities, "Legacy has_unsupported_capabilities must remain True"

    # New P1 behavior: genuinely unsupported component types is False
    assert not res.has_unsupported_component_types, "GanttChartVisual is a valid registered component"
    assert res.has_recoverable_diagnostics, "Unsupported component operation must be recoverable via bounded repair"

    # Repair prompt must be generated and list the valid supported operations
    repair_prompt = get_prompt2_scene_ir_repair(
        outline={"topic": "SJF"},
        failed_output=scene_bad_op.to_dict(),
        processing_result=res,
        level="intermediate",
    )
    assert repair_prompt is not None
    assert "UNSUPPORTED_COMPONENT_OPERATION" in repair_prompt
    assert "animate_add_block" in repair_prompt, "Repair prompt must list valid allowed operations"


def test_unsupported_component_type_remains_fail_closed():
    """
    P1: An unregistered, non-existent component class must remain fatal (has_recoverable_diagnostics == False).
    """
    scene_unreg = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="alien_comp",
                component_type="AlienQuantumVisual3D",  # Unregistered class!
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
                        "type": ActionType.ENTER,
                        "target_entity": "alien_comp",
                    }
                ],
            }
        ],
    )

    res = process_stage2_scene_ir_output(scene_unreg.to_dict())
    assert not res.is_valid
    assert any(e["error_code"] == "UNSUPPORTED_COMPONENT_TYPE" for e in res.payload_errors)
    assert res.has_unsupported_capabilities
    assert res.has_unsupported_component_types, "Unregistered component class must be flagged"
    assert not res.has_recoverable_diagnostics, "Unregistered component class must NOT allow repair (fail closed)"


# =============================================================================
# P2: Stage 1 JSON Repair Tests
# =============================================================================

def test_stage1_json_repair_reproduces_and_formats_diagnostic():
    """
    P2: Reproduces the exact SMOKE-MOD-FCFS-01 Stage 1 failure and verifies that
    get_prompt1_outline_repair formats the exact error message, line, and column.
    """
    corrupted_json_sample = '''{
  "title": "First Come First Served CPU Scheduling",
  "sections": [
    {
      "id": "section_1",
      "title": "Overview",
      "lecture_script": ["FCFS scheduling"],
      "state_transitions": [
        {
          "step": "P2 completes execution at t=7",
          "result": "{\\"ready_queue\\": [], \\"cpu\\": null, \\"completed_processes\\": [{\\"pid\\": \\"P1\\"}]}",
          "state_after": {"ready_queue": [], "cpu": null, "completed_processes": [{"pid": "P1"}, {\\"pid\\": "P2"}]}
        }
      ]
    }
  ]
}'''

    with pytest.raises(json.JSONDecodeError) as exc_info:
        json.loads(corrupted_json_sample)

    err = exc_info.value
    err_msg = str(err)
    lineno = err.lineno
    colno = err.colno

    repair_prompt = get_prompt1_outline_repair(
        knowledge_point="FCFS CPU Scheduling",
        failed_output=corrupted_json_sample,
        error_message=err_msg,
        line=lineno,
        column=colno,
    )

    assert "JSON Parsing Error" in repair_prompt
    assert f"line {lineno}" in repair_prompt
    assert f"column {colno}" in repair_prompt
    assert "FCFS CPU Scheduling" in repair_prompt
    assert "Bounded Repair Instructions" in repair_prompt
    assert "Do NOT mix escaped quotes" in repair_prompt


def test_stage1_json_repair_parses_cleanly_after_correction():
    """
    P2: Verifies that once the unescaped backslash is corrected, the JSON parses without error.
    """
    fixed_json_sample = '''{
  "title": "First Come First Served CPU Scheduling",
  "sections": [
    {
      "id": "section_1",
      "title": "Overview",
      "lecture_script": ["FCFS scheduling"],
      "state_transitions": [
        {
          "step": "P2 completes execution at t=7",
          "result": "{\\"ready_queue\\": [], \\"cpu\\": null, \\"completed_processes\\": [{\\"pid\\": \\"P1\\"}]}",
          "state_after": {"ready_queue": [], "cpu": null, "completed_processes": [{"pid": "P1"}, {"pid": "P2"}]}
        }
      ]
    }
  ]
}'''
    data = json.loads(fixed_json_sample)
    assert data["title"] == "First Come First Served CPU Scheduling"
    assert data["sections"][0]["state_transitions"][0]["state_after"]["completed_processes"][1]["pid"] == "P2"


# =============================================================================
# P3: Input Value Preservation Tests
# =============================================================================

def test_stage2_prompt_enforces_array_value_preservation():
    """
    P3: The Stage 2 prompt for sequence/array search must explicitly require all values
    from inputs.array (including boundary and non-target elements like 38) to be declared.
    """
    bs_outline = {
        "topic": "Binary Search",
        "title": "Binary Search",
        "sections": [
            {
                "id": "section_1",
                "title": "Worked Example",
                "worked_example_spec": {
                    "inputs": {
                        "array": [2, 5, 8, 12, 16, 23, 38],
                        "target": 23
                    }
                }
            }
        ]
    }

    prompt = get_prompt2_scene_ir(bs_outline, level="intermediate", format_as_sequence=True)

    assert "For sequence and array search: Every value in 'worked_example_spec.inputs.array' MUST be explicitly preserved" in prompt
    assert "38" in prompt
    assert "Inactive, boundary, and non-target elements" in prompt


def test_stage2_prompt_domain_isolation():
    """
    P3: General topics and CPU scheduling topics must NOT leak Binary Search array requirements.
    """
    general_outline = {
        "topic": "Photosynthesis in C3 Plants",
        "title": "Photosynthesis",
        "sections": [{"id": "section_1", "title": "Light Reactions"}]
    }

    gen_prompt = get_prompt2_scene_ir(general_outline, level="intermediate", format_as_sequence=True)
    assert "inputs.array" not in gen_prompt
    assert "38" not in gen_prompt
    assert "For sequence and array search" not in gen_prompt
    assert "fcfs" not in gen_prompt.lower()
    assert "sjf" not in gen_prompt.lower()
    assert "dijkstra" not in gen_prompt.lower()

    fcfs_outline = {
        "topic": "FCFS CPU Scheduling",
        "title": "FCFS CPU Scheduling",
        "sections": [{"id": "section_1", "title": "Worked Example"}]
    }

    fcfs_prompt = get_prompt2_scene_ir(fcfs_outline, level="intermediate", format_as_sequence=True)
    assert "For sequence and array search" not in fcfs_prompt
    assert "38" not in fcfs_prompt
    assert "For CPU scheduling: Every process specified in 'worked_example_spec'" in fcfs_prompt
