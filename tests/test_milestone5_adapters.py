"""
Unit and Integration Tests for Milestone 5.1: Pipeline Switch & Architecture Bridge.
=====================================================================================

Validates:
1. pipeline_mode configuration in RunConfig and TeachingVideoAgent (legacy default).
2. Explicit mode selection (legacy and modern).
3. CLI argument parsing with --pipeline-mode / --pipeline_mode.
4. Checkpoint persistence of pipeline_mode.
5. Checkpoint compatibility validation (rejection of cross-pipeline resumes).
6. Backward compatibility for historical legacy checkpoints lacking pipeline_mode.
7. Fail-closed dispatch behavior for modern pipeline mode (raises NotImplementedError).
8. SectionProxy dual access (attribute and dict), field mutation, deletion, defaults, and error semantics.
9. Bidirectional Section <-> SectionProxy conversion.
10. scene_ir_to_storyboard_section_obj integration with typed SceneIR models.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agent import RunConfig, Section, TeachingVideoAgent, build_and_parse_args
from src.ir.adapter import (
    SceneIRAdapterError,
    SectionProxy,
    scene_ir_to_storyboard_section,
    scene_ir_to_storyboard_section_obj,
)
from src.ir.fixtures import (
    get_binary_search_fixture,
    get_chemical_kinetics_fixture,
    get_pythagorean_theorem_fixture,
)
from src.ir.models import (
    ActionDeclaration,
    ActionType,
    EntityDeclaration,
    SceneIR,
    TimelineEvent,
    VisualTier,
)
from src.ir.payload_validator import PayloadErrorCode, SceneIRPayloadValidator


# =============================================================================
# 1. Configuration & Mode Selection Tests
# =============================================================================

def test_run_config_default_pipeline_mode():
    """RunConfig should default pipeline_mode to 'legacy'."""
    cfg = RunConfig()
    assert cfg.pipeline_mode == "legacy"


def test_agent_default_pipeline_mode():
    """TeachingVideoAgent should default to 'legacy' pipeline mode when unspecified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = TeachingVideoAgent(
            idx=0,
            topic="Test Topic",
            folder=tmpdir,
        )
        assert agent.pipeline_mode == "legacy"


def test_agent_explicit_legacy_mode():
    """TeachingVideoAgent should respect explicit 'legacy' pipeline mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = RunConfig(pipeline_mode="legacy")
        agent = TeachingVideoAgent(
            idx=0,
            topic="Test Topic",
            folder=tmpdir,
            cfg=cfg,
        )
        assert agent.pipeline_mode == "legacy"

        # Also via direct constructor argument
        agent2 = TeachingVideoAgent(
            idx=0,
            topic="Test Topic",
            folder=tmpdir,
            pipeline_mode="legacy",
        )
        assert agent2.pipeline_mode == "legacy"


def test_agent_explicit_modern_mode():
    """TeachingVideoAgent should accept 'modern' pipeline mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = RunConfig(pipeline_mode="modern")
        agent = TeachingVideoAgent(
            idx=0,
            topic="Test Topic",
            folder=tmpdir,
            cfg=cfg,
        )
        assert agent.pipeline_mode == "modern"

        # Direct constructor argument takes precedence over cfg
        agent2 = TeachingVideoAgent(
            idx=0,
            topic="Test Topic",
            folder=tmpdir,
            cfg=RunConfig(pipeline_mode="legacy"),
            pipeline_mode="modern",
        )
        assert agent2.pipeline_mode == "modern"


def test_agent_pipeline_mode_normalization():
    """Agent should normalize case and whitespace for pipeline_mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = TeachingVideoAgent(
            idx=0,
            topic="Test Topic",
            folder=tmpdir,
            pipeline_mode="  MODERN  ",
        )
        assert agent.pipeline_mode == "modern"

        agent_legacy = TeachingVideoAgent(
            idx=0,
            topic="Test Topic",
            folder=tmpdir,
            pipeline_mode="  Legacy  ",
        )
        assert agent_legacy.pipeline_mode == "legacy"


def test_agent_invalid_pipeline_mode_rejection():
    """Agent must reject unknown pipeline modes with ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Invalid pipeline_mode"):
            TeachingVideoAgent(
                idx=0,
                topic="Test Topic",
                folder=tmpdir,
                pipeline_mode="quantum",
            )

        with pytest.raises(ValueError, match="Invalid pipeline_mode"):
            TeachingVideoAgent(
                idx=0,
                topic="Test Topic",
                folder=tmpdir,
                cfg=RunConfig(pipeline_mode="unsupported"),
            )


# =============================================================================
# 2. CLI Argument Handling Tests
# =============================================================================

def test_cli_default_pipeline_mode():
    """CLI parser defaults --pipeline-mode to 'legacy'."""
    test_args = ["agent.py", "--topic", "CLI Test Topic"]
    with patch("sys.argv", test_args):
        parsed = build_and_parse_args()
        assert parsed.pipeline_mode == "legacy"


def test_cli_explicit_modern_mode():
    """CLI parser accepts --pipeline-mode modern."""
    test_args = ["agent.py", "--topic", "CLI Test Topic", "--pipeline-mode", "modern"]
    with patch("sys.argv", test_args):
        parsed = build_and_parse_args()
        assert parsed.pipeline_mode == "modern"


def test_cli_underscore_alias():
    """CLI parser accepts --pipeline_mode alias."""
    test_args = ["agent.py", "--topic", "CLI Test Topic", "--pipeline_mode", "modern"]
    with patch("sys.argv", test_args):
        parsed = build_and_parse_args()
        assert parsed.pipeline_mode == "modern"


def test_cli_invalid_mode_rejected():
    """CLI parser rejects invalid choices for --pipeline-mode."""
    test_args = ["agent.py", "--topic", "CLI Test Topic", "--pipeline-mode", "experimental"]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit):
            build_and_parse_args()


# =============================================================================
# 3. Pipeline Dispatch & Fail-Closed Modern Mode Tests
# =============================================================================

def test_modern_mode_fails_closed():
    """GENERATE_VIDEO in modern mode must fail closed with NotImplementedError in Phase 5.1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = TeachingVideoAgent(
            idx=0,
            topic="Modern Pipeline Test",
            folder=tmpdir,
            pipeline_mode="modern",
        )
        with pytest.raises(NotImplementedError) as exc_info:
            agent.GENERATE_VIDEO()
        assert "Modern Scene IR pipeline execution is not yet implemented" in str(exc_info.value)


def test_legacy_mode_dispatch_structure():
    """GENERATE_VIDEO in legacy mode routes to _execute_legacy_pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = TeachingVideoAgent(
            idx=0,
            topic="Legacy Dispatch Test",
            folder=tmpdir,
            pipeline_mode="legacy",
        )
        # Mock _execute_legacy_pipeline to verify exact dispatch routing
        with patch.object(agent, "_execute_legacy_pipeline", return_value="dummy_legacy.mp4") as mock_legacy:
            result = agent.GENERATE_VIDEO()
            mock_legacy.assert_called_once()
            assert result == "dummy_legacy.mp4"


# =============================================================================
# 4. Checkpoint Persistence & Compatibility Tests
# =============================================================================

def test_checkpoint_persists_pipeline_mode():
    """save_checkpoint must write pipeline_mode into checkpoint.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_legacy = TeachingVideoAgent(
            idx=0,
            topic="Checkpoint Topic",
            folder=tmpdir,
            pipeline_mode="legacy",
        )
        agent_legacy.save_checkpoint("test_stage")
        cp_file = agent_legacy.output_dir / "checkpoint.json"
        assert cp_file.exists()
        with open(cp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("pipeline_mode") == "legacy"

    with tempfile.TemporaryDirectory() as tmpdir:
        agent_modern = TeachingVideoAgent(
            idx=0,
            topic="Checkpoint Topic",
            folder=tmpdir,
            pipeline_mode="modern",
        )
        agent_modern.save_checkpoint("test_stage")
        cp_file = agent_modern.output_dir / "checkpoint.json"
        assert cp_file.exists()
        with open(cp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("pipeline_mode") == "modern"


def test_checkpoint_resumes_matching_mode():
    """Resuming an existing checkpoint with matching pipeline_mode must succeed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent1 = TeachingVideoAgent(
            idx=0,
            topic="Resume Topic",
            folder=tmpdir,
            pipeline_mode="legacy",
        )
        agent1.save_checkpoint("test_stage")

        # Create new agent pointing to the same folder with matching mode
        agent2 = TeachingVideoAgent(
            idx=0,
            topic="Resume Topic",
            folder=tmpdir,
            pipeline_mode="legacy",
        )
        assert agent2.pipeline_mode == "legacy"


def test_checkpoint_rejects_mismatched_mode_modern_to_legacy():
    """Resuming a modern checkpoint with legacy agent mode must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_mod = TeachingVideoAgent(
            idx=0,
            topic="Mismatch Topic",
            folder=tmpdir,
            pipeline_mode="modern",
        )
        agent_mod.save_checkpoint("test_stage")

        # Now attempt to load this folder with legacy agent
        with pytest.raises(ValueError, match="Checkpoint pipeline mode mismatch"):
            TeachingVideoAgent(
                idx=0,
                topic="Mismatch Topic",
                folder=tmpdir,
                pipeline_mode="legacy",
            )


def test_checkpoint_rejects_mismatched_mode_legacy_to_modern():
    """Resuming a legacy checkpoint with modern agent mode must raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_leg = TeachingVideoAgent(
            idx=0,
            topic="Mismatch Topic",
            folder=tmpdir,
            pipeline_mode="legacy",
        )
        agent_leg.save_checkpoint("test_stage")

        # Now attempt to load this folder with modern agent
        with pytest.raises(ValueError, match="Checkpoint pipeline mode mismatch"):
            TeachingVideoAgent(
                idx=0,
                topic="Mismatch Topic",
                folder=tmpdir,
                pipeline_mode="modern",
            )


def test_checkpoint_historical_legacy_compatibility():
    """Historical checkpoints lacking 'pipeline_mode' default to 'legacy' without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-seed a historical checkpoint without pipeline_mode field
        agent_tmp = TeachingVideoAgent(
            idx=0,
            topic="Historical Topic",
            folder=tmpdir,
            pipeline_mode="legacy",
        )
        cp_file = agent_tmp.output_dir / "checkpoint.json"
        historical_data = {
            "topic": "Historical Topic",
            "level": "intermediate",
            "analogy": "",
            "stage": "outline_completed",
            "sections": [],
        }
        with open(cp_file, "w", encoding="utf-8") as f:
            json.dump(historical_data, f)

        # Loading with legacy agent must succeed (safe backward compatibility)
        agent_resumed = TeachingVideoAgent(
            idx=0,
            topic="Historical Topic",
            folder=tmpdir,
            pipeline_mode="legacy",
        )
        assert agent_resumed.pipeline_mode == "legacy"

        # Loading with modern agent must reject it because historical checkpoint is legacy
        with pytest.raises(ValueError, match="Checkpoint pipeline mode mismatch"):
            TeachingVideoAgent(
                idx=0,
                topic="Historical Topic",
                folder=tmpdir,
                pipeline_mode="modern",
            )


def test_checkpoint_rejects_present_null_value():
    """Checkpoint with present null pipeline_mode must be rejected with ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_tmp = TeachingVideoAgent(idx=0, topic="Null Mode Topic", folder=tmpdir, pipeline_mode="legacy")
        cp_file = agent_tmp.output_dir / "checkpoint.json"
        with open(cp_file, "w", encoding="utf-8") as f:
            json.dump({"topic": "Null Mode Topic", "pipeline_mode": None, "sections": []}, f)

        with pytest.raises(ValueError, match="invalid or corrupted pipeline_mode"):
            TeachingVideoAgent(idx=0, topic="Null Mode Topic", folder=tmpdir, pipeline_mode="legacy")


def test_checkpoint_rejects_present_empty_string():
    """Checkpoint with present empty or whitespace-only pipeline_mode must be rejected with ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_tmp = TeachingVideoAgent(idx=0, topic="Empty Mode Topic", folder=tmpdir, pipeline_mode="legacy")
        cp_file = agent_tmp.output_dir / "checkpoint.json"
        with open(cp_file, "w", encoding="utf-8") as f:
            json.dump({"topic": "Empty Mode Topic", "pipeline_mode": "   ", "sections": []}, f)

        with pytest.raises(ValueError, match="invalid or corrupted pipeline_mode"):
            TeachingVideoAgent(idx=0, topic="Empty Mode Topic", folder=tmpdir, pipeline_mode="legacy")


def test_checkpoint_rejects_present_unsupported_string():
    """Checkpoint with unsupported mode string (e.g. 'quantum') must be rejected with ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_tmp = TeachingVideoAgent(idx=0, topic="Corrupt Mode Topic", folder=tmpdir, pipeline_mode="legacy")
        cp_file = agent_tmp.output_dir / "checkpoint.json"
        with open(cp_file, "w", encoding="utf-8") as f:
            json.dump({"topic": "Corrupt Mode Topic", "pipeline_mode": "quantum", "sections": []}, f)

        with pytest.raises(ValueError, match="invalid or corrupted pipeline_mode"):
            TeachingVideoAgent(idx=0, topic="Corrupt Mode Topic", folder=tmpdir, pipeline_mode="legacy")


def test_checkpoint_rejects_present_non_string_value():
    """Checkpoint with present non-string mode (int, bool, list) must be rejected with ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent_tmp = TeachingVideoAgent(idx=0, topic="Numeric Mode Topic", folder=tmpdir, pipeline_mode="legacy")
        cp_file = agent_tmp.output_dir / "checkpoint.json"
        with open(cp_file, "w", encoding="utf-8") as f:
            json.dump({"topic": "Numeric Mode Topic", "pipeline_mode": 12345, "sections": []}, f)

        with pytest.raises(ValueError, match="invalid or corrupted pipeline_mode"):
            TeachingVideoAgent(idx=0, topic="Numeric Mode Topic", folder=tmpdir, pipeline_mode="legacy")


# =============================================================================
# 5. SectionProxy Unit & Integration Tests
# =============================================================================

def test_section_proxy_dual_access():
    """SectionProxy must support attribute and dictionary access identically."""
    raw = {
        "id": "section_1",
        "title": "Introduction to Algorithms",
        "lecture_lines": ["Line 1", "Line 2"],
        "animations": ["Step 1", "Step 2"],
        "story_beat": "Beat 1",
        "captions": ["Cap 1", "Cap 2"],
        "layout_plan": "Center container",
        "verified_values": ["val = 42"],
    }
    proxy = SectionProxy(raw)

    # Attribute access
    assert proxy.id == "section_1"
    assert proxy.title == "Introduction to Algorithms"
    assert proxy.lecture_lines == ["Line 1", "Line 2"]
    assert proxy.animations == ["Step 1", "Step 2"]
    assert proxy.story_beat == "Beat 1"
    assert proxy.captions == ["Cap 1", "Cap 2"]
    assert proxy.layout_plan == "Center container"
    assert proxy.verified_values == ["val = 42"]

    # Dictionary access
    assert proxy["id"] == "section_1"
    assert proxy["title"] == "Introduction to Algorithms"
    assert proxy["lecture_lines"] == ["Line 1", "Line 2"]
    assert proxy["animations"] == ["Step 1", "Step 2"]
    assert proxy["story_beat"] == "Beat 1"
    assert proxy["captions"] == ["Cap 1", "Cap 2"]
    assert proxy["layout_plan"] == "Center container"
    assert proxy["verified_values"] == ["val = 42"]


def test_section_proxy_canonical_defaults():
    """Unspecified canonical fields should return default values without raising."""
    proxy = SectionProxy({"id": "sec_min", "title": "Minimal Section"})
    assert proxy.beats is None
    assert proxy.algorithmic_operations is None
    assert proxy.resolved_actions is None
    assert proxy.story_beat == ""
    assert proxy.lecture_lines == []
    assert proxy.animations == []
    assert proxy.captions is None
    assert proxy.verified_values is None
    assert proxy.get("beats") is None
    assert proxy.get("nonexistent", "fallback") == "fallback"


def test_section_proxy_nonexistent_attributes_and_keys():
    """Non-canonical nonexistent fields must raise AttributeError / KeyError."""
    proxy = SectionProxy({"id": "sec_1", "title": "Test"})

    with pytest.raises(AttributeError):
        _ = proxy.nonexistent_field

    with pytest.raises(KeyError):
        _ = proxy["nonexistent_field"]


def test_section_proxy_mutations():
    """Mutations via attribute or dict interface must remain strictly synchronized."""
    proxy = SectionProxy({"id": "s1", "title": "Original Title"})

    # Mutate via attribute
    proxy.title = "New Title"
    assert proxy.title == "New Title"
    assert proxy["title"] == "New Title"

    # Mutate via dict
    proxy["title"] = "Updated Title"
    assert proxy.title == "Updated Title"
    assert proxy["title"] == "Updated Title"

    # Add custom field via dict
    proxy["custom_field"] = [1, 2, 3]
    assert proxy.custom_field == [1, 2, 3]
    assert proxy["custom_field"] == [1, 2, 3]

    # Mutate custom field via attr
    proxy.custom_field = [4, 5, 6]
    assert proxy["custom_field"] == [4, 5, 6]


def test_section_proxy_deletion():
    """Field deletion via del obj.attr or del obj['key']."""
    proxy = SectionProxy({"id": "s1", "extra1": "foo", "extra2": "bar"})

    # Delete via attr
    del proxy.extra1
    assert "extra1" not in proxy
    with pytest.raises(AttributeError):
        del proxy.extra1

    # Delete via dict
    del proxy["extra2"]
    assert "extra2" not in proxy
    with pytest.raises(KeyError):
        del proxy["extra2"]


def test_section_proxy_dict_methods():
    """Verify keys, values, items, len, iter, and __dict__ views."""
    data = {"id": "sec_3", "title": "Sorting", "lecture_lines": ["L1"]}
    proxy = SectionProxy(data)

    assert len(proxy) == 3
    assert set(proxy.keys()) == {"id", "title", "lecture_lines"}
    assert "id" in proxy
    assert "nonexistent" not in proxy
    assert list(proxy) == ["id", "title", "lecture_lines"]
    assert proxy.to_dict() == data

    # Checkpoint serialization compatibility: vars(proxy) and proxy.__dict__
    assert vars(proxy) == data
    assert proxy.__dict__ == data


def test_section_proxy_dataclass_interop():
    """Bidirectional conversion between Section dataclass and SectionProxy."""
    sec = Section(
        id="sec_dc",
        title="Dataclass Section",
        lecture_lines=["Line A", "Line B"],
        animations=["Anim A"],
        story_beat="Beat A",
        captions=["Cap A"],
        layout_plan="Split layout",
        verified_values=["val = 10"],
        algorithmic_operations=[{"op": "insert"}],
        beats=[{"beat_id": "b1"}],
        resolved_actions=[{"action": "create"}],
    )

    # Convert Section -> SectionProxy
    proxy = SectionProxy.from_section(sec)
    assert proxy.id == "sec_dc"
    assert proxy["title"] == "Dataclass Section"
    assert proxy.lecture_lines == ["Line A", "Line B"]
    assert proxy["beats"] == [{"beat_id": "b1"}]

    # Convert SectionProxy -> Section
    sec_roundtrip = proxy.to_section()
    assert isinstance(sec_roundtrip, Section)
    assert sec_roundtrip.id == sec.id
    assert sec_roundtrip.title == sec.title
    assert sec_roundtrip.lecture_lines == sec.lecture_lines
    assert sec_roundtrip.animations == sec.animations
    assert sec_roundtrip.beats == sec.beats
    assert sec_roundtrip.resolved_actions == sec.resolved_actions


# =============================================================================
# 6. scene_ir_to_storyboard_section_obj Tests
# =============================================================================

def test_scene_ir_to_storyboard_section_obj_binary_search():
    """scene_ir_to_storyboard_section_obj transforms SceneIR into a fully-functional SectionProxy."""
    scene = get_binary_search_fixture()
    assert isinstance(scene, SceneIR)

    proxy = scene_ir_to_storyboard_section_obj(scene)
    assert isinstance(proxy, SectionProxy)

    # Section ID & title
    assert proxy.id == "Section3Scene"
    assert proxy["id"] == "Section3Scene"
    assert proxy.title == "Narrowing the Search Interval"
    assert proxy["title"] == "Narrowing the Search Interval"

    # Lecture lines and captions
    assert len(proxy.lecture_lines) == 2
    assert "midpoint index" in proxy.lecture_lines[0]
    assert len(proxy.captions) == 2

    # Synthesized animation strings (valid 6-field grammar)
    assert len(proxy.animations) > 0
    for anim in proxy.animations:
        fields = anim.split("|")
        assert len(fields) == 6, f"Malformed animation line: {anim}"

    # Structured resolved actions
    assert len(proxy.resolved_actions) > 0
    assert any(a.get("canonical_action") == "enter" for a in proxy.resolved_actions)
    assert any(a.get("target_instance_id") == "pointer_high" for a in proxy.resolved_actions)

    # Conversion to dataclass Section
    sec = proxy.to_section()
    assert isinstance(sec, Section)
    assert sec.id == proxy.id
    assert sec.title == proxy.title
    assert sec.lecture_lines == proxy.lecture_lines


def test_scene_ir_to_storyboard_section_obj_chemical_kinetics():
    """Verify SectionProxy generation for multi-entity chemical kinetics SceneIR."""
    scene = get_chemical_kinetics_fixture()
    proxy = scene_ir_to_storyboard_section_obj(scene)
    assert isinstance(proxy, SectionProxy)
    assert proxy.id == "Section2Scene"
    assert proxy["title"] == "Energy Barrier and Activation Energy"
    assert len(proxy.lecture_lines) >= 1
    assert len(proxy.animations) >= 1


def test_scene_ir_to_storyboard_section_obj_pythagorean():
    """Verify SectionProxy generation for geometric pythagorean theorem SceneIR."""
    scene = get_pythagorean_theorem_fixture()
    proxy = scene_ir_to_storyboard_section_obj(scene)
    assert isinstance(proxy, SectionProxy)
    assert proxy.id == "Section4Scene"
    assert "Dissected Squares" in proxy.title
    assert len(proxy.lecture_lines) >= 1
    assert len(proxy.animations) >= 1


# =============================================================================
# 7. Payload Type Validation Boundary Tests (Task B)
# =============================================================================

def _make_minimal_scene(entities=None, timeline=None):
    from src.ir.models import ViewportDeclaration, ViewportType
    return SceneIR(
        scene_id="SceneTypeTest",
        section_index=1,
        title="Type Test Scene",
        duration_budget=10.0,
        viewports={
            "dock_main": ViewportDeclaration(type=ViewportType.MAIN)
        },
        entities=entities or [],
        timeline=timeline or [],
        metadata={"lecture_lines": ["Testing types."], "story_beat": "Beat"},
    )


def test_payload_validation_rejects_invalid_initial_state_type():
    """SceneIRPayloadValidator must flag INVALID_PAYLOAD_PARAM_TYPE when parameter type does not match manifest."""
    validator = SceneIRPayloadValidator()
    # ProcessCardVisual expects arrival_time: int, burst_time: int
    scene = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="pcb_bad",
                component_type="ProcessCardVisual",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="dock_main",
                initial_state={
                    "pid": "P1",
                    "arrival_time": "zero",  # Should be int, not str
                    "burst_time": 4,
                },
            )
        ]
    )
    res = validator.validate(scene)
    assert not res.is_valid
    type_errors = [e for e in res.errors if e.error_code == PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE]
    assert len(type_errors) == 1
    assert type_errors[0].parameter == "arrival_time"
    assert "Expected type 'int'" in type_errors[0].message


def test_payload_validation_accepts_valid_initial_state_types():
    """SceneIRPayloadValidator must pass when all parameter types match manifest."""
    validator = SceneIRPayloadValidator()
    scene = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="pcb_good",
                component_type="ProcessCardVisual",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="dock_main",
                initial_state={
                    "pid": "P1",
                    "arrival_time": 0,
                    "burst_time": 4,
                },
            )
        ]
    )
    res = validator.validate(scene)
    assert res.is_valid, f"Unexpected validation errors: {[e.message for e in res.errors]}"


def test_payload_validation_rejects_invalid_action_param_type():
    """Standard action parameter with invalid type (e.g. string duration) must be rejected."""
    validator = SceneIRPayloadValidator()
    scene = _make_minimal_scene(
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
                step_id="step_bad_duration",
                actions=[
                    ActionDeclaration(
                        type=ActionType.ENTER,
                        target_entity="pcb_1",
                        parameters={"duration": "two_seconds"},  # Should be numeric, not str
                    )
                ],
            )
        ],
    )
    res = validator.validate(scene)
    assert not res.is_valid
    type_errors = [e for e in res.errors if e.error_code == PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE]
    assert len(type_errors) == 1
    assert type_errors[0].parameter == "duration"


def test_payload_validation_rejects_invalid_operation_param_type():
    """Operation parameter with invalid type against manifest must be rejected."""
    validator = SceneIRPayloadValidator()
    # NumberLineVisual.animate_add_pointer expects val: float
    scene = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="num_line_1",
                component_type="NumberLineVisual",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="dock_main",
                initial_state={"length": 6.0},
            )
        ],
        timeline=[
            TimelineEvent(
                step_id="step_bad_op_param",
                actions=[
                    ActionDeclaration(
                        type=ActionType.COMPONENT_OP,
                        target_entity="num_line_1",
                        operation="animate_add_pointer",
                        parameters={"name": "pt1", "val": "three"},  # Should be float, not str
                    )
                ],
            )
        ],
    )
    res = validator.validate(scene)
    assert not res.is_valid
    type_errors = [e for e in res.errors if e.error_code == PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE]
    assert len(type_errors) == 1
    assert type_errors[0].parameter == "val"
    assert "Expected type 'float'" in type_errors[0].message


def test_payload_validation_rejects_invalid_conceptual_card_type():
    """Tier 4 ConceptualCard with non-string content must be rejected."""
    validator = SceneIRPayloadValidator()
    scene = _make_minimal_scene(
        entities=[
            EntityDeclaration(
                id="card_1",
                component_type="ConceptualCard",
                visual_tier=VisualTier.TIER4_CONCEPTUAL_CARD,
                viewport="dock_main",
                initial_state={"title": "Valid Title", "formula": 12345},  # formula must be str
            )
        ]
    )
    res = validator.validate(scene)
    assert not res.is_valid
    type_errors = [e for e in res.errors if e.error_code == PayloadErrorCode.INVALID_PAYLOAD_PARAM_TYPE]
    assert len(type_errors) == 1
    assert type_errors[0].parameter == "formula"

