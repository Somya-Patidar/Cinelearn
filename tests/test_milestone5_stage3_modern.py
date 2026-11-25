"""
Milestone 5.2.2-B Integration Tests: Modern Scene IR Deterministic Compilation & Rendering.

Verifies:
1. End-to-end modern deterministic execution flow:
   Validated SceneSequenceIR -> TimelineScheduler -> LayoutSolver -> SceneCompiler -> Manim Render -> Verification.
2. Fail-closed error handling at every modern boundary:
   - Invalid Scene IR / Validation failure -> Halt, checkpoint 'scene_ir_validation_blocked', no legacy fallback.
   - Unsupported Tier 2 / components -> Fail-closed, no legacy fallback.
   - Timeline scheduling failure -> Halt, checkpoint 'timeline_failed', no render.
   - Spatial layout failure -> Halt, checkpoint 'layout_failed', no compile/render.
   - Compiler failure -> Halt, checkpoint 'compilation_failed', no render.
   - Render failure (non-zero exit) -> Halt, checkpoint 'render_failed', no legacy fallback.
3. Strict Pipeline Mode Isolation:
   - Modern mode never invokes legacy storyboard, legacy code generation, or legacy merge.
   - Modern failure never triggers legacy fallback.
   - Legacy mode never invokes modern pipeline.
   - Checkpoint pipeline_mode mismatch safeguards remain intact.
"""

import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agent import TeachingVideoAgent, RunConfig
from src.ir.models import (
    SceneIR,
    SceneSequenceIR,
    EntityDeclaration,
    VisualTier,
)
from src.ir.fixtures import get_binary_search_fixture
from src.compiler.layout_engine import SceneSpatialLayout, LayoutDiagnostic, LayoutErrorCode, BoundingBox2D
from src.compiler.timeline_scheduler import ScheduledTimeline
from src.compiler.scene_compiler import (
    CompiledSceneOutput,
    CompilationDiagnostic,
    CompilationErrorCode,
)


@pytest.fixture
def binary_search_sequence() -> SceneSequenceIR:
    """Canonical native SceneSequenceIR fixture wrapping Binary Search."""
    scene = get_binary_search_fixture()
    return SceneSequenceIR(
        sequence_id="seq_binary_search",
        title="Binary Search Algorithm",
        topic="Binary Search",
        sections=[scene],
    )


class TestModernPipelineExecution:
    """Phase 4.A: Successful end-to-end modern deterministic execution slice."""

    def test_modern_pipeline_end_to_end_success(self, tmp_path, binary_search_sequence):
        """Valid native SceneSequenceIR passes through layout, scheduling, compilation, and rendering."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
        )
        agent.scene_sequence_ir = binary_search_sequence

        # Spy on legacy methods to guarantee pipeline isolation
        agent.generate_storyboard = MagicMock(side_effect=AssertionError("Legacy generate_storyboard called in modern mode!"))
        agent.generate_codes = MagicMock(side_effect=AssertionError("Legacy generate_codes called in modern mode!"))
        agent._execute_legacy_pipeline = MagicMock(side_effect=AssertionError("Legacy pipeline invoked from modern mode!"))

        final_video = agent.GENERATE_VIDEO()

        # 1. Output video verification
        assert final_video is not None
        final_video_path = Path(final_video)
        assert final_video_path.exists()
        assert final_video_path.stat().st_size > 1000

        # 2. Checkpoint verification
        checkpoint_path = tmp_path / "0-Binary_Search" / "checkpoint.json"
        assert checkpoint_path.exists()
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["pipeline_mode"] == "modern"
        assert cp["stage"] == "completed"
        assert "Section3Scene" in cp["section_codes"]
        assert "Section3Scene" in cp["section_videos"]

        # 3. Artifact verification
        code_file = tmp_path / "0-Binary_Search" / "Section3Scene.py"
        assert code_file.exists()
        layout_file = tmp_path / "0-Binary_Search" / "layout_Section3Scene.json"
        assert layout_file.exists()
        timing_file = tmp_path / "0-Binary_Search" / "timing_Section3Scene.json"
        assert timing_file.exists()

        # 4. Pipeline isolation verification: zero legacy calls
        assert agent.generate_storyboard.call_count == 0
        assert agent.generate_codes.call_count == 0
        assert agent._execute_legacy_pipeline.call_count == 0


class TestModernPipelineFailures:
    """Phase 4.B, C, D: Fail-closed error handling across modern execution boundaries."""

    def test_modern_pipeline_empty_sections_fails_closed(self, tmp_path):
        """SceneSequenceIR with empty sections halts fail-closed."""
        empty_seq = SceneSequenceIR(
            sequence_id="seq_empty",
            title="Empty Sequence",
            topic="Binary Search",
            sections=[],
        )
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
        )
        agent.scene_sequence_ir = empty_seq
        agent._execute_legacy_pipeline = MagicMock()

        res = agent.GENERATE_VIDEO()

        assert res is None
        assert agent._execute_legacy_pipeline.call_count == 0
        cp_path = tmp_path / "0-Binary_Search" / "checkpoint.json"
        assert cp_path.exists()
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["pipeline_mode"] == "modern"
        assert cp["stage"] == "scene_ir_empty"

    def test_modern_pipeline_validation_gate_failure(self, tmp_path):
        """SceneSequenceIR failing semantic validation halts fail-closed."""
        invalid_scene = get_binary_search_fixture()
        # Create invalid entity referencing nonexistent viewport
        invalid_scene.entities.append(
            EntityDeclaration(
                id="rogue_entity",
                component_type="PointerIndicator",
                visual_tier=VisualTier.TIER3_PRIMITIVE_TREE,
                viewport="nonexistent_viewport",
                initial_state={"target_index": 0, "label": "rogue"},
            )
        )
        invalid_seq = SceneSequenceIR(
            sequence_id="seq_invalid",
            title="Invalid",
            topic="Binary Search",
            sections=[invalid_scene],
        )

        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
        )
        agent.scene_sequence_ir = invalid_seq
        agent._execute_legacy_pipeline = MagicMock()

        res = agent.GENERATE_VIDEO()

        assert res is None
        assert agent._execute_legacy_pipeline.call_count == 0
        cp_path = tmp_path / "0-Binary_Search" / "checkpoint.json"
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["pipeline_mode"] == "modern"
        assert cp["stage"] == "scene_ir_validation_blocked"

    def test_modern_pipeline_unsupported_tier2_composition_fails_closed(self, tmp_path):
        """Tier 2 composite entity fails closed and aborts execution without fallback."""
        tier2_scene = get_binary_search_fixture()
        tier2_scene.entities.append(
            EntityDeclaration(
                id="unsupported_tier2_container",
                component_type="ArbitraryContainer",
                visual_tier=VisualTier.TIER2_COMPOSITION,
                viewport="array_viewport",
                initial_state={"children": []},
            )
        )
        tier2_seq = SceneSequenceIR(
            sequence_id="seq_tier2",
            title="Tier 2 Composition",
            topic="Binary Search",
            sections=[tier2_scene],
        )

        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
        )
        agent.scene_sequence_ir = tier2_seq
        agent._execute_legacy_pipeline = MagicMock()

        res = agent.GENERATE_VIDEO()

        assert res is None
        assert agent._execute_legacy_pipeline.call_count == 0
        cp_path = tmp_path / "0-Binary_Search" / "checkpoint.json"
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["pipeline_mode"] == "modern"
        assert cp["stage"] in ("scene_ir_validation_blocked", "compilation_failed")

    def test_modern_pipeline_timeline_failure_fails_closed(self, tmp_path, binary_search_sequence):
        """Timeline scheduler failure halts modern execution fail-closed."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
        )
        agent.scene_sequence_ir = binary_search_sequence
        agent._execute_legacy_pipeline = MagicMock()

        # Mock TimelineScheduler.schedule to return invalid timeline
        with patch("src.agent.TimelineScheduler.schedule") as mock_sched:
            mock_sched.return_value = (
                ScheduledTimeline(
                    scene_id="Section3Scene",
                    total_duration=50.0,
                    is_valid=False,
                    steps=[],
                    actions=[],
                ),
                [
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN,
                        message="Timeline budget violation: 50.0s > 14.5s",
                        severity="ERROR",
                    )
                ],
            )

            res = agent.GENERATE_VIDEO()

            assert res is None
            assert agent._execute_legacy_pipeline.call_count == 0
            cp_path = tmp_path / "0-Binary_Search" / "checkpoint.json"
            with open(cp_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            assert cp["pipeline_mode"] == "modern"
            assert cp["stage"] == "timeline_failed"

    def test_modern_pipeline_layout_failure_fails_closed(self, tmp_path, binary_search_sequence):
        """Layout solver failure halts modern execution fail-closed."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
        )
        agent.scene_sequence_ir = binary_search_sequence
        agent._execute_legacy_pipeline = MagicMock()

        # Mock HierarchicalBoxModelLayoutSolver.solve to return invalid layout
        with patch("src.agent.HierarchicalBoxModelLayoutSolver.solve") as mock_layout:
            mock_layout.return_value = SceneSpatialLayout(
                scene_id="Section3Scene",
                canvas_box=BoundingBox2D(x=0.0, y=0.0, width=14.22, height=8.0),
                is_valid=False,
                viewports={},
                entities={},
                errors=[
                    LayoutDiagnostic(
                        error_code=LayoutErrorCode.LAYOUT_BOUNDARY_BREACH,
                        message="Entity out of frame bounds",
                    )
                ],
            )

            res = agent.GENERATE_VIDEO()

            assert res is None
            assert agent._execute_legacy_pipeline.call_count == 0
            cp_path = tmp_path / "0-Binary_Search" / "checkpoint.json"
            with open(cp_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            assert cp["pipeline_mode"] == "modern"
            assert cp["stage"] == "layout_failed"

    def test_modern_pipeline_compiler_failure_fails_closed(self, tmp_path, binary_search_sequence):
        """DeterministicSceneCompiler failure halts modern execution fail-closed."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
        )
        agent.scene_sequence_ir = binary_search_sequence
        agent._execute_legacy_pipeline = MagicMock()

        # Mock DeterministicSceneCompiler.compile to return invalid compiler output
        with patch("src.agent.DeterministicSceneCompiler.compile") as mock_compile:
            mock_compile.return_value = CompiledSceneOutput(
                scene_id="Section3Scene",
                python_source="",
                is_valid=False,
                entity_variable_map={},
                diagnostics=[
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_COMPONENT,
                        message="Unsupported component in modern compiler",
                        severity="ERROR",
                    )
                ],
                metadata={},
            )

            res = agent.GENERATE_VIDEO()

            assert res is None
            assert agent._execute_legacy_pipeline.call_count == 0
            cp_path = tmp_path / "0-Binary_Search" / "checkpoint.json"
            with open(cp_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            assert cp["pipeline_mode"] == "modern"
            assert cp["stage"] == "compilation_failed"

    def test_modern_pipeline_render_failure_fails_closed(self, tmp_path, binary_search_sequence):
        """Manim non-zero exit code halts modern execution fail-closed."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
        )
        agent.scene_sequence_ir = binary_search_sequence
        agent._execute_legacy_pipeline = MagicMock()

        # Mock subprocess.run during manim render to return exit code 1
        with patch("subprocess.run") as mock_subproc:
            mock_subproc.return_value = subprocess.CompletedProcess(
                args=["manim"],
                returncode=1,
                stdout="",
                stderr="Manim execution error: LaTeX compile error",
            )

            res = agent.GENERATE_VIDEO()

            assert res is None
            assert agent._execute_legacy_pipeline.call_count == 0
            cp_path = tmp_path / "0-Binary_Search" / "checkpoint.json"
            with open(cp_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            assert cp["pipeline_mode"] == "modern"
            assert cp["stage"] == "render_failed"


class TestPipelineModeIsolationAndSafeguards:
    """Phase 4.E: Strict mode isolation and checkpoint compatibility safeguards."""

    def test_legacy_mode_does_not_invoke_modern(self, tmp_path):
        """Legacy mode calls _execute_legacy_pipeline and never _execute_modern_pipeline."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="legacy",
        )
        agent._execute_legacy_pipeline = MagicMock(return_value="legacy_video.mp4")
        agent._execute_modern_pipeline = MagicMock()

        res = agent.GENERATE_VIDEO()

        assert res == "legacy_video.mp4"
        assert agent._execute_legacy_pipeline.call_count == 1
        assert agent._execute_modern_pipeline.call_count == 0

    def test_checkpoint_mode_mismatch_prevents_resume(self, tmp_path):
        """Agent configured with pipeline_mode='modern' rejects checkpoint saved with pipeline_mode='legacy'."""
        case_dir = tmp_path / "0-Binary_Search"
        case_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = case_dir / "checkpoint.json"
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump({
                "topic": "Binary Search",
                "level": "intermediate",
                "pipeline_mode": "legacy",
                "stage": "completed",
            }, f)

        with pytest.raises(ValueError, match="Checkpoint pipeline mode mismatch"):
            TeachingVideoAgent(
                idx=0,
                topic="Binary Search",
                level="intermediate",
                folder=str(tmp_path),
                pipeline_mode="modern",
            )
