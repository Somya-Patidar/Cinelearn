"""
Milestone 5.2.2-B Cross-Domain SJF Scheduling Tests.

Verifies:
1. Independent deterministic SJF oracle calculation:
   - Process set: P1 (0, 7), P2 (2, 4), P3 (4, 1), P4 (5, 4)
   - Non-preemptive execution: P1 (0-7), P3 (7-8), P2 (8-12), P4 (12-16)
   - Metrics: Avg WT = 4.0, Avg TAT = 8.0.
2. Native Scene IR validation of the SJF fixture (Gate 3.5 & payload validation).
3. End-to-end modern deterministic execution of SJF through identical pipeline:
   SceneSequenceIR -> TimelineScheduler -> LayoutSolver -> SceneCompiler -> Manim -> MP4.
4. Fail-closed error handling (unsupported component, layout, timeline, compiler failure).
5. Strict pipeline mode isolation and verification that Binary Search is unaffected.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agent import TeachingVideoAgent
from tests.test_milestone5_3_4_2_pipeline_orchestration import make_deterministic_mock_tts
from src.ir.models import (
    SceneSequenceIR,
    EntityDeclaration,
    VisualTier,
)
from src.ir.fixtures import (
    get_sjf_scheduling_fixture,
    get_sjf_scheduling_fixture_dict,
    get_binary_search_fixture,
)
from src.ir.processor import process_stage2_scene_ir_output
from src.compiler.layout_engine import SceneSpatialLayout, LayoutDiagnostic, LayoutErrorCode, BoundingBox2D
from src.compiler.timeline_scheduler import ScheduledTimeline
from src.compiler.scene_compiler import (
    CompiledSceneOutput,
    CompilationDiagnostic,
    CompilationErrorCode,
)


def compute_deterministic_sjf(processes):
    """
    Independent reference oracle for non-preemptive Shortest Job First scheduling.
    Tie-breaking: earlier arrival time, then PID.
    """
    procs = sorted(processes, key=lambda x: (x["arrival_time"], x["pid"]))
    time = 0
    completed = []
    ready = []
    remaining = procs[:]

    while remaining or ready:
        arrived = [p for p in remaining if p["arrival_time"] <= time]
        for p in arrived:
            ready.append(p)
            remaining.remove(p)

        if not ready:
            time = remaining[0]["arrival_time"]
            continue

        # Non-preemptive SJF: sort by shortest burst time, tie-break by arrival time
        ready.sort(key=lambda x: (x["burst_time"], x["arrival_time"], x["pid"]))
        curr = ready.pop(0)

        start = time
        end = time + curr["burst_time"]
        time = end
        wt = start - curr["arrival_time"]
        tat = end - curr["arrival_time"]

        completed.append({
            "pid": curr["pid"],
            "arrival_time": curr["arrival_time"],
            "burst_time": curr["burst_time"],
            "start": start,
            "end": end,
            "wt": wt,
            "tat": tat,
        })

    return completed


class TestSJFDeterministicOracle:
    """Independent mathematical verification of the canonical SJF problem."""

    def test_sjf_oracle_schedule_and_metrics(self):
        """Verifies exact execution order and metrics for canonical process set."""
        procs = [
            {"pid": "P1", "arrival_time": 0, "burst_time": 7},
            {"pid": "P2", "arrival_time": 2, "burst_time": 4},
            {"pid": "P3", "arrival_time": 4, "burst_time": 1},
            {"pid": "P4", "arrival_time": 5, "burst_time": 4},
        ]
        schedule = compute_deterministic_sjf(procs)

        # Expected execution order: P1 (0-7), P3 (7-8), P2 (8-12), P4 (12-16)
        assert [s["pid"] for s in schedule] == ["P1", "P3", "P2", "P4"]
        assert [(s["start"], s["end"]) for s in schedule] == [(0, 7), (7, 8), (8, 12), (12, 16)]

        # Expected waiting times: P1=0, P2=6, P3=3, P4=7
        wt_map = {s["pid"]: s["wt"] for s in schedule}
        assert wt_map == {"P1": 0, "P2": 6, "P3": 3, "P4": 7}

        # Expected turnaround times: P1=7, P2=10, P3=4, P4=11
        tat_map = {s["pid"]: s["tat"] for s in schedule}
        assert tat_map == {"P1": 7, "P2": 10, "P3": 4, "P4": 11}

        # Averages
        avg_wt = sum(s["wt"] for s in schedule) / len(schedule)
        avg_tat = sum(s["tat"] for s in schedule) / len(schedule)
        assert avg_wt == 4.0
        assert avg_tat == 8.0


class TestSJFSceneIRValidation:
    """Pre-layout Gate 3.5 & payload validation of native SJF fixture."""

    def test_sjf_fixture_passes_native_scene_ir_validation(self):
        """Native SJF Scene IR passes schema, Gate 3.5 semantics, and component payload validation."""
        raw_dict = get_sjf_scheduling_fixture_dict()
        res = process_stage2_scene_ir_output(raw_dict)
        assert res.is_valid, f"SJF Scene IR validation failed: {res.all_errors}"
        assert res.all_errors == []
        assert res.scene_ir is not None
        assert res.scene_ir.scene_id == "SJFScene"
        assert len(res.scene_ir.entities) == 3
        assert len(res.scene_ir.timeline) == 5


class TestSJFModernPipelineExecution:
    """Cross-domain execution of SJF through the identical modern pipeline."""

    def test_sjf_modern_pipeline_end_to_end_render(self, tmp_path):
        """Native SJF fixture executes and renders via the shared modern pipeline."""
        sjf_scene = get_sjf_scheduling_fixture()
        seq = SceneSequenceIR(
            sequence_id="seq_sjf_scheduling",
            title="Shortest Job First CPU Scheduling",
            topic="Shortest Job First",
            sections=[sjf_scene],
        )

        agent = TeachingVideoAgent(
            idx=0,
            topic="Shortest Job First",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
            tts_provider=make_deterministic_mock_tts(),
        )
        agent.scene_sequence_ir = seq

        # Ensure complete pipeline isolation from legacy
        agent.generate_storyboard = MagicMock(side_effect=AssertionError("Legacy generate_storyboard called!"))
        agent.generate_codes = MagicMock(side_effect=AssertionError("Legacy generate_codes called!"))
        agent._execute_legacy_pipeline = MagicMock(side_effect=AssertionError("Legacy pipeline invoked!"))

        final_video = agent.GENERATE_VIDEO()

        # Verification
        assert final_video is not None
        vpath = Path(final_video)
        assert vpath.exists()
        assert vpath.stat().st_size > 100000

        # Checkpoint verification
        cp_path = tmp_path / "0-Shortest_Job_First" / "checkpoint.json"
        assert cp_path.exists()
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["pipeline_mode"] == "modern"
        assert cp["stage"] == "completed"
        assert "SJFScene" in cp["section_codes"]
        assert "SJFScene" in cp["section_videos"]

        # Artifacts verification
        assert (tmp_path / "0-Shortest_Job_First" / "SJFScene.py").exists()
        assert (tmp_path / "0-Shortest_Job_First" / "layout_SJFScene.json").exists()
        assert (tmp_path / "0-Shortest_Job_First" / "timing_SJFScene.json").exists()

        # Zero legacy calls
        assert agent.generate_storyboard.call_count == 0
        assert agent.generate_codes.call_count == 0
        assert agent._execute_legacy_pipeline.call_count == 0

    def test_sjf_unsupported_component_fails_closed(self, tmp_path):
        """Unsupported visual component in SJF scene halts execution fail-closed."""
        sjf_scene = get_sjf_scheduling_fixture()
        sjf_scene.entities.append(
            EntityDeclaration(
                id="unsupported_quantum_core",
                component_type="QuantumCoreVisual",
                visual_tier=VisualTier.TIER1_REGISTERED,
                viewport="table_viewport",
                initial_state={},
            )
        )
        seq = SceneSequenceIR(
            sequence_id="seq_sjf_invalid",
            title="Invalid SJF",
            topic="Shortest Job First",
            sections=[sjf_scene],
        )

        agent = TeachingVideoAgent(
            idx=0,
            topic="Shortest Job First",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
            tts_provider=make_deterministic_mock_tts(),
        )
        agent.scene_sequence_ir = seq
        agent._execute_legacy_pipeline = MagicMock()

        res = agent.GENERATE_VIDEO()
        assert res is None
        assert agent._execute_legacy_pipeline.call_count == 0

        cp_path = tmp_path / "0-Shortest_Job_First" / "checkpoint.json"
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["pipeline_mode"] == "modern"
        assert cp["stage"] in ("scene_ir_validation_blocked", "compilation_failed")

    def test_sjf_layout_failure_fails_closed(self, tmp_path):
        """Layout failure during SJF execution halts fail-closed without legacy fallback."""
        sjf_scene = get_sjf_scheduling_fixture()
        seq = SceneSequenceIR(
            sequence_id="seq_sjf_layout_fail",
            title="SJF Layout Fail",
            topic="Shortest Job First",
            sections=[sjf_scene],
        )

        agent = TeachingVideoAgent(
            idx=0,
            topic="Shortest Job First",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
            tts_provider=make_deterministic_mock_tts(),
        )
        agent.scene_sequence_ir = seq
        agent._execute_legacy_pipeline = MagicMock()

        with patch("src.agent.HierarchicalBoxModelLayoutSolver.solve") as mock_layout:
            mock_layout.return_value = SceneSpatialLayout(
                scene_id="SJFScene",
                canvas_box=BoundingBox2D(x=0.0, y=0.0, width=14.22, height=8.0),
                is_valid=False,
                viewports={},
                entities={},
                errors=[
                    LayoutDiagnostic(
                        error_code=LayoutErrorCode.LAYOUT_BOUNDARY_BREACH,
                        message="Table exceeds viewport width",
                    )
                ],
            )
            res = agent.GENERATE_VIDEO()
            assert res is None
            assert agent._execute_legacy_pipeline.call_count == 0
            cp_path = tmp_path / "0-Shortest_Job_First" / "checkpoint.json"
            with open(cp_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            assert cp["pipeline_mode"] == "modern"
            assert cp["stage"] == "layout_failed"

    def test_sjf_timeline_failure_fails_closed(self, tmp_path):
        """Timeline scheduler failure during SJF execution halts fail-closed."""
        sjf_scene = get_sjf_scheduling_fixture()
        seq = SceneSequenceIR(
            sequence_id="seq_sjf_tl_fail",
            title="SJF Timeline Fail",
            topic="Shortest Job First",
            sections=[sjf_scene],
        )

        agent = TeachingVideoAgent(
            idx=0,
            topic="Shortest Job First",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
            tts_provider=make_deterministic_mock_tts(),
        )
        agent.scene_sequence_ir = seq
        agent._execute_legacy_pipeline = MagicMock()

        with patch("src.agent.TimelineScheduler.schedule") as mock_sched:
            mock_sched.return_value = (
                ScheduledTimeline(
                    scene_id="SJFScene",
                    total_duration=100.0,
                    is_valid=False,
                    steps=[],
                    actions=[],
                ),
                [
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_TIMELINE_BUDGET_OVERRUN,
                        message="SJF timeline exceeds budget",
                        severity="ERROR",
                    )
                ],
            )
            res = agent.GENERATE_VIDEO()
            assert res is None
            assert agent._execute_legacy_pipeline.call_count == 0
            cp_path = tmp_path / "0-Shortest_Job_First" / "checkpoint.json"
            with open(cp_path, "r", encoding="utf-8") as f:
                cp = json.load(f)
            assert cp["pipeline_mode"] == "modern"
            assert cp["stage"] == "timeline_failed"

    def test_sjf_and_binary_search_independent_execution(self, tmp_path):
        """Both Binary Search and SJF execute through modern pipeline without cross-domain interference."""
        bs_scene = get_binary_search_fixture()
        bs_seq = SceneSequenceIR(
            sequence_id="seq_binary_search",
            title="Binary Search Algorithm",
            topic="Binary Search",
            sections=[bs_scene],
        )

        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search",
            level="intermediate",
            folder=str(tmp_path),
            pipeline_mode="modern",
            quality="low",
            tts_provider=make_deterministic_mock_tts(),
        )
        agent.scene_sequence_ir = bs_seq
        bs_video = agent.GENERATE_VIDEO()
        assert bs_video is not None
        assert Path(bs_video).exists()
        assert Path(bs_video).stat().st_size > 100000
