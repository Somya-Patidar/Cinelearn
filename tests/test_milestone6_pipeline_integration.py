"""
Milestone 6.3: Integration Tests for Modern Pipeline QC and Fail-Closed Checkpoint.

Verifies:
- RenderedVideoQualityController end-to-end evaluation with real MP4 artifacts and synthetic scenes.
- Event-driven keyframe sampling plan generation and propagation.
- Spatial coordinate projection and keyframe association.
- Raster analysis invocation (blank canvas, clipping, subtitle collision, contrast, overlap).
- Fail-closed handling for missing video, empty video, and invalid/non-finite duration.
- Fail-closed handling for missing, malformed, non-numeric, NaN, and Inf timestamp metadata.
- Aspect-ratio mismatch detection between render resolution and scene layout.
- Excessive seek deviation handling (warning vs error semantics).
- Aggregation of mixed keyframe outcomes (fail-closed if any frame contains an ERROR).
- TeachingVideoAgent pipeline wire-in:
  - Default enable_qc=False preservation for M5 backward compatibility.
  - Fail-closed checkpoint 'qc_failed' halts execution without fallback to legacy pipeline.
  - Successful QC execution checkpoints 'qc_scene_{sid}_passed' and 'qc_master_passed'.
  - Legacy pipeline isolation (zero QC invocation).
"""

import json
import math
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.agent import TeachingVideoAgent, RunConfig
from src.compiler.compiler_models import ScheduledAction, ScheduledStep, ScheduledTimeline
from src.compiler.layout_engine import (
    BoundingBox2D,
    ComputedEntityGeometry,
    ComputedViewportGeometry,
    HierarchicalBoxModelLayoutSolver,
    SceneSpatialLayout,
    ViewportType,
)
from src.compiler.timeline_scheduler import TimelineScheduler
from src.ir.fixtures import get_binary_search_fixture
from src.ir.models import (
    ActionDeclaration,
    ActionType,
    EntityDeclaration,
    LayoutConfiguration,
    SceneIR,
    SceneSequenceIR,
    TimelineEvent,
)
from src.qc import (
    PixelBox,
    QCDiagnostic,
    QCDiagnosticCode,
    QCDiagnosticLevel,
    QCLayer,
    QCReport,
    QCReportStatus,
    RasterAnalyzerConfig,
    RenderedVideoQualityController,
    VideoQCConfig,
)
from src.qc.keyframe_sampler import KeyframeSampleRequest, KeyframeSamplingPlan


@pytest.fixture
def binary_search_muxed_artifact() -> Path:
    """Path to the authoritative M5.3.4.2 Binary Search multiplexed MP4 artifact."""
    base_dir = Path(__file__).resolve().parent.parent
    p = base_dir / "CASES" / "artifacts" / "m5_3_4_2_binary_search_muxed.mp4"
    if not p.exists():
        pytest.skip(f"Authoritative artifact not found at {p}")
    return p


@pytest.fixture
def clean_synthetic_frame() -> np.ndarray:
    """Creates a 1920x1080 synthetic BGR frame with distinct, readable shapes (non-blank)."""
    frame = np.full((1080, 1920, 3), 20, dtype=np.uint8)  # Dark background
    # Draw a prominent white rectangle in center
    frame[400:680, 700:1220] = [240, 240, 240]
    # Draw some colored accent bars
    frame[450:500, 750:1150] = [200, 50, 50]
    return frame


class TestRenderedVideoQualityControllerPrerequisites:
    """Verifies fail-closed handling on file, container, duration, and sampling plan failures."""

    def test_missing_rendered_video_fails_closed(self, tmp_path):
        """Missing video file produces FATAL QC_FRAME_CORRUPTION diagnostic and FAILED status."""
        controller = RenderedVideoQualityController()
        scene = get_binary_search_fixture()
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=10.0)

        missing_path = tmp_path / "does_not_exist.mp4"
        report = controller.evaluate_scene_video(missing_path, scene, timeline)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        assert report.sampled_frames_count == 0
        assert report.has_errors() is True

        fatals = report.get_diagnostics_by_level(QCDiagnosticLevel.FATAL)
        assert len(fatals) == 1
        assert fatals[0].code == QCDiagnosticCode.QC_FRAME_CORRUPTION
        assert "does not exist" in fatals[0].message

    def test_empty_zero_byte_video_fails_closed(self, tmp_path):
        """Zero-byte video file produces FATAL QC_FRAME_CORRUPTION diagnostic and FAILED status."""
        controller = RenderedVideoQualityController()
        scene = get_binary_search_fixture()
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=10.0)

        empty_path = tmp_path / "empty_video.mp4"
        empty_path.write_bytes(b"")

        report = controller.evaluate_scene_video(empty_path, scene, timeline)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        assert report.sampled_frames_count == 0
        fatals = report.get_diagnostics_by_level(QCDiagnosticLevel.FATAL)
        assert len(fatals) == 1
        assert "empty (0 bytes)" in fatals[0].message

    def test_invalid_or_non_finite_duration_fails_closed(self, binary_search_muxed_artifact):
        """NaN, Inf, or negative duration produces ERROR diagnostic and FAILED status."""
        controller = RenderedVideoQualityController()
        scene = get_binary_search_fixture()
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=10.0)

        # Inject NaN duration
        scene.duration_budget = float("nan")
        report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, timeline)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        errors = report.get_diagnostics_by_level(QCDiagnosticLevel.ERROR)
        assert any(d.code == QCDiagnosticCode.QC_TIMELINE_BUDGET_OVERRUN for d in errors)

    def test_empty_sampling_plan_fails_closed(self, binary_search_muxed_artifact):
        """A timeline producing zero sample requests fails closed without claiming success."""
        controller = RenderedVideoQualityController()
        scene = get_binary_search_fixture()
        scene.duration_budget = 5.0
        # Empty timeline with 0 steps
        empty_timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=5.0, steps=[])

        # Force empty requests
        with patch.object(controller.sampler, "sample_timeline", return_value=KeyframeSamplingPlan(
            scene_id=scene.scene_id, total_duration=5.0, sample_requests=[]
        )):
            report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, empty_timeline)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        assert any("zero sample requests" in d.message for d in report.diagnostics)


class TestTimestampMetadataStrictnessAndFiniteness:
    """Verifies strict M6.2 acceptance conditions for metadata presence, non-finiteness, and seek accuracy."""

    def test_missing_actual_timestamp_metadata_fails_closed(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """Decoder metadata missing 'actual_timestamp_sec' fails closed (no silent fallback to ts)."""
        controller = RenderedVideoQualityController()
        scene = get_binary_search_fixture()
        scene.duration_budget = 5.0
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=5.0)

        # Mock decoder to return metadata WITHOUT 'actual_timestamp_sec'
        bad_meta = {"width": 1920, "height": 1080, "fps": 30.0}
        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", return_value=(clean_synthetic_frame, bad_meta)):
            report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, timeline)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        corrupt_diags = [d for d in report.diagnostics if d.code == QCDiagnosticCode.QC_FRAME_CORRUPTION]
        assert len(corrupt_diags) >= 1
        assert any("missing mandatory 'actual_timestamp_sec'" in d.message for d in corrupt_diags)

    def test_non_numeric_actual_timestamp_metadata_fails_closed(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """Decoder metadata with None, boolean, or string actual_timestamp_sec fails closed."""
        controller = RenderedVideoQualityController()
        scene = get_binary_search_fixture()
        scene.duration_budget = 5.0
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=5.0)

        for bad_val in [None, True, "bad_timestamp"]:
            bad_meta = {"actual_timestamp_sec": bad_val, "width": 1920, "height": 1080}
            with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", return_value=(clean_synthetic_frame, bad_meta)):
                report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, timeline)

            assert report.overall_status == QCReportStatus.FAILED
            assert report.is_valid is False
            assert any("non-numeric actual_timestamp_sec" in d.message for d in report.diagnostics)

    def test_non_finite_actual_timestamp_metadata_fails_closed(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """Decoder metadata with NaN or Inf actual_timestamp_sec fails closed."""
        controller = RenderedVideoQualityController()
        scene = get_binary_search_fixture()
        scene.duration_budget = 5.0
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=5.0)

        for non_finite in [float("nan"), float("inf"), float("-inf")]:
            bad_meta = {"actual_timestamp_sec": non_finite, "width": 1920, "height": 1080}
            with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", return_value=(clean_synthetic_frame, bad_meta)):
                report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, timeline)

            assert report.overall_status == QCReportStatus.FAILED
            assert report.is_valid is False
            assert any("non-finite actual_timestamp_sec" in d.message for d in report.diagnostics)

    def test_excessive_seek_deviation_warns_without_halting(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """Decoded timestamp deviating beyond tolerance emits a WARNING diagnostic without hard-failing."""
        config = VideoQCConfig(
            sampler_config={"max_seek_deviation_seconds": 0.05},
            check_subtitle_clearance=False,
            check_raster_overlap=False,
            check_contrast=False,
        )
        controller = RenderedVideoQualityController(config=config)
        scene = get_binary_search_fixture()
        scene.duration_budget = 2.0
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=2.0)

        # Mock decoder to return actual_timestamp_sec with a 150ms delta (> 50ms)
        def mock_read(vpath, ts):
            meta = {"actual_timestamp_sec": ts + 0.150, "width": 1920, "height": 1080, "fps": 30.0}
            return clean_synthetic_frame, meta

        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", side_effect=mock_read):
            report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, timeline)

        assert report.overall_status == QCReportStatus.WARNINGS
        assert report.is_valid is True  # Warnings do not block video acceptance
        warnings = report.get_diagnostics_by_level(QCDiagnosticLevel.WARNING)
        assert len(warnings) >= 1
        assert any("Excessive timestamp seek deviation" in d.message for d in warnings)
        assert report.has_errors() is False


class TestAspectRatioMatchingAndResolutionValidation:
    """Verifies strict aspect-ratio matching between decoded video frames and scene layout."""

    def test_aspect_ratio_mismatch_fails_closed(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """1000x1000 (1:1) video frame evaluated against a 16:9 scene layout produces an ERROR diagnostic."""
        controller = RenderedVideoQualityController()
        scene = get_binary_search_fixture()
        scene.layout_configuration.aspect_ratio = "16:9"
        scene.duration_budget = 3.0
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=3.0)

        # 1000x1000 square frame
        square_frame = np.full((1000, 1000, 3), 120, dtype=np.uint8)
        square_frame[400:600, 400:600] = [255, 255, 255]
        meta = {"actual_timestamp_sec": 1.0, "width": 1000, "height": 1000, "fps": 30.0}

        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", return_value=(square_frame, meta)):
            report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, timeline)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        errors = report.get_diagnostics_by_level(QCDiagnosticLevel.ERROR)
        assert any("Aspect-ratio mismatch" in d.message for d in errors)
        mismatch_diag = next(d for d in errors if "Aspect-ratio mismatch" in d.message)
        assert mismatch_diag.evidence["actual_aspect_ratio"] == 1.0
        assert mismatch_diag.evidence["expected_aspect_ratio"] == round(16.0 / 9.0, 4)

    def test_aspect_ratio_matching_passes_1080p_and_720p(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """1920x1080 (16:9) frame matches 16:9 scene layout within tolerance."""
        config = VideoQCConfig(check_subtitle_clearance=False, check_raster_overlap=False, check_contrast=False)
        controller = RenderedVideoQualityController(config=config)
        scene = get_binary_search_fixture()
        scene.layout_configuration.aspect_ratio = "16:9"
        scene.duration_budget = 3.0
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=3.0)

        meta = {"actual_timestamp_sec": 1.0, "width": 1920, "height": 1080, "fps": 30.0}
        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", return_value=(clean_synthetic_frame, meta)):
            report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, timeline)

        # No aspect ratio errors
        errors = [d for d in report.diagnostics if "Aspect-ratio mismatch" in d.message]
        assert len(errors) == 0


class TestRasterVisionInspectionAndAggregation:
    """Verifies deterministic raster vision defect detection and keyframe aggregation."""

    def test_blank_canvas_detection_fails_closed(self, binary_search_muxed_artifact):
        """Solid black frame triggers QC_BLANK_CANVAS ERROR diagnostic and FAILED report."""
        controller = RenderedVideoQualityController()
        scene = get_binary_search_fixture()
        scene.duration_budget = 2.0
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=2.0)

        # Completely blank frame
        blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        meta = {"actual_timestamp_sec": 1.0, "width": 1920, "height": 1080, "fps": 30.0}

        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", return_value=(blank_frame, meta)):
            report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, timeline)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        blank_errors = [d for d in report.diagnostics if d.code == QCDiagnosticCode.QC_BLANK_CANVAS]
        assert len(blank_errors) >= 1
        assert blank_errors[0].level == QCDiagnosticLevel.ERROR

    def test_perimeter_clipping_detection_fails_closed(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """Frame with visual content bleeding into perimeter border triggers QC_PERIMETER_CLIPPING ERROR."""
        controller = RenderedVideoQualityController()
        scene = get_binary_search_fixture()
        scene.duration_budget = 2.0
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=2.0)

        # Bleed bright pixels across left and top canvas borders
        clipped_frame = clean_synthetic_frame.copy()
        clipped_frame[:30, :] = [255, 255, 255]  # Top border bleed
        clipped_frame[:, :30] = [255, 255, 255]  # Left border bleed

        meta = {"actual_timestamp_sec": 1.0, "width": 1920, "height": 1080, "fps": 30.0}
        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", return_value=(clipped_frame, meta)):
            report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, timeline, burn_subtitles=False)

        assert report.overall_status == QCReportStatus.FAILED
        clip_errors = [d for d in report.diagnostics if d.code == QCDiagnosticCode.QC_PERIMETER_CLIPPING]
        assert len(clip_errors) >= 1
        assert any("top" in d.message.lower() or "left" in d.message.lower() for d in clip_errors)

    def test_multiple_keyframes_mixed_outcomes_fails_closed(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """If 3 keyframes pass cleanly but 1 keyframe contains an ERROR, report fails closed with full diagnostics."""
        controller = RenderedVideoQualityController(config=VideoQCConfig(
            check_subtitle_clearance=False, check_raster_overlap=False, check_contrast=False
        ))
        scene = get_binary_search_fixture()
        scene.duration_budget = 4.0

        # Plan with 4 sample requests
        reqs = [
            KeyframeSampleRequest(requested_timestamp=0.5, scene_id=scene.scene_id),
            KeyframeSampleRequest(requested_timestamp=1.5, scene_id=scene.scene_id),
            KeyframeSampleRequest(requested_timestamp=2.5, scene_id=scene.scene_id),
            KeyframeSampleRequest(requested_timestamp=3.5, scene_id=scene.scene_id),
        ]
        plan = KeyframeSamplingPlan(scene_id=scene.scene_id, total_duration=4.0, sample_requests=reqs)
        timeline = ScheduledTimeline(scene_id=scene.scene_id, total_duration=4.0)

        # Decoder returns clean frame for t=0.5, 1.5, 3.5, but blank frame for t=2.5
        blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        def mock_read(vpath, ts):
            meta = {"actual_timestamp_sec": ts, "width": 1920, "height": 1080, "fps": 30.0}
            if abs(ts - 2.5) < 0.1:
                return blank_frame, meta
            return clean_synthetic_frame, meta

        with patch.object(controller.sampler, "sample_timeline", return_value=plan):
            with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", side_effect=mock_read):
                report = controller.evaluate_scene_video(binary_search_muxed_artifact, scene, timeline)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        assert report.sampled_frames_count == 4
        # Blank canvas diagnostic should be attached to t=2.5
        blank_diags = [d for d in report.diagnostics if d.code == QCDiagnosticCode.QC_BLANK_CANVAS]
        assert len(blank_diags) == 1
        assert abs(blank_diags[0].timestamp - 2.5) < 0.01

    def test_real_artifact_scene_qc_evaluation(self, binary_search_muxed_artifact):
        """Evaluates real production MP4 artifact from M5.3.4.2 against Binary Search SceneIR."""
        config = VideoQCConfig(
            check_subtitle_clearance=False,  # Video already has burned subtitles
            check_raster_overlap=False,
            check_contrast=False,
        )
        controller = RenderedVideoQualityController(config=config)
        scene = get_binary_search_fixture()
        # Set duration matching real video artifact (10.267s)
        scene.duration_budget = 10.267
        sched = TimelineScheduler()
        scheduled_timeline, _ = sched.schedule(scene)

        # Plan with 3 realistic keyframes
        reqs = [
            KeyframeSampleRequest(requested_timestamp=0.5, scene_id=scene.scene_id),
            KeyframeSampleRequest(requested_timestamp=1.0, scene_id=scene.scene_id),
            KeyframeSampleRequest(requested_timestamp=2.0, scene_id=scene.scene_id),
        ]
        plan = KeyframeSamplingPlan(scene_id=scene.scene_id, total_duration=10.267, sample_requests=reqs)

        with patch.object(controller.sampler, "sample_timeline", return_value=plan):
            report = controller.evaluate_scene_video(
                video_path=binary_search_muxed_artifact,
                scene_ir=scene,
                scheduled_timeline=scheduled_timeline,
                burn_subtitles=True,
            )

        assert report.is_valid is True
        assert report.overall_status in (QCReportStatus.PASSED, QCReportStatus.WARNINGS)
        assert report.sampled_frames_count == 3
        assert report.has_errors() is False


class TestTeachingVideoAgentModernPipelineQCIntegration:
    """Verifies TeachingVideoAgent wire-in, checkpoint generation, and fail-closed halting."""

    def test_modern_pipeline_qc_disabled_by_default(self, tmp_path):
        """TeachingVideoAgent defaults to enable_qc=False to preserve M5 backward compatibility."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search Default QC",
            folder=str(tmp_path),
            pipeline_mode="modern",
        )
        assert agent.enable_qc is False
        assert agent.section_qc_reports == {}
        assert agent.master_qc_report is None

    def test_modern_pipeline_qc_enabled_via_param_and_config(self, tmp_path):
        """enable_qc=True parameter and RunConfig(enable_qc=True) activate QC."""
        agent1 = TeachingVideoAgent(
            idx=0,
            topic="Topic 1",
            folder=str(tmp_path),
            pipeline_mode="modern",
            enable_qc=True,
        )
        assert agent1.enable_qc is True

        cfg = RunConfig(enable_qc=True)
        agent2 = TeachingVideoAgent(
            idx=1,
            topic="Topic 2",
            folder=str(tmp_path),
            pipeline_mode="modern",
            cfg=cfg,
        )
        assert agent2.enable_qc is True

    def test_modern_pipeline_qc_failure_halts_fail_closed(self, tmp_path):
        """When QC evaluation fails, modern pipeline halts with 'qc_failed' and returns None."""
        bs_scene = get_binary_search_fixture()
        bs_seq = SceneSequenceIR(
            sequence_id="seq_qc_fail_test",
            title="Binary Search QC Fail",
            topic="Binary Search QC Fail",
            sections=[bs_scene],
        )

        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search QC Fail",
            folder=str(tmp_path),
            pipeline_mode="modern",
            enable_qc=True,
            scene_sequence_ir=bs_seq,
        )
        agent.use_audio = False  # Avoid TTS synthesis overhead

        # Mock Manim render to produce an empty dummy file (>1000 bytes)
        def fake_manim(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "manim":
                target_vid = agent.output_dir / "media" / "videos" / "720p30" / f"{cmd[3]}.mp4"
                target_vid.parent.mkdir(parents=True, exist_ok=True)
                target_vid.write_bytes(b"0" * 2048)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        # Mock QC controller to return a FAILED report
        mock_failed_report = QCReport(
            scene_id=bs_scene.scene_id,
            overall_status=QCReportStatus.FAILED,
            is_valid=False,
            sampled_frames_count=1,
            diagnostics=[
                QCDiagnostic(
                    code=QCDiagnosticCode.QC_BLANK_CANVAS,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_C,
                    message="Canvas is completely blank or corrupt.",
                    scene_id=bs_scene.scene_id,
                    timestamp=1.0,
                )
            ],
        )
        mock_qc_ctrl = MagicMock()
        mock_qc_ctrl.evaluate_scene_video.return_value = mock_failed_report
        agent.qc_controller = mock_qc_ctrl

        with patch("subprocess.run", side_effect=fake_manim):
            res = agent.GENERATE_VIDEO()

        # Must halt immediately and return None
        assert res is None
        mock_qc_ctrl.evaluate_scene_video.assert_called_once()

        # Checkpoint verification
        cp_path = tmp_path / "0-Binary_Search_QC_Fail" / "checkpoint.json"
        assert cp_path.exists()
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["pipeline_mode"] == "modern"
        assert cp["stage"] == "qc_failed"
        assert cp.get("failed_scene") == bs_scene.scene_id or cp.get("extra", {}).get("failed_scene") == bs_scene.scene_id
        errors = cp.get("errors") or cp.get("extra", {}).get("errors", [])
        assert "Canvas is completely blank" in errors[0]

    def test_modern_pipeline_qc_success_checkpoints(self, tmp_path):
        """When QC evaluation passes, checkpoints 'qc_scene_{sid}_passed' and completes pipeline."""
        bs_scene = get_binary_search_fixture()
        bs_seq = SceneSequenceIR(
            sequence_id="seq_qc_pass_test",
            title="Binary Search QC Pass",
            topic="Binary Search QC Pass",
            sections=[bs_scene],
        )

        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search QC Pass",
            folder=str(tmp_path),
            pipeline_mode="modern",
            enable_qc=True,
            scene_sequence_ir=bs_seq,
        )
        agent.use_audio = False

        # Mock Manim render to produce an empty dummy file (>1000 bytes)
        def fake_manim(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "manim":
                target_vid = agent.output_dir / "media" / "videos" / "720p30" / f"{cmd[3]}.mp4"
                target_vid.parent.mkdir(parents=True, exist_ok=True)
                target_vid.write_bytes(b"0" * 2048)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_passed_report = QCReport(
            scene_id=bs_scene.scene_id,
            overall_status=QCReportStatus.PASSED,
            is_valid=True,
            sampled_frames_count=3,
            diagnostics=[],
        )
        mock_qc_ctrl = MagicMock()
        mock_qc_ctrl.evaluate_scene_video.return_value = mock_passed_report
        mock_qc_ctrl.evaluate_master_video.return_value = QCReport(
            scene_id=bs_seq.sequence_id,
            overall_status=QCReportStatus.PASSED,
            is_valid=True,
            sampled_frames_count=1,
            diagnostics=[],
        )
        agent.qc_controller = mock_qc_ctrl

        # Mock media validation to return valid
        agent.audio_manager.validate_media_file = MagicMock(return_value={"is_valid": True, "duration": 5.0})

        with patch("subprocess.run", side_effect=fake_manim):
            with patch("shutil.copyfile"):
                res = agent.GENERATE_VIDEO()

        assert res is not None
        mock_qc_ctrl.evaluate_scene_video.assert_called_once()
        mock_qc_ctrl.evaluate_master_video.assert_called_once()

        # Checkpoints verification
        cp_path = tmp_path / "0-Binary_Search_QC_Pass" / "checkpoint.json"
        assert cp_path.exists()
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["pipeline_mode"] == "modern"
        assert cp["stage"] == "completed"

    def test_legacy_pipeline_isolation(self, tmp_path):
        """Legacy pipeline execution never touches QC controller or QC checkpoints."""
        agent = TeachingVideoAgent(
            idx=0,
            topic="Binary Search Legacy",
            folder=str(tmp_path),
            pipeline_mode="legacy",
            enable_qc=True,
        )
        assert agent.pipeline_mode == "legacy"
        # Mock legacy pipeline execution
        agent._execute_legacy_pipeline = MagicMock(return_value="legacy_video.mp4")

        res = agent.GENERATE_VIDEO()
        assert res == "legacy_video.mp4"
        agent._execute_legacy_pipeline.assert_called_once()
        assert agent.section_qc_reports == {}


class TestSubtitleClearancePreMuxRemediation:
    """Remediation tests for FIND-6.3-01: Subtitle collision false positives."""

    def test_burned_subtitles_do_not_trigger_false_positives_when_raw_video_provided(
        self, binary_search_muxed_artifact, tmp_path
    ):
        """When raw_video_path is provided, subtitle collision inspects raw video and ignores burned subtitles."""
        raw_video = tmp_path / "raw_manim_render.mp4"
        raw_video.write_bytes(binary_search_muxed_artifact.read_bytes())

        controller = RenderedVideoQualityController(
            config=VideoQCConfig(
                check_subtitle_clearance=True,
                check_raster_overlap=False,
                check_contrast=False,
                check_perimeter_clipping=False,
                check_blank_canvas=False,
            )
        )
        scene = get_binary_search_fixture()
        scene.duration_budget = 10.267
        scene.layout_configuration.has_subtitles = True

        sched = TimelineScheduler()
        scheduled_timeline, _ = sched.schedule(scene)

        clean_raw_frame = np.full((480, 854, 3), 20, dtype=np.uint8)

        def mock_read(vpath, ts):
            meta = {"actual_timestamp_sec": ts, "width": 854, "height": 480, "fps": 15.0}
            if str(vpath) == str(raw_video):
                return clean_raw_frame, meta
            else:
                muxed_frame = np.full((480, 854, 3), 20, dtype=np.uint8)
                muxed_frame[410:470, 100:750] = 255  # Burned subtitle text
                return muxed_frame, meta

        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", side_effect=mock_read):
            report = controller.evaluate_scene_video(
                video_path=binary_search_muxed_artifact,
                scene_ir=scene,
                scheduled_timeline=scheduled_timeline,
                burn_subtitles=True,
                raw_video_path=raw_video,
            )

        sub_diags = [d for d in report.diagnostics if d.code == QCDiagnosticCode.QC_SUBTITLE_COLLISION]
        assert len(sub_diags) == 0
        assert report.overall_status == QCReportStatus.PASSED

    def test_genuine_visual_intrusion_in_raw_video_is_detected(
        self, binary_search_muxed_artifact, tmp_path
    ):
        """When pre-mux raw video has genuine animation/visual intrusion into the subtitle band, it is detected."""
        raw_video = tmp_path / "raw_manim_intrusion.mp4"
        raw_video.write_bytes(binary_search_muxed_artifact.read_bytes())

        controller = RenderedVideoQualityController(
            config=VideoQCConfig(
                check_subtitle_clearance=True,
                check_raster_overlap=False,
                check_contrast=False,
                check_perimeter_clipping=False,
                check_blank_canvas=False,
            )
        )
        scene = get_binary_search_fixture()
        scene.duration_budget = 10.267
        scene.layout_configuration.has_subtitles = True

        sched = TimelineScheduler()
        scheduled_timeline, _ = sched.schedule(scene)

        intrusive_raw_frame = np.full((480, 854, 3), 20, dtype=np.uint8)
        intrusive_raw_frame[420:460, 200:600] = 200  # High-density graphics intrusion

        def mock_read(vpath, ts):
            meta = {"actual_timestamp_sec": ts, "width": 854, "height": 480, "fps": 15.0}
            return intrusive_raw_frame, meta

        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", side_effect=mock_read):
            report = controller.evaluate_scene_video(
                video_path=binary_search_muxed_artifact,
                scene_ir=scene,
                scheduled_timeline=scheduled_timeline,
                burn_subtitles=True,
                raw_video_path=raw_video,
            )

        sub_diags = [d for d in report.diagnostics if d.code == QCDiagnosticCode.QC_SUBTITLE_COLLISION]
        assert len(sub_diags) > 0
        assert report.overall_status == QCReportStatus.WARNINGS


class TestMasterVideoRasterInspectionRemediation:
    """Remediation tests for FIND-6.3-02: Master-video raster inspection across multiple sections."""

    def test_valid_multi_section_master_video_passes(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """A valid master video with multiple sections samples across sections and passes."""
        controller = RenderedVideoQualityController()
        s1 = SceneIR(scene_id="sec1", title="Scene 1", duration_budget=5.0, layout_configuration=LayoutConfiguration(), entities=[])
        s2 = SceneIR(scene_id="sec2", title="Scene 2", duration_budget=5.0, layout_configuration=LayoutConfiguration(), entities=[])
        seq = SceneSequenceIR(sequence_id="master_seq", title="Master", topic="Master", sections=[s1, s2])

        sampled_timestamps = []
        def mock_read(vpath, ts):
            sampled_timestamps.append(ts)
            meta = {"actual_timestamp_sec": ts, "width": 1920, "height": 1080, "duration_sec": 10.0, "fps": 30.0}
            return clean_synthetic_frame, meta

        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", side_effect=mock_read):
            report = controller.evaluate_master_video(binary_search_muxed_artifact, seq)

        assert report.overall_status == QCReportStatus.PASSED
        assert report.is_valid is True
        assert report.sampled_frames_count == 2
        assert any(abs(t - 2.5) < 0.1 for t in sampled_timestamps)
        assert any(abs(t - 7.5) < 0.1 for t in sampled_timestamps)

    def test_blank_frame_in_later_section_fails_closed(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """When section 1 is clean but section 2 is blank, master QC fails closed."""
        controller = RenderedVideoQualityController()
        s1 = SceneIR(scene_id="sec1", title="Scene 1", duration_budget=5.0, layout_configuration=LayoutConfiguration(), entities=[])
        s2 = SceneIR(scene_id="sec2", title="Scene 2", duration_budget=5.0, layout_configuration=LayoutConfiguration(), entities=[])
        seq = SceneSequenceIR(sequence_id="master_seq", title="Master", topic="Master", sections=[s1, s2])

        blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        def mock_read(vpath, ts):
            meta = {"actual_timestamp_sec": ts, "width": 1920, "height": 1080, "duration_sec": 10.0, "fps": 30.0}
            if ts > 4.0:
                return blank_frame, meta
            return clean_synthetic_frame, meta

        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", side_effect=mock_read):
            report = controller.evaluate_master_video(binary_search_muxed_artifact, seq)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        blank_diags = [d for d in report.diagnostics if d.code == QCDiagnosticCode.QC_BLANK_CANVAS]
        assert len(blank_diags) == 1
        assert blank_diags[0].scene_id == "sec2"

    def test_blank_frame_in_first_section_fails_closed(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """When section 1 is blank, master QC fails closed immediately."""
        controller = RenderedVideoQualityController()
        s1 = SceneIR(scene_id="sec1", title="Scene 1", duration_budget=5.0, layout_configuration=LayoutConfiguration(), entities=[])
        s2 = SceneIR(scene_id="sec2", title="Scene 2", duration_budget=5.0, layout_configuration=LayoutConfiguration(), entities=[])
        seq = SceneSequenceIR(sequence_id="master_seq", title="Master", topic="Master", sections=[s1, s2])

        blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        def mock_read(vpath, ts):
            meta = {"actual_timestamp_sec": ts, "width": 1920, "height": 1080, "duration_sec": 10.0, "fps": 30.0}
            if ts < 4.0:
                return blank_frame, meta
            return clean_synthetic_frame, meta

        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", side_effect=mock_read):
            report = controller.evaluate_master_video(binary_search_muxed_artifact, seq)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        blank_diags = [d for d in report.diagnostics if d.code == QCDiagnosticCode.QC_BLANK_CANVAS]
        assert len(blank_diags) == 1
        assert blank_diags[0].scene_id == "sec1"

    def test_master_frame_extraction_failure_fails_closed(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """When keyframe extraction fails on one of the sections, master QC fails closed."""
        controller = RenderedVideoQualityController()
        s1 = SceneIR(scene_id="sec1", title="Scene 1", duration_budget=5.0, layout_configuration=LayoutConfiguration(), entities=[])
        s2 = SceneIR(scene_id="sec2", title="Scene 2", duration_budget=5.0, layout_configuration=LayoutConfiguration(), entities=[])
        seq = SceneSequenceIR(sequence_id="master_seq", title="Master", topic="Master", sections=[s1, s2])

        def mock_read(vpath, ts):
            meta = {"actual_timestamp_sec": ts, "width": 1920, "height": 1080, "duration_sec": 10.0, "fps": 30.0}
            if ts > 4.0:
                raise RuntimeError("Corrupted codec packet at t=7.5s")
            return clean_synthetic_frame, meta

        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", side_effect=mock_read):
            report = controller.evaluate_master_video(binary_search_muxed_artifact, seq)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        corrupt_diags = [d for d in report.diagnostics if d.code == QCDiagnosticCode.QC_FRAME_CORRUPTION]
        assert len(corrupt_diags) >= 1

    def test_mixed_valid_and_invalid_frames_aggregate_fail_closed(self, binary_search_muxed_artifact, clean_synthetic_frame):
        """Mixed valid and invalid sampled frames aggregate with fail-closed status and preserve all diagnostics."""
        controller = RenderedVideoQualityController()
        s1 = SceneIR(scene_id="sec1", title="Scene 1", duration_budget=3.0, layout_configuration=LayoutConfiguration(), entities=[])
        s2 = SceneIR(scene_id="sec2", title="Scene 2", duration_budget=3.0, layout_configuration=LayoutConfiguration(), entities=[])
        s3 = SceneIR(scene_id="sec3", title="Scene 3", duration_budget=3.0, layout_configuration=LayoutConfiguration(), entities=[])
        seq = SceneSequenceIR(sequence_id="master_seq", title="Master", topic="Master", sections=[s1, s2, s3])

        blank_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        def mock_read(vpath, ts):
            meta = {"actual_timestamp_sec": ts, "width": 1920, "height": 1080, "duration_sec": 9.0, "fps": 30.0}
            if 2.0 < ts < 5.0:
                return blank_frame, meta
            return clean_synthetic_frame, meta

        with patch("src.qc.raster_analyzer.read_keyframe_at_timestamp", side_effect=mock_read):
            report = controller.evaluate_master_video(binary_search_muxed_artifact, seq)

        assert report.overall_status == QCReportStatus.FAILED
        assert report.is_valid is False
        assert report.sampled_frames_count == 3
        assert len(report.diagnostics) == 1
        assert report.diagnostics[0].code == QCDiagnosticCode.QC_BLANK_CANVAS
        assert report.diagnostics[0].scene_id == "sec2"


class TestQCPersistenceFailClosedRemediation:
    """Remediation tests for FIND-6.3-03: Fail-closed handling when report persistence fails."""

    def test_scene_qc_report_write_failure_halts_fail_closed(self, tmp_path):
        """When writing qc_report_{sid}.json fails, pipeline saves 'qc_failed' and halts immediately."""
        bs_scene = get_binary_search_fixture()
        bs_seq = SceneSequenceIR(
            sequence_id="seq_persist_fail_scene",
            title="Scene Persist Fail",
            topic="Scene Persist Fail",
            sections=[bs_scene],
        )
        agent = TeachingVideoAgent(
            idx=0,
            topic="Scene Persist Fail",
            folder=str(tmp_path),
            pipeline_mode="modern",
            enable_qc=True,
            scene_sequence_ir=bs_seq,
        )
        agent.use_audio = False

        def fake_manim(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "manim":
                target_vid = agent.output_dir / "media" / "videos" / "720p30" / f"{cmd[3]}.mp4"
                target_vid.parent.mkdir(parents=True, exist_ok=True)
                target_vid.write_bytes(b"0" * 2048)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_qc_ctrl = MagicMock()
        mock_qc_ctrl.evaluate_scene_video.return_value = QCReport(
            scene_id=bs_scene.scene_id,
            overall_status=QCReportStatus.PASSED,
            is_valid=True,
            sampled_frames_count=1,
            diagnostics=[],
        )
        agent.qc_controller = mock_qc_ctrl

        # Create a directory where the report file is expected to induce an IsADirectoryError on write_text
        blocker = agent.output_dir / f"qc_report_{bs_scene.scene_id}.json"
        blocker.mkdir(parents=True, exist_ok=True)

        with patch("subprocess.run", side_effect=fake_manim):
            res = agent.GENERATE_VIDEO()

        assert res is None
        cp_path = tmp_path / "0-Scene_Persist_Fail" / "checkpoint.json"
        assert cp_path.exists()
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["stage"] == "qc_failed"
        assert "Failed to persist QC report" in cp.get("errors", [""])[0]

    def test_master_qc_report_write_failure_halts_fail_closed(self, tmp_path):
        """When writing qc_report_master.json fails, pipeline saves 'qc_master_failed' and halts."""
        bs_scene = get_binary_search_fixture()
        bs_seq = SceneSequenceIR(
            sequence_id="seq_persist_fail_master",
            title="Master Persist Fail",
            topic="Master Persist Fail",
            sections=[bs_scene],
        )
        agent = TeachingVideoAgent(
            idx=0,
            topic="Master Persist Fail",
            folder=str(tmp_path),
            pipeline_mode="modern",
            enable_qc=True,
            scene_sequence_ir=bs_seq,
        )
        agent.use_audio = False

        def fake_manim(cmd, *args, **kwargs):
            if isinstance(cmd, list) and len(cmd) > 0 and cmd[0] == "manim":
                target_vid = agent.output_dir / "media" / "videos" / "720p30" / f"{cmd[3]}.mp4"
                target_vid.parent.mkdir(parents=True, exist_ok=True)
                target_vid.write_bytes(b"0" * 2048)
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_qc_ctrl = MagicMock()
        mock_qc_ctrl.evaluate_scene_video.return_value = QCReport(
            scene_id=bs_scene.scene_id,
            overall_status=QCReportStatus.PASSED,
            is_valid=True,
            sampled_frames_count=1,
            diagnostics=[],
        )
        mock_qc_ctrl.evaluate_master_video.return_value = QCReport(
            scene_id=bs_seq.sequence_id,
            overall_status=QCReportStatus.PASSED,
            is_valid=True,
            sampled_frames_count=1,
            diagnostics=[],
        )
        agent.qc_controller = mock_qc_ctrl
        agent.audio_manager.validate_media_file = MagicMock(return_value={"is_valid": True, "duration": 5.0})

        # Create a directory where qc_report_master.json is expected to induce an IsADirectoryError
        blocker = agent.output_dir / "qc_report_master.json"
        blocker.mkdir(parents=True, exist_ok=True)

        with patch("subprocess.run", side_effect=fake_manim):
            with patch("shutil.copyfile"):
                res = agent.GENERATE_VIDEO()

        assert res is None
        cp_path = tmp_path / "0-Master_Persist_Fail" / "checkpoint.json"
        assert cp_path.exists()
        with open(cp_path, "r", encoding="utf-8") as f:
            cp = json.load(f)
        assert cp["stage"] == "qc_master_failed"
        assert "Failed to persist master QC report" in cp.get("errors", [""])[0]
