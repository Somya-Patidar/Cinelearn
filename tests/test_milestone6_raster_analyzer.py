"""
Milestone 6.1: Deterministic Raster Vision Analyzer Test Suite.

Verifies:
1. QC Pydantic data models (strict validation, extra='forbid', JSON serialization).
2. Frame array validation (valid shapes, channels, dtypes, empty/corrupted inputs).
3. Blank canvas detection (uniform black/white, minimal valid title cards, low-detail scenes).
4. Perimeter clipping detection (edge breaches across left, right, top, bottom, ignore_edges).
5. Subtitle safe-zone clearance analysis and collision detection (with/without exclusion masks).
6. WCAG 2.1 relative luminance and contrast ratio evaluation (high contrast, low contrast, inconclusive unimodal).
7. Pairwise ROI overlap analysis (geometric overlap, empty intersection vs active raster collision).
8. OpenCV keyframe extraction and resource release (valid seek, invalid timestamp, missing video).
9. Real media artifact smoke testing on CASES/artifacts/m5_3_4_2_binary_search_muxed.mp4.
"""

import os
import tempfile
import pytest
import numpy as np
import cv2
from pathlib import Path
from pydantic import ValidationError

from src.qc.models import (
    QCDiagnosticLevel,
    QCLayer,
    QCDiagnosticCode,
    QCReportStatus,
    PixelBox,
    BlankCanvasEvidence,
    PerimeterClippingEvidence,
    SubtitleCollisionEvidence,
    ContrastEvidence,
    OverlapEvidence,
    QCDiagnostic,
    QCReport,
    RasterAnalyzerConfig,
)
from src.qc.raster_analyzer import (
    validate_frame_array,
    to_grayscale,
    estimate_background_color,
    read_keyframe_at_timestamp,
    detect_blank_canvas,
    detect_perimeter_clipping,
    detect_subtitle_collision,
    evaluate_roi_contrast,
    evaluate_roi_overlap,
)


# =============================================================================
# 1. QC Data Model Strictness and Serialization Tests
# =============================================================================

class TestQCDataModels:
    """Verifies strict validation, immutability of schema, and serialization."""

    def test_pixel_box_valid_and_area(self):
        box = PixelBox(x_min=10, y_min=20, x_max=110, y_max=120)
        assert box.width == 100
        assert box.height == 100
        assert box.area == 10000

    def test_pixel_box_rejects_inverted_or_zero_coordinates(self):
        with pytest.raises(ValidationError):
            PixelBox(x_min=100, y_min=20, x_max=50, y_max=120)  # x_max <= x_min
        with pytest.raises(ValidationError):
            PixelBox(x_min=10, y_min=100, x_max=50, y_max=100)  # y_max <= y_min
        with pytest.raises(ValidationError):
            PixelBox(x_min=-5, y_min=20, x_max=50, y_max=120)  # negative coordinate

    def test_pixel_box_forbids_extra_fields(self):
        with pytest.raises(ValidationError):
            PixelBox(x_min=10, y_min=20, x_max=100, y_max=200, extra_val=123)

    def test_qc_diagnostic_serialization_and_validation(self):
        diag = QCDiagnostic(
            code=QCDiagnosticCode.QC_BLANK_CANVAS,
            level=QCDiagnosticLevel.ERROR,
            layer=QCLayer.LAYER_C,
            message="Canvas is completely blank.",
            scene_id="Scene_1",
            timestamp=2.5,
            evidence={"std": 0.0, "entropy": 0.0},
        )
        data = diag.model_dump()
        assert data["code"] == "QC_BLANK_CANVAS"
        assert data["level"] == "ERROR"
        assert data["layer"] == "Layer C"
        assert data["timestamp"] == 2.5

    def test_qc_diagnostic_rejects_negative_timestamp(self):
        with pytest.raises(ValidationError):
            QCDiagnostic(
                code=QCDiagnosticCode.QC_BLANK_CANVAS,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_C,
                message="Negative timestamp",
                timestamp=-1.0,
            )

    def test_qc_report_status_methods_and_json_roundtrip(self):
        report = QCReport(
            scene_id="Scene_SJF",
            overall_status=QCReportStatus.WARNINGS,
            is_valid=True,
            sampled_frames_count=10,
            diagnostics=[
                QCDiagnostic(
                    code=QCDiagnosticCode.QC_SUBTITLE_COLLISION,
                    level=QCDiagnosticLevel.WARNING,
                    layer=QCLayer.LAYER_C,
                    message="Subtitle warning",
                )
            ],
            raster_metrics={"mean_contrast": 6.5},
        )
        assert report.has_warnings() is True
        assert report.has_errors() is False
        assert len(report.get_diagnostics_by_level(QCDiagnosticLevel.WARNING)) == 1

        # JSON serialization roundtrip
        json_str = report.model_dump_json()
        restored = QCReport.model_validate_json(json_str)
        assert restored.scene_id == "Scene_SJF"
        assert restored.overall_status == QCReportStatus.WARNINGS
        assert restored.is_valid is True


# =============================================================================
# 2. Frame Array Validation and Grayscale Extraction Tests
# =============================================================================

class TestFrameValidation:
    """Verifies defensive checks on image arrays, dimensions, and channel counts."""

    def test_valid_bgr_frame(self):
        arr = np.zeros((480, 854, 3), dtype=np.uint8)
        val = validate_frame_array(arr)
        assert val.shape == (480, 854, 3)

    def test_valid_grayscale_frame(self):
        arr = np.zeros((480, 854), dtype=np.uint8)
        val = validate_frame_array(arr)
        assert val.shape == (480, 854)

    def test_float_normalization(self):
        arr = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        val = validate_frame_array(arr)
        assert val.dtype == np.uint8
        assert 126 <= val[0, 0, 0] <= 128

    def test_rejects_none_input(self):
        with pytest.raises(ValueError, match="cannot be None"):
            validate_frame_array(None)

    def test_rejects_non_numpy_type(self):
        with pytest.raises(TypeError, match="Expected numpy.ndarray"):
            validate_frame_array([[0, 1], [1, 0]])

    def test_rejects_empty_array(self):
        with pytest.raises(ValueError, match="empty"):
            validate_frame_array(np.array([], dtype=np.uint8))

    def test_rejects_invalid_dimensions(self):
        with pytest.raises(ValueError, match="must have 2 or 3 dimensions"):
            validate_frame_array(np.zeros((10,), dtype=np.uint8))
        with pytest.raises(ValueError, match="must have 2 or 3 dimensions"):
            validate_frame_array(np.zeros((1, 2, 3, 4), dtype=np.uint8))

    def test_rejects_unsupported_channels(self):
        with pytest.raises(ValueError, match="must have 1, 3, or 4 channels"):
            validate_frame_array(np.zeros((100, 100, 5), dtype=np.uint8))

    def test_to_grayscale_conversion(self):
        # 3-channel BGR
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255  # Red
        gray = to_grayscale(bgr)
        assert gray.ndim == 2
        assert gray.shape == (100, 100)
        assert 70 <= gray[0, 0] <= 80  # standard red luminance


# =============================================================================
# 3. Blank and Corrupted Canvas Detection Tests
# =============================================================================

class TestBlankCanvasDetection:
    """Verifies blank/corrupted canvas detection with negative test protection."""

    def test_uniform_black_frame_detected_as_blank(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        is_blank, ev, diag = detect_blank_canvas(frame)
        assert is_blank is True
        assert ev.is_blank is True
        assert ev.std_luminance == 0.0
        assert ev.entropy == 0.0
        assert diag is not None
        assert diag.code == QCDiagnosticCode.QC_BLANK_CANVAS
        assert diag.level == QCDiagnosticLevel.ERROR

    def test_uniform_white_frame_detected_as_blank(self):
        frame = np.full((480, 854, 3), 255, dtype=np.uint8)
        is_blank, ev, diag = detect_blank_canvas(frame)
        assert is_blank is True
        assert ev.is_blank is True
        assert ev.std_luminance == 0.0
        assert ev.entropy == 0.0

    def test_simple_title_card_not_falsely_classified_as_blank(self):
        """A minimal valid educational title card on black background must NOT be blank."""
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        # Render a clear white title block
        cv2.putText(frame, "CineLearn Educational Video", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        is_blank, ev, diag = detect_blank_canvas(frame)
        assert is_blank is False
        assert ev.is_blank is False
        assert diag is None
        assert ev.entropy > 0.02
        assert ev.std_luminance > 10.0

    def test_low_detail_valid_geometric_shape_not_blank(self):
        """A single geometric diagram (e.g. circle) on dark background must pass."""
        frame = np.full((480, 854, 3), 15, dtype=np.uint8)  # dark background
        cv2.circle(frame, (427, 240), 60, (0, 200, 255), 3)  # yellow circle
        is_blank, ev, diag = detect_blank_canvas(frame)
        assert is_blank is False
        assert ev.is_blank is False


# =============================================================================
# 4. Perimeter Clipping Detection Tests
# =============================================================================

class TestPerimeterClippingDetection:
    """Verifies edge clipping detection and ignore_edges filtering."""

    def test_clean_frame_has_no_clipping(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        # Center square well away from perimeters
        cv2.rectangle(frame, (200, 100), (600, 350), (255, 255, 255), -1)
        is_clipped, evs, diags = detect_perimeter_clipping(frame)
        assert is_clipped is False
        assert len(diags) == 0
        assert all(not e.is_clipped for e in evs)

    def test_top_edge_clipping_detected(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        # Shape touching top border
        cv2.rectangle(frame, (200, 0), (600, 30), (255, 255, 255), -1)
        is_clipped, evs, diags = detect_perimeter_clipping(frame)
        assert is_clipped is True
        clipped_edges = [e.edge for e in evs if e.is_clipped]
        assert "top" in clipped_edges
        assert any(d.code == QCDiagnosticCode.QC_PERIMETER_CLIPPING for d in diags)

    def test_left_edge_clipping_detected(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        # Shape overflowing left border
        cv2.rectangle(frame, (0, 100), (25, 300), (0, 255, 0), -1)
        is_clipped, evs, diags = detect_perimeter_clipping(frame)
        assert is_clipped is True
        clipped_edges = [e.edge for e in evs if e.is_clipped]
        assert "left" in clipped_edges

    def test_right_edge_clipping_detected(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        # Shape touching right border
        cv2.rectangle(frame, (830, 150), (854, 350), (0, 0, 255), -1)
        is_clipped, evs, diags = detect_perimeter_clipping(frame)
        assert is_clipped is True
        assert "right" in [e.edge for e in evs if e.is_clipped]

    def test_ignore_edges_skips_designated_boundaries(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        # Left border decorated intentionally (stopping before top/bottom bands)
        cv2.rectangle(frame, (0, 50), (20, 400), (100, 100, 100), -1)
        is_clipped, evs, diags = detect_perimeter_clipping(frame, ignore_edges=["left"])
        assert is_clipped is False
        assert len(diags) == 0


# =============================================================================
# 5. Subtitle Clearance Band Analysis Tests
# =============================================================================

class TestSubtitleClearance:
    """Verifies subtitle zone collision analysis and exclusion masking."""

    def test_clean_canvas_has_no_subtitle_collision(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 50), (700, 300), (255, 255, 255), -1)
        is_col, ev, diag = detect_subtitle_collision(frame)
        assert is_col is False
        assert ev.is_colliding is False
        assert diag is None

    def test_intrusion_into_subtitle_band_detected(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        # Element drawn into bottom subtitle band (y >= 410 on 480p)
        cv2.rectangle(frame, (200, 415), (600, 465), (255, 255, 255), -1)
        is_col, ev, diag = detect_subtitle_collision(frame)
        assert is_col is True
        assert ev.is_colliding is True
        assert diag is not None
        assert diag.code == QCDiagnosticCode.QC_SUBTITLE_COLLISION
        assert diag.level == QCDiagnosticLevel.WARNING

    def test_exclusion_mask_suppresses_legitimate_content(self):
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 415), (600, 465), (255, 255, 255), -1)
        mask = np.zeros((480, 854), dtype=bool)
        mask[415:465, 200:600] = True  # Exclude known permitted element
        is_col, ev, diag = detect_subtitle_collision(frame, exclusion_mask=mask)
        assert is_col is False
        assert diag is None


# =============================================================================
# 6. WCAG 2.1 Contrast Ratio Evaluation Tests
# =============================================================================

class TestContrastEvaluation:
    """Verifies relative luminance and contrast calculations on known combinations."""

    def test_white_on_black_contrast_is_twenty_one_to_one(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (25, 25), (75, 75), (255, 255, 255), -1)
        contrast, ev, diag = evaluate_roi_contrast(frame, roi_box=(20, 20, 80, 80))
        assert ev.is_conclusive is True
        assert round(contrast, 1) == 21.0
        assert ev.is_compliant is True
        assert diag is None

    def test_dark_grey_on_black_contrast_fails_wcag(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Dark grey (#333333 = 51) on black (#000000 = 0)
        cv2.rectangle(frame, (25, 25), (75, 75), (51, 51, 51), -1)
        contrast, ev, diag = evaluate_roi_contrast(frame, roi_box=(20, 20, 80, 80))
        assert ev.is_conclusive is True
        assert contrast < 3.0  # Measured ~1.66:1
        assert ev.is_compliant is False
        assert diag is not None
        assert diag.code == QCDiagnosticCode.QC_LOW_CONTRAST
        assert diag.level == QCDiagnosticLevel.ERROR

    def test_explicit_fg_and_bg_masks(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        fg_mask = np.zeros((50, 50), dtype=bool)
        bg_mask = np.zeros((50, 50), dtype=bool)
        fg_mask[10:20, 10:20] = True
        bg_mask[30:40, 30:40] = True
        frame[fg_mask] = [255, 255, 255]  # White fg
        frame[bg_mask] = [0, 0, 0]        # Black bg
        contrast, ev, diag = evaluate_roi_contrast(frame, fg_mask=fg_mask, bg_mask=bg_mask)
        assert round(contrast, 1) == 21.0
        assert ev.is_compliant is True

    def test_unimodal_roi_returns_inconclusive(self):
        """A flat solid patch cannot be reliably separated into text foreground and background."""
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        contrast, ev, diag = evaluate_roi_contrast(frame)
        assert ev.is_conclusive is False
        assert diag is not None
        assert diag.code == QCDiagnosticCode.QC_INCONCLUSIVE_MEASUREMENT
        assert diag.level == QCDiagnosticLevel.WARNING


# =============================================================================
# 7. Pairwise ROI Overlap Evaluation Tests
# =============================================================================

class TestROIOverlapEvaluation:
    """Verifies geometric overlap and raster edge collision differentiation."""

    def test_non_overlapping_boxes(self):
        box_a = (10, 10, 50, 50)
        box_b = (60, 60, 100, 100)
        is_ov, ev, diag = evaluate_roi_overlap(box_a, box_b)
        assert is_ov is False
        assert ev.is_geometric_overlap is False
        assert ev.intersection_area == 0
        assert diag is None

    def test_geometrically_overlapping_boxes_without_raster_content(self):
        """Two boxes overlap, but the image frame has no visible content in intersection."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        box_a = (20, 20, 80, 80)
        box_b = (60, 60, 120, 120)  # Intersection is [60, 60, 80, 80] (area 400)
        # Background is pure black in intersection
        is_ov, ev, diag = evaluate_roi_overlap(box_a, box_b, frame=frame)
        assert ev.is_geometric_overlap is True
        assert ev.is_raster_overlap is False
        assert is_ov is False  # Empty raster intersection does not collide visually
        assert diag is None

    def test_overlapping_boxes_with_active_raster_content(self):
        """Two boxes overlap, and visible raster edges intersect."""
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        box_a = (20, 20, 80, 80)
        box_b = (60, 60, 120, 120)
        # Draw intersecting diagonal lines through the intersection patch
        cv2.line(frame, (60, 60), (80, 80), (255, 255, 255), 2)
        cv2.line(frame, (60, 80), (80, 60), (255, 255, 255), 2)
        is_ov, ev, diag = evaluate_roi_overlap(box_a, box_b, frame=frame)
        assert ev.is_geometric_overlap is True
        assert ev.is_raster_overlap is True
        assert is_ov is True
        assert diag is not None
        assert diag.code == QCDiagnosticCode.QC_RASTER_OVERLAP


# =============================================================================
# 8. Video Frame Extraction and Real Artifact Smoke Tests
# =============================================================================

class TestVideoExtractionAndRealArtifact:
    """Verifies OpenCV keyframe extraction and smoke tests on real M5 video artifact."""

    def test_missing_video_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_keyframe_at_timestamp("non_existent_video_file.mp4", 1.0)

    def test_negative_timestamp_raises(self, tmp_path):
        dummy_file = tmp_path / "dummy.mp4"
        dummy_file.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="must be non-negative"):
            read_keyframe_at_timestamp(dummy_file, -0.5)

    def test_real_binary_search_artifact_smoke_test(self):
        """Smoke test on the accepted M5 artifact without modifying it."""
        artifact_path = Path("CASES/artifacts/m5_3_4_2_binary_search_muxed.mp4")
        if not artifact_path.exists():
            pytest.skip("Accepted M5 artifact not present in workspace.")

        frame, meta = read_keyframe_at_timestamp(artifact_path, 2.0)
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert meta["width"] == 854
        assert meta["height"] == 480
        assert meta["fps"] == 15.0
        assert meta["duration_sec"] > 5.0

        # Run blank detection on sampled frame
        is_blank, ev_b, diag_b = detect_blank_canvas(frame)
        assert is_blank is False
        assert ev_b.std_luminance > 10.0

        # Run perimeter clipping on sampled frame
        is_clip, ev_c, diag_c = detect_perimeter_clipping(frame)
        assert isinstance(is_clip, bool)

        # Ensure capture is cleanly released (verified by opening again)
        frame2, _ = read_keyframe_at_timestamp(artifact_path, 4.0)
        assert frame2 is not None
