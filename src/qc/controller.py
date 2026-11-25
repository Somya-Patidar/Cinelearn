"""
Milestone 6.3: Rendered Video Quality Controller.

Provides deterministic post-render quality control (Layer C) integration for the
CineLearn modern pipeline. Coordinates event-driven keyframe sampling, spatial
projection, OpenCV frame extraction, raster vision analyzers, diagnostic aggregation,
and fail-closed checkpoint reporting.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.compiler.compiler_models import ScheduledTimeline
from src.compiler.layout_engine import SceneSpatialLayout
from src.compiler.spatial_ledger import SpatialLedger
from src.ir.models import SceneIR, SceneSequenceIR
from src.qc.keyframe_sampler import (
    KeyframeSampler,
    KeyframeSamplerConfig,
    KeyframeSampleRequest,
    KeyframeSampleResult,
    KeyframeSamplingPlan,
    SamplePhase,
    TimestampEvidence,
)
from src.qc.models import (
    BlankCanvasEvidence,
    ContrastEvidence,
    OverlapEvidence,
    PerimeterClippingEvidence,
    PixelBox,
    QCDiagnostic,
    QCDiagnosticCode,
    QCDiagnosticLevel,
    QCLayer,
    QCReport,
    QCReportStatus,
    RasterAnalyzerConfig,
    SubtitleCollisionEvidence,
)
from src.qc import raster_analyzer
from src.qc.spatial_projector import (
    GeometricOverlapRecord,
    ProjectedEntityGeometry,
    ProjectedSpatialLayout,
    ProjectionConfig,
    SpatialCoordinateProjector,
    SpatialProjectionResult,
    parse_aspect_ratio_to_float,
)


class VideoQCConfig(BaseModel):
    """Configuration options for RenderedVideoQualityController."""
    model_config = ConfigDict(extra="forbid")

    raster_config: RasterAnalyzerConfig = Field(default_factory=RasterAnalyzerConfig)
    sampler_config: KeyframeSamplerConfig = Field(default_factory=KeyframeSamplerConfig)
    max_aspect_ratio_tolerance: float = Field(default=0.02, ge=0.0)
    enforce_strict_aspect_ratio: bool = True
    check_blank_canvas: bool = True
    check_perimeter_clipping: bool = True
    check_subtitle_clearance: bool = True
    check_contrast: bool = True
    check_raster_overlap: bool = True


class RenderedVideoQualityController:
    """
    Authoritative controller for deterministic Rendered-Video Quality Control (Layer C).
    Coordinates temporal sampling, spatial coordinate projection, video frame decoding,
    and raster vision inspection with fail-closed checkpoint mechanics.
    """

    def __init__(self, config: Optional[VideoQCConfig] = None):
        self.config = config or VideoQCConfig()
        self.sampler = KeyframeSampler(config=self.config.sampler_config)

    def evaluate_scene_video(
        self,
        video_path: Union[str, Path],
        scene_ir: SceneIR,
        scheduled_timeline: ScheduledTimeline,
        layout: Optional[SceneSpatialLayout] = None,
        spatial_ledger: Optional[SpatialLedger] = None,
        burn_subtitles: bool = True,
        raw_video_path: Optional[Union[str, Path]] = None,
        exclusion_mask: Optional[np.ndarray] = None,
    ) -> QCReport:
        """
        Executes complete deterministic Layer C quality control for a rendered scene video.
        
        Evaluates:
        1. Media file existence, non-emptiness, and accessibility.
        2. Finite scene duration and valid event-driven keyframe sampling plan.
        3. Keyframe decoding with strict decoder timestamp metadata verification.
        4. Aspect-ratio conformance between render resolution and scene layout.
        5. Raster vision checks: blank canvas, perimeter clipping, subtitle collision.
        6. Spatial projective checks: WCAG contrast on text and raster overlap on intersecting entities.
        """
        scene_id = scene_ir.scene_id
        diagnostics: List[QCDiagnostic] = []
        vpath = Path(video_path)

        # ------------------------------------------------------------------
        # 1. Structural & File Prerequisites Check
        # ------------------------------------------------------------------
        if not vpath.exists():
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                level=QCDiagnosticLevel.FATAL,
                layer=QCLayer.LAYER_C,
                message=f"Rendered video file does not exist: {vpath}",
                scene_id=scene_id,
            )
            return QCReport(
                scene_id=scene_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=[diag],
                raster_metrics={"error": "file_not_found", "path": str(vpath)},
            )

        if vpath.stat().st_size == 0:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                level=QCDiagnosticLevel.FATAL,
                layer=QCLayer.LAYER_C,
                message=f"Rendered video file is empty (0 bytes): {vpath}",
                scene_id=scene_id,
            )
            return QCReport(
                scene_id=scene_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=[diag],
                raster_metrics={"error": "file_empty", "path": str(vpath)},
            )

        # Duration validation
        dur_val = scene_ir.duration_budget if scene_ir.duration_budget else scheduled_timeline.total_duration
        if dur_val is None or isinstance(dur_val, bool) or not isinstance(dur_val, (int, float)):
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_TIMELINE_BUDGET_OVERRUN,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_B,
                message=f"Invalid or non-numeric scene duration: {dur_val!r}",
                scene_id=scene_id,
            )
            return QCReport(
                scene_id=scene_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=[diag],
                raster_metrics={"error": "invalid_duration"},
            )

        scene_duration = float(dur_val)
        if math.isnan(scene_duration) or math.isinf(scene_duration) or scene_duration <= 0.0:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_TIMELINE_BUDGET_OVERRUN,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_B,
                message=f"Scene duration must be positive and finite, got {scene_duration}",
                scene_id=scene_id,
            )
            return QCReport(
                scene_id=scene_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=[diag],
                raster_metrics={"error": "non_finite_duration"},
            )

        # ------------------------------------------------------------------
        # 2. Event-Driven Keyframe Sampling Plan
        # ------------------------------------------------------------------
        try:
            sampling_plan = self.sampler.sample_timeline(scheduled_timeline, scene_duration=scene_duration)
            diagnostics.extend(sampling_plan.diagnostics)
        except Exception as e:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_TIMELINE_BUDGET_OVERRUN,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_B,
                message=f"Failed to generate keyframe sampling plan: {e}",
                scene_id=scene_id,
            )
            return QCReport(
                scene_id=scene_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=[diag] + diagnostics,
                raster_metrics={"error": "sampling_plan_generation_failed"},
            )

        if not sampling_plan.sample_requests:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_TIMELINE_BUDGET_OVERRUN,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_B,
                message=f"Keyframe sampling plan produced zero sample requests for scene {scene_id}",
                scene_id=scene_id,
            )
            return QCReport(
                scene_id=scene_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=[diag] + diagnostics,
                raster_metrics={"error": "empty_sampling_plan"},
            )

        # ------------------------------------------------------------------
        # 3. Keyframe Extraction & Decoding
        # ------------------------------------------------------------------
        extraction_results = self.sampler.extract_keyframes(str(vpath), sampling_plan)
        for res in extraction_results:
            diagnostics.extend(res.diagnostics)

        successful_extractions = [r for r in extraction_results if r.frame_extracted]
        failed_extractions = [r for r in extraction_results if not r.frame_extracted]

        if not successful_extractions:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_C,
                message=(
                    f"All {len(extraction_results)} keyframe extractions failed for scene '{scene_id}'."
                ),
                scene_id=scene_id,
            )
            diagnostics.append(diag)
            return QCReport(
                scene_id=scene_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=diagnostics,
                raster_metrics={
                    "total_requests": len(extraction_results),
                    "successful_frames": 0,
                    "failed_frames": len(failed_extractions),
                },
            )

        # ------------------------------------------------------------------
        # 4. Aspect-Ratio Matching & Resolution Validation
        # ------------------------------------------------------------------
        first_res = successful_extractions[0]
        frame_h, frame_w = first_res.frame_shape[0], first_res.frame_shape[1]
        ar_str = scene_ir.layout_configuration.aspect_ratio if scene_ir.layout_configuration else "16:9"
        try:
            expected_ar = parse_aspect_ratio_to_float(ar_str)
        except Exception:
            expected_ar = 16.0 / 9.0

        if frame_h <= 0 or frame_w <= 0:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_C,
                message=f"Invalid decoded frame dimensions: width={frame_w}, height={frame_h}",
                scene_id=scene_id,
            )
            diagnostics.append(diag)
        else:
            actual_ar = frame_w / frame_h
            ar_diff = abs(actual_ar - expected_ar)
            if ar_diff > self.config.max_aspect_ratio_tolerance:
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_C,
                    message=(
                        f"Aspect-ratio mismatch: Render resolution {frame_w}x{frame_h} "
                        f"(aspect ratio {actual_ar:.4f}) does not match scene layout "
                        f"aspect ratio '{ar_str}' ({expected_ar:.4f}, delta={ar_diff:.4f} > {self.config.max_aspect_ratio_tolerance})."
                    ),
                    scene_id=scene_id,
                    evidence={
                        "frame_width": frame_w,
                        "frame_height": frame_h,
                        "actual_aspect_ratio": round(actual_ar, 4),
                        "expected_aspect_ratio": round(expected_ar, 4),
                        "layout_aspect_ratio_str": ar_str,
                        "delta": round(ar_diff, 4),
                    },
                )
                diagnostics.append(diag)

        # ------------------------------------------------------------------
        # 5. Spatial Projector Initialization
        # ------------------------------------------------------------------
        proj_config = ProjectionConfig(
            frame_width=frame_w,
            frame_height=frame_h,
            aspect_ratio=ar_str,
        )
        projector = SpatialCoordinateProjector(config=proj_config)

        # ------------------------------------------------------------------
        # 6. Per-Keyframe Deterministic Raster Analysis
        # ------------------------------------------------------------------
        for res in successful_extractions:
            req = res.request
            ts = req.requested_timestamp

            try:
                frame, meta = raster_analyzer.read_keyframe_at_timestamp(vpath, ts)
                frame = raster_analyzer.validate_frame_array(frame)
            except Exception as e:
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_C,
                    message=f"Failed to read/validate frame at t={ts:.3f}s: {e}",
                    scene_id=scene_id,
                    timestamp=ts,
                )
                diagnostics.append(diag)
                continue

            try:
                # 6.1 Blank Canvas Detection
                if self.config.check_blank_canvas:
                    is_blank, blank_ev, blank_diag = raster_analyzer.detect_blank_canvas(
                        frame,
                        config=self.config.raster_config,
                        scene_id=scene_id,
                        timestamp=ts,
                    )
                    if blank_diag:
                        diagnostics.append(blank_diag)

                # 6.2 Perimeter Clipping Detection
                if self.config.check_perimeter_clipping:
                    ignore_edges = ["bottom"] if burn_subtitles else []
                    is_clipped, clip_evs, clip_diags = raster_analyzer.detect_perimeter_clipping(
                        frame,
                        config=self.config.raster_config,
                        scene_id=scene_id,
                        timestamp=ts,
                        ignore_edges=ignore_edges,
                    )
                    diagnostics.extend(clip_diags)

                # 6.3 Subtitle Safe-Zone Clearance Inspection
                if (
                    self.config.check_subtitle_clearance
                    and scene_ir.layout_configuration
                    and scene_ir.layout_configuration.has_subtitles
                ):
                    target_sub_frame = frame
                    # FIND-6.3-01: Prefer inspecting pre-mux raw video to avoid false collisions from burned subtitles
                    if raw_video_path:
                        raw_p = Path(raw_video_path)
                        if raw_p.exists() and raw_p.resolve() != vpath.resolve():
                            try:
                                raw_f, _ = raster_analyzer.read_keyframe_at_timestamp(raw_p, ts)
                                target_sub_frame = raster_analyzer.validate_frame_array(raw_f)
                            except Exception as e:
                                diag = QCDiagnostic(
                                    code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                                    level=QCDiagnosticLevel.WARNING,
                                    layer=QCLayer.LAYER_C,
                                    message=f"Failed to read pre-mux raw video frame at t={ts:.3f}s for subtitle clearance: {e}",
                                    scene_id=scene_id,
                                    timestamp=ts,
                                )
                                diagnostics.append(diag)
                                target_sub_frame = None

                    if target_sub_frame is not None:
                        is_colliding, sub_ev, sub_diag = raster_analyzer.detect_subtitle_collision(
                            target_sub_frame,
                            config=self.config.raster_config,
                            scene_id=scene_id,
                            timestamp=ts,
                            exclusion_mask=exclusion_mask if (target_sub_frame is frame and burn_subtitles) else None,
                        )
                        if sub_diag:
                            diagnostics.append(sub_diag)

                # 6.4 Projective Spatial Layout & Contrast / Overlap Inspection
                if layout is not None and layout.is_valid:
                    proj_res = projector.project_layout_at_time(
                        layout,
                        timestamp=ts,
                        ledger=spatial_ledger,
                    )
                    diagnostics.extend(proj_res.diagnostics)
                    p_layout = proj_res.layout

                    # 6.4.1 Contrast evaluation on active text/label entities
                    if self.config.check_contrast:
                        for eid, p_geom in p_layout.projected_entities.items():
                            ent_decl = next((e for e in scene_ir.entities if e.id == eid), None)
                            is_text = (
                                p_geom.layer_role in ("label", "text", "metric", "title", "caption")
                                or (ent_decl and ent_decl.component_type in ("text", "title", "label", "caption", "metric", "code_block"))
                            )
                            if is_text and p_geom.pixel_box:
                                roi_tuple = (
                                    p_geom.pixel_box.x_min,
                                    p_geom.pixel_box.y_min,
                                    p_geom.pixel_box.x_max,
                                    p_geom.pixel_box.y_max,
                                )
                                ratio, c_ev, c_diag = raster_analyzer.evaluate_roi_contrast(
                                    frame=frame,
                                    roi_box=roi_tuple,
                                    config=self.config.raster_config,
                                    scene_id=scene_id,
                                    timestamp=ts,
                                )
                                if c_diag:
                                    c_diag.entity_id = eid
                                    diagnostics.append(c_diag)

                    # 6.4.2 Raster overlap evaluation on active intersecting entity pairs
                    if self.config.check_raster_overlap:
                        for overlap in p_layout.geometric_overlaps:
                            box_a = p_layout.get_pixel_box(overlap.entity_a)
                            box_b = p_layout.get_pixel_box(overlap.entity_b)
                            if box_a and box_b:
                                is_overlap, ol_ev, ol_diag = raster_analyzer.evaluate_roi_overlap(
                                    box_a=(box_a.x_min, box_a.y_min, box_a.x_max, box_a.y_max),
                                    box_b=(box_b.x_min, box_b.y_min, box_b.x_max, box_b.y_max),
                                    frame=frame,
                                    config=self.config.raster_config,
                                    scene_id=scene_id,
                                    timestamp=ts,
                                )
                                if ol_diag:
                                    diagnostics.append(ol_diag)

            except Exception as e:
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_C,
                    message=f"Raster vision analysis failed unexpectedly at t={ts:.3f}s: {e}",
                    scene_id=scene_id,
                    timestamp=ts,
                )
                diagnostics.append(diag)

        # ------------------------------------------------------------------
        # 7. Aggregate QC Report & Status Determination
        # ------------------------------------------------------------------
        has_errors = any(d.level in (QCDiagnosticLevel.ERROR, QCDiagnosticLevel.FATAL) for d in diagnostics)
        has_warnings = any(d.level == QCDiagnosticLevel.WARNING for d in diagnostics)

        if has_errors:
            overall_status = QCReportStatus.FAILED
            is_valid = False
        elif has_warnings:
            overall_status = QCReportStatus.WARNINGS
            is_valid = True
        else:
            overall_status = QCReportStatus.PASSED
            is_valid = True

        report = QCReport(
            scene_id=scene_id,
            overall_status=overall_status,
            is_valid=is_valid,
            sampled_frames_count=len(successful_extractions),
            diagnostics=diagnostics,
            raster_metrics={
                "total_requests": len(extraction_results),
                "successful_frames": len(successful_extractions),
                "failed_frames": len(failed_extractions),
                "resolution": [frame_w, frame_h] if successful_extractions else None,
            },
        )
        return report

    def evaluate_master_video(
        self,
        video_path: Union[str, Path],
        sequence_ir: SceneSequenceIR,
    ) -> QCReport:
        """
        Executes Layer C quality control on the assembled master video artifact.
        Verifies container integrity, non-zero duration, samples representative keyframes
        across all sections in sequence_ir, and detects blank or corrupted visual frames.
        """
        seq_id = sequence_ir.sequence_id
        diagnostics: List[QCDiagnostic] = []
        vpath = Path(video_path)

        if not vpath.exists():
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                level=QCDiagnosticLevel.FATAL,
                layer=QCLayer.LAYER_C,
                message=f"Master video file does not exist: {vpath}",
                scene_id=seq_id,
            )
            return QCReport(
                scene_id=seq_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=[diag],
            )

        if vpath.stat().st_size == 0:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                level=QCDiagnosticLevel.FATAL,
                layer=QCLayer.LAYER_C,
                message=f"Master video file is empty (0 bytes): {vpath}",
                scene_id=seq_id,
            )
            return QCReport(
                scene_id=seq_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=[diag],
            )

        # Probe initial frame to obtain video duration and ensure decodability
        try:
            init_frame, init_meta = raster_analyzer.read_keyframe_at_timestamp(vpath, 0.0)
            init_frame = raster_analyzer.validate_frame_array(init_frame)
            video_duration = float(init_meta.get("duration_sec", 0.0))
        except Exception as e:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                level=QCDiagnosticLevel.FATAL,
                layer=QCLayer.LAYER_C,
                message=f"Master video initial frame decode failed: {e}",
                scene_id=seq_id,
                timestamp=0.0,
            )
            return QCReport(
                scene_id=seq_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=[diag],
            )

        if video_duration <= 0.0 or math.isnan(video_duration) or math.isinf(video_duration):
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_C,
                message=f"Master video reported invalid or non-finite duration: {video_duration}",
                scene_id=seq_id,
            )
            return QCReport(
                scene_id=seq_id,
                overall_status=QCReportStatus.FAILED,
                is_valid=False,
                sampled_frames_count=0,
                diagnostics=[diag],
            )

        # Construct multi-section sampling timestamps across all sections
        sections = sequence_ir.sections or []
        sample_targets: List[Tuple[str, float]] = []

        has_valid_budgets = (
            bool(sections)
            and all(
                isinstance(s.duration_budget, (int, float))
                and not isinstance(s.duration_budget, bool)
                and s.duration_budget > 0
                for s in sections
            )
        )

        if sections and has_valid_budgets:
            accum = 0.0
            for sec in sections:
                dur = float(sec.duration_budget)
                target_ts = accum + 0.5 * dur
                safe_ts = max(0.0, min(target_ts, max(0.0, video_duration - 0.05)))
                sample_targets.append((sec.scene_id, round(safe_ts, 3)))
                accum += dur
        elif sections:
            sec_len = video_duration / len(sections)
            for idx, sec in enumerate(sections):
                target_ts = (idx + 0.5) * sec_len
                safe_ts = max(0.0, min(target_ts, max(0.0, video_duration - 0.05)))
                sample_targets.append((sec.scene_id, round(safe_ts, 3)))
        else:
            safe_ts = max(0.0, min(1.0, max(0.0, video_duration - 0.05)))
            sample_targets.append((seq_id, round(safe_ts, 3)))

        # Deduplicate timestamps while preserving order
        seen_ts = set()
        dedup_targets: List[Tuple[str, float]] = []
        for sid, ts in sample_targets:
            if ts not in seen_ts:
                seen_ts.add(ts)
                dedup_targets.append((sid, ts))

        sampled_count = 0
        for target_scene_id, ts in dedup_targets:
            try:
                frame, meta = raster_analyzer.read_keyframe_at_timestamp(vpath, ts)
                frame = raster_analyzer.validate_frame_array(frame)
                sampled_count += 1
            except Exception as e:
                diag = QCDiagnostic(
                    code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                    level=QCDiagnosticLevel.ERROR,
                    layer=QCLayer.LAYER_C,
                    message=f"Master video frame decode failed for section '{target_scene_id}' at t={ts:.3f}s: {e}",
                    scene_id=target_scene_id,
                    timestamp=ts,
                )
                diagnostics.append(diag)
                continue

            # Raster check: Blank canvas detection on master frame
            if self.config.check_blank_canvas:
                try:
                    is_blank, blank_ev, blank_diag = raster_analyzer.detect_blank_canvas(
                        frame,
                        config=self.config.raster_config,
                        scene_id=target_scene_id,
                        timestamp=ts,
                    )
                    if blank_diag:
                        diagnostics.append(blank_diag)
                except Exception as e:
                    diag = QCDiagnostic(
                        code=QCDiagnosticCode.QC_FRAME_CORRUPTION,
                        level=QCDiagnosticLevel.ERROR,
                        layer=QCLayer.LAYER_C,
                        message=f"Blank canvas check failed unexpectedly on master video at t={ts:.3f}s: {e}",
                        scene_id=target_scene_id,
                        timestamp=ts,
                    )
                    diagnostics.append(diag)

        has_errors = any(d.level in (QCDiagnosticLevel.ERROR, QCDiagnosticLevel.FATAL) for d in diagnostics)
        has_warnings = any(d.level == QCDiagnosticLevel.WARNING for d in diagnostics)

        if has_errors:
            overall_status = QCReportStatus.FAILED
            is_valid = False
        elif has_warnings:
            overall_status = QCReportStatus.WARNINGS
            is_valid = True
        else:
            overall_status = QCReportStatus.PASSED
            is_valid = True

        return QCReport(
            scene_id=seq_id,
            overall_status=overall_status,
            is_valid=is_valid,
            sampled_frames_count=sampled_count,
            diagnostics=diagnostics,
            raster_metrics={
                "total_targets": len(dedup_targets),
                "sampled_frames": sampled_count,
                "video_duration": video_duration,
            },
        )
