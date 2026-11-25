"""
Milestone 6.1: Deterministic Raster Vision Analyzer.

Provides pixel-level and region-of-interest (ROI) computer vision analysis
for rendered video keyframes using OpenCV, NumPy, and PIL:
- Frame validation and in-memory extraction via OpenCV.
- Blank and corrupted canvas detection (intensity variance, entropy, coverage).
- Perimeter clipping detection (outer border band energy).
- Subtitle band clearance and collision inspection.
- Relative luminance and WCAG 2.1 contrast ratio evaluation.
- Pairwise geometric and visible raster ROI overlap analysis.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.qc.models import (
    QCDiagnosticLevel,
    QCLayer,
    QCDiagnosticCode,
    PixelBox,
    BlankCanvasEvidence,
    PerimeterClippingEvidence,
    SubtitleCollisionEvidence,
    ContrastEvidence,
    OverlapEvidence,
    QCDiagnostic,
    RasterAnalyzerConfig,
)


def validate_frame_array(frame: Any) -> np.ndarray:
    """
    Validates that the input is a valid 2D (grayscale) or 3D (BGR/RGB/BGRA) NumPy image array.
    Normalizes float inputs in [0.0, 1.0] to uint8 [0, 255].
    Raises ValueError or TypeError for invalid/empty inputs.
    """
    if frame is None:
        raise ValueError("Input frame cannot be None.")
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(frame).__name__}.")
    if frame.size == 0:
        raise ValueError("Input frame array is empty (size 0).")
    if frame.ndim not in (2, 3):
        raise ValueError(f"Input frame must have 2 or 3 dimensions, got {frame.ndim} (shape: {frame.shape}).")

    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Frame dimensions must be positive, got height={h}, width={w}.")

    if frame.ndim == 3:
        c = frame.shape[2]
        if c not in (1, 3, 4):
            raise ValueError(f"3D frame must have 1, 3, or 4 channels, got {c}.")

    # Normalize float image to uint8 if needed
    if np.issubdtype(frame.dtype, np.floating):
        if np.max(frame) <= 1.0 and np.min(frame) >= 0.0:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    elif frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.integer):
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported frame array dtype: {frame.dtype}.")

    return frame


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Converts a validated 2D or 3D image array to a 2D grayscale uint8 array."""
    valid_frame = validate_frame_array(frame)
    if valid_frame.ndim == 2:
        return valid_frame
    c = valid_frame.shape[2]
    if c == 1:
        return valid_frame[:, :, 0]
    elif c == 3:
        return cv2.cvtColor(valid_frame, cv2.COLOR_BGR2GRAY)
    elif c == 4:
        return cv2.cvtColor(valid_frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Cannot convert {c}-channel frame to grayscale.")


def estimate_background_color(frame: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Estimates dominant background color from four corner patches of the frame.
    Returns estimated background color as a 1D float array matching channel count.
    """
    valid_frame = validate_frame_array(frame)
    h, w = valid_frame.shape[:2]
    ps_y = min(patch_size, max(2, h // 10))
    ps_x = min(patch_size, max(2, w // 10))

    if valid_frame.ndim == 2:
        corners = [
            valid_frame[:ps_y, :ps_x],
            valid_frame[:ps_y, w - ps_x:],
            valid_frame[h - ps_y:, :ps_x],
            valid_frame[h - ps_y:, w - ps_x:],
        ]
        sampled = np.concatenate([c.ravel() for c in corners])
        return np.array([np.median(sampled)], dtype=np.float32)
    else:
        corners = [
            valid_frame[:ps_y, :ps_x, :],
            valid_frame[:ps_y, w - ps_x:, :],
            valid_frame[h - ps_y:, :ps_x, :],
            valid_frame[h - ps_y:, w - ps_x:, :],
        ]
        c = valid_frame.shape[2]
        sampled = np.concatenate([c_patch.reshape(-1, c) for c_patch in corners], axis=0)
        return np.median(sampled, axis=0)


def read_keyframe_at_timestamp(
    video_path: Union[str, Path],
    timestamp: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extracts a single BGR video frame at the requested timestamp (seconds) using OpenCV.
    Ensures safe resource release in all scenarios via try-finally.

    Note on Seeking Behavior:
    OpenCV POS_MSEC seeks to the nearest keyframe / decoded frame depending on codec
    GOP structure. The returned actual_timestamp_msec records the empirical position.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if vpath.stat().st_size == 0:
        raise ValueError(f"Video file is empty (0 bytes): {video_path}")
    if timestamp < 0.0:
        raise ValueError(f"Timestamp must be non-negative, got {timestamp}.")

    cap = cv2.VideoCapture(str(vpath))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = total_frames / fps if fps > 0.0 else 0.0

        target_msec = timestamp * 1000.0
        cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
        ret, frame = cap.read()

        if not ret or frame is None:
            raise RuntimeError(
                f"Failed to decode frame from '{video_path}' at t={timestamp:.3f}s "
                f"(total frames: {total_frames}, duration: {duration_sec:.2f}s)."
            )

        actual_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        metadata = {
            "video_path": str(vpath),
            "requested_timestamp_sec": timestamp,
            "actual_timestamp_sec": round(actual_msec / 1000.0, 4),
            "width": w,
            "height": h,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 3),
        }
        return frame, metadata
    finally:
        cap.release()


def detect_blank_canvas(
    frame: np.ndarray,
    config: Optional[RasterAnalyzerConfig] = None,
    scene_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> Tuple[bool, BlankCanvasEvidence, Optional[QCDiagnostic]]:
    """
    Detects whether a frame is an unrendered or blank canvas by evaluating:
    1. Standard deviation of pixel intensities.
    2. Shannon histogram entropy.
    3. Foreground pixel coverage relative to estimated background.

    Distinguishes intentionally dark or minimal title frames from blank canvases
    by checking for localized edge energy.
    """
    cfg = config or RasterAnalyzerConfig()
    gray = to_grayscale(frame)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Calculate histogram entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total_pixels = float(gray.size)
    probs = hist[hist > 0] / total_pixels
    entropy = float(-np.sum(probs * np.log2(probs))) if len(probs) > 0 else 0.0

    # Estimate background & foreground coverage
    if bg_color is not None:
        bg_lum = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    else:
        bg_est = estimate_background_color(gray)
        bg_lum = float(bg_est[0])

    diff = np.abs(gray.astype(np.float32) - bg_lum)
    fg_mask = diff > 15.0
    fg_coverage = float(np.count_nonzero(fg_mask) / total_pixels)

    # Edge detection to protect intentionally simple text cards
    edges = cv2.Canny(gray, 50, 150)
    edge_count = int(np.count_nonzero(edges))

    # Blank canvas determination
    # Blank if std and entropy are below threshold, OR coverage is near zero with low std,
    # UNLESS localized edge structures prove intentional visual text/diagram presence.
    has_localized_edges = edge_count >= 30 and entropy >= 0.20
    is_blank = False

    if std_val < cfg.blank_std_threshold and entropy < cfg.blank_entropy_threshold:
        is_blank = not has_localized_edges
    elif fg_coverage < cfg.min_fg_coverage and std_val < (cfg.blank_std_threshold * 1.5):
        is_blank = not has_localized_edges

    evidence = BlankCanvasEvidence(
        mean_luminance=round(mean_val, 2),
        std_luminance=round(std_val, 3),
        entropy=round(entropy, 4),
        foreground_coverage=round(fg_coverage, 4),
        is_blank=is_blank,
        std_threshold=cfg.blank_std_threshold,
        entropy_threshold=cfg.blank_entropy_threshold,
    )

    diagnostic = None
    if is_blank:
        diagnostic = QCDiagnostic(
            code=QCDiagnosticCode.QC_BLANK_CANVAS,
            level=QCDiagnosticLevel.ERROR,
            layer=QCLayer.LAYER_C,
            message=f"Blank or unrendered canvas detected (std={std_val:.2f}, entropy={entropy:.3f}, fg={fg_coverage:.3%}).",
            scene_id=scene_id,
            timestamp=timestamp,
            evidence=evidence.model_dump(),
        )

    return is_blank, evidence, diagnostic


def detect_perimeter_clipping(
    frame: np.ndarray,
    config: Optional[RasterAnalyzerConfig] = None,
    scene_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    ignore_edges: Optional[List[str]] = None,
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> Tuple[bool, List[PerimeterClippingEvidence], List[QCDiagnostic]]:
    """
    Inspects outer border bands of the frame (left, right, top, bottom) for foreground
    activity and edge gradients touching screen perimeters.

    Note: Pixel activity in perimeter bands is empirical evidence of margin breach;
    it indicates potential clipping rather than semantic certainty.
    """
    cfg = config or RasterAnalyzerConfig()
    gray = to_grayscale(frame)
    h, w = gray.shape[:2]
    ignored = set(ignore_edges or [])

    band_h = max(2, int(h * cfg.border_ratio))
    band_w = max(2, int(w * cfg.border_ratio))

    if bg_color is not None:
        bg_lum = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    else:
        bg_est = estimate_background_color(gray)
        bg_lum = float(bg_est[0])

    edge_slices = {
        "left": (slice(0, h), slice(0, band_w), band_w),
        "right": (slice(0, h), slice(w - band_w, w), band_w),
        "top": (slice(0, band_h), slice(0, w), band_h),
        "bottom": (slice(h - band_h, h), slice(0, w), band_h),
    }

    evidences: List[PerimeterClippingEvidence] = []
    diagnostics: List[QCDiagnostic] = []
    any_clipped = False

    for edge_name, (s_y, s_x, band_px) in edge_slices.items():
        if edge_name in ignored:
            continue

        band_patch = gray[s_y, s_x]
        diff = np.abs(band_patch.astype(np.float32) - bg_lum)
        fg_pixels = np.count_nonzero(diff > 18.0)
        fg_density = float(fg_pixels / band_patch.size)

        # Gradient check along border: sharp boundary cutting through border band
        sobel_x = cv2.Sobel(band_patch, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(band_patch, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge_energy = float(np.mean(grad_mag > 30.0))

        # Flag clipping if foreground density in the border band exceeds threshold
        is_edge_clipped = fg_density > cfg.border_max_density
        if is_edge_clipped:
            any_clipped = True

        ev = PerimeterClippingEvidence(
            edge=edge_name,  # type: ignore[arg-type]
            band_thickness_pixels=band_px,
            edge_pixel_density=round(fg_density, 4),
            threshold=cfg.border_max_density,
            is_clipped=is_edge_clipped,
        )
        evidences.append(ev)

        if is_edge_clipped:
            diag = QCDiagnostic(
                code=QCDiagnosticCode.QC_PERIMETER_CLIPPING,
                level=QCDiagnosticLevel.ERROR,
                layer=QCLayer.LAYER_C,
                message=f"Perimeter clipping detected on {edge_name} edge (density={fg_density:.2%}, threshold={cfg.border_max_density:.2%}).",
                scene_id=scene_id,
                timestamp=timestamp,
                evidence=ev.model_dump(),
            )
            diagnostics.append(diag)

    return any_clipped, evidences, diagnostics


def detect_subtitle_collision(
    frame: np.ndarray,
    config: Optional[RasterAnalyzerConfig] = None,
    scene_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    bg_color: Optional[Tuple[int, int, int]] = None,
    exclusion_mask: Optional[np.ndarray] = None,
) -> Tuple[bool, SubtitleCollisionEvidence, Optional[QCDiagnostic]]:
    """
    Inspects the designated subtitle safe band (default lower 15% of frame) for unexpected
    canvas visual intrusion that could collide with burned or rendered subtitles.
    """
    cfg = config or RasterAnalyzerConfig()
    gray = to_grayscale(frame)
    h, w = gray.shape[:2]

    y_min = int(h * cfg.subtitle_band_y_min_ratio)
    y_max = int(h * cfg.subtitle_band_y_max_ratio)
    x_min = int(w * 0.10)
    x_max = int(w * 0.90)

    sub_patch = gray[y_min:y_max, x_min:x_max]

    if bg_color is not None:
        bg_lum = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    else:
        bg_est = estimate_background_color(gray)
        bg_lum = float(bg_est[0])

    diff = np.abs(sub_patch.astype(np.float32) - bg_lum)
    fg_mask = diff > 18.0

    if exclusion_mask is not None:
        ex_patch = exclusion_mask[y_min:y_max, x_min:x_max]
        fg_mask = fg_mask & (~ex_patch)

    intrusion_density = float(np.count_nonzero(fg_mask) / sub_patch.size)
    is_colliding = intrusion_density > cfg.subtitle_max_density

    evidence = SubtitleCollisionEvidence(
        subtitle_band_y_min=y_min,
        subtitle_band_y_max=y_max,
        collision_pixel_density=round(intrusion_density, 4),
        threshold=cfg.subtitle_max_density,
        is_colliding=is_colliding,
    )

    diagnostic = None
    if is_colliding:
        diagnostic = QCDiagnostic(
            code=QCDiagnosticCode.QC_SUBTITLE_COLLISION,
            level=QCDiagnosticLevel.WARNING,
            layer=QCLayer.LAYER_C,
            message=f"Visual content intrudes into subtitle safe band (density={intrusion_density:.2%}, threshold={cfg.subtitle_max_density:.2%}).",
            scene_id=scene_id,
            timestamp=timestamp,
            evidence=evidence.model_dump(),
        )

    return is_colliding, evidence, diagnostic


def _srgb_to_linear(c_norm: float) -> float:
    """Converts a normalized sRGB channel value [0, 1] to linear luminance."""
    if c_norm <= 0.04045:
        return c_norm / 12.92
    return ((c_norm + 0.055) / 1.055) ** 2.4


def calculate_relative_luminance_bgr(bgr: np.ndarray) -> float:
    """Calculates relative luminance L in [0.0, 1.0] from a BGR array using WCAG 2.1."""
    b, g, r = [float(x) / 255.0 for x in bgr[:3]]
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)
    return float(0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin)


def evaluate_roi_contrast(
    frame: np.ndarray,
    roi_box: Optional[Tuple[int, int, int, int]] = None,
    fg_mask: Optional[np.ndarray] = None,
    bg_mask: Optional[np.ndarray] = None,
    is_large_text: bool = False,
    config: Optional[RasterAnalyzerConfig] = None,
    scene_id: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> Tuple[float, ContrastEvidence, Optional[QCDiagnostic]]:
    """
    Evaluates relative luminance contrast ratio according to WCAG 2.1 formulas.
    If explicit fg_mask/bg_mask are supplied, computes exact contrast between them.
    Otherwise, uses bimodal Otsu clustering on the ROI grayscale. If the ROI histogram
    is unimodal or poorly separated, returns an inconclusive result rather than a fabricated score.
    """
    cfg = config or RasterAnalyzerConfig()
    valid_frame = validate_frame_array(frame)
    required_ratio = cfg.large_text_contrast_threshold if is_large_text else cfg.normal_text_contrast_threshold

    if roi_box is not None:
        x0, y0, x1, y1 = roi_box
        h, w = valid_frame.shape[:2]
        x0 = max(0, min(x0, w - 1))
        x1 = max(1, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(1, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid ROI crop bounds: [{x0}, {y0}, {x1}, {y1}] on frame size {w}x{h}.")
        patch = valid_frame[y0:y1, x0:x1]
    else:
        patch = valid_frame

    gray_patch = to_grayscale(patch)

    is_conclusive = True
    method = "explicit_masks" if (fg_mask is not None and bg_mask is not None) else "bimodal_otsu"

    if fg_mask is not None and bg_mask is not None:
        if roi_box is not None:
            sub_fg = fg_mask[y0:y1, x0:x1]
            sub_bg = bg_mask[y0:y1, x0:x1]
        else:
            sub_fg = fg_mask
            sub_bg = bg_mask

        if np.count_nonzero(sub_fg) == 0 or np.count_nonzero(sub_bg) == 0:
            is_conclusive = False
            l_fg = 0.5
            l_bg = 0.5
        else:
            if patch.ndim == 3 and patch.shape[2] >= 3:
                fg_bgr = np.mean(patch[sub_fg], axis=0)
                bg_bgr = np.mean(patch[sub_bg], axis=0)
                l_fg = calculate_relative_luminance_bgr(fg_bgr)
                l_bg = calculate_relative_luminance_bgr(bg_bgr)
            else:
                l_fg = _srgb_to_linear(float(np.mean(patch[sub_fg])) / 255.0)
                l_bg = _srgb_to_linear(float(np.mean(patch[sub_bg])) / 255.0)
    else:
        # Bimodal Otsu clustering to separate text foreground and background
        thresh_val, otsu_mask = cv2.threshold(gray_patch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        class_1 = otsu_mask > 0
        class_0 = ~class_1
        c1_count = np.count_nonzero(class_1)
        c0_count = np.count_nonzero(class_0)

        # Check bimodal separation confidence: if one class has < 1% pixels or std is tiny
        if c1_count < (gray_patch.size * 0.01) or c0_count < (gray_patch.size * 0.01) or np.std(gray_patch) < 3.0:
            is_conclusive = False
            l_fg = 0.5
            l_bg = 0.5
        else:
            if patch.ndim == 3 and patch.shape[2] >= 3:
                bgr_1 = np.mean(patch[class_1], axis=0)
                bgr_0 = np.mean(patch[class_0], axis=0)
                l1 = calculate_relative_luminance_bgr(bgr_1)
                l0 = calculate_relative_luminance_bgr(bgr_0)
            else:
                l1 = _srgb_to_linear(float(np.mean(patch[class_1])) / 255.0)
                l0 = _srgb_to_linear(float(np.mean(patch[class_0])) / 255.0)

            # Assign foreground as the minority class (typical of text on canvas)
            if c1_count <= c0_count:
                l_fg, l_bg = l1, l0
            else:
                l_fg, l_bg = l0, l1

    if is_conclusive:
        l_high = max(l_fg, l_bg)
        l_low = min(l_fg, l_bg)
        contrast = float((l_high + 0.05) / (l_low + 0.05))
    else:
        contrast = 1.0

    is_compliant = (contrast >= required_ratio) if is_conclusive else False

    evidence = ContrastEvidence(
        contrast_ratio=round(contrast, 2),
        foreground_luminance=round(l_fg, 4),
        background_luminance=round(l_bg, 4),
        required_ratio=required_ratio,
        is_compliant=is_compliant,
        is_conclusive=is_conclusive,
        is_large_text=is_large_text,
        method=method,
    )

    diagnostic = None
    if not is_conclusive:
        diagnostic = QCDiagnostic(
            code=QCDiagnosticCode.QC_INCONCLUSIVE_MEASUREMENT,
            level=QCDiagnosticLevel.WARNING,
            layer=QCLayer.LAYER_C,
            message="Contrast evaluation inconclusive: ROI is unimodal or has insufficient foreground/background separation.",
            scene_id=scene_id,
            timestamp=timestamp,
            evidence=evidence.model_dump(),
        )
    elif not is_compliant:
        diagnostic = QCDiagnostic(
            code=QCDiagnosticCode.QC_LOW_CONTRAST,
            level=QCDiagnosticLevel.ERROR,
            layer=QCLayer.LAYER_C,
            message=f"Low contrast detected ({contrast:.2f}:1 < required {required_ratio:.1f}:1).",
            scene_id=scene_id,
            timestamp=timestamp,
            evidence=evidence.model_dump(),
        )

    return contrast, evidence, diagnostic


def evaluate_roi_overlap(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
    frame: Optional[np.ndarray] = None,
    config: Optional[RasterAnalyzerConfig] = None,
    scene_id: Optional[str] = None,
    timestamp: Optional[float] = None,
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> Tuple[bool, OverlapEvidence, Optional[QCDiagnostic]]:
    """
    Evaluates pairwise ROI overlap by:
    1. Calculating geometric intersection area and overlap ratios.
    2. (If frame is provided) analyzing actual visible raster edge density inside the intersection patch.
    Distinguishes empty geometric intersections from visible collisions.
    """
    cfg = config or RasterAnalyzerConfig()
    p_box_a = PixelBox(x_min=box_a[0], y_min=box_a[1], x_max=box_a[2], y_max=box_a[3])
    p_box_b = PixelBox(x_min=box_b[0], y_min=box_b[1], x_max=box_b[2], y_max=box_b[3])

    ix0 = max(p_box_a.x_min, p_box_b.x_min)
    iy0 = max(p_box_a.y_min, p_box_b.y_min)
    ix1 = min(p_box_a.x_max, p_box_b.x_max)
    iy1 = min(p_box_a.y_max, p_box_b.y_max)

    has_geom_overlap = False
    intersection_area = 0
    ratio_a = 0.0
    ratio_b = 0.0
    p_inter: Optional[PixelBox] = None

    if ix1 > ix0 and iy1 > iy0:
        p_inter = PixelBox(x_min=ix0, y_min=iy0, x_max=ix1, y_max=iy1)
        intersection_area = p_inter.area
        ratio_a = float(intersection_area / p_box_a.area)
        ratio_b = float(intersection_area / p_box_b.area)
        has_geom_overlap = max(ratio_a, ratio_b) >= cfg.geometric_overlap_threshold

    raster_density: Optional[float] = None
    has_raster_overlap: Optional[bool] = None

    if has_geom_overlap and frame is not None and p_inter is not None:
        gray = to_grayscale(frame)
        h, w = gray.shape[:2]
        cx0 = max(0, min(p_inter.x_min, w - 1))
        cx1 = max(1, min(p_inter.x_max, w))
        cy0 = max(0, min(p_inter.y_min, h - 1))
        cy1 = max(1, min(p_inter.y_max, h))

        if cx1 > cx0 and cy1 > cy0:
            inter_patch = gray[cy0:cy1, cx0:cx1]
            edges = cv2.Canny(inter_patch, 50, 150)
            edge_count = np.count_nonzero(edges)
            raster_density = float(edge_count / inter_patch.size)
            has_raster_overlap = raster_density >= cfg.raster_overlap_density_threshold
        else:
            has_raster_overlap = False
            raster_density = 0.0

    # Overlap is confirmed if geometric overlap occurs AND raster inspection confirms content
    # (or raster inspection wasn't requested so geometric alone holds)
    is_overlapping = bool(has_raster_overlap if has_raster_overlap is not None else has_geom_overlap)

    evidence = OverlapEvidence(
        box_a=p_box_a,
        box_b=p_box_b,
        intersection_box=p_inter,
        intersection_area=intersection_area,
        overlap_ratio_a=round(ratio_a, 4),
        overlap_ratio_b=round(ratio_b, 4),
        is_geometric_overlap=has_geom_overlap,
        raster_edge_density=round(raster_density, 4) if raster_density is not None else None,
        is_raster_overlap=has_raster_overlap,
    )

    diagnostic = None
    if is_overlapping:
        diagnostic = QCDiagnostic(
            code=QCDiagnosticCode.QC_RASTER_OVERLAP,
            level=QCDiagnosticLevel.ERROR,
            layer=QCLayer.LAYER_C,
            message=(
                f"Visible ROI overlap detected (intersection_area={intersection_area}px, "
                f"overlap_a={ratio_a:.1%}, overlap_b={ratio_b:.1%}"
                + (f", raster_edge_density={raster_density:.2%}" if raster_density is not None else "")
                + ")."
            ),
            scene_id=scene_id,
            timestamp=timestamp,
            evidence=evidence.model_dump(),
        )

    return is_overlapping, evidence, diagnostic
