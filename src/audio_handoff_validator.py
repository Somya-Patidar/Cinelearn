"""
Audio and Render Handoff Validator for CineLearn (Step 2B.4C).

Ensures that only the exact storyboard that passed numerical fidelity validation
is used for audio generation, Stage 3 code generation, rendering, and final assembly.
Enforces deterministic storyboard fingerprinting, audio integrity, narration consistency,
and handoff gates across all pipeline transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydub import AudioSegment


# ============================================================================
# Status & Issue Enums
# ============================================================================

class HandoffStatus(str, Enum):
    """Overall status of audio and render handoff validation."""
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class HandoffIssueCategory(str, Enum):
    """Categorization of handoff discrepancies."""
    MUTATED_STORYBOARD = "MUTATED_STORYBOARD"
    STALE_AUDIO = "STALE_AUDIO"
    MISSING_AUDIO_SECTION = "MISSING_AUDIO_SECTION"
    INVALID_AUDIO_FILE = "INVALID_AUDIO_FILE"
    SECTION_ORDER_MISMATCH = "SECTION_ORDER_MISMATCH"
    MISSING_STORYBOARD_GATE = "MISSING_STORYBOARD_GATE"
    UNREADABLE_METADATA = "UNREADABLE_METADATA"
    TIMING_CONTRACT_VIOLATION = "TIMING_CONTRACT_VIOLATION"
    SYNCHRONIZATION_FAILED = "SYNCHRONIZATION_FAILED"


@dataclass
class HandoffIssue:
    """Individual issue detected during audio/render handoff validation."""
    section_id: str
    category: HandoffIssueCategory
    message: str
    actual_value: Optional[str] = None
    expected_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "category": self.category.value,
            "message": self.message,
            "actual_value": self.actual_value,
            "expected_value": self.expected_value,
        }


@dataclass
class AudioHandoffReport:
    """Overall outcome of the audio and render handoff audit."""
    overall_status: HandoffStatus
    is_safe_for_stage3: bool
    storyboard_fingerprint: str
    total_sections: int
    validated_sections: int
    issues: List[HandoffIssue] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "is_safe_for_stage3": self.is_safe_for_stage3,
            "storyboard_fingerprint": self.storyboard_fingerprint,
            "total_sections": self.total_sections,
            "validated_sections": self.validated_sections,
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
        }


# ============================================================================
# Deterministic Fingerprinting Helpers
# ============================================================================

def compute_storyboard_fingerprint(storyboard_or_sections: Union[Dict[str, Any], List[Any]]) -> str:
    """
    Computes a deterministic SHA-256 fingerprint of the semantic storyboard content.
    Includes section ID, title, lecture lines, captions, animations, and verified values.
    """
    sections_data = []
    if isinstance(storyboard_or_sections, dict):
        raw_sections = storyboard_or_sections.get("sections", [])
    elif isinstance(storyboard_or_sections, list):
        raw_sections = storyboard_or_sections
    else:
        raw_sections = []

    for s in raw_sections:
        if hasattr(s, "__dict__"):
            s_dict = s.__dict__
        elif isinstance(s, dict):
            s_dict = s
        else:
            s_dict = {}

        vvals = s_dict.get("verified_values")
        if isinstance(vvals, dict):
            canon_vvals = sorted((str(k), str(v)) for k, v in vvals.items())
        elif isinstance(vvals, list):
            canon_vvals = sorted(str(x) for x in vvals)
        else:
            canon_vvals = []

        sections_data.append({
            "id": str(s_dict.get("id", "")).strip(),
            "title": str(s_dict.get("title", "")).strip(),
            "lecture_lines": [str(line).strip() for line in (s_dict.get("lecture_lines") or [])],
            "captions": [str(c).strip() for c in (s_dict.get("captions") or [])],
            "animations": [str(a).strip() for a in (s_dict.get("animations") or [])],
            "story_beat": str(s_dict.get("story_beat") or s_dict.get("story_step") or "").strip(),
            "layout_plan": str(s_dict.get("layout_plan", "")).strip(),
            "verified_values": canon_vvals,
        })

    serialized = json.dumps(sections_data, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"sb_{digest[:16]}"


def compute_narration_hash(
    lecture_lines: List[str],
    audio_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Computes a SHA-256 hash of the exact narration lines and optional synthesis configuration.

    Maintains 100% backward compatibility: if audio_config is None, produces the exact
    canonical 16-character hex digest of the newline-joined stripped lines.
    If audio_config is provided (e.g. voice, rate, language), the sorted configuration
    is incorporated into the digest, ensuring cache invalidation on synthesis setting changes.
    """
    clean_lines = [str(line).strip() for line in (lecture_lines or [])]
    text = "\n".join(clean_lines)
    if audio_config:
        cfg_str = json.dumps(audio_config, sort_keys=True, ensure_ascii=False)
        payload = f"{text}\n---AUDIO_CONFIG---\n{cfg_str}"
    else:
        payload = text
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_scene_ir_narration_hash(
    scene_ir: Any,
    audio_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Computes deterministic narration cache identity directly from a SceneIR object.

    Extracts ordered, non-empty narration cues from the SceneIR's timeline events
    and hashes them along with optional synthesis configuration.
    """
    if hasattr(scene_ir, "get_ordered_narration_cues"):
        cues = scene_ir.get_ordered_narration_cues()
    elif isinstance(scene_ir, dict):
        timeline = scene_ir.get("timeline", [])
        extracted = []
        for idx, event in enumerate(timeline):
            anchor = event.get("narration_anchor") if isinstance(event, dict) else getattr(event, "narration_anchor", None)
            if anchor:
                cue = anchor.get("text_cue") if isinstance(anchor, dict) else getattr(anchor, "text_cue", "")
                line_idx = anchor.get("lecture_line_index", idx) if isinstance(anchor, dict) else getattr(anchor, "lecture_line_index", idx)
                if cue and str(cue).strip():
                    extracted.append((line_idx, idx, str(cue)))
        extracted.sort(key=lambda item: (item[0], item[1]))
        cues = [item[2] for item in extracted]
    else:
        cues = []
    return compute_narration_hash(cues, audio_config=audio_config)


# ============================================================================
# Audio Handoff Validator Engine
# ============================================================================

class AudioHandoffValidator:
    """
    Validates the consistency, completeness, and non-staleness of audio artifacts
    before Stage 3 code generation, rendering, and final assembly.
    """

    MIN_AUDIO_BYTES = 500  # Minimum byte size for readable audio segment
    MAX_DURATION_DISCREPANCY_SEC = 0.5  # Max allowed divergence between decoded audio and metadata duration

    def validate_handoff(
        self,
        sections: List[Any],
        section_audios: Dict[str, str],
        section_timestamps: Dict[str, Dict[str, Any]],
        expected_fingerprint: Optional[str] = None,
        fidelity_report: Optional[Any] = None,
    ) -> AudioHandoffReport:
        """
        Validates audio artifacts against the current storyboard sections.
        Ensures fidelity gate has passed, storyboard is unmutated, audio files exist
        and are non-empty/decodable, narration hashes match, and ordering is preserved.
        """
        current_fp = compute_storyboard_fingerprint(sections)
        issues: List[HandoffIssue] = []

        if not sections:
            return AudioHandoffReport(
                overall_status=HandoffStatus.NOT_APPLICABLE,
                is_safe_for_stage3=True,
                storyboard_fingerprint=current_fp,
                total_sections=0,
                validated_sections=0,
                issues=[],
                summary="No sections present to validate.",
            )

        # 1. Gate: Verify Stage 2 Numerical Fidelity Gate was not bypassed or failed
        if fidelity_report is not None:
            is_fidelity_safe = getattr(fidelity_report, "is_safe_for_audio", False)
            if not is_fidelity_safe:
                status_v = getattr(fidelity_report, "overall_status", "FAILED")
                if hasattr(status_v, "value"):
                    status_v = status_v.value
                issues.append(
                    HandoffIssue(
                        section_id="all",
                        category=HandoffIssueCategory.MISSING_STORYBOARD_GATE,
                        message=f"Storyboard has not passed numerical fidelity gate ({status_v}). Downstream handoff blocked.",
                        actual_value=str(status_v),
                        expected_value="VERIFIED",
                    )
                )

        # 2. Gate: Verify Storyboard was not mutated after fidelity validation
        if expected_fingerprint and current_fp != expected_fingerprint:
            issues.append(
                HandoffIssue(
                    section_id="all",
                    category=HandoffIssueCategory.MUTATED_STORYBOARD,
                    message="Storyboard content was mutated after numerical fidelity validation was established.",
                    actual_value=current_fp,
                    expected_value=expected_fingerprint,
                )
            )

        # 3. Section by section audio audit
        validated_count = 0
        for i, sec in enumerate(sections):
            if hasattr(sec, "id"):
                sid = sec.id
                lecture_lines = getattr(sec, "lecture_lines", []) or []
            elif isinstance(sec, dict):
                sid = sec.get("id", f"section_{i+1}")
                lecture_lines = sec.get("lecture_lines", []) or []
            else:
                sid = f"section_{i+1}"
                lecture_lines = []

            # A. Check section audio registration
            if sid not in section_audios or not section_audios[sid]:
                issues.append(
                    HandoffIssue(
                        section_id=sid,
                        category=HandoffIssueCategory.MISSING_AUDIO_SECTION,
                        message=f"Section '{sid}' is missing registered audio file path.",
                    )
                )
                continue

            # B. Check section timestamps registration
            if sid not in section_timestamps or not section_timestamps[sid]:
                issues.append(
                    HandoffIssue(
                        section_id=sid,
                        category=HandoffIssueCategory.MISSING_AUDIO_SECTION,
                        message=f"Section '{sid}' is missing timestamps metadata.",
                    )
                )
                continue

            audio_file_path = Path(section_audios[sid])
            meta = section_timestamps[sid]

            # C. Check section ordering and identifier consistency
            meta_sid = meta.get("section_id")
            if meta_sid and meta_sid != sid:
                issues.append(
                    HandoffIssue(
                        section_id=sid,
                        category=HandoffIssueCategory.SECTION_ORDER_MISMATCH,
                        message=f"Section audio metadata ID '{meta_sid}' does not match expected section ID '{sid}'.",
                        actual_value=meta_sid,
                        expected_value=sid,
                    )
                )
                continue

            meta_audio_path = meta.get("audio_path")
            if meta_audio_path:
                try:
                    p_meta = Path(meta_audio_path)
                    if not p_meta.is_absolute():
                        if p_meta.exists():
                            p_meta = p_meta.resolve()
                        else:
                            p_meta = (audio_file_path.parent / p_meta).resolve()
                    else:
                        p_meta = p_meta.resolve()
                    if p_meta != audio_file_path.resolve():
                        issues.append(
                            HandoffIssue(
                                section_id=sid,
                                category=HandoffIssueCategory.INVALID_AUDIO_FILE,
                                message=f"Section '{sid}' audio metadata path '{meta_audio_path}' does not match registered path '{audio_file_path}'.",
                                actual_value=str(meta_audio_path),
                                expected_value=str(audio_file_path),
                            )
                        )
                        continue
                except Exception as e:
                    issues.append(
                        HandoffIssue(
                            section_id=sid,
                            category=HandoffIssueCategory.INVALID_AUDIO_FILE,
                            message=f"Section '{sid}' audio metadata path '{meta_audio_path}' failed path resolution: {e}",
                            actual_value=str(meta_audio_path),
                            expected_value=str(audio_file_path),
                        )
                    )
                    continue

            # D. Check physical audio file existence and non-empty size
            if not audio_file_path.exists():
                issues.append(
                    HandoffIssue(
                        section_id=sid,
                        category=HandoffIssueCategory.INVALID_AUDIO_FILE,
                        message=f"Audio file '{audio_file_path}' does not exist on disk.",
                    )
                )
                continue

            try:
                f_size = audio_file_path.stat().st_size
            except Exception as e:
                f_size = 0

            if f_size < self.MIN_AUDIO_BYTES:
                issues.append(
                    HandoffIssue(
                        section_id=sid,
                        category=HandoffIssueCategory.INVALID_AUDIO_FILE,
                        message=f"Audio file '{audio_file_path}' is empty or too small ({f_size} bytes).",
                        actual_value=str(f_size),
                        expected_value=f">={self.MIN_AUDIO_BYTES}",
                    )
                )
                continue

            # E. Verify audio file readability
            try:
                audio_seg = AudioSegment.from_file(str(audio_file_path))
                seg_duration = len(audio_seg) / 1000.0
                if seg_duration <= 0.05:
                    issues.append(
                        HandoffIssue(
                            section_id=sid,
                            category=HandoffIssueCategory.INVALID_AUDIO_FILE,
                            message=f"Audio file '{audio_file_path}' has non-positive or near-zero duration ({seg_duration:.3f}s).",
                            actual_value=f"{seg_duration:.3f}",
                            expected_value="> 0.05",
                        )
                    )
                    continue
            except Exception as e:
                issues.append(
                    HandoffIssue(
                        section_id=sid,
                        category=HandoffIssueCategory.INVALID_AUDIO_FILE,
                        message=f"Audio file '{audio_file_path}' failed audio decoding: {e}",
                    )
                )
                continue

            # F. Verify narration consistency / detect stale audio
            expected_narr_hash = compute_narration_hash(lecture_lines)
            actual_narr_hash = meta.get("narration_hash")
            if actual_narr_hash:
                if actual_narr_hash != expected_narr_hash:
                    issues.append(
                        HandoffIssue(
                            section_id=sid,
                            category=HandoffIssueCategory.STALE_AUDIO,
                            message=f"Section '{sid}' audio narration hash does not match current storyboard lecture lines.",
                            actual_value=actual_narr_hash,
                            expected_value=expected_narr_hash,
                        )
                    )
                    continue
            else:
                # Fallback text comparison against line timestamps
                meta_lines = meta.get("lines") or meta.get("line_timestamps") or []
                if len(meta_lines) != len(lecture_lines):
                    issues.append(
                        HandoffIssue(
                            section_id=sid,
                            category=HandoffIssueCategory.STALE_AUDIO,
                            message=f"Section '{sid}' audio line count ({len(meta_lines)}) does not match current storyboard line count ({len(lecture_lines)}).",
                            actual_value=str(len(meta_lines)),
                            expected_value=str(len(lecture_lines)),
                        )
                    )
                    continue

                line_differs = False
                for li, (line_m, exp_line) in enumerate(zip(meta_lines, lecture_lines)):
                    raw_t = line_m.get("raw_text") or line_m.get("text", "")
                    if raw_t.strip() != exp_line.strip():
                        issues.append(
                            HandoffIssue(
                                section_id=sid,
                                category=HandoffIssueCategory.STALE_AUDIO,
                                message=f"Section '{sid}' line {li+1} narration changed; existing audio is stale.",
                                actual_value=raw_t.strip(),
                                expected_value=exp_line.strip(),
                            )
                        )
                        line_differs = True
                        break
                if line_differs:
                    continue

            # G. Duration validity in metadata and physical duration reconciliation
            raw_duration = meta.get("total_duration")
            if raw_duration is None:
                issues.append(
                    HandoffIssue(
                        section_id=sid,
                        category=HandoffIssueCategory.UNREADABLE_METADATA,
                        message=f"Section '{sid}' audio metadata is missing 'total_duration'.",
                        expected_value="positive float",
                    )
                )
                continue

            try:
                tot_dur = float(raw_duration)
            except (TypeError, ValueError):
                issues.append(
                    HandoffIssue(
                        section_id=sid,
                        category=HandoffIssueCategory.UNREADABLE_METADATA,
                        message=f"Section '{sid}' audio metadata has non-numeric duration: {raw_duration!r}.",
                        actual_value=str(raw_duration),
                        expected_value="positive float",
                    )
                )
                continue

            if math.isnan(tot_dur) or math.isinf(tot_dur) or tot_dur <= 0.0:
                issues.append(
                    HandoffIssue(
                        section_id=sid,
                        category=HandoffIssueCategory.INVALID_AUDIO_FILE,
                        message=f"Section '{sid}' audio metadata reports non-positive or non-finite duration ({tot_dur}).",
                        actual_value=str(tot_dur),
                        expected_value="> 0.0",
                    )
                )
                continue

            duration_diff = abs(seg_duration - tot_dur)
            if duration_diff > self.MAX_DURATION_DISCREPANCY_SEC:
                issues.append(
                    HandoffIssue(
                        section_id=sid,
                        category=HandoffIssueCategory.INVALID_AUDIO_FILE,
                        message=(
                            f"Section '{sid}' decoded audio duration ({seg_duration:.3f}s) disagrees with "
                            f"metadata total_duration ({tot_dur:.3f}s) by {duration_diff:.3f}s "
                            f"(tolerance: {self.MAX_DURATION_DISCREPANCY_SEC}s)."
                        ),
                        actual_value=f"{seg_duration:.3f}s",
                        expected_value=f"{tot_dur:.3f}s",
                    )
                )
                continue

            # H. Timing Contract Integrity Check (if present in metadata)
            timing_contract = meta.get("timing_contract")
            if timing_contract and isinstance(timing_contract, dict):
                contract_segments = timing_contract.get("segments") or []
                if len(contract_segments) != len(lecture_lines):
                    issues.append(
                        HandoffIssue(
                            section_id=sid,
                            category=HandoffIssueCategory.TIMING_CONTRACT_VIOLATION,
                            message=(
                                f"Section '{sid}' timing contract segment count ({len(contract_segments)}) "
                                f"does not match storyboard lecture line count ({len(lecture_lines)})."
                            ),
                            actual_value=str(len(contract_segments)),
                            expected_value=str(len(lecture_lines)),
                        )
                    )
                    continue

                prev_end = -0.001
                contract_valid = True
                for s_idx, c_seg in enumerate(contract_segments):
                    s_dur = c_seg.get("actual_audio_duration")
                    s_start = c_seg.get("timeline_start", 0.0)
                    if s_dur is not None and s_dur <= 0.05:
                        issues.append(
                            HandoffIssue(
                                section_id=sid,
                                category=HandoffIssueCategory.TIMING_CONTRACT_VIOLATION,
                                message=f"Section '{sid}' segment {s_idx+1} reports non-positive audio duration ({s_dur}s).",
                                actual_value=str(s_dur),
                                expected_value="> 0.05s",
                            )
                        )
                        contract_valid = False
                        break
                    if s_start < prev_end - 0.05:
                        issues.append(
                            HandoffIssue(
                                section_id=sid,
                                category=HandoffIssueCategory.TIMING_CONTRACT_VIOLATION,
                                message=f"Section '{sid}' segment {s_idx+1} has backward or overlapping start ({s_start}s < {prev_end}s).",
                                actual_value=str(s_start),
                                expected_value=f">={prev_end}s",
                            )
                        )
                        contract_valid = False
                        break
                    prev_end = s_start
                if not contract_valid:
                    continue

            validated_count += 1

        is_safe = (len(issues) == 0)
        overall_status = HandoffStatus.VALIDATED if is_safe else HandoffStatus.FAILED
        summary = (
            f"Audio and render handoff: {validated_count}/{len(sections)} sections validated. "
            f"{len(issues)} issue(s) detected. Fingerprint: {current_fp}."
        )

        return AudioHandoffReport(
            overall_status=overall_status,
            is_safe_for_stage3=is_safe,
            storyboard_fingerprint=current_fp,
            total_sections=len(sections),
            validated_sections=validated_count,
            issues=issues,
            summary=summary,
        )

    def validate_synchronized_media(
        self,
        video_path: Union[str, Path],
        expected_duration: Optional[float] = None,
        tolerance: float = 0.5,
    ) -> Tuple[bool, List[str]]:
        """
        Validates assembled video media via ffprobe ensuring both audio and video streams exist,
        are non-empty, and overall duration is consistent.
        """
        import subprocess

        p = Path(video_path)
        issues = []
        if not p.exists():
            return False, [f"Media file '{video_path}' does not exist on disk."]
        if p.stat().st_size < 1000:
            return False, [f"Media file '{video_path}' is empty or truncated ({p.stat().st_size} bytes)."]

        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,codec_type,codec_name,duration",
            "-show_entries", "format=duration,size",
            "-of", "json",
            str(p)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return False, [f"ffprobe failed to inspect '{video_path}': {res.stderr}"]

        try:
            data = json.loads(res.stdout)
        except Exception as e:
            return False, [f"ffprobe JSON decoding error for '{video_path}': {e}"]

        streams = data.get("streams", [])
        has_video = any(s.get("codec_type") == "video" for s in streams)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)

        if not has_video:
            issues.append(f"Media file '{video_path}' contains no video stream.")
        if not has_audio:
            issues.append(f"Media file '{video_path}' contains no audio stream.")

        raw_dur = data.get("format", {}).get("duration")
        try:
            actual_dur = float(raw_dur)
            if actual_dur <= 0.05:
                issues.append(f"Media file '{video_path}' has non-positive duration: {actual_dur}s.")
            elif expected_duration is not None and expected_duration > 0:
                diff = abs(actual_dur - expected_duration)
                if diff > tolerance:
                    issues.append(
                        f"Media file '{video_path}' duration ({actual_dur:.2f}s) differs from expected "
                        f"duration ({expected_duration:.2f}s) by {diff:.2f}s (tolerance: {tolerance:.2f}s)."
                    )
        except (TypeError, ValueError):
            issues.append(f"Media file '{video_path}' has unreadable duration: {raw_dur!r}.")

        return len(issues) == 0, issues
