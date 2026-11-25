"""
Adaptive Duration Management for CineLearn (Code2Video).

Implements adaptive video duration with soft targets and hard maximums:
- Duration estimation from planned narration, speaking rate, section count,
  worked examples, visual transitions, equations, and pauses.
- Learning level-aware defaults and configurable targets.
- Intelligent compression of nonessential repetition when soft target is exceeded
  without omitting prerequisites or core concepts.
- Multi-part lesson reorganization when hard maximum is exceeded to prevent silent truncation.
- Line-by-line audio and animation timing validation.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class LineDurationEstimate:
    line_index: int
    text: str
    word_count: int
    speech_duration: float
    pause_duration: float
    visual_overhead: float
    total_duration: float
    has_equation: bool = False
    has_worked_example: bool = False


@dataclass
class SectionDurationEstimate:
    section_id: str
    title: str
    lines: List[LineDurationEstimate]
    total_duration: float
    line_count: int


@dataclass
class LessonDurationEstimate:
    total_duration_sec: float
    total_duration_min: float
    soft_target_sec: float
    hard_max_sec: float
    status: str  # "OPTIMAL", "SOFT_TARGET_EXCEEDED", "HARD_MAX_EXCEEDED"
    level: str
    sections: List[SectionDurationEstimate]
    recommendations: List[str] = field(default_factory=list)
    multi_part_plan: Optional[Dict[str, Any]] = None


class AdaptiveDurationManager:
    """
    Manages duration estimation, adaptive compression, multi-part reorganization,
    and line-level timing validation for CineLearn.
    """

    # Level-based default speaking rates (WPM)
    # Beginners need slower, more deliberate pacing; advanced can absorb higher density.
    DEFAULT_WPM_BY_LEVEL = {
        "beginner": 135.0,
        "intermediate": 145.0,
        "advanced": 155.0,
    }

    # Level-based default soft target durations in seconds
    DEFAULT_SOFT_TARGETS = {
        "beginner": 210.0,      # ~3.5 minutes
        "intermediate": 280.0,  # ~4.67 minutes
        "advanced": 360.0,      # ~6.0 minutes
    }

    # Level-based default hard maximum durations in seconds
    DEFAULT_HARD_MAX = {
        "beginner": 360.0,      # ~6.0 minutes
        "intermediate": 480.0,  # ~8.0 minutes
        "advanced": 600.0,      # ~10.0 minutes
    }

    # Level-based default minimum durations in seconds (to prevent shallow/rushed videos)
    DEFAULT_MIN_DURATION = {
        "beginner": 180.0,      # ~3.0 minutes
        "intermediate": 240.0,  # ~4.0 minutes
        "advanced": 300.0,      # ~5.0 minutes
    }

    # Tolerance window around optional user target: ±2.5 minutes (150 seconds)
    TARGET_TOLERANCE_SEC: float = 150.0

    # Cognitive overload threshold for a single educational video: 12 minutes (720 seconds)
    COGNITIVE_OVERLOAD_THRESHOLD_SEC: float = 720.0

    # Common repetitive filler phrases to streamline if duration exceeds target
    REPETITIVE_PATTERNS = [
        r"\b(?:as we have already (?:seen|noted|discussed|stated|observed))\b,?\s*",
        r"\b(?:to reiterate what was (?:said|stated|explained) earlier)\b,?\s*",
        r"\b(?:as mentioned in the previous (?:step|section|part))\b,?\s*",
        r"\b(?:it goes without saying that)\b,?\s*",
        r"\b(?:it is interesting to note that)\b,?\s*",
        r"\b(?:needless to say)\b,?\s*",
        r"\b(?:as everyone knows)\b,?\s*",
    ]

    def __init__(
        self,
        level: str = "intermediate",
        soft_target_duration: float = 0.0,
        hard_max_duration: float = 0.0,
        enforce_min_duration: bool = False,
        speaking_wpm: float = 0.0,
        inter_line_pause: float = 0.5,
        section_end_pause: float = 1.0,
    ):
        self.level = (level or "intermediate").lower()
        if self.level not in self.DEFAULT_WPM_BY_LEVEL:
            self.level = "intermediate"

        # Speaking rate
        self.speaking_wpm = speaking_wpm if speaking_wpm > 0 else self.DEFAULT_WPM_BY_LEVEL[self.level]

        # Flexible user target vs adaptive topic-dependent duration
        self.has_user_target = (soft_target_duration is not None and soft_target_duration > 0)
        self.user_target_sec = float(soft_target_duration) if self.has_user_target else 0.0

        # Duration bounds
        self.soft_target_sec = (
            soft_target_duration if (soft_target_duration and soft_target_duration > 0)
            else self.DEFAULT_SOFT_TARGETS[self.level]
        )
        self.hard_max_sec = (
            hard_max_duration if (hard_max_duration and hard_max_duration > 0)
            else self.DEFAULT_HARD_MAX[self.level]
        )

        # Enforce consistency: hard_max must be >= soft_target
        if self.hard_max_sec < self.soft_target_sec:
            self.hard_max_sec = self.soft_target_sec * 1.35

        self.min_duration_sec = self.DEFAULT_MIN_DURATION.get(self.level, 240.0)
        self.enforce_min_duration = enforce_min_duration
        self.inter_line_pause = inter_line_pause
        self.section_end_pause = section_end_pause

    def _has_equation(self, text: str, animation_desc: str = "") -> bool:
        """Detect mathematical symbols, formulas, or equations."""
        combined = f"{text} {animation_desc}"
        equation_indicators = [
            r"[=<>±×÷\^]",
            r"(?:\d+\s*[\+\-\*\/]\s*\d+)",  # arithmetic expressions like 2 + 3
            r"\\[a-zA-Z]+",  # LaTeX commands like \det, \frac, \lambda
            r"\[\s*[-0-9\s,;]{2,}\s*\]",  # Bracketed coordinates or numbers like [1, 2]
            r"\b(?:det|sin|cos|tan|log|exp)\s*\(",
        ]
        return any(re.search(pat, combined, re.IGNORECASE) for pat in equation_indicators)

    def _has_worked_example(self, text: str, animation_desc: str = "") -> bool:
        """Detect worked calculation steps, substitutions, or numerical examples."""
        combined = f"{text} {animation_desc}"
        example_indicators = [
            r"\b(?:example|worked example|calculate|compute|solve|substitute|plug in|evaluat)\b",
            r"\b(?:step \d+|let's calculate|for instance|suppose we have)\b",
            r"\d+\s*[\*\+\-\/]\s*\d+",  # Explicit arithmetic like 2 * 3
        ]
        return any(re.search(pat, combined, re.IGNORECASE) for pat in example_indicators)

    def estimate_line(
        self,
        text: str,
        animation_desc: str = "",
        is_last_line: bool = False,
        wpm_override: Optional[float] = None,
        pause_override: Optional[float] = None,
    ) -> LineDurationEstimate:
        """
        Estimate duration for an individual lecture line considering:
        - Word count and speech rate
        - Mathematical equation absorption time
        - Worked example demonstration time
        - Animation visual transition overhead
        - Inter-line / section-end pauses
        """
        wpm = wpm_override or self.speaking_wpm
        words = text.split()
        word_count = len(words)

        # Base spoken duration in seconds
        speech_sec = max(1.2, word_count / (wpm / 60.0))

        # Check for equations & worked examples
        has_eq = self._has_equation(text, animation_desc)
        has_ex = self._has_worked_example(text, animation_desc)

        # Visual overhead: visual absorption, equation rendering, animation settle
        visual_overhead = 0.0
        if has_eq:
            visual_overhead += 0.8
        if has_ex:
            visual_overhead += 1.0

        # Animation visual complexity overhead
        anim_words = len(animation_desc.split())
        if anim_words > 12:
            visual_overhead += 0.6
        elif anim_words > 6:
            visual_overhead += 0.3

        # Pause duration
        if pause_override is not None:
            pause_sec = pause_override
        else:
            pause_sec = self.section_end_pause if is_last_line else self.inter_line_pause

        total = speech_sec + visual_overhead + pause_sec

        return LineDurationEstimate(
            line_index=0,  # to be assigned by caller
            text=text,
            word_count=word_count,
            speech_duration=round(speech_sec, 3),
            pause_duration=round(pause_sec, 3),
            visual_overhead=round(visual_overhead, 3),
            total_duration=round(total, 3),
            has_equation=has_eq,
            has_worked_example=has_ex,
        )

    def estimate_section(
        self,
        section_data: Dict[str, Any],
        wpm_override: Optional[float] = None,
        pause_override: Optional[float] = None,
    ) -> SectionDurationEstimate:
        """Estimate total duration and line breakdown for a storyboard section."""
        sid = section_data.get("id", "section_unknown")
        title = section_data.get("title", "")
        lines = section_data.get("lecture_lines", [])
        anims = section_data.get("animations", [])

        line_estimates: List[LineDurationEstimate] = []
        total_sec = 0.0

        for i, line_text in enumerate(lines):
            anim_desc = anims[i] if i < len(anims) else ""
            is_last = (i == len(lines) - 1)
            est = self.estimate_line(
                text=line_text,
                animation_desc=anim_desc,
                is_last_line=is_last,
                wpm_override=wpm_override,
                pause_override=pause_override,
            )
            est.line_index = i
            line_estimates.append(est)
            total_sec += est.total_duration

        return SectionDurationEstimate(
            section_id=sid,
            title=title,
            lines=line_estimates,
            total_duration=round(total_sec, 3),
            line_count=len(lines),
        )

    def estimate_lesson(
        self,
        storyboard_data: Dict[str, Any],
        wpm_override: Optional[float] = None,
        pause_override: Optional[float] = None,
    ) -> LessonDurationEstimate:
        """
        Estimate entire lesson duration from storyboard sections.
        Categorizes result against soft target and hard maximum.
        """
        sections_data = storyboard_data.get("sections", [])
        section_estimates: List[SectionDurationEstimate] = []
        total_sec = 0.0

        for sec in sections_data:
            sec_est = self.estimate_section(
                sec, wpm_override=wpm_override, pause_override=pause_override
            )
            section_estimates.append(sec_est)
            total_sec += sec_est.total_duration

        total_min = total_sec / 60.0
        recommendations: List[str] = []
        status = "OPTIMAL"

        if total_sec > self.hard_max_sec:
            status = "HARD_MAX_EXCEEDED"
            recommendations.append(
                f"Lesson duration ({total_min:.1f}m) exceeds maximum single-video threshold ({self.hard_max_sec / 60.0:.1f}m). "
                f"Reorganizing lesson into coherent multi-part structure instead of truncating essential content."
            )
        elif self.has_user_target and total_sec > self.soft_target_sec:
            status = "SOFT_TARGET_EXCEEDED"
            recommendations.append(
                f"Lesson duration ({total_min:.1f}m) exceeds user target window ({self.user_target_sec / 60.0:.1f}m ± 2.5m). "
                f"Compressing non-essential repetition while fully preserving prerequisites and core concepts."
            )
        elif not self.has_user_target and total_sec > self.soft_target_sec:
            status = "SOFT_TARGET_EXCEEDED"
            recommendations.append(
                f"Lesson duration ({total_min:.1f}m) is extensive for a single video. "
                f"Streamlining non-essential repetition while fully preserving prerequisites and core concepts."
            )
        elif self.enforce_min_duration and total_sec < self.min_duration_sec:
            status = "UNDER_LENGTH_WARNING"
            recommendations.append(
                f"Lesson duration ({total_min:.1f}m) is below the recommended minimum ({self.min_duration_sec / 60.0:.1f}m for {self.level.capitalize()} level). "
                f"Expand lecture lines to 5-8 substantive sentences per section with worked examples and formal definitions."
            )
        else:
            if total_sec < self.min_duration_sec:
                recommendations.append(
                    f"Lesson duration ({total_min:.1f}m) is brief for a {self.level.capitalize()} topic (recommended floor: {self.min_duration_sec / 60.0:.1f}m). "
                    f"Consider expanding lecture lines to 5-8 substantive sentences per section."
                )
            elif self.has_user_target:
                recommendations.append(
                    f"Lesson duration ({total_min:.1f}m) is within user target window ({self.user_target_sec / 60.0:.1f}m ± 2.5m)."
                )
            else:
                recommendations.append(
                    f"Lesson duration ({total_min:.1f}m) naturally reflects topic complexity and depth ({self.level.capitalize()} level)."
                )

        return LessonDurationEstimate(
            total_duration_sec=round(total_sec, 3),
            total_duration_min=round(total_min, 3),
            soft_target_sec=self.soft_target_sec,
            hard_max_sec=self.hard_max_sec,
            status=status,
            level=self.level,
            sections=section_estimates,
            recommendations=recommendations,
        )

    def audit_timings(
        self,
        storyboard_est_sec: float,
        section_audios_meta: List[Dict[str, Any]],
        final_video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive duration audit comparing:
        1. Initial storyboard duration estimate
        2. Measured TTS narration duration (ground-truth spoken audio)
        3. Actual final rendered video duration (from video file inspection via ffprobe)
        """
        import subprocess

        # 1. Sum measured audio durations across sections
        measured_audio_sec = 0.0
        section_timings = []
        for s in section_audios_meta:
            sid = s.get("section_id", "")
            dur = float(s.get("total_duration", 0.0))
            measured_audio_sec += dur
            section_timings.append({"section_id": sid, "duration_sec": round(dur, 2)})

        # 2. Inspect rendered video duration if file exists
        video_dur_sec = None
        if final_video_path and Path(final_video_path).exists():
            try:
                probe_cmd = [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(final_video_path)
                ]
                res = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                if res.returncode == 0 and res.stdout.strip():
                    video_dur_sec = round(float(res.stdout.strip()), 2)
            except Exception:
                video_dur_sec = None

        audio_diff_sec = round(abs(measured_audio_sec - storyboard_est_sec), 2)
        video_diff_sec = round(abs(video_dur_sec - measured_audio_sec), 2) if video_dur_sec else None

        return {
            "storyboard_estimate_sec": round(storyboard_est_sec, 2),
            "storyboard_estimate_min": round(storyboard_est_sec / 60.0, 2),
            "measured_audio_duration_sec": round(measured_audio_sec, 2),
            "measured_audio_duration_min": round(measured_audio_sec / 60.0, 2),
            "rendered_video_duration_sec": video_dur_sec,
            "rendered_video_duration_min": round(video_dur_sec / 60.0, 2) if video_dur_sec else None,
            "storyboard_vs_audio_diff_sec": audio_diff_sec,
            "audio_vs_video_diff_sec": video_diff_sec,
            "section_timings": section_timings,
            "status": "VALIDATED" if (video_dur_sec or measured_audio_sec > 0) else "ESTIMATED_ONLY"
        }

    def adapt_storyboard_if_needed(
        self, storyboard_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], LessonDurationEstimate, Dict[str, Any]]:
        """
        Analyzes storyboard duration and executes adaptive actions:
        1. If within soft target: Keep intact.
        2. If soft target exceeded: Streamline repetitive filler without dropping prerequisites or concepts.
        3. If hard maximum exceeded: Reorganize into a coherent multi-part lesson plan.

        Returns:
            (adapted_storyboard, final_estimate, adaptation_report)
        """
        initial_est = self.estimate_lesson(storyboard_data)
        report: Dict[str, Any] = {
            "initial_duration_sec": initial_est.total_duration_sec,
            "initial_status": initial_est.status,
            "level": self.level,
            "soft_target_sec": self.soft_target_sec,
            "hard_max_sec": self.hard_max_sec,
            "actions_taken": [],
        }

        # Case 1: Optimal
        if initial_est.status == "OPTIMAL":
            report["actions_taken"].append("No adaptation needed; duration is within soft target.")
            report["final_duration_sec"] = initial_est.total_duration_sec
            report["final_status"] = "OPTIMAL"
            return storyboard_data, initial_est, report

        # Case 1b: Under-length warning
        if initial_est.status == "UNDER_LENGTH_WARNING":
            report["actions_taken"].append(
                f"Duration ({initial_est.total_duration_min:.1f}m) is below the minimum recommended floor ({self.min_duration_sec / 60.0:.1f}m). "
                f"Flagged for explanatory expansion to ensure complete educational flow."
            )
            report["final_duration_sec"] = initial_est.total_duration_sec
            report["final_status"] = "UNDER_LENGTH_WARNING"
            return storyboard_data, initial_est, report

        # Case 2: Soft target exceeded -> Non-destructive compression
        adapted_sb = {
            "sections": [
                {
                    "id": s["id"],
                    "title": s["title"],
                    "lecture_lines": list(s.get("lecture_lines", [])),
                    "animations": list(s.get("animations", [])),
                }
                for s in storyboard_data.get("sections", [])
            ]
        }

        if initial_est.status in ("SOFT_TARGET_EXCEEDED", "HARD_MAX_EXCEEDED"):
            pruned_lines_count = 0
            for sec in adapted_sb["sections"]:
                new_lines = []
                for line in sec["lecture_lines"]:
                    cleaned_line = line
                    for pat in self.REPETITIVE_PATTERNS:
                        cleaned_line = re.sub(pat, "", cleaned_line, flags=re.IGNORECASE).strip()
                    # Capitalize first letter if needed
                    if cleaned_line and cleaned_line[0].islower():
                        cleaned_line = cleaned_line[0].upper() + cleaned_line[1:]
                    if cleaned_line != line:
                        pruned_lines_count += 1
                    new_lines.append(cleaned_line)
                sec["lecture_lines"] = new_lines

            report["actions_taken"].append(
                f"Streamlined repetitive narrative filler in {pruned_lines_count} lines without modifying core definitions or prerequisites."
            )

            # Re-estimate after text streamlining with slightly tighter inter-line pause (0.4s)
            mid_est = self.estimate_lesson(adapted_sb, pause_override=0.4)

            # If still exceeding hard maximum, construct multi-part reorganization
            if mid_est.total_duration_sec > self.hard_max_sec:
                report["actions_taken"].append(
                    "Duration still exceeds hard maximum after narrative compression. "
                    "Constructed multi-part lesson reorganization plan."
                )
                multi_part = self._build_multi_part_reorganization(adapted_sb)
                mid_est.multi_part_plan = multi_part
                mid_est.status = "HARD_MAX_EXCEEDED"
                report["multi_part_plan"] = multi_part
                report["final_duration_sec"] = mid_est.total_duration_sec
                report["final_status"] = "HARD_MAX_EXCEEDED"
                return adapted_sb, mid_est, report

            report["final_duration_sec"] = mid_est.total_duration_sec
            report["final_status"] = mid_est.status
            return adapted_sb, mid_est, report

        return storyboard_data, initial_est, report

    def _build_multi_part_reorganization(self, storyboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Organizes sections into coherent parts instead of truncating essential content.
        Typically Part 1: Foundations & Core Mechanics, Part 2: Applications & Worked Examples.
        """
        sections = storyboard_data.get("sections", [])
        total_sections = len(sections)
        split_point = max(2, total_sections // 2)

        part1_sections = sections[:split_point]
        part2_sections = sections[split_point:]

        est1 = self.estimate_lesson({"sections": part1_sections})
        est2 = self.estimate_lesson({"sections": part2_sections})

        return {
            "is_multi_part": True,
            "total_parts": 2,
            "parts": [
                {
                    "part": 1,
                    "title": "Part 1: Conceptual Foundations and Core Principles",
                    "section_ids": [s["id"] for s in part1_sections],
                    "section_count": len(part1_sections),
                    "estimated_duration_sec": est1.total_duration_sec,
                    "estimated_duration_min": round(est1.total_duration_sec / 60.0, 2),
                },
                {
                    "part": 2,
                    "title": "Part 2: Worked Examples, Calculations and Synthesis",
                    "section_ids": [s["id"] for s in part2_sections],
                    "section_count": len(part2_sections),
                    "estimated_duration_sec": est2.total_duration_sec,
                    "estimated_duration_min": round(est2.total_duration_sec / 60.0, 2),
                },
            ],
            "approval_required": True,
            "recommendation": (
                "The topic has sufficient conceptual depth that a single video would exceed maximum learner cognitive load. "
                "The lesson is pre-organized into two balanced modules."
            ),
        }

    def validate_line_timings(
        self, section_audio_meta: Dict[str, Any], section_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates timing at the individual narration-line level:
        - Compares measured TTS durations with line budgets
        - Flags timing anomalies (speech rate too high/low, audio lines too long/short)
        - If section Manim code is provided, checks animation block wait/run_time balance
        """
        lines_meta = section_audio_meta.get("lines", [])
        section_id = section_audio_meta.get("section_id", "section")
        total_duration = section_audio_meta.get("total_duration", 0.0)

        checks: List[Dict[str, Any]] = []
        warnings: List[str] = []
        is_valid = True

        for item in lines_meta:
            idx = item.get("line_index", 0)
            raw_text = item.get("raw_text", "")
            dur = item.get("duration", 0.0)
            words = len(raw_text.split())

            line_check = {
                "line_index": idx,
                "text": raw_text[:40] + ("..." if len(raw_text) > 40 else ""),
                "duration_sec": round(dur, 2),
                "word_count": words,
                "words_per_second": round(words / dur, 2) if dur > 0 else 0,
                "status": "PASS",
            }

            # Check 1: Audio line duration bounds
            if dur < 1.0:
                line_check["status"] = "WARN_TOO_FAST"
                msg = f"Line {idx+1} audio duration is very short ({dur:.2f}s). May feel rushed."
                warnings.append(msg)
                line_check["warning"] = msg
            elif dur > 16.0:
                line_check["status"] = "WARN_TOO_LONG"
                msg = f"Line {idx+1} audio duration is very long ({dur:.2f}s, {words} words). Consider splitting line."
                warnings.append(msg)
                line_check["warning"] = msg

            # Check 2: Speaking rate anomalies
            wps = words / dur if dur > 0 else 0
            if wps > 4.2:
                msg = f"Line {idx+1} speech density ({wps:.1f} words/s) is abnormally rapid."
                warnings.append(msg)
                line_check["pacing_warning"] = msg
            elif wps < 1.0 and words > 4:
                msg = f"Line {idx+1} speech density ({wps:.1f} words/s) is abnormally slow."
                warnings.append(msg)
                line_check["pacing_warning"] = msg

            checks.append(line_check)

        # Check 3: If Manim code is provided, analyze code timing allocations
        code_timing_checks = []
        if section_code:
            code_timing_checks = self._validate_code_line_timings(section_code, lines_meta)
            for ctc in code_timing_checks:
                if ctc.get("mismatch_sec", 0) > 3.0:
                    warnings.append(
                        f"Code animation for Line {ctc['line_index']+1} has {ctc['mismatch_sec']:.1f}s timing discrepancy "
                        f"against measured TTS ({ctc['code_duration']:.1f}s vs {ctc['tts_duration']:.1f}s)."
                    )

        return {
            "section_id": section_id,
            "total_duration": total_duration,
            "is_valid": len(warnings) == 0,
            "line_count": len(lines_meta),
            "line_checks": checks,
            "code_timing_checks": code_timing_checks,
            "warnings": warnings,
        }

    def _validate_code_line_timings(
        self, code: str, lines_meta: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Statically inspects Manim code blocks for each lecture line to estimate animation time."""
        results = []
        # Split code into line blocks: # === Animation for Lecture Line X ===
        blocks = re.split(r"#\s*===+\s*Animation for Lecture Line\s*(\d+)\s*===+", code, flags=re.IGNORECASE)

        if len(blocks) < 3:
            return results

        # blocks[0] is header, blocks[1] is line 1 index, blocks[2] is line 1 code, etc.
        i = 1
        while i < len(blocks) - 1:
            try:
                line_num = int(blocks[i])
                block_code = blocks[i + 1]
            except (ValueError, IndexError):
                i += 2
                continue

            line_idx = line_num - 1
            tts_dur = 4.0
            if line_idx < len(lines_meta):
                tts_dur = lines_meta[line_idx].get("duration", 4.0)

            # Sum run_times and wait times
            run_times = [float(m) for m in re.findall(r"run_time\s*=\s*([0-9.]+)", block_code)]
            waits = [float(m) for m in re.findall(r"self\.wait\(\s*([0-9.]+)\s*\)", block_code)]
            total_code_dur = sum(run_times) + sum(waits)

            results.append({
                "line_index": line_idx,
                "code_duration": round(total_code_dur, 2),
                "tts_duration": round(tts_dur, 2),
                "mismatch_sec": round(abs(total_code_dur - tts_dur), 2),
            })
            i += 2

        return results
