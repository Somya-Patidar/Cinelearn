"""
Audio Manager for Code2Video (Educational Video Generation).
Handles audio input/output, text-to-speech generation, timestamp generation,
synchronized subtitles, concise caption extraction, and context-appropriate animation selection.
"""

import os
import sys
import re
import json
import array
import random
import asyncio
import subprocess
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from datetime import timedelta

import srt
from pydub import AudioSegment
try:
    from src.audio_handoff_validator import compute_narration_hash, compute_scene_ir_narration_hash
except ImportError:
    from audio_handoff_validator import compute_narration_hash, compute_scene_ir_narration_hash

try:
    from src.timing_contract import SceneTimingContract, NarrationSegmentTiming, AnimationRetimer, PacingAuditResult
except ImportError:
    from timing_contract import SceneTimingContract, NarrationSegmentTiming, AnimationRetimer, PacingAuditResult

try:
    from src.ir.models import AudioReference, SceneIR, SceneSequenceIR
except ImportError:
    try:
        from ir.models import AudioReference, SceneIR, SceneSequenceIR
    except ImportError:
        AudioReference = None
        SceneIR = None
        SceneSequenceIR = None


# Stop words that should not appear at the end of a truncated concise caption
STOP_ENDINGS = {
    "in", "of", "to", "the", "a", "an", "that", "and", "or", "for", "with", "by",
    "from", "at", "on", "into", "behind", "as", "is", "are", "was", "were", "under",
    "over", "across", "through", "about", "how", "which", "who", "what", "where", "when",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "their", "its"
}


def split_clauses_safe(text: str) -> List[str]:
    """Split clauses by commas, semicolons, dashes without breaking bracketed coordinates like [3, 2]."""
    placeholder = "___COMMA___"
    protected = re.sub(r"\[([^\]]*)\]", lambda m: "[" + m.group(1).replace(",", placeholder) + "]", text)
    protected = re.sub(r"\(([^\)]*)\)", lambda m: "(" + m.group(1).replace(",", placeholder) + ")", protected)
    protected = re.sub(r"(\d+),(\d+)", r"\1___NUMCOMMA___\2", protected)
    return [
        p.replace(placeholder, ",").replace("___NUMCOMMA___", ",").strip()
        for p in re.split(r"[,;—]", protected)
        if p.strip()
    ]


def extract_concise_caption(text: str, max_words: int = 5, max_chars: int = 35) -> str:
    """
    Extract a concise key phrase / important concept from a lecture line.
    Avoids displaying the entire script on screen; extracts only important sentences/key phrases.
    """
    if not text or not text.strip():
        return ""

    clean = text.strip()
    # Strip leading bullets, numbers, dashes
    clean = re.sub(r"^[0-9]+[\.\)\-:\s]+", "", clean)
    clean = re.sub(r"^[-*•]\s*", "", clean)

    # Strip common conversational / spoken introductory filler
    fillers = [
        r"^(?:let us|let\'s|we can|we will|we\'ll|now|here|first|second|third|finally|next|then)\s+",
        r"^(?:in this section,? we will|as we can see|as shown|to begin with|in other words|importantly)\s+",
        r"^(?:specifically|recall that|remember that|notice that|observe that|it turns out that)\s+",
        r"^(?:it is easy to see that|we can observe that|note that|we have|consider that)\s+",
        r"^(?:to understand this,?|to see this,?|notice how|see how|as an example)\s+",
    ]
    for pat in fillers:
        clean = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()

    # If there is a colon separating concept and detail, e.g. "Rule 1: origin never moves"
    if ":" in clean:
        parts = clean.split(":", 1)
        prefix = parts[0].strip()
        rest = parts[1].strip()
        if len(prefix.split()) <= 3 and len(prefix) > 2:
            rest_words = rest.split()
            suffix = " ".join(rest_words[:3]) if len(rest_words) > 3 else rest
            candidate = f"{prefix}: {suffix}".strip(" .,;:")
            if len(candidate.split()) <= max_words and len(candidate) <= max_chars:
                return candidate

    # Check for clause splits (commas, semicolons, dashes) safely
    parts = split_clauses_safe(clean)
    if len(parts) > 1:
        # If first clause starts with subordinate conjunction, main clause is second
        if parts[0].lower().startswith(("to ", "when ", "if ", "while ", "since ", "because ", "although ")):
            candidate = parts[1]
            words = candidate.split()
            if 2 <= len(words) <= max_words:
                clean = candidate
        else:
            # Check if first clause is self-contained
            words = parts[0].split()
            if 2 <= len(words) <= max_words:
                clean = parts[0]

    # Split into words and limit length
    words = clean.split()
    if len(words) > max_words:
        words = words[:max_words]

    # Remove dangling stop words from the end
    while words and words[-1].lower().rstrip(".,;:!?") in STOP_ENDINGS:
        words.pop()

    result = " ".join(words).strip(" .,;:!?")

    if len(result) > max_chars:
        # Truncate at word boundary
        truncated = result[:max_chars].rsplit(" ", 1)[0].strip(" .,;:!?")
        if truncated:
            result = truncated

    return result if result else text[:max_chars].strip()


def get_context_animation_type(text: str, animation_description: str = "") -> str:
    """
    Determine context-appropriate animation type (fade, slide, zoom, highlight)
    based on the educational content rather than choosing random effects.

    - Highlight: Key formulas, rules, theorems, critical vocabulary, emphasis
    - Zoom: Central entities, core geometric objects, diagrams, focal points
    - Slide: Sequential steps, movements, directional flow, vector translations
    - Fade: Introductions, definitions, background explanations, gentle context
    """
    combined = f"{text} {animation_description}".lower()

    # Highlight triggers
    highlight_patterns = [
        r"\b(?:important|critical|key|formula|equation|matrix|determinant|eigen|note|rule|theorem)\b",
        r"\b(?:highlight|indicate|flash|pulse|glow|color|label|symbol)\b",
        r"\b(?:\[.*\]|\=|\+|\-|\*)\b"
    ]
    for pat in highlight_patterns:
        if re.search(pat, combined):
            return "highlight"

    # Zoom triggers
    zoom_patterns = [
        r"\b(?:robot|circle|square|point|origin|center|core|focus|zoom|magnif|grow|scale|shape|diagram)\b",
        r"\b(?:mesh|grid|plane|space|primitives)\b"
    ]
    for pat in zoom_patterns:
        if re.search(pat, combined):
            return "zoom"

    # Slide triggers
    slide_patterns = [
        r"\b(?:move|shift|step|slide|translate|vector|direction|arrow|extend|flow|along|reach)\b",
        r"\b(?:horizontal|vertical|transform|shear|rotate|connect|then|next)\b"
    ]
    for pat in slide_patterns:
        if re.search(pat, combined):
            return "slide"

    # Default to Fade for conceptual, introductory, or descriptive statements
    return "fade"


class AcousticPauseEngine:
    """
    Acoustic Pause Engine for CineLearn.
    Generates natural ambient room-tone presence and calculates context-sensitive pedagogical pauses.
    Eliminates digital zero-amplitude vacuum cutoffs and acoustic clicks between narration lines.

    Note on audio calibration:
    Digital zero-amplitude silence is -inf dBFS, whereas 0 dBFS represents full-scale digital clipping.
    AcousticPauseEngine generates gentle comfort presence at calibrated -58 dBFS (~0.125% of full-scale amplitude).
    """

    def __init__(self, ambient_dbfs: float = -58.0, sample_rate: int = 44100):
        self.ambient_dbfs = ambient_dbfs
        self.sample_rate = sample_rate

    def validate_audio_presence(
        self,
        audio: AudioSegment,
        expected_min_dbfs: float = -70.0,
        expected_max_dbfs: float = -45.0
    ) -> Dict[str, Any]:
        """Automated acoustic check verifying audio segment has natural presence without DC bias or clipping."""
        if len(audio) == 0:
            return {"is_valid": False, "error": "Audio segment is empty"}
        dbfs = audio.dBFS
        is_presence_valid = (dbfs != float("-inf") and expected_min_dbfs <= dbfs <= expected_max_dbfs)
        return {
            "is_valid": is_presence_valid,
            "measured_dbfs": round(dbfs, 2) if dbfs != float("-inf") else None,
            "duration_ms": len(audio),
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
        }

    def generate_room_tone(
        self,
        duration_sec: float,
        sample_rate: Optional[int] = None,
        dbfs: Optional[float] = None,
    ) -> AudioSegment:
        """
        Generates deterministic, subtle ambient room presence at specified dBFS.
        Uses deterministic pseudorandom samples (fixed seed) so caching and audio digests remain repeatable.
        """
        sr = sample_rate or self.sample_rate
        db = dbfs or self.ambient_dbfs
        dur_ms = max(50, int(duration_sec * 1000))
        num_samples = int(sr * (dur_ms / 1000.0))

        rng = random.Random(42)
        amp = max(1, int(32767.0 * (10.0 ** (db / 20.0))))
        raw_data = array.array("h", [rng.randint(-amp, amp) for _ in range(num_samples * 2)]).tobytes()
        return AudioSegment(data=raw_data, sample_width=2, frame_rate=sr, channels=2)

    def calculate_pause_duration(
        self,
        prev_text: str = "",
        next_text: str = "",
        is_section_end: bool = False,
        story_beat_change: bool = False,
        explicit_pause: Optional[float] = None,
    ) -> float:
        """
        Calculates context-sensitive pedagogical pause duration:
        - Explicit override preserved if specified > 0
        - Section boundary / major transition: 1.0s - 1.2s
        - Story beat change: 0.8s - 1.0s
        - Period / question / exclamation mark: 0.5s - 0.6s
        - Comma / clause / colon / dash: 0.3s - 0.4s
        - Default: 0.5s
        """
        if explicit_pause is not None and explicit_pause > 0:
            return float(explicit_pause)
        if is_section_end:
            return 1.0
        if story_beat_change:
            return 0.8
        clean = (prev_text or "").strip()
        if clean.endswith((".", "!", "?")):
            return 0.55
        if clean.endswith((",", ";", ":", "—", "-")):
            return 0.35
        return 0.50

    def create_natural_pause(
        self,
        duration_sec: float,
        sample_rate: Optional[int] = None,
        dbfs: Optional[float] = None,
    ) -> AudioSegment:
        """Creates an AudioSegment with natural room tone for the specified duration."""
        if duration_sec <= 0.001:
            return AudioSegment.empty()
        return self.generate_room_tone(duration_sec, sample_rate, dbfs)

    def crossfade_speech(
        self,
        seg1: AudioSegment,
        seg2: AudioSegment,
        pause_sec: float = 0.5,
        fade_ms: int = 40,
    ) -> AudioSegment:
        """Joins two speech segments with smooth boundary fades and natural acoustic pause."""
        s1 = seg1.fade_out(fade_ms) if len(seg1) > fade_ms * 2 else seg1
        s2 = seg2.fade_in(fade_ms) if len(seg2) > fade_ms * 2 else seg2
        sr = seg1.frame_rate if hasattr(seg1, "frame_rate") else 44100
        pause = self.create_natural_pause(pause_sec, sample_rate=sr)
        return s1 + pause + s2


class TTSFailureError(RuntimeError):
    """
    Raised when speech synthesis fails in fail-closed mode.
    Ensures callers do not mistake room tone or empty audio for synthesized speech.
    """
    def __init__(
        self,
        message: str,
        cue: Optional[str] = None,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.cue = cue
        self.provider = provider
        self.original_error = original_error


class SpeechSynthesisResult(float):
    """
    Typed result of a speech synthesis operation, subclassing float for 100% backward compatibility.

    Can be used directly as a float (duration in seconds) by legacy callers:
        duration = generate_speech(...)
        assert duration > 0.0
    While providing rich inspection attributes for modern native pipelines:
        result.is_speech (bool)
        result.is_fallback (bool)
        result.output_path (str)
        result.provider (str)
        result.error (Optional[str])
    """
    duration: float
    is_speech: bool
    is_fallback: bool
    output_path: Optional[str]
    provider: Optional[str]
    error: Optional[str]
    cue: Optional[str]

    def __new__(
        cls,
        duration: float,
        is_speech: bool = True,
        is_fallback: bool = False,
        output_path: Optional[str] = None,
        provider: Optional[str] = None,
        error: Optional[str] = None,
        cue: Optional[str] = None,
    ):
        instance = super().__new__(cls, float(duration))
        instance.duration = float(duration)
        instance.is_speech = bool(is_speech)
        instance.is_fallback = bool(is_fallback)
        instance.output_path = output_path
        instance.provider = provider
        instance.error = error
        instance.cue = cue
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration": self.duration,
            "is_speech": self.is_speech,
            "is_fallback": self.is_fallback,
            "output_path": self.output_path,
            "provider": self.provider,
            "error": self.error,
            "cue": self.cue,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __repr__(self) -> str:
        return (
            f"SpeechSynthesisResult(duration={self.duration:.3f}, "
            f"is_speech={self.is_speech}, is_fallback={self.is_fallback}, "
            f"provider='{self.provider}')"
        )


class AudioManager:
    """
    Manages speech generation, audio input/output, timestamp generation,
    synchronized subtitles, and video-audio alignment for Code2Video.
    """

    def __init__(self, tts_voice: str = "en-US-GuyNeural", tts_rate: str = "-4%"):
        self.tts_voice = tts_voice
        self.tts_rate = tts_rate
        self.pause_engine = AcousticPauseEngine()
        self._last_speech_was_fallback: bool = False

    async def _generate_speech_edge_tts(
        self, text: str, output_path: str, voice: str, rate: Optional[str] = None
    ) -> float:
        """Generate audio file via edge-tts and return duration in seconds."""
        import edge_tts

        rate = rate or self.tts_rate
        communicator = edge_tts.Communicate(text, voice, rate=rate)
        with open(output_path, "wb") as f:
            async for chunk in communicator.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])

        # Determine exact duration using pydub
        audio = AudioSegment.from_file(output_path)
        return len(audio) / 1000.0

    def generate_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        allow_fallback: bool = True,
    ) -> SpeechSynthesisResult:
        """
        Synchronous call to generate speech file for a text line.

        Args:
            text: Text cue to synthesize.
            output_path: Destination path for speech audio file.
            voice: Voice override identifier.
            rate: Rate override identifier.
            allow_fallback: If True (legacy default), falls back to calibrated
                room-tone presence on Edge-TTS failure. If False (strict native mode),
                raises TTSFailureError fail-closed without writing fallback silence.

        Returns:
            SpeechSynthesisResult (subclasses float): duration in seconds with metadata attributes.

        Raises:
            TTSFailureError: If synthesis fails and allow_fallback is False.
        """
        voice = voice or self.tts_voice
        rate = rate or self.tts_rate
        output_path = str(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        try:
            dur = asyncio.run(self._generate_speech_edge_tts(text, output_path, voice, rate=rate))
            self._last_speech_was_fallback = False
            return SpeechSynthesisResult(
                duration=dur,
                is_speech=True,
                is_fallback=False,
                output_path=output_path,
                provider="edge-tts",
                cue=text,
            )
        except Exception as e:
            self._last_speech_was_fallback = True
            if not allow_fallback:
                if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                    try:
                        os.unlink(output_path)
                    except OSError:
                        pass
                raise TTSFailureError(
                    f"Edge-TTS speech synthesis failed for cue '{text}': {e}",
                    cue=text,
                    provider="edge-tts",
                    original_error=e,
                ) from e

            print(f"⚠️ edge-tts failed ({e}), creating fallback silence audio")
            # Fallback: estimate duration based on words (avg 150 wpm = 2.5 words/sec)
            words_count = max(1, len(text.split()))
            est_duration_sec = max(1.5, words_count / 2.5)
            # Create natural room tone audio file
            silence = self.pause_engine.create_natural_pause(est_duration_sec)
            silence.export(output_path, format="mp3")
            return SpeechSynthesisResult(
                duration=est_duration_sec,
                is_speech=False,
                is_fallback=True,
                output_path=output_path,
                provider="room-tone-fallback",
                error=str(e),
                cue=text,
            )

    def generate_speech_structured(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        allow_fallback: bool = False,
    ) -> SpeechSynthesisResult:
        """
        Structured invocation of speech synthesis defaulting to strict fail-closed mode.
        """
        return self.generate_speech(
            text=text,
            output_path=output_path,
            voice=voice,
            rate=rate,
            allow_fallback=allow_fallback,
        )

    def generate_section_audio(
        self,
        section_id: str,
        lecture_lines: List[str],
        animations: List[str] = None,
        captions: List[str] = None,
        output_dir: Path = None,
        voice: str = None,
        inter_line_pause: float = 0.5,
        section_end_pause: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generates audio for each lecture line in a section, combines them with
        pedagogical pauses, and computes fine-grained timestamps.

        Returns metadata dict containing:
        - audio_path: path to section combined audio
        - total_duration: float duration in seconds
        - line_timestamps: list of line timestamp dicts
        - captions: list of concise captions
        - animations_context: list of context-appropriate animation types
        """
        output_dir = Path(output_dir) if output_dir else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = output_dir / "audio_lines"
        audio_dir.mkdir(parents=True, exist_ok=True)

        target_voice = voice or self.tts_voice
        target_rate = self.tts_rate
        target_inter = round(float(inter_line_pause), 3)
        target_end = round(float(section_end_pause), 3)
        target_dbfs = round(float(self.pause_engine.ambient_dbfs), 1)
        expected_audio_config = {
            "voice": target_voice,
            "rate": target_rate,
            "inter_line_pause": target_inter,
            "section_end_pause": target_end,
            "ambient_dbfs": target_dbfs,
        }

        # Fast cache check: if section audio and timestamps already exist, reuse immediately ONLY IF narration and config match
        ts_path = output_dir / f"{section_id}_timestamps.json"
        section_audio_path = output_dir / f"{section_id}_audio.mp3"
        current_narr_hash = compute_narration_hash(lecture_lines)
        is_cache_valid = False
        if ts_path.exists() and section_audio_path.exists() and section_audio_path.stat().st_size > 1000:
            try:
                with open(ts_path, "r", encoding="utf-8") as f:
                    cached_meta = json.load(f)
                    cached_hash = cached_meta.get("narration_hash")
                    cached_cfg = cached_meta.get("audio_config")
                    config_matches = (cached_cfg is None or cached_cfg == expected_audio_config)
                    if cached_hash == current_narr_hash and config_matches:
                        is_cache_valid = True
                    elif not cached_hash and cached_meta.get("lines") and len(cached_meta["lines"]) == len(lecture_lines):
                        # Fallback text equality check for legacy timestamp files without hash
                        if all(
                            (line_m.get("raw_text") or line_m.get("text", "")).strip() == exp_l.strip()
                            for line_m, exp_l in zip(cached_meta["lines"], lecture_lines)
                        ) and config_matches:
                            cached_meta["narration_hash"] = current_narr_hash
                            cached_meta["audio_config"] = expected_audio_config
                            is_cache_valid = True
                    if is_cache_valid:
                        return cached_meta
            except Exception:
                pass

        # If cache was stale, invalidate and purge old audio artifacts so stale speech is never reused
        if not is_cache_valid:
            if ts_path.exists():
                try:
                    ts_path.unlink()
                except Exception:
                    pass
            if section_audio_path.exists():
                try:
                    section_audio_path.unlink()
                except Exception:
                    pass
            if audio_dir.exists():
                for old_line_audio in audio_dir.glob(f"{section_id}_line_*.mp3"):
                    try:
                        old_line_audio.unlink()
                    except Exception:
                        pass

        voice = voice or self.tts_voice
        animations = animations or []

        combined_audio = AudioSegment.empty()
        line_timestamps = []
        concise_captions = []
        anim_types = []

        current_time = 0.0

        for i, line in enumerate(lecture_lines):
            # 1. Concise caption extraction (do not display full script on screen)
            if captions and i < len(captions) and captions[i]:
                caption = captions[i]
            else:
                caption = extract_concise_caption(line)
            concise_captions.append(caption)

            # 2. Context-appropriate animation type
            anim_desc = animations[i] if i < len(animations) else ""
            anim_type = get_context_animation_type(line, anim_desc)
            anim_types.append(anim_type)

            # 3. Generate speech for this line (check cache first)
            line_audio_path = audio_dir / f"{section_id}_line_{i+1}.mp3"
            if line_audio_path.exists() and line_audio_path.stat().st_size > 500:
                duration = len(AudioSegment.from_file(str(line_audio_path))) / 1000.0
            else:
                duration = self.generate_speech(line, str(line_audio_path), voice)

            line_start = current_time
            line_end = line_start + duration
            pause = inter_line_pause if i < len(lecture_lines) - 1 else section_end_pause

            line_timestamps.append({
                "section_id": section_id,
                "line_index": i,
                "raw_text": line,
                "caption": caption,
                "animation_type": anim_type,
                "start": round(line_start, 3),
                "end": round(line_end, 3),
                "duration": round(duration, 3),
                "total_budget": round(duration + pause, 3)
            })

            # Append audio
            line_segment = AudioSegment.from_file(str(line_audio_path))
            combined_audio += line_segment

            # Add inter-line natural acoustic pause
            if i < len(lecture_lines) - 1:
                next_l = lecture_lines[i + 1] if i + 1 < len(lecture_lines) else ""
                actual_pause = self.pause_engine.calculate_pause_duration(
                    prev_text=line,
                    next_text=next_l,
                    explicit_pause=inter_line_pause,
                )
                if actual_pause > 0:
                    sr = combined_audio.frame_rate if len(combined_audio) > 0 else 44100
                    combined_audio += self.pause_engine.create_natural_pause(actual_pause, sample_rate=sr)
                    current_time = line_end + actual_pause
                else:
                    current_time = line_end
            else:
                current_time = line_end

        # Add section end natural pause
        if section_end_pause > 0:
            sr = combined_audio.frame_rate if len(combined_audio) > 0 else 44100
            combined_audio += self.pause_engine.create_natural_pause(section_end_pause, sample_rate=sr)
            current_time += section_end_pause

        # Build and update structured timing contract
        contract = SceneTimingContract.from_storyboard_section({
            "id": section_id,
            "lecture_lines": lecture_lines,
            "animations": animations,
            "captions": concise_captions,
        })
        line_durs = [item["duration"] for item in line_timestamps]
        line_paths = [str(audio_dir / f"{section_id}_line_{i+1}.mp3") for i in range(len(lecture_lines))]
        contract.update_actual_audio_durations(line_durs, line_paths)
        contract.audio_path = str(section_audio_path)

        section_audio_path = output_dir / f"{section_id}_audio.mp3"
        tmp_audio_path = output_dir / f"{section_id}_audio.tmp.{os.getpid()}.mp3"
        ts_path = output_dir / f"{section_id}_timestamps.json"
        tmp_ts_path = output_dir / f"{section_id}_timestamps.tmp.{os.getpid()}.json"
        committed = False
        try:
            combined_audio.export(str(tmp_audio_path), format="mp3")
            total_duration = len(combined_audio) / 1000.0

            section_meta = {
                "section_id": section_id,
                "audio_path": str(section_audio_path),
                "total_duration": round(total_duration, 3),
                "lines": line_timestamps,
                "concise_captions": concise_captions,
                "animation_types": anim_types,
                "narration_hash": current_narr_hash,
                "audio_config": expected_audio_config,
                "timing_contract": contract.to_dict(),
            }

            # Save section timestamps and audio atomically
            with open(tmp_ts_path, "w", encoding="utf-8") as f:
                json.dump(section_meta, f, indent=2, ensure_ascii=False)

            tmp_audio_path.replace(section_audio_path)
            tmp_ts_path.replace(ts_path)
            committed = True
        finally:
            if not committed:
                if tmp_audio_path.exists():
                    try:
                        tmp_audio_path.unlink()
                    except Exception:
                        pass
                if tmp_ts_path.exists():
                    try:
                        tmp_ts_path.unlink()
                    except Exception:
                        pass

        # Generate section SRT
        srt_path = output_dir / f"{section_id}.srt"
        self.export_srt(line_timestamps, str(srt_path))

        return section_meta

    def generate_scene_ir_audio(
        self,
        scene_ir: Any,
        output_dir: Union[str, Path],
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        inter_cue_pause: float = 0.5,
        scene_end_pause: float = 1.0,
        tts_provider: Optional[Callable[[str, str, str, Optional[str]], float]] = None,
        force_refresh: bool = False,
        provider_config: Optional[Dict[str, Any]] = None,
        allow_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Generates TTS speech for a native SceneIR document without storyboard adapters.
        Updates scene_ir in-place with AudioReference and measured cue-level timestamp ranges.
        Returns the section audio metadata dictionary.
        """
        synthesizer = NativeSceneAudioSynthesizer(
            audio_manager=self,
            tts_provider=tts_provider,
            voice=voice or self.tts_voice,
            rate=rate or self.tts_rate,
            inter_cue_pause=inter_cue_pause,
            scene_end_pause=scene_end_pause,
            allow_fallback=allow_fallback,
        )
        _, meta = synthesizer.synthesize_scene_ir(
            scene_ir=scene_ir,
            output_dir=output_dir,
            voice=voice,
            rate=rate,
            inter_cue_pause=inter_cue_pause,
            scene_end_pause=scene_end_pause,
            tts_provider=tts_provider,
            force_refresh=force_refresh,
            provider_config=provider_config,
            allow_fallback=allow_fallback,
        )
        return meta

    def generate_scene_sequence_audio(
        self,
        sequence_ir: Any,
        output_dir: Union[str, Path],
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        inter_cue_pause: float = 0.5,
        scene_end_pause: float = 1.0,
        tts_provider: Optional[Callable[[str, str, str, Optional[str]], float]] = None,
        force_refresh: bool = False,
        provider_config: Optional[Dict[str, Any]] = None,
        allow_fallback: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generates TTS speech for all sections in a SceneSequenceIR document.
        Updates each section in-place with AudioReference and measured cue ranges.
        Returns a list of section metadata dictionaries.
        """
        synthesizer = NativeSceneAudioSynthesizer(
            audio_manager=self,
            tts_provider=tts_provider,
            voice=voice or self.tts_voice,
            rate=rate or self.tts_rate,
            inter_cue_pause=inter_cue_pause,
            scene_end_pause=scene_end_pause,
            allow_fallback=allow_fallback,
        )
        _, metas = synthesizer.synthesize_sequence_ir(
            sequence_ir=sequence_ir,
            output_dir=output_dir,
            voice=voice,
            rate=rate,
            inter_cue_pause=inter_cue_pause,
            scene_end_pause=scene_end_pause,
            tts_provider=tts_provider,
            force_refresh=force_refresh,
            provider_config=provider_config,
            allow_fallback=allow_fallback,
        )
        return metas

    def handle_external_audio(
        self,
        section_id: str,
        audio_file_path: Path,
        lecture_lines: List[str],
        animations: List[str] = None,
        captions: List[str] = None,
        output_dir: Path = None,
    ) -> Dict[str, Any]:
        """
        Handles user-provided external audio for a section.
        Extracts duration and maps timestamps proportionally across lines.
        """
        output_dir = Path(output_dir) if output_dir else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        audio = AudioSegment.from_file(str(audio_file_path))
        total_duration = len(audio) / 1000.0

        dest_audio = output_dir / f"{section_id}_audio.mp3"
        tmp_dest_audio = output_dir / f"{section_id}_audio.tmp.{os.getpid()}.mp3"
        ts_path = output_dir / f"{section_id}_timestamps.json"
        tmp_ts_path = output_dir / f"{section_id}_timestamps.tmp.{os.getpid()}.json"
        committed = False
        try:
            audio.export(str(tmp_dest_audio), format="mp3")

            animations = animations or []
            n_lines = max(1, len(lecture_lines))
            # Distribute time proportionally to character length
            char_counts = [max(1, len(line)) for line in lecture_lines]
            total_chars = sum(char_counts)
            ratios = [c / total_chars for c in char_counts]

            line_timestamps = []
            concise_captions = []
            anim_types = []
            current_time = 0.0

            for i, line in enumerate(lecture_lines):
                caption = captions[i] if (captions and i < len(captions)) else extract_concise_caption(line)
                concise_captions.append(caption)

                anim_desc = animations[i] if i < len(animations) else ""
                anim_type = get_context_animation_type(line, anim_desc)
                anim_types.append(anim_type)

                dur = total_duration * ratios[i]
                l_start = current_time
                l_end = l_start + dur
                current_time = l_end

                line_timestamps.append({
                    "section_id": section_id,
                    "line_index": i,
                    "raw_text": line,
                    "caption": caption,
                    "animation_type": anim_type,
                    "start": round(l_start, 3),
                    "end": round(l_end, 3),
                    "duration": round(dur, 3),
                })

            contract = SceneTimingContract.from_storyboard_section({
                "id": section_id,
                "lecture_lines": lecture_lines,
                "animations": animations,
                "captions": concise_captions,
            })
            contract.update_actual_audio_durations([item["duration"] for item in line_timestamps])
            contract.audio_path = str(dest_audio)

            section_meta = {
                "section_id": section_id,
                "audio_path": str(dest_audio),
                "total_duration": round(total_duration, 3),
                "lines": line_timestamps,
                "concise_captions": concise_captions,
                "animation_types": anim_types,
                "narration_hash": compute_narration_hash(lecture_lines),
                "timing_contract": contract.to_dict(),
            }

            with open(tmp_ts_path, "w", encoding="utf-8") as f:
                json.dump(section_meta, f, indent=2, ensure_ascii=False)

            tmp_dest_audio.replace(dest_audio)
            tmp_ts_path.replace(ts_path)
            committed = True
        finally:
            if not committed:
                if tmp_dest_audio.exists():
                    try:
                        tmp_dest_audio.unlink()
                    except Exception:
                        pass
                if tmp_ts_path.exists():
                    try:
                        tmp_ts_path.unlink()
                    except Exception:
                        pass

        srt_path = output_dir / f"{section_id}.srt"
        self.export_srt(line_timestamps, str(srt_path))

        return section_meta

    @staticmethod
    def get_media_duration(file_path: str) -> float:
        """Get exact duration in seconds using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
        # Fallback to pydub if audio
        try:
            return len(AudioSegment.from_file(file_path)) / 1000.0
        except Exception:
            return 0.0

    @staticmethod
    def extract_rendered_scene_timings(
        section_id: str,
        video_path: Optional[str] = None,
        code_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts actual rendered animation timing per lecture line.
        1. Checks for ground-truth execution metadata from TeachingScene
           (timing_{scene_name}.json or layout_{scene_name}.json).
        2. Falls back to static code block inspection (summing run_times and waits).
        """
        base_dir = Path(output_dir) if output_dir else (Path(video_path).parent if video_path else Path("."))
        scene_name = f"{section_id.title().replace('_', '')}Scene"

        # 1. Check execution metadata from TeachingScene
        timing_file = base_dir / f"timing_{scene_name}.json"
        layout_file = base_dir / f"layout_{scene_name}.json"
        sec_timing_file = base_dir / f"{section_id}_timing.json"
        for cand in [timing_file, sec_timing_file, layout_file]:
            if cand.exists():
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    lt = data.get("line_timings")
                    if lt and isinstance(lt, list) and len(lt) > 0:
                        return lt
                except Exception:
                    pass

        # 2. Check code block timings via AST / regex inspection
        if code_path is None:
            cand_code = base_dir / f"{section_id}.py"
            if cand_code.exists():
                code_path = str(cand_code)

        if code_path and os.path.exists(code_path):
            try:
                with open(code_path, "r", encoding="utf-8") as f:
                    code_text = f.read()
                blocks = re.split(r"#\s*===+\s*Animation for Lecture Line\s*(\d+)\s*===+", code_text, flags=re.IGNORECASE)
                if len(blocks) >= 3:
                    results = []
                    current_t = 0.0
                    for i in range(1, len(blocks) - 1, 2):
                        idx = int(blocks[i]) - 1
                        block_code = blocks[i + 1]
                        run_times = [float(m) for m in re.findall(r"run_time\s*=\s*([0-9.]+)", block_code)]
                        waits = [float(m) for m in re.findall(r"self\.wait\(\s*([0-9.]+)\s*\)", block_code)]
                        dur = max(0.1, sum(run_times) + sum(waits))
                        results.append({
                            "line_index": idx,
                            "start_time": round(current_t, 3),
                            "end_time": round(current_t + dur, 3),
                            "duration": round(dur, 3),
                        })
                        current_t += dur
                    return results
            except Exception:
                pass

        return []

    @staticmethod
    def validate_media_file(
        file_path: str,
        expected_duration: Optional[float] = None,
        tolerance: float = 0.5,
        require_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Deterministically validates an assembled media file via ffprobe:
        - Verifies physical file exists and size >= 1000 bytes.
        - Verifies video stream is present with valid codec and dimensions.
        - Verifies audio stream is present with valid sampling.
        - Validates overall duration against expected tolerance.
        """
        p = Path(file_path)
        if not p.exists():
            return {"is_valid": False, "error": f"File '{file_path}' does not exist"}
        if p.stat().st_size < 1000:
            return {"is_valid": False, "error": f"File '{file_path}' is too small ({p.stat().st_size} bytes)"}

        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,codec_type,codec_name,width,height,sample_rate,channels",
            "-show_entries", "format=duration,size,bit_rate",
            "-of", "json",
            str(file_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return {"is_valid": False, "error": f"ffprobe error: {res.stderr}"}

        try:
            info = json.loads(res.stdout)
        except Exception as e:
            return {"is_valid": False, "error": f"ffprobe JSON decode error: {e}"}

        streams = info.get("streams", [])
        fmt = info.get("format", {})
        has_video = any(s.get("codec_type") == "video" for s in streams)
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        raw_dur = fmt.get("duration")
        duration = float(raw_dur) if raw_dur else 0.0

        if not has_video:
            return {"is_valid": False, "error": "No video stream found in file"}
        if require_audio and not has_audio:
            return {"is_valid": False, "error": "No audio stream found in file"}
        if duration <= 0.05:
            return {"is_valid": False, "error": f"Invalid media duration: {duration}s"}

        if expected_duration is not None and expected_duration > 0:
            diff = abs(duration - expected_duration)
            if diff > tolerance:
                return {
                    "is_valid": False,
                    "error": f"Duration discrepancy ({duration:.2f}s vs expected {expected_duration:.2f}s, diff {diff:.2f}s > {tolerance}s)",
                    "duration": duration,
                }

        return {
            "is_valid": True,
            "duration": round(duration, 3),
            "has_video": has_video,
            "has_audio": has_audio,
            "streams_count": len(streams),
        }

    def mux_video_and_audio(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        srt_path: Optional[Union[str, Path]] = None,
        burn_subtitles: bool = True,
        expected_duration: Optional[float] = None,
        tolerance: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Hardened FFmpeg A/V multiplexing for modern native Scene IR pipeline.
        - Validates existence and stream validity of raw video and speech audio inputs.
        - Aligns video and audio durations (freeze-frame last frame or natural room-tone pad).
        - Optionally burns synchronized subtitles via FFmpeg subtitles filter.
        - Stages multiplexing into PID- and thread-isolated temporary files.
        - Deterministically validates multiplexed media via ffprobe and full decode check.
        - Atomically replaces target output on verified success.
        - Cleans up all intermediate temporary files on success and failure.
        - Returns structured result dict with metadata and error diagnostics.
        """
        v_path = Path(video_path).resolve()
        a_path = Path(audio_path).resolve()
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Validate inputs
        if not v_path.exists() or v_path.stat().st_size == 0:
            return {"success": False, "error": f"Video input does not exist or is empty: {v_path}"}
        v_val = self.validate_media_file(str(v_path), require_audio=False)
        if not v_val.get("is_valid"):
            return {"success": False, "error": f"Invalid video input: {v_val.get('error')}"}

        if not a_path.exists() or a_path.stat().st_size == 0:
            return {"success": False, "error": f"Audio input does not exist or is empty: {a_path}"}
        a_dur = self.get_media_duration(str(a_path))
        if a_dur <= 0.05:
            return {"success": False, "error": f"Audio input has invalid duration ({a_dur:.3f}s): {a_path}"}

        v_dur = float(v_val.get("duration", 0.0))
        tid = threading.get_ident()
        pid = os.getpid()

        tmp_v = out_path.parent / f"{out_path.stem}.tmp_v.{pid}.{tid}.mp4"
        tmp_a = out_path.parent / f"{out_path.stem}.tmp_a.{pid}.{tid}.mp3"
        tmp_out = out_path.parent / f"{out_path.stem}.tmp_out.{pid}.{tid}.mp4"
        temps_to_clean: List[Path] = []

        try:
            # 2. Temporal alignment
            aligned_video = v_path
            aligned_audio = a_path

            if v_dur < a_dur - 0.05:
                # Video shorter than audio -> freeze last frame
                freeze_dur = a_dur - v_dur
                cmd_freeze = [
                    "ffmpeg", "-y",
                    "-i", str(v_path),
                    "-vf", f"tpad=stop_mode=clone:stop_duration={freeze_dur:.3f}",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                    str(tmp_v),
                ]
                res_f = subprocess.run(cmd_freeze, capture_output=True, text=True)
                if res_f.returncode != 0:
                    return {"success": False, "error": f"FFmpeg video freeze-frame failed ({res_f.returncode}): {res_f.stderr[-500:]}"}
                temps_to_clean.append(tmp_v)
                aligned_video = tmp_v
            elif a_dur < v_dur - 0.05:
                # Audio shorter than video -> pad with natural room tone
                pad_sec = v_dur - a_dur
                orig_audio = AudioSegment.from_file(str(a_path))
                sr = orig_audio.frame_rate if len(orig_audio) > 0 else 44100
                padded = orig_audio + self.pause_engine.create_natural_pause(pad_sec, sample_rate=sr)
                padded.export(str(tmp_a), format="mp3")
                temps_to_clean.append(tmp_a)
                aligned_audio = tmp_a

            # 3. Multiplexing
            sub_filter = None
            if burn_subtitles and srt_path:
                srt_p = Path(srt_path).resolve()
                if srt_p.exists() and srt_p.stat().st_size > 0:
                    escaped_srt = str(srt_p).replace("\\", "/").replace(":", "\\:")
                    sub_filter = (
                        f"subtitles='{escaped_srt}':"
                        "force_style='FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,MarginV=25,Alignment=2'"
                    )

            cmd_mux = [
                "ffmpeg", "-y",
                "-i", str(aligned_video),
                "-i", str(aligned_audio),
                *(["-vf", sub_filter] if sub_filter else []),
                "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(tmp_out),
            ]
            temps_to_clean.append(tmp_out)
            res_mux = subprocess.run(cmd_mux, capture_output=True, text=True)
            if res_mux.returncode != 0:
                cmd_copy = [
                    "ffmpeg", "-y",
                    "-i", str(aligned_video),
                    "-i", str(aligned_audio),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    str(tmp_out),
                ]
                res_copy = subprocess.run(cmd_copy, capture_output=True, text=True)
                if res_copy.returncode != 0:
                    return {"success": False, "error": f"FFmpeg A/V mux failed ({res_mux.returncode}): {res_mux.stderr[-500:]}"}

            # 4. Deterministic verification via ffprobe and decode pass
            val_res = self.validate_media_file(
                str(tmp_out),
                expected_duration=expected_duration,
                tolerance=tolerance,
                require_audio=True,
            )
            if not val_res.get("is_valid"):
                return {"success": False, "error": f"Muxed media validation failed: {val_res.get('error')}"}

            decode_cmd = ["ffmpeg", "-v", "error", "-i", str(tmp_out), "-f", "null", "-"]
            res_dec = subprocess.run(decode_cmd, capture_output=True, text=True)
            if res_dec.returncode != 0:
                return {"success": False, "error": f"Muxed stream decode check failed: {res_dec.stderr[-500:]}"}

            # 5. Atomic replacement
            tmp_out.replace(out_path)
            if tmp_out in temps_to_clean:
                temps_to_clean.remove(tmp_out)

            return {
                "success": True,
                "output_path": str(out_path),
                "duration": val_res.get("duration"),
                "has_video": val_res.get("has_video", True),
                "has_audio": val_res.get("has_audio", True),
            }

        except Exception as e:
            return {"success": False, "error": f"A/V multiplexing exception: {str(e)}"}
        finally:
            for tmp in temps_to_clean:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception:
                        pass

    def sync_video_and_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        srt_path: str = None,
        burn_subtitles: bool = True,
        timing_contract: Optional[Any] = None,
        section_id: Optional[str] = None,
    ) -> str:
        """
        Temporally synchronizes video and audio using the generic timing contract.
        - Reconciles narration speech and visual animations beat-by-beat.
        - Reconstructs synchronized audio stream where each line speech starts at its visual beat.
        - Aligns durations down to the millisecond (freeze-frame or silence padding).
        - Optionally burns synchronized concise subtitles onto the video.
        - Merges audio and video streams and deterministically validates via ffprobe.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        video_path = str(video_path)
        audio_path = str(audio_path)
        output_path = str(output_path)
        output_dir = Path(output_path).parent

        # 1. Infer section_id if not explicitly provided
        if not section_id:
            m = re.search(r"(section_\d+)", Path(video_path).name.lower()) or re.search(r"(section_\d+)", Path(audio_path).name.lower())
            if m:
                section_id = m.group(1)

        # 2. Load or construct timing contract if available
        if timing_contract is None and section_id:
            for cand_dir in [output_dir, Path(audio_path).parent, Path(video_path).parent]:
                ts_cand = cand_dir / f"{section_id}_timestamps.json"
                if ts_cand.exists():
                    try:
                        with open(ts_cand, "r", encoding="utf-8") as f:
                            ts_data = json.load(f)
                        timing_contract = SceneTimingContract.from_dict(ts_data)
                        break
                    except Exception:
                        pass

        # 3. Synchronize audio stream with visual line boundaries if timing contract is present
        v_dur = self.get_media_duration(video_path)
        if timing_contract and hasattr(timing_contract, "segments") and len(timing_contract.segments) > 0 and section_id:
            # Check for individual line audio files
            audio_dir = output_dir / "audio_lines"
            if not audio_dir.exists() and (Path(audio_path).parent / "audio_lines").exists():
                audio_dir = Path(audio_path).parent / "audio_lines"

            line_files = []
            for i, seg in enumerate(timing_contract.segments):
                cand = Path(seg.audio_file_path) if seg.audio_file_path else (audio_dir / f"{section_id}_line_{i+1}.mp3")
                if cand.exists() and cand.stat().st_size > 300:
                    line_files.append(cand)
                else:
                    break

            if len(line_files) == len(timing_contract.segments):
                # Extract rendered visual line timings from execution metadata or AST
                rendered_timings = self.extract_rendered_scene_timings(section_id, video_path, output_dir=output_dir)
                timing_contract.reconcile_with_rendered_video(v_dur, rendered_timings)

                # Reconstruct temporally aligned audio track
                synced_audio = AudioSegment.empty()
                cur_pos_ms = 0
                for i, seg in enumerate(timing_contract.segments):
                    line_seg = AudioSegment.from_file(str(line_files[i]))
                    target_start_ms = int(seg.timeline_start * 1000)
                    if target_start_ms > cur_pos_ms:
                        gap_sec = (target_start_ms - cur_pos_ms) / 1000.0
                        synced_audio += self.pause_engine.create_natural_pause(gap_sec, sample_rate=line_seg.frame_rate)
                        cur_pos_ms = target_start_ms
                    synced_audio += line_seg
                    cur_pos_ms += len(line_seg)

                # Pad trailing natural acoustic presence if needed
                target_total_ms = int(max(v_dur, timing_contract.total_assembled_duration) * 1000)
                if len(synced_audio) < target_total_ms:
                    pad_sec = (target_total_ms - len(synced_audio)) / 1000.0
                    sr = synced_audio.frame_rate if len(synced_audio) > 0 else 44100
                    synced_audio += self.pause_engine.create_natural_pause(pad_sec, sample_rate=sr)

                final_dur_sec = len(synced_audio) / 1000.0
                timing_contract.total_assembled_duration = final_dur_sec
                timing_contract.total_audio_duration = final_dur_sec

                # Export reconstructed synchronized audio atomically
                tmp_sync_audio = output_dir / f"{section_id}_audio.tmp.{os.getpid()}.mp3"
                final_sync_audio = output_dir / f"{section_id}_audio.mp3"
                synced_audio.export(str(tmp_sync_audio), format="mp3")
                tmp_sync_audio.replace(final_sync_audio)
                audio_path = str(final_sync_audio)

                # Atomically update timestamps JSON and SRT with reconciled timings
                ts_path = output_dir / f"{section_id}_timestamps.json"
                tmp_ts_path = output_dir / f"{section_id}_timestamps.tmp.{os.getpid()}.json"
                contract_dict = timing_contract.to_dict()
                contract_dict["audio_path"] = str(final_sync_audio)
                contract_dict["total_duration"] = round(final_dur_sec, 3)
                contract_dict["total_audio_duration"] = round(final_dur_sec, 3)
                with open(tmp_ts_path, "w", encoding="utf-8") as f:
                    json.dump(contract_dict, f, indent=2, ensure_ascii=False)
                tmp_ts_path.replace(ts_path)

                if srt_path:
                    self.export_srt(contract_dict["lines"], str(srt_path))

        a_dur = self.get_media_duration(audio_path)

        temp_video = str(output_path) + ".temp_v.mp4"
        temp_audio = str(output_path) + ".temp_a.mp3"

        # 4. Final duration alignment (freeze-frame or silence padding)
        if v_dur < a_dur - 0.05:
            # Video is shorter than audio -> hold last frame
            freeze_duration = a_dur - v_dur
            cmd_freeze = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vf", f"tpad=stop_mode=clone:stop_duration={freeze_duration:.3f}",
                "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                temp_video
            ]
            subprocess.run(cmd_freeze, capture_output=True, check=True)
            aligned_video = temp_video
            aligned_audio = audio_path
        elif a_dur < v_dur - 0.05:
            # Audio is shorter than video -> pad audio with natural room tone
            pad_sec = v_dur - a_dur
            orig_audio = AudioSegment.from_file(audio_path)
            padded = orig_audio + self.pause_engine.create_natural_pause(pad_sec, sample_rate=orig_audio.frame_rate)
            padded.export(temp_audio, format="mp3")
            aligned_video = video_path
            aligned_audio = temp_audio
        else:
            aligned_video = video_path
            aligned_audio = audio_path

        # 5. Multiplex audio and video (and burn subtitles if requested)
        if burn_subtitles and srt_path and os.path.exists(srt_path):
            escaped_srt = os.path.abspath(srt_path).replace("\\", "/").replace(":", "\\:")
            sub_filter = (
                f"subtitles='{escaped_srt}':"
                "force_style='FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,MarginV=25,Alignment=2'"
            )
            cmd_mux = [
                "ffmpeg", "-y",
                "-i", str(aligned_video),
                "-i", str(aligned_audio),
                "-vf", sub_filter,
                "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(output_path)
            ]
        else:
            cmd_mux = [
                "ffmpeg", "-y",
                "-i", str(aligned_video),
                "-i", str(aligned_audio),
                "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(output_path)
            ]

        res = subprocess.run(cmd_mux, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"⚠️ FFmpeg mux error ({res.stderr[:200]}), falling back to direct copy")
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", str(aligned_video),
                "-i", str(aligned_audio),
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output_path)
            ]
            subprocess.run(cmd_fallback, capture_output=True, check=True)

        # 6. Validate merged file via ffprobe
        val_res = self.validate_media_file(output_path, require_audio=True)
        if not val_res.get("is_valid"):
            print(f"⚠️ Video assembly validation warning: {val_res.get('error')}")

        # Clean temp files
        for tmp in [temp_video, temp_audio]:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

        return str(output_path)

    @staticmethod
    def export_srt(line_timestamps: List[Dict[str, Any]], output_path: str):
        """
        Export line timestamps to standard SubRip (.srt) subtitle file.
        Uses atomic file creation (PID-staged tmp file then os.replace)
        to prevent partial writes or corrupted subtitle artifacts.
        """
        subs = []
        for i, item in enumerate(line_timestamps, 1):
            start = timedelta(seconds=item["start"])
            end = timedelta(seconds=item["end"])
            # Display concise caption as the primary subtitle line
            caption = item.get("caption") or item.get("raw_text") or item.get("text") or ""
            subs.append(srt.Subtitle(index=i, start=start, end=end, content=str(caption)))

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.parent / f"{out_p.name}.tmp.{os.getpid()}"

        try:
            content = srt.compose(subs)
            with open(tmp_p, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_p), str(out_p))
        except Exception:
            if tmp_p.exists():
                try:
                    tmp_p.unlink()
                except Exception:
                    pass
            raise

    @staticmethod
    def export_vtt(line_timestamps: List[Dict[str, Any]], output_path: str):
        """Export line timestamps to WebVTT (.vtt) subtitle file."""
        def format_vtt_time(seconds: float) -> str:
            td = timedelta(seconds=seconds)
            total_seconds = int(td.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            secs = total_seconds % 60
            millis = int(td.microseconds / 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

        lines = ["WEBVTT\n"]
        for i, item in enumerate(line_timestamps, 1):
            start = format_vtt_time(item["start"])
            end = format_vtt_time(item["end"])
            caption = item.get("caption") or item.get("raw_text")
            lines.append(f"{i}\n{start} --> {end}\n{caption}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


class NativeSceneAudioSynthesizer:
    """
    Dedicated Text-to-Speech (TTS) synthesizer for CineLearn SceneIR and SceneSequenceIR documents.
    Operates directly on native IR models without storyboard adapters or SectionProxy dependencies.
    Provides deterministic cache identity, atomic artifact generation, and verified duration measurement.
    """

    def __init__(
        self,
        audio_manager: Optional[AudioManager] = None,
        tts_provider: Optional[Callable[[str, str, str, Optional[str]], float]] = None,
        voice: str = "en-US-GuyNeural",
        rate: str = "-4%",
        inter_cue_pause: float = 0.5,
        scene_end_pause: float = 1.0,
        provider_name: Optional[str] = None,
        allow_fallback: bool = False,
    ):
        self.audio_manager = audio_manager or AudioManager(tts_voice=voice, tts_rate=rate)
        self.tts_provider = tts_provider
        self.voice = voice
        self.rate = rate
        self.inter_cue_pause = inter_cue_pause
        self.scene_end_pause = scene_end_pause
        self.provider_name = provider_name
        self.allow_fallback = allow_fallback

    def synthesize_scene_ir(
        self,
        scene_ir: Any,
        output_dir: Union[str, Path],
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        inter_cue_pause: Optional[float] = None,
        scene_end_pause: Optional[float] = None,
        tts_provider: Optional[Callable[[str, str, str, Optional[str]], float]] = None,
        force_refresh: bool = False,
        provider_config: Optional[Dict[str, Any]] = None,
        allow_fallback: Optional[bool] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Synthesizes TTS narration for a single SceneIR document.
        
        Flow:
        1. Authoritatively extracts ordered narration cues via scene_ir.get_ordered_narration_cues().
        2. Computes deterministic narration cache identity using compute_scene_ir_narration_hash().
        3. If valid cached audio artifact exists and matches configuration, returns cache hit.
        4. On cache miss or invalidation, synthesizes each cue, measures actual duration, and stitches
           cues with natural acoustic pauses.
        5. Atomically writes the stitched audio, JSON timestamps, and SRT subtitles.
        6. Populates SceneIR.audio_reference and NarrationAnchor.audio_timestamp_range.
        """
        if not hasattr(scene_ir, "get_ordered_narration_cues") or not hasattr(scene_ir, "timeline"):
            raise TypeError(f"Expected SceneIR object with get_ordered_narration_cues(), got {type(scene_ir).__name__}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = output_dir / "audio_lines"
        audio_dir.mkdir(parents=True, exist_ok=True)

        scene_id = getattr(scene_ir, "scene_id", "scene_0")
        ordered_cues = scene_ir.get_ordered_narration_cues()

        # Handle empty narration scene cleanly and safely
        if not ordered_cues:
            scene_ir.audio_reference = None
            for event in getattr(scene_ir, "timeline", []):
                if getattr(event, "narration_anchor", None):
                    event.narration_anchor.audio_timestamp_range = None
            empty_meta = {
                "scene_id": scene_id,
                "has_narration": False,
                "total_duration": 0.0,
                "audio_path": None,
                "timestamps_path": None,
                "lines": [],
                "narration_hash": "",
                "audio_config": None,
                "cache_hit": False,
                "is_speech": True,
                "is_fallback": False,
                "message": "No narration cues present in SceneIR",
            }
            return scene_ir, empty_meta

        # Identify corresponding timeline events for ordered cues
        valid_events = []
        for idx, event in enumerate(scene_ir.timeline):
            anchor = getattr(event, "narration_anchor", None)
            if anchor and getattr(anchor, "text_cue", None):
                cue = anchor.text_cue
                if str(cue).strip():
                    valid_events.append((anchor.lecture_line_index, idx, event))
        valid_events.sort(key=lambda item: (item[0], item[1]))

        # Target configuration parameters
        target_voice = voice or self.voice
        target_rate = rate or self.rate
        target_inter = round(float(self.inter_cue_pause if inter_cue_pause is None else inter_cue_pause), 3)
        target_end = round(float(self.scene_end_pause if scene_end_pause is None else scene_end_pause), 3)
        target_provider = tts_provider or self.tts_provider
        target_allow_fallback = self.allow_fallback if allow_fallback is None else allow_fallback

        prov_name = self.provider_name
        if prov_name is None:
            if target_provider is not None:
                prov_name = getattr(target_provider, "provider_name", getattr(target_provider, "__name__", "custom_provider"))
            else:
                prov_name = "edge-tts"

        ambient_dbfs = round(float(self.audio_manager.pause_engine.ambient_dbfs), 1)
        expected_audio_config = {
            "provider": prov_name,
            "voice": target_voice,
            "rate": target_rate,
            "format": "mp3",
            "inter_cue_pause": target_inter,
            "scene_end_pause": target_end,
            "ambient_dbfs": ambient_dbfs,
            "allow_fallback": target_allow_fallback,
        }
        if provider_config:
            expected_audio_config.update(provider_config)

        # Narration cache hash computation
        current_narr_hash = compute_scene_ir_narration_hash(scene_ir, audio_config=expected_audio_config)

        scene_audio_path = output_dir / f"{scene_id}_audio.mp3"
        ts_path = output_dir / f"{scene_id}_timestamps.json"
        srt_path = output_dir / f"{scene_id}.srt"

        # Check existing cache
        is_cache_valid = False
        cached_meta = None
        if (
            not force_refresh
            and ts_path.exists()
            and ts_path.stat().st_size > 0
            and scene_audio_path.exists()
            and scene_audio_path.stat().st_size >= 1000
            and srt_path.exists()
            and srt_path.stat().st_size > 0
        ):
            try:
                # Subtitle artifact hardening: verify SRT contains valid non-empty content
                with open(srt_path, "r", encoding="utf-8") as sf:
                    srt_content = sf.read().strip()
                if not srt_content:
                    raise ValueError("Cached SRT subtitle artifact is empty.")

                with open(ts_path, "r", encoding="utf-8") as f:
                    cached_meta = json.load(f)
                cached_hash = cached_meta.get("narration_hash")
                cached_cfg = cached_meta.get("audio_config")
                cached_lines = cached_meta.get("lines", [])

                # Check for cached fallback status:
                is_fallback_cached = cached_meta.get("is_fallback", False) or any(
                    isinstance(line, dict) and line.get("is_fallback", False) for line in cached_lines
                )

                # Strict fail-closed policy: reject any fallback room-tone artifact
                if not target_allow_fallback and is_fallback_cached:
                    is_cache_valid = False
                elif (
                    cached_hash == current_narr_hash
                    and cached_cfg == expected_audio_config
                    and isinstance(cached_lines, list)
                    and len(cached_lines) == len(ordered_cues)
                ):
                    cached_seg = AudioSegment.from_file(str(scene_audio_path))
                    if len(cached_seg) > 0:
                        is_cache_valid = True
            except Exception:
                is_cache_valid = False

        if is_cache_valid and cached_meta:
            cached_meta["cache_hit"] = True
            cached_meta["srt_path"] = str(srt_path)
            cached_lines = cached_meta.get("lines", [])
            if AudioReference is not None:
                scene_ir.audio_reference = AudioReference(
                    audio_path=str(scene_audio_path),
                    timestamps_path=str(ts_path),
                    total_duration=cached_meta["total_duration"],
                    alignment_status=cached_meta.get("alignment_status", "unaligned"),
                )
            for k, item in enumerate(valid_events):
                if k < len(cached_lines):
                    line_data = cached_lines[k]
                    item[2].narration_anchor.audio_timestamp_range = (line_data["start"], line_data["end"])
            return scene_ir, cached_meta

        # On cache miss or invalidation: purge stale artifacts
        for p in [ts_path, scene_audio_path, srt_path]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
        if audio_dir.exists():
            for old_f in audio_dir.glob(f"{scene_id}_cue_*.mp3"):
                try:
                    old_f.unlink()
                except Exception:
                    pass

        # Synthesize cues
        combined_audio = AudioSegment.empty()
        line_timestamps = []
        current_time = 0.0
        tmp_files_to_clean = []

        try:
            for k, cue in enumerate(ordered_cues):
                line_audio_path = audio_dir / f"{scene_id}_cue_{k+1}.mp3"
                tmp_line_audio = audio_dir / f"{scene_id}_cue_{k+1}.tmp.{os.getpid()}.mp3"
                tmp_files_to_clean.append(tmp_line_audio)

                cue_is_fallback = False
                cue_is_speech = True

                # Synthesize cue via provider or default AudioManager.generate_speech
                if target_provider is not None:
                    prov_res = target_provider(cue, str(tmp_line_audio), target_voice, target_rate)
                else:
                    prov_res = self.audio_manager.generate_speech(
                        cue,
                        str(tmp_line_audio),
                        voice=target_voice,
                        rate=target_rate,
                        allow_fallback=target_allow_fallback,
                    )

                # Inspect provider result for fallback/speech flags
                if isinstance(prov_res, SpeechSynthesisResult):
                    cue_is_fallback = prov_res.is_fallback
                    cue_is_speech = prov_res.is_speech
                elif isinstance(prov_res, dict):
                    cue_is_fallback = bool(prov_res.get("is_fallback", False))
                    cue_is_speech = bool(prov_res.get("is_speech", not cue_is_fallback))
                elif getattr(self.audio_manager, "_last_speech_was_fallback", False) and target_provider is None:
                    cue_is_fallback = True
                    cue_is_speech = False

                if not target_allow_fallback and (cue_is_fallback or not cue_is_speech):
                    raise TTSFailureError(
                        f"Speech synthesis returned room-tone fallback for cue '{cue}', which is rejected in fail-closed mode.",
                        cue=cue,
                        provider=prov_name,
                    )

                if not tmp_line_audio.exists() or tmp_line_audio.stat().st_size == 0:
                    raise RuntimeError(f"TTS provider produced missing or empty audio file for cue {k+1}: '{cue}'")

                try:
                    line_seg = AudioSegment.from_file(str(tmp_line_audio))
                except Exception as decode_err:
                    raise RuntimeError(f"Corrupted audio produced by TTS provider for cue {k+1}: {decode_err}")

                duration_sec = len(line_seg) / 1000.0
                if duration_sec <= 0.0:
                    raise RuntimeError(f"TTS provider produced zero-duration audio for cue {k+1}: '{cue}'")

                # Atomically commit line audio
                tmp_line_audio.replace(line_audio_path)
                if tmp_line_audio in tmp_files_to_clean:
                    tmp_files_to_clean.remove(tmp_line_audio)

                cue_start = round(current_time, 3)
                cue_end = round(cue_start + duration_sec, 3)
                caption = extract_concise_caption(cue) if extract_concise_caption else cue

                line_timestamps.append({
                    "scene_id": scene_id,
                    "cue_index": k,
                    "line_index": k,
                    "raw_text": cue,
                    "caption": caption,
                    "start": cue_start,
                    "end": cue_end,
                    "duration": round(duration_sec, 3),
                    "is_speech": cue_is_speech,
                    "is_fallback": cue_is_fallback,
                })

                combined_audio += line_seg

                # Natural acoustic pause between cues
                if k < len(ordered_cues) - 1:
                    next_cue = ordered_cues[k + 1]
                    actual_pause = self.audio_manager.pause_engine.calculate_pause_duration(
                        prev_text=cue,
                        next_text=next_cue,
                        explicit_pause=target_inter,
                    )
                    if actual_pause > 0:
                        sr = combined_audio.frame_rate if len(combined_audio) > 0 else 44100
                        combined_audio += self.audio_manager.pause_engine.create_natural_pause(actual_pause, sample_rate=sr)
                        current_time = cue_end + actual_pause
                    else:
                        current_time = cue_end
                else:
                    current_time = cue_end

            # Trailing scene end natural pause
            if target_end > 0:
                sr = combined_audio.frame_rate if len(combined_audio) > 0 else 44100
                combined_audio += self.audio_manager.pause_engine.create_natural_pause(target_end, sample_rate=sr)
                current_time += target_end

            total_duration = len(combined_audio) / 1000.0
            if total_duration <= 0.0:
                raise RuntimeError(f"Assembled audio for scene '{scene_id}' has zero total duration")

            tmp_audio_path = output_dir / f"{scene_id}_audio.tmp.{os.getpid()}.mp3"
            tmp_ts_path = output_dir / f"{scene_id}_timestamps.tmp.{os.getpid()}.json"
            tmp_files_to_clean.extend([tmp_audio_path, tmp_ts_path])

            combined_audio.export(str(tmp_audio_path), format="mp3")
            if not tmp_audio_path.exists() or tmp_audio_path.stat().st_size < 1000:
                raise RuntimeError(f"Exported combined audio artifact for '{scene_id}' is missing or below 1000 bytes")

            scene_meta = {
                "scene_id": scene_id,
                "audio_path": str(scene_audio_path),
                "timestamps_path": str(ts_path),
                "total_duration": round(total_duration, 3),
                "lines": line_timestamps,
                "narration_hash": current_narr_hash,
                "audio_config": expected_audio_config,
                "cache_hit": False,
                "is_speech": all(l.get("is_speech", True) for l in line_timestamps),
                "is_fallback": any(l.get("is_fallback", False) for l in line_timestamps),
            }

            with open(tmp_ts_path, "w", encoding="utf-8") as f:
                json.dump(scene_meta, f, indent=2, ensure_ascii=False)

            tmp_audio_path.replace(scene_audio_path)
            tmp_ts_path.replace(ts_path)
            if tmp_audio_path in tmp_files_to_clean:
                tmp_files_to_clean.remove(tmp_audio_path)
            if tmp_ts_path in tmp_files_to_clean:
                tmp_files_to_clean.remove(tmp_ts_path)

            srt_path = output_dir / f"{scene_id}.srt"
            self.audio_manager.export_srt(line_timestamps, str(srt_path))
            scene_meta["srt_path"] = str(srt_path)
            scene_meta["alignment_status"] = "unaligned"

            if AudioReference is not None:
                scene_ir.audio_reference = AudioReference(
                    audio_path=str(scene_audio_path),
                    timestamps_path=str(ts_path),
                    total_duration=round(total_duration, 3),
                    alignment_status="unaligned",
                )
            for k, item in enumerate(valid_events):
                line_data = line_timestamps[k]
                item[2].narration_anchor.audio_timestamp_range = (line_data["start"], line_data["end"])

            return scene_ir, scene_meta

        except Exception as e:
            # Fail closed: clear false audio metadata
            scene_ir.audio_reference = None
            for item in valid_events:
                if item[2].narration_anchor:
                    item[2].narration_anchor.audio_timestamp_range = None

            # Clean up any partially committed cue files and scene artifacts
            if audio_dir.exists():
                for cue_f in audio_dir.glob(f"{scene_id}_cue_*.mp3"):
                    try:
                        cue_f.unlink()
                    except Exception:
                        pass
            for art_p in [scene_audio_path, ts_path, srt_path]:
                if art_p.exists():
                    try:
                        art_p.unlink()
                    except Exception:
                        pass
            raise e
        finally:
            for tmp_p in tmp_files_to_clean:
                if tmp_p.exists():
                    try:
                        tmp_p.unlink()
                    except Exception:
                        pass

    def synthesize_sequence_ir(
        self,
        sequence_ir: Any,
        output_dir: Union[str, Path],
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        inter_cue_pause: Optional[float] = None,
        scene_end_pause: Optional[float] = None,
        tts_provider: Optional[Callable[[str, str, str, Optional[str]], float]] = None,
        force_refresh: bool = False,
        provider_config: Optional[Dict[str, Any]] = None,
        allow_fallback: Optional[bool] = None,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """Synthesizes TTS narration sequentially across all sections of a SceneSequenceIR."""
        output_dir = Path(output_dir)
        metas = []
        for section in getattr(sequence_ir, "sections", []):
            _, meta = self.synthesize_scene_ir(
                scene_ir=section,
                output_dir=output_dir,
                voice=voice,
                rate=rate,
                inter_cue_pause=inter_cue_pause,
                scene_end_pause=scene_end_pause,
                tts_provider=tts_provider,
                force_refresh=force_refresh,
                provider_config=provider_config,
                allow_fallback=allow_fallback,
            )
            metas.append(meta)
        return sequence_ir, metas

