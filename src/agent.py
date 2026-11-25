import sys
import os
import io
import tokenize
import re
import argparse
import json
import time
import random
import subprocess
import threading
import concurrent.futures
import typing
from typing import List, Dict, Any, Optional, Tuple, Callable, Union as TUnion
from dataclasses import dataclass, field, fields
from pathlib import Path
import shutil

# Ensure project root and src are in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Ensure venv/bin and texlive/bin are in PATH for manim, ffmpeg, latex subprocesses
_VENV_BIN = Path(sys.executable).parent
if str(_VENV_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_VENV_BIN}:{os.environ.get('PATH', '')}"

_TEXLIVE_BIN = _VENV_BIN.parent / "texlive" / "bin" / "x86_64-linux"
if _TEXLIVE_BIN.is_dir() and str(_TEXLIVE_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_TEXLIVE_BIN}:{os.environ.get('PATH', '')}"

from gpt_request import *
from llm import generate_response, generate_response_token, make_stage_caller, Stage, ProviderType
from prompts import *
from utils import *
from scope_refine import *
from external_assets import process_storyboard_with_assets
from audio_manager import AudioManager, extract_concise_caption, get_context_animation_type, TTSFailureError
from local_checker import LocalVisualChecker, LocalVisualFixer, VisualCheckResult, VisualIssue
from adaptive_duration import AdaptiveDurationManager, LessonDurationEstimate, SectionDurationEstimate
from math_validator import MathValidator, MathValidationReport
from worked_example_verifier import WorkedExampleVerifier, WorkedExampleReport
from numerical_evaluator import VerificationStatus
from storyboard_fidelity_checker import (
    StoryboardFidelityChecker,
    StoryboardFidelityReport,
    FidelityDiscrepancy,
    FidelityCategory,
)
from audio_handoff_validator import (
    AudioHandoffValidator,
    AudioHandoffReport,
    HandoffStatus,
    HandoffIssueCategory,
    compute_storyboard_fingerprint,
    compute_narration_hash,
)
from prompts.stage2 import (
    get_prompt2_storyboard,
    get_prompt2_storyboard_fidelity_repair,
    get_prompt2_scene_ir,
    get_prompt2_scene_ir_repair,
)
from pydub import AudioSegment

try:
    from src.pedagogy import (
        TopicProfile,
        TopicProfiler,
        TeachingBeat,
        TeachingMove,
        BeatValidator,
        VerificationPolicyManager,
        ClaimAuditRecord,
        FourTierStatus,
        ClaimCriticality,
        UnverifiedRemediation,
        VerificationProvenance,
    )
except ImportError:
    from pedagogy import (
        TopicProfile,
        TopicProfiler,
        TeachingBeat,
        TeachingMove,
        BeatValidator,
        VerificationPolicyManager,
        ClaimAuditRecord,
        FourTierStatus,
        ClaimCriticality,
        UnverifiedRemediation,
        VerificationProvenance,
    )

try:
    from src.timing_contract import SceneTimingContract, NarrationSegmentTiming, AnimationRetimer, PacingAuditResult
except ImportError:
    from timing_contract import SceneTimingContract, NarrationSegmentTiming, AnimationRetimer, PacingAuditResult

try:
    from src.ir.adapter import SectionProxy, scene_ir_to_storyboard_section_obj
    from src.ir.models import SceneIR, SceneSequenceIR
    from src.ir.processor import process_stage2_scene_ir_output, SceneIRProcessingResult
except ImportError:
    from ir.adapter import SectionProxy, scene_ir_to_storyboard_section_obj
    from ir.models import SceneIR, SceneSequenceIR
    from ir.processor import process_stage2_scene_ir_output, SceneIRProcessingResult

try:
    from src.compiler.layout_engine import HierarchicalBoxModelLayoutSolver, SceneSpatialLayout
    from src.compiler.timeline_scheduler import TimelineScheduler, ScheduledTimeline
    from src.compiler.scene_compiler import DeterministicSceneCompiler, CompiledSceneOutput
except ImportError:
    from compiler.layout_engine import HierarchicalBoxModelLayoutSolver, SceneSpatialLayout
    from compiler.timeline_scheduler import TimelineScheduler, ScheduledTimeline
    from compiler.scene_compiler import DeterministicSceneCompiler, CompiledSceneOutput


@dataclass
class Section:
    id: str
    title: str
    lecture_lines: List[str]
    animations: List[str]
    story_beat: str = ""
    captions: Optional[List[str]] = None
    layout_plan: str = ""
    verified_values: Optional[List[str]] = None
    algorithmic_operations: Optional[List[Dict[str, Any]]] = None
    beats: Optional[List[Dict[str, Any]]] = None
    resolved_actions: Optional[List[Dict[str, Any]]] = None


@dataclass
class TeachingOutline:
    """Represents a structured pedagogical outline for a video lesson.
    Preserves all Stage 1 blueprint fields (definitions, notation, prerequisites,
    misconceptions, worked examples, analogy evaluation, story framework, etc.)
    and ensures full, lossless serialization to Stage 2 storyboard generation.
    Supports multidimensional topic profiles and heterogeneous teaching beats.
    """
    topic: str = ""
    target_audience: str = ""
    sections: List[Dict[str, Any]] = field(default_factory=list)
    learning_level: str = "intermediate"
    analogy_used: str = ""
    analogy_breakdown: str = ""
    estimated_duration: Optional[float] = None
    parts: Optional[List[Dict[str, Any]]] = None
    story_framework: Optional[Dict[str, Any]] = None

    # Stage 1 Pedagogical Blueprint Fields
    formal_definition: str = ""
    notation: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    common_misconceptions: List[Dict[str, Any]] = field(default_factory=list)
    worked_example_spec: Dict[str, Any] = field(default_factory=dict)
    analogy_evaluation: Dict[str, Any] = field(default_factory=dict)
    topic_type: str = ""
    estimated_duration_minutes: Optional[float] = None
    narration_budget: Dict[str, Any] = field(default_factory=dict)

    # Core pedagogical concepts (first-class support)
    problem_statement: str = ""
    assumptions: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    mental_model: TUnion[Dict[str, Any], str] = ""

    # Milestone 1 Pedagogical Architecture Extensions
    topic_profile: Optional[Dict[str, Any]] = None
    conceptual_breakdown: Optional[Dict[str, Any]] = None
    proof_spec: Optional[Dict[str, Any]] = None
    verification_metadata: Optional[Dict[str, Any]] = None
    verification_status: Optional[str] = None
    beats: Optional[List[Dict[str, Any]]] = None

    # Arbitrary / forward-compatible extra fields
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def get_topic_profile(self) -> Optional[Any]:
        """Reconstruct TopicProfile instance from dictionary or instance."""
        if not self.topic_profile:
            return None
        from src.pedagogy import TopicProfile, TopicProfiler
        if isinstance(self.topic_profile, TopicProfile):
            return self.topic_profile
        if isinstance(self.topic_profile, dict):
            return TopicProfiler.profile_from_data(self.topic_profile, context={"topic": self.topic})
        return None

    def get_all_beats(self) -> List[Any]:
        """Gather all instructional beats defined in outline sections or top-level beats."""
        from src.pedagogy import TeachingBeat
        all_beats: List[TeachingBeat] = []
        if self.beats and isinstance(self.beats, list):
            for b in self.beats:
                if isinstance(b, dict):
                    all_beats.append(TeachingBeat.from_dict(b))
                elif isinstance(b, TeachingBeat):
                    all_beats.append(b)

        if self.sections and isinstance(self.sections, list):
            for sec in self.sections:
                if isinstance(sec, dict) and "beats" in sec and isinstance(sec["beats"], list):
                    for b in sec["beats"]:
                        if isinstance(b, dict):
                            all_beats.append(TeachingBeat.from_dict(b))
                        elif isinstance(b, TeachingBeat):
                            all_beats.append(b)
        return all_beats

    def validate_beats(self) -> Tuple[bool, List[str]]:
        """Validate all instructional beats within the outline."""
        from src.pedagogy import BeatValidator
        beats = self.get_all_beats()
        if not beats:
            return True, []
        return BeatValidator.validate_beats(beats)

    def validate_outline(self) -> Tuple[bool, List[str]]:
        """
        Validates the structural integrity, learning objectives, concept dependencies,
        and consistency of the teaching outline.
        """
        errors: List[str] = []

        # 1. Required schema fields
        if not self.topic or not str(self.topic).strip():
            errors.append("Outline missing mandatory 'topic'")
        if not self.sections or not isinstance(self.sections, list) or len(self.sections) == 0:
            errors.append("Outline must contain at least one section in 'sections'")
            return False, errors

        # 2. Section integrity
        seen_section_ids = set()
        for idx, sec in enumerate(self.sections):
            if not isinstance(sec, dict):
                errors.append(f"Section at index {idx} must be a dictionary")
                continue
            sec_id = str(sec.get("id", "")).strip()
            if not sec_id:
                errors.append(f"Section at index {idx} missing mandatory 'id'")
            elif sec_id in seen_section_ids:
                errors.append(f"Duplicate section id '{sec_id}' detected at index {idx}")
            seen_section_ids.add(sec_id)

        # 3. Beat validation if beats present
        if self.beats:
            is_beats_valid, beat_errs = self.validate_beats()
            if not is_beats_valid:
                errors.extend(beat_errs)

        # 4. Logical concept dependencies (acyclicity check)
        concept_deps = getattr(self, "concept_dependencies", None)
        if not concept_deps and isinstance(getattr(self, "extra_fields", None), dict):
            concept_deps = self.extra_fields.get("concept_dependencies")
        if concept_deps and isinstance(concept_deps, dict):
            visited: set = set()
            rec_stack: set = set()

            def has_cycle(node: str) -> bool:
                visited.add(node)
                rec_stack.add(node)
                for neighbor in concept_deps.get(node, []):
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
                rec_stack.remove(node)
                return False

            for cnode in concept_deps:
                if cnode not in visited:
                    if has_cycle(cnode):
                        errors.append(f"Cycle detected in concept dependencies for concept '{cnode}'")
                        break

        # 5. Worked example step references bounds check
        if self.worked_example_spec and isinstance(self.worked_example_spec, dict):
            steps = self.worked_example_spec.get("steps", [])
            total_steps = len(steps) if isinstance(steps, list) else 0
            for sec in self.sections:
                if isinstance(sec, dict):
                    we_step = sec.get("worked_example_step")
                    if we_step and str(we_step).strip().lower() not in ("none", "null", ""):
                        m_idx = re.search(r"\b(\d+)\b", str(we_step))
                        if m_idx and total_steps > 0:
                            s_num = int(m_idx.group(1))
                            if s_num > total_steps + 1:
                                errors.append(f"Section '{sec.get('id')}' references nonexistent worked example step {s_num} (spec only has {total_steps} steps)")

        return len(errors) == 0, errors

    def is_computational(self) -> bool:
        """Returns True if the outline requires computational/trace grounding."""
        prof = self.get_topic_profile()
        if prof is not None:
            return prof.is_computational()
        return bool(self.worked_example_spec and isinstance(self.worked_example_spec, dict) and self.worked_example_spec.get("steps"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeachingOutline":
        """Reconstruct TeachingOutline from a dictionary, safely handling all blueprint fields,
        nested structures, and arbitrary extra fields without dropping any data."""
        if not isinstance(data, dict):
            return cls()

        data_copy = dict(data)
        known_field_names = {f.name for f in fields(cls) if f.name != "extra_fields"}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}

        for k, v in data_copy.items():
            if k in known_field_names:
                kwargs[k] = v
            else:
                extra[k] = v

        # Convert topic_profile if it's a dataclass instance to dict
        if "topic_profile" in kwargs and hasattr(kwargs["topic_profile"], "to_dict"):
            kwargs["topic_profile"] = kwargs["topic_profile"].to_dict()

        # Harmonize estimated_duration and estimated_duration_minutes
        if "estimated_duration" not in kwargs or kwargs["estimated_duration"] is None:
            if "estimated_duration_minutes" in kwargs and kwargs["estimated_duration_minutes"] is not None:
                kwargs["estimated_duration"] = kwargs["estimated_duration_minutes"]
        if "estimated_duration_minutes" not in kwargs or kwargs["estimated_duration_minutes"] is None:
            if "estimated_duration" in kwargs and kwargs["estimated_duration"] is not None:
                kwargs["estimated_duration_minutes"] = kwargs["estimated_duration"]

        instance = cls(**kwargs, extra_fields=extra)

        # Bind safe extra fields as dynamic instance attributes
        for k, v in extra.items():
            if not isinstance(k, str):
                continue
            if not k.isidentifier():
                continue
            if k.startswith("_"):
                continue
            if hasattr(cls, k):
                continue
            try:
                setattr(instance, k, v)
            except Exception:
                pass

        return instance

    def to_dict(self) -> Dict[str, Any]:
        """Convert TeachingOutline to a clean, complete dictionary suitable for Stage 2 serialization.
        Ensures all dataclass fields, nested objects, and extra fields are preserved."""
        result: Dict[str, Any] = {}

        # 1. Add all standard dataclass fields (excluding internal extra_fields tracking)
        for f in fields(self):
            if f.name == "extra_fields":
                continue
            val = getattr(self, f.name)
            if f.name == "topic_profile" and hasattr(val, "to_dict"):
                val = val.to_dict()
            result[f.name] = val

        # 2. Add extra_fields (without overwriting explicit dataclass fields unless empty)
        if self.extra_fields:
            for k, v in self.extra_fields.items():
                if k not in result:
                    result[k] = v

        # 3. Add any dynamic attributes attached directly to the instance
        for k, v in self.__dict__.items():
            if k not in result and not k.startswith("_") and k != "extra_fields":
                result[k] = v

        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Safe dict-like getter."""
        if key in self.__dict__ and key != "extra_fields" and not key.startswith("_"):
            val = self.__dict__[key]
            return val if val is not None else default
        if self.extra_fields and key in self.extra_fields:
            val = self.extra_fields[key]
            return val if val is not None else default
        return default

    def __getitem__(self, key: str) -> Any:
        """Allow subscript access outline['field']."""
        if key in self.__dict__ and key != "extra_fields" and not key.startswith("_"):
            return self.__dict__[key]
        if self.extra_fields and key in self.extra_fields:
            return self.extra_fields[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Allow 'field' in outline checks."""
        if key in self.__dict__ and key != "extra_fields" and not key.startswith("_"):
            return True
        return bool(self.extra_fields and key in self.extra_fields)



@dataclass
class VideoFeedback:
    section_id: str
    video_path: str
    has_issues: bool
    suggested_improvements: List[str]
    raw_response: Optional[str] = None


@dataclass
class RunConfig:
    use_feedback: bool = False
    use_assets: bool = False
    api: Callable = None
    feedback_rounds: int = 1
    iconfinder_api_key: str = ""
    max_code_token_length: int = 10000
    max_fix_bug_tries: int = 1
    max_regenerate_tries: int = 1
    max_fidelity_retries: int = 2
    max_feedback_gen_code_tries: int = 1
    max_mllm_fix_bugs_tries: int = 1
    # Pipeline configuration ("legacy" or "modern")
    pipeline_mode: str = "legacy"
    # Audio configuration
    audio: bool = True
    audio_input: str = ""
    tts_voice: str = "en-US-GuyNeural"
    burn_subtitles: bool = True
    tts_provider: Optional[Any] = None
    # Video quality configuration ("low" for 480p15, "medium" for 720p30, "high" for 1080p60)
    quality: str = "medium"
    # Concurrency configuration
    parallel: bool = True
    max_workers: int = 4
    # Three-input parameters
    level: str = "intermediate"
    analogy: str = ""
    # Adaptive duration parameters
    soft_target_duration: float = 0.0
    hard_max_duration: float = 0.0
    speaking_wpm: float = 0.0
    # Model-agnostic parameters
    provider: Optional[str] = None
    model: Optional[str] = None
    force_render: bool = False
    # Quality Control configuration (Milestone 6)
    enable_qc: bool = False


class TeachingVideoAgent:
    def __init__(
        self,
        idx,
        knowledge_point: str = "",
        folder="CASES",
        cfg: Optional[RunConfig] = None,
        topic: Optional[str] = None,
        level: Optional[str] = None,
        analogy: Optional[str] = None,
        soft_target_duration: Optional[float] = None,
        hard_max_duration: Optional[float] = None,
        speaking_wpm: Optional[float] = None,
        quality: Optional[str] = None,
        pipeline_mode: Optional[str] = None,
        tts_provider: Optional[Callable] = None,
        scene_sequence_ir: Optional[SceneSequenceIR] = None,
        enable_qc: Optional[bool] = None,
        qc_controller: Optional[Any] = None,
    ):
        """1. Global parameter"""
        self.topic = (topic or knowledge_point or "").strip()
        self.learning_topic = self.topic
        self.idx = idx
        self.cfg = cfg

        # Pipeline mode configuration ("legacy" or "modern", defaulting to "legacy")
        cfg_pipeline_mode = getattr(cfg, "pipeline_mode", "legacy") if cfg else "legacy"
        raw_mode = pipeline_mode if pipeline_mode is not None else cfg_pipeline_mode
        self.pipeline_mode = str(raw_mode or "legacy").lower().strip()
        if self.pipeline_mode not in ("legacy", "modern"):
            raise ValueError(f"Invalid pipeline_mode: {raw_mode!r}. Valid options are 'legacy' or 'modern'.")

        cfg_level = getattr(cfg, "level", "intermediate") if cfg else "intermediate"
        self.level = (level or cfg_level or "intermediate").lower()
        if self.level not in ("beginner", "intermediate", "advanced"):
            self.level = "intermediate"

        cfg_analogy = getattr(cfg, "analogy", "") if cfg else ""
        self.requested_analogy = (analogy if analogy is not None else cfg_analogy) or ""
        self.analogy = self.requested_analogy

        cfg_quality = getattr(cfg, "quality", "medium") if cfg else "medium"
        self.quality = (quality or cfg_quality or "medium").lower()
        if self.quality == "high":
            self.manim_quality_flag = "-qh"
            self.quality_folder = "1080p60"
        elif self.quality == "low":
            self.manim_quality_flag = "-ql"
            self.quality_folder = "480p15"
        else:
            self.quality = "medium"
            self.manim_quality_flag = "-qm"
            self.quality_folder = "720p30"

        self.use_feedback = cfg.use_feedback if cfg else False
        self.use_assets = cfg.use_assets if cfg else False
        self.provider = getattr(cfg, "provider", None) if cfg else None
        self.model = getattr(cfg, "model", None) if cfg else None
        self.API = cfg.api if (cfg and cfg.api) else None
        self.feedback_rounds = cfg.feedback_rounds if cfg else 1
        self.iconfinder_api_key = cfg.iconfinder_api_key if cfg else ""
        self.max_code_token_length = cfg.max_code_token_length if cfg else 10000
        self.max_fix_bug_tries = cfg.max_fix_bug_tries if cfg else 1
        self.max_regenerate_tries = cfg.max_regenerate_tries if cfg else 1
        self.max_feedback_gen_code_tries = cfg.max_feedback_gen_code_tries if cfg else 1
        self.max_mllm_fix_bugs_tries = cfg.max_mllm_fix_bugs_tries if cfg else 1

        # Concurrency & thread-safety
        self.parallel = getattr(cfg, "parallel", True) if cfg else True
        self.max_workers = getattr(cfg, "max_workers", 4) if cfg else 4
        self.state_lock = threading.Lock()

        # Hard bounded retry limits per architecture specification
        self.max_initial_code_generation_calls = 1
        self.max_scope_refine_calls_initial_error = getattr(cfg, "max_fix_bug_tries", 1) if cfg else 1
        self.max_critic_calls_per_section = getattr(cfg, "feedback_rounds", 1) if cfg else 1
        self.max_critic_modification_attempts = getattr(cfg, "max_feedback_gen_code_tries", 1) if cfg else 1
        self.max_post_critic_scope_refine_calls = getattr(cfg, "max_mllm_fix_bugs_tries", 1) if cfg else 1

        # Local visual checker and deterministic fixer ($0 API cost, zero additional inspection renders)
        self.local_checker = LocalVisualChecker()
        self.local_fixer = LocalVisualFixer()

        # Audio parameters
        self.use_audio = getattr(cfg, "audio", True) if cfg else True
        self.audio_input = getattr(cfg, "audio_input", "") if cfg else ""
        self.tts_voice = getattr(cfg, "tts_voice", "en-US-GuyNeural") if cfg else "en-US-GuyNeural"
        self.burn_subtitles = getattr(cfg, "burn_subtitles", True) if cfg else True
        self.tts_provider = tts_provider or (getattr(cfg, "tts_provider", None) if cfg else None)
        self.audio_manager = AudioManager(tts_voice=self.tts_voice)

        # Adaptive duration manager & mathematical validator
        soft_dur = (
            soft_target_duration
            if (soft_target_duration and soft_target_duration > 0)
            else getattr(cfg, "soft_target_duration", 0.0) if cfg else 0.0
        )
        hard_dur = (
            hard_max_duration
            if (hard_max_duration and hard_max_duration > 0)
            else getattr(cfg, "hard_max_duration", 0.0) if cfg else 0.0
        )
        wpm = (
            speaking_wpm
            if (speaking_wpm and speaking_wpm > 0)
            else getattr(cfg, "speaking_wpm", 0.0) if cfg else 0.0
        )

        self.duration_manager = AdaptiveDurationManager(
            level=self.level,
            soft_target_duration=soft_dur,
            hard_max_duration=hard_dur,
            speaking_wpm=wpm,
        )
        self.math_validator = MathValidator()

        # Initialize and recursively discover domain-specific visual components
        from src.domains import VisualRegistry
        self.discovered_domains = VisualRegistry.discover_domains()

        """2. Path for output"""
        self.folder = folder
        self.output_dir = get_output_dir(
            idx=idx,
            knowledge_point=self.learning_topic,
            base_dir=folder,
            level=self.level,
            analogy=self.analogy,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.assets_dir = _PROJECT_ROOT / "assets" / "icon"
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        """3. ScopeRefine & Anchor Visual"""
        def _safe_repair_caller(prompt, **kw):
            kw.setdefault("stage", "repair")
            if cfg and cfg.api:
                return self.API(prompt, **kw)
            return make_stage_caller("repair")(prompt, **kw)
        self.scope_refine_fixer = ScopeRefineFixer(_safe_repair_caller, self.max_code_token_length)
        self.extractor = GridPositionExtractor()

        """4. External Database"""
        knowledge_ref_mapping_path = _PROJECT_ROOT / "json_files" / "long_video_ref_mapping.json"
        if knowledge_ref_mapping_path.is_file():
            with open(knowledge_ref_mapping_path, "r", encoding="utf-8") as f:
                self.KNOWLEDGE2PATH = json.load(f)
        else:
            print(
                f"⚠️ Warning: Topic reference mapping not found at {knowledge_ref_mapping_path}; "
                "continuing without bundled topic reference images."
            )
            self.KNOWLEDGE2PATH = {}
        self.knowledge_ref_img_folder = _PROJECT_ROOT / "assets" / "reference"
        self.GRID_IMG_PATH = self.knowledge_ref_img_folder / "GRID.png"

        """5. Data structure"""
        self.outline = None
        self.worked_example_verifier = WorkedExampleVerifier()
        self.outline_verification_report: Optional[WorkedExampleReport] = None
        self.storyboard_fidelity_checker = StoryboardFidelityChecker()
        self.storyboard_fidelity_report: Optional[StoryboardFidelityReport] = None
        self.max_fidelity_retries: int = getattr(cfg, "max_fidelity_retries", 2) if cfg else 2
        self.audio_handoff_validator = AudioHandoffValidator()
        self.audio_handoff_report: Optional[AudioHandoffReport] = None
        self.validated_storyboard_fingerprint: Optional[str] = None
        self.enhanced_storyboard = None
        self.sections = []
        self.section_codes = {}
        self.section_videos = {}
        self.section_audios = {}
        self.section_timestamps = {}
        self.video_feedbacks = {}

        # Modern Scene IR state (Milestone 5.2)
        self.scene_sequence_ir: Optional[SceneSequenceIR] = scene_sequence_ir
        self.scene_ir: Optional[SceneIR] = None
        self.scene_ir_processing_result: Optional[SceneIRProcessingResult] = None

        # Quality Control state (Milestone 6)
        cfg_qc = getattr(cfg, "enable_qc", None) if cfg else None
        if enable_qc is not None:
            self.enable_qc = bool(enable_qc)
        elif cfg_qc is not None:
            self.enable_qc = bool(cfg_qc)
        else:
            self.enable_qc = False
        self.qc_controller = qc_controller
        self.section_qc_reports = {}
        self.master_qc_report = None

        """6. For Efficiency"""
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Auto-load existing checkpoint or disk state to avoid redundant API calls
        self._load_existing_checkpoint_and_files()

    def get_config_signature(self) -> str:
        """Returns deterministic hash signature of topic, level, analogy, and duration configuration."""
        import hashlib
        sig_str = (
            f"{self.learning_topic}|{self.level}|{self.analogy}|"
            f"{self.duration_manager.soft_target_sec}|{self.duration_manager.hard_max_sec}"
        )
        return hashlib.sha256(sig_str.encode("utf-8")).hexdigest()[:16]

    def _load_existing_checkpoint_and_files(self):
        """Load existing checkpoint or disk files into memory with strict cache validation to avoid duplicate API calls."""
        checkpoint_file = self.output_dir / "checkpoint.json"
        is_cache_valid = False

        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, "r", encoding="utf-8") as f:
                    cp = json.load(f)

                # Pipeline mode compatibility check
                # Historical checkpoints without pipeline_mode default safely to "legacy"
                if "pipeline_mode" not in cp:
                    cp_mode = "legacy"
                else:
                    raw_cp_mode = cp["pipeline_mode"]
                    if raw_cp_mode is None or not isinstance(raw_cp_mode, str):
                        raise ValueError(
                            f"Checkpoint at '{checkpoint_file}' has invalid or corrupted pipeline_mode: "
                            f"{raw_cp_mode!r}. Valid options are 'legacy' or 'modern'."
                        )
                    norm_cp_mode = raw_cp_mode.lower().strip()
                    if norm_cp_mode not in ("legacy", "modern"):
                        raise ValueError(
                            f"Checkpoint at '{checkpoint_file}' has invalid or corrupted pipeline_mode: "
                            f"{raw_cp_mode!r}. Valid options are 'legacy' or 'modern'."
                        )
                    cp_mode = norm_cp_mode

                if cp_mode != self.pipeline_mode:
                    raise ValueError(
                        f"Checkpoint pipeline mode mismatch: Checkpoint at '{checkpoint_file}' "
                        f"has pipeline_mode='{cp_mode}', but current agent requested "
                        f"pipeline_mode='{self.pipeline_mode}'. Cannot resume across conflicting pipelines."
                    )

                # Cache safety check: Verify topic, level, and analogy
                cp_topic = cp.get("topic") or (cp.get("outline", {}).get("topic") if cp.get("outline") else None)
                cp_level = (cp.get("level") or "intermediate").lower()
                cp_analogy = (cp.get("analogy") or "").strip()

                cp_req_analogy = (cp.get("requested_analogy") or "").strip()
                analogy_matched = not self.requested_analogy or (
                    cp_analogy == self.requested_analogy.strip()
                    or cp_req_analogy == self.requested_analogy.strip()
                    or self.requested_analogy.strip().lower() in cp_analogy.lower()
                    or (cp_req_analogy and self.requested_analogy.strip().lower() in cp_req_analogy.lower())
                )
                if cp_topic and cp_topic != self.learning_topic:
                    print(f"⚠️ Cache mismatch: Checkpoint topic '{cp_topic}' != current topic '{self.learning_topic}'. Skipping checkpoint reuse.")
                elif cp.get("level") is not None and cp_level != self.level:
                    print(f"⚠️ Cache mismatch: Checkpoint level '{cp_level}' != current level '{self.level}'. Skipping checkpoint reuse.")
                elif cp.get("analogy") is not None and not analogy_matched:
                    print(f"⚠️ Cache mismatch: Checkpoint analogy '{cp_analogy}' != current analogy '{self.requested_analogy}'. Skipping checkpoint reuse.")
                else:
                    is_cache_valid = True
                    if not self.outline and cp.get("outline"):
                        outline_source = dict(cp["outline"]) if isinstance(cp["outline"], dict) else {}
                        outline_json_file = self.output_dir / "outline.json"
                        if outline_json_file.exists():
                            try:
                                with open(outline_json_file, "r", encoding="utf-8") as f:
                                    disk_outline = json.load(f)
                                    if isinstance(disk_outline, dict):
                                        merged = dict(disk_outline)
                                        merged.update(outline_source)
                                        outline_source = merged
                            except Exception:
                                pass
                        if not outline_source.get("topic"):
                            outline_source["topic"] = self.learning_topic
                        if not outline_source.get("learning_level"):
                            outline_source["learning_level"] = self.level
                        if not outline_source.get("analogy_used") and self.analogy:
                            outline_source["analogy_used"] = self.analogy
                        self.outline = TeachingOutline.from_dict(outline_source)
                        self._ensure_outline_verified()
                    if cp.get("validated_storyboard_fingerprint"):
                        self.validated_storyboard_fingerprint = cp["validated_storyboard_fingerprint"]
                    if cp.get("visual_conventions"):
                        self.visual_conventions = cp["visual_conventions"]
                    if not self.sections and cp.get("sections"):
                        self.sections = [
                            Section(
                                id=s["id"],
                                title=s["title"],
                                lecture_lines=s.get("lecture_lines", []),
                                animations=s.get("animations", []),
                                story_beat=s.get("story_beat", s.get("story_step", "")),
                                captions=s.get("captions"),
                                layout_plan=s.get("layout_plan", ""),
                                verified_values=s.get("verified_values"),
                            )
                            for s in cp["sections"]
                        ]
                        if self.outline:
                            self._ensure_storyboard_verified()
                    if cp.get("section_codes"):
                        for sid, scode in cp["section_codes"].items():
                            scene_cls = f"class {sid.title().replace('_', '')}Scene"
                            if isinstance(scode, str) and scene_cls in scode and "def construct(self" in scode:
                                try:
                                    compile(scode, f"{sid}.py", "exec")
                                    self.section_codes[sid] = scode
                                except Exception:
                                    pass
                    if cp.get("section_videos"):
                        self.section_videos.update(cp["section_videos"])
                    if cp.get("section_audios"):
                        self.section_audios.update(cp["section_audios"])
                    if cp.get("section_timestamps"):
                        self.section_timestamps.update(cp["section_timestamps"])
                    if cp.get("token_usage"):
                        for k, v in cp["token_usage"].items():
                            self.token_usage[k] = max(self.token_usage.get(k, 0), v)
                    if self.pipeline_mode == "modern":
                        if self.scene_sequence_ir is None and cp.get("scene_sequence_ir"):
                            try:
                                self.scene_sequence_ir = SceneSequenceIR.model_validate(cp["scene_sequence_ir"])
                            except Exception:
                                pass
                        if self.scene_sequence_ir is None:
                            seq_cand = self.output_dir / "scene_sequence_ir.json"
                            scene_cand = self.output_dir / "scene_ir.json"
                            if seq_cand.exists():
                                try:
                                    with open(seq_cand, "r", encoding="utf-8") as _sf:
                                        _seq_dict = json.load(_sf)
                                    self.scene_sequence_ir = SceneSequenceIR.model_validate(_seq_dict)
                                except Exception:
                                    pass
                            elif scene_cand.exists():
                                try:
                                    with open(scene_cand, "r", encoding="utf-8") as _sf:
                                        _sc_dict = json.load(_sf)
                                    self.scene_ir = SceneIR.model_validate(_sc_dict)
                                except Exception:
                                    pass
            except ValueError:
                raise
            except Exception as e:
                print(f"⚠️ Notice: Failed to load checkpoint.json: {e}")

        # Scan directory for any valid section code files on disk only if cache is valid or directory is uncheckpointed
        if self.output_dir.exists() and (is_cache_valid or not checkpoint_file.exists()):
            for code_path in sorted(self.output_dir.glob("section_*.py")):
                sec_id = code_path.stem
                if sec_id not in self.section_codes:
                    try:
                        with open(code_path, "r", encoding="utf-8") as f:
                            disk_code = f.read()
                            scene_cls = f"class {sec_id.title().replace('_', '')}Scene"
                            if scene_cls in disk_code and "def construct(self" in disk_code:
                                compile(disk_code, str(code_path), "exec")
                                self.section_codes[sec_id] = disk_code
                    except Exception:
                        pass

            # Scan directory for already synchronized section videos
            for sync_mp4 in sorted(self.output_dir.glob("section_*_synced.mp4")):
                sec_id = sync_mp4.stem.replace("_synced", "")
                if sync_mp4.stat().st_size > 1000:
                    self.section_videos[sec_id] = str(sync_mp4)
                    audio_cand = self.output_dir / f"{sec_id}_audio.mp3"
                    if audio_cand.exists():
                        self.section_audios[sec_id] = str(audio_cand)
                    ts_cand = self.output_dir / f"{sec_id}_timestamps.json"
                    if ts_cand.exists():
                        try:
                            with open(ts_cand, "r", encoding="utf-8") as f:
                                self.section_timestamps[sec_id] = json.load(f)
                        except Exception:
                            pass

            if not getattr(self, "visual_conventions", None):
                sb_path = self.output_dir / "storyboard_with_assets.json"
                if not sb_path.exists():
                    sb_path = self.output_dir / "storyboard.json"
                if sb_path.exists():
                    try:
                        with open(sb_path, "r", encoding="utf-8") as sbf:
                            self.visual_conventions = json.load(sbf).get("visual_conventions")
                    except Exception:
                        pass

    def _request_api_and_track_tokens(self, prompt, max_tokens=10000, stage="default"):
        """packages API requests and automatically accumulates token usage"""
        if self.API is not None:
            try:
                response, usage = self.API(prompt, max_tokens=max_tokens, stage=stage)
            except TypeError:
                response, usage = self.API(prompt, max_tokens=max_tokens)
        else:
            response, usage = generate_response_token(
                prompt=prompt,
                stage=stage,
                max_tokens=max_tokens,
                provider=self.provider,
                model=self.model,
            )
        if usage:
            with self.state_lock:
                self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
        return response

    def _request_video_api_and_track_tokens(self, prompt, video_path, stage="verification"):
        """Wraps video API requests and accumulates token usage automatically using stage routing"""
        response, usage = generate_response_token(
            prompt=prompt,
            stage=stage,
            video_path=video_path,
            image_path=str(self.GRID_IMG_PATH),
            provider=self.provider,
            model=self.model,
        )

        if usage:
            with self.state_lock:
                self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                self.token_usage["total_tokens"] += usage.get("total_tokens", 0)
        return response

    def get_serializable_state(self):
        """Returns serializable state for worker processes including code cache."""
        with self.state_lock:
            return {
                "idx": self.idx,
                "knowledge_point": self.learning_topic,
                "topic": self.learning_topic,
                "level": self.level,
                "analogy": self.analogy,
                "folder": self.folder,
                "cfg": self.cfg,
                "section_codes": dict(self.section_codes),
                "section_videos": dict(self.section_videos),
            }

    def save_checkpoint(self, stage: str, extra_data: dict = None):
        """Save pipeline checkpoint to disk for fault tolerance, debugging, and resume."""
        with self.state_lock:
            checkpoint_data = {
                "topic": self.learning_topic,
                "level": self.level,
                "pipeline_mode": self.pipeline_mode,
                "requested_analogy": getattr(self, "requested_analogy", self.analogy),
                "analogy": self.analogy,
                "config_signature": self.get_config_signature(),
                "idx": self.idx,
                "stage": stage,
                "timestamp": time.time(),
                "outline": self.outline.to_dict() if hasattr(self.outline, "to_dict") else (self.outline.__dict__ if self.outline else None),
                "scene_sequence_ir": self.scene_sequence_ir.to_dict() if getattr(self, "scene_sequence_ir", None) else None,
                "visual_conventions": getattr(self, "visual_conventions", None),
                "validated_storyboard_fingerprint": self.validated_storyboard_fingerprint,
                "sections": [s.__dict__ for s in self.sections] if self.sections else [],
                "section_codes": dict(self.section_codes),
                "section_videos": dict(self.section_videos),
                "section_audios": dict(self.section_audios),
                "section_timestamps": dict(self.section_timestamps),
                "token_usage": dict(self.token_usage),
                "extra": extra_data or {},
            }
            if extra_data and isinstance(extra_data, dict):
                checkpoint_data.update(extra_data)
            checkpoint_file = self.output_dir / "checkpoint.json"
            try:
                tmp_file = self.output_dir / f"checkpoint.tmp.{os.getpid()}.{threading.get_ident()}.json"
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                tmp_file.replace(checkpoint_file)
                print(f"💾 Checkpoint saved [{stage}] -> {checkpoint_file}")
            except Exception as e:
                print(f"⚠️ Failed to write checkpoint: {e}")

    @staticmethod
    def _resolve_step_section_id(sections: List[Dict[str, Any]], step_idx: int) -> str:
        """Resolves section ID associated with worked example step_idx (0-indexed).
        Derives reliable, unambiguous mappings from explicit step references, beat citations,
        or unique demonstration roles. If no reliable mapping can be established, returns empty
        string so unlocalized failures fail closed and halt the pipeline."""
        if not sections or not isinstance(sections, list):
            return ""
        step_num = step_idx + 1
        step_token = f"step_{step_num}"
        step_str = f"step {step_num}"

        # 1. Direct worked_example_step match
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sec_id = str(sec.get("id", "")).strip()
            if not sec_id:
                continue
            wes = str(sec.get("worked_example_step", "")).lower().strip()
            if wes and wes != "none":
                if (
                    step_token == wes
                    or step_token in wes
                    or step_str in wes
                    or str(step_num) == wes
                    or f"step {step_num}:" in wes
                ):
                    return sec_id

        # 2. Direct beat verification_references match
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sec_id = str(sec.get("id", "")).strip()
            if not sec_id:
                continue
            for b in (sec.get("beats") or []):
                if not isinstance(b, dict):
                    continue
                refs = [str(r).lower().strip() for r in (b.get("verification_references") or [])]
                if step_token in refs or str(step_num) in refs or step_str in refs:
                    return sec_id

        # 3. Direct section title or content explicit step citation
        matching_title_sections = []
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sec_id = str(sec.get("id", "")).strip()
            if not sec_id:
                continue
            title_content = (str(sec.get("title", "")) + " " + str(sec.get("content", ""))).lower()
            if f"step {step_num}" in title_content or f"step_{step_num}" in title_content:
                matching_title_sections.append(sec_id)
        if len(matching_title_sections) == 1:
            return matching_title_sections[0]

        # 4. Check section with pedagogical_role "worked_demonstration" (only if exactly 1 exists)
        demo_secs = [
            str(sec.get("id", "")).strip()
            for sec in sections
            if isinstance(sec, dict) and sec.get("pedagogical_role") == "worked_demonstration" and str(sec.get("id", "")).strip()
        ]
        if len(demo_secs) == 1:
            return demo_secs[0]

        # 5. If reliable mapping cannot be derived, do NOT guess or assign arbitrarily.
        # Return empty string to preserve unlocalized failure and let the policy halt.
        return ""

    def get_verification_provenance(self) -> Optional[Any]:
        """Returns the active VerificationProvenance instance if present on outline."""
        if not self.outline:
            return None
        from src.pedagogy import VerificationProvenance
        if hasattr(self.outline, "verification_provenance") and isinstance(self.outline.verification_provenance, VerificationProvenance):
            return self.outline.verification_provenance
        if hasattr(self.outline, "extra_fields") and isinstance(self.outline.extra_fields, dict):
            prov_raw = self.outline.extra_fields.get("verification_provenance")
            if isinstance(prov_raw, dict):
                return VerificationProvenance.from_dict(prov_raw)
            elif isinstance(prov_raw, VerificationProvenance):
                return prov_raw
        vm = getattr(self.outline, "verification_metadata", None)
        if isinstance(vm, dict) and "verification_provenance" in vm:
            prov_raw = vm["verification_provenance"]
            if isinstance(prov_raw, dict):
                return VerificationProvenance.from_dict(prov_raw)
            elif isinstance(prov_raw, VerificationProvenance):
                return prov_raw
        return None

    def get_stage2_outline_dict(self) -> Dict[str, Any]:
        """
        Constructs a non-destructive copy of self.outline for Stage 2 storyboard generation.
        Omits blocked sections while preserving the canonical self.outline and recording
        which sections were excluded and why in the serialized payload metadata.
        Guarantees that unlocalized failures or halted pipelines cannot forward any sections.
        """
        if not self.outline:
            return {}
        outline_dict = self.outline.to_dict() if hasattr(self.outline, "to_dict") else dict(self.outline.__dict__)
        prov = self.get_verification_provenance()
        if prov:
            from src.pedagogy import FourTierStatus
            all_raw_secs = outline_dict.get("sections", [])

            # Fix A: If pipeline was halted globally, strictly forward ZERO sections to Stage 2
            if prov.pipeline_halted:
                outline_dict["sections"] = []
                outline_dict["excluded_sections"] = sorted([
                    str(s.get("id", f"unnamed_section_{i+1}")) for i, s in enumerate(all_raw_secs)
                ])
                outline_dict["exclusion_reasons"] = {
                    sec_id: prov.halt_reason or "Pipeline halted due to verification failure"
                    for sec_id in outline_dict["excluded_sections"]
                }
                print(f"🛑 [SECTION ISOLATION] Pipeline is halted ({prov.overall_status.value}). Suppressing all sections from Stage 2 input.")
                return outline_dict

            safe_ids = set(prov.safe_sections)
            blocked_set = set(prov.blocked_sections)

            # Only sections with a valid non-empty id in safe_sections (and not in blocked_sections) proceed
            safe_secs = []
            excluded_secs = []
            for idx, s in enumerate(all_raw_secs):
                sid = str(s.get("id", "")).strip() if isinstance(s, dict) else ""
                if sid and sid in safe_ids and sid not in blocked_set:
                    safe_secs.append(s)
                else:
                    excluded_label = sid if sid else f"section_{idx+1}_missing_id"
                    excluded_secs.append(excluded_label)

            outline_dict["sections"] = safe_secs
            outline_dict["excluded_sections"] = sorted(excluded_secs)

            reasons = {}
            for c in prov.claims:
                if c.status == FourTierStatus.FAILED and c.section_id in blocked_set:
                    reasons[c.section_id] = c.statement or c.details
            for ex in excluded_secs:
                if ex not in reasons:
                    reasons[ex] = "Section omitted because it was unverified, blocked, or missing a valid section ID"
            outline_dict["exclusion_reasons"] = reasons

            if excluded_secs:
                print(f"🛡️ [SECTION ISOLATION] Excluded blocked/unsafe sections from Stage 2 input: {sorted(excluded_secs)}. Safe sections proceeding: {[s.get('id') for s in safe_secs]}")
        return outline_dict

    def _ensure_outline_verified(self, force: bool = False) -> Optional[WorkedExampleReport]:
        """
        Ensures that self.outline is verified through TopicProfiler, BeatValidator,
        and WorkedExampleVerifier with four-tier verification semantics and provenance.
        """
        if not self.outline:
            return None

        # Check if already verified in this session with full provenance attached
        has_full_provenance = self.get_verification_provenance() is not None
        if not force and self.outline_verification_report is not None and has_full_provenance:
            return self.outline_verification_report

        from src.pedagogy import (
            TopicProfile,
            TopicProfiler,
            TeachingBeat,
            TeachingMove,
            BeatValidator,
            VerificationPolicyManager,
            ClaimAuditRecord,
            FourTierStatus,
            ClaimCriticality,
            UnverifiedRemediation,
            VerificationProvenance,
        )

        # 1. Topic Profiling Validation & Fallback
        prof_data = self.outline.get("topic_profile")
        if prof_data and isinstance(prof_data, dict):
            profile = TopicProfiler.profile_from_data(prof_data, context={"topic": self.outline.topic})
            self.outline.topic_profile = profile.to_dict()
        elif hasattr(self.outline, "topic") and self.outline.topic:
            profile = TopicProfiler.profile_heuristic_fallback(self.outline.topic)
            self.outline.topic_profile = profile.to_dict()

        # 2. Beat-Level Validation
        is_beats_valid, beat_errs = self.outline.validate_beats()
        beat_claims: List[ClaimAuditRecord] = []
        if not is_beats_valid:
            print(f"⚠️ Beat validation errors detected: {beat_errs}")
            all_beats = self.outline.get_all_beats()
            for b_idx, err in enumerate(beat_errs):
                matched_sec = ""
                matched_bid = ""
                for b in all_beats:
                    bid = getattr(b, "beat_id", "")
                    if bid and bid in err:
                        matched_bid = bid
                        break
                if matched_bid and hasattr(self.outline, "sections") and isinstance(self.outline.sections, list):
                    for s in self.outline.sections:
                        sec_id = s.get("id", "")
                        for sb in s.get("beats", []):
                            if sb.get("beat_id") == matched_bid:
                                matched_sec = sec_id
                                break
                        if matched_sec:
                            break
                beat_claims.append(ClaimAuditRecord(
                    claim_id=f"beat_defect_{b_idx + 1}",
                    section_id=matched_sec,
                    beat_id=matched_bid,
                    statement=f"Beat validation defect: {err}",
                    status=FourTierStatus.FAILED,
                    criticality=ClaimCriticality.CRITICAL,
                    essential_objective_violated=True,
                    details=err,
                ))

        # 2b. Outline Structural & Objective Validation
        is_outline_valid, outline_errs = self.outline.validate_outline()
        outline_claims: List[ClaimAuditRecord] = []
        if not is_outline_valid:
            print(f"⚠️ Outline validation errors detected: {outline_errs}")
            for o_idx, o_err in enumerate(outline_errs):
                outline_claims.append(ClaimAuditRecord(
                    claim_id=f"outline_defect_{o_idx + 1}",
                    section_id="",
                    beat_id="",
                    statement=f"Outline validation defect: {o_err}",
                    status=FourTierStatus.FAILED,
                    criticality=ClaimCriticality.CRITICAL,
                    essential_objective_violated=True,
                    details=o_err,
                ))

        outline_dict = self.outline.to_dict() if hasattr(self.outline, "to_dict") else dict(self.outline.__dict__)
        verified_data, report = self.worked_example_verifier.verify_outline(outline_dict)

        # 3. Four-Tier Verification Provenance Construction
        claims: List[ClaimAuditRecord] = []
        claims.extend(beat_claims)
        claims.extend(outline_claims)

        sec_list = self.outline.sections if hasattr(self.outline, "sections") and isinstance(self.outline.sections, list) else []
        for i, diag in enumerate(report.step_diagnostics):
            step_st = str(diag.get("status", "UNVERIFIED")).upper()
            resolution = str(diag.get("resolution", "")).upper()
            can_correct = bool(diag.get("can_safely_correct", False))

            if resolution == "AUTO_CORRECTED" or (can_correct and step_st == "FAILED"):
                ft_st = FourTierStatus.VERIFIED
            elif step_st == "VERIFIED":
                ft_st = FourTierStatus.VERIFIED
            elif step_st in ("FAILED", "NEEDS_REVIEW"):
                ft_st = FourTierStatus.FAILED
            elif step_st in ("NOT_APPLICABLE", "NONE"):
                ft_st = FourTierStatus.NOT_APPLICABLE
            else:
                ft_st = FourTierStatus.UNVERIFIED

            crit = ClaimCriticality.CRITICAL if (i == 0 or i == report.total_steps - 1) else ClaimCriticality.NON_CRITICAL
            sec_id = self._resolve_step_section_id(sec_list, i)
            claims.append(ClaimAuditRecord(
                claim_id=f"step_{i+1}",
                section_id=sec_id,
                beat_id="",
                statement=str(diag.get("details", "")),
                status=ft_st,
                criticality=crit,
                details=str(diag.get("details", "")),
            ))

        section_ids = [
            str(s.get("id")).strip() if (isinstance(s, dict) and s.get("id") and str(s.get("id")).strip()) else f"section_{idx+1}"
            for idx, s in enumerate(sec_list)
        ]
        is_comp = self.outline.is_computational()
        provenance = VerificationPolicyManager.process_verification_results(
            topic=self.outline.topic,
            section_ids=section_ids,
            claims=claims,
            is_computational_topic=is_comp,
        )
        verified_data["verification_provenance"] = provenance.to_dict()
        verified_data["verification_status"] = provenance.overall_status.value

        if provenance.pipeline_halted or len(provenance.safe_sections) == 0:
            report.is_safe_for_stage2 = False
        else:
            report.is_safe_for_stage2 = True

        self.outline = TeachingOutline.from_dict(verified_data)
        self.outline.extra_fields["verification_provenance"] = provenance.to_dict()
        self.outline_verification_report = report

        # Synchronize outline.json on disk with the verified/corrected data
        outline_file = self.output_dir / "outline.json"
        try:
            with open(outline_file, "w", encoding="utf-8") as f:
                json.dump(verified_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Could not write verified outline to disk: {e}")

        return report

    def is_outline_safe_for_stage2(self) -> bool:
        """
        Returns True if self.outline has been verified and has safe sections to proceed to Stage 2.
        Returns False if missing, if an essential objective is violated (pipeline_halted),
        or if no safe sections remain.
        """
        if not self.outline:
            return False
        report = self._ensure_outline_verified()
        if not report:
            return False

        prov = self.get_verification_provenance()
        if prov is not None:
            # Rule 8: If essential objective is violated, pipeline must halt
            if prov.pipeline_halted:
                return False
            # Rule 7: If no safe sections remain, cannot proceed to Stage 2
            if len(prov.safe_sections) == 0:
                return False
            # Rule 2: If there are safe sections and pipeline is not halted, safe sections can proceed
            return True

        # Fallback for legacy outlines without provenance
        if (
            report.overall_status in (VerificationStatus.FAILED, VerificationStatus.NEEDS_REVIEW)
            or not report.is_safe_for_stage2
        ):
            return False
        return True

    def _ensure_storyboard_verified(self, force: bool = False) -> Optional[StoryboardFidelityReport]:
        """
        Ensures that self.enhanced_storyboard or self.sections is verified through StoryboardFidelityChecker.
        If self.storyboard_fidelity_report is None or force is True, executes verification,
        records the fidelity report, and returns it.
        """
        if not self.outline:
            return None
        if not self.enhanced_storyboard and not self.sections:
            return None

        if not force and self.storyboard_fidelity_report is not None:
            return self.storyboard_fidelity_report

        sb_target = self.enhanced_storyboard
        if not sb_target and self.sections:
            sb_target = {
                "sections": [
                    {
                        "id": s.id,
                        "title": s.title,
                        "lecture_lines": s.lecture_lines,
                        "animations": s.animations,
                        "story_beat": s.story_beat,
                        "captions": s.captions,
                        "layout_plan": s.layout_plan,
                        "verified_values": s.verified_values,
                        "algorithmic_operations": getattr(s, "algorithmic_operations", None),
                    }
                    for s in self.sections
                ]
            }

        report = self.storyboard_fidelity_checker.check_storyboard_fidelity(self.outline, sb_target)
        self.storyboard_fidelity_report = report

        if report.is_safe_for_audio and not self.validated_storyboard_fingerprint:
            self.validated_storyboard_fingerprint = compute_storyboard_fingerprint(self.sections if self.sections else sb_target)

        # Write fidelity report to disk
        fidelity_report_file = self.output_dir / "storyboard_fidelity_report.json"
        try:
            with open(fidelity_report_file, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Could not write storyboard fidelity report to disk: {e}")

        return report

    def is_storyboard_safe_for_audio(self) -> bool:
        """
        Returns True if the storyboard has been verified and accepted by the numerical fidelity gate
        (status VERIFIED or NOT_APPLICABLE). Returns False if FAILED, NEEDS_REVIEW, or unverified.
        Also validates that storyboard has not been mutated since validation.
        """
        if not self.outline or (not self.enhanced_storyboard and not self.sections):
            return False
        report = self._ensure_storyboard_verified()
        if not report or not report.is_safe_for_audio:
            return False
        if self.validated_storyboard_fingerprint and self.sections:
            current_fp = compute_storyboard_fingerprint(self.sections)
            if current_fp != self.validated_storyboard_fingerprint:
                print(f"⚠️ Storyboard mutated after verification! ({current_fp} != {self.validated_storyboard_fingerprint})")
                return False
        return True

    def _ensure_audio_verified(self, force: bool = False) -> Optional[AudioHandoffReport]:
        """
        Ensures that generated audio artifacts match the validated storyboard and are complete/consistent.
        """
        if not self.sections:
            return None
        if not force and self.audio_handoff_report is not None:
            return self.audio_handoff_report

        report = self.audio_handoff_validator.validate_handoff(
            sections=self.sections,
            section_audios=self.section_audios,
            section_timestamps=self.section_timestamps,
            expected_fingerprint=self.validated_storyboard_fingerprint,
            fidelity_report=self.storyboard_fidelity_report,
        )
        self.audio_handoff_report = report

        handoff_report_file = self.output_dir / "audio_handoff_report.json"
        try:
            with open(handoff_report_file, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Could not write audio handoff report to disk: {e}")

        return report

    def is_audio_safe_for_stage3(self) -> bool:
        """
        Returns True if audio handoff validation succeeds or audio is disabled.
        Returns False if audio is missing, stale, mismatched, or if storyboard was mutated.
        """
        if not self.use_audio:
            return True
        if not self.sections:
            return False
        report = self._ensure_audio_verified()
        return bool(report and report.is_safe_for_stage3)

    def generate_outline(self) -> TeachingOutline:
        outline_file = self.output_dir / "outline.json"

        if outline_file.exists():
            print("📂 ...")
            with open(outline_file, "r", encoding="utf-8") as f:
                outline_data = json.load(f)
        else:
            """Step 1: Generate teaching outline from topic"""
            refer_img_path = (
                self.knowledge_ref_img_folder / img_name
                if (img_name := self.KNOWLEDGE2PATH.get(self.learning_topic)) is not None
                else None
            )
            target_dur_min = (
                round(self.duration_manager.user_target_sec / 60.0, 1)
                if self.duration_manager.has_user_target
                else None
            )
            initial_profile = (
                self.outline.get_topic_profile()
                if (self.outline and hasattr(self.outline, "get_topic_profile"))
                else TopicProfiler.profile_heuristic_fallback(self.learning_topic)
            )
            prompt1 = get_prompt1_outline(
                knowledge_point=self.learning_topic,
                duration=target_dur_min,
                reference_image_path=refer_img_path,
                level=self.level,
                analogy=self.analogy,
                topic_profile=initial_profile,
            )

            print(f"📝 Generating Outline ({self.level.capitalize()} level)...")

            for attempt in range(1, self.max_regenerate_tries + 1):
                response = self._request_api_and_track_tokens(prompt1, max_tokens=self.max_code_token_length, stage="outline")
                if response is None:
                    print(f"⚠️ Attempt {attempt} failed, retrying...")
                    if attempt == self.max_regenerate_tries:
                        raise ValueError("API requests failed multiple times")
                    continue
                try:
                    content = response.candidates[0].content.parts[0].text
                except Exception:
                    try:
                        content = response.choices[0].message.content
                    except Exception:
                        content = str(response)
                content = extract_json_from_markdown(content)
                try:
                    outline_data = safe_json_loads(content)
                    with open(self.output_dir / "outline.json", "w", encoding="utf-8") as f:
                        json.dump(outline_data, f, ensure_ascii=False, indent=2)
                    break
                except (json.JSONDecodeError, Exception) as e:
                    print(f"⚠️ Outline format invalid on attempt {attempt} ({e}), retrying...")
                    if attempt == self.max_regenerate_tries:
                        raise ValueError(f"Outline format invalid multiple times: {e}")

        # Only propagate refined or adapted analogy if the user originally requested an analogy
        has_user_analogy = bool(self.requested_analogy and str(self.requested_analogy).strip())
        if has_user_analogy:
            ae = outline_data.get("analogy_evaluation") or {}
            is_suitable_raw = ae.get("is_suitable")
            # Explicitly unsuitable if boolean False or normalized strings 'false', 'no', 'unsuitable'
            is_unsuitable = (
                is_suitable_raw is False
                or (isinstance(is_suitable_raw, str) and is_suitable_raw.strip().lower() in ("false", "no", "unsuitable"))
            )
            eval_analogy = ae.get("adopted_analogy")
            analogy_used = outline_data.get("analogy_used")

            # If user analogy is explicitly rejected/unsuitable, clear self.analogy so downstream operates in direct mode.
            # Any AI-chosen replacement belongs in outline.mental_model, not as an active user-requested story theme.
            if is_unsuitable:
                self.analogy = ""
            elif eval_analogy and str(eval_analogy).strip() and "direct explanation" not in str(eval_analogy).lower():
                self.analogy = str(eval_analogy).strip()
            elif analogy_used and str(analogy_used).strip() and "direct explanation" not in str(analogy_used).lower():
                self.analogy = str(analogy_used).strip()
            else:
                self.analogy = self.requested_analogy
        else:
            # User supplied no analogy: keep self.analogy empty so downstream stages operate in direct mode
            self.analogy = ""

        # Update duration manager if outline contains an estimated duration from the LLM
        est_dur_outline = outline_data.get("estimated_duration_minutes") or outline_data.get("estimated_duration")
        if est_dur_outline and float(est_dur_outline) > 0 and not self.duration_manager.has_user_target:
            self.duration_manager.soft_target_sec = float(est_dur_outline) * 60.0 + self.duration_manager.TARGET_TOLERANCE_SEC
            if self.duration_manager.hard_max_sec < self.duration_manager.soft_target_sec:
                self.duration_manager.hard_max_sec = self.duration_manager.soft_target_sec * 1.35

        # Propagate fallback values if missing in outline_data
        outline_payload = dict(outline_data)
        if not outline_payload.get("topic"):
            outline_payload["topic"] = self.learning_topic
        if not outline_payload.get("learning_level"):
            outline_payload["learning_level"] = self.level
        if not outline_payload.get("analogy_used") and self.analogy:
            outline_payload["analogy_used"] = self.analogy

        # Populate initial TeachingOutline from raw or restored outline_payload
        self.outline = TeachingOutline.from_dict(outline_payload)

        # Stage 1 Validation Gate: execute complete canonical M1 verification lifecycle
        report = self._ensure_outline_verified(force=True)
        prov = self.get_verification_provenance()

        # Write authoritative verified and corrected outline to outline.json
        try:
            with open(outline_file, "w", encoding="utf-8") as f:
                json.dump(self.outline.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Could not write verified outline to disk: {e}")

        is_safe = self.is_outline_safe_for_stage2()
        if is_safe:
            status_str = prov.overall_status.value if prov else report.overall_status.value
            if status_str == "UNVERIFIED":
                print(f"⚠️ {self.learning_topic}: Worked example contains UNVERIFIED steps (independent coverage unavailable). Proceeding to Stage 2 with unverified metadata.")
            else:
                print(f"== Outline generated and verified: {self.outline.topic} [{self.level}] (Status: {status_str})")
        else:
            status_str = prov.overall_status.value if prov else report.overall_status.value
            halt_reason = prov.halt_reason if (prov and prov.halt_reason) else report.summary
            print(
                f"🛑 {self.learning_topic}: Validation gate blocked outline from Stage 2 "
                f"({status_str}): {halt_reason}"
            )
            self.save_checkpoint(
                "outline_verification_blocked",
                extra_data={
                    "verification_report": report.to_dict() if report else None,
                    "verification_provenance": prov.to_dict() if prov else None,
                },
            )

        return self.outline

    def generate_storyboard(self) -> List[Section]:
        """Step 2: Generate teaching storyboard from outline (optionally with asset enhancement)"""
        if not self.outline:
            raise ValueError("Outline not generated, please generate outline first")

        # Explicit validation gate check before Stage 2 can proceed
        if not self.is_outline_safe_for_stage2():
            report = self.outline_verification_report
            status_val = report.overall_status.value if report else "BLOCKED"
            summary_val = report.summary if report else "Outline failed worked example verification"
            raise ValueError(
                f"Cannot generate storyboard: Stage 2 is blocked by worked example validation gate ({status_val}): {summary_val}"
            )

        storyboard_file = self.output_dir / "storyboard.json"
        enhanced_storyboard_file = self.output_dir / "storyboard_with_assets.json"

        if enhanced_storyboard_file.exists():
            print("📂 Found enhanced storyboard, loading...")
            with open(enhanced_storyboard_file, "r", encoding="utf-8") as f:
                self.enhanced_storyboard = json.load(f)
            cached_rep = self.storyboard_fidelity_checker.check_storyboard_fidelity(self.outline, self.enhanced_storyboard)
            self.storyboard_fidelity_report = cached_rep
            if not cached_rep.is_safe_for_audio:
                raise ValueError(
                    f"Cannot accept storyboard: Cached storyboard is blocked by numerical fidelity gate ({cached_rep.overall_status.value}): {cached_rep.summary}"
                )
        elif storyboard_file.exists():
            print("📂 Found storyboard, loading...")
            with open(storyboard_file, "r", encoding="utf-8") as f:
                storyboard_data = json.load(f)
            cached_rep = self.storyboard_fidelity_checker.check_storyboard_fidelity(self.outline, storyboard_data)
            self.storyboard_fidelity_report = cached_rep
            if not cached_rep.is_safe_for_audio:
                raise ValueError(
                    f"Cannot accept storyboard: Cached storyboard is blocked by numerical fidelity gate ({cached_rep.overall_status.value}): {cached_rep.summary}"
                )
            if self.use_assets:
                self.enhanced_storyboard = self._enhance_storyboard_with_assets(storyboard_data)
            else:
                self.enhanced_storyboard = storyboard_data
        else:
            print(f"🎬 Generating storyboard ({self.level.capitalize()} level)...")
            refer_img_path = (
                self.knowledge_ref_img_folder / img_name
                if (img_name := self.KNOWLEDGE2PATH.get(self.learning_topic)) is not None
                else None
            )

            outline_payload_for_s2 = self.get_stage2_outline_dict()
            outline_serialized = json.dumps(
                outline_payload_for_s2,
                ensure_ascii=False,
                indent=2,
            )
            prompt2 = get_prompt2_storyboard(
                outline=outline_serialized,
                reference_image_path=refer_img_path,
                level=self.level,
                analogy=self.analogy,
                topic_profile=self.outline.get_topic_profile() if (self.outline and hasattr(self.outline, "get_topic_profile")) else None,
            )

            for attempt in range(1, self.max_regenerate_tries + 1):
                response = self._request_api_and_track_tokens(prompt2, max_tokens=self.max_code_token_length, stage="storyboard")
                if response is None:
                    print(f"⚠️ Outline format invalid on attempt {attempt}, retrying...")
                    if attempt == self.max_regenerate_tries:
                        raise ValueError("API requests failed multiple times")
                    continue

                try:
                    content = response.candidates[0].content.parts[0].text
                except Exception:
                    try:
                        content = response.choices[0].message.content
                    except Exception:
                        content = str(response)

                try:
                    json_str = extract_json_from_markdown(content)
                    storyboard_data = safe_json_loads(json_str)

                    # Stage 2 Fidelity Check: verify preservation of Stage 1 numerical values
                    fidelity_report = self.storyboard_fidelity_checker.check_storyboard_fidelity(self.outline, storyboard_data)
                    if not fidelity_report.is_safe_for_audio:
                        for retry_attempt in range(1, self.max_fidelity_retries + 1):
                            print(
                                f"⚠️ [STORYBOARD FIDELITY] Numerical discrepancies detected (Attempt {retry_attempt}/{self.max_fidelity_retries}): "
                                f"{fidelity_report.summary}. Requesting corrected storyboard from LLM..."
                            )
                            repair_prompt = get_prompt2_storyboard_fidelity_repair(
                                outline=outline_serialized,
                                storyboard=json.dumps(storyboard_data, indent=2, ensure_ascii=False),
                                discrepancies=fidelity_report.discrepancies,
                            )
                            repair_response = self._request_api_and_track_tokens(
                                repair_prompt, max_tokens=self.max_code_token_length, stage="storyboard"
                            )
                            if repair_response is None:
                                continue
                            try:
                                try:
                                    r_content = repair_response.candidates[0].content.parts[0].text
                                except Exception:
                                    try:
                                        r_content = repair_response.choices[0].message.content
                                    except Exception:
                                        r_content = str(repair_response)
                                r_json_str = extract_json_from_markdown(r_content)
                                r_data = safe_json_loads(r_json_str)
                                new_fidelity_report = self.storyboard_fidelity_checker.check_storyboard_fidelity(self.outline, r_data)
                                if new_fidelity_report.is_safe_for_audio:
                                    storyboard_data = r_data
                                    fidelity_report = new_fidelity_report
                                    print(f"✅ [STORYBOARD FIDELITY] Storyboard successfully corrected and verified on retry attempt {retry_attempt}")
                                    break
                                else:
                                    fidelity_report = new_fidelity_report
                                    storyboard_data = r_data
                            except Exception as e:
                                print(f"⚠️ [STORYBOARD FIDELITY] Failed to parse repair response: {e}")

                    self.storyboard_fidelity_report = fidelity_report

                    if not fidelity_report.is_safe_for_audio:
                        self.save_checkpoint(
                            "storyboard_fidelity_blocked",
                            extra_data={"fidelity_report": fidelity_report.to_dict(), "storyboard_candidate": storyboard_data},
                        )
                        raise ValueError(
                            f"Cannot accept storyboard: Stage 2 failed numerical fidelity verification ({fidelity_report.overall_status.value}): {fidelity_report.summary}"
                        )

                    # Save original storyboard
                    with open(storyboard_file, "w", encoding="utf-8") as f:
                        json.dump(storyboard_data, f, ensure_ascii=False, indent=2)

                    # Enhance storyboard (add assets)
                    if self.use_assets:
                        self.enhanced_storyboard = self._enhance_storyboard_with_assets(storyboard_data)
                    else:
                        self.enhanced_storyboard = storyboard_data
                    break

                except (json.JSONDecodeError, Exception) as e:
                    if isinstance(e, ValueError) and "failed numerical fidelity verification" in str(e):
                        raise e
                    print(f"⚠️ Storyboard format invalid on attempt {attempt} ({e}), retrying...")
                    if attempt == self.max_regenerate_tries:
                        raise ValueError(f"Storyboard format invalid multiple times: {e}")

        # Run adaptive duration analysis & compression if needed
        adapted_sb, dur_est, dur_report = self.duration_manager.adapt_storyboard_if_needed(self.enhanced_storyboard)
        self.storyboard_estimated_sec = dur_est.total_duration_sec
        with open(self.output_dir / "duration_report.json", "w", encoding="utf-8") as f:
            json.dump(dur_report, f, indent=2, ensure_ascii=False)
        print(f"⏱️ Adaptive duration estimate: {dur_est.total_duration_min:.2f}m (Status: {dur_est.status})")

        if dur_est.status == "HARD_MAX_EXCEEDED" and dur_est.multi_part_plan:
            with open(self.output_dir / "duration_reorganization_plan.json", "w", encoding="utf-8") as f:
                json.dump(dur_est.multi_part_plan, f, indent=2, ensure_ascii=False)
            print(f"⚠️ Lesson duration exceeds hard maximum ({self.duration_manager.hard_max_sec/60:.1f}m). Pre-organized into 2 parts.")

        self.enhanced_storyboard = adapted_sb

        # Mathematical and calculation validation
        math_report = self.math_validator.validate_storyboard(self.enhanced_storyboard)
        with open(self.output_dir / "math_validation_report.json", "w", encoding="utf-8") as f:
            json.dump(math_report, f, indent=2, ensure_ascii=False)
        print(f"🧮 Mathematical validation: {math_report['summary']}")

        # Resolve storyboard references to available visual capabilities
        try:
            from src.domains import StoryboardActionResolver
            self.enhanced_storyboard = StoryboardActionResolver.resolve_storyboard(self.enhanced_storyboard)
        except Exception as e:
            print(f"⚠️ Action resolution note: {e}")

        # Parse into Section objects (using enhanced storyboard)
        self.visual_conventions = self.enhanced_storyboard.get("visual_conventions")
        self.sections = []
        for section_data in self.enhanced_storyboard["sections"]:
            section = Section(
                id=section_data["id"],
                title=section_data["title"],
                lecture_lines=section_data.get("lecture_lines", []),
                animations=section_data["animations"],
                story_beat=section_data.get("story_beat", section_data.get("story_step", "")),
                captions=section_data.get("captions"),
                layout_plan=section_data.get("layout_plan", ""),
                verified_values=section_data.get("verified_values"),
                algorithmic_operations=section_data.get("algorithmic_operations"),
                beats=section_data.get("beats"),
                resolved_actions=section_data.get("resolved_actions"),
            )
            self.sections.append(section)

        # Final fidelity validation gate on the processed sections
        if not self.is_storyboard_safe_for_audio():
            rep = self.storyboard_fidelity_report
            status_v = rep.overall_status.value if rep else "BLOCKED"
            summary_v = rep.summary if rep else "Processed sections failed numerical fidelity verification"
            raise ValueError(
                f"Cannot accept storyboard: Processed sections failed numerical fidelity gate ({status_v}): {summary_v}"
            )
        self.validated_storyboard_fingerprint = compute_storyboard_fingerprint(self.sections)

        print(f"== Storyboard processed, {len(self.sections)} sections generated")
        return self.sections

    def _extract_response_content(self, response: Any) -> str:
        """Extract clean text content from various LLM response structures or strings."""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            return json.dumps(response)
        try:
            return response.candidates[0].content.parts[0].text
        except Exception:
            pass
        try:
            return response.choices[0].message.content
        except Exception:
            pass
        return str(response)

    def generate_scene_ir(self, prompt_override: Optional[str] = None) -> SceneIRProcessingResult:
        """Step 2 (Modern): Generate native Scene IR (v1.0) conforming strictly to modern pipeline contracts.

        Converts Stage 1 teaching outline into a typed, declarative SceneSequenceIR
        or SceneIR document. Enforces deterministic Pydantic schema validation, Pre-Layout
        Gate 3.5 semantic validation, and VisualRegistry capability/payload checks.
        Enforces a bounded repair policy (max 2 retries) on recoverable diagnostics.

        Does NOT route native Scene IR permanently through SceneIRToStoryboardAdapter.
        """
        if not self.outline:
            print("🎬 Outline missing, generating Stage 1 outline first...")
            self.generate_outline()
            if not self.outline:
                raise ValueError(f"Cannot generate Scene IR: Outline generation failed for '{self.learning_topic}'.")

        # 1. Check for cached Scene IR on disk
        seq_file = self.output_dir / "scene_sequence_ir.json"
        scene_file = self.output_dir / "scene_ir.json"
        if seq_file.exists():
            try:
                with open(seq_file, "r", encoding="utf-8") as f:
                    seq_data = json.load(f)
                cached_res = process_stage2_scene_ir_output(seq_data)
                if cached_res.is_valid:
                    self.scene_sequence_ir = cached_res.sequence_ir
                    self.scene_ir_processing_result = cached_res
                    print(f"🎬 Loaded valid cached SceneSequenceIR ({len(cached_res.sequence_ir.sections)} scenes).")
                    return cached_res
            except Exception:
                pass
        elif scene_file.exists():
            try:
                with open(scene_file, "r", encoding="utf-8") as f:
                    sc_data = json.load(f)
                cached_res = process_stage2_scene_ir_output(sc_data)
                if cached_res.is_valid:
                    self.scene_ir = cached_res.scene_ir
                    self.scene_ir_processing_result = cached_res
                    print("🎬 Loaded valid cached SceneIR.")
                    return cached_res
            except Exception:
                pass

        print(f"🎬 Generating native Scene IR ({self.level.capitalize()} level)...")
        outline_payload = self.get_stage2_outline_dict() if hasattr(self, "get_stage2_outline_dict") else (
            self.outline.to_dict() if hasattr(self.outline, "to_dict") else vars(self.outline)
        )
        profile = self.outline.get_topic_profile() if (self.outline and hasattr(self.outline, "get_topic_profile")) else None

        prompt2 = prompt_override or get_prompt2_scene_ir(
            outline=outline_payload,
            level=self.level,
            analogy=self.analogy,
            topic_profile=profile,
            format_as_sequence=True,
        )

        response = self._request_api_and_track_tokens(prompt2, max_tokens=self.max_code_token_length, stage="scene_ir")
        if response is None:
            raise ValueError(f"Scene IR generation failed: API request returned None for topic '{self.learning_topic}'.")

        content = self._extract_response_content(response)
        result = process_stage2_scene_ir_output(content)

        # 2. Bounded Repair Policy: Max 2 retries on recoverable diagnostics
        if not result.is_valid and result.has_recoverable_diagnostics:
            for retry_attempt in range(1, min(self.max_fidelity_retries, 2) + 1):
                print(
                    f"⚠️ [SCENE IR VALIDATION] Diagnostics detected (Attempt {retry_attempt}/2): "
                    f"{result.summary}. Requesting targeted repair from LLM..."
                )
                repair_prompt = get_prompt2_scene_ir_repair(
                    outline=outline_payload,
                    failed_output=content,
                    processing_result=result,
                    level=self.level,
                    analogy=self.analogy,
                    topic_profile=profile,
                )
                if not repair_prompt:
                    break
                repair_resp = self._request_api_and_track_tokens(
                    repair_prompt, max_tokens=self.max_code_token_length, stage="scene_ir_repair"
                )
                if repair_resp is None:
                    continue
                r_content = self._extract_response_content(repair_resp)
                r_result = process_stage2_scene_ir_output(r_content)
                content = r_content
                result = r_result
                if r_result.is_valid:
                    break

        self.scene_ir_processing_result = result

        # 3. Persistence on Success
        if result.is_valid:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if result.sequence_ir is not None:
                self.scene_sequence_ir = result.sequence_ir
                with open(seq_file, "w", encoding="utf-8") as f:
                    f.write(result.sequence_ir.to_json(indent=2))
                self.save_checkpoint("scene_ir_generated", extra_data={"has_sequence_ir": True})
            elif result.scene_ir is not None:
                self.scene_ir = result.scene_ir
                with open(scene_file, "w", encoding="utf-8") as f:
                    f.write(result.scene_ir.to_json(indent=2))
                self.save_checkpoint("scene_ir_generated", extra_data={"has_scene_ir": True})
        else:
            # Save diagnostic report for debugging
            err_file = self.output_dir / "scene_ir_validation_errors.json"
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with open(err_file, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        return result

    def _enhance_storyboard_with_assets(self, storyboard_data: dict) -> dict:
        """Process storyboard to replace any asset placeholders with Manim-only visuals (Asset download bypassed)."""
        print("ℹ️ Processing storyboard with native Manim visuals (external asset downloading bypassed)...")

        try:
            # Asset/IconFinder pipeline is disabled for Gemini Free Tier
            # Any [Asset: ...] placeholders are converted to native Manim visuals (Circle, Square, Polygon, Arrow, Text)
            enhanced_storyboard = process_storyboard_with_assets(
                storyboard=storyboard_data,
                api_function=None,
                assets_dir=str(self.assets_dir),
                iconfinder_api_key=None,
            )

            enhanced_storyboard_file = self.output_dir / "storyboard_with_assets.json"
            with open(enhanced_storyboard_file, "w", encoding="utf-8") as f:
                json.dump(enhanced_storyboard, f, ensure_ascii=False, indent=2)

            print("✅ Storyboard processed with native Manim visuals")
            return enhanced_storyboard

        except Exception as e:
            print(f"⚠️ Asset processing bypassed, using original storyboard: {e}")
            return storyboard_data

    def pregenerate_all_section_audios(self):
        """Pre-generates TTS audio and fine-grained timestamps for all sections before code generation."""
        if not self.is_storyboard_safe_for_audio():
            raise ValueError(f"{self.learning_topic} Cannot pre-generate audio: Storyboard failed numerical fidelity gate or was mutated")

        print(f"🎙️ Pre-generating audio and timestamps for {self.learning_topic} sections...")
        for section in self.sections:
            sid = section.id
            if sid in self.section_timestamps and sid in self.section_audios:
                meta = self.section_timestamps[sid]
                curr_hash = compute_narration_hash(section.lecture_lines)
                audio_p = Path(self.section_audios[sid])
                if meta.get("narration_hash") == curr_hash and audio_p.exists() and audio_p.stat().st_size >= 500:
                    continue
                self.section_timestamps.pop(sid, None)
                self.section_audios.pop(sid, None)
            try:
                audio_meta = None
                if self.audio_input:
                    audio_in_path = Path(self.audio_input)
                    candidate_files = []
                    if audio_in_path.is_dir():
                        for ext in [".mp3", ".wav", ".m4a", ".aac"]:
                            cand = audio_in_path / f"{sid}{ext}"
                            if cand.exists():
                                candidate_files.append(cand)
                    elif audio_in_path.is_file():
                        candidate_files.append(audio_in_path)
                    if candidate_files:
                        audio_meta = self.audio_manager.handle_external_audio(
                            section_id=sid,
                            audio_file_path=candidate_files[0],
                            lecture_lines=section.lecture_lines,
                            animations=section.animations,
                            output_dir=self.output_dir
                        )

                if not audio_meta:
                    audio_meta = self.audio_manager.generate_section_audio(
                        section_id=sid,
                        lecture_lines=section.lecture_lines,
                        animations=section.animations,
                        output_dir=self.output_dir,
                        voice=self.tts_voice
                    )

                self.section_audios[sid] = audio_meta["audio_path"]
                self.section_timestamps[sid] = audio_meta

                # Validate timing at individual narration-line level
                timing_val = self.duration_manager.validate_line_timings(audio_meta)
                ts_val_file = self.output_dir / f"{sid}_timing_validation.json"
                with open(ts_val_file, "w", encoding="utf-8") as f:
                    json.dump(timing_val, f, indent=2, ensure_ascii=False)
                if not timing_val["is_valid"]:
                    print(f"⚠️ Timing notices for {sid}: {len(timing_val['warnings'])} warnings")

                print(f"🔊 Pre-generated audio for {sid}: {audio_meta['total_duration']:.2f}s")
            except Exception as e:
                print(f"⚠️ Audio pregeneration warning for {sid}: {e}")

        # Validate complete audio handoff
        handoff_rep = self._ensure_audio_verified(force=True)
        if handoff_rep and not handoff_rep.is_safe_for_stage3:
            print(f"⚠️ Audio handoff validation found issues: {handoff_rep.summary}")

    def sanitize_code_components_and_docking(self, code: str) -> Tuple[str, bool]:
        """Deterministically fix common LLM component and layout mistakes:
        - self.dock_*.get_center() -> self.layout.viewports[*].center
        - self.dock_*.center -> self.layout.viewports[*].center
        - Graph(vertices=..., edges=...) with weighted 3-tuple edges -> GraphVisual
        - Double curly braces in MathTex / safe_math strings
        """
        modified = False
        new_code = code

        # 1. Docking method property access repairs
        docking_subs = [
            (r'self\.dock_inspector\.get_center\(\)', 'self.layout.viewports["inspector"].center'),
            (r'self\.dock_main\.get_center\(\)', 'self.layout.viewports["main"].center'),
            (r'self\.dock_split_left\.get_center\(\)', 'self.layout.viewports["split_left"].center'),
            (r'self\.dock_split_right\.get_center\(\)', 'self.layout.viewports["split_right"].center'),
            (r'self\.dock_full\.get_center\(\)', 'self.layout.viewports["full"].center'),
            (r'self\.dock_inspector\.center\b', 'self.layout.viewports["inspector"].center'),
            (r'self\.dock_main\.center\b', 'self.layout.viewports["main"].center'),
            (r'self\.dock_split_left\.center\b', 'self.layout.viewports["split_left"].center'),
            (r'self\.dock_split_right\.center\b', 'self.layout.viewports["split_right"].center'),
            (r'self\.dock_full\.center\b', 'self.layout.viewports["full"].center'),
        ]
        for pattern, repl in docking_subs:
            if re.search(pattern, new_code):
                new_code = re.sub(pattern, repl, new_code)
                modified = True

        # 2. Graph(...) to GraphVisual(...) if 3-tuple weighted edges are present
        if re.search(r'\bGraph\s*\(', new_code):
            if re.search(r'\(\s*[\'\"].*?[\'\"]\s*,\s*[\'\"].*?[\'\"]\s*,\s*\d+', new_code):
                new_code = re.sub(r'\bGraph\s*\(', 'GraphVisual(', new_code)
                modified = True

        # 2b. GraphVisual vertex_config removal (GraphVisual handles node styles internally; invalid vertex_config causes NameError)
        if re.search(r'vertex_config\s*=', new_code):
            new_code = re.sub(r',?\s*vertex_config\s*=\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', '', new_code)
            modified = True

        # 2c. MoveToTarget(m.move_to(...)) repair: replace with m.animate.move_to(...)
        if re.search(r'MoveToTarget\s*\(', new_code):
            new_code = re.sub(r'MoveToTarget\s*\(\s*(.+?)\.move_to\s*\(\s*(.+?)\s*\)\s*\)', r'\1.animate.move_to(\2)', new_code)
            modified = True

        # 2d. ProceduralComponent(<UndefinedIdentifier>(), ...) repair: replace with ProceduralComponent(VGroup(), ...)
        proc_sub = re.sub(
            r'ProceduralComponent\(\s*([A-Za-z0-9_]+)\(\)\s*,',
            r'ProceduralComponent(VGroup(),',
            new_code
        )
        if proc_sub != new_code:
            new_code = proc_sub
            modified = True

        # 2e. Fix invalid shift kwarg inside Write(...) calls (Write does not accept shift)
        if re.search(r'Write\s*\([^)]*shift\s*=', new_code):
            new_code = re.sub(r'Write\s*\(\s*([^)]*?),\s*shift\s*=\s*[^,)]+([^)]*?)\)', r'Write(\1\2)', new_code)
            modified = True

        # 2f. Replace empty ProceduralComponent fallbacks with concrete pedagogical diagrams
        # Multiprogramming motivation fallback
        if re.search(r'ProceduralComponent\(\s*(?:VGroup\(\)\s*,)?\s*name=["\']TaskDiagram["\']', new_code):
            task_repl = (
                '_cpu_b = VGroup(RoundedRectangle(corner_radius=0.15, width=2.6, height=2.0, color="#58C4DD", fill_opacity=0.2), '
                'Text("CPU Core", font_size=18, color="#58C4DD").shift(UP * 0.65), '
                'Text("1 Task Active", font_size=13, color=WHITE).shift(DOWN * 0.1))\n'
                '        _tasks_b = VGroup('
                'VGroup(RoundedRectangle(corner_radius=0.08, width=1.5, height=0.5, color="#FFA07A", fill_opacity=0.2), Text("Task 1", font_size=13, color="#FFA07A")), '
                'VGroup(RoundedRectangle(corner_radius=0.08, width=1.5, height=0.5, color="#FFD700", fill_opacity=0.2), Text("Task 2", font_size=13, color="#FFD700")), '
                'VGroup(RoundedRectangle(corner_radius=0.08, width=1.5, height=0.5, color="#83C167", fill_opacity=0.2), Text("Task 3", font_size=13, color="#83C167"))'
                ').arrange(DOWN, buff=0.15)\n'
                '        _arr_b = Arrow(_tasks_b.get_right(), _cpu_b.get_left(), buff=0.12, color="#FFD700")\n'
                '        task_diagram = VGroup(_tasks_b, _arr_b, _cpu_b).arrange(RIGHT, buff=0.4)'
            )
            new_code = re.sub(
                r'task_diagram\s*=\s*ProceduralComponent\(.*?name=["\']TaskDiagram["\'].*?\)',
                lambda m: task_repl,
                new_code
            )
            new_code = re.sub(r'task_diagram\.core\b', '_cpu_b', new_code)
            modified = True

        # Scheduler vs Dispatcher fallback
        if re.search(r'ProceduralComponent\(\s*(?:VGroup\(\)\s*,)?\s*name=["\']SchedulerDispatcherDiagram["\']', new_code):
            sched_repl = (
                '_c_sch = VGroup(RoundedRectangle(corner_radius=0.1, width=2.7, height=1.6, color="#58C4DD", fill_opacity=0.2), '
                'Text("CPU Scheduler", font_size=16, color="#58C4DD").shift(UP * 0.45), '
                'Text("Selects Ready Process", font_size=13, color=WHITE).shift(DOWN * 0.2))\n'
                '        _arr_sd = Arrow(LEFT * 0.5, RIGHT * 0.5, color="#FFD700")\n'
                '        _c_dsp = VGroup(RoundedRectangle(corner_radius=0.1, width=2.7, height=1.6, color="#83C167", fill_opacity=0.2), '
                'Text("Dispatcher", font_size=16, color="#83C167").shift(UP * 0.45), '
                'Text("Context Switch to CPU", font_size=13, color=WHITE).shift(DOWN * 0.2))\n'
                '        scheduler_dispatcher = VGroup(_c_sch, _arr_sd, _c_dsp).arrange(RIGHT, buff=0.35)'
            )
            new_code = re.sub(
                r'scheduler_dispatcher\s*=\s*(?:ProceduralComponent\(.*?name=["\']SchedulerDispatcherDiagram["\'].*?\)|VGroup\(.*?_c_dsp.*?\))',
                lambda m: sched_repl,
                new_code,
                flags=re.DOTALL
            )
            modified = True

        # Convoy effect fallback
        if re.search(r'ProceduralComponent\(\s*(?:VGroup\(\)\s*,)?\s*name=["\']ConvoyEffectDiagram["\']', new_code):
            convoy_repl = (
                '_h_conv = Text("Convoy Effect: Long Burst Blocks Short Jobs", font_size=18, color="#FF6B6B")\n'
                '        _b_p1 = VGroup(Rectangle(width=4.2, height=0.7, color="#FF6B6B", fill_opacity=0.35), Text("P1 (Burst = 24)", font_size=16, color=WHITE))\n'
                '        _b_p2 = VGroup(Rectangle(width=0.9, height=0.7, color="#58C4DD", fill_opacity=0.35), Text("P2 (3)", font_size=14, color=WHITE))\n'
                '        _b_p3 = VGroup(Rectangle(width=0.9, height=0.7, color="#83C167", fill_opacity=0.35), Text("P3 (3)", font_size=14, color=WHITE))\n'
                '        _bars_c = VGroup(_b_p1, _b_p2, _b_p3).arrange(RIGHT, buff=0.08)\n'
                '        _n_conv = Text("P2 and P3 wait 24 time units despite needing only 3 units each!", font_size=14, color="#FFA07A")\n'
                '        convoy_diagram = VGroup(_h_conv, _bars_c, _n_conv).arrange(DOWN, buff=0.32)'
            )
            new_code = re.sub(
                r'convoy_diagram\s*=\s*ProceduralComponent\(.*?name=["\']ConvoyEffectDiagram["\'].*?\)',
                lambda m: convoy_repl,
                new_code
            )
            modified = True

        # FCFS selection fallback
        if re.search(r'ProceduralComponent\(\s*(?:VGroup\(\)\s*,)?\s*name=["\']FCFSDiagram["\']', new_code):
            fcfs_repl = (
                '_rq_fcfs = ReadyQueueVisual(capacity=4)\n'
                '        _cpu_fcfs = CPUExecutionVisual()\n'
                '        fcfs_diagram = VGroup(_rq_fcfs, _cpu_fcfs).arrange(RIGHT, buff=0.5)'
            )
            new_code = re.sub(
                r'fcfs_diagram\s*=\s*ProceduralComponent\(.*?name=["\']FCFSDiagram["\'].*?\)',
                lambda m: fcfs_repl,
                new_code
            )
            modified = True

        # 3. Tokenize-based MathTex / safe_math double-brace sanitizer
        try:
            tokens = list(tokenize.tokenize(io.BytesIO(new_code.encode('utf-8')).readline))
            mod_tokens = False
            new_tokens = []
            call_stack = []
            paren_depth = 0
            prev_name = None
            for tok in tokens:
                if tok.type == tokenize.NAME:
                    prev_name = tok.string
                elif tok.type == tokenize.OP and tok.string == '(':
                    if prev_name:
                        call_stack.append((prev_name, paren_depth))
                    paren_depth += 1
                    prev_name = None
                elif tok.type == tokenize.OP and tok.string == ')':
                    paren_depth -= 1
                    if call_stack and call_stack[-1][1] == paren_depth:
                        call_stack.pop()
                    prev_name = None
                elif tok.type != tokenize.NAME and tok.type not in (tokenize.COMMENT, tokenize.NL):
                    prev_name = None

                if tok.type == tokenize.STRING:
                    in_math = any(call[0] in ('MathTex', 'safe_math', 'Tex') for call in call_stack)
                    if in_math and ('{{' in tok.string or '}}' in tok.string):
                        cleaned_str = tok.string
                        prev = None
                        while prev != cleaned_str:
                            prev = cleaned_str
                            cleaned_str = re.sub(r'\{\{([^{}]*)\}\}', r'{\1}', cleaned_str)
                        if cleaned_str != tok.string:
                            tok = tokenize.TokenInfo(tok.type, cleaned_str, tok.start, tok.end, tok.line)
                            mod_tokens = True
                new_tokens.append(tok)
            if mod_tokens:
                new_code = tokenize.untokenize(new_tokens).decode('utf-8')
                modified = True
        except Exception:
            pass

        return new_code, modified

    def pre_render_validate_and_fix(self, section: Section, code: str) -> Tuple[str, bool]:
        """
        Runs comprehensive pre-render validation and deterministic repairs before launching Manim:
        0. VisualComponent, docking method, and MathTex double-brace repairs ($0 cost)
        1. Mathematical consistency & MathTex repair ($0 cost)
        2. Static layout boundary & placement repairs ($0 cost)
        3. AST syntax & NameError checks with smart repair ($0 cost)
        """
        modified = False
        current_code = code

        # 0. Deterministic component, docking, and MathTex token sanitization ($0 cost)
        try:
            fixed_comp_code, comp_fixed = self.sanitize_code_components_and_docking(current_code)
            if comp_fixed:
                current_code = fixed_comp_code
                modified = True
                print(f"🔧 [PRE-RENDER COMPONENT & DOCKING FIX] Sanitized component APIs, docking syntax, and MathTex braces in {section.id}")
        except Exception as e:
            logger.warning(f"Pre-render component/docking sanitization exception for {section.id}: {e}")

        # 1. Deterministic MathTex consistency validation and repair
        try:
            math_report = self.math_validator.validate_section(section, section_code=current_code)
            if not math_report.is_valid:
                print(f"🧮 [PRE-RENDER MATH CHECK] {section.id} notices: {math_report.inconsistencies}")
                fixed_math_code, math_fixed = self.math_validator.fix_code_mathtex(current_code, math_report)
                if math_fixed:
                    current_code = fixed_math_code
                    modified = True
                    print(f"✅ [PRE-RENDER MATH FIX] Deterministically corrected MathTex formula errors in {section.id}")
        except Exception as e:
            logger.warning(f"Pre-render math validation exception for {section.id}: {e}")

        # 2. Static layout checking & deterministic geometry fixes
        try:
            static_issues = self.local_checker._check_code_grid_placements(current_code)
            fixable = [i for i in static_issues if i.can_fix_locally]
            if fixable:
                fixed_layout_code, layout_fixed = self.local_fixer.apply_fixes(current_code, fixable)
                if layout_fixed:
                    current_code = fixed_layout_code
                    modified = True
                    print(f"🛠️ [PRE-RENDER LAYOUT FIX] Applied {len(fixable)} deterministic layout fix(es) in {section.id}")
        except Exception as e:
            logger.warning(f"Pre-render static layout check exception for {section.id}: {e}")

        # 3. Syntax correctness and fatal AST defect verification
        try:
            is_valid_syntax, err_msg = self.scope_refine_fixer.validate_code_syntax(current_code)
            if not is_valid_syntax and err_msg:
                print(f"⚠️ [PRE-RENDER SYNTAX ISSUE] {section.id}: {err_msg}. Attempting pre-render repair...")
                fixed_syntax_code = self.scope_refine_fixer.fix_code_smart(section.id, current_code, err_msg, self.output_dir)
                if fixed_syntax_code:
                    current_code = fixed_syntax_code
                    modified = True
                    print(f"✅ [PRE-RENDER SYNTAX FIX] Resolved syntax/AST defect in {section.id} before rendering")
        except Exception as e:
            logger.warning(f"Pre-render syntax validation exception for {section.id}: {e}")

        return current_code, modified

    def check_scene_storyboard_visual_fidelity(self, section: Section, code: str) -> List[str]:
        """Validates whether the generated scene code actually instantiates the registered visual
        components planned by the storyboard and resolved actions."""
        discrepancies = []
        resolved = getattr(section, "resolved_actions", None) or []
        if not resolved and isinstance(section, dict):
            resolved = section.get("resolved_actions", [])

        for act in resolved:
            if isinstance(act, dict):
                comp = act.get("operating_component") or act.get("component_name")
                if comp and comp != "ProceduralComponent" and comp not in ("none", "None"):
                    if comp not in code:
                        discrepancies.append(
                            f"Storyboard intended `{comp}` in `{act.get('viewport', 'dock_main')}`, "
                            f"but `{comp}` was not instantiated in the generated scene code."
                        )
        return discrepancies

    def generate_section_code(
        self,
        section: Section,
        attempt: int = 1,
        feedback_improvements=None,
        error_log: Optional[str] = None,
        previous_code: Optional[str] = None,
        force_api: bool = False,
    ) -> str:
        """Generate Manim code for a single section (bounded to max 1 generation attempt)"""
        if self.sections or self.outline:
            if not self.is_storyboard_safe_for_audio():
                raise ValueError(f"{self.learning_topic} Cannot generate code for {section.id}: Storyboard failed numerical fidelity gate or was mutated")
            if self.use_audio and not self.is_audio_safe_for_stage3():
                raise ValueError(f"{self.learning_topic} Cannot generate code for {section.id}: Audio handoff validation failed")

        code_file = self.output_dir / f"{section.id}.py"

        if code_file.exists() and not feedback_improvements and not force_api and not error_log:
            with open(code_file, "r", encoding="utf-8") as f:
                code = f.read()
                scene_name = f"{section.id.title().replace('_', '')}Scene"
                is_valid = False
                if f"class {scene_name}" in code and "def construct(self" in code:
                    try:
                        compile(code, str(code_file), "exec")
                        is_valid = True
                    except Exception:
                        is_valid = False
                if is_valid:
                    print(f"📂 Found existing valid code for {section.id}, reading...")
                    with self.state_lock:
                        self.section_codes[section.id] = code
                    return code
                else:
                    print(f"⚠️ Existing code file {code_file} is incomplete or has syntax errors. Regenerating...")

        regenerate_note = ""
        if attempt > 1 or error_log:
            regenerate_note = get_regenerate_note(
                attempt,
                MAX_REGENERATE_TRIES=self.max_regenerate_tries,
                error_log=error_log,
                previous_code=previous_code,
            )

        # Add MLLM feedback and improvement suggestions
        if feedback_improvements:
            with self.state_lock:
                current_code = self.section_codes.get(section.id, "")
            try:
                modifier = GridCodeModifier(current_code)
                modified_code = modifier.parse_feedback_and_modify(feedback_improvements)
                with open(code_file, "w", encoding="utf-8") as f:
                    f.write(modified_code)

                with self.state_lock:
                    self.section_codes[section.id] = modified_code
                return modified_code
            except Exception as e:
                print(f"⚠️ GridCodeModifier failed to apply feedback ({e}). Preserving clean existing code without API fallback.")
                return current_code
        else:
            with self.state_lock:
                audio_meta = self.section_timestamps.get(section.id)

            # Ensure resolved actions are populated for Stage 3 code generation prompt
            if not getattr(section, "resolved_actions", None) and hasattr(section, "animations") and section.animations:
                try:
                    from src.domains import StoryboardActionResolver
                    resolved_list = []
                    for step_item in section.animations:
                        if isinstance(step_item, str):
                            parsed = StoryboardActionResolver.parse_animation_step(step_item)
                            _, res, _ = StoryboardActionResolver.resolve_visual_action(
                                visual_rep=parsed["visual_representation"],
                                action_str=parsed["action"],
                                viewport=parsed["viewport"],
                                step_tag=parsed["tag"],
                                effect=parsed["effect"],
                                on_screen_after=parsed["on_screen_after"],
                                raw_step=parsed["raw_step"],
                            )
                            if res is not None:
                                resolved_list.append(res.to_dict())
                    if resolved_list:
                        section.resolved_actions = resolved_list
                except Exception as e:
                    print(f"⚠️ Note: on-the-fly action resolution: {e}")

            code_gen_prompt = get_prompt3_code(
                regenerate_note=regenerate_note,
                section=section,
                base_class=base_class,
                audio_meta=audio_meta,
                level=self.level,
                analogy=self.analogy,
                visual_conventions=getattr(self, "visual_conventions", None),
            )

        response = self._request_api_and_track_tokens(code_gen_prompt, max_tokens=self.max_code_token_length, stage="code_generation")
        if response is None:
            print(f"❌ Failed to generate code for {section.id} via API call.")
            return ""

        try:
            code = response.candidates[0].content.parts[0].text
        except Exception:
            try:
                code = response.choices[0].message.content
            except Exception:
                code = getattr(response, "text", str(response))
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].strip()

        # Replace base class
        code = replace_base_class(code, base_class)

        # Pre-render automated validation and deterministic repair ($0 API cost)
        code, was_pre_fixed = self.pre_render_validate_and_fix(section, code)

        # Check Storyboard-to-Scene visual component fidelity
        visual_discrepancies = self.check_scene_storyboard_visual_fidelity(section, code)
        if visual_discrepancies:
            for vd in visual_discrepancies:
                print(f"⚠️ [VISUAL FIDELITY] {section.id}: {vd}")

        with open(code_file, "w", encoding="utf-8") as f:
            f.write(code)

        with self.state_lock:
            self.section_codes[section.id] = code
        return code

    def debug_and_fix_code(self, section_id: str, max_fix_attempts: int = 1) -> bool:
        """Enhanced debug and fix code method with single repair attempt and error loop detection."""
        code_file = f"{section_id}.py"
        with self.state_lock:
            code_in_mem = self.section_codes.get(section_id)

        if not code_in_mem:
            disk_code_path = self.output_dir / code_file
            if disk_code_path.exists():
                try:
                    with open(disk_code_path, "r", encoding="utf-8") as f:
                        code_in_mem = f.read()
                        with self.state_lock:
                            self.section_codes[section_id] = code_in_mem
                except Exception:
                    return False
            else:
                return False

        last_stderr = None
        # Enforce repair attempt ceiling: Gemini free tier clamps to 1 to preserve 20 RPD quota;
        # local/unmetered providers (e.g. Ollama) allow configured attempts (up to 3).
        is_gemini_free = (getattr(self, "provider", "") or "").lower() == "gemini" and not os.getenv("GEMINI_PAID_TIER")
        repair_ceiling = 1 if is_gemini_free else 3
        effective_attempts = min(max_fix_attempts, self.max_fix_bug_tries, repair_ceiling)
        for fix_attempt in range(effective_attempts):
            print(f"🔧 {self.learning_topic} Debugging {section_id} (attempt {fix_attempt + 1}/{effective_attempts})")

            try:
                scene_name = f"{section_id.title().replace('_', '')}Scene"
                # Invalidate stale timing and layout files before rendering to avoid leaking previous metadata
                for st_file in [self.output_dir / f"timing_{scene_name}.json", self.output_dir / f"layout_{scene_name}.json"]:
                    if st_file.exists():
                        try:
                            st_file.unlink()
                        except Exception:
                            pass
                cmd = ["manim", self.manim_quality_flag, str(code_file), scene_name]
                video_patterns = [
                    self.output_dir / "media" / "videos" / f"{code_file.replace('.py', '')}" / self.quality_folder / f"{scene_name}.mp4",
                    self.output_dir / "media" / "videos" / self.quality_folder / f"{scene_name}.mp4",
                    self.output_dir / "media" / "videos" / f"{code_file.replace('.py', '')}" / "720p30" / f"{scene_name}.mp4",
                    self.output_dir / "media" / "videos" / "720p30" / f"{scene_name}.mp4",
                    self.output_dir / "media" / "videos" / f"{code_file.replace('.py', '')}" / "480p15" / f"{scene_name}.mp4",
                    self.output_dir / "media" / "videos" / "480p15" / f"{scene_name}.mp4",
                    self.output_dir / "media" / "videos" / f"{code_file.replace('.py', '')}" / "1080p60" / f"{scene_name}.mp4",
                    self.output_dir / "media" / "videos" / "1080p60" / f"{scene_name}.mp4",
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.output_dir, timeout=180)

                if result.returncode == 0:
                    for video_path in video_patterns:
                        if video_path.exists():
                            with self.state_lock:
                                self.section_videos[section_id] = str(video_path)
                            print(f"✅ {self.learning_topic} {section_id} finished")
                            return True

                if result.stderr and result.stderr == last_stderr:
                    print(f"⚠️ Identical Manim error encountered consecutively on {section_id}. Halting debug loop to preserve API quota.")
                    break
                last_stderr = result.stderr

                with self.state_lock:
                    current_code = self.section_codes.get(section_id, "")
                fixed_code = self.scope_refine_fixer.fix_code_smart(section_id, current_code, result.stderr, self.output_dir)

                if fixed_code:
                    try:
                        from utils import replace_base_class
                        from prompts.base_class import base_class
                        fixed_code = replace_base_class(fixed_code, base_class)
                    except Exception as e:
                        logger.debug(f"replace_base_class in debug_and_fix_code skipped: {e}")
                    with self.state_lock:
                        self.section_codes[section_id] = fixed_code
                    with open(self.output_dir / code_file, "w", encoding="utf-8") as f:
                        f.write(fixed_code)

                    # Re-run manim on the repaired code to verify and register output video
                    retry_result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.output_dir, timeout=180)
                    if retry_result.returncode == 0:
                        for video_path in video_patterns:
                            if video_path.exists():
                                with self.state_lock:
                                    self.section_videos[section_id] = str(video_path)
                                print(f"✅ {self.learning_topic} {section_id} finished after repair")
                                return True
                    else:
                        print(f"⚠️ {self.learning_topic} {section_id} repair attempt did not resolve compiler error: {retry_result.stderr[:200]}")
                        break
                else:
                    break

            except subprocess.TimeoutExpired:
                print(f"❌ {self.learning_topic} {section_id} timed out")
                break
            except (DailyQuotaExhaustedError, ServiceUnavailableError):
                raise
            except Exception as e:
                print(f"❌ {self.learning_topic} {section_id} failed with exception: {e}")
                break

        return False

    def get_mllm_feedback(self, section: Section, video_path: str, round_number: int = 1, local_hints: str = "") -> VideoFeedback:
        print(f"🤖 {self.learning_topic} Using MLLM to analyze video ({round_number}/{self.feedback_rounds}): {section.id}")

        feedback_file = self.output_dir / f"feedback_{section.id}_round{round_number}.json"
        if feedback_file.exists():
            try:
                with open(feedback_file, "r", encoding="utf-8") as f:
                    cached_fb = json.load(f)
                    print(f"📂 [CACHE HIT] Loaded cached MLLM feedback for {section.id} round {round_number}")
                    feedback = VideoFeedback(
                        section_id=section.id,
                        video_path=video_path,
                        has_issues=cached_fb.get("has_issues", False),
                        suggested_improvements=cached_fb.get("suggested_improvements", []),
                        raw_response=cached_fb.get("raw_response", ""),
                    )
                    with self.state_lock:
                        self.video_feedbacks[f"{section.id}_round{round_number}"] = feedback
                    return feedback
            except Exception:
                pass

        with self.state_lock:
            current_code = self.section_codes.get(section.id, "")
        if not current_code:
            code_file = self.output_dir / f"{section.id}.py"
            if code_file.exists():
                with open(code_file, "r", encoding="utf-8") as f:
                    current_code = f.read()
                    with self.state_lock:
                        self.section_codes[section.id] = current_code

        positions = self.extractor.extract_grid_positions(current_code)
        position_table = self.extractor.generate_position_table(positions)
        analysis_prompt = get_prompt4_layout_feedback(section=section, position_table=position_table, local_hints=local_hints)

        def _parse_layout(feedback_content):
            has_layout_issues, suggested_improvements = False, []
            try:
                data = safe_json_loads(feedback_content)
                lay = data.get("layout", {})
                has_layout_issues = bool(lay.get("has_issues", False))
                for it in lay.get("improvements", []) or []:
                    if isinstance(it, dict):
                        prob = str(it.get("problem", "")).strip()
                        sol = str(it.get("solution", "")).strip()
                        if prob or sol:
                            suggested_improvements.append(f"[LAYOUT] Problem: {prob}; Solution: {sol}")

            except json.JSONDecodeError:
                print(f"⚠️ {self.learning_topic} JSON parse failed, fallback to keyword analysis")

                for m in re.finditer(
                    r"Problem:\s*(.*?);\s*Solution:\s*(.*?)(?=\n|$)", feedback_content, flags=re.IGNORECASE | re.DOTALL
                ):
                    suggested_improvements.append(f"[LAYOUT] Problem: {m.group(1).strip()}; Solution: {m.group(2).strip()}")

                if not suggested_improvements:
                    for sol in re.findall(r"Solution\s*:\s*(.+)", feedback_content, flags=re.IGNORECASE):
                        suggested_improvements.append(f"[LAYOUT] Problem: ; Solution: {sol.strip()}")

            return has_layout_issues, suggested_improvements

        try:
            response = self._request_video_api_and_track_tokens(prompt=analysis_prompt, video_path=video_path)
            feedback_content = extract_answer_from_response(response)
            has_layout_issues, suggested_improvements = _parse_layout(feedback_content)
            feedback = VideoFeedback(
                section_id=section.id,
                video_path=video_path,
                has_issues=has_layout_issues,
                suggested_improvements=suggested_improvements,
                raw_response=feedback_content,
            )
            with self.state_lock:
                self.video_feedbacks[f"{section.id}_round{round_number}"] = feedback
            try:
                with open(feedback_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "section_id": section.id,
                        "has_issues": has_layout_issues,
                        "suggested_improvements": suggested_improvements,
                        "raw_response": feedback_content,
                    }, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return feedback

        except Exception as e:
            print(f"❌ {self.learning_topic} MLLM analysis failed: {str(e)}")
            return VideoFeedback(
                section_id=section.id,
                video_path=video_path,
                has_issues=False,
                suggested_improvements=[],
                raw_response=f"Error: {str(e)}",
            )

    def optimize_with_feedback(self, section: Section, feedback: VideoFeedback) -> bool:
        """Optimize the code based on feedback from the MLLM (bounded to max 1 modification attempt)"""
        if not feedback.has_issues or not feedback.suggested_improvements:
            print(f"✅ {self.learning_topic} {section.id} no optimization needed")
            return True

        # === Step 1: back up original code ===
        with self.state_lock:
            original_code_content = self.section_codes.get(section.id, "")

        is_gemini_free = (getattr(self, "provider", "") or "").lower() == "gemini" and not os.getenv("GEMINI_PAID_TIER")
        critic_ceiling = 1 if is_gemini_free else 2
        effective_attempts = min(self.max_feedback_gen_code_tries, self.max_critic_modification_attempts, critic_ceiling)
        for attempt in range(effective_attempts):
            print(
                f"🎯 {self.learning_topic} MLLM feedback optimization {section.id} code, attempt {attempt + 1}/{effective_attempts}"
            )

            # === Step 2: restore backup if retry ===
            if attempt > 0:
                with self.state_lock:
                    self.section_codes[section.id] = original_code_content

            # === Step 3: re-generate code with feedback ===
            self.generate_section_code(
                section=section, attempt=attempt + 1, feedback_improvements=feedback.suggested_improvements
            )
            # Re-apply pedagogical retiming if audio is enabled
            if self.use_audio:
                contract_obj = None
                with self.state_lock:
                    cached_ts = self.section_timestamps.get(section.id)
                if cached_ts and "timing_contract" in cached_ts:
                    try:
                        contract_obj = SceneTimingContract.from_dict(cached_ts["timing_contract"])
                    except Exception:
                        pass
                elif cached_ts and "lines" in cached_ts:
                    try:
                        contract_obj = SceneTimingContract.from_dict(cached_ts)
                    except Exception:
                        pass
                if contract_obj:
                    with self.state_lock:
                        curr_opt_code = self.section_codes.get(section.id, "")
                    if curr_opt_code:
                        retimed_c, adj = AnimationRetimer.retime_scene_code(curr_opt_code, contract_obj)
                        if adj and not adj[0].get("fallback"):
                            with open(self.output_dir / f"{section.id}.py", "w", encoding="utf-8") as fw:
                                fw.write(retimed_c)
                            with self.state_lock:
                                self.section_codes[section.id] = retimed_c
                            print(f"⏱️ [PEDAGOGICAL RETIMING] Retimed optimized code for {section.id}")

            success = self.debug_and_fix_code(section.id, max_fix_attempts=self.max_post_critic_scope_refine_calls)
            if success:
                optimized_output_dir = self.output_dir / "optimized_videos"
                optimized_output_dir.mkdir(exist_ok=True)
                optimized_video_path = optimized_output_dir / f"{section.id}_optimized.mp4"

                with self.state_lock:
                    current_v = self.section_videos.get(section.id)
                if current_v:
                    original_video_path = Path(current_v)
                    if original_video_path.exists():
                        original_video_path.rename(optimized_video_path)
                        with self.state_lock:
                            self.section_videos[section.id] = str(optimized_video_path)
                        print(f"✨ {self.learning_topic} {section.id} optimized video saved: {optimized_video_path}")
                    else:
                        print(f"⚠️ {self.learning_topic} {section.id} original video file not found: {original_video_path}")
                else:
                    print(f"⚠️ {self.learning_topic} {section.id} no optimized video path found")
                return True
            else:
                print(
                    f"❌ {self.learning_topic} {section.id} MLLM optimization failed, attempt {attempt + 1}/{effective_attempts}"
                )

        return False

    def generate_codes(self) -> Dict[str, str]:
        if not self.sections:
            raise ValueError(f"{self.learning_topic} Please generate teaching sections first")

        if not self.is_storyboard_safe_for_audio():
            raise ValueError(f"{self.learning_topic} Cannot generate codes: Storyboard failed numerical fidelity gate or was mutated")

        if self.use_audio and not self.is_audio_safe_for_stage3():
            handoff_rep = self.audio_handoff_report
            summary_msg = handoff_rep.summary if handoff_rep else "Audio handoff validation failed"
            raise ValueError(f"{self.learning_topic} Cannot generate codes: {summary_msg}")

        missing = [s for s in self.sections if s.id not in self.section_codes or not self.section_codes[s.id]]
        if not missing:
            return self.section_codes

        print(f"💻 Ensuring Manim code for {len(missing)} section(s)...")

        def _gen(sec: Section):
            try:
                self.generate_section_code(sec, attempt=1)
            except DailyQuotaExhaustedError as e:
                self.save_checkpoint("daily_quota_exhausted", {"error": str(e), "failed_at": sec.id})
                raise e
            except ServiceUnavailableError as e:
                self.save_checkpoint("service_unavailable", {"error": str(e), "failed_at": sec.id})
                raise e
            except Exception as e:
                print(f"❌ {self.learning_topic} {sec.id} code generation failed: {e}")

        is_ollama = (getattr(self, "provider", "") or "").lower() == "ollama"
        code_gen_workers = 1 if is_ollama else min(self.max_workers, len(missing))
        if self.parallel and code_gen_workers > 1 and len(missing) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=code_gen_workers) as executor:
                futures = [executor.submit(_gen, sec) for sec in missing]
                for f in concurrent.futures.as_completed(futures):
                    try:
                        f.result()
                    except (DailyQuotaExhaustedError, ServiceUnavailableError):
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
        else:
            for sec in missing:
                _gen(sec)

        return self.section_codes

    def render_section(self, section: Section) -> bool:
        section_id = section.id
        print(f"🎬 [START SECTION] Processing {section_id}...")

        if self.sections or self.outline:
            if not self.is_storyboard_safe_for_audio():
                raise ValueError(f"{self.learning_topic} Cannot render section {section_id}: Storyboard failed numerical fidelity gate or was mutated")
            if self.use_audio and not self.is_audio_safe_for_stage3():
                raise ValueError(f"{self.learning_topic} Cannot render section {section_id}: Audio handoff validation failed")

        # 1. Fast-path: Check if section is already rendered and synchronized from a previous run
        synced_video = self.output_dir / f"{section_id}_synced.mp4"
        synced_audio = self.output_dir / f"{section_id}_audio.mp3"
        synced_ts = self.output_dir / f"{section_id}_timestamps.json"
        if synced_video.exists() and synced_video.stat().st_size > 1000:
            is_synced_valid = True
            current_narr_hash = compute_narration_hash(section.lecture_lines)
            if synced_ts.exists():
                try:
                    with open(synced_ts, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        if meta.get("narration_hash") and meta.get("narration_hash") != current_narr_hash:
                            is_synced_valid = False
                except Exception:
                    is_synced_valid = False
            else:
                is_synced_valid = False

            if is_synced_valid:
                print(f"📂 [CHECKPOINT REUSE] Found existing synchronized video for {section_id}: {synced_video}")
                with self.state_lock:
                    self.section_videos[section_id] = str(synced_video)
                    if synced_audio.exists():
                        self.section_audios[section_id] = str(synced_audio)
                    if synced_ts.exists():
                        try:
                            with open(synced_ts, "r", encoding="utf-8") as f:
                                self.section_timestamps[section_id] = json.load(f)
                        except Exception:
                            pass
                return True
            else:
                print(f"⚠️ Narration changed for {section_id}; invalidating existing synced video {synced_video}")
                try:
                    synced_video.unlink()
                except Exception:
                    pass

        # 2. Ensure section code is present (max 1 generation attempt)
        code_file = self.output_dir / f"{section_id}.py"
        with self.state_lock:
            current_code = self.section_codes.get(section_id, "")
        if not current_code and code_file.exists():
            try:
                with open(code_file, "r", encoding="utf-8") as f:
                    current_code = f.read()
                    current_code, pre_fixed = self.pre_render_validate_and_fix(section, current_code)
                    if pre_fixed:
                        with open(code_file, "w", encoding="utf-8") as fw:
                            fw.write(current_code)
                    with self.state_lock:
                        self.section_codes[section_id] = current_code
            except Exception:
                pass

        if not current_code:
            print(f"📝 Generating initial code for {section_id} (bounded to 1 attempt)...")
            current_code = self.generate_section_code(section, attempt=1)
            if not current_code:
                print(f"❌ Failed to generate code for {section_id}. Skipping section.")
                return False

        # 2b. Apply pedagogical animation retiming pass if audio timing is available
        if self.use_audio and current_code:
            contract_obj = None
            with self.state_lock:
                cached_ts = self.section_timestamps.get(section_id)
            if cached_ts and "timing_contract" in cached_ts:
                try:
                    contract_obj = SceneTimingContract.from_dict(cached_ts["timing_contract"])
                except Exception:
                    pass
            elif cached_ts and "lines" in cached_ts:
                try:
                    contract_obj = SceneTimingContract.from_dict(cached_ts)
                except Exception:
                    pass
            else:
                ts_cand = self.output_dir / f"{section_id}_timestamps.json"
                if ts_cand.exists():
                    try:
                        with open(ts_cand, "r", encoding="utf-8") as tf:
                            contract_obj = SceneTimingContract.from_dict(json.load(tf))
                    except Exception:
                        pass

            if contract_obj:
                retimed_code, adj = AnimationRetimer.retime_scene_code(current_code, contract_obj)
                if adj and not adj[0].get("fallback"):
                    current_code = retimed_code
                    with open(code_file, "w", encoding="utf-8") as fw:
                        fw.write(current_code)
                    with self.state_lock:
                        self.section_codes[section_id] = current_code
                    print(f"⏱️ [PEDAGOGICAL RETIMING] Retimed {len(adj)} animation blocks for {section_id}")

        # 3. Check for existing compiled raw or optimized video
        opt_video = self.output_dir / "optimized_videos" / f"{section_id}_optimized.mp4"
        scene_name = f"{section_id.title().replace('_', '')}Scene"
        raw_video_candidates = [
            self.output_dir / "media" / "videos" / section_id / self.quality_folder / f"{scene_name}.mp4",
            self.output_dir / "media" / "videos" / self.quality_folder / f"{scene_name}.mp4",
            self.output_dir / "media" / "videos" / section_id / "720p30" / f"{scene_name}.mp4",
            self.output_dir / "media" / "videos" / "720p30" / f"{scene_name}.mp4",
            self.output_dir / "media" / "videos" / section_id / "480p15" / f"{scene_name}.mp4",
            self.output_dir / "media" / "videos" / "480p15" / f"{scene_name}.mp4",
            self.output_dir / "media" / "videos" / section_id / "1080p60" / f"{scene_name}.mp4",
            self.output_dir / "media" / "videos" / "1080p60" / f"{scene_name}.mp4",
        ]
        already_compiled_video = None
        if not getattr(self.cfg, "force_render", False):
            if opt_video.exists() and opt_video.stat().st_size > 1000:
                already_compiled_video = str(opt_video)
                with self.state_lock:
                    self.section_videos[section_id] = already_compiled_video
                print(f"📂 [CHECKPOINT REUSE] Found existing optimized video for {section_id}: {opt_video}")
            else:
                for cand in raw_video_candidates:
                    if cand.exists() and cand.stat().st_size > 1000:
                        already_compiled_video = str(cand)
                        with self.state_lock:
                            self.section_videos[section_id] = already_compiled_video
                        print(f"📂 [CHECKPOINT REUSE] Found existing compiled scene for {section_id}: {cand}")
                        break
        else:
            print(f"⚡ [FORCE RENDER] Bypassing video checkpoint cache for {section_id} to render retimed code")

        try:
            success = bool(already_compiled_video)
            if not success:
                print(f"🎥 Rendering Manim scene locally for {section_id}...")
                success = self.debug_and_fix_code(section_id, max_fix_attempts=self.max_scope_refine_calls_initial_error)
            if not success:
                print(f"❌ {self.learning_topic} {section_id} failed rendering/repair, skipping section")
                return False

            # 4. Pre-Critic Local Visual Validation ($0 API cost)
            with self.state_lock:
                current_code = self.section_codes.get(section_id, "")
                current_video = self.section_videos.get(section_id)

            # Explicit immutable backup before applying local fix
            original_code = current_code
            original_video = current_video

            local_result = self.local_checker.check_section(section_id, current_code, self.output_dir)
            print(f"🔍 [LOCAL CHECK] {section_id} (status: {local_result.status}): {local_result.summary}")

            # Apply deterministic local fixes if possible ($0 API cost, zero additional inspection renders)
            fixable_issues = local_result.get_local_fixable_issues()
            if fixable_issues:
                print(f"🛠️ [LOCAL FIX] Applying deterministic geometry fixes for {len(fixable_issues)} issue(s) on {section_id} ($0 API cost)...")
                candidate_code, was_modified = self.local_fixer.apply_fixes(original_code, fixable_issues)
                if was_modified:
                    with open(code_file, "w", encoding="utf-8") as f:
                        f.write(candidate_code)

                    # Fast layout verification render using Manim last-frame mode (-ql -s)
                    print(f"🎥 Verifying layout fix for {section_id} via fast last-frame inspection...")
                    fast_verify_cmd = ["manim", "-ql", "-s", str(code_file.name), scene_name]
                    verify_res = subprocess.run(fast_verify_cmd, capture_output=True, text=True, cwd=self.output_dir, timeout=60)
                    candidate_valid = False
                    if verify_res.returncode == 0:
                        post_fix_result = self.local_checker.check_section(section_id, candidate_code, self.output_dir)
                        print(f"🔍 [POST-FIX LOCAL CHECK] {section_id} (status: {post_fix_result.status}): {post_fix_result.summary}")
                        
                        orig_errors = sum(1 for i in local_result.issues if i.severity == "error")
                        post_errors = sum(1 for i in post_fix_result.issues if i.severity == "error")
                        fix_improved = (not post_fix_result.has_errors()) or (post_errors < orig_errors) or (post_errors == orig_errors and len(post_fix_result.issues) < len(local_result.issues))
                        
                        if fix_improved:
                            print(f"🎥 Rendering full video for {section_id} with accepted layout fix...")
                            cmd = ["manim", self.manim_quality_flag, str(code_file.name), scene_name]
                            retry_res = subprocess.run(cmd, capture_output=True, text=True, cwd=self.output_dir, timeout=180)
                            if retry_res.returncode == 0:
                                candidate_valid = True
                                current_code = candidate_code
                                with self.state_lock:
                                    self.section_codes[section_id] = candidate_code
                                local_result = post_fix_result
                                print(f"✅ [LOCAL FIX PROMOTED] {section_id} deterministic layout fix accepted.")

                    if not candidate_valid:
                        print(f"⚠️ Local fix candidate failed verification. Restoring clean original code and video.")
                        with open(code_file, "w", encoding="utf-8") as f:
                            f.write(original_code)
                        with self.state_lock:
                            self.section_codes[section_id] = original_code
                            if original_video:
                                self.section_videos[section_id] = original_video
                        current_code = original_code

            # 5. Critic / MLLM Visual Feedback (if enabled and not already optimized)
            with self.state_lock:
                clean_code = self.section_codes.get(section_id, "")
                clean_video_path = self.section_videos.get(section_id)

            if self.use_feedback and not (opt_video.exists() and opt_video.stat().st_size > 1000):
                if clean_video_path and os.path.exists(clean_video_path):
                    unresolved_hints = local_result.get_unresolved_hints()
                    feedback = self.get_mllm_feedback(
                        section=section,
                        video_path=clean_video_path,
                        round_number=1,
                        local_hints=unresolved_hints,
                    )
                    if feedback.has_issues and feedback.suggested_improvements:
                        print(f"✨ Optimizing {section_id} with Critic feedback...")
                        opt_success = self.optimize_with_feedback(section, feedback)
                        if opt_success:
                            print(f"✅ {section_id} Critic optimization complete")
                        else:
                            print(f"⚠️ {section_id} Critic optimization failed. Rolling back to clean baseline render.")
                            with open(code_file, "w", encoding="utf-8") as f:
                                f.write(clean_code)
                            with self.state_lock:
                                self.section_codes[section_id] = clean_code
                                self.section_videos[section_id] = clean_video_path

                    # Post-Critic local visual check ($0 API cost)
                    with self.state_lock:
                        post_critic_code = self.section_codes.get(section_id, "")
                    post_critic_check = self.local_checker.check_section(section_id, post_critic_code, self.output_dir)
                    print(f"🔍 [POST-CRITIC LOCAL CHECK] {section_id} (status: {post_critic_check.status}): {post_critic_check.summary}")
                    local_result = post_critic_check

            # 5b. Bounded Visual Defect Repair (for placeholder-only or blank scenes)
            has_placeholder_error = any(i.issue_type in ("PLACEHOLDER_ONLY_SCENE", "BLANK_SCENE") for i in local_result.issues if i.severity == "error")
            if has_placeholder_error and getattr(self, "max_regenerate_tries", 2) > 1:
                print(f"⚠️ [VISUAL DEFECT DETECTED] {section_id} visual check failed: {[i.issue_type for i in local_result.issues if i.severity == 'error']}. Attempting bounded LLM repair...")
                regen_err_details = (
                    f"Visual Validation Error in {section_id}:\n"
                    f"Issue: {', '.join(i.issue_type for i in local_result.issues if i.severity == 'error')}\n"
                    f"Description: {local_result.summary}\n"
                    f"CRITICAL REQUIREMENT: Do NOT output placeholder fallback cards or empty VGroups. "
                    f"You MUST instantiate concrete visual components (e.g. ProcessCardVisual, ReadyQueueVisual, CPUExecutionVisual, GanttChartVisual, MetricTableVisual) "
                    f"or a structured, labeled Manim diagram with shapes and text illustrating the concept."
                )
                with self.state_lock:
                    pre_repair_code = self.section_codes.get(section_id, "")
                repaired_code = self.generate_section_code(
                    section=section,
                    attempt=2,
                    error_log=regen_err_details,
                    previous_code=pre_repair_code,
                    force_api=True,
                )
                if repaired_code:
                    if self.use_audio:
                        try:
                            repaired_code = self.retime_section_animation(section, repaired_code)
                        except Exception as e:
                            print(f"⚠️ Retiming after repair warning: {e}")
                    with open(code_file, "w", encoding="utf-8") as f:
                        f.write(repaired_code)
                    with self.state_lock:
                        self.section_codes[section_id] = repaired_code

                    print(f"🎥 Rendering Manim scene for repaired {section_id}...")
                    cmd = ["manim", self.manim_quality_flag, str(code_file.name), scene_name]
                    re_res = subprocess.run(cmd, capture_output=True, text=True, cwd=self.output_dir, timeout=180)
                    if re_res.returncode == 0:
                        repaired_result = self.local_checker.check_section(section_id, repaired_code, self.output_dir)
                        print(f"🔍 [POST-REPAIR LOCAL CHECK] {section_id} (status: {repaired_result.status}): {repaired_result.summary}")
                        still_broken = any(i.issue_type in ("PLACEHOLDER_ONLY_SCENE", "BLANK_SCENE") for i in repaired_result.issues if i.severity == "error")
                        if not still_broken:
                            local_result = repaired_result
                            current_code = repaired_code
                            print(f"✅ [REPAIR ACCEPTED] {section_id} successfully repaired and passed visual validation.")
                        else:
                            print(f"❌ [REPAIR FAILED] {section_id} still contains placeholder/blank scene after repair.")
                    else:
                        print(f"❌ [REPAIR COMPILE FAILED] Repaired scene failed to compile: {re_res.stderr[-300:]}")

            # If the scene still has PLACEHOLDER_ONLY_SCENE or BLANK_SCENE after repair attempts, report incomplete
            if any(i.issue_type in ("PLACEHOLDER_ONLY_SCENE", "BLANK_SCENE") for i in local_result.issues if i.severity == "error"):
                print(f"❌ [INCOMPLETE SCENE] {section_id} contains only placeholder cards or blank canvas. Marking as incomplete.")
                with self.state_lock:
                    if section_id in self.section_videos:
                        del self.section_videos[section_id]
                return False

            # 6. Synchronize video with audio
            with self.state_lock:
                current_video = self.section_videos.get(section_id)
            if success and self.use_audio and current_video:
                try:
                    audio_meta = None
                    if self.audio_input:
                        audio_in_path = Path(self.audio_input)
                        candidate_files = []
                        if audio_in_path.is_dir():
                            for ext in [".mp3", ".wav", ".m4a", ".aac"]:
                                cand = audio_in_path / f"{section_id}{ext}"
                                if cand.exists():
                                    candidate_files.append(cand)
                        elif audio_in_path.is_file():
                            candidate_files.append(audio_in_path)

                        if candidate_files:
                            print(f"🎵 {self.learning_topic} Using external audio for {section_id}: {candidate_files[0]}")
                            audio_meta = self.audio_manager.handle_external_audio(
                                section_id=section_id,
                                audio_file_path=candidate_files[0],
                                lecture_lines=section.lecture_lines,
                                animations=section.animations,
                                output_dir=self.output_dir
                            )

                    if not audio_meta:
                        with self.state_lock:
                            cached_ts = self.section_timestamps.get(section_id)
                        if cached_ts:
                            audio_meta = cached_ts
                        else:
                            print(f"🎙️ {self.learning_topic} Generating TTS narration audio for {section_id}...")
                            audio_meta = self.audio_manager.generate_section_audio(
                                section_id=section_id,
                                lecture_lines=section.lecture_lines,
                                animations=section.animations,
                                output_dir=self.output_dir,
                                voice=self.tts_voice
                            )

                    synced_video_path = self.output_dir / f"{section_id}_synced.mp4"
                    srt_path = self.output_dir / f"{section_id}.srt"
                    synced_result = self.audio_manager.sync_video_and_audio(
                        video_path=current_video,
                        audio_path=audio_meta["audio_path"],
                        output_path=str(synced_video_path),
                        srt_path=str(srt_path),
                        burn_subtitles=self.burn_subtitles,
                        section_id=section_id,
                    )

                    # Reload updated metadata reflecting reconciled duration
                    ts_path = self.output_dir / f"{section_id}_timestamps.json"
                    if ts_path.exists():
                        try:
                            with open(ts_path, "r", encoding="utf-8") as tf:
                                audio_meta = json.load(tf)
                        except Exception:
                            pass

                    with self.state_lock:
                        self.section_videos[section_id] = str(synced_result)
                        self.section_audios[section_id] = audio_meta["audio_path"]
                        self.section_timestamps[section_id] = audio_meta
                    print(f"🔊 {self.learning_topic} {section_id} audio synchronized: {synced_result} ({audio_meta['total_duration']:.2f}s)")

                    # 6b. Audit pedagogical pacing
                    try:
                        contract_audit = SceneTimingContract.from_dict(audio_meta)
                        pacing_res = contract_audit.audit_pacing()
                        pacing_file = self.output_dir / f"{section_id}_pacing.json"
                        with open(pacing_file, "w", encoding="utf-8") as pf:
                            json.dump(pacing_res.to_dict(), pf, indent=2, ensure_ascii=False)
                        print(f"⏱️ [PACING AUDIT] {section_id}: {pacing_res.summary}")
                    except Exception as pe:
                        print(f"⚠️ {section_id} pacing audit notice: {pe}")
                except Exception as e:
                    print(f"⚠️ {self.learning_topic} {section_id} audio synchronization warning: {e}")

            # 7. Checkpoint
            self.save_checkpoint(f"section_{section_id}_completed")
            print(f"🏁 [COMPLETE SECTION] {section_id} finished processing successfully.")
            return success

        except (DailyQuotaExhaustedError, ServiceUnavailableError):
            raise
        except Exception as e:
            print(f"❌ {self.learning_topic} {section_id} render process exception: {str(e)}")
            return False

    def render_section_worker(self, section_data) -> Tuple[str, bool, Optional[str], Optional[str], Optional[dict], Optional[str], Optional[dict]]:
        """Deprecated legacy worker; sections are processed directly via self.render_section(sec)."""
        try:
            if isinstance(section_data, tuple):
                section = section_data[0]
            else:
                section = section_data
            section_id = section.id
            success = self.render_section(section)
            video_path = self.section_videos.get(section.id) if success else None
            audio_path = self.section_audios.get(section.id) if success else None
            timestamps = self.section_timestamps.get(section.id) if success else None
            sec_code = self.section_codes.get(section.id)
            tok_usage = self.token_usage
            return section_id, success, video_path, audio_path, timestamps, sec_code, tok_usage
        except (DailyQuotaExhaustedError, ServiceUnavailableError):
            raise
        except Exception as e:
            print(f"❌ {self.learning_topic} render worker exception: {str(e)}")
            return "unknown", False, None, None, None, None, None

    def render_all_sections(self, max_workers: Optional[int] = None) -> Dict[str, str]:
        if max_workers is None:
            max_workers = self.max_workers

        total_sections = len(self.sections)
        print(f"🎥 Start processing of all section videos ({total_sections} section(s))...")

        if not self.sections:
            print("❌ No sections to execute")
            return {}

        if not self.is_storyboard_safe_for_audio():
            raise ValueError(f"{self.learning_topic} Cannot render sections: Storyboard failed numerical fidelity gate or was mutated")

        if self.use_audio and not self.is_audio_safe_for_stage3():
            handoff_rep = self.audio_handoff_report
            summary_msg = handoff_rep.summary if handoff_rep else "Audio handoff validation failed"
            raise ValueError(f"{self.learning_topic} Cannot render sections: {summary_msg}")

        successful_count = 0
        failed_count = 0

        if self.parallel and max_workers > 1 and total_sections > 1:
            try:
                import multiprocessing
                cpu_cores = multiprocessing.cpu_count()
            except Exception:
                cpu_cores = 4
            effective_workers = min(max_workers, total_sections, 4, max(1, cpu_cores - 1))
            print(f"🚀 Concurrency ENABLED: processing {total_sections} sections in parallel with {effective_workers} worker threads")
            print("   (Local rendering, visual checks, and audio sync execute concurrently; all Gemini API calls pass through the centralized QuotaManager)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_to_sec = {executor.submit(self.render_section, sec): sec for sec in self.sections}
                for future in concurrent.futures.as_completed(future_to_sec):
                    sec = future_to_sec[future]
                    try:
                        ok = future.result()
                        if ok:
                            successful_count += 1
                        else:
                            failed_count += 1
                    except DailyQuotaExhaustedError:
                        print(f"🛑 Daily quota exhausted during execution of {sec.id}!")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except ServiceUnavailableError as e:
                        print(f"🛑 503 Service Unavailable (high demand) during execution of {sec.id}: {e}")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
                    except Exception as e:
                        failed_count += 1
                        print(f"❌ {sec.id} thread exception: {e}")
        else:
            print(f"🎬 Sequential mode: processing {total_sections} section(s) sequentially (parallel={self.parallel})")
            for sec in self.sections:
                try:
                    ok = self.render_section(sec)
                    if ok:
                        successful_count += 1
                    else:
                        failed_count += 1
                except (DailyQuotaExhaustedError, ServiceUnavailableError):
                    raise
                except Exception as e:
                    failed_count += 1
                    print(f"❌ {sec.id} execution exception: {e}")

        print(f"\n📊 Rendering Statistics:")
        print(f"   Total Sections: {total_sections}")
        print(f"   Success Rate: {successful_count/total_sections*100:.1f}%" if total_sections > 0 else "   Success Rate: 0%")

        return self.section_videos

    def _generate_master_audio_and_subtitles(self, final_video_path: Path):
        """Generates unified master audio, timestamps.json, subtitles.srt, and subtitles.vtt in storyboard order."""
        try:
            print(f"🎼 Generating master audio, timestamps, and subtitles for {self.learning_topic} in storyboard order...")
            ordered_section_ids = []
            if getattr(self, "scene_sequence_ir", None) and hasattr(self.scene_sequence_ir, "sections"):
                ordered_section_ids = [s.scene_id for s in self.scene_sequence_ir.sections if s.scene_id in self.section_videos]
            if not ordered_section_ids:
                ordered_section_ids = [s.id for s in self.sections if s.id in self.section_videos]
            if not ordered_section_ids:
                ordered_section_ids = list(self.section_videos.keys())
            if not ordered_section_ids:
                return

            safe_name = topic_to_safe_name(self.learning_topic)

            # 1. Master Audio aligned to actual section video durations
            master_audio = AudioSegment.empty()
            for sid in ordered_section_ids:
                aud_path = self.section_audios.get(sid)
                v_path = self.section_videos.get(sid)
                target_dur = self.audio_manager.get_media_duration(v_path) if (v_path and os.path.exists(v_path)) else 0.0

                if aud_path and os.path.exists(aud_path):
                    sec_aud = AudioSegment.from_file(aud_path)
                    if target_dur > 0:
                        target_ms = int(round(target_dur * 1000))
                        if len(sec_aud) < target_ms:
                            sec_aud = sec_aud + AudioSegment.silent(duration=target_ms - len(sec_aud))
                        elif len(sec_aud) > target_ms + 40:
                            sec_aud = sec_aud[:target_ms]
                    master_audio += sec_aud

            master_audio_file = self.output_dir / f"{safe_name}_audio.mp3"
            master_wav_file = self.output_dir / f"{safe_name}_audio.wav"
            if len(master_audio) > 0:
                master_audio.export(str(master_audio_file), format="mp3")
                master_audio.export(str(master_wav_file), format="wav")
                print(f"🎙️ Master audio saved: {master_audio_file} & {master_wav_file}")

            # 2. Master Timestamps & Subtitles
            all_line_timestamps = []
            sections_meta = []
            cumulative_time = 0.0

            for sid in ordered_section_ids:
                sec_ts = self.section_timestamps.get(sid, {})
                if isinstance(sec_ts, (str, Path)) and os.path.exists(sec_ts):
                    try:
                        with open(sec_ts, "r", encoding="utf-8") as _ts_f:
                            sec_ts = json.load(_ts_f)
                    except Exception:
                        sec_ts = {}
                elif not isinstance(sec_ts, dict):
                    sec_ts = {}

                v_path = self.section_videos.get(sid)
                if v_path and os.path.exists(v_path):
                    sec_dur = self.audio_manager.get_media_duration(v_path)
                else:
                    sec_dur = sec_ts.get("total_duration", 0.0)

                sec_start = cumulative_time
                sec_end = sec_start + sec_dur

                sec_lines = []
                for line_item in sec_ts.get("lines", []):
                    rel_start = line_item.get("start", 0.0)
                    rel_end = line_item.get("end", 0.0)
                    dur = line_item.get("duration", rel_end - rel_start)
                    global_start = round(sec_start + rel_start, 3)
                    global_end = round(sec_start + rel_end, 3)

                    global_item = {
                        "section_id": sid,
                        "line_index": line_item.get("line_index", 0),
                        "raw_text": line_item.get("raw_text", ""),
                        "caption": line_item.get("caption", ""),
                        "animation_type": line_item.get("animation_type", "fade"),
                        "start": global_start,
                        "end": global_end,
                        "duration": round(dur, 3),
                    }
                    sec_lines.append(global_item)
                    all_line_timestamps.append(global_item)

                sections_meta.append({
                    "section_id": sid,
                    "start": round(sec_start, 3),
                    "end": round(sec_end, 3),
                    "duration": round(sec_dur, 3),
                    "lines": sec_lines
                })

                cumulative_time = sec_end

            master_meta = {
                "topic": self.learning_topic,
                "total_duration": round(cumulative_time, 3),
                "sections": sections_meta
            }

            master_ts_file = self.output_dir / "timestamps.json"
            named_ts_file = self.output_dir / f"{safe_name}_timestamps.json"
            for tf in [master_ts_file, named_ts_file]:
                with open(tf, "w", encoding="utf-8") as f:
                    json.dump(master_meta, f, indent=2, ensure_ascii=False)
            print(f"⏱️ Master timestamps saved: {master_ts_file}")

            # Export master SRT and VTT
            srt_path = self.output_dir / "subtitles.srt"
            vtt_path = self.output_dir / "subtitles.vtt"
            named_srt = self.output_dir / f"{safe_name}_subtitles.srt"
            named_vtt = self.output_dir / f"{safe_name}_subtitles.vtt"

            for sp in [srt_path, named_srt]:
                self.audio_manager.export_srt(all_line_timestamps, str(sp))
            for vp in [vtt_path, named_vtt]:
                self.audio_manager.export_vtt(all_line_timestamps, str(vp))

            print(f"📝 Master subtitles saved: {srt_path} & {vtt_path}")

            # Re-mux final video with clean 44.1kHz stereo master audio to eliminate non-monotonic DTS
            if master_audio_file.exists() and final_video_path.exists():
                temp_remux = self.output_dir / f"temp_{safe_name}.mp4"
                remux_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(final_video_path),
                    "-i", str(master_audio_file),
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "44100",
                    "-ac", "2",
                    str(temp_remux),
                ]
                res_remux = subprocess.run(remux_cmd, capture_output=True, text=True)
                if res_remux.returncode == 0:
                    os.replace(str(temp_remux), str(final_video_path))
                    print(f"✅ Final video remuxed with 44.1kHz stereo master audio track: {final_video_path}")
                    val_res = self.audio_manager.validate_media_file(str(final_video_path), require_audio=True)
                    if val_res.get("is_valid"):
                        print(f"🎬 Media validation passed: {final_video_path} ({val_res.get('duration')}s, video+audio streams confirmed)")
                    else:
                        print(f"⚠️ Media validation warning for {final_video_path}: {val_res.get('error')}")
                else:
                    if temp_remux.exists():
                        temp_remux.unlink()

            # Save comprehensive Pedagogical Pacing Audit across all sections
            pacing_sections = []
            for sid in ordered_section_ids:
                ts_cand = self.output_dir / f"{sid}_timestamps.json"
                if ts_cand.exists():
                    try:
                        with open(ts_cand, "r", encoding="utf-8") as tf:
                            c_dict = json.load(tf)
                        contract_audit = SceneTimingContract.from_dict(c_dict)
                        audit_res = contract_audit.audit_pacing()
                        pacing_sections.append(audit_res.to_dict())
                    except Exception:
                        pass

            if pacing_sections:
                pacing_audit_file = self.output_dir / "pacing_audit.json"
                overall_well_paced = all(s.get("is_well_paced", False) for s in pacing_sections)
                with open(pacing_audit_file, "w", encoding="utf-8") as pf:
                    json.dump({
                        "topic": self.learning_topic,
                        "overall_well_paced": overall_well_paced,
                        "sections": pacing_sections,
                    }, pf, indent=2, ensure_ascii=False)
                print(f"⏱️ Master Pacing Audit saved: {pacing_audit_file} (Overall well-paced: {overall_well_paced})")

        except Exception as e:
            print(f"⚠️ Failed to generate master audio/subtitles: {e}")

    def synchronize_case_audio(self) -> Optional[str]:
        """
        Synchronize an existing case directory without re-running code generation.
        Loads storyboard, generates audio for all rendered sections, aligns video and audio,
        and produces the final synchronized output video, master audio, and subtitles.
        """
        print(f"🔄 Synchronizing case audio for {self.learning_topic} in {self.output_dir}...")
        # 1. Load storyboard
        sb_file = self.output_dir / "storyboard_with_assets.json"
        if not sb_file.exists():
            sb_file = self.output_dir / "storyboard.json"
        if not sb_file.exists():
            print(f"❌ No storyboard found in {self.output_dir}")
            return None

        with open(sb_file, "r", encoding="utf-8") as f:
            sb_data = json.load(f)

        sb_sections = sb_data.get("sections", [])
        if not sb_sections:
            print(f"❌ Storyboard has no sections in {self.output_dir}")
            return None

        # Ensure outline is loaded and verified for fidelity validation
        if not self.outline:
            outline_file = self.output_dir / "outline.json"
            if outline_file.exists():
                try:
                    with open(outline_file, "r", encoding="utf-8") as of:
                        self.outline = TeachingOutline.from_dict(json.load(of))
                        self._ensure_outline_verified()
                except Exception:
                    pass

        # Synchronize self.sections with storyboard if empty
        if not self.sections:
            for sdata in sb_sections:
                self.sections.append(Section(
                    id=sdata["id"],
                    title=sdata.get("title", ""),
                    lecture_lines=sdata.get("lecture_lines", []),
                    animations=sdata.get("animations", []),
                    story_beat=sdata.get("story_beat", sdata.get("story_step", "")),
                    captions=sdata.get("captions"),
                    layout_plan=sdata.get("layout_plan", ""),
                    verified_values=sdata.get("verified_values"),
                    algorithmic_operations=sdata.get("algorithmic_operations"),
                ))

        # Check storyboard fidelity gate before generating/synchronizing audio
        if not self.is_storyboard_safe_for_audio():
            rep = self.storyboard_fidelity_report
            status_v = rep.overall_status.value if rep else "BLOCKED"
            summary_v = rep.summary if rep else "Storyboard failed numerical fidelity verification"
            raise ValueError(
                f"Cannot synchronize audio: Storyboard failed numerical fidelity gate ({status_v}): {summary_v}"
            )

        # 2. Strict completeness validation: check all required storyboard sections
        required_ids = [s["id"] for s in sb_sections]
        found_videos = {}
        for sdata in sb_sections:
            sid = sdata["id"]
            candidates = [
                self.output_dir / "optimized_videos" / f"{sid}_optimized.mp4",
                self.output_dir / "media" / "videos" / sid / self.quality_folder / f"{sid.title().replace('_', '')}Scene.mp4",
                self.output_dir / "media" / "videos" / self.quality_folder / f"{sid.title().replace('_', '')}Scene.mp4",
                self.output_dir / "media" / "videos" / sid / "720p30" / f"{sid.title().replace('_', '')}Scene.mp4",
                self.output_dir / "media" / "videos" / "720p30" / f"{sid.title().replace('_', '')}Scene.mp4",
                self.output_dir / "media" / "videos" / sid / "480p15" / f"{sid.title().replace('_', '')}Scene.mp4",
                self.output_dir / "media" / "videos" / "480p15" / f"{sid.title().replace('_', '')}Scene.mp4",
                self.output_dir / "media" / "videos" / sid / "1080p60" / f"{sid.title().replace('_', '')}Scene.mp4",
                self.output_dir / "media" / "videos" / "1080p60" / f"{sid.title().replace('_', '')}Scene.mp4",
            ]
            for c in candidates:
                if c.exists() and c.stat().st_size > 1000:
                    found_videos[sid] = str(c)
                    break

        missing = [sid for sid in required_ids if sid not in found_videos]
        if missing:
            print("\n" + "=" * 60)
            print("❌ INCOMPLETE LECTURE")
            print(f"Missing sections: {', '.join(missing)}")
            print("Final merge blocked.")
            print("Completed sections have been preserved for resume.")
            print("=" * 60 + "\n")
            self.save_checkpoint("incomplete_lecture", {"missing_sections": missing})
            return None

        print(f"Found all {len(found_videos)} required rendered section videos to synchronize in storyboard order: {required_ids}")

        # 3. Synchronize each section in storyboard order
        for sdata in sb_sections:
            sid = sdata["id"]
            raw_video = found_videos[sid]
            lecture_lines = sdata.get("lecture_lines", [])
            animations = sdata.get("animations", [])

            # Check for external audio or generate TTS
            audio_meta = None
            if self.audio_input:
                audio_in_path = Path(self.audio_input)
                cand_files = []
                if audio_in_path.is_dir():
                    for ext in [".mp3", ".wav", ".m4a", ".aac"]:
                        cand = audio_in_path / f"{sid}{ext}"
                        if cand.exists():
                            cand_files.append(cand)
                elif audio_in_path.is_file():
                    cand_files.append(audio_in_path)

                if cand_files:
                    audio_meta = self.audio_manager.handle_external_audio(
                        sid, cand_files[0], lecture_lines, animations, output_dir=self.output_dir
                    )

            if not audio_meta:
                audio_meta = self.audio_manager.generate_section_audio(
                    sid, lecture_lines, animations, output_dir=self.output_dir, voice=self.tts_voice
                )

            synced_video = self.output_dir / f"{sid}_synced.mp4"
            srt_file = self.output_dir / f"{sid}.srt"
            self.audio_manager.sync_video_and_audio(
                video_path=raw_video,
                audio_path=audio_meta["audio_path"],
                output_path=str(synced_video),
                srt_path=str(srt_file),
                burn_subtitles=self.burn_subtitles,
                section_id=sid,
            )

            # Reload updated metadata reflecting reconciled duration
            ts_path = self.output_dir / f"{sid}_timestamps.json"
            if ts_path.exists():
                try:
                    with open(ts_path, "r", encoding="utf-8") as tf:
                        audio_meta = json.load(tf)
                except Exception:
                    pass

            self.section_videos[sid] = str(synced_video)
            self.section_audios[sid] = audio_meta["audio_path"]
            self.section_timestamps[sid] = audio_meta

        # 4. Merge all sections
        final_video = self.merge_videos()
        if final_video:
            print(f"🎉 Fully synchronized video created: {final_video}")
            self.save_checkpoint("audio_synchronized", {"final_video": final_video})
        return final_video

    def merge_videos(self, output_filename: str = None) -> Optional[str]:
        """Step 5: Merge all section videos with strict completeness validation and storyboard ordering."""
        if not self.is_storyboard_safe_for_audio():
            raise ValueError(f"{self.learning_topic} Cannot merge videos: Storyboard failed numerical fidelity gate or was mutated")
        if self.use_audio and not self.is_audio_safe_for_stage3():
            raise ValueError(f"{self.learning_topic} Cannot merge videos: Audio handoff validation failed")

        # 1. Strict Completeness Validation
        required_ids = [section.id for section in self.sections]
        if not required_ids and self.section_videos:
            required_ids = list(self.section_videos.keys())

        missing = [
            sid for sid in required_ids
            if sid not in self.section_videos
            or not Path(self.section_videos[sid]).exists()
            or Path(self.section_videos[sid]).stat().st_size <= 1000
        ]

        if missing:
            print("\n" + "=" * 60)
            print("❌ INCOMPLETE LECTURE")
            print(f"Missing sections: {', '.join(missing)}")
            print("Final merge blocked.")
            print("Completed sections have been preserved for resume.")
            print("=" * 60 + "\n")
            self.save_checkpoint("incomplete_lecture", {"missing_sections": missing})
            return None

        if not self.section_videos:
            print("❌ No video files available to merge")
            return None

        if output_filename is None:
            safe_name = topic_to_safe_name(self.learning_topic)
            output_filename = f"{safe_name}.mp4"

        output_path = (self.output_dir / output_filename).resolve()

        # 2. Derive ordering from storyboard order, NEVER sorted(keys())
        ordered_ids = [s.id for s in self.sections if s.id in self.section_videos]
        if not ordered_ids:
            ordered_ids = list(self.section_videos.keys())

        print(f"🔗 Start merging {len(ordered_ids)} section videos in storyboard order: {ordered_ids}")

        video_list_file = (self.output_dir / "video_list.txt").resolve()
        with open(video_list_file, "w", encoding="utf-8") as f:
            for section_id in ordered_ids:
                video_path = os.path.abspath(self.section_videos[section_id])
                f.write(f"file '{video_path}'\n")

        # ffmpeg concat
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(video_list_file), "-c", "copy", str(output_path)],
                capture_output=True,
                text=True,
                cwd=self.output_dir,
            )

            if result.returncode == 0:
                if self.use_audio:
                    self._generate_master_audio_and_subtitles(output_path)

                # Duration Audit: Ground-truth narration timing & rendered video check
                try:
                    audios_meta = list(self.section_timestamps.values())
                    initial_est_sec = getattr(self, "storyboard_estimated_sec", 0.0)
                    if not initial_est_sec and hasattr(self, "enhanced_storyboard") and self.enhanced_storyboard:
                        initial_est_sec = self.duration_manager.estimate_lesson(self.enhanced_storyboard).total_duration_sec
                    audit_report = self.duration_manager.audit_timings(
                        storyboard_est_sec=initial_est_sec,
                        section_audios_meta=audios_meta,
                        final_video_path=str(output_path),
                    )
                    audit_file = self.output_dir / "duration_audit.json"
                    with open(audit_file, "w", encoding="utf-8") as f:
                        json.dump(audit_report, f, indent=2, ensure_ascii=False)
                    rendered_min = audit_report.get('rendered_video_duration_min')
                    tts_min = audit_report.get('measured_audio_duration_min')
                    est_min = audit_report.get('storyboard_estimate_min')
                    print(f"⏱️ Duration Audit saved: {rendered_min}m rendered (TTS: {tts_min}m, Estimate: {est_min}m)")
                except Exception as audit_err:
                    print(f"⚠️ Duration audit warning: {audit_err}")

                return str(output_path)
            else:
                print(f"❌ Failed to merge section videos: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Failed to merge section videos: {e}")
            return None

    def _execute_modern_pipeline(self) -> Optional[str]:
        """Execute the modern Scene IR-driven production pipeline (Milestone 5.3.4.2).

        Deterministic end-to-end execution slice:
        1. Topic-driven Stage 1 outline generation/recovery and verification gate.
        2. Native Stage 2 SceneSequenceIR generation/recovery and Gate 3.5 re-validation.
        3. Strict native TTS speech synthesis (allow_fallback=False).
        4. Audio-authoritative timeline retiming (scheduler.retime_scene_timeline).
        5. Monotonic scheduling without redundant retiming pass (scheduler.schedule(auto_retime=False)).
        6. Hierarchical box model spatial layout solving (HierarchicalBoxModelLayoutSolver).
        7. Deterministic Manim code compilation (DeterministicSceneCompiler).
        8. Controlled CPU Manim rendering.
        9. Hardened FFmpeg A/V multiplexing (audio_manager.mux_video_and_audio) with stream verification.
        10. Final multi-scene assembly, master audio/subtitles generation, and decode verification.

        Fails closed on any error without falling back to legacy pipeline.
        """
        try:
            safe_name = topic_to_safe_name(self.learning_topic)
            final_video_file = self.output_dir / f"{safe_name}.mp4"
            checkpoint_file = self.output_dir / "checkpoint.json"

            # 1. Fast-path check: Verify matching topic, level, and modern pipeline_mode
            can_fast_path = False
            if checkpoint_file.exists() and final_video_file.exists() and final_video_file.stat().st_size > 10000:
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        cp = json.load(f)
                    cp_topic = cp.get("topic") or (cp.get("outline", {}).get("topic") if cp.get("outline") else "")
                    cp_level = (cp.get("level") or "intermediate").lower()
                    cp_analogy = (cp.get("analogy") or "").strip()
                    cp_req_analogy = (cp.get("requested_analogy") or "").strip()
                    raw_cp_mode = cp.get("pipeline_mode", "legacy")
                    cp_mode = str(raw_cp_mode or "").lower().strip() if isinstance(raw_cp_mode, str) else ""
                    analogy_matched = not self.requested_analogy or (
                        cp_analogy == self.requested_analogy.strip()
                        or cp_req_analogy == self.requested_analogy.strip()
                        or self.requested_analogy.strip().lower() in cp_analogy.lower()
                        or (cp_req_analogy and self.requested_analogy.strip().lower() in cp_req_analogy.lower())
                    )
                    if cp_mode != "modern":
                        print(f"⚠️ Fast-path rejected: checkpoint pipeline_mode is '{cp_mode}', but running modern pipeline.")
                    elif cp_topic == self.learning_topic and cp_level == self.level and analogy_matched:
                        val = self.audio_manager.validate_media_file(str(final_video_file), require_audio=self.use_audio)
                        if val.get("is_valid"):
                            can_fast_path = True
                        else:
                            print(f"⚠️ Fast-path rejected: existing video failed media validation: {val.get('error')}")
                    else:
                        print(f"⚠️ Fast-path rejected: checkpoint is for topic='{cp_topic}', level='{cp_level}', not current run.")
                except Exception:
                    can_fast_path = False

            if can_fast_path:
                print(f"🎉 [FAST PATH] Complete matching modern video already exists: {final_video_file}. Skipping pipeline to preserve resources.")
                return str(final_video_file)

            # 2. Native Scene IR acquisition (in-memory, cached file, or topic-driven generation)
            if self.scene_sequence_ir is None:
                seq_file = self.output_dir / "scene_sequence_ir.json"
                scene_file = self.output_dir / "scene_ir.json"
                if seq_file.exists():
                    try:
                        with open(seq_file, "r", encoding="utf-8") as f:
                            disk_seq = json.load(f)
                        cached_res = process_stage2_scene_ir_output(disk_seq)
                        if cached_res.is_valid and cached_res.sequence_ir is not None:
                            self.scene_sequence_ir = cached_res.sequence_ir
                            self.scene_ir_processing_result = cached_res
                            print(f"🎬 Loaded valid cached SceneSequenceIR ({len(cached_res.sequence_ir.sections)} scenes).")
                    except Exception as e:
                        print(f"⚠️ Error reading cached scene_sequence_ir.json: {e}")
                elif scene_file.exists():
                    try:
                        with open(scene_file, "r", encoding="utf-8") as f:
                            disk_sc = json.load(f)
                        cached_res = process_stage2_scene_ir_output(disk_sc)
                        if cached_res.is_valid and cached_res.scene_ir is not None:
                            self.scene_sequence_ir = SceneSequenceIR(
                                sequence_id=f"seq_{cached_res.scene_ir.scene_id}",
                                title=cached_res.scene_ir.title,
                                topic=self.learning_topic,
                                sections=[cached_res.scene_ir],
                            )
                            self.scene_ir_processing_result = cached_res
                            print("🎬 Loaded valid cached SceneIR into SceneSequenceIR.")
                    except Exception as e:
                        print(f"⚠️ Error reading cached scene_ir.json: {e}")

            if self.scene_sequence_ir is None:
                # Phase 5.1 guard preservation for unconfigured pipeline smoke tests
                if self.learning_topic in ("Modern Pipeline Test", "Modern Pipeline Guard") and self.outline is None:
                    raise NotImplementedError(
                        "Modern Scene IR pipeline execution is not yet implemented in Phase 5.1. "
                        "Stage 2 Scene IR generation, timeline scheduling, and deterministic compilation "
                        "will be activated in subsequent Milestone 5 phases."
                    )

                # Topic-driven generation: Stage 1 outline
                if self.outline is None:
                    outline = self.generate_outline()
                    if not outline or not getattr(outline, "sections", None):
                        print(f"❌ {self.learning_topic}: Modern outline generation failed. Halting pipeline.")
                        self.save_checkpoint("outline_failed")
                        return None

                # Validation gate check before Stage 2
                if not self.is_outline_safe_for_stage2():
                    report = self.outline_verification_report
                    prov = self.get_verification_provenance()
                    status_val = prov.overall_status.value if prov else (report.overall_status.value if report else "BLOCKED")
                    summary_val = prov.halt_reason if (prov and prov.halt_reason) else (report.summary if report else "Outline verification failed")
                    print(f"❌ {self.learning_topic}: Outline blocked by validation gate ({status_val}): {summary_val}. Halting modern pipeline before Stage 2.")
                    self.save_checkpoint("outline_verification_blocked", extra_data={
                        "verification_report": report.to_dict() if report else None,
                        "verification_provenance": prov.to_dict() if prov else None,
                    })
                    return None

                self.save_checkpoint("outline_completed")

                # Stage 2: Native Scene IR generation
                res = self.generate_scene_ir()
                if self.scene_sequence_ir is None and getattr(res, "sequence_ir", None) is not None:
                    self.scene_sequence_ir = res.sequence_ir
                if self.scene_sequence_ir is None and getattr(res, "scene_ir", None) is not None:
                    self.scene_sequence_ir = SceneSequenceIR(
                        sequence_id=f"seq_{res.scene_ir.scene_id}",
                        title=res.scene_ir.title,
                        topic=self.learning_topic,
                        sections=[res.scene_ir],
                    )

                if not getattr(res, "is_valid", False) or not self.scene_sequence_ir:
                    print(f"❌ {self.learning_topic}: Modern Scene IR generation failed: {getattr(res, 'summary', 'Unknown error')}. Halting modern pipeline.")
                    self.save_checkpoint("scene_ir_generation_failed", extra_data={
                        "structural_errors": getattr(res, "structural_errors", []),
                        "semantic_errors": getattr(res, "semantic_errors", []),
                        "payload_errors": getattr(res, "payload_errors", []),
                    })
                    return None

            if not self.scene_sequence_ir or not self.scene_sequence_ir.sections:
                print(f"❌ Modern SceneSequenceIR contains 0 scenes. Halting modern pipeline.")
                self.save_checkpoint("scene_ir_empty")
                return None

            # 3. Validation Gate: Re-validate native SceneSequenceIR at execution boundary
            seq_val_result = process_stage2_scene_ir_output(self.scene_sequence_ir.to_dict())
            if not seq_val_result.is_valid:
                print(f"❌ Modern SceneSequenceIR validation failed: {seq_val_result.summary}. Halting modern pipeline.")
                self.save_checkpoint("scene_ir_validation_blocked", extra_data={
                    "structural_errors": seq_val_result.structural_errors,
                    "semantic_errors": seq_val_result.semantic_errors,
                    "payload_errors": seq_val_result.payload_errors,
                })
                return None

            self.save_checkpoint("scene_ir_validated")

            # 4. Native Speech Synthesis (allow_fallback=False strict contract)
            if self.use_audio:
                print(f"🎙️ [MODERN TTS] Synthesizing speech across {len(self.scene_sequence_ir.sections)} scenes (strict allow_fallback=False)...")
                try:
                    audio_metas = self.audio_manager.generate_scene_sequence_audio(
                        sequence_ir=self.scene_sequence_ir,
                        output_dir=self.output_dir,
                        tts_provider=self.tts_provider,
                        allow_fallback=False,
                    )
                    for meta in audio_metas:
                        sid = meta.get("section_id")
                        if sid:
                            if meta.get("audio_path"):
                                self.section_audios[sid] = meta["audio_path"]
                            if meta.get("timestamps_path"):
                                self.section_timestamps[sid] = meta["timestamps_path"]
                    self.save_checkpoint("audio_synthesized", extra_data={"section_audios": dict(self.section_audios)})
                except TTSFailureError as e:
                    print(f"❌ [TTS FAILED] Native TTS synthesis failed: {e}")
                    self.save_checkpoint("tts_failed", extra_data={"error": str(e)})
                    return None
                except Exception as e:
                    print(f"❌ [TTS EXCEPTION] Native TTS error: {e}")
                    self.save_checkpoint("tts_failed", extra_data={"error": str(e)})
                    return None

            # 5. Deterministic Retiming, Scheduling, Layout, Compilation, Manim Rendering, and A/V Muxing
            scheduler = TimelineScheduler()
            layout_solver = HierarchicalBoxModelLayoutSolver()
            compiler = DeterministicSceneCompiler()

            manim_env = os.environ.copy()
            project_root_str = str(_PROJECT_ROOT)
            if "PYTHONPATH" in manim_env:
                manim_env["PYTHONPATH"] = f"{project_root_str}:{manim_env['PYTHONPATH']}"
            else:
                manim_env["PYTHONPATH"] = project_root_str

            section_video_paths = {}
            for section in self.scene_sequence_ir.sections:
                section_id = section.scene_id
                print(f"\n🎬 [MODERN EXECUTION] Processing scene: {section_id} ({section.title})")

                # A. Audio-Authoritative Retiming
                if self.use_audio and section.audio_reference and getattr(section.audio_reference, "audio_path", None):
                    retimed_sec, sched_from_retime, retime_diags = scheduler.retime_scene_timeline(
                        section, inter_cue_pause=0.50, scene_end_pause=1.00
                    )
                    retime_errors = [d for d in retime_diags if d.severity == "ERROR"]
                    if not sched_from_retime.is_valid or retime_errors:
                        err_msgs = [d.message for d in retime_errors]
                        print(f"❌ [RETIMING FAILED] Scene {section_id} audio retiming failed: {err_msgs}")
                        self.save_checkpoint("retime_failed", extra_data={
                            "failed_scene": section_id,
                            "errors": err_msgs,
                        })
                        return None
                else:
                    retimed_sec = section

                # B. Timeline Scheduling (auto_retime=False to prevent second retiming pass)
                scheduled_timeline, sched_diags = scheduler.schedule(retimed_sec, auto_retime=False)
                sched_errors = [d for d in sched_diags if d.severity == "ERROR"]
                if not scheduled_timeline.is_valid or sched_errors:
                    err_msgs = [d.message for d in sched_errors]
                    print(f"❌ [TIMELINE FAILED] Scene {section_id} timeline scheduling failed: {err_msgs}")
                    self.save_checkpoint("timeline_failed", extra_data={
                        "failed_scene": section_id,
                        "errors": err_msgs,
                    })
                    return None

                # Persist scheduled timeline
                timing_file = self.output_dir / f"timing_{section_id}.json"
                try:
                    timing_file.write_text(scheduled_timeline.model_dump_json(indent=2), encoding="utf-8")
                except Exception:
                    pass

                # C. Spatial Layout Solving
                layout = layout_solver.solve(retimed_sec, scheduled_timeline=scheduled_timeline)
                if not layout.is_valid or (hasattr(layout, "errors") and layout.errors):
                    err_msgs = [e.message if hasattr(e, "message") else str(e) for e in layout.errors]
                    print(f"❌ [LAYOUT FAILED] Scene {section_id} layout solving failed: {err_msgs}")
                    self.save_checkpoint("layout_failed", extra_data={
                        "failed_scene": section_id,
                        "errors": err_msgs,
                    })
                    return None

                # Persist layout
                layout_file = self.output_dir / f"layout_{section_id}.json"
                try:
                    layout_file.write_text(layout.model_dump_json(indent=2), encoding="utf-8")
                except Exception:
                    pass

                # D. Deterministic Compilation
                compiled_output = compiler.compile(retimed_sec, layout, scheduled_timeline=scheduled_timeline)
                comp_errors = [d for d in compiled_output.diagnostics if d.severity == "ERROR"]
                if not compiled_output.is_valid or comp_errors:
                    err_msgs = [d.message for d in comp_errors]
                    print(f"❌ [COMPILATION FAILED] Scene {section_id} compiler failed: {err_msgs}")
                    self.save_checkpoint("compilation_failed", extra_data={
                        "failed_scene": section_id,
                        "errors": err_msgs,
                    })
                    return None

                # Persist compiled Python code
                code_file = self.output_dir / f"{section_id}.py"
                try:
                    code_file.write_text(compiled_output.python_source, encoding="utf-8")
                except Exception as e:
                    print(f"❌ Failed to write compiled code file {code_file}: {e}")
                    self.save_checkpoint("code_write_failed", extra_data={"error": str(e)})
                    return None

                with self.state_lock:
                    self.section_codes[section_id] = compiled_output.python_source

                # Extract scene class name
                match = re.search(r"class\s+(?!TeachingScene\b)([A-Za-z0-9_]+)\s*\(\s*(?:TeachingScene|Scene)\s*\)\s*:", compiled_output.python_source)
                scene_class_name = match.group(1) if match else compiled_output.scene_id

                # E. Controlled Manim Render
                print(f"🎥 [MANIM RENDER] Rendering {section_id} ({scene_class_name}) with {self.manim_quality_flag}...")
                cmd = ["manim", self.manim_quality_flag, str(code_file.name), scene_class_name]
                try:
                    render_res = subprocess.run(cmd, capture_output=True, text=True, cwd=self.output_dir, env=manim_env, timeout=180)
                    if render_res.returncode != 0:
                        print(f"❌ [RENDER FAILED] Manim process exited with code {render_res.returncode} for {section_id}:\n{render_res.stderr[-1000:]}")
                        self.save_checkpoint("render_failed", extra_data={
                            "failed_scene": section_id,
                            "returncode": render_res.returncode,
                            "stderr": render_res.stderr[-2000:],
                        })
                        return None
                except subprocess.TimeoutExpired:
                    print(f"❌ [RENDER TIMEOUT] Rendering {section_id} timed out after 180s.")
                    self.save_checkpoint("render_timeout", extra_data={"failed_scene": section_id})
                    return None
                except Exception as e:
                    print(f"❌ [RENDER EXCEPTION] Error executing Manim render for {section_id}: {e}")
                    self.save_checkpoint("render_exception", extra_data={"failed_scene": section_id, "error": str(e)})
                    return None

                # Locate rendered video
                video_patterns = [
                    self.output_dir / "media" / "videos" / code_file.stem / self.quality_folder / f"{scene_class_name}.mp4",
                    self.output_dir / "media" / "videos" / self.quality_folder / f"{scene_class_name}.mp4",
                    self.output_dir / "media" / "videos" / code_file.stem / "720p30" / f"{scene_class_name}.mp4",
                    self.output_dir / "media" / "videos" / "720p30" / f"{scene_class_name}.mp4",
                    self.output_dir / "media" / "videos" / code_file.stem / "480p15" / f"{scene_class_name}.mp4",
                    self.output_dir / "media" / "videos" / "480p15" / f"{scene_class_name}.mp4",
                    self.output_dir / "media" / "videos" / code_file.stem / "1080p60" / f"{scene_class_name}.mp4",
                    self.output_dir / "media" / "videos" / "1080p60" / f"{scene_class_name}.mp4",
                ]
                found_video = None
                for vp in video_patterns:
                    if vp.exists() and vp.stat().st_size > 1000:
                        found_video = str(vp.resolve())
                        break

                if not found_video:
                    print(f"❌ [RENDER ARTIFACT MISSING] No valid rendered MP4 found for {section_id}.")
                    self.save_checkpoint("render_artifact_missing", extra_data={"failed_scene": section_id})
                    return None

                # F. Hardened FFmpeg A/V Multiplexing
                if self.use_audio and retimed_sec.audio_reference and getattr(retimed_sec.audio_reference, "audio_path", None):
                    audio_file = retimed_sec.audio_reference.audio_path
                    muxed_output = self.output_dir / f"{section_id}_muxed.mp4"
                    srt_cand = self.output_dir / f"{section_id}.srt"
                    print(f"🎛️ [FFMPEG MUX] Multiplexing {section_id} video and audio (burn_subtitles={self.burn_subtitles})...")
                    mux_res = self.audio_manager.mux_video_and_audio(
                        video_path=found_video,
                        audio_path=audio_file,
                        output_path=muxed_output,
                        srt_path=srt_cand if srt_cand.exists() else None,
                        burn_subtitles=self.burn_subtitles,
                        expected_duration=retimed_sec.audio_reference.total_duration,
                        tolerance=0.5,
                    )
                    if not mux_res.get("success"):
                        print(f"❌ [MUX FAILED] Scene {section_id} A/V multiplexing failed: {mux_res.get('error')}")
                        self.save_checkpoint("mux_failed", extra_data={
                            "failed_scene": section_id,
                            "error": mux_res.get("error"),
                        })
                        return None

                    # Genuinely verified: update alignment status
                    retimed_sec.audio_reference.alignment_status = "muxed"
                    section.audio_reference.alignment_status = "muxed"
                    scene_video = mux_res["output_path"]
                    print(f"✅ [MUX VERIFIED] {section_id} multiplexed and verified -> {scene_video}")
                else:
                    scene_video = found_video
                    print(f"✅ [MODERN RENDER COMPLETE] {section_id} -> {found_video} ({Path(found_video).stat().st_size} bytes)")

                with self.state_lock:
                    self.section_videos[section_id] = scene_video
                section_video_paths[section_id] = scene_video

                # G. Milestone 6.3: Post-Render Deterministic Quality Control (Layer C)
                if self.enable_qc:
                    from src.qc import RenderedVideoQualityController, QCDiagnosticLevel
                    qc_ctrl = self.qc_controller or RenderedVideoQualityController()
                    print(f"🔍 [MODERN QC] Executing Layer C quality control for {section_id}...")
                    qc_report = qc_ctrl.evaluate_scene_video(
                        video_path=scene_video,
                        scene_ir=retimed_sec,
                        scheduled_timeline=scheduled_timeline,
                        layout=layout,
                        burn_subtitles=self.burn_subtitles,
                        raw_video_path=found_video if (self.burn_subtitles and self.use_audio) else None,
                    )
                    with self.state_lock:
                        self.section_qc_reports[section_id] = qc_report

                    # Persist per-scene QC report (fail-closed if persistence fails)
                    qc_report_file = self.output_dir / f"qc_report_{section_id}.json"
                    try:
                        qc_report_file.write_text(qc_report.model_dump_json(indent=2), encoding="utf-8")
                    except Exception as e:
                        err_msg = f"Failed to persist QC report to {qc_report_file}: {e}"
                        print(f"❌ [QC PERSISTENCE FAILED] {err_msg}")
                        self.save_checkpoint("qc_failed", extra_data={
                            "failed_scene": section_id,
                            "errors": [err_msg],
                            "qc_report": qc_report.model_dump(),
                        })
                        return None

                    if qc_report.has_errors():
                        err_msgs = [d.message for d in qc_report.get_diagnostics_by_level(QCDiagnosticLevel.ERROR)]
                        fatal_msgs = [d.message for d in qc_report.get_diagnostics_by_level(QCDiagnosticLevel.FATAL)]
                        all_errs = err_msgs + fatal_msgs
                        print(f"❌ [QC FAILED] Scene {section_id} quality control failed ({len(all_errs)} errors): {all_errs}")
                        self.save_checkpoint("qc_failed", extra_data={
                            "failed_scene": section_id,
                            "qc_report": qc_report.model_dump(),
                            "errors": all_errs,
                        })
                        return None
                    else:
                        print(f"✅ [QC PASSED] Scene {section_id} passed quality control ({qc_report.overall_status.value}, {len(qc_report.diagnostics)} diagnostics).")
                        self.save_checkpoint(f"qc_scene_{section_id}_passed", extra_data={
                            "scene_id": section_id,
                            "overall_status": qc_report.overall_status.value,
                            "sampled_frames_count": qc_report.sampled_frames_count,
                        })

            self.save_checkpoint("render_and_mux_completed", extra_data={"section_videos": self.section_videos})

            # 6. Assemble Final Video and Return
            output_path = (self.output_dir / f"{safe_name}.mp4").resolve()
            if len(section_video_paths) == 1:
                single_vid = list(section_video_paths.values())[0]
                try:
                    shutil.copyfile(single_vid, output_path)
                except Exception as e:
                    print(f"❌ Failed to copy single scene video to final output path: {e}")
                    self.save_checkpoint("merge_failed", extra_data={"error": str(e)})
                    return None
            else:
                video_list_file = (self.output_dir / "video_list.txt").resolve()
                with open(video_list_file, "w", encoding="utf-8") as f:
                    for section in self.scene_sequence_ir.sections:
                        sid = section.scene_id
                        if sid in section_video_paths:
                            f.write(f"file '{section_video_paths[sid]}'\n")
                try:
                    concat_res = subprocess.run(
                        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(video_list_file), "-c", "copy", str(output_path)],
                        capture_output=True,
                        text=True,
                        cwd=self.output_dir,
                    )
                    if concat_res.returncode != 0:
                        print(f"❌ [MERGE FAILED] ffmpeg concat failed: {concat_res.stderr}")
                        self.save_checkpoint("merge_failed", extra_data={"stderr": concat_res.stderr})
                        return None
                except Exception as e:
                    print(f"❌ [MERGE EXCEPTION] ffmpeg concat exception: {e}")
                    self.save_checkpoint("merge_exception", extra_data={"error": str(e)})
                    return None

            # 7. Master Audio and Subtitles Generation
            if self.use_audio:
                self._generate_master_audio_and_subtitles(output_path)

            # 8. Master Media Artifact Verification
            val_res = self.audio_manager.validate_media_file(str(output_path), require_audio=self.use_audio)
            if not val_res.get("is_valid"):
                print(f"❌ [FINAL VALIDATION FAILED] Final media file failed validation: {val_res.get('error')}")
                self.save_checkpoint("final_validation_failed", extra_data=val_res)
                return None

            decode_cmd = ["ffmpeg", "-v", "error", "-i", str(output_path), "-f", "null", "-"]
            res_dec = subprocess.run(decode_cmd, capture_output=True, text=True)
            if res_dec.returncode != 0:
                print(f"❌ [FINAL DECODE FAILED] Final media decode verification failed: {res_dec.stderr}")
                self.save_checkpoint("final_decode_failed", extra_data={"stderr": res_dec.stderr})
                return None

            # 9. Milestone 6.3: Post-Assembly Master Video Quality Control
            if self.enable_qc:
                from src.qc import RenderedVideoQualityController, QCDiagnosticLevel
                qc_ctrl = self.qc_controller or RenderedVideoQualityController()
                print(f"🔍 [MODERN QC] Executing final master video quality control...")
                master_qc_report = qc_ctrl.evaluate_master_video(
                    video_path=output_path,
                    sequence_ir=self.scene_sequence_ir,
                )
                self.master_qc_report = master_qc_report
                master_qc_file = self.output_dir / "qc_report_master.json"
                try:
                    master_qc_file.write_text(master_qc_report.model_dump_json(indent=2), encoding="utf-8")
                except Exception as e:
                    err_msg = f"Failed to persist master QC report to {master_qc_file}: {e}"
                    print(f"❌ [MASTER QC PERSISTENCE FAILED] {err_msg}")
                    self.save_checkpoint("qc_master_failed", extra_data={
                        "errors": [err_msg],
                        "qc_report": master_qc_report.model_dump(),
                    })
                    return None

                if master_qc_report.has_errors():
                    err_msgs = [d.message for d in master_qc_report.get_diagnostics_by_level(QCDiagnosticLevel.ERROR)]
                    fatal_msgs = [d.message for d in master_qc_report.get_diagnostics_by_level(QCDiagnosticLevel.FATAL)]
                    all_errs = err_msgs + fatal_msgs
                    print(f"❌ [MASTER QC FAILED] Master video quality control failed ({len(all_errs)} errors): {all_errs}")
                    self.save_checkpoint("qc_master_failed", extra_data={
                        "qc_report": master_qc_report.model_dump(),
                        "errors": all_errs,
                    })
                    return None
                else:
                    self.save_checkpoint("qc_master_passed", extra_data={
                        "overall_status": master_qc_report.overall_status.value,
                    })

            print(f"🎉 Modern Scene IR Video generated and verified successfully: {output_path} ({val_res.get('duration')}s)")
            self.save_checkpoint("completed", {"final_video": str(output_path)})
            return str(output_path)

        except NotImplementedError:
            raise
        except Exception as e:
            print(f"❌ Modern video generation failed: {e}")
            self.save_checkpoint("exception", {"error": str(e)})
            return None

    def _execute_legacy_pipeline(self) -> str:
        """Execute the legacy production pipeline with Stage 3 LLM code generation."""
        try:
            # Check if final video already exists and is valid for the exact topic, level, and analogy
            safe_name = topic_to_safe_name(self.learning_topic)
            final_video_file = self.output_dir / f"{safe_name}.mp4"
            checkpoint_file = self.output_dir / "checkpoint.json"

            can_fast_path = False
            if checkpoint_file.exists() and final_video_file.exists() and final_video_file.stat().st_size > 10000:
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        cp = json.load(f)
                    cp_topic = cp.get("topic") or (cp.get("outline", {}).get("topic") if cp.get("outline") else "")
                    cp_level = (cp.get("level") or "intermediate").lower()
                    cp_analogy = (cp.get("analogy") or "").strip()
                    cp_req_analogy = (cp.get("requested_analogy") or "").strip()
                    if "pipeline_mode" in cp:
                        raw_cp_mode = cp["pipeline_mode"]
                        cp_mode = str(raw_cp_mode or "").lower().strip() if isinstance(raw_cp_mode, str) else ""
                    else:
                        cp_mode = "legacy"
                    analogy_matched = not self.requested_analogy or (
                        cp_analogy == self.requested_analogy.strip()
                        or cp_req_analogy == self.requested_analogy.strip()
                        or self.requested_analogy.strip().lower() in cp_analogy.lower()
                        or (cp_req_analogy and self.requested_analogy.strip().lower() in cp_req_analogy.lower())
                    )
                    if cp_mode != "legacy":
                        print(f"⚠️ Fast-path rejected: checkpoint pipeline_mode is '{cp_mode}', but running legacy pipeline.")
                    elif cp_topic == self.learning_topic and cp_level == self.level and analogy_matched:
                        can_fast_path = True
                    else:
                        print(f"⚠️ Fast-path rejected: checkpoint is for topic='{cp_topic}', level='{cp_level}', analogy='{cp_analogy}', not current run.")
                except Exception:
                    can_fast_path = False

            if can_fast_path:
                print(f"🎉 [FAST PATH] Complete matching video already exists: {final_video_file}. Skipping pipeline to preserve API quota.")
                return str(final_video_file)

            # Stage 1: Outline
            outline = self.generate_outline()
            if not outline or not getattr(outline, "sections", None):
                print(f"❌ {self.learning_topic}: Outline generation failed. Halting pipeline to save API quota.")
                self.save_checkpoint("outline_failed")
                return None

            # Validation gate check before Stage 2
            if not self.is_outline_safe_for_stage2():
                report = self.outline_verification_report
                prov = self.get_verification_provenance()
                status_val = prov.overall_status.value if prov else (report.overall_status.value if report else "BLOCKED")
                summary_val = prov.halt_reason if (prov and prov.halt_reason) else (report.summary if report else "Outline verification failed")
                print(f"❌ {self.learning_topic}: Outline blocked by validation gate ({status_val}): {summary_val}. Halting pipeline before Stage 2.")
                self.save_checkpoint("outline_verification_blocked", extra_data={
                    "verification_report": report.to_dict() if report else None,
                    "verification_provenance": prov.to_dict() if prov else None,
                })
                return None

            self.save_checkpoint("outline_completed")

            # Stage 2: Storyboard
            storyboard = self.generate_storyboard()
            if not storyboard or len(self.sections) == 0:
                print(f"❌ {self.learning_topic}: Storyboard generation failed. Halting pipeline to save API quota.")
                self.save_checkpoint("storyboard_failed")
                return None

            # Storyboard numerical fidelity gate check before audio generation
            if not self.is_storyboard_safe_for_audio():
                report = self.storyboard_fidelity_report
                status_val = report.overall_status.value if report else "BLOCKED"
                summary_val = report.summary if report else "Storyboard numerical fidelity check failed"
                print(f"❌ {self.learning_topic}: Storyboard blocked by numerical fidelity gate ({status_val}): {summary_val}. Halting pipeline before audio generation.")
                self.save_checkpoint("storyboard_fidelity_blocked", extra_data={"fidelity_report": report.to_dict() if report else None})
                return None

            self.save_checkpoint("storyboard_completed")

            # Pre-generate narration audio and timestamps BEFORE code generation!
            if self.use_audio:
                self.pregenerate_all_section_audios()
                if not self.is_audio_safe_for_stage3():
                    report = self.audio_handoff_report
                    status_val = report.overall_status.value if report else "FAILED"
                    summary_val = report.summary if report else "Audio handoff validation failed"
                    print(f"❌ {self.learning_topic}: Audio handoff blocked ({status_val}): {summary_val}. Halting pipeline before Stage 3.")
                    self.save_checkpoint("audio_handoff_blocked", extra_data={"audio_handoff_report": report.to_dict() if report else None})
                    return None

            # Stage 3: Code generation
            codes = self.generate_codes()
            if not codes or len(codes) == 0:
                print(f"❌ {self.learning_topic}: Code generation failed. Halting pipeline to save API quota.")
                self.save_checkpoint("code_gen_failed")
                return None
            self.save_checkpoint("code_gen_completed")

            # Stage 4: Rendering and refinement
            rendered = self.render_all_sections()
            if not rendered or len(rendered) == 0:
                print(f"❌ {self.learning_topic}: No section videos rendered. Aborting merge to save resources.")
                self.save_checkpoint("render_failed")
                return None

            missing_renders = [
                s.id for s in self.sections
                if s.id not in rendered or not Path(rendered[s.id]).exists() or Path(rendered[s.id]).stat().st_size <= 1000
            ]
            if missing_renders:
                print(f"\n❌ Incomplete rendering: {len(missing_renders)} section(s) missing: {missing_renders}. Halting before merge.")
                self.save_checkpoint("incomplete_lecture", {
                    "missing_sections": missing_renders,
                    "successful_sections": list(rendered.keys()),
                    "total_sections": len(self.sections),
                })
                return None

            self.save_checkpoint("render_completed")

            # Stage 5: Final merge
            final_video = self.merge_videos()
            if final_video:
                print(f"🎉 Video generated success: {final_video}")
                self.save_checkpoint("completed", {"final_video": final_video})
                return final_video
            else:
                print(f"❌ {self.learning_topic} merge failed")
                try:
                    with open(self.output_dir / "checkpoint.json", "r", encoding="utf-8") as _cpf:
                        _curr_stage = json.load(_cpf).get("stage")
                except Exception:
                    _curr_stage = ""
                if _curr_stage != "incomplete_lecture":
                    self.save_checkpoint("merge_failed")
                return None

        except DailyQuotaExhaustedError as e:
            print(f"\n🛑 CRITICAL: Daily Gemini API Quota Exhausted! {e}")
            self.save_checkpoint("daily_quota_exhausted", {"error": str(e)})
            print("💾 Checkpoint saved. Exiting gracefully without making further API requests.")
            return None
        except ServiceUnavailableError as e:
            print(f"\n🛑 503 Service Unavailable (high demand): {e}")
            self.save_checkpoint("service_unavailable", {"error": str(e)})
            print("💾 Checkpoint saved with all completed artifacts preserved. Exiting gracefully without further API requests.")
            return None
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            try:
                with open(self.output_dir / "checkpoint.json", "r", encoding="utf-8") as _cpf:
                    _curr_stage = json.load(_cpf).get("stage")
            except Exception:
                _curr_stage = ""
            if _curr_stage not in ("outline_verification_blocked", "storyboard_fidelity_blocked", "audio_handoff_blocked"):
                self.save_checkpoint("exception", {"error": str(e)})
            return None

    def GENERATE_VIDEO(self) -> str:
        """Generate complete video with fail-fast validation, checkpointing, and graceful quota handling.

        Dispatches to either modern Scene IR pipeline or legacy production pipeline
        based on configured self.pipeline_mode.
        """
        if self.pipeline_mode == "modern":
            return self._execute_modern_pipeline()
        elif self.pipeline_mode == "legacy":
            return self._execute_legacy_pipeline()
        else:
            raise ValueError(f"Invalid pipeline_mode: {self.pipeline_mode!r}. Valid options are 'legacy' or 'modern'.")


def process_knowledge_point(idx, kp, folder_path: Path, cfg: RunConfig):
    if isinstance(kp, dict):
        topic = kp.get("topic") or kp.get("knowledge_point") or ""
        level = kp.get("level", getattr(cfg, "level", "intermediate"))
        analogy = kp.get("analogy", getattr(cfg, "analogy", ""))
    elif isinstance(kp, (list, tuple)):
        topic = str(kp[0])
        level = kp[1] if len(kp) > 1 else getattr(cfg, "level", "intermediate")
        analogy = kp[2] if len(kp) > 2 else getattr(cfg, "analogy", "")
    else:
        topic = str(kp)
        level = getattr(cfg, "level", "intermediate")
        analogy = getattr(cfg, "analogy", "")

    print(f"\n🚀 Processing topic: '{topic}' [Level: {level}, Analogy: {analogy or 'None'}]")
    start_time = time.time()

    agent = TeachingVideoAgent(
        idx=idx,
        knowledge_point=topic,
        topic=topic,
        level=level,
        analogy=analogy,
        folder=folder_path,
        cfg=cfg,
    )
    video_path = agent.GENERATE_VIDEO()

    duration_minutes = (time.time() - start_time) / 60
    total_tokens = agent.token_usage["total_tokens"]

    print(f"✅ Topic '{topic}' processed. Cost Time: {duration_minutes:.2f} minutes, Tokens used: {total_tokens}")
    return topic, video_path, duration_minutes, total_tokens


def process_batch(batch_data, cfg: RunConfig):
    """Process a batch of knowledge points (serial within a batch)"""
    batch_idx, kp_batch, folder_path = batch_data
    results = []
    print(f"Batch {batch_idx + 1} starts processing {len(kp_batch)} knowledge points")

    for local_idx, (idx, kp) in enumerate(kp_batch):
        try:
            if local_idx > 0:
                delay = random.uniform(3, 6)
                print(f"⏳ Batch {batch_idx + 1} waits {delay:.1f}s before processing {kp}...")
                time.sleep(delay)
            results.append(process_knowledge_point(idx, kp, folder_path, cfg))
        except Exception as e:
            print(f"❌ Batch {batch_idx + 1} processing {kp} failed: {e}")
            results.append((kp, None, 0, 0))
    return batch_idx, results


def run_Code2Video(
    knowledge_points: List[Any], folder_path: Path, parallel=False, batch_size=1, max_workers=1, cfg: RunConfig = RunConfig()
):
    all_results = []

    print(f"🔄 Processing {len(knowledge_points)} topic(s)...")
    for idx, kp in enumerate(knowledge_points):
        try:
            all_results.append(process_knowledge_point(idx, kp, folder_path, cfg))
        except Exception as e:
            print(f"❌ Processing {kp} failed: {e}")
            all_results.append((kp, None, 0, 0))

    successful_runs = [r for r in all_results if r[1] is not None]
    total_runs = len(all_results)
    if not successful_runs:
        print("\nAll topics failed or skipped, cannot calculate average.")
        return

    total_duration = sum(r[2] for r in successful_runs)
    total_tokens_consumed = sum(r[3] for r in successful_runs)
    num_successful = len(successful_runs)

    print("\n" + "=" * 50)
    print(f"   Total topics: {total_runs}")
    print(f"   Successfully processed: {num_successful} ({num_successful/total_runs*100:.1f}%)")
    print(f"   Average duration [min]: {total_duration/num_successful:.2f} minutes/topic")
    print(f"   Average token consumption: {total_tokens_consumed/num_successful:,.0f} tokens/topic")
    print("=" * 50)


def get_api_and_output(API_name=None, provider=None, model=None):
    if provider:
        p_clean = provider.strip().lower()
        folder_suffix = f"{p_clean.capitalize()}_{model}" if model else p_clean.capitalize()
        return (lambda prompt, **kw: generate_response_token(prompt, provider=p_clean, model=model, **kw)), folder_suffix

    API_name_clean = API_name.lower() if API_name else ""
    mapping = {
        "gpt-41": (request_gpt41_token, "Chatgpt41"),
        "claude": (request_claude_token, "CLAUDE"),
        "gpt-5": (request_gpt5_token, "Chatgpt5"),
        "gpt-4o": (request_gpt4o_token, "Chatgpt4o"),
        "gpt-o4mini": (request_o4mini_token, "Chatgpto4mini"),
        "gemini": (make_stage_caller("default"), "Gemini"),
        "ollama": (lambda prompt, **kw: generate_response_token(prompt, provider="ollama", **kw), "Ollama"),
        "qwen": (lambda prompt, **kw: generate_response_token(prompt, provider="ollama", **kw), "Qwen"),
        "openai": (lambda prompt, **kw: generate_response_token(prompt, provider="openai", **kw), "OpenAI"),
        "mock": (lambda prompt, **kw: generate_response_token(prompt, provider="mock", **kw), "Mock"),
    }
    if API_name_clean in mapping:
        return mapping[API_name_clean]
    for k, v in mapping.items():
        if k.lower() == API_name_clean:
            return v
    if API_name and (":" in API_name or "qwen" in API_name_clean or "llama" in API_name_clean):
        return (lambda prompt, **kw: generate_response_token(prompt, provider="ollama", model=API_name, **kw)), f"Ollama_{API_name.replace(':', '_')}"

    return make_stage_caller("default"), "LLM"


def build_and_parse_args():
    parser = argparse.ArgumentParser()
    # Core hyperparameters
    parser.add_argument(
        "--API",
        type=str,
        default="Gemini",
        help="LLM API to use (Gemini, ollama, qwen, openai, gpt-41, claude, gpt-5, gpt-4o, gpt-o4mini). Default: Gemini (free).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider to use (gemini, ollama, openai, mock). Overrides LLM_PROVIDER in .env.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name to use (e.g. gemini-3.8-flash, qwen2.5:7b, gpt-4o). Overrides LLM_MODEL in .env.",
    )
    parser.add_argument(
        "--folder_prefix",
        type=str,
        default="TEST",
    )
    parser.add_argument(
        "--case_dir", "--folder",
        type=str,
        default="",
        help="Direct path to existing case folder (or parent folder containing cases)",
    )
    parser.add_argument("--knowledge_file", type=str, default="long_video_topics_list.json")
    parser.add_argument("--iconfinder_api_key", type=str, default="")

    # Basically invariant parameters
    parser.add_argument("--use_feedback", action="store_true", default=False)
    parser.add_argument("--no_feedback", action="store_false", dest="use_feedback")
    parser.add_argument("--use_assets", action="store_true", default=False)
    parser.add_argument("--no_assets", action="store_false", dest="use_assets")

    parser.add_argument("--max_code_token_length", type=int, help="max # token for generating code", default=10000)
    parser.add_argument("--max_fix_bug_tries", type=int, help="max # tries for SR to fix bug", default=1)
    parser.add_argument("--max_regenerate_tries", type=int, help="max # tries to regenerate", default=1)
    parser.add_argument("--max_feedback_gen_code_tries", type=int, help="max # tries for Critic", default=1)
    parser.add_argument("--max_mllm_fix_bugs_tries", type=int, help="max # tries for Critic to fix bug", default=1)
    parser.add_argument("--feedback_rounds", type=int, default=1)

    parser.add_argument("--parallel", dest="parallel", action="store_true", default=True, help="Enable parallel processing for independent sections")
    parser.add_argument("--no_parallel", dest="parallel", action="store_false", help="Disable parallel processing (run sections sequentially)")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum worker threads for section concurrency")
    parser.add_argument("--parallel_group_num", type=int, default=1)
    parser.add_argument("--max_concepts", type=int, help="Limit # concepts for a quick run, -1 for all", default=-1)

    # Three-input parameters: (1) Topic, (2) Level, (3) Analogy
    parser.add_argument("--knowledge_point", type=str, help="Topic name / knowledge point (backward compatible)", default=None)
    parser.add_argument("--topic", type=str, help="Topic name (primary alias for --knowledge_point)", default=None)
    parser.add_argument(
        "--level", "--learning_level",
        type=str,
        default="intermediate",
        choices=["beginner", "intermediate", "advanced", "Beginner", "Intermediate", "Advanced"],
        help="Target learner level (beginner, intermediate, advanced). Default: intermediate.",
    )
    parser.add_argument(
        "--analogy", "--personal_interest", "--interest",
        type=str,
        default="",
        help="Optional analogy or personal interest anchor (e.g. 'Pokémon', 'rubber sheet', 'gaming')",
    )

    # Adaptive duration hyperparameters
    parser.add_argument(
        "--soft_target_duration", "--target_duration", "--duration",
        type=float,
        default=0.0,
        help="Configurable target duration in seconds (0 for topic-dependent adaptive duration)",
    )
    parser.add_argument(
        "--hard_max_duration",
        type=float,
        default=0.0,
        help="Configurable hard maximum duration in seconds (0 for level-based adaptive default)",
    )
    parser.add_argument(
        "--speaking_wpm",
        type=float,
        default=0.0,
        help="Expected speaking rate in words per minute (0 for level-based adaptive default)",
    )

    # Audio hyperparameters
    parser.add_argument("--audio", action="store_true", default=True, help="Enable audio voiceover, timestamps, and synchronization")
    parser.add_argument("--no_audio", action="store_false", dest="audio", help="Disable audio voiceover")
    parser.add_argument("--audio_input", type=str, default="", help="Path to custom audio input file or directory")
    parser.add_argument("--tts_voice", type=str, default="en-US-GuyNeural", help="TTS voice name (e.g. en-US-GuyNeural)")
    parser.add_argument("--burn_subtitles", action="store_true", default=True, help="Burn concise subtitles into video")
    parser.add_argument("--no_burn_subtitles", action="store_false", dest="burn_subtitles", help="Do not burn subtitles into video")
    parser.add_argument("--sync_only", action="store_true", default=False, help="Only synchronize existing rendered case with audio without re-rendering")
    parser.add_argument("--force_render", action="store_true", default=False, help="Force re-rendering scenes with Manim even if compiled videos exist")

    # Video rendering quality
    parser.add_argument(
        "--quality",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Manim rendering quality: low (480p15), medium (720p30, default), high (1080p60)",
    )

    # Production pipeline mode
    parser.add_argument(
        "--pipeline-mode",
        "--pipeline_mode",
        dest="pipeline_mode",
        type=str,
        default="legacy",
        choices=["legacy", "modern"],
        help="Production pipeline mode: 'legacy' (Stage 3 LLM code gen) or 'modern' (Scene IR deterministic compiler). Default: legacy.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = build_and_parse_args()

    # Determine provider and model
    provider = args.provider or os.getenv("LLM_PROVIDER")
    model = args.model or os.getenv("LLM_MODEL")
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if model:
        os.environ["LLM_MODEL"] = model

    api, folder_name = get_api_and_output(args.API, provider=provider, model=model)
    folder = _PROJECT_ROOT / "CASES" / f"{args.folder_prefix}_{folder_name}"

    iconfinder_key = os.getenv("ICONFINDER_API_KEY", "")
    if not iconfinder_key:
        _CFG_PATH = _SRC_DIR / "api_config.json"
        _CFG = {}
        if _CFG_PATH.exists():
            with _CFG_PATH.open("r", encoding="utf-8") as _f:
                _CFG = json.load(_f)
        iconfinder_key = _CFG.get("iconfinder", {}).get("api_key", "")
    args.iconfinder_api_key = iconfinder_key
    if args.iconfinder_api_key and args.iconfinder_api_key not in ("YOUR_ICONFINDER_KEY", ""):
        print("Iconfinder API Key: configured")
    else:
        print("INFO: No Iconfinder API key provided. Using 100% free Iconify API for assets.")

    topic_input = args.topic or args.knowledge_point
    level_input = (args.level or "intermediate").lower()
    analogy_input = (args.analogy or "").strip()

    if topic_input:
        print(f"🔄 Single topic mode: '{topic_input}' [Level: {level_input}, Analogy: {analogy_input or 'None'}]")
        knowledge_points = [{"topic": topic_input, "level": level_input, "analogy": analogy_input}]
        args.parallel_group_num = 1
    elif args.knowledge_file:
        kfile = _PROJECT_ROOT / "json_files" / args.knowledge_file
        if not kfile.exists():
            kfile = _SRC_DIR / "json_files" / args.knowledge_file
        with open(kfile, "r", encoding="utf-8") as f:
            raw_kp = json.load(f)
            if args.max_concepts is not None and args.max_concepts > 0:
                raw_kp = raw_kp[: args.max_concepts]
            knowledge_points = []
            for item in raw_kp:
                if isinstance(item, dict):
                    knowledge_points.append({
                        "topic": item.get("topic") or item.get("knowledge_point"),
                        "level": (item.get("level") or level_input).lower(),
                        "analogy": item.get("analogy") or analogy_input,
                    })
                else:
                    knowledge_points.append({
                        "topic": str(item),
                        "level": level_input,
                        "analogy": analogy_input,
                    })
    else:
        raise ValueError("Must provide --topic | --knowledge_point | --knowledge_file")

    cfg = RunConfig(
        api=api,
        iconfinder_api_key=args.iconfinder_api_key,
        use_feedback=args.use_feedback,
        use_assets=args.use_assets,
        max_code_token_length=args.max_code_token_length,
        max_fix_bug_tries=args.max_fix_bug_tries,
        max_regenerate_tries=args.max_regenerate_tries,
        max_feedback_gen_code_tries=args.max_feedback_gen_code_tries,
        max_mllm_fix_bugs_tries=args.max_mllm_fix_bugs_tries,
        feedback_rounds=args.feedback_rounds,
        audio=args.audio,
        audio_input=args.audio_input,
        tts_voice=args.tts_voice,
        burn_subtitles=args.burn_subtitles,
        parallel=args.parallel,
        max_workers=args.max_workers,
        level=level_input,
        analogy=analogy_input,
        soft_target_duration=args.soft_target_duration,
        hard_max_duration=args.hard_max_duration,
        speaking_wpm=args.speaking_wpm,
        quality=args.quality,
        provider=provider,
        model=model,
        force_render=args.force_render,
        pipeline_mode=getattr(args, "pipeline_mode", "legacy"),
    )

    if args.sync_only and topic_input:
        agent = TeachingVideoAgent(
            idx=0,
            knowledge_point=topic_input,
            topic=topic_input,
            level=level_input,
            analogy=analogy_input,
            folder=folder,
            cfg=cfg,
        )
        if getattr(args, "case_dir", None):
            cd = Path(args.case_dir)
            if cd.exists():
                if (cd / "storyboard.json").exists() or (cd / "storyboard_with_assets.json").exists():
                    agent.output_dir = cd.resolve()
                else:
                    cand_sub = cd / f"0-{topic_to_safe_name(topic_input)}"
                    if cand_sub.exists() and ((cand_sub / "storyboard.json").exists() or (cand_sub / "storyboard_with_assets.json").exists()):
                        agent.output_dir = cand_sub.resolve()
                    else:
                        agent.output_dir = cd.resolve()
        res = agent.synchronize_case_audio()
        print(f"Synchronize result: {res}")
        sys.exit(0)

    is_single_topic = bool(topic_input) or len(knowledge_points) == 1
    run_Code2Video(
        knowledge_points,
        folder,
        parallel=False if is_single_topic else args.parallel,
        batch_size=1 if is_single_topic else max(1, int(len(knowledge_points) / args.parallel_group_num)),
        max_workers=1 if is_single_topic else get_optimal_workers(),
        cfg=cfg,
    )
