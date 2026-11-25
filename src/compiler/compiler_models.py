"""
CineLearn — Milestone 4B: Deterministic Scene Compiler Models
=============================================================
Defines typed models, error codes, and structured diagnostics for compiling
validated Scene IR and spatial layouts into deterministic Manim Python source.
"""

from __future__ import annotations

from enum import Enum
import keyword
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ir.models import ActionDeclaration


def is_safe_keyword_identifier(k: Any) -> bool:
    """Verifies that k is a safe Python identifier, not a reserved keyword, and not a dunder."""
    if not isinstance(k, str) or not k.isidentifier():
        return False
    if keyword.iskeyword(k) or (hasattr(keyword, "issoftkeyword") and keyword.issoftkeyword(k)):
        return False
    if k.startswith("__"):
        return False
    return True


def canonicalize_time(val: Any) -> float:
    """
    Normalizes a numerical time or duration value to canonical floating-point
    representation rounded to integer milliseconds (3 decimal places).
    Strictly rejects boolean values, non-numeric types, non-finite values,
    oversized integers (>64-bit), and negative numbers.
    """
    if isinstance(val, bool):
        raise ValueError("Timestamp cannot be a boolean value.")
    if not isinstance(val, (int, float)):
        raise ValueError(f"Timestamp must be numeric, got {type(val).__name__}.")
    if isinstance(val, int) and val.bit_length() > 64:
        raise ValueError(f"Integer timestamp exceeds 64-bit precision: {val}.")
    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"Timestamp must be finite, got {val}.")
    if val < 0.0:
        raise ValueError(f"Timestamp cannot be negative, got {val}.")
    res = round(float(val), 3)
    return 0.0 if res == 0.0 else res


class CompilationErrorCode(str, Enum):
    """Machine-readable compiler diagnostic and error codes."""
    COMPILER_MISSING_GEOMETRY = "COMPILER_MISSING_GEOMETRY"
    COMPILER_UNKNOWN_ENTITY = "COMPILER_UNKNOWN_ENTITY"
    COMPILER_UNSUPPORTED_COMPONENT = "COMPILER_UNSUPPORTED_COMPONENT"
    COMPILER_UNSUPPORTED_OPERATION = "COMPILER_UNSUPPORTED_OPERATION"
    COMPILER_UNSUPPORTED_FEATURE = "COMPILER_UNSUPPORTED_FEATURE"
    COMPILER_INVALID_PARAMETER = "COMPILER_INVALID_PARAMETER"
    COMPILER_CONTENT_OVERFLOW = "COMPILER_CONTENT_OVERFLOW"
    COMPILER_SYNTAX_ERROR = "COMPILER_SYNTAX_ERROR"
    COMPILER_OUT_OF_BOUNDS = "COMPILER_OUT_OF_BOUNDS"
    # Milestone 4C.4 Timing & Lifecycle error codes
    COMPILER_TIMELINE_BUDGET_OVERRUN = "COMPILER_TIMELINE_BUDGET_OVERRUN"
    COMPILER_TIMELINE_TARGET_CONFLICT = "COMPILER_TIMELINE_TARGET_CONFLICT"
    COMPILER_TIMELINE_ANCHOR_OVERLAP = "COMPILER_TIMELINE_ANCHOR_OVERLAP"
    TIMELINE_PACING_RUSHED = "TIMELINE_PACING_RUSHED"
    TIMELINE_PACING_UNDERUTILIZED = "TIMELINE_PACING_UNDERUTILIZED"
    TIMELINE_LIFECYCLE_REDUNDANT_ENTER = "TIMELINE_LIFECYCLE_REDUNDANT_ENTER"
    TIMELINE_LIFECYCLE_UNMATCHED_EXIT = "TIMELINE_LIFECYCLE_UNMATCHED_EXIT"
    TIMELINE_LIFECYCLE_DUPLICATE_ENTER = "TIMELINE_LIFECYCLE_DUPLICATE_ENTER"
    TIMELINE_LIFECYCLE_DUPLICATE_EXIT = "TIMELINE_LIFECYCLE_DUPLICATE_EXIT"
    TIMELINE_LIFECYCLE_REDUNDANT_ACTION = "TIMELINE_LIFECYCLE_REDUNDANT_ACTION"
    TIMELINE_TIMESTAMP_NORMALIZED = "TIMELINE_TIMESTAMP_NORMALIZED"
    TIMELINE_BOUNDED_BY_SCENE_END = "TIMELINE_BOUNDED_BY_SCENE_END"
    COMPILER_REDUNDANT_DECLARATION = "COMPILER_REDUNDANT_DECLARATION"
    COMPILER_UNUSED_ENTITY = "COMPILER_UNUSED_ENTITY"


class CompilationDiagnostic(BaseModel):
    """Structured diagnostic reporting a compilation defect, error, or warning."""
    model_config = ConfigDict(extra="forbid")

    error_code: CompilationErrorCode
    message: str
    entity_id: Optional[str] = None
    field_path: str = ""
    severity: str = Field(default="ERROR", description="ERROR or WARNING")
    details: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "entity_id": self.entity_id,
            "field_path": self.field_path,
            "severity": self.severity,
            "details": self.details,
        }


class CompiledSceneOutput(BaseModel):
    """
    Structured result of deterministic scene compilation.
    Contains the verified Manim Python source code, entity-to-variable mappings,
    validation status, and inspectable compiler diagnostics.
    """
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    python_source: str
    is_valid: bool = Field(default=True, description="True if code compiles with zero fatal errors")
    entity_variable_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from entity ID to Python variable name in generated script"
    )
    diagnostics: List[CompilationDiagnostic] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        """Returns True if any ERROR severity diagnostics are present."""
        return any(d.severity == "ERROR" for d in self.diagnostics)

    @property
    def error_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for d in self.diagnostics if d.severity == "WARNING")

    def get_diagnostics_by_code(self, code: CompilationErrorCode) -> List[CompilationDiagnostic]:
        """Returns diagnostics matching a specific error code."""
        return [d for d in self.diagnostics if d.error_code == code]


class ScheduledAction(BaseModel):
    """
    Authoritative normalized execution time representation for a compiled action.
    Constructed exclusively by TimelineScheduler in production; direct construction
    permitted in unit tests with identical strict validation.
    """
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    action_index: int = Field(ge=0)
    target_entity: str = Field(min_length=1)
    action: ActionDeclaration
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    duration: float = Field(ge=0.100)
    is_concurrent: bool = False

    @model_validator(mode="before")
    @classmethod
    def reject_booleans_before_coercion(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for fld in ("action_index", "start_time", "end_time", "duration"):
                if fld in data and isinstance(data[fld], bool):
                    raise ValueError(f"Field '{fld}' must be numeric, not Boolean.")
        return data

    @model_validator(mode="after")
    def verify_action_integrity(self) -> ScheduledAction:
        # 1. Strict Boolean Guards: Prevent bool-to-numeric coercion
        for fld_name, val in [
            ("action_index", self.action_index),
            ("start_time", self.start_time),
            ("end_time", self.end_time),
            ("duration", self.duration),
        ]:
            if isinstance(val, bool):
                raise ValueError(f"Field '{fld_name}' must be numeric, not Boolean.")

        # 2. Strict Integer Guard for action_index
        if not isinstance(self.action_index, int) or self.action_index < 0:
            raise ValueError(f"action_index must be non-negative integer, got {self.action_index}")

        # 3. Finite Numbers Assertion
        for fld_name, fld_val in [
            ("start_time", self.start_time),
            ("end_time", self.end_time),
            ("duration", self.duration),
        ]:
            if not math.isfinite(fld_val):
                raise ValueError(f"Field '{fld_name}' must be finite, got {fld_val}")

        # 4. Target Entity Identity Match
        if self.target_entity != self.action.target_entity:
            raise ValueError(
                f"target_entity '{self.target_entity}' != action.target_entity '{self.action.target_entity}'"
            )

        # 5. Canonical Value Invariants
        if self.start_time != round(self.start_time, 3):
            raise ValueError(f"start_time {self.start_time} is not canonical (must have <=3 decimals)")
        if self.duration != round(self.duration, 3):
            raise ValueError(f"duration {self.duration} is not canonical (must have <=3 decimals)")

        # 6. Strict Canonical Equality for End-Time
        expected_canonical_end = round(self.start_time + self.duration, 3)
        if self.end_time != expected_canonical_end:
            raise ValueError(
                f"end_time {self.end_time} must exactly equal canonicalize_time(start_time + duration) ({expected_canonical_end})"
            )

        return self


class ScheduledStep(BaseModel):
    """A timeline step scheduled against an authoritative audio interval or sequential flow."""
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1)
    step_index: int = Field(ge=0)
    t_start: float = Field(ge=0.0)
    t_end: float = Field(ge=0.0)
    delta_t: float = Field(ge=0.0)
    is_concurrent: bool = False
    actions: List[ScheduledAction] = Field(default_factory=list)
    resting_pause: float = Field(ge=0.0, default=0.0)
    lead_in_pause: float = Field(ge=0.0, default=0.0)
    constituent_step_ids: List[str] = Field(default_factory=list)

    @property
    def active_duration(self) -> float:
        """The total active animation duration of this step."""
        if self.is_concurrent:
            return max((a.duration for a in self.actions), default=0.0)
        return round(sum(a.duration for a in self.actions), 3)


class ScheduledTimeline(BaseModel):
    """The complete scheduled timeline for a scene."""
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    total_duration: float = Field(ge=0.0)
    is_valid: bool = True
    steps: List[ScheduledStep] = Field(default_factory=list)
    actions: List[ScheduledAction] = Field(default_factory=list)
    diagnostics: List[CompilationDiagnostic] = Field(default_factory=list)

    def get_step_for_event(self, event_step_id: str) -> Optional[ScheduledStep]:
        """Resolves any constituent event step ID to its parent ScheduledStep."""
        for step in self.steps:
            if event_step_id == step.step_id or event_step_id in step.constituent_step_ids:
                return step
        return None

