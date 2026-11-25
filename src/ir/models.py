"""
CineLearn — Typed Scene Intermediate Representation (Scene IR) Data Models.
===========================================================================

Defines the universal, domain-independent Scene IR schema (v1.0) conforming to
docs/architecture/scene_ir_and_spatial_composition_specification.md.

Establishes typed models for:
- SceneIR (Section Scene root) & SceneSequenceIR (Multi-section sequence)
- ViewportDeclaration, LayoutConfiguration, AudioReference, InheritedEntityReference
- EntityDeclaration, VisualTier, EntityStyling, VisualContent variants
- ActionDeclaration, ActionType, EasingType, NarrationAnchor, TimelineEvent
"""

from __future__ import annotations

from enum import Enum
import json
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================================
# Enums
# ============================================================================

class ViewportType(str, Enum):
    """Standard anchor viewports supported on the 16:9 canvas."""
    MAIN = "viewport_main"
    SPLIT_LEFT = "viewport_split_left"
    SPLIT_RIGHT = "viewport_split_right"
    INSPECTOR = "viewport_inspector"
    FULL = "viewport_full"


class FlowDirection(str, Enum):
    """Layout flow axis inside a viewport flex container."""
    COLUMN = "column"
    ROW = "row"
    SINGLE = "single"
    ABSOLUTE_GRID = "absolute_grid"


class Alignment(str, Enum):
    """Alignment along the viewport flow axis."""
    CENTER = "center"
    START = "start"
    END = "end"
    SPACE_BETWEEN = "space_between"


class VisualTier(str, Enum):
    """4-Tier visual capability hierarchy."""
    TIER1_REGISTERED = "tier1_registered"
    TIER2_COMPOSITION = "tier2_composition"
    TIER3_PRIMITIVE_TREE = "tier3_primitive_tree"
    TIER4_CONCEPTUAL_CARD = "tier4_conceptual_card"


class ActionType(str, Enum):
    """Declarative visual action types."""
    ENTER = "enter"
    EXIT = "exit"
    TRANSFORM = "transform"
    MOVE = "move"
    HIGHLIGHT = "highlight"
    COMPONENT_OP = "component_op"


class EasingType(str, Enum):
    """Animation pacing and easing functions."""
    LINEAR = "linear"
    SMOOTH = "smooth"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"


# ============================================================================
# Layout & Viewport Models
# ============================================================================

class ViewportDeclaration(BaseModel):
    """Declaration of a bounded layout viewport on the canvas."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: ViewportType
    flow: FlowDirection = FlowDirection.SINGLE
    alignment: Alignment = Alignment.CENTER
    padding: float = Field(default=0.20, ge=0.0)
    sub_slots: List[str] = Field(default_factory=list)

    @field_validator("padding")
    @classmethod
    def validate_padding(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Viewport padding must be non-negative.")
        return round(float(v), 3)


class LayoutConfiguration(BaseModel):
    """Canvas layout options and safe-zone reservations."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    aspect_ratio: str = "16:9"
    has_title_banner: bool = True
    has_caption_column: bool = True
    has_subtitles: bool = True
    safety_padding: float = Field(default=0.20, ge=0.0)

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        v_str = str(v).strip()
        if v_str not in ("16:9", "4:3", "9:16", "1:1"):
            raise ValueError(f"Unsupported aspect_ratio '{v_str}'. Expected 16:9, 4:3, 9:16, or 1:1.")
        return v_str


# ============================================================================
# Audio & State Continuity Models
# ============================================================================

class AudioReference(BaseModel):
    """Pre-generated TTS audio file and word-level timestamps reference."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    audio_path: str = Field(min_length=1)
    timestamps_path: str = Field(min_length=1)
    total_duration: float = Field(gt=0.0)
    alignment_status: Optional[str] = Field(default="unaligned")

    @field_validator("audio_path", "timestamps_path")
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Path must be a string.")
        s = v.strip()
        if not s:
            raise ValueError("Path cannot be empty or whitespace only.")
        return s

    @field_validator("total_duration", mode="before")
    @classmethod
    def validate_duration(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("total_duration cannot be boolean.")
        if not isinstance(v, (int, float)):
            raise ValueError("total_duration must be numeric.")
        if math.isnan(float(v)) or math.isinf(float(v)):
            raise ValueError("total_duration must be a finite number.")
        if float(v) <= 0.0:
            raise ValueError("total_duration must be strictly positive.")
        return round(float(v), 3)

    @field_validator("alignment_status")
    @classmethod
    def validate_alignment_status(cls, v: Optional[str]) -> str:
        if v is None:
            return "unaligned"
        if not isinstance(v, str):
            raise ValueError("alignment_status must be a string.")
        s = v.strip().lower()
        if s not in {"unaligned", "retimed", "muxed"}:
            raise ValueError(f"Invalid alignment_status '{v}'. Must be one of: 'unaligned', 'retimed', 'muxed'.")
        return s


class InheritedEntityReference(BaseModel):
    """Cross-section continuity reference to an entity from a prior section."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source_section: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    carry_over_state: str = "terminal_state"


# ============================================================================
# Entity Styling & Typed Visual Content Payloads
# ============================================================================

class EntityStyling(BaseModel):
    """Declarative styling properties for visual entities."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    color: Optional[str] = None
    fill_opacity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    stroke_width: Optional[float] = Field(default=None, ge=0.0)
    stroke_color: Optional[str] = None
    font_size: Optional[int] = Field(default=None, gt=0)


class ConceptualCardContent(BaseModel):
    """Typed payload for Tier 4 Conceptual Cards (definitions, formulas, badges)."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    title: Optional[str] = None
    formula: Optional[str] = None
    body: Optional[str] = None
    badge: Optional[str] = None
    subtext: Optional[str] = None


class PrimitiveShapeContent(BaseModel):
    """Typed payload for native geometric primitives (rectangles, circles, axes)."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    shape_type: Optional[str] = None
    width: Optional[float] = None
    height: Optional[float] = None
    radius: Optional[float] = None
    label: Optional[str] = None


class CompositeGraphContent(BaseModel):
    """Typed payload for composite node-graphs, arrays, and relational structures."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    elements: List[Dict[str, Any]] = Field(default_factory=list)
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================================
# Visual Entity Declaration
# ============================================================================

class EntityDeclaration(BaseModel):
    """
    Universal declarative visual entity in the scene.
    
    Can represent:
    - Tier 1: Registered component (e.g. ReactionEnergyDiagram, FunctionCurve)
    - Tier 2: Multi-component composition
    - Tier 3: Primitive tree (e.g. CompositeNodeGraph, PointerIndicator)
    - Tier 4: Conceptual card (e.g. ConceptualCard)
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    component_type: str = Field(min_length=1)
    visual_tier: VisualTier
    viewport: str = Field(min_length=1)
    slot: Optional[str] = None
    parent_id: Optional[str] = None
    z_index: int = 10
    visual_priority: int = Field(default=1, ge=1, le=3)
    initial_state: Dict[str, Any] = Field(default_factory=dict)
    styling: Optional[EntityStyling] = None

    @field_validator("id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        v_str = str(v).strip()
        if not v_str:
            raise ValueError("Entity id cannot be empty.")
        return v_str


# ============================================================================
# Timeline & Action Models
# ============================================================================

class NarrationAnchor(BaseModel):
    """Binding of a visual timeline step to a spoken narration segment."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    lecture_line_index: int = Field(ge=0)
    text_cue: str = Field(default="")
    audio_timestamp_range: Tuple[float, float]

    @field_validator("lecture_line_index", mode="before")
    @classmethod
    def validate_line_index(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("lecture_line_index cannot be boolean.")
        if not isinstance(v, int):
            raise ValueError("lecture_line_index must be an integer.")
        if v < 0:
            raise ValueError("lecture_line_index must be non-negative.")
        return v

    @field_validator("audio_timestamp_range", mode="before")
    @classmethod
    def validate_timestamp_range(cls, v: Any) -> Tuple[float, float]:
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            raise ValueError("audio_timestamp_range must contain exactly [start_time, end_time].")
        t0, t1 = v[0], v[1]
        if isinstance(t0, bool) or isinstance(t1, bool):
            raise ValueError("Audio timestamps cannot be booleans.")
        if not isinstance(t0, (int, float)) or not isinstance(t1, (int, float)):
            raise ValueError("Audio timestamps must be numeric.")
        if math.isnan(float(t0)) or math.isnan(float(t1)) or math.isinf(float(t0)) or math.isinf(float(t1)):
            raise ValueError("Audio timestamps must be finite.")
        t_start, t_end = float(t0), float(t1)
        if t_start < 0.0 or t_end < 0.0:
            raise ValueError("Audio timestamps must be non-negative.")
        if t_start > t_end:
            raise ValueError(f"Inverted audio timestamp range: start {t_start} > end {t_end}.")
        return (round(t_start, 3), round(t_end, 3))


class ActionDeclaration(BaseModel):
    """A discrete visual animation or state mutation applied to a target entity."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: ActionType
    target_entity: str = Field(min_length=1)
    operation: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    duration: float = Field(default=1.0, ge=0.0)
    easing: EasingType = EasingType.SMOOTH
    post_action_pause: float = Field(default=0.0, ge=0.0)

    @field_validator("duration", "post_action_pause")
    @classmethod
    def validate_non_negative_time(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Duration and pause values must be non-negative.")
        return round(float(v), 3)


class TimelineEvent(BaseModel):
    """An event in the scene timeline anchored to narration with ordered visual actions."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    step_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    narration_anchor: Optional[NarrationAnchor] = None
    actions: List[ActionDeclaration] = Field(default_factory=list)


# ============================================================================
# Scene IR Root & Multi-Section Models
# ============================================================================

SUPPORTED_SCENE_IR_VERSIONS = {"1.0", "1", "scene_ir_v1"}

class SceneIR(BaseModel):
    """
    Universal, typed root for a Section Scene Intermediate Representation (v1.0).
    
    Conforms strictly to docs/architecture/scene_ir_and_spatial_composition_specification.md.
    Provides complete declarative specification for layout solving, timeline scheduling,
    and deterministic Manim code compilation.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_url: Optional[str] = Field(
        default="https://cinelearn.org/schemas/scene_ir_v1.json",
        alias="$schema",
    )
    version: str = Field(default="1.0")
    scene_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    section_index: int = Field(default=1, ge=0)
    title: str = Field(default="")
    duration_budget: float = Field(
        gt=0.0,
        description=(
            "Total duration budget for this scene section IN SECONDS (e.g. 30.0 to 120.0 seconds). "
            "Must be greater than or equal to the total scheduled duration of all timeline actions and narration anchors."
        ),
    )
    audio_reference: Optional[AudioReference] = None
    layout_configuration: LayoutConfiguration = Field(default_factory=LayoutConfiguration)
    viewports: Dict[str, ViewportDeclaration] = Field(default_factory=dict)
    inherited_entities: List[InheritedEntityReference] = Field(default_factory=list)
    entities: List[EntityDeclaration] = Field(default_factory=list)
    timeline: List[TimelineEvent] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        v_str = str(v).strip().lower()
        if v_str not in SUPPORTED_SCENE_IR_VERSIONS:
            raise ValueError(
                f"Unsupported Scene IR schema version '{v}'. Supported versions: {sorted(SUPPORTED_SCENE_IR_VERSIONS)}"
            )
        return v_str

    @field_validator("duration_budget", mode="before")
    @classmethod
    def validate_duration_budget(cls, v: Any) -> float:
        if isinstance(v, bool):
            raise ValueError("duration_budget cannot be boolean.")
        if not isinstance(v, (int, float)):
            raise ValueError("duration_budget must be numeric.")
        if math.isnan(float(v)) or math.isinf(float(v)):
            raise ValueError("duration_budget must be a finite number.")
        if float(v) <= 0.0:
            raise ValueError("duration_budget must be strictly positive.")
        return round(float(v), 3)

    # ------------------------------------------------------------------------
    # Narration Extraction Helpers
    # ------------------------------------------------------------------------

    def get_ordered_narration_cues(self) -> List[str]:
        """
        Extract valid, non-empty narration cues ordered deterministically by
        narration_anchor.lecture_line_index (with stable secondary sort by timeline event position).
        
        Preserves exact cue text without normalization.
        Events without narration anchors or with empty/whitespace-only cues are omitted.
        Duplicate lecture_line_indices are resolved stably by event order.
        """
        extracted = []
        for idx, event in enumerate(self.timeline):
            if event.narration_anchor and event.narration_anchor.text_cue:
                cue = event.narration_anchor.text_cue
                if cue.strip():
                    extracted.append((event.narration_anchor.lecture_line_index, idx, cue))

        extracted.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in extracted]

    @property
    def narration_cues(self) -> List[str]:
        """Convenience property for ordered narration cues."""
        return self.get_ordered_narration_cues()

    # ------------------------------------------------------------------------
    # Serialization and Convenience Methods
    # ------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Lossless dictionary serialization using model aliases ($schema)."""
        return self.model_dump(by_alias=True, exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        """Deterministic, formatted JSON export with stable key ordering."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneIR":
        """Reconstruct SceneIR from dictionary with strict Pydantic validation."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "SceneIR":
        """Reconstruct SceneIR from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class SceneSequenceIR(BaseModel):
    """
    Root document representing an entire multi-section video presentation.
    Wraps ordered SceneIR sections and presentation-level metadata.
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version: str = Field(default="1.0")
    sequence_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    title: str = Field(default="")
    topic: str = Field(default="")
    sections: List[SceneIR] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        v_str = str(v).strip().lower()
        if v_str not in SUPPORTED_SCENE_IR_VERSIONS:
            raise ValueError(f"Unsupported sequence version '{v}'.")
        return v_str

    @property
    def total_scenes(self) -> int:
        """Total number of scenes in this sequence."""
        return len(self.sections)

    @property
    def total_duration(self) -> float:
        """Sum of duration budgets across all scenes in this sequence."""
        return round(sum(s.duration_budget for s in self.sections), 3)

    def get_ordered_narration_cues(self) -> List[str]:
        """
        Extract all ordered narration cues across all sections in presentation sequence order.
        """
        all_cues = []
        for section in self.sections:
            all_cues.extend(section.get_ordered_narration_cues())
        return all_cues

    @property
    def narration_cues(self) -> List[str]:
        """Convenience property for sequence-wide ordered narration cues."""
        return self.get_ordered_narration_cues()

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneSequenceIR":
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> "SceneSequenceIR":
        return cls.from_dict(json.loads(json_str))
