"""
Generic Timing Contract for CineLearn (Audio Integration and Temporal Synchronization).
========================================================================================

Establishes a reusable, domain-agnostic timing contract connecting:
- Storyboard sections and teaching beats.
- Narration segments (spoken lines / explanations).
- Associated visual actions and component animations.
- Actual measured TTS audio durations.
- Actual rendered animation durations.
- Intentional pauses and synchronization markers.

Enforces deterministic temporal alignment across all educational subjects
(OS, DSA, Mathematics, Chemistry, Physics, etc.) without domain coupling.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union


@dataclass
class VisualActionTiming:
    """Represents the timing and identity of a single visual action within a narration segment."""
    action_id: str                          # e.g., "L1-1", "L1-2", "act_01"
    effect: str = ""                        # e.g., "FadeIn", "animate_enqueue", "Indicate"
    component_name: str = ""                # e.g., "ProcessCardVisual", "GraphVisual", "MathTex"
    canonical_action: str = ""              # e.g., "animate_enqueue", "construct"
    estimated_duration: float = 1.0         # Estimated animation run time in seconds
    actual_duration: Optional[float] = None # Measured duration from execution or code AST
    start_offset: float = 0.0               # Offset from segment start in seconds
    relative_order: int = 0                 # Ordering within the segment (0, 1, 2...)
    viewport: str = "dock_main"             # Viewport where action renders
    parameters: Dict[str, Any] = field(default_factory=dict)
    raw_step: str = ""                      # Raw storyboard animation string if available
    action_type: str = "transition"         # "transition", "creation", "emphasis", "hold"
    min_duration: float = 0.5               # Minimum duration clamp in seconds
    max_duration: float = 4.0               # Maximum duration clamp in seconds
    allow_retiming: bool = True             # Whether this action can be safely retimed
    pacing_weight: float = 1.0              # Relative weight for retiming duration allocation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "effect": self.effect,
            "component_name": self.component_name,
            "canonical_action": self.canonical_action,
            "estimated_duration": round(self.estimated_duration, 3),
            "actual_duration": round(self.actual_duration, 3) if self.actual_duration is not None else None,
            "start_offset": round(self.start_offset, 3),
            "relative_order": self.relative_order,
            "viewport": self.viewport,
            "parameters": self.parameters,
            "raw_step": self.raw_step,
            "action_type": self.action_type,
            "min_duration": round(self.min_duration, 3),
            "max_duration": round(self.max_duration, 3),
            "allow_retiming": self.allow_retiming,
            "pacing_weight": round(self.pacing_weight, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VisualActionTiming:
        return cls(
            action_id=str(data.get("action_id", "")),
            effect=str(data.get("effect", "")),
            component_name=str(data.get("component_name", "")),
            canonical_action=str(data.get("canonical_action", "")),
            estimated_duration=float(data.get("estimated_duration", 1.0)),
            actual_duration=float(data["actual_duration"]) if data.get("actual_duration") is not None else None,
            start_offset=float(data.get("start_offset", 0.0)),
            relative_order=int(data.get("relative_order", 0)),
            viewport=str(data.get("viewport", "dock_main")),
            parameters=dict(data.get("parameters") or {}),
            raw_step=str(data.get("raw_step", "")),
            action_type=str(data.get("action_type", "transition")),
            min_duration=float(data.get("min_duration", 0.5)),
            max_duration=float(data.get("max_duration", 4.0)),
            allow_retiming=bool(data.get("allow_retiming", True)),
            pacing_weight=float(data.get("pacing_weight", 1.0)),
        )


@dataclass
class NarrationSegmentTiming:
    """
    Represents a discrete narration segment and its linked visual actions.
    Supports 1:1, 1:many, and many:1 relationships between narration and visuals.
    """
    segment_id: str                         # e.g., "sec1_seg1" or "L1"
    section_id: str                         # e.g., "section_1"
    segment_index: int                      # 0-indexed line position
    narration_text: str                     # Spoken script line
    caption: str = ""                       # Concise on-screen subtitle / phrase
    teaching_beat: str = ""                 # Pedagogical purpose of this explanation
    associated_action_ids: List[str] = field(default_factory=list)
    visual_actions: List[VisualActionTiming] = field(default_factory=list)
    estimated_duration: float = 4.0         # Heuristic duration (e.g. ~150 wpm)
    actual_audio_duration: Optional[float] = None # Exact measured duration from decoded TTS audio
    rendered_visual_duration: Optional[float] = None # Actual duration occupied in rendered video
    pause_duration: float = 0.5             # Intentional pause / hold after this segment (seconds)
    sync_mode: str = "hold"                 # "hold" (pad visual), "extend_pause" (pad audio silence)
    audio_file_path: Optional[str] = None   # Path to individual line audio file
    timeline_start: float = 0.0             # Start timestamp in final assembled timeline (seconds)
    timeline_end: float = 0.0               # End timestamp in final assembled timeline (seconds)
    target_slot_duration: float = 0.0       # Effective slot duration = max(audio_dur, visual_dur) + pause
    pacing_preference: str = "balanced"     # "balanced", "visual_lead", "steady_hold", "inspection_emphasis"
    inspection_pause: float = 1.0           # Desired post-animation inspection pause (seconds)
    natural_pause_type: str = "sentence"    # "clause", "sentence", "transition", "inspection"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "section_id": self.section_id,
            "segment_index": self.segment_index,
            "narration_text": self.narration_text,
            "caption": self.caption,
            "teaching_beat": self.teaching_beat,
            "associated_action_ids": list(self.associated_action_ids),
            "visual_actions": [a.to_dict() for a in self.visual_actions],
            "estimated_duration": round(self.estimated_duration, 3),
            "actual_audio_duration": round(self.actual_audio_duration, 3) if self.actual_audio_duration is not None else None,
            "rendered_visual_duration": round(self.rendered_visual_duration, 3) if self.rendered_visual_duration is not None else None,
            "pause_duration": round(self.pause_duration, 3),
            "sync_mode": self.sync_mode,
            "audio_file_path": self.audio_file_path,
            "timeline_start": round(self.timeline_start, 3),
            "timeline_end": round(self.timeline_end, 3),
            "target_slot_duration": round(self.target_slot_duration, 3),
            "pacing_preference": self.pacing_preference,
            "inspection_pause": round(self.inspection_pause, 3),
            "natural_pause_type": self.natural_pause_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NarrationSegmentTiming:
        actions = [
            VisualActionTiming.from_dict(a) if isinstance(a, dict) else a
            for a in (data.get("visual_actions") or [])
        ]
        return cls(
            segment_id=str(data.get("segment_id", "")),
            section_id=str(data.get("section_id", "")),
            segment_index=int(data.get("segment_index", 0)),
            narration_text=str(data.get("narration_text", "")),
            caption=str(data.get("caption", "")),
            teaching_beat=str(data.get("teaching_beat", "")),
            associated_action_ids=list(data.get("associated_action_ids") or []),
            visual_actions=actions,
            estimated_duration=float(data.get("estimated_duration", 4.0)),
            actual_audio_duration=float(data["actual_audio_duration"]) if data.get("actual_audio_duration") is not None else None,
            rendered_visual_duration=float(data["rendered_visual_duration"]) if data.get("rendered_visual_duration") is not None else None,
            pause_duration=float(data.get("pause_duration", 0.5)),
            sync_mode=str(data.get("sync_mode", "hold")),
            audio_file_path=data.get("audio_file_path"),
            timeline_start=float(data.get("timeline_start", 0.0)),
            timeline_end=float(data.get("timeline_end", 0.0)),
            target_slot_duration=float(data.get("target_slot_duration", 0.0)),
            pacing_preference=str(data.get("pacing_preference", "balanced")),
            inspection_pause=float(data.get("inspection_pause", 1.0)),
            natural_pause_type=str(data.get("natural_pause_type", "sentence")),
        )


class AlignmentStatus(str, Enum):
    """Lifecycle synchronization status for scene audio and visual timeline."""
    UNALIGNED = "unaligned"
    RETIMED = "retimed"
    MUXED = "muxed"


@dataclass
class SceneTimingContract:
    """
    Complete temporal contract for a section / scene.
    Coordinates narration audio, rendered visual animations, pauses, and timestamps.
    """
    section_id: str
    title: str = ""
    story_beat: str = ""
    segments: List[NarrationSegmentTiming] = field(default_factory=list)
    total_audio_duration: float = 0.0
    total_rendered_video_duration: float = 0.0
    total_assembled_duration: float = 0.0
    is_synchronized: bool = False
    alignment_status: AlignmentStatus = AlignmentStatus.UNALIGNED
    narration_hash: str = ""
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes contract to dictionary.
        Maintains complete backward compatibility with existing Cinelearn
        section timestamp schema (`lines`, `concise_captions`, `total_duration`, etc.)
        while including full timing contract details.
        """
        legacy_lines = []
        concise_captions = []
        animation_types = []

        for seg in self.segments:
            concise_captions.append(seg.caption)
            anim_type = seg.visual_actions[0].effect if seg.visual_actions else "fade"
            animation_types.append(anim_type)

            dur = seg.actual_audio_duration if seg.actual_audio_duration is not None else seg.estimated_duration
            legacy_lines.append({
                "section_id": self.section_id,
                "line_index": seg.segment_index,
                "raw_text": seg.narration_text,
                "caption": seg.caption,
                "animation_type": anim_type,
                "start": round(seg.timeline_start, 3),
                "end": round(seg.timeline_end, 3),
                "duration": round(dur, 3),
                "total_budget": round(seg.target_slot_duration, 3),
                "associated_actions": list(seg.associated_action_ids),
                "rendered_visual_duration": round(seg.rendered_visual_duration, 3) if seg.rendered_visual_duration is not None else None,
            })

        return {
            "section_id": self.section_id,
            "title": self.title,
            "story_beat": self.story_beat,
            "audio_path": self.audio_path,
            "video_path": self.video_path,
            "total_duration": round(self.total_assembled_duration or self.total_audio_duration, 3),
            "total_audio_duration": round(self.total_audio_duration, 3),
            "total_rendered_video_duration": round(self.total_rendered_video_duration, 3),
            "total_assembled_duration": round(self.total_assembled_duration, 3),
            "is_synchronized": self.is_synchronized,
            "alignment_status": self.alignment_status.value if isinstance(self.alignment_status, AlignmentStatus) else str(self.alignment_status),
            "narration_hash": self.narration_hash,
            "lines": legacy_lines,
            "concise_captions": concise_captions,
            "animation_types": animation_types,
            "segments": [s.to_dict() for s in self.segments],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SceneTimingContract:
        """Constructs a SceneTimingContract from a serialized dictionary."""
        section_id = str(data.get("section_id", "section_1"))
        title = str(data.get("title", ""))
        story_beat = str(data.get("story_beat", ""))
        audio_path = data.get("audio_path")
        video_path = data.get("video_path")
        narration_hash = str(data.get("narration_hash", ""))
        total_audio = float(data.get("total_audio_duration", data.get("total_duration", 0.0)))
        total_video = float(data.get("total_rendered_video_duration", 0.0))
        total_assembled = float(data.get("total_assembled_duration", data.get("total_duration", 0.0)))
        is_synchronized = bool(data.get("is_synchronized", False))
        align_raw = data.get("alignment_status", AlignmentStatus.UNALIGNED.value)
        try:
            alignment_status = AlignmentStatus(align_raw)
        except (ValueError, TypeError):
            alignment_status = AlignmentStatus.UNALIGNED
        metadata = dict(data.get("metadata") or {})

        segments: List[NarrationSegmentTiming] = []
        if data.get("segments"):
            for sdata in data["segments"]:
                segments.append(NarrationSegmentTiming.from_dict(sdata))
        elif data.get("lines"):
            # Reconstruct from legacy lines format
            captions = data.get("concise_captions") or []
            for i, line_item in enumerate(data["lines"]):
                cap = line_item.get("caption") or (captions[i] if i < len(captions) else "")
                dur = float(line_item.get("duration", 4.0))
                budget = float(line_item.get("total_budget", dur + 0.5))
                pause = max(0.0, budget - dur)
                seg = NarrationSegmentTiming(
                    segment_id=f"{section_id}_seg_{i+1}",
                    section_id=section_id,
                    segment_index=i,
                    narration_text=line_item.get("raw_text") or line_item.get("text", ""),
                    caption=cap,
                    associated_action_ids=list(line_item.get("associated_actions") or [f"L{i+1}-1"]),
                    actual_audio_duration=dur,
                    rendered_visual_duration=line_item.get("rendered_visual_duration"),
                    pause_duration=pause,
                    timeline_start=float(line_item.get("start", 0.0)),
                    timeline_end=float(line_item.get("end", dur)),
                    target_slot_duration=budget,
                )
                segments.append(seg)

        return cls(
            section_id=section_id,
            title=title,
            story_beat=story_beat,
            segments=segments,
            total_audio_duration=total_audio,
            total_rendered_video_duration=total_video,
            total_assembled_duration=total_assembled,
            is_synchronized=is_synchronized,
            alignment_status=alignment_status,
            narration_hash=narration_hash,
            audio_path=audio_path,
            video_path=video_path,
            metadata=metadata,
        )

    @classmethod
    def from_scene_ir(cls, scene_ir: Any) -> SceneTimingContract:
        """
        Builds a generic SceneTimingContract directly from a typed SceneIR object.

        Accepts native SceneIR directly without using SectionProxy, SceneIRToStoryboardAdapter,
        or legacy storyboard dictionaries.

        Invariants enforced:
        - Preserves exact scene identity (scene_id) and timeline event ordering.
        - Maps visual actions to their declaring viewport and component type.
        - Clearly distinguishes known/measured audio durations from estimated or unavailable values:
          `actual_audio_duration` is None unless `scene_ir.audio_reference` is populated.
        - Never invents measured audio durations or treats LLM estimates as authoritative.
        """
        from src.ir.models import SceneIR

        if isinstance(scene_ir, dict):
            scene_ir = SceneIR.from_dict(scene_ir)
        elif not isinstance(scene_ir, SceneIR):
            raise TypeError(f"Expected SceneIR or dict, got {type(scene_ir).__name__}")

        sec_id = scene_ir.scene_id
        title = scene_ir.title or ""
        story_beat = ""
        if isinstance(scene_ir.metadata, dict):
            story_beat = str(scene_ir.metadata.get("story_beat") or scene_ir.metadata.get("story_step") or "")

        # Map entity declarations by ID -> (component_type, viewport)
        entity_info_map: Dict[str, Tuple[str, str]] = {}
        for ent in scene_ir.entities:
            entity_info_map[ent.id] = (ent.component_type, ent.viewport)

        # Authoritative audio presence: ONLY consider audio measured if audio_reference is present
        has_measured_audio = scene_ir.audio_reference is not None and scene_ir.audio_reference.total_duration > 0.0
        audio_ref_path = scene_ir.audio_reference.audio_path if scene_ir.audio_reference else None
        total_audio_dur = scene_ir.audio_reference.total_duration if scene_ir.audio_reference else 0.0

        segments: List[NarrationSegmentTiming] = []
        cues_for_hash: List[str] = []

        for ev_idx, event in enumerate(scene_ir.timeline):
            anchor = event.narration_anchor
            step_id = event.step_id

            if anchor and anchor.text_cue:
                narration_text = anchor.text_cue
                if narration_text.strip():
                    cues_for_hash.append(narration_text)
                line_idx = anchor.lecture_line_index
            else:
                narration_text = ""
                line_idx = ev_idx

            # Actions in this step
            visual_actions: List[VisualActionTiming] = []
            action_ids: List[str] = []
            curr_offset = 0.0

            for a_idx, action in enumerate(event.actions):
                act_id = f"{step_id}_act_{a_idx+1}"
                action_ids.append(act_id)
                comp_type, vport = entity_info_map.get(action.target_entity, ("ProceduralComponent", "dock_main"))
                effect = action.operation or action.type.value
                act_dur = float(action.duration) if action.duration is not None else 1.0

                vat = VisualActionTiming(
                    action_id=act_id,
                    effect=effect,
                    component_name=comp_type,
                    canonical_action=action.operation or action.type.value,
                    estimated_duration=round(act_dur, 3),
                    actual_duration=None,  # Not measured until rendered
                    start_offset=round(curr_offset, 3),
                    relative_order=a_idx,
                    viewport=vport or "dock_main",
                    parameters=dict(action.parameters or {}),
                    raw_step=f"{step_id} | {action.type.value} | {action.target_entity}",
                    action_type=action.type.value,
                )
                visual_actions.append(vat)
                curr_offset += act_dur

            # Timings
            if anchor and anchor.audio_timestamp_range:
                t_start = round(float(anchor.audio_timestamp_range[0]), 3)
                t_end = round(float(anchor.audio_timestamp_range[1]), 3)
                slot_budget = round(max(0.0, t_end - t_start), 3)
            else:
                t_start = 0.0
                t_end = round(curr_offset, 3)
                slot_budget = t_end

            # Audio duration: ONLY authoritative if audio_reference is present
            if has_measured_audio and anchor and anchor.audio_timestamp_range:
                actual_audio_dur: Optional[float] = slot_budget
            else:
                actual_audio_dur = None

            total_act_dur = round(curr_offset, 3)
            est_dur = slot_budget if slot_budget > 0 else (total_act_dur or 1.0)
            target_slot = max(est_dur, total_act_dur)
            pause_dur = max(0.0, round(target_slot - (actual_audio_dur if actual_audio_dur is not None else est_dur), 3))

            seg = NarrationSegmentTiming(
                segment_id=f"{sec_id}_{step_id}",
                section_id=sec_id,
                segment_index=line_idx,
                narration_text=narration_text,
                caption="",
                teaching_beat=step_id,
                associated_action_ids=action_ids if action_ids else [f"{step_id}_default"],
                visual_actions=visual_actions,
                estimated_duration=est_dur,
                actual_audio_duration=actual_audio_dur,
                rendered_visual_duration=None,
                pause_duration=pause_dur,
                timeline_start=t_start,
                timeline_end=t_end,
                target_slot_duration=target_slot,
            )
            segments.append(seg)

        clean_lines = [c.strip() for c in cues_for_hash if c.strip()]
        text_full = "\n".join(clean_lines)
        narr_hash = hashlib.sha256(text_full.encode("utf-8")).hexdigest()[:16]

        alignment_status = AlignmentStatus.UNALIGNED
        if scene_ir.audio_reference and getattr(scene_ir.audio_reference, "alignment_status", None):
            try:
                alignment_status = AlignmentStatus(scene_ir.audio_reference.alignment_status)
            except (ValueError, TypeError):
                alignment_status = AlignmentStatus.UNALIGNED

        return cls(
            section_id=sec_id,
            title=title,
            story_beat=story_beat,
            segments=segments,
            total_audio_duration=round(total_audio_dur, 3),
            total_rendered_video_duration=0.0,
            total_assembled_duration=round(scene_ir.duration_budget, 3),
            is_synchronized=has_measured_audio,
            alignment_status=alignment_status,
            narration_hash=narr_hash,
            audio_path=audio_ref_path,
            video_path=None,
            metadata={"scene_ir_version": scene_ir.version, "duration_budget": scene_ir.duration_budget},
        )

    @classmethod
    def from_storyboard_section(cls, section: Any) -> SceneTimingContract:
        """
        Builds a generic SceneTimingContract from a storyboard Section object or dictionary.
        Parses `animations` into visual actions and maps them to narration lines (L1, L2, etc.).
        """
        if isinstance(section, dict):
            sec_id = str(section.get("id", "section_1"))
            title = str(section.get("title", ""))
            lecture_lines = [str(l).strip() for l in (section.get("lecture_lines") or [])]
            captions = [str(c).strip() for c in (section.get("captions") or [])]
            animations = [str(a).strip() for a in (section.get("animations") or [])]
            story_beat = str(section.get("story_beat") or section.get("story_step", ""))
        else:
            sec_id = getattr(section, "id", "section_1")
            title = getattr(section, "title", "")
            lecture_lines = [str(l).strip() for l in (getattr(section, "lecture_lines", None) or [])]
            captions = [str(c).strip() for c in (getattr(section, "captions", None) or [])]
            animations = [str(a).strip() for a in (getattr(section, "animations", None) or [])]
            story_beat = getattr(section, "story_beat", None) or getattr(section, "story_step", "") or ""

        # Compute narration hash
        text_full = "\n".join(lecture_lines)
        narr_hash = hashlib.sha256(text_full.encode("utf-8")).hexdigest()[:16]

        # Parse animation steps and group them by line tag (e.g. L1-1, L1-2 -> line 1)
        # Format: step_tag | effect | component | parameters | viewport | on_screen_after
        line_actions_map: Dict[int, List[VisualActionTiming]] = {i: [] for i in range(len(lecture_lines))}

        for anim_idx, raw_anim in enumerate(animations):
            parts = [p.strip() for p in raw_anim.split("|")]
            step_tag = parts[0] if parts else f"L1-{anim_idx+1}"
            effect = parts[1] if len(parts) > 1 else "FadeIn"
            comp = parts[2] if len(parts) > 2 else "ProceduralComponent"
            viewport = "dock_main"
            canonical_action = ""
            pacing_pref = "balanced"

            for p in parts[3:]:
                if p.startswith("viewport:"):
                    viewport = p.replace("viewport:", "").strip()
                elif p.startswith("action:"):
                    canonical_action = p.replace("action:", "").strip()
                elif p.startswith("pacing:"):
                    pacing_pref = p.replace("pacing:", "").strip()

            # Identify target line from tag (e.g., "L2-1" -> line_idx 1)
            line_idx = 0
            m = re.match(r"L(\d+)", step_tag, flags=re.IGNORECASE)
            if m:
                line_idx = max(0, min(len(lecture_lines) - 1, int(m.group(1)) - 1))
            else:
                # If no L tag, distribute across lines
                line_idx = min(anim_idx, max(0, len(lecture_lines) - 1))

            act_timing = VisualActionTiming(
                action_id=step_tag,
                effect=effect,
                component_name=comp,
                canonical_action=canonical_action,
                estimated_duration=1.2,
                relative_order=len(line_actions_map[line_idx]),
                viewport=viewport,
                parameters={"pacing": pacing_pref} if pacing_pref != "balanced" else {},
                raw_step=raw_anim,
            )
            line_actions_map[line_idx].append(act_timing)

        # Build narration segments
        segments: List[NarrationSegmentTiming] = []
        current_time = 0.0

        for i, line in enumerate(lecture_lines):
            cap = captions[i] if i < len(captions) and captions[i] else f"Line {i+1}"
            actions = line_actions_map.get(i, [])
            action_ids = [a.action_id for a in actions] if actions else [f"L{i+1}-1"]

            # Check if this segment represents an intentional pedagogical inspection hold
            is_intentional_inspection = any(
                "inspect" in str(a.canonical_action).lower()
                or "hold" in str(a.canonical_action).lower()
                or "inspect" in str(a.parameters.get("pacing", "")).lower()
                for a in actions
            )
            seg_pacing_pref = "inspection_emphasis" if is_intentional_inspection else "balanced"

            # Estimate duration based on word count (avg 145 wpm = 2.4 words/sec)
            words = max(1, len(line.split()))
            est_dur = max(2.5, words / 2.4)
            is_last = (i == len(lecture_lines) - 1)
            pause = 1.0 if is_last else 0.5
            budget = est_dur + pause

            seg = NarrationSegmentTiming(
                segment_id=f"{sec_id}_seg_{i+1}",
                section_id=sec_id,
                segment_index=i,
                narration_text=line,
                caption=cap,
                teaching_beat=story_beat if i == 0 else "",
                associated_action_ids=action_ids,
                visual_actions=actions,
                estimated_duration=round(est_dur, 3),
                pause_duration=pause,
                timeline_start=round(current_time, 3),
                timeline_end=round(current_time + est_dur, 3),
                target_slot_duration=round(budget, 3),
                pacing_preference=seg_pacing_pref,
            )
            segments.append(seg)
            current_time += budget

        return cls(
            section_id=sec_id,
            title=title,
            story_beat=story_beat,
            segments=segments,
            total_audio_duration=round(current_time, 3),
            total_assembled_duration=round(current_time, 3),
            narration_hash=narr_hash,
            metadata={"source": "storyboard_section"},
        )

    def update_actual_audio_durations(self, line_durations: List[float], audio_paths: Optional[List[str]] = None) -> None:
        """
        Updates each segment with actual measured audio durations from decoded TTS files.
        Recalculates target slot budgets and preliminary segment start/end timestamps.
        """
        current_time = 0.0
        for i, dur in enumerate(line_durations):
            if i >= len(self.segments):
                break
            seg = self.segments[i]
            seg.actual_audio_duration = float(dur)
            if audio_paths and i < len(audio_paths):
                seg.audio_file_path = audio_paths[i]

            pause = seg.pause_duration
            slot_dur = dur + pause
            seg.timeline_start = current_time
            seg.timeline_end = current_time + dur
            seg.target_slot_duration = slot_dur
            current_time += slot_dur

        self.total_audio_duration = round(current_time, 3)
        self.total_assembled_duration = round(current_time, 3)

    def reconcile_with_rendered_video(
        self,
        actual_video_duration: float,
        rendered_line_timings: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Core synchronization engine reconciling narration and visual animations.

        Handles all six required temporal scenarios:
        A. Narration longer than animation: Visual holds stable state until speech concludes.
        B. Animation longer than narration: Visual completes, audio pauses until next beat.
        C. Multiple actions during one narration segment: Actions execute during speech window.
        D. Multiple narration segments in one scene: Ordered sequentially, aligned 1:1.
        E. Intentional pauses: Preserved at transitions without drift accumulation.
        F. Dynamic scene durations: Uses final rendered durations after repair/re-render.

        Returns a reconciliation report detailing adjustments made.
        """
        self.total_rendered_video_duration = round(actual_video_duration, 3)
        num_segments = len(self.segments)

        if num_segments == 0:
            self.total_assembled_duration = round(actual_video_duration, 3)
            self.is_synchronized = True
            return {"status": "NO_SEGMENTS", "adjustments": []}

        # 1. Determine actual rendered visual durations per line
        visual_starts: List[float] = []
        visual_durations: List[float] = []

        has_explicit_timings = (
            rendered_line_timings is not None
            and len(rendered_line_timings) == num_segments
            and all("start_time" in item for item in rendered_line_timings)
        )

        if has_explicit_timings:
            for i, item in enumerate(rendered_line_timings):
                v_start = float(item["start_time"])
                visual_starts.append(v_start)
                if "duration" in item and float(item["duration"]) > 0:
                    v_dur = float(item["duration"])
                elif i < num_segments - 1:
                    v_dur = max(0.1, float(rendered_line_timings[i + 1]["start_time"]) - v_start)
                else:
                    v_dur = max(0.1, actual_video_duration - v_start)
                visual_durations.append(v_dur)
        else:
            # Proportional allocation fallback based on target budgets
            total_target = sum(s.target_slot_duration for s in self.segments) or 1.0
            cur_v = 0.0
            for seg in self.segments:
                prop = seg.target_slot_duration / total_target
                seg_v_dur = actual_video_duration * prop
                visual_starts.append(cur_v)
                visual_durations.append(seg_v_dur)
                cur_v += seg_v_dur

        # 2. Reconcile each segment's narration with its visual duration
        adjustments = []
        reconciled_audio_timeline: List[Dict[str, float]] = []
        cur_timeline = 0.0

        for i, seg in enumerate(self.segments):
            a_dur = seg.actual_audio_duration if seg.actual_audio_duration is not None else seg.estimated_duration
            v_dur = visual_durations[i]
            v_start = visual_starts[i]
            seg.rendered_visual_duration = round(v_dur, 3)

            # Scenario A vs Scenario B analysis:
            # Effective slot duration must be large enough for BOTH the visual animation and the speech
            effective_slot = max(a_dur, v_dur)
            seg_start = cur_timeline
            speech_end = seg_start + a_dur

            if a_dur > v_dur + 0.1:
                # Scenario A: Narration is longer than animation -> visual holds state
                hold_time = a_dur - v_dur
                seg.sync_mode = "hold"
                adjustments.append({
                    "segment_index": i,
                    "type": "NARRATION_LONGER_THAN_ANIMATION",
                    "narration_sec": round(a_dur, 2),
                    "animation_sec": round(v_dur, 2),
                    "visual_hold_sec": round(hold_time, 2),
                })
            elif v_dur > a_dur + 0.1:
                # Scenario B: Animation is longer than narration -> speech ends, wait for animation
                wait_time = v_dur - a_dur
                seg.sync_mode = "extend_pause"
                adjustments.append({
                    "segment_index": i,
                    "type": "ANIMATION_LONGER_THAN_NARRATION",
                    "narration_sec": round(a_dur, 2),
                    "animation_sec": round(v_dur, 2),
                    "audio_pause_sec": round(wait_time, 2),
                })
            else:
                seg.sync_mode = "balanced"

            # Apply intentional pause at segment boundary
            post_pause = seg.pause_duration
            slot_total = effective_slot + post_pause
            seg_start = max(v_start, cur_timeline)
            speech_end = seg_start + a_dur

            seg.timeline_start = round(seg_start, 3)
            seg.timeline_end = round(speech_end, 3)
            seg.target_slot_duration = round(slot_total, 3)

            reconciled_audio_timeline.append({
                "segment_index": i,
                "audio_start": round(seg_start, 3),
                "audio_end": round(speech_end, 3),
                "audio_duration": round(a_dur, 3),
                "slot_duration": round(slot_total, 3),
                "visual_start": round(v_start, 3),
                "visual_duration": round(v_dur, 3),
            })
            cur_timeline = seg_start + slot_total

        self.total_assembled_duration = round(cur_timeline, 3)
        self.is_synchronized = True

        return {
            "status": "SYNCHRONIZED",
            "section_id": self.section_id,
            "total_audio_sec": round(self.total_audio_duration, 2),
            "total_video_sec": round(actual_video_duration, 2),
            "total_assembled_sec": round(self.total_assembled_duration, 2),
            "adjustments": adjustments,
            "timeline": reconciled_audio_timeline,
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Deterministically validates the internal consistency of the timing contract.
        Returns (is_valid, list_of_issues).
        """
        issues = []
        if not self.section_id:
            issues.append("Missing section_id")
        if not self.segments:
            issues.append("Contract contains 0 narration segments")

        prev_end = -0.001
        for i, seg in enumerate(self.segments):
            if seg.segment_index != i:
                issues.append(f"Segment {i} has mismatched segment_index {seg.segment_index}")
            if not seg.narration_text.strip():
                issues.append(f"Segment {i} has empty narration text")
            if seg.actual_audio_duration is not None and seg.actual_audio_duration <= 0.05:
                issues.append(f"Segment {i} has non-positive actual_audio_duration: {seg.actual_audio_duration}")
            if seg.timeline_start < prev_end - 0.05:
                issues.append(f"Segment {i} start ({seg.timeline_start}s) is earlier than previous end ({prev_end}s)")
            prev_end = seg.timeline_start

        return (len(issues) == 0, issues)

    def audit_pacing(
        self,
        min_anim_speech_ratio: float = 0.25,
        max_hold_sec: float = 5.0,
        min_inspection_sec: float = 0.5,
    ) -> PacingAuditResult:
        """
        Deterministically evaluates pedagogical pacing across the scene.
        Identifies:
        - Rushed animations: meaningful visual action finishes too rapidly compared to speech (< 25% of speech and < 1.0s)
        - Excessive holds: stable visual state lingers statically without activity (> 5.0s)
        - Inspection windows: checks whether learner receives adequate post-action absorption time
        - Duration ratios: computes animation-to-speech ratio per segment
        """
        rushed = []
        excessive_holds = []
        inspection_windows = []
        ratios = []
        warnings = []

        for i, seg in enumerate(self.segments):
            a_dur = seg.actual_audio_duration if seg.actual_audio_duration is not None else seg.estimated_duration
            v_dur = seg.rendered_visual_duration if seg.rendered_visual_duration is not None else 0.0
            ratio = round(v_dur / max(0.1, a_dur), 2)
            ratios.append(ratio)

            # Check rushed animations:
            # If segment has meaningful actions and v_dur is very small relative to substantial speech
            has_meaningful_action = any(
                a.action_type in ("transition", "creation") and a.allow_retiming
                for a in seg.visual_actions
            ) if seg.visual_actions else True

            if has_meaningful_action and a_dur >= 3.0 and v_dur < 1.0 and (v_dur / a_dur) < min_anim_speech_ratio:
                act_str = ", ".join(seg.associated_action_ids) if seg.associated_action_ids else f"L{i+1}-1"
                issue = {
                    "segment_index": i,
                    "segment_id": seg.segment_id,
                    "caption": seg.caption,
                    "associated_actions": list(seg.associated_action_ids),
                    "speech_sec": round(a_dur, 2),
                    "animation_sec": round(v_dur, 2),
                    "ratio": ratio,
                    "reason": f"Segment {i} ('{seg.caption}', actions: {act_str}) is rushed: animation ({v_dur:.2f}s) is too brief for speech ({a_dur:.2f}s, ratio {ratio:.2f} < {min_anim_speech_ratio})"
                }
                rushed.append(issue)
                warnings.append(issue["reason"])

            # Check excessive holds vs intentional inspection windows:
            # If visual was completed early and held statically for > max_hold_sec
            is_intentional_hold = (
                seg.pacing_preference in ("inspection_emphasis", "steady_hold")
                or any("inspect" in str(a.canonical_action).lower() or "hold" in str(a.canonical_action).lower() for a in seg.visual_actions)
            )
            if seg.sync_mode == "hold" and (a_dur - v_dur) > max_hold_sec:
                hold_sec = round(a_dur - v_dur, 2)
                act_str = ", ".join(seg.associated_action_ids) if seg.associated_action_ids else f"L{i+1}-1"
                if is_intentional_hold:
                    # Legitimate pedagogical absorption window, record as intentional
                    inspection_windows.append({
                        "segment_index": i,
                        "segment_id": seg.segment_id,
                        "caption": seg.caption,
                        "associated_actions": list(seg.associated_action_ids),
                        "inspection_sec": hold_sec,
                        "adequate": True,
                        "intentional": True,
                        "reason": f"Segment {i} ('{seg.caption}') has intentional pedagogical hold of {hold_sec:.2f}s (preference: {seg.pacing_preference})"
                    })
                else:
                    issue = {
                        "segment_index": i,
                        "segment_id": seg.segment_id,
                        "caption": seg.caption,
                        "associated_actions": list(seg.associated_action_ids),
                        "hold_sec": hold_sec,
                        "reason": f"Segment {i} ('{seg.caption}', actions: {act_str}) has excessive static hold of {hold_sec:.2f}s (> {max_hold_sec}s)"
                    }
                    excessive_holds.append(issue)
                    warnings.append(issue["reason"])

            # Record inspection window
            post_pause = max(0.0, seg.target_slot_duration - max(a_dur, v_dur))
            if not is_intentional_hold or (a_dur - v_dur) <= max_hold_sec:
                inspection_windows.append({
                    "segment_index": i,
                    "inspection_sec": round(post_pause, 2),
                    "adequate": post_pause >= min_inspection_sec
                })

        is_well_paced = (len(rushed) == 0 and len(excessive_holds) == 0)
        summary = (
            f"Pacing PASSED for {self.section_id} ({len(self.segments)} segments, avg ratio {sum(ratios)/max(1, len(ratios)):.2f})"
            if is_well_paced
            else f"Pacing WARNINGS for {self.section_id}: {len(rushed)} rushed, {len(excessive_holds)} excessive holds"
        )

        return PacingAuditResult(
            is_well_paced=is_well_paced,
            section_id=self.section_id,
            total_audio_sec=round(self.total_audio_duration, 2),
            total_video_sec=round(self.total_rendered_video_duration, 2),
            animation_to_speech_ratios=ratios,
            excessive_holds=excessive_holds,
            rushed_animations=rushed,
            inspection_windows=inspection_windows,
            pacing_warnings=warnings,
            summary=summary,
        )


@dataclass
class PacingAuditResult:
    """Audit result evaluating the pedagogical pacing and timing health of a scene."""
    is_well_paced: bool
    section_id: str
    total_audio_sec: float
    total_video_sec: float
    animation_to_speech_ratios: List[float] = field(default_factory=list)
    excessive_holds: List[Dict[str, Any]] = field(default_factory=list)
    rushed_animations: List[Dict[str, Any]] = field(default_factory=list)
    inspection_windows: List[Dict[str, Any]] = field(default_factory=list)
    pacing_warnings: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_well_paced": self.is_well_paced,
            "section_id": self.section_id,
            "total_audio_sec": round(self.total_audio_sec, 2),
            "total_video_sec": round(self.total_video_sec, 2),
            "animation_to_speech_ratios": [round(r, 2) for r in self.animation_to_speech_ratios],
            "excessive_holds": self.excessive_holds,
            "rushed_animations": self.rushed_animations,
            "inspection_windows": self.inspection_windows,
            "pacing_warnings": self.pacing_warnings,
            "summary": self.summary,
        }


class AnimationRetimer:
    """
    Dynamic Animation Retimer for CineLearn Manim Scenes.
    Dynamically adjusts animation run_times and wait intervals in generated scene code
    to improve pedagogical pacing and eliminate abrupt freezes while preserving syntax validity.
    """

    # Transient animation calls that should remain quick / snappy (not stretched)
    TRANSIENT_CALLS = {
        "activate_caption", "lecture", "set_color", "set_opacity",
        "Indicate", "Circumscribe", "Flash", "ShowPassingFlash"
    }

    # Generic Manim animation primitives (domain-agnostic defaults)
    GENERIC_MAJOR_CALLS = {"Create", "Write", "ReplacementTransform"}
    GENERIC_STANDARD_CALLS = {
        "Transform", "DrawBorderThenFill", "GrowFromCenter", "SpinInFromNothing",
        "FadeIn", "MoveToTarget"
    }
    GENERIC_MINOR_CALLS: Set[str] = set()

    @classmethod
    def get_registered_action_weights(cls) -> Dict[str, float]:
        """Queries registered domain/component action weights from VisualRegistry."""
        try:
            try:
                from domains import VisualRegistry
            except ImportError:
                from src.domains import VisualRegistry
            if not VisualRegistry._registry:
                VisualRegistry.discover_domains()
            return VisualRegistry.get_action_weights()
        except Exception:
            return {}

    # Backward compatibility aliases
    MAJOR_TRANSITION_CALLS = GENERIC_MAJOR_CALLS
    STANDARD_TRANSFORMATION_CALLS = GENERIC_STANDARD_CALLS
    MINOR_AUXILIARY_CALLS = GENERIC_MINOR_CALLS

    @classmethod
    def find_method_calls(cls, text: str, method_name: str) -> List[Tuple[int, int, str]]:
        """
        Scans text for `self.<method_name>(...)` calls using balanced parentheses tracking.
        Correctly respects single/double/triple quotes and python '#' comments.
        Returns a list of (start_idx, end_idx, inner_arguments_str).
        """
        prefix = f"self.{method_name}("
        idx = 0
        calls = []
        n = len(text)
        while True:
            pos = text.find(prefix, idx)
            if pos == -1:
                break
            arg_start = pos + len(prefix)
            depth = 1
            i = arg_start
            in_quote = None
            in_comment = False
            while i < n and depth > 0:
                ch = text[i]
                if in_comment:
                    if ch == '\n':
                        in_comment = False
                elif in_quote:
                    if ch == '\\':
                        i += 2
                        continue
                    if in_quote in ('"""', "'''"):
                        if text[i:i+3] == in_quote:
                            in_quote = None
                            i += 2
                    elif ch == in_quote:
                        in_quote = None
                else:
                    if ch == '#':
                        in_comment = True
                    elif text[i:i+3] in ('"""', "'''"):
                        in_quote = text[i:i+3]
                        i += 2
                    elif ch in ('"', "'"):
                        in_quote = ch
                    elif ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                        if depth == 0:
                            calls.append((pos, i + 1, text[arg_start:i]))
                            break
                i += 1
            idx = (i + 1) if depth == 0 else (pos + len(prefix))
        return calls

    @classmethod
    def get_action_weight(cls, call_arg_str: str) -> float:
        """
        Computes pedagogical importance weight for an animation call.
        Consumes semantic weights registered in domain/component metadata,
        falling back to generic Manim action properties.
        """
        # 1. Query registered domain / component metadata weights
        reg_weights = cls.get_registered_action_weights()
        for action_name, weight in reg_weights.items():
            if action_name in call_arg_str:
                return float(weight)

        # 2. Check generic standard Manim primitives
        if any(m in call_arg_str for m in cls.GENERIC_MAJOR_CALLS):
            return 1.5
        if any(s in call_arg_str for s in cls.GENERIC_STANDARD_CALLS):
            return 1.0
        if any(a in call_arg_str for a in cls.GENERIC_MINOR_CALLS):
            return 0.6
        return 1.0

    @classmethod
    def is_transient_call(cls, play_arg: str) -> bool:
        """Determines if a play argument is a transient focus/caption animation."""
        if not any(t in play_arg for t in cls.TRANSIENT_CALLS):
            return False
        # If play_arg matches any domain action weight or generic meaningful action, not transient
        reg_weights = cls.get_registered_action_weights()
        if any(act in play_arg for act in reg_weights):
            return False
        if any(m in play_arg for m in cls.GENERIC_MAJOR_CALLS | cls.GENERIC_STANDARD_CALLS):
            return False
        return True

    @classmethod
    def retime_scene_code(
        cls,
        code_str: str,
        contract: SceneTimingContract,
        min_anim_dur: float = 0.8,
        max_anim_dur: float = 3.5,
        target_anim_ratio: float = 0.50,
        min_wait_sec: float = 0.6,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retimes a Manim scene class script using timing contract segment budgets.
        Adjusts run_time and wait statements in each `# === Animation for Lecture Line N ===` block.
        Ensures strict AST syntax validity, falling back safely to original code if modified syntax fails.
        """
        if not code_str or not contract or not contract.segments:
            return code_str, []

        pattern = r"(#\s*===+\s*Animation for Lecture Line\s*(\d+)\s*===+)"
        splits = re.split(pattern, code_str, flags=re.IGNORECASE)
        if len(splits) < 4:
            return code_str, []

        reconstructed = [splits[0]]
        adjustments = []

        i = 1
        while i < len(splits) - 2:
            header = splits[i]
            line_str = splits[i + 1]
            content = splits[i + 2]
            try:
                line_idx = int(line_str) - 1
            except ValueError:
                line_idx = -1

            if 0 <= line_idx < len(contract.segments):
                seg = contract.segments[line_idx]
                speech_dur = seg.actual_audio_duration if seg.actual_audio_duration is not None else seg.estimated_duration
                slot_budget = seg.target_slot_duration if seg.target_slot_duration > 0 else (speech_dur + 0.5)

                mod_content, adj = cls._retime_block(
                    content=content,
                    line_idx=line_idx,
                    speech_dur=speech_dur,
                    slot_budget=slot_budget,
                    min_anim_dur=min_anim_dur,
                    max_anim_dur=max_anim_dur,
                    target_anim_ratio=target_anim_ratio,
                    min_wait_sec=min_wait_sec,
                )
                reconstructed.append(header)
                reconstructed.append(mod_content)
                if adj:
                    adjustments.append(adj)
            else:
                reconstructed.append(header)
                reconstructed.append(content)
            i += 3

        retimed_code = "".join(reconstructed)

        import textwrap
        try:
            ast.parse(retimed_code)
            return retimed_code, adjustments
        except IndentationError:
            try:
                ast.parse(textwrap.dedent(retimed_code))
                return retimed_code, adjustments
            except Exception as e:
                return code_str, [{"error": f"AST SyntaxError during retiming: {e}", "fallback": True}]
        except SyntaxError as e:
            return code_str, [{"error": f"AST SyntaxError during retiming: {e}", "fallback": True}]

    @classmethod
    def _retime_block(
        cls,
        content: str,
        line_idx: int,
        speech_dur: float,
        slot_budget: float,
        min_anim_dur: float,
        max_anim_dur: float,
        target_anim_ratio: float,
        min_wait_sec: float,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Retimes an individual lecture line animation block with pedagogical weighting and wait preservation."""
        play_calls = cls.find_method_calls(content, "play")
        if not play_calls:
            return content, None

        # Filter transient calls (captions, highlights) vs meaningful animations
        meaningful_calls = []
        for c in play_calls:
            arg_str = c[2]
            if not cls.is_transient_call(arg_str):
                meaningful_calls.append(c)

        if not meaningful_calls:
            meaningful_calls = [c for c in play_calls if "activate_caption" not in c[2]]
        if not meaningful_calls:
            meaningful_calls = play_calls

        # Compute pedagogical weights: major transitions receive higher share
        weights = [cls.get_action_weight(c[2]) for c in meaningful_calls]
        sum_w = sum(weights) or 1.0

        total_anim_budget = min(max_anim_dur, max(min_anim_dur, speech_dur * target_anim_ratio))
        durations = []
        for w in weights:
            raw_d = total_anim_budget * (w / sum_w)
            d = max(0.4, min(max_anim_dur, round(raw_d, 2)))
            durations.append(d)

        # Apply retimed run_time in reverse order to preserve string positions
        mod_content = content
        for idx_call in reversed(range(len(meaningful_calls))):
            start, end, inner_args = meaningful_calls[idx_call]
            dur = durations[idx_call]
            if re.search(r"\brun_time\s*=\s*[0-9.]+", inner_args):
                new_inner = re.sub(r"\brun_time\s*=\s*[0-9.]+", f"run_time={dur:.2f}", inner_args)
            else:
                clean_inner = inner_args.rstrip()
                comment_suffix = ""
                m_cmt = re.search(r"(#[^\n]*)$", clean_inner)
                if m_cmt:
                    comment_suffix = " " + m_cmt.group(1)
                    clean_inner = clean_inner[:m_cmt.start()].rstrip()
                clean_inner = clean_inner.rstrip(",")
                new_inner = clean_inner + f", run_time={dur:.2f}" + comment_suffix
            new_call = f"self.play({new_inner})"
            mod_content = mod_content[:start] + new_call + mod_content[end:]

        # Handle self.wait(...) calls: preserve intentional mid-block pauses, update only the final wait
        wait_calls = cls.find_method_calls(mod_content, "wait")
        actual_anim_dur_sum = sum(durations)
        mid_waits_total = 0.0

        if wait_calls:
            # All waits except the last are intentional mid-block inspection pauses
            for w in wait_calls[:-1]:
                m = re.match(r"^\s*([0-9.]+)\s*$", w[2])
                if m:
                    mid_waits_total += float(m.group(1))
                else:
                    mid_waits_total += 0.5

            last_start, last_end, last_inner = wait_calls[-1]
            clock_wait_pat = r"max\(\s*[0-9.]+\s*,\s*([0-9.]+)\s*-\s*\((.*?)\)\s*\)"
            if re.search(clock_wait_pat, last_inner):
                new_last_inner = re.sub(
                    clock_wait_pat,
                    rf"max(0.1, {slot_budget:.2f} - (\2))",
                    last_inner
                )
                mod_content = mod_content[:last_start] + f"self.wait({new_last_inner})" + mod_content[last_end:]
                target_wait = max(min_wait_sec, round(slot_budget - (actual_anim_dur_sum + mid_waits_total), 2))
            else:
                remaining_wait = slot_budget - (actual_anim_dur_sum + mid_waits_total)
                target_wait = max(min_wait_sec, round(remaining_wait, 2))
                mod_content = mod_content[:last_start] + f"self.wait({target_wait:.2f})" + mod_content[last_end:]
        else:
            remaining_wait = slot_budget - actual_anim_dur_sum
            target_wait = max(min_wait_sec, round(remaining_wait, 2))
            mod_content = mod_content.rstrip() + f"\n        self.wait({target_wait:.2f})\n"

        adj = {
            "line_index": line_idx,
            "speech_sec": round(speech_dur, 2),
            "slot_budget_sec": round(slot_budget, 2),
            "retimed_anim_sec": round(actual_anim_dur_sum, 2),
            "retimed_wait_sec": round(target_wait, 2),
            "meaningful_actions_count": len(meaningful_calls),
        }
        return mod_content, adj
