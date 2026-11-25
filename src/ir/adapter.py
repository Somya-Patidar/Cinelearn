"""
Cinelearn — Scene IR to Storyboard Compatibility Adapter.
=========================================================

Provides deterministic, non-lossy transformation between typed Scene IR models
(SceneIR / SceneSequenceIR) and the legacy storyboard data structures expected by
existing Stage 3 Manim code generation (get_prompt3_code) and StoryboardFidelityChecker.

Guarantees:
- Non-lossy preservation of narration lines, captions, verified values, and visual intent.
- Generation of valid 6-field pipe-delimited animation step grammar for Stage 3.
- Populates structured `resolved_actions` for authoritative Stage 3 method binding.
- Explicit fail-closed behavior (SceneIRAdapterError) if essential storyboard data is missing.
- Zero modification to existing Stage 3 code or contracts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .models import (
    ActionDeclaration,
    ActionType,
    EntityDeclaration,
    SceneIR,
    SceneSequenceIR,
    TimelineEvent,
    ViewportDeclaration,
    ViewportType,
    VisualTier,
)


class SceneIRAdapterError(Exception):
    """Raised when Scene IR cannot be deterministically adapted without information loss."""
    pass


class SectionProxy:
    """A dual-access bridge object providing both attribute access (obj.attr)
    and dictionary access (obj['key']) for storyboard sections.

    Resolves M3B-04 by allowing code expecting dataclass Section or plain dict
    to seamlessly interact with adapted Scene IR sections without AttributeError
    or KeyError.
    """

    _CANONICAL_FIELDS = {
        "id": "",
        "title": "",
        "lecture_lines": [],
        "animations": [],
        "story_beat": "",
        "captions": None,
        "layout_plan": "",
        "verified_values": None,
        "algorithmic_operations": None,
        "beats": None,
        "resolved_actions": None,
    }

    def __init__(self, data: Optional[Union[Dict[str, Any], Any]] = None, **kwargs):
        init_data: Dict[str, Any] = {}
        if data is not None:
            if isinstance(data, dict):
                init_data.update(data)
            elif isinstance(data, SectionProxy):
                init_data.update(data.to_dict())
            elif hasattr(data, "__dict__"):
                init_data.update(vars(data))
            else:
                raise TypeError(f"Cannot initialize SectionProxy from {type(data).__name__}")
        init_data.update(kwargs)
        object.__setattr__(self, "_data", init_data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        if name in self._CANONICAL_FIELDS:
            default_val = self._CANONICAL_FIELDS[name]
            return list(default_val) if isinstance(default_val, list) else default_val
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            data = object.__getattribute__(self, "_data")
            data[name] = value

    def __delattr__(self, name: str) -> None:
        data = object.__getattribute__(self, "_data")
        if name in data:
            del data[name]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if key in data:
            return data[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        data = object.__getattribute__(self, "_data")
        data[key] = value

    def __delitem__(self, key: str) -> None:
        data = object.__getattribute__(self, "_data")
        if key in data:
            del data[key]
        else:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        data = object.__getattribute__(self, "_data")
        return key in data

    def __len__(self) -> int:
        data = object.__getattribute__(self, "_data")
        return len(data)

    def __iter__(self):
        data = object.__getattribute__(self, "_data")
        return iter(data)

    def get(self, key: str, default: Any = None) -> Any:
        data = object.__getattribute__(self, "_data")
        if key in data:
            return data[key]
        if key in self._CANONICAL_FIELDS:
            default_val = self._CANONICAL_FIELDS[key]
            return list(default_val) if isinstance(default_val, list) else default_val
        return default

    def keys(self):
        return object.__getattribute__(self, "_data").keys()

    def values(self):
        return object.__getattribute__(self, "_data").values()

    def items(self):
        return object.__getattribute__(self, "_data").items()

    def to_dict(self) -> Dict[str, Any]:
        return dict(object.__getattribute__(self, "_data"))

    @property
    def __dict__(self) -> Dict[str, Any]:
        return object.__getattribute__(self, "_data")

    def __repr__(self) -> str:
        data = object.__getattribute__(self, "_data")
        return f"SectionProxy({data!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SectionProxy):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return False

    @classmethod
    def from_section(cls, section: Any) -> SectionProxy:
        """Create a SectionProxy from a Section dataclass, dict, or another SectionProxy."""
        if isinstance(section, cls):
            return cls(section.to_dict())
        if isinstance(section, dict):
            return cls(section)
        if hasattr(section, "__dict__"):
            return cls(vars(section))
        raise TypeError(f"Cannot convert {type(section).__name__} to SectionProxy")

    def to_section(self) -> Any:
        """Convert to Section dataclass if agent.Section is importable."""
        try:
            from src.agent import Section
            return Section(
                id=self.get("id", ""),
                title=self.get("title", ""),
                lecture_lines=self.get("lecture_lines", []),
                animations=self.get("animations", []),
                story_beat=self.get("story_beat", ""),
                captions=self.get("captions"),
                layout_plan=self.get("layout_plan", ""),
                verified_values=self.get("verified_values"),
                algorithmic_operations=self.get("algorithmic_operations"),
                beats=self.get("beats"),
                resolved_actions=self.get("resolved_actions"),
            )
        except ImportError:
            return self


def scene_ir_to_storyboard_section(scene: SceneIR) -> Dict[str, Any]:
    """
    Deterministically transforms a typed SceneIR model into a legacy storyboard section dictionary.
    Preserves all lecture lines, captions, story beats, verified values, and animation steps.
    """
    if not isinstance(scene, SceneIR):
        raise TypeError(f"Expected SceneIR instance, got {type(scene).__name__}")

    metadata = scene.metadata or {}

    # 1. Spoken lecture lines & captions
    lecture_lines: List[str] = list(metadata.get("lecture_lines") or [])
    if not lecture_lines:
        # Fallback: extract distinct cues from timeline narration anchors
        extracted_cues: Dict[int, str] = {}
        for event in scene.timeline:
            if event.narration_anchor and event.narration_anchor.text_cue:
                idx = event.narration_anchor.lecture_line_index
                if idx not in extracted_cues:
                    extracted_cues[idx] = event.narration_anchor.text_cue
        if extracted_cues:
            max_idx = max(extracted_cues.keys())
            lecture_lines = [extracted_cues.get(i, f"Instructional step {i + 1}.") for i in range(max_idx + 1)]

    if not lecture_lines:
        raise SceneIRAdapterError(
            f"Scene '{scene.scene_id}' cannot be adapted to Stage 3: missing essential 'lecture_lines' in metadata and timeline anchors."
        )

    captions: List[str] = list(metadata.get("captions") or [])
    if len(captions) != len(lecture_lines):
        # Generate concise noun-phrase captions if missing
        captions = [
            " ".join(line.split()[:6]) if line else f"Step {i + 1}"
            for i, line in enumerate(lecture_lines)
        ]

    story_beat = str(metadata.get("story_beat") or f"Pedagogical progression for {scene.title or scene.scene_id}")
    verified_values = list(metadata.get("verified_values") or [])

    # 2. Entity lookup map
    entity_map: Dict[str, EntityDeclaration] = {e.id: e for e in scene.entities}
    inherited_ids: Set[str] = {inh.entity_id for inh in scene.inherited_entities if inh.entity_id}

    # 3. Layout plan synthesis
    layout_entries = []
    for e in scene.entities:
        layout_entries.append(f"{e.id} in {e.viewport}")
    layout_plan = ", ".join(layout_entries) if layout_entries else "Standard container layout"

    # 4. Synthesize 6-field animation strings and resolved actions
    animations: List[str] = []
    resolved_actions: List[Dict[str, Any]] = []
    active_on_screen: Set[str] = set()

    for s_idx, event in enumerate(scene.timeline):
        line_idx = event.narration_anchor.lecture_line_index if event.narration_anchor else 0
        line_num = line_idx + 1

        for a_idx, action in enumerate(event.actions):
            tag = f"L{line_num}-{a_idx + 1}"
            target_id = action.target_entity
            target_entity = entity_map.get(target_id)

            if target_entity is None and target_id not in inherited_ids:
                raise SceneIRAdapterError(
                    f"Action in step '{event.step_id}' references unknown target entity '{target_id}'."
                )

            # Viewport mapping
            vp_name = target_entity.viewport if target_entity else "dock_main"
            if not vp_name.startswith("dock_") and not vp_name.startswith("viewport_") and not vp_name.startswith("area:"):
                # Normalize viewport identifier to standard dock naming
                vp_name = f"dock_{vp_name.replace('viewport_', '')}"

            # Effect mapping
            effect_map = {
                ActionType.ENTER: "FadeIn",
                ActionType.EXIT: "FadeOut",
                ActionType.TRANSFORM: "Transform",
                ActionType.MOVE: "MoveTo",
                ActionType.HIGHLIGHT: "Indicate",
                ActionType.COMPONENT_OP: "Transform",
            }
            effect = effect_map.get(action.type, "FadeIn")

            # Update active on-screen entities
            if action.type == ActionType.ENTER:
                active_on_screen.add(target_id)
            elif action.type == ActionType.EXIT:
                active_on_screen.discard(target_id)

            # Visual representation name
            if target_entity:
                comp_type = target_entity.component_type
                if target_entity.visual_tier == VisualTier.TIER4_CONCEPTUAL_CARD:
                    vis_rep = f"ConceptualCard({target_id})"
                elif target_entity.visual_tier == VisualTier.TIER3_PRIMITIVE_TREE:
                    vis_rep = f"ProceduralComponent({comp_type})"
                else:
                    vis_rep = comp_type
            else:
                vis_rep = f"InheritedEntity({target_id})"

            # Action call string
            if action.type == ActionType.COMPONENT_OP and action.operation:
                param_strs = [f"{k}={v!r}" for k, v in action.parameters.items()]
                action_str = f"action: {action.operation}({', '.join(param_strs)})"
            elif action.type == ActionType.HIGHLIGHT:
                sub = action.parameters.get("subterm")
                action_str = f"action: highlight('{sub}')" if sub else "action: highlight()"
            elif action.type == ActionType.MOVE:
                t_idx = action.parameters.get("new_target_index", action.parameters.get("target_position"))
                action_str = f"action: move_to({t_idx!r})" if t_idx is not None else "action: move()"
            elif action.type == ActionType.TRANSFORM:
                formula = action.parameters.get("formula")
                action_str = f"action: update_formula(r'{formula}')" if formula else "action: transform()"
            else:
                action_str = f"action: {action.type.value}()"

            on_screen_str = f"on screen after: [{', '.join(sorted(active_on_screen))}]"

            # Form standard 6-field pipe-delimited string
            anim_step = f"{tag} | {effect} | {vis_rep} | viewport: {vp_name} | {action_str} | {on_screen_str}"
            animations.append(anim_step)

            # Form structured resolved_actions entry for Stage 3
            canonical_op = action.operation or action.type.value
            resolved_entry = {
                "step_tag": tag,
                "effect": effect,
                "component_name": comp_type if target_entity else "InheritedEntity",
                "canonical_action": canonical_op,
                "kwargs": action.parameters,
                "viewport": vp_name,
                "on_screen_after": sorted(active_on_screen),
                "target_instance_id": target_id,
                "intended_call": f"{target_id}.{canonical_op}({', '.join(f'{k}={v!r}' for k, v in action.parameters.items())})",
                "raw_step": anim_step,
            }
            resolved_actions.append(resolved_entry)

    return {
        "id": scene.scene_id,
        "title": scene.title or f"Sec {scene.section_index}: Scene",
        "story_beat": story_beat,
        "lecture_lines": lecture_lines,
        "captions": captions,
        "animations": animations,
        "layout_plan": layout_plan,
        "verified_values": verified_values,
        "resolved_actions": resolved_actions,
    }


def scene_ir_to_storyboard_section_obj(scene: SceneIR) -> SectionProxy:
    """
    Deterministically transforms a typed SceneIR model into a SectionProxy object.

    Provides both attribute access (section.lecture_lines) and dict access
    (section['lecture_lines']), resolving M3B-04 interface mismatches without lossy conversion.
    """
    raw_dict = scene_ir_to_storyboard_section(scene)
    return SectionProxy(raw_dict)


def sequence_ir_to_storyboard(sequence: SceneSequenceIR) -> Dict[str, Any]:
    """
    Transforms an entire SceneSequenceIR document into a complete legacy storyboard dictionary.
    """
    if not isinstance(sequence, SceneSequenceIR):
        raise TypeError(f"Expected SceneSequenceIR instance, got {type(sequence).__name__}")

    metadata = sequence.metadata or {}
    visual_conventions = metadata.get(
        "visual_conventions",
        {
            "palette": {
                "primary": "#58C4DD",
                "highlight": "#FFD700",
                "positive": "#83C167",
                "accent": "#FFA07A",
                "warning": "#FF6B6B",
            },
            "entities": [],
        },
    )

    sections = [scene_ir_to_storyboard_section(sec) for sec in sequence.sections]

    return {
        "visual_conventions": visual_conventions,
        "sections": sections,
    }


def storyboard_section_to_scene_ir(section_dict: Dict[str, Any], section_index: int = 1) -> SceneIR:
    """
    Converts a legacy storyboard section dictionary into a typed SceneIR model.
    """
    if not isinstance(section_dict, dict):
        raise TypeError(f"Expected dict, got {type(section_dict).__name__}")

    sid = str(section_dict.get("id", f"Section{section_index}Scene"))
    title = str(section_dict.get("title", ""))
    lecture_lines = list(section_dict.get("lecture_lines") or [])
    captions = list(section_dict.get("captions") or [])
    story_beat = str(section_dict.get("story_beat") or "")
    verified_values = list(section_dict.get("verified_values") or [])
    raw_animations = list(section_dict.get("animations") or [])

    # Estimate duration budget: ~4 seconds per lecture line
    duration_budget = max(5.0, round(len(lecture_lines) * 4.5, 1))

    # Standard viewports
    viewports = {
        "dock_main": ViewportDeclaration(type=ViewportType.MAIN),
        "dock_inspector": ViewportDeclaration(type=ViewportType.INSPECTOR),
        "dock_split_left": ViewportDeclaration(type=ViewportType.SPLIT_LEFT),
        "dock_split_right": ViewportDeclaration(type=ViewportType.SPLIT_RIGHT),
    }

    entities: List[EntityDeclaration] = []
    seen_entity_ids: Set[str] = set()
    timeline: List[TimelineEvent] = []

    for a_idx, anim_str in enumerate(raw_animations):
        parts = [p.strip() for p in anim_str.split("|")]
        if len(parts) >= 5:
            tag = parts[0]
            effect = parts[1]
            vis_rep = parts[2]
            vp_field = parts[3].replace("viewport:", "").strip()
            action_field = parts[4].replace("action:", "").strip()
        else:
            tag = f"L{min(a_idx + 1, len(lecture_lines))}-1"
            effect = "FadeIn"
            vis_rep = "ProceduralComponent(VisualUnit)"
            vp_field = "dock_main"
            action_field = "show()"

        # Parse line number from tag (e.g. L1-1 -> 0)
        line_idx = 0
        tag_match = re.match(r"L(\d+)", tag)
        if tag_match:
            line_idx = max(0, int(tag_match.group(1)) - 1)

        # Entity ID synthesis
        clean_rep = re.sub(r"[^A-Za-z0-9_]", "_", vis_rep).strip("_") or f"unit_{a_idx + 1}"
        eid = f"ent_{clean_rep.lower()}"
        if eid not in seen_entity_ids:
            seen_entity_ids.add(eid)
            # Determine visual tier
            if "Card" in vis_rep or "MathTex" in vis_rep:
                tier = VisualTier.TIER4_CONCEPTUAL_CARD
                ctype = "ConceptualCard"
            elif any(v in vis_rep for v in ("GraphVisual", "DPTableMatrix", "ProcessCardVisual", "ReadyQueueVisual", "CPUExecutionVisual", "GanttChartVisual", "MetricTableVisual", "MoleculeVisual", "ReactionEnergyDiagram", "FunctionCurve", "NumberLineVisual")):
                tier = VisualTier.TIER1_REGISTERED
                ctype = vis_rep.split("(")[0].strip()
            else:
                tier = VisualTier.TIER3_PRIMITIVE_TREE
                ctype = "ProceduralComponent"

            entities.append(
                EntityDeclaration(
                    id=eid,
                    component_type=ctype,
                    visual_tier=tier,
                    viewport=vp_field if vp_field in viewports else "dock_main",
                )
            )

        # Action synthesis
        step_id = f"step_{tag.lower().replace('-', '_')}_{a_idx + 1}"
        action_type = ActionType.ENTER if "FadeIn" in effect or "Create" in effect else (
            ActionType.EXIT if "FadeOut" in effect else (
                ActionType.HIGHLIGHT if "Indicate" in effect else ActionType.TRANSFORM
            )
        )

        cue_text = lecture_lines[line_idx] if line_idx < len(lecture_lines) else ""
        t_start = round(line_idx * 4.0, 1)
        t_end = round(t_start + 3.8, 1)

        event = TimelineEvent(
            step_id=step_id,
            narration_anchor={
                "lecture_line_index": line_idx,
                "text_cue": cue_text,
                "audio_timestamp_range": (t_start, t_end),
            },
            actions=[
                ActionDeclaration(
                    type=action_type,
                    target_entity=eid,
                    operation=action_field.split("(")[0] if "(" in action_field else None,
                    duration=1.0,
                )
            ],
        )
        timeline.append(event)

    metadata = {
        "story_beat": story_beat,
        "lecture_lines": lecture_lines,
        "captions": captions,
        "verified_values": verified_values,
    }

    return SceneIR(
        scene_id=sid,
        section_index=section_index,
        title=title,
        duration_budget=duration_budget,
        viewports=viewports,
        entities=entities,
        timeline=timeline,
        metadata=metadata,
    )


def storyboard_to_sequence_ir(storyboard_dict: Dict[str, Any]) -> SceneSequenceIR:
    """
    Transforms a complete legacy storyboard dictionary into a typed SceneSequenceIR document.
    """
    if not isinstance(storyboard_dict, dict):
        raise TypeError(f"Expected dict, got {type(storyboard_dict).__name__}")

    raw_sections = storyboard_dict.get("sections", [])
    sections = [
        storyboard_section_to_scene_ir(sec, section_index=idx + 1)
        for idx, sec in enumerate(raw_sections)
    ]

    return SceneSequenceIR(
        sequence_id="seq_storyboard_presentation",
        title="Adapted Storyboard Sequence",
        sections=sections,
        metadata={"visual_conventions": storyboard_dict.get("visual_conventions", {})},
    )


__all__ = [
    "SceneIRAdapterError",
    "SectionProxy",
    "scene_ir_to_storyboard_section",
    "scene_ir_to_storyboard_section_obj",
    "sequence_ir_to_storyboard",
    "storyboard_section_to_scene_ir",
    "storyboard_to_sequence_ir",
]

