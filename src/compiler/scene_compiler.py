"""
CineLearn — Milestone 4B: Deterministic Scene Compiler Orchestrator
===================================================================
Translates validated Scene IR documents and computed SceneSpatialLayout
spatial geometry into deterministic, syntactically valid Manim Python scripts
subclassing TeachingScene.

Guarantees:
- Strict geometry consumption directly from ComputedEntityGeometry.world_box.
- Safe parameter serialization and LaTeX escaping (no code injection).
- Deterministic naming and stable AST emission.
- Explicit diagnostics for missing geometry, unknown entities, and deferred features.
"""

from __future__ import annotations

import ast
import keyword
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ir.models import (
    ActionDeclaration,
    ActionType,
    EasingType,
    EntityDeclaration,
    SceneIR,
    VisualTier,
)
from src.compiler.compiler_models import (
    CompilationDiagnostic,
    CompilationErrorCode,
    CompiledSceneOutput,
    ScheduledAction,
    ScheduledStep,
    ScheduledTimeline,
    is_safe_keyword_identifier,
)
from src.compiler.layout_models import (
    VIEWPORT_BOUNDARY_TOLERANCE,
    BoundingBox2D,
    ComputedEntityGeometry,
    LayoutErrorCode,
    SceneSpatialLayout,
)
from src.compiler.primitive_compiler import PrimitiveCompiler
from src.compiler.card_compiler import ConceptualCardCompiler, _safe_color
from src.compiler.timeline_scheduler import TimelineScheduler


def _split_top_level_comma(s: str) -> List[str]:
    """Splits a string by commas, respecting parentheses and brackets."""
    parts = []
    curr = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
            curr.append(ch)
        elif ch in ")]}":
            depth -= 1
            curr.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(curr).strip())
            curr = []
        else:
            curr.append(ch)
    if curr:
        parts.append("".join(curr).strip())
    return [p for p in parts if p]


class DeterministicSceneCompiler:
    """
    Compiler orchestrator for CineLearn Scene IR and Spatial Layout.
    Produces deterministic, standalone Manim Python source code.
    """

    def __init__(
        self,
        primitive_compiler: Optional[PrimitiveCompiler] = None,
        card_compiler: Optional[ConceptualCardCompiler] = None,
        registry_manifest: Optional[Dict[str, Any]] = None,
    ):
        self.primitive_compiler = primitive_compiler or PrimitiveCompiler()
        self.card_compiler = card_compiler or ConceptualCardCompiler()
        self._registry_manifest: Optional[Dict[str, Any]] = registry_manifest

    def _resolve_registered_component_symbol(self, ctype: str) -> Optional[str]:
        """
        Validates component_type against authoritative VisualRegistry manifest
        and returns the trusted Python class name symbol.
        Returns None if component_type is not registered or not a trusted identifier.
        """
        if not ctype or not isinstance(ctype, str):
            return None
        c_clean = ctype.strip()
        if self._registry_manifest is None:
            try:
                from src.domains import VisualRegistry
                VisualRegistry.discover_domains("src.domains")
                self._registry_manifest = VisualRegistry.get_schema_manifest()
            except Exception:
                self._registry_manifest = {}

        for domain, comps in self._registry_manifest.items():
            if isinstance(comps, list):
                for comp in comps:
                    cid = comp.get("component_id")
                    if cid and is_safe_keyword_identifier(cid):
                        if c_clean == cid or c_clean.lower() == cid.lower():
                            return cid
                        aliases = comp.get("aliases", [])
                        if isinstance(aliases, list):
                            for alias in aliases:
                                if isinstance(alias, str) and (c_clean == alias or c_clean.lower() == alias.lower()):
                                    return cid
        return None

    def _allocate_entity_variable_names(self, entities: List[EntityDeclaration]) -> Dict[str, str]:
        """
        Deterministically allocates unique Python variable names for all entities in the scene,
        resolving normalization collisions across different entity IDs.
        """
        allocated: Set[str] = set()
        var_map: Dict[str, str] = {}

        for entity in entities:
            clean = re.sub(r"[^a-zA-Z0-9_]", "_", entity.id.strip()).strip("_")
            clean = re.sub(r"_+", "_", clean)
            if not clean:
                clean = "entity"
            elif clean[0].isdigit():
                clean = f"entity_{clean}"

            base_var = f"ent_{clean}"
            stripped_base = base_var.replace("ent_", "")
            if keyword.iskeyword(stripped_base) or (hasattr(keyword, "issoftkeyword") and keyword.issoftkeyword(stripped_base)):
                base_var = f"{base_var}_var"

            var_name = base_var
            if var_name in allocated:
                counter = 2
                while f"{base_var}_{counter}" in allocated:
                    counter += 1
                var_name = f"{base_var}_{counter}"

            allocated.add(var_name)
            var_map[entity.id] = var_name

        return var_map

    def compile(
        self,
        scene: SceneIR,
        layout: SceneSpatialLayout,
        scheduled_timeline: Optional[ScheduledTimeline] = None,
    ) -> CompiledSceneOutput:
        """
        Compiles a validated SceneIR document and its computed SceneSpatialLayout
        into a complete, valid Manim Python source file.
        """
        diagnostics: List[CompilationDiagnostic] = []

        # -------------------------------------------------------------
        # Phase 1: Contract & Pre-Compilation Validation
        # -------------------------------------------------------------
        # 1. Authoritative Scheduling (Sequence A)
        if scheduled_timeline is None:
            scheduler = TimelineScheduler()
            scheduled_timeline, sched_diags = scheduler.schedule(scene)
            diagnostics.extend(sched_diags)
        else:
            diagnostics.extend(scheduled_timeline.diagnostics)

        # Allocate unique, collision-free variable names for all entities
        entity_var_map = self._allocate_entity_variable_names(scene.entities)

        # Propagate layout diagnostics into compilation diagnostics
        LAYOUT_TO_COMPILER_CODE = {
            LayoutErrorCode.LAYOUT_OVERCROWDING_ERROR: CompilationErrorCode.COMPILER_CONTENT_OVERFLOW,
            LayoutErrorCode.LAYOUT_BOUNDARY_BREACH: CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
            LayoutErrorCode.LAYOUT_PARENT_BOUNDS_EXCEEDED: CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
            LayoutErrorCode.LAYOUT_FORBIDDEN_OVERLAP: CompilationErrorCode.COMPILER_CONTENT_OVERFLOW,
            LayoutErrorCode.LAYOUT_INVALID_VIEWPORT: CompilationErrorCode.COMPILER_INVALID_PARAMETER,
            LayoutErrorCode.LAYOUT_CYCLIC_HIERARCHY: CompilationErrorCode.COMPILER_INVALID_PARAMETER,
            LayoutErrorCode.LAYOUT_UNSATISFIABLE_CONSTRAINTS: CompilationErrorCode.COMPILER_INVALID_PARAMETER,
            LayoutErrorCode.LAYOUT_DIMENSION_UNCERTAINTY: CompilationErrorCode.COMPILER_INVALID_PARAMETER,
        }

        for l_err in layout.errors:
            c_code = LAYOUT_TO_COMPILER_CODE.get(l_err.error_code, CompilationErrorCode.COMPILER_OUT_OF_BOUNDS)
            field_path = (
                f"entities.{l_err.entity_id}"
                if l_err.entity_id
                else (f"viewports.{l_err.viewport_id}" if l_err.viewport_id else "layout")
            )
            details = dict(l_err.details)
            if l_err.viewport_id and "viewport_id" not in details:
                details["viewport_id"] = l_err.viewport_id
            if l_err.related_entities and "related_entities" not in details:
                details["related_entities"] = l_err.related_entities
            diagnostics.append(
                CompilationDiagnostic(
                    error_code=c_code,
                    message=f"[Layout] {l_err.message}",
                    entity_id=l_err.entity_id,
                    field_path=field_path,
                    severity="ERROR",
                    details=details,
                )
            )

        for l_warn in layout.warnings:
            c_code = LAYOUT_TO_COMPILER_CODE.get(l_warn.error_code, CompilationErrorCode.COMPILER_INVALID_PARAMETER)
            field_path = (
                f"entities.{l_warn.entity_id}"
                if l_warn.entity_id
                else (f"viewports.{l_warn.viewport_id}" if l_warn.viewport_id else "layout")
            )
            details = dict(l_warn.details)
            if l_warn.viewport_id and "viewport_id" not in details:
                details["viewport_id"] = l_warn.viewport_id
            diagnostics.append(
                CompilationDiagnostic(
                    error_code=c_code,
                    message=f"[Layout] {l_warn.message}",
                    entity_id=l_warn.entity_id,
                    field_path=field_path,
                    severity="WARNING",
                    details=details,
                )
            )

        # If scheduling had fatal errors or layout is invalid, fail closed immediately
        if not scheduled_timeline.is_valid or not layout.is_valid or any(d.severity == "ERROR" for d in diagnostics):
            return CompiledSceneOutput(
                scene_id=scene.scene_id,
                python_source="",
                is_valid=False,
                entity_variable_map=entity_var_map,
                diagnostics=diagnostics,
                metadata=scene.metadata,
            )

        # Validate that every entity in scene has a computed geometry allocation
        for idx, entity in enumerate(scene.entities):
            geom = layout.entities.get(entity.id)
            if geom is None:
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_MISSING_GEOMETRY,
                        message=f"Entity '{entity.id}' has no computed geometry in SceneSpatialLayout.",
                        entity_id=entity.id,
                        field_path=f"entities[{idx}]",
                        severity="ERROR",
                    )
                )

            # Check for unsupported composite grouping via group_id
            init_state = entity.initial_state or {}
            if "group_id" in init_state:
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_FEATURE,
                        message=(
                            f"Entity '{entity.id}' requested composite grouping via group_id='{init_state['group_id']}'. "
                            f"Composite grouping is explicitly unsupported in Scene IR v1.0."
                        ),
                        entity_id=entity.id,
                        field_path=f"entities[{idx}].initial_state.group_id",
                        severity="ERROR",
                    )
                )

        # -------------------------------------------------------------
        # Phase 2: Entity Code Compilation
        # -------------------------------------------------------------
        entity_code_blocks: List[str] = []
        entered_entities: Set[str] = set()

        # Identify which entities have explicit "enter" actions in the timeline
        for event in scene.timeline:
            for action in event.actions:
                if action.type == ActionType.ENTER:
                    entered_entities.add(action.target_entity)

        # Compile each entity deterministically
        for entity_idx, entity in enumerate(scene.entities):
            var_name = entity_var_map[entity.id]
            geom = layout.entities.get(entity.id)
            if geom is None:
                continue

            code_block, e_diags = self._compile_entity(entity, geom, var_name, entity_idx=entity_idx)
            diagnostics.extend(e_diags)
            entity_code_blocks.append(code_block)

        # -------------------------------------------------------------
        # Phase 3: Timeline & Animation Scheduling Compilation
        # -------------------------------------------------------------
        timeline_code_blocks: List[str] = []
        entity_map = {e.id: e for e in scene.entities}
        entity_state_tracker: Dict[str, Dict[str, Any]] = {
            e.id: dict(e.initial_state or {}) for e in scene.entities
        }
        entity_position_tracker: Dict[str, BoundingBox2D] = {}
        entity_viewport_tracker: Dict[str, str] = {e.id: e.viewport for e in scene.entities}
        for e in scene.entities:
            geom = layout.entities.get(e.id)
            if geom:
                entity_position_tracker[e.id] = geom.world_box

        # Build explicit mapping from constituent event step IDs to parent ScheduledStep
        event_to_step_map: Dict[str, ScheduledStep] = {}
        for s in scheduled_timeline.steps:
            for cid in s.constituent_step_ids:
                event_to_step_map[cid] = s
            event_to_step_map[s.step_id] = s

        # Group timeline events by ScheduledStep
        for sched_step in scheduled_timeline.steps:
            constituent_ids = set(sched_step.constituent_step_ids) or {sched_step.step_id}
            step_events = [ev for ev in scene.timeline if ev.step_id in constituent_ids]
            if not step_events:
                continue

            step_lines: List[str] = []

            # 1. Lead-in pause (emitted ONCE before the group begins)
            if sched_step.lead_in_pause > 0.001:
                step_lines.append(f"self.wait({round(sched_step.lead_in_pause, 3)})")

            # 2. Narration cues and step headers for each constituent event
            for ev in step_events:
                ev_idx = scene.timeline.index(ev) if ev in scene.timeline else 0
                step_lines.append(f"# --- Step {ev_idx + 1}: {ev.step_id} ---")
                if ev.narration_anchor:
                    anchor = ev.narration_anchor
                    cue = anchor.text_cue.replace("\n", " ").strip()
                    t_range = f"[{anchor.audio_timestamp_range[0]:.2f}s - {anchor.audio_timestamp_range[1]:.2f}s]"
                    step_lines.append(f"# Narration Cue: {cue!r} {t_range}")

            # 3. Actions compilation
            actions_in_step: List[Tuple[TimelineEvent, int, ActionDeclaration]] = []
            for ev in step_events:
                for a_idx, action in enumerate(ev.actions):
                    actions_in_step.append((ev, a_idx, action))

            if sched_step.is_concurrent:
                # Concurrent actions: collect setup, animations, post lines
                concurrent_setup: List[str] = []
                concurrent_anims: List[str] = []
                concurrent_post: List[str] = []

                for ev, a_idx, action in actions_in_step:
                    target_id = action.target_entity
                    target_var = entity_var_map.get(target_id)
                    ev_idx = scene.timeline.index(ev) if ev in scene.timeline else 0

                    if not target_var:
                        is_inherited = any(inh.entity_id == target_id for inh in scene.inherited_entities)
                        if is_inherited:
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION,
                                    message=(
                                        f"Timeline action references inherited entity '{target_id}', but cross-section "
                                        f"inherited entity reconstitution is explicitly unsupported in Milestone 4B Phase 1."
                                    ),
                                    entity_id=target_id,
                                    field_path=f"timeline[{ev_idx}].actions[{a_idx}].target_entity",
                                    severity="ERROR",
                                )
                            )
                        else:
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_UNKNOWN_ENTITY,
                                    message=f"Timeline action references undeclared entity '{target_id}'.",
                                    entity_id=target_id,
                                    field_path=f"timeline[{ev_idx}].actions[{a_idx}].target_entity",
                                    severity="ERROR",
                                )
                            )
                        continue

                    setup, anims, post, act_diags = self._compile_action_elements(
                        action=action,
                        target_var=target_var,
                        layout=layout,
                        entity_map=entity_map,
                        entity_var_map=entity_var_map,
                        entity_state_tracker=entity_state_tracker,
                        entity_position_tracker=entity_position_tracker,
                        entity_viewport_tracker=entity_viewport_tracker,
                        entered_entities=entered_entities,
                        step_idx=ev_idx,
                        a_idx=a_idx,
                        is_concurrent=True,
                    )
                    diagnostics.extend(act_diags)
                    concurrent_setup.extend(setup)
                    concurrent_anims.extend(anims)
                    concurrent_post.extend(post)

                step_lines.extend(concurrent_setup)
                if concurrent_anims:
                    step_lines.append(f"self.play({', '.join(concurrent_anims)})")
                elif actions_in_step:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION,
                            message=f"Step '{sched_step.step_id}' concurrent actions produced no valid animation expressions.",
                            field_path=f"step('{sched_step.step_id}')",
                            severity="ERROR",
                        )
                    )
                step_lines.extend(concurrent_post)

            else:
                for ev, a_idx, action in actions_in_step:
                    target_id = action.target_entity
                    target_var = entity_var_map.get(target_id)
                    ev_idx = scene.timeline.index(ev) if ev in scene.timeline else 0

                    if not target_var:
                        is_inherited = any(inh.entity_id == target_id for inh in scene.inherited_entities)
                        if is_inherited:
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION,
                                    message=(
                                        f"Timeline action references inherited entity '{target_id}', but cross-section "
                                        f"inherited entity reconstitution is explicitly unsupported in Milestone 4B Phase 1."
                                    ),
                                    entity_id=target_id,
                                    field_path=f"timeline[{ev_idx}].actions[{a_idx}].target_entity",
                                    severity="ERROR",
                                )
                            )
                        else:
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_UNKNOWN_ENTITY,
                                    message=f"Timeline action references undeclared entity '{target_id}'.",
                                    entity_id=target_id,
                                    field_path=f"timeline[{ev_idx}].actions[{a_idx}].target_entity",
                                    severity="ERROR",
                                )
                            )
                        continue

                    setup, anims, post, act_diags = self._compile_action_elements(
                        action=action,
                        target_var=target_var,
                        layout=layout,
                        entity_map=entity_map,
                        entity_var_map=entity_var_map,
                        entity_state_tracker=entity_state_tracker,
                        entity_position_tracker=entity_position_tracker,
                        entity_viewport_tracker=entity_viewport_tracker,
                        entered_entities=entered_entities,
                        step_idx=ev_idx,
                        a_idx=a_idx,
                        is_concurrent=False,
                    )
                    diagnostics.extend(act_diags)
                    step_lines.extend(setup)
                    for anim in anims:
                        step_lines.append(anim)
                    step_lines.extend(post)
                    if action.post_action_pause > 0.0:
                        step_lines.append(f"self.wait({round(action.post_action_pause, 2)})")

            # 4. Resting pause (emitted ONCE after the group completes)
            if sched_step.resting_pause > 0.001:
                explicit_pauses = sum(
                    float(getattr(a, "post_action_pause", 0.0) or 0.0)
                    for _, _, a in actions_in_step
                    if getattr(a, "post_action_pause", 0.0) > 0.0
                )
                net_resting_pause = round(max(0.0, sched_step.resting_pause - explicit_pauses), 3)
                if net_resting_pause > 0.001:
                    step_lines.append(f"self.wait({net_resting_pause})")

            timeline_code_blocks.append("\n".join(step_lines))

        # -------------------------------------------------------------
        # Phase 4: Full Python Source Emission
        # -------------------------------------------------------------
        python_source = self._emit_python_source(
            scene=scene,
            layout=layout,
            entity_code_blocks=entity_code_blocks,
            timeline_code_blocks=timeline_code_blocks,
            entered_entities=entered_entities,
            entity_var_map=entity_var_map,
        )

        # -------------------------------------------------------------
        # Phase 5: Python AST Verification
        # -------------------------------------------------------------
        is_valid = True
        try:
            ast.parse(python_source)
        except SyntaxError as e:
            is_valid = False
            diagnostics.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_SYNTAX_ERROR,
                    message=f"Generated Manim script failed AST validation: {e.msg} (line {e.lineno}, offset {e.offset})",
                    field_path=f"line:{e.lineno}",
                    severity="ERROR",
                    details={"lineno": e.lineno, "offset": e.offset, "text": e.text},
                )
            )

        # Mark invalid if any ERROR diagnostics exist
        if any(d.severity == "ERROR" for d in diagnostics):
            is_valid = False

        return CompiledSceneOutput(
            scene_id=scene.scene_id,
            python_source=python_source,
            is_valid=is_valid,
            entity_variable_map=entity_var_map,
            diagnostics=diagnostics,
            metadata={
                "title": scene.title,
                "entity_count": len(scene.entities),
                "timeline_steps": len(scene.timeline),
                "duration_budget": scene.duration_budget,
            },
        )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------
    def _generate_var_name(self, entity_id: str) -> str:
        """Produces a deterministic, clean Python variable name from entity_id."""
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", entity_id.strip()).strip("_")
        clean = re.sub(r"_+", "_", clean)
        if not clean:
            clean = "entity"
        elif clean[0].isdigit():
            clean = f"entity_{clean}"

        base_var = f"ent_{clean}"
        stripped = base_var.replace("ent_", "")
        if keyword.iskeyword(stripped) or (hasattr(keyword, "issoftkeyword") and keyword.issoftkeyword(stripped)):
            base_var = f"{base_var}_var"
        return base_var

    def _compile_entity(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
        entity_idx: Optional[int] = None,
    ) -> Tuple[str, List[CompilationDiagnostic]]:
        """Delegates entity compilation to specialized tier compiler."""
        tier = entity.visual_tier

        # Tier 3 Primitives
        if self.primitive_compiler.can_compile(entity):
            return self.primitive_compiler.compile_primitive(entity, geom, var_name)

        # Tier 4 Conceptual Cards
        if self.card_compiler.can_compile(entity):
            return self.card_compiler.compile_card(entity, geom, var_name, entity_idx=entity_idx)

        # Tier 1 Registered Components
        if tier == VisualTier.TIER1_REGISTERED:
            return self._compile_tier1_component(entity, geom, var_name)

        # Fallback unknown component
        return (
            f"# Unrecognized entity: {entity.id} ({entity.component_type})\n"
            f"{var_name} = VGroup()\n"
            f"{var_name}.set_z_index({geom.z_index})"
        ), [
            CompilationDiagnostic(
                error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_COMPONENT,
                message=f"Entity '{entity.id}' references unsupported component type '{entity.component_type}'.",
                entity_id=entity.id,
                field_path="component_type",
                severity="ERROR",
            )
        ]

    def _compile_tier1_component(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
    ) -> Tuple[str, List[CompilationDiagnostic]]:
        """Compiles a registered Tier 1 visual component using its trusted constructor."""
        diagnostics: List[CompilationDiagnostic] = []
        ctype = entity.component_type
        init = entity.initial_state or {}

        # Validate component type against authoritative VisualRegistry manifest
        trusted_symbol = self._resolve_registered_component_symbol(ctype)
        if not trusted_symbol:
            diagnostics.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_COMPONENT,
                    message=(
                        f"Entity '{entity.id}' references Tier 1 component_type '{ctype}', which is not registered "
                        f"in VisualRegistry or is not a trusted identifier."
                    ),
                    entity_id=entity.id,
                    field_path="component_type",
                    severity="ERROR",
                )
            )
            code = (
                f"# Unsupported Tier 1 Component: {var_name}\n"
                f"{var_name} = VGroup()\n"
                f"{var_name}.set_z_index({geom.z_index})"
            )
            return code, diagnostics

        cx = round(geom.world_box.x, 4)
        cy = round(geom.world_box.y, 4)

        # Safely format constructor arguments with strict key validation
        kwarg_strs = []
        has_invalid_param_key = False
        for k, v in sorted(init.items()):
            if not is_safe_keyword_identifier(k):
                has_invalid_param_key = True
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                        message=(
                            f"Parameter key {k!r} for entity '{entity.id}' is invalid. "
                            f"Keyword argument keys must be valid, non-reserved Python identifiers."
                        ),
                        entity_id=entity.id,
                        field_path=f"initial_state.{k}",
                        severity="ERROR",
                    )
                )
            else:
                kwarg_strs.append(f"{k}={v!r}")

        if has_invalid_param_key:
            code = (
                f"# Tier 1 Component with invalid parameters: {var_name}\n"
                f"{var_name} = VGroup()\n"
                f"{var_name}.set_z_index({geom.z_index})"
            )
            return code, diagnostics

        kwargs_formatted = ", ".join(kwarg_strs)

        code = (
            f"# Tier 1 Registered Component: {trusted_symbol}\n"
            f"try:\n"
            f"    {var_name} = {trusted_symbol}({kwargs_formatted})\n"
            f"except Exception:\n"
            f"    {var_name} = ProceduralComponent(name={trusted_symbol!r})\n"
            f"{var_name}.move_to(np.array([{cx}, {cy}, 0.0]))\n"
            f"{var_name}.set_z_index({geom.z_index})"
        )
        return code, diagnostics

    def _compile_action_elements(
        self,
        action: ActionDeclaration,
        target_var: str,
        layout: SceneSpatialLayout,
        entity_map: Optional[Dict[str, EntityDeclaration]] = None,
        entity_var_map: Optional[Dict[str, str]] = None,
        entity_state_tracker: Optional[Dict[str, Dict[str, Any]]] = None,
        entity_position_tracker: Optional[Dict[str, BoundingBox2D]] = None,
        entity_viewport_tracker: Optional[Dict[str, str]] = None,
        entered_entities: Optional[Set[str]] = None,
        step_idx: int = 0,
        a_idx: int = 0,
        is_concurrent: bool = False,
    ) -> Tuple[List[str], List[str], List[str], List[CompilationDiagnostic]]:
        """
        Compiles an ActionDeclaration into structured setup lines, animation expressions,
        post-play update lines, and compilation diagnostics.
        """
        diagnostics: List[CompilationDiagnostic] = []
        setup_lines: List[str] = []
        anim_exprs: List[str] = []
        post_lines: List[str] = []
        dur = round(max(0.1, action.duration), 2)
        params = action.parameters or {}

        if action.type == ActionType.ENTER:
            if is_concurrent:
                anim_exprs.append(f"FadeIn({target_var}, run_time={dur})")
            else:
                anim_exprs.append(f"self.play(FadeIn({target_var}), run_time={dur})")

        elif action.type == ActionType.EXIT:
            if is_concurrent:
                anim_exprs.append(f"FadeOut({target_var}, run_time={dur})")
            else:
                anim_exprs.append(f"self.play(FadeOut({target_var}), run_time={dur})")

        elif action.type == ActionType.HIGHLIGHT:
            color = params.get("color", "#FFD700")
            if is_concurrent:
                anim_exprs.append(f"Indicate({target_var}, color={color!r}, run_time={dur})")
            else:
                anim_exprs.append(f"self.play(Indicate({target_var}, color={color!r}), run_time={dur})")

        elif action.type == ActionType.TRANSFORM:
            # 1. Parameter key safety validation
            has_invalid_param_key = False
            for k in params:
                if not is_safe_keyword_identifier(k):
                    has_invalid_param_key = True
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Action TRANSFORM on entity '{action.target_entity}' contains invalid parameter key '{k}'.",
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.{k}",
                            severity="ERROR",
                        )
                    )
            if has_invalid_param_key:
                return setup_lines, anim_exprs, post_lines, diagnostics

            # Category A2: Entity-to-Entity Structural Replacement
            if "replacement_entity" in params:
                repl_id = str(params["replacement_entity"]).strip()
                repl_var = entity_var_map.get(repl_id) if entity_var_map else None
                if not repl_var:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_UNKNOWN_ENTITY,
                            message=f"Action TRANSFORM on entity '{action.target_entity}' references unknown replacement_entity '{repl_id}'.",
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.replacement_entity",
                            severity="ERROR",
                        )
                    )
                else:
                    target_entity_decl = entity_map.get(action.target_entity) if entity_map else None
                    repl_entity_decl = entity_map.get(repl_id) if entity_map else None
                    is_incompatible = False
                    if target_entity_decl and repl_entity_decl:
                        if target_entity_decl.visual_tier != repl_entity_decl.visual_tier:
                            is_incompatible = True
                        elif target_entity_decl.component_type != repl_entity_decl.component_type:
                            if target_entity_decl.visual_tier == VisualTier.TIER1_REGISTERED:
                                is_incompatible = True

                    transform_mode = params.get("transform_mode")
                    if is_incompatible or transform_mode == "cross_fade":
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION,
                                message=(
                                    f"Transform between incompatible entities '{action.target_entity}' "
                                    f"({getattr(target_entity_decl, 'component_type', 'unknown')}) and '{repl_id}' "
                                    f"({getattr(repl_entity_decl, 'component_type', 'unknown')}). "
                                    f"Falling back to safe cross-fade sequence."
                                ),
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters",
                                severity="WARNING",
                            )
                        )
                        t_fadeout = round(0.4 * dur, 2)
                        t_fadein = round(0.6 * dur, 2)
                        setup_lines.append(f"# Fallback Cross-Fade: {action.target_entity} -> {repl_id}")
                        if is_concurrent:
                            anim_exprs.append(f"FadeOut({target_var}, run_time={t_fadeout})")
                            anim_exprs.append(f"FadeIn({repl_var}, run_time={t_fadein})")
                        else:
                            anim_exprs.append(
                                f"self.play(FadeOut({target_var}, run_time={t_fadeout}), FadeIn({repl_var}, run_time={t_fadein}))"
                            )
                    else:
                        setup_lines.append(f"# Entity-to-Entity Structural Replacement: {action.target_entity} -> {repl_id}")
                        if is_concurrent:
                            anim_exprs.append(f"ReplacementTransform({target_var}, {repl_var}, run_time={dur})")
                        else:
                            anim_exprs.append(f"self.play(ReplacementTransform({target_var}, {repl_var}), run_time={dur})")
                    if entered_entities is not None:
                        entered_entities.add(repl_id)

            else:
                target_entity_decl = entity_map.get(action.target_entity) if entity_map else None
                is_card = bool(
                    target_entity_decl
                    and (
                        target_entity_decl.visual_tier == VisualTier.TIER4_CONCEPTUAL_CARD
                        or self.card_compiler.can_compile(target_entity_decl)
                    )
                )

                if is_card:
                    curr_state = (
                        entity_state_tracker.get(action.target_entity, {})
                        if entity_state_tracker
                        else dict(getattr(target_entity_decl, "initial_state", {}) or {})
                    )
                    geom = layout.entities.get(action.target_entity)
                    if not geom:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_MISSING_GEOMETRY,
                                message=f"Entity '{action.target_entity}' has no computed geometry in layout.",
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].target_entity",
                                severity="ERROR",
                            )
                        )
                    else:
                        card_lines, card_diags, new_state = self.card_compiler.compile_card_transform(
                            entity=target_entity_decl,
                            geom=geom,
                            var_name=target_var,
                            action=action,
                            current_state=curr_state,
                            step_idx=step_idx,
                            a_idx=a_idx,
                        )
                        diagnostics.extend(card_diags)
                        if entity_state_tracker is not None:
                            entity_state_tracker[action.target_entity] = new_state

                        if is_concurrent:
                            in_post = False
                            for line in card_lines:
                                if line.startswith("self.play("):
                                    in_post = True
                                    inner = line[len("self.play("):-1].strip()
                                    if ", run_time=" in inner:
                                        anims_part, _ = inner.rsplit(", run_time=", 1)
                                        anims_part = anims_part.strip()
                                        for pa in _split_top_level_comma(anims_part):
                                            if ".animate." in pa:
                                                pa = pa.replace(".animate.", f".animate(run_time={dur}).", 1)
                                            elif "run_time=" not in pa and pa.endswith(")"):
                                                pa = pa[:-1] + f", run_time={dur})"
                                            anim_exprs.append(pa)
                                    else:
                                        anim_exprs.append(inner)
                                elif not in_post:
                                    setup_lines.append(line)
                                else:
                                    post_lines.append(line)
                        else:
                            anim_exprs.extend(card_lines)

                elif target_entity_decl and target_entity_decl.component_type == "PrimitiveShape":
                    method_calls: List[str] = []
                    applied_names: List[str] = []
                    has_param_error = False

                    # Check for unsupported parameter keys
                    allowed_params = {
                        "color", "fill_color", "stroke_color",
                        "scale",
                        "opacity", "fill_opacity",
                        "duration", "easing", "target_entity",
                    }
                    unsupported_keys = sorted([k for k in params.keys() if k not in allowed_params])
                    if unsupported_keys:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION,
                                message=(
                                    f"Action TRANSFORM on PrimitiveShape '{action.target_entity}' provided unsupported "
                                    f"parameters: {unsupported_keys}. Supported parameters are color, scale, opacity, "
                                    f"or replacement_entity."
                                ),
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters",
                                severity="ERROR",
                                details={"unsupported_keys": unsupported_keys},
                            )
                        )
                        has_param_error = True

                    # 1. Color mutation
                    if "color" in params or "fill_color" in params or "stroke_color" in params:
                        raw_color = params.get("color") or params.get("fill_color") or params.get("stroke_color")
                        if not isinstance(raw_color, str) or not raw_color.strip():
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                                    message=f"Action TRANSFORM on PrimitiveShape '{action.target_entity}' has invalid color: {raw_color!r}.",
                                    entity_id=action.target_entity,
                                    field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.color",
                                    severity="ERROR",
                                )
                            )
                            has_param_error = True
                        else:
                            color_val = _safe_color(raw_color)
                            method_calls.append(f".set_color({color_val!r})")
                            applied_names.append("color")

                    # 2. Scale mutation
                    if "scale" in params:
                        raw_scale = params["scale"]
                        if isinstance(raw_scale, bool) or not isinstance(raw_scale, (int, float)):
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                                    message=f"Action TRANSFORM on PrimitiveShape '{action.target_entity}' has non-numeric scale: {raw_scale!r}.",
                                    entity_id=action.target_entity,
                                    field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.scale",
                                    severity="ERROR",
                                )
                            )
                            has_param_error = True
                        elif math.isnan(raw_scale) or math.isinf(raw_scale) or raw_scale <= 0.0:
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                                    message=f"Action TRANSFORM on PrimitiveShape '{action.target_entity}' scale must be finite and > 0, got {raw_scale}.",
                                    entity_id=action.target_entity,
                                    field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.scale",
                                    severity="ERROR",
                                )
                            )
                            has_param_error = True
                        else:
                            scale_factor = float(raw_scale)
                            method_calls.append(f".scale({scale_factor})")
                            applied_names.append("scale")

                    # 3. Opacity mutation
                    if "opacity" in params or "fill_opacity" in params:
                        raw_op = params.get("opacity") if "opacity" in params else params.get("fill_opacity")
                        if isinstance(raw_op, bool) or not isinstance(raw_op, (int, float)):
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                                    message=f"Action TRANSFORM on PrimitiveShape '{action.target_entity}' has non-numeric opacity.",
                                    entity_id=action.target_entity,
                                    field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.opacity",
                                    severity="ERROR",
                                )
                            )
                            has_param_error = True
                        elif math.isnan(raw_op) or math.isinf(raw_op) or not (0.0 <= float(raw_op) <= 1.0):
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                                    message=f"Action TRANSFORM on PrimitiveShape '{action.target_entity}' opacity must be in [0.0, 1.0], got {raw_op}.",
                                    entity_id=action.target_entity,
                                    field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.opacity",
                                    severity="ERROR",
                                )
                            )
                            has_param_error = True
                        else:
                            op_val = float(raw_op)
                            method_calls.append(f".set_opacity({op_val})")
                            applied_names.append("opacity")

                    if not has_param_error:
                        if method_calls:
                            chain_str = "".join(method_calls)
                            setup_lines.append(f"# Transform PrimitiveShape ({', '.join(applied_names)}): {action.target_entity}")
                            if is_concurrent:
                                anim_exprs.append(f"{target_var}.animate(run_time={dur}){chain_str}")
                            else:
                                anim_exprs.append(f"self.play({target_var}.animate{chain_str}, run_time={dur})")
                        else:
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION,
                                    message=(
                                        f"Action TRANSFORM on PrimitiveShape '{action.target_entity}' provided no supported "
                                        f"transformation parameters (color, scale, opacity)."
                                    ),
                                    entity_id=action.target_entity,
                                    field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters",
                                    severity="ERROR",
                                )
                            )

                elif target_entity_decl and target_entity_decl.visual_tier == VisualTier.TIER1_REGISTERED:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION,
                            message=(
                                f"Action TRANSFORM on Tier 1 registered component '{action.target_entity}' "
                                f"({target_entity_decl.component_type}) is unsupported without replacement_entity. "
                                f"Registered components must be mutated via ActionType.COMPONENT_OP."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].type",
                            severity="ERROR",
                        )
                    )

                else:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION,
                            message=(
                                f"Action TRANSFORM on entity '{action.target_entity}' "
                                f"({getattr(target_entity_decl, 'component_type', 'unknown')}) cannot be performed with "
                                f"parameters {list(params.keys())}."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters",
                            severity="ERROR",
                        )
                    )

        elif action.type == ActionType.MOVE:
            # 1. Parameter key safety validation
            for k in params:
                if not is_safe_keyword_identifier(k):
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Action MOVE on entity '{action.target_entity}' contains invalid parameter key '{k}'.",
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters",
                            severity="ERROR",
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics

            # 2. Authoritative geometry resolution
            geom = layout.entities.get(action.target_entity)
            if not geom:
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_MISSING_GEOMETRY,
                        message=f"Entity '{action.target_entity}' has no computed geometry in layout.",
                        entity_id=action.target_entity,
                        field_path=f"timeline[{step_idx}].actions[{a_idx}].target_entity",
                        severity="ERROR",
                    )
                )
                return setup_lines, anim_exprs, post_lines, diagnostics

            curr_box = (
                entity_position_tracker.get(action.target_entity)
                if entity_position_tracker and action.target_entity in entity_position_tracker
                else geom.world_box
            )
            target_entity_decl = entity_map.get(action.target_entity) if entity_map else None

            # 3. Check for trajectory parameter
            if "trajectory" in params or "path" in params:
                traj_key = "trajectory" if "trajectory" in params else "path"
                traj_val = params[traj_key]
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_OPERATION,
                        message=(
                            f"Action MOVE on entity '{action.target_entity}' requested trajectory modeling ('{traj_val}'). "
                            f"Continuous swept-volume collision modeling along custom animation paths is unsupported; "
                            f"validating static destination bounding box and compiling straight-line movement."
                        ),
                        entity_id=action.target_entity,
                        field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.{traj_key}",
                        severity="WARNING",
                        details={
                            "requested_trajectory": traj_val,
                            "validation_performed": "static_destination_enclosure",
                            "swept_volume_validated": False,
                        },
                    )
                )

            # 4. Resolve destination coordinates (x_dest, y_dest)
            x_dest: Optional[float] = None
            y_dest: Optional[float] = None
            move_mode = "shift"  # "shift" or "move_to"
            pending_state_update: Optional[Tuple[str, Any]] = None

            def _parse_coords(val: Any) -> Optional[Tuple[float, float]]:
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    if isinstance(val[0], bool) or isinstance(val[1], bool):
                        return None
                    try:
                        x_val = float(val[0])
                        y_val = float(val[1])
                        if math.isfinite(x_val) and math.isfinite(y_val):
                            return (x_val, y_val)
                    except (ValueError, TypeError):
                        return None
                elif isinstance(val, dict):
                    x_k = "x" if "x" in val else ("dx" if "dx" in val else None)
                    y_k = "y" if "y" in val else ("dy" if "dy" in val else None)
                    if x_k and y_k:
                        if isinstance(val[x_k], bool) or isinstance(val[y_k], bool):
                            return None
                        try:
                            x_val = float(val[x_k])
                            y_val = float(val[y_k])
                            if math.isfinite(x_val) and math.isfinite(y_val):
                                return (x_val, y_val)
                        except (ValueError, TypeError):
                            return None
                return None

            # Priority A: Absolute Destination (destination, target_position, position, to)
            abs_key = None
            for candidate in ("destination", "target_position", "position", "to"):
                if candidate in params:
                    abs_key = candidate
                    break

            if abs_key:
                coords = _parse_coords(params[abs_key])
                if coords is None:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=(
                                f"Action MOVE on entity '{action.target_entity}' provided invalid or non-finite "
                                f"coordinates for '{abs_key}': {params[abs_key]!r}."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.{abs_key}",
                            severity="ERROR",
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics
                x_dest, y_dest = coords
                move_mode = "move_to"

            # Priority B: Relative Shift (shift, relative_vector, offset)
            elif any(k in params for k in ("shift", "relative_vector", "offset")):
                rel_key = next(k for k in ("shift", "relative_vector", "offset") if k in params)
                coords = _parse_coords(params[rel_key])
                if coords is None:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=(
                                f"Action MOVE on entity '{action.target_entity}' provided invalid or non-finite "
                                f"vector for '{rel_key}': {params[rel_key]!r}."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.{rel_key}",
                            severity="ERROR",
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics
                dx, dy = coords
                x_dest = curr_box.x + dx
                y_dest = curr_box.y + dy
                move_mode = "shift"

            # Priority C: Viewport Sub-Slot (target_slot, slot)
            elif "target_slot" in params or "slot" in params:
                slot_key = "target_slot" if "target_slot" in params else "slot"
                slot_name = str(params[slot_key]).strip()
                vp_id = getattr(target_entity_decl, "viewport", None)
                vp_geom = layout.viewports.get(vp_id) if vp_id else None
                if not vp_geom or slot_name not in vp_geom.sub_slots:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=(
                                f"Action MOVE on entity '{action.target_entity}' references unknown sub-slot "
                                f"'{slot_name}' in viewport '{vp_id}'."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.{slot_key}",
                            severity="ERROR",
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics
                slot_box = vp_geom.sub_slots[slot_name]
                x_dest = slot_box.x
                y_dest = slot_box.y
                move_mode = "move_to"

            # Priority D: Target Viewport Center (target_viewport, viewport)
            elif "target_viewport" in params or "viewport" in params:
                vp_key = "target_viewport" if "target_viewport" in params else "viewport"
                target_vp_name = str(params[vp_key]).strip()
                vp_geom = layout.viewports.get(target_vp_name)
                if not vp_geom:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=(
                                f"Action MOVE on entity '{action.target_entity}' references unknown target viewport "
                                f"'{target_vp_name}'."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.{vp_key}",
                            severity="ERROR",
                            details={"target_viewport": target_vp_name, "available_viewports": list(layout.viewports.keys())},
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics
                x_dest = vp_geom.world_box.x
                y_dest = vp_geom.world_box.y
                move_mode = "move_to"

            # Priority E: Array Pointer Index Shift (new_target_index)
            elif "new_target_index" in params:
                val = params["new_target_index"]
                if isinstance(val, bool):
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Action MOVE parameter 'new_target_index' cannot be a boolean: {val!r}.",
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                            severity="ERROR",
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics

                try:
                    new_idx = int(val)
                except (ValueError, TypeError):
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=f"Action MOVE parameter 'new_target_index' must be an integer: {params['new_target_index']!r}.",
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                            severity="ERROR",
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics

                if new_idx < 0:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
                            message=f"Action MOVE parameter 'new_target_index' ({new_idx}) cannot be negative.",
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                            severity="ERROR",
                            details={"new_target_index": new_idx},
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics

                curr_state = (
                    entity_state_tracker.get(action.target_entity, {})
                    if entity_state_tracker
                    else dict(getattr(target_entity_decl, "initial_state", {}) or {})
                )
                raw_curr_idx = curr_state.get("target_index", 0)
                try:
                    curr_idx = int(raw_curr_idx)
                except (ValueError, TypeError):
                    curr_idx = 0

                if curr_idx < 0:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
                            message=f"Current pointer index ({curr_idx}) is negative.",
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                            severity="ERROR",
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics

                parent_id = (
                    params.get("parent_id")
                    or params.get("target_component")
                    or params.get("target_array")
                    or getattr(target_entity_decl, "parent_id", None)
                    or curr_state.get("parent_id")
                    or curr_state.get("target_component")
                    or curr_state.get("target_array")
                )
                if not parent_id:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_MISSING_GEOMETRY,
                            message=(
                                f"Action MOVE on pointer '{action.target_entity}' specifies 'new_target_index'={new_idx}, "
                                f"but entity '{action.target_entity}' has no 'parent_id' and no target component was specified."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                            severity="ERROR",
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics

                parent_geom = layout.entities.get(parent_id)
                if not parent_geom:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_MISSING_GEOMETRY,
                            message=(
                                f"Action MOVE on entity '{action.target_entity}' references parent component '{parent_id}', "
                                f"which has no computed geometry in layout."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                            severity="ERROR",
                            details={"referenced_parent": parent_id},
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics

                parent_decl = entity_map.get(parent_id) if entity_map else None
                if not parent_decl:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_MISSING_GEOMETRY,
                            message=(
                                f"Action MOVE on entity '{action.target_entity}' references undeclared parent component '{parent_id}'."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                            severity="ERROR",
                            details={"referenced_parent": parent_id},
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics

                parent_state = parent_decl.initial_state or {}
                dx = None

                # Case 1: Explicit cell coordinates
                cell_positions = (
                    parent_state.get("cell_positions")
                    or parent_state.get("element_positions")
                    or params.get("cell_positions")
                )
                if cell_positions and isinstance(cell_positions, (list, tuple)):
                    num_cells = len(cell_positions)
                    if new_idx >= num_cells:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
                                message=f"Action MOVE target index {new_idx} out of bounds for parent '{parent_id}' with {num_cells} cells.",
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                severity="ERROR",
                            )
                        )
                        return setup_lines, anim_exprs, post_lines, diagnostics
                    if curr_idx >= num_cells:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
                                message=f"Action MOVE current index {curr_idx} out of bounds for parent '{parent_id}' with {num_cells} cells.",
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                severity="ERROR",
                            )
                        )
                        return setup_lines, anim_exprs, post_lines, diagnostics
                    try:
                        dx = round(float(cell_positions[new_idx]) - float(cell_positions[curr_idx]), 4)
                    except (ValueError, TypeError):
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_MISSING_GEOMETRY,
                                message=f"Parent component '{parent_id}' has malformed cell_positions entries.",
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                severity="ERROR",
                            )
                        )
                        return setup_lines, anim_exprs, post_lines, diagnostics

                # Case 2: Unequal cell widths
                cell_widths = (
                    parent_state.get("cell_widths")
                    or parent_state.get("column_widths")
                    or params.get("cell_widths")
                    or params.get("column_widths")
                )
                if dx is None and cell_widths and isinstance(cell_widths, (list, tuple)):
                    num_cells = len(cell_widths)
                    if new_idx >= num_cells or curr_idx >= num_cells:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
                                message=f"Target index {new_idx} out of bounds for parent '{parent_id}' with {num_cells} cells.",
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                severity="ERROR",
                            )
                        )
                        return setup_lines, anim_exprs, post_lines, diagnostics
                    try:
                        c_widths = [float(w) for w in cell_widths]
                    except (ValueError, TypeError):
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_MISSING_GEOMETRY,
                                message=f"Parent component '{parent_id}' has malformed cell_widths entries.",
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                severity="ERROR",
                            )
                        )
                        return setup_lines, anim_exprs, post_lines, diagnostics
                    buff = float(parent_state.get("buff", parent_state.get("spacing", parent_state.get("cell_spacing", 0.04))))
                    if new_idx == curr_idx:
                        dx = 0.0
                    elif new_idx > curr_idx:
                        dx = sum((c_widths[k] + c_widths[k + 1]) / 2.0 + buff for k in range(curr_idx, new_idx))
                    else:
                        dx = -sum((c_widths[k] + c_widths[k + 1]) / 2.0 + buff for k in range(new_idx, curr_idx))
                    dx = round(dx, 4)

                # Case 3: Elements / cells / nodes list
                elements = parent_state.get("elements") or parent_state.get("cells") or parent_state.get("nodes")
                if dx is None and elements and isinstance(elements, list):
                    num_cells = len(elements)
                    if num_cells == 0:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
                                message=f"Parent component '{parent_id}' has 0 cells.",
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                severity="ERROR",
                            )
                        )
                        return setup_lines, anim_exprs, post_lines, diagnostics
                    if new_idx >= num_cells or curr_idx >= num_cells:
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
                                message=(
                                    f"Action MOVE on entity '{action.target_entity}' target index {new_idx} is out of bounds "
                                    f"for parent '{parent_id}' with {num_cells} cells (valid range: 0..{num_cells - 1})."
                                ),
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                severity="ERROR",
                                details={"new_target_index": new_idx, "max_valid_index": num_cells - 1, "parent_id": parent_id},
                            )
                        )
                        return setup_lines, anim_exprs, post_lines, diagnostics

                    if all(isinstance(e, dict) and "width" in e for e in elements):
                        try:
                            c_widths = [float(e["width"]) for e in elements]
                            buff = float(parent_state.get("buff", parent_state.get("spacing", parent_state.get("cell_spacing", 0.04))))
                            if new_idx == curr_idx:
                                dx = 0.0
                            elif new_idx > curr_idx:
                                dx = sum((c_widths[k] + c_widths[k + 1]) / 2.0 + buff for k in range(curr_idx, new_idx))
                            else:
                                dx = -sum((c_widths[k] + c_widths[k + 1]) / 2.0 + buff for k in range(new_idx, curr_idx))
                            dx = round(dx, 4)
                        except (ValueError, TypeError):
                            pass

                    elif all(isinstance(e, dict) and ("x" in e or "position" in e or "center_x" in e) for e in elements):
                        try:
                            c_positions = [float(e.get("x", e.get("position", e.get("center_x")))) for e in elements]
                            dx = round(c_positions[new_idx] - c_positions[curr_idx], 4)
                        except (ValueError, TypeError):
                            pass

                    if dx is None:
                        pitch_val = (
                            parent_state.get("cell_pitch")
                            or parent_state.get("pitch")
                            or parent_state.get("cell_spacing")
                            or params.get("cell_pitch")
                        )
                        if pitch_val is not None:
                            try:
                                pitch = float(pitch_val)
                            except (ValueError, TypeError):
                                pitch = None
                        elif "cell_width" in parent_state:
                            try:
                                pitch = float(parent_state["cell_width"]) + float(parent_state.get("buff", 0.0))
                            except (ValueError, TypeError):
                                pitch = None
                        else:
                            parent_w = parent_geom.world_box.width
                            if parent_w <= 0:
                                diagnostics.append(
                                    CompilationDiagnostic(
                                        error_code=CompilationErrorCode.COMPILER_MISSING_GEOMETRY,
                                        message=f"Parent component '{parent_id}' has non-positive width: {parent_w}.",
                                        entity_id=action.target_entity,
                                        field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                        severity="ERROR",
                                    )
                                )
                                return setup_lines, anim_exprs, post_lines, diagnostics
                            cell_w = round(parent_w / num_cells, 4)
                            buff = float(parent_state.get("buff", 0.04))
                            pitch = round(cell_w + buff, 4)

                        if pitch is None or pitch <= 0 or not math.isfinite(pitch):
                            diagnostics.append(
                                CompilationDiagnostic(
                                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                                    message=f"Parent component '{parent_id}' has invalid cell pitch: {pitch!r}.",
                                    entity_id=action.target_entity,
                                    field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                    severity="ERROR",
                                )
                            )
                            return setup_lines, anim_exprs, post_lines, diagnostics

                        dx = round((new_idx - curr_idx) * pitch, 4)

                # Case 4: Explicit pitch/width without elements list
                if dx is None and (
                    "cell_pitch" in parent_state
                    or "pitch" in parent_state
                    or "cell_spacing" in parent_state
                    or "cell_width" in parent_state
                    or "cell_pitch" in params
                ):
                    pitch_val = (
                        parent_state.get("cell_pitch")
                        or parent_state.get("pitch")
                        or parent_state.get("cell_spacing")
                        or params.get("cell_pitch")
                    )
                    if pitch_val is not None:
                        try:
                            pitch = float(pitch_val)
                        except (ValueError, TypeError):
                            pitch = None
                    else:
                        try:
                            pitch = float(parent_state["cell_width"]) + float(parent_state.get("buff", 0.0))
                        except (ValueError, TypeError):
                            pitch = None

                    if pitch is None or pitch <= 0 or not math.isfinite(pitch):
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                                message=f"Parent component '{parent_id}' has invalid non-positive cell pitch: {pitch!r}.",
                                entity_id=action.target_entity,
                                field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                severity="ERROR",
                            )
                        )
                        return setup_lines, anim_exprs, post_lines, diagnostics

                    cap = parent_state.get("capacity") or parent_state.get("max_cells") or parent_state.get("num_cells")
                    if cap is not None:
                        try:
                            max_c = int(cap)
                            if new_idx >= max_c or curr_idx >= max_c:
                                diagnostics.append(
                                    CompilationDiagnostic(
                                        error_code=CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
                                        message=f"Target index {new_idx} out of bounds for parent '{parent_id}' with capacity {max_c}.",
                                        entity_id=action.target_entity,
                                        field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                                        severity="ERROR",
                                    )
                                )
                                return setup_lines, anim_exprs, post_lines, diagnostics
                        except (ValueError, TypeError):
                            pass

                    dx = round((new_idx - curr_idx) * pitch, 4)

                if dx is None:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_MISSING_GEOMETRY,
                            message=(
                                f"Action MOVE on entity '{action.target_entity}' cannot derive cell pitch from parent component '{parent_id}': "
                                f"parent provides no cell/element definitions or explicit pitch geometry."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.new_target_index",
                            severity="ERROR",
                            details={"referenced_parent": parent_id},
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics

                axis = str(params.get("axis", "x")).lower()
                if axis in ("y", "vertical", "col", "column"):
                    x_dest = curr_box.x
                    y_dest = curr_box.y + dx
                else:
                    x_dest = curr_box.x + dx
                    y_dest = curr_box.y

                move_mode = "shift"
                pending_state_update = ("target_index", new_idx)

            else:
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                        message=(
                            f"Action MOVE on entity '{action.target_entity}' provided no recognized movement destination "
                            f"or shift vector (e.g. destination, shift, slot, target_viewport, new_target_index)."
                        ),
                        entity_id=action.target_entity,
                        field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters",
                        severity="ERROR",
                    )
                )
                return setup_lines, anim_exprs, post_lines, diagnostics

            # 5. Boundary & Viewport Enclosure Validation
            dest_box = BoundingBox2D(
                x=round(x_dest, 4),
                y=round(y_dest, 4),
                width=curr_box.width,
                height=curr_box.height,
            )

            # Determine enclosing boundary and track viewport migration
            explicit_vp = params.get("target_viewport") or params.get("viewport")
            current_vp = (
                entity_viewport_tracker.get(action.target_entity)
                if entity_viewport_tracker and action.target_entity in entity_viewport_tracker
                else getattr(target_entity_decl, "viewport", None)
            )

            if explicit_vp:
                target_vp_id = str(explicit_vp).strip()
                if target_vp_id not in layout.viewports:
                    diagnostics.append(
                        CompilationDiagnostic(
                            error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                            message=(
                                f"Action MOVE on entity '{action.target_entity}' references unknown target viewport "
                                f"'{target_vp_id}'. Available: {list(layout.viewports.keys())}."
                            ),
                            entity_id=action.target_entity,
                            field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.target_viewport",
                            severity="ERROR",
                            details={"target_viewport": target_vp_id, "available_viewports": list(layout.viewports.keys())},
                        )
                    )
                    return setup_lines, anim_exprs, post_lines, diagnostics
                active_target_vp = target_vp_id
            else:
                active_target_vp = current_vp

            vp_geom = layout.viewports.get(active_target_vp) if active_target_vp else None
            boundary = vp_geom.world_box if vp_geom else layout.canvas_box
            boundary_name = f"viewport '{active_target_vp}'" if vp_geom else "canvas"

            if not boundary.contains_box(dest_box, tolerance=VIEWPORT_BOUNDARY_TOLERANCE):
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_OUT_OF_BOUNDS,
                        message=(
                            f"Action MOVE destination for entity '{action.target_entity}' (center ({dest_box.x:.2f}, {dest_box.y:.2f}), "
                            f"size {dest_box.width:.2f}x{dest_box.height:.2f}) breaches {boundary_name} boundaries "
                            f"(bounds: [{boundary.left:.2f}, {boundary.right:.2f}] x [{boundary.bottom:.2f}, {boundary.top:.2f}])."
                        ),
                        entity_id=action.target_entity,
                        field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters",
                        severity="ERROR",
                        details={
                            "destination_box": dest_box.to_dict(),
                            "boundary_box": boundary.to_dict(),
                            "viewport_id": active_target_vp,
                        },
                    )
                )
                return setup_lines, anim_exprs, post_lines, diagnostics

            # 6. Animation Code Emission
            dur = round(max(0.1, action.duration), 2)
            EASING_MAP = {
                EasingType.SMOOTH: "smooth",
                EasingType.LINEAR: "linear",
                EasingType.EASE_IN: "rate_functions.ease_in_quad",
                EasingType.EASE_OUT: "rate_functions.ease_out_quad",
            }
            if action.easing is not None and action.easing not in EASING_MAP:
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                        message=f"Action MOVE on entity '{action.target_entity}' has unsupported easing: {action.easing!r}.",
                        entity_id=action.target_entity,
                        field_path=f"timeline[{step_idx}].actions[{a_idx}].easing",
                        severity="ERROR",
                    )
                )
                return setup_lines, anim_exprs, post_lines, diagnostics

            rate_fn = EASING_MAP.get(action.easing, "smooth")

            if move_mode == "move_to":
                setup_lines.append(f"# Move {action.target_entity} to ({dest_box.x:.2f}, {dest_box.y:.2f})")
                target_call = f"move_to(np.array([{round(dest_box.x, 4)}, {round(dest_box.y, 4)}, 0.0]))"
            else:
                dx = round(dest_box.x - curr_box.x, 4)
                dy = round(dest_box.y - curr_box.y, 4)
                setup_lines.append(f"# Shift {action.target_entity} by ({dx:.2f}, {dy:.2f})")
                target_call = f"shift(np.array([{dx}, {dy}, 0.0]))"

            if is_concurrent:
                animate_prefix = (
                    f"{target_var}.animate(rate_func={rate_fn}, run_time={dur})"
                    if rate_fn != "smooth"
                    else f"{target_var}.animate(run_time={dur})"
                )
                anim_exprs.append(f"{animate_prefix}.{target_call}")
            else:
                animate_prefix = (
                    f"{target_var}.animate(rate_func={rate_fn})"
                    if rate_fn != "smooth"
                    else f"{target_var}.animate"
                )
                anim_exprs.append(f"self.play({animate_prefix}.{target_call}, run_time={dur})")

            # Update tracked position, viewport ownership, and state ONLY on success
            if entity_position_tracker is not None:
                entity_position_tracker[action.target_entity] = dest_box
            if entity_viewport_tracker is not None and active_target_vp is not None:
                entity_viewport_tracker[action.target_entity] = active_target_vp
            if pending_state_update is not None and entity_state_tracker is not None:
                if action.target_entity not in entity_state_tracker:
                    entity_state_tracker[action.target_entity] = {}
                p_key, p_val = pending_state_update
                entity_state_tracker[action.target_entity][p_key] = p_val

        elif action.type == ActionType.COMPONENT_OP:
            op_name = action.operation
            if not op_name:
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                        message=f"Action COMPONENT_OP on entity '{action.target_entity}' missing required 'operation' field.",
                        entity_id=action.target_entity,
                        field_path="action.operation",
                        severity="WARNING",
                    )
                )
                if is_concurrent:
                    anim_exprs.append(f"Indicate({target_var}, run_time={dur})")
                else:
                    anim_exprs.append(f"self.play(Indicate({target_var}), run_time={dur})")
            elif not is_safe_keyword_identifier(op_name):
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                        message=(
                            f"Action COMPONENT_OP operation {op_name!r} on entity '{action.target_entity}' is not "
                            f"a valid, non-reserved Python identifier."
                        ),
                        entity_id=action.target_entity,
                        field_path="action.operation",
                        severity="ERROR",
                    )
                )
                if is_concurrent:
                    anim_exprs.append(f"Indicate({target_var}, run_time={dur})")
                else:
                    anim_exprs.append(f"self.play(Indicate({target_var}), run_time={dur})")
            else:
                # Format parameters safely with strict key validation
                kwarg_strs = []
                has_invalid_key = False
                for k, v in sorted(params.items()):
                    if not is_safe_keyword_identifier(k):
                        has_invalid_key = True
                        diagnostics.append(
                            CompilationDiagnostic(
                                error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                                message=(
                                    f"Parameter key {k!r} in COMPONENT_OP on entity '{action.target_entity}' is invalid. "
                                    f"Keyword argument keys must be valid, non-reserved Python identifiers."
                                ),
                                entity_id=action.target_entity,
                                field_path=f"action.parameters.{k}",
                                severity="ERROR",
                            )
                        )
                    else:
                        kwarg_strs.append(f"{k}={v!r}")

                if has_invalid_key:
                    if is_concurrent:
                        anim_exprs.append(f"Indicate({target_var}, run_time={dur})")
                    else:
                        anim_exprs.append(f"self.play(Indicate({target_var}), run_time={dur})")
                else:
                    kw_str = ", ".join(kwarg_strs)
                    setup_lines.append(f"# Component Operation: {op_name}({kw_str})")
                    if is_concurrent:
                        anim_var = f"_anim_{target_var}_{a_idx}"
                        setup_lines.append(
                            f"{anim_var} = (getattr({target_var}, {op_name!r})({kw_str}) "
                            f"if hasattr({target_var}, {op_name!r}) and callable(getattr({target_var}, {op_name!r})) "
                            f"else Indicate({target_var}))"
                        )
                        setup_lines.append(
                            f"if hasattr({anim_var}, 'set_run_time'):\n"
                            f"    {anim_var}.set_run_time({dur})\n"
                            f"elif hasattr({anim_var}, 'run_time'):\n"
                            f"    {anim_var}.run_time = {dur}"
                        )
                        anim_exprs.append(anim_var)
                    else:
                        anim_exprs.append(
                            f"if hasattr({target_var}, {op_name!r}) and callable(getattr({target_var}, {op_name!r})):\n"
                            f"    self.play(getattr({target_var}, {op_name!r})({kw_str}), run_time={dur})\n"
                            f"else:\n"
                            f"    self.play(Indicate({target_var}), run_time={dur})"
                        )

        return setup_lines, anim_exprs, post_lines, diagnostics

    def _compile_action(
        self,
        action: ActionDeclaration,
        target_var: str,
        layout: SceneSpatialLayout,
        entity_map: Optional[Dict[str, EntityDeclaration]] = None,
        entity_var_map: Optional[Dict[str, str]] = None,
        entity_state_tracker: Optional[Dict[str, Dict[str, Any]]] = None,
        entity_position_tracker: Optional[Dict[str, BoundingBox2D]] = None,
        entity_viewport_tracker: Optional[Dict[str, str]] = None,
        entered_entities: Optional[Set[str]] = None,
        step_idx: int = 0,
        a_idx: int = 0,
    ) -> Tuple[List[str], List[CompilationDiagnostic]]:
        """Compiles an ActionDeclaration into Manim self.play / self.wait calls."""
        setup, anims, post, diags = self._compile_action_elements(
            action=action,
            target_var=target_var,
            layout=layout,
            entity_map=entity_map,
            entity_var_map=entity_var_map,
            entity_state_tracker=entity_state_tracker,
            entity_position_tracker=entity_position_tracker,
            entity_viewport_tracker=entity_viewport_tracker,
            entered_entities=entered_entities,
            step_idx=step_idx,
            a_idx=a_idx,
            is_concurrent=False,
        )
        lines = list(setup) + list(anims) + list(post)
        if action.post_action_pause > 0.0:
            pause_time = round(action.post_action_pause, 2)
            lines.append(f"self.wait({pause_time})")
        return lines, diags

    def _emit_python_source(
        self,
        scene: SceneIR,
        layout: SceneSpatialLayout,
        entity_code_blocks: List[str],
        timeline_code_blocks: List[str],
        entered_entities: Set[str],
        entity_var_map: Dict[str, str],
    ) -> str:
        """Emits the complete, formatted Python script as a string."""
        raw_scene_name = scene.scene_id.strip()
        scene_class_name = re.sub(r"[^a-zA-Z0-9_]", "", raw_scene_name)
        if not scene_class_name:
            scene_class_name = "CompiledScene"
        elif not scene_class_name.endswith("Scene"):
            scene_class_name = f"{scene_class_name}Scene"

        # Collect narration lines for setup_layout
        lecture_lines = []
        for event in scene.timeline:
            if event.narration_anchor and event.narration_anchor.text_cue:
                cue = event.narration_anchor.text_cue.strip()
                if cue and cue not in lecture_lines:
                    lecture_lines.append(cue)

        # Indent helper
        def indent(code: str, spaces: int = 8) -> str:
            prefix = " " * spaces
            return "\n".join(f"{prefix}{line}" if line.strip() else "" for line in code.splitlines())

        # Entities that are never entered in timeline actions should be added upfront
        static_additions: List[str] = []
        for entity in scene.entities:
            if entity.id not in entered_entities:
                vname = entity_var_map.get(entity.id)
                if vname:
                    static_additions.append(f"self.add({vname})")

        static_add_code = "\n".join(f"        {sa}" for sa in static_additions)
        entities_code_joined = "\n\n".join(indent(b, 8) for b in entity_code_blocks)
        timeline_code_joined = "\n\n".join(indent(b, 8) for b in timeline_code_blocks)

        source = f'''"""
Auto-generated Manim Scene compiled by CineLearn Deterministic Scene Compiler.
Scene ID: {scene.scene_id}
Title: {scene.title}
Target Aspect Ratio: {scene.layout_configuration.aspect_ratio}
Duration Budget: {scene.duration_budget}s
"""

from manim import *
import numpy as np

# Import TeachingScene base class and registered domain components
try:
    from prompts.base_class import TeachingScene
except ImportError:
    class TeachingScene(Scene):
        def setup_layout(self, title_text, lecture_lines, captions=None):
            t = Text(str(title_text), font_size=24, color=WHITE).to_edge(UP, buff=0.3)
            self.add(t)

try:
    from src.domains import VisualRegistry, ProceduralComponent
    from src.domains.cs import *
    from src.domains.chemistry import *
    from src.domains.math import *
except ImportError:
    pass


class {scene_class_name}(TeachingScene):
    def construct(self):
        # -------------------------------------------------------------
        # 1. Canvas and Lecture Layout Setup
        # -------------------------------------------------------------
        self.setup_layout(
            title_text={scene.title!r},
            lecture_lines={lecture_lines!r},
        )

        # -------------------------------------------------------------
        # 2. Entity Declarations & Geometric Instantiations
        # -------------------------------------------------------------
{entities_code_joined}

        # Static Entity Additions (entities without explicit enter actions)
{static_add_code if static_add_code else "        pass"}

        # -------------------------------------------------------------
        # 3. Scheduled Timeline Animations
        # -------------------------------------------------------------
{timeline_code_joined}

        # Final pacing pause
        self.wait(1.0)
'''
        return source
