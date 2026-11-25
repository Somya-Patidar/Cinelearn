"""
CineLearn — Milestone 4A: Hierarchical Box Model Layout Solver
=============================================================
Deterministic, domain-independent layout engine operating on validated Scene IR.
Implements dynamic canvas computation, viewport planning, hierarchical DAG resolution,
flexbox box-model allocation, parent-child coordinate transformations,
the 8-step overcrowding mitigation hierarchy, and time-aware spatial verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from src.compiler.compiler_models import (
    CompilationDiagnostic,
    ScheduledAction,
    ScheduledTimeline,
)
from src.compiler.dimension_estimator import EstimatedDimensions, IntrinsicDimensionEstimator
from src.compiler.layout_models import (
    CONTAINER_BORDER_TOLERANCE,
    VIEWPORT_BOUNDARY_TOLERANCE,
    BoundingBox2D,
    ComputedEntityGeometry,
    ComputedViewportGeometry,
    LayoutDiagnostic,
    LayoutErrorCode,
    LayoutFailureReport,
    SceneSpatialLayout,
)
from src.compiler.spatial_ledger import SpatialLedger
from src.compiler.timeline_scheduler import (
    TimelineScheduler,
    build_disjoint_lifespan_intervals,
    determine_initial_presence,
)
from src.ir.models import (
    Alignment,
    EntityDeclaration,
    FlowDirection,
    LayoutConfiguration,
    SceneIR,
    SceneSequenceIR,
    ViewportDeclaration,
    ViewportType,
    VisualTier,
)


class HierarchicalBoxModelLayoutSolver:
    """
    Deterministic layout solver for Scene IR documents.
    Computes 2D bounding boxes and spatial allocations without depending on Manim.
    """

    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        self.dimension_estimator = IntrinsicDimensionEstimator(manifest=manifest)

    def solve(
        self,
        scene: SceneIR,
        scheduled_timeline: Optional[ScheduledTimeline] = None,
    ) -> SceneSpatialLayout:
        """Computes complete spatial layout for a SceneIR document."""
        errors: List[LayoutDiagnostic] = []
        warnings: List[LayoutDiagnostic] = []
        mitigations_applied: List[str] = []
        failure_report: Optional[LayoutFailureReport] = None

        # Authoritative Sequence A: Schedule timeline if not provided
        if scheduled_timeline is None:
            scheduler = TimelineScheduler()
            scheduled_timeline, _ = scheduler.schedule(scene)

        # -----------------------------------------------------------------
        # 1. Compute Active Visual Canvas Boundaries Dynamically
        # -----------------------------------------------------------------
        canvas_box = self.compute_canvas_box(scene.layout_configuration)

        # -----------------------------------------------------------------
        # 2. Plan Viewport Bounding Boxes and Sub-Slots
        # -----------------------------------------------------------------
        computed_viewports = self.plan_viewports(scene.viewports, canvas_box)

        # -----------------------------------------------------------------
        # 3. Resolve Entity Hierarchy & Detect DAG Cycles
        # -----------------------------------------------------------------
        cycle_error = self._check_hierarchy_cycles(scene.entities)
        if cycle_error:
            errors.append(cycle_error)
            return SceneSpatialLayout(
                scene_id=scene.scene_id,
                canvas_box=canvas_box,
                viewports=computed_viewports,
                entities={},
                is_valid=False,
                errors=errors,
                warnings=warnings,
            )

        top_level_entities: List[EntityDeclaration] = []
        child_entities: Dict[str, List[EntityDeclaration]] = {}
        for ent in scene.entities:
            if ent.parent_id:
                child_entities.setdefault(ent.parent_id, []).append(ent)
            else:
                top_level_entities.append(ent)

        # -----------------------------------------------------------------
        # 4. Estimate Intrinsic Dimensions for All Entities
        # -----------------------------------------------------------------
        dimensions: Dict[str, EstimatedDimensions] = {}
        for ent in scene.entities:
            dim = self.dimension_estimator.estimate(ent)
            dimensions[ent.id] = dim
            if dim.is_uncertain:
                warnings.append(
                    LayoutDiagnostic(
                        error_code=LayoutErrorCode.LAYOUT_DIMENSION_UNCERTAINTY,
                        message=f"Entity '{ent.id}' ({ent.component_type}) has uncertain dimensions (applied 25% safety margin).",
                        entity_id=ent.id,
                        details=dim.details,
                    )
                )

        # -----------------------------------------------------------------
        # 5. Flexbox Allocation for Top-Level Viewport Entities
        # -----------------------------------------------------------------
        computed_entities: Dict[str, ComputedEntityGeometry] = {}

        # Group top-level entities by assigned viewport
        entities_by_viewport: Dict[str, List[EntityDeclaration]] = {}
        for ent in top_level_entities:
            vp_key = ent.viewport
            # Fallback if viewport name does not directly match declared keys
            if vp_key not in computed_viewports:
                matched = self._match_viewport_key(vp_key, computed_viewports)
                if matched:
                    vp_key = matched
                else:
                    errors.append(
                        LayoutDiagnostic(
                            error_code=LayoutErrorCode.LAYOUT_INVALID_VIEWPORT,
                            message=f"Entity '{ent.id}' assigned to undeclared viewport '{ent.viewport}'. Available: {list(computed_viewports.keys())}",
                            entity_id=ent.id,
                            viewport_id=ent.viewport,
                            details={"assigned_viewport": ent.viewport, "available_viewports": list(computed_viewports.keys())},
                        )
                    )
                    continue
            entities_by_viewport.setdefault(vp_key, []).append(ent)

        # Allocate each viewport's children
        for vp_id, vp_geom in computed_viewports.items():
            vp_ents = entities_by_viewport.get(vp_id, [])
            vp_decl = scene.viewports.get(vp_id)
            vp_geom.assigned_entities = [e.id for e in vp_ents]

            if not vp_ents:
                continue

            # Check if entities declare sub-slots
            sub_slot_groups: Dict[str, List[EntityDeclaration]] = {}
            unslotted_ents: List[EntityDeclaration] = []
            for e in vp_ents:
                if e.slot and e.slot in vp_geom.sub_slots:
                    sub_slot_groups.setdefault(e.slot, []).append(e)
                else:
                    unslotted_ents.append(e)

            # Allocate slotted entities within their sub-slot boxes
            for slot_name, slot_ents in sub_slot_groups.items():
                slot_box = vp_geom.sub_slots[slot_name]
                allocated, mit_list, fail_rep = self._allocate_box_children(
                    container_box=slot_box,
                    entities=slot_ents,
                    flow=FlowDirection.COLUMN if slot_box.height > slot_box.width else FlowDirection.ROW,
                    alignment=Alignment.CENTER,
                    padding=0.10,
                    dimensions=dimensions,
                    viewport_id=vp_id,
                    scene_id=scene.scene_id,
                    timeline=scheduled_timeline,
                )
                computed_entities.update(allocated)
                mitigations_applied.extend(mit_list)
                if fail_rep:
                    failure_report = fail_rep
                    errors.append(
                        LayoutDiagnostic(
                            error_code=LayoutErrorCode.LAYOUT_OVERCROWDING_ERROR,
                            message=f"Sub-slot '{slot_name}' in viewport '{vp_id}' overcrowded: deficit {fail_rep.spatial_deficit:.2f} units.",
                            viewport_id=vp_id,
                            details=fail_rep.model_dump(),
                        )
                    )

            # Allocate unslotted entities in the main viewport area
            if unslotted_ents:
                flow_dir = vp_decl.flow if vp_decl else FlowDirection.SINGLE
                align = vp_decl.alignment if vp_decl else Alignment.CENTER
                pad = vp_decl.padding if vp_decl else 0.20

                allocated, mit_list, fail_rep = self._allocate_box_children(
                    container_box=vp_geom.world_box,
                    entities=unslotted_ents,
                    flow=flow_dir,
                    alignment=align,
                    padding=pad,
                    dimensions=dimensions,
                    viewport_id=vp_id,
                    scene_id=scene.scene_id,
                    timeline=scheduled_timeline,
                    inactive_viewports=[v for v, g in computed_viewports.items() if not g.assigned_entities and v != vp_id],
                )
                computed_entities.update(allocated)
                mitigations_applied.extend(mit_list)
                if fail_rep:
                    failure_report = fail_rep
                    errors.append(
                        LayoutDiagnostic(
                            error_code=LayoutErrorCode.LAYOUT_OVERCROWDING_ERROR,
                            message=f"Viewport '{vp_id}' overcrowded: deficit {fail_rep.spatial_deficit:.2f} units.",
                            viewport_id=vp_id,
                            details=fail_rep.model_dump(),
                        )
                    )

        # -----------------------------------------------------------------
        # 6. Allocate Hierarchical Child Entities (Parent Containment)
        # Iterative topological resolution to support arbitrary depth and declaration order
        # -----------------------------------------------------------------
        remaining_parents = dict(child_entities)
        max_iterations = len(child_entities) + 5
        iteration = 0

        while remaining_parents and iteration < max_iterations:
            iteration += 1
            # Find parents that have already been allocated
            current_batch = [pid for pid in remaining_parents if pid in computed_entities]
            if not current_batch:
                # No more progress can be made (unresolvable missing parent or cycle)
                break

            for parent_id in current_batch:
                children = remaining_parents.pop(parent_id)
                parent_geom = computed_entities[parent_id]
                parent_decl = next((e for e in scene.entities if e.id == parent_id), None)

                # Determine container stroke width and border padding
                stroke_pts = 2.0
                if parent_decl:
                    if parent_decl.styling and parent_decl.styling.stroke_width is not None:
                        stroke_pts = float(parent_decl.styling.stroke_width)
                    elif parent_decl.initial_state and "stroke_width" in parent_decl.initial_state:
                        try:
                            stroke_pts = float(parent_decl.initial_state["stroke_width"])
                        except (ValueError, TypeError):
                            stroke_pts = 2.0

                border_pad = 0.10
                if parent_decl and parent_decl.initial_state and "padding" in parent_decl.initial_state:
                    try:
                        border_pad = float(parent_decl.initial_state["padding"])
                    except (ValueError, TypeError):
                        border_pad = 0.10

                # Clearance from outer container boundary to prevent visual border collision
                stroke_inset = round(max(0.04, (stroke_pts / 200.0) + border_pad), 4)

                p_box = parent_geom.world_box

                # Compute available inner content dimensions after stroke inset
                raw_inner_w = round(p_box.width - 2.0 * stroke_inset, 4)
                raw_inner_h = round(p_box.height - 2.0 * stroke_inset, 4)

                # Explicit check: If container is too small or stroke/inset consumes all available space,
                # emit LAYOUT_PARENT_BOUNDS_EXCEEDED with structured diagnostics.
                if raw_inner_w <= 0.05 or raw_inner_h <= 0.05:
                    for child in children:
                        errors.append(
                            LayoutDiagnostic(
                                error_code=LayoutErrorCode.LAYOUT_PARENT_BOUNDS_EXCEEDED,
                                message=(
                                    f"Parent container '{parent_id}' ({p_box.width:.2f}x{p_box.height:.2f}) with stroke {stroke_pts:.1f}pt "
                                    f"and inset {stroke_inset:.2f} has insufficient inner space ({raw_inner_w:.2f}x{raw_inner_h:.2f}) "
                                    f"to allocate child '{child.id}'."
                                ),
                                entity_id=child.id,
                                viewport_id=parent_geom.viewport_id,
                                related_entities=[parent_id],
                                details={
                                    "parent_box": p_box.to_dict(),
                                    "stroke_width": stroke_pts,
                                    "stroke_inset": stroke_inset,
                                    "raw_inner_width": raw_inner_w,
                                    "raw_inner_height": raw_inner_h,
                                    "reason": "stroke_and_padding_consume_container",
                                },
                            )
                        )
                    continue

                content_box = BoundingBox2D(
                    x=p_box.x,
                    y=p_box.y,
                    width=min(p_box.width, max(0.05, raw_inner_w)),
                    height=min(p_box.height, max(0.05, raw_inner_h)),
                )

                # Allocate children inside parent box
                child_alloc, _, _ = self._allocate_box_children(
                    container_box=content_box,
                    entities=children,
                    flow=FlowDirection.COLUMN if content_box.height > content_box.width else FlowDirection.ROW,
                    alignment=Alignment.CENTER,
                    padding=0.08,
                    dimensions=dimensions,
                    viewport_id=parent_geom.viewport_id,
                    scene_id=scene.scene_id,
                    timeline=scheduled_timeline,
                    parent_id=parent_id,
                )

                raw_bound_w = round(p_box.width - max(0.02, stroke_pts / 100.0), 4)
                raw_bound_h = round(p_box.height - max(0.02, stroke_pts / 100.0), 4)
                inner_bound = BoundingBox2D(
                    x=p_box.x,
                    y=p_box.y,
                    width=min(p_box.width, max(0.02, raw_bound_w)),
                    height=min(p_box.height, max(0.02, raw_bound_h)),
                )

                for cid, cgeom in child_alloc.items():
                    computed_entities[cid] = cgeom
                    # Check strict parent containment and allocation limits
                    raw_dim = dimensions[cid]
                    if (raw_dim.width > p_box.width or raw_dim.height > p_box.height) and cgeom.scale < 0.70:
                        errors.append(
                            LayoutDiagnostic(
                                error_code=LayoutErrorCode.LAYOUT_PARENT_BOUNDS_EXCEEDED,
                                message=f"Child entity '{cid}' (size {raw_dim.width:.1f}x{raw_dim.height:.1f}) exceeds allocation of parent '{parent_id}' ({p_box.width:.1f}x{p_box.height:.1f}, scale {cgeom.scale:.2f} < 0.70).",
                                entity_id=cid,
                                viewport_id=parent_geom.viewport_id,
                                related_entities=[parent_id],
                                details={
                                    "child_box": cgeom.world_box.to_dict(),
                                    "parent_box": p_box.to_dict(),
                                    "stroke_width": stroke_pts,
                                    "stroke_inset": stroke_inset,
                                    "scale": cgeom.scale,
                                },
                            )
                        )
                    elif not inner_bound.contains_box(cgeom.world_box, tolerance=CONTAINER_BORDER_TOLERANCE):
                        errors.append(
                            LayoutDiagnostic(
                                error_code=LayoutErrorCode.LAYOUT_PARENT_BOUNDS_EXCEEDED,
                                message=f"Child entity '{cid}' visually overlaps or exceeds border of parent '{parent_id}'.",
                                entity_id=cid,
                                viewport_id=parent_geom.viewport_id,
                                related_entities=[parent_id],
                                details={
                                    "child_box": cgeom.world_box.to_dict(),
                                    "parent_box": p_box.to_dict(),
                                    "stroke_width": stroke_pts,
                                    "stroke_inset": stroke_inset,
                                },
                            )
                        )

        # Flag any unresolvable children (whose parent was never allocated or doesn't exist)
        if remaining_parents:
            for unres_parent_id, orphan_children in remaining_parents.items():
                for o_child in orphan_children:
                    errors.append(
                        LayoutDiagnostic(
                            error_code=LayoutErrorCode.LAYOUT_UNSATISFIABLE_CONSTRAINTS,
                            message=f"Child entity '{o_child.id}' references unresolved parent '{unres_parent_id}'.",
                            entity_id=o_child.id,
                            viewport_id=o_child.viewport,
                            related_entities=[unres_parent_id],
                            details={"unresolved_parent_id": unres_parent_id},
                        )
                    )

        # -----------------------------------------------------------------
        # 7. Time-Aware Spatial Ledger & Collision Verification
        # -----------------------------------------------------------------
        world_boxes = {eid: geom.world_box for eid, geom in computed_entities.items()}
        ledger = SpatialLedger(scene_duration=scene.duration_budget or 30.0)
        ledger.populate_from_scene_ir(scene, world_boxes, scheduled_timeline=scheduled_timeline)

        ledger_valid, _, ledger_diags = ledger.validate_spatial_integrity(
            canvas_box=canvas_box,
            overlap_threshold=0.05,
            viewports=computed_viewports,
        )

        for diag in ledger_diags:
            if diag.error_code in (LayoutErrorCode.LAYOUT_FORBIDDEN_OVERLAP, LayoutErrorCode.LAYOUT_BOUNDARY_BREACH):
                errors.append(diag)
            else:
                warnings.append(diag)

        is_valid = (len(errors) == 0) and (failure_report is None)

        return SceneSpatialLayout(
            scene_id=scene.scene_id,
            canvas_box=canvas_box,
            viewports=computed_viewports,
            entities=computed_entities,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            mitigations_applied=mitigations_applied,
            failure_report=failure_report,
            metadata=scene.metadata,
        )

    def solve_sequence(self, sequence: SceneSequenceIR) -> Dict[str, SceneSpatialLayout]:
        """Computes spatial layout for all sections in a SceneSequenceIR."""
        layouts: Dict[str, SceneSpatialLayout] = {}
        for section in sequence.sections:
            layout = self.solve(section)
            layouts[section.scene_id] = layout
        return layouts

    @staticmethod
    def _parse_aspect_ratio(aspect_str: str) -> float:
        """Parses aspect ratio string (e.g. '16:9', '4:3', '9:16', '1:1') to float."""
        if not aspect_str:
            return 16.0 / 9.0
        try:
            parts = str(aspect_str).split(":")
            if len(parts) == 2:
                w = float(parts[0].strip())
                h = float(parts[1].strip())
                if w > 0 and h > 0:
                    return w / h
            return float(aspect_str)
        except Exception:
            return 16.0 / 9.0

    def compute_canvas_box(self, config: LayoutConfiguration) -> BoundingBox2D:
        """
        Dynamically calculates the active visual canvas boundaries from layout configuration,
        aspect ratio, and safety padding.
        Eradicates fixed hardcoded constants.
        """
        aspect = self._parse_aspect_ratio(config.aspect_ratio)
        half_height = 4.0
        half_width = half_height * aspect

        # Vertical margins from safety padding (default safety_padding=0.20 -> v_pad=0.40)
        v_pad = 2.0 * max(0.0, float(config.safety_padding))
        top_max = half_height - v_pad
        bottom_min = -half_height + v_pad

        # Top banner reserved zone: ~1.05 units
        top = top_max - 1.05 if config.has_title_banner else top_max

        # Subtitles reserved zone: ~0.95 units
        bottom = bottom_min + 0.95 if config.has_subtitles else bottom_min

        # Horizontal margins from safety padding and aspect ratio
        pad_ratio = (config.safety_padding / 0.20) if config.safety_padding > 0 else 1.0
        h_pad = pad_ratio * (7.111111111111111 - 6.60) * (aspect / (16.0 / 9.0))
        right = round(half_width - h_pad, 4)

        # Full active horizontal span before caption reservation
        full_active_width = 2.0 * right

        # Caption column reserved zone: ~33.7% of horizontal active span
        if config.has_caption_column:
            caption_reservation = full_active_width * (4.45 / 13.20)
            left = round(-right + caption_reservation, 4)
        else:
            left = round(-right, 4)

        return BoundingBox2D.from_min_max(left=left, right=right, bottom=round(bottom, 4), top=round(top, 4))

    def plan_viewports(
        self,
        viewports: Dict[str, ViewportDeclaration],
        canvas_box: BoundingBox2D,
    ) -> Dict[str, ComputedViewportGeometry]:
        """
        Computes 2D bounding boxes and sub-slots for declared viewports
        within the active visual canvas.
        """
        computed: Dict[str, ComputedViewportGeometry] = {}
        has_inspector = any(
            v.type in (ViewportType.INSPECTOR, "viewport_inspector") or "inspector" in k.lower()
            for k, v in viewports.items()
        )

        inspector_height = 1.05
        inspector_gap = 0.10

        if has_inspector:
            main_bottom = canvas_box.bottom + inspector_height + inspector_gap
        else:
            main_bottom = canvas_box.bottom

        split_gap = 0.25
        split_w = (canvas_box.width - split_gap) / 2.0

        for vp_id, decl in viewports.items():
            vp_type = decl.type

            if vp_type in (ViewportType.INSPECTOR, "viewport_inspector"):
                vbox = BoundingBox2D.from_min_max(
                    left=canvas_box.left,
                    right=canvas_box.right,
                    bottom=canvas_box.bottom,
                    top=canvas_box.bottom + inspector_height,
                )
            elif vp_type in (ViewportType.SPLIT_LEFT, "viewport_split_left"):
                vbox = BoundingBox2D.from_min_max(
                    left=canvas_box.left,
                    right=canvas_box.left + split_w,
                    bottom=main_bottom,
                    top=canvas_box.top,
                )
            elif vp_type in (ViewportType.SPLIT_RIGHT, "viewport_split_right"):
                vbox = BoundingBox2D.from_min_max(
                    left=canvas_box.right - split_w,
                    right=canvas_box.right,
                    bottom=main_bottom,
                    top=canvas_box.top,
                )
            elif vp_type in (ViewportType.FULL, "viewport_full"):
                vbox = BoundingBox2D.from_min_max(
                    left=canvas_box.left,
                    right=canvas_box.right,
                    bottom=canvas_box.bottom,
                    top=canvas_box.top,
                )
            else:
                # Default: ViewportType.MAIN
                vbox = BoundingBox2D.from_min_max(
                    left=canvas_box.left,
                    right=canvas_box.right,
                    bottom=main_bottom,
                    top=canvas_box.top,
                )

            # Sub-slot partitioning
            sub_slots_map: Dict[str, BoundingBox2D] = {}
            if decl.sub_slots:
                n_slots = len(decl.sub_slots)
                # Partition horizontally if wide, vertically if tall
                if vbox.aspect_ratio >= 1.2:
                    slot_w = (vbox.width - (0.10 * (n_slots - 1))) / n_slots
                    cur_left = vbox.left
                    for sname in decl.sub_slots:
                        sbox = BoundingBox2D.from_min_max(
                            left=cur_left,
                            right=cur_left + slot_w,
                            bottom=vbox.bottom,
                            top=vbox.top,
                        )
                        sub_slots_map[sname] = sbox
                        cur_left += slot_w + 0.10
                else:
                    slot_h = (vbox.height - (0.10 * (n_slots - 1))) / n_slots
                    cur_top = vbox.top
                    for sname in decl.sub_slots:
                        sbox = BoundingBox2D.from_min_max(
                            left=vbox.left,
                            right=vbox.right,
                            bottom=cur_top - slot_h,
                            top=cur_top,
                        )
                        sub_slots_map[sname] = sbox
                        cur_top -= slot_h + 0.10

            computed[vp_id] = ComputedViewportGeometry(
                viewport_id=vp_id,
                viewport_type=vp_type,
                world_box=vbox,
                flow=decl.flow,
                alignment=decl.alignment,
                padding=decl.padding,
                sub_slots=sub_slots_map,
                assigned_entities=[],
            )

        return computed

    def _allocate_box_children(
        self,
        container_box: BoundingBox2D,
        entities: List[EntityDeclaration],
        flow: FlowDirection,
        alignment: Alignment,
        padding: float,
        dimensions: Dict[str, EstimatedDimensions],
        viewport_id: str,
        scene_id: str,
        timeline: Any,
        parent_id: Optional[str] = None,
        inactive_viewports: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, ComputedEntityGeometry], List[str], Optional[LayoutFailureReport]]:
        """
        Allocates entity bounding boxes within a container box using Flexbox rules
        and the 8-Step Overcrowding Mitigation Hierarchy.
        """
        allocated: Dict[str, ComputedEntityGeometry] = {}
        mitigations: List[str] = []
        failure_report: Optional[LayoutFailureReport] = None

        if not entities:
            return allocated, mitigations, failure_report

        # -------------------------------------------------------------
        # Single Entity Allocation (Center in container)
        # -------------------------------------------------------------
        if len(entities) == 1:
            ent = entities[0]
            dim = dimensions[ent.id]
            w = min(dim.width, container_box.width * (1.0 - padding))
            h = min(dim.height, container_box.height * (1.0 - padding))
            # Preserve aspect ratio if scaled down (except for flexible conceptual cards where width is unconstrained by dock height)
            if ent.visual_tier == VisualTier.TIER4_CONCEPTUAL_CARD or ent.component_type in (
                "ConceptualCard", "FormulaCard", "DefinitionCard", "TextCard", "MathTex"
            ):
                scale = 1.0
            elif w < dim.width or h < dim.height:
                scale = min(w / dim.width, h / dim.height)
                w = dim.width * scale
                h = dim.height * scale
            else:
                scale = 1.0

            world_box = BoundingBox2D(x=container_box.x, y=container_box.y, width=w, height=h)
            local_box = BoundingBox2D(x=0.0, y=0.0, width=w, height=h)

            allocated[ent.id] = ComputedEntityGeometry(
                entity_id=ent.id,
                viewport_id=viewport_id,
                parent_id=parent_id,
                local_box=local_box,
                world_box=world_box,
                scale=round(scale, 4),
                z_index=ent.z_index,
                visual_priority=ent.visual_priority,
                slot=ent.slot,
                is_uncertain=dim.is_uncertain,
            )
            return allocated, mitigations, failure_report

        # -------------------------------------------------------------
        # Multiple Sibling Allocation
        # -------------------------------------------------------------
        # Step 6 Pre-check: Check if entities have non-overlapping lifespans (sequential replacement)
        if flow == FlowDirection.SINGLE:
            is_sequential = self._check_sequential_lifespans(entities, timeline)
            if is_sequential:
                mitigations.append("Step 6: sequential_visual_replacement")
                for ent in entities:
                    dim = dimensions[ent.id]
                    scale = min(
                        container_box.width * (1.0 - padding) / dim.width,
                        container_box.height * (1.0 - padding) / dim.height,
                        1.0,
                    )
                    w = dim.width * scale
                    h = dim.height * scale
                    wbox = BoundingBox2D(x=container_box.x, y=container_box.y, width=w, height=h)
                    lbox = BoundingBox2D(x=0.0, y=0.0, width=w, height=h)
                    allocated[ent.id] = ComputedEntityGeometry(
                        entity_id=ent.id,
                        viewport_id=viewport_id,
                        parent_id=parent_id,
                        local_box=lbox,
                        world_box=wbox,
                        scale=round(scale, 4),
                        z_index=ent.z_index,
                        visual_priority=ent.visual_priority,
                        slot=ent.slot,
                        is_uncertain=dim.is_uncertain,
                    )
                return allocated, mitigations, failure_report
            else:
                # Multiple simultaneous entities assigned to FlowDirection.SINGLE:
                # Automatically adapt to column flow (Step 2)
                flow = FlowDirection.COLUMN
                mitigations.append("Step 2: change_flow_direction (single -> column)")

        # Attempt flex packing
        gap = 0.20
        inner_w = container_box.width * (1.0 - padding)
        inner_h = container_box.height * (1.0 - padding)

        # Calculate requirements
        if flow == FlowDirection.COLUMN:
            req_extent = sum(dimensions[e.id].height for e in entities) + gap * (len(entities) - 1)
            avail_extent = inner_h
            is_overcrowded = req_extent > avail_extent
        else:
            # ROW
            req_extent = sum(dimensions[e.id].width for e in entities) + gap * (len(entities) - 1)
            avail_extent = inner_w
            is_overcrowded = req_extent > avail_extent

        # -------------------------------------------------------------
        # Execute Overcrowding Mitigation Hierarchy if Needed
        # -------------------------------------------------------------
        scale_factor = 1.0
        if is_overcrowded:
            # Step 2: Try changing direction if container aspect allows
            if flow == FlowDirection.COLUMN and container_box.aspect_ratio > 1.8:
                alt_req = sum(dimensions[e.id].width for e in entities) + gap * (len(entities) - 1)
                if alt_req <= inner_w:
                    flow = FlowDirection.ROW
                    mitigations.append("Step 2: change_flow_direction (column -> row)")
                    is_overcrowded = False

        if is_overcrowded:
            # Step 5: Tighten gaps and padding
            gap = 0.08
            inner_w = container_box.width * 0.95
            inner_h = container_box.height * 0.95
            if flow == FlowDirection.COLUMN:
                req_extent = sum(dimensions[e.id].height for e in entities) + gap * (len(entities) - 1)
                avail_extent = inner_h
            else:
                req_extent = sum(dimensions[e.id].width for e in entities) + gap * (len(entities) - 1)
                avail_extent = inner_w

            if req_extent <= avail_extent:
                mitigations.append("Step 5: simplify_composition_hierarchy (tighten gaps)")
                is_overcrowded = False

        if is_overcrowded:
            # Step 7: Proportional Readability-Floored Scaling
            scale_needed = avail_extent / req_extent if req_extent > 0 else 1.0
            # Readability floor: scale >= 0.70
            if scale_needed >= 0.70:
                scale_factor = scale_needed
                mitigations.append(f"Step 7: proportional_scaling_floored (scale={scale_factor:.2f})")
                is_overcrowded = False
            else:
                # Step 8: Mitigation Failed -> Emit Structured Failure Report
                mitigations.append("Step 8: emit_structured_failure_report")
                spatial_deficit = req_extent - avail_extent
                failure_report = LayoutFailureReport(
                    status="LAYOUT_OVERCROWDING_ERROR",
                    scene_id=scene_id,
                    viewport_id=viewport_id,
                    conflicting_entities=[e.id for e in entities],
                    available_dimensions={"width": container_box.width, "height": container_box.height},
                    required_dimensions={"width": inner_w, "height": req_extent},
                    spatial_deficit=round(spatial_deficit, 4),
                    mitigations_attempted=mitigations,
                    recommended_action="SPLIT_INTO_SEQUENTIAL_BEATS",
                )
                scale_factor = max(scale_needed, 0.40)  # Apply best-effort placement

        # -------------------------------------------------------------
        # Deterministic Coordinates Computation
        # -------------------------------------------------------------
        if flow == FlowDirection.COLUMN:
            # Stack from top to bottom
            scaled_heights = [dimensions[e.id].height * scale_factor for e in entities]
            total_h = sum(scaled_heights) + gap * (len(entities) - 1)

            if alignment == Alignment.START:
                start_y = container_box.top - (container_box.height * padding / 2.0)
            elif alignment == Alignment.END:
                start_y = container_box.bottom + total_h + (container_box.height * padding / 2.0)
            else:
                # CENTER
                start_y = container_box.y + (total_h / 2.0)

            cur_top = start_y
            for idx, ent in enumerate(entities):
                dim = dimensions[ent.id]
                ew = min(dim.width * scale_factor, container_box.width * 0.95)
                eh = scaled_heights[idx]

                ey = cur_top - (eh / 2.0)
                ex = container_box.x
                wbox = BoundingBox2D(x=ex, y=ey, width=ew, height=eh)
                lbox = BoundingBox2D(x=ex - container_box.x, y=ey - container_box.y, width=ew, height=eh)

                allocated[ent.id] = ComputedEntityGeometry(
                    entity_id=ent.id,
                    viewport_id=viewport_id,
                    parent_id=parent_id,
                    local_box=lbox,
                    world_box=wbox,
                    scale=round(scale_factor, 4),
                    z_index=ent.z_index,
                    visual_priority=ent.visual_priority,
                    slot=ent.slot,
                    is_uncertain=dim.is_uncertain,
                )
                cur_top -= (eh + gap)

        else:
            # ROW: Pack from left to right
            scaled_widths = [dimensions[e.id].width * scale_factor for e in entities]
            total_w = sum(scaled_widths) + gap * (len(entities) - 1)

            if alignment == Alignment.START:
                start_x = container_box.left + (container_box.width * padding / 2.0)
            elif alignment == Alignment.END:
                start_x = container_box.right - total_w - (container_box.width * padding / 2.0)
            else:
                # CENTER
                start_x = container_box.x - (total_w / 2.0)

            cur_left = start_x
            for idx, ent in enumerate(entities):
                dim = dimensions[ent.id]
                ew = scaled_widths[idx]
                eh = min(dim.height * scale_factor, container_box.height * 0.95)

                ex = cur_left + (ew / 2.0)
                ey = container_box.y
                wbox = BoundingBox2D(x=ex, y=ey, width=ew, height=eh)
                lbox = BoundingBox2D(x=ex - container_box.x, y=ey - container_box.y, width=ew, height=eh)

                allocated[ent.id] = ComputedEntityGeometry(
                    entity_id=ent.id,
                    viewport_id=viewport_id,
                    parent_id=parent_id,
                    local_box=lbox,
                    world_box=wbox,
                    scale=round(scale_factor, 4),
                    z_index=ent.z_index,
                    visual_priority=ent.visual_priority,
                    slot=ent.slot,
                    is_uncertain=dim.is_uncertain,
                )
                cur_left += (ew + gap)

        return allocated, mitigations, failure_report

    def _check_hierarchy_cycles(self, entities: List[EntityDeclaration]) -> Optional[LayoutDiagnostic]:
        """Validates that parent-child relationships form an acyclic graph (DAG)."""
        adjacency: Dict[str, List[str]] = {e.id: [] for e in entities}
        for e in entities:
            if e.parent_id:
                if e.parent_id == e.id:
                    return LayoutDiagnostic(
                        error_code=LayoutErrorCode.LAYOUT_CYCLIC_HIERARCHY,
                        message=f"Entity '{e.id}' is set as its own parent.",
                        entity_id=e.id,
                        related_entities=[e.id],
                        details={"self_cycle": e.id},
                    )
                if e.parent_id in adjacency:
                    adjacency[e.parent_id].append(e.id)

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for child in adjacency.get(node, []):
                if child not in visited:
                    if dfs(child):
                        return True
                elif child in rec_stack:
                    return True
            rec_stack.remove(node)
            return False

        for eid in adjacency:
            if eid not in visited:
                if dfs(eid):
                    return LayoutDiagnostic(
                        error_code=LayoutErrorCode.LAYOUT_CYCLIC_HIERARCHY,
                        message=f"Circular containment hierarchy cycle detected involving '{eid}'.",
                        entity_id=eid,
                        details={"detected_at": eid},
                    )
        return None

    def _check_sequential_lifespans(
        self,
        entities: List[EntityDeclaration],
        timeline: Any,
    ) -> bool:
        """
        Checks if a group of entities have strictly non-overlapping active intervals.
        Supports multi-lifecycle entities with multiple re-entries, active replacement,
        and disjoint spans using authoritative TimelineScheduler algorithms.
        """
        if not timeline or not entities:
            return False

        if len(entities) <= 1:
            return True

        # Extract scheduled actions from ScheduledTimeline or raw timeline events
        if isinstance(timeline, ScheduledTimeline):
            scheduled_actions = timeline.actions
            scene_duration = timeline.total_duration or 30.0
            dummy_scene = None
        elif isinstance(timeline, list) and timeline and isinstance(timeline[0], ScheduledAction):
            scheduled_actions = timeline
            scene_duration = max((sa.end_time for sa in scheduled_actions), default=30.0)
            dummy_scene = None
        else:
            # timeline is raw TimelineEvent list: schedule via TimelineScheduler
            events = timeline if isinstance(timeline, list) else []
            dummy_scene = SceneIR(
                scene_id="sequential_check_dummy",
                duration_budget=1000.0,
                layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
                viewports={},
                entities=entities,
                timeline=events,
            )
            scheduler = TimelineScheduler()
            sched_tl, _ = scheduler.schedule(dummy_scene)
            scheduled_actions = sched_tl.actions
            scene_duration = sched_tl.total_duration or 1000.0

        entity_intervals: Dict[str, List[Tuple[float, float]]] = {}
        target_ids = [e.id for e in entities]

        for ent in entities:
            if dummy_scene is None:
                # Build minimal scene context for initial presence determination
                scene_ctx = SceneIR(
                    scene_id="presence_ctx",
                    duration_budget=scene_duration,
                    layout_configuration=LayoutConfiguration(aspect_ratio="16:9"),
                    viewports={},
                    entities=[ent],
                    timeline=[],
                )
            else:
                scene_ctx = dummy_scene

            init_presence, _ = determine_initial_presence(ent, scene_ctx, scheduled_actions=scheduled_actions)
            diags: List[CompilationDiagnostic] = []
            intervals = build_disjoint_lifespan_intervals(
                eid=ent.id,
                scheduled_actions=scheduled_actions,
                scene_duration=scene_duration,
                initial_presence=init_presence,
                diagnostics=diags,
            )
            if not intervals:
                intervals = [(0.0, scene_duration)]
            entity_intervals[ent.id] = intervals

        # Check all distinct entity pairs for overlapping intervals
        for i in range(len(target_ids)):
            for j in range(i + 1, len(target_ids)):
                ints_a = entity_intervals.get(target_ids[i], [])
                ints_b = entity_intervals.get(target_ids[j], [])
                for sa, ea in ints_a:
                    for sb, eb in ints_b:
                        overlap_s = max(sa, sb)
                        overlap_e = min(ea, eb)
                        # Disjoint intervals overlap if overlap_e > overlap_s
                        if overlap_e > overlap_s:
                            return False
        return True

    def _match_viewport_key(
        self,
        key: str,
        viewports: Dict[str, ComputedViewportGeometry],
    ) -> Optional[str]:
        """Matches a loose or prefixed viewport name against declared viewports."""
        k = key.lower()
        if k in viewports:
            return k
        for vp_id in viewports:
            if vp_id.lower() == k:
                return vp_id
            if k.replace("dock_", "").replace("viewport_", "") == vp_id.lower().replace("dock_", "").replace("viewport_", ""):
                return vp_id
            if "split_left" in k and "split_left" in vp_id:
                return vp_id
            if "split_right" in k and "split_right" in vp_id:
                return vp_id
            if "inspector" in k and "inspector" in vp_id:
                return vp_id
            if "main" in k and "main" in vp_id:
                return vp_id
        return None
