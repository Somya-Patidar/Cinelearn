"""
CineLearn — Milestone 4A: Spatial Ledger & Time-Aware Collision Engine
======================================================================
Implements time-aware 2D bounding box occupancy tracking, formal 7-category
collision classification, and spatial invariant validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.compiler.compiler_models import (
    CompilationDiagnostic,
    ScheduledAction,
    ScheduledTimeline,
)
from src.compiler.layout_models import (
    CONTAINER_BORDER_TOLERANCE,
    VIEWPORT_BOUNDARY_TOLERANCE,
    BoundingBox2D,
    CollisionCategory,
    CollisionRecord,
    LayoutDiagnostic,
    LayoutErrorCode,
)
from src.compiler.timeline_scheduler import (
    TimelineScheduler,
    build_disjoint_lifespan_intervals,
    determine_initial_presence,
)
from src.ir.models import ActionType, SceneIR


class EntitySpatialSpan(BaseModel):
    """Spatial and temporal lifespan footprint of an entity in a scene."""
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    box: BoundingBox2D
    t_start: float = Field(ge=0.0)
    t_end: float = Field(ge=0.0)
    z_index: int = 10
    layer_role: Optional[str] = None
    parent_id: Optional[str] = None
    group_id: Optional[str] = None
    slot: Optional[str] = None
    viewport_id: str = "main"
    move_duration: Optional[float] = None
    move_windows: List[Tuple[float, float]] = Field(default_factory=list)
    intervals: List[Tuple[float, float]] = Field(default_factory=list)


class SpatialTimeSlice(BaseModel):
    """Snapshot of spatial occupancy and active entities at a discrete timestamp."""
    model_config = ConfigDict(extra="forbid")

    timestamp: float
    active_entities: List[str]
    bounds: Dict[str, BoundingBox2D]
    collisions: List[CollisionRecord]


class SpatialLedger:
    """
    Time-aware spatial occupancy ledger.
    Tracks entity bounding boxes across time intervals [t_start, t_end],
    classifies spatial intersections using the formal 7-category collision taxonomy,
    and reports fatal layout defects.
    """

    def __init__(self, scene_duration: float = 30.0):
        self.scene_duration = float(scene_duration)
        self.spans: Dict[str, EntitySpatialSpan] = {}

    def register_span(
        self,
        entity_id: str,
        box: BoundingBox2D,
        t_start: float = 0.0,
        t_end: Optional[float] = None,
        z_index: int = 10,
        layer_role: Optional[str] = None,
        parent_id: Optional[str] = None,
        group_id: Optional[str] = None,
        slot: Optional[str] = None,
        viewport_id: str = "main",
        move_duration: Optional[float] = None,
        move_windows: Optional[List[Tuple[float, float]]] = None,
        intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> EntitySpatialSpan:
        """Registers a spatial-temporal lifespan footprint for an entity."""
        end_time = float(t_end if t_end is not None else self.scene_duration)
        effective_intervals = (
            list(intervals)
            if intervals is not None and len(intervals) > 0
            else [(max(0.0, float(t_start)), max(0.0, end_time))]
        )
        effective_move_windows = list(move_windows) if move_windows is not None else []
        if move_duration is not None and not effective_move_windows:
            start_t = min(i[0] for i in effective_intervals)
            effective_move_windows = [(start_t, start_t + float(move_duration))]

        span = EntitySpatialSpan(
            entity_id=entity_id,
            box=box,
            t_start=min(i[0] for i in effective_intervals),
            t_end=max(i[1] for i in effective_intervals),
            z_index=z_index,
            layer_role=layer_role,
            parent_id=parent_id,
            group_id=group_id,
            slot=slot,
            viewport_id=viewport_id,
            move_duration=move_duration,
            move_windows=effective_move_windows,
            intervals=effective_intervals,
        )
        self.spans[entity_id] = span
        return span

    def populate_from_scene_ir(
        self,
        scene: SceneIR,
        computed_geometries: Dict[str, BoundingBox2D],
        scheduled_timeline: Optional[ScheduledTimeline] = None,
    ) -> None:
        """
        Populates entity spatial spans from a validated Scene IR document,
        authoritative ScheduledTimeline, and computed bounding boxes.
        """
        self.scene_duration = float(scene.duration_budget or 30.0)

        # Ensure we have an authoritative ScheduledTimeline
        if scheduled_timeline is None:
            scheduler = TimelineScheduler()
            scheduled_timeline, _ = scheduler.schedule(scene)

        # Track entity movement windows from scheduled actions
        move_windows: Dict[str, List[Tuple[float, float]]] = {}
        move_durations: Dict[str, float] = {}

        for sa in scheduled_timeline.actions:
            if sa.action.type == ActionType.MOVE:
                move_windows.setdefault(sa.target_entity, []).append((sa.start_time, sa.end_time))
                move_durations[sa.target_entity] = sa.duration

        # Register each entity with its computed bounds and lifespan intervals
        for entity in scene.entities:
            eid = entity.id
            if eid not in computed_geometries:
                continue

            box = computed_geometries[eid]

            # Reconstruct disjoint lifespan intervals via authoritative state machine
            init_presence, _ = determine_initial_presence(entity, scene, scheduled_actions=scheduled_timeline.actions)
            diags: List[CompilationDiagnostic] = []
            intervals = build_disjoint_lifespan_intervals(
                eid=eid,
                scheduled_actions=scheduled_timeline.actions,
                scene_duration=self.scene_duration,
                initial_presence=init_presence,
                diagnostics=diags,
            )

            if not intervals:
                intervals = [(0.0, self.scene_duration)]

            t_start = intervals[0][0]
            t_end = intervals[-1][1]

            layer_role = None
            opacity_val = getattr(entity.styling, "fill_opacity", getattr(entity.styling, "opacity", None)) if entity.styling else None
            if opacity_val is not None and opacity_val < 0.9:
                layer_role = "overlay"
            elif "badge" in (entity.initial_state or {}):
                layer_role = "badge"
            elif entity.visual_tier.value in ("tier4_conceptual_card", "TIER4_CONCEPTUAL_CARD"):
                layer_role = "card"

            self.register_span(
                entity_id=eid,
                box=box,
                t_start=t_start,
                t_end=t_end,
                z_index=entity.z_index,
                layer_role=layer_role,
                parent_id=entity.parent_id,
                slot=entity.slot,
                viewport_id=entity.viewport,
                move_duration=move_durations.get(eid),
                move_windows=move_windows.get(eid, []),
                intervals=intervals,
            )

    def check_collisions(
        self,
        overlap_threshold: float = 0.05,
    ) -> List[CollisionRecord]:
        """
        Examines all entity pairs in the ledger and classifies intersections
        according to the 7 formal collision categories.
        """
        records: List[CollisionRecord] = []
        entity_ids = sorted(self.spans.keys())

        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                id_a = entity_ids[i]
                id_b = entity_ids[j]
                span_a = self.spans[id_a]
                span_b = self.spans[id_b]

                # Check spatial AABB intersection
                if not span_a.box.intersects(span_b.box, tolerance=0.001):
                    continue

                overlap_ratio = span_a.box.overlap_ratio(span_b.box)
                intersect_area = span_a.box.intersection_area(span_b.box)

                # Check temporal overlap across active lifespan intervals
                intervals_a = span_a.intervals if span_a.intervals else [(span_a.t_start, span_a.t_end)]
                intervals_b = span_b.intervals if span_b.intervals else [(span_b.t_start, span_b.t_end)]

                overlapping_intervals: List[Tuple[float, float]] = []
                for a0, a1 in intervals_a:
                    for b0, b1 in intervals_b:
                        ov_start = max(a0, b0)
                        ov_end = min(a1, b1)
                        if ov_end > ov_start + 0.001:
                            overlapping_intervals.append((ov_start, ov_end))

                # Category 6: Sequential Replacement (no temporal overlap between active lifespans)
                if not overlapping_intervals:
                    records.append(
                        CollisionRecord(
                            entity_a=id_a,
                            entity_b=id_b,
                            category=CollisionCategory.SEQUENTIAL_REPLACEMENT,
                            is_fatal=False,
                            overlap_area=round(intersect_area, 4),
                            overlap_ratio=round(overlap_ratio, 4),
                            time_range=None,
                            description=(
                                f"Entities '{id_a}' and '{id_b}' occupy overlapping screen area "
                                f"sequentially with disjoint lifespans: "
                                f"'{id_a}' {intervals_a} vs '{id_b}' {intervals_b}."
                            ),
                        )
                    )
                    continue

                # Below minimum threshold (e.g. boundary touching or < 5% overlap)
                if overlap_ratio <= overlap_threshold:
                    continue

                t_overlap_start = min(w[0] for w in overlapping_intervals)
                t_overlap_end = max(w[1] for w in overlapping_intervals)
                time_range = (t_overlap_start, t_overlap_end)

                # Category 2: Intentional Containment (parent-child or ancestor-descendant relationship)
                def _is_ancestor(ancestor_id: str, desc_id: str) -> bool:
                    curr = self.spans.get(desc_id)
                    visited = set()
                    while curr and curr.parent_id and curr.parent_id not in visited:
                        if curr.parent_id == ancestor_id:
                            return True
                        visited.add(curr.parent_id)
                        curr = self.spans.get(curr.parent_id)
                    return False

                is_a_desc_of_b = _is_ancestor(id_b, id_a)
                is_b_desc_of_a = _is_ancestor(id_a, id_b)

                if is_a_desc_of_b or is_b_desc_of_a:
                    parent_span = span_b if is_a_desc_of_b else span_a
                    child_span = span_a if is_a_desc_of_b else span_b
                    # Check whether child is within parent boundaries (tight tolerance ensures border containment)
                    is_contained = parent_span.box.contains_box(child_span.box, tolerance=CONTAINER_BORDER_TOLERANCE)
                    records.append(
                        CollisionRecord(
                            entity_a=id_a,
                            entity_b=id_b,
                            category=CollisionCategory.INTENTIONAL_CONTAINMENT,
                            is_fatal=not is_contained,
                            overlap_area=round(intersect_area, 4),
                            overlap_ratio=round(overlap_ratio, 4),
                            time_range=time_range,
                            description=(
                                f"Child '{child_span.entity_id}' contained in parent '{parent_span.entity_id}' "
                                f"({'properly contained' if is_contained else 'CHILD EXCEEDS PARENT BOUNDS'})."
                            ),
                        )
                    )
                    continue

                # Category 3: Intentional Layering (distinct z_indices with overlay/badge role)
                if span_a.z_index != span_b.z_index:
                    is_layer_role = bool(
                        span_a.layer_role in ("overlay", "badge", "halo", "highlight", "card_badge")
                        or span_b.layer_role in ("overlay", "badge", "halo", "highlight", "card_badge")
                    )
                    if is_layer_role:
                        records.append(
                            CollisionRecord(
                                entity_a=id_a,
                                entity_b=id_b,
                                category=CollisionCategory.INTENTIONAL_LAYERING,
                                is_fatal=False,
                                overlap_area=round(intersect_area, 4),
                                overlap_ratio=round(overlap_ratio, 4),
                                time_range=time_range,
                                description=(
                                    f"Intentional layering: '{id_a}' (z={span_a.z_index}) "
                                    f"and '{id_b}' (z={span_b.z_index})."
                                ),
                            )
                        )
                        continue

                # Category 4: Composite Grouping (members of same primitive tree or group)
                if span_a.group_id and span_a.group_id == span_b.group_id:
                    records.append(
                        CollisionRecord(
                            entity_a=id_a,
                            entity_b=id_b,
                            category=CollisionCategory.COMPOSITE_GROUPING,
                            is_fatal=False,
                            overlap_area=round(intersect_area, 4),
                            overlap_ratio=round(overlap_ratio, 4),
                            time_range=time_range,
                            description=f"Declared composite group '{span_a.group_id}' internal overlap.",
                        )
                    )
                    continue

                # Category 5: Temporary Transition (trajectory intersection during short move animation <= 0.8s)
                raw_move_windows = [
                    (float(w[0]), float(w[1]))
                    for w in (span_a.move_windows + span_b.move_windows)
                    if len(w) >= 2 and 0.0 < (float(w[1]) - float(w[0])) <= 0.80 + 0.001
                ]

                is_temp_transition = False
                trans_dur = None
                if raw_move_windows and overlapping_intervals:
                    # Deterministically sort move windows for the interacting pair
                    sorted_windows = sorted(raw_move_windows, key=lambda w: (w[0], w[1]))
                    merged_move_windows: List[Tuple[float, float]] = []
                    curr_s, curr_e = sorted_windows[0]
                    for nxt_s, nxt_e in sorted_windows[1:]:
                        if nxt_s <= curr_e + 0.05:
                            curr_e = max(curr_e, nxt_e)
                        else:
                            merged_move_windows.append((curr_s, curr_e))
                            curr_s, curr_e = nxt_s, nxt_e
                    merged_move_windows.append((curr_s, curr_e))

                    # Every overlapping interval must be bounded within merged move windows and <= 0.8s
                    all_intervals_covered = True
                    for ov_s, ov_e in overlapping_intervals:
                        if (ov_e - ov_s) > 0.80 + 0.001:
                            all_intervals_covered = False
                            break
                        covered = any(
                            (mw[0] - 0.05 <= ov_s and ov_e <= mw[1] + 0.05)
                            for mw in merged_move_windows
                        )
                        if not covered:
                            all_intervals_covered = False
                            break

                    if all_intervals_covered:
                        is_temp_transition = True
                        trans_dur = max(ov_e - ov_s for ov_s, ov_e in overlapping_intervals)

                if is_temp_transition:
                    records.append(
                        CollisionRecord(
                            entity_a=id_a,
                            entity_b=id_b,
                            category=CollisionCategory.TEMPORARY_TRANSITION,
                            is_fatal=False,
                            overlap_area=round(intersect_area, 4),
                            overlap_ratio=round(overlap_ratio, 4),
                            time_range=time_range,
                            description=f"Temporary transition trajectory intersection ({trans_dur:.2f}s <= 0.8s).",
                        )
                    )
                    continue

                # Category 7: Distinct Sub-Slots in same viewport
                if (
                    span_a.viewport_id == span_b.viewport_id
                    and span_a.slot
                    and span_b.slot
                    and span_a.slot != span_b.slot
                ):
                    records.append(
                        CollisionRecord(
                            entity_a=id_a,
                            entity_b=id_b,
                            category=CollisionCategory.SHARED_SUB_SLOT,
                            is_fatal=False,
                            overlap_area=round(intersect_area, 4),
                            overlap_ratio=round(overlap_ratio, 4),
                            time_range=time_range,
                            description=f"Distinct sub-slots '{span_a.slot}' and '{span_b.slot}'.",
                        )
                    )
                    continue

                # Category 1: Fatal Forbidden Overlap
                records.append(
                    CollisionRecord(
                        entity_a=id_a,
                        entity_b=id_b,
                        category=CollisionCategory.FORBIDDEN_OVERLAP,
                        is_fatal=True,
                        overlap_area=round(intersect_area, 4),
                        overlap_ratio=round(overlap_ratio, 4),
                        time_range=time_range,
                        description=(
                            f"FORBIDDEN OVERLAP (Fatal): Entities '{id_a}' and '{id_b}' occupy overlapping screen "
                            f"space ({overlap_ratio * 100:.1f}%) simultaneously during [{t_overlap_start:.1f}s - {t_overlap_end:.1f}s]."
                        ),
                    )
                )

        return records

    def validate_spatial_integrity(
        self,
        canvas_box: BoundingBox2D,
        overlap_threshold: float = 0.05,
        viewports: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[CollisionRecord], List[LayoutDiagnostic]]:
        """
        Validates the overall spatial layout.
        Checks for canvas and viewport boundary breaches and forbidden collisions.
        """
        diagnostics: List[LayoutDiagnostic] = []
        collisions = self.check_collisions(overlap_threshold=overlap_threshold)

        # 1. Boundary breach checks
        for eid, span in self.spans.items():
            # A. Active canvas boundary
            if not canvas_box.contains_box(span.box, tolerance=VIEWPORT_BOUNDARY_TOLERANCE):
                breach_left = max(0.0, canvas_box.left - span.box.left)
                breach_right = max(0.0, span.box.right - canvas_box.right)
                breach_bottom = max(0.0, canvas_box.bottom - span.box.bottom)
                breach_top = max(0.0, span.box.top - canvas_box.top)
                max_breach = max(breach_left, breach_right, breach_bottom, breach_top)

                diagnostics.append(
                    LayoutDiagnostic(
                        error_code=LayoutErrorCode.LAYOUT_BOUNDARY_BREACH,
                        message=(
                            f"Entity '{eid}' breaches active canvas boundaries by {max_breach:.2f} units "
                            f"(box=[{span.box.left:.2f}, {span.box.right:.2f}, {span.box.bottom:.2f}, {span.box.top:.2f}], "
                            f"canvas=[{canvas_box.left:.2f}, {canvas_box.right:.2f}, {canvas_box.bottom:.2f}, {canvas_box.top:.2f}])."
                        ),
                        entity_id=eid,
                        viewport_id=span.viewport_id,
                        details={
                            "breach": max_breach,
                            "boundary": "canvas",
                            "box": span.box.to_dict(),
                            "canvas_box": canvas_box.to_dict(),
                        },
                    )
                )

            # B. Viewport boundary (if viewports provided and entity has assigned viewport)
            if viewports and span.viewport_id in viewports:
                vp_val = viewports[span.viewport_id]
                vp_box = vp_val.world_box if hasattr(vp_val, "world_box") else vp_val
                if isinstance(vp_box, BoundingBox2D) and not vp_box.contains_box(span.box, tolerance=VIEWPORT_BOUNDARY_TOLERANCE):
                    v_breach_left = max(0.0, vp_box.left - span.box.left)
                    v_breach_right = max(0.0, span.box.right - vp_box.right)
                    v_breach_bottom = max(0.0, vp_box.bottom - span.box.bottom)
                    v_breach_top = max(0.0, span.box.top - vp_box.top)
                    v_max_breach = max(v_breach_left, v_breach_right, v_breach_bottom, v_breach_top)

                    diagnostics.append(
                        LayoutDiagnostic(
                            error_code=LayoutErrorCode.LAYOUT_BOUNDARY_BREACH,
                            message=(
                                f"Entity '{eid}' breaches assigned viewport '{span.viewport_id}' boundaries by {v_max_breach:.2f} units "
                                f"(box=[{span.box.left:.2f}, {span.box.right:.2f}, {span.box.bottom:.2f}, {span.box.top:.2f}], "
                                f"viewport=[{vp_box.left:.2f}, {vp_box.right:.2f}, {vp_box.bottom:.2f}, {vp_box.top:.2f}])."
                            ),
                            entity_id=eid,
                            viewport_id=span.viewport_id,
                            details={
                                "breach": v_max_breach,
                                "boundary": f"viewport '{span.viewport_id}'",
                                "box": span.box.to_dict(),
                                "viewport_box": vp_box.to_dict(),
                            },
                        )
                    )

        # 2. Fatal collision checks
        for coll in collisions:
            if coll.is_fatal:
                if coll.category == CollisionCategory.INTENTIONAL_CONTAINMENT:
                    err_code = LayoutErrorCode.LAYOUT_PARENT_BOUNDS_EXCEEDED
                else:
                    err_code = LayoutErrorCode.LAYOUT_FORBIDDEN_OVERLAP

                span_a = self.spans.get(coll.entity_a)
                vp_id = span_a.viewport_id if span_a else None

                diagnostics.append(
                    LayoutDiagnostic(
                        error_code=err_code,
                        message=coll.description,
                        entity_id=coll.entity_a,
                        viewport_id=vp_id,
                        related_entities=[coll.entity_b],
                        details={
                            "category": coll.category.value,
                            "overlap_ratio": coll.overlap_ratio,
                            "overlap_area": coll.overlap_area,
                            "time_range": coll.time_range,
                            "box_a": span_a.box.to_dict() if span_a else {},
                            "box_b": self.spans[coll.entity_b].box.to_dict() if coll.entity_b in self.spans else {},
                        },
                    )
                )

        is_valid = len(diagnostics) == 0
        return is_valid, collisions, diagnostics

    def query_time_slice(self, timestamp: float) -> SpatialTimeSlice:
        """Returns all visible entities and active collisions at a discrete timestamp."""
        t = float(timestamp)
        active: List[str] = []
        bounds: Dict[str, BoundingBox2D] = {}

        for eid, span in self.spans.items():
            intervals = span.intervals if span.intervals else [(span.t_start, span.t_end)]
            if any(t0 <= t <= t1 for t0, t1 in intervals):
                active.append(eid)
                bounds[eid] = span.box

        # Filter collisions active at timestamp t
        all_colls = self.check_collisions()
        active_colls = [
            c for c in all_colls
            if c.time_range and (c.time_range[0] <= t <= c.time_range[1])
        ]

        return SpatialTimeSlice(
            timestamp=t,
            active_entities=active,
            bounds=bounds,
            collisions=active_colls,
        )
