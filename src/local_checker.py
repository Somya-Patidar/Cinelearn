"""
Local Visual Checker and Deterministic Fixer for CineLearn / Manim Educational Videos.

Provides lightweight, 100% local visual validation:
- Zero Gemini / external API calls ($0 cost, 0 tokens)
- Reuses normal successful Manim render (zero additional inspection renders)
- Detects:
    1. Out-of-screen elements (Manim CE v0.18.1 default frame 14.2222 x 8.0 with safety margin,
       plus the shared layout contract: title buffer y <= 2.60, subtitle buffer y >= -2.75,
       lecture column x >= -2.15, right edge x <= 6.60)
    2. Obvious bounding-box overlaps between distinct visual elements
    3. Bad placement / duplicate grid assignments / excessive scaling
- Deterministic local code correction before Critic (shifting grid positions, reducing scale)
- Post-Critic visual validation to catch issues introduced by visual feedback
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass
class VisualIssue:
    issue_type: str  # "OUT_OF_SCREEN", "OVERLAP", "BAD_PLACEMENT", "SYNC_MISMATCH", "COMPLEXITY_OVERLOAD", etc.
    severity: str    # "error", "warning", "info"
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    can_fix_locally: bool = False
    suggested_fix: Optional[Dict[str, Any]] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualCheckResult:
    is_clean: bool
    issues: List[VisualIssue] = field(default_factory=list)
    summary: str = ""
    status: str = "CLEAN"  # "CLEAN", "FIXED", "ISSUES_FOUND", "METADATA_MISSING"

    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def get_local_fixable_issues(self) -> List[VisualIssue]:
        return [i for i in self.issues if i.can_fix_locally]

    def get_unresolved_hints(self) -> str:
        """Format unresolved local findings as concise hints for the Critic prompt."""
        unresolved = [i for i in self.issues if not i.can_fix_locally]
        if not unresolved:
            return ""
        hints = []
        for i in unresolved:
            hints.append(f"LOCAL VALIDATION WARNING: {i.description}")
        return "\n".join(hints)


class LocalVisualChecker:
    """
    Lightweight local geometry and layout validator:
    - Evaluates mobject bounding boxes extracted during the standard Manim render
    - Statically inspects Manim code and grid placement annotations
    - Detects boundary breaches, mutual collisions, and grid coordinate conflicts
    """

    # Manim standard 16:9 frame dimensions
    FRAME_WIDTH: float = 14.2222
    FRAME_HEIGHT: float = 8.0
    X_MIN: float = -7.1111
    X_MAX: float = 7.1111
    Y_MIN: float = -4.0
    Y_MAX: float = 4.0

    # Layout safe zones (identical to the contract in the stage 2/3/4 prompts)
    DEFAULT_SAFETY_MARGIN: float = 0.35
    SAFE_TOP: float = 2.60        # Below title buffer
    SAFE_BOTTOM: float = -2.75    # Above subtitle buffer
    SAFE_LEFT: float = -2.15      # Right of the lecture caption column
    SAFE_RIGHT: float = 6.60      # Inside right screen edge

    # Class groups used by the overlap filter (Manim CE v0.18.1 class names)
    LINE_TYPES = {
        "Arrow", "Vector", "Line", "DashedLine", "DoubleArrow", "CurvedArrow", "CurvedDoubleArrow",
        "TangentLine", "Elbow", "Underline", "Arc", "ArcBetweenPoints", "NumberLine", "Cross",
    }
    CONTAINER_TYPES = {
        "SurroundingRectangle", "BackgroundRectangle", "RoundedRectangle", "Rectangle", "Square",
        "Circle", "Ellipse", "Annulus", "Polygon", "RegularPolygon", "Triangle", "Brace",
        "Table", "MobjectTable", "IntegerTable", "DecimalTable", "Axes", "NumberPlane",
    }

    # Composite domain components treated as single coherent visual units (context-aware complexity)
    COMPOSITE_COMPONENT_TYPES = {
        "GraphVisual", "DPTableMatrix", "MoleculeVisual", "ReactionEnergyDiagram",
        "FunctionCurve", "NumberLineVisual", "VisualComponent", "ProceduralComponent",
        "TableMatrixGroup", "SafeGraphWrapper", "Table", "MobjectTable", "IntegerTable", "DecimalTable",
        "ProcessCardVisual", "ReadyQueueVisual", "CPUExecutionVisual", "GanttChartVisual", "MetricTableVisual",
    }

    # Classes whose position is fixed by their constructor arguments (start/end points)
    POSITIONED_BY_CONSTRUCTION = {"Line", "Arrow", "DashedLine", "DoubleArrow"}
    TRACKED_CLASSES = (
        "Text", "MathTex", "Tex", "Circle", "Ellipse", "Square", "Rectangle", "RoundedRectangle",
        "Triangle", "RegularPolygon", "Polygon", "Dot", "Arrow", "DoubleArrow", "Line", "DashedLine",
        "Matrix", "Table", "MobjectTable", "IntegerTable", "Axes", "NumberPlane", "NumberLine",
        "Integer", "DecimalNumber", "VGroup", "Group",
    )

    def __init__(self, safety_margin: float = 0.35):
        self.safety_margin = safety_margin
        self.right_limit = self.X_MAX - safety_margin
        self.left_limit = self.X_MIN + safety_margin
        self.top_limit = self.Y_MAX - safety_margin
        self.bottom_limit = self.Y_MIN + safety_margin

    def check_section(
        self,
        section_id: str,
        code: str,
        output_dir: Path,
        timestamps_data: Optional[Dict[str, Any]] = None,
    ) -> VisualCheckResult:
        """
        Runs comprehensive local visual and multi-frame synchronization checks on a section:
        1. Reads runtime layout metadata from layout_{SceneName}.json if present.
        2. Evaluates multi-frame keyframes, context-aware complexity, and trajectory collisions.
        3. Statically inspects grid placement commands in the section code.
        4. Evaluates narration synchronization against timestamps metadata.
        5. Returns aggregated issues and local fixability flags.
        """
        issues: List[VisualIssue] = []

        # 1. Runtime bounding box inspection
        scene_name = f"{section_id.title().replace('_', '')}Scene"
        layout_candidates = [
            output_dir / f"layout_{scene_name}.json",
            output_dir / f"layout_{section_id}.json",
            output_dir / "layout_report.json",
        ]

        layout_data = None
        layout_found = False
        for cand in layout_candidates:
            if cand.exists():
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        layout_data = json.load(f)
                    layout_found = True
                    break
                except Exception:
                    pass

        frame_w = None
        frame_h = None
        if layout_data:
            cfg = layout_data.get("config") or {}
            frame_w = layout_data.get("frame_width") or cfg.get("frame_width")
            frame_h = layout_data.get("frame_height") or cfg.get("frame_height")
            if isinstance(layout_data.get("mobjects"), list):
                mobjects = layout_data["mobjects"]
                issues.extend(self._check_mobjects_geometry(mobjects, frame_width=frame_w, frame_height=frame_h))
                # Terminal frame visual complexity
                issues.extend(self.check_visual_complexity(mobjects, checkpoint_name="terminal_frame"))

            # Multi-frame intermediate checkpoints inspection (geometry, complexity, and trajectory collisions)
            checkpoints = layout_data.get("checkpoints", [])
            if checkpoints:
                cp_issues = self.check_multiframe_checkpoints(checkpoints, frame_width=frame_w, frame_height=frame_h)
                issues.extend(cp_issues)

            # Check for completely blank scene canvas (no visible mobjects across final frame and all checkpoints)
            has_visible_canvas = False
            for mob in layout_data.get("mobjects", []):
                if self._is_meaningful_canvas_mobject(mob):
                    has_visible_canvas = True
                    break
            if not has_visible_canvas:
                for cp in layout_data.get("checkpoints", []):
                    for mob in cp.get("mobjects", []):
                        if self._is_meaningful_canvas_mobject(mob):
                            has_visible_canvas = True
                            break
                    if has_visible_canvas:
                        break
            if not has_visible_canvas and (layout_data.get("mobjects") is not None or layout_data.get("checkpoints")):
                issues.append(VisualIssue(
                    issue_type="BLANK_SCENE",
                    severity="error",
                    description="Scene produces a blank canvas with zero visible visual elements across all keyframes",
                    details={"section_id": section_id},
                    can_fix_locally=False,
                    suggested_fix={"action": "instantiate_domain_component"},
                    evidence={"metric": "blank_scene", "visible_count": 0}
                ))

            # Check for placeholder scenes (where structural visuals are placeholders with no genuine visual diagram)
            all_canvas_mobs = []
            for mob in layout_data.get("mobjects", []):
                if self._is_meaningful_canvas_mobject(mob):
                    all_canvas_mobs.append(mob)
            for cp in layout_data.get("checkpoints", []):
                for mob in cp.get("mobjects", []):
                    if self._is_meaningful_canvas_mobject(mob):
                        all_canvas_mobs.append(mob)

            has_substantive_visual = False
            has_placeholder = False
            for mob in all_canvas_mobs:
                if mob.get("is_placeholder", False):
                    has_placeholder = True
                else:
                    cls_m = mob.get("class", "")
                    # Math formulas, tables, curves, or geometric groups count as substantive visual content
                    is_plain_text = cls_m in ("Text", "Paragraph", "Title")
                    if not is_plain_text or float(mob.get("width", 0)) * float(mob.get("height", 0)) > 2.0:
                        has_substantive_visual = True

            if has_placeholder and not has_substantive_visual:
                issues.append(VisualIssue(
                    issue_type="PLACEHOLDER_ONLY_SCENE",
                    severity="error",
                    description="Scene relies on placeholder fallback card without genuine educational diagram or visual components",
                    details={"section_id": section_id, "placeholder_names": [m.get("placeholder_name", m.get("class")) for m in all_canvas_mobs if m.get("is_placeholder")]},
                    can_fix_locally=False,
                    suggested_fix={"action": "replace_placeholder_with_diagram_or_component"},
                    evidence={"metric": "placeholder_only", "has_substantive_visual": False}
                ))
        else:
            # Metadata missing: record explicit warning and incomplete validation
            issues.append(VisualIssue(
                issue_type="METADATA_MISSING",
                severity="warning",
                description="Layout metadata JSON not found; geometry validation incomplete",
                can_fix_locally=False,
                evidence={"metric": "metadata_presence", "status": "missing"}
            ))

        # 2. Static code analysis (grid placement conflicts & excessive scaling)
        code_issues = self._check_code_grid_placements(code)
        issues.extend(code_issues)

        # 3. Narration synchronization analysis (evaluates visual events vs narration timestamps)
        ts_data = timestamps_data
        if ts_data is None:
            ts_candidates = [
                output_dir / f"{section_id}_timestamps.json",
                output_dir / f"{section_id}_audio_timestamps.json",
                output_dir / "timestamps.json",
            ]
            for cand in ts_candidates:
                if cand.exists():
                    try:
                        with open(cand, "r", encoding="utf-8") as f:
                            ts_data = json.load(f)
                        break
                    except Exception:
                        pass
        if ts_data and isinstance(ts_data, dict) and "lines" in ts_data:
            sync_issues = self.check_narration_synchronization(code, ts_data)
            issues.extend(sync_issues)

        if not layout_found:
            status = "METADATA_MISSING"
            is_clean = False
            summary = "Layout metadata missing; geometry validation incomplete."
            if code_issues:
                summary += f" Detected {len(code_issues)} static code issue(s)."
        elif len(issues) == 0:
            status = "CLEAN"
            is_clean = True
            summary = "Clean layout: no boundary breaches, collisions, or synchronization issues detected."
        else:
            status = "ISSUES_FOUND"
            is_clean = False
            summary = (
                f"Detected {len(issues)} local visual layout and synchronization issue(s): " +
                ", ".join(sorted(set(i.issue_type for i in issues)))
            )

        return VisualCheckResult(is_clean=is_clean, issues=issues, summary=summary, status=status)

    def check_visual_complexity(
        self,
        mobjects: List[Dict[str, Any]],
        checkpoint_name: str = "frame",
        timestamp: Optional[float] = None,
        max_focal_units: int = 4,
    ) -> List[VisualIssue]:
        """
        Evaluates visual complexity in a context-aware manner (Requirement 2).
        - Composite components (GraphVisual, DPTableMatrix, MoleculeVisual, ReactionEnergyDiagram, FunctionCurve, etc.)
          count as 1 single focal unit, NOT dozens of individual nodes/edges/atoms.
        - Submobjects and labels strictly contained inside a container or composite component are absorbed.
        - Relational connectors (arrows, vectors, connecting lines) are excluded from the focal unit count.
        - Generates COMPLEXITY_OVERLOAD with structured evidence if independent units exceed max_focal_units.
        """
        issues: List[VisualIssue] = []
        canvas_mobjects = [m for m in mobjects if m.get("tag") == "canvas"]
        if not canvas_mobjects:
            return issues

        # Filter out connectors (arrows, vectors, line connectors)
        non_connector_mobjects = [
            m for m in canvas_mobjects
            if m.get("class") not in self.LINE_TYPES and "Tip" not in m.get("class", "")
        ]

        focal_groups = []
        visited = set()

        for idx, m in enumerate(non_connector_mobjects):
            if idx in visited:
                continue
            cls = m.get("class", "Mobject")

            # Composite domain components count as a single focal unit and absorb inner elements
            if cls in self.COMPOSITE_COMPONENT_TYPES:
                group = [m]
                visited.add(idx)
                for jdx, other in enumerate(non_connector_mobjects):
                    if jdx not in visited and self._contains(m, other, tol=0.25):
                        group.append(other)
                        visited.add(jdx)
                focal_groups.append(group)

            # Container shapes (Circle, Rectangle, Box) containing a label/text
            elif cls in self.CONTAINER_TYPES:
                group = [m]
                visited.add(idx)
                for jdx, other in enumerate(non_connector_mobjects):
                    if jdx not in visited and self._contains(m, other, tol=0.10):
                        group.append(other)
                        visited.add(jdx)
                focal_groups.append(group)

            else:
                # Standalone text, formula, or shape
                focal_groups.append([m])
                visited.add(idx)

        num_focal_units = len(focal_groups)
        if num_focal_units > max_focal_units:
            unit_classes = [g[0].get("class", "Mobject") for g in focal_groups]
            time_str = f" @ t={timestamp:.2f}s" if timestamp is not None else ""
            issues.append(VisualIssue(
                issue_type="COMPLEXITY_OVERLOAD",
                severity="warning",
                description=(
                    f"[{checkpoint_name}{time_str}] Visual complexity overload: {num_focal_units} independent focal units on screen "
                    f"(recommended limit: {max_focal_units}). Units: {', '.join(unit_classes[:6])}"
                ),
                details={
                    "checkpoint": checkpoint_name,
                    "timestamp": timestamp,
                    "num_focal_units": num_focal_units,
                    "max_focal_units": max_focal_units,
                    "unit_classes": unit_classes,
                },
                can_fix_locally=False,
                evidence={
                    "checkpoint": checkpoint_name,
                    "timestamp": timestamp,
                    "metric": "focal_units",
                    "expected": f"<= {max_focal_units}",
                    "actual": num_focal_units,
                    "component_units": unit_classes,
                }
            ))

        return issues

    def check_trajectory_collisions(
        self,
        checkpoints: List[Dict[str, Any]],
        overlap_threshold: float = 0.35,
    ) -> List[VisualIssue]:
        """
        Inspects consecutive checkpoints (CP_k -> CP_{k+1}) to detect dynamic collisions
        introduced when visual objects move or are added during animation transitions.
        """
        issues: List[VisualIssue] = []
        if not checkpoints or len(checkpoints) < 2:
            return issues

        for k in range(len(checkpoints) - 1):
            cp_prev = checkpoints[k]
            cp_curr = checkpoints[k + 1]

            cp_curr_name = cp_curr.get("checkpoint", f"step_{k+2}")
            cp_curr_time = cp_curr.get("time", 0.0)

            mobs_prev = [m for m in cp_prev.get("mobjects", []) if m.get("tag") == "canvas"]
            mobs_curr = [m for m in cp_curr.get("mobjects", []) if m.get("tag") == "canvas"]

            # Map previous mobjects by identifier (class + approximate width/height if index differs)
            prev_centers = {}
            for m in mobs_prev:
                m_id = (m.get("class", ""), round(m.get("width", 0.0), 2), round(m.get("height", 0.0), 2))
                prev_centers[m_id] = m.get("center", [0.0, 0.0, 0.0])

            # Check each current mobject
            for i, m1 in enumerate(mobs_curr):
                cls1 = m1.get("class", "")
                w1 = m1.get("width", 0.0)
                h1 = m1.get("height", 0.0)
                area1 = w1 * h1
                if area1 < 0.005:
                    continue

                m1_id = (cls1, round(w1, 2), round(h1, 2))
                c1 = m1.get("center", [0.0, 0.0, 0.0])

                is_new = m1_id not in prev_centers
                has_moved = False
                movement_dist = 0.0
                if not is_new:
                    prev_c = prev_centers[m1_id]
                    movement_dist = ((c1[0] - prev_c[0])**2 + (c1[1] - prev_c[1])**2)**0.5
                    has_moved = movement_dist > 0.15

                # If object is newly introduced or has moved, test for collision with all other objects
                if is_new or has_moved:
                    for j, m2 in enumerate(mobs_curr):
                        if i == j:
                            continue
                        cls2 = m2.get("class", "")
                        w2 = m2.get("width", 0.0)
                        h2 = m2.get("height", 0.0)
                        area2 = w2 * h2
                        if area2 < 0.005:
                            continue

                        if self._is_intentional_composition(cls1, cls2, m1, m2):
                            continue

                        x_overlap = max(0.0, min(m1.get("right", 0.0), m2.get("right", 0.0)) -
                                             max(m1.get("left", 0.0), m2.get("left", 0.0)))
                        y_overlap = max(0.0, min(m1.get("top", 0.0), m2.get("top", 0.0)) -
                                             max(m1.get("bottom", 0.0), m2.get("bottom", 0.0)))
                        overlap_area = x_overlap * y_overlap
                        min_area = min(area1, area2)

                        if min_area > 0 and (overlap_area / min_area) > overlap_threshold:
                            ratio = round(overlap_area / min_area, 2)
                            action_str = "newly introduced" if is_new else f"moved ({movement_dist:.2f} units)"
                            issues.append(VisualIssue(
                                issue_type="OVERLAP",
                                severity="error" if ratio > 0.6 else "warning",
                                description=(
                                    f"[{cp_curr_name} @ t={cp_curr_time:.2f}s] Dynamic trajectory collision: {cls1} ({action_str}) "
                                    f"collides with {cls2} (overlap ratio: {ratio*100:.0f}%)"
                                ),
                                details={
                                    "checkpoint": cp_curr_name,
                                    "timestamp": cp_curr_time,
                                    "m1": m1,
                                    "m2": m2,
                                    "overlap_ratio": ratio,
                                    "action": action_str,
                                },
                                can_fix_locally=False,
                                evidence={
                                    "checkpoint": cp_curr_name,
                                    "timestamp": cp_curr_time,
                                    "m1": cls1,
                                    "m2": cls2,
                                    "overlap_ratio": ratio,
                                    "action": action_str,
                                    "metric": "trajectory_overlap",
                                }
                            ))

        return issues

    def check_multiframe_checkpoints(
        self,
        checkpoints: List[Dict[str, Any]],
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None,
    ) -> List[VisualIssue]:
        """
        Comprehensive multi-frame inspection across intermediate animation checkpoints:
        1. Evaluates bounding box geometry, boundary clearances, and mutual overlaps at each checkpoint.
        2. Evaluates context-aware visual complexity (focal units) per checkpoint.
        3. Analyzes dynamic trajectory collisions across checkpoint transitions.
        """
        issues: List[VisualIssue] = []
        if not checkpoints or not isinstance(checkpoints, list):
            return issues

        for cp in checkpoints:
            cp_name = cp.get("checkpoint", "checkpoint")
            cp_time = cp.get("time", 0.0)
            cp_mobs = cp.get("mobjects", [])
            if isinstance(cp_mobs, list) and cp_mobs:
                geom_issues = self._check_mobjects_geometry(cp_mobs, frame_width=frame_width, frame_height=frame_height)
                for ci in geom_issues:
                    ci.description = f"[{cp_name} @ t={cp_time:.1f}s] {ci.description}"
                    if not ci.evidence:
                        ci.evidence = {}
                    ci.evidence["checkpoint"] = cp_name
                    ci.evidence["timestamp"] = cp_time
                issues.extend(geom_issues)

                comp_issues = self.check_visual_complexity(cp_mobs, checkpoint_name=cp_name, timestamp=cp_time)
                issues.extend(comp_issues)

        traj_issues = self.check_trajectory_collisions(checkpoints)
        issues.extend(traj_issues)

        return issues

    def check_narration_synchronization(
        self,
        code: str,
        timestamps_data: Dict[str, Any],
        tolerance_sec: float = 0.75,
    ) -> List[VisualIssue]:
        """
        Evaluates actual visual animation events against narration timestamps (Requirement 3).
        - Segments Manim code into lecture line blocks.
        - Parses visual animation calls (self.play), durations (run_time), self.wait, and caption activations.
        - Detects:
            1. Dead line / missing visual animations during spoken narration.
            2. Caption highlight misalignments (out-of-order activations).
            3. Duration budget overruns (animation duration exceeds line budget).
            4. Duration underruns (animations finish prematurely, leaving dead screen).
            5. Primary visual action onset lag (visual starts after narration ends).
        - Every issue includes structured evidence (Requirement 1).
        """
        issues: List[VisualIssue] = []
        lines_meta = timestamps_data.get("lines", [])
        if not lines_meta or not isinstance(lines_meta, list):
            return issues

        # Segment code into lecture line blocks
        code_lines = code.split("\n")
        header_pat = re.compile(r'#\s*===+\s*(?:Animation for\s+)?Lecture Line\s*(\d+)\s*===+', re.IGNORECASE)

        # Map block index (0-based) to line range (start_line_idx, end_line_idx)
        block_markers = []
        for line_idx, line_str in enumerate(code_lines):
            m = header_pat.search(line_str)
            if m:
                block_num = int(m.group(1)) - 1
                block_markers.append((block_num, line_idx))

        # If header comments exist, partition lines
        blocks = {}
        if block_markers:
            for idx, (b_num, start_idx) in enumerate(block_markers):
                end_idx = block_markers[idx + 1][1] if idx + 1 < len(block_markers) else len(code_lines)
                blocks[b_num] = {
                    "start_line": start_idx + 1,
                    "end_line": end_idx,
                    "text": "\n".join(code_lines[start_idx:end_idx]),
                }
        else:
            # Fallback partitioning: search for self.activate_caption(N) or self.lecture[N]
            cap_pat = re.compile(r'self\.(?:activate_caption\(\s*(\d+)|lecture\[\s*(\d+)\s*\])')
            cap_markers = []
            for line_idx, line_str in enumerate(code_lines):
                m = cap_pat.search(line_str)
                if m:
                    c_num = int(m.group(1) if m.group(1) is not None else m.group(2))
                    cap_markers.append((c_num, line_idx))
            for idx, (c_num, start_idx) in enumerate(cap_markers):
                end_idx = cap_markers[idx + 1][1] if idx + 1 < len(cap_markers) else len(code_lines)
                blocks[c_num] = {
                    "start_line": start_idx + 1,
                    "end_line": end_idx,
                    "text": "\n".join(code_lines[start_idx:end_idx]),
                }

        # Analyze each lecture line
        for i, line_info in enumerate(lines_meta):
            line_idx = line_info.get("line_index", i)
            spoken_duration = float(line_info.get("duration", 0.0))
            spoken_start = float(line_info.get("start", 0.0))
            spoken_end = float(line_info.get("end", spoken_start + spoken_duration))
            total_budget = float(line_info.get("total_budget", spoken_duration + 0.5))

            if line_idx not in blocks:
                # Missing block in code for this lecture line
                issues.append(VisualIssue(
                    issue_type="SYNC_MISSING_BLOCK",
                    severity="error",
                    description=f"Lecture line {line_idx + 1} has narration ({spoken_duration:.1f}s) but no corresponding code block found",
                    details={"line_index": line_idx, "spoken_duration": spoken_duration},
                    can_fix_locally=False,
                    evidence={
                        "line_index": line_idx,
                        "spoken_duration": spoken_duration,
                        "spoken_start": spoken_start,
                        "metric": "block_presence",
                    }
                ))
                continue

            block = blocks[line_idx]
            block_text = block["text"]
            block_start_line = block["start_line"]

            # Parse self.play and self.wait calls in true chronological appearance order
            raw_events = []
            # Find play calls with positions
            start = 0
            while True:
                idx = block_text.find("self.play(", start)
                if idx == -1:
                    break
                i = idx + len("self.play(")
                depth = 1
                while i < len(block_text) and depth > 0:
                    if block_text[i] == '(':
                        depth += 1
                    elif block_text[i] == ')':
                        depth -= 1
                    i += 1
                call_text = block_text[idx:i]
                rt_match = re.search(r'run_time\s*=\s*([0-9.]+)', call_text)
                rt = float(rt_match.group(1)) if rt_match else 1.0
                anim_names = re.findall(
                    r'\b(FadeIn|FadeOut|Create|Write|Transform|ReplacementTransform|Indicate|GrowFromCenter|animate_\w+)\b',
                    call_text
                )
                raw_events.append((idx, {
                    "type": "play",
                    "duration": rt,
                    "animations": anim_names,
                    "call_snippet": call_text[:60].replace("\n", " ")
                }))
                start = max(i, idx + len("self.play("))

            # Find wait calls with positions
            for m in re.finditer(r'self\.wait\(\s*([0-9.]+)?\s*\)', block_text):
                idx = m.start()
                w_dur = float(m.group(1)) if m.group(1) else 1.0
                raw_events.append((idx, {
                    "type": "wait",
                    "duration": w_dur,
                    "call_snippet": f"self.wait({w_dur})"
                }))

            # Sort events by appearance order in code
            raw_events.sort(key=lambda x: x[0])
            events = [e[1] for e in raw_events]
            play_calls = [e for e in events if e["type"] == "play"]
            total_anim_duration = sum(e["duration"] for e in events)

            # Check 1: Dead line check (narration occurs with no visual animation)
            if spoken_duration > 1.5 and len(play_calls) == 0:
                issues.append(VisualIssue(
                    issue_type="SYNC_MISSING_VISUAL",
                    severity="warning",
                    description=(
                        f"Lecture line {line_idx + 1} has {spoken_duration:.1f}s narration but zero animation events "
                        f"(screen remains static during spoken explanation)"
                    ),
                    details={
                        "line_index": line_idx,
                        "spoken_duration": spoken_duration,
                        "block_start_line": block_start_line,
                    },
                    can_fix_locally=False,
                    evidence={
                        "line_index": line_idx,
                        "code_line": block_start_line,
                        "spoken_duration": spoken_duration,
                        "animations_found": 0,
                        "metric": "visual_presence",
                    }
                ))

            # Check 2: Caption highlight alignment
            cap_activations = re.findall(r'self\.activate_caption\(\s*(\d+)', block_text)
            if not cap_activations:
                cap_activations = re.findall(r'self\.lecture\[\s*(\d+)\s*\]', block_text)
            for ca in cap_activations:
                cap_idx = int(ca)
                if cap_idx != line_idx:
                    issues.append(VisualIssue(
                        issue_type="SYNC_CAPTION_MISMATCH",
                        severity="error",
                        description=(
                            f"Lecture line {line_idx + 1} activates caption beat index {cap_idx + 1} instead of {line_idx + 1} "
                            f"(caption out of sync with narration)"
                        ),
                        details={
                            "line_index": line_idx,
                            "expected_caption_index": line_idx,
                            "actual_caption_index": cap_idx,
                            "block_start_line": block_start_line,
                        },
                        can_fix_locally=True,
                        evidence={
                            "line_index": line_idx,
                            "code_line": block_start_line,
                            "expected_caption": line_idx,
                            "actual_caption": cap_idx,
                            "metric": "caption_alignment",
                        }
                    ))

            # Check 3: Duration budget overrun
            if total_anim_duration > total_budget + tolerance_sec:
                delta = round(total_anim_duration - total_budget, 2)
                issues.append(VisualIssue(
                    issue_type="SYNC_DURATION_OVERRUN",
                    severity="error" if delta > 2.5 else "warning",
                    description=(
                        f"Lecture line {line_idx + 1} animation duration ({total_anim_duration:.2f}s) overruns narration budget "
                        f"({total_budget:.2f}s) by {delta:.2f}s"
                    ),
                    details={
                        "line_index": line_idx,
                        "total_anim_duration": total_anim_duration,
                        "total_budget": total_budget,
                        "delta": delta,
                        "block_start_line": block_start_line,
                    },
                    can_fix_locally=False,
                    evidence={
                        "line_index": line_idx,
                        "code_line": block_start_line,
                        "expected": total_budget,
                        "actual": total_anim_duration,
                        "delta": f"+{delta}s",
                        "metric": "duration_budget",
                    }
                ))

            # Check 4: Duration underrun / premature end
            if spoken_duration >= 4.0 and total_anim_duration < spoken_duration - 2.0:
                gap = round(spoken_duration - total_anim_duration, 2)
                issues.append(VisualIssue(
                    issue_type="SYNC_DURATION_UNDERRUN",
                    severity="warning",
                    description=(
                        f"Lecture line {line_idx + 1} animation finishes at {total_anim_duration:.2f}s, leaving {gap:.2f}s "
                        f"of unillustrated narration before the line ends"
                    ),
                    details={
                        "line_index": line_idx,
                        "total_anim_duration": total_anim_duration,
                        "spoken_duration": spoken_duration,
                        "gap": gap,
                        "block_start_line": block_start_line,
                    },
                    can_fix_locally=False,
                    evidence={
                        "line_index": line_idx,
                        "code_line": block_start_line,
                        "expected": spoken_duration,
                        "actual": total_anim_duration,
                        "delta": f"-{gap}s",
                        "metric": "duration_underrun",
                    }
                ))

            # Check 5: Primary visual action onset alignment
            t_rel = 0.0
            for ev in events:
                if ev["type"] == "play" and ev.get("animations"):
                    # Check if this is a primary animation (not just caption coloring)
                    is_caption_only = any("lecture" in ev.get("call_snippet", "").lower() for a in ev["animations"])
                    if not is_caption_only:
                        t_abs = spoken_start + t_rel
                        if t_abs > spoken_end + 0.5:
                            anim_repr = ", ".join(ev["animations"][:3])
                            issues.append(VisualIssue(
                                issue_type="SYNC_EVENT_LAG",
                                severity="warning",
                                description=(
                                    f"Lecture line {line_idx + 1} primary animation '{anim_repr}' triggers late at t={t_abs:.2f}s, "
                                    f"after spoken narration ended at t={spoken_end:.2f}s"
                                ),
                                details={
                                    "line_index": line_idx,
                                    "event_onset": t_abs,
                                    "spoken_end": spoken_end,
                                    "animation": anim_repr,
                                    "block_start_line": block_start_line,
                                },
                                can_fix_locally=False,
                                evidence={
                                    "line_index": line_idx,
                                    "code_line": block_start_line,
                                    "event_onset": round(t_abs, 2),
                                    "narration_end": round(spoken_end, 2),
                                    "animation": anim_repr,
                                    "metric": "event_onset",
                                }
                            ))
                            break
                t_rel += ev["duration"]

        return issues

    def _check_mobjects_geometry(
        self,
        mobjects: List[Dict[str, Any]],
        frame_width: Optional[float] = None,
        frame_height: Optional[float] = None
    ) -> List[VisualIssue]:
        issues: List[VisualIssue] = []
        canvas_mobjects = [m for m in mobjects if m.get("tag") == "canvas"]

        # Dynamic frame boundaries if metadata specifies custom dimensions
        fw = float(frame_width) if frame_width is not None else self.FRAME_WIDTH
        fh = float(frame_height) if frame_height is not None else self.FRAME_HEIGHT
        right_limit = min((fw / 2.0) - self.safety_margin, self.SAFE_RIGHT)
        left_limit = -(fw / 2.0) + self.safety_margin
        safe_top = min(self.SAFE_TOP, (fh / 2.0) - self.safety_margin)
        safe_bottom = max(self.SAFE_BOTTOM, -(fh / 2.0) + self.safety_margin)

        # A. Out-of-screen & safe boundary checks
        for m in canvas_mobjects:
            cls_name = m.get("class", "Mobject")
            left = m.get("left", 0.0)
            right = m.get("right", 0.0)
            top = m.get("top", 0.0)
            bottom = m.get("bottom", 0.0)
            width = m.get("width", 0.0)
            height = m.get("height", 0.0)

            # Skip tiny invisible anchor dots or zero-sized helper objects
            if width < 0.02 and height < 0.02:
                continue

            # Right frame boundary overflow
            if right > right_limit:
                overflow = round(right - right_limit, 2)
                issues.append(VisualIssue(
                    issue_type="OUT_OF_SCREEN",
                    severity="error" if overflow > 0.3 else "warning",
                    description=f"{cls_name} extends {overflow} units beyond right screen boundary (x={right:.2f} > {right_limit:.2f})",
                    details={"boundary": "right", "overflow": overflow, "mobject": m},
                    can_fix_locally=True,
                    suggested_fix={"action": "shift_left_or_scale", "object_class": cls_name, "overflow": overflow},
                    evidence={"boundary": "right", "overflow": overflow, "class": cls_name, "coord": right, "limit": right_limit, "metric": "boundary_limit"}
                ))

            # Left screen boundary breach (frame edge), else lecture column intrusion (see elif below)
            if left < left_limit:
                overflow = round(left_limit - left, 2)
                issues.append(VisualIssue(
                    issue_type="OUT_OF_SCREEN",
                    severity="error" if overflow > 0.3 else "warning",
                    description=f"{cls_name} extends {overflow} units beyond left screen boundary (x={left:.2f} < {left_limit:.2f})",
                    details={"boundary": "left", "overflow": overflow, "mobject": m},
                    can_fix_locally=True,
                    suggested_fix={"action": "shift_right", "object_class": cls_name, "overflow": overflow},
                    evidence={"boundary": "left", "overflow": overflow, "class": cls_name, "coord": left, "limit": left_limit, "metric": "boundary_limit"}
                ))

            # Lecture caption column intrusion (hint only: cannot be fixed by a local regex edit)
            elif left < self.SAFE_LEFT - 0.05:
                overflow = round(self.SAFE_LEFT - left, 2)
                issues.append(VisualIssue(
                    issue_type="OUT_OF_SCREEN",
                    severity="error" if overflow > 0.3 else "warning",
                    description=f"{cls_name} intrudes {overflow} units into the left lecture-caption column (x={left:.2f} < {self.SAFE_LEFT:.2f})",
                    details={"boundary": "lecture_column", "overflow": overflow, "mobject": m},
                    can_fix_locally=False,
                    suggested_fix={"action": "shift_right", "object_class": cls_name, "overflow": overflow},
                    evidence={"boundary": "lecture_column", "overflow": overflow, "class": cls_name, "coord": left, "limit": self.SAFE_LEFT, "metric": "boundary_limit"}
                ))

            # Title buffer intrusion (y > safe_top)
            if top > safe_top:
                overflow = round(top - safe_top, 2)
                issues.append(VisualIssue(
                    issue_type="OUT_OF_SCREEN",
                    severity="warning",
                    description=f"{cls_name} intrudes {overflow} units into the top title reserved area (y={top:.2f} > {safe_top:.2f})",
                    details={"boundary": "top_title", "overflow": overflow, "mobject": m},
                    can_fix_locally=True,
                    suggested_fix={"action": "shift_down", "object_class": cls_name, "overflow": overflow},
                    evidence={"boundary": "top_title", "overflow": overflow, "class": cls_name, "coord": top, "limit": safe_top, "metric": "boundary_limit"}
                ))

            # Subtitle buffer intrusion (y < safe_bottom)
            if bottom < safe_bottom:
                overflow = round(safe_bottom - bottom, 2)
                issues.append(VisualIssue(
                    issue_type="OUT_OF_SCREEN",
                    severity="warning",
                    description=f"{cls_name} intrudes {overflow} units into the bottom subtitle reserved area (y={bottom:.2f} < {safe_bottom:.2f})",
                    details={"boundary": "bottom_subtitle", "overflow": overflow, "mobject": m},
                    can_fix_locally=True,
                    suggested_fix={"action": "shift_up", "object_class": cls_name, "overflow": overflow},
                    evidence={"boundary": "bottom_subtitle", "overflow": overflow, "class": cls_name, "coord": bottom, "limit": safe_bottom, "metric": "boundary_limit"}
                ))

        # B. Obvious overlap checks between distinct canvas elements
        n = len(canvas_mobjects)
        for i in range(n):
            m1 = canvas_mobjects[i]
            cls1 = m1.get("class", "")
            w1 = m1.get("width", 0.0)
            h1 = m1.get("height", 0.0)
            area1 = w1 * h1
            if area1 < 0.005:
                continue

            for j in range(i + 1, n):
                m2 = canvas_mobjects[j]
                cls2 = m2.get("class", "")
                w2 = m2.get("width", 0.0)
                h2 = m2.get("height", 0.0)
                area2 = w2 * h2
                if area2 < 0.005:
                    continue

                # Filter out intentional compositions
                if self._is_intentional_composition(cls1, cls2, m1, m2):
                    continue

                # Compute 2D axis-aligned bounding box intersection
                x_overlap = max(0.0, min(m1.get("right", 0.0), m2.get("right", 0.0)) -
                                     max(m1.get("left", 0.0), m2.get("left", 0.0)))
                y_overlap = max(0.0, min(m1.get("top", 0.0), m2.get("top", 0.0)) -
                                     max(m1.get("bottom", 0.0), m2.get("bottom", 0.0)))
                overlap_area = x_overlap * y_overlap
                min_area = min(area1, area2)

                # Meaningful overlap threshold: > 35% of the smaller object
                if min_area > 0 and (overlap_area / min_area) > 0.35:
                    ratio = round(overlap_area / min_area, 2)
                    c1 = m1.get("center", [0.0, 0.0, 0.0])
                    c2 = m2.get("center", [0.0, 0.0, 0.0])
                    cell1 = self.coordinates_to_grid_cell(c1[0], c1[1])
                    cell2 = self.coordinates_to_grid_cell(c2[0], c2[1])
                    issues.append(VisualIssue(
                        issue_type="OVERLAP",
                        severity="error" if ratio > 0.6 else "warning",
                        description=f"Obvious overlap between {cls1} and {cls2} (overlap ratio: {ratio*100:.0f}%)",
                        details={"m1": m1, "m2": m2, "overlap_ratio": ratio, "cell": cell2, "cell1": cell1, "cell2": cell2},
                        can_fix_locally=True,
                        suggested_fix={"action": "separate_grid_cells", "cls1": cls1, "cls2": cls2, "cell": cell2},
                        evidence={"overlap_ratio": ratio, "m1": cls1, "m2": cls2, "cell1": cell1, "cell2": cell2, "metric": "bounding_box_overlap"}
                    ))

        # C. Intra-group and composite component internal overlap detection
        for m in canvas_mobjects:
            children = m.get("children", [])
            parent_cls = m.get("class", "Component")
            if not children or len(children) < 2:
                continue

            # Stacking check: >= 2 non-line elements at identical position
            center_counts: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
            for ch in children:
                c = ch.get("center", [0.0, 0.0])
                key = (round(c[0], 2), round(c[1], 2))
                center_counts.setdefault(key, []).append(ch)

            for (cx, cy), group in center_counts.items():
                if len(group) >= 2:
                    non_line_group = [
                        ch for ch in group
                        if ch.get("class", "") not in self.LINE_TYPES and "Tip" not in ch.get("class", "")
                    ]
                    LABEL_TYPES = {"Text", "MathTex", "Tex", "SafeMathTex", "Dot", "Integer", "DecimalNumber", "Paragraph"}
                    card_or_box_items = [
                        ch for ch in non_line_group
                        if ch.get("class", "") not in LABEL_TYPES and ("Card" in ch.get("class", "") or "Visual" in ch.get("class", "") or ch.get("width", 0) > 0.5)
                    ]
                    stacked_labels = [
                        ch for ch in non_line_group
                        if ch.get("class", "") in LABEL_TYPES
                    ]
                    if len(card_or_box_items) >= 2 or len(stacked_labels) >= 2 or len(non_line_group) >= 3:
                        cell = self.coordinates_to_grid_cell(cx, cy)
                        issues.append(VisualIssue(
                            issue_type="OVERLAP",
                            severity="error",
                            description=f"Severe intra-group stacking in {parent_cls}: {len(non_line_group)} elements placed at identical position ({cx}, {cy}) in cell {cell}",
                            details={"mobject": m, "stacked_count": len(non_line_group), "cell": cell, "center": [cx, cy]},
                            can_fix_locally=False,
                            suggested_fix={"action": "distribute_grid_cells", "cell": cell},
                            evidence={"stacked_count": len(non_line_group), "cell": cell, "center": [cx, cy], "metric": "intra_group_stacking"}
                        ))

            # Pairwise child bounding box overlap checking
            num_ch = len(children)
            for ci in range(num_ch):
                ch1 = children[ci]
                c_cls1 = ch1.get("class", "")
                w1 = ch1.get("width", 0.0)
                h1 = ch1.get("height", 0.0)
                area1 = w1 * h1
                if area1 < 0.01:
                    continue

                for cj in range(ci + 1, num_ch):
                    ch2 = children[cj]
                    c_cls2 = ch2.get("class", "")
                    w2 = ch2.get("width", 0.0)
                    h2 = ch2.get("height", 0.0)
                    area2 = w2 * h2
                    if area2 < 0.01:
                        continue

                    if self._is_intentional_child_composition(c_cls1, c_cls2, ch1, ch2, parent_cls):
                        continue

                    left1 = ch1.get("left", ch1.get("center", [0, 0])[0] - w1 / 2.0)
                    right1 = ch1.get("right", ch1.get("center", [0, 0])[0] + w1 / 2.0)
                    top1 = ch1.get("top", ch1.get("center", [0, 0])[1] + h1 / 2.0)
                    bottom1 = ch1.get("bottom", ch1.get("center", [0, 0])[1] - h1 / 2.0)

                    left2 = ch2.get("left", ch2.get("center", [0, 0])[0] - w2 / 2.0)
                    right2 = ch2.get("right", ch2.get("center", [0, 0])[0] + w2 / 2.0)
                    top2 = ch2.get("top", ch2.get("center", [0, 0])[1] + h2 / 2.0)
                    bottom2 = ch2.get("bottom", ch2.get("center", [0, 0])[1] - h2 / 2.0)

                    x_ov = max(0.0, min(right1, right2) - max(left1, left2))
                    y_ov = max(0.0, min(top1, top2) - max(bottom1, bottom2))
                    ov_area = x_ov * y_ov
                    min_a = min(area1, area2)

                    if min_a > 0 and (ov_area / min_a) > 0.45:
                        ratio = round(ov_area / min_a, 2)
                        c1 = ch1.get("center", [0.0, 0.0])
                        cell = self.coordinates_to_grid_cell(c1[0], c1[1])
                        issues.append(VisualIssue(
                            issue_type="OVERLAP",
                            severity="error" if ratio > 0.65 else "warning",
                            description=f"Internal overlap within {parent_cls}: {c_cls1} and {c_cls2} collide (overlap ratio: {ratio*100:.0f}%)",
                            details={"parent": parent_cls, "child1": ch1, "child2": ch2, "overlap_ratio": ratio, "cell": cell},
                            can_fix_locally=False,
                            suggested_fix={"action": "clean_up_internal_components", "parent": parent_cls},
                            evidence={"overlap_ratio": ratio, "parent": parent_cls, "child1": c_cls1, "child2": c_cls2, "cell": cell, "metric": "internal_child_overlap"}
                        ))

        # D. Readability checks for text mobjects (appropriate for 854x480 resolution)
        for m in canvas_mobjects:
            cls_name = m.get("class", "")
            if any(t in cls_name for t in ("Text", "MathTex", "Tex", "Paragraph", "Title", "SafeMathTex")):
                h = m.get("height", 0.0)
                w = m.get("width", 0.0)
                is_math = any(t in cls_name for t in ("MathTex", "Tex", "SafeMathTex"))
                # Math formulas are naturally horizontal and thin (~0.12-0.14 height for standard single-line formulas)
                min_h = 0.09 if is_math else 0.12
                if 0.02 < h < min_h:
                    issues.append(VisualIssue(
                        issue_type="READABILITY",
                        severity="warning",
                        description=f"{cls_name} element has tiny rendered height ({h:.2f} units < {min_h}), illegible at 854x480 resolution",
                        details={"mobject": m, "height": h},
                        can_fix_locally=False,
                        suggested_fix={"action": "increase_font_size", "object_class": cls_name},
                        evidence={"height": h, "threshold": min_h, "class": cls_name, "metric": "font_readability"}
                    ))

        # E. Unlabelled geometric shapes acting as diagram elements
        label_mobs = [
            m for m in canvas_mobjects
            if any(t in m.get("class", "") for t in ("Text", "Math", "Tex", "SafeMathTex", "Label"))
        ]
        RAW_SHAPE_TYPES = {"Rectangle", "RoundedRectangle", "Square", "Circle"}
        for m in canvas_mobjects:
            cls = m.get("class", "")
            if cls in RAW_SHAPE_TYPES:
                mw = m.get("width", 0.0)
                mh = m.get("height", 0.0)
                if mw > 0.8 and mh > 0.4:
                    has_child_label = any(
                        any(t in ch.get("class", "") for t in ("Text", "Math", "Tex", "SafeMathTex", "Label"))
                        for ch in m.get("children", [])
                    )
                    has_overlapping_label = any(
                        self._contains(m, lbl, tol=0.15) for lbl in label_mobs
                    )
                    if not has_child_label and not has_overlapping_label:
                        issues.append(VisualIssue(
                            issue_type="UNLABELLED_DIAGRAM_ELEMENT",
                            severity="warning",
                            description=f"Unlabelled {cls} (w={mw:.2f}, h={mh:.2f}) lacks explanatory text or mathematical label",
                            details={"mobject": m},
                            can_fix_locally=False,
                            suggested_fix={"action": "add_explanatory_label", "target": cls},
                            evidence={"metric": "unlabelled_shape", "class": cls, "width": mw, "height": mh}
                        ))

        return issues

    def _is_meaningful_canvas_mobject(self, mob: Dict[str, Any]) -> bool:
        """Determines whether a canvas mobject represents meaningful visible content,
        properly accounting for mathematical formulas, text, shapes, and composite components.
        Fixes false-positive BLANK_SCENE errors on single-line formulas."""
        if mob.get("tag") != "canvas":
            return False
        cls_name = mob.get("class", "")
        w = float(mob.get("width", 0.0))
        h = float(mob.get("height", 0.0))

        # Textual / mathematical expressions (e.g. SafeMathTex formulas have height ~0.135, width ~1.5)
        is_textual = any(t in cls_name for t in ("Tex", "Text", "Math", "Formula", "Label", "SafeMathTex", "SafeTable", "Table"))
        if is_textual:
            return (w > 0.25 and h > 0.06) or (w > 0.12 and h > 0.10)

        # Standard shapes or domain components
        return (w > 0.15 and h > 0.15) or ((w * h > 0.04) and (max(w, h) > 0.20))

    @staticmethod
    def coordinates_to_grid_cell(x: float, y: float) -> str:
        """Reverse-maps 2D canvas coordinates (x, y) to closest discrete 6x6 grid cell ('A1'..'F6')."""
        col = int(round((x - 0.0) / 1.05)) + 1
        col = max(1, min(6, col))
        row_idx = int(round((1.7 - y) / 0.85))
        row_idx = max(0, min(5, row_idx))
        rows = ["A", "B", "C", "D", "E", "F"]
        return f"{rows[row_idx]}{col}"

    @staticmethod
    def _contains(outer: dict, inner: dict, tol: float = 0.03) -> bool:
        try:
            return (outer["left"] - tol <= inner["left"] and outer["right"] + tol >= inner["right"] and
                    outer["bottom"] - tol <= inner["bottom"] and outer["top"] + tol >= inner["top"])
        except KeyError:
            return False

    def _is_intentional_child_composition(self, cls1: str, cls2: str, ch1: dict, ch2: dict, parent_cls: str) -> bool:
        """Returns True if the overlap between two child mobjects inside a composite is intentional."""
        if cls1 in self.LINE_TYPES or cls2 in self.LINE_TYPES or "Tip" in cls1 or "Tip" in cls2:
            return True

        # Text/Label inside a box or cell
        LABEL_TYPES = {"Text", "MathTex", "Tex", "SafeMathTex", "Dot", "Integer", "DecimalNumber", "Paragraph"}
        CONTAINER_TYPES = {"Rectangle", "RoundedRectangle", "Square", "Circle", "VGroup", "Cell"}

        # Text inside container
        if cls1 in LABEL_TYPES and (cls2 in CONTAINER_TYPES or cls2 in self.CONTAINER_TYPES):
            if self._contains(ch2, ch1, tol=0.15):
                return True
        if cls2 in LABEL_TYPES and (cls1 in CONTAINER_TYPES or cls1 in self.CONTAINER_TYPES):
            if self._contains(ch1, ch2, tol=0.15):
                return True

        # If both are labels, overlapping text is almost always an error
        if cls1 in LABEL_TYPES and cls2 in LABEL_TYPES:
            return False

        # If both are process cards, overlapping cards is always an error
        if "ProcessCard" in cls1 and "ProcessCard" in cls2:
            return False

        # Dedicated domain component internal compositions
        if parent_cls in ("ReadyQueueVisual", "CPUExecutionVisual", "GanttChartVisual", "MetricTableVisual", "ProcessCardVisual"):
            if ("ProcessCard" in cls1) != ("ProcessCard" in cls2):
                return True
            if any(r in cls1 for r in ("Rectangle", "VGroup", "Group")) and any(r in cls2 for r in ("Rectangle", "VGroup", "Group")):
                if self._contains(ch1, ch2, tol=0.25) or self._contains(ch2, ch1, tol=0.25):
                    return True

        # Container enclosing a subcontainer (e.g. core_box enclosing slot_box)
        w1, h1 = ch1.get("width", 0.0), ch1.get("height", 0.0)
        w2, h2 = ch2.get("width", 0.0), ch2.get("height", 0.0)
        if (w1 > 1.08 * w2 and h1 > 1.08 * h2) and self._contains(ch1, ch2, tol=0.20):
            return True
        if (w2 > 1.08 * w1 and h2 > 1.08 * h1) and self._contains(ch2, ch1, tol=0.20):
            return True

        # Dedicated wrappers
        WRAPPER_TYPES = {"SurroundingRectangle", "BackgroundRectangle", "Brace"}
        if cls1 in WRAPPER_TYPES or cls2 in WRAPPER_TYPES:
            return True

        return False

    def _is_intentional_composition(self, cls1: str, cls2: str, m1: dict, m2: dict) -> bool:
        """Returns True if the overlap is likely intentional (arrows, boxes, labels, containers)."""
        # Arrows or lines touching targets (arrow tips are separate submobjects in Manim CE 0.18)
        if cls1 in self.LINE_TYPES or cls2 in self.LINE_TYPES:
            return True
        if "Tip" in cls1 or "Tip" in cls2:
            return True

        # Coordinate systems (Axes, NumberPlane) containing plots, labels, or math
        COORDINATE_SYSTEMS = {"Axes", "NumberPlane", "ThreeDAxes"}
        if (cls1 in COORDINATE_SYSTEMS and self._contains(m1, m2, tol=0.15)) or (cls2 in COORDINATE_SYSTEMS and self._contains(m2, m1, tol=0.15)):
            return True

        # Composite domain components containing internal elements or labels
        if (cls1 in self.COMPOSITE_COMPONENT_TYPES and self._contains(m1, m2, tol=0.20)) or (cls2 in self.COMPOSITE_COMPONENT_TYPES and self._contains(m2, m1, tol=0.20)):
            return True

        # Dedicated enclosing wrappers
        WRAPPER_TYPES = {"SurroundingRectangle", "BackgroundRectangle", "Brace"}
        if (cls1 in WRAPPER_TYPES and self._contains(m1, m2, tol=0.1)) or (cls2 in WRAPPER_TYPES and self._contains(m2, m1, tol=0.1)):
            return True

        # Label or marker dot inside a container shape (Circle, Square, Rectangle, etc.)
        LABEL_TYPES = {"Text", "MathTex", "Tex", "SafeMathTex", "Dot", "Integer", "DecimalNumber"}
        if cls1 in self.CONTAINER_TYPES and cls2 in LABEL_TYPES and self._contains(m1, m2):
            return True
        if cls2 in self.CONTAINER_TYPES and cls1 in LABEL_TYPES and self._contains(m2, m1):
            return True

        # Table cells containing text/numbers
        TABLE_TYPES = {"Table", "MobjectTable", "IntegerTable", "DecimalTable"}
        if (cls1 in TABLE_TYPES and self._contains(m1, m2)) or (cls2 in TABLE_TYPES and self._contains(m2, m1)):
            return True

        # Process card dispatched into CPU core execution box or queue slot
        if (cls1 == "CPUExecutionVisual" and "ProcessCard" in cls2) or (cls2 == "CPUExecutionVisual" and "ProcessCard" in cls1):
            return True
        if ("ProcessCard" in cls1 or "ProcessCard" in cls2) and (
            any(k in (cls1, cls2) for k in ("CPUExecutionVisual", "ReadyQueueVisual", "RoundedRectangle", "Rectangle", "VGroup", "Group"))
        ):
            if not m1 or not m2 or self._contains(m1, m2, tol=0.25) or self._contains(m2, m1, tol=0.25):
                return True

        # Concentric status badges, borders, or transforms inside components
        if cls1 in ("RoundedRectangle", "Rectangle") and cls2 in ("RoundedRectangle", "Rectangle"):
            if self._contains(m1, m2, tol=0.20) or self._contains(m2, m1, tol=0.20):
                return True

        # VGroup / Group containment: intentional if one contains the other (nested groups / parent-child)
        if "Group" in cls1 or "Group" in cls2:
            if self._contains(m1, m2, tol=0.20) or self._contains(m2, m1, tol=0.20):
                return True
            ch1_classes = {c.get("class") for c in m1.get("children", [])}
            ch2_classes = {c.get("class") for c in m2.get("children", [])}
            if (ch1_classes == {"Line"} or ch2_classes == {"Line"}) and (
                any("Card" in c for c in (ch1_classes | ch2_classes)) or
                any("Text" in c for c in (ch1_classes | ch2_classes))
            ):
                return True
            if any("ProcessCard" in c for c in ch1_classes) or any("ProcessCard" in c for c in ch2_classes):
                if any("Line" in c or "Rectangle" in c for c in (ch1_classes | ch2_classes)):
                    return True
            if cls1 in ("VGroup", "Group") and cls2 in ("VGroup", "Group"):
                return False

        return False

    @staticmethod
    def _iter_call_texts(code: str, prefix: str):
        """Yield the full text of every `prefix ... )` call, honouring nested parentheses."""
        start = 0
        while True:
            idx = code.find(prefix, start)
            if idx == -1:
                return
            i = idx + len(prefix)
            depth = 1
            while i < len(code) and depth > 0:
                ch = code[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                i += 1
            yield code[idx:i]
            start = max(i, idx + len(prefix))

    def _check_code_grid_placements(self, code: str) -> List[VisualIssue]:
        """Statically inspects code for duplicate grid assignments and extreme scale factors."""
        issues: List[VisualIssue] = []
        lines = code.split("\n")

        # Pre-extract defined objects
        def_pat = re.compile(
            r'^\s*([a-zA-Z_]\w*)\s*=\s*(' + "|".join(self.TRACKED_CLASSES) + r')\((.*)$',
            re.MULTILINE
        )
        defined_objects: Dict[str, str] = {}
        for m in def_pat.finditer(code):
            name, cls, rest = m.group(1), m.group(2), m.group(3)
            # Lines/arrows built from explicit end points are positioned by construction
            if cls in self.POSITIONED_BY_CONSTRUCTION and re.search(
                    r'get_(?:center|left|right|top|bottom|start|end)\(|np\.array|\[', rest):
                continue
            defined_objects[name] = cls

        # Check for to_edge(LEFT) which intrudes into lecture caption area
        for line_num, line in enumerate(lines, 1):
            m = re.search(r'\b([a-zA-Z_]\w*)\.to_edge\(\s*LEFT\b', line)
            if m:
                obj_name = m.group(1).strip()
                # Never flag or rewrite framework lecture captions (self.lecture) or title positioning
                if obj_name in ("lecture", "title", "self") or line.strip().startswith("self.lecture"):
                    continue
                issues.append(VisualIssue(
                    issue_type="OUT_OF_SCREEN",
                    severity="warning",
                    description=f"Object '{obj_name}' uses to_edge(LEFT) which intrudes into lecture captions",
                    details={"object": obj_name, "line": line_num},
                    can_fix_locally=True,
                    suggested_fix={"action": "fix_to_edge_left", "object": obj_name, "line": line_num}
                ))

        # Check for next_to(self.lecture) which risks left margin collision
        for line_num, line in enumerate(lines, 1):
            m = re.search(r'\b([a-zA-Z_]\w*)\.next_to\(\s*self\.lecture', line)
            if m:
                obj_name = m.group(1).strip()
                if obj_name in ("lecture", "title", "self") or line.strip().startswith("self.lecture"):
                    continue
                issues.append(VisualIssue(
                    issue_type="BAD_PLACEMENT",
                    severity="warning",
                    description=f"Object '{obj_name}' positioned next_to self.lecture risking left margin collisions",
                    details={"object": obj_name, "line": line_num},
                    can_fix_locally=True,
                    suggested_fix={"action": "fix_next_to_lecture", "object": obj_name, "line": line_num}
                ))

        # Check for redundant duplicate title creation
        has_setup_layout = any("self.setup_layout(" in l for l in lines)
        if has_setup_layout:
            for line_num, line in enumerate(lines, 1):
                m = re.search(r'^\s*([a-zA-Z_]\w*)\s*=\s*(?:Text|Title)\([^)]*(?:Sec\s*\d+|Title|Redundant Title|to_edge\(UP\))', line)
                if not m and re.search(r'^\s*(title)\s*=\s*(?:Text|Title)\(', line):
                    m = re.search(r'^\s*(title)\s*=\s*(?:Text|Title)\(', line)
                if m:
                    title_var = m.group(1)
                    line_context = "\n".join(lines[max(0, line_num - 1):min(len(lines), line_num + 2)])
                    if "to_edge(UP)" in line_context or "to_corner(UP" in line_context or "Sec " in line:
                        issues.append(VisualIssue(
                            issue_type="BAD_PLACEMENT",
                            severity="warning",
                            description=f"Redundant title '{title_var}' created when self.setup_layout() already manages the title",
                            details={"object": title_var, "line": line_num},
                            can_fix_locally=True,
                            suggested_fix={"action": "fix_duplicate_title", "object": title_var, "line": line_num}
                        ))

        grid_call_pat = re.compile(
            r'self\.place_at_grid\(\s*([^,]+),\s*[\'"]([A-F][1-6])[\'"](?:,\s*scale_factor=([0-9.]+))?\s*\)'
        )

        grid_occupancy: Dict[str, List[Tuple[int, str, float]]] = {}
        for line_num, line in enumerate(lines, 1):
            m = grid_call_pat.search(line)
            if m:
                obj_name = m.group(1).strip()
                cell = m.group(2).strip().upper()
                scale = float(m.group(3)) if m.group(3) else 1.0

                if cell not in grid_occupancy:
                    grid_occupancy[cell] = []
                grid_occupancy[cell].append((line_num, obj_name, scale))

                # Check for row A placement with large scale factor (likely title collision)
                if cell.startswith("A") and scale > 1.05:
                    issues.append(VisualIssue(
                        issue_type="BAD_PLACEMENT",
                        severity="warning",
                        description=f"Object '{obj_name}' placed in Row A ({cell}) with large scale ({scale}) may intrude into title",
                        details={"cell": cell, "line": line_num, "object": obj_name, "scale": scale},
                        can_fix_locally=True,
                        suggested_fix={"action": "move_to_row_b_or_scale_down", "line": line_num, "cell": cell}
                    ))

                # Check for col 6 placement with large scale factor (likely right boundary overflow)
                if cell.endswith("6") and scale > 1.05:
                    issues.append(VisualIssue(
                        issue_type="OUT_OF_SCREEN",
                        severity="warning",
                        description=f"Object '{obj_name}' placed in Column 6 ({cell}) with large scale ({scale}) may breach right screen edge",
                        details={"cell": cell, "line": line_num, "object": obj_name, "scale": scale},
                        can_fix_locally=True,
                        suggested_fix={"action": "move_to_col_5_or_scale_down", "line": line_num, "cell": cell}
                    ))

        # Check for multiple active assignments to the exact same cell
        for cell, entries in grid_occupancy.items():
            if len(entries) > 1:
                obj_names = [e[1] for e in entries]
                first_line = entries[0][0]
                second_line = entries[1][0]
                # Check whether clear_canvas or FadeOut appears between the two assignments
                code_between = "\n".join(lines[first_line:second_line])
                if "clear_canvas" not in code_between and "FadeOut" not in code_between and "transition_to" not in code_between:
                    # Check if this is an intentional shape + label composition at the same cell
                    is_shape_and_label = False
                    if len(entries) == 2:
                        name1, name2 = entries[0][1], entries[1][1]
                        cls1 = defined_objects.get(name1, "")
                        cls2 = defined_objects.get(name2, "")
                        label_keywords = ("lbl", "label", "text", "txt", "caption", "name")
                        is_label1 = cls1 in ("Text", "MathTex", "Tex", "SafeMathTex") or any(k in name1.lower() for k in label_keywords)
                        is_label2 = cls2 in ("Text", "MathTex", "Tex", "SafeMathTex") or any(k in name2.lower() for k in label_keywords)
                        is_shape1 = cls1 in ("Circle", "Square", "Rectangle", "RoundedRectangle", "Dot", "Polygon", "Ellipse") or any(k in name1.lower() for k in ("circ", "rect", "box", "node", "shape"))
                        is_shape2 = cls2 in ("Circle", "Square", "Rectangle", "RoundedRectangle", "Dot", "Polygon", "Ellipse") or any(k in name2.lower() for k in ("circ", "rect", "box", "node", "shape"))
                        if (is_shape1 and is_label2) or (is_shape2 and is_label1):
                            is_shape_and_label = True

                    if not is_shape_and_label:
                        issues.append(VisualIssue(
                            issue_type="OVERLAP",
                            severity="warning",
                            description=f"Multiple objects {obj_names} placed at identical grid cell '{cell}' without clearing canvas",
                            details={"cell": cell, "lines": [first_line, second_line], "objects": obj_names},
                            can_fix_locally=True,
                            suggested_fix={"action": "relocate_duplicate_cell", "line": second_line, "cell": cell}
                        ))

        # Check for unpositioned objects played in animations (risking frame center 0,0 collision)
        played_objects = set()
        for call_text in self._iter_call_texts(code, "self.play("):
            for candidate in re.findall(r'\b([a-zA-Z_]\w*)\b', call_text):
                if candidate in defined_objects:
                    played_objects.add(candidate)

        positioned = set()
        # (a) object handed to a placement helper
        for m in re.finditer(r'self\.(?:place_at_grid|place_in_area|safe_label)\(\s*([a-zA-Z_]\w*)', code):
            positioned.add(m.group(1))
        # (b) receiver of a positioning method:  traveler.move_to(...), label[0].next_to(...)
        for m in re.finditer(
                r'\b([a-zA-Z_]\w*)(?:\[[^\]]*\])*\.(?:move_to|next_to|to_edge|to_corner|shift|align_to|set_x|set_y|set_z)\(',
                code):
            positioned.add(m.group(1))
        # (c) positioned in the same statement:  name = Text(...).next_to(...)
        for m in re.finditer(
                r'([a-zA-Z_]\w*)\s*=\s*[A-Z]\w*\(.*?\)\.(?:next_to|move_to|to_edge|to_corner|shift|align_to)\(', code):
            positioned.add(m.group(1))
        # (d) multiline VGroup / Group definitions whose arguments contain positioning or Line/Arrow
        for m in re.finditer(r'^\s*([a-zA-Z_]\w*)\s*=\s*(?:VGroup|Group)\(', code, re.MULTILINE):
            grp_name = m.group(1)
            start_pos = m.end() - 1
            depth = 1
            i = start_pos + 1
            while i < len(code) and depth > 0:
                if code[i] == '(':
                    depth += 1
                elif code[i] == ')':
                    depth -= 1
                i += 1
            grp_body = code[start_pos:i]
            if any(k in grp_body for k in [".next_to(", ".move_to(", ".shift(", "Line(", "Arrow(", "Dot(", "ORIGIN", "UP", "DOWN", "LEFT", "RIGHT", "np.array"]):
                positioned.add(grp_name)
        # (e) group membership: children of a placed group are placed with it; a group whose
        #     tracked children are all placed is placed as well
        children: Dict[str, set] = {}
        for m in re.finditer(r'^\s*([a-zA-Z_]\w*)\s*=\s*(?:VGroup|Group)\((.*)\)', code, re.MULTILINE):
            children[m.group(1)] = set(re.findall(r'(?<![\w.])\*?([a-zA-Z_]\w*)', m.group(2)))
        grew = True
        while grew:
            grew = False
            for parent, kids in children.items():
                if parent in positioned:
                    fresh = kids - positioned
                    if fresh:
                        positioned |= fresh
                        grew = True
                else:
                    tracked_kids = {k for k in kids if k in defined_objects}
                    if tracked_kids and tracked_kids <= positioned:
                        positioned.add(parent)
                        grew = True

        ignore_names = {"self", "FadeOut", "FadeIn", "Create", "Write", "Indicate", "Transform", "GrowFromCenter"}
        for obj in sorted(played_objects):
            if obj not in positioned and obj not in ignore_names:
                line_idx = 1
                for idx, l in enumerate(lines, 1):
                    if re.search(rf'\b{obj}\s*=', l):
                        line_idx = idx
                        break
                issues.append(VisualIssue(
                    issue_type="BAD_PLACEMENT",
                    severity="warning",
                    description=f"Object '{obj}' animated without position specification; defaults to (0,0) risking overlap",
                    details={"object": obj, "line": line_idx},
                    can_fix_locally=True,
                    suggested_fix={"action": "add_grid_placement", "object": obj, "line": line_idx}
                ))

        # Check for long MathTex equations without scale_to_fit_width
        long_tex_pat = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*MathTex\(\s*r?["\'](.*?)["\']', re.MULTILINE)
        for match in long_tex_pat.finditer(code):
            obj_name = match.group(1)
            raw_tex = match.group(2)
            if len(raw_tex) > 45:
                fitted_by_area = re.search(rf'self\.place_in_area\(\s*{re.escape(obj_name)}\b', code)
                if f"{obj_name}.scale" not in code and f"{obj_name}.scale_to_fit" not in code and not fitted_by_area:
                    line_idx = 1
                    for idx, l in enumerate(lines, 1):
                        if f"{obj_name} = MathTex" in l:
                            line_idx = idx
                            break
                    issues.append(VisualIssue(
                        issue_type="OUT_OF_SCREEN",
                        severity="warning",
                        description=f"Long MathTex equation '{obj_name}' ({len(raw_tex)} chars) without scaling may extend beyond screen margins",
                        details={"object": obj_name, "line": line_idx, "length": len(raw_tex)},
                        can_fix_locally=True,
                        suggested_fix={"action": "add_scale_to_fit_width", "object": obj_name, "line": line_idx}
                    ))

        # Check for edges (Line/Arrow) defined before their referenced nodes are placed with place_at_grid
        grid_lines: Dict[str, int] = {}
        for idx, l in enumerate(lines, 1):
            m = re.search(r'self\.place_at_grid\(\s*([^,]+),', l)
            if m:
                grid_lines[m.group(1).strip()] = idx

        for idx, l in enumerate(lines, 1):
            if any(k in l for k in ["Line(", "Arrow(", "DashedLine(", "DoubleArrow("]):
                for obj in re.findall(r'([a-zA-Z_]\w*(?:\[[^\]]+\])*)\.get_center\(\)', l):
                    base_obj = re.sub(r'\[0\]$', '', obj.strip())
                    line_pos = grid_lines.get(obj.strip(), grid_lines.get(base_obj))
                    if line_pos and line_pos > idx:
                        issues.append(VisualIssue(
                            issue_type="EDGES_BEFORE_NODES",
                            severity="error",
                            description=f"Edge on line {idx} references node '{obj}' before its grid placement on line {line_pos}; causes zero-length collapsed edges and overlapping labels",
                            details={"edge_line": idx, "node": obj, "place_line": line_pos},
                            can_fix_locally=True,
                            suggested_fix={"action": "reorder_nodes_before_edges", "edge_line": idx}
                        ))
                        break

        # Check if a node dictionary/list is defined and positioned, but never added/played into the scene
        has_nodes_def = re.search(r'^\s*(nodes|node_group)\s*=\s*(?:\{|\[)', code, re.MULTILINE)
        if has_nodes_def:
            node_var = has_nodes_def.group(1)
            is_added = (
                re.search(rf'self\.(?:play|add)\([^)]*\b{re.escape(node_var)}\b', code) or
                re.search(rf'for\s+\w+\s+in\s+{re.escape(node_var)}(?:\.values\(\))?\s*:\s*self\.add', code)
            )
            if not is_added:
                issues.append(VisualIssue(
                    issue_type="NODES_NOT_DISPLAYED",
                    severity="warning",
                    description=f"Collection '{node_var}' is defined and placed, but never added or animated into the scene",
                    details={"var_name": node_var},
                    can_fix_locally=True,
                    suggested_fix={"action": "add_missing_nodes", "var_name": node_var}
                ))

        # Check for clutter accumulation across lecture lines without cleanup
        blocks = re.split(r'#\s*===+\s*Animation for Lecture Line\s*\d+\s*===+', code, flags=re.IGNORECASE)
        if len(blocks) >= 4:
            accumulating_blocks = 0
            for blk in blocks[1:]:
                has_intro = any(k in blk for k in ["FadeIn", "Create", "Write", "GrowFromCenter"])
                has_cleanup = any(k in blk for k in ["FadeOut", "clear_canvas", "transition_to"])
                if has_intro and not has_cleanup:
                    accumulating_blocks += 1
                else:
                    accumulating_blocks = 0
                if accumulating_blocks >= 3:
                    issues.append(VisualIssue(
                        issue_type="CLUTTER_ACCUMULATION",
                        severity="warning",
                        description="Visual objects accumulate across 3+ consecutive lecture lines without FadeOut or clear_canvas transitions",
                        can_fix_locally=False
                    ))
                    break

        return issues


class LocalVisualFixer:
    """
    Applies deterministic code transformations to safely resolve local layout issues:
    - Moves Row A objects to Row B or reduces scale
    - Moves Column 6 objects to Column 5 or reduces scale
    - Relocates colliding duplicate grid assignments to adjacent free cells
    - Clamps oversized scale factors
    - Automatically adds place_at_grid for unpositioned objects
    - Adds scale_to_fit_width for long equations
    """

    def apply_fixes(self, code: str, issues: List[VisualIssue]) -> Tuple[str, bool]:
        """
        Applies deterministic corrections to code. Returns (modified_code, was_modified).
        """
        if not issues:
            return code, False

        modified = False
        new_code = code

        for issue in issues:
            if not issue.can_fix_locally:
                continue

            fix_info = issue.suggested_fix or {}
            action = fix_info.get("action", "")

            # 1. Title collision / Row A -> Row B
            if action in ("shift_down", "move_to_row_b_or_scale_down"):
                new_code, changed = self._fix_row_a_intrusion(new_code, issue)
                modified = modified or changed

            # 2. Right boundary breach / Column 6 -> Column 5
            elif action in ("shift_left_or_scale", "move_to_col_5_or_scale_down"):
                new_code, changed = self._fix_col_6_overflow(new_code, issue)
                modified = modified or changed

            # 3. Duplicate cell collision
            elif action in ("relocate_duplicate_cell", "separate_grid_cells"):
                new_code, changed = self._fix_duplicate_grid_cell(new_code, issue)
                modified = modified or changed

            # 4. Bottom boundary intrusion
            elif action == "shift_up":
                new_code, changed = self._fix_row_f_intrusion(new_code, issue)
                modified = modified or changed

            # 5. Unpositioned object placement
            elif action == "add_grid_placement":
                new_code, changed = self._fix_unpositioned_object(new_code, issue)
                modified = modified or changed

            # 6. Long equation scaling
            elif action == "add_scale_to_fit_width":
                new_code, changed = self._fix_equation_scale(new_code, issue)
                modified = modified or changed

            # 7. Edges defined before nodes placed
            elif action == "reorder_nodes_before_edges":
                new_code, changed = self._fix_reorder_nodes_before_edges(new_code, issue)
                modified = modified or changed

            # 8. Missing node rendering
            elif action == "add_missing_nodes":
                new_code, changed = self._fix_missing_nodes(new_code, issue)
                modified = modified or changed

            # 9. to_edge(LEFT) intrusion fix
            elif action == "fix_to_edge_left":
                new_code, changed = self._fix_to_edge_left(new_code, issue)
                modified = modified or changed

            # 10. next_to(self.lecture) intrusion fix
            elif action == "fix_next_to_lecture":
                new_code, changed = self._fix_next_to_lecture(new_code, issue)
                modified = modified or changed

            # 11. Duplicate title neutralization
            elif action == "fix_duplicate_title":
                new_code, changed = self._fix_duplicate_title(new_code, issue)
                modified = modified or changed

        # Safeguard: verify that deterministic layout edits produced syntactically valid Python code
        if modified:
            try:
                import ast
                import textwrap
                try:
                    ast.parse(new_code)
                except IndentationError:
                    ast.parse(textwrap.dedent(new_code))
            except Exception:
                # Layout fixes broke syntax/compilation; immediately rollback to clean original
                return code, False

        return new_code, modified

    def _fix_unpositioned_object(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """Inserts self.place_at_grid call for unpositioned object, avoiding grid collisions and syntax breaks."""
        obj = issue.details.get("object", "")
        if not obj:
            return code, False

        # If place_at_grid for this object is already present, do nothing
        if re.search(rf'self\.place_at_grid\(\s*{re.escape(obj)}\b', code):
            return code, False

        lines = code.split("\n")
        changed = False

        used_cells = set(re.findall(r'self\.place_at_grid\([^,]+,\s*[\'"]([A-F][1-6])[\'"]', code))
        candidate_cells = ['C3', 'C4', 'B3', 'B4', 'D3', 'D4', 'C2', 'C5']
        target_cell = next((c for c in candidate_cells if c not in used_cells), 'C3')

        for i, line in enumerate(lines):
            # Locate definition line: obj = ...
            if re.search(rf'^\s*{re.escape(obj)}\s*=', line):
                indent = re.match(r'^\s*', line).group(0)
                open_parens = line.count('(') - line.count(')')
                open_brackets = line.count('[') - line.count(']')
                open_braces = line.count('{') - line.count('}')
                insert_idx = i + 1
                while (open_parens > 0 or open_brackets > 0 or open_braces > 0) and insert_idx < len(lines):
                    next_line = lines[insert_idx]
                    open_parens += next_line.count('(') - next_line.count(')')
                    open_brackets += next_line.count('[') - next_line.count(']')
                    open_braces += next_line.count('{') - next_line.count('}')
                    insert_idx += 1

                grid_call = f"{indent}self.place_at_grid({obj}, '{target_cell}', scale_factor=0.85)"
                lines.insert(insert_idx, grid_call)
                changed = True
                break

        return "\n".join(lines), changed

    def _fix_equation_scale(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """Appends scale_to_fit_width for long MathTex object."""
        obj = issue.details.get("object", "")
        if not obj:
            return code, False
        lines = code.split("\n")
        changed = False

        for i, line in enumerate(lines):
            if f"{obj} = MathTex" in line:
                indent = re.match(r'^\s*', line).group(0)
                scale_call = f"{indent}{obj}.scale_to_fit_width(5.8)"
                lines.insert(i + 1, scale_call)
                changed = True
                break

        return "\n".join(lines), changed

    def _fix_row_a_intrusion(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """Replaces Row A placements with Row B or lowers scale factor."""
        lines = code.split("\n")
        changed = False

        # Pattern: self.place_at_grid(obj, 'A{col}', scale_factor=...)
        for i, line in enumerate(lines):
            m = re.search(r'(self\.place_at_grid\([^,]+,\s*[\'"])A([1-6])([\'"])(?:,\s*scale_factor=([0-9.]+))?(\s*\))', line)
            if m:
                prefix = m.group(1)
                col = m.group(2)
                suffix = m.group(3)
                old_scale = float(m.group(4)) if m.group(4) else 1.0
                end = m.group(5)

                new_scale = round(min(old_scale, 0.85), 2)
                replacement = f"{prefix}B{col}{suffix}, scale_factor={new_scale}{end}"
                lines[i] = line[:m.start()] + replacement + line[m.end():]
                changed = True
                break

        return "\n".join(lines), changed

    def _fix_col_6_overflow(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """Replaces Column 6 placements with Column 5 or scales down."""
        lines = code.split("\n")
        changed = False

        for i, line in enumerate(lines):
            m = re.search(r'(self\.place_at_grid\([^,]+,\s*[\'"])([A-F])6([\'"])(?:,\s*scale_factor=([0-9.]+))?(\s*\))', line)
            if m:
                prefix = m.group(1)
                row = m.group(2)
                suffix = m.group(3)
                old_scale = float(m.group(4)) if m.group(4) else 1.0
                end = m.group(5)

                new_scale = round(min(old_scale, 0.85), 2)
                replacement = f"{prefix}{row}5{suffix}, scale_factor={new_scale}{end}"
                lines[i] = line[:m.start()] + replacement + line[m.end():]
                changed = True
                break

        return "\n".join(lines), changed

    def _fix_row_f_intrusion(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """Replaces Row F placements with Row E to avoid subtitle buffer."""
        lines = code.split("\n")
        changed = False

        for i, line in enumerate(lines):
            m = re.search(r'(self\.place_at_grid\([^,]+,\s*[\'"])F([1-6])([\'"])(?:,\s*scale_factor=([0-9.]+))?(\s*\))', line)
            if m:
                prefix = m.group(1)
                col = m.group(2)
                suffix = m.group(3)
                old_scale = float(m.group(4)) if m.group(4) else 1.0
                end = m.group(5)

                new_scale = round(min(old_scale, 0.85), 2)
                replacement = f"{prefix}E{col}{suffix}, scale_factor={new_scale}{end}"
                lines[i] = line[:m.start()] + replacement + line[m.end():]
                changed = True
                break

            m_area = re.search(r'(self\.place_in_area\([^,]+,\s*[\'"][A-F][1-6][\'"],\s*[\'"])F([1-6])([\'"])', line)
            if m_area:
                lines[i] = line[:m_area.start(2)-1] + f"E{m_area.group(2)}" + line[m_area.end():]
                changed = True
                break

        return "\n".join(lines), changed

    def _fix_duplicate_grid_cell(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """Moves duplicate grid assignments to adjacent free cells."""
        details = issue.details or {}
        target_line = details.get("line") or (details.get("lines", [0, 0])[1] if "lines" in details else None)
        cell = details.get("cell", "")
        if not cell and "m2" in details:
            center = details["m2"].get("center")
            if center and len(center) >= 2:
                cell = LocalVisualChecker.coordinates_to_grid_cell(center[0], center[1])
        if not cell or len(cell) < 2:
            return code, False

        row_char, col_char = cell[0], cell[1]
        lines = code.split("\n")
        changed = False

        # Find adjacent safe cell (try right col, then down row, then left col)
        col_num = int(col_char)
        row_idx = ord(row_char) - ord("A")
        rows = ["A", "B", "C", "D", "E", "F"]

        candidate_cells = []
        if col_num < 5:
            candidate_cells.append(f"{row_char}{col_num + 1}")
        if row_idx < 4:
            candidate_cells.append(f"{rows[row_idx + 1]}{col_num}")
        if col_num > 1:
            candidate_cells.append(f"{row_char}{col_num - 1}")

        new_cell = candidate_cells[0] if candidate_cells else f"{row_char}{col_num}"
        if new_cell == cell:
            return code, False

        # If a specific target line was identified:
        if target_line and 1 <= target_line <= len(lines):
            line = lines[target_line - 1]
            if cell in line:
                lines[target_line - 1] = line.replace(f"'{cell}'", f"'{new_cell}'").replace(f'"{cell}"', f'"{new_cell}"')
                changed = True
                return "\n".join(lines), changed

        # Fallback: find second occurrence of the cell in place_at_grid or place_in_area
        occurrences = 0
        for i, line in enumerate(lines):
            if f"'{cell}'" in line or f'"{cell}"' in line:
                occurrences += 1
                if occurrences >= 2:
                    lines[i] = line.replace(f"'{cell}'", f"'{new_cell}'").replace(f'"{cell}"', f'"{new_cell}"')
                    changed = True
                    break

        return "\n".join(lines), changed

    def _fix_to_edge_left(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """Replaces diag.to_edge(LEFT) with safe grid placement."""
        obj = issue.details.get("object", "")
        if obj in ("lecture", "self.lecture", "title", "self") or "self.lecture" in issue.description:
            return code, False
        lines = code.split("\n")
        changed = False
        for i, line in enumerate(lines):
            if re.search(rf'\b{re.escape(obj)}\.to_edge\(\s*LEFT', line):
                indent = re.match(r'^\s*', line).group(0)
                lines[i] = f"{indent}self.place_at_grid({obj}, 'C3', scale_factor=0.85)"
                changed = True
                break
        return "\n".join(lines), changed

    def _fix_next_to_lecture(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """Replaces diag.next_to(self.lecture[...]) with safe grid placement."""
        obj = issue.details.get("object", "")
        lines = code.split("\n")
        changed = False
        for i, line in enumerate(lines):
            if re.search(rf'\b{re.escape(obj)}\.next_to\(\s*self\.lecture', line):
                indent = re.match(r'^\s*', line).group(0)
                lines[i] = f"{indent}self.place_at_grid({obj}, 'C3', scale_factor=0.85)"
                changed = True
                break
        return "\n".join(lines), changed

    def _fix_duplicate_title(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """Neutralizes duplicate title definition."""
        obj = issue.details.get("object", "title")
        lines = code.split("\n")
        changed = False
        for i, line in enumerate(lines):
            if re.search(rf'^\s*{re.escape(obj)}\s*=\s*(?:Text|Title)\(', line):
                indent = re.match(r'^\s*', line).group(0)
                lines[i] = f"{indent}{obj} = VGroup()  # [AUTO-FIX] Neutralized duplicate title"
                changed = True
                # Clean up chained or trailing .to_edge(UP)
                j = i + 1
                while j < len(lines) and (f"{obj}.to_edge(" in lines[j] or f"{obj}.to_corner(" in lines[j]):
                    lines[j] = f"{indent}pass  # Neutralized title positioning"
                    j += 1
                break
        return "\n".join(lines), changed

    def _fix_reorder_nodes_before_edges(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """
        Reorders self.place_at_grid calls for nodes to execute BEFORE edge (Line/Arrow) definitions.
        Prevents zero-length edges and collapsed overlapping weights at (0, 0, 0).
        """
        lines = code.split("\n")
        grid_lines: Dict[str, int] = {}
        for idx, l in enumerate(lines):
            m = re.search(r'self\.place_at_grid\(\s*([^,]+),', l)
            if m:
                grid_lines[m.group(1).strip()] = idx

        earliest_edge_idx = None
        nodes_to_move = set()

        for idx, l in enumerate(lines):
            if any(k in l for k in ["Line(", "Arrow(", "DashedLine(", "DoubleArrow("]):
                for obj in re.findall(r'([a-zA-Z_]\w*(?:\[[^\]]+\])*)\.get_center\(\)', l):
                    base_obj = re.sub(r'\[0\]$', '', obj.strip())
                    line_pos = grid_lines.get(obj.strip(), grid_lines.get(base_obj))
                    if line_pos and line_pos > idx:
                        if earliest_edge_idx is None or idx < earliest_edge_idx:
                            earliest_edge_idx = idx
                        nodes_to_move.add(line_pos)

        if not nodes_to_move or earliest_edge_idx is None:
            return code, False

        insert_idx = earliest_edge_idx
        for i in range(earliest_edge_idx, -1, -1):
            if re.search(r'^\s*[a-zA-Z_]\w*\s*=\s*(?:\{|\[|Line\(|Arrow\()', lines[i]):
                insert_idx = i
                break

        moved_statements = [lines[i] for i in sorted(nodes_to_move)]
        new_lines = [lines[i] for i in range(len(lines)) if i not in nodes_to_move]

        removed_before = sum(1 for i in nodes_to_move if i < insert_idx)
        new_insert_idx = insert_idx - removed_before

        indent = re.match(r'^\s*', new_lines[new_insert_idx]).group(0) if new_insert_idx < len(new_lines) else "        "
        formatted_moved = [indent + s.strip() for s in moved_statements]

        result_lines = (
            new_lines[:new_insert_idx] +
            [indent + "# Auto-placed nodes before connecting edges:"] +
            formatted_moved +
            new_lines[new_insert_idx:]
        )
        return "\n".join(result_lines), True

    def _fix_missing_nodes(self, code: str, issue: VisualIssue) -> Tuple[str, bool]:
        """
        Adds missing node rendering when nodes are defined and placed but forgotten in self.add/self.play.
        """
        var_name = issue.suggested_fix.get("var_name", "nodes")
        is_dict = bool(re.search(rf'^\s*{re.escape(var_name)}\s*=\s*\{{', code, re.MULTILINE))
        add_stmt = f"for node in {var_name}.values(): self.add(node)" if is_dict else f"for node in {var_name}: self.add(node)"

        lines = code.split("\n")
        target_line = None
        for idx, l in enumerate(lines):
            if re.search(r'for\s+\w+\s+in\s+(?:edges|edge_labels|weights)', l):
                target_line = idx
                break
            elif re.search(r'self\.play\([^)]*(?:edges|weights|FadeIn\(arrows)', l):
                target_line = idx
                break

        if target_line is not None:
            indent = re.match(r'^\s*', lines[target_line]).group(0)
            lines.insert(target_line, f"{indent}{add_stmt}")
            return "\n".join(lines), True

        return code, False