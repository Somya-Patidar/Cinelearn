"""
CineLearn — Milestone 4B: Tier 3 Declarative Visual Primitives Compiler
========================================================================
Compiles declarative visual primitives (PrimitiveShape, PointerIndicator,
CompositeNodeGraph, CompositeDissectionSquare, container/boundary) into
deterministic, syntactically safe Manim Python source code directly consuming
authoritative ComputedEntityGeometry world bounding boxes.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ir.models import EntityDeclaration, VisualTier
from src.compiler.compiler_models import (
    CompilationDiagnostic,
    CompilationErrorCode,
)
from src.compiler.layout_models import ComputedEntityGeometry


# Recognized shape types for PrimitiveShape
SUPPORTED_SHAPE_TYPES: Set[str] = {
    "rectangle",
    "square",
    "circle",
    "rounded_rectangle",
    "ellipse",
    "polygon",
}

SUPPORTED_PRIMITIVES: Set[str] = {
    "PrimitiveShape",
    "PointerIndicator",
    "CompositeNodeGraph",
    "CompositeDissectionSquare",
    "container",
    "boundary",
    "ProceduralComponent",
}


def _safe_color(val: Optional[str], default: str = "#58C4DD") -> str:
    """Sanitizes color strings, falling back to default if invalid."""
    if not val or not isinstance(val, str):
        return default
    v = val.strip()
    if re.match(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$", v):
        return v
    # Standard named Manim colors
    named_colors = {
        "WHITE", "BLACK", "RED", "GREEN", "BLUE", "YELLOW", "GOLD",
        "GRAY", "GREY", "ORANGE", "PURPLE", "LIGHT_GRAY", "DARK_GRAY",
        "LIGHT_BLUE", "LIGHT_SALMON", "GREEN_SCREEN", "GOLD_ACCENT",
    }
    if v.upper() in named_colors:
        return v.upper()
    return default


class PrimitiveCompiler:
    """
    Compiler handler for Tier 3 Declarative Visual Primitives.
    Converts primitive entity declarations and computed world-space bounding boxes
    into deterministic Manim code snippets.
    """

    def can_compile(self, entity: EntityDeclaration) -> bool:
        """Determines if this entity is a supported Tier 3 primitive."""
        if entity.visual_tier == VisualTier.TIER3_PRIMITIVE_TREE:
            return True
        return entity.component_type in SUPPORTED_PRIMITIVES

    def compile_primitive(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
    ) -> Tuple[str, List[CompilationDiagnostic]]:
        """
        Compiles a Tier 3 primitive into Manim Python code.
        Directly uses geom.world_box for positioning and sizing.
        """
        diagnostics: List[CompilationDiagnostic] = []
        ctype = entity.component_type
        init = entity.initial_state or {}
        styling = entity.styling

        if ctype == "PrimitiveShape":
            code, diags = self._compile_primitive_shape(entity, geom, var_name)
            diagnostics.extend(diags)
            return code, diagnostics

        elif ctype == "PointerIndicator":
            code, diags = self._compile_pointer_indicator(entity, geom, var_name)
            diagnostics.extend(diags)
            return code, diagnostics

        elif ctype == "CompositeNodeGraph":
            code, diags = self._compile_composite_node_graph(entity, geom, var_name)
            diagnostics.extend(diags)
            return code, diagnostics

        elif ctype == "CompositeDissectionSquare":
            code, diags = self._compile_dissection_square(entity, geom, var_name)
            diagnostics.extend(diags)
            return code, diagnostics

        elif ctype in ("container", "boundary"):
            code, diags = self._compile_boundary_container(entity, geom, var_name)
            diagnostics.extend(diags)
            return code, diagnostics

        elif ctype == "ProceduralComponent":
            code, diags = self._compile_procedural_component(entity, geom, var_name)
            diagnostics.extend(diags)
            return code, diagnostics

        # Fallback for unrecognized Tier 3 primitive
        diagnostics.append(
            CompilationDiagnostic(
                error_code=CompilationErrorCode.COMPILER_UNSUPPORTED_COMPONENT,
                message=f"Unsupported Tier 3 primitive component type '{ctype}' for entity '{entity.id}'.",
                entity_id=entity.id,
                field_path="component_type",
                severity="ERROR",
            )
        )
        # Emit a safe placeholder to allow parsing
        cx = round(geom.world_box.x, 4)
        cy = round(geom.world_box.y, 4)
        w = round(max(0.2, geom.world_box.width), 4)
        h = round(max(0.2, geom.world_box.height), 4)
        code = f"{var_name} = Rectangle(width={w}, height={h}, stroke_color='#FF6B6B').move_to(np.array([{cx}, {cy}, 0.0]))"
        return code, diagnostics

    # -------------------------------------------------------------------------
    # PrimitiveShape Compilation
    # -------------------------------------------------------------------------
    def _compile_primitive_shape(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
    ) -> Tuple[str, List[CompilationDiagnostic]]:
        diagnostics: List[CompilationDiagnostic] = []
        init = entity.initial_state or {}
        styling = entity.styling

        raw_shape_type = str(init.get("shape_type", "rectangle")).lower()
        if raw_shape_type not in SUPPORTED_SHAPE_TYPES:
            diagnostics.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                    message=f"Entity '{entity.id}' has unrecognized shape_type '{raw_shape_type}', falling back to 'rectangle'.",
                    entity_id=entity.id,
                    field_path="initial_state.shape_type",
                    severity="WARNING",
                )
            )
            raw_shape_type = "rectangle"

        cx = round(geom.world_box.x, 4)
        cy = round(geom.world_box.y, 4)
        w = round(max(0.05, geom.world_box.width), 4)
        h = round(max(0.05, geom.world_box.height), 4)

        # Color & Styling resolution
        color_val = styling.color if (styling and styling.color) else init.get("color", "#58C4DD")
        color = _safe_color(color_val, "#58C4DD")

        stroke_color_val = styling.stroke_color if (styling and styling.stroke_color) else init.get("stroke_color", color)
        stroke_color = _safe_color(stroke_color_val, color)

        stroke_width = styling.stroke_width if (styling and styling.stroke_width is not None) else float(init.get("stroke_width", 2.0))
        stroke_width = round(max(0.5, min(stroke_width, 10.0)), 2)

        fill_opacity = styling.fill_opacity if (styling and styling.fill_opacity is not None) else float(init.get("fill_opacity", 0.25))
        fill_opacity = round(max(0.0, min(fill_opacity, 1.0)), 2)

        lines: List[str] = []

        # Construct primary shape mobject
        if raw_shape_type == "rectangle":
            lines.append(
                f"_shape_{var_name} = Rectangle("
                f"width={w}, height={h}, "
                f"stroke_color={stroke_color!r}, stroke_width={stroke_width}, "
                f"fill_color={color!r}, fill_opacity={fill_opacity}"
                f").move_to(np.array([{cx}, {cy}, 0.0]))"
            )
        elif raw_shape_type == "square":
            side = round(min(w, h), 4)
            lines.append(
                f"_shape_{var_name} = Square("
                f"side_length={side}, "
                f"stroke_color={stroke_color!r}, stroke_width={stroke_width}, "
                f"fill_color={color!r}, fill_opacity={fill_opacity}"
                f").move_to(np.array([{cx}, {cy}, 0.0]))"
            )
        elif raw_shape_type == "circle":
            rad = round(min(w, h) / 2.0, 4)
            lines.append(
                f"_shape_{var_name} = Circle("
                f"radius={rad}, "
                f"stroke_color={stroke_color!r}, stroke_width={stroke_width}, "
                f"fill_color={color!r}, fill_opacity={fill_opacity}"
                f").move_to(np.array([{cx}, {cy}, 0.0]))"
            )
        elif raw_shape_type == "rounded_rectangle":
            cr = round(min(0.15, h * 0.2), 4)
            lines.append(
                f"_shape_{var_name} = RoundedRectangle("
                f"corner_radius={cr}, width={w}, height={h}, "
                f"stroke_color={stroke_color!r}, stroke_width={stroke_width}, "
                f"fill_color={color!r}, fill_opacity={fill_opacity}"
                f").move_to(np.array([{cx}, {cy}, 0.0]))"
            )
        elif raw_shape_type == "ellipse":
            lines.append(
                f"_shape_{var_name} = Ellipse("
                f"width={w}, height={h}, "
                f"stroke_color={stroke_color!r}, stroke_width={stroke_width}, "
                f"fill_color={color!r}, fill_opacity={fill_opacity}"
                f").move_to(np.array([{cx}, {cy}, 0.0]))"
            )
        else:
            # Default polygon/rectangle fallback
            lines.append(
                f"_shape_{var_name} = Rectangle("
                f"width={w}, height={h}, "
                f"stroke_color={stroke_color!r}, stroke_width={stroke_width}, "
                f"fill_color={color!r}, fill_opacity={fill_opacity}"
                f").move_to(np.array([{cx}, {cy}, 0.0]))"
            )

        label_str = init.get("label")
        if label_str and isinstance(label_str, str) and label_str.strip():
            lbl = label_str.strip()
            font_size = styling.font_size if (styling and styling.font_size) else int(init.get("font_size", 18))
            font_size = max(12, min(font_size, 36))
            lines.append(
                f"_lbl_{var_name} = Text({lbl!r}, font_size={font_size}, color='#FFFFFF')"
                f".move_to(_shape_{var_name}.get_center())"
            )
            # Safe width clamping for label inside shape
            max_lbl_w = round(w * 0.85, 4)
            lines.append(
                f"if _lbl_{var_name}.width > {max_lbl_w}:\n"
                f"    _lbl_{var_name}.scale_to_fit_width({max_lbl_w})"
            )
            lines.append(f"{var_name} = VGroup(_shape_{var_name}, _lbl_{var_name})")
        else:
            lines.append(f"{var_name} = _shape_{var_name}")

        lines.append(f"{var_name}.set_z_index({geom.z_index})")
        return "\n".join(lines), diagnostics

    # -------------------------------------------------------------------------
    # PointerIndicator Compilation
    # -------------------------------------------------------------------------
    def _compile_pointer_indicator(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
    ) -> Tuple[str, List[CompilationDiagnostic]]:
        diagnostics: List[CompilationDiagnostic] = []
        init = entity.initial_state or {}
        styling = entity.styling

        cx = round(geom.world_box.x, 4)
        cy = round(geom.world_box.y, 4)
        w = round(max(0.1, geom.world_box.width), 4)
        h = round(max(0.1, geom.world_box.height), 4)

        direction = str(init.get("direction", "up")).lower()
        color_val = styling.color if (styling and styling.color) else init.get("color", "#FFD700")
        color = _safe_color(color_val, "#FFD700")
        label = init.get("label")

        lines: List[str] = []

        # Compute start and end vectors based on direction within the bounding box
        half_w = round(w / 2.0, 4)
        half_h = round(h / 2.0, 4)

        if direction in ("up", "arrow_up"):
            start_pt = f"np.array([{cx}, {round(cy - half_h, 4)}, 0.0])"
            end_pt = f"np.array([{cx}, {round(cy + half_h, 4)}, 0.0])"
        elif direction in ("down", "arrow_down"):
            start_pt = f"np.array([{cx}, {round(cy + half_h, 4)}, 0.0])"
            end_pt = f"np.array([{cx}, {round(cy - half_h, 4)}, 0.0])"
        elif direction in ("left", "arrow_left"):
            start_pt = f"np.array([{round(cx + half_w, 4)}, {cy}, 0.0])"
            end_pt = f"np.array([{round(cx - half_w, 4)}, {cy}, 0.0])"
        elif direction in ("right", "arrow_right"):
            start_pt = f"np.array([{round(cx - half_w, 4)}, {cy}, 0.0])"
            end_pt = f"np.array([{round(cx + half_w, 4)}, {cy}, 0.0])"
        else:
            # Default pointing UP
            start_pt = f"np.array([{cx}, {round(cy - half_h, 4)}, 0.0])"
            end_pt = f"np.array([{cx}, {round(cy + half_h, 4)}, 0.0])"

        lines.append(
            f"_arrow_{var_name} = Arrow("
            f"start={start_pt}, end={end_pt}, "
            f"buff=0.02, color={color!r}, stroke_width=3.0, max_tip_length_to_length_ratio=0.35"
            f")"
        )

        if label and isinstance(label, str) and label.strip():
            lbl = label.strip()
            # Place label at the base of the arrow
            buff_dir = "DOWN" if direction in ("up", "arrow_up") else ("UP" if direction in ("down", "arrow_down") else ("RIGHT" if direction in ("left", "arrow_left") else "LEFT"))
            lines.append(
                f"_lbl_{var_name} = Text({lbl!r}, font_size=14, color={color!r})"
                f".next_to(_arrow_{var_name}, {buff_dir}, buff=0.08)"
            )
            lines.append(f"{var_name} = VGroup(_arrow_{var_name}, _lbl_{var_name})")
        else:
            lines.append(f"{var_name} = _arrow_{var_name}")

        lines.append(f"{var_name}.set_z_index({geom.z_index})")
        return "\n".join(lines), diagnostics

    # -------------------------------------------------------------------------
    # CompositeNodeGraph Compilation
    # -------------------------------------------------------------------------
    def _compile_composite_node_graph(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
    ) -> Tuple[str, List[CompilationDiagnostic]]:
        diagnostics: List[CompilationDiagnostic] = []
        init = entity.initial_state or {}

        cx = round(geom.world_box.x, 4)
        cy = round(geom.world_box.y, 4)
        w = round(max(0.5, geom.world_box.width), 4)
        h = round(max(0.4, geom.world_box.height), 4)

        elements: List[Dict[str, Any]] = init.get("elements", [])
        nodes: List[Dict[str, Any]] = init.get("nodes", [])

        lines: List[str] = []

        if elements and isinstance(elements, list):
            # Render as sequential / array cells
            num_cells = len(elements)
            cell_w = round(w / max(1, num_cells), 4)
            cell_h = round(h * 0.75, 4)

            lines.append(f"_cells_{var_name} = []")
            for idx, elem in enumerate(elements):
                val = str(elem.get("val", elem.get("value", str(idx))))
                state = str(elem.get("state", "default")).lower()

                # Visual encoding for cell state
                if state == "active":
                    stroke_c, fill_c, fill_op = "'#83C167'", "'#1a3320'", 0.75
                elif state == "midpoint":
                    stroke_c, fill_c, fill_op = "'#FFD700'", "'#332e1a'", 0.85
                elif state == "inactive":
                    stroke_c, fill_c, fill_op = "'#555555'", "'#222222'", 0.35
                else:
                    stroke_c, fill_c, fill_op = "'#58C4DD'", "'#101720'", 0.50

                lines.append(f"# Array Cell {idx}: {val} ({state})")
                lines.append(
                    f"_box_{var_name}_{idx} = Rectangle("
                    f"width={cell_w}, height={cell_h}, "
                    f"stroke_color={stroke_c}, stroke_width=2.0, "
                    f"fill_color={fill_c}, fill_opacity={fill_op}"
                    f")"
                )
                lines.append(
                    f"_val_{var_name}_{idx} = Text({val!r}, font_size=18, color='#FFFFFF')"
                    f".move_to(_box_{var_name}_{idx}.get_center())"
                )
                # Cell index subscript below cell
                lines.append(
                    f"_idx_{var_name}_{idx} = Text({str(idx)!r}, font_size=11, color='#888888')"
                    f".next_to(_box_{var_name}_{idx}, DOWN, buff=0.06)"
                )
                lines.append(
                    f"_cgrp_{var_name}_{idx} = VGroup("
                    f"_box_{var_name}_{idx}, _val_{var_name}_{idx}, _idx_{var_name}_{idx}"
                    f")"
                )
                lines.append(f"_cells_{var_name}.append(_cgrp_{var_name}_{idx})")

            lines.append(
                f"{var_name} = VGroup(*_cells_{var_name})"
                f".arrange(RIGHT, buff=0.04)"
                f".move_to(np.array([{cx}, {cy}, 0.0]))"
            )

        elif nodes and isinstance(nodes, list):
            # Render as graph nodes and connections
            num_nodes = len(nodes)
            lines.append(f"_nodes_{var_name} = []")
            node_rad = round(min(0.35, h / 3.0), 4)

            for idx, nd in enumerate(nodes):
                nid = str(nd.get("id", nd.get("name", str(idx))))
                nlbl = str(nd.get("label", nid))
                nstate = str(nd.get("state", "default")).lower()
                ncolor = "'#83C167'" if nstate == "active" else ("'#FFD700'" if nstate == "highlight" else "'#58C4DD'")

                lines.append(
                    f"_nc_{var_name}_{idx} = Circle("
                    f"radius={node_rad}, stroke_color={ncolor}, stroke_width=2.5, "
                    f"fill_color='#101720', fill_opacity=0.85"
                    f")"
                )
                lines.append(
                    f"_nl_{var_name}_{idx} = Text({nlbl!r}, font_size=14, color='#FFFFFF')"
                    f".move_to(_nc_{var_name}_{idx}.get_center())"
                )
                lines.append(
                    f"_ngrp_{var_name}_{idx} = VGroup(_nc_{var_name}_{idx}, _nl_{var_name}_{idx})"
                )
                lines.append(f"_nodes_{var_name}.append(_ngrp_{var_name}_{idx})")

            lines.append(
                f"{var_name} = VGroup(*_nodes_{var_name})"
                f".arrange(RIGHT, buff=0.35)"
                f".move_to(np.array([{cx}, {cy}, 0.0]))"
            )
        else:
            # Fallback simple node graph container
            lines.append(
                f"{var_name} = Rectangle("
                f"width={w}, height={h}, "
                f"stroke_color='#58C4DD', stroke_width=1.5, "
                f"fill_color='#101720', fill_opacity=0.3"
                f").move_to(np.array([{cx}, {cy}, 0.0]))"
            )

        lines.append(f"{var_name}.set_z_index({geom.z_index})")
        return "\n".join(lines), diagnostics

    # -------------------------------------------------------------------------
    # CompositeDissectionSquare Compilation
    # -------------------------------------------------------------------------
    def _compile_dissection_square(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
    ) -> Tuple[str, List[CompilationDiagnostic]]:
        diagnostics: List[CompilationDiagnostic] = []
        init = entity.initial_state or {}
        styling = entity.styling

        cx = round(geom.world_box.x, 4)
        cy = round(geom.world_box.y, 4)
        side = round(min(geom.world_box.width, geom.world_box.height), 4)

        color_val = styling.color if (styling and styling.color) else init.get("color", "#58C4DD")
        color = _safe_color(color_val, "#58C4DD")

        lines: List[str] = [
            f"# Composite Dissection Square: side={side}",
            f"_outer_{var_name} = Square("
            f"side_length={side}, stroke_color={color!r}, stroke_width=3.0, "
            f"fill_color='#101720', fill_opacity=0.25"
            f").move_to(np.array([{cx}, {cy}, 0.0]))",
        ]

        # Optional inner regions / label
        labels = init.get("labels", [])
        if labels and isinstance(labels, list):
            lines.append(f"_dlabels_{var_name} = []")
            for l_idx, lbl in enumerate(labels[:4]):
                lines.append(
                    f"_dl_{var_name}_{l_idx} = Text({str(lbl)!r}, font_size=14, color='#FFFFFF')"
                )
                lines.append(f"_dlabels_{var_name}.append(_dl_{var_name}_{l_idx})")
            lines.append(
                f"_dgrp_{var_name} = VGroup(*_dlabels_{var_name}).arrange_in_grid(rows=2, cols=2, buff=0.3)"
                f".move_to(_outer_{var_name}.get_center())"
            )
            lines.append(f"{var_name} = VGroup(_outer_{var_name}, _dgrp_{var_name})")
        else:
            lines.append(f"{var_name} = _outer_{var_name}")

        lines.append(f"{var_name}.set_z_index({geom.z_index})")
        return "\n".join(lines), diagnostics

    # -------------------------------------------------------------------------
    # Boundary & Container Compilation
    # -------------------------------------------------------------------------
    def _compile_boundary_container(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
    ) -> Tuple[str, List[CompilationDiagnostic]]:
        diagnostics: List[CompilationDiagnostic] = []
        init = entity.initial_state or {}
        styling = entity.styling

        cx = round(geom.world_box.x, 4)
        cy = round(geom.world_box.y, 4)
        w = round(max(0.2, geom.world_box.width), 4)
        h = round(max(0.2, geom.world_box.height), 4)

        color_val = styling.color if (styling and styling.color) else init.get("color", "#58C4DD")
        color = _safe_color(color_val, "#58C4DD")

        title = init.get("title", init.get("label"))
        lines: List[str] = [
            f"_boundary_{var_name} = Rectangle("
            f"width={w}, height={h}, "
            f"stroke_color={color!r}, stroke_width=1.8, stroke_opacity=0.75, "
            f"fill_color='#0D1117', fill_opacity=0.15"
            f").move_to(np.array([{cx}, {cy}, 0.0]))"
        ]

        if title and isinstance(title, str) and title.strip():
            t_str = title.strip()
            lines.append(
                f"_header_{var_name} = Text({t_str!r}, font_size=15, color={color!r})"
                f".next_to(_boundary_{var_name}.get_top(), DOWN, buff=0.12)"
            )
            lines.append(f"{var_name} = VGroup(_boundary_{var_name}, _header_{var_name})")
        else:
            lines.append(f"{var_name} = _boundary_{var_name}")

        lines.append(f"{var_name}.set_z_index({geom.z_index})")
        return "\n".join(lines), diagnostics

    # -------------------------------------------------------------------------
    # ProceduralComponent Compilation
    # -------------------------------------------------------------------------
    def _compile_procedural_component(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
    ) -> Tuple[str, List[CompilationDiagnostic]]:
        diagnostics: List[CompilationDiagnostic] = []
        init = entity.initial_state or {}

        cx = round(geom.world_box.x, 4)
        cy = round(geom.world_box.y, 4)
        w = round(max(0.2, geom.world_box.width), 4)
        h = round(max(0.2, geom.world_box.height), 4)
        name = str(init.get("name", entity.id))

        code = (
            f"# ProceduralComponent: {name}\n"
            f"_pbox_{var_name} = RoundedRectangle("
            f"corner_radius=0.15, width={w}, height={h}, "
            f"stroke_color='#58C4DD', stroke_width=2.0, "
            f"fill_color='#101720', fill_opacity=0.85"
            f").move_to(np.array([{cx}, {cy}, 0.0]))\n"
            f"_plbl_{var_name} = Text({name!r}, font_size=16, color='#FFFFFF')"
            f".move_to(_pbox_{var_name}.get_center())\n"
            f"{var_name} = VGroup(_pbox_{var_name}, _plbl_{var_name})\n"
            f"{var_name}.set_z_index({geom.z_index})"
        )
        return code, diagnostics
