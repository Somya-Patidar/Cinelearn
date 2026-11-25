"""
CineLearn — Milestone 4A: Intrinsic Dimension Estimator
======================================================
Deterministic dimension estimation for visual components, primitive trees,
and conceptual text/card representations with uncertainty detection and
a mandatory 15% safety margin.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field

from src.ir.models import EntityDeclaration, VisualTier


class EstimatedDimensions(BaseModel):
    """Result of intrinsic dimension estimation."""
    model_config = ConfigDict(extra="forbid")

    width: float = Field(gt=0, description="Estimated width in Manim canvas units")
    height: float = Field(gt=0, description="Estimated height in Manim canvas units")
    aspect_ratio: float = Field(gt=0, description="Aspect ratio (width / height)")
    is_uncertain: bool = Field(default=False, description="True if estimation has high variance (e.g. complex LaTeX)")
    source: str = Field(description="Estimation strategy applied")
    safety_margin_applied: float = Field(default=0.15, description="Fractional safety buffer applied")
    details: Dict[str, Any] = Field(default_factory=dict)


# Known default dimensions for Tier 1 registered components from manifests
KNOWN_COMPONENT_DEFAULTS: Dict[str, Dict[str, float]] = {
    "ReactionEnergyDiagram": {"width": 6.0, "height": 3.6, "aspect": 1.5, "min_scale": 0.55},
    "MoleculeVisual": {"width": 5.0, "height": 3.5, "aspect": 1.4, "min_scale": 0.55},
    "GraphVisual": {"width": 5.4, "height": 3.0, "aspect": 1.8, "min_scale": 0.50},
    "DPTableMatrix": {"width": 5.6, "height": 3.5, "aspect": 1.6, "min_scale": 0.55},
    "FunctionCurve": {"width": 6.0, "height": 3.5, "aspect": 1.7, "min_scale": 0.55},
    "NumberLineVisual": {"width": 6.6, "height": 1.6, "aspect": 2.4, "min_scale": 0.55},
    "ProcessCardVisual": {"width": 3.2, "height": 1.8, "aspect": 1.5, "min_scale": 0.65},
    "ReadyQueueVisual": {"width": 6.0, "height": 1.8, "aspect": 1.8, "min_scale": 0.60},
    "CPUExecutionVisual": {"width": 5.5, "height": 2.4, "aspect": 1.6, "min_scale": 0.60},
    "GanttChartVisual": {"width": 6.8, "height": 2.2, "aspect": 1.8, "min_scale": 0.55},
    "MetricTableVisual": {"width": 7.0, "height": 2.2, "aspect": 2.0, "min_scale": 0.55},
}

# Regex to detect complex LaTeX structures that have high geometric variance
COMPLEX_LATEX_PATTERN = re.compile(
    r"(\\frac|\\sum|\\int|\\matrix|\\begin\{|\\sqrt|\\prod|\\left|\\over|\\lim)",
    re.IGNORECASE
)


class IntrinsicDimensionEstimator:
    """
    Estimates 2D intrinsic dimensions (width, height) of Scene IR visual entities.
    Prioritizes explicit initial_state parameters, then registry component manifests,
    then text measurement heuristics with safety margins, and finally primitive defaults.
    """

    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        if manifest is None:
            try:
                from src.domains import VisualRegistry
                manifest = VisualRegistry.get_schema_manifest()
            except Exception:
                manifest = {}
        self.manifest = manifest or {}
        self._component_specs: Dict[str, Dict[str, Any]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Indexes component specifications from manifest."""
        for domain, comps in self.manifest.items():
            if isinstance(comps, list):
                for comp in comps:
                    cid = comp.get("component_id")
                    if cid:
                        self._component_specs[cid] = comp
                        for alias in comp.get("aliases", []):
                            self._component_specs[alias] = comp

    def estimate(self, entity: EntityDeclaration) -> EstimatedDimensions:
        """Computes intrinsic dimensions for any visual entity declaration."""
        init = entity.initial_state or {}
        ctype = entity.component_type
        tier = entity.visual_tier

        # -------------------------------------------------------------
        # Strategy 1: Explicit geometric parameters in initial_state
        # -------------------------------------------------------------
        if "width" in init and "height" in init:
            try:
                w = float(init["width"])
                h = float(init["height"])
                if w > 0 and h > 0:
                    return EstimatedDimensions(
                        width=w,
                        height=h,
                        aspect_ratio=w / h,
                        is_uncertain=False,
                        source="explicit_width_height",
                        safety_margin_applied=0.0,
                        details={"initial_state": init},
                    )
            except (ValueError, TypeError):
                pass

        if "side_length" in init:
            try:
                s = float(init["side_length"])
                if s > 0:
                    return EstimatedDimensions(
                        width=s,
                        height=s,
                        aspect_ratio=1.0,
                        is_uncertain=False,
                        source="explicit_side_length",
                        safety_margin_applied=0.0,
                        details={"side_length": s},
                    )
            except (ValueError, TypeError):
                pass

        if "radius" in init:
            try:
                r = float(init["radius"])
                if r > 0:
                    diam = 2.0 * r
                    return EstimatedDimensions(
                        width=diam,
                        height=diam,
                        aspect_ratio=1.0,
                        is_uncertain=False,
                        source="explicit_radius",
                        safety_margin_applied=0.0,
                        details={"radius": r},
                    )
            except (ValueError, TypeError):
                pass

        if "length" in init and ctype in ("NumberLineVisual", "NumberLine"):
            try:
                length = float(init["length"])
                h = 1.6
                return EstimatedDimensions(
                    width=length,
                    height=h,
                    aspect_ratio=length / h,
                    is_uncertain=False,
                    source="explicit_length",
                    safety_margin_applied=0.0,
                    details={"length": length},
                )
            except (ValueError, TypeError):
                pass

        # -------------------------------------------------------------
        # Strategy 2: Registered Tier 1 Components from Registry / Defaults
        # -------------------------------------------------------------
        if tier == VisualTier.TIER1_REGISTERED or ctype in KNOWN_COMPONENT_DEFAULTS or ctype in self._component_specs:
            spec = self._component_specs.get(ctype, {})
            defaults = KNOWN_COMPONENT_DEFAULTS.get(ctype, {})

            # Default dimensions from init_params in spec
            init_params = spec.get("init_params", {})
            w = defaults.get("width")
            h = defaults.get("height")

            if "width" in init_params and init_params["width"].get("default") is not None:
                try:
                    w = float(init_params["width"]["default"])
                except Exception:
                    pass
            if "height" in init_params and init_params["height"].get("default") is not None:
                try:
                    h = float(init_params["height"]["default"])
                except Exception:
                    pass

            if w is not None and h is not None and w > 0 and h > 0:
                return EstimatedDimensions(
                    width=w,
                    height=h,
                    aspect_ratio=w / h,
                    is_uncertain=False,
                    source="registered_component_manifest",
                    safety_margin_applied=0.0,
                    details={"component": ctype, "manifest_defaults": {"width": w, "height": h}},
                )

            # If only preferred_aspect is known
            aspect = float(spec.get("preferred_aspect") or defaults.get("aspect") or 1.6)
            h = 3.0
            w = h * aspect
            return EstimatedDimensions(
                width=w,
                height=h,
                aspect_ratio=aspect,
                is_uncertain=False,
                source="registered_component_aspect",
                safety_margin_applied=0.0,
                details={"component": ctype, "aspect": aspect},
            )

        # -------------------------------------------------------------
        # Strategy 3: Tier 4 Conceptual Cards, MathTex, Text Cards
        # -------------------------------------------------------------
        if tier == VisualTier.TIER4_CONCEPTUAL_CARD or ctype in ("ConceptualCard", "MathTex", "TextCard"):
            return self._estimate_conceptual_card(entity)

        # -------------------------------------------------------------
        # Strategy 4: Tier 3 Primitive Trees & Shapes
        # -------------------------------------------------------------
        if tier == VisualTier.TIER3_PRIMITIVE_TREE or ctype in (
            "CompositeNodeGraph", "PointerIndicator", "CompositeDissectionSquare",
            "PrimitiveShape", "ProceduralComponent"
        ):
            return self._estimate_primitive_tree(entity)

        # -------------------------------------------------------------
        # Strategy 5: Generic Fallback with Uncertainty
        # -------------------------------------------------------------
        return EstimatedDimensions(
            width=4.0,
            height=2.5,
            aspect_ratio=4.0 / 2.5,
            is_uncertain=True,
            source="generic_fallback",
            safety_margin_applied=0.20,
            details={"warning": "Unregistered component type without explicit dimensions"},
        )

    def _estimate_conceptual_card(self, entity: EntityDeclaration) -> EstimatedDimensions:
        """
        Estimates dimensions for ConceptualCard and text components using
        character counting, font size heuristics, and mandatory 15% safety buffer.
        """
        init = entity.initial_state or {}
        font_size = float(init.get("font_size", 24))

        title = str(init.get("title", "") or "")
        formula = str(init.get("formula", "") or init.get("math_str", "") or "")
        body = str(init.get("body", "") or init.get("text", "") or "")
        badge = str(init.get("badge", "") or "")

        # Manim text heuristic: height = font_size / 135.0
        unit_height = font_size / 135.0
        unit_char_width = unit_height * 0.55

        # Check for complex LaTeX that triggers dimension uncertainty
        is_uncertain = bool(COMPLEX_LATEX_PATTERN.search(formula))
        safety_margin = 0.25 if is_uncertain else 0.15

        # Width calculation: maximum length among components
        title_w = len(title) * unit_char_width * 1.05 if title else 0.0
        formula_w = len(formula) * unit_char_width * 1.15 if formula else 0.0
        body_lines = body.split("\n") if body else []
        body_w = max((len(line) * unit_char_width for line in body_lines), default=0.0)
        badge_w = len(badge) * unit_char_width * 0.90 if badge else 0.0

        content_w = max(title_w, formula_w, body_w, badge_w, 2.8)
        # Apply safety margin + internal card padding (0.4 units)
        total_w = (content_w * (1.0 + safety_margin)) + 0.40

        # Height calculation: vertical stack of title, badge, formula, body + margins
        total_h = 0.40  # Base card padding
        if title:
            total_h += max(unit_height * 1.3, 0.45)
        if badge:
            total_h += 0.35
        if formula:
            formula_lines = formula.count("\\\\") + formula.count("\n") + 1
            total_h += formula_lines * max(unit_height * 1.8, 0.65)
        if body_lines:
            total_h += len(body_lines) * max(unit_height * 1.2, 0.38)

        total_h = total_h * (1.0 + safety_margin)

        # Enforce minimum readability bounds
        total_w = max(total_w, 3.2)
        total_h = max(total_h, 1.4)

        return EstimatedDimensions(
            width=round(total_w, 4),
            height=round(total_h, 4),
            aspect_ratio=round(total_w / total_h, 4),
            is_uncertain=is_uncertain,
            source="text_heuristic_with_safety_margin",
            safety_margin_applied=safety_margin,
            details={
                "title_chars": len(title),
                "formula_chars": len(formula),
                "has_complex_latex": is_uncertain,
                "font_size": font_size,
            },
        )

    def _estimate_primitive_tree(self, entity: EntityDeclaration) -> EstimatedDimensions:
        """Estimates dimensions for Tier 3 primitive shapes and composite trees."""
        init = entity.initial_state or {}
        ctype = entity.component_type

        if ctype == "CompositeNodeGraph":
            nodes = init.get("nodes") or init.get("elements") or []
            node_count = len(nodes) if isinstance(nodes, list) else 4
            # Scaling graph width based on node count
            w = min(max(3.5, node_count * 0.9), 6.5)
            h = 3.2
            return EstimatedDimensions(
                width=w,
                height=h,
                aspect_ratio=w / h,
                is_uncertain=False,
                source="node_graph_heuristic",
                safety_margin_applied=0.10,
                details={"node_count": node_count},
            )

        if ctype == "PointerIndicator":
            size = float(init.get("size", 0.8))
            return EstimatedDimensions(
                width=round(size * 0.75, 4),
                height=round(size * 1.25, 4),
                aspect_ratio=0.6,
                is_uncertain=False,
                source="pointer_primitive_size",
                safety_margin_applied=0.0,
                details={"size": size},
            )

        if ctype == "CompositeDissectionSquare":
            side = float(init.get("side_length", 4.0))
            return EstimatedDimensions(
                width=side,
                height=side,
                aspect_ratio=1.0,
                is_uncertain=False,
                source="dissection_square_side",
                safety_margin_applied=0.0,
                details={"side_length": side},
            )

        if ctype == "PrimitiveShape":
            shape = init.get("shape_type", "rectangle")
            w = float(init.get("width", 3.0))
            h = float(init.get("height", 2.0))
            return EstimatedDimensions(
                width=w,
                height=h,
                aspect_ratio=w / h if h > 0 else 1.0,
                is_uncertain=False,
                source="primitive_shape",
                safety_margin_applied=0.0,
                details={"shape_type": shape},
            )

        # ProceduralComponent default
        return EstimatedDimensions(
            width=4.5,
            height=2.8,
            aspect_ratio=4.5 / 2.8,
            is_uncertain=False,
            source="procedural_component_default",
            safety_margin_applied=0.15,
            details={"component_type": ctype},
        )
