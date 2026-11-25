"""
CineLearn — Milestone 4B: Tier 4 Conceptual Cards Compiler
===========================================================
Compiles Tier 4 conceptual cards (Formula cards, Definition cards,
Boundary-Invariant cards) into deterministic Manim Python source code,
enforcing safe LaTeX sanitization, strict geometry consumption from
ComputedEntityGeometry, and content overflow diagnostics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.ir.models import ActionDeclaration, EntityDeclaration, VisualTier
from src.compiler.compiler_models import (
    CompilationDiagnostic,
    CompilationErrorCode,
    is_safe_keyword_identifier,
)
from src.compiler.layout_models import ComputedEntityGeometry


SUPPORTED_CARD_COMPONENTS: Set[str] = {
    "ConceptualCard",
    "MathTex",
    "TextCard",
    "FormulaCard",
    "DefinitionCard",
}

# Color palette for badges and cards
BADGE_COLORS: Dict[str, str] = {
    "INVARIANT": "#FFD700",
    "GOVERNING_LAW": "#83C167",
    "INTERVAL_HALVED": "#58C4DD",
    "WARNING": "#FF6B6B",
    "DEFINITION": "#58C4DD",
    "THEOREM": "#FFA07A",
    "DEFAULT": "#58C4DD",
}


def sanitize_latex(tex: str) -> str:
    """
    Sanitizes LaTeX string for safe inclusion in Manim MathTex.
    Normalizes excessive braces and cleans backslashes without altering mathematical meaning.
    """
    if not tex:
        return ""
    cleaned = str(tex).strip()
    # Normalize duplicate consecutive backslashes if accidentally escaped
    # But preserve standard LaTeX commands
    # Clean up double braces {{var}} -> {var} if not needed
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = re.sub(r"\{\{([^{}]*)\}\}", r"{\1}", cleaned)
    return cleaned


def _safe_color(val: Optional[str], default: str = "#58C4DD") -> str:
    """Sanitizes color string, ensuring valid hex or Manim named color."""
    if not val or not isinstance(val, str):
        return default
    v = val.strip()
    if re.match(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$", v):
        return v
    named_colors = {
        "WHITE", "BLACK", "RED", "GREEN", "BLUE", "YELLOW", "GOLD",
        "GRAY", "GREY", "ORANGE", "PURPLE", "LIGHT_GRAY", "DARK_GRAY",
        "LIGHT_BLUE", "LIGHT_SALMON", "GREEN_SCREEN", "GOLD_ACCENT",
    }
    if v.upper() in named_colors:
        return v.upper()
    return default


def should_use_matching_tex(
    src_formula: Optional[str],
    tgt_formula: str,
    mode: Optional[str] = None,
) -> bool:
    """
    Determines whether TransformMatchingTex or ReplacementTransform should be used.
    Returns True for TransformMatchingTex, False for ReplacementTransform.

    Rules:
    1. Input presence validation: If src_formula or tgt_formula is absent, empty,
       or not a string/whitespace-only, returns False. (FINDING-4C2-01)
    2. Explicit mode evaluation:
       - If mode is 'replacement', returns False.
       - If mode is 'matching_tex' or 'matching', returns True.
    3. Identical expressions (clean_src == clean_tgt) return True.
    4. Extract alphanumeric mathematical tokens, LaTeX command tokens, and digits.
    5. Meaningful overlap evaluation (FINDING-4C2-03):
       - If no common tokens, returns False.
       - Reject false positives: solitary single character or digit lacks structural overlap (returns False).
       - Solitary multi-character LaTeX macro (e.g. \\sqrt, \\frac, \\alpha, \\int) returns True.
       - If multiple tokens overlap:
         - If both formulas contain variables, but share no variables and no macros (only numbers),
           returns False.
         - Otherwise returns True.
    """
    # 1. Validate presence and basic validity of LaTeX inputs before selecting mode (FINDING-4C2-01)
    if not src_formula or not isinstance(src_formula, str) or not src_formula.strip():
        return False
    if not tgt_formula or not isinstance(tgt_formula, str) or not tgt_formula.strip():
        return False

    clean_src = src_formula.strip()
    clean_tgt = tgt_formula.strip()

    # 2. Explicit mode overrides (preceded by input presence check)
    if mode == "replacement":
        return False
    if mode in ("matching_tex", "matching"):
        return True

    # 3. Identical expressions always match
    if clean_src == clean_tgt:
        return True

    # 4. Extract alphanumeric mathematical tokens, LaTeX command tokens, and digits
    token_pattern = re.compile(r"\\[a-zA-Z]+|[a-zA-Z]|[0-9]+")
    src_tokens = set(token_pattern.findall(clean_src))
    tgt_tokens = set(token_pattern.findall(clean_tgt))

    common = src_tokens.intersection(tgt_tokens)
    if not common:
        return False

    # 5. Meaningful overlap evaluation (FINDING-4C2-03)
    macros = {t for t in common if t.startswith("\\")}

    # Solitary common token: only a structural/operator LaTeX macro is considered meaningful.
    # A solitary single variable or number lacks structural overlap and triggers replacement.
    if len(common) == 1:
        tok = next(iter(common))
        return tok.startswith("\\")

    # If both formulas have variables, but share no variables and no macros (only numbers), reject
    src_vars = {t for t in src_tokens if t.isalpha()}
    tgt_vars = {t for t in tgt_tokens if t.isalpha()}
    if src_vars and tgt_vars and not (src_vars & tgt_vars) and not macros:
        return False

    return True


class ConceptualCardCompiler:
    """
    Compiler handler for Tier 4 Conceptual Cards.
    Converts ConceptualCard, MathTex, and TextCard declarations into
    deterministic Manim code with background cards, titles, formula MathTex,
    badges, and inner padding constraints.
    """

    def can_compile(self, entity: EntityDeclaration) -> bool:
        """Determines if this entity is a supported Tier 4 card."""
        if entity.visual_tier == VisualTier.TIER4_CONCEPTUAL_CARD:
            return True
        return entity.component_type in SUPPORTED_CARD_COMPONENTS

    def compile_card(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
        entity_idx: Optional[int] = None,
    ) -> Tuple[str, List[CompilationDiagnostic]]:
        """
        Compiles a Tier 4 conceptual card into Manim Python code.
        Directly uses geom.world_box for positioning and sizing.
        """
        diagnostics: List[CompilationDiagnostic] = []
        init = entity.initial_state or {}
        styling = entity.styling

        cx = round(geom.world_box.x, 4)
        cy = round(geom.world_box.y, 4)
        w = round(max(0.4, geom.world_box.width), 4)
        h = round(max(0.3, geom.world_box.height), 4)

        # Extract content parameters
        title = init.get("title")
        formula = init.get("formula", init.get("math_str"))
        body = init.get("body", init.get("text"))
        badge = init.get("badge")
        subtext = init.get("subtext")

        # Styling parameters
        color_val = styling.color if (styling and styling.color) else init.get("color", "#58C4DD")
        card_stroke = _safe_color(color_val, "#58C4DD")
        fill_opacity = styling.fill_opacity if (styling and styling.fill_opacity is not None) else float(init.get("fill_opacity", 0.85))
        fill_opacity = round(max(0.1, min(fill_opacity, 1.0)), 2)

        init_prefix = f"entities[{entity_idx}].initial_state" if entity_idx is not None else "initial_state"

        # Check for empty content
        if not title and not formula and not body and not badge:
            diagnostics.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                    message=f"ConceptualCard entity '{entity.id}' has no title, formula, body, or badge content.",
                    entity_id=entity.id,
                    field_path=init_prefix,
                    severity="WARNING",
                )
            )

        # Inner dimensions available for text/math
        inner_w = round(w * 0.88, 4)
        inner_h = round(h * 0.82, 4)

        # Content Overflow Check
        self._check_content_overflow(
            entity, w, h, inner_w, inner_h, title, formula, body, diagnostics, field_prefix=init_prefix
        )

        lines: List[str] = [
            f"# ConceptualCard: {entity.id} ({entity.component_type})",
            f"_bg_{var_name} = RoundedRectangle(",
            f"    corner_radius=0.15, width={w}, height={h},",
            f"    stroke_color={card_stroke!r}, stroke_width=2.0,",
            f"    fill_color='#101720', fill_opacity={fill_opacity}",
            f").move_to(np.array([{cx}, {cy}, 0.0]))",
        ]

        sub_elements: List[str] = [f"_bg_{var_name}"]

        # -------------------------------------------------------------
        # Badge Pill Component (if present)
        # -------------------------------------------------------------
        if badge and isinstance(badge, str) and badge.strip():
            b_text = badge.strip().upper()
            b_color = BADGE_COLORS.get(b_text, BADGE_COLORS["DEFAULT"])
            lines.append(f"# Badge: {b_text}")
            lines.append(
                f"_badge_txt_{var_name} = Text({b_text!r}, font_size=11, color='#FFFFFF', weight=BOLD)"
            )
            lines.append(
                f"_badge_box_{var_name} = RoundedRectangle("
                f"corner_radius=0.08, width=max(0.8, _badge_txt_{var_name}.width + 0.25), "
                f"height=0.32, stroke_color={b_color!r}, stroke_width=1.0, "
                f"fill_color={b_color!r}, fill_opacity=0.85"
                f")"
            )
            lines.append(
                f"_badge_grp_{var_name} = VGroup(_badge_box_{var_name}, _badge_txt_{var_name})"
            )
            # Position at top-right inside card
            lines.append(
                f"_badge_grp_{var_name}.next_to(_bg_{var_name}.get_corner(UR), DL, buff=0.12)"
            )
            sub_elements.append(f"_badge_grp_{var_name}")

        # -------------------------------------------------------------
        # Title Component (if present)
        # -------------------------------------------------------------
        if title and isinstance(title, str) and title.strip():
            t_str = title.strip()
            title_color = "#FFD700" if (badge and "INVARIANT" in str(badge)) else "#FFFFFF"
            lines.append(f"# Card Title: {t_str}")
            lines.append(
                f"_title_{var_name} = Text({t_str!r}, font_size=17, color={title_color!r}, weight=BOLD)"
            )
            # Safe width clamping for title
            max_title_w = round(inner_w * (0.65 if badge else 0.95), 4)
            lines.append(
                f"if _title_{var_name}.width > {max_title_w}:\n"
                f"    _title_{var_name}.scale_to_fit_width({max_title_w})"
            )
            lines.append(
                f"_title_{var_name}.next_to(_bg_{var_name}.get_top(), DOWN, buff=0.10)"
            )
            if badge:
                # Align left-ish to avoid overlapping the badge in UR corner
                lines.append(
                    f"_title_{var_name}.shift(LEFT * (_title_{var_name}.get_center()[0] - (_bg_{var_name}.get_left()[0] + _title_{var_name}.width/2.0 + 0.2)))"
                )
            sub_elements.append(f"_title_{var_name}")

        # -------------------------------------------------------------
        # Formula Component (Variant A / Variant C)
        # -------------------------------------------------------------
        if formula and isinstance(formula, str) and formula.strip():
            clean_tex = sanitize_latex(formula)
            lines.append(f"# Card Formula: {clean_tex}")
            # Use raw string repr or SafeMathTex
            lines.append(
                f"_formula_{var_name} = MathTex({clean_tex!r}, font_size=20, color='#58C4DD')"
            )
            # Scale formula to fit inner dimensions
            lines.append(
                f"if _formula_{var_name}.width > {inner_w}:\n"
                f"    _formula_{var_name}.scale_to_fit_width({inner_w})"
            )
            max_fh = round(inner_h * (0.60 if bool(title) else 0.85), 4)
            lines.append(
                f"_max_h_{var_name} = {max_fh}\n"
                f"if _formula_{var_name}.height > _max_h_{var_name}:\n"
                f"    _formula_{var_name}.scale_to_fit_height(_max_h_{var_name})"
            )
            if title:
                lines.append(
                    f"_formula_{var_name}.next_to(_title_{var_name}, DOWN, buff=0.12)"
                )
                lines.append(
                    f"_formula_{var_name}.set_x(_bg_{var_name}.get_center()[0])"
                )
            else:
                lines.append(
                    f"_formula_{var_name}.move_to(_bg_{var_name}.get_center())"
                )
            sub_elements.append(f"_formula_{var_name}")

        # -------------------------------------------------------------
        # Body Text Component (Variant B)
        # -------------------------------------------------------------
        elif body and isinstance(body, str) and body.strip():
            b_str = body.strip()
            lines.append(f"# Card Body Text: {b_str[:40]}...")
            lines.append(
                f"_body_{var_name} = Text({b_str!r}, font_size=13, color='#DDDDDD', line_spacing=0.85)"
            )
            lines.append(
                f"if _body_{var_name}.width > {inner_w}:\n"
                f"    _body_{var_name}.scale_to_fit_width({inner_w})"
            )
            max_bh = round(inner_h * (0.60 if bool(title) else 0.85), 4)
            lines.append(
                f"_max_body_h_{var_name} = {max_bh}\n"
                f"if _body_{var_name}.height > _max_body_h_{var_name}:\n"
                f"    _body_{var_name}.scale_to_fit_height(_max_body_h_{var_name})"
            )
            if title:
                lines.append(
                    f"_body_{var_name}.next_to(_title_{var_name}, DOWN, buff=0.12)"
                )
                lines.append(
                    f"_body_{var_name}.set_x(_bg_{var_name}.get_center()[0])"
                )
            else:
                lines.append(
                    f"_body_{var_name}.move_to(_bg_{var_name}.get_center())"
                )
            sub_elements.append(f"_body_{var_name}")

        # -------------------------------------------------------------
        # Subtext Component (Invariant rules / footnotes)
        # -------------------------------------------------------------
        if subtext and isinstance(subtext, str) and subtext.strip():
            s_str = subtext.strip()
            lines.append(f"# Card Subtext: {s_str}")
            lines.append(
                f"_subtext_{var_name} = Text({s_str!r}, font_size=11, color='#AAAAAA')"
            )
            lines.append(
                f"if _subtext_{var_name}.width > {inner_w}:\n"
                f"    _subtext_{var_name}.scale_to_fit_width({inner_w})"
            )
            lines.append(
                f"_subtext_{var_name}.next_to(_bg_{var_name}.get_bottom(), UP, buff=0.15)"
            )
            sub_elements.append(f"_subtext_{var_name}")

        # Assemble into final composite VGroup
        lines.append(f"{var_name} = VGroup({', '.join(sub_elements)})")
        lines.append(f"{var_name}.set_z_index({geom.z_index})")

        return "\n".join(lines), diagnostics

    # -------------------------------------------------------------------------
    # Content Overflow Diagnostics
    # -------------------------------------------------------------------------
    def _check_content_overflow(
        self,
        entity: EntityDeclaration,
        w: float,
        h: float,
        inner_w: float,
        inner_h: float,
        title: Optional[Any],
        formula: Optional[Any],
        body: Optional[Any],
        diagnostics: List[CompilationDiagnostic],
        field_prefix: str = "initial_state",
    ) -> None:
        """
        Estimates whether declared text or formula content will exceed card bounding
        box and force scaling below the minimum readable floor (< 12pt).
        """
        # Character-based length estimation heuristics
        # Standard character width at font size 18 is approx 0.09 Manim units
        overflow_reasons: List[str] = []
        overflow_fields: List[str] = []

        if title and isinstance(title, str):
            t_chars = len(title.strip())
            est_title_w = t_chars * 0.095
            if est_title_w > inner_w * 1.5:
                overflow_reasons.append(f"Title length ({t_chars} chars) significantly exceeds card width ({inner_w:.2f})")
                overflow_fields.append("title")

        if formula and isinstance(formula, str):
            f_chars = len(formula.strip())
            # LaTeX formulas can be wide
            est_form_w = f_chars * 0.08
            if est_form_w > inner_w * 2.2:
                overflow_reasons.append(f"Formula length ({f_chars} chars) exceeds card capacity and requires severe scaling")
                overflow_fields.append("formula")

        if body and isinstance(body, str):
            b_chars = len(body.strip())
            est_body_w = b_chars * 0.075
            est_lines = max(1.0, est_body_w / max(1.0, inner_w))
            est_body_h = est_lines * 0.35
            if est_body_h > inner_h * 1.4:
                overflow_reasons.append(f"Body text ({b_chars} chars across est. {est_lines:.1f} lines) exceeds card height ({inner_h:.2f})")
                overflow_fields.append("body")

        if overflow_reasons:
            if len(overflow_fields) == 1:
                target_field_path = f"{field_prefix}.{overflow_fields[0]}"
            else:
                target_field_path = field_prefix

            diagnostics.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_CONTENT_OVERFLOW,
                    message=(
                        f"ConceptualCard entity '{entity.id}' content exceeds card bounds "
                        f"({w:.2f}x{h:.2f}): {'; '.join(overflow_reasons)}. "
                        f"Effective font size may drop below minimum legibility floor (12pt)."
                    ),
                    entity_id=entity.id,
                    field_path=target_field_path,
                    severity="WARNING",
                    details={
                        "reasons": overflow_reasons,
                        "overflow_fields": overflow_fields,
                        "card_width": w,
                        "card_height": h,
                        "field_prefix": field_prefix,
                    },
                )
            )

    def compile_card_transform(
        self,
        entity: EntityDeclaration,
        geom: ComputedEntityGeometry,
        var_name: str,
        action: ActionDeclaration,
        current_state: Dict[str, Any],
        step_idx: int = 0,
        a_idx: int = 0,
    ) -> Tuple[List[str], List[CompilationDiagnostic], Dict[str, Any]]:
        """
        Compiles a TRANSFORM action on a Tier 4 Conceptual Card into genuine
        Manim visual replacement/morphing calls (TransformMatchingTex, ReplacementTransform).
        Preserves card identity and geometry, updates only specified fields,
        replaces old submobjects in the parent VGroup, and clamps text/math widths.
        """
        diagnostics: List[CompilationDiagnostic] = []
        params = action.parameters or {}

        # 1. Parameter key safety validation
        for k in params:
            if not is_safe_keyword_identifier(k):
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                        message=f"Action TRANSFORM on entity '{entity.id}' contains invalid parameter key '{k}'.",
                        entity_id=entity.id,
                        field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters",
                        severity="ERROR",
                    )
                )
                return [], diagnostics, current_state

        # 2. Extract transform target values
        new_formula = params.get("formula") or params.get("new_formula") or params.get("math_str") or params.get("target_formula")
        new_title = params.get("title") or params.get("new_title")
        new_body = params.get("body") or params.get("new_body") or params.get("text") or params.get("new_text")
        new_badge = params.get("badge") or params.get("new_badge")
        new_subtext = params.get("subtext") or params.get("new_subtext")
        new_color = params.get("color")
        transform_mode = params.get("transform_mode")

        # 3. Validate parameter types
        type_checks = [
            ("formula", new_formula),
            ("title", new_title),
            ("body", new_body),
            ("badge", new_badge),
            ("subtext", new_subtext),
        ]
        for field_name, val in type_checks:
            if val is not None and not isinstance(val, (str, int, float)):
                diagnostics.append(
                    CompilationDiagnostic(
                        error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                        message=f"Action TRANSFORM on '{entity.id}' has non-scalar value for '{field_name}': {type(val).__name__}.",
                        entity_id=entity.id,
                        field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters.{field_name}",
                        severity="ERROR",
                    )
                )
                return [], diagnostics, current_state

        has_any_field = any([
            new_formula is not None,
            new_title is not None,
            new_body is not None,
            new_badge is not None,
            new_subtext is not None,
            new_color is not None,
        ])
        if not has_any_field:
            diagnostics.append(
                CompilationDiagnostic(
                    error_code=CompilationErrorCode.COMPILER_INVALID_PARAMETER,
                    message=f"Action TRANSFORM on ConceptualCard '{entity.id}' provided no recognized transformation parameters (e.g. formula, title, body, badge).",
                    entity_id=entity.id,
                    field_path=f"timeline[{step_idx}].actions[{a_idx}].parameters",
                    severity="ERROR",
                )
            )
            return [], diagnostics, current_state

        # 4. Inner bounding dimensions
        w = round(max(0.4, geom.world_box.width), 4)
        h = round(max(0.3, geom.world_box.height), 4)
        inner_w = round(w * 0.88, 4)
        inner_h = round(h * 0.82, 4)
        dur = round(max(0.1, action.duration), 2)

        # 5. Check content overflow on new content
        self._check_content_overflow(
            entity,
            w,
            h,
            inner_w,
            inner_h,
            new_title,
            new_formula,
            new_body,
            diagnostics,
            field_prefix=f"timeline[{step_idx}].actions[{a_idx}].parameters",
        )

        suffix = f"{step_idx}_{a_idx}"
        lines: List[str] = [f"# Transform ConceptualCard Content: {entity.id}"]
        play_animations: List[str] = []
        post_updates: List[str] = []
        updated_state = dict(current_state)

        # -------------------------------------------------------------
        # Badge Update
        # -------------------------------------------------------------
        if new_badge is not None:
            b_text = str(new_badge).strip().upper()
            b_color = BADGE_COLORS.get(b_text, BADGE_COLORS["DEFAULT"])
            _btxt = f"_new_badge_txt_{var_name}_{suffix}"
            _bbox = f"_new_badge_box_{var_name}_{suffix}"
            _bgrp = f"_new_badge_grp_{var_name}_{suffix}"
            lines.append(f"# Transform Badge: {b_text}")
            lines.append(f"{_btxt} = Text({b_text!r}, font_size=11, color='#FFFFFF', weight=BOLD)")
            lines.append(
                f"{_bbox} = RoundedRectangle("
                f"corner_radius=0.08, width=max(0.8, {_btxt}.width + 0.25), "
                f"height=0.32, stroke_color={b_color!r}, stroke_width=1.0, "
                f"fill_color={b_color!r}, fill_opacity=0.85)"
            )
            lines.append(f"{_bgrp} = VGroup({_bbox}, {_btxt})")
            lines.append(f"{_bgrp}.next_to(_bg_{var_name}.get_corner(UR), DL, buff=0.12)")

            if "badge" in current_state:
                play_animations.append(f"ReplacementTransform(_badge_grp_{var_name}, {_bgrp})")
                post_updates.append(f"{var_name}.remove(_badge_grp_{var_name}).add({_bgrp})")
                post_updates.append(f"_badge_grp_{var_name} = {_bgrp}")
            else:
                play_animations.append(f"FadeIn({_bgrp})")
                post_updates.append(f"{var_name}.add({_bgrp})")
                post_updates.append(f"_badge_grp_{var_name} = {_bgrp}")
            updated_state["badge"] = b_text

        # -------------------------------------------------------------
        # Title Update
        # -------------------------------------------------------------
        if new_title is not None:
            t_str = str(new_title).strip()
            active_badge = updated_state.get("badge")
            title_color = "#FFD700" if (active_badge and "INVARIANT" in str(active_badge)) else "#FFFFFF"
            _t_var = f"_new_title_{var_name}_{suffix}"
            max_title_w = round(inner_w * (0.65 if active_badge else 0.95), 4)
            lines.append(f"# Transform Title: {t_str}")
            lines.append(f"{_t_var} = Text({t_str!r}, font_size=17, color={title_color!r}, weight=BOLD)")
            lines.append(f"if {_t_var}.width > {max_title_w}:")
            lines.append(f"    {_t_var}.scale_to_fit_width({max_title_w})")
            lines.append(f"{_t_var}.next_to(_bg_{var_name}.get_top(), DOWN, buff=0.10)")
            if active_badge:
                lines.append(
                    f"{_t_var}.shift(LEFT * ({_t_var}.get_center()[0] - (_bg_{var_name}.get_left()[0] + {_t_var}.width/2.0 + 0.2)))"
                )

            if "title" in current_state:
                play_animations.append(f"ReplacementTransform(_title_{var_name}, {_t_var})")
                post_updates.append(f"{var_name}.remove(_title_{var_name}).add({_t_var})")
                post_updates.append(f"_title_{var_name} = {_t_var}")
            else:
                play_animations.append(f"FadeIn({_t_var})")
                post_updates.append(f"{var_name}.add({_t_var})")
                post_updates.append(f"_title_{var_name} = {_t_var}")
            updated_state["title"] = t_str

        # -------------------------------------------------------------
        # Formula Update
        # -------------------------------------------------------------
        if new_formula is not None:
            clean_tex = sanitize_latex(str(new_formula))
            _f_var = f"_new_formula_{var_name}_{suffix}"
            lines.append(f"# Transform Formula: {clean_tex}")
            lines.append(f"{_f_var} = MathTex({clean_tex!r}, font_size=20, color='#58C4DD')")
            lines.append(f"if {_f_var}.width > {inner_w}:")
            lines.append(f"    {_f_var}.scale_to_fit_width({inner_w})")
            max_fh = round(inner_h * (0.60 if bool(updated_state.get('title')) else 0.85), 4)
            lines.append(f"_max_h_{var_name}_{suffix} = {max_fh}")
            lines.append(f"if {_f_var}.height > _max_h_{var_name}_{suffix}:")
            lines.append(f"    {_f_var}.scale_to_fit_height(_max_h_{var_name}_{suffix})")
            has_title = bool(updated_state.get("title"))
            active_title_ref = f"_new_title_{var_name}_{suffix}" if new_title is not None else f"_title_{var_name}"
            if has_title:
                lines.append(f"{_f_var}.next_to({active_title_ref}, DOWN, buff=0.12)")
                lines.append(f"{_f_var}.set_x(_bg_{var_name}.get_center()[0])")
            else:
                lines.append(f"{_f_var}.move_to(_bg_{var_name}.get_center())")

            src_formula = current_state.get("formula") or current_state.get("math_str")
            use_matching = should_use_matching_tex(src_formula, clean_tex, mode=transform_mode)

            if src_formula:
                if use_matching:
                    play_animations.append(f"TransformMatchingTex(_formula_{var_name}, {_f_var})")
                else:
                    play_animations.append(f"ReplacementTransform(_formula_{var_name}, {_f_var})")
                post_updates.append(f"{var_name}.remove(_formula_{var_name}).add({_f_var})")
                post_updates.append(f"_formula_{var_name} = {_f_var}")
            elif "body" in current_state:
                play_animations.append(f"ReplacementTransform(_body_{var_name}, {_f_var})")
                post_updates.append(f"{var_name}.remove(_body_{var_name}).add({_f_var})")
                post_updates.append(f"_formula_{var_name} = {_f_var}")
                updated_state.pop("body", None)
            else:
                play_animations.append(f"FadeIn({_f_var})")
                post_updates.append(f"{var_name}.add({_f_var})")
                post_updates.append(f"_formula_{var_name} = {_f_var}")
            updated_state["formula"] = clean_tex

        # -------------------------------------------------------------
        # Body Text Update
        # -------------------------------------------------------------
        if new_body is not None:
            b_str = str(new_body).strip()
            _b_var = f"_new_body_{var_name}_{suffix}"
            lines.append(f"# Transform Body Text: {b_str[:40]}...")
            lines.append(f"{_b_var} = Text({b_str!r}, font_size=13, color='#DDDDDD', line_spacing=0.85)")
            lines.append(f"if {_b_var}.width > {inner_w}:")
            lines.append(f"    {_b_var}.scale_to_fit_width({inner_w})")
            max_bh = round(inner_h * (0.60 if bool(updated_state.get('title')) else 0.85), 4)
            lines.append(f"_max_body_h_{var_name}_{suffix} = {max_bh}")
            lines.append(f"if {_b_var}.height > _max_body_h_{var_name}_{suffix}:")
            lines.append(f"    {_b_var}.scale_to_fit_height(_max_body_h_{var_name}_{suffix})")
            has_title = bool(updated_state.get("title"))
            active_title_ref = f"_new_title_{var_name}_{suffix}" if new_title is not None else f"_title_{var_name}"
            if has_title:
                lines.append(f"{_b_var}.next_to({active_title_ref}, DOWN, buff=0.12)")
                lines.append(f"{_b_var}.set_x(_bg_{var_name}.get_center()[0])")
            else:
                lines.append(f"{_b_var}.move_to(_bg_{var_name}.get_center())")

            if "body" in current_state:
                play_animations.append(f"ReplacementTransform(_body_{var_name}, {_b_var})")
                post_updates.append(f"{var_name}.remove(_body_{var_name}).add({_b_var})")
                post_updates.append(f"_body_{var_name} = {_b_var}")
            elif "formula" in current_state or "math_str" in current_state:
                play_animations.append(f"ReplacementTransform(_formula_{var_name}, {_b_var})")
                post_updates.append(f"{var_name}.remove(_formula_{var_name}).add({_b_var})")
                post_updates.append(f"_body_{var_name} = {_b_var}")
                updated_state.pop("formula", None)
                updated_state.pop("math_str", None)
            else:
                play_animations.append(f"FadeIn({_b_var})")
                post_updates.append(f"{var_name}.add({_b_var})")
                post_updates.append(f"_body_{var_name} = {_b_var}")
            updated_state["body"] = b_str

        # -------------------------------------------------------------
        # Subtext Update
        # -------------------------------------------------------------
        if new_subtext is not None:
            s_str = str(new_subtext).strip()
            _s_var = f"_new_subtext_{var_name}_{suffix}"
            lines.append(f"# Transform Subtext: {s_str}")
            lines.append(f"{_s_var} = Text({s_str!r}, font_size=11, color='#AAAAAA')")
            lines.append(f"if {_s_var}.width > {inner_w}:")
            lines.append(f"    {_s_var}.scale_to_fit_width({inner_w})")
            lines.append(f"{_s_var}.next_to(_bg_{var_name}.get_bottom(), UP, buff=0.15)")

            if "subtext" in current_state:
                play_animations.append(f"ReplacementTransform(_subtext_{var_name}, {_s_var})")
                post_updates.append(f"{var_name}.remove(_subtext_{var_name}).add({_s_var})")
                post_updates.append(f"_subtext_{var_name} = {_s_var}")
            else:
                play_animations.append(f"FadeIn({_s_var})")
                post_updates.append(f"{var_name}.add({_s_var})")
                post_updates.append(f"_subtext_{var_name} = {_s_var}")
            updated_state["subtext"] = s_str

        # -------------------------------------------------------------
        # Card Stroke Color Update
        # -------------------------------------------------------------
        if new_color is not None:
            c_val = _safe_color(new_color)
            play_animations.append(f"_bg_{var_name}.animate.set_stroke(color={c_val!r})")
            updated_state["color"] = c_val

        # -------------------------------------------------------------
        # Emission
        # -------------------------------------------------------------
        if play_animations:
            anims_joined = ", ".join(play_animations)
            lines.append(f"self.play({anims_joined}, run_time={dur})")
            lines.extend(post_updates)

        return lines, diagnostics, updated_state
