"""
Reaction Energy Diagram Visual Component
========================================
Configurable visual component for chemical potential energy profiles,
activation energy barriers (Ea), reaction enthalpy (ΔH), transition states,
and catalytic pathway comparisons.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from manim import (
    VGroup,
    VMobject,
    Arrow,
    DoubleArrow,
    DashedLine,
    Line,
    Dot,
    Text,
    Indicate,
    FadeIn,
    Transform,
    Animation,
    AnimationGroup,
    ShowPassingFlash,
    MoveAlongPath,
    UP,
    DOWN,
    RIGHT,
    LEFT,
)

from .. import VisualComponent, VisualRegistry


@VisualRegistry.register("chemistry", "ReactionEnergyDiagram")
class ReactionEnergyDiagram(VisualComponent):
    """
    Configurable visual component for potential energy reaction coordinate diagrams.
    Visualizes exothermic/endothermic profiles, activation energy (Ea),
    reaction enthalpy (ΔH), transition state complexes, and catalyzed pathways.
    """
    component_id = "ReactionEnergyDiagram"
    domain = "chemistry"
    aliases = ["reaction energy", "reaction energy diagram", "energy diagram"]
    preferred_aspect = 1.5

    def __init__(
        self,
        e_reactants: float = 1.4,
        e_transition_state: float = 3.4,
        e_products: float = 0.8,
        catalyzed_e_ts: Optional[float] = 2.4,
        reactants_label: str = "Reactants",
        products_label: str = "Products",
        ts_label: str = "Transition State [‡]",
        width: float = 6.0,
        height: float = 3.6,
        curve_color: str = "#58C4DD",
        catalyzed_color: str = "#83C167",
        text_color: str = "#FFFFFF",
        accent_color: str = "#FFD700",
        show_axes: bool = True,
        show_annotations: bool = True,
        font_size: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.e_reactants = float(e_reactants)
        self.e_ts = float(e_transition_state)
        self.e_products = float(e_products)
        self.catalyzed_e_ts = float(catalyzed_e_ts) if catalyzed_e_ts is not None else None
        self.reactants_text = reactants_label
        self.products_text = products_label
        self.ts_text = ts_label
        self.diag_width = float(width)
        self.diag_height = float(height)
        self.curve_color = curve_color
        self.catalyzed_color = catalyzed_color
        self.text_color = text_color
        self.accent_color = accent_color
        self.show_axes = show_axes
        self.show_annotations = show_annotations
        self.font_size = font_size

        # Groups for z-ordering
        self.axes_group = VGroup()
        self.curves_group = VGroup()
        self.annotations_group = VGroup()
        self.token_group = VGroup()

        self._build_axes()
        self._build_curves()
        self._build_annotations()
        self._build_reaction_token()

        self.add(self.axes_group, self.curves_group, self.annotations_group, self.token_group)
        self.center()

    @property
    def ea(self) -> float:
        return self.e_ts - self.e_reactants

    @property
    def delta_h(self) -> float:
        return self.e_products - self.e_reactants

    def _build_axes(self) -> None:
        """Constructs L-shaped reaction coordinate and potential energy axes."""
        if not self.show_axes:
            return

        origin = np.array([0.0, 0.0, 0.0])
        x_end = np.array([self.diag_width, 0.0, 0.0])
        y_end = np.array([0.0, self.diag_height, 0.0])

        x_axis = Arrow(start=origin, end=x_end, color="#777777", buff=0.0, stroke_width=2.5, max_tip_length_to_length_ratio=0.04)
        y_axis = Arrow(start=origin, end=y_end, color="#777777", buff=0.0, stroke_width=2.5, max_tip_length_to_length_ratio=0.04)

        x_lbl = Text("Reaction Progress", font_size=max(12, int(self.font_size * 0.85)), color="#AAAAAA")
        x_lbl.next_to(x_axis, DOWN, buff=0.15)

        y_lbl = Text("Potential Energy (E)", font_size=max(12, int(self.font_size * 0.85)), color="#AAAAAA")
        y_lbl.rotate(np.pi / 2.0).next_to(y_axis, LEFT, buff=0.15)

        self.axes_group.add(x_axis, y_axis, x_lbl, y_lbl)
        self.register_semantic_handle("axes", self.axes_group)

    def _generate_curve_points(self, ts_height: float) -> List[np.ndarray]:
        """Calculates control points for potential energy profile."""
        w = self.diag_width
        x_r_start = 0.5
        x_r_end = 1.6
        x_ts = w / 2.0
        x_p_start = w - 1.6
        x_p_end = w - 0.4

        pts = [
            np.array([x_r_start, self.e_reactants, 0.0]),
            np.array([x_r_end, self.e_reactants, 0.0]),
            np.array([w * 0.32, (self.e_reactants * 0.3 + ts_height * 0.7), 0.0]),
            np.array([x_ts, ts_height, 0.0]),
            np.array([w * 0.68, (ts_height * 0.7 + self.e_products * 0.3), 0.0]),
            np.array([x_p_start, self.e_products, 0.0]),
            np.array([x_p_end, self.e_products, 0.0]),
        ]
        return pts

    def _build_curves(self) -> None:
        """Constructs the uncatalyzed and optional catalyzed potential energy curves."""
        # 1. Main uncatalyzed profile
        pts = self._generate_curve_points(self.e_ts)
        self.main_curve = VMobject(color=self.curve_color, stroke_width=3.8)
        self.main_curve.set_points_smoothly(pts)
        self.curves_group.add(self.main_curve)
        self.register_semantic_handle("uncatalyzed_curve", self.main_curve)

        # 2. Catalyzed pathway (if configured)
        if self.catalyzed_e_ts is not None:
            cat_pts = self._generate_curve_points(self.catalyzed_e_ts)
            self.cat_curve = VMobject(color=self.catalyzed_color, stroke_width=3.0)
            self.cat_curve.set_points_smoothly(cat_pts)
            self.curves_group.add(self.cat_curve)
            self.register_semantic_handle("catalyzed_curve", self.cat_curve)

    def _build_annotations(self) -> None:
        """Constructs Reactants, Products, TS labels, Ea double arrow, and ΔH arrow."""
        w = self.diag_width
        x_ts = w / 2.0

        # State text labels
        lbl_react = Text(self.reactants_text, font_size=self.font_size, color=self.text_color)
        lbl_react.move_to(np.array([1.05, self.e_reactants + 0.32, 0.0]))

        lbl_prod = Text(self.products_text, font_size=self.font_size, color=self.text_color)
        lbl_prod.move_to(np.array([w - 1.05, self.e_products + 0.32, 0.0]))

        lbl_ts = Text(self.ts_text, font_size=max(12, int(self.font_size * 0.9)), color=self.accent_color)
        lbl_ts.move_to(np.array([x_ts, self.e_ts + 0.35, 0.0]))

        self.annotations_group.add(lbl_react, lbl_prod, lbl_ts)
        self.register_semantic_handle("reactants_label", lbl_react)
        self.register_semantic_handle("products_label", lbl_prod)
        self.register_semantic_handle("ts_label", lbl_ts)

        if not self.show_annotations:
            return

        # Reference dashed line across from Reactants energy level
        dash_ref = DashedLine(
            start=np.array([1.6, self.e_reactants, 0.0]),
            end=np.array([w - 0.4, self.e_reactants, 0.0]),
            color="#555555",
            stroke_width=1.5,
            dash_length=0.08,
        )
        self.annotations_group.add(dash_ref)
        self.register_semantic_handle("reactants_ref_line", dash_ref)

        # Activation Energy (Ea) Arrow
        x_ea = x_ts - 0.35
        ea_arrow = DoubleArrow(
            start=np.array([x_ea, self.e_reactants, 0.0]),
            end=np.array([x_ea, self.e_ts, 0.0]),
            color=self.accent_color,
            buff=0.0,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.15,
        )
        ea_label = Text("Ea", font_size=14, color=self.accent_color).next_to(ea_arrow, LEFT, buff=0.08)
        self.annotations_group.add(ea_arrow, ea_label)
        self.register_semantic_handle("ea_arrow", ea_arrow)
        self.register_semantic_handle("ea_label", ea_label)

        # Reaction Enthalpy (ΔH) Arrow
        x_dh = w - 0.8
        is_exothermic = self.e_products < self.e_reactants
        dh_color = "#83C167" if is_exothermic else "#FF6B6B"
        dh_arrow = Arrow(
            start=np.array([x_dh, self.e_reactants, 0.0]),
            end=np.array([x_dh, self.e_products, 0.0]),
            color=dh_color,
            buff=0.0,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.22,
        )
        dh_text = "ΔH < 0" if is_exothermic else "ΔH > 0"
        dh_label = Text(dh_text, font_size=13, color=dh_color).next_to(dh_arrow, RIGHT, buff=0.08)
        self.annotations_group.add(dh_arrow, dh_label)
        self.register_semantic_handle("delta_h_arrow", dh_arrow)
        self.register_semantic_handle("delta_h_label", dh_label)

    def _build_reaction_token(self) -> None:
        """Constructs particle token representing progressing reactant molecules."""
        start_pt = self.main_curve.get_start()
        self.token = Dot(point=start_pt, color=self.accent_color, radius=0.11)
        self.token_group.add(self.token)
        self.register_semantic_handle("particle", self.token)

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------
    def animate_progress_reaction(
        self,
        color: str = "#FFD700",
        run_time: float = 2.0,
    ) -> AnimationGroup:
        """Animates reactant particle travelling up over activation barrier to products."""
        token = self.get_semantic_handle("particle")
        if token is None:
            token = Dot(point=self.main_curve.get_start(), color=color)
            self.add(token)

        trail = self.main_curve.copy().set_color(color).set_stroke(width=self.main_curve.get_stroke_width() * 1.5)

        return AnimationGroup(
            MoveAlongPath(token, self.main_curve, run_time=run_time),
            ShowPassingFlash(trail, time_width=0.4, run_time=run_time),
        )

    def animate_highlight_transition_state(
        self,
        color: str = "#FFD700",
        run_time: float = 0.6,
    ) -> Animation:
        """Highlights the high-energy Transition State peak."""
        ts_lbl = self.get_semantic_handle("ts_label")
        if ts_lbl is None:
            return Indicate(self, color=color, run_time=run_time)
        return Indicate(ts_lbl, color=color, scale_factor=1.2, run_time=run_time)

    def animate_show_activation_barrier(
        self,
        color: str = "#FFD700",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Indicates and emphasizes the activation energy barrier (Ea)."""
        ea_arrow = self.get_semantic_handle("ea_arrow")
        ea_lbl = self.get_semantic_handle("ea_label")
        if ea_arrow is None or ea_lbl is None:
            return AnimationGroup(Indicate(self, color=color, run_time=run_time))
        return AnimationGroup(
            Indicate(ea_arrow, color=color, run_time=run_time),
            Indicate(ea_lbl, color=color, run_time=run_time),
        )

    def animate_show_enthalpy(
        self,
        color: str = "#83C167",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Indicates and emphasizes the reaction enthalpy (ΔH)."""
        dh_arrow = self.get_semantic_handle("delta_h_arrow")
        dh_lbl = self.get_semantic_handle("delta_h_label")
        if dh_arrow is None or dh_lbl is None:
            return AnimationGroup(Indicate(self, color=color, run_time=run_time))
        return AnimationGroup(
            Indicate(dh_arrow, color=color, run_time=run_time),
            Indicate(dh_lbl, color=color, run_time=run_time),
        )

    def animate_add_catalyst(
        self,
        lower_e_ts: Optional[float] = None,
        color: str = "#83C167",
        run_time: float = 1.2,
    ) -> AnimationGroup:
        """
        Dynamically introduces the catalyzed pathway with lower activation barrier,
        illustrating catalytic rate enhancement.
        """
        cat_curve = self.get_semantic_handle("catalyzed_curve")
        if cat_curve is None:
            e_cat = lower_e_ts or (self.e_ts * 0.7 + self.e_reactants * 0.3)
            cat_pts = self._generate_curve_points(e_cat)
            cat_curve = VMobject(color=color, stroke_width=3.2)
            cat_curve.set_points_smoothly(cat_pts)
            self.curves_group.add(cat_curve)
            self.register_semantic_handle("catalyzed_curve", cat_curve)

        return AnimationGroup(
            FadeIn(cat_curve, run_time=run_time),
            Indicate(cat_curve, color=color, run_time=run_time),
        )
