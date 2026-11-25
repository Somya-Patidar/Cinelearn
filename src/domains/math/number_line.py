"""
Number Line Visual Component
============================
Configurable 1D number line component for real intervals, inequalities,
domain constraints, limit sequences, and point markers.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from manim import (
    VGroup,
    Line,
    Arrow,
    Dot,
    Circle,
    Text,
    Indicate,
    Create,
    FadeIn,
    Transform,
    Animation,
    AnimationGroup,
    UP,
    DOWN,
    RIGHT,
    LEFT,
)

from .. import VisualComponent, VisualRegistry


@VisualRegistry.register("math", "NumberLineVisual")
class NumberLineVisual(VisualComponent):
    """
    Configurable 1D number line component.
    Supports closed/open intervals, numerical inequalities, limit sequences,
    and named value pointers without external LaTeX dependencies.
    """
    component_id = "NumberLineVisual"
    domain = "math"
    aliases = ["number line", "number_line"]
    preferred_aspect = 2.4

    def __init__(
        self,
        x_range: Tuple[float, float, float] = (-5.0, 5.0, 1.0),
        length: float = 6.6,
        include_ticks: bool = True,
        include_numbers: bool = True,
        line_color: str = "#888888",
        tick_color: str = "#777777",
        text_color: str = "#CCCCCC",
        font_size: int = 14,
        intervals: Optional[Sequence[Tuple[float, float, bool, bool]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.x_min = float(x_range[0])
        self.x_max = float(x_range[1])
        self.step = float(x_range[2]) if len(x_range) > 2 else 1.0
        self.line_length = float(length)
        self.include_ticks = include_ticks
        self.include_numbers = include_numbers
        self.line_color = line_color
        self.tick_color = tick_color
        self.text_color = text_color
        self.font_size = font_size

        self.start_pt = np.array([-self.line_length / 2.0, 0.0, 0.0])
        self.end_pt = np.array([self.line_length / 2.0, 0.0, 0.0])

        self.line_group = VGroup()
        self.ticks_group = VGroup()
        self.numbers_group = VGroup()
        self.intervals_group = VGroup()
        self.pointers_group = VGroup()

        self._build_number_line()

        if intervals:
            for idx, item in enumerate(intervals):
                s, e = item[0], item[1]
                cs = item[2] if len(item) > 2 else True
                ce = item[3] if len(item) > 3 else True
                self._add_interval_geometry(s, e, cs, ce, f"interval_{idx}")

        self.add(
            self.line_group,
            self.ticks_group,
            self.numbers_group,
            self.intervals_group,
            self.pointers_group,
        )
        self.center()

    @property
    def x_range(self) -> Tuple[float, float, float]:
        return (self.x_min, self.x_max, self.step)

    def n2p(self, val: float) -> np.ndarray:
        """Converts scalar number to 3D point coordinates along the line."""
        span = max(1e-4, self.x_max - self.x_min)
        alpha = (float(val) - self.x_min) / span
        return self.start_pt + alpha * (self.end_pt - self.start_pt)

    def _build_number_line(self) -> None:
        """Constructs native line and ticks."""
        main_axis = Line(
            start=self.start_pt + LEFT * 0.15,
            end=self.end_pt + RIGHT * 0.15,
            color=self.line_color,
            stroke_width=2.5,
        )
        self.line_group.add(main_axis)
        self.register_semantic_handle("line", main_axis)

        # Generate ticks and numbers
        num_steps = int(round((self.x_max - self.x_min) / self.step))
        for i in range(num_steps + 1):
            val = self.x_min + i * self.step
            pt = self.n2p(val)

            if self.include_ticks:
                is_major = (val == 0.0 or val == self.x_min or val == self.x_max)
                tick_h = 0.10 if is_major else 0.07
                tick = Line(pt + UP * tick_h, pt + DOWN * tick_h, color=self.tick_color, stroke_width=2.0)
                self.ticks_group.add(tick)

            if self.include_numbers:
                val_str = f"{int(val)}" if abs(val - round(val)) < 1e-4 else f"{val:.1f}"
                lbl = Text(val_str, font_size=self.font_size, color=self.text_color)
                lbl.next_to(pt, DOWN, buff=0.15)
                self.numbers_group.add(lbl)

        self.register_semantic_handle("ticks", self.ticks_group)
        self.register_semantic_handle("numbers", self.numbers_group)

    def _add_interval_geometry(
        self,
        start: float,
        end: float,
        closed_start: bool = True,
        closed_end: bool = True,
        handle_name: str = "interval",
        color: str = "#83C167",
    ) -> VGroup:
        """Constructs an interval segment with closed (filled) or open (hollow) endpoint dots."""
        p_s = self.n2p(start)
        p_e = self.n2p(end)

        segment = Line(p_s, p_e, color=color, stroke_width=5.0)

        r = 0.08
        if closed_start:
            dot_s = Dot(p_s, color=color, radius=r)
        else:
            dot_s = Circle(radius=r, color=color, fill_color="#000000", fill_opacity=1.0, stroke_width=2.5).move_to(p_s)

        if closed_end:
            dot_e = Dot(p_e, color=color, radius=r)
        else:
            dot_e = Circle(radius=r, color=color, fill_color="#000000", fill_opacity=1.0, stroke_width=2.5).move_to(p_e)

        int_grp = VGroup(segment, dot_s, dot_e)
        self.intervals_group.add(int_grp)
        self.register_semantic_handle(handle_name, int_grp)
        return int_grp

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------
    def animate_highlight_number(
        self,
        val: float,
        color: str = "#FFD700",
        run_time: float = 0.6,
    ) -> AnimationGroup:
        """Places a highlighted dot and visual pulse at number val."""
        pt = self.n2p(val)
        dot = Dot(pt, color=color, radius=0.12)
        self.pointers_group.add(dot)
        self.register_semantic_handle(f"highlight_{val}", dot)
        return AnimationGroup(
            FadeIn(dot, run_time=run_time * 0.5),
            Indicate(dot, color=color, scale_factor=1.3, run_time=run_time),
        )

    def animate_show_interval(
        self,
        start: float,
        end: float,
        closed_start: bool = True,
        closed_end: bool = True,
        color: str = "#83C167",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Animates drawing a continuous numerical interval."""
        handle_name = f"interval_{start}_{end}"
        int_grp = self._add_interval_geometry(start, end, closed_start, closed_end, handle_name, color)
        return AnimationGroup(
            Create(int_grp[0], run_time=run_time),
            FadeIn(int_grp[1], run_time=run_time * 0.6),
            FadeIn(int_grp[2], run_time=run_time * 0.6),
        )

    def animate_add_pointer(
        self,
        name: str,
        val: float,
        color: str = "#FFD700",
        run_time: float = 0.6,
    ) -> AnimationGroup:
        """Adds a named vertical pointer arrow and label pointing to val."""
        pt = self.n2p(val)
        arrow = Arrow(
            start=pt + UP * 0.75,
            end=pt + UP * 0.12,
            color=color,
            buff=0.0,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.25,
        )
        lbl = Text(name, font_size=13, color=color).next_to(arrow, UP, buff=0.08)
        ptr_grp = VGroup(arrow, lbl)
        self.pointers_group.add(ptr_grp)
        self.register_semantic_handle(f"pointer_{name}", ptr_grp)
        return AnimationGroup(FadeIn(ptr_grp, run_time=run_time))

    def animate_move_pointer(
        self,
        name: str,
        new_val: float,
        run_time: float = 0.8,
    ) -> Animation:
        """Slides an existing pointer to a new coordinate along the number line."""
        ptr = self.get_semantic_handle(f"pointer_{name}")
        if ptr is None:
            return Indicate(self.line_group, run_time=run_time)

        target_x = self.n2p(new_val)[0]
        dx = target_x - ptr.get_bottom()[0]
        return ptr.animate(run_time=run_time).shift(RIGHT * dx)

    def animate_approach_limit(
        self,
        target_val: float,
        start_val: float,
        color: str = "#58C4DD",
        num_steps: int = 4,
        run_time: float = 1.2,
    ) -> AnimationGroup:
        """Visualizes sequence steps approaching a limit value."""
        anims = []
        step_dt = run_time / max(1, num_steps)
        current_v = start_val
        for i in range(num_steps):
            current_v = current_v + (target_val - current_v) * 0.55
            pt = self.n2p(current_v)
            dot = Dot(pt, color=color, radius=0.09)
            self.pointers_group.add(dot)
            anims.append(FadeIn(dot, run_time=step_dt))

        target_pt = self.n2p(target_val)
        target_dot = Dot(target_pt, color="#FFD700", radius=0.12)
        self.pointers_group.add(target_dot)
        anims.append(Indicate(target_dot, color="#FFD700", run_time=step_dt * 1.5))

        return AnimationGroup(*anims, lag_ratio=0.3)
