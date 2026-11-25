"""
Function Curve Visual Component
===============================
Configurable visual component for mathematical function plotting,
coordinate axes, tangent lines, derivatives, roots, extrema,
and integral area representations (Riemann sums and shaded area).
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from manim import (
    VGroup,
    Axes,
    Line,
    Dot,
    Text,
    Indicate,
    Create,
    FadeIn,
    FadeOut,
    Transform,
    Animation,
    AnimationGroup,
    UP,
    DOWN,
    RIGHT,
    LEFT,
)

from .. import VisualComponent, VisualRegistry


# Common Mathematical Function Presets
FUNCTION_PRESETS: Dict[str, Tuple[Callable[[float], float], str]] = {
    "x^2": (lambda x: x**2, "f(x) = x²"),
    "x^3": (lambda x: x**3, "f(x) = x³"),
    "cubic": (lambda x: x**3 - 3 * x, "f(x) = x³ - 3x"),
    "quadratic": (lambda x: x**2 - 2 * x - 1, "f(x) = x² - 2x - 1"),
    "sin": (lambda x: np.sin(x), "f(x) = sin(x)"),
    "sin(x)": (lambda x: np.sin(x), "f(x) = sin(x)"),
    "cos": (lambda x: np.cos(x), "f(x) = cos(x)"),
    "cos(x)": (lambda x: np.cos(x), "f(x) = cos(x)"),
    "exp": (lambda x: np.exp(x), "f(x) = eˣ"),
    "exp(x)": (lambda x: np.exp(x), "f(x) = eˣ"),
    "gauss": (lambda x: np.exp(-x**2), "f(x) = e^(-x²)"),
    "log": (lambda x: np.log(np.maximum(x, 1e-4)), "f(x) = ln(x)"),
    "ln": (lambda x: np.log(np.maximum(x, 1e-4)), "f(x) = ln(x)"),
    "sqrt": (lambda x: np.sqrt(np.maximum(x, 0.0)), "f(x) = √x"),
}


@VisualRegistry.register("math", "FunctionCurve")
class FunctionCurve(VisualComponent):
    """
    Configurable visual component for 2D function plotting and calculus.
    Supports polynomial, trigonometric, exponential, and user-defined functions,
    axes, tangent lines, roots, extrema, area under curves, and Riemann sums.
    """
    component_id = "FunctionCurve"
    domain = "math"
    aliases = ["function curve", "function plot", "curve", "plot"]
    preferred_aspect = 1.45

    def __init__(
        self,
        function: Union[str, Callable[[float], float]] = "cubic",
        x_range: Tuple[float, float, float] = (-3.0, 3.0, 1.0),
        y_range: Tuple[float, float, float] = (-4.0, 4.0, 1.0),
        x_length: float = 6.2,
        y_length: float = 4.0,
        curve_color: str = "#58C4DD",
        curve_label: Optional[str] = None,
        secondary_function: Optional[Union[str, Callable[[float], float]]] = None,
        secondary_color: str = "#FFA07A",
        secondary_label: Optional[str] = None,
        roots: Optional[Sequence[float]] = None,
        show_axes: bool = True,
        font_size: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.x_range = x_range
        self.y_range = y_range
        self.x_length = float(x_length)
        self.y_length = float(y_length)
        self.curve_color = curve_color
        self.secondary_color = secondary_color
        self.show_axes = show_axes
        self.font_size = font_size
        self.explicit_roots = list(roots) if roots is not None else None

        # Parse functions
        self.func, default_label = self._parse_function(function)
        self.curve_label_text = curve_label if curve_label is not None else default_label

        self.sec_func = None
        self.sec_label_text = None
        if secondary_function is not None:
            self.sec_func, sec_def_label = self._parse_function(secondary_function)
            self.sec_label_text = secondary_label if secondary_label is not None else sec_def_label

        # Groups for z-ordering
        self.axes_group = VGroup()
        self.plots_group = VGroup()
        self.annotations_group = VGroup()

        self._build_axes()
        self._build_primary_curve()
        if self.sec_func is not None:
            self._build_secondary_curve()

        self.add(self.axes_group, self.plots_group, self.annotations_group)
        self.center()

    def _parse_function(self, func_spec: Union[str, Callable]) -> Tuple[Callable[[float], float], str]:
        """Parses function string, preset, or callable into a safe vectorized evaluator."""
        if callable(func_spec):
            return func_spec, "f(x)"

        s = str(func_spec).strip()
        if s.lower() in FUNCTION_PRESETS:
            return FUNCTION_PRESETS[s.lower()]

        # Safe custom expression evaluation using numpy namespace
        safe_dict = {
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "ln": np.log,
            "sqrt": np.sqrt,
            "abs": np.abs,
            "pi": np.pi,
            "e": np.e,
        }

        def _custom_fn(x):
            try:
                local_scope = {"x": x}
                # Support elementwise numpy arrays or scalar floats
                val = eval(s, {"__builtins__": {}}, {**safe_dict, **local_scope})
                return val
            except Exception:
                return 0.0

        return _custom_fn, f"f(x) = {s}"

    def _build_axes(self) -> None:
        """Constructs Manim Axes with x and y labels."""
        if not self.show_axes:
            return

        self.axes = Axes(
            x_range=self.x_range,
            y_range=self.y_range,
            x_length=self.x_length,
            y_length=self.y_length,
            axis_config={"color": "#666666", "stroke_width": 2.0},
            tips=True,
        )

        x_lbl = Text("x", font_size=max(12, int(self.font_size * 0.9)), color="#AAAAAA")
        x_lbl.next_to(self.axes.x_axis.get_end(), RIGHT, buff=0.12)

        y_lbl = Text("y", font_size=max(12, int(self.font_size * 0.9)), color="#AAAAAA")
        y_lbl.next_to(self.axes.y_axis.get_end(), UP, buff=0.12)

        self.axes_group.add(self.axes, x_lbl, y_lbl)
        self.register_semantic_handle("axes", self.axes)
        self.register_semantic_handle("x_axis", self.axes.x_axis)
        self.register_semantic_handle("y_axis", self.axes.y_axis)

    def _build_primary_curve(self) -> None:
        """Plots the primary function curve."""
        x_min, x_max, _ = self.x_range
        y_min, y_max, _ = self.y_range

        # Clip function to prevent off-axis spillovers
        def _safe_f(x):
            val = self.func(x)
            return np.clip(val, y_min - 1.5, y_max + 1.5)

        self.main_curve = self.axes.plot(_safe_f, x_range=[x_min, x_max], color=self.curve_color, stroke_width=3.6)
        self.plots_group.add(self.main_curve)

        self.register_semantic_handle("curve", self.main_curve)
        self.register_semantic_handle("main_curve", self.main_curve)

        if self.curve_label_text:
            lbl = Text(self.curve_label_text, font_size=self.font_size, color=self.curve_color)
            lbl.next_to(self.main_curve.get_end(), UP + RIGHT, buff=0.15)
            # Ensure label stays inside container limits
            if lbl.get_top()[1] > self.axes.y_axis.get_top()[1]:
                lbl.next_to(self.main_curve.get_end(), DOWN + RIGHT, buff=0.15)
            self.plots_group.add(lbl)
            self.register_semantic_handle("curve_label", lbl)

    def _build_secondary_curve(self) -> None:
        """Plots the optional secondary curve for comparison or intersections."""
        if self.sec_func is None:
            return

        x_min, x_max, _ = self.x_range
        y_min, y_max, _ = self.y_range

        def _safe_g(x):
            val = self.sec_func(x)
            return np.clip(val, y_min - 1.5, y_max + 1.5)

        self.sec_curve = self.axes.plot(_safe_g, x_range=[x_min, x_max], color=self.secondary_color, stroke_width=3.0)
        self.plots_group.add(self.sec_curve)

        self.register_semantic_handle("secondary_curve", self.sec_curve)

        if self.sec_label_text:
            lbl = Text(self.sec_label_text, font_size=self.font_size, color=self.secondary_color)
            lbl.next_to(self.sec_curve.get_end(), UP + RIGHT, buff=0.15)
            self.plots_group.add(lbl)
            self.register_semantic_handle("secondary_curve_label", lbl)

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------
    def animate_plot_curve(self, run_time: float = 1.5) -> Animation:
        """Animates drawing the function curve along the coordinate axes."""
        return Create(self.main_curve, run_time=run_time)

    def animate_tangent(
        self,
        x_val: float,
        delta: float = 1.0,
        color: str = "#FFD700",
        show_slope: bool = True,
        run_time: float = 1.0,
    ) -> AnimationGroup:
        """
        Calculates and animates the tangent line and point at x = x_val,
        displaying numerical slope m = f'(x).
        """
        h = 1e-5
        y0 = float(self.func(x_val))
        slope = float((self.func(x_val + h) - self.func(x_val - h)) / (2.0 * h))

        p_center = self.axes.c2p(x_val, y0)
        p_start = self.axes.c2p(x_val - delta, y0 - slope * delta)
        p_end = self.axes.c2p(x_val + delta, y0 + slope * delta)

        tangent_line = Line(p_start, p_end, color=color, stroke_width=3.5)
        tangent_dot = Dot(p_center, color=color, radius=0.10)

        self.annotations_group.add(tangent_line, tangent_dot)
        self.register_semantic_handle("tangent_line", tangent_line)
        self.register_semantic_handle("tangent_point", tangent_dot)

        anims = [Create(tangent_line, run_time=run_time), FadeIn(tangent_dot, run_time=run_time * 0.7)]

        if show_slope:
            slope_str = f"m = {slope:.2f}"
            slope_lbl = Text(slope_str, font_size=13, color=color).next_to(tangent_dot, UP + RIGHT, buff=0.12)
            self.annotations_group.add(slope_lbl)
            self.register_semantic_handle("slope_label", slope_lbl)
            anims.append(FadeIn(slope_lbl, run_time=run_time * 0.8))

        return AnimationGroup(*anims)

    def animate_show_roots(
        self,
        color: str = "#FFD700",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Identifies and highlights zeroes/roots where f(x) = 0."""
        # Find roots
        roots = []
        if self.explicit_roots is not None:
            roots = self.explicit_roots
        else:
            # Numerical sign-change search
            x_min, x_max, _ = self.x_range
            grid_x = np.linspace(x_min, x_max, 200)
            grid_y = np.array([self.func(x) for x in grid_x])
            for i in range(len(grid_x) - 1):
                if grid_y[i] * grid_y[i + 1] <= 0:
                    r = grid_x[i] - grid_y[i] * (grid_x[i + 1] - grid_x[i]) / (grid_y[i + 1] - grid_y[i] + 1e-9)
                    roots.append(round(float(r), 2))

        anims = []
        for idx, r in enumerate(roots):
            p = self.axes.c2p(r, 0.0)
            dot = Dot(p, color=color, radius=0.11)
            lbl = Text(f"x={r}", font_size=12, color=color).next_to(dot, DOWN, buff=0.12)
            root_group = VGroup(dot, lbl)
            self.annotations_group.add(root_group)
            self.register_semantic_handle(f"root_{idx}", root_group)
            anims.append(FadeIn(root_group, run_time=run_time))
            anims.append(Indicate(dot, color=color, run_time=run_time))

        if not anims:
            return AnimationGroup(Indicate(self.main_curve, color=color, run_time=run_time))
        return AnimationGroup(*anims, lag_ratio=0.2)

    def animate_show_extrema(
        self,
        color: str = "#FFA07A",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Identifies and highlights local minima and maxima (f'(x) = 0)."""
        x_min, x_max, _ = self.x_range
        grid_x = np.linspace(x_min + 0.1, x_max - 0.1, 150)
        h = 1e-4
        derivs = np.array([(self.func(x + h) - self.func(x - h)) / (2.0 * h) for x in grid_x])

        extrema = []
        for i in range(len(grid_x) - 1):
            if derivs[i] * derivs[i + 1] < 0:
                x_ext = grid_x[i]
                y_ext = float(self.func(x_ext))
                extrema.append((x_ext, y_ext))

        anims = []
        for idx, (xe, ye) in enumerate(extrema):
            p = self.axes.c2p(xe, ye)
            dot = Dot(p, color=color, radius=0.11)
            lbl = Text(f"({xe:.1f}, {ye:.1f})", font_size=12, color=color).next_to(dot, UP, buff=0.12)
            ext_group = VGroup(dot, lbl)
            self.annotations_group.add(ext_group)
            self.register_semantic_handle(f"extrema_{idx}", ext_group)
            anims.append(FadeIn(ext_group, run_time=run_time))

        if not anims:
            return AnimationGroup(Indicate(self.main_curve, color=color, run_time=run_time))
        return AnimationGroup(*anims, lag_ratio=0.2)

    def animate_shade_area(
        self,
        x_start: float,
        x_end: float,
        color: str = "#58C4DD",
        opacity: float = 0.35,
        run_time: float = 1.0,
    ) -> Animation:
        """Visualizes the definite integral ∫[a,b] f(x)dx by shading the area under curve."""
        area = self.axes.get_area(self.main_curve, x_range=[x_start, x_end], color=color, opacity=opacity)
        self.annotations_group.add(area)
        self.register_semantic_handle("area", area)
        return FadeIn(area, run_time=run_time)

    def animate_riemann_rectangles(
        self,
        x_start: float,
        x_end: float,
        dx: float = 0.4,
        color: str = "#83C167",
        run_time: float = 1.2,
    ) -> Animation:
        """Constructs and draws Riemann approximation rectangles for definite integration."""
        rects = self.axes.get_riemann_rectangles(
            self.main_curve,
            x_range=[x_start, x_end],
            dx=dx,
            color=color,
            stroke_color=color,
            fill_opacity=0.45,
            stroke_width=1.5,
        )
        self.annotations_group.add(rects)
        self.register_semantic_handle("riemann_rectangles", rects)
        return FadeIn(rects, run_time=run_time)

    def animate_show_intersection(
        self,
        color: str = "#FFD700",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Finds and highlights intersections between primary and secondary curves."""
        if self.sec_func is None:
            return AnimationGroup(Indicate(self.main_curve, color=color, run_time=run_time))

        x_min, x_max, _ = self.x_range
        grid_x = np.linspace(x_min, x_max, 200)
        diff = np.array([self.func(x) - self.sec_func(x) for x in grid_x])

        intersections = []
        for i in range(len(grid_x) - 1):
            if diff[i] * diff[i + 1] <= 0:
                xi = grid_x[i]
                yi = float(self.func(xi))
                intersections.append((xi, yi))

        anims = []
        for idx, (xi, yi) in enumerate(intersections):
            p = self.axes.c2p(xi, yi)
            dot = Dot(p, color=color, radius=0.12)
            lbl = Text(f"({xi:.1f}, {yi:.1f})", font_size=12, color=color).next_to(dot, UP, buff=0.12)
            grp = VGroup(dot, lbl)
            self.annotations_group.add(grp)
            self.register_semantic_handle(f"intersection_{idx}", grp)
            anims.append(FadeIn(grp, run_time=run_time))
            anims.append(Indicate(dot, color=color, run_time=run_time))

        if not anims:
            return AnimationGroup(Indicate(self.main_curve, color=color, run_time=run_time))
        return AnimationGroup(*anims, lag_ratio=0.2)

    def animate_transform_function(
        self,
        new_func: Union[str, Callable[[float], float]],
        new_label: Optional[str] = None,
        run_time: float = 1.2,
    ) -> AnimationGroup:
        """Morphs the curve smoothly into a new mathematical function."""
        parsed_fn, def_lbl = self._parse_function(new_func)
        self.func = parsed_fn

        x_min, x_max, _ = self.x_range
        y_min, y_max, _ = self.y_range

        def _safe_new(x):
            return np.clip(parsed_fn(x), y_min - 1.5, y_max + 1.5)

        new_curve = self.axes.plot(_safe_new, x_range=[x_min, x_max], color=self.curve_color, stroke_width=3.6)

        anims = [Transform(self.main_curve, new_curve, run_time=run_time)]

        old_lbl = self.get_semantic_handle("curve_label")
        if old_lbl is not None:
            new_txt = new_label or def_lbl
            new_lbl_mob = Text(new_txt, font_size=self.font_size, color=self.curve_color)
            new_lbl_mob.next_to(new_curve.get_end(), UP + RIGHT, buff=0.15)
            anims.append(Transform(old_lbl, new_lbl_mob, run_time=run_time))

        return AnimationGroup(*anims)
