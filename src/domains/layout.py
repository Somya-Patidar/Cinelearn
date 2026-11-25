"""
CineLearn Generic Container Layout Engine
=========================================
Domain-independent spatial layout management for educational video generation.
Operates strictly on generic 2D bounding boxes, viewports, anchors, and aspect ratios.
Does NOT depend on any domain-specific shapes, topics, or components.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class RectangularViewport:
    """
    Represents a bounded 2D rectangular screen region with explicit coordinate boundaries.
    All coordinates use Manim standard world units: (0, 0, 0) at screen center.
    """

    def __init__(self, name: str, left: float, right: float, bottom: float, top: float):
        if left >= right:
            raise ValueError(f"Invalid viewport '{name}': left ({left}) must be < right ({right})")
        if bottom >= top:
            raise ValueError(f"Invalid viewport '{name}': bottom ({bottom}) must be < top ({top})")
        self.name = name
        self.left = float(left)
        self.right = float(right)
        self.bottom = float(bottom)
        self.top = float(top)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.left + self.right) / 2.0, (self.bottom + self.top) / 2.0, 0.0])

    def contains_point(self, point: np.ndarray, tolerance: float = 0.01) -> bool:
        x, y = point[0], point[1]
        return (self.left - tolerance <= x <= self.right + tolerance and
                self.bottom - tolerance <= y <= self.top + tolerance)

    def contains_box(self, box: Dict[str, float], tolerance: float = 0.05) -> bool:
        return (box["left"] >= self.left - tolerance and
                box["right"] <= self.right + tolerance and
                box["bottom"] >= self.bottom - tolerance and
                box["top"] <= self.top + tolerance)

    def clamp_mobject(self, mobject: Any) -> Any:
        """Clamps a Manim mobject strictly within the viewport and absolute canvas boundaries."""
        try:
            w = float(mobject.width) if hasattr(mobject, "width") else float(mobject.get_width())
            h = float(mobject.height) if hasattr(mobject, "height") else float(mobject.get_height())
            if w > self.width:
                mobject.scale_to_fit_width(self.width * 0.98)
            if h > self.height:
                mobject.scale_to_fit_height(self.height * 0.98)

            top_y = float(mobject.get_top()[1])
            bottom_y = float(mobject.get_bottom()[1])
            left_x = float(mobject.get_left()[0])
            right_x = float(mobject.get_right()[0])

            shift_x = 0.0
            shift_y = 0.0

            if top_y > self.top:
                shift_y -= (top_y - self.top)
            elif bottom_y < self.bottom:
                shift_y += (self.bottom - bottom_y)

            if left_x < self.left:
                shift_x += (self.left - left_x)
            elif right_x > self.right:
                shift_x -= (right_x - self.right)

            if shift_x != 0.0 or shift_y != 0.0:
                mobject.shift(np.array([shift_x, shift_y, 0.0]))

            # Absolute hard barrier against caption column intrusion (X < -2.15)
            curr_left = float(mobject.get_left()[0])
            if curr_left < -2.15:
                mobject.shift(np.array([-2.15 - curr_left, 0.0, 0.0]))

            # Absolute hard barrier against subtitle zone intrusion (Y < -2.65)
            curr_bottom = float(mobject.get_bottom()[1])
            if curr_bottom < -2.65:
                mobject.shift(np.array([0.0, -2.65 - curr_bottom, 0.0]))

            # Absolute hard barrier against top title intrusion (Y > 2.55)
            curr_top = float(mobject.get_top()[1])
            if curr_top > 2.55:
                mobject.shift(np.array([0.0, 2.55 - curr_top, 0.0]))

            # Absolute hard barrier against right screen edge (X > 6.60)
            curr_right = float(mobject.get_right()[0])
            if curr_right > 6.60:
                mobject.shift(np.array([6.60 - curr_right, 0.0, 0.0]))
        except Exception:
            pass
        return mobject

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "left": self.left,
            "right": self.right,
            "bottom": self.bottom,
            "top": self.top,
            "width": self.width,
            "height": self.height,
            "center": [float(self.center[0]), float(self.center[1]), 0.0],
        }


class GenericContainerLayout:
    """
    Core layout engine that manages bounded rectangular viewports on the Manim canvas.
    Widescreen standard 16:9 canvas dimensions: X in [-7.11, 7.11], Y in [-4.00, 4.00].
    
    Safe Zones & Buffers:
    - Top strip: Section Title reserved zone (Y in [2.70, 3.80])
    - Left column: Lecture Bullet list reserved zone (X < -2.15)
    - Bottom strip: Subtitle / Captions reserved zone (Y < -2.75)
    - Active Visual Canvas: X in [-2.15, 6.60], Y in [-2.65, 2.55]
    """

    def __init__(self, scene: Optional[Any] = None):
        self.scene = scene

        # Standard bounded viewports within the active visual canvas
        self.header = RectangularViewport("header", left=-2.50, right=2.50, bottom=2.70, top=3.80)
        self.main_canvas = RectangularViewport("main_canvas", left=-2.15, right=6.60, bottom=-1.55, top=2.55)
        self.inspector = RectangularViewport("inspector", left=-2.15, right=6.60, bottom=-2.65, top=-1.65)
        self.split_left = RectangularViewport("split_left", left=-2.15, right=2.10, bottom=-2.65, top=2.55)
        self.split_right = RectangularViewport("split_right", left=2.35, right=6.60, bottom=-2.65, top=2.55)
        self.full_canvas = RectangularViewport("full_canvas", left=-2.15, right=6.60, bottom=-2.65, top=2.55)

        self.viewports = {
            "header": self.header,
            "main_canvas": self.main_canvas,
            "main": self.main_canvas,
            "inspector": self.inspector,
            "split_left": self.split_left,
            "split_right": self.split_right,
            "full_canvas": self.full_canvas,
            "full": self.full_canvas,
        }
        self.active_docked: Dict[str, List[Any]] = {
            "main": [],
            "inspector": [],
            "split_left": [],
            "split_right": [],
            "full": [],
        }

    def _get_active_in_viewport(self, vp_key: str) -> List[Any]:
        """Returns currently active mobjects in the viewport (excluding removed/faded mobjects)."""
        docked = self.active_docked.get(vp_key, [])
        if not self.scene:
            return docked
        active = []
        scene_mobs = getattr(self.scene, "mobjects", [])
        for m in docked:
            if m in scene_mobs or any(m in getattr(sm, "submobjects", []) for sm in scene_mobs):
                active.append(m)
        return active

    def _auto_fit_to_viewport(self, mobject: Any, viewport: RectangularViewport, padding: float = 0.04) -> Any:
        """
        Scales mobject so it fits strictly within the viewport without exceeding boundaries.
        Works with VisualComponent instances, standard Manim VGroups, SVGs, or procedural shapes.
        """
        try:
            if hasattr(mobject, "auto_fit_container") and callable(mobject.auto_fit_container):
                mobject.auto_fit_container(viewport.width, viewport.height, padding=padding)
            else:
                w = float(mobject.width) if hasattr(mobject, "width") else float(mobject.get_width())
                h = float(mobject.height) if hasattr(mobject, "height") else float(mobject.get_height())
                if w > 0 and h > 0:
                    target_w = viewport.width * (1.0 - padding)
                    target_h = viewport.height * (1.0 - padding)
                    scale_factor = min(target_w / w, target_h / h, 1.0)
                    min_floor = float(getattr(mobject, "min_scale_threshold", 0.65) or 0.65)
                    scale_factor = max(scale_factor, min_floor)
                    mobject.scale(scale_factor)
        except Exception:
            pass
        return mobject

    def _dock_to_viewport(
        self,
        vp_key: str,
        viewport: RectangularViewport,
        mobject: Any,
        auto_scale: bool = True,
        padding: float = 0.04,
        default_direction: np.ndarray = np.array([0, -1, 0]),
        default_buff: float = 0.20,
        **kwargs,
    ) -> Any:
        if auto_scale:
            self._auto_fit_to_viewport(mobject, viewport, padding=padding)

        active_mobs = self._get_active_in_viewport(vp_key)

        if "next_to" in kwargs and hasattr(kwargs["next_to"], "get_center"):
            direction = kwargs.get("direction", default_direction)
            buff = kwargs.get("buff", default_buff)
            mobject.next_to(kwargs["next_to"], direction, buff=buff)
        elif active_mobs:
            # Another object is already active in this viewport: arrange relative to prevent stacking collisions
            prev_mob = active_mobs[-1]
            direction = kwargs.get("direction", default_direction)
            buff = kwargs.get("buff", default_buff)
            mobject.next_to(prev_mob, direction, buff=buff)
        else:
            mobject.move_to(viewport.center)

        viewport.clamp_mobject(mobject)
        if vp_key in self.active_docked:
            self.active_docked[vp_key].append(mobject)
        self._register_with_scene(mobject)
        return mobject

    def clamp_mobject(self, mobject: Any) -> Any:
        """Clamps mobject to ensure it stays strictly within the safe visual canvas."""
        return self.full_canvas.clamp_mobject(mobject)

    def dock_main(self, mobject: Any, auto_scale: bool = True, padding: float = 0.04, **kwargs) -> Any:
        """Docks a primary visual into the main canvas viewport."""
        return self._dock_to_viewport(
            "main", self.main_canvas, mobject, auto_scale=auto_scale, padding=padding,
            default_direction=np.array([0, -1, 0]), default_buff=0.20, **kwargs
        )

    def dock_inspector(self, mobject: Any, auto_scale: bool = True, padding: float = 0.04, **kwargs) -> Any:
        """
        Docks a secondary inspector/table/equation component into the bottom inspector zone.
        Anchored to top of inspector zone (Y = -1.65) and clamped at Y >= -2.65.
        Guaranteed to NEVER intrude into the subtitle safe zone (Y < -2.75).
        """
        return self._dock_to_viewport(
            "inspector", self.inspector, mobject, auto_scale=auto_scale, padding=padding,
            default_direction=np.array([1, 0, 0]), default_buff=0.25, **kwargs
        )

    def dock_split_left(self, mobject: Any, auto_scale: bool = True, padding: float = 0.04, **kwargs) -> Any:
        """Docks a component into the left split viewport for side-by-side comparison."""
        return self._dock_to_viewport(
            "split_left", self.split_left, mobject, auto_scale=auto_scale, padding=padding,
            default_direction=np.array([0, -1, 0]), default_buff=0.20, **kwargs
        )

    def dock_split_right(self, mobject: Any, auto_scale: bool = True, padding: float = 0.04, **kwargs) -> Any:
        """Docks a component into the right split viewport for side-by-side comparison."""
        return self._dock_to_viewport(
            "split_right", self.split_right, mobject, auto_scale=auto_scale, padding=padding,
            default_direction=np.array([0, -1, 0]), default_buff=0.20, **kwargs
        )

    def dock_full(self, mobject: Any, auto_scale: bool = True, padding: float = 0.04, **kwargs) -> Any:
        """Docks a single large visual occupying the entire active canvas."""
        return self._dock_to_viewport(
            "full", self.full_canvas, mobject, auto_scale=auto_scale, padding=padding,
            default_direction=np.array([0, -1, 0]), default_buff=0.20, **kwargs
        )

    def _register_with_scene(self, mobject: Any) -> None:
        if self.scene and hasattr(self.scene, "canvas_mobjects"):
            if mobject not in self.scene.canvas_mobjects:
                self.scene.canvas_mobjects.append(mobject)

    @staticmethod
    def get_bounding_box(mobject: Any) -> Dict[str, float]:
        """Extracts standard 2D axis-aligned bounding box from any Manim mobject."""
        try:
            w = float(mobject.width) if hasattr(mobject, "width") else float(mobject.get_width())
            h = float(mobject.height) if hasattr(mobject, "height") else float(mobject.get_height())
            return {
                "left": float(mobject.get_left()[0]),
                "right": float(mobject.get_right()[0]),
                "bottom": float(mobject.get_bottom()[1]),
                "top": float(mobject.get_top()[1]),
                "width": w,
                "height": h,
                "center": [float(mobject.get_center()[0]), float(mobject.get_center()[1]), 0.0],
            }
        except Exception:
            return {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0, "width": 0.0, "height": 0.0, "center": [0.0, 0.0, 0.0]}

    @classmethod
    def check_overlap(cls, m1: Any, m2: Any, threshold: float = 0.15) -> Tuple[bool, float]:
        """
        Computes 2D axis-aligned intersection between two mobjects.
        Returns (has_overlap, overlap_ratio_of_smaller_object).
        """
        b1 = cls.get_bounding_box(m1)
        b2 = cls.get_bounding_box(m2)

        x_overlap = max(0.0, min(b1["right"], b2["right"]) - max(b1["left"], b2["left"]))
        y_overlap = max(0.0, min(b1["top"], b2["top"]) - max(b1["bottom"], b2["bottom"]))
        overlap_area = x_overlap * y_overlap

        area1 = b1["width"] * b1["height"]
        area2 = b2["width"] * b2["height"]
        min_area = min(area1, area2)

        if min_area <= 0.001:
            return False, 0.0

        ratio = overlap_area / min_area
        return (ratio > threshold), float(ratio)
