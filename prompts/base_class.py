import os, sys
from pathlib import Path

# Ensure TeX Live 2026 binaries under $CONDA_PREFIX/texlive are discovered by Manim
_CONDA_PREFIX = os.environ.get("CONDA_PREFIX") or str(Path(sys.executable).parent.parent)
_TEXLIVE_BIN = Path(_CONDA_PREFIX) / "texlive" / "bin" / "x86_64-linux"
if _TEXLIVE_BIN.is_dir() and str(_TEXLIVE_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_TEXLIVE_BIN}:{os.environ.get('PATH', '')}"

base_class = r"""
import os, sys
from pathlib import Path

# Ensure TeX Live 2026 binaries under $CONDA_PREFIX/texlive are discovered by Manim
_CONDA_PREFIX = os.environ.get("CONDA_PREFIX") or str(Path(sys.executable).parent.parent)
_TEXLIVE_BIN = Path(_CONDA_PREFIX) / "texlive" / "bin" / "x86_64-linux"
if _TEXLIVE_BIN.is_dir() and str(_TEXLIVE_BIN) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_TEXLIVE_BIN}:{os.environ.get('PATH', '')}"

from manim import *
import math
import numpy as np
import re
from typing import Any, Optional, Tuple, Dict, List, Callable

# Ensure project root is in sys.path for shared domain layout and visual component imports
_candidates = [
    Path(__file__).resolve() if ("__file__" in locals() and __file__) else Path.cwd(),
    Path.cwd(),
    Path("/home/shreya/Somya/Cinelearn"),
]
for _base in _candidates:
    for _parent in [_base] + list(_base.parents):
        if (_parent / "src" / "domains").is_dir():
            if str(_parent) not in sys.path:
                sys.path.insert(0, str(_parent))
            break

from src.domains.layout import RectangularViewport, GenericContainerLayout
from src.domains import VisualComponent, ProceduralComponent, VisualRegistry, is_safe_svg

# Trigger automatic recursive discovery of all domain components
VisualRegistry.discover_domains("src.domains")

# Directly export all registered domain components into the scene execution namespace
from src.domains.cs import (
    GraphVisual,
    DPTableMatrix,
    ProcessCardVisual,
    ReadyQueueVisual,
    CPUExecutionVisual,
    GanttChartVisual,
    MetricTableVisual,
    OSTraceVisualAdapter,
)
from src.domains.chemistry import MoleculeVisual, ReactionEnergyDiagram
from src.domains.math import FunctionCurve, NumberLineVisual

LIGHT_SALMON = '#FFA07A'
LIGHT_BLUE = '#58C4DD'
GREEN_SCREEN = '#83C167'
GOLD_ACCENT = '#FFD700'
LIGHT_GREEN = '#83C167'
LIGHT_RED = '#FF6B6B'

def _split_clauses_safe(text):
    placeholder = "___COMMA___"
    protected = re.sub(r"\[([^\]]*)\]", lambda m: "[" + m.group(1).replace(",", placeholder) + "]", text)
    protected = re.sub(r"\(([^\)]*)\)", lambda m: "(" + m.group(1).replace(",", placeholder) + ")", protected)
    protected = re.sub(r"(\d+),(\d+)", r"\1___NUMCOMMA___\2", protected)
    return [
        p.replace(placeholder, ",").replace("___NUMCOMMA___", ",").strip()
        for p in re.split(r"[,;—]", protected)
        if p.strip()
    ]

def _extract_concise_caption(text, max_words=5, max_chars=32):
    clean = re.sub(r"^[0-9]+[\.\)\-:\s]+", "", text.strip())
    clean = re.sub(r"^[-*•]\s*", "", clean)
    fillers = [
        r"^(?:let us|let\'s|we can|we will|we\'ll|now|here|first|second|third|finally|next|then)\s+",
        r"^(?:in this section,? we will|as we can see|as shown|to begin with|in other words|importantly)\s+",
        r"^(?:specifically|recall that|remember that|notice that|observe that|it turns out that)\s+",
        r"^(?:it is easy to see that|we can observe that|note that|we have|consider that)\s+",
        r"^(?:to understand this,?|to see this,?|notice how|see how)\s+",
    ]
    for pat in fillers:
        clean = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()
    if ":" in clean:
        parts = clean.split(":", 1)
        prefix = parts[0].strip()
        rest = parts[1].strip()
        if len(prefix.split()) <= 3 and len(prefix) > 2:
            rest_words = rest.split()
            suffix = " ".join(rest_words[:2]) if len(rest_words) > 2 else rest
            candidate = f"{prefix}: {suffix}".strip(" .,;:")
            if len(candidate.split()) <= max_words and len(candidate) <= max_chars:
                return candidate
    parts = _split_clauses_safe(clean)
    if len(parts) > 1 and parts[0].lower().startswith(("to ", "when ", "if ", "while ")):
        cand = parts[1].split()
        if 2 <= len(cand) <= max_words:
            clean = parts[1]
    elif len(parts) > 1 and 2 <= len(parts[0].split()) <= max_words:
        clean = parts[0]
    words = clean.split()
    if len(words) > max_words:
        words = words[:max_words]
    stop_words = {"in", "of", "to", "the", "a", "an", "that", "and", "or", "for", "with", "by", "from", "at", "on", "into", "is", "are", "was", "were"}
    while words and words[-1].lower().rstrip(".,;:!?") in stop_words:
        words.pop()
    res = " ".join(words).strip(" .,;:!?")
    if len(res) > max_chars:
        res = res[:max_chars].rsplit(" ", 1)[0].strip(" .,;:!?")
    return res or text[:max_chars].strip()

class CaptionText(Text):
    '''Caption item representing a concise key phrase, starting subtle and activating on demand.'''
    def __init__(self, text, raw_text="", **kwargs):
        super().__init__(text, **kwargs)
        self.raw_text = raw_text
        self.set_opacity(0.35)
    def set_color(self, color):
        super().set_color(color)
        self.set_opacity(1.0)
        return self

import os
from pathlib import Path

# Add corner aliases to Mobject for robust LLM compatibility
if not hasattr(Mobject, "get_bottom_right"):
    Mobject.get_bottom_right = lambda self: self.get_corner(DOWN + RIGHT)
if not hasattr(Mobject, "get_bottom_left"):
    Mobject.get_bottom_left = lambda self: self.get_corner(DOWN + LEFT)
if not hasattr(Mobject, "get_top_right"):
    Mobject.get_top_right = lambda self: self.get_corner(UP + RIGHT)
if not hasattr(Mobject, "get_top_left"):
    Mobject.get_top_left = lambda self: self.get_corner(UP + LEFT)

_OriginalImageMobject = ImageMobject

class SafeImageMobject(Group):
    '''Safe drop-in replacement for ImageMobject when offline or assets disabled.'''
    def __init__(self, filename_or_array, **kwargs):
        try:
            if isinstance(filename_or_array, (str, Path)) and not os.path.exists(str(filename_or_array)):
                label = Path(str(filename_or_array)).stem.replace("_", " ").title()
                box = RoundedRectangle(corner_radius=0.1, width=1.6, height=1.0, color=LIGHT_BLUE, fill_opacity=0.3)
                txt = Text(label, font_size=16, color=WHITE).scale_to_fit_width(1.4)
                super().__init__(box, txt, **kwargs)
                return
            super().__init__(_OriginalImageMobject(filename_or_array, **kwargs))
        except Exception:
            label = str(filename_or_array)
            box = RoundedRectangle(corner_radius=0.1, width=1.6, height=1.0, color=LIGHT_BLUE, fill_opacity=0.3)
            txt = Text(str(label)[:12], font_size=16, color=WHITE).scale_to_fit_width(1.4)
            super().__init__(box, txt, **kwargs)

ImageMobject = SafeImageMobject

_OriginalSVGMobject = SVGMobject

class SafeSVGMobject(VGroup):
    '''Safe drop-in replacement for SVGMobject that validates/sanitizes SVG or falls back to native vector placeholder.'''
    def __init__(self, file_name=None, **kwargs):
        super().__init__()
        actual_path = None
        if file_name is not None and os.path.exists(str(file_name)):
            actual_path = str(file_name)
        if actual_path and is_safe_svg(actual_path):
            try:
                loaded = _OriginalSVGMobject(actual_path, **kwargs)
                self.add(*loaded.submobjects)
                return
            except Exception:
                pass

        label = Path(str(file_name or "icon")).stem.replace("_", " ").title()
        box = RoundedRectangle(corner_radius=0.15, width=2.0, height=1.4, color=LIGHT_BLUE, fill_opacity=0.25, stroke_width=2)
        txt = Text(label[:16], font_size=18, color=WHITE).scale_to_fit_width(1.8)
        sub1 = Dot(color=GREEN_SCREEN, radius=0.08).next_to(txt, DOWN, buff=0.1).shift(LEFT * 0.4)
        sub2 = Dot(color=RED, radius=0.08).next_to(txt, DOWN, buff=0.1)
        sub3 = Dot(color=GOLD_ACCENT, radius=0.08).next_to(txt, DOWN, buff=0.1).shift(RIGHT * 0.4)
        self.add(box, txt, sub1, sub2, sub3)

    def __getitem__(self, value):
        if isinstance(value, int) and (value >= len(self.submobjects) or value < -len(self.submobjects)):
            dummy = Dot(radius=0.01, fill_opacity=0).move_to(self.get_center())
            self.add(dummy)
            return dummy
        return super().__getitem__(value)

SVGMobject = SafeSVGMobject

_OriginalLine = Line

class SafeLine(VGroup):
    '''Safe drop-in replacement for Line that handles dash_length/dashed_ratio gracefully.'''
    def __new__(cls, *args, **kwargs):
        if "dash_length" in kwargs or "dashed_ratio" in kwargs:
            return DashedLine(*args, **kwargs)
        return _OriginalLine(*args, **kwargs)

Line = SafeLine

_OriginalTable = Table

class SafeTable(VGroup):
    '''Safe drop-in replacement for Table that handles row_labels mismatch gracefully.'''
    def __new__(cls, table, **kwargs):
        row_labels = kwargs.get("row_labels")
        if row_labels and len(table) != len(row_labels):
            kwargs.pop("row_labels", None)
        try:
            return _OriginalTable(table, **kwargs)
        except Exception:
            kwargs.pop("row_labels", None)
            kwargs.pop("col_labels", None)
            try:
                return _OriginalTable(table, **kwargs)
            except Exception:
                return VGroup()

Table = SafeTable

_OriginalMathTex = MathTex

class SafeMathTex(VGroup):
    '''Safe drop-in replacement for MathTex that protects against double-brace splitting,
    normalizes accent quotes, and falls back cleanly to Text if LaTeX compilation fails.'''
    def __init__(self, *tex_strings, **kwargs):
        super().__init__()
        cleaned_strings = []
        for s in tex_strings:
            if isinstance(s, str):
                cleaned = s
                prev = None
                while prev != cleaned:
                    prev = cleaned
                    cleaned = re.sub(r'\{\{([^{}]*)\}\}', r'{\1}', cleaned)
                cleaned = cleaned.replace(r"\'", "'")
                cleaned_strings.append(cleaned)
            else:
                cleaned_strings.append(s)
        try:
            loaded = _OriginalMathTex(*cleaned_strings, **kwargs)
            self.add(*loaded.submobjects)
        except Exception:
            raw_text = " ".join(str(s) for s in cleaned_strings)
            fallback = Text(raw_text, font_size=kwargs.get("font_size", 24), color=kwargs.get("color", kwargs.get("fill_color", WHITE)))
            self.add(fallback)

MathTex = SafeMathTex

_OriginalGraph = Graph

class SafeGraphWrapper:
    '''Safe wrapper for Graph that routes weighted 3-tuple edges to GraphVisual.'''
    def __new__(cls, *args, **kwargs):
        edges = kwargs.get("edges", None)
        if edges is None and len(args) > 1:
            edges = args[1]
        if edges and any(isinstance(e, (list, tuple)) and len(e) >= 3 for e in edges):
            return GraphVisual(*args, **kwargs)
        try:
            return _OriginalGraph(*args, **kwargs)
        except Exception:
            return GraphVisual(*args, **kwargs)

Graph = SafeGraphWrapper

class DockingHandler:
    '''Callable docking helper supporting both method call syntax and .get_center()/.center property access.'''
    def __init__(self, scene, viewport_name):
        self._scene = scene
        self._viewport_name = viewport_name

    def __call__(self, mobject, auto_scale=True, padding=0.04, **kwargs):
        method = getattr(self._scene.layout, f"dock_{self._viewport_name}")
        return method(mobject, auto_scale=auto_scale, padding=padding, **kwargs)

    def get_center(self):
        return self._scene.layout.viewports[self._viewport_name].center

    @property
    def center(self):
        return self._scene.layout.viewports[self._viewport_name].center

class GridPosition(np.ndarray):
    # NumPy array wrapper that supports .get_center() returning itself for Manim positioning compat.
    def get_center(self):
        return self

class SafeLectureGroup(VGroup):
    '''Safe wrapper for lecture caption list preventing off-by-one index crashes.'''
    def __getitem__(self, idx):
        if isinstance(idx, int) and idx >= len(self.submobjects):
            return self.submobjects[-1] if self.submobjects else Dot(radius=0.01)
        return super().__getitem__(idx)

class TeachingScene(Scene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = GenericContainerLayout(self)
        self.dock_main = DockingHandler(self, "main")
        self.dock_inspector = DockingHandler(self, "inspector")
        self.dock_split_left = DockingHandler(self, "split_left")
        self.dock_split_right = DockingHandler(self, "split_right")
        self.dock_full = DockingHandler(self, "full")

    def setup_layout(self, title_text, lecture_lines, captions=None):
        # BASE BACKGROUND
        self.camera.background_color = "#000000"
        self.canvas_mobjects = []

        # 1. SMART TITLE WRAPPING & SCALING
        raw_title = str(title_text).strip()
        if len(raw_title) > 34:
            if ":" in raw_title:
                prefix, suffix = raw_title.split(":", 1)
                wrapped_title = f"{prefix.strip()}:\n{suffix.strip()}"
            else:
                words = raw_title.split()
                mid = len(words) // 2
                wrapped_title = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        else:
            wrapped_title = raw_title

        self.title = Text(wrapped_title, font_size=24, color=WHITE, line_spacing=0.85)
        if self.title.width > 10.5:
            self.title.scale_to_fit_width(10.5)
        if self.title.height > 0.95:
            self.title.scale_to_fit_height(0.95)
        self.title.to_edge(UP, buff=0.22)
        self.title.set_z_index(100)
        self.add(self.title)

        # 2. CONCISE CAPTION LIST (Left side)
        self.raw_lecture_lines = lecture_lines
        if captions and len(captions) == len(lecture_lines):
            self.concise_captions = captions
        else:
            self.concise_captions = [_extract_concise_caption(line) for line in lecture_lines]

        caption_texts = []
        for cap, line in zip(self.concise_captions, lecture_lines):
            ct = CaptionText(f"• {cap}", raw_text=line, font_size=18, color=WHITE)
            if ct.width > 3.8:
                ct.scale_to_fit_width(3.8)
            caption_texts.append(ct)

        self.lecture = SafeLectureGroup(*caption_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        if self.lecture.height > 4.2:
            self.lecture.scale_to_fit_height(4.2)
        self.lecture.to_edge(LEFT, buff=0.3)
        if self.lecture.get_top()[1] > 2.6:
            self.lecture.shift(DOWN * (self.lecture.get_top()[1] - 2.6))
        self.lecture.set_z_index(90)
        self.add(self.lecture)

        # 3. FINE-GRAINED 6x6 ANIMATION GRID (Right side only)
        # Positioned safely between x in [0.0, 5.25] and y in [-2.55, 1.7]
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.0 + j * 1.05
                y = 1.7 - i * 0.85
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

        self.animation_area = Rectangle(width=6.6, height=5.0, stroke_opacity=0, fill_opacity=0).move_to(np.array([2.6, -0.4, 0]))
        self.layout = GenericContainerLayout(self)
        self.dock_main = DockingHandler(self, "main")
        self.dock_inspector = DockingHandler(self, "inspector")
        self.dock_split_left = DockingHandler(self, "split_left")
        self.dock_split_right = DockingHandler(self, "split_right")
        self.dock_full = DockingHandler(self, "full")

    def dock_main(self, mobject, auto_scale=True, padding=0.04, **kwargs):
        return self.layout.dock_main(mobject, auto_scale=auto_scale, padding=padding, **kwargs)

    def dock_inspector(self, mobject, auto_scale=True, padding=0.04, **kwargs):
        return self.layout.dock_inspector(mobject, auto_scale=auto_scale, padding=padding, **kwargs)

    def dock_split_left(self, mobject, auto_scale=True, padding=0.04, **kwargs):
        return self.layout.dock_split_left(mobject, auto_scale=auto_scale, padding=padding, **kwargs)

    def dock_split_right(self, mobject, auto_scale=True, padding=0.04, **kwargs):
        return self.layout.dock_split_right(mobject, auto_scale=auto_scale, padding=padding, **kwargs)

    def dock_full(self, mobject, auto_scale=True, padding=0.04, **kwargs):
        return self.layout.dock_full(mobject, auto_scale=auto_scale, padding=padding, **kwargs)

    def _clamp_to_safe_area(self, mobject, max_width=8.4, max_height=5.2):
        # Enforce size limits and safe boundary margins:
        # Title buffer: y <= 2.65, Subtitle buffer: y >= -2.65, Caption column: x >= -2.15, Right edge: x <= 6.65
        try:
            # 1. Size constraint: scale down if wider or taller than safe canvas
            if hasattr(mobject, "width") and mobject.width > max_width:
                mobject.scale_to_fit_width(max_width)
            if hasattr(mobject, "height") and mobject.height > max_height:
                mobject.scale_to_fit_height(max_height)

            # 2. Boundary clamping (strictly inside safe buffers)
            top_y = mobject.get_top()[1]
            if top_y > 2.55:
                mobject.shift(DOWN * (top_y - 2.55))
            bottom_y = mobject.get_bottom()[1]
            if bottom_y < -2.65:
                mobject.shift(UP * (-2.65 - bottom_y))
            left_x = mobject.get_left()[0]
            if left_x < -2.15:
                mobject.shift(RIGHT * (-2.15 - left_x))
            right_x = mobject.get_right()[0]
            if right_x > 6.65:
                mobject.shift(LEFT * (right_x - 6.65))
        except Exception:
            pass
        return mobject

    def _parse_grid_pos(self, key_str, default="C3"):
        # Safely parses a grid coordinate like 'B2' or 'C7', clamping to valid 6x6 limits.
        s = str(key_str).strip().upper()
        if not s:
            return self.grid.get(default, np.array([2.6, 0.0, 0]))
        if s in self.grid:
            return self.grid[s]
        # Attempt to clamp invalid row or col e.g. C7 -> C6, G3 -> F3, or C11 -> row C col 1, C12 -> row C col 2
        m = re.match(r"^([A-Z])(\d+)$", s)
        if m:
            row = m.group(1)
            digits = m.group(2)
            if len(digits) == 2:
                r_offset = int(digits[0])
                c_offset = int(digits[1])
                row = chr(ord('A') + min(5, max(0, r_offset - 1)))
                col = min(6, max(1, c_offset))
            else:
                col = int(digits)
                if row not in ["A", "B", "C", "D", "E", "F"]:
                    row = "F" if row > "F" else "A"
                if col < 1:
                    col = 1
                elif col > 6:
                    col = 6
            clamped = f"{row}{col}"
            if clamped in self.grid:
                return self.grid[clamped]
        return self.grid.get(default, np.array([2.6, 0.0, 0]))

    def create_table_matrix(self, matrix_values, cell_size=0.8, color="#58C4DD", text_color="#FFFFFF", font_size=20, start_cell="C3"):
        # Creates a 2D matrix/grid of cells with values properly positioned relative to each other
        rows = len(matrix_values)
        cols = len(matrix_values[0]) if rows > 0 else 0
        row_groups = []
        for r in range(rows):
            rg = VGroup()
            for c in range(cols):
                box = Square(side_length=cell_size, color=color, fill_opacity=0.2)
                val = str(matrix_values[r][c])
                txt = Text(val, font_size=font_size, color=text_color).move_to(box.get_center())
                cell = VGroup(box, txt)
                cell.move_to(np.array([c * cell_size * 1.1, -r * cell_size * 1.1, 0]))
                rg.add(cell)
            row_groups.append(rg)

        class TableMatrixGroup(VGroup):
            def __init__(self, r_groups, num_r=0, num_c=0, **kwargs):
                super().__init__(*r_groups, **kwargs)
                self.num_rows = num_r
                self.num_cols = num_c

            def __getitem__(self, idx):
                if isinstance(idx, int):
                    if idx < len(self.submobjects):
                        return self.submobjects[idx]
                    r = idx // max(1, self.num_cols)
                    c = idx % max(1, self.num_cols)
                    if r < len(self.submobjects) and c < len(self.submobjects[r].submobjects):
                        return self.submobjects[r].submobjects[c]
                    return self.submobjects[-1] if self.submobjects else Dot(radius=0.01)
                return super().__getitem__(idx)

        grid_group = TableMatrixGroup(row_groups, num_r=rows, num_c=cols)
        grid_group.move_to(self.place_at_grid(start_cell))
        self._clamp_to_safe_area(grid_group)
        if str(start_cell).upper().startswith("F") and grid_group.get_bottom()[1] < -2.65:
            grid_group.shift(UP * (-2.65 - grid_group.get_bottom()[1]))
        return grid_group

    def place_at_grid(self, mobject, grid_pos=None, scale_factor=1.0):
        # Places mobject at a discrete grid cell with cell-aware width and height constraints.
        if grid_pos is None and isinstance(mobject, str):
            grid_pos = mobject
            mobject = None
        pos_key = str(grid_pos).strip().upper()
        if "-" in pos_key:
            parts = pos_key.split("-", 1)
            return self.place_in_area(mobject, parts[0], parts[1], scale_factor=scale_factor)
        
        target_pos = self._parse_grid_pos(pos_key, default="C3")
        if mobject is None:
            return target_pos.view(GridPosition)
        
        # Scale and auto-fit to cell bounds (1 cell = 1.05w x 0.85h)
        mobject.scale(scale_factor)
        if hasattr(mobject, "width") and mobject.width > 2.2:
            mobject.scale_to_fit_width(2.2)
        if hasattr(mobject, "height") and mobject.height > 1.2:
            mobject.scale_to_fit_height(1.2)
            
        mobject.move_to(target_pos)
        self._clamp_to_safe_area(mobject)
        if pos_key.startswith("F") and mobject.get_bottom()[1] < -2.65:
            mobject.shift(UP * (-2.65 - mobject.get_bottom()[1]))
        if mobject not in self.canvas_mobjects:
            self.canvas_mobjects.append(mobject)
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right=None, scale_factor=1.0):
        # Places mobject centered inside an area with area-aware scaling. Never collapses to center.
        # 1. Handle hyphenated area passed as top_left (e.g. 'B2-D4')
        if isinstance(top_left, str) and "-" in top_left:
            parts = top_left.split("-", 1)
            top_left = parts[0].strip()
            bottom_right = parts[1].strip()

        # 2. Handle single cell passed without bottom_right (e.g. place_in_area(obj, 'C3'))
        if bottom_right is None or (isinstance(bottom_right, (int, float)) and scale_factor == 1.0):
            if isinstance(bottom_right, (int, float)):
                scale_factor = bottom_right
            if isinstance(top_left, str) and top_left.lower() in ("center", "middle"):
                center = self.animation_area.get_center() if hasattr(self, "animation_area") else np.array([2.6, -0.4, 0])
                if mobject is None:
                    return center.view(GridPosition)
                mobject.scale(scale_factor)
                mobject.move_to(center)
                self._clamp_to_safe_area(mobject)
                if mobject not in self.canvas_mobjects:
                    self.canvas_mobjects.append(mobject)
                return mobject
            # Single cell provided: delegate safely to place_at_grid
            return self.place_at_grid(mobject, top_left, scale_factor=scale_factor)

        tl_pos = self._parse_grid_pos(top_left, default="B2")
        br_pos = self._parse_grid_pos(bottom_right, default="E5")

        center_x = (tl_pos[0] + br_pos[0]) / 2.0
        center_y = (tl_pos[1] + br_pos[1]) / 2.0
        center = np.array([center_x, center_y, 0])
        if mobject is None:
            return center.view(GridPosition)

        # Calculate bounding box of the area to prevent spillover
        area_w = max(1.2, abs(br_pos[0] - tl_pos[0]) + 1.05)
        area_h = max(0.9, abs(tl_pos[1] - br_pos[1]) + 0.85)

        mobject.scale(scale_factor)
        if hasattr(mobject, "width") and mobject.width > area_w * 0.95:
            mobject.scale_to_fit_width(area_w * 0.95)
        if hasattr(mobject, "height") and mobject.height > area_h * 0.95:
            mobject.scale_to_fit_height(area_h * 0.95)

        mobject.move_to(center)
        self._clamp_to_safe_area(mobject)
        if mobject not in self.canvas_mobjects:
            self.canvas_mobjects.append(mobject)
        return mobject

    def safe_label(self, label_or_text, target_mobj=None, direction=UP, buff=0.15, role="body", color=WHITE):
        # Constructs or positions an adaptive label with role-based font sizing and boundary protection.
        if isinstance(label_or_text, str):
            role_sizes = {
                "title": (24, 4.2),
                "header": (22, 3.2),
                "body": (18, 2.4),
                "label": (18, 2.4),
                "data": (16, 1.8),
                "value": (16, 1.8),
                "index": (14, 1.2),
                "badge": (13, 1.0),
            }
            font_size, max_w = role_sizes.get(str(role).lower(), (18, 2.4))
            lbl = Text(str(label_or_text), font_size=font_size, color=color)
            if lbl.width > max_w:
                lbl.scale_to_fit_width(max_w)
        else:
            lbl = label_or_text

        if target_mobj is not None:
            lbl.next_to(target_mobj, direction, buff=buff)
            if lbl.get_top()[1] > 2.65:
                lbl.next_to(target_mobj, DOWN, buff=buff)
            self._clamp_to_safe_area(lbl)

        if lbl not in self.canvas_mobjects:
            self.canvas_mobjects.append(lbl)
        return lbl

    def create_array(self, values, cell_size=0.8, indices=True, color=LIGHT_BLUE, text_color=WHITE):
        # Creates an aligned array VGroup with cell boxes, values, and optional indices.
        cells = VGroup()
        for i, val in enumerate(values):
            box = Square(side_length=cell_size, color=color, fill_opacity=0.2)
            val_text = Text(str(val), font_size=int(22 * cell_size / 0.8), color=text_color)
            if val_text.width > cell_size * 0.8:
                val_text.scale_to_fit_width(cell_size * 0.8)
            val_text.move_to(box.get_center())
            cell_group = VGroup(box, val_text)
            if indices:
                idx_text = Text(str(i), font_size=int(14 * cell_size / 0.8), color=GRAY).next_to(box, UP, buff=0.1)
                cell_group.add(idx_text)
            cells.add(cell_group)
        cells.arrange(RIGHT, buff=0.0)
        return cells

    def create_pointer(self, target_mob, label="ptr", direction=DOWN, color=GOLD_ACCENT):
        # Creates an aligned pointer arrow with label pointing to target_mob without overlap.
        arrow = Arrow(
            start=target_mob.get_boundary_point(direction) + direction * 0.7,
            end=target_mob.get_boundary_point(direction) + direction * 0.08,
            color=color,
            buff=0.05,
            stroke_width=3,
            max_tip_length_to_length_ratio=0.35,
        )
        ptr_label = Text(label, font_size=15, color=color).next_to(arrow, direction, buff=0.08)
        return VGroup(arrow, ptr_label)

    def safe_text(self, text, font_size=22, max_width=6.0, color=WHITE, line_spacing=0.85):
        '''Construct multi-line text with word wrapping to prevent microscopic scaling.'''
        raw_str = str(text).strip()
        clean_str = raw_str.replace("\\n", "\n")
        paragraphs = clean_str.split("\n")
        lines = []
        for p in paragraphs:
            words = p.split()
            cur_line = []
            cur_len = 0
            for w in words:
                if cur_len + len(w) > 28 and cur_line:
                    lines.append(" ".join(cur_line))
                    cur_line = [w]
                    cur_len = len(w)
                else:
                    cur_line.append(w)
                    cur_len += len(w) + 1
            if cur_line:
                lines.append(" ".join(cur_line))
        wrapped = "\n".join(lines)
        font_sz = max(16, font_size)
        txt = Text(wrapped, font_size=font_sz, color=color, line_spacing=line_spacing)
        if txt.width > max_width:
            txt.scale_to_fit_width(max_width)
        return txt

    def safe_math(self, tex_str, max_width=6.5, color=WHITE, font_size=24):
        '''Construct MathTex with automatic double-brace protection and width limit.'''
        clean_tex = str(tex_str)
        # Collapse accidental double curly braces {{ and }} that trigger Manim substring splitting
        import re
        prev = None
        while prev != clean_tex:
            prev = clean_tex
            clean_tex = re.sub(r'\{\{([^{}]*)\}\}', r'{\1}', clean_tex)
        try:
            m = MathTex(clean_tex, color=color, font_size=font_size)
        except Exception:
            m = Text(str(tex_str), font_size=font_size, color=color)
        if hasattr(m, "width") and m.width > max_width:
            m.scale_to_fit_width(max_width)
        return m

    def clear_canvas(self, *keep, run_time=0.6):
        # Fades out all registered canvas mobjects except those explicitly preserved in keep
        fade_outs = []
        remaining = []
        for m in list(self.canvas_mobjects):
            if m in self.mobjects and m not in keep:
                fade_outs.append(FadeOut(m, run_time=run_time))
            else:
                remaining.append(m)
        self.canvas_mobjects = remaining
        if fade_outs:
            self.play(*fade_outs)

    def transition_to(self, new_mobjects, persist=None, run_time=0.8):
        # Cleanly cross-dissolves from existing canvas mobjects to new visual elements
        persist = persist or []
        fade_outs = []
        remaining = []
        for m in list(self.canvas_mobjects):
            if m in self.mobjects and m not in persist:
                fade_outs.append(FadeOut(m, run_time=run_time))
            else:
                remaining.append(m)
        new_list = new_mobjects if isinstance(new_mobjects, (list, tuple)) else [new_mobjects]
        fade_ins = [FadeIn(m, run_time=run_time) for m in new_list]
        self.canvas_mobjects = remaining + list(new_list)
        if fade_outs and fade_ins:
            self.play(*fade_outs, *fade_ins)
        elif fade_ins:
            self.play(*fade_ins)
        elif fade_outs:
            self.play(*fade_outs)

    def activate_caption(self, index, color="#FFD700", run_time=0.6):
        # Highlights the active caption line while subtly dimming other lines
        # Record temporal synchronization marker for this narration line
        renderer = getattr(self, "renderer", None)
        t_now = float(getattr(renderer, "time", 0.0)) if renderer else 0.0
        if not hasattr(self, "_line_execution_timings"):
            self._line_execution_timings = {}
        if index not in self._line_execution_timings:
            self._line_execution_timings[index] = {"start_time": t_now}
        anims = []
        for i, cap in enumerate(self.lecture):
            if i == index:
                anims.append(cap.animate(run_time=run_time).set_opacity(1.0).set_color(color))
            else:
                anims.append(cap.animate(run_time=run_time).set_opacity(0.35).set_color(WHITE))
        return anims

    def mark_line_beat(self, index):
        # Explicit temporal anchor for a lecture line
        renderer = getattr(self, "renderer", None)
        t_now = float(getattr(renderer, "time", 0.0)) if renderer else 0.0
        if not hasattr(self, "_line_execution_timings"):
            self._line_execution_timings = {}
        if index not in self._line_execution_timings:
            self._line_execution_timings[index] = {"start_time": t_now}

    # Context-appropriate animations: fade, slide, zoom, highlight
    def animate_fade(self, mobject, run_time=1.0, **kwargs):
        return FadeIn(mobject, run_time=run_time, **kwargs)

    def animate_slide(self, mobject, direction=RIGHT, run_time=1.0, **kwargs):
        return FadeIn(mobject, shift=direction, run_time=run_time, **kwargs)

    def animate_zoom(self, mobject, run_time=1.0, **kwargs):
        return GrowFromCenter(mobject, run_time=run_time, **kwargs)

    def animate_highlight(self, mobject, color="#FFFF00", run_time=1.0, **kwargs):
        return Indicate(mobject, color=color, run_time=run_time, **kwargs)

    def _extract_active_mobjects(self):
        mobs = []
        for idx, m in enumerate(self.mobjects):
            try:
                tag = "canvas"
                if hasattr(self, "title") and (m is self.title or self.title in getattr(m, "submobjects", [])):
                    tag = "title"
                elif hasattr(self, "lecture") and (m is self.lecture or self.lecture in getattr(m, "submobjects", [])):
                    tag = "lecture"

                left = float(m.get_left()[0])
                right = float(m.get_right()[0])
                top = float(m.get_top()[1])
                bottom = float(m.get_bottom()[1])
                width = float(m.width) if hasattr(m, "width") else float(m.get_width())
                height = float(m.height) if hasattr(m, "height") else float(m.get_height())
                center = [float(c) for c in m.get_center()]
                cls_name = m.__class__.__name__

                if width < 0.01 and height < 0.01:
                    continue

                is_ph = bool(
                    getattr(m, "_is_placeholder", False)
                    or getattr(m, "is_placeholder", False)
                )

                mob_entry = {
                    "index": idx,
                    "class": cls_name,
                    "tag": tag,
                    "left": round(left, 3),
                    "right": round(right, 3),
                    "top": round(top, 3),
                    "bottom": round(bottom, 3),
                    "width": round(width, 3),
                    "height": round(height, 3),
                    "center": [round(c, 3) for c in center],
                    "is_placeholder": is_ph,
                }
                if is_ph:
                    mob_entry["placeholder_name"] = str(getattr(m, "procedural_name", "") or cls_name)

                # Record children for composite components and VGroups on canvas to detect internal stacking
                if tag == "canvas" and hasattr(m, "submobjects") and len(m.submobjects) > 1:
                    children_info = []
                    for c_idx, child in enumerate(m.submobjects[:50]):
                        try:
                            cw = float(child.width) if hasattr(child, "width") else float(child.get_width())
                            ch = float(child.height) if hasattr(child, "height") else float(child.get_height())
                            if cw < 0.01 and ch < 0.01:
                                continue
                            cc = [round(float(c), 3) for c in child.get_center()]
                            c_left = round(cc[0] - cw / 2.0, 3)
                            c_right = round(cc[0] + cw / 2.0, 3)
                            c_bottom = round(cc[1] - ch / 2.0, 3)
                            c_top = round(cc[1] + ch / 2.0, 3)
                            children_info.append({
                                "child_index": c_idx,
                                "class": child.__class__.__name__,
                                "center": cc,
                                "width": round(cw, 3),
                                "height": round(ch, 3),
                                "left": c_left,
                                "right": c_right,
                                "top": c_top,
                                "bottom": c_bottom,
                            })
                        except Exception:
                            continue
                    if children_info:
                        mob_entry["children"] = children_info

                mobs.append(mob_entry)
            except Exception:
                continue
        return mobs

    def checkpoint_layout(self, name=None):
        # Record an intermediate animation layout checkpoint (all checkpoints recorded)
        if not hasattr(self, "_layout_checkpoints"):
            self._layout_checkpoints = []
        chk_name = name or f"step_{len(self._layout_checkpoints) + 1}"
        mobs = self._extract_active_mobjects()
        if mobs:
            renderer = getattr(self, "renderer", None)
            current_time = float(getattr(renderer, "time", 0.0)) if renderer else float(getattr(self, "time", 0.0))
            self._layout_checkpoints.append({
                "checkpoint": chk_name,
                "time": round(current_time, 2),
                "mobjects": mobs
            })

    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        try:
            step_num = len(getattr(self, "_layout_checkpoints", [])) + 1
            self.checkpoint_layout(name=f"step_{step_num}")
        except Exception:
            pass

    def tear_down(self):
        super().tear_down()
        try:
            self._save_layout_inspection()
        except Exception:
            pass

    def _save_layout_inspection(self):
        import json
        import os
        import sys
        from pathlib import Path
        scene_name = self.__class__.__name__
        mobs = self._extract_active_mobjects()
        checkpoints = getattr(self, "_layout_checkpoints", [])

        # Determine target output directory
        target_dir = getattr(self, "output_dir", None)
        if target_dir is None:
            try:
                mod_file = sys.modules[self.__class__.__module__].__file__
                if mod_file:
                    target_dir = Path(mod_file).parent
            except Exception:
                pass
        if target_dir is None:
            env_dir = os.environ.get("CINELEARN_OUTPUT_DIR")
            if env_dir:
                target_dir = Path(env_dir)
        if target_dir is None:
            try:
                from manim import config
                if config.output_dir:
                    target_dir = Path(config.output_dir)
            except Exception:
                pass
        if target_dir is None:
            target_dir = Path.cwd()
        else:
            target_dir = Path(target_dir)

        # Compute ground-truth line execution timings
        renderer = getattr(self, "renderer", None)
        total_time = float(getattr(renderer, "time", 0.0)) if renderer else 0.0
        raw_timings = getattr(self, "_line_execution_timings", {})
        line_timings = []
        sorted_indices = sorted(raw_timings.keys())
        for k, idx in enumerate(sorted_indices):
            s_time = float(raw_timings[idx].get("start_time", 0.0))
            if k < len(sorted_indices) - 1:
                next_idx = sorted_indices[k + 1]
                e_time = float(raw_timings[next_idx].get("start_time", total_time))
            else:
                e_time = total_time
            dur = max(0.1, e_time - s_time)
            line_timings.append({
                "line_index": idx,
                "start_time": round(s_time, 3),
                "end_time": round(e_time, 3),
                "duration": round(dur, 3)
            })

        layout_data = {
            "scene": scene_name,
            "frame_width": 14.222,
            "frame_height": 8.0,
            "safe_bounds": {"left": -2.15, "right": 6.65, "top": 2.65, "bottom": -2.85},
            "total_duration": round(total_time, 3),
            "line_timings": line_timings,
            "mobjects": mobs,
            "checkpoints": checkpoints
        }
        filename = f"layout_{scene_name}.json"
        timing_filename = f"timing_{scene_name}.json"

        # Save to resolved case directory
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            with open(target_dir / filename, "w", encoding="utf-8") as f:
                json.dump(layout_data, f, indent=2)
            with open(target_dir / timing_filename, "w", encoding="utf-8") as f:
                json.dump({
                    "scene": scene_name,
                    "total_duration": round(total_time, 3),
                    "line_timings": line_timings
                }, f, indent=2)
        except Exception:
            pass

        # Also write to CWD if different to guarantee compatibility with callers expecting CWD output
        try:
            cwd_path = Path.cwd().resolve()
            if target_dir.resolve() != cwd_path:
                with open(cwd_path / filename, "w", encoding="utf-8") as f:
                    json.dump(layout_data, f, indent=2)
                with open(cwd_path / timing_filename, "w", encoding="utf-8") as f:
                    json.dump({
                        "scene": scene_name,
                        "total_duration": round(total_time, 3),
                        "line_timings": line_timings
                    }, f, indent=2)
        except Exception:
            pass
"""

# Execute base_class in module scope so standalone scripts or tests importing prompts.base_class
# have direct access to TeachingScene, Layout, and all 6 domain visual components.
exec(base_class, globals())

