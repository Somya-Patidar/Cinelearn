"""
DP Table Matrix Visual Component
================================
Configurable, reusable visual component for dynamic programming tables,
2D recurrence matrices, lookup grids, and memoization caches.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from manim import (
    VGroup,
    Rectangle,
    Text,
    Indicate,
    Transform,
    Animation,
    AnimationGroup,
    UP,
    LEFT,
)

from .. import VisualComponent, VisualRegistry


@VisualRegistry.register("cs", "DPTableMatrix")
class DPTableMatrix(VisualComponent):
    """
    Configurable visual component for dynamic programming matrices and grids.
    Supports row/column headers, cell updates, dependency highlighting,
    and row/column batch operations.
    """
    component_id = "DPTableMatrix"
    domain = "cs"
    aliases = ["dp table", "dp_table", "table matrix", "matrix table", "table"]
    preferred_aspect = 1.6
    suggested_viewport = "dock_inspector"
    supported_handles = ["cell_<r>_<c>", "row_header_<r>", "col_header_<c>"]
    action_weights = {
        "animate_update_cell": 0.6,
        "animate_compute_dependency": 0.6,
        "animate_highlight_cell": 0.6,
    }

    def __init__(
        self,
        matrix_values: Sequence[Sequence[Any]],
        row_labels: Optional[Sequence[Any]] = None,
        col_labels: Optional[Sequence[Any]] = None,
        cell_width: float = 0.75,
        cell_height: float = 0.60,
        cell_color: str = "#58C4DD",
        fill_color: str = "#111111",
        fill_opacity: float = 0.35,
        stroke_width: float = 2.0,
        text_color: str = "#FFFFFF",
        header_color: str = "#FFA07A",
        font_size: int = 18,
        table_title: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.matrix_values = [list(row) for row in matrix_values]
        self.num_rows = len(self.matrix_values)
        self.num_cols = len(self.matrix_values[0]) if self.num_rows > 0 else 0
        self.row_labels = list(row_labels) if row_labels is not None else None
        self.col_labels = list(col_labels) if col_labels is not None else None
        self.cell_width = float(cell_width)
        self.cell_height = float(cell_height)
        self.cell_color = cell_color
        self.fill_color = fill_color
        self.fill_opacity = fill_opacity
        self.stroke_width = stroke_width
        self.text_color = text_color
        self.header_color = header_color
        self.font_size = font_size
        self.table_title = table_title

        self.grid_group = VGroup()
        self.headers_group = VGroup()

        self._build_grid()
        self._build_headers()

        self.add(self.grid_group, self.headers_group)
        self.center()

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.num_rows, self.num_cols)

    def _build_grid(self) -> None:
        """Constructs cells and registers semantic handles for cells, boxes, and values."""
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                center_x = c * self.cell_width
                center_y = -r * self.cell_height

                box = Rectangle(
                    width=self.cell_width,
                    height=self.cell_height,
                    color=self.cell_color,
                    fill_color=self.fill_color,
                    fill_opacity=self.fill_opacity,
                    stroke_width=self.stroke_width,
                )
                box.move_to(np.array([center_x, center_y, 0.0]))

                val_str = str(self.matrix_values[r][c])
                val_txt = Text(val_str, font_size=self.font_size, color=self.text_color)
                if val_txt.width > self.cell_width * 0.85:
                    val_txt.scale_to_fit_width(self.cell_width * 0.85)
                if val_txt.height > self.cell_height * 0.80:
                    val_txt.scale_to_fit_height(self.cell_height * 0.80)
                val_txt.move_to(box.get_center())

                cell_group = VGroup(box, val_txt)
                self.grid_group.add(cell_group)

                self.register_semantic_handle(f"cell_{r}_{c}", cell_group)
                self.register_semantic_handle(f"box_{r}_{c}", box)
                self.register_semantic_handle(f"val_{r}_{c}", val_txt)

        # Composite row and column handles
        for r in range(self.num_rows):
            row_cells = [self.get_semantic_handle(f"cell_{r}_{c}") for c in range(self.num_cols)]
            self.register_semantic_handle(f"row_{r}", VGroup(*[c for c in row_cells if c is not None]))

        for c in range(self.num_cols):
            col_cells = [self.get_semantic_handle(f"cell_{r}_{c}") for r in range(self.num_rows)]
            self.register_semantic_handle(f"col_{c}", VGroup(*[cell for cell in col_cells if cell is not None]))

    def _build_headers(self) -> None:
        """Constructs column and row headers and table title."""
        # Column headers (above row 0)
        if self.col_labels and self.num_cols > 0:
            for c in range(min(len(self.col_labels), self.num_cols)):
                center_x = c * self.cell_width
                center_y = 0.5 * self.cell_height + 0.22

                h_txt = Text(
                    str(self.col_labels[c]),
                    font_size=max(12, int(self.font_size * 0.85)),
                    color=self.header_color,
                )
                if h_txt.width > self.cell_width * 0.90:
                    h_txt.scale_to_fit_width(self.cell_width * 0.90)
                h_txt.move_to(np.array([center_x, center_y, 0.0]))

                self.headers_group.add(h_txt)
                self.register_semantic_handle(f"col_header_{c}", h_txt)

        # Row headers (left of col 0)
        if self.row_labels and self.num_rows > 0:
            for r in range(min(len(self.row_labels), self.num_rows)):
                center_x = -0.5 * self.cell_width - 0.28
                center_y = -r * self.cell_height

                r_txt = Text(
                    str(self.row_labels[r]),
                    font_size=max(12, int(self.font_size * 0.85)),
                    color=self.header_color,
                )
                if r_txt.width > 1.2:
                    r_txt.scale_to_fit_width(1.2)
                r_txt.move_to(np.array([center_x, center_y, 0.0]))

                self.headers_group.add(r_txt)
                self.register_semantic_handle(f"row_header_{r}", r_txt)

        # Table title
        if self.table_title:
            title_txt = Text(
                self.table_title,
                font_size=max(14, int(self.font_size * 0.95)),
                color=self.header_color,
            )
            title_txt.next_to(self.grid_group, UP, buff=0.45 if self.col_labels else 0.25)
            self.headers_group.add(title_txt)
            self.register_semantic_handle("table_title", title_txt)

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------
    def animate_highlight_cell(
        self,
        row: int,
        col: int,
        color: str = "#FFD700",
        scale: float = 1.15,
        run_time: float = 0.6,
    ) -> Animation:
        """Highlights a specific table cell."""
        cell = self.get_semantic_handle(f"cell_{row}_{col}")
        if cell is None:
            return Indicate(self, color=color, run_time=run_time)
        return Indicate(cell, color=color, scale_factor=scale, run_time=run_time)

    def animate_update_cell(
        self,
        row: int,
        col: int,
        new_val: Any,
        highlight_color: str = "#FFD700",
        run_time: float = 0.6,
    ) -> AnimationGroup:
        """Updates a cell's value and indicates the cell box."""
        old_val_mob = self.get_semantic_handle(f"val_{row}_{col}")
        box = self.get_semantic_handle(f"box_{row}_{col}")
        if old_val_mob is None or box is None:
            return AnimationGroup(Indicate(self, run_time=run_time))

        self.matrix_values[row][col] = new_val
        new_val_str = str(new_val)
        new_val_mob = Text(new_val_str, font_size=self.font_size, color=highlight_color)
        if new_val_mob.width > self.cell_width * 0.85:
            new_val_mob.scale_to_fit_width(self.cell_width * 0.85)
        if new_val_mob.height > self.cell_height * 0.80:
            new_val_mob.scale_to_fit_height(self.cell_height * 0.80)
        new_val_mob.move_to(box.get_center())

        self._semantic_handles[f"val_{row}_{col}"] = new_val_mob

        return AnimationGroup(
            Indicate(box, color=highlight_color, run_time=run_time),
            Transform(old_val_mob, new_val_mob, run_time=run_time),
        )

    def animate_highlight_row(
        self,
        row: int,
        color: str = "#FFA07A",
        run_time: float = 0.6,
    ) -> Animation:
        """Highlights all cells in row `row`."""
        row_grp = self.get_semantic_handle(f"row_{row}")
        if row_grp is None:
            return Indicate(self, color=color, run_time=run_time)
        return Indicate(row_grp, color=color, run_time=run_time)

    def animate_highlight_col(
        self,
        col: int,
        color: str = "#FFA07A",
        run_time: float = 0.6,
    ) -> Animation:
        """Highlights all cells in column `col`."""
        col_grp = self.get_semantic_handle(f"col_{col}")
        if col_grp is None:
            return Indicate(self, color=color, run_time=run_time)
        return Indicate(col_grp, color=color, run_time=run_time)

    def animate_compute_dependency(
        self,
        target_rc: Tuple[int, int],
        dependency_rcs: Sequence[Tuple[int, int]],
        highlight_color: str = "#83C167",
        target_color: str = "#FFD700",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """
        Highlights prerequisite subproblem cells, then highlights the computed target cell.
        Classic recurrence visualization (e.g. dp[i][w] = max(dp[i-1][w], dp[i-1][w-wt] + val)).
        """
        dep_anims = []
        for r, c in dependency_rcs:
            cell = self.get_semantic_handle(f"cell_{r}_{c}")
            if cell is not None:
                dep_anims.append(Indicate(cell, color=highlight_color, run_time=run_time * 0.6))

        target_cell = self.get_semantic_handle(f"cell_{target_rc[0]}_{target_rc[1]}")
        target_anim = (
            Indicate(target_cell, color=target_color, run_time=run_time * 0.7)
            if target_cell is not None
            else Indicate(self, run_time=run_time * 0.7)
        )

        all_anims = dep_anims + [target_anim]
        return AnimationGroup(*all_anims, lag_ratio=0.35)

    def animate_fill_row(
        self,
        row: int,
        values: Sequence[Any],
        highlight_color: str = "#FFD700",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Batch-updates an entire row with new values."""
        anims = []
        for c, val in enumerate(values):
            if c < self.num_cols:
                anims.append(
                    self.animate_update_cell(
                        row,
                        c,
                        val,
                        highlight_color=highlight_color,
                        run_time=run_time,
                    )
                )
        return AnimationGroup(*anims, lag_ratio=0.1)
