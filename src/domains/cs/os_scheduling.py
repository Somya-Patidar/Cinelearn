"""
CineLearn Operating Systems CPU Scheduling Visual Component Suite
==================================================================
Reusable visual components and trace adapter for OS CPU scheduling demonstrations:
- ProcessCardVisual: Process Control Block (PCB) visual representation
- ReadyQueueVisual: FIFO queue container with enqueue/dequeue/peek semantics
- CPUExecutionVisual: CPU core execution unit with dispatch/execute/complete semantics
- GanttChartVisual: Proportional timeline Gantt chart with time cursor
- MetricTableVisual: Accounting matrix (CT, TAT, WT) without premature metric leakage
- OSTraceVisualAdapter: Deterministic bridge between ComputationTrace and visual components
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from manim import (
    Mobject,
    VMobject,
    VGroup,
    Rectangle,
    RoundedRectangle,
    Line,
    Arrow,
    Text,
    Indicate,
    Transform,
    FadeIn,
    FadeOut,
    Animation,
    AnimationGroup,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
)

from .. import (
    VisualComponent,
    VisualRegistry,
    DomainTraceGuidanceRegistry,
    DomainFidelityVerifier,
    DomainFidelityRegistry,
)

# Theme & Palette Constants
STATE_COLORS: Dict[str, str] = {
    "NEW": "#A0A0A0",
    "NOT_ARRIVED": "#707070",
    "READY": "#FFA07A",       # Amber / light salmon
    "RUNNING": "#83C167",     # Green / active
    "TERMINATED": "#58C4DD",  # Muted cyan/blue
    "IDLE": "#888888",
}

STATE_BG_COLORS: Dict[str, str] = {
    "NEW": "#222222",
    "NOT_ARRIVED": "#181818",
    "READY": "#2D1E12",
    "RUNNING": "#122A16",
    "TERMINATED": "#12202A",
    "IDLE": "#1A1A1A",
}

PROCESS_PALETTE: List[str] = [
    "#58C4DD",  # Light blue
    "#FFA07A",  # Salmon
    "#83C167",  # Green
    "#FFD700",  # Gold
    "#C9A0FF",  # Lavender
    "#FF6B6B",  # Coral red
]


# ============================================================================
# 1. ProcessCardVisual (Process Control Block)
# ============================================================================

@VisualRegistry.register("cs", "ProcessCardVisual")
class ProcessCardVisual(VisualComponent):
    """
    Visual representation of an OS Process Control Block (PCB).
    Displays PID, arrival time (AT), burst time (BT), remaining burst time (RT),
    and current lifecycle state with color-coded badges and border styling.
    """
    component_id = "ProcessCardVisual"
    domain = "cs"
    aliases = ["pcb", "process card", "process_card", "process_control_block", "process"]
    preferred_aspect = 1.6
    suggested_viewport = "dock_split_left"
    supported_handles = ["card", "state_badge", "pid_text", "at_text", "bt_text", "rt_text", "attributes"]
    action_weights = {
        "animate_set_state": 1.0,
        "animate_update_remaining_time": 0.8,
        "animate_highlight": 0.6,
    }

    def __init__(
        self,
        pid: str,
        arrival_time: int = 0,
        burst_time: int = 1,
        remaining_time: Optional[int] = None,
        state: str = "NEW",
        width: float = 1.5,
        height: float = 0.95,
        font_size: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pid = f"P{pid}" if isinstance(pid, int) else str(pid or "P1")
        self.arrival_time = int(arrival_time)
        self.burst_time = int(burst_time)
        self.remaining_time = int(remaining_time if remaining_time is not None else burst_time)
        self.current_state = str(state).upper()
        self.card_width = float(width)
        self.card_height = float(height)
        self.font_size = font_size

        self.card_body: Optional[RoundedRectangle] = None
        self.pid_label: Optional[Text] = None
        self.state_badge_box: Optional[RoundedRectangle] = None
        self.state_badge_text: Optional[Text] = None
        self.remaining_time_label: Optional[Text] = None
        self.meta_labels: Optional[VGroup] = None

        self._build_card()

    def _get_state_color(self, state: str) -> str:
        return STATE_COLORS.get(state.upper(), "#A0A0A0")

    def _get_state_bg(self, state: str) -> str:
        return STATE_BG_COLORS.get(state.upper(), "#222222")

    def _build_card(self) -> None:
        col = self._get_state_color(self.current_state)
        bg = self._get_state_bg(self.current_state)

        # Outer PCB container
        self.card_body = RoundedRectangle(
            corner_radius=0.08,
            width=self.card_width,
            height=self.card_height,
            stroke_color=col,
            stroke_width=2.0,
            fill_color=bg,
            fill_opacity=0.88,
        )
        self.register_semantic_handle("card_body", self.card_body)
        self.add(self.card_body)

        # PID header (upper left)
        self.pid_label = Text(
            self.pid,
            font_size=max(14, int(self.font_size * 1.15)),
            weight="BOLD",
            color="#FFFFFF",
        )
        if self.pid_label.width > self.card_width * 0.45:
            self.pid_label.scale_to_fit_width(self.card_width * 0.45)
        self.pid_label.move_to(
            self.card_body.get_center() + np.array([-self.card_width * 0.26, self.card_height * 0.25, 0.0])
        )
        self.register_semantic_handle("pid_label", self.pid_label)
        self.add(self.pid_label)

        # State badge (upper right)
        badge_w = self.card_width * 0.46
        badge_h = self.card_height * 0.28
        self.state_badge_box = RoundedRectangle(
            corner_radius=0.04,
            width=badge_w,
            height=badge_h,
            stroke_width=1.0,
            stroke_color=col,
            fill_color=col,
            fill_opacity=0.25,
        )
        self.state_badge_text = Text(
            self.current_state,
            font_size=max(12, int(self.font_size * 0.75)),
            color=col,
        )
        if self.state_badge_text.width > badge_w * 0.88:
            self.state_badge_text.scale_to_fit_width(badge_w * 0.88)
        self.state_badge_box.move_to(
            self.card_body.get_center() + np.array([self.card_width * 0.24, self.card_height * 0.25, 0.0])
        )
        self.state_badge_text.move_to(self.state_badge_box.get_center())

        state_badge = VGroup(self.state_badge_box, self.state_badge_text)
        self.register_semantic_handle("state_badge", state_badge)
        self.register_semantic_handle("state_badge_box", self.state_badge_box)
        self.register_semantic_handle("state_badge_text", self.state_badge_text)
        self.add(self.state_badge_box, self.state_badge_text)

        # Divider line
        div = Line(
            self.card_body.get_center() + np.array([-self.card_width * 0.44, 0.06, 0.0]),
            self.card_body.get_center() + np.array([self.card_width * 0.44, 0.06, 0.0]),
            stroke_width=0.7,
            stroke_color="#555555",
        )
        self.add(div)

        # Meta properties row (lower half)
        at_label = Text(f"AT:{self.arrival_time}", font_size=max(12, int(self.font_size * 0.75)), color="#CCCCCC")
        bt_label = Text(f"BT:{self.burst_time}", font_size=max(12, int(self.font_size * 0.75)), color="#CCCCCC")
        self.remaining_time_label = Text(
            f"Rem:{self.remaining_time}",
            font_size=max(12, int(self.font_size * 0.75)),
            color="#FFD700" if self.remaining_time > 0 else "#83C167",
        )

        row_y = self.card_body.get_center()[1] - self.card_height * 0.24
        at_label.move_to(np.array([self.card_body.get_center()[0] - self.card_width * 0.30, row_y, 0.0]))
        bt_label.move_to(np.array([self.card_body.get_center()[0], row_y, 0.0]))
        self.remaining_time_label.move_to(
            np.array([self.card_body.get_center()[0] + self.card_width * 0.28, row_y, 0.0])
        )

        self.meta_labels = VGroup(at_label, bt_label, self.remaining_time_label)
        self.register_semantic_handle("at_label", at_label)
        self.register_semantic_handle("bt_label", bt_label)
        self.register_semantic_handle("remaining_time_label", self.remaining_time_label)
        self.add(at_label, bt_label, self.remaining_time_label)

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------

    def animate_set_state(self, new_state: str, run_time: float = 0.5) -> AnimationGroup:
        """Transitions PCB state, updating border, badge colors, and text."""
        new_state = new_state.upper()
        self.current_state = new_state
        col = self._get_state_color(new_state)
        bg = self._get_state_bg(new_state)

        new_badge_box = RoundedRectangle(
            corner_radius=0.04,
            width=self.card_width * 0.46,
            height=self.card_height * 0.28,
            stroke_width=1.0,
            stroke_color=col,
            fill_color=col,
            fill_opacity=0.25,
        ).move_to(self.state_badge_box.get_center())

        new_badge_text = Text(
            new_state,
            font_size=max(12, int(self.font_size * 0.75)),
            color=col,
        )
        if new_badge_text.width > (self.card_width * 0.46) * 0.88:
            new_badge_text.scale_to_fit_width((self.card_width * 0.46) * 0.88)
        new_badge_text.move_to(new_badge_box.get_center())

        anim_box = Transform(self.state_badge_box, new_badge_box, run_time=run_time)
        anim_text = Transform(self.state_badge_text, new_badge_text, run_time=run_time)
        anim_border = self.card_body.animate(run_time=run_time).set_stroke(color=col).set_fill(color=bg)

        return AnimationGroup(anim_box, anim_text, anim_border)

    def animate_update_remaining_time(self, new_remaining: int, run_time: float = 0.4) -> AnimationGroup:
        """Updates remaining burst time with numerical animation."""
        self.remaining_time = int(new_remaining)
        col = "#FFD700" if self.remaining_time > 0 else "#83C167"
        new_text = Text(
            f"Rem:{self.remaining_time}",
            font_size=max(12, int(self.font_size * 0.75)),
            color=col,
        ).move_to(self.remaining_time_label.get_center())

        anim = Transform(self.remaining_time_label, new_text, run_time=run_time)
        return AnimationGroup(anim)

    def animate_highlight(self, color: str = "#FFD700", run_time: float = 0.5) -> Animation:
        """Emphasizes card during scheduling or dispatch."""
        return Indicate(self.card_body, color=color, scale_factor=1.08, run_time=run_time)

    def set_state_instant(self, new_state: str) -> None:
        """Instantaneous state update without animations."""
        new_state = new_state.upper()
        self.current_state = new_state
        col = self._get_state_color(new_state)
        bg = self._get_state_bg(new_state)
        self.card_body.set_stroke(color=col).set_fill(color=bg)
        self.state_badge_box.set_stroke(color=col).set_fill(color=col, opacity=0.25)
        new_text = Text(new_state, font_size=max(12, int(self.font_size * 0.75)), color=col)
        if new_text.width > (self.card_width * 0.46) * 0.88:
            new_text.scale_to_fit_width((self.card_width * 0.46) * 0.88)
        new_text.move_to(self.state_badge_box.get_center())
        self.state_badge_text.become(new_text)
        self.state_badge_text.text = new_state

    def set_remaining_time_instant(self, new_remaining: int) -> None:
        """Instantaneous remaining time update."""
        self.remaining_time = int(new_remaining)
        col = "#FFD700" if self.remaining_time > 0 else "#83C167"
        new_text = Text(
            f"Rem:{self.remaining_time}",
            font_size=max(12, int(self.font_size * 0.75)),
            color=col,
        ).move_to(self.remaining_time_label.get_center())
        self.remaining_time_label.become(new_text)
        self.remaining_time_label.text = f"Rem:{self.remaining_time}"

    # Aliases for flexible LLM invocation
    set_state = animate_set_state
    update_remaining_time = animate_update_remaining_time
    highlight = animate_highlight


# ============================================================================
# 2. ReadyQueueVisual (FIFO Ready Queue)
# ============================================================================

@VisualRegistry.register("cs", "ReadyQueueVisual")
class ReadyQueueVisual(VisualComponent):
    """
    Visual container representing an OS FIFO Ready Queue.
    Features labeled queue boundaries (HEAD and TAIL), slot dividers,
    and semantic enqueue, dequeue, and peek animations.
    """
    component_id = "ReadyQueueVisual"
    domain = "cs"
    aliases = ["ready queue", "ready_queue", "queue", "fifo queue", "fifo_queue"]
    preferred_aspect = 3.2
    suggested_viewport = "dock_split_left"
    supported_handles = ["rail_top", "rail_bottom", "head_label", "tail_label", "slot_0", "slot_1", "slot_2", "slot_3", "slot_4"]
    action_weights = {
        "animate_enqueue": 1.5,
        "animate_dequeue": 1.5,
        "animate_peek": 0.6,
    }

    def __init__(
        self,
        capacity: int = 5,
        slot_width: float = 1.1,
        slot_height: float = 0.85,
        title: str = "READY QUEUE (FIFO)",
        stroke_color: str = "#58C4DD",
        font_size: int = 16,
        initial_pids: Optional[Sequence[str]] = None,
        processes: Optional[Sequence[Union[Dict[str, Any], Any, str]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.capacity = max(1, int(capacity))
        self.slot_width = float(slot_width)
        self.slot_height = float(slot_height)
        self.title_text = title
        self.stroke_color = stroke_color
        self.font_size = font_size

        self.total_queue_width = self.capacity * self.slot_width
        self.queue_items: List[VisualComponent] = []
        self.top_rail: Optional[Line] = None
        self.bottom_rail: Optional[Line] = None

        self._build_queue()

        # Handle initial processes/pids if provided
        pids_to_add: List[str] = []
        if initial_pids:
            pids_to_add = [str(p) for p in initial_pids]
        elif processes:
            for p in processes:
                if isinstance(p, dict) and "pid" in p:
                    pids_to_add.append(str(p["pid"]))
                elif hasattr(p, "pid"):
                    pids_to_add.append(str(getattr(p, "pid")))
                else:
                    pids_to_add.append(str(p))
        if pids_to_add:
            self.set_queue_pids(pids_to_add)

    def get_slot_center(self, index: int) -> np.ndarray:
        """Returns slot center coordinates in world space (Slot 0 is HEAD on left)."""
        idx = max(0, min(int(index), self.capacity - 1))
        div_left = self.get_semantic_handle(f"slot_divider_{idx}")
        div_right = self.get_semantic_handle(f"slot_divider_{idx + 1}")
        if div_left is not None and div_right is not None:
            cx = (div_left.get_center()[0] + div_right.get_center()[0]) / 2.0
            cy = (div_left.get_center()[1] + div_right.get_center()[1]) / 2.0
            return np.array([cx, cy, 0.0])
        left_edge = -self.total_queue_width / 2.0
        x = left_edge + (idx + 0.5) * self.slot_width
        return self.get_center() + np.array([x, 0.0, 0.0])

    def _get_slot_center(self, index: int) -> np.ndarray:
        """Local slot center offset relative to queue center."""
        return self.get_slot_center(index) - self.get_center()

    def _build_queue(self) -> None:
        container_grp = VGroup()

        # Top and bottom rail tracks
        w = self.total_queue_width
        h = self.slot_height
        self.top_rail = Line(
            np.array([-w / 2.0, h / 2.0, 0.0]),
            np.array([w / 2.0, h / 2.0, 0.0]),
            stroke_width=2.5,
            stroke_color=self.stroke_color,
        )
        self.bottom_rail = Line(
            np.array([-w / 2.0, -h / 2.0, 0.0]),
            np.array([w / 2.0, -h / 2.0, 0.0]),
            stroke_width=2.5,
            stroke_color=self.stroke_color,
        )
        container_grp.add(self.top_rail, self.bottom_rail)

        # Slot partition markers
        for i in range(self.capacity + 1):
            x = -w / 2.0 + i * self.slot_width
            divider = Line(
                np.array([x, -h / 2.0 + 0.06, 0.0]),
                np.array([x, h / 2.0 - 0.06, 0.0]),
                stroke_width=1.0,
                stroke_color="#444444",
            )
            container_grp.add(divider)
            self.register_semantic_handle(f"slot_divider_{i}", divider)
            if i < self.capacity:
                self.register_semantic_handle(f"slot_{i}", divider)

        self.register_semantic_handle("container", container_grp)
        self.add(container_grp)

        # Queue Title
        title = Text(self.title_text, font_size=max(14, int(self.font_size * 1.0)), color="#FFFFFF")
        title.move_to(np.array([0.0, h / 2.0 + 0.35, 0.0]))
        self.register_semantic_handle("title", title)
        self.add(title)

        # HEAD indicator (left, pointing out to CPU)
        head_label = Text("HEAD", font_size=max(12, int(self.font_size * 0.75)), color="#83C167")
        head_label.move_to(np.array([-w / 2.0 - 0.45, 0.15, 0.0]))
        head_arrow = Arrow(
            np.array([-w / 2.0 + 0.15, -0.15, 0.0]),
            np.array([-w / 2.0 - 0.45, -0.15, 0.0]),
            buff=0.0,
            stroke_width=2.0,
            color="#83C167",
            max_tip_length_to_length_ratio=0.35,
        )
        head_grp = VGroup(head_label, head_arrow)
        self.register_semantic_handle("head", head_grp)
        self.add(head_grp)

        # TAIL indicator (right, pointing into queue)
        tail_label = Text("TAIL", font_size=max(12, int(self.font_size * 0.75)), color="#FFA07A")
        tail_label.move_to(np.array([w / 2.0 + 0.45, 0.15, 0.0]))
        tail_arrow = Arrow(
            np.array([w / 2.0 + 0.45, -0.15, 0.0]),
            np.array([w / 2.0 - 0.15, -0.15, 0.0]),
            buff=0.0,
            stroke_width=2.0,
            color="#FFA07A",
            max_tip_length_to_length_ratio=0.35,
        )
        tail_grp = VGroup(tail_label, tail_arrow)
        self.register_semantic_handle("tail", tail_grp)
        self.add(tail_grp)

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------

    def add_process(self, process: Union[ProcessCardVisual, str], **kwargs) -> None:
        """Directly adds a process card or PID to the queue without animation."""
        if isinstance(process, ProcessCardVisual):
            self.queue_items.append(process)
            self.add(process)
        else:
            pid = str(process)
            self.set_queue_pids([*getattr(self, "queue_pids", []), pid])

    def animate_enqueue(
        self,
        item: Optional[Union[ProcessCardVisual, str]] = None,
        run_time: float = 0.6,
        pid: Optional[str] = None,
        arrival_time: Optional[float] = None,
        burst_time: Optional[float] = None,
        **kwargs,
    ) -> AnimationGroup:
        """Enqueues a process at TAIL and places it in the next open slot."""
        if item is None:
            item = pid if pid is not None else kwargs.get("process", "P1")
        idx = len(self.queue_items)
        if idx >= self.capacity:
            # Over capacity fallback: replace or indicate
            return AnimationGroup(Indicate(self.get_semantic_handle("container"), color="#FF6B6B"))

        target_pos = self.get_slot_center(idx)

        div_left = self.get_semantic_handle(f"slot_divider_{idx}")
        div_right = self.get_semantic_handle(f"slot_divider_{idx + 1}")
        slot_w = abs(div_right.get_center()[0] - div_left.get_center()[0]) if (div_left and div_right) else self.slot_width
        slot_h = abs(self.top_rail.get_center()[1] - self.bottom_rail.get_center()[1]) if self.top_rail else self.slot_height

        if isinstance(item, str):
            card = ProcessCardVisual(
                pid=item,
                state="READY",
                width=slot_w * 0.90,
                height=slot_h * 0.88,
            )
        else:
            card = item
            if card.width > slot_w * 0.90:
                card.scale_to_fit_width(slot_w * 0.90)
            if card.height > slot_h * 0.88:
                card.scale_to_fit_height(slot_h * 0.88)

        # Start from right of TAIL
        tail = self.get_semantic_handle("tail")
        start_x = tail.get_right()[0] + 0.5 if tail else (self.get_center()[0] + self.total_queue_width / 2.0 + 0.8)
        start_pos = np.array([start_x, target_pos[1], 0.0])
        card.move_to(start_pos)
        self.add(card)
        self.queue_items.append(card)

        anim = card.animate(run_time=run_time).move_to(target_pos)
        return AnimationGroup(anim)

    def animate_dequeue(self, animate_head_exit: bool = True, run_time: float = 0.6) -> AnimationGroup:
        """Dequeues the HEAD item (slot 0) towards CPU and shifts subsequent items forward."""
        if not self.queue_items:
            return AnimationGroup(Indicate(self.get_semantic_handle("head"), color="#888888"))

        head_item = self.queue_items.pop(0)
        anims: List[Animation] = []

        if animate_head_exit:
            head = self.get_semantic_handle("head")
            raw_exit_x = head.get_left()[0] - 0.2 if head else (self.get_center()[0] - self.total_queue_width / 2.0 - 0.2)
            exit_x = max(-2.10, raw_exit_x)
            exit_pos = np.array([exit_x, head_item.get_center()[1], 0.0])
            anims.append(head_item.animate(run_time=run_time).move_to(exit_pos).set_opacity(0.0).scale(0.6))

        # Shift remaining queue items leftward by 1 slot
        for i, item in enumerate(self.queue_items):
            target_pos = self.get_slot_center(i)
            anims.append(item.animate(run_time=run_time).move_to(target_pos))

        return AnimationGroup(*anims)

    def animate_peek(self, color: str = "#FFD700", run_time: float = 0.5) -> Animation:
        """Highlights the process currently at the HEAD of the queue."""
        if self.queue_items:
            return Indicate(self.queue_items[0], color=color, scale_factor=1.06, run_time=run_time)
        return Indicate(self.get_semantic_handle("head"), color=color, run_time=run_time)

    def get_head_item(self) -> Optional[VisualComponent]:
        """Returns the item currently at the HEAD of the queue, or None if empty."""
        return self.queue_items[0] if self.queue_items else None

    def set_queue_pids(self, pids: Sequence[str]) -> None:
        """Directly positions process cards for the given sequence of PIDs."""
        for item in self.queue_items:
            self.remove(item)
        self.queue_items.clear()

        for i, pid in enumerate(pids[:self.capacity]):
            card = ProcessCardVisual(
                pid=pid,
                state="READY",
                width=self.slot_width * 0.90,
                height=self.slot_height * 0.88,
            )
            pos = self._get_slot_center(i) + self.get_center()
            card.move_to(pos)
            self.queue_items.append(card)
            self.add(card)

    # Aliases for flexible LLM invocation
    enqueue = animate_enqueue
    dequeue = animate_dequeue
    peek = animate_peek


# ============================================================================
# 3. CPUExecutionVisual (CPU Core Unit)
# ============================================================================

@VisualRegistry.register("cs", "CPUExecutionVisual")
class CPUExecutionVisual(VisualComponent):
    """
    Visual representation of a CPU core execution unit.
    Features core boundary box, dynamic status badge ('IDLE' vs 'EXECUTING: {pid}'),
    slot dock for running ProcessCardVisual, and dispatch/complete animations.
    """
    component_id = "CPUExecutionVisual"
    domain = "cs"
    aliases = ["cpu", "cpu core", "cpu_core", "cpu execution", "cpu_exec", "execution unit"]
    preferred_aspect = 1.4
    suggested_viewport = "dock_split_right"
    supported_handles = ["core_box", "dock_box", "status_badge", "core_label", "exec_slot"]
    action_weights = {
        "animate_dispatch": 1.5,
        "animate_execute_tick": 1.5,
        "animate_complete": 1.5,
        "animate_set_idle": 1.0,
    }

    def __init__(
        self,
        width: float = 2.4,
        height: float = 1.9,
        title: str = "CPU CORE",
        stroke_color: str = "#58C4DD",
        font_size: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.core_width = float(width)
        self.core_height = float(height)
        self.title_text = title
        self.stroke_color = stroke_color
        self.font_size = font_size

        self.cpu_state: str = "IDLE"
        self.current_process: Optional[VisualComponent] = None
        self.active_card: Optional[ProcessCardVisual] = None
        self.current_pid: Optional[str] = None

        self.core_box: Optional[RoundedRectangle] = None
        self.status_badge_box: Optional[RoundedRectangle] = None
        self.status_badge_text: Optional[Text] = None
        self.slot_box: Optional[RoundedRectangle] = None

        self._build_cpu()

    def _build_cpu(self) -> None:
        # Outer core enclosure
        self.core_box = RoundedRectangle(
            corner_radius=0.15,
            width=self.core_width,
            height=self.core_height,
            stroke_color=self.stroke_color,
            stroke_width=2.5,
            fill_color="#101720",
            fill_opacity=0.92,
        )
        self.register_semantic_handle("core_box", self.core_box)
        self.add(self.core_box)

        # Header Title
        title = Text(
            self.title_text,
            font_size=max(14, int(self.font_size * 1.05)),
            weight="BOLD",
            color="#FFFFFF",
        )
        title.move_to(
            self.core_box.get_center() + np.array([0.0, self.core_height * 0.36, 0.0])
        )
        self.register_semantic_handle("title", title)
        self.add(title)

        # Status Badge (upper row below title)
        badge_w = self.core_width * 0.85
        badge_h = 0.30
        self.status_badge_box = RoundedRectangle(
            corner_radius=0.05,
            width=badge_w,
            height=badge_h,
            stroke_width=1.0,
            stroke_color="#888888",
            fill_color="#1F242C",
            fill_opacity=0.9,
        )
        self.status_badge_box.move_to(
            self.core_box.get_center() + np.array([0.0, self.core_height * 0.16, 0.0])
        )

        self.status_badge_text = Text(
            "STATUS: IDLE",
            font_size=max(12, int(self.font_size * 0.75)),
            color="#888888",
        )
        self.status_badge_text.move_to(self.status_badge_box.get_center())

        status_badge = VGroup(self.status_badge_box, self.status_badge_text)
        self.register_semantic_handle("status_badge", status_badge)
        self.register_semantic_handle("status_badge_box", self.status_badge_box)
        self.register_semantic_handle("status_badge_text", self.status_badge_text)
        self.add(self.status_badge_box, self.status_badge_text)

        # Execution Dock Slot (lower area)
        self.slot_box = RoundedRectangle(
            corner_radius=0.08,
            width=self.core_width * 0.88,
            height=self.core_height * 0.52,
            stroke_width=1.0,
            stroke_color="#334455",
            fill_color="#0A0E14",
            fill_opacity=0.6,
        )
        self.slot_box.move_to(
            self.core_box.get_center() + np.array([0.0, -self.core_height * 0.20, 0.0])
        )
        self.register_semantic_handle("slot", self.slot_box)
        self.add(self.slot_box)

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------

    def animate_dispatch(
        self,
        process_card: Optional[Union[ProcessCardVisual, str]] = None,
        item: Optional[Union[ProcessCardVisual, str]] = None,
        pid: Optional[Union[ProcessCardVisual, str]] = None,
        run_time: float = 0.6,
    ) -> AnimationGroup:
        """Dispatches a process into the CPU slot and updates status badge to RUNNING."""
        target = process_card if process_card is not None else (item if item is not None else pid)
        if target is None:
            target = "P"
        slot_center = self.slot_box.get_center()
        slot_w = self.slot_box.width
        slot_h = self.slot_box.height

        anims: List[Animation] = []

        # If there is already an active card in the execution slot, fade it out cleanly and remove
        old_card = self.active_card if self.active_card is not None else (self.current_process if isinstance(self.current_process, Mobject) else None)
        if old_card is not None and old_card != target:
            anims.append(FadeOut(old_card, run_time=run_time * 0.5))
            if old_card in self.submobjects:
                self.remove(old_card)
            self.active_card = None

        if isinstance(target, str):
            card = ProcessCardVisual(
                pid=target,
                state="RUNNING",
                width=slot_w * 0.88,
                height=slot_h * 0.85,
            )
            card.move_to(slot_center + LEFT * 1.5)
            self.add(card)
        else:
            card = target
            if card.width > slot_w * 0.88:
                card.scale_to_fit_width(slot_w * 0.88)
            if card.height > slot_h * 0.85:
                card.scale_to_fit_height(slot_h * 0.85)

        self.active_card = card
        self.current_process = card
        self.cpu_state = "RUNNING"
        pid_name = getattr(card, "pid", str(target))
        self.current_pid = pid_name

        # Target badge
        new_badge_box = RoundedRectangle(
            corner_radius=0.05,
            width=self.core_width * 0.85,
            height=0.30,
            stroke_width=1.2,
            stroke_color="#83C167",
            fill_color="#122A16",
            fill_opacity=0.9,
        ).move_to(self.status_badge_box.get_center())

        new_badge_text = Text(
            f"EXECUTING: {pid_name}",
            font_size=max(12, int(self.font_size * 0.75)),
            color="#83C167",
        )
        if new_badge_text.width > self.core_width * 0.80:
            new_badge_text.scale_to_fit_width(self.core_width * 0.80)
        new_badge_text.move_to(new_badge_box.get_center())

        anims.extend([
            card.animate(run_time=run_time).move_to(slot_center),
            Transform(self.status_badge_box, new_badge_box, run_time=run_time),
            Transform(self.status_badge_text, new_badge_text, run_time=run_time),
            Indicate(self.core_box, color="#83C167", scale_factor=1.03, run_time=run_time),
        ])

        if hasattr(card, "animate_set_state") and getattr(card, "current_state", "") != "RUNNING":
            anims.append(card.animate_set_state("RUNNING", run_time=run_time))

        return AnimationGroup(*anims)

    def animate_execute_tick(
        self,
        remaining_time: Optional[int] = None,
        tick_count: Optional[int] = None,
        ticks: Optional[int] = None,
        run_time: float = 0.4,
    ) -> AnimationGroup:
        """Prorates execution tick with CPU core pulse and optional remaining burst update."""
        rem = remaining_time if remaining_time is not None else (tick_count if tick_count is not None else ticks)
        anims: List[Animation] = [
            Indicate(self.core_box, color="#83C167", scale_factor=1.03, run_time=run_time)
        ]
        if self.current_process and rem is not None:
            if hasattr(self.current_process, "animate_update_remaining_time"):
                anims.append(
                    self.current_process.animate_update_remaining_time(rem, run_time=run_time)
                )
        return AnimationGroup(*anims)

    def animate_complete(
        self,
        process: Optional[Any] = None,
        pid: Optional[str] = None,
        run_time: float = 0.5,
    ) -> AnimationGroup:
        """Transitions current process to TERMINATED and marks completion."""
        if isinstance(process, (int, float)):
            run_time = float(process)
            process = None
        if process is None and pid is not None:
            process = pid

        if isinstance(process, ProcessCardVisual):
            self.active_card = process
            self.current_process = process
            self.current_pid = getattr(process, "pid", None)
        elif isinstance(process, str):
            self.current_pid = process
            # Preserve self.active_card if it exists so Mobject isn't overwritten with string
            if self.active_card is not None:
                self.current_process = self.active_card
            else:
                self.current_process = process

        self.cpu_state = "COMPLETED"
        anims: List[Animation] = []
        pid_name = self.current_pid or (getattr(self.current_process, "pid", "") if self.current_process else "")

        target_card = self.active_card if self.active_card is not None else (self.current_process if isinstance(self.current_process, Mobject) else None)
        if target_card and hasattr(target_card, "animate_set_state"):
            anims.append(target_card.animate_set_state("TERMINATED", run_time=run_time))

        new_badge_box = RoundedRectangle(
            corner_radius=0.05,
            width=self.core_width * 0.85,
            height=0.30,
            stroke_width=1.0,
            stroke_color="#58C4DD",
            fill_color="#12202A",
            fill_opacity=0.9,
        ).move_to(self.status_badge_box.get_center())

        new_badge_text = Text(
            f"COMPLETED: {pid_name}" if pid_name else "COMPLETED",
            font_size=max(12, int(self.font_size * 0.75)),
            color="#58C4DD",
        )
        new_badge_text.move_to(new_badge_box.get_center())

        anims.append(Transform(self.status_badge_box, new_badge_box, run_time=run_time))
        anims.append(Transform(self.status_badge_text, new_badge_text, run_time=run_time))

        return AnimationGroup(*anims)

    def animate_set_idle(
        self,
        process: Optional[Any] = None,
        run_time: float = 0.4,
    ) -> AnimationGroup:
        """Clears execution slot and restores IDLE badge."""
        if isinstance(process, (int, float)):
            run_time = float(process)
        self.cpu_state = "IDLE"
        anims: List[Animation] = []

        card_to_remove = self.active_card if self.active_card is not None else (self.current_process if isinstance(self.current_process, Mobject) else None)
        if card_to_remove is not None:
            anims.append(FadeOut(card_to_remove, run_time=run_time))
            if card_to_remove in self.submobjects:
                self.remove(card_to_remove)
            self.active_card = None

        self.current_process = None
        self.current_pid = None

        new_badge_box = RoundedRectangle(
            corner_radius=0.05,
            width=self.core_width * 0.85,
            height=0.30,
            stroke_width=1.0,
            stroke_color="#888888",
            fill_color="#1F242C",
            fill_opacity=0.9,
        ).move_to(self.status_badge_box.get_center())

        new_badge_text = Text(
            "STATUS: IDLE",
            font_size=max(12, int(self.font_size * 0.75)),
            color="#888888",
        )
        new_badge_text.move_to(new_badge_box.get_center())

        anims.append(Transform(self.status_badge_box, new_badge_box, run_time=run_time))
        anims.append(Transform(self.status_badge_text, new_badge_text, run_time=run_time))

        return AnimationGroup(*anims)

    def set_state_instant(self, cpu_state: str, running_pid: Optional[str] = None) -> None:
        """Instantaneous CPU state synchronization."""
        self.cpu_state = cpu_state.upper()
        if self.cpu_state == "IDLE":
            if self.active_card is not None and self.active_card in self.submobjects:
                self.remove(self.active_card)
            self.active_card = None
            self.current_process = None
            self.current_pid = None
        col = "#83C167" if self.cpu_state == "RUNNING" else "#888888"
        bg = "#122A16" if self.cpu_state == "RUNNING" else "#1F242C"
        lbl = f"EXECUTING: {running_pid}" if running_pid and self.cpu_state == "RUNNING" else "STATUS: IDLE"

        self.status_badge_box.set_stroke(color=col).set_fill(color=bg, opacity=0.9)
        new_text = Text(lbl, font_size=max(12, int(self.font_size * 0.75)), color=col)
        if new_text.width > self.core_width * 0.80:
            new_text.scale_to_fit_width(self.core_width * 0.80)
        new_text.move_to(self.status_badge_box.get_center())
        self.status_badge_text.become(new_text)
        self.status_badge_text.text = lbl

    # Aliases for flexible LLM invocation
    dispatch = animate_dispatch
    execute_tick = animate_execute_tick
    complete = animate_complete
    set_idle = animate_set_idle


# ============================================================================
# 4. GanttChartVisual (Execution Timeline)
# ============================================================================

@VisualRegistry.register("cs", "GanttChartVisual")
class GanttChartVisual(VisualComponent):
    """
    Visual timeline showing process execution intervals and CPU idle gaps.
    Features proportional block widths, PID labels, idle interval styling,
    and a vertical simulation time cursor.
    """
    component_id = "GanttChartVisual"
    domain = "cs"
    aliases = ["gantt", "gantt chart", "gantt_chart", "timeline", "cpu timeline"]
    preferred_aspect = 4.0
    suggested_viewport = "dock_inspector"
    supported_handles = ["baseline", "time_marker", "bar_<pid>", "convoys", "container", "track"]
    action_weights = {
        "animate_add_block": 1.0,
        "animate_update_time_marker": 0.6,
        "animate_highlight": 0.6,
    }

    def __init__(
        self,
        total_time: int = 10,
        width: float = 6.2,
        height: float = 1.0,
        title: str = "GANTT CHART (CPU TIMELINE)",
        font_size: int = 16,
        bar_height: float = 0.42,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.total_time = max(1, int(total_time))
        self.chart_width = float(width)
        self.chart_height = float(height)
        self.title_text = title
        self.font_size = font_size
        self.bar_height = float(bar_height)

        self.current_marker_time: float = 0.0
        self.blocks_group = VGroup()
        self.axis_group = VGroup()
        self.axis_group.x_axis = self.axis_group
        self.axis_group.axes = self.axis_group
        self.cursor: Optional[VGroup] = None

        self._pid_color_map: Dict[str, str] = {}
        self._color_index = 0

        self._build_chart()

    @property
    def axes(self) -> VGroup:
        return self.axis_group

    @property
    def x_axis(self) -> VGroup:
        return self.axis_group

    @property
    def process_bars(self) -> List[Any]:
        return self.blocks_group.submobjects if self.blocks_group.submobjects else [self.blocks_group]

    def _get_process_color(self, pid: Optional[str]) -> str:
        if pid is None or pid.upper() == "IDLE":
            return "#444444"
        if pid not in self._pid_color_map:
            self._pid_color_map[pid] = PROCESS_PALETTE[self._color_index % len(PROCESS_PALETTE)]
            self._color_index += 1
        return self._pid_color_map[pid]

    def _time_to_x(self, t: float) -> float:
        """Converts timeline time to x coordinate on the chart track."""
        t_clamped = max(0.0, min(float(t), float(self.total_time)))
        if hasattr(self, "track") and self.track is not None:
            left_x = self.track.get_left()[0]
            w = self.track.width
            return left_x + (t_clamped / self.total_time) * w
        return -self.chart_width / 2.0 + (t_clamped / self.total_time) * self.chart_width

    def _build_chart(self) -> None:
        w = self.chart_width
        h = self.bar_height

        # Title
        title = Text(self.title_text, font_size=max(14, int(self.font_size * 0.95)), color="#FFFFFF")
        title.move_to(np.array([0.0, h / 2.0 + 0.32, 0.0]))
        self.register_semantic_handle("title", title)
        self.add(title)

        # Baseline axis line
        baseline_y = -h / 2.0 - 0.08
        baseline = Line(
            np.array([-w / 2.0, baseline_y, 0.0]),
            np.array([w / 2.0, baseline_y, 0.0]),
            stroke_width=2.0,
            stroke_color="#888888",
        )
        self.axis_group.add(baseline)

        # Time ticks and labels
        for t in range(self.total_time + 1):
            x = self._time_to_x(t)
            tick = Line(
                np.array([x, baseline_y - 0.04, 0.0]),
                np.array([x, baseline_y + 0.04, 0.0]),
                stroke_width=1.5,
                stroke_color="#AAAAAA",
            )
            t_lbl = Text(str(t), font_size=max(12, int(self.font_size * 0.75)), color="#CCCCCC")
            t_lbl.move_to(np.array([x, baseline_y - 0.16, 0.0]))
            self.axis_group.add(tick, t_lbl)

        # Background track outline
        self.track = Rectangle(
            width=w,
            height=h,
            stroke_width=1.0,
            stroke_color="#334455",
            fill_color="#0D1117",
            fill_opacity=0.6,
        ).move_to(np.array([0.0, 0.0, 0.0]))
        self.axis_group.add(self.track)
        self.register_semantic_handle("track", self.track)

        self.register_semantic_handle("axis", self.axis_group)
        self.add(self.axis_group)

        # Scheduled blocks container
        self.register_semantic_handle("blocks_group", self.blocks_group)
        self.add(self.blocks_group)

        # Time Marker / Cursor (vertical line)
        cursor_x = self._time_to_x(0.0)
        cursor_line = Line(
            np.array([cursor_x, baseline_y - 0.05, 0.0]),
            np.array([cursor_x, h / 2.0 + 0.12, 0.0]),
            stroke_width=2.5,
            stroke_color="#FFD700",
        )
        cursor_tip = Arrow(
            np.array([cursor_x, h / 2.0 + 0.22, 0.0]),
            np.array([cursor_x, h / 2.0 + 0.04, 0.0]),
            buff=0.0,
            stroke_width=2.0,
            color="#FFD700",
            max_tip_length_to_length_ratio=0.5,
        )
        self.cursor = VGroup(cursor_line, cursor_tip)
        self.register_semantic_handle("cursor", self.cursor)
        self.add(self.cursor)

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------

    def animate_add_block(
        self,
        pid: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        color: Optional[str] = None,
        run_time: float = 0.6,
        **kwargs,
    ) -> AnimationGroup:
        """Animates addition of a Gantt interval block and advances the time cursor."""
        if start_time is None:
            start_time = kwargs.get("start", kwargs.get("s", 0))
        if end_time is None:
            end_time = kwargs.get("end", kwargs.get("e", 0))
        start_time = int(start_time)
        end_time = int(end_time)
        duration = max(1, end_time - start_time)
        track_w = self.track.width if (hasattr(self, "track") and self.track is not None) else self.chart_width
        track_h = self.track.height if (hasattr(self, "track") and self.track is not None) else self.bar_height
        track_y = self.track.get_center()[1] if (hasattr(self, "track") and self.track is not None) else self.get_center()[1]

        block_w = (duration / self.total_time) * track_w
        x1 = self._time_to_x(start_time)
        x2 = self._time_to_x(end_time)
        block_x = (x1 + x2) / 2.0

        is_idle = pid is None or str(pid).upper() == "IDLE"
        block_col = "#555555" if is_idle else (color or self._get_process_color(pid))
        fill_opacity = 0.45 if is_idle else 0.85

        block_rect = Rectangle(
            width=block_w,
            height=track_h,
            stroke_width=1.5,
            stroke_color=block_col,
            fill_color=block_col,
            fill_opacity=fill_opacity,
        ).move_to(np.array([block_x, track_y, 0.0]))

        label_text = "IDLE" if is_idle else str(pid)
        block_label = Text(
            label_text,
            font_size=max(13, int(self.font_size * 0.85)),
            weight="BOLD" if not is_idle else "NORMAL",
            color="#FFFFFF" if not is_idle else "#888888",
        )
        if block_label.width > block_w * 0.85:
            block_label.scale_to_fit_width(block_w * 0.85)
        if block_label.height > track_h * 0.80:
            block_label.scale_to_fit_height(track_h * 0.80)
        block_label.move_to(block_rect.get_center())

        block_grp = VGroup(block_rect, block_label)
        self.blocks_group.add(block_grp)
        handle_name = f"block_{label_text}_{start_time}"
        self.register_semantic_handle(handle_name, block_grp)

        # Move cursor to end_time
        anim_cursor = self.animate_update_time_marker(end_time, run_time=run_time)

        return AnimationGroup(FadeIn(block_grp, run_time=run_time), anim_cursor)

    def animate_update_time_marker(self, new_time: Optional[float] = None, run_time: float = 0.4, **kwargs) -> AnimationGroup:
        """Slides the vertical cursor line horizontally to new simulation time."""
        if new_time is None:
            new_time = kwargs.get("new_time", kwargs.get("time", kwargs.get("t", kwargs.get("target_time", kwargs.get("start", kwargs.get("start_time", 0.0))))))
        self.current_marker_time = float(new_time)
        target_x = self._time_to_x(new_time)
        curr_x = self.cursor.get_center()[0]
        shift_vec = np.array([target_x - curr_x, 0.0, 0.0])
        return AnimationGroup(self.cursor.animate(run_time=run_time).shift(shift_vec))

    def animate_highlight(self, color: str = "#FFD700", run_time: float = 0.5) -> Animation:
        """Highlights the Gantt chart timeline track or container."""
        target = getattr(self, "track", self)
        return Indicate(target, color=color, scale_factor=1.03, run_time=run_time)

    def set_gantt_blocks(self, blocks: Sequence[Union[Dict[str, Any], Any]]) -> None:
        """Direct batch initialization or instantaneous update of timeline blocks."""
        for b in self.blocks_group:
            self.remove(b)
        self.blocks_group.submobjects.clear()

        max_t = 0
        for block in blocks:
            if isinstance(block, dict):
                pid = block.get("pid")
                s = int(block.get("start_time", 0))
                e = int(block.get("end_time", 0))
            else:
                pid = getattr(block, "pid", None)
                s = int(getattr(block, "start_time", 0))
                e = int(getattr(block, "end_time", 0))

            max_t = max(max_t, e)
            duration = max(1, e - s)
            track_w = self.track.width if (hasattr(self, "track") and self.track is not None) else self.chart_width
            track_h = self.track.height if (hasattr(self, "track") and self.track is not None) else self.bar_height
            track_y = self.track.get_center()[1] if (hasattr(self, "track") and self.track is not None) else self.get_center()[1]

            block_w = (duration / self.total_time) * track_w
            x1 = self._time_to_x(s)
            x2 = self._time_to_x(e)
            block_x = (x1 + x2) / 2.0

            is_idle = pid is None or str(pid).upper() == "IDLE"
            block_col = "#555555" if is_idle else self._get_process_color(pid)
            fill_opacity = 0.45 if is_idle else 0.85

            block_rect = Rectangle(
                width=block_w,
                height=track_h,
                stroke_width=1.5,
                stroke_color=block_col,
                fill_color=block_col,
                fill_opacity=fill_opacity,
            ).move_to(np.array([block_x, track_y, 0.0]))

            lbl_txt = "IDLE" if is_idle else str(pid)
            block_lbl = Text(
                lbl_txt,
                font_size=max(13, int(self.font_size * 0.85)),
                weight="BOLD" if not is_idle else "NORMAL",
                color="#FFFFFF" if not is_idle else "#888888",
            )
            if block_lbl.width > block_w * 0.85:
                block_lbl.scale_to_fit_width(block_w * 0.85)
            block_lbl.move_to(block_rect.get_center())

            bg = VGroup(block_rect, block_lbl)
            self.blocks_group.add(bg)
            self.register_semantic_handle(f"block_{lbl_txt}_{s}", bg)

        # Reposition cursor
        target_x = self.get_center()[0] + self._time_to_x(max_t)
        curr_x = self.cursor.get_center()[0]
        self.cursor.shift(np.array([target_x - curr_x, 0.0, 0.0]))
        self.current_marker_time = float(max_t)

    # Aliases for flexible LLM invocation
    add_block = animate_add_block
    update_time_marker = animate_update_time_marker


# ============================================================================
# 5. MetricTableVisual (Accounting Matrix)
# ============================================================================

@VisualRegistry.register("cs", "MetricTableVisual")
class MetricTableVisual(VisualComponent):
    """
    Evolving tabular matrix tracking per-process metrics: PID, AT, BT, CT, TAT, WT.
    Enforces honest accounting: CT, TAT, WT display as '-' until process completes.
    Summary row displays average TAT and WT once simulation terminates.
    """
    component_id = "MetricTableVisual"
    domain = "cs"
    aliases = ["metric table", "metric_table", "metrics table", "metrics_table", "metrics", "accounting table"]
    preferred_aspect = 2.4
    suggested_viewport = "dock_main"
    supported_handles = ["header_row", "summary_row", "cell_<pid>_CT", "cell_<pid>_TAT", "cell_<pid>_WT", "cell_<pid>_RT", "cell_avg_tat", "cell_avg_wt", "cell_avg_rt"]
    action_weights = {
        "animate_update_response_time": 0.6,
        "animate_update_process_metrics": 0.6,
        "animate_show_averages": 0.6,
        "highlight_cell_avg_wt": 0.6,
        "highlight_cell_avg_tat": 0.6,
        "highlight_cell_avg_rt": 0.6,
    }

    COLUMNS = ["PID", "AT", "BT", "CT", "TAT", "WT", "RT"]

    def __init__(
        self,
        processes: Optional[Sequence[Union[Dict[str, Any], Any]]] = None,
        width: float = 6.2,
        height: float = 2.0,
        title: Optional[str] = "PROCESS SCHEDULING METRICS",
        font_size: int = 16,
        header_color: str = "#FFA07A",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.table_width = float(width)
        self.table_height = float(height)
        self.title_text = title
        self.font_size = font_size
        self.header_color = header_color

        if processes is None:
            processes = [
                {"pid": "P1", "arrival_time": 0, "burst_time": 3},
                {"pid": "P2", "arrival_time": 1, "burst_time": 4},
                {"pid": "P3", "arrival_time": 2, "burst_time": 2},
            ]

        # Normalize process specifications
        self.processes: List[Dict[str, Any]] = []
        for p in processes:
            if isinstance(p, dict):
                self.processes.append({
                    "pid": str(p.get("pid", "P?")),
                    "arrival_time": int(p.get("arrival_time", 0)),
                    "burst_time": int(p.get("burst_time", 1)),
                })
            elif isinstance(p, str):
                self.processes.append({
                    "pid": str(p),
                    "arrival_time": 0,
                    "burst_time": 1,
                })
            else:
                self.processes.append({
                    "pid": str(getattr(p, "pid", "P?")),
                    "arrival_time": int(getattr(p, "arrival_time", 0)),
                    "burst_time": int(getattr(p, "burst_time", 1)),
                })

        self.num_rows = len(self.processes)
        # Total rows = header + process rows + summary row
        self.total_rows = self.num_rows + 2
        self.row_height = self.table_height / max(1, self.total_rows)
        self.col_width = self.table_width / len(self.COLUMNS)

        self._build_table()

    def _get_cell_center(self, row: int, col: int) -> np.ndarray:
        """Row 0: Header; Row 1..N: Processes; Row N+1: Summary."""
        top_y = self.table_height / 2.0
        left_x = -self.table_width / 2.0
        x = left_x + (col + 0.5) * self.col_width
        y = top_y - (row + 0.5) * self.row_height
        return np.array([x, y, 0.0])

    def _build_table(self) -> None:
        table_grp = VGroup()

        # Title
        if self.title_text:
            title = Text(
                self.title_text,
                font_size=max(14, int(self.font_size * 1.05)),
                weight="BOLD",
                color="#FFFFFF",
            )
            title.move_to(np.array([0.0, self.table_height / 2.0 + 0.28, 0.0]))
            self.register_semantic_handle("title", title)
            self.add(title)

        # 1. Header Row (row 0)
        header_grp = VGroup()
        for c, col_name in enumerate(self.COLUMNS):
            box = Rectangle(
                width=self.col_width,
                height=self.row_height,
                stroke_width=1.0,
                stroke_color="#555555",
                fill_color="#221A1A",
                fill_opacity=0.9,
            ).move_to(self._get_cell_center(0, c))

            txt = Text(
                col_name,
                font_size=max(13, int(self.font_size * 0.85)),
                weight="BOLD",
                color=self.header_color,
            )
            txt.move_to(box.get_center())
            header_grp.add(box, txt)
            self.register_semantic_handle(f"header_{col_name}", txt)

        self.register_semantic_handle("header", header_grp)
        table_grp.add(header_grp)

        # 2. Process Rows (row 1 to num_rows)
        for r, p in enumerate(self.processes, start=1):
            pid = p["pid"]
            row_grp = VGroup()

            cell_values = {
                "PID": pid,
                "AT": str(p["arrival_time"]),
                "BT": str(p["burst_time"]),
                "CT": "-",
                "TAT": "-",
                "WT": "-",
                "RT": "-",
            }

            for c, col_name in enumerate(self.COLUMNS):
                val = cell_values[col_name]
                box = Rectangle(
                    width=self.col_width,
                    height=self.row_height,
                    stroke_width=1.0,
                    stroke_color="#334455",
                    fill_color="#101720",
                    fill_opacity=0.75,
                ).move_to(self._get_cell_center(r, c))

                val_col = "#CCCCCC" if col_name in ("PID", "AT", "BT") else "#666666"
                txt = Text(
                    val,
                    font_size=max(12, int(self.font_size * 0.80)),
                    color=val_col,
                )
                txt.move_to(box.get_center())

                row_grp.add(box, txt)
                self.register_semantic_handle(f"cell_{pid}_{col_name}", txt)
                self.register_semantic_handle(f"box_{pid}_{col_name}", box)

            self.register_semantic_handle(f"row_{pid}", row_grp)
            table_grp.add(row_grp)

        # 3. Summary Row (row num_rows + 1)
        summary_r = self.num_rows + 1
        summary_grp = VGroup()

        # First 4 columns merged span for 'AVERAGE'
        span_w = self.col_width * 4.0
        span_center_x = -self.table_width / 2.0 + span_w / 2.0
        span_center_y = self.table_height / 2.0 - (summary_r + 0.5) * self.row_height
        avg_box = Rectangle(
            width=span_w,
            height=self.row_height,
            stroke_width=1.0,
            stroke_color="#555555",
            fill_color="#1C2028",
            fill_opacity=0.9,
        ).move_to(np.array([span_center_x, span_center_y, 0.0]))

        avg_lbl = Text(
            "AVERAGE",
            font_size=max(12, int(self.font_size * 0.80)),
            weight="BOLD",
            color="#FFD700",
        )
        avg_lbl.move_to(avg_box.get_center())
        summary_grp.add(avg_box, avg_lbl)

        # Column 4: TAT Average
        tat_box = Rectangle(
            width=self.col_width,
            height=self.row_height,
            stroke_width=1.0,
            stroke_color="#555555",
            fill_color="#1C2028",
            fill_opacity=0.9,
        ).move_to(self._get_cell_center(summary_r, 4))
        avg_tat_txt = Text("-", font_size=max(12, int(self.font_size * 0.80)), color="#666666")
        avg_tat_txt.move_to(tat_box.get_center())
        summary_grp.add(tat_box, avg_tat_txt)
        self.register_semantic_handle("cell_avg_tat", avg_tat_txt)

        # Column 5: WT Average
        wt_box = Rectangle(
            width=self.col_width,
            height=self.row_height,
            stroke_width=1.0,
            stroke_color="#555555",
            fill_color="#1C2028",
            fill_opacity=0.9,
        ).move_to(self._get_cell_center(summary_r, 5))
        avg_wt_txt = Text("-", font_size=max(12, int(self.font_size * 0.80)), color="#666666")
        avg_wt_txt.move_to(wt_box.get_center())
        summary_grp.add(wt_box, avg_wt_txt)
        self.register_semantic_handle("cell_avg_wt", avg_wt_txt)

        # Column 6: RT Average
        rt_box = Rectangle(
            width=self.col_width,
            height=self.row_height,
            stroke_width=1.0,
            stroke_color="#555555",
            fill_color="#1C2028",
            fill_opacity=0.9,
        ).move_to(self._get_cell_center(summary_r, 6))
        avg_rt_txt = Text("-", font_size=max(12, int(self.font_size * 0.80)), color="#666666")
        avg_rt_txt.move_to(rt_box.get_center())
        summary_grp.add(rt_box, avg_rt_txt)
        self.register_semantic_handle("cell_avg_rt", avg_rt_txt)

        self.register_semantic_handle("summary_row", summary_grp)
        table_grp.add(summary_grp)

        self.add(table_grp)

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------

    def animate_update_response_time(
        self,
        pid: str,
        rt: Optional[int] = None,
        response_time: Optional[int] = None,
        highlight_color: str = "#83C167",
        run_time: float = 0.6,
    ) -> AnimationGroup:
        """Reveals honest Response Time (RT) when process is first dispatched to CPU."""
        if rt is None and response_time is not None:
            rt = response_time
        rt = 0 if rt is None else int(rt)

        anims: List[Animation] = []
        old_txt = self.get_semantic_handle(f"cell_{pid}_RT")
        box = self.get_semantic_handle(f"box_{pid}_RT")

        if old_txt is not None:
            new_txt = Text(
                str(rt),
                font_size=max(12, int(self.font_size * 0.80)),
                color=highlight_color,
            ).move_to(old_txt.get_center())
            anims.append(Transform(old_txt, new_txt, run_time=run_time))
            self.register_semantic_handle(f"cell_{pid}_RT", new_txt)

        if box is not None:
            anims.append(Indicate(box, color=highlight_color, scale_factor=1.05, run_time=run_time))

        return AnimationGroup(*anims)

    def animate_update_process_metrics(
        self,
        pid: str,
        ct: Optional[int] = None,
        tat: Optional[int] = None,
        wt: Optional[int] = None,
        rt: Optional[int] = None,
        completion_time: Optional[int] = None,
        turnaround_time: Optional[int] = None,
        waiting_time: Optional[int] = None,
        response_time: Optional[int] = None,
        highlight_color: str = "#83C167",
        run_time: float = 0.6,
        **kwargs,
    ) -> AnimationGroup:
        """Reveals honest metrics (CT, TAT, WT, and optionally RT) for a newly terminated process."""
        if ct is None:
            ct = kwargs.get("CT", kwargs.get("ct", completion_time))
        if tat is None:
            tat = kwargs.get("TAT", kwargs.get("tat", turnaround_time))
        if wt is None:
            wt = kwargs.get("WT", kwargs.get("wt", waiting_time))
        if rt is None:
            rt = kwargs.get("RT", kwargs.get("rt", response_time))

        ct = 0 if ct is None else int(ct)
        tat = 0 if tat is None else int(tat)
        wt = 0 if wt is None else int(wt)
        anims: List[Animation] = []
        updates = [("CT", ct), ("TAT", tat), ("WT", wt)]
        if rt is not None:
            updates.append(("RT", int(rt)))

        for col_name, val in updates:
            old_txt = self.get_semantic_handle(f"cell_{pid}_{col_name}")
            box = self.get_semantic_handle(f"box_{pid}_{col_name}")

            if old_txt is not None:
                new_txt = Text(
                    str(val),
                    font_size=max(12, int(self.font_size * 0.80)),
                    color=highlight_color,
                ).move_to(old_txt.get_center())

                anims.append(Transform(old_txt, new_txt, run_time=run_time))
                self.register_semantic_handle(f"cell_{pid}_{col_name}", new_txt)

            if box is not None:
                anims.append(Indicate(box, color=highlight_color, scale_factor=1.05, run_time=run_time))

        return AnimationGroup(*anims)

    def animate_show_averages(
        self,
        avg_tat: Optional[float] = None,
        avg_wt: Optional[float] = None,
        avg_rt: Optional[float] = None,
        avg_turnaround_time: Optional[float] = None,
        avg_waiting_time: Optional[float] = None,
        avg_response_time: Optional[float] = None,
        highlight_color: str = "#FFD700",
        run_time: float = 0.6,
        **kwargs,
    ) -> AnimationGroup:
        """Reveals system-level average TAT, WT, and optionally RT upon simulation completion."""
        if avg_tat is None:
            avg_tat = avg_turnaround_time if avg_turnaround_time is not None else kwargs.get("tat", 0.0)
        if avg_wt is None:
            avg_wt = avg_waiting_time if avg_waiting_time is not None else kwargs.get("wt", 0.0)
        if avg_rt is None:
            avg_rt = avg_response_time if avg_response_time is not None else kwargs.get("rt", None)
        anims: List[Animation] = []

        tat_handle = self.get_semantic_handle("cell_avg_tat")
        if tat_handle is not None:
            tat_str = f"{avg_tat:.2f}" if isinstance(avg_tat, float) else str(avg_tat)
            new_tat = Text(
                tat_str,
                font_size=max(12, int(self.font_size * 0.80)),
                color=highlight_color,
            ).move_to(tat_handle.get_center())
            anims.append(Transform(tat_handle, new_tat, run_time=run_time))
            self.register_semantic_handle("cell_avg_tat", new_tat)

        wt_handle = self.get_semantic_handle("cell_avg_wt")
        if wt_handle is not None:
            wt_str = f"{avg_wt:.2f}" if isinstance(avg_wt, float) else str(avg_wt)
            new_wt = Text(
                wt_str,
                font_size=max(12, int(self.font_size * 0.80)),
                color=highlight_color,
            ).move_to(wt_handle.get_center())
            anims.append(Transform(wt_handle, new_wt, run_time=run_time))
            self.register_semantic_handle("cell_avg_wt", new_wt)

        if avg_rt is not None:
            rt_handle = self.get_semantic_handle("cell_avg_rt")
            if rt_handle is not None:
                rt_str = f"{avg_rt:.2f}" if isinstance(avg_rt, float) else str(avg_rt)
                new_rt = Text(
                    rt_str,
                    font_size=max(12, int(self.font_size * 0.80)),
                    color=highlight_color,
                ).move_to(rt_handle.get_center())
                anims.append(Transform(rt_handle, new_rt, run_time=run_time))
                self.register_semantic_handle("cell_avg_rt", new_rt)

        summary_row = self.get_semantic_handle("summary_row")
        if summary_row is not None:
            anims.append(Indicate(summary_row, color=highlight_color, scale_factor=1.03, run_time=run_time))

        return AnimationGroup(*anims)

    def animate_highlight_cell(
        self,
        cell_name_or_handle: str,
        color: str = "#FFD700",
        run_time: float = 0.5,
    ) -> Animation:
        """Highlights a specific metric cell or summary handle."""
        handle = self.get_semantic_handle(cell_name_or_handle)
        if handle is not None:
            return Indicate(handle, color=color, scale_factor=1.05, run_time=run_time)
        return Wait(run_time)

    def animate_highlight_row(
        self,
        row: int = 0,
        color: str = "#FFD700",
        run_time: float = 0.5,
        **kwargs,
    ) -> Animation:
        """Highlights a specific process row (0=P1, 1=P2, 2=P3)."""
        pid = self.processes[row]["pid"] if 0 <= row < len(self.processes) else None
        handle = self.get_semantic_handle(f"row_{pid}") if pid else self.get_semantic_handle("summary_row")
        if handle is not None:
            return Indicate(handle, color=color, scale_factor=1.03, run_time=run_time)
        return Wait(run_time)

    def animate_highlight(
        self,
        color: str = "#FFD700",
        run_time: float = 0.5,
        **kwargs,
    ) -> Animation:
        """Highlights the entire table or summary row."""
        return Indicate(self, color=color, scale_factor=1.02, run_time=run_time)

    def highlight_cell_avg_wt(self, color: str = "#FFD700", run_time: float = 0.5) -> Animation:
        """Convenience method to highlight average waiting time cell."""
        return self.animate_highlight_cell("cell_avg_wt", color=color, run_time=run_time)

    def highlight_cell_avg_tat(self, color: str = "#FFD700", run_time: float = 0.5) -> Animation:
        """Convenience method to highlight average turnaround time cell."""
        return self.animate_highlight_cell("cell_avg_tat", color=color, run_time=run_time)

    def highlight_cell_avg_rt(self, color: str = "#FFD700", run_time: float = 0.5) -> Animation:
        """Convenience method to highlight average response time cell."""
        return self.animate_highlight_cell("cell_avg_rt", color=color, run_time=run_time)

    def set_metrics_from_state(
        self,
        process_metrics: Dict[str, Dict[str, Optional[int]]],
        avg_tat: Optional[float] = None,
        avg_wt: Optional[float] = None,
        avg_rt: Optional[float] = None,
    ) -> None:
        """Direct batch synchronization of table to snapshot state."""
        for pid, m in process_metrics.items():
            for col_name, val_key in [("CT", "completion_time"), ("TAT", "turnaround_time"), ("WT", "waiting_time"), ("RT", "response_time")]:
                val = m.get(val_key)
                txt_handle = self.get_semantic_handle(f"cell_{pid}_{col_name}")
                if txt_handle is not None:
                    txt_str = "-" if val is None else str(val)
                    col = "#666666" if val is None else "#83C167"
                    new_txt = Text(txt_str, font_size=max(12, int(self.font_size * 0.80)), color=col)
                    new_txt.move_to(txt_handle.get_center())
                    txt_handle.become(new_txt)
                    txt_handle.text = txt_str

        # Averages
        tat_handle = self.get_semantic_handle("cell_avg_tat")
        if tat_handle is not None:
            tat_str = "-" if avg_tat is None else f"{avg_tat:.2f}"
            col = "#666666" if avg_tat is None else "#FFD700"
            new_tat = Text(tat_str, font_size=max(12, int(self.font_size * 0.80)), color=col)
            new_tat.move_to(tat_handle.get_center())
            tat_handle.become(new_tat)
            tat_handle.text = tat_str

        wt_handle = self.get_semantic_handle("cell_avg_wt")
        if wt_handle is not None:
            wt_str = "-" if avg_wt is None else f"{avg_wt:.2f}"
            col = "#666666" if avg_wt is None else "#FFD700"
            new_wt = Text(wt_str, font_size=max(12, int(self.font_size * 0.80)), color=col)
            new_wt.move_to(wt_handle.get_center())
            wt_handle.become(new_wt)
            wt_handle.text = wt_str

        rt_handle = self.get_semantic_handle("cell_avg_rt")
        if rt_handle is not None:
            rt_str = "-" if avg_rt is None else f"{avg_rt:.2f}"
            col = "#666666" if avg_rt is None else "#FFD700"
            new_rt = Text(rt_str, font_size=max(12, int(self.font_size * 0.80)), color=col)
            new_rt.move_to(rt_handle.get_center())
            rt_handle.become(new_rt)
            rt_handle.text = rt_str

    # Aliases for flexible LLM invocation
    update_response_time = animate_update_response_time
    update_process_metrics = animate_update_process_metrics
    show_averages = animate_show_averages


# ============================================================================
# 6. OSTraceVisualAdapter (Trace-to-Visual Bridge)
# ============================================================================

class OSTraceVisualAdapter:
    """
    Deterministic bridge between ComputationTrace / StateSnapshot and CineLearn visual components.
    Inspects snapshot transitions to extract visual deltas (arrivals, dispatches, completions,
    Gantt blocks, and metric reveals) without coupling visual components directly to simulators.
    """

    def __init__(self, trace: Any):
        if not hasattr(trace, "snapshots"):
            raise TypeError("Expected ComputationTrace instance with 'snapshots' attribute")
        self.trace = trace
        self.processes: List[Dict[str, Any]] = copy.deepcopy(
            getattr(trace, "metadata", {}).get("processes", [])
        )

    @property
    def total_steps(self) -> int:
        return len(self.trace.snapshots)

    def get_process_specs(self) -> List[Dict[str, Any]]:
        """Returns normalized process specifications from trace metadata."""
        return list(self.processes)

    def get_total_simulation_time(self) -> int:
        """Determines total elapsed timeline duration across all snapshots."""
        max_t = 0
        for snap in self.trace.snapshots:
            st = snap.state
            max_t = max(max_t, int(st.get("current_time", 0)))
            for b in st.get("gantt_history", []):
                max_t = max(max_t, int(b.get("end_time", 0)))
        return max_t

    def get_snapshot(self, step_index: int) -> Any:
        """Retrieves StateSnapshot at step_index."""
        if 0 <= step_index < len(self.trace.snapshots):
            return self.trace.snapshots[step_index]
        raise IndexError(f"Step index {step_index} out of range [0, {len(self.trace.snapshots)-1}]")

    def get_state_at_step(self, step_index: int) -> Dict[str, Any]:
        """Retrieves state dictionary at step_index."""
        return self.get_snapshot(step_index).state

    def get_visual_delta(self, from_step: int, to_step: int) -> Dict[str, Any]:
        """
        Computes the visual transition delta between step from_step and to_step:
        - time_delta: simulation clock advancement
        - arrivals: processes newly entering READY queue
        - dispatched_pid: process allocated to CPU (if state changed to RUNNING)
        - completions: processes newly transitioned to TERMINATED
        - new_gantt_blocks: timeline blocks added in this step
        - metric_updates: newly revealed {pid: {CT, TAT, WT}}
        - averages_revealed: (avg_tat, avg_wt) if revealed at to_step, else None
        """
        st_from = self.get_state_at_step(from_step)
        st_to = self.get_state_at_step(to_step)

        time_start = int(st_from.get("current_time", 0))
        time_end = int(st_to.get("current_time", 0))

        states_from: Dict[str, str] = st_from.get("process_states", {})
        states_to: Dict[str, str] = st_to.get("process_states", {})

        # Arrivals: transitioned to READY
        arrivals = [
            pid for pid, s_to in states_to.items()
            if s_to == "READY" and states_from.get(pid) in ("NOT_ARRIVED", "NEW")
        ]

        # Dispatch: running_pid changed or newly RUNNING
        dispatched_pid = None
        if st_to.get("cpu_state") == "RUNNING":
            if st_from.get("cpu_state") != "RUNNING" or st_to.get("running_pid") != st_from.get("running_pid"):
                dispatched_pid = st_to.get("running_pid")

        # Completions: transitioned to TERMINATED
        completions = [
            pid for pid, s_to in states_to.items()
            if s_to == "TERMINATED" and states_from.get(pid) != "TERMINATED"
        ]

        # Gantt delta
        g_from = st_from.get("gantt_history", [])
        g_to = st_to.get("gantt_history", [])
        new_gantt_blocks = g_to[len(g_from):]

        # Metric updates for newly completed processes
        metrics_from: Dict[str, Dict] = st_from.get("process_metrics", {})
        metrics_to: Dict[str, Dict] = st_to.get("process_metrics", {})
        metric_updates = {}
        for pid in completions:
            m = metrics_to.get(pid, {})
            metric_updates[pid] = {
                "ct": m.get("completion_time"),
                "tat": m.get("turnaround_time"),
                "wt": m.get("waiting_time"),
                "rt": m.get("response_time"),
            }

        # Response time reveals (newly known upon first CPU dispatch)
        response_time_updates = {}
        for p_id, m_to in metrics_to.items():
            m_fr = metrics_from.get(p_id, {})
            if m_fr.get("response_time") is None and m_to.get("response_time") is not None:
                response_time_updates[p_id] = m_to.get("response_time")

        # Averages revealed
        averages_revealed = None
        if (st_from.get("average_turnaround_time") is None and
                st_to.get("average_turnaround_time") is not None):
            averages_revealed = (
                float(st_to["average_turnaround_time"]),
                float(st_to["average_waiting_time"]),
                float(st_to.get("average_response_time", 0.0)),
            )

        return {
            "time_start": time_start,
            "time_end": time_end,
            "arrivals": arrivals,
            "dispatched_pid": dispatched_pid,
            "completions": completions,
            "new_gantt_blocks": new_gantt_blocks,
            "metric_updates": metric_updates,
            "response_time_updates": response_time_updates,
            "averages_revealed": averages_revealed,
            "ready_queue": list(st_to.get("ready_queue", [])),
            "cpu_state": st_to.get("cpu_state", "IDLE"),
            "running_pid": st_to.get("running_pid"),
        }

    def sync_all(self, components: Dict[str, Any], step_index: int) -> None:
        """
        Instantaneously synchronizes a dictionary of visual components to step_index:
        - 'queue': ReadyQueueVisual
        - 'cpu': CPUExecutionVisual
        - 'gantt': GanttChartVisual
        - 'table': MetricTableVisual
        """
        st = self.get_state_at_step(step_index)

        if "queue" in components and hasattr(components["queue"], "set_queue_pids"):
            components["queue"].set_queue_pids(st.get("ready_queue", []))

        if "cpu" in components and hasattr(components["cpu"], "set_state_instant"):
            components["cpu"].set_state_instant(st.get("cpu_state", "IDLE"), st.get("running_pid"))

        if "gantt" in components and hasattr(components["gantt"], "set_gantt_blocks"):
            components["gantt"].set_gantt_blocks(st.get("gantt_history", []))

        if "table" in components and hasattr(components["table"], "set_metrics_from_state"):
            components["table"].set_metrics_from_state(
                st.get("process_metrics", {}),
                st.get("average_turnaround_time"),
                st.get("average_waiting_time"),
                st.get("average_response_time"),
            )


# ============================================================================
# 7. Domain Decoupling Registrations (Guidance Formatting & Fidelity Verifier)
# ============================================================================

def format_fcfs_storyboard_guidance(algo_state: Dict[str, Any]) -> str:
    """Formats verified FCFS simulation facts into prompt guidance for Stage 2 Storyboards."""
    processes = algo_state.get("processes", [])
    gantt = algo_state.get("gantt_chart", [])
    metrics = algo_state.get("process_metrics", {})
    avg_tat = algo_state.get("average_turnaround_time", 0.0)
    avg_wt = algo_state.get("average_waiting_time", 0.0)
    avg_rt = algo_state.get("average_response_time", 0.0)
    total_time = algo_state.get("total_simulation_time", 0)

    proc_strs = [
        f"{p.get('pid')}: Arrival Time = {p.get('arrival_time')}, Burst Time = {p.get('burst_time')}"
        for p in processes
    ]
    procs_formatted = "\n".join(f"  * {s}" for s in proc_strs)

    gantt_strs = []
    for b in gantt:
        if b.get("is_idle"):
            gantt_strs.append(f"  * Time [{b.get('start_time')}, {b.get('end_time')}]: CPU IDLE")
        else:
            gantt_strs.append(
                f"  * Time [{b.get('start_time')}, {b.get('end_time')}]: Process {b.get('pid')} executing (duration {b.get('duration')})"
            )
    gantt_formatted = "\n".join(gantt_strs)

    metric_strs = []
    for pid, m in metrics.items():
        rt_part = f", Response Time (RT) = {m.get('response_time')}" if m.get("response_time") is not None else ""
        metric_strs.append(
            f"  * {pid}: Completion Time (CT) = {m.get('completion_time')}, "
            f"Turnaround Time (TAT = CT - AT) = {m.get('turnaround_time')}, "
            f"Waiting Time (WT = TAT - BT) = {m.get('waiting_time')}{rt_part}"
        )
    metrics_formatted = "\n".join(metric_strs)

    return f"""
### Authoritative FCFS CPU Scheduling Simulation & Trace Facts (MANDATORY VERIFIED TRACE)
The outline contains a deterministically verified FCFS CPU scheduling simulation. Stage 2 MUST faithfully visualize this exact execution without altering arrival times, changing queue order, preempting running processes, or fabricating numbers.

#### 1. Immutable Verified Facts (MUST PRESERVE EXACTLY)
- Processes & Problem Input Specifications (Authoritative Ground Truth):
{procs_formatted}
- Authoritative Gantt Chart Timeline (Total Time = {total_time}):
{gantt_formatted}
- Authoritative Per-Process Metrics:
{metrics_formatted}
- Summary Averages: Average Waiting Time = {avg_wt:.2f}, Average Turnaround Time = {avg_tat:.2f}, Average Response Time = {avg_rt:.2f}

#### 2. Pedagogical Organization & 12 Core Educational Beats (MANDATORY TEACHING PROGRESSION)
A genuinely good educational video MUST teach these 12 concepts in natural pedagogical progression:
1. **CPU Scheduling Motivation**: In multiprogramming, multiple processes in memory compete for a single CPU core. Illustrate this with a visual diagram showing a CPU core with tasks waiting and switching to maximize CPU utilization across CPU and I/O bursts.
2. **Process Control Block (PCB)**: An OS kernel data structure containing PID, Arrival Time (AT), Burst Time (BT), and current state. Use `ProcessCardVisual(pid="P1", arrival_time=0, burst_time=3, state="NEW")` in `dock_split_left` or `dock_main`.
3. **Process States & Lifecycle Transitions**: Explicitly trace the 4-state lifecycle (NEW -> READY -> RUNNING -> TERMINATED) and explain trigger conditions. Never duplicate consecutive state transitions.
4. **Process Input Specification (CRITICAL BEAT BEFORE EXECUTION)**:
   - MUST introduce the problem input data table/specification BEFORE any CPU dispatch or execution occurs:
     * P1: Arrival Time = 0, Burst Time = 3
     * P2: Arrival Time = 1, Burst Time = 4
     * P3: Arrival Time = 2, Burst Time = 2
   - Note: There is no dedicated 3-column table component in the registry. Display this input specification using `ProcessCardVisual` cards for P1, P2, P3 showing AT and BT, or procedural presentation. Do not assume `MetricTableVisual` has an input-only mode.
5. **Ready Queue**: A FIFO queue holding processes in READY state waiting for CPU. Use `ReadyQueueVisual(capacity=4)` in `dock_split_left`. Visibly distinguish HEAD (left, towards CPU) and TAIL (right, entrance).
6. **CPU Scheduler vs Dispatcher**: Decision vs execution flow:
   - Scheduler: Selects which process from the Ready Queue runs next based on scheduling policy.
   - Dispatcher: Performs the actual context switch and allocates the CPU core to the selected process.
7. **FCFS Selection & Non-Preemptive Execution**:
   - FCFS policy: The process that requests the CPU first gets the CPU first (earliest arrival time). Tie-breaking is stable arrival order.
   - Non-preemption rule: Once a process starts running on the CPU, it holds the CPU until completion. Arriving processes enter the Ready Queue; they NEVER interrupt or preempt the executing process.
   - Use `ReadyQueueVisual` in `dock_split_left` and `CPUExecutionVisual` in `dock_split_right`.
8. **Chronological Process Execution (COMPLETE P1, P2, P3 COVERAGE)**:
   - Every process (P1, P2, AND P3) MUST be explicitly dispatched, executed, and completed in animations:
     * Step 1: P1 arrives at t=0, dispatched to CPU, runs from t=0 to 3.
     * Step 2: P2 arrives at t=1 (waits in Ready Queue), P3 arrives at t=2 (waits in Ready Queue).
     * Step 3: P1 completes at t=3. P2 is dispatched from queue head, runs from t=3 to 7.
     * Step 4: P2 completes at t=7. P3 is dispatched from queue head, runs from t=7 to 9.
     * Step 5: P3 completes at t=9. System idle/terminated.
   - DO NOT stop after P1! P2 and P3 must both be visibly dispatched and executed to completion.
9. **Gantt Chart Construction**: Build the horizontal timeline bar step-by-step using `GanttChartVisual` as a primary visual, marking start, duration, and completion times for [0, 3], [3, 7], and [7, 9].
10. **Scheduling Metric Formulas (EXPLAIN FORMULAS EXPLICITLY)**:
    - Display and teach the mathematical formulas clearly in narration and visuals:
      * Completion Time (CT): Time when process finishes execution (P1=3, P2=7, P3=9).
      * Turnaround Time (TAT = CT - AT): Total elapsed time from arrival to completion (P1: 3-0=3, P2: 7-1=6, P3: 9-2=7).
      * Waiting Time (WT = TAT - BT): Time spent waiting in the ready queue (P1: 3-3=0, P2: 6-4=2, P3: 7-2=5).
      * Response Time (RT = First Run Time - AT): In non-preemptive FCFS, RT == WT (P1=0, P2=2, P3=5).
11. **Summary Averages Calculation**:
    - Use `MetricTableVisual` in `dock_main` to display all process metrics. Call `animate_show_averages()` to reveal averages with exact arithmetic:
      * Average CT = (3 + 7 + 9) / 3 = 19/3 = 6.33
      * Average TAT = (3 + 6 + 7) / 3 = 16/3 = 5.33
      * Average WT = (0 + 2 + 5) / 3 = 7/3 = 2.33
      * Average RT = (0 + 2 + 5) / 3 = 7/3 = 2.33
12. **FCFS Limitations & The Convoy Effect**:
    - Explain and visualize the Convoy Effect: a long CPU-bound process arriving first forces subsequent short interactive processes to wait, drastically inflating average waiting time and underutilizing I/O. Contrast with preemptive scheduling (e.g. Round Robin).

#### 3. Connected Component State Transitions (One Coherent Scheduling Event)
When a process is dispatched, your animations MUST communicate one synchronized scheduling transition:
- Ready Queue: `ready_queue.animate_dequeue()` (process leaves queue head, subsequent processes shift forward).
- CPU Core: `cpu_exec.animate_dispatch(pid=...)` (process enters execution dock, badge updates to EXECUTING: {pid}).
- Gantt Chart: `gantt_chart.animate_add_block(pid=..., start=..., end=...)` (timeline adds block and advances time marker).
- CPU Execution: `cpu_exec.animate_execute_tick()` for burst duration, then `cpu_exec.animate_complete(pid=...)`.
- Metric Update: When a process completes, update its row in the metric table: `metric_table.animate_update_process_metrics(pid=..., ct=..., tat=..., wt=...)`.

#### 4. Strict Component Method Contracts & Action Semantics
Specify ONLY valid, registered component methods. Do NOT invent unsupported methods:
- **`ProcessCardVisual`**:
  * Supported methods: `animate_set_state(new_state)` (valid states: "NEW", "READY", "RUNNING", "TERMINATED"), `animate_update_remaining_time(rem)`, `animate_highlight()`.
  * FORBIDDEN (DO NOT CALL): `highlight_pid_at()`, `highlight_bt()`, `highlight_state()`, `initialize_structure()`.
  * State transitions must follow: NEW -> READY -> RUNNING -> TERMINATED.
  * NEVER duplicate consecutive state transitions (e.g. do not call `set_state("RUNNING")` twice in a row).
- **`ReadyQueueVisual`**:
  * Supported methods: `animate_enqueue(item=...)`, `animate_dequeue()`.
  * Enqueue and dequeue must follow chronological order.
- **`CPUExecutionVisual`**:
  * Supported methods: `animate_dispatch(pid=...)`, `animate_execute_tick(remaining_time=...)`, `animate_complete(pid=...)`, `animate_idle()`.
  * FORBIDDEN (DO NOT CALL): `highlight_scheduler()`, `highlight_dispatcher()`, `highlight_context_switch()`.
  * NEVER dispatch the same process twice while it is running or after it completed.
- **`GanttChartVisual`**:
  * Supported methods: `animate_add_block(pid=..., start_time=..., end_time=...)`, `animate_advance_time(new_time=...)`.
- **`MetricTableVisual`**:
  * Supported methods: `animate_update_process_metrics(pid=..., ct=..., tat=..., wt=...)`, `animate_show_averages()`, `highlight_cell_avg_wt()`, `highlight_cell_avg_tat()`, `highlight_cell_avg_rt()`.

#### 4. Highlighting Discipline (NO UNNECESSARY PULSING)
- Animate with purpose: Use animations to show state changes, queue movements, and timeline progress.
- Highlight ONLY when directing attention to a specific concept being explained.
- DO NOT force an `Indicate` or `highlight` action onto every single narration line.
- Avoid consecutive repetitive highlights on the same component.

#### 5. Viewport Allocation & Layout Discipline
- **`MetricTableVisual`**: Belongs in `dock_main`. NEVER dock it to `dock_inspector` (which is only 1.0 unit high and causes extreme downscaling).
- **`GanttChartVisual`**: When constructing or explaining the timeline as the primary visual, place it in `dock_main` or ensure adequate viewport space.
- - **CRITICAL LAYOUT INVARIANT**: NEVER dock both `MetricTableVisual` AND `GanttChartVisual` to `dock_inspector` at the same time! `dock_inspector` is only 1.0 unit high; `MetricTableVisual` belongs in `dock_main`.
- Space out mathematical formula cards so they do not accumulate in a single cramped dock.
"""


class FCFSStoryboardFidelityVerifier(DomainFidelityVerifier):
    """Domain fidelity verifier for FCFS CPU scheduling storyboards."""

    def can_verify(self, model: Any, sections: List[Dict[str, Any]]) -> bool:
        return bool(
            getattr(model, "is_fcfs", False)
            or (getattr(model, "algo_state", {}) or {}).get("algorithm") == "fcfs"
        )

    def verify(self, model: Any, sections: List[Dict[str, Any]]) -> Any:
        from src.storyboard_fidelity_checker import StoryboardFidelityChecker
        return StoryboardFidelityChecker()._check_fcfs_fidelity(model, sections)


# Register FCFS domain guidance formatter and fidelity verifier
DomainTraceGuidanceRegistry.register("fcfs", format_fcfs_storyboard_guidance)
DomainFidelityRegistry.register("fcfs", FCFSStoryboardFidelityVerifier())

