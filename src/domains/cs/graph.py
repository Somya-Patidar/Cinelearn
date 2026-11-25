"""
Graph Visual Component
======================
Configurable, reusable visual component for arbitrary graph structures,
networks, and algorithms (e.g. Dijkstra, BFS, DFS, MST, Shortest Path).
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import networkx as nx
from manim import (
    VGroup,
    Circle,
    Text,
    Line,
    Arrow,
    Indicate,
    Transform,
    Animation,
    AnimationGroup,
    ShowPassingFlash,
    CurvedArrow,
)

from .. import (
    VisualComponent,
    VisualRegistry,
    DomainTraceGuidanceRegistry,
    DomainFidelityVerifier,
    DomainFidelityRegistry,
)


class VertexCollection(dict):
    """Hybrid collection allowing both dictionary lookup by vertex id (graph.vertices['A'])
    and integer indexing (graph.vertices[0]), matching Manim native Graph compatibility."""
    def __init__(self, keys, node_getter):
        super().__init__()
        self._keys = list(keys)
        for k in self._keys:
            self[k] = node_getter(k)

    def __getitem__(self, item):
        if isinstance(item, int) and item not in self:
            return self[self._keys[item]]
        return super().__getitem__(item)


class EdgeCollection(dict):
    """Hybrid collection allowing lookup by edge tuple (graph.edges[('A', 'B')])
    and integer indexing (graph.edges[0]), matching Manim native Graph compatibility."""
    def __init__(self, raw_edges, edge_getter):
        super().__init__()
        self._list = list(raw_edges)
        for e in self._list:
            if isinstance(e, (tuple, list)) and len(e) >= 2:
                u, v = e[0], e[1]
                mob = edge_getter(u, v)
                self[(u, v)] = mob
                self[(v, u)] = mob

    def __getitem__(self, item):
        if isinstance(item, int) and item not in self:
            return self._list[item]
        return super().__getitem__(item)


@VisualRegistry.register("cs", "GraphVisual")
class GraphVisual(VisualComponent):
    """
    Configurable and reusable visual component for graphs and networks.
    Supports directed/undirected graphs, weighted edges, custom layouts,
    and semantic animation methods for path traversal, node visitation,
    and edge relaxation.
    """
    component_id = "GraphVisual"
    domain = "cs"
    aliases = ["graph", "network", "node graph", "dijkstra graph"]
    preferred_aspect = 1.35
    suggested_viewport = "dock_main"
    supported_handles = ["vertex_<id>", "edge_<u_v>", "weight_<u_v>"]
    action_weights = {
        "animate_relax_edge": 1.5,
        "animate_visit_node": 1.0,
        "animate_traverse_edge": 1.0,
        "animate_highlight_node": 0.6,
    }

    def __init__(
        self,
        vertices: Sequence[Any],
        edges: Sequence[Union[Tuple[Any, Any], Tuple[Any, Any, Any], Dict[str, Any]]],
        layout: Union[str, Dict[Any, Sequence[float]]] = "spring",
        directed: bool = False,
        node_radius: float = 0.36,
        node_color: str = "#58C4DD",
        node_fill_opacity: float = 0.25,
        node_stroke_width: float = 3.0,
        edge_color: str = "#83C167",
        edge_width: float = 3.0,
        text_color: str = "#FFFFFF",
        font_size: int = 18,
        layout_scale: float = 2.0,
        root_vertex: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vertices_list = list(vertices)
        self.raw_edges = list(edges)
        self.directed = directed
        self.node_radius = node_radius
        self.node_color = node_color
        self.node_fill_opacity = node_fill_opacity
        self.node_stroke_width = node_stroke_width
        self.edge_color = edge_color
        self.edge_width = edge_width
        self.text_color = text_color
        self.font_size = font_size
        self.layout_scale = layout_scale
        self.root_vertex = root_vertex

        self.positions: Dict[Any, np.ndarray] = self._compute_layout(layout)

        # Groups for z-ordering: edges behind nodes
        self.edge_group = VGroup()
        self.node_group = VGroup()
        self.weight_group = VGroup()

        self._build_nodes()
        self._build_edges()

        # Assemble into this VGroup: edges first (behind), then weights, then nodes (in front)
        self.add(self.edge_group, self.weight_group, self.node_group)
        self.center()

        # Hybrid accessors for full compatibility with Manim native Graph APIs
        self.vertices = VertexCollection(
            self.vertices_list,
            lambda v: self.get_semantic_handle(f"node_{str(v)}") or self.get_semantic_handle(f"circle_{str(v)}")
        )
        self.edges = EdgeCollection(
            self.raw_edges,
            lambda u, v: self.get_semantic_handle(f"edge_{u}_{v}") or self.get_semantic_handle(f"edge_{v}_{u}")
        )

    def _compute_layout(self, layout: Union[str, Dict[Any, Sequence[float]]]) -> Dict[Any, np.ndarray]:
        """Calculates 3D coordinates for all vertices according to layout strategy."""
        num_v = len(self.vertices_list)
        if num_v == 0:
            return {}

        # 1. Explicit coordinate mapping
        if isinstance(layout, dict):
            positions = {}
            for v in self.vertices_list:
                if v in layout:
                    coords = layout[v]
                    if len(coords) == 2:
                        positions[v] = np.array([float(coords[0]), float(coords[1]), 0.0])
                    else:
                        positions[v] = np.array([float(coords[0]), float(coords[1]), float(coords[2])])
                else:
                    positions[v] = np.array([0.0, 0.0, 0.0])
            return positions

        layout_type = str(layout).lower().strip()

        # 2. Circular layout
        if layout_type == "circular" or num_v <= 2:
            positions = {}
            radius = float(self.layout_scale)
            for i, v in enumerate(self.vertices_list):
                angle = (np.pi / 2.0) - (2.0 * np.pi * i / max(1, num_v))
                positions[v] = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            return positions

        # 3. Tree / Hierarchical layout
        if layout_type == "tree":
            return self._compute_tree_layout()

        # 4. Grid layout
        if layout_type == "grid":
            positions = {}
            cols = int(np.ceil(np.sqrt(num_v)))
            rows = int(np.ceil(num_v / cols))
            spacing_x = float(self.layout_scale) * 2.0 / max(1, cols)
            spacing_y = float(self.layout_scale) * 1.5 / max(1, rows)
            for i, v in enumerate(self.vertices_list):
                r = i // cols
                c = i % cols
                x = (c - (cols - 1) / 2.0) * spacing_x
                y = ((rows - 1) / 2.0 - r) * spacing_y
                positions[v] = np.array([x, y, 0.0])
            return positions

        # 5. Default: Force-directed Spring layout via networkx
        try:
            G = nx.DiGraph() if self.directed else nx.Graph()
            G.add_nodes_from(self.vertices_list)
            for edge in self.raw_edges:
                u, v, _ = self._parse_edge(edge)
                if u in self.vertices_list and v in self.vertices_list:
                    G.add_edge(u, v)

            nx_pos = nx.spring_layout(G, seed=42, scale=float(self.layout_scale))
            positions = {}
            for v in self.vertices_list:
                pt = nx_pos.get(v, np.array([0.0, 0.0]))
                positions[v] = np.array([float(pt[0]), float(pt[1]), 0.0])
            return positions
        except Exception:
            # Safe circular fallback
            positions = {}
            radius = float(self.layout_scale)
            for i, v in enumerate(self.vertices_list):
                angle = (np.pi / 2.0) - (2.0 * np.pi * i / max(1, num_v))
                positions[v] = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            return positions

    def _compute_tree_layout(self) -> Dict[Any, np.ndarray]:
        """Hierarchical BFS level-order tree layout."""
        num_v = len(self.vertices_list)
        root = self.root_vertex if (self.root_vertex in self.vertices_list) else self.vertices_list[0]

        adj: Dict[Any, List[Any]] = {v: [] for v in self.vertices_list}
        for edge in self.raw_edges:
            u, v, _ = self._parse_edge(edge)
            if u in adj and v in adj:
                adj[u].append(v)
                if not self.directed:
                    adj[v].append(u)

        # BFS for levels
        visited = {root}
        queue = [(root, 0)]
        levels: Dict[int, List[Any]] = {}

        while queue:
            curr, depth = queue.pop(0)
            levels.setdefault(depth, []).append(curr)
            for nxt in adj.get(curr, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))

        # Add any unreached nodes to last level
        for v in self.vertices_list:
            if v not in visited:
                max_d = max(levels.keys()) if levels else 0
                levels.setdefault(max_d + 1, []).append(v)

        max_depth = max(levels.keys()) if levels else 0
        depth_spacing = (float(self.layout_scale) * 1.8) / max(1, max_depth)
        positions = {}

        for depth, nodes in levels.items():
            y = (max_depth / 2.0 - depth) * depth_spacing
            n_nodes = len(nodes)
            width_span = float(self.layout_scale) * 2.2
            spacing_x = width_span / max(1, n_nodes)
            for idx, node in enumerate(nodes):
                x = (idx - (n_nodes - 1) / 2.0) * spacing_x
                positions[node] = np.array([x, y, 0.0])

        return positions

    def _parse_edge(self, edge: Any) -> Tuple[Any, Any, Optional[Any]]:
        """Parses edge representation into (u, v, weight)."""
        if isinstance(edge, (tuple, list)):
            if len(edge) >= 3:
                return edge[0], edge[1], edge[2]
            elif len(edge) == 2:
                return edge[0], edge[1], None
        elif isinstance(edge, dict):
            return edge.get("u"), edge.get("v"), edge.get("weight")
        return None, None, None

    def _build_nodes(self) -> None:
        """Constructs Manim circle and text label for each node."""
        for v in self.vertices_list:
            pos = self.positions.get(v, np.array([0.0, 0.0, 0.0]))

            circle = Circle(
                radius=self.node_radius,
                color=self.node_color,
                fill_color=self.node_color,
                fill_opacity=self.node_fill_opacity,
                stroke_width=self.node_stroke_width,
            )
            circle.move_to(pos)

            label = Text(str(v), font_size=self.font_size, color=self.text_color)
            if label.width > self.node_radius * 1.55:
                label.scale_to_fit_width(self.node_radius * 1.55)
            label.move_to(circle.get_center())

            node_group = VGroup(circle, label)
            self.node_group.add(node_group)

            # Register semantic handles
            v_key = str(v)
            self.register_semantic_handle(f"node_{v_key}", node_group)
            self.register_semantic_handle(f"circle_{v_key}", circle)
            self.register_semantic_handle(f"label_{v_key}", label)

    def _build_edges(self) -> None:
        """Constructs Manim lines/arrows and weight labels between nodes."""
        for edge_spec in self.raw_edges:
            u, v, weight = self._parse_edge(edge_spec)
            if u is None or v is None or u not in self.positions or v not in self.positions:
                continue

            pos_u = self.positions[u]
            pos_v = self.positions[v]
            diff = pos_v - pos_u
            dist = float(np.linalg.norm(diff))

            u_key = str(u)
            v_key = str(v)

            # Self-loop
            if u == v or dist < 1e-4:
                loop = CurvedArrow(
                    start_point=pos_u + np.array([-self.node_radius * 0.7, self.node_radius * 0.7, 0.0]),
                    end_point=pos_u + np.array([self.node_radius * 0.7, self.node_radius * 0.7, 0.0]),
                    angle=-3.5,
                    color=self.edge_color,
                    stroke_width=self.edge_width,
                )
                self.edge_group.add(loop)
                self.register_semantic_handle(f"edge_{u_key}_{v_key}", loop)
                continue

            unit_vec = diff / dist
            start_pt = pos_u + unit_vec * self.node_radius
            end_pt = pos_v - unit_vec * self.node_radius

            if self.directed:
                edge_mob = Arrow(
                    start=start_pt,
                    end=end_pt,
                    color=self.edge_color,
                    buff=0.0,
                    stroke_width=self.edge_width,
                    max_tip_length_to_length_ratio=0.22,
                )
            else:
                edge_mob = Line(
                    start=start_pt,
                    end=end_pt,
                    color=self.edge_color,
                    stroke_width=self.edge_width,
                )

            self.edge_group.add(edge_mob)
            self.register_semantic_handle(f"edge_{u_key}_{v_key}", edge_mob)
            if not self.directed:
                self.register_semantic_handle(f"edge_{v_key}_{u_key}", edge_mob)

            # Edge weight label
            if weight is not None:
                mid_pt = (start_pt + end_pt) / 2.0
                normal = np.array([-unit_vec[1], unit_vec[0], 0.0])
                weight_pos = mid_pt + normal * (0.16 + self.node_radius * 0.3)

                w_txt = Text(
                    str(weight),
                    font_size=max(12, int(self.font_size * 0.8)),
                    color=self.text_color,
                )
                w_txt.move_to(weight_pos)
                self.weight_group.add(w_txt)
                self.register_semantic_handle(f"weight_{u_key}_{v_key}", w_txt)
                if not self.directed:
                    self.register_semantic_handle(f"weight_{v_key}_{u_key}", w_txt)

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------
    def animate_highlight_node(
        self,
        node_id: Any,
        color: str = "#FFD700",
        scale: float = 1.15,
        run_time: float = 0.6,
    ) -> Animation:
        """Indicates/highlights the target node."""
        target = self.get_semantic_handle(f"node_{node_id}")
        if target is None:
            return Indicate(self, color=color, run_time=run_time)
        return Indicate(target, color=color, scale_factor=scale, run_time=run_time)

    def animate_select_node(
        self,
        node_id: Any,
        stroke_color: str = "#FFD700",
        stroke_width: float = 5.0,
        run_time: float = 0.5,
    ) -> Animation:
        """Permanently highlights the border of a selected node."""
        circle = self.get_semantic_handle(f"circle_{node_id}")
        if circle is None:
            return Indicate(self, color=stroke_color, run_time=run_time)
        return circle.animate(run_time=run_time).set_stroke(color=stroke_color, width=stroke_width)

    def animate_visit_node(
        self,
        node_id: Any,
        fill_color: str = "#83C167",
        fill_opacity: float = 0.55,
        run_time: float = 0.6,
    ) -> Animation:
        """Marks a node as visited/finalized by coloring its fill area."""
        circle = self.get_semantic_handle(f"circle_{node_id}")
        if circle is None:
            return Indicate(self, color=fill_color, run_time=run_time)
        return circle.animate(run_time=run_time).set_fill(fill_color, opacity=fill_opacity)

    def animate_highlight_edge(
        self,
        u: Any,
        v: Any,
        color: str = "#FFD700",
        stroke_width: float = 5.0,
        run_time: float = 0.6,
    ) -> Animation:
        """Highlights the edge connecting u and v."""
        edge = self.get_semantic_handle(f"edge_{u}_{v}")
        if edge is None:
            edge = self.get_semantic_handle(f"edge_{v}_{u}")
        if edge is None:
            return Indicate(self, color=color, run_time=run_time)
        return Indicate(edge, color=color, run_time=run_time)

    def animate_traverse_edge(
        self,
        u: Any,
        v: Any,
        color: str = "#FFD700",
        run_time: float = 0.8,
    ) -> Animation:
        """Animates a passing light flash traversing the edge from u to v."""
        edge = self.get_semantic_handle(f"edge_{u}_{v}")
        if edge is None:
            edge = self.get_semantic_handle(f"edge_{v}_{u}")
        if edge is None:
            return Indicate(self, color=color, run_time=run_time)
        edge_flash = edge.copy().set_color(color).set_stroke(width=self.edge_width * 1.6)
        return ShowPassingFlash(edge_flash, time_width=0.4, run_time=run_time)

    def animate_update_node_label(
        self,
        node_id: Any,
        new_label: Any,
        color: Optional[str] = None,
        run_time: float = 0.6,
    ) -> Animation:
        """Updates the text label inside a node (e.g. updating tentative distance in Dijkstra)."""
        old_label = self.get_semantic_handle(f"label_{node_id}")
        circle = self.get_semantic_handle(f"circle_{node_id}")
        txt_color = color or self.text_color
        if old_label is None or circle is None:
            return Indicate(self, run_time=run_time)

        new_label_mob = Text(str(new_label), font_size=self.font_size, color=txt_color)
        if new_label_mob.width > self.node_radius * 1.55:
            new_label_mob.scale_to_fit_width(self.node_radius * 1.55)
        new_label_mob.move_to(circle.get_center())

        # Update handle reference
        self._semantic_handles[f"label_{node_id}"] = new_label_mob
        return Transform(old_label, new_label_mob, run_time=run_time)

    def animate_update_edge_weight(
        self,
        u: Any,
        v: Any,
        new_weight: Any,
        color: Optional[str] = None,
        run_time: float = 0.6,
    ) -> Animation:
        """Updates the weight label for edge (u, v)."""
        old_w = self.get_semantic_handle(f"weight_{u}_{v}")
        if old_w is None:
            old_w = self.get_semantic_handle(f"weight_{v}_{u}")
        if old_w is None:
            return Indicate(self, run_time=run_time)

        txt_color = color or self.text_color
        new_w_mob = Text(
            str(new_weight),
            font_size=max(12, int(self.font_size * 0.8)),
            color=txt_color,
        )
        new_w_mob.move_to(old_w.get_center())
        self._semantic_handles[f"weight_{u}_{v}"] = new_w_mob
        return Transform(old_w, new_w_mob, run_time=run_time)

    def animate_relax_edge(
        self,
        u: Any,
        v: Any,
        new_v_label: Optional[Any] = None,
        edge_color: str = "#FFD700",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Compound animation relaxing edge (u, v) and updating destination node's label."""
        anims = [self.animate_highlight_edge(u, v, color=edge_color, run_time=run_time * 0.6)]
        if new_v_label is not None:
            anims.append(
                self.animate_update_node_label(
                    v,
                    new_v_label,
                    color=edge_color,
                    run_time=run_time * 0.7,
                )
            )
        return AnimationGroup(*anims, lag_ratio=0.35)


# ============================================================================
# Domain Decoupling Registrations (Dijkstra Guidance & Fidelity Verifier)
# ============================================================================

def format_dijkstra_storyboard_guidance(algo_state: Dict[str, Any]) -> str:
    """Formats verified Dijkstra trace into prompt guidance for Stage 2 Storyboards."""
    source = algo_state.get("source", "A")
    vertices = algo_state.get("vertices", [])
    edges = algo_state.get("edges", [])
    directed = algo_state.get("directed", False)
    settled_order = algo_state.get("settled_order", [])
    final_distances = algo_state.get("final_distances", {})
    shortest_paths = algo_state.get("shortest_paths", {})
    canonical_trace = algo_state.get("canonical_trace", [])

    edge_strs = [f"({e.get('u')}, {e.get('v')}, w={e.get('weight')})" for e in edges]
    edges_formatted = ", ".join(edge_strs)

    trace_lines = []
    for step in canonical_trace:
        stype = step.get("step_type", "iteration")
        if stype == "initialization":
            dists = ", ".join(f"{k}: {v}" for k, v in step.get("tentative_distances", {}).items())
            queue = ", ".join(step.get("remaining_queue", []))
            trace_lines.append(
                f"  * Beat 1 (Setup): Source node '{source}', Initial Tentative Distances = {{{dists}}}, Unsettled Queue = [{queue}]"
            )
        else:
            ext_node = step.get("extracted_node")
            ext_dist = step.get("extracted_dist")
            relax_strs = []
            for r in step.get("relaxations", []):
                u, v, w = r.get("u"), r.get("v"), r.get("weight")
                prev_d = r.get("previous_dist")
                cand_d = r.get("candidate_dist")
                upd = r.get("is_updated")
                prev_str = "inf" if prev_d == "inf" or (isinstance(prev_d, float) and np.isinf(prev_d)) else prev_d
                status_str = f"UPDATED to {cand_d}" if upd else f"NO UPDATE (candidate {cand_d} >= current {prev_str})"
                relax_strs.append(f"examine ({u}, {v}, w={w}): previous={prev_str}, candidate={cand_d} -> {status_str}")
            relax_formatted = "; ".join(relax_strs) if relax_strs else "no unvisited outgoing edges"
            dists = ", ".join(f"{k}: {v}" for k, v in step.get("tentative_distances", {}).items())
            trace_lines.append(
                f"  * Extraction Beat (Node '{ext_node}', dist={ext_dist}):\n"
                f"      - Outgoing Relaxations: {relax_formatted}\n"
                f"      - Updated Distances: {{{dists}}}\n"
                f"      - Settled Nodes: {step.get('settled_nodes', [])}"
            )
    trace_str = "\n".join(trace_lines)

    final_dists_str = ", ".join(f"{k}: {v}" for k, v in final_distances.items())
    paths_str = ", ".join(f"{k}: {' -> '.join(p)}" for k, p in shortest_paths.items())

    return f"""
### Authoritative Dijkstra Worked Example Execution (MANDATORY VERIFIED TRACE)
The outline contains a deterministically verified Dijkstra simulation on graph G. Stage 2 MUST faithfully visualize this exact execution without omitting steps, altering numbers, or skipping intermediate states.

#### 1. Immutable Verified Facts (MUST PRESERVE EXACTLY)
- Graph Topology: Vertices V = {vertices}, Source s = '{source}', Directed = {directed}
- Edge Weights: [{edges_formatted}]
- Extraction Order: {settled_order}
- Canonical Trace Steps ({len(canonical_trace)} steps):
{trace_str}
- Authoritative Final Distances: {{{final_dists_str}}}
- Shortest Paths: {{{paths_str}}}

#### 2. Required Teaching Sequence (Beats to Cover in Worked Example)
1. Initial graph, source node '{source}', initial distances (dist[{source}]=0, others=inf), and initial unsettled set.
2. First minimum-distance extraction (extract node '{settled_order[0]}' with distance 0).
3. Examination of outgoing edges and relaxation decisions.
4. Distance updates, explicitly showing previous distance, candidate distance, and whether an update occurred.
5. Updated graph/table/queue state following relaxation.
6. Subsequent extractions and relaxations ({', '.join(settled_order[1:])}), repeated until all reachable nodes are settled.
7. Final shortest distances ({{{final_dists_str}}}) and predecessor/shortest-path interpretation.

#### 3. Visual Progression & Action Requirements (Visible State Transitions - NOT Just Narration)
The storyboard MUST describe visible state transitions in 'animations', NOT merely spoken narration about them:
- Highlight Extracted Node: 'action: extract_min(\'{source}\')' or 'action: select_node(\'{source}\')' / 'action: highlight_node(\'{source}\')'.
- Highlight Examined Edge: 'action: highlight_edge(\'u\', \'v\')' or 'action: traverse_edge(\'u\', \'v\')'.
- Visible Distance Update (when relaxation improves distance):
  * 'action: relax_edge(u=\'u\', v=\'v\', candidate=D, updated=True, new_dist=D)' OR 'action: update_node_label(\'v\', D)' / 'action: update_distance(\'v\', D)'.
- Visible Unsuccessful Relaxation (candidate >= current distance):
  * 'action: relax_edge(u=\'u\', v=\'v\', candidate=C, updated=False)' OR 'action: no_update(\'v\')' / 'action: keep_distance(\'v\')'.
- Mark Settled Nodes: 'action: visit_node(\'u\')' or 'action: mark_settled(\'u\')'.
- Update Distance Table & Queue: update table entries visibly in 'dock_inspector'.
- Final Shortest Path Tree: 'action: highlight_path(...)'.
"""


class DijkstraStoryboardFidelityVerifier(DomainFidelityVerifier):
    """Domain fidelity verifier for Dijkstra storyboards."""

    def can_verify(self, model: Any, sections: List[Dict[str, Any]]) -> bool:
        return bool(
            getattr(model, "is_dijkstra", False)
            or (getattr(model, "algo_state", {}) or {}).get("algorithm") == "dijkstra"
        )

    def verify(self, model: Any, sections: List[Dict[str, Any]]) -> Any:
        from src.storyboard_fidelity_checker import StoryboardFidelityChecker
        return StoryboardFidelityChecker()._check_dijkstra_fidelity(model, sections)


# Register Dijkstra domain guidance formatter and fidelity verifier
DomainTraceGuidanceRegistry.register("dijkstra", format_dijkstra_storyboard_guidance)
DomainFidelityRegistry.register("dijkstra", DijkstraStoryboardFidelityVerifier())

