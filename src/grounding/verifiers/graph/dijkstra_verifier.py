"""
Deterministic Dijkstra Trace Verifier for CineLearn (Milestone 2B.6A).

Provides independent, deterministic ground-truth simulation of Dijkstra's
shortest-path algorithm, validating step-by-step node extractions, distance
relaxations, settled-node sets, priority-queue states, and final shortest distances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from grounding.trace_engine import StateSnapshot, ComputationTrace
except ImportError:
    from src.grounding.trace_engine import StateSnapshot, ComputationTrace
import copy

from numerical_evaluator import (
    DiagnosticCode,
    VerificationStatus,
)
from worked_example_verifier import (
    SpecializedVerifier,
    SpecializedVerifierRegistry,
    WorkedExampleReport,
)


# ============================================================================
# Algorithmic Trace Data Structures
# ============================================================================

@dataclass
class RelaxationDetail:
    """Represents candidate edge relaxation during an extract-min step."""
    edge: Tuple[str, str]
    weight: float
    previous_dist: float
    candidate_dist: float
    updated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge": list(self.edge),
            "weight": self.weight,
            "previous_dist": "inf" if math.isinf(self.previous_dist) else self.previous_dist,
            "candidate_dist": "inf" if math.isinf(self.candidate_dist) else self.candidate_dist,
            "updated": self.updated,
        }


@dataclass
class DijkstraStepState:
    """Ground-truth canonical state of Dijkstra's algorithm at a single step."""
    step_index: int
    step_type: str  # "initialization" | "iteration" | "stale_skip"
    extracted_node: Optional[str]
    extracted_dist: Optional[float]
    relaxations: List[RelaxationDetail] = field(default_factory=list)
    tentative_distances: Dict[str, float] = field(default_factory=dict)
    settled_nodes: List[str] = field(default_factory=list)
    remaining_queue: List[str] = field(default_factory=list)
    predecessors: Dict[str, Optional[str]] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_type": self.step_type,
            "extracted_node": self.extracted_node,
            "extracted_dist": (
                "inf" if (self.extracted_dist is not None and math.isinf(self.extracted_dist))
                else self.extracted_dist
            ),
            "relaxations": [r.to_dict() for r in self.relaxations],
            "tentative_distances": {
                k: ("inf" if math.isinf(v) else v)
                for k, v in self.tentative_distances.items()
            },
            "settled_nodes": list(self.settled_nodes),
            "remaining_queue": list(self.remaining_queue),
            "predecessors": dict(self.predecessors),
            "description": self.description,
        }

    def to_state_snapshot(self) -> StateSnapshot:
        """Converts step state to generic M2.1 StateSnapshot representation."""
        st: Dict[str, Any] = {
            "step_index": self.step_index,
            "step_type": self.step_type,
            "extracted_node": self.extracted_node,
            "extracted_dist": (
                "inf" if (self.extracted_dist is not None and math.isinf(self.extracted_dist))
                else self.extracted_dist
            ),
            "relaxations": [r.to_dict() for r in self.relaxations],
            "tentative_distances": {
                k: ("inf" if math.isinf(v) else v)
                for k, v in self.tentative_distances.items()
            },
            "settled_nodes": list(self.settled_nodes),
            "remaining_queue": list(self.remaining_queue),
            "predecessors": dict(self.predecessors),
            "description": self.description,
        }
        for k, v in self.tentative_distances.items():
            st[f"dist_{k}"] = "inf" if math.isinf(v) else (int(v) if v.is_integer() else v)
            st[f"settled_{k}"] = (k in self.settled_nodes)
        if self.extracted_node is not None:
            st["current_node"] = self.extracted_node
            if self.extracted_dist is not None:
                ed = self.extracted_dist
                st[f"dist_{self.extracted_node}"] = "inf" if math.isinf(ed) else (int(ed) if ed.is_integer() else ed)

        return StateSnapshot(
            step_index=self.step_index,
            state=st,
            description=self.description,
            step_type=f"DIJKSTRA_{self.step_type.upper()}",
            metadata={"active_variables": ["extracted_node", "settled_nodes", "tentative_distances"]},
        )


@dataclass
class DijkstraStepVerification:
    """Outcome of verifying a single claimed step against ground-truth simulation."""
    step_index: int
    status: VerificationStatus
    diagnostic: DiagnosticCode
    claimed_step: str
    claimed_result: str
    expected_state: Optional[DijkstraStepState]
    extracted_node_match: bool
    distances_match: bool
    settled_match: bool
    queue_match: bool
    discrepancies: List[str] = field(default_factory=list)
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "status": self.status.value,
            "diagnostic": self.diagnostic.value,
            "claimed_step": self.claimed_step,
            "claimed_result": self.claimed_result,
            "extracted_node_match": self.extracted_node_match,
            "distances_match": self.distances_match,
            "settled_match": self.settled_match,
            "queue_match": self.queue_match,
            "discrepancies": self.discrepancies,
            "details": self.details,
            "expected_state": self.expected_state.to_dict() if self.expected_state else None,
        }


@dataclass
class DijkstraTraceReport:
    """Overall outcome of validating a complete Dijkstra worked-example trace."""
    overall_status: VerificationStatus
    total_steps: int
    verified_steps: int
    failed_steps: int
    review_steps: int
    not_applicable_steps: int
    expected_trace: List[DijkstraStepState] = field(default_factory=list)
    step_verifications: List[DijkstraStepVerification] = field(default_factory=list)
    final_distances: Dict[str, float] = field(default_factory=dict)
    settled_nodes: List[str] = field(default_factory=list)
    predecessors: Dict[str, Optional[str]] = field(default_factory=dict)
    shortest_paths: Dict[str, List[str]] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""
    vertices: List[str] = field(default_factory=list)
    directed: bool = False
    discrepancies: List[str] = field(default_factory=list)
    is_safe_for_stage2: bool = True
    final_result_status: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    summary: str = ""
    computation_trace: Optional[ComputationTrace] = None

    def to_computation_trace(self) -> ComputationTrace:
        """Converts expected trace to an authoritative M2.1 ComputationTrace."""
        snaps = [s.to_state_snapshot() for s in self.expected_trace]
        return ComputationTrace(
            algorithm_or_domain="dijkstra_shortest_path",
            metadata={
                "source": self.source,
                "vertices": list(self.vertices),
                "directed": self.directed,
                "edges": list(self.edges),
            },
            snapshots=snaps,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "total_steps": self.total_steps,
            "verified_steps": self.verified_steps,
            "failed_steps": self.failed_steps,
            "review_steps": self.review_steps,
            "not_applicable_steps": self.not_applicable_steps,
            "final_result_status": self.final_result_status.value,
            "is_safe_for_stage2": self.is_safe_for_stage2,
            "summary": self.summary,
            "discrepancies": self.discrepancies,
            "source": self.source,
            "vertices": self.vertices,
            "directed": self.directed,
            "edges": self.edges,
            "final_distances": {
                k: ("inf" if math.isinf(v) else v)
                for k, v in self.final_distances.items()
            },
            "settled_nodes": list(self.settled_nodes),
            "predecessors": dict(self.predecessors),
            "shortest_paths": dict(self.shortest_paths),
            "step_diagnostics": [s.to_dict() for s in self.step_verifications],
            "expected_trace": [s.to_dict() for s in self.expected_trace],
        }


# ============================================================================
# Deterministic Dijkstra Trace Verifier Engine
# ============================================================================

class DijkstraTraceVerifier(SpecializedVerifier):
    """
    Simulates Dijkstra's algorithm from ground truth and verifies step-by-step
    algorithmic traces in worked examples.

    Tie-breaking rule:
        Lexicographical order on vertex labels:
        min(unsettled, key=lambda v: (distances[v], v))
        When tentative distances are equal, the vertex with the lexicographically
        smaller string identifier is extracted first.

    Stale queue entries:
        In priority-queue implementations where relaxed keys are added without
        updating existing elements in-place, stale entries (where the vertex is
        already settled) are skipped and do not perform relaxations.
    """

    verifier_id: str = "dijkstra"

    @classmethod
    def can_handle(cls, spec: Any, topic: str = "") -> bool:
        """Alias for can_verify to ensure full contract compatibility."""
        return cls.can_verify(spec, topic)

    @classmethod
    def can_verify(cls, spec: Dict[str, Any], topic: str = "") -> bool:
        """
        Determines whether the given worked_example_spec / topic represents a Dijkstra problem.
        Guarantees:
        - Unrelated graph algorithms (BFS, DFS, Bellman-Ford, Kruskal, Prim, Topological Sort) are NOT claimed.
        - Non-graph domains (Knapsack, FCFS, Binary Search) are NOT claimed.
        - Ambiguous specifications without positive Dijkstra indicators are safely rejected.
        """
        if not isinstance(spec, dict):
            return False

        topic_lower = topic.strip().lower()
        inputs = spec.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}

        algo = str(spec.get("algorithm") or inputs.get("algorithm") or spec.get("domain", "")).strip().lower()

        # 1. Reject explicit non-Dijkstra algorithms
        non_dijkstra_algos = {
            "bfs", "breadth_first_search", "breadth_first",
            "dfs", "depth_first_search", "depth_first",
            "bellman_ford", "bellman-ford", "floyd_warshall", "floyd-warshall",
            "kruskal", "prim", "mst", "minimum_spanning_tree",
            "topological_sort", "toposort",
            "a_star", "astar",
            "ford_fulkerson", "bipartite", "cycle_detection",
        }
        if algo in non_dijkstra_algos:
            return False

        # If topic clearly specifies an unrelated graph algorithm without mentioning Dijkstra
        unrelated_topic_keywords = [
            "breadth first", "breadth-first", "depth first", "depth-first",
            "bellman-ford", "bellman ford", "floyd-warshall", "floyd warshall",
            "kruskal", "prim's", "prims", "minimum spanning tree",
            "topological sort", "toposort", "a* algorithm", "a-star",
        ]
        if "dijkstra" not in topic_lower:
            if any(kw in topic_lower for kw in unrelated_topic_keywords):
                return False

        # 2. Reject non-graph domains
        if "capacity" in spec or "capacity" in inputs:
            return False
        if "processes" in spec or "processes" in inputs or "os_scheduling" in algo:
            return False
        if "array" in spec and "target" in spec:
            return False

        # 3. Positive Dijkstra indicators
        if "dijkstra" in topic_lower:
            return True
        if algo in {"dijkstra", "dijkstra_shortest_path", "shortest_path_dijkstra"}:
            return True
        setup = str(spec.get("setup", "")).lower()
        if "dijkstra" in setup:
            return True
        if "dijkstra" in str(inputs).lower():
            return True

        # Explicit graph definition with vertices/edges and starting node,
        # but only if no conflicting algorithm or domain was specified
        if ("vertices" in inputs or "edges" in inputs) and ("s" in inputs or "source" in inputs):
            return True
        if re.search(r"\b(?:nodes|vertices)\b", setup, re.IGNORECASE) and re.search(r"\b(?:start|source)\b", setup, re.IGNORECASE):
            if re.search(r"\b[A-Za-z0-9_]+\s*[-—–]\s*[A-Za-z0-9_]+\s*\(", setup):
                return True

        return False

    @classmethod
    def simulate(
        cls,
        vertices: List[str],
        edges: List[Tuple[str, str, float]],
        source: str,
        directed: bool = False,
    ) -> Tuple[
        List[DijkstraStepState],
        Dict[str, float],
        List[str],
        Dict[str, Optional[str]],
        Dict[str, List[str]],
    ]:
        """
        Executes deterministic, independent simulation of Dijkstra's algorithm.

        Returns:
            (expected_trace, final_distances, settled_order, predecessors, shortest_paths)
        Raises:
            ValueError on invalid input (negative weight, unknown vertex, invalid source).
        """
        if not vertices:
            raise ValueError("Vertex set cannot be empty.")

        vertex_set = set(str(v).strip() for v in vertices)
        cleaned_vertices = sorted(list(vertex_set))

        clean_source = str(source).strip()
        if clean_source not in vertex_set:
            raise ValueError(f"Source vertex '{clean_source}' does not exist in vertices: {cleaned_vertices}")

        # Build adjacency map and validate edges, consolidating parallel edges to minimum weight
        adj: Dict[str, List[Tuple[str, float]]] = {v: [] for v in cleaned_vertices}
        edge_map: Dict[Tuple[str, str], float] = {}
        for edge in edges:
            if len(edge) != 3:
                raise ValueError(f"Malformed edge tuple: {edge} (must be (u, v, weight))")
            u, v, w = str(edge[0]).strip(), str(edge[1]).strip(), float(edge[2])
            if u not in vertex_set:
                raise ValueError(f"Edge {edge} references unknown vertex '{u}'")
            if v not in vertex_set:
                raise ValueError(f"Edge {edge} references unknown vertex '{v}'")
            if not math.isfinite(w):
                raise ValueError(f"Edge weight must be finite: {w} on edge ({u}, {v})")
            if w < 0:
                raise ValueError(f"Dijkstra does not support negative edge weights: ({u}, {v}, {w})")

            if (u, v) not in edge_map or w < edge_map[(u, v)]:
                edge_map[(u, v)] = w
            if not directed and u != v:
                if (v, u) not in edge_map or w < edge_map[(v, u)]:
                    edge_map[(v, u)] = w

        for (u, v), w in edge_map.items():
            adj[u].append((v, w))

        # Sort outgoing edges by neighbor name for deterministic iteration order
        for u in adj:
            adj[u].sort(key=lambda item: item[0])

        distances: Dict[str, float] = {v: (0.0 if v == clean_source else float("inf")) for v in cleaned_vertices}
        predecessors: Dict[str, Optional[str]] = {v: None for v in cleaned_vertices}
        settled: Set[str] = set()
        settled_order: List[str] = []
        expected_trace: List[DijkstraStepState] = []

        # Step 0: Initialization
        step_0 = DijkstraStepState(
            step_index=0,
            step_type="initialization",
            extracted_node=None,
            extracted_dist=None,
            relaxations=[],
            tentative_distances=dict(distances),
            settled_nodes=[],
            remaining_queue=list(cleaned_vertices),
            predecessors=dict(predecessors),
            description=(
                f"Initialize dist[{clean_source}] = 0, all other distances = ∞, "
                f"Q = {{{', '.join(cleaned_vertices)}}}"
            ),
        )
        expected_trace.append(step_0)

        # Main algorithm loop
        while len(settled) < len(cleaned_vertices):
            unsettled = [v for v in cleaned_vertices if v not in settled]
            if not unsettled:
                break

            # Deterministic tie-breaking: (distance, vertex_name)
            min_v = min(unsettled, key=lambda v: (distances[v], v))
            min_dist = distances[min_v]

            # If the minimum distance among unsettled nodes is infinity,
            # remaining nodes are unreachable from source.
            if math.isinf(min_dist):
                break

            settled.add(min_v)
            settled_order.append(min_v)

            relaxations: List[RelaxationDetail] = []
            for neighbor, weight in adj[min_v]:
                if neighbor in settled:
                    continue

                prev_d = distances[neighbor]
                cand_d = min_dist + weight
                updated = False
                if cand_d < prev_d:
                    distances[neighbor] = cand_d
                    predecessors[neighbor] = min_v
                    updated = True

                relaxations.append(
                    RelaxationDetail(
                        edge=(min_v, neighbor),
                        weight=weight,
                        previous_dist=prev_d,
                        candidate_dist=cand_d,
                        updated=updated,
                    )
                )

            remaining_q = [v for v in cleaned_vertices if v not in settled]
            step_k = DijkstraStepState(
                step_index=len(expected_trace),
                step_type="iteration",
                extracted_node=min_v,
                extracted_dist=min_dist,
                relaxations=relaxations,
                tentative_distances=dict(distances),
                settled_nodes=list(settled_order),
                remaining_queue=remaining_q,
                predecessors=dict(predecessors),
                description=(
                    f"Extract min node {min_v} (dist {min_dist:g}). "
                    f"Relaxed {len(relaxations)} outgoing edges."
                ),
            )
            expected_trace.append(step_k)

        # Shortest path reconstruction
        shortest_paths: Dict[str, List[str]] = {}
        for v in cleaned_vertices:
            if math.isinf(distances[v]):
                shortest_paths[v] = []
                continue
            path = []
            curr: Optional[str] = v
            visited_in_path = set()
            while curr is not None and curr not in visited_in_path:
                visited_in_path.add(curr)
                path.append(curr)
                if curr == clean_source:
                    break
                curr = predecessors.get(curr)
            path.reverse()
            if path and path[0] == clean_source:
                shortest_paths[v] = path
            else:
                shortest_paths[v] = [v] if v == clean_source else []

        return expected_trace, distances, settled_order, predecessors, shortest_paths

    @classmethod
    def parse_graph_from_spec(
        cls, spec: Dict[str, Any]
    ) -> Tuple[List[str], List[Tuple[str, str, float]], str, bool]:
        """
        Extracts vertices, weighted edges, source, and directed flag from a worked_example_spec.
        Supports structured dictionaries, inputs mappings, and prose descriptions.
        """
        setup = str(spec.get("setup", "")).strip()
        inputs = spec.get("inputs", {}) if isinstance(spec.get("inputs"), dict) else {}

        vertices: List[str] = []
        edges: List[Tuple[str, str, float]] = []
        source: Optional[str] = None
        directed: bool = False

        # 1. Directed flag detection
        full_context = f"{setup} {json.dumps(inputs)}"
        if re.search(r"\bdirected\b", full_context, re.IGNORECASE) and not re.search(r"\bundirected\b", full_context, re.IGNORECASE):
            directed = True

        # 2. Source detection
        if inputs.get("source"):
            source = str(inputs["source"]).strip()
        elif inputs.get("s"):
            source = str(inputs["s"]).strip()
        elif spec.get("source"):
            source = str(spec["source"]).strip()
        else:
            src_match = re.search(
                r"\b(?:start(?:ing)?|source)\s+(?:node\s+)?(?:is\s+)?['\"]?([A-Za-z0-9_]+)['\"]?",
                setup,
                re.IGNORECASE,
            )
            if src_match:
                source = src_match.group(1).strip()

        # 3. Vertices detection
        if isinstance(inputs.get("vertices"), list):
            vertices = [str(v).strip() for v in inputs["vertices"]]
        elif isinstance(spec.get("vertices"), list):
            vertices = [str(v).strip() for v in spec["vertices"]]
        else:
            # Check setup text for 'nodes (A, B, C, D, E)' or 'vertices: A, B, C'
            v_match = re.search(r"\b(?:nodes|vertices)\s*\(([^)]+)\)", setup, re.IGNORECASE)
            if not v_match:
                v_match = re.search(r"\b(?:nodes|vertices)\s*(?:are|:)\s*([A-Za-z0-9_,\s]+?)(?:\.|\band\b|$)", setup, re.IGNORECASE)
            if not v_match and isinstance(inputs.get("G"), str):
                v_match = re.search(r"\b(?:nodes|vertices)\s+([A-Za-z0-9_,\s]+?)(?:\.|\band\b|$)", inputs["G"], re.IGNORECASE)

            if v_match:
                raw_list = v_match.group(1).replace("and", ",")
                vertices = [v.strip() for v in raw_list.split(",") if v.strip()]

        # 4. Edges detection
        if isinstance(inputs.get("edges"), list):
            for e in inputs["edges"]:
                if isinstance(e, (tuple, list)) and len(e) >= 3:
                    edges.append((str(e[0]).strip(), str(e[1]).strip(), float(e[2])))
                elif isinstance(e, dict):
                    edges.append((str(e.get("u") or e.get("source")).strip(), str(e.get("v") or e.get("target")).strip(), float(e.get("weight", 1))))
        elif isinstance(spec.get("edges"), list):
            for e in spec["edges"]:
                if isinstance(e, (tuple, list)) and len(e) >= 3:
                    edges.append((str(e[0]).strip(), str(e[1]).strip(), float(e[2])))
        else:
            # Pattern A: 'A-B (1)' or 'A - B (1.5)'
            for m in re.finditer(r"([A-Za-z0-9_]+)\s*[-—–]\s*([A-Za-z0-9_]+)\s*\(\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\)", setup):
                edges.append((m.group(1).strip(), m.group(2).strip(), float(m.group(3))))

            # Pattern B: 'A-B: 1' or 'A-B = 1'
            if not edges:
                for m in re.finditer(r"([A-Za-z0-9_]+)\s*[-—–]\s*([A-Za-z0-9_]+)\s*[:=]\s*(-?[0-9]+(?:\.[0-9]+)?)\b", setup):
                    edges.append((m.group(1).strip(), m.group(2).strip(), float(m.group(3))))

            # Pattern C: Python tuples '("A", "B", 1)' or '(A, B, 1)'
            if not edges:
                for m in re.finditer(r"\(\s*['\"]?([A-Za-z0-9_]+)['\"]?\s*,\s*['\"]?([A-Za-z0-9_]+)['\"]?\s*,\s*(-?[0-9]+(?:\.[0-9]+)?)\s*\)", setup):
                    edges.append((m.group(1).strip(), m.group(2).strip(), float(m.group(3))))

            # Pattern D: Directed arrow 'A -> B (1)'
            if not edges:
                for m in re.finditer(r"([A-Za-z0-9_]+)\s*(?:->|→)\s*([A-Za-z0-9_]+)\s*(?:\(|\:)?\s*(-?[0-9]+(?:\.[0-9]+)?)\)?", setup):
                    edges.append((m.group(1).strip(), m.group(2).strip(), float(m.group(3))))
                    directed = True

        # Fallback: if vertices was not explicitly found, infer from edge endpoints
        if not vertices and edges:
            inferred = set()
            for u, v, _ in edges:
                inferred.add(u)
                inferred.add(v)
            vertices = sorted(list(inferred))

        # Fallback source from step 0 if still missing
        if not source and spec.get("steps") and isinstance(spec["steps"], list) and len(spec["steps"]) > 0:
            first_step = str(spec["steps"][0].get("step", ""))
            src_m = re.search(r"\bdist\[([A-Za-z0-9_]+)\]\s*=\s*0\b", first_step)
            if src_m:
                source = src_m.group(1).strip()

        return vertices, edges, source or "", directed

    @classmethod
    def parse_claimed_step(cls, step_data: Any, step_index: int) -> Dict[str, Any]:
        """
        Parses a step from worked_example_spec (string or dict) into normalized claimed state.
        """
        if not isinstance(step_data, dict):
            step_dict = {"step": str(step_data), "result": ""}
        else:
            step_dict = step_data

        step_desc = str(step_dict.get("step", "")).strip()
        raw_result = str(step_dict.get("result", "")).strip()
        full_text = f"{step_desc} {raw_result}"

        # 1. Extracted node
        extracted_node = step_dict.get("extracted_node")
        if extracted_node:
            extracted_node = str(extracted_node).strip()
        elif re.search(r"\binit(?:ialize|ialization)?\b", step_desc, re.IGNORECASE) and not re.search(r"\bextract", step_desc, re.IGNORECASE):
            extracted_node = None
        else:
            m = re.search(
                r"\b(?:extract|pop|select|visit)(?:ing)?\s+(?:node\s+)?([A-Za-z0-9_]+)",
                full_text,
                re.IGNORECASE,
            )
            if m:
                extracted_node = m.group(1).strip()
            else:
                m2 = re.search(r"\bnode\s+([A-Za-z0-9_]+)\s+(?:from|extracted|popped|selected)", full_text, re.IGNORECASE)
                if m2:
                    extracted_node = m2.group(1).strip()

        # 2. Tentative distances
        claimed_distances: Dict[str, float] = {}
        if isinstance(step_dict.get("distances"), dict):
            for k, v in step_dict["distances"].items():
                if v in ("∞", "inf", "infinity", float("inf")) or str(v).lower() in ("inf", "infinity", "none"):
                    claimed_distances[str(k).strip()] = float("inf")
                else:
                    try:
                        claimed_distances[str(k).strip()] = float(v)
                    except (ValueError, TypeError):
                        pass
        else:
            # Check for dict syntax: dist = {A: 0, B: 1, ...}
            m_dict = re.search(r"\bdist(?:ances)?\s*=\s*\{([^}]+)\}", full_text, re.IGNORECASE)
            if m_dict:
                for pair in m_dict.group(1).split(","):
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if v in ("∞", "inf", "infinity", "float('inf')"):
                            claimed_distances[k] = float("inf")
                        else:
                            try:
                                claimed_distances[k] = float(v)
                            except ValueError:
                                pass

            # Also check bracket syntax: dist[A] = 0, dist[B] = 1
            for m in re.finditer(r"\bdist\[([A-Za-z0-9_]+)\]\s*=\s*([^,;\s]+)", full_text):
                k = m.group(1).strip()
                v = m.group(2).strip().strip("'\"")
                if v in ("∞", "inf", "infinity", "float('inf')"):
                    claimed_distances[k] = float("inf")
                else:
                    try:
                        claimed_distances[k] = float(v)
                    except ValueError:
                        pass

        # 3. Remaining queue
        claimed_queue: Optional[List[str]] = None
        if isinstance(step_dict.get("queue"), list):
            claimed_queue = [str(x).strip() for x in step_dict["queue"]]
        else:
            m_q = re.search(r"\bQ\s*=\s*\{([^}]*)\}", full_text)
            if m_q:
                q_body = m_q.group(1).strip()
                if q_body:
                    claimed_queue = [x.strip() for x in q_body.split(",") if x.strip()]
                else:
                    claimed_queue = []

        # 4. Settled set
        claimed_settled: Optional[List[str]] = None
        if isinstance(step_dict.get("settled"), list):
            claimed_settled = [str(x).strip() for x in step_dict["settled"]]
        else:
            m_s = re.search(r"\b(?:settled|visited|S)\s*=\s*\{([^}]*)\}", full_text, re.IGNORECASE)
            if m_s:
                s_body = m_s.group(1).strip()
                if s_body:
                    claimed_settled = [x.strip() for x in s_body.split(",") if x.strip()]
                else:
                    claimed_settled = []

        # 5. Stale entry discard indicator
        is_stale_discard = bool(
            re.search(r"\b(?:stale|already settled|discard(?:ed)?|skip(?:ped)?)\b", full_text, re.IGNORECASE)
        )

        return {
            "step_index": step_index,
            "step_desc": step_desc,
            "raw_result": raw_result,
            "extracted_node": extracted_node,
            "distances": claimed_distances,
            "queue": claimed_queue,
            "settled": claimed_settled,
            "is_stale_discard": is_stale_discard,
        }

    @classmethod
    def parse_claimed_final_distances(cls, final_result_str: Any) -> Dict[str, float]:
        """Parses final distance mappings from final_result string or dictionary."""
        res: Dict[str, float] = {}
        if isinstance(final_result_str, dict):
            for k, v in final_result_str.items():
                if v in ("∞", "inf", "infinity", float("inf")):
                    res[str(k).strip()] = float("inf")
                else:
                    try:
                        res[str(k).strip()] = float(v)
                    except (ValueError, TypeError):
                        pass
            return res

        s = str(final_result_str or "").strip()
        m_dict = re.search(r"\{([^}]+)\}", s)
        if m_dict:
            for pair in m_dict.group(1).split(","):
                if ":" in pair:
                    k, v = pair.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if v in ("∞", "inf", "infinity"):
                        res[k] = float("inf")
                    else:
                        try:
                            res[k] = float(v)
                        except ValueError:
                            pass
        else:
            for m in re.finditer(r"\b([A-Za-z0-9_]+)\s*[:=]\s*([0-9\.]+|∞|inf)\b", s):
                k = m.group(1).strip()
                v = m.group(2).strip()
                if v in ("∞", "inf"):
                    res[k] = float("inf")
                else:
                    try:
                        res[k] = float(v)
                    except ValueError:
                        pass
        return res

    @classmethod
    def verify_trace(
        cls,
        vertices: List[str],
        edges: List[Tuple[str, str, float]],
        source: str,
        claimed_steps: List[Any],
        directed: bool = False,
        claimed_final: Optional[Union[Dict[str, float], str]] = None,
    ) -> DijkstraTraceReport:
        """
        Validates claimed algorithmic steps against ground-truth Dijkstra simulation.
        Checks extract-min order, tentative distances, relaxations, settled sets,
        priority queue, and final distances.
        """
        discrepancies: List[str] = []

        # 1. Run simulation
        try:
            expected_trace, expected_final_distances, expected_settled, predecessors, shortest_paths = cls.simulate(
                vertices=vertices,
                edges=edges,
                source=source,
                directed=directed,
            )
        except ValueError as e:
            err_msg = str(e)
            diag = DiagnosticCode.INVALID_GRAPH_DEFINITION
            if "negative" in err_msg.lower():
                diag = DiagnosticCode.NEGATIVE_EDGE_WEIGHT
            return DijkstraTraceReport(
                overall_status=VerificationStatus.FAILED,
                total_steps=len(claimed_steps),
                verified_steps=0,
                failed_steps=max(1, len(claimed_steps)),
                review_steps=0,
                not_applicable_steps=0,
                edges=[{"u": str(e[0]), "v": str(e[1]), "weight": float(e[2])} for e in edges if len(e) >= 3],
                source=source,
                vertices=vertices,
                directed=directed,
                discrepancies=[err_msg],
                is_safe_for_stage2=False,
                final_result_status=VerificationStatus.FAILED,
                summary=f"Dijkstra verification failed: {err_msg}",
            )

        step_verifications: List[DijkstraStepVerification] = []
        verified_steps_count = 0
        failed_steps_count = 0

        # Track currently settled nodes to handle stale queue entries
        simulated_settled: Set[str] = set()

        for idx, claimed_raw in enumerate(claimed_steps):
            claimed_parsed = cls.parse_claimed_step(claimed_raw, idx)
            claimed_step_desc = claimed_parsed["step_desc"]
            claimed_raw_result = claimed_parsed["raw_result"]

            # Handle step index exceeding expected trace
            if idx >= len(expected_trace):
                # Check if this is a legitimate stale-queue discard step
                if claimed_parsed["is_stale_discard"] and claimed_parsed["extracted_node"] in simulated_settled:
                    step_verif = DijkstraStepVerification(
                        step_index=idx,
                        status=VerificationStatus.VERIFIED,
                        diagnostic=DiagnosticCode.ALGORITHMIC_VERIFIED,
                        claimed_step=claimed_step_desc,
                        claimed_result=claimed_raw_result,
                        expected_state=None,
                        extracted_node_match=True,
                        distances_match=True,
                        settled_match=True,
                        queue_match=True,
                        details=f"Stale queue entry for '{claimed_parsed['extracted_node']}' correctly skipped.",
                    )
                    step_verifications.append(step_verif)
                    verified_steps_count += 1
                    continue

                step_disc = [f"Step {idx}: Claimed step index exceeds expected algorithm steps ({len(expected_trace)})."]
                discrepancies.extend(step_disc)
                step_verif = DijkstraStepVerification(
                    step_index=idx,
                    status=VerificationStatus.FAILED,
                    diagnostic=DiagnosticCode.ALGORITHMIC_STATE_MISMATCH,
                    claimed_step=claimed_step_desc,
                    claimed_result=claimed_raw_result,
                    expected_state=None,
                    extracted_node_match=False,
                    distances_match=False,
                    settled_match=False,
                    queue_match=False,
                    discrepancies=step_disc,
                    details="Extra step not matching simulation trace.",
                )
                step_verifications.append(step_verif)
                failed_steps_count += 1
                continue

            exp_step = expected_trace[idx]
            step_disc: List[str] = []
            diag_code = DiagnosticCode.ALGORITHMIC_VERIFIED

            # A. Extracted Node Verification
            node_match = True
            if exp_step.step_type == "initialization":
                if claimed_parsed["extracted_node"] is not None:
                    node_match = False
                    msg = f"Step 0 expected graph initialization, but claimed extracted node '{claimed_parsed['extracted_node']}'."
                    step_disc.append(msg)
                    diag_code = DiagnosticCode.INVALID_EXTRACT_MIN
            else:
                simulated_settled.add(exp_step.extracted_node)
                if claimed_parsed["extracted_node"] is None:
                    node_match = False
                    msg = (
                        f"Step {idx}: Unparseable or missing extract-min node in claimed iteration step "
                        f"(expected node '{exp_step.extracted_node}')."
                    )
                    step_disc.append(msg)
                    diag_code = DiagnosticCode.INVALID_EXTRACT_MIN
                elif claimed_parsed["extracted_node"] != exp_step.extracted_node:
                    node_match = False
                    msg = (
                        f"Step {idx}: Invalid extract-min node. Claimed '{claimed_parsed['extracted_node']}', "
                        f"but minimum tentative distance node is '{exp_step.extracted_node}' (tentative dist {exp_step.extracted_dist:g})."
                    )
                    step_disc.append(msg)
                    diag_code = DiagnosticCode.INVALID_EXTRACT_MIN

            # B. Tentative Distances Verification
            distances_match = True
            missing_vertices = [v for v in exp_step.tentative_distances if v not in claimed_parsed["distances"]]
            if not claimed_parsed["distances"]:
                distances_match = False
                msg = f"Step {idx}: Missing or unparseable distance snapshot in claimed step."
                step_disc.append(msg)
                if diag_code == DiagnosticCode.ALGORITHMIC_VERIFIED:
                    diag_code = DiagnosticCode.ALGORITHMIC_STATE_MISMATCH
            elif missing_vertices:
                distances_match = False
                msg = f"Step {idx}: Incomplete distance snapshot: missing vertices {missing_vertices}."
                step_disc.append(msg)
                if diag_code == DiagnosticCode.ALGORITHMIC_VERIFIED:
                    diag_code = DiagnosticCode.ALGORITHMIC_STATE_MISMATCH
            else:
                for v, exp_d in exp_step.tentative_distances.items():
                    c_d = claimed_parsed["distances"][v]
                    mismatch = False
                    if math.isinf(exp_d) != math.isinf(c_d):
                        mismatch = True
                    elif not math.isinf(exp_d) and abs(c_d - exp_d) > 1e-5:
                        mismatch = True

                    if mismatch:
                        distances_match = False
                        c_disp = "∞" if math.isinf(c_d) else f"{c_d:g}"
                        e_disp = "∞" if math.isinf(exp_d) else f"{exp_d:g}"
                        msg = f"Step {idx}: Tentative distance mismatch for vertex '{v}': claimed {c_disp}, expected {e_disp}."
                        step_disc.append(msg)
                        if diag_code == DiagnosticCode.ALGORITHMIC_VERIFIED:
                            diag_code = DiagnosticCode.ALGORITHMIC_STATE_MISMATCH

            # C. Settled Nodes Verification (if claimed)
            settled_match = True
            if claimed_parsed["settled"] is not None:
                claimed_set = set(claimed_parsed["settled"])
                exp_set = set(exp_step.settled_nodes)
                if claimed_set != exp_set:
                    settled_match = False
                    msg = f"Step {idx}: Settled set mismatch: claimed {sorted(list(claimed_set))}, expected {sorted(list(exp_set))}."
                    step_disc.append(msg)
                    if diag_code == DiagnosticCode.ALGORITHMIC_VERIFIED:
                        diag_code = DiagnosticCode.ALGORITHMIC_STATE_MISMATCH

            # D. Remaining Priority Queue Verification (if claimed)
            queue_match = True
            if claimed_parsed["queue"] is not None:
                claimed_q_set = set(claimed_parsed["queue"])
                exp_q_set = set(exp_step.remaining_queue)
                if claimed_q_set != exp_q_set:
                    queue_match = False
                    msg = f"Step {idx}: Priority queue mismatch: claimed {sorted(list(claimed_q_set))}, expected {sorted(list(exp_q_set))}."
                    step_disc.append(msg)
                    if diag_code == DiagnosticCode.ALGORITHMIC_VERIFIED:
                        diag_code = DiagnosticCode.ALGORITHMIC_STATE_MISMATCH

            # Build step outcome
            is_step_valid = node_match and distances_match and settled_match and queue_match
            if is_step_valid:
                step_status = VerificationStatus.VERIFIED
                verified_steps_count += 1
                details = f"Step {idx} verified against ground truth: {exp_step.description}"
            else:
                step_status = VerificationStatus.FAILED
                failed_steps_count += 1
                details = "; ".join(step_disc)
                discrepancies.extend(step_disc)

            step_verif = DijkstraStepVerification(
                step_index=idx,
                status=step_status,
                diagnostic=diag_code if not is_step_valid else DiagnosticCode.EXACT_MATCH,
                claimed_step=claimed_step_desc,
                claimed_result=claimed_raw_result,
                expected_state=exp_step,
                extracted_node_match=node_match,
                distances_match=distances_match,
                settled_match=settled_match,
                queue_match=queue_match,
                discrepancies=step_disc,
                details=details,
            )
            step_verifications.append(step_verif)

        # 2. Check for Incomplete Trace
        if len(claimed_steps) < len(expected_trace):
            missing_count = len(expected_trace) - len(claimed_steps)
            missing_nodes = [s.extracted_node for s in expected_trace[len(claimed_steps):] if s.extracted_node]
            inc_msg = (
                f"Claimed trace is incomplete: {len(claimed_steps)} steps provided, "
                f"but {len(expected_trace)} steps required to settle all nodes. "
                f"Missing node extractions: {missing_nodes}."
            )
            discrepancies.append(inc_msg)
            failed_steps_count += missing_count

        # 3. Final Shortest Distances Verification
        final_status = VerificationStatus.VERIFIED
        if claimed_final is not None:
            claimed_final_dict = cls.parse_claimed_final_distances(claimed_final)
            for v, exp_d in expected_final_distances.items():
                if v in claimed_final_dict:
                    c_d = claimed_final_dict[v]
                    mismatch = False
                    if math.isinf(exp_d) != math.isinf(c_d):
                        mismatch = True
                    elif not math.isinf(exp_d) and abs(c_d - exp_d) > 1e-5:
                        mismatch = True

                    if mismatch:
                        final_status = VerificationStatus.FAILED
                        c_disp = "∞" if math.isinf(c_d) else f"{c_d:g}"
                        e_disp = "∞" if math.isinf(exp_d) else f"{exp_d:g}"
                        msg = f"Final distance mismatch for vertex '{v}': claimed {c_disp}, expected {e_disp}."
                        discrepancies.append(msg)

        # Overall Status
        has_failure = failed_steps_count > 0 or final_status == VerificationStatus.FAILED or len(discrepancies) > 0
        overall_status = VerificationStatus.FAILED if has_failure else VerificationStatus.VERIFIED
        is_safe = (overall_status == VerificationStatus.VERIFIED)

        summary = (
            f"Dijkstra trace verification: {verified_steps_count} verified, "
            f"{failed_steps_count} failed out of {len(expected_trace)} canonical steps."
        )

        return DijkstraTraceReport(
            overall_status=overall_status,
            total_steps=len(expected_trace),
            verified_steps=verified_steps_count,
            failed_steps=failed_steps_count,
            review_steps=0,
            not_applicable_steps=0,
            expected_trace=expected_trace,
            step_verifications=step_verifications,
            final_distances=expected_final_distances,
            settled_nodes=expected_settled,
            predecessors=predecessors,
            shortest_paths=shortest_paths,
            edges=[{"u": u, "v": v, "weight": w} for u, v, w in edges],
            source=source,
            vertices=vertices,
            directed=directed,
            discrepancies=discrepancies,
            is_safe_for_stage2=is_safe,
            final_result_status=final_status,
            summary=summary,
        )

    @classmethod
    def verify_worked_example_spec(
        cls,
        spec: Dict[str, Any],
        topic: str = "",
    ) -> Tuple[Dict[str, Any], WorkedExampleReport]:
        """
        End-to-end integration entry point for verifying a Dijkstra worked_example_spec.
        Updates spec with verified step dictionaries and structured verification metadata.
        """
        spec_copy = dict(spec)
        raw_steps = spec_copy.get("steps", [])

        # Parse graph from spec
        try:
            vertices, edges, source, directed = cls.parse_graph_from_spec(spec_copy)
        except Exception as e:
            err_rep = WorkedExampleReport(
                overall_status=VerificationStatus.FAILED,
                total_steps=len(raw_steps),
                verified_steps=0,
                corrected_steps=0,
                failed_steps=max(1, len(raw_steps)),
                review_steps=0,
                not_applicable_steps=0,
                final_result_status=VerificationStatus.FAILED,
                summary=f"Failed to parse Dijkstra graph from spec: {e}",
                is_safe_for_stage2=False,
                step_diagnostics=[{
                    "step_index": 0,
                    "status": VerificationStatus.FAILED.value,
                    "diagnostic": DiagnosticCode.INVALID_GRAPH_DEFINITION.value,
                    "details": str(e),
                }],
            )
            spec_copy["verification_metadata"] = err_rep.to_dict()
            return spec_copy, err_rep

        if not vertices or not source:
            err_rep = WorkedExampleReport(
                overall_status=VerificationStatus.FAILED,
                total_steps=len(raw_steps),
                verified_steps=0,
                corrected_steps=0,
                failed_steps=max(1, len(raw_steps)),
                review_steps=0,
                not_applicable_steps=0,
                final_result_status=VerificationStatus.FAILED,
                summary="Dijkstra graph definition missing vertices or starting source vertex.",
                is_safe_for_stage2=False,
                step_diagnostics=[{
                    "step_index": 0,
                    "status": VerificationStatus.FAILED.value,
                    "diagnostic": DiagnosticCode.INVALID_GRAPH_DEFINITION.value,
                    "details": "Missing vertices or source in graph spec",
                }],
            )
            spec_copy["verification_metadata"] = err_rep.to_dict()
            return spec_copy, err_rep

        claimed_final = spec_copy.get("final_result")

        # Execute trace verification
        trace_report = cls.verify_trace(
            vertices=vertices,
            edges=edges,
            source=source,
            claimed_steps=raw_steps,
            directed=directed,
            claimed_final=claimed_final,
        )

        # Annotate each step with structured algorithmic calculations
        verified_steps_output: List[Dict[str, Any]] = []
        for i, step_data in enumerate(raw_steps):
            step_out = dict(step_data) if isinstance(step_data, dict) else {"step": str(step_data), "result": ""}
            verif = trace_report.step_verifications[i] if i < len(trace_report.step_verifications) else None
            exp = trace_report.expected_trace[i] if i < len(trace_report.expected_trace) else None

            if verif and exp:
                out_var = f"dist_{exp.extracted_node}" if exp.extracted_node else "dist_init"
                structured_calc = {
                    "step_index": i,
                    "status": verif.status.value,
                    "diagnostic": verif.diagnostic.value,
                    "step_type": exp.step_type,
                    "extracted_node": exp.extracted_node,
                    "extracted_dist": exp.extracted_dist,
                    "computed_value": exp.extracted_dist if exp.extracted_dist is not None else 0.0,
                    "output_variable": out_var,
                    "relaxations": [r.to_dict() for r in exp.relaxations],
                    "tentative_distances": {
                        k: ("inf" if math.isinf(v) else v)
                        for k, v in exp.tentative_distances.items()
                    },
                    "settled_nodes": list(exp.settled_nodes),
                    "remaining_queue": list(exp.remaining_queue),
                    "predecessors": dict(exp.predecessors),
                    "is_evaluable": True,
                    "can_safely_correct": False,
                    "details": verif.details,
                    "discrepancies": verif.discrepancies,
                }
            elif verif:
                structured_calc = {
                    "step_index": i,
                    "status": verif.status.value,
                    "diagnostic": verif.diagnostic.value,
                    "is_evaluable": True,
                    "can_safely_correct": False,
                    "details": verif.details,
                    "discrepancies": verif.discrepancies,
                }
            else:
                structured_calc = {
                    "step_index": i,
                    "status": VerificationStatus.FAILED.value,
                    "diagnostic": DiagnosticCode.INCOMPLETE_TRACE.value,
                    "is_evaluable": True,
                    "can_safely_correct": False,
                    "details": "Step unverified due to trace mismatch",
                }

            step_out["structured_calc"] = structured_calc
            verified_steps_output.append(step_out)

        spec_copy["steps"] = verified_steps_output

        # Reconcile final result string if correct or preserve original
        exp_final_str = ", ".join(
            f"{k}: {'∞' if math.isinf(v) else int(v) if v.is_integer() else v}"
            for k, v in trace_report.final_distances.items()
        )
        canonical_final_str = f"dist = {{{exp_final_str}}}"

        if trace_report.final_result_status == VerificationStatus.VERIFIED and not claimed_final:
            spec_copy["final_result"] = canonical_final_str

        # Build algorithmic state dict
        algo_state = {
            "algorithm": "dijkstra",
            "source": source,
            "vertices": vertices,
            "directed": directed,
            "edges": [
                {"u": u, "v": v, "weight": w}
                for u, v, w in edges
            ],
            "final_distances": {
                k: ("inf" if math.isinf(v) else v)
                for k, v in trace_report.final_distances.items()
            },
            "settled_order": trace_report.settled_nodes,
            "predecessors": trace_report.predecessors,
            "shortest_paths": trace_report.shortest_paths,
            "canonical_trace": [s.to_dict() for s in trace_report.expected_trace],
            "tie_breaking_rule": "Lexicographical order on vertex labels: (tentative_distance, vertex_label)",
        }

        # Build authoritative M2.1 ComputationTrace and StateSnapshots
        comp_trace = trace_report.to_computation_trace()
        trace_report.computation_trace = comp_trace

        # Construct standard WorkedExampleReport
        report = WorkedExampleReport(
            overall_status=trace_report.overall_status,
            total_steps=trace_report.total_steps,
            verified_steps=trace_report.verified_steps,
            corrected_steps=0,
            failed_steps=trace_report.failed_steps,
            review_steps=trace_report.review_steps,
            not_applicable_steps=trace_report.not_applicable_steps,
            step_diagnostics=[s.to_dict() for s in trace_report.step_verifications],
            final_result_status=trace_report.final_result_status,
            summary=trace_report.summary,
            is_safe_for_stage2=trace_report.is_safe_for_stage2,
            verified_algorithmic_state=algo_state,
            verification_method="dijkstra_trace_verifier",
            state_snapshots={s.step_index: copy.deepcopy(s.state) for s in comp_trace.snapshots},
        )

        spec_copy["verification_metadata"] = report.to_dict()

        return spec_copy, report


# Register with generic verifier registry
SpecializedVerifierRegistry.register(DijkstraTraceVerifier)

