"""
Top-level backward-compatible re-export module for Dijkstra Trace Verifier.

Part of CineLearn Milestone 2.3: Dijkstra Trace Registry Consolidation.
Re-exports canonical classes from grounding.verifiers.graph.dijkstra_verifier
preserving 100% compatibility with all existing callers and tests.
"""

from __future__ import annotations

try:
    from grounding.verifiers.graph.dijkstra_verifier import (
        RelaxationDetail,
        DijkstraStepState,
        DijkstraStepVerification,
        DijkstraTraceReport,
        DijkstraTraceVerifier,
    )
except ImportError:
    from src.grounding.verifiers.graph.dijkstra_verifier import (
        RelaxationDetail,
        DijkstraStepState,
        DijkstraStepVerification,
        DijkstraTraceReport,
        DijkstraTraceVerifier,
    )

__all__ = [
    "RelaxationDetail",
    "DijkstraStepState",
    "DijkstraStepVerification",
    "DijkstraTraceReport",
    "DijkstraTraceVerifier",
]
