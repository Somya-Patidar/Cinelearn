"""
CineLearn Grounding & Trace Engine Package.

Provides domain-independent state snapshots and computation trace infrastructure
for verifiable step-by-step educational animations.
"""

from __future__ import annotations

try:
    from grounding.trace_engine import StateSnapshot, ComputationTrace
except ImportError:
    from src.grounding.trace_engine import StateSnapshot, ComputationTrace

try:
    from grounding.verifiers.os.fcfs_verifier import (
        ProcessSpec,
        GanttBlock,
        ProcessMetrics,
        ProcessState,
        FCFSSimulator,
        FCFSSimulationResult,
        FCFSTraceVerifier,
    )
    from grounding.verifiers.dp_verifier import (
        KnapsackSpec,
        KnapsackCellResult,
        KnapsackBacktrackStep,
        KnapsackSimulationResult,
        KnapsackSimulator,
        DPTraceVerifier,
    )
    from grounding.verifiers.graph.dijkstra_verifier import (
        RelaxationDetail,
        DijkstraStepState,
        DijkstraStepVerification,
        DijkstraTraceReport,
        DijkstraTraceVerifier,
    )
except ImportError:
    try:
        from src.grounding.verifiers.os.fcfs_verifier import (
            ProcessSpec,
            GanttBlock,
            ProcessMetrics,
            ProcessState,
            FCFSSimulator,
            FCFSSimulationResult,
            FCFSTraceVerifier,
        )
        from src.grounding.verifiers.dp_verifier import (
            KnapsackSpec,
            KnapsackCellResult,
            KnapsackBacktrackStep,
            KnapsackSimulationResult,
            KnapsackSimulator,
            DPTraceVerifier,
        )
        from src.grounding.verifiers.graph.dijkstra_verifier import (
            RelaxationDetail,
            DijkstraStepState,
            DijkstraStepVerification,
            DijkstraTraceReport,
            DijkstraTraceVerifier,
        )
    except ImportError:
        pass

__all__ = [
    "StateSnapshot",
    "ComputationTrace",
    "ProcessSpec",
    "GanttBlock",
    "ProcessMetrics",
    "ProcessState",
    "FCFSSimulator",
    "FCFSSimulationResult",
    "FCFSTraceVerifier",
    "KnapsackSpec",
    "KnapsackCellResult",
    "KnapsackBacktrackStep",
    "KnapsackSimulationResult",
    "KnapsackSimulator",
    "DPTraceVerifier",
    "RelaxationDetail",
    "DijkstraStepState",
    "DijkstraStepVerification",
    "DijkstraTraceReport",
    "DijkstraTraceVerifier",
]
