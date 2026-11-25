"""
Top-level export for FCFS CPU Scheduling Verifier & Simulator (Milestone 2.2).
Re-exports canonical interfaces from grounding.verifiers.os.fcfs_verifier.
"""

from __future__ import annotations

try:
    from grounding.verifiers.os.fcfs_verifier import (
        ProcessSpec,
        GanttBlock,
        ProcessMetrics,
        ProcessState,
        FCFSSimulator,
        FCFSSimulationResult,
        FCFSTraceVerifier,
        validate_process_list,
    )
except ImportError:
    from src.grounding.verifiers.os.fcfs_verifier import (
        ProcessSpec,
        GanttBlock,
        ProcessMetrics,
        ProcessState,
        FCFSSimulator,
        FCFSSimulationResult,
        FCFSTraceVerifier,
        validate_process_list,
    )

__all__ = [
    "ProcessSpec",
    "GanttBlock",
    "ProcessMetrics",
    "ProcessState",
    "FCFSSimulator",
    "FCFSSimulationResult",
    "FCFSTraceVerifier",
    "validate_process_list",
]
