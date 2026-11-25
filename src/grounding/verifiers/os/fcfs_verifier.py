"""
Deterministic First-Come, First-Served (FCFS) CPU Scheduling Simulator & Trace Verifier.

Part of CineLearn Milestone 2.2: Operating Systems Deep Specialization.
Implements deterministic CPU scheduling simulation, timeline snapshots, Gantt tracking,
honest metric accounting, and outline fidelity verification using the generic
StateSnapshot and ComputationTrace infrastructure.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    from grounding.trace_engine import StateSnapshot, ComputationTrace
except ImportError:
    from src.grounding.trace_engine import StateSnapshot, ComputationTrace

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
# Core Data Models
# ============================================================================

@dataclass(frozen=True)
class ProcessSpec:
    """
    Canonical specification for an individual schedulable process.
    Immutable, validated upon construction.
    """
    pid: str
    arrival_time: int
    burst_time: int

    def __post_init__(self):
        if not isinstance(self.pid, str) or not self.pid.strip():
            raise ValueError("Process PID must be a non-empty string")
        if not isinstance(self.arrival_time, int) or isinstance(self.arrival_time, bool):
            raise TypeError(f"Arrival time for process '{self.pid}' must be an integer, got {type(self.arrival_time).__name__}")
        if self.arrival_time < 0:
            raise ValueError(f"Arrival time for process '{self.pid}' must be a non-negative integer, got {self.arrival_time}")
        if not isinstance(self.burst_time, int) or isinstance(self.burst_time, bool):
            raise TypeError(f"Burst time for process '{self.pid}' must be an integer, got {type(self.burst_time).__name__}")
        if self.burst_time <= 0:
            raise ValueError(f"Burst time for process '{self.pid}' must be a positive integer, got {self.burst_time}")


@dataclass(frozen=True)
class GanttBlock:
    """Represents a continuous interval of CPU allocation or idle time."""
    pid: Optional[str]  # None indicates CPU idle interval
    start_time: int
    end_time: int
    is_idle: bool = False

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "is_idle": self.is_idle,
        }


@dataclass(frozen=True)
class ProcessMetrics:
    """Calculated outcome metrics for a terminated process."""
    pid: str
    arrival_time: int
    burst_time: int
    completion_time: int
    turnaround_time: int
    waiting_time: int
    response_time: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "arrival_time": self.arrival_time,
            "burst_time": self.burst_time,
            "completion_time": self.completion_time,
            "turnaround_time": self.turnaround_time,
            "waiting_time": self.waiting_time,
            "response_time": self.response_time,
        }


class ProcessState(str, Enum):
    NOT_ARRIVED = "NOT_ARRIVED"
    READY = "READY"
    RUNNING = "RUNNING"
    TERMINATED = "TERMINATED"


@dataclass
class FCFSSimulationResult:
    """Consolidated outcome of deterministic FCFS simulation."""
    processes: List[ProcessSpec]
    gantt_chart: List[GanttBlock]
    process_metrics: Dict[str, ProcessMetrics]
    average_turnaround_time: float
    average_waiting_time: float
    total_simulation_time: int
    total_idle_time: int
    computation_trace: ComputationTrace
    average_response_time: float = 0.0


# ============================================================================
# Input Validation Helper
# ============================================================================

def validate_process_list(processes: Sequence[ProcessSpec]) -> None:
    """
    Validates a list of ProcessSpec instances with fail-closed rules.
    Raises ValueError or TypeError on any defect.
    """
    if not processes:
        raise ValueError("Process list cannot be empty")
    seen_pids = set()
    for p in processes:
        if not isinstance(p, ProcessSpec):
            raise TypeError(f"Expected ProcessSpec instance, got {type(p).__name__}")
        if p.pid in seen_pids:
            raise ValueError(f"Duplicate process ID detected: '{p.pid}'")
        seen_pids.add(p.pid)


# ============================================================================
# Deterministic FCFS Simulator
# ============================================================================

class FCFSSimulator:
    """
    Deterministic First-Come, First-Served CPU Scheduling Simulator.

    Guarantees:
    - Stable input order tie-breaking for simultaneous arrivals.
    - Canonical same-tick event ordering: Completions -> Arrivals -> Dispatches.
    - Explicit CPU idle gap representation in Gantt history.
    - Honest metric accounting: metrics remain None until termination;
      subsystem averages remain None until the final simulation termination snapshot.
    - Emits standard ComputationTrace with complete StateSnapshot instances.
    """

    @classmethod
    def run(cls, processes: Sequence[Union[ProcessSpec, Dict[str, Any]]]) -> FCFSSimulationResult:
        """
        Executes deterministic discrete-event FCFS simulation on the given processes.
        """
        norm_procs: List[ProcessSpec] = []
        for p in processes:
            if isinstance(p, dict):
                norm_procs.append(
                    ProcessSpec(
                        pid=str(p.get("pid", "")),
                        arrival_time=int(p.get("arrival_time", 0)),
                        burst_time=int(p.get("burst_time", 1)),
                    )
                )
            else:
                norm_procs.append(p)

        validate_process_list(norm_procs)
        original_processes: List[ProcessSpec] = list(norm_procs)
        process_map: Dict[str, ProcessSpec] = {p.pid: p for p in original_processes}

        # Subsystem tracking state
        process_states: Dict[str, str] = {p.pid: ProcessState.NOT_ARRIVED.value for p in original_processes}
        ready_queue: List[str] = []
        completed_pids: List[str] = []
        gantt_blocks: List[GanttBlock] = []

        # Honest metric store: response_time known at first dispatch; CT, TAT, WT strictly None until completion
        process_metrics_raw: Dict[str, Dict[str, Optional[int]]] = {
            p.pid: {
                "completion_time": None,
                "turnaround_time": None,
                "waiting_time": None,
                "response_time": None,
            }
            for p in original_processes
        }

        trace = ComputationTrace(
            algorithm_or_domain="fcfs_cpu_scheduling",
            metadata={
                "processes": [
                    {"pid": p.pid, "arrival_time": p.arrival_time, "burst_time": p.burst_time}
                    for p in original_processes
                ],
                "tie_breaking_rule": "stable_input_order",
            },
        )

        step_counter = 0

        # CPU Subsystem Registers
        current_time = 0
        cpu_state = "IDLE"
        running_pid: Optional[str] = None
        running_start_time: int = 0
        running_burst_remaining: int = 0

        def _build_snapshot_state(
            avg_tat: Optional[float] = None,
            avg_wt: Optional[float] = None,
            avg_rt: Optional[float] = None,
        ) -> Dict[str, Any]:
            st = {
                "current_time": current_time,
                "cpu_state": cpu_state,
                "running_pid": running_pid,
                "ready_queue": list(ready_queue),
                "completed_pids": list(completed_pids),
                "process_states": dict(process_states),
                "gantt_history": [b.to_dict() for b in gantt_blocks],
                "process_metrics": copy.deepcopy(process_metrics_raw),
                "average_turnaround_time": avg_tat,
                "average_waiting_time": avg_wt,
                "average_response_time": avg_rt,
            }
            # Flat convenience variables for downstream expression evaluators
            for pid, m in process_metrics_raw.items():
                st[f"{pid}_completion_time"] = m["completion_time"]
                st[f"{pid}_turnaround_time"] = m["turnaround_time"]
                st[f"{pid}_waiting_time"] = m["waiting_time"]
                st[f"{pid}_response_time"] = m["response_time"]
            return st

        def _record_snapshot(
            description: str,
            avg_tat: Optional[float] = None,
            avg_wt: Optional[float] = None,
            avg_rt: Optional[float] = None,
        ) -> None:
            nonlocal step_counter
            snap = StateSnapshot(
                step_index=step_counter,
                state=_build_snapshot_state(avg_tat, avg_wt, avg_rt),
                description=description,
                step_type="SCHEDULING_EVENT",
                metadata={
                    "timestamp": float(current_time),
                    "active_variables": [
                        "current_time",
                        "cpu_state",
                        "running_pid",
                        "ready_queue",
                        "completed_pids",
                    ],
                },
            )
            trace.add_snapshot(snap)
            step_counter += 1

        # Initial clock tick t = 0
        # Phase 1: Completions (none at t=0)
        # Phase 2: Arrivals at t=0
        arrivals_at_zero = [
            p for p in original_processes
            if p.arrival_time == 0 and process_states[p.pid] == ProcessState.NOT_ARRIVED.value
        ]
        for p in arrivals_at_zero:
            process_states[p.pid] = ProcessState.READY.value
            ready_queue.append(p.pid)

        # Phase 3: Dispatch at t=0
        if ready_queue:
            running_pid = ready_queue.pop(0)
            process_states[running_pid] = ProcessState.RUNNING.value
            cpu_state = "RUNNING"
            running_start_time = 0
            running_burst_remaining = process_map[running_pid].burst_time
            if process_metrics_raw[running_pid]["response_time"] is None:
                process_metrics_raw[running_pid]["response_time"] = running_start_time - process_map[running_pid].arrival_time
            desc = f"At time 0: Process '{running_pid}' arrived and was dispatched to CPU."
            if ready_queue:
                desc += f" Ready queue: {ready_queue}."
        else:
            cpu_state = "IDLE"
            running_pid = None
            desc = "Initial state at time 0: CPU idle awaiting process arrivals."

        _record_snapshot(desc)

        # Discrete-event simulation loop
        while len(completed_pids) < len(original_processes):
            if running_pid is not None:
                # Scheduled completion of running process
                t_comp = running_start_time + process_map[running_pid].burst_time

                # Earliest arrival among unarrived processes
                unarrived = [
                    p for p in original_processes
                    if process_states[p.pid] == ProcessState.NOT_ARRIVED.value
                ]
                earliest_arr = min((p.arrival_time for p in unarrived), default=None)

                if earliest_arr is not None and earliest_arr < t_comp:
                    # An arrival occurs while process is actively running
                    current_time = earliest_arr
                    # Running process continues non-preemptively
                    running_burst_remaining = t_comp - current_time

                    # Phase 1: Completions (none, running process has remaining burst)
                    # Phase 2: Arrivals at current_time
                    arrived_procs = [
                        p for p in original_processes
                        if p.arrival_time == current_time and process_states[p.pid] == ProcessState.NOT_ARRIVED.value
                    ]
                    arrived_pids = []
                    for p in arrived_procs:
                        process_states[p.pid] = ProcessState.READY.value
                        ready_queue.append(p.pid)
                        arrived_pids.append(p.pid)

                    # Phase 3: Dispatch (CPU already busy)
                    desc = (
                        f"At time {current_time}: Process(es) {arrived_pids} arrived and entered ready queue. "
                        f"Running '{running_pid}' continues execution."
                    )
                    _record_snapshot(desc)

                else:
                    # Running process completes at t_comp
                    current_time = t_comp
                    running_burst_remaining = 0

                    # Phase 1: Completions First
                    finishing_pid = running_pid
                    gantt_blocks.append(
                        GanttBlock(
                            pid=finishing_pid,
                            start_time=running_start_time,
                            end_time=current_time,
                            is_idle=False,
                        )
                    )
                    p_info = process_map[finishing_pid]
                    ct = current_time
                    tat = ct - p_info.arrival_time
                    wt = tat - p_info.burst_time
                    process_metrics_raw[finishing_pid]["completion_time"] = ct
                    process_metrics_raw[finishing_pid]["turnaround_time"] = tat
                    process_metrics_raw[finishing_pid]["waiting_time"] = wt
                    process_states[finishing_pid] = ProcessState.TERMINATED.value
                    completed_pids.append(finishing_pid)
                    running_pid = None
                    cpu_state = "IDLE"

                    # Phase 2: Arrivals Second (at current_time)
                    arrived_procs = [
                        p for p in original_processes
                        if p.arrival_time == current_time and process_states[p.pid] == ProcessState.NOT_ARRIVED.value
                    ]
                    arrived_pids = []
                    for p in arrived_procs:
                        process_states[p.pid] = ProcessState.READY.value
                        ready_queue.append(p.pid)
                        arrived_pids.append(p.pid)

                    # Phase 3: Dispatch Third
                    if ready_queue:
                        running_pid = ready_queue.pop(0)
                        process_states[running_pid] = ProcessState.RUNNING.value
                        cpu_state = "RUNNING"
                        running_start_time = current_time
                        running_burst_remaining = process_map[running_pid].burst_time
                        if process_metrics_raw[running_pid]["response_time"] is None:
                            process_metrics_raw[running_pid]["response_time"] = running_start_time - process_map[running_pid].arrival_time

                        desc = (
                            f"At time {current_time}: Process '{finishing_pid}' completed. "
                            f"Dispatched '{running_pid}' to CPU."
                        )
                        if arrived_pids:
                            desc += f" Newly arrived: {arrived_pids}."
                        _record_snapshot(desc)

                    else:
                        # Ready queue empty
                        if len(completed_pids) == len(original_processes):
                            # All processes terminated: finalize simulation
                            avg_tat = sum(m["turnaround_time"] for m in process_metrics_raw.values()) / len(original_processes)
                            avg_wt = sum(m["waiting_time"] for m in process_metrics_raw.values()) / len(original_processes)
                            avg_rt = sum(m["response_time"] for m in process_metrics_raw.values()) / len(original_processes)
                            desc = (
                                f"At time {current_time}: Final process '{finishing_pid}' completed. "
                                f"All {len(original_processes)} processes terminated. Simulation complete."
                            )
                            _record_snapshot(desc, avg_tat=avg_tat, avg_wt=avg_wt, avg_rt=avg_rt)
                            break
                        else:
                            # Subsystem enters idle state awaiting future arrivals
                            desc = (
                                f"At time {current_time}: Process '{finishing_pid}' completed. "
                                f"Ready queue empty; CPU enters IDLE."
                            )
                            _record_snapshot(desc)

            else:
                # CPU is IDLE with ready_queue empty and unarrived processes remaining
                unarrived = [
                    p for p in original_processes
                    if process_states[p.pid] == ProcessState.NOT_ARRIVED.value
                ]
                next_arr = min(p.arrival_time for p in unarrived)

                # Record idle block from current_time to next_arr
                idle_start = current_time
                current_time = next_arr
                gantt_blocks.append(
                    GanttBlock(
                        pid=None,
                        start_time=idle_start,
                        end_time=current_time,
                        is_idle=True,
                    )
                )

                # Phase 1: Completions (none during idle)
                # Phase 2: Arrivals at next_arr
                arrived_procs = [
                    p for p in original_processes
                    if p.arrival_time == current_time and process_states[p.pid] == ProcessState.NOT_ARRIVED.value
                ]
                arrived_pids = []
                for p in arrived_procs:
                    process_states[p.pid] = ProcessState.READY.value
                    ready_queue.append(p.pid)
                    arrived_pids.append(p.pid)

                # Phase 3: Dispatch
                if ready_queue:
                    running_pid = ready_queue.pop(0)
                    process_states[running_pid] = ProcessState.RUNNING.value
                    cpu_state = "RUNNING"
                    running_start_time = current_time
                    running_burst_remaining = process_map[running_pid].burst_time
                    if process_metrics_raw[running_pid]["response_time"] is None:
                        process_metrics_raw[running_pid]["response_time"] = running_start_time - process_map[running_pid].arrival_time
                    desc = (
                        f"At time {current_time}: After CPU idle period [{idle_start}, {current_time}], "
                        f"Process '{running_pid}' arrived and dispatched to CPU."
                    )
                else:
                    desc = f"At time {current_time}: CPU idle period ended."

                _record_snapshot(desc)

        # Construct final ProcessMetrics objects
        final_metrics: Dict[str, ProcessMetrics] = {}
        for pid in [p.pid for p in original_processes]:
            raw = process_metrics_raw[pid]
            p_spec = process_map[pid]
            final_metrics[pid] = ProcessMetrics(
                pid=pid,
                arrival_time=p_spec.arrival_time,
                burst_time=p_spec.burst_time,
                completion_time=raw["completion_time"],  # type: ignore[arg-type]
                turnaround_time=raw["turnaround_time"],  # type: ignore[arg-type]
                waiting_time=raw["waiting_time"],        # type: ignore[arg-type]
                response_time=raw["response_time"] if raw["response_time"] is not None else 0,
            )

        avg_turnaround = sum(m.turnaround_time for m in final_metrics.values()) / len(original_processes)
        avg_waiting = sum(m.waiting_time for m in final_metrics.values()) / len(original_processes)
        avg_response = sum(m.response_time for m in final_metrics.values()) / len(original_processes)
        total_time = current_time
        total_idle = sum(b.duration for b in gantt_blocks if b.is_idle)

        return FCFSSimulationResult(
            processes=original_processes,
            gantt_chart=gantt_blocks,
            process_metrics=final_metrics,
            average_turnaround_time=avg_turnaround,
            average_waiting_time=avg_waiting,
            average_response_time=avg_response,
            total_simulation_time=total_time,
            total_idle_time=total_idle,
            computation_trace=trace,
        )

    # Convenience alias
    simulate = run


# ============================================================================
# Specialized Verifier Registry Integration
# ============================================================================

class FCFSTraceVerifier(SpecializedVerifier):
    """
    Independent Specialized Verifier for FCFS CPU Scheduling worked example outlines.
    Registers cleanly with SpecializedVerifierRegistry.
    """

    verifier_id: str = "fcfs_cpu_scheduling"

    @classmethod
    def can_handle(cls, spec: Any, topic: str = "") -> bool:
        """Alias for can_verify to ensure full contract compatibility."""
        return cls.can_verify(spec, topic)

    @classmethod
    def can_verify(cls, spec: Dict[str, Any], topic: str = "") -> bool:
        """
        Determines whether the given worked_example_spec or topic represents FCFS scheduling.
        Selectively claims FCFS topics without false positive matching on other domains.
        """
        if not isinstance(spec, dict):
            return False

        topic_clean = topic.strip().lower()
        if any(kw in topic_clean for kw in [
            "fcfs",
            "first come first serve",
            "first-come first-served",
            "first come, first served",
        ]):
            return True

        algo = str(spec.get("algorithm", "")).strip().lower()
        if algo in ["fcfs", "fcfs_cpu_scheduling", "first_come_first_served"]:
            return True

        domain = str(spec.get("domain", "")).strip().lower()
        if domain in ["os_scheduling", "cpu_scheduling_fcfs"]:
            return True

        setup = str(spec.get("setup", "")).strip().lower()
        if "fcfs" in setup or "first-come" in setup:
            return True

        inputs = spec.get("inputs")
        if isinstance(inputs, list):
            if any(isinstance(v, dict) and any(k in v for k in ("AT", "at", "arrival", "arrival_time", "BT", "bt", "burst", "burst_time")) for v in inputs):
                if "fcfs" in topic_clean or not topic_clean or "first" in topic_clean or "scheduling" in topic_clean:
                    return True
        elif isinstance(inputs, dict):
            inp_algo = str(inputs.get("algorithm", "")).strip().lower()
            if inp_algo in ["fcfs", "first_come_first_served"]:
                return True
            if "processes" in inputs and ("fcfs" in topic_clean or not topic_clean):
                procs = inputs.get("processes")
                if isinstance(procs, list) and len(procs) > 0 and isinstance(procs[0], dict):
                    if "burst_time" in procs[0] and ("fcfs" in topic_clean or inp_algo == "fcfs"):
                        return True
            # Recognize dictionary keyed by PID, e.g. {"P1": {"AT": 0, "BT": 3}, ...}
            dict_of_procs = any(
                isinstance(v, dict) and any(k in v for k in ("AT", "at", "arrival", "arrival_time", "BT", "bt", "burst", "burst_time"))
                for k, v in inputs.items()
            )
            if dict_of_procs and ("fcfs" in topic_clean or not topic_clean or "first" in topic_clean or "scheduling" in topic_clean):
                return True

        return False

    @classmethod
    def _normalize_single_process(cls, item: Any, default_pid: str = "") -> ProcessSpec:
        """
        Normalizes a process specification element (ProcessSpec or Dict) into a canonical ProcessSpec.
        Supports field aliases:
        - PID: 'pid', 'PID', 'id', 'ID', 'process', 'name', 'Process'
        - Arrival time: 'arrival_time', 'arrival', 'AT', 'at', 'arr', 'arrivalTime'
        - Burst time: 'burst_time', 'burst', 'BT', 'bt', 'burstTime'
        Validates non-negative integer arrival times and positive integer burst times fail-closed.
        """
        if isinstance(item, ProcessSpec):
            return item
        if isinstance(item, str):
            import re
            m_at = re.search(r"(?:AT|arrival\w*)\s*[:=]\s*(\d+)", item, re.I)
            m_bt = re.search(r"(?:BT|burst\w*)\s*[:=]\s*(\d+)", item, re.I)
            if m_at and m_bt:
                item = {"pid": default_pid, "AT": int(m_at.group(1)), "BT": int(m_bt.group(1))}
            else:
                raise ValueError(f"Could not parse arrival and burst times from string: '{item}'")
        if not isinstance(item, dict):
            raise TypeError(f"Invalid process specification element: expected dict or ProcessSpec, got {type(item).__name__}")

        # Extract PID
        pid_val = None
        for k in ("pid", "PID", "id", "ID", "process", "name", "Process"):
            if k in item and item[k] is not None:
                pid_val = str(item[k]).strip()
                break
        if not pid_val:
            pid_val = default_pid.strip()
        if not pid_val:
            raise ValueError(f"Process specification missing PID: {item}")

        # Extract Arrival Time
        arr_raw = None
        for k in ("arrival_time", "arrival", "AT", "at", "arr", "arrivalTime"):
            if k in item and item[k] is not None:
                arr_raw = item[k]
                break
        if arr_raw is None:
            raise ValueError(f"Process '{pid_val}' missing 'arrival_time' / 'AT' field in: {item}")
        if isinstance(arr_raw, bool) or not isinstance(arr_raw, (int, str)):
            raise TypeError(f"Arrival time for process '{pid_val}' must be an integer, got {type(arr_raw).__name__}")
        try:
            arr_int = int(arr_raw)
            if isinstance(arr_raw, str) and "." in arr_raw:
                raise ValueError
        except Exception:
            raise TypeError(f"Arrival time for process '{pid_val}' must be an integer, got '{arr_raw}'")
        if arr_int < 0:
            raise ValueError(f"Arrival time for process '{pid_val}' must be a non-negative integer, got {arr_int}")

        # Extract Burst Time
        burst_raw = None
        for k in ("burst_time", "burst", "BT", "bt", "burstTime"):
            if k in item and item[k] is not None:
                burst_raw = item[k]
                break
        if burst_raw is None:
            raise ValueError(f"Process '{pid_val}' missing 'burst_time' / 'BT' field in: {item}")
        if isinstance(burst_raw, bool) or not isinstance(burst_raw, (int, str)):
            raise TypeError(f"Burst time for process '{pid_val}' must be an integer, got {type(burst_raw).__name__}")
        try:
            burst_int = int(burst_raw)
            if isinstance(burst_raw, str) and "." in burst_raw:
                raise ValueError
        except Exception:
            raise TypeError(f"Burst time for process '{pid_val}' must be an integer, got '{burst_raw}'")
        if burst_int <= 0:
            raise ValueError(f"Burst time for process '{pid_val}' must be a positive integer, got {burst_int}")

        return ProcessSpec(pid=pid_val, arrival_time=arr_int, burst_time=burst_int)

    @classmethod
    def parse_processes_from_spec(cls, spec: Dict[str, Any]) -> List[ProcessSpec]:
        """
        Extracts and constructs validated ProcessSpec instances from worked example spec.
        Accepts both representations:
        1. List of process dicts/specs under 'inputs.processes', 'processes', or 'inputs.process_list'
        2. Dictionary of process dicts keyed by PID under 'inputs' (e.g. {"P1": {"AT": 0, "BT": 3}, ...})
        Preserves stable process identity and input order.
        Rejects duplicate PIDs, negative arrival times, non-positive burst times, and malformed structures fail-closed.
        """
        if not isinstance(spec, dict):
            raise TypeError(f"Specification must be a dictionary, got {type(spec).__name__}")

        raw_candidates: Any = None
        inputs = spec.get("inputs")

        if isinstance(inputs, list):
            raw_candidates = inputs
        elif isinstance(inputs, dict):
            if "processes" in inputs:
                raw_candidates = inputs["processes"]
            elif "process_list" in inputs:
                raw_candidates = inputs["process_list"]
            else:
                # Check if inputs is a dict-of-dicts keyed by PID
                meta_keys = {"algorithm", "domain", "type", "setup", "note", "description"}
                proc_items = {k: v for k, v in inputs.items() if k not in meta_keys}
                if proc_items and all(isinstance(v, (dict, str)) for v in proc_items.values()):
                    raw_candidates = proc_items

        if raw_candidates is None and "processes" in spec:
            raw_candidates = spec["processes"]
        if raw_candidates is None and "process_list" in spec:
            raw_candidates = spec["process_list"]

        if raw_candidates is None:
            raise ValueError("Specification must contain processes under 'inputs.processes', 'processes', or dictionary 'inputs'")

        parsed: List[ProcessSpec] = []
        seen_pids: Set[str] = set()

        if isinstance(raw_candidates, dict):
            for pid_key, p_val in raw_candidates.items():
                if not isinstance(p_val, (dict, str)):
                    raise TypeError(f"Expected process dictionary or string for key '{pid_key}', got {type(p_val).__name__}")
                spec_obj = cls._normalize_single_process(p_val, default_pid=str(pid_key).strip())
                if spec_obj.pid in seen_pids:
                    raise ValueError(f"Duplicate process ID detected: '{spec_obj.pid}'")
                seen_pids.add(spec_obj.pid)
                parsed.append(spec_obj)
        elif isinstance(raw_candidates, (list, tuple)):
            for idx, item in enumerate(raw_candidates):
                spec_obj = cls._normalize_single_process(item, default_pid=f"P{idx+1}")
                if spec_obj.pid in seen_pids:
                    raise ValueError(f"Duplicate process ID detected: '{spec_obj.pid}'")
                seen_pids.add(spec_obj.pid)
                parsed.append(spec_obj)
        else:
            raise TypeError(f"Processes must be a list or dictionary, got {type(raw_candidates).__name__}")

        if not parsed:
            raise ValueError("Process list cannot be empty")

        validate_process_list(parsed)
        return parsed

    @classmethod
    def verify_worked_example_spec(
        cls,
        spec: Dict[str, Any],
        topic: str = "",
    ) -> Tuple[Dict[str, Any], WorkedExampleReport]:
        """
        Public verifier entry point called by WorkedExampleVerifier.
        Executes deterministic simulation, audits claimed steps and metrics,
        and constructs authoritative WorkedExampleReport.
        """
        spec_copy = dict(spec)
        raw_steps = spec_copy.get("steps", [])

        # 1. Parse and validate inputs (fail-closed)
        try:
            processes = cls.parse_processes_from_spec(spec_copy)
            validate_process_list(processes)
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
                summary=f"FCFS input specification validation failed: {e}",
                is_safe_for_stage2=False,
                verification_method="fcfs_trace_verifier",
                step_diagnostics=[{
                    "step_index": 0,
                    "status": VerificationStatus.FAILED.value,
                    "diagnostic": DiagnosticCode.INVALID_INPUT_VALUES.value,
                    "details": str(e),
                }],
            )
            spec_copy["verification_metadata"] = err_rep.to_dict()
            return spec_copy, err_rep

        # 2. Run authoritative deterministic simulation
        sim_result = FCFSSimulator.run(processes)
        trace = sim_result.computation_trace
        snapshots = trace.snapshots

        # 3. Step audit against authoritative trace snapshots
        verified_steps_output: List[Dict[str, Any]] = []
        step_diagnostics: List[Dict[str, Any]] = []
        verified_count = 0
        failed_count = 0

        for i, step_data in enumerate(raw_steps):
            step_out = dict(step_data) if isinstance(step_data, dict) else {"step": str(step_data)}
            snap: Optional[StateSnapshot] = snapshots[i] if i < len(snapshots) else None

            step_failed = False
            fail_reasons: List[str] = []

            # Audit step claims if present
            claimed_running = step_out.get("running_pid") or step_out.get("claimed_running")
            claimed_time = step_out.get("current_time") or step_out.get("claimed_time")

            if snap is not None:
                # Time mismatch check
                if claimed_time is not None and int(claimed_time) != snap.state.get("current_time"):
                    step_failed = True
                    fail_reasons.append(
                        f"Time mismatch: claimed {claimed_time}, expected {snap.state.get('current_time')}"
                    )

                # Running process mismatch check
                if claimed_running is not None and str(claimed_running) != str(snap.state.get("running_pid")):
                    step_failed = True
                    fail_reasons.append(
                        f"Running process mismatch: claimed '{claimed_running}', expected '{snap.state.get('running_pid')}'"
                    )

                # Premature / predictive metric claims check
                claimed_metrics = step_out.get("claimed_metrics", {})
                if isinstance(claimed_metrics, dict):
                    for pid, m_dict in claimed_metrics.items():
                        expected_m = snap.state.get("process_metrics", {}).get(pid, {})
                        for m_key, m_val in m_dict.items():
                            exp_val = expected_m.get(m_key)
                            if exp_val is None:
                                step_failed = True
                                fail_reasons.append(
                                    f"Premature metric claim for process '{pid}': {m_key}={m_val} claimed, but process not yet terminated at this step"
                                )
                            elif exp_val != m_val:
                                step_failed = True
                                fail_reasons.append(
                                    f"Metric mismatch for '{pid}.{m_key}': claimed {m_val}, expected {exp_val}"
                                )

            if step_failed:
                failed_count += 1
                diag = {
                    "step_index": i,
                    "status": VerificationStatus.FAILED.value,
                    "diagnostic": DiagnosticCode.ALGORITHMIC_STATE_MISMATCH.value,
                    "details": "; ".join(fail_reasons),
                }
                step_out["verification_status"] = VerificationStatus.FAILED.value
                step_out["diagnostics"] = diag
            else:
                verified_count += 1
                diag = {
                    "step_index": i,
                    "status": VerificationStatus.VERIFIED.value,
                    "diagnostic": DiagnosticCode.ALGORITHMIC_VERIFIED.value,
                    "details": "Step claims match authoritative FCFS simulation state",
                }
                step_out["verification_status"] = VerificationStatus.VERIFIED.value
                step_out["diagnostics"] = diag

            if snap is not None:
                step_out["authoritative_state"] = copy.deepcopy(snap.state)

            verified_steps_output.append(step_out)
            step_diagnostics.append(diag)

        spec_copy["steps"] = verified_steps_output

        # 4. Final Result Audit
        claimed_final = spec_copy.get("final_result")
        exp_wt = sim_result.average_waiting_time
        exp_tat = sim_result.average_turnaround_time
        exp_rt = sim_result.average_response_time
        canonical_final_str = (
            f"Average Waiting Time = {exp_wt:.2f}, Average Turnaround Time = {exp_tat:.2f}, Average Response Time = {exp_rt:.2f}"
        )

        final_result_status = VerificationStatus.VERIFIED
        if claimed_final:
            cf_str = str(claimed_final)
            has_per_process_claim = False
            for pid, m in sim_result.process_metrics.items():
                m_dict = m.to_dict()
                pattern = rf"\b{re.escape(pid)}\b[^;\n]*"
                m_match = re.search(pattern, cf_str, re.IGNORECASE)
                if m_match:
                    has_per_process_claim = True
                    chunk = m_match.group(0)
                    for col, key in [("CT", "completion_time"), ("TAT", "turnaround_time"), ("WT", "waiting_time"), ("RT", "response_time")]:
                        c_match = re.search(rf"\b{col}\s*[:=]\s*(\d+)", chunk, re.IGNORECASE)
                        if c_match:
                            val = int(c_match.group(1))
                            if val != m_dict[key]:
                                final_result_status = VerificationStatus.FAILED
                                break
                if final_result_status == VerificationStatus.FAILED:
                    break

            wt_match = re.search(r"\b(?:average|avg)[^=:\d]*(?:waiting|wt)[^=:\d]*[:=]\s*([0-9]*\.?[0-9]+)", cf_str, re.IGNORECASE)
            tat_match = re.search(r"\b(?:average|avg)[^=:\d]*(?:turnaround|tat)[^=:\d]*[:=]\s*([0-9]*\.?[0-9]+)", cf_str, re.IGNORECASE)
            rt_match = re.search(r"\b(?:average|avg)[^=:\d]*(?:response|rt)[^=:\d]*[:=]\s*([0-9]*\.?[0-9]+)", cf_str, re.IGNORECASE)

            if wt_match:
                if abs(float(wt_match.group(1)) - exp_wt) >= 0.05:
                    final_result_status = VerificationStatus.FAILED
            if tat_match:
                if abs(float(tat_match.group(1)) - exp_tat) >= 0.05:
                    final_result_status = VerificationStatus.FAILED
            if rt_match:
                if abs(float(rt_match.group(1)) - exp_rt) >= 0.05:
                    final_result_status = VerificationStatus.FAILED

            if not has_per_process_claim and not wt_match and not tat_match:
                nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", cf_str)]
                if len(nums) >= 2:
                    matches = any(
                        (abs(nums[0] - exp_wt) < 0.05 and abs(nums[1] - exp_tat) < 0.05) or
                        (abs(nums[0] - exp_tat) < 0.05 and abs(nums[1] - exp_wt) < 0.05)
                        for _ in [None]
                    )
                    if not matches:
                        final_result_status = VerificationStatus.FAILED
                elif len(nums) == 1:
                    if not (abs(nums[0] - exp_wt) < 0.05 or abs(nums[0] - exp_tat) < 0.05):
                        final_result_status = VerificationStatus.FAILED
        else:
            spec_copy["final_result"] = canonical_final_str

        # 5. Algorithmic State Payload
        algo_state = {
            "algorithm": "fcfs",
            "processes": [
                {"pid": p.pid, "arrival_time": p.arrival_time, "burst_time": p.burst_time}
                for p in sim_result.processes
            ],
            "gantt_chart": [b.to_dict() for b in sim_result.gantt_chart],
            "process_metrics": {pid: m.to_dict() for pid, m in sim_result.process_metrics.items()},
            "average_turnaround_time": exp_tat,
            "average_waiting_time": exp_wt,
            "average_response_time": sim_result.average_response_time,
            "total_simulation_time": sim_result.total_simulation_time,
            "total_idle_time": sim_result.total_idle_time,
            "tie_breaking_rule": "stable_input_order",
        }

        # Overall Status
        overall_status = VerificationStatus.FAILED if (failed_count > 0 or final_result_status == VerificationStatus.FAILED) else VerificationStatus.VERIFIED

        report = WorkedExampleReport(
            overall_status=overall_status,
            total_steps=len(raw_steps),
            verified_steps=verified_count,
            corrected_steps=0,
            failed_steps=failed_count,
            review_steps=0,
            not_applicable_steps=0,
            step_diagnostics=step_diagnostics,
            final_result_status=final_result_status,
            summary=(
                f"FCFS CPU Scheduling verification {'PASSED' if overall_status == VerificationStatus.VERIFIED else 'FAILED'}: "
                f"{verified_count}/{len(raw_steps)} steps verified, Avg WT={exp_wt:.2f}, Avg TAT={exp_tat:.2f}, Avg RT={sim_result.average_response_time:.2f}"
            ),
            is_safe_for_stage2=(overall_status == VerificationStatus.VERIFIED),
            verified_algorithmic_state=algo_state,
            verification_method="fcfs_trace_verifier",
            state_snapshots={s.step_index: copy.deepcopy(s.state) for s in snapshots},
        )

        spec_copy["verification_metadata"] = report.to_dict()
        return spec_copy, report


# ============================================================================
# Automatic Module-Level Registration
# ============================================================================
SpecializedVerifierRegistry.register(FCFSTraceVerifier)
