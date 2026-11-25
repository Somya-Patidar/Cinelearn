"""
Generic Temporal Trace Contract & Snapshot Engine.

Defines domain-independent state snapshots and chronological computation-trace
contracts for CineLearn worked examples, algorithm animations, and process simulations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union


@dataclass
class StateSnapshot:
    """
    Domain-independent representation of an execution state snapshot at a specific step.

    Captures the complete active state dictionary at a step boundary (scalars,
    collections, pointers, arrays, settled sets, and structured status values).
    Guarantees isolation through deep-copying.
    """
    step_index: int
    state: Dict[str, Any]
    description: str = ""
    step_type: str = "state_transition"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.step_index, int) or isinstance(self.step_index, bool):
            raise TypeError(
                f"step_index must be an integer, got {type(self.step_index).__name__}: {self.step_index}"
            )
        if not isinstance(self.state, dict):
            raise TypeError(
                f"state must be a dict, got {type(self.state).__name__}: {self.state}"
            )
        # Deep-copy state to preserve isolation against downstream mutations
        self.state = copy.deepcopy(self.state)
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata) if self.metadata else {}
        self.metadata = copy.deepcopy(self.metadata)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.state[key]

    def __contains__(self, key: str) -> bool:
        return key in self.state

    def __len__(self) -> int:
        return len(self.state)

    def __iter__(self) -> Iterator[str]:
        return iter(self.state)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes snapshot to a clean dictionary."""
        return {
            "step_index": self.step_index,
            "state": copy.deepcopy(self.state),
            "description": self.description,
            "step_type": self.step_type,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StateSnapshot:
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dictionary, got {type(data).__name__}")
        if "step_index" not in data:
            raise KeyError("Missing required key 'step_index' in StateSnapshot data")
        return cls(
            step_index=int(data["step_index"]),
            state=data.get("state", {}),
            description=str(data.get("description", "")),
            step_type=str(data.get("step_type", "state_transition")),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ComputationTrace:
    """
    Minimal reusable computation-trace contract for ordered state transitions/snapshots.

    Maintains chronological ordering, initial state preservation, and deterministic
    state query interfaces across diverse educational domains (OS scheduling, DP,
    graph algorithms, sorting, and discrete state machines).
    """
    algorithm_or_domain: str
    initial_state: Dict[str, Any] = field(default_factory=dict)
    snapshots: List[StateSnapshot] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.initial_state, dict):
            raise TypeError(
                f"initial_state must be a dict, got {type(self.initial_state).__name__}"
            )
        self.initial_state = copy.deepcopy(self.initial_state)

        cleaned_snaps: List[StateSnapshot] = []
        for s in self.snapshots:
            if isinstance(s, StateSnapshot):
                cleaned_snaps.append(s)
            elif isinstance(s, dict):
                cleaned_snaps.append(StateSnapshot.from_dict(s))
            else:
                raise TypeError(f"Expected StateSnapshot or dict, got {type(s).__name__}")

        self.snapshots = sorted(cleaned_snaps, key=lambda snap: snap.step_index)

    def add_snapshot(
        self,
        snapshot_or_state: Union[StateSnapshot, Dict[str, Any]],
        step_index: Optional[int] = None,
        description: str = "",
        step_type: str = "state_transition",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateSnapshot:
        """
        Appends or updates a state snapshot in chronological order.
        If a snapshot with the same step_index exists, it is replaced idempotently.
        """
        if isinstance(snapshot_or_state, StateSnapshot):
            snap = snapshot_or_state
        elif isinstance(snapshot_or_state, dict):
            if step_index is None:
                # If step_index not provided, infer next sequential index
                max_idx = max([s.step_index for s in self.snapshots], default=-1)
                idx = max_idx + 1
            else:
                idx = step_index
            snap = StateSnapshot(
                step_index=idx,
                state=snapshot_or_state,
                description=description,
                step_type=step_type,
                metadata=metadata or {},
            )
        else:
            raise TypeError(
                f"snapshot_or_state must be StateSnapshot or dict, got {type(snapshot_or_state).__name__}"
            )

        # Check for existing snapshot at this step_index
        for i, existing in enumerate(self.snapshots):
            if existing.step_index == snap.step_index:
                self.snapshots[i] = snap
                return snap

        if self.snapshots and snap.step_index < self.snapshots[-1].step_index:
            raise ValueError(
                f"Snapshots must preserve chronological ordering: cannot add step {snap.step_index} "
                f"after existing step {self.snapshots[-1].step_index}"
            )

        self.snapshots.append(snap)
        return snap

    def get_snapshot(self, step_index: int) -> StateSnapshot:
        """
        Retrieves snapshot at step_index.
        - step_index == -1: returns initial state snapshot.
        - 0 <= step_index <= max_step: returns snapshot at that step.
        - Invalid / out-of-range: raises IndexError.
        """
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            raise TypeError(f"step_index must be an integer, got {type(step_index).__name__}")

        if step_index == -1:
            return StateSnapshot(
                step_index=-1,
                state=copy.deepcopy(self.initial_state),
                description="Initial state before execution",
                step_type="initialization",
            )

        if step_index < -1:
            raise IndexError(
                f"Step index {step_index} is invalid. Minimum valid index is -1 (initial state)."
            )

        for s in self.snapshots:
            if s.step_index == step_index:
                return copy.deepcopy(s)

        max_idx = self.snapshots[-1].step_index if self.snapshots else -1
        raise IndexError(
            f"Step index {step_index} is out of range. "
            f"Valid step indices are -1 to {max_idx}."
        )

    def get_state_at_step(self, step_index: int) -> Dict[str, Any]:
        """Returns complete state dict at step_index."""
        return copy.deepcopy(self.get_snapshot(step_index).state)

    def has_step(self, step_index: int) -> bool:
        """Returns True if a snapshot exists for step_index or if step_index == -1."""
        if step_index == -1:
            return True
        return any(s.step_index == step_index for s in self.snapshots)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self) -> Iterator[StateSnapshot]:
        return iter(self.snapshots)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_or_domain": self.algorithm_or_domain,
            "initial_state": copy.deepcopy(self.initial_state),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ComputationTrace:
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data).__name__}")
        snaps = [StateSnapshot.from_dict(s) for s in data.get("snapshots", [])]
        return cls(
            algorithm_or_domain=str(data.get("algorithm_or_domain", "general")),
            initial_state=data.get("initial_state", {}),
            snapshots=snaps,
            metadata=data.get("metadata", {}),
        )
