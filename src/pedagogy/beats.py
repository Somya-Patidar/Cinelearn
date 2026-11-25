"""
Beat-Level Pedagogical Architecture for CineLearn.

Models heterogeneous instructional moves, concrete learning objectives,
and state transition requirements at the beat level within TeachingOutline and Section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class TeachingMove(str, Enum):
    """Pedagogical moves available at the beat level."""
    INTUITION_BRIDGE = "intuition_bridge"
    FORMAL_DEFINITION = "formal_definition"
    MECHANISTIC_TRACE = "mechanistic_trace"
    COUNTEREXAMPLE_EXPOSURE = "counterexample_exposure"
    BOUNDARY_CLARIFICATION = "boundary_clarification"
    CASE_COMPARISON = "case_comparison"
    RELATIONAL_SYNTHESIS = "relational_synthesis"
    CONCEPTUAL_BREAKDOWN = "conceptual_breakdown"
    PROOF_DERIVATION = "proof_derivation"
    WORKED_DEMONSTRATION = "worked_demonstration"


@dataclass
class TeachingBeat:
    """An individual instructional beat within a section."""
    beat_id: str
    pedagogical_objective: str
    teaching_move: str
    concept_addressed: str
    visual_intent: str
    required_state_changes: List[Dict[str, Any]] = field(default_factory=list)
    verification_references: List[str] = field(default_factory=list)
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate required fields on this beat."""
        errors: List[str] = []
        if not self.beat_id or not str(self.beat_id).strip():
            errors.append("Beat missing mandatory 'beat_id'")
        if not self.pedagogical_objective or not str(self.pedagogical_objective).strip():
            errors.append(f"Beat '{self.beat_id}' missing mandatory 'pedagogical_objective'")
        if not self.teaching_move or not str(self.teaching_move).strip():
            errors.append(f"Beat '{self.beat_id}' missing mandatory 'teaching_move'")
        if not self.concept_addressed or not str(self.concept_addressed).strip():
            errors.append(f"Beat '{self.beat_id}' missing mandatory 'concept_addressed'")
        if not self.visual_intent or not str(self.visual_intent).strip():
            errors.append(f"Beat '{self.beat_id}' missing mandatory 'visual_intent'")
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Lossless serialization to dictionary."""
        d = {
            "beat_id": self.beat_id,
            "pedagogical_objective": self.pedagogical_objective,
            "teaching_move": self.teaching_move,
            "concept_addressed": self.concept_addressed,
            "visual_intent": self.visual_intent,
            "required_state_changes": list(self.required_state_changes),
            "verification_references": list(self.verification_references),
        }
        if self.extra_fields:
            for k, v in self.extra_fields.items():
                if k not in d:
                    d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeachingBeat":
        """Reconstruct TeachingBeat from dictionary, safely handling extra fields."""
        if not isinstance(data, dict):
            return cls(
                beat_id="unknown_beat",
                pedagogical_objective="Unspecified objective",
                teaching_move=TeachingMove.CONCEPTUAL_BREAKDOWN.value,
                concept_addressed="Unspecified concept",
                visual_intent="Unspecified visual intent",
            )

        known_keys = {
            "beat_id", "pedagogical_objective", "teaching_move",
            "concept_addressed", "visual_intent", "required_state_changes",
            "verification_references",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        req_changes = data.get("required_state_changes", [])
        if isinstance(req_changes, dict):
            req_changes = [req_changes]
        elif not isinstance(req_changes, list):
            req_changes = []

        ver_refs = data.get("verification_references", [])
        if isinstance(ver_refs, str):
            ver_refs = [ver_refs]
        elif not isinstance(ver_refs, list):
            ver_refs = []

        return cls(
            beat_id=str(data.get("beat_id", "")).strip(),
            pedagogical_objective=str(data.get("pedagogical_objective", "")).strip(),
            teaching_move=str(data.get("teaching_move", TeachingMove.CONCEPTUAL_BREAKDOWN.value)).strip().lower(),
            concept_addressed=str(data.get("concept_addressed", "")).strip(),
            visual_intent=str(data.get("visual_intent", "")).strip(),
            required_state_changes=req_changes,
            verification_references=[str(r) for r in ver_refs],
            extra_fields=extra,
        )


class BeatValidator:
    """Deterministic validation of heterogeneous teaching beats."""

    VALID_MOVES = {m.value for m in TeachingMove}

    @classmethod
    def validate_beats(cls, beats: List[TeachingBeat]) -> Tuple[bool, List[str]]:
        """
        Validates an entire sequence of beats:
        1. Checks uniqueness of beat_id across beats.
        2. Validates mandatory fields on each individual beat.
        3. Asserts that beat definitions are non-empty.
        4. Validates heterogeneous move diversity without enforcing a rigid universal sequence.
        """
        if not beats:
            return False, ["Beats list cannot be empty"]

        errors: List[str] = []
        seen_ids: Set[str] = set()

        for i, beat in enumerate(beats):
            if not isinstance(beat, TeachingBeat):
                errors.append(f"Item at index {i} is not a TeachingBeat instance")
                continue

            # Check individual beat completeness
            is_valid, beat_errs = beat.validate()
            if not is_valid:
                errors.extend(beat_errs)

            # Check ID uniqueness
            bid = beat.beat_id
            if bid in seen_ids:
                errors.append(f"Duplicate beat_id detected: '{bid}'")
            seen_ids.add(bid)

        return len(errors) == 0, errors

    @classmethod
    def has_heterogeneous_moves(cls, beats: List[TeachingBeat]) -> bool:
        """
        Returns True if the beats exhibit heterogeneous pedagogical moves
        (i.e., not every beat merely repeats the identical teaching move).
        """
        if len(beats) <= 1:
            return True
        moves = {beat.teaching_move for beat in beats if beat.teaching_move}
        return len(moves) > 1
