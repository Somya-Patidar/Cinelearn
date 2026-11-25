"""
Four-Tier Verification Semantics & Criticality Progression Policy for CineLearn.

Implements strict four-tier verification routing (VERIFIED, FAILED, UNVERIFIED, NOT_APPLICABLE),
criticality assessment, multi-tiered remediation (qualify, isolate, remove, halt),
section-level fault isolation, and verification provenance tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class FourTierStatus(str, Enum):
    """The four distinct, immutable verification outcomes."""
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"
    UNVERIFIED = "UNVERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ClaimCriticality(str, Enum):
    """Criticality level of an unverified claim."""
    CRITICAL = "CRITICAL"          # Central claim essential to the learning objective
    NON_CRITICAL = "NON_CRITICAL"  # Supplementary or descriptive background detail


class UnverifiedRemediation(str, Enum):
    """Remediation action for an unverified claim."""
    QUALIFY = "QUALIFY"  # Rephrase with explicit hedging ("heuristically", "in standard approximations")
    ISOLATE = "ISOLATE"  # Confine to an isolated callout card, preventing contamination of core derivations
    REMOVE = "REMOVE"    # Omit supplementary claim from beat
    HALT = "HALT"        # Essential learning objective violated without unresolved claim; pipeline must halt


@dataclass
class ClaimAuditRecord:
    """Audit record for a single pedagogical, factual, or computational claim."""
    claim_id: str
    section_id: str
    beat_id: str
    statement: str
    status: FourTierStatus
    criticality: ClaimCriticality = ClaimCriticality.NON_CRITICAL
    remediation: Optional[UnverifiedRemediation] = None
    essential_objective_violated: bool = False
    details: str = ""
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "claim_id": self.claim_id,
            "section_id": self.section_id,
            "beat_id": self.beat_id,
            "statement": self.statement,
            "status": self.status.value,
            "criticality": self.criticality.value,
            "remediation": self.remediation.value if self.remediation else None,
            "essential_objective_violated": self.essential_objective_violated,
            "details": self.details,
        }
        if self.extra_fields:
            for k, v in self.extra_fields.items():
                if k not in d:
                    d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimAuditRecord":
        if not isinstance(data, dict):
            return cls(
                claim_id="unknown_claim",
                section_id="",
                beat_id="",
                statement="",
                status=FourTierStatus.UNVERIFIED,
            )

        status_str = str(data.get("status", FourTierStatus.UNVERIFIED.value)).upper()
        try:
            status = FourTierStatus(status_str)
        except Exception:
            status = FourTierStatus.UNVERIFIED

        crit_str = str(data.get("criticality", ClaimCriticality.NON_CRITICAL.value)).upper()
        try:
            criticality = ClaimCriticality(crit_str)
        except Exception:
            criticality = ClaimCriticality.NON_CRITICAL

        remed_raw = data.get("remediation")
        remediation = None
        if remed_raw:
            try:
                remediation = UnverifiedRemediation(str(remed_raw).upper())
            except Exception:
                remediation = None

        known_keys = {
            "claim_id", "section_id", "beat_id", "statement", "status",
            "criticality", "remediation", "essential_objective_violated", "details"
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            claim_id=str(data.get("claim_id", "")),
            section_id=str(data.get("section_id", "")),
            beat_id=str(data.get("beat_id", "")),
            statement=str(data.get("statement", "")),
            status=status,
            criticality=criticality,
            remediation=remediation,
            essential_objective_violated=bool(data.get("essential_objective_violated", False)),
            details=str(data.get("details", "")),
            extra_fields=extra,
        )


@dataclass
class VerificationProvenance:
    """Immutable audit trail and provenance record preserved downstream."""
    topic: str
    overall_status: FourTierStatus
    claims: List[ClaimAuditRecord] = field(default_factory=list)
    blocked_sections: List[str] = field(default_factory=list)
    safe_sections: List[str] = field(default_factory=list)
    pipeline_halted: bool = False
    halt_reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "overall_status": self.overall_status.value,
            "claims": [c.to_dict() for c in self.claims],
            "blocked_sections": list(self.blocked_sections),
            "safe_sections": list(self.safe_sections),
            "pipeline_halted": self.pipeline_halted,
            "halt_reason": self.halt_reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationProvenance":
        if not isinstance(data, dict):
            return cls(topic="", overall_status=FourTierStatus.NOT_APPLICABLE)

        status_str = str(data.get("overall_status", FourTierStatus.NOT_APPLICABLE.value)).upper()
        try:
            overall = FourTierStatus(status_str)
        except Exception:
            overall = FourTierStatus.NOT_APPLICABLE

        raw_claims = data.get("claims", [])
        claims = [ClaimAuditRecord.from_dict(c) for c in raw_claims if isinstance(c, dict)]

        return cls(
            topic=str(data.get("topic", "")),
            overall_status=overall,
            claims=claims,
            blocked_sections=list(data.get("blocked_sections", [])),
            safe_sections=list(data.get("safe_sections", [])),
            pipeline_halted=bool(data.get("pipeline_halted", False)),
            halt_reason=str(data.get("halt_reason", "")),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
        )


class VerificationPolicyManager:
    """
    Coordinates verification decisions, enforces 4-tier semantics,
    manages criticality-based progression, and generates audit provenance.
    """

    @classmethod
    def process_verification_results(
        cls,
        topic: str,
        section_ids: List[str],
        claims: List[ClaimAuditRecord],
        is_computational_topic: bool = True,
    ) -> VerificationProvenance:
        """
        Processes claims across an outline according to the approved rules:
        1. VERIFIED: independently proven -> proceeds.
        2. FAILED: confirmed error -> blocks affected section; unaffected sections remain unblocked.
        3. UNVERIFIED: never silently promoted to VERIFIED.
           - Non-critical: proceeds with audit metadata.
           - Critical: qualified, rephrased, isolated, or removed.
           - Pipeline halted only if essential objective cannot be delivered correctly.
        4. NOT_APPLICABLE: non-computational topic -> proceeds cleanly.
        """
        if not is_computational_topic and not claims:
            return VerificationProvenance(
                topic=topic,
                overall_status=FourTierStatus.NOT_APPLICABLE,
                claims=[],
                blocked_sections=[],
                safe_sections=list(section_ids),
                pipeline_halted=False,
                halt_reason="",
            )

        valid_section_ids = {str(s).strip() for s in section_ids if s and str(s).strip()}
        blocked_secs: Set[str] = set()
        safe_secs: Set[str] = set(valid_section_ids)
        pipeline_halted = False
        halt_reasons: List[str] = []

        has_failed = False
        has_unverified = False
        has_verified = False

        for claim in claims:
            # Fix B: Check essential_objective_violated independently of verification status.
            # Any claim violating an essential learning objective must halt the pipeline globally.
            if claim.essential_objective_violated:
                pipeline_halted = True
                claim_sec = str(claim.section_id or "").strip()
                sec_desc = f"section '{claim_sec}'" if claim_sec else "unlocalized"
                halt_reasons.append(
                    f"Essential learning objective violated in claim '{claim.claim_id}' ({sec_desc}): {claim.statement}"
                )
                if claim_sec and claim_sec in valid_section_ids:
                    blocked_secs.add(claim_sec)

            # Rule 1: Never silently promote UNVERIFIED to VERIFIED
            if claim.status == FourTierStatus.UNVERIFIED:
                has_unverified = True
                if claim.criticality == ClaimCriticality.CRITICAL:
                    if claim.essential_objective_violated:
                        claim.remediation = UnverifiedRemediation.HALT
                    else:
                        # Remediate by qualifying or isolating; section remains operable with audit tag
                        if not claim.remediation:
                            claim.remediation = UnverifiedRemediation.QUALIFY
                else:
                    # Non-critical unverified detail proceeds with audit tags
                    if not claim.remediation:
                        claim.remediation = UnverifiedRemediation.QUALIFY

            elif claim.status == FourTierStatus.FAILED:
                has_failed = True
                claim_sec = str(claim.section_id or "").strip()
                if claim_sec and claim_sec in valid_section_ids:
                    # Fix A: Valid localized failure blocks the corresponding section
                    blocked_secs.add(claim_sec)
                    halt_reasons.append(f"Failed check in section '{claim_sec}': {claim.statement}")
                else:
                    # Fix A: Unmatched, empty, or unresolvable section ID cannot be localized; must fail closed and halt
                    pipeline_halted = True
                    halt_reasons.append(
                        f"Unlocalized failed check (unmatched section_id='{claim.section_id}'): {claim.statement}"
                    )

            elif claim.status == FourTierStatus.VERIFIED:
                has_verified = True

        # Unaffected sections are preserved only when the pipeline has not halted
        safe_secs = safe_secs - blocked_secs

        # Determine overall status and enforce global halt guarantees
        if pipeline_halted or (has_failed and len(safe_secs) == 0):
            overall_status = FourTierStatus.FAILED
            pipeline_halted = True
            # When pipeline is halted, no sections may proceed to Stage 2
            safe_secs = set()
        elif has_failed:
            overall_status = FourTierStatus.FAILED
            # Content affected is blocked, but if some sections are unaffected, pipeline can isolate them
        elif has_unverified:
            overall_status = FourTierStatus.UNVERIFIED
        elif has_verified:
            overall_status = FourTierStatus.VERIFIED
        else:
            overall_status = FourTierStatus.NOT_APPLICABLE

        return VerificationProvenance(
            topic=topic,
            overall_status=overall_status,
            claims=claims,
            blocked_sections=sorted(list(blocked_secs)),
            safe_sections=sorted(list(safe_secs)),
            pipeline_halted=pipeline_halted,
            halt_reason="; ".join(halt_reasons) if halt_reasons else "",
        )
