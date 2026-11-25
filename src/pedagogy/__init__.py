"""
Pedagogical architecture and teaching design module for CineLearn.
Implements multidimensional topic profiling, heterogeneous teaching beats,
and four-tier verification semantics with criticality progression policies.
"""

from .profiler import (
    TopicProfile,
    TopicProfiler,
    StateEvolution,
    FormalismLevel,
    ComputationRequirement,
    PrimaryVisualMode,
    ProfileStatus,
)
from .beats import (
    TeachingBeat,
    TeachingMove,
    BeatValidator,
)
from .verification_policy import (
    FourTierStatus,
    VerificationPolicyManager,
    ClaimCriticality,
    UnverifiedRemediation,
    ClaimAuditRecord,
    VerificationProvenance,
)

__all__ = [
    "TopicProfile",
    "TopicProfiler",
    "StateEvolution",
    "FormalismLevel",
    "ComputationRequirement",
    "PrimaryVisualMode",
    "ProfileStatus",
    "TeachingBeat",
    "TeachingMove",
    "BeatValidator",
    "FourTierStatus",
    "VerificationPolicyManager",
    "ClaimCriticality",
    "UnverifiedRemediation",
    "ClaimAuditRecord",
    "VerificationProvenance",
]
