"""
Milestone 1 Test Suite: Topic Profiling, Teaching Beats, and Verification Semantics.

Covers:
1. Profile schema and bounds validation.
2. Semantic consistency validation.
3. Invalid and contradictory profiles producing UNCLASSIFIED with confidence 0.0.
4. Heterogeneous teaching beats within one outline.
5. Four-tier verification routing (VERIFIED, FAILED, UNVERIFIED, NOT_APPLICABLE).
6. Critical versus non-critical UNVERIFIED handling.
7. Verification provenance preservation.
8. TeachingOutline serialization and deserialization.
9. Backward compatibility with legacy outlines and fixtures.
10. NOT_APPLICABLE bypass for non-computational topics.
11. No false arithmetic or missing-variable errors for qualitative topics.
12. Live model integration tests using configured Qwen 32B model on 3 canonical topics:
    - Photosynthesis: Light Reactions and Calvin Cycle
    - Binary Search
    - Pythagorean Theorem
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pytest

try:
    from src.pedagogy import (
        TopicProfile,
        TopicProfiler,
        StateEvolution,
        FormalismLevel,
        ComputationRequirement,
        PrimaryVisualMode,
        ProfileStatus,
        TeachingBeat,
        TeachingMove,
        BeatValidator,
        VerificationPolicyManager,
        ClaimAuditRecord,
        FourTierStatus,
        ClaimCriticality,
        UnverifiedRemediation,
        VerificationProvenance,
    )
except ImportError:
    from pedagogy import (
        TopicProfile,
        TopicProfiler,
        StateEvolution,
        FormalismLevel,
        ComputationRequirement,
        PrimaryVisualMode,
        ProfileStatus,
        TeachingBeat,
        TeachingMove,
        BeatValidator,
        VerificationPolicyManager,
        ClaimAuditRecord,
        FourTierStatus,
        ClaimCriticality,
        UnverifiedRemediation,
        VerificationProvenance,
    )

from src.agent import TeachingOutline, Section, TeachingVideoAgent, RunConfig
from src.numerical_evaluator import VerificationStatus
from src.worked_example_verifier import WorkedExampleVerifier, WorkedExampleReport, StepType, CalculationStepAdapter


# ============================================================================
# 1. Profile Schema & Bounds Validation
# ============================================================================

class TestProfileSchemaAndBoundsValidation:
    """Verifies typed profile schema, mandatory field enforcement, and bounds."""

    def test_valid_profile_schema_passes(self):
        valid_payload = {
            "state_evolution": "discrete_state_machine",
            "formalism_level": "rigorous_mathematical",
            "computation_requirement": "deterministic_trace",
            "primary_visual_mode": "network_graph",
            "rationale_and_evidence": "Dijkstra algorithm evolves via discrete node relaxation and priority queue state updates.",
            "profile_status": "CLASSIFIED",
            "confidence": 0.95,
        }
        is_valid, errors = TopicProfiler.validate_schema(valid_payload)
        assert is_valid is True
        assert len(errors) == 0

        profile = TopicProfiler.profile_from_data(valid_payload)
        assert profile.profile_status == ProfileStatus.CLASSIFIED
        assert profile.state_evolution == StateEvolution.DISCRETE_STATE_MACHINE
        assert profile.computation_requirement == ComputationRequirement.DETERMINISTIC_TRACE
        assert profile.confidence == 0.95

    def test_missing_required_fields_detected(self):
        incomplete_payload = {
            "state_evolution": "discrete_state_machine",
            # missing formalism_level
            "computation_requirement": "deterministic_trace",
            # missing primary_visual_mode
            "rationale_and_evidence": "Some rationale",
        }
        is_valid, errors = TopicProfiler.validate_schema(incomplete_payload)
        assert is_valid is False
        assert any("formalism_level" in e for e in errors)
        assert any("primary_visual_mode" in e for e in errors)

    def test_invalid_enums_detected(self):
        invalid_enum_payload = {
            "state_evolution": "invalid_evolution_type",
            "formalism_level": "rigorous_mathematical",
            "computation_requirement": "deterministic_trace",
            "primary_visual_mode": "network_graph",
            "rationale_and_evidence": "Valid rationale text",
            "confidence": 0.9,
        }
        is_valid, errors = TopicProfiler.validate_schema(invalid_enum_payload)
        assert is_valid is False
        assert any("invalid_evolution_type" in e for e in errors)

    def test_confidence_bounds_enforced(self):
        out_of_bounds_high = {
            "state_evolution": "discrete_state_machine",
            "formalism_level": "rigorous_mathematical",
            "computation_requirement": "deterministic_trace",
            "primary_visual_mode": "network_graph",
            "rationale_and_evidence": "Valid rationale",
            "confidence": 1.5,
        }
        is_valid, errors = TopicProfiler.validate_schema(out_of_bounds_high)
        assert is_valid is False
        assert any("outside allowable bounds" in e for e in errors)

        out_of_bounds_low = dict(out_of_bounds_high, confidence=-0.2)
        is_valid, errors = TopicProfiler.validate_schema(out_of_bounds_low)
        assert is_valid is False


# ============================================================================
# 2. Semantic Consistency Validation
# ============================================================================

class TestSemanticConsistencyValidation:
    """Verifies that schema validity alone is NOT treated as proof of correctness."""

    def test_uncertain_dimension_with_high_confidence_is_contradiction(self):
        contradictory_payload = {
            "state_evolution": "uncertain",
            "formalism_level": "rigorous_mathematical",
            "computation_requirement": "deterministic_trace",
            "primary_visual_mode": "network_graph",
            "rationale_and_evidence": "We are uncertain about evolution.",
            "confidence": 0.95,
        }
        profile = TopicProfile.from_dict(contradictory_payload)
        is_consistent, contradictions = TopicProfiler.validate_semantic_consistency(profile)
        assert is_consistent is False
        assert any("UNCERTAIN dimensions but claims confidence" in c for c in contradictions)

    def test_discrete_state_machine_with_none_not_applicable_is_contradiction(self):
        contradictory_payload = {
            "state_evolution": "discrete_state_machine",
            "formalism_level": "rigorous_mathematical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "network_graph",
            "rationale_and_evidence": "Claiming discrete state mutations without computation.",
            "confidence": 0.9,
        }
        profile = TopicProfile.from_dict(contradictory_payload)
        is_consistent, contradictions = TopicProfiler.validate_semantic_consistency(profile)
        assert is_consistent is False
        assert any("discrete_state_machine but computation_requirement claims none_not_applicable" in c for c in contradictions)

    def test_state_table_matrix_with_none_not_applicable_is_contradiction(self):
        contradictory_payload = {
            "state_evolution": "static_structural",
            "formalism_level": "rigorous_mathematical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "state_table_matrix",
            "rationale_and_evidence": "Matrix table visual without computation.",
            "confidence": 0.85,
        }
        profile = TopicProfile.from_dict(contradictory_payload)
        is_consistent, contradictions = TopicProfiler.validate_semantic_consistency(profile)
        assert is_consistent is False
        assert any("state_table_matrix but computation_requirement claims none_not_applicable" in c for c in contradictions)

    def test_contextual_algorithm_with_none_not_applicable_is_contradiction(self):
        payload = {
            "state_evolution": "static_structural",
            "formalism_level": "qualitative_intuitive",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "structured_concept_card",
            "rationale_and_evidence": "A general discussion without formulas.",
            "confidence": 0.9,
        }
        profile = TopicProfile.from_dict(payload)
        context = {"topic": "Dijkstra's Shortest Path Algorithm"}
        is_consistent, contradictions = TopicProfiler.validate_semantic_consistency(profile, context=context)
        assert is_consistent is False
        assert any("Topic mentions algorithmic constructs" in c for c in contradictions)

    def test_empty_rationale_is_rejected(self):
        payload = {
            "state_evolution": "continuous_dynamic",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "arithmetic_calculation",
            "primary_visual_mode": "coordinate_geometry",
            "rationale_and_evidence": "",  # Empty
            "confidence": 0.9,
        }
        profile = TopicProfile.from_dict(payload)
        is_consistent, contradictions = TopicProfiler.validate_semantic_consistency(profile)
        assert is_consistent is False
        assert any("rationale_and_evidence must cite" in c for c in contradictions)

    def test_overlapping_characteristics_supported(self):
        overlapping_payload = {
            "state_evolution": "discrete_state_machine",
            "formalism_level": "rigorous_mathematical",
            "computation_requirement": "deterministic_trace",
            "primary_visual_mode": "network_graph",
            "rationale_and_evidence": "Graph algorithm requiring dual visual modes for graph and queue.",
            "secondary_visual_modes": ["state_table_matrix", "coordinate_geometry"],
            "secondary_characteristics": ["greedy_relaxation", "optimal_substructure"],
            "confidence": 0.95,
        }
        profile = TopicProfiler.profile_from_data(overlapping_payload)
        assert profile.profile_status == ProfileStatus.CLASSIFIED
        assert "state_table_matrix" in profile.secondary_visual_modes
        assert "greedy_relaxation" in profile.secondary_characteristics


# ============================================================================
# 3. Explicit UNCLASSIFIED Fallback
# ============================================================================

class TestUnclassifiedFallback:
    """Verifies that flawed profiles trigger explicit UNCLASSIFIED fallback with confidence 0.0."""

    def test_schema_invalid_produces_unclassified(self):
        invalid_data = {"bogus": "value"}
        profile = TopicProfiler.profile_from_data(invalid_data)
        assert profile.profile_status == ProfileStatus.UNCLASSIFIED
        assert profile.confidence == 0.0
        assert len(profile.evidence_notes) > 0
        assert profile.is_unclassified() is True

    def test_semantic_contradiction_produces_unclassified(self):
        contradictory_payload = {
            "state_evolution": "discrete_state_machine",
            "formalism_level": "rigorous_mathematical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "state_table_matrix",
            "rationale_and_evidence": "Contradictory algorithm profile.",
            "confidence": 0.95,
        }
        profile = TopicProfiler.profile_from_data(contradictory_payload)
        assert profile.profile_status == ProfileStatus.UNCLASSIFIED
        assert profile.confidence == 0.0
        assert any("Contradiction" in note for note in profile.evidence_notes)

    def test_unclassified_does_not_impose_hardcoded_visual_strategy(self):
        fallback = TopicProfiler.build_unclassified_fallback(
            rationale="Uncertain topic",
            evidence_notes=["Unknown domain"],
        )
        assert fallback.primary_visual_mode == PrimaryVisualMode.UNCERTAIN
        assert fallback.state_evolution == StateEvolution.UNCERTAIN
        assert fallback.computation_requirement == ComputationRequirement.UNCERTAIN
        # Does not force structured_concept_card or assume non-computational
        assert fallback.primary_visual_mode != PrimaryVisualMode.STRUCTURED_CONCEPT_CARD


# ============================================================================
# 4. Heterogeneous Teaching Beats
# ============================================================================

class TestHeterogeneousTeachingBeats:
    """Verifies beat-level representation, validation, and heterogeneous move diversity."""

    def test_beat_validation_passes_valid_heterogeneous_moves(self):
        beats = [
            TeachingBeat(
                beat_id="beat_1_1",
                pedagogical_objective="Establish intuitive real-world navigation analogy",
                teaching_move=TeachingMove.INTUITION_BRIDGE.value,
                concept_addressed="Shortest path intuition",
                visual_intent="Show road network with distance annotations",
            ),
            TeachingBeat(
                beat_id="beat_1_2",
                pedagogical_objective="State formal greedy relaxation invariant",
                teaching_move=TeachingMove.FORMAL_DEFINITION.value,
                concept_addressed="Non-negative edge weight invariant",
                visual_intent="Highlight node extraction and distance equation MathTex",
                verification_references=["invariant_greedy_choice"],
            ),
            TeachingBeat(
                beat_id="beat_1_3",
                pedagogical_objective="Demonstrate discrete priority queue transition",
                teaching_move=TeachingMove.MECHANISTIC_TRACE.value,
                concept_addressed="Tentative distance relaxation",
                visual_intent="Animate edge AC relaxation and table update",
                required_state_changes=[{"entity": "dist_table", "operation": "update", "detail": "dist[C] = 4"}],
                verification_references=["trace_step_1"],
            ),
            TeachingBeat(
                beat_id="beat_1_4",
                pedagogical_objective="Expose failure mode with negative edge weights",
                teaching_move=TeachingMove.COUNTEREXAMPLE_EXPOSURE.value,
                concept_addressed="Negative weight cycle failure",
                visual_intent="Introduce -5 edge and show contradictory recalculation",
                verification_references=["counterexample_1"],
            ),
        ]
        is_valid, errors = BeatValidator.validate_beats(beats)
        assert is_valid is True
        assert len(errors) == 0
        assert BeatValidator.has_heterogeneous_moves(beats) is True

    def test_beat_missing_required_fields_fails_validation(self):
        incomplete_beat = TeachingBeat(
            beat_id="beat_bad",
            pedagogical_objective="",  # missing
            teaching_move="formal_definition",
            concept_addressed="concept",
            visual_intent="visual",
        )
        is_valid, errors = BeatValidator.validate_beats([incomplete_beat])
        assert is_valid is False
        assert any("missing mandatory 'pedagogical_objective'" in e for e in errors)

    def test_duplicate_beat_id_fails_validation(self):
        beats = [
            TeachingBeat("b1", "Obj 1", "formal_definition", "C1", "V1"),
            TeachingBeat("b1", "Obj 2", "intuition_bridge", "C2", "V2"),  # Duplicate ID
        ]
        is_valid, errors = BeatValidator.validate_beats(beats)
        assert is_valid is False
        assert any("Duplicate beat_id detected: 'b1'" in e for e in errors)

    def test_heterogeneous_moves_detected(self):
        uniform_beats = [
            TeachingBeat("b1", "Obj 1", "formal_definition", "C1", "V1"),
            TeachingBeat("b2", "Obj 2", "formal_definition", "C2", "V2"),
        ]
        assert BeatValidator.has_heterogeneous_moves(uniform_beats) is False

        diverse_beats = [
            TeachingBeat("b1", "Obj 1", "formal_definition", "C1", "V1"),
            TeachingBeat("b2", "Obj 2", "mechanistic_trace", "C2", "V2"),
        ]
        assert BeatValidator.has_heterogeneous_moves(diverse_beats) is True


# ============================================================================
# 5. Four-Tier Verification Routing
# ============================================================================

class TestFourTierVerificationRouting:
    """Verifies that VERIFIED, FAILED, UNVERIFIED, and NOT_APPLICABLE remain distinct."""

    def test_verified_step_routes_cleanly(self):
        claims = [
            ClaimAuditRecord(
                claim_id="c1",
                section_id="section_1",
                beat_id="b1",
                statement="k1 = 0.0001087 s^-1",
                status=FourTierStatus.VERIFIED,
                criticality=ClaimCriticality.CRITICAL,
            )
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Reaction Kinetics",
            section_ids=["section_1"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.overall_status == FourTierStatus.VERIFIED
        assert prov.pipeline_halted is False
        assert "section_1" in prov.safe_sections

    def test_failed_step_blocks_affected_section_leaving_unaffected_safe(self):
        claims = [
            ClaimAuditRecord(
                claim_id="c1",
                section_id="section_1",
                beat_id="b1",
                statement="Definition valid",
                status=FourTierStatus.VERIFIED,
            ),
            ClaimAuditRecord(
                claim_id="c2",
                section_id="section_2",
                beat_id="b2",
                statement="2 + 2 = 5 contradiction",
                status=FourTierStatus.FAILED,
                criticality=ClaimCriticality.CRITICAL,
            ),
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Arithmetic",
            section_ids=["section_1", "section_2"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.overall_status == FourTierStatus.FAILED
        assert "section_2" in prov.blocked_sections
        assert "section_1" in prov.safe_sections  # Section 1 is preserved and not unnecessarily blocked

    def test_unverified_never_silently_promoted_to_verified(self):
        claims = [
            ClaimAuditRecord(
                claim_id="c1",
                section_id="section_1",
                beat_id="b1",
                statement="Heuristic observation about search pattern",
                status=FourTierStatus.UNVERIFIED,
                criticality=ClaimCriticality.NON_CRITICAL,
            )
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Search Heuristics",
            section_ids=["section_1"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.overall_status == FourTierStatus.UNVERIFIED
        assert prov.overall_status != FourTierStatus.VERIFIED  # Strict invariant

    def test_not_applicable_bypasses_cleanly(self):
        prov = VerificationPolicyManager.process_verification_results(
            topic="Photosynthesis",
            section_ids=["section_1", "section_2"],
            claims=[],
            is_computational_topic=False,
        )
        assert prov.overall_status == FourTierStatus.NOT_APPLICABLE
        assert prov.pipeline_halted is False
        assert len(prov.safe_sections) == 2


# ============================================================================
# 6. Critical vs Non-Critical UNVERIFIED Handling
# ============================================================================

class TestCriticalVsNonCriticalUnverified:
    """Verifies that non-critical unverified details proceed while critical violations halt."""

    def test_non_critical_unverified_detail_proceeds_with_audit_tags(self):
        claims = [
            ClaimAuditRecord(
                claim_id="c1",
                section_id="section_1",
                beat_id="b1",
                statement="Typical enzyme active site diameter is ~2nm",
                status=FourTierStatus.UNVERIFIED,
                criticality=ClaimCriticality.NON_CRITICAL,
            )
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Enzyme Kinetics",
            section_ids=["section_1"],
            claims=claims,
            is_computational_topic=False,
        )
        assert prov.overall_status == FourTierStatus.UNVERIFIED
        assert prov.pipeline_halted is False
        assert "section_1" in prov.safe_sections
        assert claims[0].remediation == UnverifiedRemediation.QUALIFY

    def test_critical_unverified_with_objective_violated_halts_pipeline(self):
        claims = [
            ClaimAuditRecord(
                claim_id="c_crit",
                section_id="section_3",
                beat_id="b3",
                statement="Core shortest distance claims 0 -> 4 -> 7",
                status=FourTierStatus.UNVERIFIED,
                criticality=ClaimCriticality.CRITICAL,
                essential_objective_violated=True,  # Cannot deliver lesson correctly without this
            )
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Dijkstra Algorithm",
            section_ids=["section_1", "section_2", "section_3"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.pipeline_halted is True
        assert prov.overall_status == FourTierStatus.FAILED
        assert claims[0].remediation == UnverifiedRemediation.HALT
        assert "section_3" in prov.blocked_sections

    def test_critical_unverified_remediated_via_qualification_proceeds(self):
        claims = [
            ClaimAuditRecord(
                claim_id="c_crit_remediated",
                section_id="section_1",
                beat_id="b1",
                statement="Empirically observed convergence rate in practice",
                status=FourTierStatus.UNVERIFIED,
                criticality=ClaimCriticality.CRITICAL,
                essential_objective_violated=False,  # Can be safely qualified
                remediation=UnverifiedRemediation.QUALIFY,
            )
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Optimization",
            section_ids=["section_1"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.pipeline_halted is False
        assert prov.overall_status == FourTierStatus.UNVERIFIED

    def test_failed_claim_with_essential_objective_violated_halts_globally(self):
        """Fix B: FAILED claim with essential_objective_violated must halt pipeline globally."""
        claims = [
            ClaimAuditRecord(
                claim_id="c_fail_crit",
                section_id="section_1",
                beat_id="b1",
                statement="Essential proof invariant contradicted",
                status=FourTierStatus.FAILED,
                criticality=ClaimCriticality.CRITICAL,
                essential_objective_violated=True,
            )
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Mathematical Proof",
            section_ids=["section_1", "section_2"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.pipeline_halted is True
        assert prov.overall_status == FourTierStatus.FAILED
        assert len(prov.safe_sections) == 0

    def test_unverified_claim_with_essential_objective_violated_halts_globally(self):
        """Fix B: UNVERIFIED claim with essential_objective_violated must halt pipeline globally."""
        claims = [
            ClaimAuditRecord(
                claim_id="c_unv_crit",
                section_id="section_2",
                beat_id="b2",
                statement="Core numerical identity cannot be verified",
                status=FourTierStatus.UNVERIFIED,
                criticality=ClaimCriticality.CRITICAL,
                essential_objective_violated=True,
            )
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Core Identity",
            section_ids=["section_1", "section_2"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.pipeline_halted is True
        assert prov.overall_status == FourTierStatus.FAILED
        assert claims[0].remediation == UnverifiedRemediation.HALT
        assert len(prov.safe_sections) == 0

    def test_failed_claim_without_essential_objective_violated_isolates_section(self):
        """Fix B: FAILED claim without essential_objective_violated blocks section without halting pipeline if safe sections exist."""
        claims = [
            ClaimAuditRecord(
                claim_id="c_fail_isolated",
                section_id="section_1",
                beat_id="b1",
                statement="Side computation discrepancy in supplementary example",
                status=FourTierStatus.FAILED,
                criticality=ClaimCriticality.NON_CRITICAL,
                essential_objective_violated=False,
            )
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Supplementary Calculation",
            section_ids=["section_1", "section_2"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.pipeline_halted is False
        assert prov.overall_status == FourTierStatus.FAILED
        assert prov.blocked_sections == ["section_1"]
        assert prov.safe_sections == ["section_2"]

    def test_unverified_claim_without_essential_objective_violated_does_not_halt(self):
        """Fix B: UNVERIFIED claim without essential_objective_violated qualifies/isolates and does not halt."""
        claims = [
            ClaimAuditRecord(
                claim_id="c_unv_safe",
                section_id="section_1",
                beat_id="b1",
                statement="Descriptive historical note",
                status=FourTierStatus.UNVERIFIED,
                criticality=ClaimCriticality.NON_CRITICAL,
                essential_objective_violated=False,
            )
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Historical Context",
            section_ids=["section_1", "section_2"],
            claims=claims,
            is_computational_topic=False,
        )
        assert prov.pipeline_halted is False
        assert prov.overall_status == FourTierStatus.UNVERIFIED
        assert "section_1" in prov.safe_sections
        assert "section_2" in prov.safe_sections


# ============================================================================
# 7. Verification Provenance Preservation
# ============================================================================

class TestVerificationProvenancePreservation:
    """Verifies that verification provenance survives serialization and roundtrips."""

    def test_provenance_roundtrips_losslessly(self):
        prov = VerificationProvenance(
            topic="Calculus Limits",
            overall_status=FourTierStatus.VERIFIED,
            claims=[
                ClaimAuditRecord(
                    claim_id="c1",
                    section_id="section_1",
                    beat_id="b1",
                    statement="lim x->0 (sin x)/x = 1",
                    status=FourTierStatus.VERIFIED,
                    criticality=ClaimCriticality.CRITICAL,
                )
            ],
            blocked_sections=[],
            safe_sections=["section_1", "section_2"],
            pipeline_halted=False,
            halt_reason="",
        )
        prov_dict = prov.to_dict()
        reconstructed = VerificationProvenance.from_dict(prov_dict)
        assert reconstructed.topic == "Calculus Limits"
        assert reconstructed.overall_status == FourTierStatus.VERIFIED
        assert len(reconstructed.claims) == 1
        assert reconstructed.claims[0].claim_id == "c1"
        assert reconstructed.safe_sections == ["section_1", "section_2"]


# ============================================================================
# 8. TeachingOutline Serialization & Backward Compatibility
# ============================================================================

class TestTeachingOutlineSerializationAndCompatibility:
    """Verifies that TeachingOutline handles new fields while preserving 100% legacy compatibility."""

    def test_teaching_outline_roundtrip_with_new_fields(self):
        profile_dict = {
            "state_evolution": "discrete_state_machine",
            "formalism_level": "rigorous_mathematical",
            "computation_requirement": "deterministic_trace",
            "primary_visual_mode": "network_graph",
            "rationale_and_evidence": "Algorithmic state transitions",
            "profile_status": "CLASSIFIED",
            "confidence": 0.95,
        }
        beat_dict = {
            "beat_id": "sec1_beat1",
            "pedagogical_objective": "Explain relaxation",
            "teaching_move": "formal_definition",
            "concept_addressed": "Triangle inequality",
            "visual_intent": "Show graph nodes",
            "required_state_changes": [{"entity": "node_C", "operation": "update"}],
            "verification_references": ["eq_1"],
        }
        outline = TeachingOutline(
            topic="Dijkstra's Algorithm",
            learning_level="intermediate",
            topic_profile=profile_dict,
            sections=[
                {
                    "id": "section_1",
                    "title": "Introduction",
                    "beats": [beat_dict],
                }
            ],
        )

        # Check helper methods
        assert outline.is_computational() is True
        prof = outline.get_topic_profile()
        assert prof is not None
        assert prof.primary_visual_mode == PrimaryVisualMode.NETWORK_GRAPH

        beats = outline.get_all_beats()
        assert len(beats) == 1
        assert beats[0].beat_id == "sec1_beat1"

        is_valid, errs = outline.validate_beats()
        assert is_valid is True

        # Serialization roundtrip
        out_dict = outline.to_dict()
        assert "topic_profile" in out_dict
        assert out_dict["topic_profile"]["primary_visual_mode"] == "network_graph"

        reloaded = TeachingOutline.from_dict(out_dict)
        assert reloaded.topic == "Dijkstra's Algorithm"
        assert reloaded.get_topic_profile().confidence == 0.95
        assert len(reloaded.get_all_beats()) == 1

    def test_legacy_outline_fixture_backward_compatibility(self):
        legacy_data = {
            "topic": "Legacy Pythagorean Theorem",
            "target_audience": "high school students",
            "learning_level": "beginner",
            "formal_definition": "In a right triangle, a^2 + b^2 = c^2.",
            "sections": [
                {"id": "section_1", "title": "Overview", "lecture_lines": ["Line 1"]}
            ],
            # No topic_profile, no beats, no conceptual_breakdown
        }
        outline = TeachingOutline.from_dict(legacy_data)
        assert outline.topic == "Legacy Pythagorean Theorem"
        assert outline.topic_profile is None
        assert outline.get_all_beats() == []
        is_valid, errs = outline.validate_beats()
        assert is_valid is True  # Empty beats is safe for legacy outlines

        d = outline.to_dict()
        assert d["topic"] == "Legacy Pythagorean Theorem"
        assert d["formal_definition"] == "In a right triangle, a^2 + b^2 = c^2."


# ============================================================================
# 9. Non-Computational Topic Bypass (No False Arithmetic / Missing Variable Errors)
# ============================================================================

class TestNonComputationalTopicBypass:
    """Verifies that qualitative/biological topics bypass arithmetic AST evaluators cleanly."""

    def test_photosynthesis_bypasses_ast_evaluator_cleanly(self):
        verifier = WorkedExampleVerifier()
        # Simulated outline for Photosynthesis with stoichiometric prose
        outline_data = {
            "topic": "Photosynthesis: Light Reactions and Calvin Cycle",
            "topic_profile": {
                "state_evolution": "multi_stage_process",
                "formalism_level": "mechanistic_empirical",
                "computation_requirement": "none_not_applicable",
                "primary_visual_mode": "process_flowchart",
                "rationale_and_evidence": "Biochemical pathway without arithmetic evaluation.",
                "profile_status": "CLASSIFIED",
                "confidence": 0.95,
            },
            "worked_example_spec": {
                "setup": "Calvin Cycle stoichiometric balance",
                "inputs": {"CO2": 6, "RuBP": 6},
                "steps": [
                    {
                        "step": "Carbon fixation via RuBisCO",
                        "step_type": "conceptual_explanation",
                        "result": "6 CO2 + 6 RuBP form 12 3-PGA molecules",
                    },
                    {
                        "step": "Determine glucose yield",
                        "step_type": "conceptual_explanation",
                        "result": "12 ATP / 6 ATP per glucose = 2 glucose molecules",  # Contained 'per' which caused the historical failure
                    },
                ],
                "final_result": "1 net glucose synthesized per 6 CO2 fixed",
            },
        }

        updated_outline, report = verifier.verify_outline(outline_data)
        assert report.overall_status == VerificationStatus.NOT_APPLICABLE
        assert report.is_safe_for_stage2 is True
        assert report.failed_steps == 0
        assert report.review_steps == 0  # Zero MISSING_VARIABLE errors
        assert "NOT_APPLICABLE" in report.summary

    def test_step_adapter_handles_per_phrase_as_conceptual(self):
        step_dict = {
            "step": "Calculate ATP consumption",
            "result": "12 ATP / 6 ATP per glucose = 2 glucose",
        }
        adapted = CalculationStepAdapter.adapt_step(step_dict, index=0)
        assert adapted.step_type == StepType.CONCEPTUAL_EXPLANATION
        assert adapted.is_evaluable is False


# ============================================================================
# 10. Secondary Profile Fields Validation (Gap D)
# ============================================================================

class TestSecondaryProfileFieldsValidation:
    """Verifies schema validation and semantic consistency for secondary profile fields."""

    def test_valid_secondary_fields_pass(self):
        payload = {
            "state_evolution": "multi_stage_process",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "process_flowchart",
            "rationale_and_evidence": "Valid biological mechanism with secondary visual representation.",
            "profile_status": "CLASSIFIED",
            "confidence": 0.95,
            "secondary_visual_modes": ["molecular_spatial", "structured_concept_card"],
            "secondary_characteristics": ["cyclic_pathway", "membrane_transport"],
        }
        is_valid, errors = TopicProfiler.validate_schema(payload)
        assert is_valid is True
        assert len(errors) == 0

        prof = TopicProfiler.profile_from_data(payload)
        assert prof.profile_status == ProfileStatus.CLASSIFIED
        assert prof.secondary_visual_modes == ["molecular_spatial", "structured_concept_card"]
        assert prof.secondary_characteristics == ["cyclic_pathway", "membrane_transport"]

    def test_invalid_secondary_visual_mode_produces_unclassified(self):
        payload = {
            "state_evolution": "multi_stage_process",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "process_flowchart",
            "rationale_and_evidence": "Biological pathway.",
            "secondary_visual_modes": ["invalid_mode_xyz"],
        }
        is_valid, errors = TopicProfiler.validate_schema(payload)
        assert is_valid is False
        assert any("Invalid secondary_visual_mode: 'invalid_mode_xyz'" in e for e in errors)

        prof = TopicProfiler.profile_from_data(payload)
        assert prof.profile_status == ProfileStatus.UNCLASSIFIED
        assert prof.confidence == 0.0

    def test_non_list_secondary_fields_rejected(self):
        payload_bad_vm = {
            "state_evolution": "multi_stage_process",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "process_flowchart",
            "rationale_and_evidence": "Biological pathway.",
            "secondary_visual_modes": "not_a_list",
        }
        is_valid, errors = TopicProfiler.validate_schema(payload_bad_vm)
        assert is_valid is False
        assert any("Field 'secondary_visual_modes' must be a list of strings" in e for e in errors)

        payload_bad_char = {
            "state_evolution": "multi_stage_process",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "process_flowchart",
            "rationale_and_evidence": "Biological pathway.",
            "secondary_characteristics": 12345,
        }
        is_valid, errors = TopicProfiler.validate_schema(payload_bad_char)
        assert is_valid is False
        assert any("Field 'secondary_characteristics' must be a list of strings" in e for e in errors)

    def test_empty_string_secondary_characteristic_rejected(self):
        payload = {
            "state_evolution": "multi_stage_process",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "process_flowchart",
            "rationale_and_evidence": "Biological pathway.",
            "secondary_characteristics": ["valid_characteristic", ""],
        }
        is_valid, errors = TopicProfiler.validate_schema(payload)
        assert is_valid is False
        assert any("Invalid secondary_characteristic: item must be a non-empty string" in e for e in errors)

    def test_absent_secondary_fields_default_gracefully(self):
        payload = {
            "state_evolution": "multi_stage_process",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "process_flowchart",
            "rationale_and_evidence": "Minimal valid profile without optional secondary fields.",
        }
        is_valid, errors = TopicProfiler.validate_schema(payload)
        assert is_valid is True
        prof = TopicProfiler.profile_from_data(payload)
        assert prof.secondary_visual_modes == []
        assert prof.secondary_characteristics == []

    def test_secondary_state_table_contradicts_none_not_applicable(self):
        payload = {
            "state_evolution": "continuous_dynamic",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "coordinate_geometry",
            "rationale_and_evidence": "Contradictory secondary visual mode without computation.",
            "secondary_visual_modes": ["state_table_matrix"],
        }
        prof = TopicProfiler.profile_from_data(payload)
        assert prof.profile_status == ProfileStatus.UNCLASSIFIED
        assert prof.confidence == 0.0
        assert any("state_table_matrix" in note for note in prof.evidence_notes)

    def test_from_dict_malformed_secondary_visual_mode_returns_unclassified(self):
        """Improvement D: TopicProfile.from_dict directly rejects malformed secondary visual modes."""
        payload = {
            "state_evolution": "multi_stage_process",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "process_flowchart",
            "rationale_and_evidence": "Valid rationale.",
            "secondary_visual_modes": ["invalid_mode_not_in_enum"],
        }
        prof = TopicProfile.from_dict(payload)
        assert prof.profile_status == ProfileStatus.UNCLASSIFIED
        assert prof.confidence == 0.0
        assert any("Invalid secondary_visual_mode: 'invalid_mode_not_in_enum'" in n for n in prof.evidence_notes)

    def test_from_dict_malformed_secondary_characteristics_returns_unclassified(self):
        """Improvement D: TopicProfile.from_dict directly rejects non-string or empty secondary characteristics."""
        payload = {
            "state_evolution": "multi_stage_process",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "process_flowchart",
            "rationale_and_evidence": "Valid rationale.",
            "secondary_characteristics": ["valid_tag", ""],
        }
        prof = TopicProfile.from_dict(payload)
        assert prof.profile_status == ProfileStatus.UNCLASSIFIED
        assert prof.confidence == 0.0
        assert any("Invalid secondary_characteristic" in n for n in prof.evidence_notes)

    def test_from_dict_non_list_secondary_fields_returns_unclassified(self):
        """Improvement D: TopicProfile.from_dict directly rejects non-list secondary fields."""
        payload = {
            "state_evolution": "multi_stage_process",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "process_flowchart",
            "rationale_and_evidence": "Valid rationale.",
            "secondary_visual_modes": "not_a_list",
        }
        prof = TopicProfile.from_dict(payload)
        assert prof.profile_status == ProfileStatus.UNCLASSIFIED
        assert prof.confidence == 0.0

    def test_from_dict_valid_secondary_fields_preserved_losslessly(self):
        """Improvement D: Valid secondary fields round-trip cleanly without dropping values."""
        payload = {
            "state_evolution": "multi_stage_process",
            "formalism_level": "mechanistic_empirical",
            "computation_requirement": "none_not_applicable",
            "primary_visual_mode": "process_flowchart",
            "rationale_and_evidence": "Valid rationale.",
            "profile_status": "CLASSIFIED",
            "confidence": 0.95,
            "secondary_visual_modes": ["molecular_spatial"],
            "secondary_characteristics": ["cyclic_pathway"],
        }
        prof = TopicProfile.from_dict(payload)
        assert prof.profile_status == ProfileStatus.CLASSIFIED
        assert prof.confidence == 0.95
        assert prof.secondary_visual_modes == ["molecular_spatial"]
        assert prof.secondary_characteristics == ["cyclic_pathway"]


# ============================================================================
# 11. Beat Validation Acceptance Gate (Gap C)
# ============================================================================

class TestBeatValidationGate:
    """Verifies that malformed beats halt the outline gate while legacy outlines pass cleanly."""

    def test_malformed_beat_objective_blocks_stage2(self, tmp_path):
        agent = TeachingVideoAgent(1, knowledge_point="Sorting Algorithms", folder=str(tmp_path / "CASES"))
        agent.output_dir = tmp_path / "test_run"
        agent.output_dir.mkdir(parents=True, exist_ok=True)

        outline_data = {
            "topic": "Sorting Algorithms",
            "sections": [
                {
                    "id": "section_1",
                    "title": "Overview",
                    "beats": [
                        {
                            "beat_id": "b1",
                            "pedagogical_objective": "",  # Empty objective defect
                            "teaching_move": "formal_definition",
                            "concept_addressed": "Invariants",
                            "visual_intent": "Concept card",
                        }
                    ],
                }
            ],
        }
        with open(agent.output_dir / "outline.json", "w", encoding="utf-8") as f:
            json.dump(outline_data, f)

        outline = agent.generate_outline()
        assert agent.is_outline_safe_for_stage2() is False
        prov = agent.get_verification_provenance()
        assert prov is not None
        assert prov.pipeline_halted is True
        assert any("Beat validation defect" in c.statement for c in prov.claims)

    def test_duplicate_beat_id_blocks_stage2(self, tmp_path):
        agent = TeachingVideoAgent(1, knowledge_point="Graph Search", folder=str(tmp_path / "CASES"))
        agent.output_dir = tmp_path / "test_run"
        agent.output_dir.mkdir(parents=True, exist_ok=True)

        outline_data = {
            "topic": "Graph Search",
            "sections": [
                {
                    "id": "section_1",
                    "title": "Intro",
                    "beats": [
                        TeachingBeat("dup_b1", "Obj 1", "formal_definition", "C1", "V1").to_dict(),
                        TeachingBeat("dup_b1", "Obj 2", "intuition_bridge", "C2", "V2").to_dict(),  # Duplicate
                    ],
                }
            ],
        }
        with open(agent.output_dir / "outline.json", "w", encoding="utf-8") as f:
            json.dump(outline_data, f)

        outline = agent.generate_outline()
        assert agent.is_outline_safe_for_stage2() is False
        prov = agent.get_verification_provenance()
        assert prov is not None
        assert prov.pipeline_halted is True

    def test_legacy_outline_without_beats_passes_gate(self, tmp_path):
        agent = TeachingVideoAgent(1, knowledge_point="Legacy Topic", folder=str(tmp_path / "CASES"))
        agent.output_dir = tmp_path / "test_run"
        agent.output_dir.mkdir(parents=True, exist_ok=True)

        legacy_data = {
            "topic": "Legacy Topic",
            "sections": [
                {"id": "section_1", "title": "Overview", "lecture_lines": ["Line 1"]}
            ],
            "worked_example_spec": {"steps": []},
        }
        with open(agent.output_dir / "outline.json", "w", encoding="utf-8") as f:
            json.dump(legacy_data, f)

        outline = agent.generate_outline()
        assert agent.is_outline_safe_for_stage2() is True
        prov = agent.get_verification_provenance()
        assert prov is not None
        assert prov.pipeline_halted is False


# ============================================================================
# 12. Canonical Verification Lifecycle & Provenance Attachment (Major Issue A)
# ============================================================================

class TestCanonicalVerificationLifecycle:
    """Verifies that the canonical verification lifecycle runs through generate_outline() and is cached properly."""

    def test_normal_pipeline_attaches_provenance_and_persists_to_disk(self, tmp_path):
        agent = TeachingVideoAgent(1, knowledge_point="Binary Search", folder=str(tmp_path / "CASES"))
        agent.output_dir = tmp_path / "test_run"
        agent.output_dir.mkdir(parents=True, exist_ok=True)

        outline_data = {
            "topic": "Binary Search",
            "sections": [
                {
                    "id": "section_1",
                    "title": "Intro",
                    "worked_example_step": "Step 1",
                    "beats": [
                        TeachingBeat("b1", "Initialize pointers", "formal_definition", "Pointers", "Show array").to_dict()
                    ],
                }
            ],
            "worked_example_spec": {"steps": []},
        }
        with open(agent.output_dir / "outline.json", "w", encoding="utf-8") as f:
            json.dump(outline_data, f)

        # 1. Execute generate_outline() through normal pipeline entry point
        outline = agent.generate_outline()
        assert outline is not None

        # 2. Assert verification provenance is attached to outline in memory
        prov = agent.get_verification_provenance()
        assert prov is not None
        assert prov.topic == "Binary Search"
        assert prov.overall_status in (FourTierStatus.NOT_APPLICABLE, FourTierStatus.VERIFIED, FourTierStatus.UNVERIFIED)

        # 3. Assert provenance is persisted in outline.json on disk
        with open(agent.output_dir / "outline.json", "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        assert "verification_provenance" in disk_data
        assert disk_data["verification_provenance"]["overall_status"] == prov.overall_status.value

        # 4. Verify that is_outline_safe_for_stage2() passes and uses cached report without repeating verification
        assert agent.is_outline_safe_for_stage2() is True

        # Calling _ensure_outline_verified() with force=False returns cached report
        cached_report = agent._ensure_outline_verified(force=False)
        assert cached_report is agent.outline_verification_report


# ============================================================================
# 13. Section-Level Fault Isolation (Major Issue B)
# ============================================================================

class TestSectionLevelFaultIsolation:
    """Verifies section-level fault isolation, Stage 2 payload filtering, and essential objective halts."""

    def test_all_sections_safe_progression(self, tmp_path):
        agent = TeachingVideoAgent(1, knowledge_point="Reaction Kinetics", folder=str(tmp_path / "CASES"))
        agent.output_dir = tmp_path / "test_run"
        agent.output_dir.mkdir(parents=True, exist_ok=True)

        outline_data = {
            "topic": "Reaction Kinetics",
            "sections": [
                {"id": "section_1", "title": "Intro"},
                {"id": "section_2", "title": "Mechanism"},
            ],
            "worked_example_spec": {"steps": []},
        }
        with open(agent.output_dir / "outline.json", "w", encoding="utf-8") as f:
            json.dump(outline_data, f)

        agent.generate_outline()
        assert agent.is_outline_safe_for_stage2() is True

        s2_dict = agent.get_stage2_outline_dict()
        assert len(s2_dict["sections"]) == 2
        assert "excluded_sections" not in s2_dict or len(s2_dict["excluded_sections"]) == 0

    def test_non_essential_section_failure_filters_blocked_section_leaving_safe_sections(self, tmp_path):
        agent = TeachingVideoAgent(1, knowledge_point="Multi Section Lesson", folder=str(tmp_path / "CASES"))
        agent.output_dir = tmp_path / "test_run"
        agent.output_dir.mkdir(parents=True, exist_ok=True)

        outline = TeachingOutline(
            topic="Multi Section Lesson",
            sections=[
                {"id": "section_1", "title": "Safe Section 1", "worked_example_step": "Step 1"},
                {"id": "section_2", "title": "Failing Section 2", "worked_example_step": "Step 2"},
                {"id": "section_3", "title": "Safe Section 3", "worked_example_step": "Step 3"},
            ],
            worked_example_spec={
                "steps": [
                    {"step": "Step 1", "result": "10"},
                    {"step": "Step 2", "result": "2 + 2 = 5 contradiction"},
                    {"step": "Step 3", "result": "30"},
                ]
            },
        )
        agent.outline = outline

        # Construct claims: section_2 has a non-essential FAILED claim; section_1 and section_3 are VERIFIED
        claims = [
            ClaimAuditRecord("c1", "section_1", "b1", "Valid 1", FourTierStatus.VERIFIED),
            ClaimAuditRecord("c2", "section_2", "b2", "Calculation 2+2=5 error", FourTierStatus.FAILED, ClaimCriticality.NON_CRITICAL),
            ClaimAuditRecord("c3", "section_3", "b3", "Valid 3", FourTierStatus.VERIFIED),
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Multi Section Lesson",
            section_ids=["section_1", "section_2", "section_3"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.overall_status == FourTierStatus.FAILED
        assert prov.pipeline_halted is False
        assert prov.blocked_sections == ["section_2"]
        assert prov.safe_sections == ["section_1", "section_3"]

        agent.outline.extra_fields["verification_provenance"] = prov.to_dict()
        agent.outline_verification_report = WorkedExampleReport(
            overall_status=VerificationStatus.FAILED,
            total_steps=3,
            verified_steps=2,
            corrected_steps=0,
            failed_steps=1,
            review_steps=0,
            not_applicable_steps=0,
            is_safe_for_stage2=True,
        )

        # 1. Gate allows safe sections to proceed
        assert agent.is_outline_safe_for_stage2() is True

        # 2. Stage 2 input excludes section_2 and preserves section_1 and section_3
        s2_dict = agent.get_stage2_outline_dict()
        s2_ids = [s["id"] for s in s2_dict["sections"]]
        assert s2_ids == ["section_1", "section_3"]
        assert s2_dict["excluded_sections"] == ["section_2"]
        assert "section_2" in s2_dict["exclusion_reasons"]

        # 3. Canonical outline is NOT mutated
        canonical_ids = [s["id"] for s in agent.outline.sections]
        assert canonical_ids == ["section_1", "section_2", "section_3"]

    def test_all_sections_blocked_halts_before_stage2(self, tmp_path):
        agent = TeachingVideoAgent(1, knowledge_point="Total Failure", folder=str(tmp_path / "CASES"))
        agent.output_dir = tmp_path / "test_run"
        agent.output_dir.mkdir(parents=True, exist_ok=True)

        agent.outline = TeachingOutline(
            topic="Total Failure",
            sections=[{"id": "section_1", "title": "Single Failing Section"}],
        )
        claims = [
            ClaimAuditRecord("c1", "section_1", "b1", "Error", FourTierStatus.FAILED, ClaimCriticality.NON_CRITICAL)
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Total Failure",
            section_ids=["section_1"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.pipeline_halted is True
        assert prov.safe_sections == []

        agent.outline.extra_fields["verification_provenance"] = prov.to_dict()
        agent.outline_verification_report = WorkedExampleReport(
            overall_status=VerificationStatus.FAILED,
            total_steps=1,
            verified_steps=0,
            corrected_steps=0,
            failed_steps=1,
            review_steps=0,
            not_applicable_steps=0,
            is_safe_for_stage2=False,
        )
        assert agent.is_outline_safe_for_stage2() is False

    def test_unlocalized_failed_claim_with_empty_section_id_halts_pipeline(self):
        """Fix A: FAILED claim with empty section_id cannot be localized and must halt pipeline globally."""
        claims = [
            ClaimAuditRecord("c1", "section_1", "b1", "Valid intro", FourTierStatus.VERIFIED),
            ClaimAuditRecord("c2", "", "b2", "Unmapped worked example calculation failure", FourTierStatus.FAILED, ClaimCriticality.NON_CRITICAL),
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Unlocalized Failure Test",
            section_ids=["section_1", "section_2"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.pipeline_halted is True
        assert prov.overall_status == FourTierStatus.FAILED
        assert len(prov.safe_sections) == 0
        assert any("Unlocalized failed check" in r for r in prov.halt_reason.split("; "))

    def test_unlocalized_failed_claim_with_unresolvable_section_id_halts_pipeline(self):
        """Fix A: FAILED claim with unresolvable/hallucinated section_id must fail closed and halt globally."""
        claims = [
            ClaimAuditRecord("c1", "section_1", "b1", "Valid intro", FourTierStatus.VERIFIED),
            ClaimAuditRecord("c2", "section_nonexistent", "b2", "Failed calculation in unknown section", FourTierStatus.FAILED),
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Hallucinated Section Test",
            section_ids=["section_1", "section_2"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.pipeline_halted is True
        assert prov.overall_status == FourTierStatus.FAILED
        assert len(prov.safe_sections) == 0

    def test_get_stage2_outline_dict_clears_sections_when_pipeline_halted(self, tmp_path):
        """Fix A: get_stage2_outline_dict forwards zero sections when pipeline is halted."""
        agent = TeachingVideoAgent(1, knowledge_point="Halted Pipeline Test", folder=str(tmp_path / "CASES"))
        agent.output_dir = tmp_path / "test_run"
        agent.output_dir.mkdir(parents=True, exist_ok=True)

        agent.outline = TeachingOutline(
            topic="Halted Pipeline Test",
            sections=[
                {"id": "section_1", "title": "Section 1"},
                {"id": "section_2", "title": "Section 2"},
            ],
        )
        claims = [
            ClaimAuditRecord("c1", "", "b1", "Unlocalized calculation failure", FourTierStatus.FAILED)
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Halted Pipeline Test",
            section_ids=["section_1", "section_2"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.pipeline_halted is True

        agent.outline.extra_fields["verification_provenance"] = prov.to_dict()
        agent.outline_verification_report = WorkedExampleReport(
            overall_status=VerificationStatus.FAILED,
            total_steps=1,
            verified_steps=0,
            corrected_steps=0,
            failed_steps=1,
            review_steps=0,
            not_applicable_steps=0,
            is_safe_for_stage2=False,
        )

        assert agent.is_outline_safe_for_stage2() is False
        s2_dict = agent.get_stage2_outline_dict()
        assert s2_dict["sections"] == []
        assert "section_1" in s2_dict["excluded_sections"]
        assert "section_2" in s2_dict["excluded_sections"]

    def test_get_stage2_outline_dict_excludes_sections_with_missing_ids(self, tmp_path):
        """Fix A: Sections with missing or empty IDs cannot be forwarded as safe sections."""
        agent = TeachingVideoAgent(1, knowledge_point="Missing ID Test", folder=str(tmp_path / "CASES"))
        agent.output_dir = tmp_path / "test_run"
        agent.output_dir.mkdir(parents=True, exist_ok=True)

        agent.outline = TeachingOutline(
            topic="Missing ID Test",
            sections=[
                {"id": "section_1", "title": "Valid Section 1"},
                {"title": "Section Without ID"},  # Missing ID
            ],
        )
        claims = [
            ClaimAuditRecord("c1", "section_1", "b1", "Valid intro", FourTierStatus.VERIFIED)
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Missing ID Test",
            section_ids=["section_1"],
            claims=claims,
            is_computational_topic=False,
        )
        agent.outline.extra_fields["verification_provenance"] = prov.to_dict()

        s2_dict = agent.get_stage2_outline_dict()
        assert len(s2_dict["sections"]) == 1
        assert s2_dict["sections"][0]["id"] == "section_1"
        assert any("missing_id" in ex for ex in s2_dict["excluded_sections"])

    def test_resolve_step_section_id_avoids_arbitrary_mapping_and_returns_empty(self):
        """Fix A: _resolve_step_section_id returns empty string when unambiguous mapping cannot be derived."""
        sections = [
            {"id": "section_1", "title": "Conceptual Overview", "pedagogical_role": "formal_definition_and_purpose"},
            {"id": "section_2", "title": "Intuitive Analogy", "pedagogical_role": "intuitive_interpretation"},
        ]
        # Step 0 has no worked_example_step declared, no beats, no title mention, and no demonstration role
        resolved = TeachingVideoAgent._resolve_step_section_id(sections, step_idx=0)
        assert resolved == ""  # Must NOT arbitrarily map to section_1!

        # Step matching explicit worked_example_step maps reliably
        sections_with_wes = [
            {"id": "sec_demo", "worked_example_step": "Step 1: finding mid"},
        ]
        assert TeachingVideoAgent._resolve_step_section_id(sections_with_wes, step_idx=0) == "sec_demo"

    def test_critical_objective_violated_halts_even_if_another_section_is_safe(self, tmp_path):
        agent = TeachingVideoAgent(1, knowledge_point="Critical Failure", folder=str(tmp_path / "CASES"))
        agent.output_dir = tmp_path / "test_run"
        agent.output_dir.mkdir(parents=True, exist_ok=True)

        agent.outline = TeachingOutline(
            topic="Critical Failure",
            sections=[
                {"id": "section_1", "title": "Safe Section"},
                {"id": "section_2", "title": "Essential Theory Section"},
            ],
        )
        claims = [
            ClaimAuditRecord("c1", "section_1", "b1", "Valid intro", FourTierStatus.VERIFIED),
            ClaimAuditRecord("c2", "section_2", "b2", "Essential invariant unverified and cannot be qualified",
                             FourTierStatus.UNVERIFIED, ClaimCriticality.CRITICAL, essential_objective_violated=True),
        ]
        prov = VerificationPolicyManager.process_verification_results(
            topic="Critical Failure",
            section_ids=["section_1", "section_2"],
            claims=claims,
            is_computational_topic=True,
        )
        assert prov.pipeline_halted is True

        agent.outline.extra_fields["verification_provenance"] = prov.to_dict()
        agent.outline_verification_report = WorkedExampleReport(
            overall_status=VerificationStatus.UNVERIFIED,
            total_steps=2,
            verified_steps=1,
            corrected_steps=0,
            failed_steps=0,
            review_steps=0,
            not_applicable_steps=0,
            unverified_steps=1,
            is_safe_for_stage2=False,
        )
        assert agent.is_outline_safe_for_stage2() is False


# ============================================================================
# 10. Live Model Integration Tests (Configured Qwen 32B Model)
# ============================================================================

@pytest.mark.llm_integration
class TestLiveModelEvaluationMilestone1:
    """
    Live model integration tests using configured Qwen 32B model on 3 canonical topics:
    - Photosynthesis: Light Reactions and Calvin Cycle
    - Binary Search
    - Pythagorean Theorem
    """

    @pytest.fixture(autouse=True)
    def verify_configured_qwen32b(self):
        """Enforces that configured Qwen 32B model is reachable; fails closed if unavailable."""
        from src.llm import get_llm
        try:
            provider = get_llm(stage="outline")
            model_name = getattr(provider.config, "model", "")
            base_url = getattr(provider.config, "base_url", "")
            if "32b" not in model_name.lower():
                pytest.fail(f"BLOCKED: Configured model must be Qwen 32B, but got '{model_name}'")
            # Verify connectivity
            import urllib.request
            req = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                if not any("32b" in m.lower() for m in models):
                    pytest.fail(f"BLOCKED: Model 'qwen2.5-coder:32b' not found in Ollama endpoint models: {models}")
        except Exception as e:
            pytest.fail(f"BLOCKED: Configured Qwen 32B model is unreachable: {e}")

    def test_live_stage1_photosynthesis(self, tmp_path):
        """Validates biological multi-stage process profiling, heterogeneous beats, and NOT_APPLICABLE bypass."""
        from prompts.stage1 import get_prompt1_outline
        from src.llm import generate_response

        topic = "Photosynthesis: Light Reactions and Calvin Cycle"
        prompt = get_prompt1_outline(topic, duration=4.5, level="intermediate")
        resp = generate_response(prompt, stage="outline", max_tokens=6000)

        assert resp.text and len(resp.text) > 100
        # Parse JSON
        from src.agent import extract_json_from_markdown, safe_json_loads
        content = extract_json_from_markdown(resp.text)
        data = safe_json_loads(content)
        assert isinstance(data, dict)

        # 1. Profile assertions
        assert "topic_profile" in data
        prof_data = data["topic_profile"]
        profile = TopicProfiler.profile_from_data(prof_data, context={"topic": topic})
        assert profile.profile_status == ProfileStatus.CLASSIFIED
        assert profile.state_evolution in (StateEvolution.MULTI_STAGE_PROCESS, StateEvolution.CONTINUOUS_DYNAMIC)
        assert profile.computation_requirement in (ComputationRequirement.NONE_NOT_APPLICABLE, ComputationRequirement.UNCERTAIN)

        # 2. Beat structure assertions
        outline = TeachingOutline.from_dict(data)
        beats = outline.get_all_beats()
        assert len(beats) >= 3, f"Expected at least 3 beats across sections, got {len(beats)}"
        assert BeatValidator.has_heterogeneous_moves(beats) is True

        # 3. Validation gate clearance
        verifier = WorkedExampleVerifier()
        updated_dict, report = verifier.verify_outline(outline.to_dict())
        assert report.overall_status in (VerificationStatus.NOT_APPLICABLE, VerificationStatus.VERIFIED, VerificationStatus.UNVERIFIED)
        assert report.is_safe_for_stage2 is True
        assert report.failed_steps == 0
        assert report.review_steps == 0

    def test_live_stage1_binary_search(self, tmp_path):
        """Validates discrete algorithmic state machine profiling, pointer convergence, and state beats."""
        from prompts.stage1 import get_prompt1_outline
        from src.llm import generate_response

        topic = "Binary Search"
        prompt = get_prompt1_outline(topic, duration=4.5, level="intermediate")
        resp = generate_response(prompt, stage="outline", max_tokens=6000)

        assert resp.text and len(resp.text) > 100
        from src.agent import extract_json_from_markdown, safe_json_loads
        content = extract_json_from_markdown(resp.text)
        data = safe_json_loads(content)
        assert isinstance(data, dict)

        # 1. Profile assertions
        assert "topic_profile" in data
        profile = TopicProfiler.profile_from_data(data["topic_profile"], context={"topic": topic})
        assert profile.profile_status == ProfileStatus.CLASSIFIED
        assert profile.state_evolution == StateEvolution.DISCRETE_STATE_MACHINE
        assert profile.computation_requirement == ComputationRequirement.DETERMINISTIC_TRACE

        # 2. Beat structure assertions
        outline = TeachingOutline.from_dict(data)
        beats = outline.get_all_beats()
        assert len(beats) >= 3
        assert BeatValidator.has_heterogeneous_moves(beats) is True

        # 3. Validation gate clearance
        verifier = WorkedExampleVerifier()
        updated_dict, report = verifier.verify_outline(outline.to_dict())
        assert report.is_safe_for_stage2 is True

    def test_live_stage1_pythagorean_theorem(self, tmp_path):
        """Validates deductive proof profiling and proof beats without forced numerical calculation."""
        from prompts.stage1 import get_prompt1_outline
        from src.llm import generate_response

        topic = "The Pythagorean Theorem"
        prompt = get_prompt1_outline(topic, duration=4.5, level="intermediate")
        resp = generate_response(prompt, stage="outline", max_tokens=6000)

        assert resp.text and len(resp.text) > 100
        from src.agent import extract_json_from_markdown, safe_json_loads
        content = extract_json_from_markdown(resp.text)
        data = safe_json_loads(content)
        assert isinstance(data, dict)

        # 1. Profile assertions
        assert "topic_profile" in data
        profile = TopicProfiler.profile_from_data(data["topic_profile"], context={"topic": topic})
        assert profile.profile_status == ProfileStatus.CLASSIFIED
        assert profile.formalism_level == FormalismLevel.RIGOROUS_MATHEMATICAL
        assert profile.computation_requirement in (ComputationRequirement.SYMBOLIC_IDENTITY, ComputationRequirement.NONE_NOT_APPLICABLE, ComputationRequirement.ARITHMETIC_CALCULATION)

        # 2. Beat structure assertions
        outline = TeachingOutline.from_dict(data)
        beats = outline.get_all_beats()
        assert len(beats) >= 3
        assert BeatValidator.has_heterogeneous_moves(beats) is True

        # 3. Validation gate clearance
        verifier = WorkedExampleVerifier()
        updated_dict, report = verifier.verify_outline(outline.to_dict())
        assert report.is_safe_for_stage2 is True
