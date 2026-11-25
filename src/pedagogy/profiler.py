"""
Multidimensional Topic Profiler for CineLearn.

Implements typed multidimensional topic profiling with deterministic validation,
semantic consistency auditing beyond schema checks, overlapping topic characteristic
support, and explicit UNCLASSIFIED fallback preserving uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class StateEvolution(str, Enum):
    """How entities and states evolve throughout the instructional lesson."""
    DISCRETE_STATE_MACHINE = "discrete_state_machine"
    CONTINUOUS_DYNAMIC = "continuous_dynamic"
    MULTI_STAGE_PROCESS = "multi_stage_process"
    STATIC_STRUCTURAL = "static_structural"
    CONCEPTUAL_AXIOMATIC = "conceptual_axiomatic"
    UNCERTAIN = "uncertain"


class FormalismLevel(str, Enum):
    """The depth of mathematical, empirical, or deductive rigor required."""
    RIGOROUS_MATHEMATICAL = "rigorous_mathematical"
    MECHANISTIC_EMPIRICAL = "mechanistic_empirical"
    SYSTEM_STRUCTURAL = "system_structural"
    QUALITATIVE_INTUITIVE = "qualitative_intuitive"
    UNCERTAIN = "uncertain"


class ComputationRequirement(str, Enum):
    """Authoritative computation and simulation grounding requirements."""
    DETERMINISTIC_TRACE = "deterministic_trace"
    ARITHMETIC_CALCULATION = "arithmetic_calculation"
    SYMBOLIC_IDENTITY = "symbolic_identity"
    NONE_NOT_APPLICABLE = "none_not_applicable"
    UNCERTAIN = "uncertain"


class PrimaryVisualMode(str, Enum):
    """The predominant visual paradigm best suited for explaining the topic."""
    NETWORK_GRAPH = "network_graph"
    COORDINATE_GEOMETRY = "coordinate_geometry"
    STATE_TABLE_MATRIX = "state_table_matrix"
    MOLECULAR_SPATIAL = "molecular_spatial"
    PROCESS_FLOWCHART = "process_flowchart"
    STRUCTURED_CONCEPT_CARD = "structured_concept_card"
    UNCERTAIN = "uncertain"


class ProfileStatus(str, Enum):
    """Classification status: verified classified vs explicit uncertainty fallback."""
    CLASSIFIED = "CLASSIFIED"
    UNCLASSIFIED = "UNCLASSIFIED"


@dataclass
class TopicProfile:
    """Typed multidimensional profile capturing inherent characteristics of a topic."""
    state_evolution: StateEvolution
    formalism_level: FormalismLevel
    computation_requirement: ComputationRequirement
    primary_visual_mode: PrimaryVisualMode
    rationale_and_evidence: str
    profile_status: ProfileStatus = ProfileStatus.CLASSIFIED
    confidence: float = 1.0

    # Overlapping characteristics support
    secondary_visual_modes: List[str] = field(default_factory=list)
    secondary_characteristics: List[str] = field(default_factory=list)
    evidence_notes: List[str] = field(default_factory=list)
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def is_computational(self) -> bool:
        """Returns True if the topic requires active numerical or trace computation."""
        return self.computation_requirement in (
            ComputationRequirement.DETERMINISTIC_TRACE,
            ComputationRequirement.ARITHMETIC_CALCULATION,
        )

    def requires_trace(self) -> bool:
        """Returns True if the topic requires chronological state snapshot tracing."""
        return self.computation_requirement == ComputationRequirement.DETERMINISTIC_TRACE

    def is_unclassified(self) -> bool:
        """Returns True if profile is in an unclassified / uncertain fallback state."""
        return self.profile_status == ProfileStatus.UNCLASSIFIED

    def to_dict(self) -> Dict[str, Any]:
        """Lossless serialization to dictionary."""
        d = {
            "state_evolution": self.state_evolution.value,
            "formalism_level": self.formalism_level.value,
            "computation_requirement": self.computation_requirement.value,
            "primary_visual_mode": self.primary_visual_mode.value,
            "rationale_and_evidence": self.rationale_and_evidence,
            "profile_status": self.profile_status.value,
            "confidence": round(float(self.confidence), 3),
            "secondary_visual_modes": list(self.secondary_visual_modes),
            "secondary_characteristics": list(self.secondary_characteristics),
            "evidence_notes": list(self.evidence_notes),
        }
        if self.extra_fields:
            for k, v in self.extra_fields.items():
                if k not in d:
                    d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TopicProfile":
        """Reconstruct TopicProfile from dictionary, safely handling string enums."""
        if not isinstance(data, dict):
            return TopicProfiler.build_unclassified_fallback(
                rationale="Data provided is not a dictionary",
                evidence_notes=["Input data was not a valid dictionary"],
            )

        def _safe_enum(enum_cls, val, default):
            try:
                return enum_cls(str(val).strip().lower())
            except Exception:
                return default

        state_evo = _safe_enum(StateEvolution, data.get("state_evolution"), StateEvolution.UNCERTAIN)
        form_level = _safe_enum(FormalismLevel, data.get("formalism_level"), FormalismLevel.UNCERTAIN)
        comp_req = _safe_enum(ComputationRequirement, data.get("computation_requirement"), ComputationRequirement.UNCERTAIN)
        vis_mode = _safe_enum(PrimaryVisualMode, data.get("primary_visual_mode"), PrimaryVisualMode.UNCERTAIN)

        status_str = str(data.get("profile_status", "")).strip().upper()
        if status_str == "UNCLASSIFIED":
            status = ProfileStatus.UNCLASSIFIED
        elif status_str == "CLASSIFIED":
            status = ProfileStatus.CLASSIFIED
        else:
            status = ProfileStatus.UNCLASSIFIED if (
                state_evo == StateEvolution.UNCERTAIN
                or form_level == FormalismLevel.UNCERTAIN
                or comp_req == ComputationRequirement.UNCERTAIN
                or vis_mode == PrimaryVisualMode.UNCERTAIN
            ) else ProfileStatus.CLASSIFIED

        conf_raw = data.get("confidence", 1.0)
        try:
            conf = float(conf_raw)
            conf = max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            conf = 0.0

        known_keys = {
            "state_evolution", "formalism_level", "computation_requirement",
            "primary_visual_mode", "rationale_and_evidence", "profile_status",
            "confidence", "secondary_visual_modes", "secondary_characteristics",
            "evidence_notes",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        deserialization_errors: List[str] = []

        sec_vm_raw = data.get("secondary_visual_modes", [])
        sec_vm: List[str] = []
        if sec_vm_raw is not None and not isinstance(sec_vm_raw, list):
            deserialization_errors.append(f"Field 'secondary_visual_modes' must be a list of strings, got {type(sec_vm_raw).__name__}")
        elif isinstance(sec_vm_raw, list):
            valid_modes = {e.value for e in PrimaryVisualMode}
            for idx, x in enumerate(sec_vm_raw):
                if not isinstance(x, str):
                    deserialization_errors.append(f"Invalid secondary_visual_mode item at index {idx}: expected string, got {type(x).__name__}")
                else:
                    val = x.strip().lower()
                    if val in valid_modes:
                        sec_vm.append(val)
                    else:
                        deserialization_errors.append(f"Invalid secondary_visual_mode: '{x}'")

        sec_char_raw = data.get("secondary_characteristics", [])
        sec_char: List[str] = []
        if sec_char_raw is not None and not isinstance(sec_char_raw, list):
            deserialization_errors.append(f"Field 'secondary_characteristics' must be a list of strings, got {type(sec_char_raw).__name__}")
        elif isinstance(sec_char_raw, list):
            for idx, x in enumerate(sec_char_raw):
                if not isinstance(x, str) or not x.strip():
                    deserialization_errors.append(f"Invalid secondary_characteristic at index {idx}: non-empty string required, got '{x}'")
                else:
                    sec_char.append(x.strip())

        if deserialization_errors:
            existing_notes = list(data.get("evidence_notes", [])) if isinstance(data.get("evidence_notes"), list) else []
            combined_notes = existing_notes + deserialization_errors
            return TopicProfiler.build_unclassified_fallback(
                rationale=f"Deserialization schema validation failed: {'; '.join(deserialization_errors)}",
                evidence_notes=combined_notes,
                raw_data=data,
            )

        return cls(
            state_evolution=state_evo,
            formalism_level=form_level,
            computation_requirement=comp_req,
            primary_visual_mode=vis_mode,
            rationale_and_evidence=str(data.get("rationale_and_evidence", "")),
            profile_status=status,
            confidence=conf,
            secondary_visual_modes=sec_vm,
            secondary_characteristics=sec_char,
            evidence_notes=list(data.get("evidence_notes", [])),
            extra_fields=extra,
        )


class TopicProfiler:
    """
    Deterministic validation, consistency auditing, and fallback generation
    for multidimensional topic profiles.
    """

    ALGORITHMIC_KEYWORDS = {
        "algorithm", "pointer", "relaxation", "dijkstra", "knapsack",
        "dynamic programming", "binary search", "traversal", "graph search",
        "sorting", "recurrence", "queue", "heap", "stack", "memoization",
    }

    MATHEMATICAL_PROOF_KEYWORDS = {
        "theorem", "proof", "derivation", "pythagorean", "calculus",
        "derivative", "limit", "integral", "differentiable", "continuous",
        "axiom", "lemma", "corollary", "invariant",
    }

    BIOLOGICAL_PROCESS_KEYWORDS = {
        "photosynthesis", "calvin cycle", "cellular respiration",
        "biochemical", "chloroplast", "atp synthase", "thylakoid",
        "enzyme", "metabolism", "protein synthesis", "translation",
    }

    @staticmethod
    def validate_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validates schema bounds, types, enum memberships, and required fields.
        Returns (is_valid, error_list).
        """
        if not isinstance(data, dict):
            return False, ["Profile payload must be a JSON dictionary"]

        errors: List[str] = []
        required_fields = [
            "state_evolution",
            "formalism_level",
            "computation_requirement",
            "primary_visual_mode",
            "rationale_and_evidence",
        ]
        for rf in required_fields:
            if rf not in data or data[rf] is None or str(data[rf]).strip() == "":
                errors.append(f"Missing mandatory profile field: '{rf}'")

        # Validate enums
        if "state_evolution" in data and data["state_evolution"] is not None:
            val = str(data["state_evolution"]).strip().lower()
            if val not in {e.value for e in StateEvolution}:
                errors.append(f"Invalid state_evolution: '{val}'")

        if "formalism_level" in data and data["formalism_level"] is not None:
            val = str(data["formalism_level"]).strip().lower()
            if val not in {e.value for e in FormalismLevel}:
                errors.append(f"Invalid formalism_level: '{val}'")

        if "computation_requirement" in data and data["computation_requirement"] is not None:
            val = str(data["computation_requirement"]).strip().lower()
            if val not in {e.value for e in ComputationRequirement}:
                errors.append(f"Invalid computation_requirement: '{val}'")

        if "primary_visual_mode" in data and data["primary_visual_mode"] is not None:
            val = str(data["primary_visual_mode"]).strip().lower()
            if val not in {e.value for e in PrimaryVisualMode}:
                errors.append(f"Invalid primary_visual_mode: '{val}'")

        # Validate confidence bounds
        if "confidence" in data and data["confidence"] is not None:
            try:
                c = float(data["confidence"])
                if c < 0.0 or c > 1.0:
                    errors.append(f"Confidence {c} outside allowable bounds [0.0, 1.0]")
            except (ValueError, TypeError):
                errors.append("Confidence must be a numeric float")

        # Validate secondary_visual_modes
        if "secondary_visual_modes" in data and data["secondary_visual_modes"] is not None:
            sec_vm = data["secondary_visual_modes"]
            if not isinstance(sec_vm, list):
                errors.append("Field 'secondary_visual_modes' must be a list of strings")
            else:
                valid_modes = {e.value for e in PrimaryVisualMode}
                for item in sec_vm:
                    if not isinstance(item, str):
                        errors.append(f"Invalid secondary_visual_mode item: expected string, got {type(item).__name__}")
                    elif item.strip().lower() not in valid_modes:
                        errors.append(f"Invalid secondary_visual_mode: '{item}'")

        # Validate secondary_characteristics
        if "secondary_characteristics" in data and data["secondary_characteristics"] is not None:
            sec_char = data["secondary_characteristics"]
            if not isinstance(sec_char, list):
                errors.append("Field 'secondary_characteristics' must be a list of strings")
            else:
                for item in sec_char:
                    if not isinstance(item, str) or not item.strip():
                        errors.append(f"Invalid secondary_characteristic: item must be a non-empty string, got '{item}'")

        return len(errors) == 0, errors

    @classmethod
    def validate_semantic_consistency(
        cls,
        profile: TopicProfile,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validates semantic consistency beyond schema checks.
        Detects contradictions between dimensions, context keywords, and certainty invariants.
        Returns (is_consistent, contradiction_list).
        """
        contradictions: List[str] = []

        # Rule 1: Uncertainty Invariant
        # If any dimension is UNCERTAIN, the profile cannot claim high confidence or CLASSIFIED status.
        has_uncertain_dimension = (
            profile.state_evolution == StateEvolution.UNCERTAIN
            or profile.formalism_level == FormalismLevel.UNCERTAIN
            or profile.computation_requirement == ComputationRequirement.UNCERTAIN
            or profile.primary_visual_mode == PrimaryVisualMode.UNCERTAIN
        )
        if has_uncertain_dimension and profile.confidence > 0.3:
            contradictions.append(
                f"Contradiction: Profile has UNCERTAIN dimensions but claims confidence {profile.confidence:.2f}"
            )

        # Rule 2: Discrete State Machine vs No Computation Contradiction
        # A discrete state machine algorithm inherently evolves through discrete state updates.
        if (
            profile.state_evolution == StateEvolution.DISCRETE_STATE_MACHINE
            and profile.computation_requirement == ComputationRequirement.NONE_NOT_APPLICABLE
        ):
            contradictions.append(
                "Contradiction: state_evolution is discrete_state_machine but computation_requirement claims none_not_applicable"
            )

        # Rule 3: State Table Matrix vs No Computation Contradiction
        # A DP state table matrix visual mode requires matrix cell computation or recurrence.
        all_vis_modes = {profile.primary_visual_mode.value} | set(profile.secondary_visual_modes)
        if (
            PrimaryVisualMode.STATE_TABLE_MATRIX.value in all_vis_modes
            and profile.computation_requirement == ComputationRequirement.NONE_NOT_APPLICABLE
        ):
            contradictions.append(
                "Contradiction: visual modes include state_table_matrix but computation_requirement claims none_not_applicable"
            )

        # Rule 4: Contextual Contradictions (When topic context is provided)
        if context and isinstance(context, dict):
            topic_str = str(context.get("topic", "")).lower()
            outline_notes = str(context.get("outline_text", "")).lower()
            combined = f"{topic_str} {outline_notes}"

            # Check if topic is an explicit algorithm requiring discrete state traces
            is_algo = any(kw in combined for kw in cls.ALGORITHMIC_KEYWORDS)
            if is_algo:
                if profile.computation_requirement == ComputationRequirement.NONE_NOT_APPLICABLE:
                    contradictions.append(
                        f"Context contradiction: Topic mentions algorithmic constructs ('{topic_str}') but profile claims computation_requirement is none_not_applicable"
                    )

            # Check if topic is a biological process
            is_bio = any(kw in combined for kw in cls.BIOLOGICAL_PROCESS_KEYWORDS)
            if is_bio:
                # Biological pathways should not claim deterministic_trace as primary computation
                if profile.computation_requirement == ComputationRequirement.DETERMINISTIC_TRACE:
                    contradictions.append(
                        f"Context contradiction: Biological topic '{topic_str}' claimed computation_requirement: deterministic_trace without numerical algorithmic justification"
                    )

        # Rule 5: Empty rationale check
        if not profile.rationale_and_evidence or len(profile.rationale_and_evidence.strip()) < 5:
            contradictions.append(
                "Semantic violation: rationale_and_evidence must cite concrete technical topic characteristics"
            )

        return len(contradictions) == 0, contradictions

    @classmethod
    def build_unclassified_fallback(
        cls,
        rationale: str,
        evidence_notes: List[str],
        raw_data: Optional[Dict[str, Any]] = None,
    ) -> TopicProfile:
        """
        Constructs an explicit UNCLASSIFIED fallback profile with confidence 0.0.
        Preserves uncertainty without imposing an arbitrary visual or pedagogical strategy.
        """
        extra: Dict[str, Any] = {}
        if raw_data and isinstance(raw_data, dict):
            extra["raw_unclassified_input"] = raw_data

        return TopicProfile(
            state_evolution=StateEvolution.UNCERTAIN,
            formalism_level=FormalismLevel.UNCERTAIN,
            computation_requirement=ComputationRequirement.UNCERTAIN,
            primary_visual_mode=PrimaryVisualMode.UNCERTAIN,
            rationale_and_evidence=rationale,
            profile_status=ProfileStatus.UNCLASSIFIED,
            confidence=0.0,
            secondary_visual_modes=[],
            secondary_characteristics=[],
            evidence_notes=evidence_notes,
            extra_fields=extra,
        )

    @classmethod
    def profile_from_data(
        cls,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> TopicProfile:
        """
        Main entry point for profiling raw LLM or outline data.
        1. Validates schema.
        2. Reconstructs TopicProfile.
        3. Validates semantic consistency.
        4. If invalid or contradictory, returns explicit UNCLASSIFIED fallback.
        """
        is_schema_valid, schema_errors = cls.validate_schema(data)
        if not is_schema_valid:
            return cls.build_unclassified_fallback(
                rationale=f"Schema validation failed: {'; '.join(schema_errors)}",
                evidence_notes=schema_errors,
                raw_data=data,
            )

        profile = TopicProfile.from_dict(data)

        # Check semantic consistency
        is_consistent, contradictions = cls.validate_semantic_consistency(profile, context=context)
        if not is_consistent:
            evidence = profile.evidence_notes + contradictions
            return cls.build_unclassified_fallback(
                rationale=f"Semantic consistency checks failed: {'; '.join(contradictions)}",
                evidence_notes=evidence,
                raw_data=data,
            )

        return profile

    @classmethod
    def profile_heuristic_fallback(cls, topic: str, description: str = "") -> TopicProfile:
        """
        Deterministic, offline heuristic profiler for testing, offline bootstrapping,
        or when live model inference is unavailable.
        """
        t_low = (topic + " " + description).lower()

        # Check biological / natural science
        if any(kw in t_low for kw in cls.BIOLOGICAL_PROCESS_KEYWORDS):
            return TopicProfile(
                state_evolution=StateEvolution.MULTI_STAGE_PROCESS,
                formalism_level=FormalismLevel.MECHANISTIC_EMPIRICAL,
                computation_requirement=ComputationRequirement.NONE_NOT_APPLICABLE,
                primary_visual_mode=PrimaryVisualMode.PROCESS_FLOWCHART,
                rationale_and_evidence=f"Biological mechanism ({topic}) with multi-stage biochemical pathways and no arithmetic requirement.",
                profile_status=ProfileStatus.CLASSIFIED,
                confidence=0.90,
                secondary_visual_modes=["molecular_spatial"],
                secondary_characteristics=["cyclic_pathway", "membrane_transport"],
            )

        # Check dynamic programming / matrix table
        if "knapsack" in t_low or "dynamic programming" in t_low or "dp" in t_low:
            return TopicProfile(
                state_evolution=StateEvolution.DISCRETE_STATE_MACHINE,
                formalism_level=FormalismLevel.RIGOROUS_MATHEMATICAL,
                computation_requirement=ComputationRequirement.DETERMINISTIC_TRACE,
                primary_visual_mode=PrimaryVisualMode.STATE_TABLE_MATRIX,
                rationale_and_evidence=f"Dynamic programming recurrence ({topic}) requiring discrete memoization table state simulation.",
                profile_status=ProfileStatus.CLASSIFIED,
                confidence=0.95,
                secondary_visual_modes=["coordinate_geometry"],
                secondary_characteristics=["state_recurrence", "optimal_substructure"],
            )

        # Check graph algorithms
        if "dijkstra" in t_low or "shortest path" in t_low or "graph" in t_low:
            return TopicProfile(
                state_evolution=StateEvolution.DISCRETE_STATE_MACHINE,
                formalism_level=FormalismLevel.RIGOROUS_MATHEMATICAL,
                computation_requirement=ComputationRequirement.DETERMINISTIC_TRACE,
                primary_visual_mode=PrimaryVisualMode.NETWORK_GRAPH,
                rationale_and_evidence=f"Graph algorithm ({topic}) with priority queue relaxation and node state mutations.",
                profile_status=ProfileStatus.CLASSIFIED,
                confidence=0.95,
                secondary_visual_modes=["state_table_matrix"],
                secondary_characteristics=["greedy_relaxation", "priority_queue"],
            )

        # Check search algorithms
        if "binary search" in t_low:
            return TopicProfile(
                state_evolution=StateEvolution.DISCRETE_STATE_MACHINE,
                formalism_level=FormalismLevel.RIGOROUS_MATHEMATICAL,
                computation_requirement=ComputationRequirement.DETERMINISTIC_TRACE,
                primary_visual_mode=PrimaryVisualMode.COORDINATE_GEOMETRY,
                rationale_and_evidence=f"Searching algorithm ({topic}) with interval halving, pointer convergence, and state snapshots.",
                profile_status=ProfileStatus.CLASSIFIED,
                confidence=0.92,
                secondary_visual_modes=["structured_concept_card"],
                secondary_characteristics=["pointer_convergence", "logarithmic_reduction"],
            )

        # Check chemical kinetics
        if "arrhenius" in t_low or "activation energy" in t_low or "reaction kinetics" in t_low:
            return TopicProfile(
                state_evolution=StateEvolution.CONTINUOUS_DYNAMIC,
                formalism_level=FormalismLevel.MECHANISTIC_EMPIRICAL,
                computation_requirement=ComputationRequirement.ARITHMETIC_CALCULATION,
                primary_visual_mode=PrimaryVisualMode.COORDINATE_GEOMETRY,
                rationale_and_evidence=f"Chemical kinetics ({topic}) with continuous reaction coordinate and exponential temperature dependency.",
                profile_status=ProfileStatus.CLASSIFIED,
                confidence=0.90,
                secondary_visual_modes=["molecular_spatial"],
                secondary_characteristics=["exponential_scaling", "potential_energy_surface"],
            )

        # Check deductive proof / geometry
        if "pythagorean" in t_low or "proof" in t_low or "theorem" in t_low:
            return TopicProfile(
                state_evolution=StateEvolution.STATIC_STRUCTURAL,
                formalism_level=FormalismLevel.RIGOROUS_MATHEMATICAL,
                computation_requirement=ComputationRequirement.SYMBOLIC_IDENTITY,
                primary_visual_mode=PrimaryVisualMode.COORDINATE_GEOMETRY,
                rationale_and_evidence=f"Geometric theorem ({topic}) requiring deductive proof dissection and algebraic identity $a^2 + b^2 = c^2$.",
                profile_status=ProfileStatus.CLASSIFIED,
                confidence=0.92,
                secondary_visual_modes=["structured_concept_card"],
                secondary_characteristics=["geometric_dissection", "algebraic_invariance"],
            )

        # Check calculus
        if "derivative" in t_low or "rate of change" in t_low:
            return TopicProfile(
                state_evolution=StateEvolution.CONTINUOUS_DYNAMIC,
                formalism_level=FormalismLevel.RIGOROUS_MATHEMATICAL,
                computation_requirement=ComputationRequirement.ARITHMETIC_CALCULATION,
                primary_visual_mode=PrimaryVisualMode.COORDINATE_GEOMETRY,
                rationale_and_evidence=f"Calculus concept ({topic}) with difference quotients, local linearity, and tangent line transitions.",
                profile_status=ProfileStatus.CLASSIFIED,
                confidence=0.90,
                secondary_visual_modes=["structured_concept_card"],
                secondary_characteristics=["limiting_process", "local_tangent"],
            )

        # Check concurrency / OS
        if "race condition" in t_low or "mutex" in t_low or "operating system" in t_low:
            return TopicProfile(
                state_evolution=StateEvolution.DISCRETE_STATE_MACHINE,
                formalism_level=FormalismLevel.SYSTEM_STRUCTURAL,
                computation_requirement=ComputationRequirement.DETERMINISTIC_TRACE,
                primary_visual_mode=PrimaryVisualMode.PROCESS_FLOWCHART,
                rationale_and_evidence=f"Concurrent system ({topic}) with interleaving thread execution and critical section synchronization.",
                profile_status=ProfileStatus.CLASSIFIED,
                confidence=0.88,
                secondary_visual_modes=["structured_concept_card"],
                secondary_characteristics=["interleaving_threads", "mutual_exclusion"],
            )

        # Fallback to explicit UNCLASSIFIED
        return cls.build_unclassified_fallback(
            rationale=f"Insufficient domain-specific keywords to classify topic '{topic}' with certainty.",
            evidence_notes=[f"Topic '{topic}' did not match recognized canonical domain heuristics"],
        )
