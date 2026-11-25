"""
Storyboard Numerical Fidelity Checker for CineLearn (Milestone 4).

Deterministically verifies that Stage 2 storyboard visual components, MathTex formulas,
displayed text, verified_values declarations, and spoken narration faithfully preserve
the verified numerical calculations, variables, and units produced by Stage 1.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from numerical_evaluator import (
    ClaimedPrecision,
    DiagnosticCode,
    FormulaNormalizer,
    FormulaTrustStatus,
    PrecisionExtractor,
    SafeASTEvaluator,
    ToleranceEvaluator,
    TrustedFormulaRegistry,
    VerificationStatus,
)


# ============================================================================
# Discrepancy & Fidelity Categories
# ============================================================================

class FidelityCategory(str, Enum):
    """Classification of storyboard numerical fidelity discrepancies."""
    VALUE_MISMATCH = "VALUE_MISMATCH"
    HALLUCINATION_RELAPSE = "HALLUCINATION_RELAPSE"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    MISSING_REQUIRED_RESULT = "MISSING_REQUIRED_RESULT"
    UNSUPPORTED_FORMULA = "UNSUPPORTED_FORMULA"
    AMBIGUOUS_CLAIM = "AMBIGUOUS_CLAIM"
    # Algorithmic trace fidelity categories (Milestone 2B.6B)
    MISSING_ALGORITHMIC_STEP = "MISSING_ALGORITHMIC_STEP"
    ALGORITHMIC_ORDER_MISMATCH = "ALGORITHMIC_ORDER_MISMATCH"
    ALGORITHMIC_STATE_MISMATCH = "ALGORITHMIC_STATE_MISMATCH"
    INSUFFICIENT_VISUAL_COVERAGE = "INSUFFICIENT_VISUAL_COVERAGE"
    GRAPH_SPEC_MISMATCH = "GRAPH_SPEC_MISMATCH"
    # Generic component and trace fidelity categories
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    INVALID_COMPONENT_REFERENCE = "INVALID_COMPONENT_REFERENCE"
    TRACE_REFERENCE_MISMATCH = "TRACE_REFERENCE_MISMATCH"
    INVALID_ANIMATION_SYNTAX = "INVALID_ANIMATION_SYNTAX"
    PREMATURE_METRIC_LEAK = "PREMATURE_METRIC_LEAK"
    INVALID_DISPATCH_ORDER = "INVALID_DISPATCH_ORDER"
    UNAUTHORIZED_PREEMPTION = "UNAUTHORIZED_PREEMPTION"
    GANTT_TIMELINE_MISMATCH = "GANTT_TIMELINE_MISMATCH"
    MISSING_INPUT_SPECIFICATION = "MISSING_INPUT_SPECIFICATION"
    INVALID_STATE_TRANSITION = "INVALID_STATE_TRANSITION"
    INVALID_REPEATED_DISPATCH = "INVALID_REPEATED_DISPATCH"
    INCOMPLETE_PROCESS_COVERAGE = "INCOMPLETE_PROCESS_COVERAGE"
    EXCESSIVE_HIGHLIGHTING = "EXCESSIVE_HIGHLIGHTING"
    VIEWPORT_CONFLICT = "VIEWPORT_CONFLICT"


@dataclass
class FidelityDiscrepancy:
    """Individual discrepancy detected between storyboard claim and Stage 1 verified outline."""
    section_id: str
    location: str
    category: FidelityCategory
    actual_claim: str
    expected_value: Optional[str] = None
    expected_unit: Optional[str] = None
    reason: str = ""
    suggested_correction: Optional[str] = None
    severity: VerificationStatus = VerificationStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "location": self.location,
            "category": self.category.value,
            "actual_claim": self.actual_claim,
            "expected_value": self.expected_value,
            "expected_unit": self.expected_unit,
            "reason": self.reason,
            "suggested_correction": self.suggested_correction,
            "severity": self.severity.value,
        }


@dataclass
class StoryboardFidelityReport:
    """Overall outcome report of storyboard numerical fidelity inspection."""
    overall_status: VerificationStatus
    total_checks: int
    verified_claims: int
    discrepancies: List[FidelityDiscrepancy] = field(default_factory=list)
    summary: str = ""
    is_safe_for_audio: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "total_checks": self.total_checks,
            "verified_claims": self.verified_claims,
            "discrepancies": [d.to_dict() for d in self.discrepancies],
            "summary": self.summary,
            "is_safe_for_audio": self.is_safe_for_audio,
        }


# ============================================================================
# Authoritative Target Representation
# ============================================================================

@dataclass
class TargetVariable:
    """Authoritative variable verified in Stage 1 outline."""
    name: str
    float_value: float
    formatted_value: str
    unit: Optional[str] = None
    is_input: bool = False
    step_index: Optional[int] = None
    context_tags: Set[str] = field(default_factory=set)


class AuthoritativeOutlineModel:
    """
    Extracts and indexes authoritative numerical targets, inputs, step calculations,
    final results, and known rejected hallucinations from the Stage 1 outline.
    """

    def __init__(self, outline_dict: Dict[str, Any]):
        self.outline_dict = outline_dict
        self.targets: Dict[str, TargetVariable] = {}
        self.step_targets: Dict[int, Dict[str, TargetVariable]] = {}
        self.step_snapshots: Dict[int, Dict[str, Any]] = {}
        self.step_number_to_index: Dict[int, int] = {}
        self.value_to_targets: Dict[float, List[TargetVariable]] = {}
        self.rejected_hallucinations: Dict[float, Dict[str, Any]] = {}
        self.rejected_raw_strings: List[str] = []
        self.required_results: List[Dict[str, Any]] = []
        self.inputs: Dict[str, float] = {}
        self.governing_principles: List[str] = []
        self.step_formula_asts: Set[str] = set()
        self.step_target_map: Dict[str, TargetVariable] = {}
        self.step_formulas: List[str] = []
        self.has_worked_example: bool = False
        self.algo_state: Optional[Dict[str, Any]] = None
        self.is_dijkstra: bool = False
        self.is_fcfs: bool = False
        self.fcfs_state: Optional[Dict[str, Any]] = None

        self._build()

    def _normalize_var_name(self, name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r"[\{\}\$\\]", "", s)
        s = s.replace("_", "")
        return s

    def _build(self):
        # Check for verified algorithmic state (e.g. Dijkstra, FCFS)
        algo_state = None
        vm = self.outline_dict.get("verification_metadata")
        if isinstance(vm, dict) and vm.get("verified_algorithmic_state"):
            algo_state = vm["verified_algorithmic_state"]
        else:
            spec = self.outline_dict.get("worked_example_spec", {})
            if isinstance(spec, dict):
                spec_vm = spec.get("verification_metadata")
                if isinstance(spec_vm, dict) and spec_vm.get("verified_algorithmic_state"):
                    algo_state = spec_vm["verified_algorithmic_state"]

        if not algo_state and self.outline_dict.get("verified_algorithmic_state"):
            algo_state = self.outline_dict.get("verified_algorithmic_state")

        if not algo_state:
            spec = self.outline_dict.get("worked_example_spec", {})
            topic = str(self.outline_dict.get("topic", ""))
            try:
                from worked_example_verifier import WorkedExampleVerifier
                if isinstance(spec, dict) and WorkedExampleVerifier.is_dijkstra_spec(spec, topic):
                    from dijkstra_verifier import DijkstraTraceVerifier
                    _, rep = DijkstraTraceVerifier.verify_worked_example_spec(spec, topic=topic)
                    algo_state = rep.verified_algorithmic_state
            except Exception:
                pass

        if not algo_state:
            spec = self.outline_dict.get("worked_example_spec", {})
            topic = str(self.outline_dict.get("topic", ""))
            try:
                from fcfs_verifier import FCFSTraceVerifier
                if isinstance(spec, dict) and FCFSTraceVerifier.can_verify(spec, topic):
                    _, rep = FCFSTraceVerifier.verify_worked_example_spec(spec, topic=topic)
                    algo_state = rep.verified_algorithmic_state
            except Exception:
                pass

        if algo_state and isinstance(algo_state, dict):
            algo = algo_state.get("algorithm")
            if algo == "dijkstra":
                self.algo_state = algo_state
                self.is_dijkstra = True
                self.has_worked_example = True
                return
            elif algo == "fcfs":
                self.algo_state = algo_state
                self.is_fcfs = True
                self.fcfs_state = algo_state
                self.has_worked_example = True
                return

        spec = self.outline_dict.get("worked_example_spec")
        if not spec or not isinstance(spec, dict) or not spec.get("steps"):
            self.has_worked_example = False
            return

        self.has_worked_example = True
        gov_p = self.outline_dict.get("governing_principles")
        if isinstance(gov_p, list):
            self.governing_principles = [str(p) for p in gov_p]

        # 1. Inputs
        raw_inputs = spec.get("inputs", {})
        input_units = spec.get("input_units") or spec.get("units") or {}
        running_targets: Dict[str, TargetVariable] = {}
        running_state: Dict[str, Any] = {}
        if isinstance(raw_inputs, dict):
            for k, v in raw_inputs.items():
                unit = input_units.get(k) if isinstance(input_units, dict) else None
                f_val = None
                if isinstance(v, dict):
                    f_val = float(v.get("value", 0))
                    unit = v.get("unit") or unit
                elif isinstance(v, str):
                    clean_v, _, u = PrecisionExtractor.sanitize_claimed_string(v)
                    try:
                        f_val = float(clean_v)
                        unit = u or unit
                    except Exception:
                        pass
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    f_val = float(v)

                running_state[k] = v
                if f_val is not None and math.isfinite(f_val):
                    self.inputs[k] = f_val
                    norm_k = self._normalize_var_name(k)
                    ctx = {norm_k}
                    if norm_k in ("t1", "t2"):
                        ctx.add(f"{int(f_val)}")
                    t_var = TargetVariable(
                        name=k,
                        float_value=f_val,
                        formatted_value=str(v),
                        unit=unit,
                        is_input=True,
                        step_index=None,
                        context_tags=ctx,
                    )
                    self.targets[norm_k] = t_var
                    running_targets[norm_k] = t_var
                    self.value_to_targets.setdefault(f_val, []).append(t_var)

        self.step_targets[-1] = copy.deepcopy(running_targets)
        self.step_snapshots[-1] = copy.deepcopy(running_state)

        # 2. Steps
        steps = spec.get("steps", [])
        if isinstance(steps, list):
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue

                step_desc = step.get("step", "")
                result_str = str(step.get("result", "")).strip()
                orig_res = step.get("original_result")
                structured = step.get("structured_calc", {})
                if not isinstance(structured, dict):
                    structured = {}

                # Map 1-based step numbers to 0-based index i
                step_num = i + 1
                if "step_number" in step and isinstance(step["step_number"], int):
                    step_num = step["step_number"]
                else:
                    m_step = re.search(r"\bstep[_\s]*(\d+)\b", step_desc, re.IGNORECASE)
                    if m_step:
                        step_num = int(m_step.group(1))

                self.step_number_to_index[step_num] = i

                running_targets = dict(running_targets)
                running_state = dict(running_state)

                # Check if state_snapshot is in structured
                snap_state = structured.get("state_snapshot")
                if isinstance(snap_state, dict):
                    for var_k, var_v in snap_state.items():
                        running_state[var_k] = var_v
                        if isinstance(var_v, (int, float)) and not isinstance(var_v, bool) and math.isfinite(var_v):
                            norm_vk = self._normalize_var_name(var_k)
                            t_v = TargetVariable(
                                name=var_k,
                                float_value=float(var_v),
                                formatted_value=str(var_v),
                                is_input=False,
                                step_index=i,
                                context_tags={norm_vk},
                            )
                            running_targets[norm_vk] = t_v
                            self.targets[norm_vk] = t_v
                            self.value_to_targets.setdefault(float(var_v), []).append(t_v)

                # Check if state_after is in step
                state_after = step.get("state_after")
                if isinstance(state_after, dict):
                    for var_k, var_v in state_after.items():
                        running_state[var_k] = var_v
                        if isinstance(var_v, (int, float)) and not isinstance(var_v, bool) and math.isfinite(var_v):
                            norm_vk = self._normalize_var_name(var_k)
                            t_v = TargetVariable(
                                name=var_k,
                                float_value=float(var_v),
                                formatted_value=str(var_v),
                                is_input=False,
                                step_index=i,
                                context_tags={norm_vk},
                            )
                            running_targets[norm_vk] = t_v
                            self.targets[norm_vk] = t_v
                            self.value_to_targets.setdefault(float(var_v), []).append(t_v)

                out_var = structured.get("output_variable") or f"step_{i+1}_val"
                norm_out_var = self._normalize_var_name(out_var)
                comp_val = structured.get("computed_value")
                corr_val = structured.get("corrected_value")
                unit = structured.get("unit")
                orig_claimed = structured.get("original_claimed")

                ctx = {norm_out_var}
                if "t1" in norm_out_var or "t1" in step_desc.lower() or "first temperature" in step_desc.lower():
                    ctx.add("t1")
                if "t2" in norm_out_var or "t2" in step_desc.lower() or "second temperature" in step_desc.lower():
                    ctx.add("t2")

                t_var = None
                if comp_val is not None and isinstance(comp_val, (int, float)) and math.isfinite(comp_val):
                    f_val = float(comp_val)
                    disp_val = str(corr_val) if corr_val is not None else f"{f_val:.4g}"
                    t_var = TargetVariable(
                        name=out_var,
                        float_value=f_val,
                        formatted_value=disp_val,
                        unit=unit,
                        is_input=False,
                        step_index=i,
                        context_tags=ctx,
                    )
                    running_targets[norm_out_var] = t_var
                    running_state[out_var] = f_val
                    self.targets[norm_out_var] = t_var
                    self.value_to_targets.setdefault(f_val, []).append(t_var)

                    self.required_results.append({
                        "step_index": i,
                        "var_name": out_var,
                        "target": t_var,
                        "description": step_desc,
                    })

                # Check result_str for variable assignments: e.g. 'low = 0, high = 9' or 'mid = 4'
                if result_str:
                    assigns = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,;]+)", result_str)
                    for a_var, a_val_str in assigns:
                        clean_a, _, a_unit = PrecisionExtractor.sanitize_claimed_string(a_val_str)
                        try:
                            a_fval = float(clean_a)
                            if math.isfinite(a_fval):
                                norm_a = self._normalize_var_name(a_var)
                                running_state[a_var] = a_fval
                                a_target = TargetVariable(
                                    name=a_var,
                                    float_value=a_fval,
                                    formatted_value=clean_a,
                                    unit=a_unit or unit,
                                    is_input=False,
                                    step_index=i,
                                    context_tags={norm_a},
                                )
                                running_targets[norm_a] = a_target
                                self.targets[norm_a] = a_target
                                self.value_to_targets.setdefault(a_fval, []).append(a_target)
                        except Exception:
                            pass

                self.step_targets[i] = copy.deepcopy(running_targets)
                self.step_snapshots[i] = copy.deepcopy(running_state)

                # Record verified step formula expressions and canonical ASTs
                calc_candidates = []
                if "=" in result_str:
                    res_parts = result_str.split("=")
                    calc_candidates.append(res_parts[0].strip())
                if "formula" in structured:
                    calc_candidates.append(str(structured["formula"]).strip())
                if ":" in step_desc:
                    desc_after = step_desc.split(":", 1)[1].strip()
                    calc_candidates.append(desc_after)

                for cand in calc_candidates:
                    if not cand:
                        continue
                    clean_c = FormulaNormalizer.sanitize_latex(cand).strip()
                    clean_c = re.sub(
                        r"([0-9]+(?:\.[0-9]+)?)\s*\*\s*10\*\*(?:\(([+-]?[0-9]+)\)|([+-]?[0-9]+))",
                        lambda m: f"{m.group(1)}e{m.group(2) or m.group(3)}",
                        clean_c,
                    )
                    self.step_formulas.append(clean_c)
                    try:
                        ast_repr = FormulaNormalizer.ast_to_canonical_str(clean_c)
                        self.step_formula_asts.add(ast_repr)
                        if t_var is not None:
                            self.step_target_map[ast_repr] = t_var
                    except Exception:
                        pass

                # Check for rejected hallucination in this step
                if orig_res or orig_claimed:
                    claim_str = str(orig_claimed or orig_res)
                    try:
                        prec = PrecisionExtractor.extract_precision(claim_str)
                        if comp_val is not None:
                            match, _, _ = ToleranceEvaluator.evaluate(prec, comp_val)
                            if not match:
                                self.rejected_hallucinations[prec.claimed_float] = {
                                    "rejected_float": prec.claimed_float,
                                    "rejected_string": claim_str,
                                    "corrected_value": corr_val or str(comp_val),
                                    "var_name": out_var,
                                    "step_index": i,
                                }
                                clean_s, _, _ = PrecisionExtractor.sanitize_claimed_string(claim_str)
                                self.rejected_raw_strings.append(clean_s)
                    except Exception:
                        pass

        # 3. Final Result
        final_res = str(spec.get("final_result", "")).strip()
        orig_final = spec.get("original_final_result")
        if orig_final and orig_final != final_res:
            try:
                prec_orig = PrecisionExtractor.extract_precision(str(orig_final))
                self.rejected_hallucinations[prec_orig.claimed_float] = {
                    "rejected_float": prec_orig.claimed_float,
                    "rejected_string": str(orig_final),
                    "corrected_value": final_res,
                    "var_name": "final_result",
                    "step_index": None,
                }
                clean_s, _, _ = PrecisionExtractor.sanitize_claimed_string(str(orig_final))
                self.rejected_raw_strings.append(clean_s)
            except Exception:
                pass

    def resolve_step_index(self, step_ref: Optional[Any]) -> Optional[int]:
        """Resolves a 0-based or 1-based step reference to the canonical 0-based step index."""
        if step_ref is None:
            return None
        if isinstance(step_ref, tuple) and len(step_ref) == 2:
            kind, val = step_ref
            if kind == "index" and isinstance(val, int):
                return val
            if kind == "number" and isinstance(val, int):
                return self.step_number_to_index.get(val, val - 1)
        if isinstance(step_ref, int):
            if step_ref == -1:
                return -1
            if step_ref in self.step_number_to_index:
                return self.step_number_to_index[step_ref]
            if step_ref in self.step_targets:
                return step_ref
            return step_ref
        if isinstance(step_ref, str):
            clean = step_ref.strip().lower()
            if clean in ("initial", "inputs", "input", "-1"):
                return -1
            m = re.search(r"\bstep[_\s]*(\d+)\b", clean)
            if m:
                val = int(m.group(1))
                return self.step_number_to_index.get(val, val - 1)
            if clean.isdigit():
                val = int(clean)
                return self.step_number_to_index.get(val, val - 1)
        return None

    def has_step(self, step_ref: Optional[Union[int, str]]) -> bool:
        """Returns True if the step reference corresponds to a known step in the worked example."""
        if step_ref is None:
            return False
        idx = self.resolve_step_index(step_ref)
        if idx is None:
            return False
        return idx in self.step_targets or idx in self.step_snapshots

    def get_targets_at_step(self, step_ref: Optional[Union[int, str]]) -> Dict[str, TargetVariable]:
        """Returns the dictionary of target variables as of the specified step."""
        idx = self.resolve_step_index(step_ref)
        if idx is not None and idx in self.step_targets:
            return self.step_targets[idx]
        return self.targets

    def get_target_at_step(
        self, norm_name: str, step_ref: Optional[Union[int, str]] = None
    ) -> Optional[TargetVariable]:
        """Looks up a target variable at a specific step."""
        targets = self.get_targets_at_step(step_ref)
        return targets.get(norm_name)

    def get_state_at_step(self, step_ref: Optional[Union[int, str]] = None) -> Dict[str, Any]:
        """Returns the complete variable state dictionary as of the specified step."""
        idx = self.resolve_step_index(step_ref)
        if idx is not None and idx in self.step_snapshots:
            return copy.deepcopy(self.step_snapshots[idx])
        if self.step_snapshots:
            max_idx = max(self.step_snapshots.keys())
            return copy.deepcopy(self.step_snapshots[max_idx])
        return {}


# ============================================================================
# Unit Normalization & Helper
# ============================================================================

def normalize_unit(unit_str: Optional[str]) -> Optional[str]:
    """Canonicalizes unit expressions for robust comparison."""
    if not unit_str:
        return None
    s = unit_str.strip()
    s = re.sub(r"[\{\}\$\\]", "", s)
    s = s.replace("text", "").replace("mathrm", "")
    s = s.replace(" ", "")

    if s in ("1/s", "s^-1", "s^{-1}", "persecond", "s-1"):
        return "s^-1"
    if s in ("j/mol", "jmol^-1", "joules/mole", "joulespermole", "j/(mol*k)", "j/molk"):
        return "J/mol"
    if s in ("kj/mol", "kjmol^-1", "kilojoules/mole", "kilojoulespermole"):
        return "kJ/mol"
    if s in ("k", "kelvin"):
        return "K"
    if s in ("s", "sec", "seconds", "second"):
        return "s"
    if s in ("m/s", "ms^-1"):
        return "m/s"
    return s


def units_are_compatible(unit_a: Optional[str], unit_b: Optional[str]) -> Tuple[bool, bool]:
    """
    Checks if two units are compatible.
    Returns: (is_compatible, is_prefix_scaled).
    """
    norm_a = normalize_unit(unit_a)
    norm_b = normalize_unit(unit_b)
    if not norm_a or not norm_b:
        return True, False
    if norm_a == norm_b:
        return True, False
    if {norm_a, norm_b} == {"J/mol", "kJ/mol"}:
        return False, True
    if {norm_a, norm_b} == {"s^-1", "s"}:
        return False, False
    return False, False


# ============================================================================
# Storyboard Fidelity Checker Engine
# ============================================================================

class StoryboardFidelityChecker:
    """
    Deterministic checker ensuring that Stage 2 storyboard elements preserve
    the verified calculations, numerical precision, and units of Stage 1.
    """

    def __init__(self, evaluator: Optional[SafeASTEvaluator] = None):
        self.evaluator = evaluator or SafeASTEvaluator()

    @staticmethod
    def _extract_precision(text: str) -> ClaimedPrecision:
        """Sanitizes LaTeX backslash spacing commands (e.g. `\\ `) and extracts precision."""
        clean = re.sub(r"\\(?:\s+|$)", " ", text).strip()
        return PrecisionExtractor.extract_precision(clean)

    def check_storyboard_fidelity(
        self,
        outline: Union[Dict[str, Any], Any],
        storyboard: Union[Dict[str, Any], List[Any], Any],
    ) -> StoryboardFidelityReport:
        """
        Public entry point: audits storyboard structure against verified outline.
        Accepts outline/storyboard as dictionaries, lists, or dataclasses.
        """
        outline_dict = outline.to_dict() if hasattr(outline, "to_dict") else (
            dict(outline) if isinstance(outline, dict) else {}
        )

        model = AuthoritativeOutlineModel(outline_dict)

        # Non-numerical or conceptual outlines pass immediately
        if not model.has_worked_example:
            return StoryboardFidelityReport(
                overall_status=VerificationStatus.NOT_APPLICABLE,
                total_checks=0,
                verified_claims=0,
                discrepancies=[],
                summary="Storyboard contains no numerical worked-example specification; fidelity checks not applicable.",
                is_safe_for_audio=True,
            )

        # Normalize storyboard into sections list
        sections = self._extract_sections(storyboard)
        if not sections:
            return StoryboardFidelityReport(
                overall_status=VerificationStatus.NEEDS_REVIEW,
                total_checks=0,
                verified_claims=0,
                discrepancies=[
                    FidelityDiscrepancy(
                        section_id="all",
                        location="storyboard",
                        category=FidelityCategory.AMBIGUOUS_CLAIM,
                        actual_claim="Empty sections list",
                        reason="No sections found in storyboard to verify",
                        severity=VerificationStatus.NEEDS_REVIEW,
                    )
                ],
                summary="Storyboard contains no sections to verify",
                is_safe_for_audio=False,
            )

        # Delegate domain-specific verification through DomainFidelityRegistry
        try:
            from src.domains import DomainFidelityRegistry
            domain_verifier = DomainFidelityRegistry.find_verifier_for_model(model, sections)
            if domain_verifier is not None:
                return domain_verifier.verify(model, sections)
        except Exception:
            pass

        # Fallback routes for algorithmic outlines
        if model.is_dijkstra:
            return self._check_dijkstra_fidelity(model, sections)

        if getattr(model, "is_fcfs", False):
            return self._check_fcfs_fidelity(model, sections)

        discrepancies: List[FidelityDiscrepancy] = []
        total_checks = 0
        verified_claims = 0

        # Validate component actions against VisualRegistry manifests
        try:
            from src.domains.action_resolver import StoryboardActionResolver
            comp_discs = StoryboardActionResolver.validate_storyboard_actions(
                {"sections": sections}, authoritative_trace=getattr(model, "algo_state", None)
            )
            for cd in comp_discs:
                discrepancies.append(FidelityDiscrepancy(
                    section_id=cd.get("section_id", "unknown"),
                    location=cd.get("location", ""),
                    category=getattr(FidelityCategory, cd.get("category", "VALUE_MISMATCH"), FidelityCategory.VALUE_MISMATCH),
                    actual_claim=cd.get("actual_claim", ""),
                    expected_value=cd.get("expected_value"),
                    reason=cd.get("reason", ""),
                    suggested_correction=cd.get("suggested_correction"),
                    severity=VerificationStatus.FAILED,
                ))
                total_checks += 1
        except Exception:
            pass

        # Set of matched target variables across all sections to check completeness
        matched_targets_global: Set[str] = set()

        for sec in sections:
            sec_id = sec.get("id", "unknown_section")

            # 1. Inspect verified_values declaration
            v_disc, v_checks, v_ver, v_matched = self._check_verified_values(sec, model)
            discrepancies.extend(v_disc)
            total_checks += v_checks
            verified_claims += v_ver
            matched_targets_global.update(v_matched)

            # 2. Inspect animations (MathTex and Text objects)
            a_disc, a_checks, a_ver, a_matched = self._check_animations(sec, model)
            discrepancies.extend(a_disc)
            total_checks += a_checks
            verified_claims += a_ver
            matched_targets_global.update(a_matched)

            # 3. Inspect lecture_lines and captions (Narration)
            n_disc, n_checks, n_ver, n_matched = self._check_narration(sec, model)
            discrepancies.extend(n_disc)
            total_checks += n_checks
            verified_claims += n_ver
            matched_targets_global.update(n_matched)

        # 4. Global completeness: verify that the worked example's final result is communicated
        final_disc = self._check_required_results(model, matched_targets_global)
        if final_disc:
            discrepancies.append(final_disc)
            total_checks += 1

        # Determine overall report status
        has_failed = any(d.severity == VerificationStatus.FAILED for d in discrepancies)
        has_review = any(d.severity == VerificationStatus.NEEDS_REVIEW for d in discrepancies)

        if has_failed:
            overall_status = VerificationStatus.FAILED
            is_safe = False
        elif has_review:
            overall_status = VerificationStatus.NEEDS_REVIEW
            is_safe = False
        else:
            overall_status = VerificationStatus.VERIFIED
            is_safe = True

        summary = (
            f"Storyboard numerical fidelity: {verified_claims} verified claims, "
            f"{len(discrepancies)} discrepancy/discrepancies flagged across {total_checks} checks."
        )

        return StoryboardFidelityReport(
            overall_status=overall_status,
            total_checks=total_checks,
            verified_claims=verified_claims,
            discrepancies=discrepancies,
            summary=summary,
            is_safe_for_audio=is_safe,
        )

    def _check_dijkstra_fidelity(
        self, model: AuthoritativeOutlineModel, sections: List[Dict[str, Any]]
    ) -> StoryboardFidelityReport:
        """
        Validates Dijkstra storyboard coverage, extraction sequence, relaxation accuracy,
        visual state transitions, and final distances against the verified canonical trace.
        """
        algo_state = model.algo_state or {}
        vertices = [str(v) for v in algo_state.get("vertices", [])]
        source = str(algo_state.get("source", ""))
        directed = bool(algo_state.get("directed", False))
        edges = algo_state.get("edges", [])
        settled_order = [str(v) for v in algo_state.get("settled_order", [])]
        canonical_trace = algo_state.get("canonical_trace", [])
        final_distances_raw = algo_state.get("final_distances", {})
        final_distances: Dict[str, float] = {}
        for k, v in final_distances_raw.items():
            if v == "inf" or (isinstance(v, float) and math.isinf(v)):
                final_distances[str(k)] = math.inf
            else:
                try:
                    final_distances[str(k)] = float(v)
                except Exception:
                    final_distances[str(k)] = math.inf

        discrepancies: List[FidelityDiscrepancy] = []
        total_checks = 0
        verified_claims = 0

        # Structured collections across all sections
        observed_extractions: List[Dict[str, Any]] = []
        observed_relaxations: List[Dict[str, Any]] = []
        observed_node_updates: Dict[str, List[Dict[str, Any]]] = {}
        observed_final_distances: Dict[str, float] = {}
        observed_graph_vertices: Set[str] = set()
        observed_graph_source: Optional[str] = None
        visual_algorithmic_actions_count = 0
        narration_algo_mentions_count = 0

        # Iterate over sections
        for sec in sections:
            sec_id = sec.get("id", "section")
            anims = sec.get("animations", [])
            lines = sec.get("lecture_lines", [])
            vvals = sec.get("verified_values", {})
            algo_ops = sec.get("algorithmic_operations") or sec.get("algorithmic_beats") or []

            # 1. Check algorithmic_operations (if provided via schema extension)
            if isinstance(algo_ops, list):
                for op_item in algo_ops:
                    if not isinstance(op_item, dict):
                        continue
                    op_type = str(op_item.get("op", "")).lower()
                    if op_type in ("extract_min", "select_node", "extract"):
                        node = str(op_item.get("node", ""))
                        if node:
                            observed_extractions.append({
                                "node": node,
                                "dist": op_item.get("distance"),
                                "sec_id": sec_id,
                                "loc": f"{sec_id}.algorithmic_operations",
                                "has_visual": True,
                            })
                            visual_algorithmic_actions_count += 1
                    elif op_type in ("relax", "relax_edge"):
                        u = str(op_item.get("u", "")) if op_item.get("u") else None
                        v = str(op_item.get("v", "")) if op_item.get("v") else None
                        cand = op_item.get("candidate") or op_item.get("candidate_dist")
                        upd = op_item.get("updated")
                        if upd is None and op_item.get("is_updated") is not None:
                            upd = bool(op_item.get("is_updated"))
                        new_d = op_item.get("new_dist")
                        observed_relaxations.append({
                            "u": u,
                            "v": v,
                            "candidate": float(cand) if cand is not None else None,
                            "updated": upd,
                            "new_dist": float(new_d) if new_d is not None else None,
                            "sec_id": sec_id,
                            "loc": f"{sec_id}.algorithmic_operations",
                            "has_visual": True,
                        })
                        visual_algorithmic_actions_count += 1
                    elif op_type in ("no_update", "keep_distance"):
                        v = str(op_item.get("v", "")) if op_item.get("v") else None
                        cand = op_item.get("candidate")
                        observed_relaxations.append({
                            "u": str(op_item.get("u", "")) if op_item.get("u") else None,
                            "v": v,
                            "candidate": float(cand) if cand is not None else None,
                            "updated": False,
                            "new_dist": None,
                            "sec_id": sec_id,
                            "loc": f"{sec_id}.algorithmic_operations",
                            "has_visual": True,
                        })
                        visual_algorithmic_actions_count += 1

            # 2. Check animations
            if isinstance(anims, list):
                for i, anim in enumerate(anims):
                    anim_str = str(anim)
                    loc = f"{sec_id}.animations[{i}]"
                    parts = [p.strip() for p in anim_str.split("|")]
                    action_str = parts[4] if len(parts) >= 5 else anim_str
                    rep_str = parts[2] if len(parts) >= 3 else ""

                    # Inspect action field
                    if action_str.lower().startswith("action:"):
                        action_body = action_str[7:].strip()
                    else:
                        action_body = action_str.strip()

                    # Check for graph initialization
                    if any(term in action_body.lower() or term in rep_str.lower() for term in ("graphvisual", "initialize_graph", "init_graph", "weightedgraph")):
                        v_match = re.search(r"vertices\s*[=:]\s*\[([^\]]+)\]", action_body) or re.search(r"vertices\s*[=:]\s*\[([^\]]+)\]", rep_str)
                        if v_match:
                            found_v = re.findall(r"['\"]?([A-Za-z0-9_]+)['\"]?", v_match.group(1))
                            observed_graph_vertices.update(found_v)
                        s_match = re.search(r"source\s*[=:]\s*['\"]?([A-Za-z0-9_]+)['\"]?", action_body) or re.search(r"source\s*[=:]\s*['\"]?([A-Za-z0-9_]+)['\"]?", rep_str)
                        if s_match:
                            observed_graph_source = s_match.group(1).strip()

                    # Check for node extraction actions
                    ext_patterns = [
                        r"extract_min\(['\"]?([A-Za-z0-9_]+)['\"]?(?:,\s*([0-9.]+))?\)",
                        r"extract_min\(node=['\"]?([A-Za-z0-9_]+)['\"]?(?:,\s*dist=([0-9.]+))?\)",
                        r"select_node\(['\"]?([A-Za-z0-9_]+)['\"]?\)",
                        r"pick_min\(['\"]?([A-Za-z0-9_]+)['\"]?\)",
                        r"extract\s+node\s+['\"]?([A-Za-z0-9_]+)['\"]?",
                        r"extract\s+minimum\s+['\"]?([A-Za-z0-9_]+)['\"]?",
                    ]
                    for pat in ext_patterns:
                        m = re.search(pat, action_body, re.IGNORECASE)
                        if m:
                            node = m.group(1).strip()
                            d_val = float(m.group(2)) if len(m.groups()) >= 2 and m.group(2) else None
                            # Deduplicate consecutive repeat of same node
                            if not observed_extractions or observed_extractions[-1]["node"] != node:
                                observed_extractions.append({
                                    "node": node,
                                    "dist": d_val,
                                    "sec_id": sec_id,
                                    "loc": loc,
                                    "has_visual": True,
                                })
                            visual_algorithmic_actions_count += 1
                            break

                    # Check for edge relaxations
                    relax_m = re.search(
                        r"relax_edge\((?:u=)?['\"]?([A-Za-z0-9_]+)['\"]?,\s*(?:v=)?['\"]?([A-Za-z0-9_]+)['\"]?"
                        r"(?:,\s*(?:candidate=|new_v_label=)?([0-9.]+))?"
                        r"(?:,\s*updated=(True|False))?"
                        r"(?:,\s*new_dist=([0-9.]+))?",
                        action_body,
                        re.IGNORECASE,
                    )
                    if relax_m:
                        u = relax_m.group(1).strip()
                        v = relax_m.group(2).strip()
                        cand = float(relax_m.group(3)) if relax_m.group(3) else None
                        upd_str = relax_m.group(4)
                        upd = (upd_str.lower() == "true") if upd_str else (True if cand is not None else None)
                        new_d = float(relax_m.group(5)) if relax_m.group(5) else (cand if upd else None)
                        observed_relaxations.append({
                            "u": u,
                            "v": v,
                            "candidate": cand,
                            "updated": upd,
                            "new_dist": new_d,
                            "sec_id": sec_id,
                            "loc": loc,
                            "has_visual": True,
                        })
                        visual_algorithmic_actions_count += 1
                    else:
                        # Other edge examination actions
                        edge_m = re.search(
                            r"(?:highlight_edge|examine_edge|traverse_edge)\(['\"]?([A-Za-z0-9_]+)['\"]?,\s*['\"]?([A-Za-z0-9_]+)['\"]?\)",
                            action_body,
                            re.IGNORECASE,
                        )
                        if edge_m:
                            observed_relaxations.append({
                                "u": edge_m.group(1).strip(),
                                "v": edge_m.group(2).strip(),
                                "candidate": None,
                                "updated": None,
                                "new_dist": None,
                                "sec_id": sec_id,
                                "loc": loc,
                                "has_visual": True,
                            })
                            visual_algorithmic_actions_count += 1

                    # Check for node distance/label updates
                    upd_m = re.search(
                        r"(?:update_node_label|update_distance|update_cell)\(['\"]?([A-Za-z0-9_]+)['\"]?,\s*([0-9.]+|\\infty|inf)\)",
                        action_body,
                        re.IGNORECASE,
                    )
                    if upd_m:
                        target_v = upd_m.group(1).strip()
                        raw_d = upd_m.group(2).strip()
                        d_val = math.inf if raw_d in ("\\infty", "inf") else float(raw_d)
                        observed_node_updates.setdefault(target_v, []).append({
                            "dist": d_val,
                            "sec_id": sec_id,
                            "loc": loc,
                            "has_visual": True,
                        })
                        visual_algorithmic_actions_count += 1

                    # Check for unsuccessful relaxation / no_update action
                    no_upd_m = re.search(
                        r"(?:no_update|keep_distance)\(['\"]?([A-Za-z0-9_]+)['\"]?(?:,\s*(?:candidate=)?([0-9.]+))?",
                        action_body,
                        re.IGNORECASE,
                    )
                    if no_upd_m:
                        target_v = no_upd_m.group(1).strip()
                        cand = float(no_upd_m.group(2)) if no_upd_m.group(2) else None
                        observed_relaxations.append({
                            "u": None,
                            "v": target_v,
                            "candidate": cand,
                            "updated": False,
                            "new_dist": None,
                            "sec_id": sec_id,
                            "loc": loc,
                            "has_visual": True,
                        })
                        visual_algorithmic_actions_count += 1

                    # Inspect MathTex and Text literals
                    for lit in self._extract_display_strings(anim_str):
                        lit_clean = lit.strip()
                        # e.g. dist(B) = 4 or dist_B = 4
                        m_dist = re.search(r"dist(?:\[|\()([A-Za-z0-9_]+)(?:\]|\))\s*=\s*([0-9.]+|\\infty|inf)", lit_clean)
                        if m_dist:
                            v = m_dist.group(1).strip()
                            raw_val = m_dist.group(2).strip()
                            val = math.inf if raw_val in ("\\infty", "inf") else float(raw_val)
                            observed_node_updates.setdefault(v, []).append({
                                "dist": val,
                                "sec_id": sec_id,
                                "loc": loc,
                                "has_visual": True,
                            })
                            visual_algorithmic_actions_count += 1
                        # e.g. 0 + 4 < inf => dist(B) = 4
                        m_arrow = re.search(r"([0-9.]+)\s*\+\s*([0-9.]+)\s*<\s*([0-9.]+|\\infty|inf)\s*\\implies\s*dist\(([A-Za-z0-9_]+)\)\s*=\s*([0-9.]+)", lit_clean)
                        if m_arrow:
                            v = m_arrow.group(4).strip()
                            cand = float(m_arrow.group(5))
                            observed_node_updates.setdefault(v, []).append({
                                "dist": cand,
                                "sec_id": sec_id,
                                "loc": loc,
                                "has_visual": True,
                            })
                            visual_algorithmic_actions_count += 1
                        # Unsuccessful relaxation MathTex: e.g. 1 + 5 >= 4 => no update
                        if any(phrase in lit_clean.lower() for phrase in ("no update", "not updated", "keep")):
                            m_no = re.search(r"([0-9.]+)\s*\+\s*([0-9.]+)\s*(?:\\ge|>=)\s*([0-9.]+)", lit_clean)
                            cand = (float(m_no.group(1)) + float(m_no.group(2))) if m_no else None
                            observed_relaxations.append({
                                "u": None,
                                "v": None,
                                "candidate": cand,
                                "updated": False,
                                "new_dist": None,
                                "sec_id": sec_id,
                                "loc": loc,
                                "has_visual": True,
                            })
                            visual_algorithmic_actions_count += 1

            # 3. Check verified_values in section
            if isinstance(vvals, (dict, list)):
                vvals_items = vvals.items() if isinstance(vvals, dict) else [(f"val_{idx}", str(v)) for idx, v in enumerate(vvals)]
                for k, v in vvals_items:
                    k_str = str(k).strip()
                    v_str = str(v).strip()
                    full_str = f"{k_str} = {v_str}" if "=" not in v_str else v_str
                    # source = A
                    m_s = re.search(r"source(?:_node)?\s*=\s*['\"]?([A-Za-z0-9_]+)['\"]?", full_str, re.IGNORECASE)
                    if m_s:
                        observed_graph_source = m_s.group(1).strip()
                    # extract_min(A) = 0
                    m_ext = re.search(r"extract(?:_min)?\(?([A-Za-z0-9_]+)\)?\s*=\s*([0-9.]+)", full_str, re.IGNORECASE)
                    if m_ext:
                        node = m_ext.group(1).strip()
                        d_val = float(m_ext.group(2))
                        matched_ext = next((e for e in observed_extractions if e["node"] == node), None)
                        if matched_ext:
                            if matched_ext.get("dist") is None:
                                matched_ext["dist"] = d_val
                        else:
                            observed_extractions.append({
                                "node": node,
                                "dist": d_val,
                                "sec_id": sec_id,
                                "loc": f"{sec_id}.verified_values",
                                "has_visual": False,
                            })
                    # dist(B) = 4 or dist_B = 4
                    m_d = re.search(r"dist(?:_|\()([A-Za-z0-9_]+)(?:\))?\s*=\s*([0-9.]+|\\infty|inf)", full_str, re.IGNORECASE)
                    if m_d:
                        target_v = m_d.group(1).strip()
                        raw_d = m_d.group(2).strip()
                        d_val = math.inf if raw_d in ("\\infty", "inf") else float(raw_d)
                        observed_node_updates.setdefault(target_v, []).append({
                            "dist": d_val,
                            "sec_id": sec_id,
                            "loc": f"{sec_id}.verified_values",
                            "has_visual": True,
                        })
                        if sec == sections[-1] or "final" in sec.get("title", "").lower() or "final" in full_str.lower():
                            observed_final_distances[target_v] = d_val

            # 4. Check spoken narration in lecture_lines
            if isinstance(lines, list):
                for line in lines:
                    line_s = str(line).strip().lower()
                    if any(term in line_s for term in ("extract", "relaxation", "relax", "tentative distance", "shortest path")):
                        narration_algo_mentions_count += 1

        # --------------------------------------------------------------------
        # Run Verification Checks
        # --------------------------------------------------------------------

        # Check 1: Narration-only without visual state transitions
        total_checks += 1
        if narration_algo_mentions_count > 0 and visual_algorithmic_actions_count == 0:
            discrepancies.append(
                FidelityDiscrepancy(
                    section_id=sections[0].get("id", "section_1") if sections else "section",
                    location="animations",
                    category=FidelityCategory.INSUFFICIENT_VISUAL_COVERAGE,
                    actual_claim="Narration describes algorithmic operations, but visual animations contain no state transitions",
                    expected_value="Visual actions in animations (extract_min, relax_edge, update_node_label)",
                    reason="Storyboard describes Dijkstra operations in spoken narration but does not provide corresponding visual state transitions in animations",
                    suggested_correction="Add explicit visual actions (e.g. action: extract_min('A'), action: relax_edge('A', 'B', 4)) to animations",
                    severity=VerificationStatus.FAILED,
                )
            )
        else:
            verified_claims += 1

        # Check 2: Graph topology and source agreement
        total_checks += 1
        if observed_graph_source and observed_graph_source != source:
            discrepancies.append(
                FidelityDiscrepancy(
                    section_id=sections[0].get("id", "section_1") if sections else "section",
                    location="animations/visual_conventions",
                    category=FidelityCategory.GRAPH_SPEC_MISMATCH,
                    actual_claim=f"source = '{observed_graph_source}'",
                    expected_value=f"source = '{source}'",
                    reason=f"Starting source vertex mismatch: claimed '{observed_graph_source}', expected '{source}'",
                    suggested_correction=f"Initialize Dijkstra starting from source vertex '{source}'",
                    severity=VerificationStatus.FAILED,
                )
            )
        elif observed_extractions and observed_extractions[0]["node"] != source:
            discrepancies.append(
                FidelityDiscrepancy(
                    section_id=observed_extractions[0].get("sec_id", "worked_example"),
                    location=observed_extractions[0].get("loc", "animations"),
                    category=FidelityCategory.GRAPH_SPEC_MISMATCH,
                    actual_claim=f"First extracted node: '{observed_extractions[0]['node']}'",
                    expected_value=f"First extracted node: '{source}'",
                    reason=f"Dijkstra must extract starting source node '{source}' first with distance 0, but observed '{observed_extractions[0]['node']}'",
                    suggested_correction=f"Extract source node '{source}' first with distance 0",
                    severity=VerificationStatus.FAILED,
                )
            )
        else:
            verified_claims += 1

        if observed_graph_vertices:
            total_checks += 1
            if observed_graph_vertices != set(vertices):
                discrepancies.append(
                    FidelityDiscrepancy(
                        section_id=sections[0].get("id", "section_1") if sections else "section",
                        location="animations/visual_conventions",
                        category=FidelityCategory.GRAPH_SPEC_MISMATCH,
                        actual_claim=f"vertices = {sorted(list(observed_graph_vertices))}",
                        expected_value=f"vertices = {vertices}",
                        reason=f"Graph vertices mismatch: expected {vertices}, observed {sorted(list(observed_graph_vertices))}",
                        suggested_correction=f"Use graph with canonical vertices {vertices}",
                        severity=VerificationStatus.FAILED,
                    )
                )
            else:
                verified_claims += 1

        # Check 3: Every expected extraction appears (No intermediate step skipping)
        observed_nodes_extracted = [e["node"] for e in observed_extractions]
        missing_extractions = [node for node in settled_order if node not in observed_nodes_extracted]
        for m_node in missing_extractions:
            total_checks += 1
            discrepancies.append(
                FidelityDiscrepancy(
                    section_id="worked_example",
                    location="animations",
                    category=FidelityCategory.MISSING_ALGORITHMIC_STEP,
                    actual_claim=f"Node '{m_node}' extraction omitted",
                    expected_value=f"action: extract_min('{m_node}')",
                    reason=f"Canonical extraction of node '{m_node}' was omitted from storyboard visual transitions. All {len(settled_order)} extraction beats must be visualized.",
                    suggested_correction=f"Add visual extraction beat for node '{m_node}': action: extract_min('{m_node}')",
                    severity=VerificationStatus.FAILED,
                )
            )
        if not missing_extractions and settled_order:
            total_checks += 1
            verified_claims += 1

        # Check 4: Order of extractions matches canonical trace
        filtered_observed_order = [n for n in observed_nodes_extracted if n in settled_order]
        dedup_observed_order = []
        for n in filtered_observed_order:
            if not dedup_observed_order or dedup_observed_order[-1] != n:
                dedup_observed_order.append(n)

        total_checks += 1
        if len(dedup_observed_order) >= 2 and dedup_observed_order != settled_order[:len(dedup_observed_order)]:
            discrepancies.append(
                FidelityDiscrepancy(
                    section_id="worked_example",
                    location="animations",
                    category=FidelityCategory.ALGORITHMIC_ORDER_MISMATCH,
                    actual_claim=f"Extraction order {dedup_observed_order}",
                    expected_value=f"Canonical order {settled_order}",
                    reason=f"Extraction order mismatch: expected canonical order {settled_order}, but storyboard extracted in order {dedup_observed_order}",
                    suggested_correction=f"Reorder extraction beats to follow canonical minimum-distance order: {settled_order}",
                    severity=VerificationStatus.FAILED,
                )
            )
        else:
            verified_claims += 1

        # Check 5: Each required relaxation and distance update accuracy
        for step in canonical_trace:
            if step.get("step_type") != "iteration":
                continue
            step_u = step.get("extracted_node")
            for r in step.get("relaxations", []):
                total_checks += 1
                edge = r.get("edge")
                if edge and isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    u = str(edge[0])
                    v = str(edge[1])
                else:
                    u = str(r.get("u") or step_u)
                    v = str(r.get("v"))

                w = r.get("weight")
                prev_d = r.get("previous_dist")
                cand_d = r.get("candidate_dist")
                is_upd = r.get("updated") if r.get("updated") is not None else r.get("is_updated")

                # Match observed relaxation specifically for edge (u, v)
                rel_match = None
                for obs_r in observed_relaxations:
                    obs_u, obs_v = obs_r.get("u"), obs_r.get("v")
                    if obs_v == v and (obs_u == u or (not directed and obs_u == v and obs_v == u)):
                        rel_match = obs_r
                        break

                if not rel_match:
                    discrepancies.append(
                        FidelityDiscrepancy(
                            section_id="worked_example",
                            location="animations",
                            category=FidelityCategory.MISSING_ALGORITHMIC_STEP,
                            actual_claim=f"Relaxation of edge ({u}, {v}) missing",
                            expected_value=f"relax_edge({u}, {v}, candidate={cand_d})",
                            reason=f"Expected relaxation of edge ({u}, {v}) with weight {w} from node '{u}' is missing from storyboard visual transitions",
                            suggested_correction=f"Add visual relaxation step for edge ({u}, {v}) with candidate distance {cand_d}",
                            severity=VerificationStatus.FAILED,
                        )
                    )
                    continue

                if is_upd:
                    # Successful update: verify resulting distance value
                    claimed_cand = rel_match.get("candidate")
                    claimed_new = rel_match.get("new_dist")
                    val_to_check = claimed_new if claimed_new is not None else claimed_cand
                    if val_to_check is not None and abs(val_to_check - cand_d) > 1e-4:
                        discrepancies.append(
                            FidelityDiscrepancy(
                                section_id=rel_match.get("sec_id", "worked_example"),
                                location=rel_match.get("loc", "animations"),
                                category=FidelityCategory.ALGORITHMIC_STATE_MISMATCH,
                                actual_claim=f"dist({v}) = {val_to_check}",
                                expected_value=f"dist({v}) = {cand_d}",
                                reason=f"Incorrect updated distance for node '{v}' during relaxation of edge ({u}, {v}): claimed {val_to_check}, expected {cand_d}",
                                suggested_correction=f"Update distance of '{v}' to {cand_d}",
                                severity=VerificationStatus.FAILED,
                            )
                        )
                    else:
                        verified_claims += 1

                else:
                    # Unsuccessful relaxation: verify it was NOT presented as updating the distance
                    if rel_match.get("updated") is True:
                        claimed_val = rel_match.get("new_dist") or rel_match.get("candidate")
                        discrepancies.append(
                            FidelityDiscrepancy(
                                section_id=rel_match.get("sec_id", "worked_example"),
                                location=rel_match.get("loc", "animations"),
                                category=FidelityCategory.ALGORITHMIC_STATE_MISMATCH,
                                actual_claim=f"dist({v}) updated to {claimed_val}",
                                expected_value=f"No update (current dist {prev_d} <= candidate {cand_d})",
                                reason=f"Unsuccessful relaxation of edge ({u}, {v}) incorrectly presented as updating distance to {claimed_val} (candidate {cand_d} >= current {prev_d})",
                                suggested_correction=f"Show that candidate distance {cand_d} does not improve upon current distance {prev_d}, so no update occurs",
                                severity=VerificationStatus.FAILED,
                            )
                        )
                    else:
                        verified_claims += 1

        # Check 6: Final distance state agrees with verified final distances
        if observed_final_distances:
            for v_name, exp_final in final_distances.items():
                total_checks += 1
                if v_name in observed_final_distances:
                    claimed_f = observed_final_distances[v_name]
                    is_match = (math.isinf(claimed_f) and math.isinf(exp_final)) or (abs(claimed_f - exp_final) < 1e-4)
                    if not is_match:
                        discrepancies.append(
                            FidelityDiscrepancy(
                                section_id=sections[-1].get("id", "final_section") if sections else "final",
                                location="final_distances",
                                category=FidelityCategory.VALUE_MISMATCH,
                                actual_claim=f"dist({v_name}) = {claimed_f}",
                                expected_value=f"dist({v_name}) = {exp_final}",
                                reason=f"Final shortest distance for vertex '{v_name}' mismatch: claimed {claimed_f}, expected {exp_final}",
                                suggested_correction=f"Display correct final shortest distance {exp_final} for vertex '{v_name}'",
                                severity=VerificationStatus.FAILED,
                            )
                        )
                    else:
                        verified_claims += 1
                else:
                    verified_claims += 1

        # Determine overall report status
        has_failed = any(d.severity == VerificationStatus.FAILED for d in discrepancies)
        has_review = any(d.severity == VerificationStatus.NEEDS_REVIEW for d in discrepancies)

        if has_failed:
            overall_status = VerificationStatus.FAILED
            is_safe = False
        elif has_review:
            overall_status = VerificationStatus.NEEDS_REVIEW
            is_safe = False
        else:
            overall_status = VerificationStatus.VERIFIED
            is_safe = True

        summary = (
            f"Storyboard Dijkstra algorithmic fidelity: {'Fully verified coverage' if is_safe else overall_status.value}. "
            f"{verified_claims} verified claims, {len(discrepancies)} discrepancy/discrepancies flagged across {total_checks} checks."
        )

        return StoryboardFidelityReport(
            overall_status=overall_status,
            total_checks=total_checks,
            verified_claims=verified_claims,
            discrepancies=discrepancies,
            summary=summary,
            is_safe_for_audio=is_safe,
        )

    def _check_fcfs_fidelity(
        self, model: AuthoritativeOutlineModel, sections: List[Dict[str, Any]]
    ) -> StoryboardFidelityReport:
        """
        Validates FCFS CPU scheduling storyboard coverage, non-preemptive execution,
        FIFO dispatch order, Gantt chart intervals, metric accounting, and visual operations
        against the authoritative deterministic simulation trace.
        """
        algo_state = model.algo_state or model.fcfs_state or {}
        procs_raw = algo_state.get("processes", [])
        known_pids = [str(p.get("pid", "")) for p in procs_raw if p.get("pid")]
        proc_map = {str(p.get("pid", "")): p for p in procs_raw if p.get("pid")}
        gantt = algo_state.get("gantt_chart", [])
        metrics = algo_state.get("process_metrics", {})
        exp_avg_tat = algo_state.get("average_turnaround_time")
        exp_avg_wt = algo_state.get("average_waiting_time")

        discrepancies: List[FidelityDiscrepancy] = []
        total_checks = 0
        verified_claims = 0

        # 1. Generic component and operation validation via StoryboardActionResolver
        try:
            from src.domains.action_resolver import StoryboardActionResolver
            resolver_discs = StoryboardActionResolver.validate_storyboard_actions(
                {"sections": sections}, authoritative_trace=algo_state
            )
            for rd in resolver_discs:
                discrepancies.append(FidelityDiscrepancy(
                    section_id=rd.get("section_id", "unknown"),
                    location=rd.get("location", ""),
                    category=getattr(FidelityCategory, rd.get("category", "VALUE_MISMATCH"), FidelityCategory.VALUE_MISMATCH),
                    actual_claim=rd.get("actual_claim", ""),
                    expected_value=rd.get("expected_value"),
                    reason=rd.get("reason", ""),
                    suggested_correction=rd.get("suggested_correction"),
                    severity=VerificationStatus.FAILED,
                ))
                total_checks += 1
        except Exception:
            pass

        # 2. Track chronological execution state across sections
        completed_pids: Set[str] = set()
        dispatched_sequence: List[str] = []
        currently_running_pid: Optional[str] = None
        visual_action_count = 0
        all_dispatched_pids: Set[str] = set()
        all_gantt_pids: Set[str] = set()
        last_process_state: Dict[str, str] = {}
        input_spec_introduced = False
        has_any_animations_field = any("animations" in s for s in sections)

        for sec_idx, sec in enumerate(sections):
            sec_id = sec.get("id", f"section_{sec_idx+1}")
            anims = sec.get("animations", [])
            lines = sec.get("lecture_lines", [])
            vvals = sec.get("verified_values", [])

            all_text_sec = " ".join(lines) + " " + " ".join(vvals if isinstance(vvals, list) else [str(vvals)])

            # Check if this section introduces the input specification
            sec_title = str(sec.get("title", "") or sec.get("section_name", ""))
            sec_content = str(sec.get("content", ""))
            sec_comb = f"{sec_title} {sec_content} {all_text_sec}".lower()
            if (
                ("input" in sec_comb or "specification" in sec_comb or "arrival" in sec_comb)
                and ("burst" in sec_comb or "duration" in sec_comb)
                and any(pid.lower() in sec_comb for pid in known_pids)
            ):
                input_spec_introduced = True
            for s_str in anims:
                if isinstance(s_str, str) and (("arrival_time" in s_str and "burst_time" in s_str) or "InputTableVisual" in s_str):
                    input_spec_introduced = True

            # Check for forbidden preemption claims in spoken narration
            for line in lines:
                if re.search(r"\bpreempt(?:s|ed|ing|ion)?\b", line, re.IGNORECASE):
                    if not re.search(r"\b(?:not|never|no|non|without|cannot|disallow|prohibit|unallowed|free\s+of)\b", line, re.IGNORECASE):
                        total_checks += 1
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"{sec_id}.narration",
                            category=FidelityCategory.UNAUTHORIZED_PREEMPTION,
                            actual_claim=f"Narration claims preemption: '{line}'",
                            expected_value="Strict non-preemptive FCFS execution",
                            reason="FCFS CPU scheduling is strictly non-preemptive. A running process cannot be preempted",
                            suggested_correction="Allow running process to run to completion before dispatching next process",
                            severity=VerificationStatus.FAILED,
                        ))

            # Parse animation steps in this section
            for step_idx, step_str in enumerate(anims):
                if not isinstance(step_str, str):
                    continue
                visual_action_count += 1

                # Check for unsupported component operations
                unsupported_patterns = [
                    (r"\bhighlight_pid_at\b", "ProcessCardVisual does not support 'highlight_pid_at'. Use animate_highlight() or semantic handle."),
                    (r"\bhighlight_bt\b", "ProcessCardVisual does not support 'highlight_bt'. Use animate_highlight() or semantic handle."),
                    (r"\bhighlight_state\b", "ProcessCardVisual does not support 'highlight_state'. Use animate_set_state() or semantic handle."),
                    (r"\binitialize_structure\b", "ProcessCardVisual does not support 'initialize_structure'. Instantiate with pid, arrival_time, burst_time."),
                    (r"\bhighlight_scheduler\b", "CPUExecutionVisual does not have 'highlight_scheduler' sub-element. Use core_box or status_badge."),
                    (r"\bhighlight_dispatcher\b", "CPUExecutionVisual does not have 'highlight_dispatcher' sub-element. Use core_box or status_badge."),
                    (r"\bhighlight_context_switch\b", "CPUExecutionVisual does not have 'highlight_context_switch' sub-element. Use core_box."),
                ]
                for pat, reason_msg in unsupported_patterns:
                    if re.search(pat, step_str):
                        total_checks += 1
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"animations[{step_idx}]",
                            category=FidelityCategory.UNSUPPORTED_OPERATION,
                            actual_claim=step_str,
                            expected_value="Valid registered component method",
                            reason=reason_msg,
                            suggested_correction="Replace with supported component method",
                            severity=VerificationStatus.FAILED,
                        ))

                # Check for duplicate consecutive state transitions
                m_state = re.search(r"(?:animate_)?set_state\s*\(\s*(?:(?:new_state|state)\s*=\s*)?['\"]?([A-Za-z]+)['\"]?", step_str)
                if m_state:
                    new_st = m_state.group(1).upper()
                    target_pid = None
                    for kpid in known_pids:
                        if re.search(rf"\b{re.escape(kpid)}\b", step_str, re.IGNORECASE):
                            target_pid = kpid
                            break
                    if not target_pid:
                        target_pid = currently_running_pid or (dispatched_sequence[-1] if dispatched_sequence else (known_pids[0] if known_pids else "P1"))
                    prev_st = last_process_state.get(target_pid)
                    total_checks += 1
                    if new_st == "RUNNING" and not input_spec_introduced:
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"animations[{step_idx}]",
                            category=FidelityCategory.MISSING_INPUT_SPECIFICATION,
                            actual_claim=f"Process execution begins at '{sec_id}' without prior input specification",
                            expected_value="Explicit introduction of process Arrival Times and Burst Times (P1: AT=0, BT=3; P2: AT=1, BT=4; P3: AT=2, BT=2) before execution",
                            reason="The worked example must introduce the process arrival and burst times before execution begins so the learner understands the problem definition",
                            suggested_correction="Add an input specification beat introducing process Arrival Times and Burst Times before running processes",
                            severity=VerificationStatus.FAILED,
                        ))
                        input_spec_introduced = True
                    if prev_st == new_st:
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"animations[{step_idx}]",
                            category=FidelityCategory.INVALID_STATE_TRANSITION,
                            actual_claim=f"Process '{target_pid}' set_state('{new_st}') called while already in '{new_st}'",
                            expected_value=f"Meaningful state change for process '{target_pid}'",
                            reason=f"Duplicate consecutive state transition: process '{target_pid}' is already in state '{new_st}'",
                            suggested_correction=f"Remove redundant set_state('{new_st}') or add intervening transition",
                            severity=VerificationStatus.FAILED,
                        ))
                    else:
                        verified_claims += 1
                    last_process_state[target_pid] = new_st

                # Check dispatch action: e.g. animate_dispatch(item='P1') or dispatch('P1') or animate_dispatch(pid='P1')
                m_disp = re.search(r"(?:animate_)?dispatch\s*\(\s*(?:(?:process|process_card|item|pid)\s*=\s*)?['\"]?([A-Za-z0-9_]+)['\"]?", step_str)
                if m_disp and not input_spec_introduced:
                    total_checks += 1
                    discrepancies.append(FidelityDiscrepancy(
                        section_id=sec_id,
                        location=f"animations[{step_idx}]",
                        category=FidelityCategory.MISSING_INPUT_SPECIFICATION,
                        actual_claim=f"Process execution begins at '{sec_id}' without prior input specification",
                        expected_value="Explicit introduction of process Arrival Times and Burst Times (P1: AT=0, BT=3; P2: AT=1, BT=4; P3: AT=2, BT=2) before execution",
                        reason="The worked example must introduce the process arrival and burst times before execution begins so the learner understands the problem definition",
                        suggested_correction="Add an input specification beat introducing process Arrival Times and Burst Times before dispatching processes",
                        severity=VerificationStatus.FAILED,
                    ))
                    input_spec_introduced = True

                if m_disp:
                    disp_raw = m_disp.group(1).strip("'\"")
                    disp_pid = None
                    for kpid in known_pids:
                        if kpid.lower() == disp_raw.lower() or kpid.lower() in disp_raw.lower():
                            disp_pid = kpid
                            break
                    if disp_pid:
                        # Check preemption: was another process running and not completed?
                        if currently_running_pid and currently_running_pid != disp_pid and currently_running_pid not in completed_pids:
                            total_checks += 1
                            discrepancies.append(FidelityDiscrepancy(
                                section_id=sec_id,
                                location=f"animations[{step_idx}]",
                                category=FidelityCategory.UNAUTHORIZED_PREEMPTION,
                                actual_claim=f"Preempted '{currently_running_pid}' to dispatch '{disp_pid}'",
                                expected_value=f"Complete '{currently_running_pid}' before dispatching '{disp_pid}'",
                                reason=f"FCFS scheduling is non-preemptive. Process '{currently_running_pid}' has not completed its burst",
                                suggested_correction=f"Call complete for '{currently_running_pid}' before dispatching '{disp_pid}'",
                                severity=VerificationStatus.FAILED,
                            ))

                        # Check invalid repeated dispatch
                        if currently_running_pid == disp_pid:
                            total_checks += 1
                            discrepancies.append(FidelityDiscrepancy(
                                section_id=sec_id,
                                location=f"animations[{step_idx}]",
                                category=FidelityCategory.INVALID_REPEATED_DISPATCH,
                                actual_claim=f"Repeated dispatch of running process '{disp_pid}'",
                                expected_value=f"Allow running process '{disp_pid}' to complete before dispatching next process",
                                reason=f"Invalid repeated dispatch: process '{disp_pid}' is already executing on the CPU",
                                suggested_correction=f"Do not dispatch '{disp_pid}' again while already running",
                                severity=VerificationStatus.FAILED,
                            ))
                        elif disp_pid in completed_pids:
                            total_checks += 1
                            discrepancies.append(FidelityDiscrepancy(
                                section_id=sec_id,
                                location=f"animations[{step_idx}]",
                                category=FidelityCategory.INVALID_REPEATED_DISPATCH,
                                actual_claim=f"Repeated dispatch of terminated process '{disp_pid}'",
                                expected_value=f"Dispatch remaining uncompleted processes",
                                reason=f"Invalid repeated dispatch: process '{disp_pid}' has already completed execution and terminated",
                                suggested_correction=f"Dispatch an uncompleted process instead of '{disp_pid}'",
                                severity=VerificationStatus.FAILED,
                            ))
                        else:
                            all_dispatched_pids.add(disp_pid)

                        currently_running_pid = disp_pid
                        dispatched_sequence.append(disp_pid)
                        total_checks += 1
                        verified_claims += 1

                # Check complete action: e.g. animate_complete(pid='P1')
                m_comp = re.search(r"(?:animate_)?complete\s*\(\s*(?:(?:process|pid)\s*=\s*)?['\"]?([A-Za-z0-9_]*)['\"]?", step_str)
                if m_comp:
                    comp_raw = m_comp.group(1).strip("'\"")
                    comp_pid = None
                    for kpid in known_pids:
                        if kpid.lower() == comp_raw.lower() or kpid.lower() in comp_raw.lower():
                            comp_pid = kpid
                            break
                    if comp_pid:
                        completed_pids.add(comp_pid)
                    elif currently_running_pid:
                        completed_pids.add(currently_running_pid)
                    elif dispatched_sequence:
                        completed_pids.add(dispatched_sequence[-1])
                    currently_running_pid = None

                # Check Gantt add block action: animate_add_block(pid='P1', start_time=0, end_time=3) or start=0, end=3
                m_block = re.search(r"(?:animate_)?add_block\s*\(\s*(?:pid\s*=\s*)?['\"]?([A-Za-z0-9_]+)['\"]?\s*,\s*(?:(?:start_time|start|s)\s*=\s*)?(\d+)\s*,\s*(?:(?:end_time|end|e)\s*=\s*)?(\d+)", step_str)
                if m_block and not input_spec_introduced:
                    total_checks += 1
                    discrepancies.append(FidelityDiscrepancy(
                        section_id=sec_id,
                        location=f"animations[{step_idx}]",
                        category=FidelityCategory.MISSING_INPUT_SPECIFICATION,
                        actual_claim=f"Gantt timeline execution begins at '{sec_id}' without prior input specification",
                        expected_value="Explicit introduction of process Arrival Times and Burst Times (P1: AT=0, BT=3; P2: AT=1, BT=4; P3: AT=2, BT=2) before execution",
                        reason="The worked example must introduce the process arrival and burst times before execution begins so the learner understands the problem definition",
                        suggested_correction="Add an input specification beat introducing process Arrival Times and Burst Times before timeline construction",
                        severity=VerificationStatus.FAILED,
                    ))
                    input_spec_introduced = True
                if m_block:
                    b_pid = m_block.group(1)
                    b_start = int(m_block.group(2))
                    b_end = int(m_block.group(3))
                    total_checks += 1

                    matched = any(
                        b.get("pid") == b_pid and b.get("start_time") == b_start and b.get("end_time") == b_end
                        for b in gantt
                    )
                    if matched:
                        verified_claims += 1
                        completed_pids.add(b_pid)
                        all_gantt_pids.add(b_pid)
                    else:
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"animations[{step_idx}]",
                            category=FidelityCategory.GANTT_TIMELINE_MISMATCH,
                            actual_claim=f"Gantt block ({b_pid}, {b_start}->{b_end})",
                            expected_value=str([(b.get("pid"), b.get("start_time"), b.get("end_time")) for b in gantt]),
                            reason=f"Gantt interval [{b_start}, {b_end}] for process '{b_pid}' contradicts authoritative trace",
                            suggested_correction=f"Align Gantt block with simulation intervals",
                            severity=VerificationStatus.FAILED,
                        ))

                # Check premature visual metric updates in animations
                if "update_process_metrics" in step_str:
                    m_p = re.search(r"pid\s*=\s*['\"]?([A-Za-z0-9_]+)['\"]?", step_str)
                    if m_p:
                        act_pid = m_p.group(1)
                        if act_pid not in completed_pids and act_pid not in dispatched_sequence:
                            total_checks += 1
                            discrepancies.append(FidelityDiscrepancy(
                                section_id=sec_id,
                                location=f"animations[{step_idx}]",
                                category=FidelityCategory.PREMATURE_METRIC_LEAK,
                                actual_claim=f"Premature update_process_metrics for '{act_pid}'",
                                expected_value=f"Update metrics after process '{act_pid}' completes execution",
                                reason=f"Premature metric leak: Process '{act_pid}' has not completed execution yet",
                                suggested_correction=f"Call update_process_metrics for '{act_pid}' only after it terminates",
                                severity=VerificationStatus.FAILED,
                            ))

            # Check lecture_lines and verified_values for premature metric claims or false numbers
            for pid in known_pids:
                m_exp = metrics.get(pid, {})
                exp_ct = m_exp.get("completion_time")
                exp_tat = m_exp.get("turnaround_time")
                exp_wt = m_exp.get("waiting_time")
                exp_rt = m_exp.get("response_time")

                other_pids = [p for p in known_pids if p != pid]
                other_pattern = "|".join(re.escape(p) for p in other_pids) or r"(?!)"

                # CT match: bounded clause or direct alias, avoiding cross-process leaks
                p_ct = rf'(?:(?:completion\s+time|CT)\s*(?:of|for)?\s*{pid}\s*(?:is|=|:)?\s*(\d+)|{pid}\b(?:(?!\b(?:{other_pattern})\b)[^.;\n]){{0,60}}?\b(?:completion\s+time|CT|completes?\s+at)\s*(?:is|=|:)?\s*(\d+))'
                ct_matches = [g1 or g2 for g1, g2 in re.findall(p_ct, all_text_sec, re.IGNORECASE)]
                for ct_val in ct_matches:
                    total_checks += 1
                    if int(ct_val) == exp_ct:
                        # Premature check: was pid completed or running in or before this section?
                        if pid not in completed_pids and pid not in dispatched_sequence:
                            discrepancies.append(FidelityDiscrepancy(
                                section_id=sec_id,
                                location=f"{sec_id}.narration",
                                category=FidelityCategory.PREMATURE_METRIC_LEAK,
                                actual_claim=f"{pid} CT = {ct_val}",
                                expected_value=f"Reveal CT after process completes at tick {exp_ct}",
                                reason=f"Premature metric leak: Completion time for '{pid}' claimed before process execution completed",
                                suggested_correction=f"State completion time after '{pid}' completes",
                                severity=VerificationStatus.FAILED,
                            ))
                        else:
                            verified_claims += 1
                    else:
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"{sec_id}.narration",
                            category=FidelityCategory.VALUE_MISMATCH,
                            actual_claim=f"{pid} CT = {ct_val}",
                            expected_value=str(exp_ct),
                            reason=f"Completion time for '{pid}' is {exp_ct}, but claimed {ct_val}",
                            suggested_correction=f"Set {pid} Completion Time = {exp_ct}",
                            severity=VerificationStatus.FAILED,
                        ))

                # TAT match: bounded clause or direct alias, avoiding cross-process leaks
                p_tat = rf'(?:(?:turnaround\s+time|TAT)\s*(?:of|for)?\s*{pid}\s*(?:is|=|:)?\s*(\d+)|{pid}\b(?:(?!\b(?:{other_pattern})\b)[^.;\n]){{0,60}}?\b(?:turnaround\s+time|TAT)\s*(?:is|=|:)?\s*(\d+))'
                tat_matches = [g1 or g2 for g1, g2 in re.findall(p_tat, all_text_sec, re.IGNORECASE)]
                for tat_val in tat_matches:
                    total_checks += 1
                    if int(tat_val) == exp_tat:
                        if pid not in completed_pids and pid not in dispatched_sequence:
                            discrepancies.append(FidelityDiscrepancy(
                                section_id=sec_id,
                                location=f"{sec_id}.narration",
                                category=FidelityCategory.PREMATURE_METRIC_LEAK,
                                actual_claim=f"{pid} TAT = {tat_val}",
                                expected_value=f"Reveal TAT after process completes at tick {exp_ct}",
                                reason=f"Premature metric leak: Turnaround time for '{pid}' claimed before process execution completed",
                                suggested_correction=f"State turnaround time after '{pid}' completes",
                                severity=VerificationStatus.FAILED,
                            ))
                        else:
                            verified_claims += 1
                    else:
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"{sec_id}.narration",
                            category=FidelityCategory.VALUE_MISMATCH,
                            actual_claim=f"{pid} TAT = {tat_val}",
                            expected_value=str(exp_tat),
                            reason=f"Turnaround time for '{pid}' is {exp_tat}, but claimed {tat_val}",
                            suggested_correction=f"Set {pid} Turnaround Time = {exp_tat}",
                            severity=VerificationStatus.FAILED,
                        ))

                # WT match: bounded clause or direct alias, avoiding cross-process leaks
                p_wt = rf'(?:(?:waiting\s+time|WT)\s*(?:of|for)?\s*{pid}\s*(?:is|=|:)?\s*(\d+)|{pid}\b(?:(?!\b(?:{other_pattern})\b)[^.;\n]){{0,60}}?\b(?:waiting\s+time|WT)\s*(?:is|=|:)?\s*(\d+))'
                wt_matches = [g1 or g2 for g1, g2 in re.findall(p_wt, all_text_sec, re.IGNORECASE)]
                for wt_val in wt_matches:
                    total_checks += 1
                    if int(wt_val) == exp_wt:
                        if pid not in completed_pids and pid not in dispatched_sequence:
                            discrepancies.append(FidelityDiscrepancy(
                                section_id=sec_id,
                                location=f"{sec_id}.narration",
                                category=FidelityCategory.PREMATURE_METRIC_LEAK,
                                actual_claim=f"{pid} WT = {wt_val}",
                                expected_value=f"Reveal WT after process completes at tick {exp_ct}",
                                reason=f"Premature metric leak: Waiting time for '{pid}' claimed before process execution completed",
                                suggested_correction=f"State waiting time after '{pid}' completes",
                                severity=VerificationStatus.FAILED,
                            ))
                        else:
                            verified_claims += 1
                    else:
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"{sec_id}.narration",
                            category=FidelityCategory.VALUE_MISMATCH,
                            actual_claim=f"{pid} WT = {wt_val}",
                            expected_value=str(exp_wt),
                            reason=f"Waiting time for '{pid}' is {exp_wt}, but claimed {wt_val}",
                            suggested_correction=f"Set {pid} Waiting Time = {exp_wt}",
                            severity=VerificationStatus.FAILED,
                        ))

                # RT match: bounded clause or direct alias, avoiding cross-process leaks
                if exp_rt is not None:
                    p_rt = rf'(?:(?:response\s+time|RT)\s*(?:of|for)?\s*{pid}\s*(?:is|=|:)?\s*(\d+)|{pid}\b(?:(?!\b(?:{other_pattern})\b)[^.;\n]){{0,60}}?\b(?:response\s+time|RT)\s*(?:is|=|:)?\s*(\d+))'
                    rt_matches = [g1 or g2 for g1, g2 in re.findall(p_rt, all_text_sec, re.IGNORECASE)]
                    for rt_val in rt_matches:
                        total_checks += 1
                        if int(rt_val) == exp_rt:
                            if pid not in dispatched_sequence and pid not in completed_pids:
                                discrepancies.append(FidelityDiscrepancy(
                                    section_id=sec_id,
                                    location=f"{sec_id}.narration",
                                    category=FidelityCategory.PREMATURE_METRIC_LEAK,
                                    actual_claim=f"{pid} RT = {rt_val}",
                                    expected_value=f"Reveal RT after process is first dispatched to CPU",
                                    reason=f"Premature metric leak: Response time for '{pid}' claimed before process was dispatched to CPU",
                                    suggested_correction=f"State response time after '{pid}' is dispatched",
                                    severity=VerificationStatus.FAILED,
                                ))
                            else:
                                verified_claims += 1
                        else:
                            discrepancies.append(FidelityDiscrepancy(
                                section_id=sec_id,
                                location=f"{sec_id}.narration",
                                category=FidelityCategory.VALUE_MISMATCH,
                                actual_claim=f"{pid} RT = {rt_val}",
                                expected_value=str(exp_rt),
                                reason=f"Response time for '{pid}' is {exp_rt}, but claimed {rt_val}",
                                suggested_correction=f"Set {pid} Response Time = {exp_rt}",
                                severity=VerificationStatus.FAILED,
                            ))

            # Check average WT and average TAT
            if exp_avg_wt is not None:
                avg_wt_matches = re.findall(r"(?:average\s+waiting\s+time|avg\s+wt)\s*(?:is|=|:)?\s*(\d+\.?\d*)", all_text_sec, re.IGNORECASE)
                for claimed_avg_wt in avg_wt_matches:
                    total_checks += 1
                    val = float(claimed_avg_wt)
                    if abs(val - float(exp_avg_wt)) < 0.05:
                        verified_claims += 1
                    else:
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"{sec_id}.narration",
                            category=FidelityCategory.VALUE_MISMATCH,
                            actual_claim=f"Average Waiting Time = {claimed_avg_wt}",
                            expected_value=f"{exp_avg_wt:.2f}",
                            reason=f"Average waiting time is {exp_avg_wt:.2f}, but claimed {claimed_avg_wt}",
                            suggested_correction=f"Set Average Waiting Time = {exp_avg_wt:.2f}",
                            severity=VerificationStatus.FAILED,
                        ))

            if exp_avg_tat is not None:
                avg_tat_matches = re.findall(r"(?:average\s+turnaround\s+time|avg\s+tat)\s*(?:is|=|:)?\s*(\d+\.?\d*)", all_text_sec, re.IGNORECASE)
                for claimed_avg_tat in avg_tat_matches:
                    total_checks += 1
                    val = float(claimed_avg_tat)
                    if abs(val - float(exp_avg_tat)) < 0.05:
                        verified_claims += 1
                    else:
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"{sec_id}.narration",
                            category=FidelityCategory.VALUE_MISMATCH,
                            actual_claim=f"Average Turnaround Time = {claimed_avg_tat}",
                            expected_value=f"{exp_avg_tat:.2f}",
                            reason=f"Average turnaround time is {exp_avg_tat:.2f}, but claimed {claimed_avg_tat}",
                            suggested_correction=f"Set Average Turnaround Time = {exp_avg_tat:.2f}",
                            severity=VerificationStatus.FAILED,
                        ))

            exp_avg_rt = algo_state.get("average_response_time")
            if exp_avg_rt is not None:
                avg_rt_matches = re.findall(r"(?:average\s+response\s+time|avg\s+rt)\s*(?:is|=|:)?\s*(\d+\.?\d*)", all_text_sec, re.IGNORECASE)
                for claimed_avg_rt in avg_rt_matches:
                    total_checks += 1
                    val = float(claimed_avg_rt)
                    if abs(val - float(exp_avg_rt)) < 0.05:
                        verified_claims += 1
                    else:
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sec_id,
                            location=f"{sec_id}.narration",
                            category=FidelityCategory.VALUE_MISMATCH,
                            actual_claim=f"Average Response Time = {claimed_avg_rt}",
                            expected_value=f"{exp_avg_rt:.2f}",
                            reason=f"Average response time is {exp_avg_rt:.2f}, but claimed {claimed_avg_rt}",
                            suggested_correction=f"Set Average Response Time = {exp_avg_rt:.2f}",
                            severity=VerificationStatus.FAILED,
                        ))

            # Check for excessive consecutive generic highlights
            if anims and len(anims) >= 3:
                consecutive_ind = 0
                max_consecutive_ind = 0
                total_ind = 0
                for s_str in anims:
                    if not isinstance(s_str, str):
                        continue
                    is_substantive = any(kw in s_str for kw in ("dispatch", "complete", "set_state", "add_block", "enqueue", "dequeue", "update_process_metrics", "show_averages"))
                    is_ind = bool(re.search(r"\b(?:Indicate|animate_highlight|highlight)\b", s_str)) and not is_substantive
                    if is_ind:
                        consecutive_ind += 1
                        total_ind += 1
                        if consecutive_ind > max_consecutive_ind:
                            max_consecutive_ind = consecutive_ind
                    else:
                        consecutive_ind = 0
                if max_consecutive_ind >= 3 or total_ind == len(anims):
                    total_checks += 1
                    discrepancies.append(FidelityDiscrepancy(
                        section_id=sec_id,
                        location=f"{sec_id}.animations",
                        category=FidelityCategory.EXCESSIVE_HIGHLIGHTING,
                        actual_claim=f"{total_ind} generic highlight/Indicate actions ({max_consecutive_ind} consecutive)",
                        expected_value="Purposeful animation communicating state changes, queue movements, or timeline progress",
                        reason=f"Excessive generic highlighting in '{sec_id}': repetitive pulsing instead of meaningful visual state transitions",
                        suggested_correction="Replace repetitive indicates with meaningful domain state changes or remove redundant pulses",
                        severity=VerificationStatus.FAILED,
                    ))

            # Check for conflicting primary-visual assignments in cramped viewports
            has_metric_inspector = any("MetricTableVisual" in s_str and "dock_inspector" in s_str for s_str in anims if isinstance(s_str, str))
            has_gantt_inspector = any("GanttChartVisual" in s_str and "dock_inspector" in s_str for s_str in anims if isinstance(s_str, str))
            if has_metric_inspector and has_gantt_inspector:
                total_checks += 1
                discrepancies.append(FidelityDiscrepancy(
                    section_id=sec_id,
                    location=f"{sec_id}.layout",
                    category=FidelityCategory.VIEWPORT_CONFLICT,
                    actual_claim="Both MetricTableVisual and GanttChartVisual docked to dock_inspector",
                    expected_value="MetricTableVisual in dock_main",
                    reason="Conflicting primary-visual assignments: dock_inspector is a 1.0-unit secondary strip; docking MetricTableVisual to dock_inspector causes extreme downscaling",
                    suggested_correction="Assign MetricTableVisual to dock_main and GanttChartVisual to dock_inspector or dock_main",
                    severity=VerificationStatus.NEEDS_REVIEW,
                ))

        # 3. Check FIFO Dispatch Order
        if dispatched_sequence:
            expected_order = sorted(known_pids, key=lambda pid: proc_map.get(pid, {}).get("arrival_time", 0))
            first_arriving = expected_order[0]
            first_disp = dispatched_sequence[0]
            at_first_arriving = proc_map.get(first_arriving, {}).get("arrival_time", 0)
            at_first_disp = proc_map.get(first_disp, {}).get("arrival_time", 0)
            if at_first_disp > at_first_arriving:
                total_checks += 1
                discrepancies.append(FidelityDiscrepancy(
                    section_id=sections[0].get("id", "section_1"),
                    location="dispatch_order",
                    category=FidelityCategory.INVALID_DISPATCH_ORDER,
                    actual_claim=f"Dispatched '{first_disp}' first (AT={at_first_disp})",
                    expected_value=f"FIFO dispatch order: {expected_order}",
                    reason=f"Inverted FIFO dispatch: Process '{first_arriving}' arrived earlier ({at_first_arriving}) than '{first_disp}' ({at_first_disp}) and must be dispatched first",
                    suggested_correction=f"Dispatch '{first_arriving}' first",
                    severity=VerificationStatus.FAILED,
                ))

            if len(dispatched_sequence) >= 2:
                for i in range(len(dispatched_sequence) - 1):
                    p_cur = dispatched_sequence[i]
                    p_next = dispatched_sequence[i + 1]
                    at_cur = proc_map.get(p_cur, {}).get("arrival_time", 0)
                    at_next = proc_map.get(p_next, {}).get("arrival_time", 0)
                    if at_cur > at_next:
                        total_checks += 1
                        discrepancies.append(FidelityDiscrepancy(
                            section_id=sections[0].get("id", "section_1"),
                            location="dispatch_order",
                            category=FidelityCategory.INVALID_DISPATCH_ORDER,
                            actual_claim=f"Dispatched '{p_cur}' (AT={at_cur}) before '{p_next}' (AT={at_next})",
                            expected_value=f"FIFO dispatch order: {expected_order}",
                            reason=f"Inverted FIFO dispatch: Process '{p_next}' arrived earlier ({at_next}) than '{p_cur}' ({at_cur}) and must be dispatched first",
                            suggested_correction=f"Dispatch '{p_next}' before '{p_cur}'",
                            severity=VerificationStatus.FAILED,
                        ))

        # 4. Check Visual Action Realization
        has_any_animations_field = any("animations" in s for s in sections)
        if has_any_animations_field and visual_action_count == 0:
            total_checks += 1
            discrepancies.append(FidelityDiscrepancy(
                section_id=sections[0].get("id", "section_1"),
                location="animations",
                category=FidelityCategory.INSUFFICIENT_VISUAL_COVERAGE,
                actual_claim="Zero visual animation steps across storyboard",
                expected_value="Visual actions realizing FCFS scheduling",
                reason="Storyboard describes FCFS operations in spoken narration but does not provide visual state transitions in animations",
                suggested_correction="Add animation steps using ProcessCardVisual, ReadyQueueVisual, CPUExecutionVisual, GanttChartVisual, or MetricTableVisual",
                severity=VerificationStatus.FAILED,
            ))
        elif visual_action_count > 0:
            verified_claims += 1
            total_checks += 1

        # 5. Check Explicit Execution Coverage Across All Processes
        if has_any_animations_field and visual_action_count > 0:
            missing_dispatch = [pid for pid in known_pids if all_dispatched_pids and pid not in all_dispatched_pids and pid not in all_gantt_pids]
            missing_completion = [pid for pid in known_pids if pid not in completed_pids and pid not in all_gantt_pids]
            if missing_dispatch or missing_completion:
                total_checks += 1
                missing_all = sorted(list(set(missing_dispatch) | set(missing_completion)))
                discrepancies.append(FidelityDiscrepancy(
                    section_id=sections[-1].get("id", "section_end") if sections else "storyboard",
                    location="execution_coverage",
                    category=FidelityCategory.INCOMPLETE_PROCESS_COVERAGE,
                    actual_claim=f"Process(es) {missing_all} not fully executed to completion",
                    expected_value=f"All processes {known_pids} explicitly dispatched and completed in animations",
                    reason=f"Incomplete FCFS execution coverage: Process(es) {missing_all} were not dispatched and executed through completion in storyboard animations",
                    suggested_correction=f"Add dispatch, execute tick, and completion animations for {missing_all}",
                    severity=VerificationStatus.FAILED,
                ))
            else:
                verified_claims += 1
                total_checks += 1

        # 6. Final Status
        has_failed = any(d.severity == VerificationStatus.FAILED for d in discrepancies)
        has_needs_review = any(d.severity == VerificationStatus.NEEDS_REVIEW for d in discrepancies)
        is_safe = not has_failed
        overall_status = (
            VerificationStatus.FAILED if has_failed
            else (VerificationStatus.NEEDS_REVIEW if has_needs_review else VerificationStatus.VERIFIED)
        )

        summary = (
            f"Storyboard FCFS algorithmic fidelity: {'Fully verified coverage' if is_safe else overall_status.value}. "
            f"{verified_claims} verified claims, {len(discrepancies)} discrepancy/discrepancies flagged across {total_checks} checks."
        )

        return StoryboardFidelityReport(
            overall_status=overall_status,
            total_checks=total_checks,
            verified_claims=verified_claims,
            discrepancies=discrepancies,
            summary=summary,
            is_safe_for_audio=is_safe,
        )

    def _extract_sections(self, storyboard: Any) -> List[Dict[str, Any]]:
        """Normalizes various storyboard inputs into a list of section dicts."""
        if isinstance(storyboard, dict):
            if "sections" in storyboard and isinstance(storyboard["sections"], list):
                res = []
                for s in storyboard["sections"]:
                    if hasattr(s, "__dict__"):
                        res.append(s.__dict__)
                    elif isinstance(s, dict):
                        res.append(s)
                return res
            return [storyboard]

        if isinstance(storyboard, list):
            res = []
            for item in storyboard:
                if hasattr(item, "__dict__"):
                    res.append(item.__dict__)
                elif isinstance(item, dict):
                    res.append(item)
            return res

        return []

    def _extract_step_ref_from_sec(self, sec: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
        """Extracts an authoritative step reference from section metadata if present."""
        if "step_index" in sec and sec["step_index"] is not None:
            return ("index", sec["step_index"])
        if "step_number" in sec and sec["step_number"] is not None:
            return ("number", sec["step_number"])
        if "step" in sec and sec["step"] is not None:
            return ("number", sec["step"])
        sec_id = str(sec.get("id", ""))
        m_id = re.search(r"\bstep[_\s]*(\d+)\b", sec_id, re.IGNORECASE)
        if m_id:
            return ("number", int(m_id.group(1)))
        title = str(sec.get("title", ""))
        m_title = re.search(r"\bstep[_\s]*(\d+)\b", title, re.IGNORECASE)
        if m_title:
            return ("number", int(m_title.group(1)))
        return None

    def _extract_step_ref_from_text(self, text: str) -> Optional[Tuple[str, Any]]:
        """Extracts an explicit step reference from string claims or variable keys."""
        m = re.search(r"\bstep[_\s]*(\d+)\b", text, re.IGNORECASE)
        if m:
            return ("number", int(m.group(1)))
        return None

    # ------------------------------------------------------------------------
    # 1. Checked Elements: verified_values
    # ------------------------------------------------------------------------
    def _check_verified_values(
        self, sec: Dict[str, Any], model: AuthoritativeOutlineModel
    ) -> Tuple[List[FidelityDiscrepancy], int, int, Set[str]]:
        disc: List[FidelityDiscrepancy] = []
        checks = 0
        verified = 0
        matched: Set[str] = set()

        sec_id = sec.get("id", "section")
        sec_step = self._extract_step_ref_from_sec(sec)
        vvals = sec.get("verified_values")
        if not vvals or not isinstance(vvals, dict):
            return disc, checks, verified, matched

        for raw_k, raw_v in vvals.items():
            checks += 1
            k_str = str(raw_k).strip()
            v_str = str(raw_v).strip()
            loc = f"verified_values['{k_str}']"

            key_step = self._extract_step_ref_from_text(k_str)
            effective_step = key_step if key_step is not None else sec_step

            # If an explicit step is referenced, verify that the step exists in the outline
            if effective_step is not None and not model.has_step(effective_step):
                ref_disp = effective_step[1] if isinstance(effective_step, tuple) else effective_step
                disc.append(
                    FidelityDiscrepancy(
                        section_id=sec_id,
                        location=loc,
                        category=FidelityCategory.AMBIGUOUS_CLAIM,
                        actual_claim=v_str,
                        reason=f"Claim references step '{ref_disp}' which is out-of-range or unavailable in the authoritative outline",
                        severity=VerificationStatus.NEEDS_REVIEW,
                    )
                )
                continue

            # Check for hallucination relapse
            relapse = self._detect_hallucination_relapse(v_str, model)
            if relapse:
                disc.append(
                    FidelityDiscrepancy(
                        section_id=sec_id,
                        location=loc,
                        category=FidelityCategory.HALLUCINATION_RELAPSE,
                        actual_claim=v_str,
                        expected_value=relapse["corrected_value"],
                        reason=f"Claim matches rejected hallucinated value '{relapse['rejected_float']}' for {relapse['var_name']}",
                        suggested_correction=f"{k_str} = {relapse['corrected_value']}",
                        severity=VerificationStatus.FAILED,
                    )
                )
                continue

            # Determine whether v_str is an assignment 'var = val' or an arithmetic equation
            claimed_value_str = v_str
            is_equation = False
            if "=" in v_str:
                eq_parts = [p.strip() for p in v_str.split("=")]
                lhs_cand = eq_parts[0]
                clean_ident = re.sub(r"[\{\}\$\\]", "", lhs_cand).replace("_", "").strip()
                has_operators = any(op in lhs_cand for op in ("+", "-", "*", "/", "^", "**", "exp(", "ln(")) or r"\frac" in lhs_cand
                if clean_ident.isalnum() and not has_operators:
                    # Variable assignment like 'k1 = 0.006545' or 'k1 = approximately ????'
                    claimed_value_str = eq_parts[-1]
                else:
                    is_equation = True

            if is_equation:
                eq_disc = self._verify_standalone_equation(sec_id, loc, v_str, model, step_ref=effective_step)
                if eq_disc:
                    disc.append(eq_disc)
                    continue
                else:
                    verified += 1
                    continue

            # Parse claimed precision and unit
            try:
                prec = self._extract_precision(claimed_value_str)
            except Exception as e:
                disc.append(
                    FidelityDiscrepancy(
                        section_id=sec_id,
                        location=loc,
                        category=FidelityCategory.AMBIGUOUS_CLAIM,
                        actual_claim=v_str,
                        reason=f"Could not parse numerical value in verified_values entry '{v_str}': {e}",
                        severity=VerificationStatus.NEEDS_REVIEW,
                    )
                )
                continue

            norm_k = model._normalize_var_name(k_str)
            target = self._find_matching_target(norm_k, claimed_value_str, model, step_ref=effective_step)

            if target:
                matched.add(target.name)
                # Check numerical value
                is_match, diag, details = ToleranceEvaluator.evaluate(prec, target.float_value)
                if not is_match:
                    disc.append(
                        FidelityDiscrepancy(
                            section_id=sec_id,
                            location=loc,
                            category=FidelityCategory.VALUE_MISMATCH,
                            actual_claim=v_str,
                            expected_value=target.formatted_value,
                            expected_unit=target.unit,
                            reason=f"Numerical mismatch with Stage 1 verified target '{target.name}': {details}",
                            suggested_correction=f"{target.formatted_value} {target.unit or ''}".strip(),
                            severity=VerificationStatus.FAILED,
                        )
                    )
                    continue

                # Check unit compatibility
                if target.unit and prec.unit:
                    is_compat, is_scaled = units_are_compatible(prec.unit, target.unit)
                    if not is_compat:
                        disc.append(
                            FidelityDiscrepancy(
                                section_id=sec_id,
                                location=loc,
                                category=FidelityCategory.UNIT_MISMATCH,
                                actual_claim=v_str,
                                expected_value=target.formatted_value,
                                expected_unit=target.unit,
                                reason=f"Unit '{prec.unit}' conflicts with Stage 1 verified unit '{target.unit}'",
                                suggested_correction=f"{target.formatted_value} {target.unit}",
                                severity=VerificationStatus.FAILED,
                            )
                        )
                        continue

                # Check contextual mix-ups (e.g. T1 vs T2)
                ctx_err = self._check_context_mixup(norm_k, v_str, target, model)
                if ctx_err:
                    disc.append(
                        FidelityDiscrepancy(
                            section_id=sec_id,
                            location=loc,
                            category=FidelityCategory.CONTEXT_MISMATCH,
                            actual_claim=v_str,
                            expected_value=target.formatted_value,
                            expected_unit=target.unit,
                            reason=ctx_err,
                            severity=VerificationStatus.FAILED,
                        )
                    )
                    continue

                verified += 1
            else:
                verified += 1

        return disc, checks, verified, matched

    # ------------------------------------------------------------------------
    # 2. Checked Elements: animations (MathTex & Text)
    # ------------------------------------------------------------------------
    def _check_animations(
        self, sec: Dict[str, Any], model: AuthoritativeOutlineModel
    ) -> Tuple[List[FidelityDiscrepancy], int, int, Set[str]]:
        disc: List[FidelityDiscrepancy] = []
        checks = 0
        verified = 0
        matched: Set[str] = set()

        sec_id = sec.get("id", "section")
        sec_step = self._extract_step_ref_from_sec(sec)
        anims = sec.get("animations", [])
        if not isinstance(anims, list):
            return disc, checks, verified, matched

        for i, anim in enumerate(anims):
            anim_str = str(anim)
            loc = f"animations[{i}]"
            anim_step = self._extract_step_ref_from_text(anim_str)

            # Extract string literals inside MathTex(...) or Text(...)
            extracted_literals = self._extract_display_strings(anim_str)
            for lit in extracted_literals:
                checks += 1
                lit_clean = lit.strip()
                lit_step = self._extract_step_ref_from_text(lit_clean)
                effective_step = lit_step if lit_step is not None else (anim_step if anim_step is not None else sec_step)

                # If an explicit step is referenced, verify that the step exists in the outline
                if effective_step is not None and not model.has_step(effective_step):
                    ref_disp = effective_step[1] if isinstance(effective_step, tuple) else effective_step
                    disc.append(
                        FidelityDiscrepancy(
                            section_id=sec_id,
                            location=loc,
                            category=FidelityCategory.AMBIGUOUS_CLAIM,
                            actual_claim=lit_clean,
                            reason=f"Animation claim references step '{ref_disp}' which is out-of-range or unavailable in the authoritative outline",
                            severity=VerificationStatus.NEEDS_REVIEW,
                        )
                    )
                    continue

                # Check for hallucination relapse
                relapse = self._detect_hallucination_relapse(lit_clean, model)
                if relapse:
                    disc.append(
                        FidelityDiscrepancy(
                            section_id=sec_id,
                            location=loc,
                            category=FidelityCategory.HALLUCINATION_RELAPSE,
                            actual_claim=lit_clean,
                            expected_value=relapse["corrected_value"],
                            reason=f"MathTex/Text contains rejected hallucinated value '{relapse['rejected_float']}' for {relapse['var_name']}",
                            suggested_correction=f"{relapse['var_name']} = {relapse['corrected_value']}",
                            severity=VerificationStatus.FAILED,
                        )
                    )
                    continue

                # Check explicit equality claims in MathTex: e.g. 'k1 = ...'
                if "=" in lit_clean:
                    eq_disc, is_v, m_name = self._verify_mathtex_equality(sec_id, loc, lit_clean, model, step_ref=effective_step)
                    if eq_disc:
                        disc.append(eq_disc)
                    elif is_v:
                        verified += 1
                        if m_name:
                            matched.add(m_name)
                else:
                    # Check if standalone text displays an incorrect target value e.g. Text("k1 = 0.0001087")
                    m_disc, is_v, m_name = self._check_labeled_value(sec_id, loc, lit_clean, model, step_ref=effective_step)
                    if m_disc:
                        disc.append(m_disc)
                    elif is_v:
                        verified += 1
                        if m_name:
                            matched.add(m_name)

        return disc, checks, verified, matched

    # ------------------------------------------------------------------------
    # 3. Checked Elements: lecture_lines & captions (Narration)
    # ------------------------------------------------------------------------
    def _check_narration(
        self, sec: Dict[str, Any], model: AuthoritativeOutlineModel
    ) -> Tuple[List[FidelityDiscrepancy], int, int, Set[str]]:
        disc: List[FidelityDiscrepancy] = []
        checks = 0
        verified = 0
        matched: Set[str] = set()

        sec_id = sec.get("id", "section")
        sec_step = self._extract_step_ref_from_sec(sec)
        lines = sec.get("lecture_lines", [])
        if not isinstance(lines, list):
            return disc, checks, verified, matched

        for i, line in enumerate(lines):
            line_str = str(line).strip()
            loc = f"lecture_lines[{i}]"
            checks += 1

            line_step = self._extract_step_ref_from_text(line_str)
            effective_step = line_step if line_step is not None else sec_step

            # If an explicit step is referenced, verify that the step exists in the outline
            if effective_step is not None and not model.has_step(effective_step):
                ref_disp = effective_step[1] if isinstance(effective_step, tuple) else effective_step
                disc.append(
                    FidelityDiscrepancy(
                        section_id=sec_id,
                        location=loc,
                        category=FidelityCategory.AMBIGUOUS_CLAIM,
                        actual_claim=line_str,
                        reason=f"Spoken claim references step '{ref_disp}' which is out-of-range or unavailable in the authoritative outline",
                        severity=VerificationStatus.NEEDS_REVIEW,
                    )
                )
                continue

            # Check for hallucination relapse in spoken text
            relapse = self._detect_hallucination_relapse(line_str, model)
            if relapse:
                disc.append(
                    FidelityDiscrepancy(
                        section_id=sec_id,
                        location=loc,
                        category=FidelityCategory.HALLUCINATION_RELAPSE,
                        actual_claim=line_str,
                        expected_value=relapse["corrected_value"],
                        reason=f"Spoken narration contains rejected hallucinated value '{relapse['rejected_float']}' for {relapse['var_name']}",
                        suggested_correction=f"Replace '{relapse['rejected_float']}' with '{relapse['corrected_value']}' in narration",
                        severity=VerificationStatus.FAILED,
                    )
                )
                continue

            # Inspect spoken numbers and their associated variable context
            line_disc, is_v, m_name = self._inspect_narration_line(sec_id, loc, line_str, model, step_ref=effective_step)
            if line_disc:
                disc.append(line_disc)
            elif is_v:
                verified += 1
                if m_name:
                    matched.add(m_name)

        return disc, checks, verified, matched

    # ------------------------------------------------------------------------
    # 4. Global Completeness: Required Results
    # ------------------------------------------------------------------------
    def _check_required_results(
        self, model: AuthoritativeOutlineModel, matched_targets: Set[str]
    ) -> Optional[FidelityDiscrepancy]:
        """Ensures the storyboard communicates the final result of the worked example."""
        if not model.required_results:
            return None

        final_step = model.required_results[-1]
        final_var = final_step["var_name"]
        norm_final = model._normalize_var_name(final_var)

        has_final = False
        for m in matched_targets:
            if m == final_var or model._normalize_var_name(m) == norm_final or "final" in m.lower():
                has_final = True
                break

        if not has_final:
            return FidelityDiscrepancy(
                section_id="storyboard",
                location="entire_storyboard",
                category=FidelityCategory.MISSING_REQUIRED_RESULT,
                actual_claim="None",
                expected_value=final_step["target"].formatted_value,
                expected_unit=final_step["target"].unit,
                reason="The storyboard worked example does not present or calculate the required final result from Stage 1",
                suggested_correction=f"Include final result {final_step['target'].name} = {final_step['target'].formatted_value} in worked example section",
                severity=VerificationStatus.FAILED,
            )

        return None

    # ------------------------------------------------------------------------
    # Extraction & Parsing Helpers
    # ------------------------------------------------------------------------
    def _extract_display_strings(self, anim_str: str) -> List[str]:
        """Extracts text and math expressions inside MathTex, Text, and SafeMathTex."""
        literals: List[str] = []
        patterns = [
            r"MathTex\s*\(\s*r?[\"'](.*?)[\"'](?:\s*,\s*r?[\"'](.*?)[\"'])*\s*\)",
            r"SafeMathTex\s*\(\s*r?[\"'](.*?)[\"'](?:\s*,\s*r?[\"'](.*?)[\"'])*\s*\)",
            r"Text\s*\(\s*r?[\"'](.*?)[\"']\s*[,)]",
        ]
        for pat in patterns:
            for match in re.finditer(pat, anim_str, re.DOTALL):
                for g in match.groups():
                    if g and str(g).strip():
                        if not re.match(r"^(font_size|color|weight|slant)\s*=", g):
                            literals.append(g.strip())

        if not literals and "MathTex" in anim_str:
            sub = anim_str[anim_str.find("MathTex") :]
            quote_match = re.search(r"MathTex\s*\(\s*r?[\"'](.*?)[\"']", sub, re.DOTALL)
            if quote_match:
                literals.append(quote_match.group(1).strip())

        return literals

    def _detect_hallucination_relapse(
        self, text: str, model: AuthoritativeOutlineModel
    ) -> Optional[Dict[str, Any]]:
        """Detects whether text contains any known rejected hallucinated value."""
        for rej_float, info in model.rejected_hallucinations.items():
            cand_strs = [str(rej_float)]
            if rej_float.is_integer():
                cand_strs.append(str(int(rej_float)))
            else:
                cand_strs.append(f"{rej_float:g}")
            if "rejected_string" in info:
                cand_strs.append(info["rejected_string"])

            for c in cand_strs:
                c_clean, _, _ = PrecisionExtractor.sanitize_claimed_string(c)
                if not c_clean:
                    continue
                pat = rf"(?<![a-zA-Z0-9_\.]){re.escape(c_clean)}(?![a-zA-Z0-9_\.]|\.[0-9])"
                if re.search(pat, text):
                    return info

        return None

    def _find_matching_target(
        self, norm_name: str, val_str: str, model: AuthoritativeOutlineModel, step_ref: Optional[Union[int, str]] = None
    ) -> Optional[TargetVariable]:
        """Identifies target variable in model by clean name or explicit alias match, scoped by step if provided."""
        # Never match compound arithmetic expressions as a variable
        if any(c in norm_name for c in ("+", "-", "*", "/", "^", "frac")):
            return None

        # Determine target pool based on step_ref
        targets = model.get_targets_at_step(step_ref) if step_ref is not None else model.targets

        # Strip step prefixes if key is like "step1mid" or "step_1_mid"
        clean_name = norm_name
        m_step = re.match(r"^step\d+(.*)$", clean_name)
        if m_step and m_step.group(1):
            clean_name = m_step.group(1)

        # 1. Direct name lookup
        if clean_name in targets:
            return targets[clean_name]
        if norm_name in targets:
            return targets[norm_name]

        # 2. Canonical aliases
        ALIASES = {
            "rateconstant": ["k", "k1", "k2"],
            "rate": ["k", "k1", "k2"],
            "k": ["k1", "k2"],
            "activationenergy": ["ea", "e_a"],
            "exponent": ["expt1", "exp_t1", "expt2", "exp_t2", "exp"],
            "gasconstant": ["r"],
            "temperature": ["t", "t1", "t2"],
        }
        for lookup_name in (clean_name, norm_name):
            if lookup_name in ALIASES:
                for cand in ALIASES[lookup_name]:
                    if cand in targets:
                        return targets[cand]

        # 3. Cleaned exact match ignoring underscores
        for k, t in targets.items():
            if clean_name == k or clean_name == model._normalize_var_name(k):
                return t
            if norm_name == k or norm_name == model._normalize_var_name(k):
                return t

        return None

    def _check_context_mixup(
        self,
        norm_name: str,
        val_str: str,
        target: TargetVariable,
        model: AuthoritativeOutlineModel,
    ) -> Optional[str]:
        """Detects if a variable was mapped to the wrong condition (e.g. T1 value assigned to T2)."""
        if "t1" in norm_name and "t2" in target.context_tags:
            return f"Variable '{norm_name}' is associated with T1, but was assigned value '{val_str}' corresponding to T2"
        if "t2" in norm_name and "t1" in target.context_tags and "t2" not in target.context_tags:
            return f"Variable '{norm_name}' is associated with T2, but was assigned value '{val_str}' corresponding to T1"

        return None

    def _verify_mathtex_equality(
        self, sec_id: str, loc: str, lit: str, model: AuthoritativeOutlineModel, step_ref: Optional[Union[int, str]] = None
    ) -> Tuple[Optional[FidelityDiscrepancy], bool, Optional[str]]:
        """Verifies an equality expression inside MathTex (e.g. 'k1 = ...' or 'LHS = RHS')."""
        parts = [p.strip() for p in lit.split("=")]
        if len(parts) < 2:
            return None, False, None

        # Case A: Three-part equation: 'k_1 = formula = 0.006545'
        if len(parts) >= 3:
            var_part = parts[0]
            formula_part = parts[1]
            val_part = parts[-1]

            norm_var = model._normalize_var_name(var_part)
            target = self._find_matching_target(norm_var, val_part, model, step_ref=step_ref)

            # Verify formula part against val_part with target context
            formula_disc = self._verify_standalone_equation(sec_id, loc, f"{formula_part} = {val_part}", model, target=target, step_ref=step_ref)
            if formula_disc:
                return formula_disc, False, None

            # Verify variable part against val_part
            if target:
                try:
                    prec = self._extract_precision(val_part)
                    is_match, diag, details = ToleranceEvaluator.evaluate(prec, target.float_value)
                    if not is_match:
                        return (
                            FidelityDiscrepancy(
                                section_id=sec_id,
                                location=loc,
                                category=FidelityCategory.VALUE_MISMATCH,
                                actual_claim=lit,
                                expected_value=target.formatted_value,
                                expected_unit=target.unit,
                                reason=f"MathTex value for '{target.name}' contradicts Stage 1 verified target: {details}",
                                suggested_correction=f"{var_part} = {target.formatted_value} {target.unit or ''}".strip(),
                                severity=VerificationStatus.FAILED,
                            ),
                            False,
                            target.name,
                        )
                    return None, True, target.name
                except Exception:
                    pass
            return None, True, None

        # Case B: Two-part equation: 'LHS = RHS'
        lhs_raw = parts[0]
        rhs_raw = parts[1]

        # Check if LHS is an arithmetic expression (has operators or fractions)
        clean_lhs = FormulaNormalizer.sanitize_latex(lhs_raw).strip()
        has_arith = any(op in clean_lhs for op in ("+", "-", "*", "/", "^", "**", "exp(", "ln(")) or r"\frac" in lhs_raw

        if has_arith:
            disc = self._verify_standalone_equation(sec_id, loc, f"{lhs_raw} = {rhs_raw}", model, step_ref=step_ref)
            if disc:
                return disc, False, None
            # Verified equation! Identify if it computed a target step
            step_targets = model.get_targets_at_step(step_ref) if step_ref is not None else model.targets
            for t_var in step_targets.values():
                try:
                    p = self._extract_precision(rhs_raw)
                    m, _, _ = ToleranceEvaluator.evaluate(p, t_var.float_value)
                    if m:
                        return None, True, t_var.name
                except Exception:
                    pass
            return None, True, None

        # LHS is a variable identifier: e.g. 'k_1' or 'k1'
        norm_lhs = model._normalize_var_name(lhs_raw)
        target = self._find_matching_target(norm_lhs, rhs_raw, model, step_ref=step_ref)

        try:
            prec = self._extract_precision(rhs_raw)
        except Exception:
            return None, False, None

        if target:
            is_match, diag, details = ToleranceEvaluator.evaluate(prec, target.float_value)
            if not is_match:
                return (
                    FidelityDiscrepancy(
                        section_id=sec_id,
                        location=loc,
                        category=FidelityCategory.VALUE_MISMATCH,
                        actual_claim=lit,
                        expected_value=target.formatted_value,
                        expected_unit=target.unit,
                        reason=f"MathTex value for '{target.name}' contradicts Stage 1 verified target: {details}",
                        suggested_correction=f"{lhs_raw} = {target.formatted_value} {target.unit or ''}".strip(),
                        severity=VerificationStatus.FAILED,
                    ),
                    False,
                    target.name,
                )

            # Check unit
            if target.unit and prec.unit:
                is_compat, _ = units_are_compatible(prec.unit, target.unit)
                if not is_compat:
                    return (
                        FidelityDiscrepancy(
                            section_id=sec_id,
                            location=loc,
                            category=FidelityCategory.UNIT_MISMATCH,
                            actual_claim=lit,
                            expected_value=target.formatted_value,
                            expected_unit=target.unit,
                            reason=f"MathTex unit '{prec.unit}' conflicts with Stage 1 verified unit '{target.unit}'",
                            suggested_correction=f"{lhs_raw} = {target.formatted_value} {target.unit}",
                            severity=VerificationStatus.FAILED,
                        ),
                        False,
                        target.name,
                    )

            # Check context
            ctx_err = self._check_context_mixup(norm_lhs, rhs_raw, target, model)
            if ctx_err:
                return (
                    FidelityDiscrepancy(
                        section_id=sec_id,
                        location=loc,
                        category=FidelityCategory.CONTEXT_MISMATCH,
                        actual_claim=lit,
                        expected_value=target.formatted_value,
                        expected_unit=target.unit,
                        reason=ctx_err,
                        severity=VerificationStatus.FAILED,
                    ),
                    False,
                    target.name,
                )

            return None, True, target.name

        return None, False, None

    def _verify_standalone_equation(
        self,
        sec_id: str,
        loc: str,
        eq_str: str,
        model: AuthoritativeOutlineModel,
        target: Optional[TargetVariable] = None,
        step_ref: Optional[Union[int, str]] = None,
    ) -> Optional[FidelityDiscrepancy]:
        """Evaluates standalone mathematical equations using SafeASTEvaluator and TrustedFormulaRegistry."""
        parts = [p.strip() for p in eq_str.split("=")]
        if len(parts) < 2:
            return None

        lhs_raw = parts[0]
        rhs_raw = parts[-1]

        # Check for hallucination relapse in equation
        relapse = self._detect_hallucination_relapse(eq_str, model)
        if relapse:
            return FidelityDiscrepancy(
                section_id=sec_id,
                location=loc,
                category=FidelityCategory.HALLUCINATION_RELAPSE,
                actual_claim=eq_str,
                expected_value=relapse["corrected_value"],
                reason=f"Equation contains rejected hallucinated value '{relapse['rejected_float']}' for {relapse['var_name']}",
                suggested_correction=f"{lhs_raw} = {relapse['corrected_value']}",
                severity=VerificationStatus.FAILED,
            )

        # Clean and preprocess formula
        clean_lhs = FormulaNormalizer.sanitize_latex(lhs_raw).strip()
        # Convert scientific notation: e.g. 7.5 * 10**(10) -> 7.5e10
        clean_lhs = re.sub(
            r"([0-9]+(?:\.[0-9]+)?)\s*\*\s*10\*\*(?:\(([+-]?[0-9]+)\)|([+-]?[0-9]+))",
            lambda m: f"{m.group(1)}e{m.group(2) or m.group(3)}",
            clean_lhs,
        )
        clean_lhs = re.sub(r"\b([A-Za-z]+)_([0-9a-zA-Z]+)\b", r"\1\2", clean_lhs)
        clean_lhs = re.sub(r"\b([A-Za-z]+)\s+([A-Za-z][0-9a-zA-Z]*)", r"\1 * \2", clean_lhs)

        # 1. Formula Trust check
        is_trusted = False
        matched_target_var = target
        trust_reason = ""

        # A. Check if clean_lhs canonical AST matches a verified outline step formula
        try:
            lhs_canonical = FormulaNormalizer.ast_to_canonical_str(clean_lhs)
            if lhs_canonical in model.step_formula_asts:
                is_trusted = True
                trust_reason = "Matches verified step formula from Stage 1 outline"
                if matched_target_var is None and lhs_canonical in model.step_target_map:
                    matched_target_var = model.step_target_map[lhs_canonical]
        except Exception:
            lhs_canonical = None

        # B. Check TrustedFormulaRegistry if not already matched
        context_inputs = dict(model.inputs)
        if "T" not in context_inputs:
            if "T1" in context_inputs:
                context_inputs["T"] = context_inputs["T1"]
            elif "T2" in context_inputs:
                context_inputs["T"] = context_inputs["T2"]

        if not is_trusted:
            trust_expr = re.sub(r"\bT[12]\b", "T", clean_lhs)
            trust_status, tmpl_id, trust_details = TrustedFormulaRegistry.check_formula_trust(
                trust_expr, context_inputs, model.governing_principles
            )
            if trust_status == FormulaTrustStatus.TRUSTED:
                is_trusted = True
                trust_reason = f"Trusted registry: {trust_details}"

        # C. Check composition substitution with intermediate verified results
        if not is_trusted:
            comp_expr = clean_lhs
            for t_k, t_v in model.targets.items():
                if not t_v.is_input and t_v.formatted_value:
                    val_str = t_v.formatted_value
                    if val_str in comp_expr:
                        comp_expr = comp_expr.replace(val_str, t_v.name)
            if comp_expr != clean_lhs:
                comp_trust = re.sub(r"\bT[12]\b", "T", comp_expr)
                status, _, _ = TrustedFormulaRegistry.check_formula_trust(
                    comp_trust, context_inputs, model.governing_principles
                )
                if status == FormulaTrustStatus.TRUSTED:
                    is_trusted = True
                    trust_reason = "Matches trusted template after intermediate target substitution"

        if not is_trusted:
            return FidelityDiscrepancy(
                section_id=sec_id,
                location=loc,
                category=FidelityCategory.UNSUPPORTED_FORMULA,
                actual_claim=eq_str,
                reason=f"Formula '{lhs_raw}' introduced in Stage 2 is not trusted by canonical registry or Stage 1 steps",
                severity=VerificationStatus.NEEDS_REVIEW,
            )

        # 2. Precision and Evaluation
        try:
            prec = self._extract_precision(rhs_raw)
        except Exception as e:
            return FidelityDiscrepancy(
                section_id=sec_id,
                location=loc,
                category=FidelityCategory.AMBIGUOUS_CLAIM,
                actual_claim=eq_str,
                reason=f"Could not parse claimed numerical RHS: {e}",
                severity=VerificationStatus.NEEDS_REVIEW,
            )

        # Check if matched target matches prec
        step_targets = model.get_targets_at_step(step_ref) if step_ref is not None else model.targets
        if matched_target_var is None:
            for t_cand in step_targets.values():
                m, _, _ = ToleranceEvaluator.evaluate(prec, t_cand.float_value)
                if m:
                    matched_target_var = t_cand
                    break

        if matched_target_var is not None:
            is_match, diag, tol_details = ToleranceEvaluator.evaluate(prec, matched_target_var.float_value)
            if is_match:
                if matched_target_var.unit and prec.unit:
                    is_compat, _ = units_are_compatible(prec.unit, matched_target_var.unit)
                    if not is_compat:
                        return FidelityDiscrepancy(
                            section_id=sec_id,
                            location=loc,
                            category=FidelityCategory.UNIT_MISMATCH,
                            actual_claim=eq_str,
                            expected_value=matched_target_var.formatted_value,
                            expected_unit=matched_target_var.unit,
                            reason=f"Equation unit '{prec.unit}' conflicts with Stage 1 verified unit '{matched_target_var.unit}'",
                            suggested_correction=f"{lhs_raw} = {matched_target_var.formatted_value} {matched_target_var.unit}",
                            severity=VerificationStatus.FAILED,
                        )
                return None

        # Direct safe evaluation if not directly matching target float
        eval_vars = {k: float(v) for k, v in model.inputs.items()}
        if "T" not in eval_vars and "T1" in eval_vars:
            eval_vars["T"] = eval_vars["T1"]
        if step_ref is not None:
            step_state = model.get_state_at_step(step_ref)
            for s_k, s_v in step_state.items():
                if isinstance(s_v, (int, float)) and not isinstance(s_v, bool):
                    eval_vars[s_k] = float(s_v)
            for t_name, t_var in step_targets.items():
                if t_name not in eval_vars:
                    eval_vars[t_name] = t_var.float_value
        else:
            for t_name, t_var in model.targets.items():
                if t_name not in eval_vars:
                    eval_vars[t_name] = t_var.float_value

        eval_res = self.evaluator.safe_evaluate(
            clean_lhs, eval_vars
        )

        if eval_res.value is None:
            expected_v = matched_target_var.formatted_value if matched_target_var else "computed value"
            return FidelityDiscrepancy(
                section_id=sec_id,
                location=loc,
                category=FidelityCategory.AMBIGUOUS_CLAIM,
                actual_claim=eq_str,
                expected_value=expected_v,
                reason=f"Evaluation failed: {eval_res.details}",
                severity=VerificationStatus.NEEDS_REVIEW,
            )

        is_match, diag, tol_details = ToleranceEvaluator.evaluate(prec, eval_res.value)
        if not is_match:
            return FidelityDiscrepancy(
                section_id=sec_id,
                location=loc,
                category=FidelityCategory.VALUE_MISMATCH,
                actual_claim=eq_str,
                expected_value=f"{eval_res.value:.4g}",
                reason=f"Arithmetic equation mismatch: {tol_details}",
                suggested_correction=f"{lhs_raw} = {eval_res.value:.4g}",
                severity=VerificationStatus.FAILED,
            )

        return None

    def _check_labeled_value(
        self, sec_id: str, loc: str, lit: str, model: AuthoritativeOutlineModel, step_ref: Optional[Union[int, str]] = None
    ) -> Tuple[Optional[FidelityDiscrepancy], bool, Optional[str]]:
        """Inspects labels like 'k1: 0.0001087' or 'k1 = 0.0001087'."""
        match = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", lit)
        if match:
            var_name = match.group(1)
            val_str = match.group(2)
            norm_name = model._normalize_var_name(var_name)
            target = self._find_matching_target(norm_name, val_str, model, step_ref=step_ref)
            if target:
                try:
                    prec = self._extract_precision(val_str)
                    is_match, _, details = ToleranceEvaluator.evaluate(prec, target.float_value)
                    if not is_match:
                        return (
                            FidelityDiscrepancy(
                                section_id=sec_id,
                                location=loc,
                                category=FidelityCategory.VALUE_MISMATCH,
                                actual_claim=lit,
                                expected_value=target.formatted_value,
                                expected_unit=target.unit,
                                reason=f"Displayed value for '{target.name}' contradicts Stage 1 verified target: {details}",
                                suggested_correction=f"{var_name} = {target.formatted_value}",
                                severity=VerificationStatus.FAILED,
                            ),
                            False,
                            target.name,
                        )
                    return None, True, target.name
                except Exception:
                    pass

        return None, False, None

    def _inspect_narration_line(
        self, sec_id: str, loc: str, line_str: str, model: AuthoritativeOutlineModel, step_ref: Optional[Union[int, str]] = None
    ) -> Tuple[Optional[FidelityDiscrepancy], bool, Optional[str]]:
        """Inspects spoken narration for numerical claims referencing target variables."""
        line_lower = line_str.lower()

        KEYWORDS = {
            "rate constant": "k",
            "activation energy": "ea",
            "exponent": "exp",
            "temperature": "t",
            "derivative": "slope",
        }

        # Filter out section numbers: 'Section 1', 'Step 2', 'Part 3'
        clean_narr = re.sub(r"\b(section|step|part|item)\s+[0-9]+", "", line_str, flags=re.IGNORECASE)
        # Filter out counts: '3 factors', '2 temperatures', '4 vertices'
        clean_narr = re.sub(r"\b[0-9]+\s+(factors|temperatures|vertices|steps|nodes|units|seconds)\b", "", clean_narr, flags=re.IGNORECASE)
        # Convert spoken negative numbers: 'negative 30.07' or 'minus 30.07' -> -30.07
        clean_narr = re.sub(r"\b(?:negative|minus)\s+([0-9]+(?:\.[0-9]+)?)", r"-\1", clean_narr, flags=re.IGNORECASE)
        # Strip clause/sentence punctuation so numbers at clause boundaries (e.g. 'mid is 8.') are extracted cleanly
        clean_narr = re.sub(r"[\.,;:!\?](?=\s|$)", " ", clean_narr)

        num_matches = re.findall(
            r"(?<![a-zA-Z0-9_\.])([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)(?![a-zA-Z0-9_\.]|\.[0-9])",
            clean_narr,
        )

        step_targets = model.get_targets_at_step(step_ref) if step_ref is not None else model.targets

        parsed_nums = []
        for num_str in num_matches:
            try:
                prec = self._extract_precision(num_str)
                parsed_nums.append((num_str, prec))
            except Exception:
                continue

        if not parsed_nums:
            return None, False, None

        for norm_name, target in step_targets.items():
            var_mentioned = (
                norm_name in line_lower or
                target.name.lower() in line_lower or
                any(kw in line_lower and kw_var in norm_name for kw, kw_var in KEYWORDS.items())
            )
            if not var_mentioned:
                continue

            # First, check if any number in parsed_nums matches this target
            matched_num = None
            for num_str, prec in parsed_nums:
                is_m, _, _ = ToleranceEvaluator.evaluate(prec, target.float_value)
                if is_m:
                    matched_num = (num_str, prec)
                    break

            if matched_num is not None:
                num_str, prec = matched_num
                if "t1" in line_lower and "t2" in target.context_tags:
                    return (
                        FidelityDiscrepancy(
                            section_id=sec_id,
                            location=loc,
                            category=FidelityCategory.CONTEXT_MISMATCH,
                            actual_claim=line_str,
                            expected_value=target.formatted_value,
                            expected_unit=target.unit,
                            reason=f"Narration states T1 with value '{num_str}' corresponding to T2",
                            severity=VerificationStatus.FAILED,
                        ),
                        False,
                        target.name,
                    )
                return None, True, target.name

            # Target was mentioned, but NO number in the line matched its target.float_value
            if not target.is_input:
                for num_str, prec in parsed_nums:
                    # Check if this number matches another known variable
                    matches_other = False
                    for other_k, other_t in step_targets.items():
                        if other_k != norm_name:
                            m_other, _, _ = ToleranceEvaluator.evaluate(prec, other_t.float_value)
                            if m_other:
                                matches_other = True
                                break
                    if not matches_other or re.search(rf"\b{re.escape(norm_name)}\b.*?{re.escape(num_str)}", line_lower):
                        return (
                            FidelityDiscrepancy(
                                section_id=sec_id,
                                location=loc,
                                category=FidelityCategory.VALUE_MISMATCH,
                                actual_claim=line_str,
                                expected_value=target.formatted_value,
                                expected_unit=target.unit,
                                reason=f"Spoken claim of '{num_str}' for '{target.name}' contradicts Stage 1 verified target '{target.formatted_value}'",
                                suggested_correction=f"Replace '{num_str}' with '{target.formatted_value}' in narration",
                                severity=VerificationStatus.FAILED,
                            ),
                            False,
                            target.name,
                        )

        return None, False, None
