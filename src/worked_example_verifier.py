"""
Worked Example Verifier for CineLearn (Milestone 2 & Milestone 2B.6C).

Provides general-purpose worked-example verification, multi-step dependency tracking,
specialized verifier registry routing, backward-compatible calculation adaptation,
safe auto-correction, and outline-level verification across arbitrary academic domains.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    from grounding.trace_engine import StateSnapshot, ComputationTrace
except ImportError:
    from src.grounding.trace_engine import StateSnapshot, ComputationTrace

from numerical_evaluator import (
    DiagnosticCode,
    FormulaNormalizer,
    FormulaTrustStatus,
    NumericalStepVerifier,
    PrecisionExtractor,
    SafeASTEvaluator,
    ToleranceEvaluator,
    TrustedFormulaRegistry,
    VerificationResult,
    VerificationStatus,
)


# ============================================================================
# Environment & Variable Dependency Enums
# ============================================================================

class VariableTrust(str, Enum):
    """Trust classification for variables in the calculation environment."""
    TRUSTED_INPUT = "TRUSTED_INPUT"
    VERIFIED_INTERMEDIATE = "VERIFIED_INTERMEDIATE"
    CORRECTED_INTERMEDIATE = "CORRECTED_INTERMEDIATE"
    UNRESOLVED = "UNRESOLVED"
    UNTRUSTED = "UNTRUSTED"


class StepType(str, Enum):
    """General-purpose classification for worked-example steps."""
    SCALAR_CALCULATION = "calculation"
    STATE_TRANSITION = "state_transition"
    SEQUENCE_OPERATION = "sequence_operation"
    LOGICAL_PROOF = "logical_proof"
    CONCEPTUAL_EXPLANATION = "conceptual_explanation"
    UNKNOWN = "unknown"


@dataclass
class VariableRecord:
    """Record of a variable's value, trust status, and lineage."""
    name: str
    value: Any  # Can be float, int, str, bool, list, dict, None
    trust: VariableTrust
    source_step_index: Optional[int] = None
    diagnostic: str = ""
    original_claimed_sample: Optional[str] = None
    data_type: str = "scalar"


class CalculationEnvironment:
    """
    Maintains variable bindings, dependency states, formula lineage, and calculation history
    across sequential worked example steps. Supports arbitrary JSON-compatible data types.
    """

    def __init__(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        units: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.active_variables: Dict[str, VariableRecord] = {}
        self.history: List[VariableRecord] = []
        # Maps previous claimed literal numbers to corrected floats
        self.claimed_to_corrected: Dict[float, float] = {}
        self.step_output_mapping: Dict[int, str] = {}
        self.step_formula_mapping: Dict[int, str] = {}
        self.claimed_to_step_index: Dict[float, int] = {}
        self.units: Dict[str, Any] = copy.deepcopy(units) if units else {}
        self.initial_state: Dict[str, Any] = copy.deepcopy(inputs) if inputs else {}
        self.current_state: Dict[str, Any] = copy.deepcopy(self.initial_state)
        self._step_snapshots: Dict[int, Dict[str, Any]] = {}
        self._max_step_index: int = -1

        if inputs:
            for k, v in inputs.items():
                is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
                val = float(v) if is_num else v
                dt = "scalar" if is_num else type(v).__name__
                rec = VariableRecord(
                    name=k,
                    value=val,
                    trust=VariableTrust.TRUSTED_INPUT,
                    source_step_index=None,
                    diagnostic="Trusted problem input",
                    data_type=dt,
                )
                self.active_variables[k] = rec
                self.history.append(rec)

    def set_step_result(
        self,
        step_index: int,
        var_name: str,
        value: Any,
        trust: VariableTrust,
        diagnostic: str = "",
        claimed_sample: Optional[str] = None,
        claimed_float: Optional[float] = None,
        formula_str: Optional[str] = None,
    ):
        """Registers the output variable of a calculation step into the environment."""
        is_num = isinstance(value, (int, float)) and not isinstance(value, bool)
        val = float(value) if is_num else value
        dt = "scalar" if is_num else type(value).__name__

        rec = VariableRecord(
            name=var_name,
            value=val,
            trust=trust,
            source_step_index=step_index,
            diagnostic=diagnostic,
            original_claimed_sample=claimed_sample,
            data_type=dt,
        )
        self.active_variables[var_name] = rec
        self.history.append(rec)
        self.step_output_mapping[step_index] = var_name
        self.current_state[var_name] = copy.deepcopy(val)
        self.record_step_snapshot(step_index, description=f"Update {var_name}")

        if formula_str:
            self.step_formula_mapping[step_index] = formula_str

        if claimed_float is not None:
            self.claimed_to_step_index[claimed_float] = step_index
            if trust == VariableTrust.CORRECTED_INTERMEDIATE and is_num:
                self.claimed_to_corrected[claimed_float] = float(value)

    def record_step_snapshot(
        self,
        step_index: int,
        updates: Optional[Dict[str, Any]] = None,
        description: str = "",
        step_type: str = "state_transition",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateSnapshot:
        """
        Records or updates the snapshot of all variables active at step_index.
        Preserves complete system state across step boundaries.
        """
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            raise TypeError(
                f"step_index must be an integer, got {type(step_index).__name__}: {step_index}"
            )
        if step_index < -1:
            raise IndexError(
                f"Step index {step_index} is invalid. Minimum valid index is -1 (initial state)."
            )

        if updates and isinstance(updates, dict):
            for k, v in updates.items():
                self.current_state[k] = copy.deepcopy(v)
                is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
                val = float(v) if is_num else v
                dt = "scalar" if is_num else type(v).__name__
                self.active_variables[k] = VariableRecord(
                    name=k,
                    value=val,
                    trust=VariableTrust.VERIFIED_INTERMEDIATE,
                    source_step_index=step_index,
                    diagnostic=description or f"Snapshot update at step {step_index}",
                    data_type=dt,
                )

        state_copy = copy.deepcopy(self.current_state)
        self._step_snapshots[step_index] = state_copy
        if step_index > self._max_step_index:
            self._max_step_index = step_index

        return StateSnapshot(
            step_index=step_index,
            state=state_copy,
            description=description,
            step_type=step_type,
            metadata=metadata or {},
        )

    def get_state_at_step(self, step_index: int) -> Dict[str, Any]:
        """
        Retrieves the complete state dictionary at a particular step.
        - step_index == -1: returns initial state before execution.
        - 0 <= step_index <= max_step: returns complete state snapshot at that step.
        - Invalid / out-of-range index: raises IndexError.
        """
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            raise TypeError(
                f"step_index must be an integer, got {type(step_index).__name__}: {step_index}"
            )

        if step_index == -1:
            return copy.deepcopy(self.initial_state)

        if step_index < -1:
            raise IndexError(
                f"Step index {step_index} is invalid. Minimum valid index is -1 (initial state)."
            )

        if self._max_step_index == -1:
            raise IndexError(
                f"Step index {step_index} is out of range. No steps have been recorded yet."
            )

        if step_index > self._max_step_index:
            raise IndexError(
                f"Step index {step_index} is out of range. "
                f"Maximum recorded step index is {self._max_step_index}."
            )

        if step_index in self._step_snapshots:
            return copy.deepcopy(self._step_snapshots[step_index])

        # If step_index was within valid range [0, max_step] but has no explicit snapshot
        # (e.g. non-evaluable or conceptual step), inherit state from most recent preceding step
        for idx in range(step_index - 1, -1, -1):
            if idx in self._step_snapshots:
                return copy.deepcopy(self._step_snapshots[idx])

        return copy.deepcopy(self.initial_state)

    def get_initial_state(self) -> Dict[str, Any]:
        """Returns a copy of the initial problem inputs."""
        return copy.deepcopy(self.initial_state)

    def get_snapshot_at_step(self, step_index: int) -> StateSnapshot:
        """Retrieves a StateSnapshot instance at step_index."""
        state = self.get_state_at_step(step_index)
        desc = "Initial state" if step_index == -1 else f"State at step {step_index}"
        stype = "initialization" if step_index == -1 else "computation"
        return StateSnapshot(step_index=step_index, state=state, description=desc, step_type=stype)

    def get_trace(self, domain_name: str = "general") -> ComputationTrace:
        """Constructs an authoritative ComputationTrace across all chronological steps."""
        trace = ComputationTrace(
            algorithm_or_domain=domain_name,
            initial_state=copy.deepcopy(self.initial_state),
        )
        for idx in sorted(self._step_snapshots.keys()):
            if idx >= 0:
                trace.add_snapshot(self.get_snapshot_at_step(idx))
        return trace

    def set_unresolved(
        self,
        step_index: int,
        var_name: str,
        diagnostic: str,
        claimed_float: Optional[float] = None,
    ):
        """Marks a step's output variable as unresolved/untrusted."""
        self.set_step_result(
            step_index=step_index,
            var_name=var_name,
            value=None,
            trust=VariableTrust.UNTRUSTED,
            diagnostic=diagnostic,
            claimed_float=claimed_float,
        )

    def get(self, name: str) -> Optional[VariableRecord]:
        return self.active_variables.get(name)

    def is_trusted(self, name: str) -> bool:
        rec = self.get(name)
        if not rec or rec.value is None:
            return False
        return rec.trust in (
            VariableTrust.TRUSTED_INPUT,
            VariableTrust.VERIFIED_INTERMEDIATE,
            VariableTrust.CORRECTED_INTERMEDIATE,
        )

    def get_trusted_values(self) -> Dict[str, float]:
        """Returns dictionary of all currently trusted numeric variable names and values."""
        res = {}
        for k, rec in self.active_variables.items():
            if (
                rec.value is not None
                and isinstance(rec.value, (int, float))
                and not isinstance(rec.value, bool)
                and rec.trust
                in (
                    VariableTrust.TRUSTED_INPUT,
                    VariableTrust.VERIFIED_INTERMEDIATE,
                    VariableTrust.CORRECTED_INTERMEDIATE,
                )
            ):
                res[k] = float(rec.value)
        return res

    def get_numeric_values(self) -> Dict[str, float]:
        """Returns dictionary of all numeric variables in environment suitable for SafeASTEvaluator."""
        res = {}
        for k, rec in self.active_variables.items():
            if rec.value is not None and isinstance(rec.value, (int, float)) and not isinstance(rec.value, bool):
                res[k] = float(rec.value)
        return res

    def get_all_variables(self) -> Dict[str, Any]:
        """Returns dictionary of all current variable names and values regardless of type."""
        return {k: rec.value for k, rec in self.active_variables.items()}

    def substitute_dependencies(self, expr_str: str) -> Tuple[str, bool, List[str]]:
        """
        Replaces literal claimed numbers in downstream expression with their corrected values.
        Returns: (substituted_expr, was_modified, reasons).
        """
        s = expr_str
        modified = False
        reasons = []

        for claimed_val, corrected_val in sorted(
            self.claimed_to_corrected.items(), key=lambda x: len(str(x[0])), reverse=True
        ):
            cand_strs = [str(claimed_val), f"{claimed_val:g}"]
            corr_s = f"{corrected_val:.8g}"

            for cand in cand_strs:
                pat = rf"(?<![a-zA-Z0-9_])(?<![0-9]\.){re.escape(cand)}(?![a-zA-Z0-9_]|\.[0-9])"
                if re.search(pat, s):
                    s = re.sub(pat, corr_s, s)
                    modified = True
                    step_idx = self.claimed_to_step_index.get(claimed_val)
                    var_name = self.step_output_mapping.get(step_idx, f"step_{step_idx}") if step_idx is not None else "dep"
                    reasons.append(f"Substituted corrected {var_name}: {cand} -> {corr_s}")
                    break

        return s, modified, reasons

    def substitute_formulas_for_trust(self, expr_str: str) -> str:
        """
        Substitutes earlier step formulas or variable names into downstream expressions
        to allow compositional formula trust checking.
        """
        s = expr_str
        for claimed_val, step_idx in self.claimed_to_step_index.items():
            formula = self.step_formula_mapping.get(step_idx)
            var_name = self.step_output_mapping.get(step_idx)
            if not formula:
                continue

            # Check if variable name is in expression
            if var_name:
                var_pat = rf"\b{re.escape(var_name)}\b"
                if re.search(var_pat, s):
                    s = re.sub(var_pat, f"({formula})", s)

            # Check if literal claimed string is in expression
            cand_strs = [str(claimed_val), f"{claimed_val:g}"]
            for cand in cand_strs:
                pat = rf"(?<![a-zA-Z0-9_])(?<![0-9]\.){re.escape(cand)}(?![a-zA-Z0-9_]|\.[0-9])"
                if re.search(pat, s):
                    s = re.sub(pat, f"({formula})", s)
                    break

        return s


# ============================================================================
# Calculation Step Adapter & Classification
# ============================================================================

@dataclass
class AdaptedStep:
    """Internal normalized representation of a worked example calculation step."""
    index: int
    description: str
    original_dict: Dict[str, Any]
    raw_result: str
    step_type: StepType = StepType.UNKNOWN
    operation: Optional[str] = None
    expression: Optional[str] = None
    claimed_value: Optional[str] = None
    output_variable: Optional[str] = None
    unit: Optional[str] = None
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    is_approximate: bool = False
    is_evaluable: bool = False
    allow_auto_correction: bool = True


class CalculationStepAdapter:
    """
    Multi-domain adapter supporting both legacy string-based results
    ('-75000 / (8.314 * 300) = -29.74') and structured calculation/transition dictionaries.
    """

    ALGORITHMIC_INDICATORS = [
        r"\bdist\[", r"\bmin\[", r"\bmax\[", r"\bQ\b", r"\bheap\b", r"\bqueue\b",
        r"\bvertices\b", r"\bweights\b", r"\binfinity\b", r"∞", r"\binitialize\b",
    ]

    @classmethod
    def infer_output_variable(
        cls, step_desc: str, lhs: str, index: int, explicit_var: Optional[str] = None
    ) -> str:
        """Infers a clean output variable identifier from description, LHS, or explicit tag."""
        if explicit_var and str(explicit_var).strip():
            return str(explicit_var).strip()

        # If LHS is a lone variable identifier (e.g. 'k1 = 0.0001087')
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", lhs.strip()):
            return lhs.strip()

        # Check step description for variable identifiers like 'k1', 'exp_T1', 'dist'
        desc_matches = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=", step_desc)
        if desc_matches:
            return desc_matches[0]

        # Check for specific variable names first (more specific than exponent/exp)
        if "k1" in step_desc or "k_1" in step_desc:
            return "k1"
        if "k2" in step_desc or "k_2" in step_desc:
            return "k2"

        # Check for keywords in description (avoid matching function call 'exp(')
        if "T1" in step_desc and ("exponent" in step_desc.lower() or "exp_t1" in step_desc.lower()):
            return "exp_T1"
        if "T2" in step_desc and ("exponent" in step_desc.lower() or "exp_t2" in step_desc.lower()):
            return "exp_T2"

        return f"step_{index + 1}_val"

    @classmethod
    def _extract_expression_from_desc(
        cls, step_desc: str, var_name: str, claimed_val: str
    ) -> Optional[str]:
        """
        Attempts to extract an arithmetic expression from the step description when result
        contains a pure assignment (e.g. 'Calculate mid = (low + high) // 2 = (0 + 9) // 2 = 4').
        """
        if not step_desc or not var_name:
            return None

        # Pattern 1: var = <expr> = ...
        pat1 = rf"\b{re.escape(var_name)}\s*=\s*([^=]+?)\s*="
        m1 = re.search(pat1, step_desc)
        if m1:
            cand = m1.group(1).strip()
            # Verify candidate has operators
            if any(op in cand for op in ("+", "-", "*", "/", "//", "**", "%")):
                return cand

        # Pattern 2: update var = <expr> = ...
        pat2 = rf"\bupdate\s+{re.escape(var_name)}\s*=\s*([^=]+?)\s*="
        m2 = re.search(pat2, step_desc, re.IGNORECASE)
        if m2:
            cand = m2.group(1).strip()
            if any(op in cand for op in ("+", "-", "*", "/", "//", "**", "%")):
                return cand

        return None

    @classmethod
    def adapt_step(cls, step_dict: Dict[str, Any], index: int) -> AdaptedStep:
        """
        Normalizes a calculation or state step dictionary into an AdaptedStep.
        Preserves original input representation losslessly.
        """
        step_desc = str(step_dict.get("step") or step_dict.get("description") or "")
        raw_result = str(step_dict.get("result", "")).strip()

        explicit_type = step_dict.get("step_type")
        operation = step_dict.get("operation")
        explicit_expr = step_dict.get("expression")
        state_before = step_dict.get("state_before") if isinstance(step_dict.get("state_before"), dict) else None
        state_after = step_dict.get("state_after") if isinstance(step_dict.get("state_after"), dict) else None
        explanation = step_dict.get("explanation")
        unit = step_dict.get("unit")
        is_approx = bool(step_dict.get("is_approximate", False))

        allow_corr = step_dict.get("allow_auto_correction", step_dict.get("can_safely_correct", True))
        is_allow_corr = False if allow_corr is False else True

        # Check explicit non-evaluable step types or narrative/stoichiometric phrases (e.g. 'per glucose')
        if explicit_type and str(explicit_type).strip().lower() in ("conceptual_explanation", "conceptual_breakdown", "logical_proof"):
            return AdaptedStep(
                index=index,
                description=step_desc,
                original_dict=dict(step_dict),
                raw_result=raw_result,
                step_type=StepType.CONCEPTUAL_EXPLANATION,
                is_evaluable=False,
                allow_auto_correction=False,
            )

        if re.search(r"\bper\b", raw_result, re.IGNORECASE) or re.search(r"\bper\b", step_desc, re.IGNORECASE):
            return AdaptedStep(
                index=index,
                description=step_desc,
                original_dict=dict(step_dict),
                raw_result=raw_result,
                step_type=StepType.CONCEPTUAL_EXPLANATION,
                is_evaluable=False,
                allow_auto_correction=False,
            )

        # Check legacy algorithmic indicators (e.g. dist[A] = 0, dist[B] = inf)
        for ind in cls.ALGORITHMIC_INDICATORS:
            if re.search(ind, raw_result, re.IGNORECASE) or re.search(ind, step_desc, re.IGNORECASE):
                if "dist[" in raw_result or "∞" in raw_result:
                    return AdaptedStep(
                        index=index,
                        description=step_desc,
                        original_dict=dict(step_dict),
                        raw_result=raw_result,
                        step_type=StepType.STATE_TRANSITION,
                        is_evaluable=False,
                        allow_auto_correction=is_allow_corr,
                    )

        # Parse result and expression
        out_var = step_dict.get("output_variable")
        claimed_val = None
        expr_str = None

        if isinstance(step_dict.get("result"), dict):
            rdict = step_dict["result"]
            if rdict:
                out_var = list(rdict.keys())[0]
                claimed_val = str(list(rdict.values())[0])
            expr_str = str(explicit_expr).strip() if explicit_expr else None
        elif explicit_expr is not None and ("claimed_value" in step_dict or "=" not in raw_result):
            expr_str = str(explicit_expr).strip()
            claimed_val = str(step_dict.get("claimed_value") or raw_result).strip()
            out_var = cls.infer_output_variable(step_desc, expr_str, index, out_var)
        elif raw_result:
            parts = [p.strip() for p in raw_result.split("=")]
            if len(parts) >= 3:
                # var = expr = claimed
                out_var = parts[0]
                expr_str = parts[1] if not explicit_expr else str(explicit_expr).strip()
                clean_rhs, is_approx_r, unit_r = PrecisionExtractor.sanitize_claimed_string(parts[2])
                claimed_val = clean_rhs
                is_approx = is_approx or is_approx_r
                if not unit:
                    unit = unit_r
            elif len(parts) == 2:
                part_0 = parts[0]
                part_1 = parts[1]
                clean_rhs, is_approx_r, unit_r = PrecisionExtractor.sanitize_claimed_string(part_1)
                claimed_val = clean_rhs
                is_approx = is_approx or is_approx_r
                if not unit:
                    unit = unit_r

                # Check if part_0 is a lone variable identifier e.g. 'mid = 4' or 'low = 5'
                if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", part_0):
                    out_var = part_0
                    if explicit_expr:
                        expr_str = str(explicit_expr).strip()
                    else:
                        expr_str = cls._extract_expression_from_desc(step_desc, part_0, clean_rhs)
                else:
                    # part_0 contains arithmetic operators e.g. '-75000 / (8.314 * 300)'
                    expr_str = part_0 if not explicit_expr else str(explicit_expr).strip()
                    out_var = cls.infer_output_variable(step_desc, expr_str, index, out_var)
            else:
                expr_str = str(explicit_expr).strip() if explicit_expr else None
                claimed_val = raw_result

        # Infer step type
        step_type = StepType.UNKNOWN
        if explicit_type:
            try:
                step_type = StepType(str(explicit_type).lower())
            except ValueError:
                step_type = StepType.UNKNOWN

        if step_type == StepType.UNKNOWN:
            if state_before is not None or state_after is not None:
                step_type = StepType.STATE_TRANSITION
            elif operation in ("assign", "update", "set", "initialize"):
                step_type = StepType.STATE_TRANSITION
            elif operation in ("index", "slice", "compare", "search", "swap"):
                step_type = StepType.SEQUENCE_OPERATION
            elif (expr_str and "arr[" in expr_str) or "arr[" in raw_result or "arr[" in step_desc:
                step_type = StepType.SEQUENCE_OPERATION
            elif raw_result and "," in raw_result and "=" in raw_result:
                step_type = StepType.STATE_TRANSITION
            elif expr_str and any(op in expr_str for op in ("+", "-", "*", "/", "//", "**", "%", "exp", "log", "sqrt")):
                step_type = StepType.SCALAR_CALCULATION
            elif out_var and claimed_val:
                step_type = StepType.STATE_TRANSITION
            elif "=" not in raw_result:
                step_type = StepType.CONCEPTUAL_EXPLANATION
            else:
                step_type = StepType.UNKNOWN

        # Determine evaluability
        is_evaluable = False
        if step_type == StepType.SCALAR_CALCULATION:
            is_evaluable = bool(expr_str and claimed_val)
        elif step_type in (StepType.STATE_TRANSITION, StepType.SEQUENCE_OPERATION):
            is_evaluable = True
        elif step_type == StepType.CONCEPTUAL_EXPLANATION:
            is_evaluable = False
        else:
            is_evaluable = bool(expr_str or claimed_val or state_before or state_after)

        return AdaptedStep(
            index=index,
            description=step_desc,
            original_dict=dict(step_dict),
            raw_result=raw_result or f"{expr_str} = {claimed_val}",
            step_type=step_type,
            operation=operation,
            expression=expr_str,
            claimed_value=claimed_val,
            output_variable=out_var,
            unit=unit,
            state_before=state_before,
            state_after=state_after,
            explanation=explanation,
            is_approximate=is_approx,
            is_evaluable=is_evaluable,
            allow_auto_correction=is_allow_corr,
        )


# ============================================================================
# Output Formatter & Specialized Verifier Registry
# ============================================================================

def format_corrected_value(val: float, sample_str: Optional[str] = None) -> str:
    """
    Formats a computed float value matching the precision/scale of the claimed sample.
    """
    if sample_str is not None:
        try:
            prec = PrecisionExtractor.extract_precision(sample_str)
            if prec.is_scientific:
                return f"{val:.4e}".replace("e+0", "e+").replace("e-0", "e-")
            if prec.is_integer and abs(val - round(val)) < 1e-9:
                return str(int(round(val)))
            if 0 < abs(val) < 1:
                return f"{val:.4g}"
            if prec.decimal_places > 0:
                fmt = f"{{:.{prec.decimal_places}f}}"
                return fmt.format(val)
        except Exception:
            pass

    if abs(val) >= 1e6 or (0 < abs(val) < 1e-4):
        return f"{val:.4e}".replace("e+0", "e+").replace("e-0", "e-")
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    if 0 < abs(val) < 1:
        return f"{val:.4g}"
    return f"{val:.6g}"


class SpecializedVerifier:
    """Protocol / base class for domain-specific independent verifiers."""
    verifier_id: str = "specialized"

    def can_verify(self, spec: Dict[str, Any], topic: str = "") -> bool:
        raise NotImplementedError

    def verify_worked_example_spec(
        self, spec: Dict[str, Any], topic: str = ""
    ) -> Tuple[Dict[str, Any], WorkedExampleReport]:
        raise NotImplementedError


class SpecializedVerifierRegistry:
    """Registry of domain-specific specialized verifiers."""
    _verifiers: List[Any] = []

    @classmethod
    def register(cls, verifier: Any) -> None:
        vid = getattr(verifier, "verifier_id", None)
        if vid:
            for existing in cls._verifiers:
                if getattr(existing, "verifier_id", None) == vid:
                    return
        if verifier not in cls._verifiers:
            cls._verifiers.append(verifier)

    @classmethod
    def _ensure_builtins(cls) -> None:
        """Lazily load known built-in domain verifiers if not yet imported."""
        try:
            from grounding.verifiers.graph import dijkstra_verifier  # noqa: F401
        except Exception:
            try:
                import dijkstra_verifier  # noqa: F401
            except Exception:
                pass
        try:
            from grounding.verifiers.os import fcfs_verifier  # noqa: F401
        except Exception:
            try:
                import fcfs_verifier  # noqa: F401
            except Exception:
                pass
        try:
            from grounding.verifiers import dp_verifier  # noqa: F401
        except Exception:
            pass

    @classmethod
    def find_verifier(cls, spec: Dict[str, Any], topic: str = "") -> Optional[Any]:
        def _matches(v: Any) -> bool:
            if hasattr(v, "can_verify"):
                return bool(v.can_verify(spec, topic))
            if hasattr(v, "can_handle"):
                return bool(v.can_handle(spec, topic))
            return False

        for v in cls._verifiers:
            try:
                if _matches(v):
                    return v
            except Exception:
                continue

        cls._ensure_builtins()
        for v in cls._verifiers:
            try:
                if _matches(v):
                    return v
            except Exception:
                continue
        return None

    @classmethod
    def clear(cls) -> None:
        cls._verifiers.clear()


@dataclass
class WorkedExampleReport:
    """Consolidated outcome of worked example verification."""
    overall_status: VerificationStatus
    total_steps: int
    verified_steps: int
    corrected_steps: int
    failed_steps: int
    review_steps: int
    not_applicable_steps: int
    unverified_steps: int = 0
    step_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    final_result_status: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    summary: str = ""
    is_safe_for_stage2: bool = True
    verified_algorithmic_state: Optional[Dict[str, Any]] = None
    verification_method: str = "generic_orchestrator"
    state_snapshots: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "overall_status": self.overall_status.value,
            "total_steps": self.total_steps,
            "verified_steps": self.verified_steps,
            "corrected_steps": self.corrected_steps,
            "failed_steps": self.failed_steps,
            "review_steps": self.review_steps,
            "not_applicable_steps": self.not_applicable_steps,
            "unverified_steps": self.unverified_steps,
            "final_result_status": self.final_result_status.value,
            "is_safe_for_stage2": self.is_safe_for_stage2,
            "summary": self.summary,
            "step_diagnostics": self.step_diagnostics,
            "verification_method": self.verification_method,
        }
        if self.verified_algorithmic_state is not None:
            d["verified_algorithmic_state"] = self.verified_algorithmic_state
        if self.state_snapshots:
            d["state_snapshots"] = {str(k): copy.deepcopy(v) for k, v in self.state_snapshots.items()}
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> WorkedExampleReport:
        return cls(
            overall_status=VerificationStatus(d.get("overall_status", "NOT_APPLICABLE")),
            total_steps=int(d.get("total_steps", 0)),
            verified_steps=int(d.get("verified_steps", 0)),
            corrected_steps=int(d.get("corrected_steps", 0)),
            failed_steps=int(d.get("failed_steps", 0)),
            review_steps=int(d.get("review_steps", 0)),
            not_applicable_steps=int(d.get("not_applicable_steps", 0)),
            unverified_steps=int(d.get("unverified_steps", 0)),
            step_diagnostics=list(d.get("step_diagnostics", [])),
            final_result_status=VerificationStatus(d.get("final_result_status", "NOT_APPLICABLE")),
            summary=str(d.get("summary", "")),
            is_safe_for_stage2=bool(d.get("is_safe_for_stage2", True)),
            verified_algorithmic_state=d.get("verified_algorithmic_state"),
            verification_method=str(d.get("verification_method", "generic_orchestrator")),
            state_snapshots={int(k): dict(v) for k, v in d.get("state_snapshots", {}).items()},
        )


# ============================================================================
# Generic Worked Example Verification Orchestrator
# ============================================================================

class WorkedExampleVerifier:
    """
    Deterministic multi-domain worked-example verification orchestrator.
    Maintains environment state, tracks dependencies, applies safe corrections,
    and supports both generic multi-type step validation and specialized verifiers.
    """

    def __init__(self, evaluator: Optional[SafeASTEvaluator] = None):
        self.evaluator = evaluator or SafeASTEvaluator()
        self.step_verifier = NumericalStepVerifier(self.evaluator)

    @classmethod
    def is_dijkstra_spec(cls, spec: Dict[str, Any], topic: str = "") -> bool:
        """Backward-compatible helper checking if spec is a Dijkstra problem."""
        verifier = SpecializedVerifierRegistry.find_verifier(spec, topic)
        if verifier is not None:
            return getattr(verifier, "verifier_id", "") == "dijkstra"
        return False

    def verify_outline(
        self, outline_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], WorkedExampleReport]:
        """
        Public entry point for verifying an outline dictionary.
        """
        if not isinstance(outline_data, dict):
            empty_rep = WorkedExampleReport(
                overall_status=VerificationStatus.NOT_APPLICABLE,
                total_steps=0,
                verified_steps=0,
                corrected_steps=0,
                failed_steps=0,
                review_steps=0,
                not_applicable_steps=0,
                unverified_steps=0,
                summary="Input outline is not a dictionary",
                is_safe_for_stage2=True,
            )
            return outline_data, empty_rep

        spec = outline_data.get("worked_example_spec")
        if not spec or not isinstance(spec, dict) or not spec.get("steps"):
            empty_rep = WorkedExampleReport(
                overall_status=VerificationStatus.NOT_APPLICABLE,
                total_steps=0,
                verified_steps=0,
                corrected_steps=0,
                failed_steps=0,
                review_steps=0,
                not_applicable_steps=0,
                unverified_steps=0,
                summary="No evaluable worked_example_spec found in outline",
                is_safe_for_stage2=True,
            )
            return outline_data, empty_rep

        governing_principles = outline_data.get("governing_principles")
        if not isinstance(governing_principles, list):
            governing_principles = None

        topic = str(outline_data.get("topic", ""))

        # Check topic_profile for non-computational or proof topics
        topic_profile = outline_data.get("topic_profile")
        is_non_computational = False
        if isinstance(topic_profile, dict):
            comp_req = str(topic_profile.get("computation_requirement", "")).lower()
            if comp_req in ("none_not_applicable", "symbolic_identity"):
                is_non_computational = True
        elif hasattr(topic_profile, "computation_requirement"):
            comp_val = getattr(topic_profile.computation_requirement, "value", str(topic_profile.computation_requirement)).lower()
            if comp_val in ("none_not_applicable", "symbolic_identity"):
                is_non_computational = True

        # Check topic keywords for biological / qualitative topics
        if not is_non_computational and topic:
            t_low = topic.lower()
            if any(kw in t_low for kw in ("photosynthesis", "calvin cycle", "cellular respiration", "biochemical")):
                is_non_computational = True

        if is_non_computational:
            spec_copy = dict(spec)
            na_steps = []
            for i, st in enumerate(spec.get("steps", [])):
                st_copy = dict(st)
                st_copy["structured_calc"] = {
                    "step_index": i,
                    "status": VerificationStatus.NOT_APPLICABLE.value,
                    "diagnostic": DiagnosticCode.NON_EVALUABLE_STEP.value,
                    "is_evaluable": False,
                    "can_safely_correct": False,
                    "details": "Non-computational or proof-oriented topic; numerical verification marked NOT_APPLICABLE",
                }
                na_steps.append(st_copy)
            spec_copy["steps"] = na_steps

            report = WorkedExampleReport(
                overall_status=VerificationStatus.NOT_APPLICABLE,
                total_steps=len(na_steps),
                verified_steps=0,
                corrected_steps=0,
                failed_steps=0,
                review_steps=0,
                not_applicable_steps=len(na_steps),
                unverified_steps=0,
                summary="Topic profile indicates non-computational topic; numerical verification marked NOT_APPLICABLE.",
                is_safe_for_stage2=True,
                verification_method="pedagogy_profile_bypass",
            )
            outline_copy = dict(outline_data)
            outline_copy["worked_example_spec"] = spec_copy
            outline_copy["verification_metadata"] = report.to_dict()
            return outline_copy, report

        updated_spec, report = self.verify_worked_example_spec(
            spec, governing_principles, topic=topic
        )

        outline_copy = dict(outline_data)
        outline_copy["worked_example_spec"] = updated_spec
        outline_copy["verification_metadata"] = report.to_dict()
        return outline_copy, report

    def verify_worked_example_spec(
        self,
        spec: Dict[str, Any],
        governing_principles: Optional[List[str]] = None,
        topic: str = "",
    ) -> Tuple[Dict[str, Any], WorkedExampleReport]:
        """
        Verifies worked_example_spec: processes steps sequentially, maintains
        dependency environment, safely auto-corrects arithmetic errors, and updates final_result.
        Routes to specialized verifiers (e.g. Dijkstra) via registry when registered.
        """
        # 1. Specialized verifier routing via registry
        specialized = SpecializedVerifierRegistry.find_verifier(spec, topic)
        if specialized is not None:
            return specialized.verify_worked_example_spec(spec, topic=topic)

        # 2. Generic Multi-Domain Orchestration
        spec_copy = dict(spec)
        raw_steps = spec_copy.get("steps", [])
        inputs = dict(spec_copy.get("inputs", {})) if isinstance(spec_copy.get("inputs"), dict) else {}

        # Initialize calculation environment with inputs (preserving all JSON types)
        env = CalculationEnvironment(inputs)

        adapted_steps: List[AdaptedStep] = [
            CalculationStepAdapter.adapt_step(step, i) for i, step in enumerate(raw_steps)
        ]

        verified_steps_output: List[Dict[str, Any]] = []
        step_diagnostics: List[Dict[str, Any]] = []

        total_steps = len(adapted_steps)
        verified_count = 0
        corrected_count = 0
        failed_count = 0
        review_count = 0
        unverified_count = 0
        na_count = 0

        for adapted in adapted_steps:
            step_dict_out = dict(adapted.original_dict)

            # Case A: Non-evaluable or conceptual step
            if adapted.step_type == StepType.CONCEPTUAL_EXPLANATION or not adapted.is_evaluable:
                na_count += 1
                env.record_step_snapshot(adapted.index, description=adapted.description, step_type=adapted.step_type.value)
                structured_meta = {
                    "step_index": adapted.index,
                    "status": VerificationStatus.NOT_APPLICABLE.value,
                    "diagnostic": DiagnosticCode.NON_EVALUABLE_STEP.value,
                    "step_type": adapted.step_type.value,
                    "is_evaluable": False,
                    "can_safely_correct": False,
                    "details": "Step represents a conceptual note or narrative explanation",
                    "state_snapshot": env.get_state_at_step(adapted.index),
                }
                step_dict_out["structured_calc"] = structured_meta
                verified_steps_output.append(step_dict_out)
                step_diagnostics.append(structured_meta)
                continue

            # Case B: State Transition / Assignment
            if adapted.step_type == StepType.STATE_TRANSITION:
                step_res_dict, count_type = self._verify_state_transition_step(adapted, env)
                if count_type == "verified":
                    verified_count += 1
                elif count_type == "failed":
                    failed_count += 1
                elif count_type == "unverified":
                    unverified_count += 1
                elif count_type == "review":
                    review_count += 1

                env.record_step_snapshot(adapted.index, description=adapted.description, step_type=adapted.step_type.value)
                step_res_dict["state_snapshot"] = env.get_state_at_step(adapted.index)
                step_dict_out["structured_calc"] = step_res_dict
                verified_steps_output.append(step_dict_out)
                step_diagnostics.append(step_res_dict)
                continue

            # Case C: Sequence / Array Operation
            if adapted.step_type == StepType.SEQUENCE_OPERATION:
                step_res_dict, count_type = self._verify_sequence_step(adapted, env)
                if count_type == "verified":
                    verified_count += 1
                elif count_type == "failed":
                    failed_count += 1
                elif count_type == "unverified":
                    unverified_count += 1
                elif count_type == "review":
                    review_count += 1

                env.record_step_snapshot(adapted.index, description=adapted.description, step_type=adapted.step_type.value)
                step_res_dict["state_snapshot"] = env.get_state_at_step(adapted.index)
                step_dict_out["structured_calc"] = step_res_dict
                verified_steps_output.append(step_dict_out)
                step_diagnostics.append(step_res_dict)
                continue

            # Case D: Scalar Calculation Step
            expr_to_eval = adapted.expression or ""
            out_var = adapted.output_variable or f"step_{adapted.index + 1}_val"
            claimed_prec = None
            try:
                claimed_prec = PrecisionExtractor.extract_precision(adapted.claimed_value or "")
                claimed_float = claimed_prec.claimed_float
            except Exception:
                claimed_float = None

            # Dependency Resolution & Corrected Literal Substitution
            sub_expr, was_sub, sub_reasons = env.substitute_dependencies(expr_to_eval)
            if was_sub:
                expr_to_eval = sub_expr

            # Check if expression references any untrusted dependencies
            ref_vars = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", FormulaNormalizer.sanitize_latex(expr_to_eval))
            MATH_IDENTIFIERS = {"exp", "ln", "log", "log10", "sqrt", "abs", "sin", "cos", "tan", "pi", "e"}
            has_unresolved_dep = False
            unresolved_var_name = None
            unresolved_detail = None
            for v in ref_vars:
                if v in MATH_IDENTIFIERS:
                    continue
                if v in env.active_variables:
                    if env.active_variables[v].trust in (VariableTrust.UNRESOLVED, VariableTrust.UNTRUSTED):
                        has_unresolved_dep = True
                        unresolved_var_name = v
                        unresolved_detail = f"Dependency '{v}' failed verification or is unresolved; downstream evaluation blocked"
                        break
                else:
                    # Variable not defined in environment
                    has_unresolved_dep = True
                    unresolved_var_name = v
                    unresolved_detail = f"Dependency '{v}' is missing or unresolved; downstream evaluation blocked"
                    break

            if has_unresolved_dep:
                review_count += 1
                env.set_unresolved(
                    adapted.index,
                    out_var,
                    unresolved_detail or f"Blocked because dependency '{unresolved_var_name}' is untrusted or unresolved",
                    claimed_float=claimed_float,
                )
                structured_meta = {
                    "step_index": adapted.index,
                    "status": VerificationStatus.NEEDS_REVIEW.value,
                    "diagnostic": DiagnosticCode.MISSING_VARIABLE.value,
                    "output_variable": out_var,
                    "original_claimed": adapted.claimed_value,
                    "can_safely_correct": False,
                    "details": unresolved_detail or f"Dependency '{unresolved_var_name}' failed verification or is unresolved",
                    "unit": adapted.unit,
                }
                step_dict_out["structured_calc"] = structured_meta
                verified_steps_output.append(step_dict_out)
                step_diagnostics.append(structured_meta)
                continue

            # Contextualize Inputs for Formula Trust Matching
            active_trusted_env = env.get_trusted_values()
            context_inputs = dict(active_trusted_env)

            step_text_full = f"{adapted.description} {adapted.raw_result}"
            out_var_lower = (adapted.output_variable or "").lower()

            is_t2_explicit = out_var_lower in ("k2", "exp_t2", "k_2", "t2") or "exp_t2" in out_var_lower
            is_t1_explicit = out_var_lower in ("k1", "exp_t1", "k_1", "t1") or "exp_t1" in out_var_lower

            t2_val_str = str(int(context_inputs["T2"])) if ("T2" in context_inputs and isinstance(context_inputs["T2"], (int, float)) and math.isfinite(context_inputs["T2"])) else None
            t1_val_str = str(int(context_inputs["T1"])) if ("T1" in context_inputs and isinstance(context_inputs["T1"], (int, float)) and math.isfinite(context_inputs["T1"])) else None

            is_t2_text = (
                re.search(r"\b(t2|k2|k_2|second\s+temperature|temperature\s+2)\b", step_text_full, re.IGNORECASE) is not None
                or (t2_val_str and re.search(rf"\b{re.escape(t2_val_str)}\b", step_text_full) is not None)
            )
            is_t1_text = (
                re.search(r"\b(t1|k1|k_1|first\s+temperature|temperature\s+1)\b", step_text_full, re.IGNORECASE) is not None
                or (t1_val_str and re.search(rf"\b{re.escape(t1_val_str)}\b", step_text_full) is not None)
            )

            if "T2" in context_inputs and (is_t2_explicit or (is_t2_text and not is_t1_explicit)):
                context_inputs = {("T" if k == "T2" else k): v for k, v in context_inputs.items()}
            elif "T1" in context_inputs and (is_t1_explicit or (is_t1_text and not is_t2_explicit)):
                context_inputs = {("T" if k == "T1" else k): v for k, v in context_inputs.items()}
            elif "T" not in context_inputs and "T1" in context_inputs and "T2" not in context_inputs:
                context_inputs = {("T" if k == "T1" else k): v for k, v in context_inputs.items()}

            # Check Formula Trust
            trust_expr = env.substitute_formulas_for_trust(adapted.expression or "")
            trust_claim_str = f"{trust_expr} = {adapted.claimed_value}"

            step_res = self.step_verifier.verify_step(
                trust_claim_str,
                inputs=context_inputs,
                governing_principles=governing_principles,
            )

            eval_vars = {k: float(v) for k, v in active_trusted_env.items()}
            direct_eval = self.evaluator.safe_evaluate(
                FormulaNormalizer.sanitize_latex(expr_to_eval), eval_vars
            )
            direct_match = False
            direct_diag = None
            direct_details = ""
            if direct_eval.value is not None:
                direct_match, direct_diag, direct_details = ToleranceEvaluator.evaluate(
                    PrecisionExtractor.extract_precision(adapted.claimed_value or ""),
                    direct_eval.value,
                )

            if step_res.status == VerificationStatus.FAILED and step_res.formula_trust == FormulaTrustStatus.TRUSTED:
                if direct_match:
                    step_res = VerificationResult(
                        status=VerificationStatus.VERIFIED,
                        formula_trust=step_res.formula_trust,
                        diagnostic=direct_diag,
                        claimed_raw=step_res.claimed_raw,
                        claimed_value=step_res.claimed_value,
                        computed_value=direct_eval.value,
                        half_ulp=step_res.half_ulp,
                        is_approximate=step_res.is_approximate,
                        can_safely_correct=False,
                        details=f"Verified via direct calculation: {direct_details}",
                        matched_template=step_res.matched_template,
                        unit=step_res.unit,
                    )

            # Untrusted Formula Handling
            if step_res.formula_trust != FormulaTrustStatus.TRUSTED:
                # Check whether topic or inputs belong to a registered physics/chemistry domain where trusted formulas are expected
                topic_lower = topic.lower()
                is_known_scientific_domain = (
                    any(dom in topic_lower for dom in ("arrhenius", "kinetics", "rate", "velocity", "difference quotient", "physics", "chemistry", "derivative", "projectile"))
                    or ("Ea" in context_inputs and ("R" in context_inputs or "T" in context_inputs or "T1" in context_inputs or "T2" in context_inputs))
                    or ("v0" in context_inputs and "g" in context_inputs)
                    or ("u" in context_inputs and "a" in context_inputs)
                    or (any("=" in str(p) or "exp(" in str(p) for p in (governing_principles or [])) and any(dom in topic_lower for dom in ("physics", "chem", "kinetics", "arrhenius", "thermo")))
                )

                if is_known_scientific_domain:
                    review_count += 1
                    env.set_unresolved(
                        step_index=adapted.index,
                        var_name=out_var,
                        diagnostic=f"Formula is untrusted: {step_res.details}",
                        claimed_float=claimed_float,
                    )
                    structured_meta = {
                        "step_index": adapted.index,
                        "status": VerificationStatus.NEEDS_REVIEW.value,
                        "formula_trust": step_res.formula_trust.value,
                        "diagnostic": DiagnosticCode.UNTRUSTED_FORMULA.value,
                        "output_variable": out_var,
                        "expression": adapted.expression,
                        "original_claimed": adapted.claimed_value,
                        "computed_value": step_res.computed_value,
                        "can_safely_correct": False,
                        "details": f"Formula is untrusted or unverified in registered domain: {step_res.details}",
                        "unit": adapted.unit,
                    }
                else:
                    # In generic domains, check if mathematical expression can be evaluated directly
                    if direct_eval.value is not None:
                        if direct_match:
                            verified_count += 1
                            if out_var:
                                env.set_step_result(
                                    step_index=adapted.index,
                                    var_name=out_var,
                                    value=direct_eval.value,
                                    trust=VariableTrust.VERIFIED_INTERMEDIATE,
                                    claimed_float=claimed_float,
                                    claimed_sample=adapted.claimed_value,
                                    diagnostic="Direct calculation verified",
                                )
                            structured_meta = {
                                "step_index": adapted.index,
                                "status": VerificationStatus.VERIFIED.value,
                                "formula_trust": FormulaTrustStatus.UNTRUSTED.value,
                                "diagnostic": DiagnosticCode.EXACT_MATCH.value,
                                "output_variable": out_var,
                                "expression": adapted.expression,
                                "claimed_value": adapted.claimed_value,
                                "computed_value": direct_eval.value,
                                "can_safely_correct": False,
                                "details": f"General arithmetic verified directly: {direct_details}",
                                "unit": adapted.unit,
                            }
                        else:
                            # Contradiction: Expression evaluates to a different number
                            failed_count += 1
                            env.set_unresolved(
                                step_index=adapted.index,
                                var_name=out_var,
                                diagnostic=f"Arithmetic mismatch: claimed {claimed_float}, computed {direct_eval.value}",
                                claimed_float=claimed_float,
                            )
                            structured_meta = {
                                "step_index": adapted.index,
                                "status": VerificationStatus.FAILED.value,
                                "formula_trust": FormulaTrustStatus.UNTRUSTED.value,
                                "diagnostic": DiagnosticCode.ARITHMETIC_MISMATCH.value,
                                "output_variable": out_var,
                                "expression": adapted.expression,
                                "original_claimed": adapted.claimed_value,
                                "computed_value": direct_eval.value,
                                "can_safely_correct": False,
                                "details": f"Calculation error: {adapted.expression} = {direct_eval.value}, but claimed {adapted.claimed_value}",
                                "unit": adapted.unit,
                            }
                    else:
                        # Cannot evaluate directly (coverage limit) -> UNVERIFIED
                        unverified_count += 1
                        if claimed_float is not None and out_var:
                            env.set_step_result(
                                step_index=adapted.index,
                                var_name=out_var,
                                value=claimed_float,
                                trust=VariableTrust.VERIFIED_INTERMEDIATE,
                                diagnostic="Unverified formula claim accepted for downstream state",
                            )
                        structured_meta = {
                            "step_index": adapted.index,
                            "status": VerificationStatus.UNVERIFIED.value,
                            "formula_trust": step_res.formula_trust.value,
                            "diagnostic": DiagnosticCode.UNVERIFIED_COVERAGE.value,
                            "output_variable": out_var,
                            "expression": adapted.expression,
                            "claimed_value": adapted.claimed_value,
                            "computed_value": step_res.computed_value,
                            "can_safely_correct": False,
                            "details": "Formula does not match registered domain template; proceeding as unverified",
                            "unit": adapted.unit,
                        }
                step_dict_out["structured_calc"] = structured_meta
                verified_steps_output.append(step_dict_out)
                step_diagnostics.append(structured_meta)
                continue

            if step_res.status == VerificationStatus.VERIFIED:
                verified_count += 1
                env.set_step_result(
                    step_index=adapted.index,
                    var_name=out_var,
                    value=step_res.computed_value,
                    trust=VariableTrust.VERIFIED_INTERMEDIATE,
                    diagnostic=step_res.diagnostic.value,
                    claimed_sample=adapted.claimed_value,
                    claimed_float=claimed_float,
                    formula_str=adapted.expression,
                )
                structured_meta = {
                    "step_index": adapted.index,
                    "status": VerificationStatus.VERIFIED.value,
                    "formula_trust": step_res.formula_trust.value,
                    "diagnostic": step_res.diagnostic.value,
                    "output_variable": out_var,
                    "expression": adapted.expression,
                    "claimed_value": adapted.claimed_value,
                    "computed_value": step_res.computed_value,
                    "half_ulp": step_res.half_ulp,
                    "can_safely_correct": False,
                    "details": step_res.details,
                    "unit": adapted.unit,
                }
                step_dict_out["structured_calc"] = structured_meta
                verified_steps_output.append(step_dict_out)
                step_diagnostics.append(structured_meta)

            elif (
                step_res.status == VerificationStatus.FAILED
                and step_res.can_safely_correct
                and adapted.allow_auto_correction
            ):
                corrected_count += 1
                computed_val = step_res.computed_value
                formatted_val = format_corrected_value(computed_val, adapted.claimed_value)

                corrected_unit_str = f" {adapted.unit}" if adapted.unit else ""
                corrected_result_str = f"{adapted.expression} = {formatted_val}{corrected_unit_str}"

                env.set_step_result(
                    step_index=adapted.index,
                    var_name=out_var,
                    value=computed_val,
                    trust=VariableTrust.CORRECTED_INTERMEDIATE,
                    diagnostic="Auto-corrected arithmetic mismatch",
                    claimed_sample=adapted.claimed_value,
                    claimed_float=claimed_float,
                    formula_str=adapted.expression,
                )

                if "original_result" not in step_dict_out:
                    step_dict_out["original_result"] = adapted.raw_result
                step_dict_out["result"] = corrected_result_str

                structured_meta = {
                    "step_index": adapted.index,
                    "status": VerificationStatus.FAILED.value,
                    "resolution": "AUTO_CORRECTED",
                    "formula_trust": step_res.formula_trust.value,
                    "diagnostic": step_res.diagnostic.value,
                    "output_variable": out_var,
                    "expression": adapted.expression,
                    "original_claimed": adapted.claimed_value,
                    "computed_value": computed_val,
                    "corrected_value": formatted_val,
                    "half_ulp": step_res.half_ulp,
                    "can_safely_correct": True,
                    "correction_reason": "Arithmetic mismatch with trusted formula and verified inputs",
                    "details": step_res.details,
                    "unit": adapted.unit,
                }
                step_dict_out["structured_calc"] = structured_meta
                verified_steps_output.append(step_dict_out)
                step_diagnostics.append(structured_meta)

            else:
                if step_res.status == VerificationStatus.FAILED:
                    failed_count += 1
                else:
                    review_count += 1
                env.set_unresolved(
                    step_index=adapted.index,
                    var_name=out_var,
                    diagnostic=f"Step failed verification ({step_res.diagnostic.value}) and cannot be safely corrected",
                    claimed_float=claimed_float,
                )
                structured_meta = {
                    "step_index": adapted.index,
                    "status": step_res.status.value,
                    "formula_trust": step_res.formula_trust.value,
                    "diagnostic": step_res.diagnostic.value,
                    "output_variable": out_var,
                    "expression": adapted.expression,
                    "original_claimed": adapted.claimed_value,
                    "computed_value": step_res.computed_value,
                    "can_safely_correct": False,
                    "details": step_res.details,
                    "unit": adapted.unit,
                }
                step_dict_out["structured_calc"] = structured_meta
                verified_steps_output.append(step_dict_out)
                step_diagnostics.append(structured_meta)

        # 3. Verify & Reconcile final_result
        final_res_raw = spec_copy.get("final_result", "")
        updated_final_res, final_status = self.verify_and_reconcile_final_result(
            final_res_raw, env
        )
        if updated_final_res != final_res_raw:
            if "original_final_result" not in spec_copy:
                spec_copy["original_final_result"] = final_res_raw
            spec_copy["final_result"] = updated_final_res

        spec_copy["steps"] = verified_steps_output

        # Determine overall report status
        if failed_count > 0:
            overall_status = VerificationStatus.FAILED
            is_safe = False
        elif review_count > 0:
            overall_status = VerificationStatus.NEEDS_REVIEW
            is_safe = False
        elif corrected_count > 0:
            overall_status = VerificationStatus.VERIFIED
            is_safe = True
        elif verified_count > 0:
            overall_status = VerificationStatus.VERIFIED
            is_safe = True
        elif unverified_count > 0:
            overall_status = VerificationStatus.UNVERIFIED
            is_safe = True
        else:
            overall_status = VerificationStatus.NOT_APPLICABLE
            is_safe = True

        summary_parts = []
        if verified_count > 0:
            summary_parts.append(f"{verified_count} verified")
        if corrected_count > 0:
            summary_parts.append(f"{corrected_count} safely corrected")
        if failed_count > 0:
            summary_parts.append(f"{failed_count} failed")
        if review_count > 0:
            summary_parts.append(f"{review_count} flagged for review")
        if unverified_count > 0:
            summary_parts.append(f"{unverified_count} unverified")
        if na_count > 0:
            summary_parts.append(f"{na_count} non-evaluable")

        summary = f"Worked example verification: {', '.join(summary_parts) if summary_parts else 'no steps'}."
        if overall_status == VerificationStatus.UNVERIFIED:
            summary += " Proceeding to Stage 2 with unverified metadata (independent coverage unavailable)."

        snaps: Dict[int, Dict[str, Any]] = {}
        for i in range(len(adapted_steps)):
            try:
                snaps[i] = env.get_state_at_step(i)
            except Exception:
                pass

        report = WorkedExampleReport(
            overall_status=overall_status,
            total_steps=total_steps,
            verified_steps=verified_count,
            corrected_steps=corrected_count,
            failed_steps=failed_count,
            review_steps=review_count,
            not_applicable_steps=na_count,
            unverified_steps=unverified_count,
            step_diagnostics=step_diagnostics,
            final_result_status=final_status,
            summary=summary,
            is_safe_for_stage2=is_safe,
            verification_method="generic_orchestrator",
            state_snapshots=snaps,
        )

        spec_copy["verification_metadata"] = report.to_dict()
        spec_copy["state_snapshots"] = {str(k): copy.deepcopy(v) for k, v in snaps.items()}
        return spec_copy, report

    def _verify_state_transition_step(
        self, adapted: AdaptedStep, env: CalculationEnvironment
    ) -> Tuple[Dict[str, Any], str]:
        """
        Verifies state transitions, assignments, and continuity.
        Returns: (step_res_dict, count_type where count_type in 'verified', 'failed', 'unverified', 'review').
        """
        # 1. State continuity check (state_before)
        if adapted.state_before:
            for k, expected_v in adapted.state_before.items():
                curr_v = env.current_state.get(k)
                if curr_v is not None and curr_v != expected_v:
                    meta = {
                        "step_index": adapted.index,
                        "status": VerificationStatus.FAILED.value,
                        "diagnostic": DiagnosticCode.STATE_CONTINUITY_MISMATCH.value,
                        "details": f"State continuity mismatch: variable '{k}' had value {curr_v}, expected {expected_v}",
                        "can_safely_correct": False,
                    }
                    env.set_unresolved(adapted.index, k, meta["details"])
                    return meta, "failed"

        # 2. Assignment with expression evaluation (e.g. mid = (low + high) // 2)
        if adapted.expression and adapted.output_variable and adapted.claimed_value:
            num_vars = env.get_numeric_values()
            try:
                # Sanitize integer division // if present
                clean_expr = FormulaNormalizer.sanitize_latex(adapted.expression)
                eval_res = self.evaluator.safe_evaluate(clean_expr, num_vars)
                if eval_res.value is not None:
                    claimed_float = float(adapted.claimed_value)
                    if abs(eval_res.value - claimed_float) < 1e-6:
                        env.set_step_result(
                            step_index=adapted.index,
                            var_name=adapted.output_variable,
                            value=eval_res.value,
                            trust=VariableTrust.VERIFIED_INTERMEDIATE,
                            diagnostic=DiagnosticCode.EXACT_MATCH.value,
                            claimed_sample=adapted.claimed_value,
                            claimed_float=claimed_float,
                            formula_str=adapted.expression,
                        )
                        meta = {
                            "step_index": adapted.index,
                            "status": VerificationStatus.VERIFIED.value,
                            "diagnostic": DiagnosticCode.EXACT_MATCH.value,
                            "output_variable": adapted.output_variable,
                            "expression": adapted.expression,
                            "claimed_value": adapted.claimed_value,
                            "computed_value": eval_res.value,
                            "details": f"Expression evaluated successfully: {clean_expr} = {eval_res.value}",
                        }
                        return meta, "verified"
                    else:
                        # Value contradiction!
                        meta = {
                            "step_index": adapted.index,
                            "status": VerificationStatus.FAILED.value,
                            "diagnostic": DiagnosticCode.ARITHMETIC_MISMATCH.value,
                            "output_variable": adapted.output_variable,
                            "expression": adapted.expression,
                            "claimed_value": adapted.claimed_value,
                            "computed_value": eval_res.value,
                            "details": f"Arithmetic contradiction: {clean_expr} evaluated to {eval_res.value}, claimed {claimed_float}",
                            "can_safely_correct": False,
                        }
                        env.set_unresolved(adapted.index, adapted.output_variable, meta["details"], claimed_float=claimed_float)
                        return meta, "failed"
            except Exception:
                pass

        # 3. Multiple variable initialization/assignment (e.g. low = 0, high = 9)
        assigns = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,;]+)", adapted.raw_result)
        if assigns:
            recorded = []
            for var_name, val_str in assigns:
                val_clean = val_str.strip()
                try:
                    num_val = float(val_clean)
                    val_to_store = num_val
                except ValueError:
                    val_to_store = val_clean

                env.set_step_result(
                    step_index=adapted.index,
                    var_name=var_name,
                    value=val_to_store,
                    trust=VariableTrust.VERIFIED_INTERMEDIATE,
                    diagnostic="State transition assignment",
                )
                recorded.append(f"{var_name} = {val_to_store}")

            meta = {
                "step_index": adapted.index,
                "status": VerificationStatus.VERIFIED.value,
                "diagnostic": DiagnosticCode.STATE_TRANSITION_VERIFIED.value,
                "details": f"State variables recorded: {', '.join(recorded)}",
            }
            return meta, "verified"

        # 4. Single variable assignment without expression
        if adapted.output_variable and adapted.claimed_value:
            try:
                num_val = float(adapted.claimed_value)
                val_to_store = num_val
            except ValueError:
                val_to_store = adapted.claimed_value

            env.set_step_result(
                step_index=adapted.index,
                var_name=adapted.output_variable,
                value=val_to_store,
                trust=VariableTrust.VERIFIED_INTERMEDIATE,
                diagnostic="Direct state assignment",
            )
            meta = {
                "step_index": adapted.index,
                "status": VerificationStatus.VERIFIED.value,
                "diagnostic": DiagnosticCode.STATE_TRANSITION_VERIFIED.value,
                "output_variable": adapted.output_variable,
                "claimed_value": adapted.claimed_value,
                "details": f"Direct assignment recorded: {adapted.output_variable} = {val_to_store}",
            }
            return meta, "verified"

        # 5. Fallback unverified state transition
        meta = {
            "step_index": adapted.index,
            "status": VerificationStatus.UNVERIFIED.value,
            "diagnostic": DiagnosticCode.UNVERIFIED_COVERAGE.value,
            "details": "State transition recorded without independent mathematical check",
        }
        return meta, "unverified"

    def _verify_sequence_step(
        self, adapted: AdaptedStep, env: CalculationEnvironment
    ) -> Tuple[Dict[str, Any], str]:
        """
        Verifies sequence / array indexing and comparison operations.
        Returns: (step_res_dict, count_type).
        """
        desc = f"{adapted.description} {adapted.raw_result}"
        # Matches arr[mid] = arr[4] = 5 or arr[4] == 5 or arr[mid] = 5
        seq_matches = list(re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]\s*(?:==|=)\s*(?:[a-zA-Z_][a-zA-Z0-9_]*\[[^\]]+\]\s*(?:==|=)\s*)?([^,\s;]+)", desc))
        if not seq_matches:
            seq_matches = list(re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\]\s*(?:==|=)\s*([^,\s;]+)", desc))

        arr_verified = False
        arr_contradiction = False
        details = "Sequence operation evaluated"

        for seq_match in seq_matches:
            seq_name = seq_match.group(1)
            idx_expr = seq_match.group(2).strip()
            claimed_elem = seq_match.group(3).strip()

            # Skip intermediate chained access e.g. claimed_elem == 'arr[4]'
            if "[" in claimed_elem:
                continue

            seq_rec = env.get(seq_name)
            if seq_rec and isinstance(seq_rec.value, list):
                arr_list = seq_rec.value
                idx_val = None
                if idx_expr.isdigit():
                    idx_val = int(idx_expr)
                elif env.get(idx_expr) and env.get(idx_expr).value is not None:
                    try:
                        idx_val = int(env.get(idx_expr).value)
                    except (ValueError, TypeError):
                        pass

                if idx_val is not None and 0 <= idx_val < len(arr_list):
                    actual_elem = arr_list[idx_val]
                    try:
                        if float(actual_elem) == float(claimed_elem):
                            arr_verified = True
                            details = f"Sequence access verified: {seq_name}[{idx_val}] == {actual_elem}"
                            break
                        else:
                            arr_contradiction = True
                            details = f"Sequence element contradiction: {seq_name}[{idx_val}] is {actual_elem}, claimed {claimed_elem}"
                    except ValueError:
                        if str(actual_elem) == str(claimed_elem):
                            arr_verified = True
                            details = f"Sequence access verified: {seq_name}[{idx_val}] == {actual_elem}"
                            break
                        else:
                            arr_contradiction = True
                            details = f"Sequence element contradiction: {seq_name}[{idx_val}] is {actual_elem}, claimed {claimed_elem}"

        if arr_contradiction:
            meta = {
                "step_index": adapted.index,
                "status": VerificationStatus.FAILED.value,
                "diagnostic": DiagnosticCode.SEQUENCE_MISMATCH.value,
                "details": details,
                "can_safely_correct": False,
            }
            if adapted.output_variable:
                env.set_unresolved(adapted.index, adapted.output_variable, details)
            return meta, "failed"

        # Check if the step also records an updated variable (e.g. low = mid + 1 = 5)
        if adapted.output_variable and adapted.claimed_value:
            try:
                num_val = float(adapted.claimed_value)
                val_to_store = num_val
            except ValueError:
                val_to_store = adapted.claimed_value

            env.set_step_result(
                step_index=adapted.index,
                var_name=adapted.output_variable,
                value=val_to_store,
                trust=VariableTrust.VERIFIED_INTERMEDIATE,
                diagnostic="Sequence operation state update",
            )

        if arr_verified:
            meta = {
                "step_index": adapted.index,
                "status": VerificationStatus.VERIFIED.value,
                "diagnostic": DiagnosticCode.SEQUENCE_ELEMENT_VERIFIED.value,
                "output_variable": adapted.output_variable,
                "claimed_value": adapted.claimed_value,
                "details": details,
            }
            return meta, "verified"
        else:
            meta = {
                "step_index": adapted.index,
                "status": VerificationStatus.UNVERIFIED.value,
                "diagnostic": DiagnosticCode.UNVERIFIED_COVERAGE.value,
                "output_variable": adapted.output_variable,
                "claimed_value": adapted.claimed_value,
                "details": "Sequence operation consistent; independent element check not fully bound",
            }
            return meta, "unverified"

    def verify_and_reconcile_final_result(
        self, final_result_str: str, env: CalculationEnvironment
    ) -> Tuple[str, VerificationStatus]:
        """
        Reconciles final_result string with values in the calculation environment.
        If dependent steps were corrected, updates the final result string accordingly.
        Returns: (reconciled_string, verification_status).
        """
        s = str(final_result_str).strip()
        if not s:
            return s, VerificationStatus.NOT_APPLICABLE

        updated_s = s

        # 1. Structured variable assignments: 'k1 = 0.0001087 s^-1, k2 = 0.00284 s^-1' or 'mid = 6'
        var_matches = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^,;]+)", s)
        if var_matches:
            has_unresolved_var = False
            all_vars_valid = True

            for var_name, claimed_val_raw in var_matches:
                rec = env.get(var_name)
                if not rec or rec.value is None or rec.trust in (VariableTrust.UNRESOLVED, VariableTrust.UNTRUSTED):
                    all_vars_valid = False
                    has_unresolved_var = True
                    break

                if isinstance(rec.value, (int, float)):
                    claimed_num = None
                    try:
                        p = PrecisionExtractor.extract_precision(claimed_val_raw)
                        claimed_num = p.claimed_float
                    except Exception:
                        pass

                    if claimed_num is not None:
                        is_match, _, _ = ToleranceEvaluator.evaluate(
                            PrecisionExtractor.extract_precision(claimed_val_raw),
                            float(rec.value),
                        )
                        if not is_match:
                            fmt_val = format_corrected_value(rec.value, claimed_val_raw)
                            p = PrecisionExtractor.extract_precision(claimed_val_raw)
                            unit_str = f" {p.unit}" if (p and p.unit) else ""
                            old_assign = f"{var_name} = {claimed_val_raw.strip()}"
                            new_assign = f"{var_name} = {fmt_val}{unit_str}".strip()
                            updated_s = updated_s.replace(old_assign, new_assign)
                else:
                    if str(rec.value) != str(claimed_val_raw).strip():
                        all_vars_valid = False

            if has_unresolved_var:
                return updated_s, VerificationStatus.NEEDS_REVIEW
            return updated_s, (VerificationStatus.VERIFIED if all_vars_valid else VerificationStatus.NEEDS_REVIEW)

        # 2. Free-text reconciliation (substitute any corrected numbers)
        for old_claimed, corrected_val in sorted(
            env.claimed_to_corrected.items(), key=lambda x: len(str(x[0])), reverse=True
        ):
            cand_strs = [str(old_claimed), f"{old_claimed:g}"]
            for cand in cand_strs:
                corr_s = format_corrected_value(corrected_val, cand)
                pat = rf"(?<![a-zA-Z0-9_])(?<![0-9]\.){re.escape(cand)}(?![a-zA-Z0-9_]|\.[0-9])"
                if re.search(pat, updated_s):
                    updated_s = re.sub(pat, corr_s, updated_s)
                    break

        if env.claimed_to_corrected:
            return updated_s, VerificationStatus.VERIFIED

        # Check if trusted intermediate numbers are present in final text
        all_trusted = env.get_trusted_values()
        if all_trusted:
            matched_any = False
            for var_name, val in all_trusted.items():
                val_s = f"{val:g}"
                if len(val_s) >= 2 and val_s in updated_s:
                    matched_any = True
                    break
            if matched_any:
                return updated_s, VerificationStatus.VERIFIED

        return updated_s, VerificationStatus.NOT_APPLICABLE


# Auto-register built-in specialized verifiers
try:
    import dijkstra_verifier
except ImportError:
    pass

