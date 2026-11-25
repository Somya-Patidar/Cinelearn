"""
Numerical Evaluator for CineLearn (Milestone 1).

Deterministic, sandbox-secure numerical evaluation, formula-trust verification,
and precision-aware tolerance comparison for instructional worked examples.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import math
import operator as op
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


# ============================================================================
# Status & Diagnostic Enums
# ============================================================================

class VerificationStatus(str, Enum):
    """Overall outcome of a mathematical step verification."""
    VERIFIED = "VERIFIED"
    CORRECTED = "CORRECTED"
    FAILED = "FAILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    UNVERIFIED = "UNVERIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FormulaTrustStatus(str, Enum):
    """Trust classification for the underlying algebraic formula."""
    TRUSTED = "TRUSTED"
    UNTRUSTED = "UNTRUSTED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DiagnosticCode(str, Enum):
    """Detailed diagnostic reason explaining verification and trust outcomes."""
    # Verified diagnostics
    EXACT_MATCH = "EXACT_MATCH"
    ROUNDING_MATCH = "ROUNDING_MATCH"
    RELATIVE_MATCH = "RELATIVE_MATCH"

    # Failed diagnostics
    ARITHMETIC_MISMATCH = "ARITHMETIC_MISMATCH"
    ROUNDING_CONTRADICTION = "ROUNDING_CONTRADICTION"

    # Needs review diagnostics
    UNTRUSTED_FORMULA = "UNTRUSTED_FORMULA"
    AMBIGUOUS_LITERAL_MAPPING = "AMBIGUOUS_LITERAL_MAPPING"
    MISSING_VARIABLE = "MISSING_VARIABLE"
    DIVISION_BY_ZERO = "DIVISION_BY_ZERO"
    OVERFLOW_ERROR = "OVERFLOW_ERROR"
    COMPLEXITY_EXCEEDED = "COMPLEXITY_EXCEEDED"
    DISALLOWED_CONSTRUCT = "DISALLOWED_CONSTRUCT"
    NON_FINITE_RESULT = "NON_FINITE_RESULT"
    PARSE_SYNTAX_ERROR = "PARSE_SYNTAX_ERROR"
    INVALID_INPUT_VALUES = "INVALID_INPUT_VALUES"

    # Not applicable diagnostics
    ALGORITHMIC_STATE_UPDATE = "ALGORITHMIC_STATE_UPDATE"
    NON_EVALUABLE_STEP = "NON_EVALUABLE_STEP"

    # Algorithmic diagnostics (Milestone 2B.6A)
    ALGORITHMIC_VERIFIED = "ALGORITHMIC_VERIFIED"
    ALGORITHMIC_STATE_MISMATCH = "ALGORITHMIC_STATE_MISMATCH"
    INVALID_GRAPH_DEFINITION = "INVALID_GRAPH_DEFINITION"
    NEGATIVE_EDGE_WEIGHT = "NEGATIVE_EDGE_WEIGHT"
    INCOMPLETE_TRACE = "INCOMPLETE_TRACE"
    INVALID_EXTRACT_MIN = "INVALID_EXTRACT_MIN"
    INVALID_RELAXATION = "INVALID_RELAXATION"

    # Generic & Unverified diagnostics (Milestone 2B.6C)
    UNVERIFIED_COVERAGE = "UNVERIFIED_COVERAGE"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    STATE_CONTINUITY_MISMATCH = "STATE_CONTINUITY_MISMATCH"
    STATE_TRANSITION_VERIFIED = "STATE_TRANSITION_VERIFIED"
    SEQUENCE_ELEMENT_VERIFIED = "SEQUENCE_ELEMENT_VERIFIED"
    SEQUENCE_MISMATCH = "SEQUENCE_MISMATCH"
    GENERIC_CONSISTENCY_PASS = "GENERIC_CONSISTENCY_PASS"


# ============================================================================
# Data Containers
# ============================================================================

@dataclass(frozen=True)
class ClaimedPrecision:
    """Metadata describing the extracted precision and scale of a claimed value."""
    claimed_raw: str
    claimed_float: float
    is_scientific: bool
    decimal_places: int
    half_ulp: float
    is_integer: bool
    is_approximate: bool
    unit: Optional[str] = None


@dataclass(frozen=True)
class EvaluationResult:
    """Result of safe AST evaluation."""
    value: Optional[float]
    status: VerificationStatus
    diagnostic: DiagnosticCode
    details: str


@dataclass(frozen=True)
class VerificationResult:
    """Complete verification outcome for an equality claim."""
    status: VerificationStatus
    formula_trust: FormulaTrustStatus
    diagnostic: DiagnosticCode
    claimed_raw: str
    claimed_value: Optional[float]
    computed_value: Optional[float]
    half_ulp: Optional[float]
    is_approximate: bool
    can_safely_correct: bool
    details: str
    matched_template: Optional[str] = None
    unit: Optional[str] = None


# ============================================================================
# 1. Safe AST Evaluator
# ============================================================================

class SafeASTEvaluator:
    """
    Evaluates mathematical expressions deterministically without eval() or exec().
    Enforces strict whitelisting, complexity bounds, and numerical safety guards.
    """

    MAX_EXPR_LENGTH = 300
    MAX_NODE_COUNT = 50
    MAX_DEPTH = 10
    MAX_NUMERIC_MAGNITUDE = 1e100
    MAX_EXPONENT_MAGNITUDE = 150

    ALLOWED_BIN_OPS = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod,
        ast.Pow: op.pow,
    }

    ALLOWED_UNARY_OPS = {
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    ALLOWED_FUNCS = {
        "exp": math.exp,
        "sqrt": math.sqrt,
        "log": math.log,
        "ln": math.log,
        "log10": math.log10,
        "abs": abs,
    }

    ALLOWED_CONSTANTS = {
        "pi": math.pi,
        "PI": math.pi,
        "e": math.e,
        "E": math.e,
    }

    def __init__(self, allow_math_functions: bool = True):
        self.allow_math_functions = allow_math_functions

    def _count_and_check_nodes(self, node: ast.AST, current_depth: int = 1) -> int:
        """Recursively count nodes and check depth limits."""
        if current_depth > self.MAX_DEPTH:
            raise ValueError(f"AST recursion depth exceeded max allowed ({self.MAX_DEPTH})")

        count = 1
        for child in ast.iter_child_nodes(node):
            count += self._count_and_check_nodes(child, current_depth + 1)
            if count > self.MAX_NODE_COUNT:
                raise ValueError(f"AST node count exceeded max allowed ({self.MAX_NODE_COUNT})")
        return count

    def _eval_node(self, node: ast.AST, variables: Dict[str, float]) -> float:
        """Recursively evaluate an AST node in the restricted mathematical grammar."""
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, variables)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                val = float(node.value)
                if not math.isfinite(val):
                    raise ValueError("Non-finite numeric constant encountered")
                if abs(val) > self.MAX_NUMERIC_MAGNITUDE:
                    raise OverflowError(f"Constant exceeds magnitude ceiling ({self.MAX_NUMERIC_MAGNITUDE})")
                return val
            raise TypeError(f"Disallowed constant type: {type(node.value).__name__}")

        elif isinstance(node, ast.Name):
            name = node.id
            if name in variables:
                val = float(variables[name])
                if not math.isfinite(val):
                    raise ValueError(f"Variable '{name}' has non-finite value: {val}")
                return val
            if name in self.ALLOWED_CONSTANTS:
                return self.ALLOWED_CONSTANTS[name]
            raise NameError(f"Undefined variable or constant: {name}")

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_UNARY_OPS:
                raise TypeError(f"Disallowed unary operator: {op_type.__name__}")
            operand_val = self._eval_node(node.operand, variables)
            res = self.ALLOWED_UNARY_OPS[op_type](operand_val)
            if not math.isfinite(res):
                raise ValueError("Non-finite result from unary operation")
            return res

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.ALLOWED_BIN_OPS:
                raise TypeError(f"Disallowed binary operator: {op_type.__name__}")

            left_val = self._eval_node(node.left, variables)
            right_val = self._eval_node(node.right, variables)

            # Division safety
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right_val == 0.0:
                raise ZeroDivisionError("Division by zero")

            # Exponentiation bounds & domain safety
            if op_type is ast.Pow:
                if abs(right_val) > self.MAX_EXPONENT_MAGNITUDE:
                    raise OverflowError(f"Exponent exceeds magnitude limit ({self.MAX_EXPONENT_MAGNITUDE})")
                if left_val < 0.0 and not right_val.is_integer():
                    raise ValueError("Complex numbers from negative base with fractional exponent not supported")

            try:
                res = float(self.ALLOWED_BIN_OPS[op_type](left_val, right_val))
            except OverflowError as e:
                raise OverflowError(f"Arithmetic overflow: {e}")

            if not math.isfinite(res):
                raise ValueError("Non-finite result from binary operation")
            if abs(res) > self.MAX_NUMERIC_MAGNITUDE:
                raise OverflowError(f"Intermediate result exceeds magnitude limit ({self.MAX_NUMERIC_MAGNITUDE})")
            return res

        elif isinstance(node, ast.Call):
            if not self.allow_math_functions:
                raise TypeError("Function calls are rejected by security policy")

            if not isinstance(node.func, ast.Name):
                raise TypeError("Disallowed attribute access in function call (e.g. obj.method)")

            func_name = node.func.id
            if func_name not in self.ALLOWED_FUNCS:
                raise NameError(f"Disallowed function call: {func_name}")

            if len(node.args) != 1:
                raise ValueError(f"Function '{func_name}' expects exactly 1 argument, got {len(node.args)}")
            if node.keywords:
                raise ValueError(f"Keyword arguments not allowed in '{func_name}'")

            arg_val = self._eval_node(node.args[0], variables)

            # Function domain guards
            if func_name == "exp" and arg_val > 100.0:
                raise OverflowError(f"exp({arg_val}) exceeds exponent safety ceiling of 100.0")
            if func_name in ("log", "ln", "log10") and arg_val <= 0.0:
                raise ValueError(f"{func_name} argument must be strictly positive, got {arg_val}")
            if func_name == "sqrt" and arg_val < 0.0:
                raise ValueError(f"sqrt argument must be non-negative, got {arg_val}")

            res = float(self.ALLOWED_FUNCS[func_name](arg_val))
            if not math.isfinite(res):
                raise ValueError(f"Non-finite result from {func_name}")
            return res

        else:
            raise TypeError(f"Disallowed AST construct: {type(node).__name__}")

    def evaluate(self, expr_str: str, variables: Optional[Dict[str, float]] = None) -> float:
        """
        Evaluate an arithmetic expression string with supplied variables.
        Raises specific exceptions upon syntax, complexity, security, or domain errors.
        """
        s = expr_str.strip()
        if not s:
            raise ValueError("Empty expression string")
        if len(s) > self.MAX_EXPR_LENGTH:
            raise ValueError(f"Expression length ({len(s)}) exceeds maximum allowed ({self.MAX_EXPR_LENGTH})")

        variables = variables or {}

        try:
            tree = ast.parse(s, mode="eval")
        except SyntaxError as e:
            raise SyntaxError(f"Invalid arithmetic syntax: {e}")

        # Check node count and depth limits
        self._count_and_check_nodes(tree)

        # Evaluate root node
        return self._eval_node(tree.body, variables)

    def safe_evaluate(
        self, expr_str: str, variables: Optional[Dict[str, float]] = None
    ) -> EvaluationResult:
        """
        Safely evaluate expression and return structured EvaluationResult without raising.
        """
        try:
            val = self.evaluate(expr_str, variables)
            return EvaluationResult(
                value=val,
                status=VerificationStatus.VERIFIED,
                diagnostic=DiagnosticCode.EXACT_MATCH,
                details=f"Evaluated successfully to {val}",
            )
        except ZeroDivisionError as e:
            return EvaluationResult(
                value=None,
                status=VerificationStatus.NEEDS_REVIEW,
                diagnostic=DiagnosticCode.DIVISION_BY_ZERO,
                details=str(e),
            )
        except OverflowError as e:
            return EvaluationResult(
                value=None,
                status=VerificationStatus.NEEDS_REVIEW,
                diagnostic=DiagnosticCode.OVERFLOW_ERROR,
                details=str(e),
            )
        except NameError as e:
            return EvaluationResult(
                value=None,
                status=VerificationStatus.NEEDS_REVIEW,
                diagnostic=DiagnosticCode.MISSING_VARIABLE,
                details=str(e),
            )
        except (TypeError, ValueError) as e:
            msg = str(e)
            if "exceeded" in msg or "ceiling" in msg or "limit" in msg:
                code = DiagnosticCode.COMPLEXITY_EXCEEDED
            elif "Disallowed" in msg:
                code = DiagnosticCode.DISALLOWED_CONSTRUCT
            elif "Non-finite" in msg:
                code = DiagnosticCode.NON_FINITE_RESULT
            else:
                code = DiagnosticCode.PARSE_SYNTAX_ERROR
            return EvaluationResult(
                value=None,
                status=VerificationStatus.NEEDS_REVIEW,
                diagnostic=code,
                details=msg,
            )
        except SyntaxError as e:
            return EvaluationResult(
                value=None,
                status=VerificationStatus.NEEDS_REVIEW,
                diagnostic=DiagnosticCode.PARSE_SYNTAX_ERROR,
                details=str(e),
            )
        except Exception as e:
            return EvaluationResult(
                value=None,
                status=VerificationStatus.NEEDS_REVIEW,
                diagnostic=DiagnosticCode.PARSE_SYNTAX_ERROR,
                details=f"Unexpected evaluation error: {e}",
            )


# ============================================================================
# 2. Precision Extraction
# ============================================================================

class PrecisionExtractor:
    """Extracts displayed precision, half-ULP, scale, and approximation metadata."""

    APPROX_PATTERNS = [r"\\approx", r"\bapprox\b", r"\baround\b", r"~", r"≈"]
    UNIT_PATTERN = r"(?:\\text\{([^}]+)\}|([a-zA-Z\^_\-/]+(?:\^[0-9\-]+)?))\s*$"

    @classmethod
    def sanitize_claimed_string(cls, raw_str: str) -> Tuple[str, bool, Optional[str]]:
        """
        Strips surrounding text/LaTeX markers, detects approximation tokens,
        and isolates physical units.
        Returns: (clean_numeric_str, is_approximate, unit_str).
        """
        s = raw_str.strip()
        is_approx = False

        # Detect approximation tokens
        for pat in cls.APPROX_PATTERNS:
            if re.search(pat, s, re.IGNORECASE):
                is_approx = True
                s = re.sub(pat, " ", s, flags=re.IGNORECASE).strip()

        # Remove LaTeX display dollar signs
        s = re.sub(r"^\$+|\$+$", "", s).strip()

        # Extract and strip unit annotations (e.g. '\text{s}^{-1}', 's^-1', 'J/mol')
        unit = None
        # Match \text{...} or \mathrm{...}
        text_unit_match = re.search(r"\\(?:text|mathrm)\{([^}]+)\}(?:\^\{?(-?[0-9]+)\}?)?", s)
        if text_unit_match:
            u_base = text_unit_match.group(1)
            u_exp = text_unit_match.group(2)
            unit = f"{u_base}^{u_exp}" if u_exp else u_base
            s = re.sub(r"\\(?:text|mathrm)\{[^}]+\}(?:\^\{?-?[0-9]+\}?)?", "", s).strip()
        else:
            # Check for trailing raw unit words like 's^-1', 'J/mol'
            raw_unit_match = re.search(r"\s+([a-zA-Z][a-zA-Z0-9\^\-_/\.]*)$", s)
            if raw_unit_match:
                candidate_unit = raw_unit_match.group(1)
                # Ensure it's not a scientific exponent token like 'e10' or 'e-4'
                if not re.match(r"^[eE][+\-]?[0-9]+$", candidate_unit):
                    unit = candidate_unit
                    s = s[: raw_unit_match.start()].strip()

        # Strip LaTeX spacing
        s = re.sub(r"\\[,;!]|\\quad|\\qquad", "", s).strip()
        return s, is_approx, unit

    @classmethod
    def extract_precision(
        cls, raw_claimed: str, integer_policy: str = "exact"
    ) -> ClaimedPrecision:
        """
        Analyzes the claimed numeric string and calculates authoritative half-ULP.
        Integer policy: 'exact' (tolerance 1e-12) or 'unit_rounding' (half_ulp 0.5).
        """
        clean_s, is_approx, unit = cls.sanitize_claimed_string(raw_claimed)

        # Parse scientific notation: e.g. 1.087e-4, 7.5 * 10^10, 7.5 \times 10^{10}
        sci_match = re.search(
            r"^([+-]?[0-9]+(?:\.[0-9]+)?)\s*(?:[eE]|\*|\\times|\*10\^|\\times\s*10\^)\s*(?:10\^|\^)?\{?([+-]?[0-9]+)\}?$",
            clean_s,
        )
        if not sci_match:
            # Also match standard e-notation like 1.087e-4
            sci_match = re.search(r"^([+-]?[0-9]+(?:\.[0-9]+)?)[eE]([+-]?[0-9]+)$", clean_s)

        if sci_match:
            significand_str = sci_match.group(1)
            exponent_str = sci_match.group(2)
            sig_float = float(significand_str)
            exp_int = int(exponent_str)
            val_float = sig_float * (10.0 ** exp_int)

            if "." in significand_str:
                d_m = len(significand_str.split(".")[1])
            else:
                d_m = 0

            # half-ULP = 0.5 * 10^(exponent - decimal_places)
            half_ulp = 0.5 * (10.0 ** (exp_int - d_m))

            return ClaimedPrecision(
                claimed_raw=raw_claimed,
                claimed_float=val_float,
                is_scientific=True,
                decimal_places=d_m,
                half_ulp=half_ulp,
                is_integer=False,
                is_approximate=is_approx,
                unit=unit,
            )

        # Parse standard decimal or integer
        try:
            val_float = float(clean_s)
        except ValueError:
            raise ValueError(f"Could not parse numeric claimed value from '{clean_s}'")

        if "." in clean_s:
            decimal_part = clean_s.split(".")[1]
            d = len(decimal_part)
            half_ulp = 0.5 * (10.0 ** (-d))
            is_integer = False
        else:
            d = 0
            is_integer = True
            half_ulp = 1e-12 if integer_policy == "exact" else 0.5

        return ClaimedPrecision(
            claimed_raw=raw_claimed,
            claimed_float=val_float,
            is_scientific=False,
            decimal_places=d,
            half_ulp=half_ulp,
            is_integer=is_integer,
            is_approximate=is_approx,
            unit=unit,
        )


# ============================================================================
# 3. Precision-Aware Tolerance Evaluator
# ============================================================================

class ToleranceEvaluator:
    """
    Evaluates claimed value against computed value using authoritative displayed precision.
    Relative tolerance NEVER overrides a rounding contradiction.
    """

    DEFAULT_RTOL = 1e-3  # 0.1%

    @classmethod
    def evaluate(
        cls,
        claimed: ClaimedPrecision,
        computed: float,
        rtol: float = DEFAULT_RTOL,
    ) -> Tuple[bool, DiagnosticCode, str]:
        """
        Determines whether claimed matches computed.
        Returns: (is_match, diagnostic_code, explanation).
        """
        if not math.isfinite(computed):
            return False, DiagnosticCode.NON_FINITE_RESULT, f"Computed value is non-finite: {computed}"

        C = claimed.claimed_float
        V = computed
        abs_diff = abs(C - V)

        # Exact match
        if C == V or abs_diff < 1e-12:
            return True, DiagnosticCode.EXACT_MATCH, f"Exact match: claimed {C} == computed {V}"

        # 1. Exact zero check
        if V == 0.0:
            if abs_diff <= claimed.half_ulp:
                return True, DiagnosticCode.EXACT_MATCH, f"Zero match within precision: {abs_diff} <= {claimed.half_ulp}"
            return False, DiagnosticCode.ROUNDING_CONTRADICTION, f"Claimed {C} contradicts computed zero (diff {abs_diff})"

        # 2. Authoritative Rounding Agreement (Condition 1)
        # If displayed precision is identifiable, rounding agreement is authoritative
        if abs_diff <= claimed.half_ulp:
            return (
                True,
                DiagnosticCode.ROUNDING_MATCH,
                f"Rounding agreement confirmed: |{C} - {V:.8g}| = {abs_diff:.4e} <= half-ulp {claimed.half_ulp:.4e}",
            )

        # 3. Check for Rounding Contradiction
        # If the number has explicit decimal precision and is NOT marked approximate,
        # relative tolerance cannot override the rounding bound.
        if not claimed.is_approximate and not (claimed.is_integer and claimed.half_ulp == 1e-12):
            return (
                False,
                DiagnosticCode.ROUNDING_CONTRADICTION,
                f"Rounding contradiction: |{C} - {V:.8g}| = {abs_diff:.4e} > half-ulp {claimed.half_ulp:.4e}. "
                f"Claimed precision forbids difference.",
            )

        # 4. Relative Tolerance Agreement (Condition 2)
        # Permitted ONLY when marked approximate or precision unconstrained
        rel_diff = abs_diff / abs(V)
        if rel_diff <= rtol:
            return (
                True,
                DiagnosticCode.RELATIVE_MATCH,
                f"Approximate relative agreement: relative error {rel_diff:.4e} <= rtol {rtol:.4e}",
            )

        return (
            False,
            DiagnosticCode.ARITHMETIC_MISMATCH,
            f"Arithmetic mismatch: claimed {C} differs from computed {V:.8g} "
            f"(abs diff {abs_diff:.4e}, rel diff {rel_diff:.4e} > {rtol:.4e})",
        )


# ============================================================================
# 4. Formula Normalization & Trusted Formula Registry
# ============================================================================

class FormulaNormalizer:
    """Normalizes LaTeX and mathematical syntax into canonical ASTs."""

    @staticmethod
    def sanitize_latex(expr_str: str) -> str:
        """Sanitizes LaTeX commands into standard arithmetic syntax."""
        s = expr_str.strip()
        # Remove display math markers
        s = re.sub(r"^\$+|\$+$", "", s).strip()
        # Spacing
        s = re.sub(r"\\[,;!]|\\quad|\\qquad", " ", s)
        # Operators
        s = s.replace(r"\times", " * ").replace(r"\cdot", " * ").replace(r"\div", " / ")
        # Fractions
        while r"\frac" in s:
            new_s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))", s)
            if new_s == s:
                break
            s = new_s
        # Roots
        while r"\sqrt" in s:
            new_s = re.sub(r"\\sqrt\[([0-9a-zA-Z_\.]+)\]\{([^}]+)\}", r"((\2)**(1/(\1)))", s)
            new_s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", new_s)
            new_s = re.sub(r"\\sqrt\b", "sqrt", new_s)
            if new_s == s:
                break
            s = new_s
        # Exponentials & Logarithms
        s = re.sub(r"\\exp\b", "exp", s)
        s = re.sub(r"\\ln\b", "ln", s)
        s = re.sub(r"\\log\b", "log", s)
        s = re.sub(r"e\^\{([^}]+)\}", r"exp(\1)", s)
        s = re.sub(r"e\^(-?[0-9a-zA-Z\.]+)", r"exp(\1)", s)
        # Powers
        s = re.sub(r"\^\{([^}]+)\}", r"**(\1)", s)
        s = re.sub(r"\^([0-9a-zA-Z\.]+)", r"**(\1)", s)
        # Standardize function calls with curly braces: func{arg} -> func(arg)
        s = re.sub(r"\b(sqrt|exp|ln|log|abs)\{([^}]+)\}", r"\1(\2)", s)
        # Brackets
        s = s.replace(r"\left", "").replace(r"\right", "")
        # Remove units / text macros
        s = re.sub(r"\\(?:text|mathrm)\{[^}]+\}(?:\^\{?-?[0-9]+\}?)?", "", s)
        return s.strip()

    @classmethod
    def canonicalize_ast(cls, node: ast.AST) -> ast.AST:
        """
        Canonicalizes commutative binary operations (+, *) by sorting operands,
        and normalizes (-A) / B to -(A / B).
        Recursively returns a canonical AST tree for structural comparison.
        """
        if isinstance(node, ast.BinOp):
            node.left = cls.canonicalize_ast(node.left)
            node.right = cls.canonicalize_ast(node.right)

            # Normalize (-A) / B -> -(A / B)
            if isinstance(node.op, ast.Div) and isinstance(node.left, ast.UnaryOp) and isinstance(node.left.op, ast.USub):
                inner_div = ast.BinOp(left=node.left.operand, op=ast.Div(), right=node.right)
                return ast.UnaryOp(op=ast.USub(), operand=inner_div)

            # Sort commutative operands
            if isinstance(node.op, (ast.Add, ast.Mult)):
                left_dump = ast.dump(node.left)
                right_dump = ast.dump(node.right)
                if left_dump > right_dump:
                    node.left, node.right = node.right, node.left
            return node

        elif isinstance(node, ast.UnaryOp):
            node.operand = cls.canonicalize_ast(node.operand)
            return node

        elif isinstance(node, ast.Call):
            node.args = [cls.canonicalize_ast(arg) for arg in node.args]
            return node

        return node

    @classmethod
    def ast_to_canonical_str(cls, expr_str: str) -> str:
        """Parses expression and returns its canonical AST dump string."""
        s = cls.sanitize_latex(expr_str)
        tree = ast.parse(s, mode="eval")
        tree.body = cls.canonicalize_ast(tree.body)
        return ast.dump(tree)


class TrustedFormulaRegistry:
    """
    Registry of canonical trusted formula templates for CineLearn domains.
    Matches formulas conservatively without blind symbolic equivalence.
    """

    # Canonical templates: identifier -> (canonical_expr, required_inputs, constraints)
    TEMPLATES: Dict[str, Dict[str, Any]] = {
        "chemistry.arrhenius_exponent": {
            "template": "-Ea / (R * T)",
            "required_inputs": ["Ea", "R", "T"],
            "constraints": {
                "Ea": lambda x: x > 0,
                "R": lambda x: x > 0,
                "T": lambda x: x > 0,
            },
        },
        "chemistry.arrhenius_rate": {
            "template": "A * exp(-Ea / (R * T))",
            "required_inputs": ["A", "Ea", "R", "T"],
            "constraints": {
                "A": lambda x: x > 0,
                "Ea": lambda x: x > 0,
                "R": lambda x: x > 0,
                "T": lambda x: x > 0,
            },
        },
        "math.difference_quotient": {
            "template": "(f_xh - f_x) / h",
            "required_inputs": ["f_xh", "f_x", "h"],
            "constraints": {
                "h": lambda x: x != 0,
            },
        },
        "physics.average_velocity": {
            "template": "delta_x / delta_t",
            "required_inputs": ["delta_x", "delta_t"],
            "constraints": {
                "delta_t": lambda x: x > 0,
            },
        },
    }

    _COMPILED_CANONICAL: Dict[str, str] = {}

    @classmethod
    def _get_canonical_template(cls, template_id: str) -> str:
        if template_id not in cls._COMPILED_CANONICAL:
            template_expr = cls.TEMPLATES[template_id]["template"]
            cls._COMPILED_CANONICAL[template_id] = FormulaNormalizer.ast_to_canonical_str(template_expr)
        return cls._COMPILED_CANONICAL[template_id]

    @classmethod
    def match_literal_to_inputs(
        cls, expr_str: str, inputs: Dict[str, Any]
    ) -> Tuple[Optional[str], FormulaTrustStatus, str]:
        """
        Conservative literal-to-input mapping.
        Replaces literal numbers in expression with input variable names.
        If any literal matches multiple distinct input keys with the same numeric value,
        mapping is ambiguous -> returns AMBIGUOUS.
        """
        clean_expr = FormulaNormalizer.sanitize_latex(expr_str)

        # Check for ambiguous inputs: multiple keys with identical values
        val_to_keys: Dict[float, List[str]] = {}
        for k, v in inputs.items():
            if isinstance(v, (int, float)):
                val_float = float(v)
                val_to_keys.setdefault(val_float, []).append(k)

        # Find numbers in expression
        num_tokens = re.findall(r"\b\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?\b", clean_expr)

        substituted = clean_expr
        for num_s in sorted(set(num_tokens), key=len, reverse=True):
            try:
                num_val = float(num_s)
            except ValueError:
                continue

            # Check if this number matches an input
            matched_keys = None
            for in_val, keys in val_to_keys.items():
                if abs(num_val - in_val) < 1e-12:
                    matched_keys = keys
                    break

            if matched_keys:
                if len(matched_keys) > 1:
                    return (
                        None,
                        FormulaTrustStatus.AMBIGUOUS,
                        f"Literal number {num_s} is ambiguous: matches multiple inputs {matched_keys}",
                    )
                # Unambiguous replacement using word boundaries
                key_name = matched_keys[0]
                substituted = re.sub(rf"\b{re.escape(num_s)}\b", key_name, substituted)

        return substituted, FormulaTrustStatus.TRUSTED, "Unambiguous literal mapping"

    @classmethod
    def check_formula_trust(
        cls,
        lhs_expr_str: str,
        inputs: Dict[str, Any],
        governing_principles: Optional[List[str]] = None,
    ) -> Tuple[FormulaTrustStatus, Optional[str], str]:
        """
        Conservative formula trust evaluation:
        1. Tests direct template match against TrustedFormulaRegistry.
        2. Tests literal-to-input mapping against TrustedFormulaRegistry.
        3. Tests clean equation matches in governing_principles.
        4. Tests elementary arithmetic on explicit inputs.
        Returns: (FormulaTrustStatus, matched_template_id_or_none, diagnostic_details).
        """
        clean_lhs = FormulaNormalizer.sanitize_latex(lhs_expr_str)

        try:
            lhs_canonical = FormulaNormalizer.ast_to_canonical_str(clean_lhs)
        except Exception as e:
            return FormulaTrustStatus.UNTRUSTED, None, f"Could not parse LHS AST: {e}"

        # 1. Direct canonical template match
        for tmpl_id, tmpl_spec in cls.TEMPLATES.items():
            if lhs_canonical == cls._get_canonical_template(tmpl_id):
                # Verify required inputs exist and satisfy constraints
                valid, reason = cls._validate_constraints(tmpl_spec, inputs)
                if not valid:
                    return FormulaTrustStatus.UNTRUSTED, None, reason
                return FormulaTrustStatus.TRUSTED, tmpl_id, f"Direct match with template {tmpl_id}"

        # 2. Literal-to-input mapping match
        sub_expr, map_status, map_reason = cls.match_literal_to_inputs(clean_lhs, inputs)
        if map_status == FormulaTrustStatus.AMBIGUOUS:
            return FormulaTrustStatus.AMBIGUOUS, None, map_reason

        if sub_expr and sub_expr != clean_lhs:
            try:
                sub_canonical = FormulaNormalizer.ast_to_canonical_str(sub_expr)
                for tmpl_id, tmpl_spec in cls.TEMPLATES.items():
                    if sub_canonical == cls._get_canonical_template(tmpl_id):
                        valid, reason = cls._validate_constraints(tmpl_spec, inputs)
                        if not valid:
                            return FormulaTrustStatus.UNTRUSTED, None, reason
                        return (
                            FormulaTrustStatus.TRUSTED,
                            tmpl_id,
                            f"Mapped literal expression to template {tmpl_id}",
                        )
            except Exception:
                pass

        # 3. Explicit parseable equation match in governing_principles (Tier 2)
        if governing_principles:
            for gp in governing_principles:
                if "=" in gp:
                    gp_parts = gp.split("=", 1)
                    gp_rhs = gp_parts[1].strip()
                    try:
                        gp_canonical = FormulaNormalizer.ast_to_canonical_str(gp_rhs)
                        if lhs_canonical == gp_canonical or (
                            sub_expr and FormulaNormalizer.ast_to_canonical_str(sub_expr) == gp_canonical
                        ):
                            return (
                                FormulaTrustStatus.TRUSTED,
                                "governing_principles.explicit_equation",
                                f"Matches explicit governing principle equation '{gp.strip()}'",
                            )
                    except Exception:
                        continue

        # 4. Elementary binary arithmetic on explicitly defined inputs
        try:
            tree = ast.parse(clean_lhs, mode="eval")
            if isinstance(tree.body, ast.BinOp) and type(tree.body.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
                left_node = tree.body.left
                right_node = tree.body.right
                # Check if operands are known inputs or numbers
                if (isinstance(left_node, (ast.Name, ast.Constant)) and
                    isinstance(right_node, (ast.Name, ast.Constant))):
                    return (
                        FormulaTrustStatus.TRUSTED,
                        "elementary.binary_arithmetic",
                        "Elementary arithmetic between grounded values",
                    )
        except Exception:
            pass

        return (
            FormulaTrustStatus.UNTRUSTED,
            None,
            "Formula does not match any trusted canonical pattern or explicit governing principle",
        )

    @classmethod
    def _validate_constraints(cls, tmpl_spec: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates that required inputs exist and satisfy positive/non-zero constraints."""
        req_inputs = tmpl_spec.get("required_inputs", [])
        constraints = tmpl_spec.get("constraints", {})

        for name in req_inputs:
            # Check direct name or case variants
            val = None
            for k in (name, name.lower(), name.upper()):
                if k in inputs:
                    val = inputs[k]
                    break
            if val is None:
                return False, f"Missing required input variable '{name}' for template"

            if not isinstance(val, (int, float)):
                return False, f"Input variable '{name}' must be numeric, got {type(val).__name__}"

            if name in constraints:
                check_fn = constraints[name]
                if not check_fn(float(val)):
                    return False, f"Input variable '{name}' ({val}) violates domain constraint"

        return True, "Constraints satisfied"


# ============================================================================
# 5. Numerical Step Verifier (Public Milestone 1 Entry Point)
# ============================================================================

class NumericalStepVerifier:
    """
    Integrates safe AST evaluation, formula trust, precision extraction,
    and tolerance verification for an individual worked example calculation step.
    """

    def __init__(self, evaluator: Optional[SafeASTEvaluator] = None):
        self.evaluator = evaluator or SafeASTEvaluator()

    @staticmethod
    def parse_step_equation(step_result_str: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Splits a step result equation string into (lhs_expr, claimed_rhs, unit).
        Example: '-75000 / (8.314 * 300) = -29.74' -> ('-75000 / (8.314 * 300)', '-29.74', None)
        """
        s = step_result_str.strip()
        if "=" not in s:
            return None, None, None

        parts = s.split("=", 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()

        # Separate unit from RHS if present
        clean_rhs, is_approx, unit = PrecisionExtractor.sanitize_claimed_string(rhs)
        return lhs, rhs, unit

    def verify_step(
        self,
        raw_result_str: str,
        inputs: Dict[str, Any],
        governing_principles: Optional[List[str]] = None,
        integer_policy: str = "exact",
        rtol: float = ToleranceEvaluator.DEFAULT_RTOL,
    ) -> VerificationResult:
        """
        Verifies a single calculation claim string (e.g. 'LHS = RHS').
        """
        lhs_raw, rhs_raw, unit = self.parse_step_equation(raw_result_str)
        if not lhs_raw or not rhs_raw:
            return VerificationResult(
                status=VerificationStatus.NOT_APPLICABLE,
                formula_trust=FormulaTrustStatus.NOT_APPLICABLE,
                diagnostic=DiagnosticCode.NON_EVALUABLE_STEP,
                claimed_raw=raw_result_str,
                claimed_value=None,
                computed_value=None,
                half_ulp=None,
                is_approximate=False,
                can_safely_correct=False,
                details="Step result does not contain an evaluable equality ($LHS = RHS$)",
                unit=None,
            )

        # 1. Formula Trust Matching
        formula_trust, tmpl_id, trust_details = TrustedFormulaRegistry.check_formula_trust(
            lhs_raw, inputs, governing_principles
        )

        # 2. Precision Extraction of Claimed RHS
        try:
            claimed_prec = PrecisionExtractor.extract_precision(rhs_raw, integer_policy=integer_policy)
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.NEEDS_REVIEW,
                formula_trust=formula_trust,
                diagnostic=DiagnosticCode.PARSE_SYNTAX_ERROR,
                claimed_raw=rhs_raw,
                claimed_value=None,
                computed_value=None,
                half_ulp=None,
                is_approximate=False,
                can_safely_correct=False,
                details=f"Could not parse claimed numerical RHS: {e}",
                unit=unit,
            )

        # 3. Numeric Evaluation of LHS
        # Convert numeric inputs to float dictionary
        eval_vars = {k: float(v) for k, v in inputs.items() if isinstance(v, (int, float))}
        eval_res = self.evaluator.safe_evaluate(
            FormulaNormalizer.sanitize_latex(lhs_raw), eval_vars
        )

        if eval_res.value is None:
            return VerificationResult(
                status=VerificationStatus.NEEDS_REVIEW,
                formula_trust=formula_trust,
                diagnostic=eval_res.diagnostic,
                claimed_raw=rhs_raw,
                claimed_value=claimed_prec.claimed_float,
                computed_value=None,
                half_ulp=claimed_prec.half_ulp,
                is_approximate=claimed_prec.is_approximate,
                can_safely_correct=False,
                details=f"Evaluation failed: {eval_res.details}",
                unit=unit,
            )

        computed_val = eval_res.value

        # 4. Precision-Aware Tolerance Evaluation
        is_match, diag_code, tol_details = ToleranceEvaluator.evaluate(
            claimed_prec, computed_val, rtol=rtol
        )

        if is_match:
            # Verified
            return VerificationResult(
                status=VerificationStatus.VERIFIED,
                formula_trust=formula_trust,
                diagnostic=diag_code,
                claimed_raw=rhs_raw,
                claimed_value=claimed_prec.claimed_float,
                computed_value=computed_val,
                half_ulp=claimed_prec.half_ulp,
                is_approximate=claimed_prec.is_approximate,
                can_safely_correct=False,
                details=tol_details,
                matched_template=tmpl_id,
                unit=unit,
            )
        else:
            # Failed arithmetic check
            # Eligible for safe correction ONLY if formula is trusted
            can_correct = (
                formula_trust == FormulaTrustStatus.TRUSTED and
                math.isfinite(computed_val)
            )

            status = VerificationStatus.FAILED
            if formula_trust != FormulaTrustStatus.TRUSTED:
                # If formula is untrusted, mark NEEDS_REVIEW to prevent any auto-correction
                status = VerificationStatus.NEEDS_REVIEW

            return VerificationResult(
                status=status,
                formula_trust=formula_trust,
                diagnostic=diag_code if formula_trust == FormulaTrustStatus.TRUSTED else DiagnosticCode.UNTRUSTED_FORMULA,
                claimed_raw=rhs_raw,
                claimed_value=claimed_prec.claimed_float,
                computed_value=computed_val,
                half_ulp=claimed_prec.half_ulp,
                is_approximate=claimed_prec.is_approximate,
                can_safely_correct=can_correct,
                details=f"{tol_details}. Formula trust: {formula_trust.value} ({trust_details})",
                matched_template=tmpl_id,
                unit=unit,
            )
