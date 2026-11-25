"""
Mathematical Validator for CineLearn (Code2Video).

Uses local SymPy and NumPy dependencies to:
1. Deterministically validate mathematical expressions, calculations, and formulas
   in lecture lines, storyboard animations, and generated Manim MathTex.
2. Cross-check consistency between spoken narration, animation descriptions, and displayed equations.
3. Explicitly flag non-deterministic or qualitative mathematical claims for review
   instead of assuming correctness.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


_MATH_FUNCS = {
    "det", "tr", "trace", "sin", "cos", "tan", "exp", "log", "ln", "sqrt",
    "lim", "gcd", "lcm", "rank", "dim", "ker", "null", "span", "max", "min", "abs",
    "alpha", "beta", "gamma", "theta", "lambda", "pi", "mu", "sigma", "omega", "phi",
}

_PROSE_STOPWORDS = {
    "first", "second", "third", "then", "next", "now", "we", "have", "calculate",
    "find", "compute", "note", "notice", "see", "observe", "get", "obtain", "yields",
    "gives", "shows", "display", "mathtex", "formula", "equation", "let", "so", "thus",
    "hence", "therefore", "where", "since", "if", "for", "here", "with", "that", "this",
    "the", "an", "a", "is", "are", "as", "to", "in", "on", "at", "by", "from", "step",
}


@dataclass
class MathClaim:
    expression: str
    claim_type: str  # "arithmetic_equality", "matrix_determinant", "matrix_vector_product", "calculus_derivative", "algebraic_identity", "qualitative_claim"
    source: str      # "lecture_line", "animation", "mathtex", "storyboard"
    status: str      # "VERIFIED_TRUE", "VERIFIED_FALSE", "NEEDS_REVIEW"
    claimed_value: Optional[str] = None
    computed_value: Optional[str] = None
    details: str = ""
    suggested_fix: Optional[str] = None


@dataclass
class MathValidationReport:
    section_id: str
    claims: List[MathClaim] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    verified_true_count: int = 0
    verified_false_count: int = 0
    needs_review_count: int = 0
    is_valid: bool = True
    summary: str = ""


class MathValidator:
    """
    Deterministically validates mathematical claims and cross-checks consistency
    using local SymPy and NumPy.
    """

    TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

    def __init__(self):
        pass

    def _clean_latex_expr(self, expr_str: str) -> str:
        """Sanitize LaTeX markup into standard mathematical syntax for SymPy parsing."""
        s = expr_str.strip()
        # Remove LaTeX display markers
        s = re.sub(r"^\$+|\$+$", "", s).strip()
        s = s.replace(r"\cdot", "*")
        s = s.replace(r"\times", "*")
        s = s.replace(r"\div", "/")
        s = s.replace(r"\pm", "±")
        s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
        s = re.sub(r"\\left|\\right", "", s)
        s = s.replace(r"\det", "det")
        s = s.replace(r"\pi", "pi")
        s = s.replace(r"\lambda", "lambda")
        # Remove residual LaTeX slashes before standard functions like \sin, \cos, \exp
        s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
        return s.strip()

    def verify_equality(self, lhs_str: str, rhs_str: str) -> Tuple[str, Optional[str], Optional[str], str]:
        """
        Verify LHS == RHS using SymPy.
        Returns (status, claimed_rhs, computed_lhs, details).
        """
        clean_lhs = self._clean_latex_expr(lhs_str)
        clean_rhs = self._clean_latex_expr(rhs_str)

        try:
            lhs_expr = parse_expr(clean_lhs, transformations=self.TRANSFORMATIONS)
            rhs_expr = parse_expr(clean_rhs, transformations=self.TRANSFORMATIONS)

            # Check exact or simplified difference
            diff = sp.simplify(lhs_expr - rhs_expr)
            if diff == 0 or diff.is_zero:
                return "VERIFIED_TRUE", str(rhs_expr), str(lhs_expr), f"Identity confirmed: {clean_lhs} == {clean_rhs}"

            # Numeric evaluation check if purely numerical
            if lhs_expr.is_number and rhs_expr.is_number:
                lhs_val = float(lhs_expr.evalf())
                rhs_val = float(rhs_expr.evalf())
                if abs(lhs_val - rhs_val) < 1e-6:
                    return "VERIFIED_TRUE", str(rhs_val), str(lhs_val), f"Numeric equality confirmed: {lhs_val} == {rhs_val}"
                else:
                    return (
                        "VERIFIED_FALSE",
                        str(rhs_val),
                        str(lhs_val),
                        f"Mathematical error: {clean_lhs} evaluates to {lhs_val}, but claimed {rhs_val}",
                    )

            # If difference is a non-zero number, it is provably false (e.g. (x+2) - (x+1) = 1 != 0)
            if diff.is_number and not diff.is_zero:
                return (
                    "VERIFIED_FALSE",
                    str(rhs_expr),
                    str(lhs_expr),
                    f"Symbolic inequality: {clean_lhs} != {clean_rhs} (constant difference {diff})",
                )

            # Check if both are polynomials in the same symbols and have differing coefficients
            symbols_lhs = lhs_expr.free_symbols
            symbols_rhs = rhs_expr.free_symbols
            if symbols_lhs and symbols_lhs == symbols_rhs and not lhs_expr.atoms(sp.Function) and not rhs_expr.atoms(sp.Function):
                try:
                    poly_diff = sp.Poly(diff, *list(symbols_lhs))
                    if not poly_diff.is_zero:
                        return (
                            "VERIFIED_FALSE",
                            str(rhs_expr),
                            str(lhs_expr),
                            f"Symbolic polynomial inequality: {clean_lhs} != {clean_rhs}",
                        )
                except Exception:
                    pass

            # For expressions containing uninstantiated functions (e.g. det(A)), definitions, or context-dependent variables
            return (
                "NEEDS_REVIEW",
                str(rhs_expr),
                str(lhs_expr),
                f"Expression '{clean_lhs} = {clean_rhs}' contains uninstantiated definitions or functions. Cannot be verified deterministically; flagged for review.",
            )

        except Exception as e:
            return (
                "NEEDS_REVIEW",
                clean_rhs,
                None,
                f"Expression could not be deterministically parsed by SymPy ({e}). Flagged for human review.",
            )

    def verify_matrix_determinant(
        self, matrix_elements: List[List[float]], claimed_det: float
    ) -> Tuple[str, float, float, str]:
        """Verify 2x2 or 3x3 determinant calculation with NumPy and SymPy."""
        try:
            arr = np.array(matrix_elements, dtype=float)
            computed = float(np.linalg.det(arr))
            if abs(computed - claimed_det) < 1e-5:
                return "VERIFIED_TRUE", claimed_det, round(computed, 4), f"Determinant verified: {computed:.4f}"
            else:
                return (
                    "VERIFIED_FALSE",
                    claimed_det,
                    round(computed, 4),
                    f"Determinant error: computed {computed:.4f}, but claimed {claimed_det}",
                )
        except Exception as e:
            return "NEEDS_REVIEW", claimed_det, 0.0, f"Matrix calculation error: {e}"

    @staticmethod
    def _clean_math_tokens_lhs(lhs: str) -> str:
        """Strip non-mathematical prose tokens from the left-hand side."""
        lhs = lhs.strip().rstrip(".,;:!?")
        if "," in lhs:
            lhs = lhs.split(",")[-1].strip()
        words = lhs.split()
        while words:
            w = words[0].lower().strip(".,:;()")
            if w in _PROSE_STOPWORDS or (len(w) > 2 and w not in _MATH_FUNCS and "(" not in words[0]):
                words.pop(0)
            else:
                break
        return " ".join(words) if words else lhs

    @staticmethod
    def _clean_math_tokens_rhs(rhs: str) -> str:
        """Strip non-mathematical prose tokens and trailing punctuation from the right-hand side."""
        rhs = rhs.strip().rstrip(".,;:!?")
        words = rhs.split()
        math_words = []
        for w in words:
            w_clean = w.lower().strip(".,:;()")
            if w_clean in _PROSE_STOPWORDS or (len(w_clean) > 2 and w_clean not in _MATH_FUNCS and "(" not in w):
                break
            math_words.append(w)
        res = " ".join(math_words) if math_words else rhs
        return res.rstrip(".,;:!?")

    def extract_and_verify_claims_from_text(
        self, text: str, source: str = "lecture_line"
    ) -> List[MathClaim]:
        """
        Extracts mathematical equality claims, calculations, and formulas from natural text.
        Verifies them deterministically with SymPy/NumPy or flags for review.
        """
        claims: List[MathClaim] = []

        # 1. Pattern: explicit equation with equals sign: e.g. "1*4 - 2*3 = -2", "2 + 2 = 4", "det(A) = ad - bc", "dist[B] = 4"
        eq_patterns = [
            r"([0-9a-zA-Z\s\+\-\*\/\^\(\)\[\]_]+)\s*=\s*([0-9a-zA-Z\s\+\-\*\/\^\(\)\[\]_]+)",
        ]

        for pat in eq_patterns:
            for match in re.finditer(pat, text):
                raw_lhs, raw_rhs = match.group(1).strip(), match.group(2).strip()
                lhs = self._clean_math_tokens_lhs(raw_lhs)
                rhs = self._clean_math_tokens_rhs(raw_rhs)
                if len(lhs) < 1 or len(rhs) < 1:
                    continue
                has_math = any(c in lhs + rhs for c in "+-*/^0123456789") or any(
                    k in (lhs + rhs).lower() for k in ["det", "sin", "cos", "sqrt", "matrix", "pi", "min", "max", "dist", "cost", "dp"]
                )
                if not has_math:
                    continue

                full_expr = f"{lhs} = {rhs}"
                status, claimed, computed, details = self.verify_equality(lhs, rhs)

                suggested_fix = None
                if status == "VERIFIED_FALSE" and computed is not None:
                    suggested_fix = f"{lhs} = {computed}"

                claims.append(
                    MathClaim(
                        expression=full_expr,
                        claim_type="arithmetic_equality" if any(c in full_expr for c in "0123456789") else "algebraic_identity",
                        source=source,
                        status=status,
                        claimed_value=claimed,
                        computed_value=computed,
                        details=details,
                        suggested_fix=suggested_fix,
                    )
                )

        # 2. Pattern: 2x2 matrix determinant claim like "matrix [[1, 2], [3, 4]] with determinant -2"
        det_match = re.search(
            r"\[\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*\].*?determinant.*?(-?\d+(?:\.\d+)?)",
            text,
            re.IGNORECASE,
        )
        if det_match:
            a, b, c, d, det_val = [float(det_match.group(i)) for i in range(1, 6)]
            status, claimed, computed, details = self.verify_matrix_determinant([[a, b], [c, d]], det_val)
            claims.append(
                MathClaim(
                    expression=f"det([[{a}, {b}], [{c}, {d}]]) = {det_val}",
                    claim_type="matrix_determinant",
                    source=source,
                    status=status,
                    claimed_value=str(claimed),
                    computed_value=str(computed),
                    details=details,
                    suggested_fix=f"det([[{a}, {b}], [{c}, {d}]]) = {computed}" if status == "VERIFIED_FALSE" else None,
                )
            )

        # 3. Qualitative mathematical assertions that cannot be verified deterministically
        qualitative_keywords = [
            r"\b(?:invariant|eigenvector|eigenvalue|kernel|null space|span|basis)s?\b",
            r"\b(?:orthogonal|invertible|singular|isomorphism|continuous|differentiable)\b",
        ]
        for q_pat in qualitative_keywords:
            if re.search(q_pat, text, re.IGNORECASE):
                # If no explicit verified equality was parsed for this specific assertion, flag it
                claims.append(
                    MathClaim(
                        expression=text.strip(),
                        claim_type="qualitative_claim",
                        source=source,
                        status="NEEDS_REVIEW",
                        details="Conceptual or qualitative mathematical claim. Deterministic proof not applicable; flagged for instructional verification.",
                    )
                )
                break

        return claims

    def extract_mathtex_from_code(self, code: str) -> List[str]:
        """Extracts LaTeX mathematical strings from Manim MathTex and Tex objects."""
        mathtex_strings: List[str] = []
        # Matches MathTex(r"...") or MathTex("...") or Tex(...)
        patterns = [
            r"MathTex\(\s*r?[\"'](.*?)[\"']\s*\)",
            r"MathTex\(\s*r?[\"\"\"|'''](.*?)[\"\"\"|''']\s*\)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, code, re.DOTALL):
                content = m.group(1).strip()
                if content:
                    mathtex_strings.append(content)
        return mathtex_strings

    def cross_check_narration_and_code(
        self, lecture_lines: List[str], code_str: str
    ) -> List[str]:
        """
        Cross-checks consistency between spoken narration claims and visual MathTex code.
        Detects contradictions (e.g. narrative claims determinant is -2, but MathTex displays 2).
        """
        inconsistencies: List[str] = []
        mathtex_list = self.extract_mathtex_from_code(code_str)

        # Extract numeric values claimed in narration
        narration_text = " ".join(lecture_lines)

        # Check determinants
        narr_det_match = re.search(r"determinant\s*(?:is|=|of)?\s*(-?\d+(?:\.\d+)?)", narration_text, re.IGNORECASE)
        if narr_det_match:
            narr_det = float(narr_det_match.group(1))
            for tex in mathtex_list:
                tex_det_match = re.search(r"\\det\s*(?:\([^\)]+\))?\s*=\s*(-?\d+(?:\.\d+)?)", tex)
                if tex_det_match:
                    tex_det = float(tex_det_match.group(1))
                    if abs(narr_det - tex_det) > 1e-4:
                        inconsistencies.append(
                            f"Consistency contradiction: Narration states determinant is {narr_det}, "
                            f"but displayed MathTex shows '\\det = {tex_det}'."
                        )

        # Check general arithmetic equality discrepancies in MathTex vs narration
        for tex in mathtex_list:
            if "=" in tex:
                parts = tex.split("=", 1)
                lhs, rhs = parts[0].strip(), parts[1].strip()
                # Verify MathTex internal correctness
                status, claimed, computed, details = self.verify_equality(lhs, rhs)
                if status == "VERIFIED_FALSE":
                    inconsistencies.append(
                        f"MathTex formula '{tex}' contains mathematical error: {details}"
                    )

        return inconsistencies

    def validate_section(
        self, section: Any, section_code: Optional[str] = None
    ) -> MathValidationReport:
        """
        Validates mathematical content of a Section object:
        - Evaluates lecture lines
        - Evaluates animation descriptions
        - Evaluates Manim code MathTex (if provided)
        - Cross-checks consistency between narration and visuals
        """
        section_id = getattr(section, "id", "section")
        lecture_lines = getattr(section, "lecture_lines", [])
        animations = getattr(section, "animations", [])

        all_claims: List[MathClaim] = []

        # Validate lecture lines
        for i, line in enumerate(lecture_lines):
            line_claims = self.extract_and_verify_claims_from_text(line, source=f"lecture_line_{i+1}")
            all_claims.extend(line_claims)

        # Validate animations
        for i, anim in enumerate(animations):
            anim_claims = self.extract_and_verify_claims_from_text(anim, source=f"animation_{i+1}")
            all_claims.extend(anim_claims)

        # Cross-check consistency if code is available
        inconsistencies: List[str] = []
        if section_code:
            inconsistencies = self.cross_check_narration_and_code(lecture_lines, section_code)

        verified_true = sum(1 for c in all_claims if c.status == "VERIFIED_TRUE")
        verified_false = sum(1 for c in all_claims if c.status == "VERIFIED_FALSE")
        needs_review = sum(1 for c in all_claims if c.status == "NEEDS_REVIEW")

        is_valid = (verified_false == 0) and (len(inconsistencies) == 0)

        summary = (
            f"Math validation for {section_id}: "
            f"{verified_true} verified true, {verified_false} false, {needs_review} needs review, "
            f"{len(inconsistencies)} inconsistencies."
        )

        return MathValidationReport(
            section_id=section_id,
            claims=all_claims,
            inconsistencies=inconsistencies,
            verified_true_count=verified_true,
            verified_false_count=verified_false,
            needs_review_count=needs_review,
            is_valid=is_valid,
            summary=summary,
        )

    def fix_code_mathtex(self, code: str, report: MathValidationReport) -> Tuple[str, bool]:
        """
        Deterministically repair mathematical errors in MathTex code blocks ($0 API cost).
        Substitutes verified true calculated values for false claimed values in MathTex.
        """
        if not report or report.is_valid:
            return code, False

        modified_code = code
        was_modified = False

        # 1. Parse discrepancies from inconsistencies
        for inc in report.inconsistencies:
            # Pattern: MathTex formula '1 + 1 = 3' contains mathematical error: Mathematical error: 1 + 1 evaluates to 2.0, but claimed 3.0
            m = re.search(r"MathTex formula '([^']+)' contains mathematical error:.*evaluates to (-?[0-9\.]+), but claimed (-?[0-9\.]+)", inc)
            if m:
                tex_raw = m.group(1)
                computed = m.group(2)
                claimed = m.group(3)
                try:
                    if float(computed).is_integer():
                        computed = str(int(float(computed)))
                    if float(claimed).is_integer():
                        claimed = str(int(float(claimed)))
                except ValueError:
                    pass

                def _replace_inc(match):
                    nonlocal was_modified
                    call = match.group(0)
                    inner = match.group(1)
                    if "=" in inner:
                        lhs_part, rhs_part = inner.split("=", 1)
                        if claimed in rhs_part:
                            new_rhs = rhs_part.replace(claimed, computed, 1)
                            new_inner = f"{lhs_part}={new_rhs}"
                            was_modified = True
                            return call.replace(inner, new_inner)
                    return call

                tex_pat = re.compile(r'MathTex\(\s*r?["\'](.*?)["\']\s*\)')
                modified_code = tex_pat.sub(_replace_inc, modified_code)

            # Determinant contradiction: Narration states determinant is -2, but displayed MathTex shows '\det = 2'
            m_det = re.search(r"Narration states determinant is (-?\d+(?:\.\d+)?).*?shows.*?(-?\d+(?:\.\d+)?)", inc)
            if m_det:
                true_val = m_det.group(1)
                wrong_val = m_det.group(2)
                try:
                    if float(true_val).is_integer():
                        true_val = str(int(float(true_val)))
                    if float(wrong_val).is_integer():
                        wrong_val = str(int(float(wrong_val)))
                except ValueError:
                    pass

                def _replace_det(match):
                    nonlocal was_modified
                    call = match.group(0)
                    tex = match.group(1)
                    if "\\det" in tex and wrong_val in tex:
                        was_modified = True
                        new_tex = re.sub(rf'(\\det\s*(?:\([^\)]+\))?\s*=\s*){re.escape(wrong_val)}', rf'\g<1>{true_val}', tex)
                        return call.replace(tex, new_tex)
                    return call

                tex_pat = re.compile(r'MathTex\(\s*r?["\'](.*?)["\']\s*\)')
                modified_code = tex_pat.sub(_replace_det, modified_code)

        # 2. Fix claims with status == "VERIFIED_FALSE"
        for claim in report.claims:
            if claim.status == "VERIFIED_FALSE" and claim.claimed_value and claim.computed_value:
                claimed = claim.claimed_value.strip()
                computed = claim.computed_value.strip()

                try:
                    if float(claimed) == float(computed):
                        continue
                    if float(computed).is_integer():
                        computed = str(int(float(computed)))
                    if float(claimed).is_integer():
                        claimed = str(int(float(claimed)))
                except ValueError:
                    pass

                def _replace_tex(m):
                    nonlocal was_modified
                    full_call = m.group(0)
                    inner_tex = m.group(1)
                    if "=" in inner_tex and claimed in inner_tex:
                        parts = inner_tex.split("=", 1)
                        lhs = parts[0]
                        rhs = parts[1]
                        if claimed in rhs:
                            new_rhs = rhs.replace(claimed, computed, 1)
                            new_inner = f"{lhs}={new_rhs}"
                            was_modified = True
                            return full_call.replace(inner_tex, new_inner)
                    return full_call

                tex_pat = re.compile(r'MathTex\(\s*r?["\'](.*?)["\']\s*\)')
                modified_code = tex_pat.sub(_replace_tex, modified_code)

        return modified_code, was_modified

    def validate_storyboard(
        self, storyboard_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validates all sections in a storyboard dictionary."""
        sections = storyboard_data.get("sections", [])
        reports: List[Dict[str, Any]] = []
        total_true = 0
        total_false = 0
        total_review = 0
        all_inconsistencies = []

        for sec in sections:
            sid = sec.get("id", "section")
            lines = sec.get("lecture_lines", [])
            anims = sec.get("animations", [])

            claims: List[MathClaim] = []
            for i, line in enumerate(lines):
                claims.extend(self.extract_and_verify_claims_from_text(line, source=f"lecture_line_{i+1}"))
            for i, anim in enumerate(anims):
                claims.extend(self.extract_and_verify_claims_from_text(anim, source=f"animation_{i+1}"))
            for i, val in enumerate(sec.get("verified_values", [])):
                claims.extend(self.extract_and_verify_claims_from_text(val, source=f"verified_value_{i+1}"))

            v_true = sum(1 for c in claims if c.status == "VERIFIED_TRUE")
            v_false = sum(1 for c in claims if c.status == "VERIFIED_FALSE")
            v_review = sum(1 for c in claims if c.status == "NEEDS_REVIEW")

            total_true += v_true
            total_false += v_false
            total_review += v_review

            reports.append({
                "section_id": sid,
                "verified_true": v_true,
                "verified_false": v_false,
                "needs_review": v_review,
                "claims": [c.__dict__ for c in claims],
            })

        return {
            "total_verified_true": total_true,
            "total_verified_false": total_false,
            "total_needs_review": total_review,
            "is_valid": total_false == 0,
            "section_reports": reports,
            "summary": f"Storyboard mathematical validation: {total_true} verified, {total_false} false, {total_review} flagged for review.",
        }
