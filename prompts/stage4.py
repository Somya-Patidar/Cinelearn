import json

MANIM_VERSION = "0.18.1"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def number_code_lines(code):
    """Prefix every line with its 1-based number so MLLM feedback can point at real lines.

    Use it when you pass the code to the feedback prompts:
        get_prompt4_layout_feedback(section, table, hints, code_with_lines=number_code_lines(code))
    """
    return "\n".join(f"{i:>4}: {line}" for i, line in enumerate(code.splitlines(), start=1))


def _code_block(code_with_lines):
    if not code_with_lines:
        return """
CODE: not provided. Set "line_number" to null and identify the object by what it looks like on screen
(for example "the right-hand formula") and, if visible, the variable name.
"""
    return f"""
CODE OF THE SCENE (line numbers are in the left margin; refer to THESE numbers only):
```
{code_with_lines}
```
"""


# ----------------------------------------------------------------------------
# 1. Layout feedback (multimodal: video / frames + optional code)
# ----------------------------------------------------------------------------
def get_prompt4_layout_feedback(section, position_table, local_hints: str = "", code_with_lines=None):
    hint_section = ""
    if local_hints and local_hints.strip():
        hint_section = f"""
LOCAL VALIDATION WARNINGS (geometry issues found by the local checker - verify them in the video, they are usually real):
{local_hints.strip()}
"""

    example = {
        "layout": {
            "has_issues": True,
            "improvements": [
                {
                    "severity": "high | medium | low",
                    "problem": "Concise description of the issue and where it is (grid cells / what is hidden)",
                    "solution": "Line X: change to self.place_in_area(obj, 'B2', 'D4', scale_factor=0.8)",
                    "line_number": 42,
                    "object_affected": "variable name or visual description",
                }
            ],
        }
    }

    return f"""
1. ANALYSIS REQUIREMENTS
- Analyze this Manim educational video ONLY for layout and spatial quality: positioning, scale, overlap, visibility and composition.
- DO NOT change lecture content, mathematical meaning, numbers, explanations or timing. Judge geometry only.
- Use the attached video (and the reference image / grid overlay if one is provided) for precise spatial analysis.
- Goal: no overlaps, no obstruction, nothing cut off, nothing stale on screen, and good use of the available space.
{hint_section}
2. CONTENT CONTEXT
- Title: {section.title}
- Lecture lines: {'; '.join(section.lecture_lines)}
- Current grid occupancy: {position_table}
{_code_block(code_with_lines)}
3. VISUAL ANCHOR SYSTEM (right side only; same contract as the code generator)
- Safe canvas: x in [-2.15, 6.60], y in [-2.75, 2.60]. Left column = lecture captions, top strip = title, bottom strip = subtitles.
- 6x6 grid, one cell is about 1.45 wide x 0.9 tall. CORE ZONE = B2..E5 (main content). EDGE RING = row A, row F, column 1, column 6 (small badges / short labels only, scale <= 0.85).
```
lecture |  A1  A2  A3  A4  A5  A6
        |  B1  B2  B3  B4  B5  B6
        |  C1  C2  C3  C4  C5  C6
        |  D1  D2  D3  D4  D5  D6
        |  E1  E2  E3  E4  E5  E6
        |  F1  F2  F3  F4  F5  F6
```
- Point placement (single objects, one-word labels): self.place_at_grid(obj, 'B2', scale_factor=0.8)
- Area placement (diagrams, tables, formulas, anything wider than one cell): self.place_in_area(obj, 'A1', 'C3', scale_factor=0.7)

4. LAYOUT CHECKLIST (check ALL, report in this priority order)
 a. Obstruction: any visual covering or touching the left captions, title or bottom subtitles.
 b. Off-screen / clipped: anything cut by a frame edge or beyond the safe canvas (especially long labels, wide formulas and column 6).
 c. Overlap: objects, labels, arrows or formulas drawn on top of each other; labels sitting on top of lines; text on a busy background.
 d. Stale elements: anything that should have faded out before the next lecture line but is still visible.
 e. Readability: text that is too small (labels under about 18 pt effective, formulas under about 22 pt), low contrast against black, more than 4 visual units at once.
 f. Space use: content that fills less than about 40 percent of the core zone while the text is small, or content pushed into one corner (enlarge / re-centre it).
 g. Proximity: each label within 1 grid unit of the object it describes.

5. MANDATORY CONSTRAINTS FOR YOUR SOLUTIONS
- Allowed fixes: change place_at_grid / place_in_area cells and scale_factor, change font_size, add or move a FadeOut (keep the total duration of each lecture line unchanged: shorten the following wait by the same amount), reorder objects, change colors (give hexadecimal codes, light colors only).
- Not allowed: changing text content, numbers or formulas; adding or removing explanatory content; any animation on lecture lines other than color changes; changing the size or position of the lecture lines or the title; images or other external assets (this pipeline uses native Manim shapes only).
- Solutions must name concrete grid cells and a scale_factor. Solutions must not collide with each other or with objects that remain on screen.

6. OUTPUT: ONLY this JSON structure (no markdown fences, no commentary)
{json.dumps(example, indent=4)}
- If the layout is already good: {{"layout": {{"has_issues": false, "improvements": []}}}}
- Report at most 3 problems: the ones that hurt the viewing experience most. One entry per root cause.
- Do not give video timestamps; identify problems by object and line number.
- "solution" must start with "Line X:" (the real code line) when code is provided, followed by the exact change.
- Concise problem descriptions, precise and actionable solutions.
"""


# ----------------------------------------------------------------------------
# 2. OPTIONAL content feedback (second MLLM pass: conceptual correctness and sync)
# ----------------------------------------------------------------------------
def get_prompt4_content_feedback(section, code_with_lines=None):
    verified = getattr(section, "verified_values", None) or []
    verified_text = "\n".join(f"- {v}" for v in verified) if verified else "(none provided)"
    animations = getattr(section, "animations", None) or []
    animation_text = "\n".join(f"- {a}" for a in animations) if animations else "(none provided)"

    example = {
        "content": {
            "has_issues": True,
            "improvements": [
                {
                    "severity": "high | medium | low",
                    "problem": "What is wrong (e.g. displayed value differs from narration, wrong formula, highlight out of sync)",
                    "solution": "Line X: the exact corrected text / value / highlight order",
                    "line_number": 57,
                    "object_affected": "variable name or visual description",
                }
            ],
        }
    }

    return f"""
1. TASK
Review this Manim educational video ONLY for conceptual correctness and narration-visual synchronization. Ignore pure layout issues.

2. CONTEXT
- Title: {section.title}
- Lecture lines (spoken): {'; '.join(section.lecture_lines)}
- Planned animation steps:
{animation_text}
- Verified values that must appear exactly:
{verified_text}
{_code_block(code_with_lines)}
3. CHECKLIST (priority order)
 a. Mathematical / factual errors in anything displayed: formulas, numbers, units, labels, arrows implying a wrong relationship.
 b. Mismatch between what is spoken and what is shown (different numbers, different symbols, a concept shown that is not being explained).
 c. Caption highlight out of sync: the highlighted caption must belong to the idea currently shown.
 d. A visual that is misleading (e.g. a picture that suggests a false property) or an analogy shown without its stated limit.
 e. Missing visual support: a lecture line with no visual at all, or a step that was planned but is not visible.

4. RULES
- Fix by changing displayed text, numbers, labels, colors or the order of highlights. Do not change the narration, and do not change positions or sizes (another pass handles layout).
- Report at most 3 issues, most serious first. Use real code line numbers when code is provided; otherwise line_number = null.

5. OUTPUT: ONLY this JSON structure (no markdown fences, no commentary)
{json.dumps(example, indent=4)}
- If everything is correct: {{"content": {{"has_issues": false, "improvements": []}}}}
"""


# ----------------------------------------------------------------------------
# 3. Turn feedback into a prompt section and apply it to the code
# ----------------------------------------------------------------------------
def _format_improvement(item):
    if isinstance(item, dict):
        line = item.get("line_number")
        where = f"Line {line}" if line not in (None, "", "X") else "Line unknown"
        obj = item.get("object_affected") or "unspecified object"
        severity = item.get("severity")
        sev = f" ({severity})" if severity else ""
        return f"[{where} | {obj}]{sev} Problem: {item.get('problem', '')} | Fix: {item.get('solution', '')}"
    return str(item)


def get_feedback_list_prefix(feedback_improvements):
    items = "\n".join(f"- {_format_improvement(imp)}" for imp in feedback_improvements)
    return f"""
MLLM FEEDBACK IMPROVEMENTS: based on analysis of the rendered video, address exactly these issues:
{items}

Focus only on these points:
- animations correspond to the lecture content
- clarity and readability of the animations
- positioning, scale and alignment
- visual hierarchy and focus
"""


def get_feedback_improve_code(feedback, code):
    return f"""
You are a Manim Community v{MANIM_VERSION} educational animation expert. Improve the current code according to the feedback.

MUST KEEP (MANDATORY):
- Apply the feedback with MINIMAL edits. Leave every part of the code that the feedback does not mention exactly as it is.
- Keep the Scene class name, the TeachingScene base class, self.setup_layout(...) and the order of the "# === Animation for Lecture Line N ===" blocks.
- Do not change narration, captions, displayed numbers, formulas or text content (unless the feedback explicitly corrects a wrong value).
- Do not apply any animation to the lecture lines except color changes; their size and position must remain unchanged. Keep every self.lecture[i] highlight call.
- Timing: the total duration of each lecture line block must not change. If you add a FadeOut or an animation, shorten the following self.wait(...) by the same amount.
- Positioning only with self.place_at_grid(...) / self.place_in_area(...). No .to_edge(...), no absolute coordinates, nothing in the left column, title strip or subtitle strip. Safe canvas: x in [-2.15, 6.60], y in [-2.75, 2.60].
- Use light colors on the black background (hex strings). No emoji, no ImageMobject / SVGMobject / external files.
- Objects must be created before they are used; every object that is faded out must have been added earlier; never call self.play() with no animations.
- Output only the updated, complete Python code in ONE ```python block. No explanation.

Feedback:
{feedback}

---

Current Code:
```python
{code}
```
"""