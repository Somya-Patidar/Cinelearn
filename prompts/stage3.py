import inspect
import json

MANIM_VERSION = "0.18.1"

# One accent color per lecture line (cycled). Hex literals are used on purpose: named constants
# such as GOLD_ACCENT only work if the environment defines them, and a NameError wastes a retry.
LINE_COLORS = ["#58C4DD", "#FFD700", "#83C167", "#FFA07A", "#FF6B6B", "#C9A0FF", "#5CD0B3"]
READ_COLOR = "#9A9A9A"  # caption color after its line has been spoken


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _norm_level(level):
    norm = (level or "intermediate").lower()
    return norm if norm in ("beginner", "intermediate", "advanced") else "intermediate"


def _clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split()).replace('"', "'")


def _source_of(base_class):
    """`base_class` may be source text or the class itself. The model MUST see the real API."""
    if base_class is None:
        return ""
    if isinstance(base_class, str):
        return base_class
    try:
        return inspect.getsource(base_class)
    except (OSError, TypeError):
        return ""


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except TypeError:
        return str(value)


def _format_resolved_actions(resolved_actions):
    """Formats structured resolved visual actions for Stage 3 code generation prompt."""
    if not resolved_actions:
        return ""
    lines = [
        "### Validated Visual Actions & Operations (STAGE 3 AUTHORITATIVE CONTRACT)",
        "The following structured visual operations have been pre-validated against registered visual components.",
        "Implement these exact operations in your Manim scene. Do not invent unsupported methods:\n",
    ]
    for act in resolved_actions:
        if isinstance(act, dict):
            tag = act.get("step_tag", "")
            eff = act.get("effect", "")
            op_comp = act.get("operating_component") or act.get("component_name", "")
            inst = act.get("target_instance_id") or (op_comp.lower() if op_comp else "unit")
            action_name = act.get("canonical_action", "")
            call_code = act.get("intended_call", "")
            vp = act.get("viewport", "dock_main")
            subject = act.get("subject_entity")
            trace = act.get("trace_ref")
            expected = act.get("expected_visual_result")
            kwargs = act.get("kwargs", {})
            param_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items()) if kwargs else "none"

            item_lines = [f"- **[{tag}] {eff}** in `{vp}`:"]
            item_lines.append(f"  * Operating Component: `{op_comp}` (Target Instance: `{inst}`)")
            inst_code = act.get("instantiation_code")
            placement_call = act.get("placement_call")
            if inst_code:
                item_lines.append(f"  * Recommended Instantiation: `{inst_code}`")
            if placement_call:
                item_lines.append(f"  * Viewport Docking: `{placement_call}`")
            if subject:
                item_lines.append(f"  * Subject Entity: `{subject}`")
            item_lines.append(f"  * Validated Operation: `{action_name}({param_str})`")
            if call_code:
                item_lines.append(f"  * Concrete Code Call: `{call_code}`")
            if expected:
                item_lines.append(f"  * Expected Visual Result: {expected}")
            if trace:
                item_lines.append(f"  * Authoritative Trace Reference: {trace}")
            lines.append("\n".join(item_lines))
        else:
            lines.append(f"- {act}")
    return "\n\n".join(lines)


# ----------------------------------------------------------------------------
# Static rule blocks (raw strings: no f-string brace escaping, backslashes stay literal)
# ----------------------------------------------------------------------------
def _format_component_manifest(manifest):
    if not manifest:
        return r"""
## Available Visual Architecture & Registry Components
No domain-specific components are currently registered in VisualRegistry.
Follow the 4-Tier Fallback Hierarchy below:
- Tier 1: Registered Domain Component (none registered currently)
- Tier 2: Multi-Component Composition across disjoint viewports (dock_main + dock_inspector)
- Tier 3: ProceduralComponent (wrap custom native Manim groups: Circle, Rectangle, Arrow, Axes, MathTex)
- Tier 4: Structured Conceptual Card (formal definition, equation card, or invariant; NEVER verbatim narration dumping)
"""
    lines = ["## Available Visual Architecture & Registry Components"]
    lines.append("The following domain visual components are registered in VisualRegistry and available for instantiation:\n")
    for domain, comps in manifest.items():
        lines.append(f"### Domain: {domain.upper()}")
        for comp in comps:
            cid = comp.get("component_id", "Component")
            desc = comp.get("description", "").strip()
            params = comp.get("init_params", {})
            actions = comp.get("semantic_actions", [])
            param_str = ", ".join(f"{k}={'required' if v.get('required') else v.get('default')}" for k, v in params.items()) if params else "none"
            action_str = ", ".join(actions) if actions else "standard transforms"
            lines.append(f"- **{cid}** (Domain: `{domain}`): {desc}")
            lines.append(f"  * Directly Available: `{cid}` is directly exported in the scene namespace! Call `{cid}(...)` directly.")
            lines.append(f"  * Or via Registry: `from src.domains import VisualRegistry; {cid} = VisualRegistry.get('{domain}', '{cid}')`")
            lines.append(f"  * Parameters: {param_str}")
            lines.append(f"  * Semantic Actions: {action_str}")
    lines.append("\n### CRITICAL COMPONENT USAGE RULES:")
    lines.append("1. GraphVisual vs native Graph:")
    lines.append("   - ALWAYS use `GraphVisual(vertices=[...], edges=[...])` for graphs, networks, and algorithms.")
    lines.append("   - DO NOT use Manim's native `Graph(...)`. Native Graph only accepts 2-tuples and crashes on weighted edges.")
    lines.append("   - GraphVisual supports 3-tuples for weighted edges: `edges=[('A', 'B', 4), ('B', 'C', 2)]`.")
    lines.append("2. Tables, Matrices and Arrays:")
    lines.append("   - For DP tables, distance arrays, or lookup matrices, use `DPTableMatrix(matrix_values=[...], col_labels=[...], row_labels=[...])`.")
    lines.append("   - DO NOT invent nonexistent methods like `ProceduralComponent.get_cell(...)`! ProceduralComponent is a raw wrapper without table methods. DPTableMatrix has `animate_update_cell`, `animate_highlight_cell`, etc.")
    lines.append("3. Mathematics & Formulas:")
    lines.append("   - NEVER use double curly braces `{{` and `}}` in MathTex strings! Double braces trigger Manim substring splitting and crash LaTeX.")
    lines.append("   - Use single braces: `MathTex(r\"f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}\")` or `self.safe_math(r\"...\")`.")
    lines.append("4. Operating Systems & CPU Scheduling Components:")
    lines.append("   - For OS scheduling topics, ALWAYS use the registered visual components:")
    lines.append("     * `ProcessCardVisual(pid, arrival_time, burst_time, state)`: Process Control Block (PCB) card with lifecycle badges.")
    lines.append("     * `ReadyQueueVisual(capacity=5)`: FIFO queue container with `animate_enqueue`, `animate_dequeue`, `animate_peek`.")
    lines.append("     * `CPUExecutionVisual(...)`: CPU core execution box with `animate_dispatch`, `animate_execute_tick`, `animate_complete`, `animate_set_idle`.")
    lines.append("     * `GanttChartVisual(total_time=...)`: Proportional timeline intervals with `animate_add_block` and `animate_update_time_marker`.")
    lines.append("     * `MetricTableVisual(processes=[...])`: Accounting matrix (CT, TAT, WT) without premature metric leakage, with `animate_update_process_metrics` and `animate_show_averages`.")
    lines.append("   - Use `OSTraceVisualAdapter` to compute step deltas and map `ComputationTrace` / `StateSnapshot` states directly to component properties.")
    lines.append("5. ProceduralComponent & Custom Diagrams:")
    lines.append("   - NEVER call `ProceduralComponent(name='...')` without passing actual submobjects! Calling `ProceduralComponent` without submobjects results in an empty placeholder.")
    lines.append("   - If creating custom diagrams, construct actual Manim shapes first: `group = VGroup(Rectangle(...), Text(...))` and pass into `ProceduralComponent(group, name='...')` or dock directly.")
    lines.append("   - When registered domain components exist for the concept (such as PCB cards, queues, CPU cores, Gantt charts, metric tables), ALWAYS prefer instantiating the registered domain component over raw shapes.")
    lines.append("\n### 4-Tier Fallback Hierarchy:")
    lines.append("- Tier 1 (Registered): Instantiate registered component directly (`GraphVisual`, `DPTableMatrix`, `ProcessCardVisual`, `ReadyQueueVisual`, `CPUExecutionVisual`, `GanttChartVisual`, `MetricTableVisual`, `MoleculeVisual`, `ReactionEnergyDiagram`, `FunctionCurve`, `NumberLineVisual`).")
    lines.append("- Tier 2 (Composition): Compose multiple components across disjoint viewports (self.dock_main for primary, self.dock_inspector for secondary).")
    lines.append("- Tier 3 (ProceduralComponent): Wrap custom Manim visual groups in ProceduralComponent(group, name='...') and dock it.")
    lines.append("- Tier 4 (Structured Conceptual Card): Use VGroup(RoundedRectangle(...), MathTex(...)) for formal theorems or governing equations.")
    return "\n".join(lines)


LAYOUT_RULES = r"""
2. LAYOUT & VIEWPORT CONTRACT
- Safe active canvas: x in [-2.15, 6.60], y in [-2.65, 2.55]. Nothing may leave it.
  * Left column (x < -2.15): lecture captions only (managed by TeachingScene).
  * Top strip (y > 2.60): section title only.
  * Bottom strip (y < -2.75): spoken subtitles only.
- CONTAINER DOCKING (RECOMMENDED FOR ALL TOP-LEVEL VISUALS):
  * self.dock_main(mobject, auto_scale=True): Docks primary visual into main canvas (Y in [-1.55, 2.55], X in [-2.15, 6.60]). Ideal for primary diagrams, ProcessCardVisuals, or full tabular matrices like MetricTableVisual.
  * self.dock_inspector(table_or_card, auto_scale=True): Docks status table, formula, or secondary visual into bottom inspector zone (Y in [-2.65, -1.65], X in [-2.15, 6.60]). Ideal for GanttChartVisual timeline. Strictly above subtitles.
  * NEVER dock both MetricTableVisual AND GanttChartVisual to dock_inspector simultaneously! Dock MetricTableVisual in dock_main and GanttChartVisual in dock_inspector so they are completely non-overlapping.
  * self.dock_split_left(mob_a) & self.dock_split_right(mob_b): Places two visuals side-by-side for comparative explanations (e.g. ReadyQueueVisual in dock_split_left, CPUExecutionVisual in dock_split_right).
  * self.dock_full(large_diagram): Occupies entire active visual canvas.
  * CRITICAL DOCKING SYNTAX:
    - `self.dock_main` and `self.dock_inspector` are METHODS, not mobject instances!
    - CORRECT: `self.dock_main(diagram)` or `self.dock_inspector(table)`
    - WRONG: `diagram.move_to(self.dock_inspector.get_center())` -> DO NOT call .get_center() on a method! Simply call `self.dock_inspector(diagram)`.
    - If you need the center coordinates of a viewport directly, use `self.layout.viewports["inspector"].center`.
- LEGACY 6x6 GRID PLACEMENT (SUPPORTED FOR LOW-LEVEL SUB-ELEMENTS):
  * Rows A-F (top to bottom), columns 1-6 (left to right). One cell is ~1.45 wide x ~0.9 tall.
  * self.place_at_grid(obj, 'C3', scale_factor=0.85) (a single point).
  * self.place_in_area(obj, 'B2', 'E5', scale_factor=0.9) (a bounded area).
  * Row F bottom clamping strictly protects subtitles (Y >= -2.65).
- Capacity: at most 4 visual units visible at once.
- ADAPTIVE TEXT SIZING: Text font_size <= 22 (title: 24, header: 22, body: 18, data: 16).
- MathTex formulas: scale_to_fit_width(5.8) if wider than safe zone.

3. POSITIONING & ANIMATION RULES
- Immediate placement: Every top-level mobject MUST be docked (via self.dock_main, self.dock_inspector, self.dock_split_*, or self.place_in_area) BEFORE calling self.play(...).
- Clean transitions: When moving to a new concept or lecture line, fade out obsolete objects explicitly (FadeOut(prev_unit) or clear_canvas()).
- Colors: background is #000000. Use only high-contrast hex colors (#58C4DD, #FFD700, #83C167, #FFA07A, #FF6B6B, WHITE).
- ANTI-TEXT-CARD RULE: NEVER put full lecture sentences in Text(...) cards. Use visual diagrams, curves, molecular models, state tables, or MathTex equations.
"""

DIAGRAM_RECIPES = r"""
8. MANIM VISUAL RECIPES (domain-independent patterns)
A. Discrete Structure / Network Diagram (nodes, connections, labels):
```python
local_pos = {"A": [-2.2, 0.8, 0], "B": [0.0, 1.2, 0], "C": [-1.0, -0.9, 0], "D": [2.2, -0.1, 0]}
nodes = {name: VGroup(Circle(radius=0.35, color="#58C4DD", fill_opacity=0.3), Text(name, font_size=20)).move_to(np.array(p)) for name, p in local_pos.items()}
edges = {("A", "B"): Line(nodes["A"][0].get_center(), nodes["B"][0].get_center(), color="#83C167", buff=0.35)}
diagram = VGroup(*nodes.values(), *edges.values())
self.dock_main(diagram)
self.play(FadeIn(diagram), run_time=1.2)
```

B. Moving Agent / Focus Indicator Token:
```python
token = VGroup(Circle(radius=0.15, color="#FFD700", fill_opacity=0.95), Dot(radius=0.04, color=BLACK))
token.move_to(nodes["A"][0])
self.play(FadeIn(token), run_time=0.4)
self.play(token.animate.move_to(nodes["B"][0]), run_time=1.0)
```

C. State Table / Inspection Matrix (Docked to Inspector Zone):
```python
# 2D status table or memoization array docked to bottom inspector
matrix_data = [["Node", "A", "B", "C"], ["Dist", "0", "4", "inf"]]
table = self.create_table_matrix(matrix_data, cell_size=0.6, font_size=16)
self.dock_inspector(table)
self.play(FadeIn(table), run_time=0.8)
```

D. Mathematical Functions & Plotted Curves (Coordinate Geometry):
```python
axes = Axes(x_range=[0, 10, 2], y_range=[0, 100, 20], x_length=5.0, y_length=3.0, tips=False)
curve = axes.plot(lambda x: x**2, color="#58C4DD")
curve_label = MathTex(r"f(x) = x^2", font_size=20).next_to(curve.get_end(), RIGHT, buff=0.15)
plot_group = VGroup(axes, curve, curve_label)
self.dock_main(plot_group)
self.play(Create(axes), Create(curve), Write(curve_label), run_time=1.4)
```

E. Scientific / Chemical / Flow Diagram (ProceduralComponent Fallback):
```python
# Procedural representation for scientific, physical, or biological concepts
mol_a = VGroup(Circle(radius=0.4, color="#FF6B6B", fill_opacity=0.5), Text("A", font_size=18)).shift(LEFT * 1.5)
arrow = Arrow(LEFT * 0.8, RIGHT * 0.8, color="#FFD700")
mol_b = VGroup(Circle(radius=0.4, color="#83C167", fill_opacity=0.5), Text("B", font_size=18)).shift(RIGHT * 1.5)
flow_diagram = ProceduralComponent(VGroup(mol_a, arrow, mol_b), name="ReactionFlow")
self.dock_main(flow_diagram)
self.play(FadeIn(flow_diagram), run_time=1.0)
```

F. Structured Conceptual Card (Formal Theorems & Governing Invariants):
```python
box = RoundedRectangle(corner_radius=0.15, width=5.5, height=1.6, color="#58C4DD", fill_opacity=0.15)
title = Text("First-Order Rate Law", font_size=18, color="#FFD700").next_to(box.get_top(), DOWN, buff=0.15)
eq = MathTex(r"[A]_t = [A]_0 \, e^{-k t}", font_size=24, color=WHITE).move_to(box.get_center())
card = VGroup(box, title, eq)
self.dock_main(card)
self.play(FadeIn(card), Write(eq), run_time=1.0)
```

G. Operating Systems & Scheduling Registered Components Usage:
```python
# 1. PCB Card (Process Attributes & State)
p1 = ProcessCardVisual(pid="P1", arrival_time=0, burst_time=4, state="READY")
self.dock_split_left(p1)

# 2. Ready Queue (FIFO Operations)
queue = ReadyQueueVisual(capacity=4)
self.dock_split_left(queue)
self.play(queue.animate_enqueue(p1), run_time=1.0)

# 3. CPU Core (Process Dispatch & Execution)
cpu = CPUExecutionVisual()
self.dock_split_right(cpu)
self.play(cpu.animate_dispatch(p1), run_time=1.0)
self.play(cpu.animate_execute_tick(time_val=1), run_time=0.8)
self.play(cpu.animate_complete(p1), run_time=0.8)

# 4. Gantt Chart Timeline (Dock to Inspector Zone)
gantt = GanttChartVisual(total_time=10)
self.dock_inspector(gantt)
self.play(gantt.animate_add_block("P1", start=0, duration=4), run_time=1.2)

# 5. Metric Table (Accounting Matrix - Dock to Main Canvas)
processes = [
    {"pid": "P1", "arrival_time": 0, "burst_time": 4, "completion_time": 4, "turnaround_time": 4, "waiting_time": 0},
    {"pid": "P2", "arrival_time": 1, "burst_time": 3, "completion_time": 7, "turnaround_time": 6, "waiting_time": 3},
]
table = MetricTableVisual(processes=processes)
self.dock_main(table)
self.play(FadeIn(table), run_time=1.0)
self.play(table.animate_update_process_metrics(0, ct=4, tat=4, wt=0), run_time=0.8)
self.play(table.animate_show_averages(avg_tat=5.0, avg_wt=1.5), run_time=1.0)
```

H. Process Lifecycle State-Machine Diagram (Custom Educational Visual):
```python
# Circles with state names and connecting transition arrows
s_new = VGroup(Circle(radius=0.42, color="#58C4DD", fill_opacity=0.25), Text("NEW", font_size=15, color="#58C4DD"))
s_ready = VGroup(Circle(radius=0.42, color="#FFD700", fill_opacity=0.25), Text("READY", font_size=15, color="#FFD700"))
s_run = VGroup(Circle(radius=0.42, color="#83C167", fill_opacity=0.25), Text("RUNNING", font_size=14, color="#83C167"))
s_term = VGroup(Circle(radius=0.42, color="#FF6B6B", fill_opacity=0.25), Text("DONE", font_size=15, color="#FF6B6B"))

s_new.move_to(np.array([-1.8, 0.8, 0]))
s_ready.move_to(np.array([0.6, 0.8, 0]))
s_run.move_to(np.array([3.0, 0.8, 0]))
s_term.move_to(np.array([3.0, -1.0, 0]))

a1 = Arrow(s_new.get_right(), s_ready.get_left(), buff=0.1, color="#58C4DD")
a2 = Arrow(s_ready.get_right(), s_run.get_left(), buff=0.1, color="#FFD700")
a3 = Arrow(s_run.get_bottom(), s_term.get_top(), buff=0.1, color="#83C167")
lbl1 = Text("Admitted", font_size=12, color=WHITE).next_to(a1, UP, buff=0.05)
lbl2 = Text("Dispatched", font_size=12, color=WHITE).next_to(a2, UP, buff=0.05)
lbl3 = Text("Exit", font_size=12, color=WHITE).next_to(a3, RIGHT, buff=0.05)

lifecycle_diagram = VGroup(s_new, s_ready, s_run, s_term, a1, a2, a3, lbl1, lbl2, lbl3)
self.dock_main(lifecycle_diagram)
self.play(FadeIn(lifecycle_diagram), run_time=1.2)
```

I. Role & Flow Comparison (e.g. Scheduler vs Dispatcher, Multiprogramming):
```python
card_sched = VGroup(
    RoundedRectangle(corner_radius=0.1, width=2.7, height=1.6, color="#58C4DD", fill_opacity=0.2),
    Text("Short-Term Scheduler", font_size=16, color="#58C4DD").shift(UP * 0.45),
    Text("Decision Maker\nSelects next ready process", font_size=13, color=WHITE).shift(DOWN * 0.2)
)
arr_flow = Arrow(LEFT * 0.5, RIGHT * 0.5, color="#FFD700")
card_disp = VGroup(
    RoundedRectangle(corner_radius=0.1, width=2.7, height=1.6, color="#83C167", fill_opacity=0.2),
    Text("Dispatcher", font_size=16, color="#83C167").shift(UP * 0.45),
    Text("Execution Hand-off\nContext switch & CPU jump", font_size=13, color=WHITE).shift(DOWN * 0.2)
)
comparison = VGroup(card_sched, arr_flow, card_disp).arrange(RIGHT, buff=0.35)
self.dock_main(comparison)
self.play(FadeIn(comparison), run_time=1.2)
```

J. Mathematical Formulas & Derivations (Stacked in Main Canvas):
```python
# Multi-line formulas MUST be arranged in a single VGroup and docked in dock_main (NEVER dock_inspector!)
f1 = MathTex(r"\text{Turnaround Time (TAT)} = \text{Completion Time (CT)} - \text{Arrival Time (AT)}", font_size=18, color="#58C4DD")
f2 = MathTex(r"\text{Waiting Time (WT)} = \text{Turnaround Time (TAT)} - \text{Burst Time (BT)}", font_size=18, color="#FFD700")
f3 = MathTex(r"\text{Average Waiting Time} = \frac{1}{N} \sum_{i=1}^N \text{WT}_i", font_size=18, color="#83C167")
formulas = VGroup(f1, f2, f3).arrange(DOWN, buff=0.35)
self.dock_main(formulas)
self.play(FadeIn(formulas), run_time=1.2)
```

K. Convoy Effect Illustration (Long Burst vs Short Burst):
```python
# Visual diagram contrasting the convoy head and starved tail
header = Text("Convoy Effect: Long Burst Blocks Short Jobs", font_size=18, color="#FF6B6B")
p1_bar = VGroup(Rectangle(width=4.2, height=0.7, color="#FF6B6B", fill_opacity=0.35), Text("P1 (Burst = 24)", font_size=16, color=WHITE))
p2_bar = VGroup(Rectangle(width=0.9, height=0.7, color="#58C4DD", fill_opacity=0.35), Text("P2 (3)", font_size=14, color=WHITE))
p3_bar = VGroup(Rectangle(width=0.9, height=0.7, color="#83C167", fill_opacity=0.35), Text("P3 (3)", font_size=14, color=WHITE))
bars = VGroup(p1_bar, p2_bar, p3_bar).arrange(RIGHT, buff=0.08)
caption_note = Text("P2 and P3 wait 24 time units despite requiring only 3 units each!", font_size=14, color="#FFA07A")
convoy_group = VGroup(header, bars, caption_note).arrange(DOWN, buff=0.32)
self.dock_main(convoy_group)
self.play(FadeIn(convoy_group), run_time=1.2)
```

L. Multiprogramming Motivation Diagram (CPU Core with Competing Tasks):
```python
# CPU Core on right, competing task cards on left with connection arrow:
cpu_box = VGroup(
    RoundedRectangle(corner_radius=0.15, width=2.6, height=2.2, color="#58C4DD", fill_opacity=0.2),
    Text("CPU Core", font_size=18, color="#58C4DD").shift(UP * 0.7),
    Text("Active Unit\n1 Task at a time", font_size=13, color=WHITE).shift(DOWN * 0.1)
)
task1 = VGroup(RoundedRectangle(corner_radius=0.08, width=1.5, height=0.55, color="#FFA07A", fill_opacity=0.2), Text("Task 1", font_size=13, color="#FFA07A"))
task2 = VGroup(RoundedRectangle(corner_radius=0.08, width=1.5, height=0.55, color="#FFD700", fill_opacity=0.2), Text("Task 2", font_size=13, color="#FFD700"))
task3 = VGroup(RoundedRectangle(corner_radius=0.08, width=1.5, height=0.55, color="#83C167", fill_opacity=0.2), Text("Task 3", font_size=13, color="#83C167"))
tasks = VGroup(task1, task2, task3).arrange(DOWN, buff=0.18)
label_compete = Text("Multiple Tasks Compete", font_size=14, color=WHITE).next_to(tasks, UP, buff=0.15)
arrow_compete = Arrow(tasks.get_right(), cpu_box.get_left(), buff=0.15, color="#FFD700")
task_group = VGroup(tasks, label_compete)
motivation_diagram = VGroup(task_group, arrow_compete, cpu_box).arrange(RIGHT, buff=0.45)
self.dock_main(motivation_diagram)
self.play(FadeIn(motivation_diagram), run_time=1.2)
```

### CRITICAL RULE FOR CUSTOM DIAGRAMS (NO EMPTY ProceduralComponent):
- NEVER write `ProceduralComponent(name="...")` or `ProceduralComponent(VGroup(), name="...")`!
  Calling ProceduralComponent without real shapes produces an empty placeholder card that FAILS visual validation!
- For custom diagrams (Multiprogramming, State Machine, Scheduler vs Dispatcher, Formulas, Convoy Effect):
  ALWAYS construct the actual Manim shapes directly (VGroups of RoundedRectangles, Circles, Arrows, and Text as shown in Recipes H, I, J, K, L) and pass them to self.dock_main(diagram)!
- Write(...) does NOT accept a `shift` parameter: use `Write(mob)` or `FadeIn(mob, shift=UP * 0.2)`.
"""

MANIM_PITFALLS = rf"""
9. MANIM COMMUNITY v{MANIM_VERSION} RULES (the installed version - do not use newer APIs)
- Start with `from manim import *`. `np` is available. No other imports.
- Create = `Create`, never `ShowCreation`. Use `FadeIn(x, shift=UP * 0.2)`, `GrowFromCenter`, `Write`, `Indicate`, `Circumscribe`, `SurroundingRectangle`, `Brace`.
- Text(...) is for plain words and numbers with units. MathTex(r"...") is for mathematics (raw strings; words inside use \text{{...}}). NEVER put emoji, arrows like ->, or non-Latin symbols in Text or MathTex: they render as blanks or crash. Use Arrow(...) mobjects or MathTex(r"\rightarrow") instead.
- Native LaTeX rendering is fully supported for MathTex, Tex, and NumberLine(include_numbers=True).
- NEVER use ImageMobject, SVGMobject, Code, ThreeDScene, Surface, external files, or fonts.
- Every self.play(...) receives Animation objects (or mob.animate...), never a bare Mobject, and never an empty argument list. Do not animate the same mobject twice inside one self.play call.
- VGroup accepts only VMobjects (Text, MathTex, shapes, Lines are fine).
- Line / Arrow need start != end. Use Arrow(a, b, buff=0.1) to avoid touching shapes.
- Do not animate lecture captions except `self.lecture[i].animate.set_color(...)`. NEVER call FadeOut(self.lecture[i]), Transform(self.lecture[i], ...), or move captions.
- Prefer FadeOut / FadeIn over Transform between different kinds of objects; use ReplacementTransform only between objects of the same kind.
- Colors are hex strings (for example "#58C4DD"); BLACK and WHITE are fine; the background is already black.
- Do not hand-type computed results. Compute them in Python and format them into labels, for example:
  total = 7 + 5
  Text(f"{{total}} km", font_size=22)
  Add `assert` only for trivial arithmetic checks of the displayed numbers, never for layout or time.
"""

OUTPUT_FORMAT = r"""
11. OUTPUT FORMAT
- Return exactly ONE ```python code block with the complete file and nothing else: `from manim import *` followed by ONE Scene class that extends TeachingScene. No explanation text before or after.
- Do not redefine TeachingScene or helper functions that already exist in the API; do not add `if __name__ == "__main__"`.
- Self-check before answering: no emoji; every name used in FadeOut / animate is defined earlier; every top-level unit is placed with place_at_grid / place_in_area and animated into the scene; nothing outside the safe canvas; each lecture block ends with the wait that fills its time slot.
"""


# ----------------------------------------------------------------------------
# Level / analogy dependent blocks
# ----------------------------------------------------------------------------
LEVEL_CODE_GUIDANCE = {
    "beginner": r"""
LEVEL-SPECIFIC VISUAL STYLING (BEGINNER)
- Clean, simple geometric figures (Circle, Square, Rectangle, Arrow, Text) with large distinct fonts (labels 22-26).
- Avoid dense coordinate systems; prefer intuitive spatial arrangements and one idea per screen.
- Warm, friendly high-contrast colors from the palette.
""",
    "intermediate": r"""
LEVEL-SPECIFIC VISUAL STYLING (INTERMEDIATE)
- Combine geometric structures with standard MathTex equations.
- Use coordinate grids, tables and step-by-step transformations where they help.
- Make the visual correspondence between each algebraic label and its geometry obvious (same color).
""",
    "advanced": r"""
LEVEL-SPECIFIC VISUAL STYLING (ADVANCED)
- Precise mathematical visualization with exact MathTex notation; show invariants, edge cases and structure.
- Compact, information-dense composition, but still within the 4-unit limit and readable font sizes.
""",
}


def _analogy_code_guidance(analogy_text):
    if analogy_text:
        return rf"""
10. STORY-DRIVEN VISUAL CONTINUITY & ENTITY REPRESENTATION (analogy: {analogy_text})
- Locations / States / Nodes: Represent key locations, states, or graph nodes with structured visual components (VGroup(Circle + Text("City A"))).
- Pathways / Edges / Connections: Render connections with Arrow, Line, or registered connectors.
- Traveler / Active State / Focus: A colored token (see recipe 8B) or highlight indicator that physically moves when the narrative advances. No emoji.
- Visual Continuity Across Scenes: Keep entities identical in shape, color, and home area across all sections (see visual_conventions).
- Visualizing Analogy Boundaries & Breakdown: Where the analogy breaks down, switch to clear visual contrast (warning color, dashed outline) AND show the exact formal statement as MathTex.
- Quantities carry units ("12 km", "30 min"), computed in Python and formatted into the label.
- Temporary route highlights, cost badges and intermediate calculations are faded out before the next step or any summary table / formula.
- Every displayed number and equation must be exactly correct.
"""
    return r"""
10. SCENARIO VISUAL REPRESENTATION
- Represent entities with native Manim shapes (nodes, arrows, tables, state indicators) and keep their colors and relative positions stable across scenes.
- Fade out temporary highlights and arrows between steps.
- Every displayed equation and number must be exactly correct.
"""


def _timing_guidance(audio_meta):
    if audio_meta and "lines" in audio_meta:
        total_dur = audio_meta.get("total_duration", 35.0)
        rows = []
        for line_item in audio_meta["lines"]:
            idx = line_item.get("line_index", 0)
            speech_dur = line_item.get("duration", 4.0)
            slot_budget = line_item.get("total_budget", speech_dur + 0.5)
            rows.append(f"  * Lecture Line {idx + 1} (index {idx}): target duration = {speech_dur:.1f}s (TARGET = {slot_budget:.2f} s)")
        breakdown = "\n".join(rows)
        return f"""6. PACING & AUDIO SYNCHRONIZATION (MANDATORY - narration audio is pre-rendered, total {total_dur:.1f} s)
{breakdown}

Do NOT do timing arithmetic by hand. Use the clock so every line fills its slot exactly:
```python
# === Animation for Lecture Line 2 ===  (TARGET 6.40 s)
t0 = self.renderer.time                                  # first statement of the block, AFTER setup_layout
self.play(
    *self.activate_caption(1, "{LINE_COLORS[1]}"),        # highlights current caption, dims others
    FadeOut(prev_unit),                                   # clear the previous unit in the same beat
    run_time=0.6,
)
self.play(FadeIn(new_unit, shift=UP * 0.2), run_time=2.4) # allocate 45-65% of slot to meaningful animation
self.wait(max(0.1, 6.40 - (self.renderer.time - t0)))   # inspection hold fills the rest of the slot
# on screen after line 2: [new_unit]
```
- PEDAGOGICAL PACING RULES:
  * Allocate 45% - 65% of the line budget to core meaningful animations (run_time = 2.0s - 3.2s for a 5-6s slot). Do NOT rush complex state transitions, enqueues, dispatches, equations, or graph relaxations with a tiny 0.6s blip! Give learners time to follow visual movement.
  * Allocate 25% - 35% of the slot (1.2s - 2.0s) as a stable visual hold / inspection window (`self.wait`) so the learner can examine the resulting diagram while the voiceover finishes.
  * Use each line's own TARGET in the final wait.
  * The caption highlight is played TOGETHER with the first visual animation of its line (never alone).
  * The first block has no previous caption; the last block ends with self.wait(1.5) after its fill-wait.
"""
    return f"""6. PACING & DURATION
- Each lecture block takes about 5 to 7 seconds: caption highlight together with the first animation (run_time 0.6-1.0), further animations run_time 1.0-1.5, then self.wait(1.0-2.0).
- When moving to line i+1, return the previous caption to "{READ_COLOR}" in the same self.play call as the new highlight.
- End the scene with self.wait(2.0).
"""


# ----------------------------------------------------------------------------
# Public entry point (new parameter `visual_conventions` is optional)
# ----------------------------------------------------------------------------
def get_prompt3_code(
    regenerate_note,
    section,
    base_class,
    audio_meta=None,
    level="intermediate",
    analogy=None,
    visual_conventions=None,
    component_manifest=None,
):
    norm_level = _norm_level(level)
    analogy_text = _clean_text(analogy)

    if component_manifest is None:
        try:
            from src.domains import VisualRegistry
            component_manifest = VisualRegistry.get_schema_manifest()
        except Exception:
            component_manifest = {}

    manifest_section = _format_component_manifest(component_manifest)

    lecture_lines = list(section.lecture_lines)
    captions = list(getattr(section, "captions", None) or lecture_lines)
    if len(captions) != len(lecture_lines):
        captions = lecture_lines
    story_beat = getattr(section, "story_beat", None) or getattr(section, "story_step", "") or "N/A"
    animation_steps = "\n".join(f"  {i + 1}. {step}" for i, step in enumerate(section.animations))
    resolved_actions = getattr(section, "resolved_actions", None)
    if not resolved_actions and isinstance(section, dict):
        resolved_actions = section.get("resolved_actions")
    resolved_actions_block = f"\n{_format_resolved_actions(resolved_actions)}\n" if resolved_actions else ""
    verified_values = getattr(section, "verified_values", None) or []
    verified_text = "\n".join(f"  - {v}" for v in verified_values) if verified_values else "  (none provided: compute values in Python and keep narration and display identical)"

    class_name = f"{section.id.title().replace('_', '')}Scene"

    color_rows = "\n".join(
        f"  * Lecture line {i + 1} -> {LINE_COLORS[i % len(LINE_COLORS)]}" for i in range(len(lecture_lines))
    )

    api_source = _source_of(base_class).strip()
    if api_source:
        api_block = f"""## TeachingScene API (authoritative source - use ONLY the helpers defined here)
```python
{api_source}
```
"""
    else:
        api_block = """## TeachingScene API (source not provided - use ONLY these helpers)
- self.setup_layout(title, captions)      # creates title and left captions; fills self.lecture (list of captions)
- self.dock_main(mobject)                  # docks primary visual into main canvas viewport
- self.dock_inspector(mobject)             # docks secondary status table/equation into bottom inspector
- self.dock_split_left(mobject)            # docks into left split viewport
- self.dock_split_right(mobject)           # docks into right split viewport
- self.dock_full(mobject)                  # docks into full canvas viewport
- self.place_at_grid(obj, 'C3', scale_factor=1.0)
- self.place_in_area(obj, 'B2', 'E5', scale_factor=1.0)
Prefer plain self.play(FadeOut(...)) over helpers you cannot verify.
"""

    contract = ""
    if visual_conventions:
        contract = f"""
## VISUAL CONTINUITY CONTRACT (from the storyboard - follow exactly)
Entities that appear in several sections must keep these shapes, colors and home areas:
{_as_text(visual_conventions)}
"""

    return f"""
You are an expert Manim animator using Manim Community Edition v{MANIM_VERSION}.
Generate ONE high-quality Manim scene class from the teaching script below, for a {norm_level.capitalize()} learner.
{regenerate_note}

{manifest_section}

{api_block}
{contract}
1. BASIC REQUIREMENTS
- Use the TeachingScene base class exactly as provided. Do not modify or re-implement it.
- Lecture captions (self.lecture[i]) may ONLY change color: no scaling, moving, fading or Transform.
- Color mapping caption -> visuals: every object introduced for lecture line i uses that line's accent color (neutral white/grey is fine for structure):
{color_rows}
- Animation vocabulary: FadeIn for definitions, FadeIn(shift=...) for sequential or directional steps, GrowFromCenter for core shapes, Create / Write for lines and equations, Indicate for key formulas.
{LAYOUT_RULES}
4. TEACHING AND NARRATION CONTEXT
- Title: {section.title}
- Story Beat / Narrative Step: {story_beat}
- Lecture lines (spoken narration, index 0..{len(lecture_lines) - 1}):
{chr(10).join(f"  [{i}] {line}" for i, line in enumerate(lecture_lines))}

- Visual Intent (Storyboard Animation Steps):
{animation_steps}
{resolved_actions_block}
- Authoritative Execution Data & Verified Values:
{verified_text}

5. STRUCTURE
One block per lecture line, separated by the comment `# === Animation for Lecture Line N ===`, in lecture order. Each block: highlight caption together with the first visual animation, perform that line's animation steps, clear what the next line does not need, and end with a comment `# on screen after line N: [...]` (max 4 units).
```python
from manim import *

class {class_name}(TeachingScene):
    def construct(self):
        self.setup_layout({section.title!r}, {captions!r})

        # === Animation for Lecture Line 1 ===
        t0 = self.renderer.time
        # ... build unit(s), place them, then play highlight + first animation together ...
        # on screen after line 1: [...]

        # === Animation for Lecture Line 2 ===
        # ...
        # at the very end: fade out every remaining diagram unit (not the title or captions)
```

{_timing_guidance(audio_meta)}
7. VISUAL HYGIENE (HIGHEST PRIORITY)
- Previous temporary units must be gone before new ones appear, unless the storyboard marks them persistent. Never render a formula or table on top of a diagram: fade the diagram out first.
- Nothing orphaned: arrows, labels, boxes and highlights belong to a VGroup that is faded out together with its parent.
- Shape + label always live in one VGroup, created together (node = VGroup(Circle(...), Text("A", font_size=22))).
- Do not shrink text below the readable sizes above; if the content is too dense, simplify the visual, never the font.
- Simplicity and reliability beat decoration: no 3D, no external assets.
{DIAGRAM_RECIPES}
{MANIM_PITFALLS}
{LEVEL_CODE_GUIDANCE[norm_level]}
{_analogy_code_guidance(analogy_text)}
{OUTPUT_FORMAT}
"""


def get_regenerate_note(attempt, MAX_REGENERATE_TRIES, error_log=None, previous_code=None):
    note = f"""
**IMPORTANT NOTE:** This is attempt {attempt}/{MAX_REGENERATE_TRIES} to generate working code.
The previous attempt failed. Fix the actual cause and keep everything else.
"""
    if error_log:
        tail = str(error_log).strip()[-2500:]
        note += f"""
Error output of the previous attempt (last lines):
```
{tail}
```
"""
    if previous_code:
        note += f"""
Previous code (fix it rather than starting from scratch):
```python
{previous_code}
```
"""
    note += """
Rules for this attempt:
1. Use only basic, well-tested Manim Community v0.18.1 functions.
2. No emoji or special symbols inside Text / MathTex; MathTex only with raw strings.
3. Every variable used in FadeOut / animate must be defined before it is used; never call self.play() with no animations.
4. Keep the same layout contract, timing and content; change only what is needed to make the code run.
"""
    return note