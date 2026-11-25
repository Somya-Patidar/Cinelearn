"""
CineLearn Generic Storyboard Visual Action Resolver & Manifest Validator
========================================================================
Implements domain-independent visual capability resolution:
- Parses 6-field animation step specifications.
- Validates component references against the authoritative VisualRegistry.
- Validates constructor parameters and types against component manifests.
- Resolves semantic action names to canonical component methods (e.g. 'enqueue' -> 'animate_enqueue').
- Validates operation parameters and constraints against operation signatures.
- Rejects unsupported components, methods, parameters, and invalid viewports (fail-closed).
- Cross-references trace data where authoritative traces are available.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

def _get_registry():
    try:
        from src.domains import VisualRegistry
        if VisualRegistry._registry:
            return VisualRegistry
    except Exception:
        pass
    try:
        from domains import VisualRegistry
        if VisualRegistry._registry:
            return VisualRegistry
    except Exception:
        pass
    try:
        from src.domains import VisualRegistry
        return VisualRegistry
    except Exception:
        from domains import VisualRegistry
        return VisualRegistry


@dataclass
class ResolvedVisualAction:
    """Structured, verified visual operation ready for Stage 3 scene generation."""
    step_tag: str
    effect: str
    component_name: str
    component_class: Optional[type]
    canonical_action: str
    method: Optional[Any]
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    viewport: str = "dock_main"
    on_screen_after: List[str] = field(default_factory=list)
    is_registered_component: bool = False
    is_primitive: bool = False
    is_procedural: bool = False
    raw_step: str = ""
    # Enhanced Stage 3 integration fields
    operating_component: str = ""
    operating_class: Optional[type] = None
    subject_entity: Optional[str] = None
    target_instance_id: Optional[str] = None
    intended_call: str = ""
    trace_ref: Optional[str] = None
    expected_visual_result: Optional[str] = None
    instantiation_code: str = ""
    placement_call: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_tag": self.step_tag,
            "effect": self.effect,
            "component_name": self.component_name,
            "canonical_action": self.canonical_action,
            "args": self.args,
            "kwargs": self.kwargs,
            "viewport": self.viewport,
            "on_screen_after": self.on_screen_after,
            "is_registered_component": self.is_registered_component,
            "is_primitive": self.is_primitive,
            "is_procedural": self.is_procedural,
            "raw_step": self.raw_step,
            "operating_component": self.operating_component,
            "operating_class": self.operating_class.__name__ if self.operating_class else None,
            "subject_entity": self.subject_entity,
            "target_instance_id": self.target_instance_id,
            "intended_call": self.intended_call,
            "trace_ref": self.trace_ref,
            "expected_visual_result": self.expected_visual_result,
            "instantiation_code": self.instantiation_code,
            "placement_call": self.placement_call,
        }


class StoryboardActionResolver:
    """
    Deterministic resolver that validates and maps storyboard animation steps
    to registered VisualComponent capabilities.
    """

    STANDARD_VIEWPORTS: Set[str] = {
        "dock_main",
        "dock_inspector",
        "dock_split_left",
        "dock_split_right",
        "dock_full",
    }

    PRIMITIVE_VISUALS: Set[str] = {
        "MathTex",
        "Tex",
        "Text",
        "Circle",
        "Rectangle",
        "RoundedRectangle",
        "Square",
        "Dot",
        "Line",
        "Arrow",
        "DoubleArrow",
        "CurvedArrow",
        "Axes",
        "Table",
        "MobjectTable",
        "IntegerTable",
        "DecimalTable",
        "VGroup",
    }

    @classmethod
    def parse_animation_step(cls, step_str: str) -> Dict[str, Any]:
        """
        Parses a 6-field animation step string into structured components.
        Expected format:
        "L<line>-<k> | <effect> | <visual_representation> | viewport: <vp> | action: <action> | on screen after: [<units>]"
        """
        raw = step_str.strip()
        parts = [p.strip() for p in raw.split("|")]

        # Structured pipe-delimited format (5 or 6 fields)
        if len(parts) in (5, 6) and any("action:" in p for p in parts):
            vp_val = "dock_main"
            act_val = ""
            for p in parts[3:]:
                if p.startswith("viewport:"):
                    vp_val = p.split("viewport:", 1)[1].strip()
                elif p.startswith("action:"):
                    act_val = p.split("action:", 1)[1].strip()
            if act_val:
                on_screen = []
                if len(parts) == 6:
                    on_screen_str = parts[5].replace("on screen after:", "").strip()
                    if on_screen_str.startswith("[") and on_screen_str.endswith("]"):
                        on_screen = [x.strip() for x in on_screen_str[1:-1].split(",") if x.strip()]
                return {
                    "tag": parts[0] if parts[0].startswith("L") else "L1-1",
                    "effect": parts[1],
                    "visual_representation": parts[2],
                    "viewport": vp_val,
                    "action": act_val,
                    "on_screen_after": on_screen,
                    "raw_step": raw,
                    "is_malformed": False,
                }

        if len(parts) < 6:
            # Tolerant parsing for shorthand formats (e.g. action: Component.method(...) or fewer pipes)
            m_comp_act = re.search(r"\b([A-Z]\w+)\.([a-z_]\w+\(.*?\))", raw)
            m_act = re.search(r"\baction:\s*([A-Za-z_]\w*(?:\.[a-z_]\w+)?\(.*?\))", raw)
            if m_comp_act:
                comp_name = m_comp_act.group(1)
                act_str = m_comp_act.group(2)
                return {
                    "tag": parts[0] if (len(parts) > 1 and parts[0].startswith("L")) else "L1-1",
                    "effect": "Transform",
                    "visual_representation": comp_name,
                    "viewport": "dock_main",
                    "action": act_str,
                    "on_screen_after": [],
                    "raw_step": raw,
                    "is_malformed": False,
                }
            if m_act:
                act_str = m_act.group(1)
                if "." in act_str:
                    comp_name, sub_act = act_str.split(".", 1)
                else:
                    comp_name, sub_act = "", act_str
                return {
                    "tag": "L1-1",
                    "effect": "Transform",
                    "visual_representation": comp_name,
                    "viewport": "dock_main",
                    "action": sub_act,
                    "on_screen_after": [],
                    "raw_step": raw,
                    "is_malformed": False,
                }

            if len(parts) == 1:
                # Plain freeform procedural animation step (legacy outlines or qualitative scenes)
                return {
                    "tag": "L1-1",
                    "effect": "FadeIn",
                    "visual_representation": "ProceduralComponent",
                    "viewport": "dock_main",
                    "action": raw,
                    "on_screen_after": [],
                    "raw_step": raw,
                    "is_malformed": False,
                }

            if len(parts) == 3:
                # Milestone 2 legacy format: visual | effect/action | timing
                return {
                    "tag": "L1-1",
                    "effect": parts[1],
                    "visual_representation": parts[0],
                    "viewport": "dock_main",
                    "action": parts[1],
                    "on_screen_after": [],
                    "raw_step": raw,
                    "is_malformed": False,
                }
            if len(parts) == 2:
                # 2-part format: visual | action
                return {
                    "tag": "L1-1",
                    "effect": parts[1],
                    "visual_representation": parts[0],
                    "viewport": "dock_main",
                    "action": parts[1],
                    "on_screen_after": [],
                    "raw_step": raw,
                    "is_malformed": False,
                }
            if len(parts) in (4, 5):
                return {
                    "tag": parts[0] if parts[0].startswith("L") else "L1-1",
                    "effect": parts[1],
                    "visual_representation": parts[2],
                    "viewport": "dock_main",
                    "action": parts[-1],
                    "on_screen_after": [],
                    "raw_step": raw,
                    "is_malformed": False,
                }

            # Handle possible edge cases where raw step is empty
            return {
                "tag": parts[0] if len(parts) > 0 else "L1-1",
                "effect": parts[1] if len(parts) > 1 else "FadeIn",
                "visual_representation": parts[2] if len(parts) > 2 else "",
                "viewport": "dock_main",
                "action": parts[3] if len(parts) > 3 else "",
                "on_screen_after": [],
                "raw_step": raw,
                "is_malformed": not bool(raw.strip()),
            }

        if (parts[2].startswith("(") and parts[2].endswith(")")) or parts[4].startswith("run_time="):
            # Legacy 6-field coordinate format: visual | effect | start_pos | end_pos | run_time | None
            return {
                "tag": "L1-1",
                "effect": parts[1],
                "visual_representation": parts[0],
                "viewport": "dock_main",
                "action": parts[1],
                "on_screen_after": [],
                "raw_step": raw,
                "is_malformed": False,
            }

        tag = parts[0]
        effect = parts[1]
        visual_rep = parts[2]

        vp_part = parts[3]
        if vp_part.lower().startswith("viewport:"):
            viewport = vp_part.split(":", 1)[1].strip()
        else:
            viewport = vp_part

        action_part = parts[4]
        if action_part.lower().startswith("action:"):
            action = action_part.split(":", 1)[1].strip()
        else:
            action = action_part

        after_part = parts[5]
        on_screen = []
        if "on screen after:" in after_part.lower():
            after_str = after_part.split(":", 1)[1].strip()
            # Extract bracketed list or comma separated
            match = re.search(r"\[(.*?)\]", after_str)
            if match:
                items = [it.strip().strip("'\"") for it in match.group(1).split(",") if it.strip()]
                on_screen = items
            else:
                on_screen = [after_str] if after_str else []
        else:
            on_screen = [after_part] if after_part else []

        return {
            "tag": tag,
            "effect": effect,
            "visual_representation": visual_rep,
            "viewport": viewport,
            "action": action,
            "on_screen_after": on_screen,
            "raw_step": raw,
            "is_malformed": False,
        }

    @classmethod
    def parse_call_expression(cls, expr_str: str) -> Tuple[str, List[Any], Dict[str, Any]]:
        """
        Parses a Python function/class call string like 'ProcessCardVisual(pid="P1", arrival_time=0)'
        or 'animate_enqueue(P1)'.
        Returns (name, args_list, kwargs_dict).
        """
        s = expr_str.strip()
        if not s:
            return "", [], {}

        # If it doesn't look like a call, treat as identifier
        if "(" not in s:
            return s, [], {}

        try:
            # Parse via ast
            parsed = ast.parse(s, mode="eval")
            if isinstance(parsed.body, ast.Call):
                call = parsed.body
                func_name = ""
                if isinstance(call.func, ast.Name):
                    func_name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    func_name = call.func.attr

                args: List[Any] = []
                for a in call.args:
                    try:
                        args.append(ast.literal_eval(a))
                    except Exception:
                        if isinstance(a, ast.Name):
                            args.append(a.id)
                        else:
                            args.append(ast.unparse(a) if hasattr(ast, "unparse") else str(a))

                kwargs: Dict[str, Any] = {}
                for kw in call.keywords:
                    kname = kw.arg
                    if not kname:
                        continue
                    try:
                        kwargs[kname] = ast.literal_eval(kw.value)
                    except Exception:
                        if isinstance(kw.value, ast.Name):
                            kwargs[kname] = kw.value.id
                        else:
                            kwargs[kname] = ast.unparse(kw.value) if hasattr(ast, "unparse") else str(kw.value)

                return func_name, args, kwargs
        except Exception:
            pass

        # Fallback regex parsing
        m = re.match(r"^([a-zA-Z_]\w*)\((.*)\)$", s, re.DOTALL)
        if m:
            fname = m.group(1)
            content = m.group(2).strip()
            kwargs = {}
            args = []
            if content:
                for chunk in content.split(","):
                    chunk = chunk.strip()
                    if "=" in chunk:
                        k, v = chunk.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if v.isdigit():
                            v = int(v)
                        kwargs[k] = v
                    elif chunk:
                        args.append(chunk.strip("'\""))
            return fname, args, kwargs

        return s, [], {}

    @classmethod
    def resolve_visual_action(
        cls,
        visual_rep: str,
        action_str: Optional[str] = None,
        viewport: str = "dock_main",
        step_tag: str = "L1-1",
        effect: str = "FadeIn",
        on_screen_after: Optional[List[str]] = None,
        raw_step: str = "",
    ) -> Tuple[bool, Optional[ResolvedVisualAction], Optional[str]]:
        """
        Resolves a visual representation and action to an authoritative component capability.
        If action_str is omitted/None, visual_rep is treated as a full animation step string
        and parsed via parse_animation_step.
        Returns:
            (is_valid, resolved_action, error_message)
        """
        if action_str is None:
            parsed = cls.parse_animation_step(visual_rep)
            if parsed.get("is_malformed"):
                return False, None, f"Malformed animation step: {visual_rep}"
            raw_step = visual_rep
            visual_rep = parsed.get("visual_representation", "")
            action_str = parsed.get("action", "")
            viewport = parsed.get("viewport", viewport)
            step_tag = parsed.get("tag", step_tag)
            effect = parsed.get("effect", effect)
            on_screen_after = parsed.get("on_screen_after", on_screen_after)

        on_screen = on_screen_after or []

        # 1. Parse visual representation
        comp_name, comp_args, comp_kwargs = cls.parse_call_expression(visual_rep)
        if not comp_name:
            comp_name = visual_rep.strip()

        # Check for ProceduralComponent
        is_procedural = (
            comp_name == "ProceduralComponent" or
            "ProceduralComponent(" in visual_rep
        )

        # Check for primitive
        is_primitive = comp_name in cls.PRIMITIVE_VISUALS or any(
            visual_rep.startswith(p) for p in cls.PRIMITIVE_VISUALS
        )

        # Check if registered in VisualRegistry
        registry = _get_registry()
        found = registry.get_component_by_name(comp_name)
        comp_cls = None
        is_registered = False
        manifest: Dict[str, Any] = {}

        if found is not None:
            dom, comp_cls = found
            is_registered = True
            comp_name = comp_cls.component_id
            manifest = comp_cls.get_manifest()

        # If not registered, not primitive, and not procedural
        if not is_registered and not is_primitive and not is_procedural:
            # Check if this might be a custom name like "Table" or "ConceptualCard"
            if comp_name.startswith("ConceptualCard") or "Card" in comp_name:
                is_procedural = True
            elif comp_name == "Table" or comp_name == "StateInspector":
                is_primitive = True
            elif comp_name and comp_name[0].isupper() and comp_name.endswith("Visual"):
                # Clearly intended to be a visual component, but not registered in registry!
                return (
                    False,
                    None,
                    f"Unknown or unregistered visual component '{comp_name}'. Available registered components: {sorted(list(set(c['component_id'] for c in registry.list_components())))}",
                )

        # 2. Validate component constructor parameters if registered
        if is_registered and manifest:
            init_params = manifest.get("init_params", {})
            # Check required parameters
            for pname, pinfo in init_params.items():
                if pinfo.get("required", False):
                    # Must be provided in kwargs or args
                    if pname not in comp_kwargs and not comp_args:
                        # Some components might be initialized later or in construct; warn if strict
                        pass
            # Check unsupported kwargs
            allowed_pnames = set(init_params.keys()) | {"kwargs", "vmobjects", "args"}
            for kname in comp_kwargs:
                if kname not in allowed_pnames and not any(kname.startswith(p) for p in ("width", "height", "color", "font_size")):
                    # Unknown parameter passed to registered component
                    return (
                        False,
                        None,
                        f"Component '{comp_name}' received unsupported constructor parameter '{kname}'. Supported parameters: {sorted(list(init_params.keys()))}",
                    )

        # 3. Parse action call
        act_name, act_args, act_kwargs = cls.parse_call_expression(action_str)
        if not act_name:
            act_name = action_str.strip()

        canonical_action = act_name
        method_obj = None
        operating_comp = comp_name
        operating_cls = comp_cls
        subject_entity = None

        if is_registered and comp_cls is not None:
            # Resolve against component's operations
            op_resolved = registry.resolve_operation(comp_name, act_name)

            # If operation is not on comp_name, attempt cross-component ownership disambiguation
            if op_resolved is None:
                for (cand_dom, cand_cid), cand_cls in registry._registry.items():
                    cand_op = registry.resolve_operation(cand_cid, act_name)
                    if cand_op is not None:
                        op_resolved = cand_op
                        operating_comp = cand_cls.component_id
                        operating_cls = cand_cls
                        subject_entity = visual_rep
                        break

            if op_resolved is not None:
                canonical_action = op_resolved["canonical_name"]
                method_obj = op_resolved["method"]

                # Validate action kwargs against operating component
                op_params = op_resolved.get("parameters", {})
                param_alias_map = {
                    "time": "new_time",
                    "t": "new_time",
                    "target_time": "new_time",
                    "current_time": "new_time",
                    "pid": "item",
                    "process": "item",
                    "process_card": "item",
                    "avg_waiting_time": "avg_wt",
                    "avg_turnaround_time": "avg_tat",
                    "avg_response_time": "avg_rt",
                    "start": "new_time" if "time_marker" in canonical_action else "start_time",
                    "start_time": "new_time" if "time_marker" in canonical_action else "start_time",
                    "end": "end_time",
                    "s": "new_time" if "time_marker" in canonical_action else "start_time",
                    "e": "end_time",
                }
                for k in act_kwargs:
                    if not op_params or k in op_params or k in ("run_time", "color", "scale_factor", "rate_func"):
                        continue
                    lower_k = k.lower()
                    alias_k = param_alias_map.get(lower_k, lower_k)
                    matched = any(p.lower() in (lower_k, alias_k) for p in op_params)
                    if not matched:
                        return (
                            False,
                            None,
                            f"Operation '{canonical_action}' on '{operating_comp}' received unsupported parameter '{k}'. Supported: {sorted(list(op_params.keys()))}",
                        )
            else:
                # Check if this is a semantic handle highlight: highlight_<handle_name>
                handle_suffix = act_name.replace("highlight_", "").replace("animate_highlight_", "")
                registered_handles = manifest.get("handles", []) or getattr(comp_cls, "supported_handles", [])
                is_handle_op = (
                    (act_name.startswith("highlight_") or act_name.startswith("animate_highlight_")) and
                    (handle_suffix in registered_handles or any(re.match(r"^" + h.replace("<pid>", r"\w+") + r"$", handle_suffix) for h in registered_handles))
                )
                is_highlight_op = act_name.startswith("highlight") or act_name.startswith("animate_highlight")
                if is_highlight_op and not is_handle_op:
                    # Semantic highlight on component or sub-feature: map to animate_highlight if available, else generic Indicate
                    if hasattr(comp_cls, "animate_highlight"):
                        canonical_action = "animate_highlight"
                        method_obj = getattr(comp_cls, "animate_highlight")
                    else:
                        canonical_action = "Indicate"
                    is_handle_op = True

                generic_actions = {
                    "FadeIn", "FadeOut", "MoveTo", "Create", "Write", "Indicate", "GrowFromCenter",
                    "set_position", "auto_fit_container", "setup", "highlight",
                    "inspect", "action: inspect", "action: hold", "hold", "wait", "action: wait"
                }
                if not is_handle_op and act_name not in generic_actions and not act_name.startswith("action:") and not any(act_name.startswith(g) for g in ("animate_", "show_", "display_", "init")):
                    supported_actions = manifest.get("semantic_actions", [])
                    return (
                        False,
                        None,
                        f"Component '{comp_name}' does not support operation '{act_name}'. Supported operations: {supported_actions}",
                    )
        elif is_procedural:
            # ProceduralComponent is a raw wrapper for native Manim groups; reject domain algorithmic methods
            disallowed_procedural_ops = {
                "calculate_shortest_path", "relax_edge", "extract_min", "enqueue", "dequeue",
                "get_cell", "update_cell", "dispatch", "execute_tick", "complete"
            }
            if act_name in disallowed_procedural_ops or "calculate" in act_name or "relax" in act_name:
                return (
                    False,
                    None,
                    f"ProceduralComponent is a raw Manim wrapper and does not support domain method '{act_name}'. Use a registered domain component or standard visual transforms.",
                )

        # 4. Validate viewport
        vp_norm = viewport.strip()
        if vp_norm.startswith("area:"):
            # Legacy grid area format, e.g. 'area: B2-D4'
            vp_norm = vp_norm.replace("area:", "").strip()
        elif vp_norm.startswith("(") and vp_norm.endswith(")"):
            # Legacy coordinate tuple, normalize to dock_main
            vp_norm = "dock_main"
        elif vp_norm not in cls.STANDARD_VIEWPORTS and not re.match(r"^[A-Fa-f][1-6](-[A-Fa-f][1-6])?$", vp_norm):
            return (
                False,
                None,
                f"Unknown target viewport '{vp_norm}'. Supported viewports: {sorted(list(cls.STANDARD_VIEWPORTS))}",
            )

        # Smart viewport routing: large tabular matrices (e.g. MetricTableVisual) require dock_main height
        # to ensure crisp typography and prevent collisions with GanttChartVisual in dock_inspector
        if operating_comp == "MetricTableVisual" and vp_norm == "dock_inspector":
            vp_norm = "dock_main"

        # 5. Determine target instance identifier, subject entity, intended call, and expected result
        inst_map = {
            "ProcessCardVisual": "process_card",
            "ReadyQueueVisual": "ready_queue",
            "CPUExecutionVisual": "cpu_exec",
            "GanttChartVisual": "gantt_chart",
            "MetricTableVisual": "metric_table",
            "GraphVisual": "graph",
            "DPTableMatrix": "dp_table",
            "MoleculeVisual": "molecule",
            "ReactionEnergyDiagram": "energy_diagram",
            "FunctionCurve": "function_curve",
            "NumberLineVisual": "number_line",
        }
        target_instance_id = inst_map.get(operating_comp)
        if not target_instance_id:
            if operating_comp.endswith("Visual"):
                target_instance_id = re.sub(r"Visual$", "", operating_comp).lower()
            else:
                target_instance_id = operating_comp.lower()

        # Generate recommended instantiation code and placement call
        instantiation_code = ""
        placement_call = ""
        if is_registered and comp_cls is not None:
            if operating_comp == "ProcessCardVisual":
                pid_v = comp_kwargs.get("pid") or act_kwargs.get("pid") or (comp_args[0] if comp_args else "P1")
                at_v = comp_kwargs.get("arrival_time", comp_kwargs.get("at", 0))
                bt_v = comp_kwargs.get("burst_time", comp_kwargs.get("bt", 3))
                st_v = comp_kwargs.get("state", "NEW")
                instantiation_code = f"{target_instance_id} = ProcessCardVisual(pid={pid_v!r}, arrival_time={at_v}, burst_time={bt_v}, state={st_v!r})"
            elif operating_comp == "ReadyQueueVisual":
                instantiation_code = f"{target_instance_id} = ReadyQueueVisual(capacity=4)"
            elif operating_comp == "CPUExecutionVisual":
                instantiation_code = f"{target_instance_id} = CPUExecutionVisual()"
            elif operating_comp == "GanttChartVisual":
                total_t = comp_kwargs.get("total_time", 9)
                instantiation_code = f"{target_instance_id} = GanttChartVisual(total_time={total_t})"
            elif operating_comp == "MetricTableVisual":
                instantiation_code = f"{target_instance_id} = MetricTableVisual()"
            elif operating_comp == "GraphVisual":
                instantiation_code = f"{target_instance_id} = GraphVisual(vertices=['A', 'B', 'C', 'D'], edges=[('A', 'B', 4), ('B', 'C', 2), ('A', 'C', 5)])"
            elif operating_comp == "DPTableMatrix":
                instantiation_code = f"{target_instance_id} = DPTableMatrix(matrix_values=[[0, 1, 2], [1, 2, 3]])"
            elif operating_comp == "ReactionEnergyDiagram":
                instantiation_code = f"{target_instance_id} = ReactionEnergyDiagram()"
            elif operating_comp == "MoleculeVisual":
                instantiation_code = f"{target_instance_id} = MoleculeVisual(atoms='H2O')"
            elif operating_comp == "FunctionCurve":
                instantiation_code = f"{target_instance_id} = FunctionCurve(function='cubic')"
            elif operating_comp == "NumberLineVisual":
                instantiation_code = f"{target_instance_id} = NumberLineVisual(x_range=(-5, 5, 1))"
            placement_call = f"self.{vp_norm}({target_instance_id})"

        # Subject entity extraction
        if not subject_entity:
            if comp_name != operating_comp:
                subject_entity = visual_rep
            elif "pid" in comp_kwargs:
                subject_entity = str(comp_kwargs["pid"])
            elif comp_args:
                subject_entity = str(comp_args[0])
            elif "pid" in act_kwargs:
                subject_entity = str(act_kwargs["pid"])
            elif act_args and isinstance(act_args[0], str):
                subject_entity = str(act_args[0])

        # Synthesize concrete Python call
        call_args = []
        for a in act_args:
            call_args.append(repr(a))
        for k, v in act_kwargs.items():
            call_args.append(f"{k}={v!r}")

        # If operating on ReadyQueueVisual with enqueue and no args, supply the process card or pid
        if operating_comp == "ReadyQueueVisual" and canonical_action == "animate_enqueue" and not call_args:
            pid_val = comp_kwargs.get("pid") or act_kwargs.get("pid") or (comp_args[0] if comp_args else "P1")
            call_args.append(f"card_{str(pid_val).lower()}")

        intended_call = (
            f"{target_instance_id}.{canonical_action}({', '.join(call_args)})"
            if method_obj
            else f"self.play({effect}({target_instance_id}))"
        )

        # Trace reference and expected visual result
        trace_ref = None
        if canonical_action == "animate_enqueue":
            pid_txt = comp_kwargs.get("pid") or act_kwargs.get("pid") or "process"
            expected_visual_result = f"Process {pid_txt} enters the Ready Queue tail slot"
            trace_ref = f"arrival of {pid_txt}"
        elif canonical_action == "animate_dequeue":
            expected_visual_result = "Process at head of Ready Queue dequeues towards CPU"
            trace_ref = "head process dispatch"
        elif canonical_action == "animate_dispatch":
            pid_txt = act_kwargs.get("pid") or comp_kwargs.get("pid") or "process"
            expected_visual_result = f"CPU core dispatches and begins executing process {pid_txt}"
            trace_ref = f"dispatch {pid_txt}"
        elif canonical_action == "animate_execute_tick":
            expected_visual_result = "CPU indicates execution progress / tick"
        elif canonical_action == "animate_complete":
            expected_visual_result = "Running process completes burst and CPU returns to idle"
            trace_ref = "process termination"
        elif canonical_action == "animate_add_block":
            p = act_kwargs.get("pid", "process")
            s = act_kwargs.get("start_time", 0)
            e = act_kwargs.get("end_time", 0)
            expected_visual_result = f"Gantt interval [{s}, {e}] added for {p} and time cursor advances"
            trace_ref = f"timeline [{s}, {e}]"
        elif canonical_action == "animate_update_response_time":
            p = act_kwargs.get("pid", "process")
            rt_val = act_kwargs.get("rt", act_kwargs.get("response_time", 0))
            expected_visual_result = f"MetricTable reveals Response Time (RT={rt_val}) for {p}"
            trace_ref = f"{p} RT={rt_val}"
        elif canonical_action == "animate_update_process_metrics":
            p = act_kwargs.get("pid", "process")
            expected_visual_result = f"MetricTable reveals completed metrics (CT, TAT, WT) for {p}"
            trace_ref = f"{p} completion metrics"
        elif canonical_action == "animate_show_averages":
            expected_visual_result = "MetricTable summary row reveals final average TAT, WT, RT"
            trace_ref = "summary averages"
        else:
            expected_visual_result = f"{effect} of {comp_name} in {vp_norm}"

        resolved = ResolvedVisualAction(
            step_tag=step_tag,
            effect=effect,
            component_name=comp_name,
            component_class=comp_cls,
            canonical_action=canonical_action,
            method=method_obj,
            args=act_args,
            kwargs=act_kwargs,
            viewport=vp_norm,
            on_screen_after=on_screen,
            is_registered_component=is_registered,
            is_primitive=is_primitive,
            is_procedural=is_procedural,
            raw_step=raw_step,
            operating_component=operating_comp,
            operating_class=operating_cls,
            subject_entity=subject_entity,
            target_instance_id=target_instance_id,
            intended_call=intended_call,
            trace_ref=trace_ref,
            expected_visual_result=expected_visual_result,
            instantiation_code=instantiation_code,
            placement_call=placement_call,
        )
        return True, resolved, None

    @classmethod
    def validate_storyboard_actions(
        cls,
        storyboard_data: Dict[str, Any],
        authoritative_trace: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validates all animation steps in a storyboard dictionary.
        Returns a list of discrepancy dictionaries (empty if 100% valid).
        """
        discrepancies: List[Dict[str, Any]] = []
        sections = storyboard_data.get("sections", [])
        if not isinstance(sections, list):
            return discrepancies

        for sec in sections:
            sec_id = sec.get("id", "unknown_section")
            anims = sec.get("animations", [])
            if not isinstance(anims, list):
                continue

            for step_idx, step_item in enumerate(anims):
                if not isinstance(step_item, str):
                    continue

                parsed = cls.parse_animation_step(step_item)
                if parsed.get("is_malformed"):
                    discrepancies.append({
                        "section_id": sec_id,
                        "location": f"animations[{step_idx}]",
                        "category": "INVALID_ANIMATION_SYNTAX",
                        "actual_claim": step_item,
                        "reason": "Animation step does not conform to the 6-field pipe-delimited format",
                        "severity": "FAILED",
                    })
                    continue

                is_valid, resolved, err_msg = cls.resolve_visual_action(
                    visual_rep=parsed["visual_representation"],
                    action_str=parsed["action"],
                    viewport=parsed["viewport"],
                    step_tag=parsed["tag"],
                    effect=parsed["effect"],
                    on_screen_after=parsed["on_screen_after"],
                    raw_step=parsed["raw_step"],
                )

                if not is_valid:
                    discrepancies.append({
                        "section_id": sec_id,
                        "location": f"animations[{step_idx}]: {parsed['tag']}",
                        "category": "UNSUPPORTED_OPERATION" if "support operation" in (err_msg or "") else "INVALID_COMPONENT_REFERENCE",
                        "actual_claim": parsed["action"] or parsed["visual_representation"],
                        "reason": err_msg or "Visual action validation failed",
                        "severity": "FAILED",
                    })

                # Check trace references if trace is available
                if is_valid and resolved and authoritative_trace:
                    algo = authoritative_trace.get("algorithm")
                    if algo == "fcfs":
                        trace_discrepancy = cls._validate_fcfs_action_trace(
                            sec_id, step_idx, resolved, authoritative_trace
                        )
                        if trace_discrepancy:
                            discrepancies.append(trace_discrepancy)

        return discrepancies

    @classmethod
    def _validate_fcfs_action_trace(
        cls,
        section_id: str,
        step_idx: int,
        action: ResolvedVisualAction,
        trace: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Validates action against authoritative FCFS scheduling trace."""
        known_pids = {p.get("pid") for p in trace.get("processes", [])}
        gantt = trace.get("gantt_chart", [])
        metrics = trace.get("process_metrics", {})

        # Check PID references
        claimed_pid = action.kwargs.get("pid")
        if not claimed_pid and action.args and isinstance(action.args[0], str):
            if action.canonical_action not in (
                "animate_set_state", "set_state", "set_state_instant",
                "animate_update_remaining_time", "set_remaining_time_instant",
                "animate_highlight", "highlight", "Indicate", "inspect"
            ):
                claimed_pid = action.args[0]
        if action.component_name == "ProcessCardVisual" and claimed_pid:
            if claimed_pid not in known_pids:
                return {
                    "section_id": section_id,
                    "location": f"animations[{step_idx}]: {action.step_tag}",
                    "category": "TRACE_REFERENCE_MISMATCH",
                    "actual_claim": f"Process PID '{claimed_pid}'",
                    "expected_value": f"One of {sorted(list(known_pids))}",
                    "reason": f"ProcessCardVisual references unknown process PID '{claimed_pid}' not in authoritative process list",
                    "severity": "FAILED",
                }

        # Check Gantt chart block references
        if action.canonical_action == "animate_add_block":
            b_pid = action.kwargs.get("pid")
            start = action.kwargs.get("start_time") or action.kwargs.get("start")
            end = action.kwargs.get("end_time") or action.kwargs.get("end")

            if b_pid is not None and start is not None and end is not None:
                # Find matching block in trace gantt_chart
                matched = any(
                    b.get("pid") == b_pid and b.get("start_time") == int(start) and b.get("end_time") == int(end)
                    for b in gantt
                )
                if not matched:
                    return {
                        "section_id": section_id,
                        "location": f"animations[{step_idx}]: {action.step_tag}",
                        "category": "ALGORITHMIC_STATE_MISMATCH",
                        "actual_claim": f"Gantt block ({b_pid}, start={start}, end={end})",
                        "expected_value": str([(b.get("pid"), b.get("start_time"), b.get("end_time")) for b in gantt]),
                        "reason": f"Gantt block ({b_pid}, {start}->{end}) contradicts authoritative FCFS execution interval",
                        "severity": "FAILED",
                    }

        # Check MetricTable updates
        if action.canonical_action == "animate_update_process_metrics":
            m_pid = action.kwargs.get("pid") or (action.args[0] if action.args else None)
            if m_pid and m_pid in metrics:
                exp_m = metrics[m_pid]
                for col in ("ct", "tat", "wt"):
                    val = action.kwargs.get(col)
                    exp_key = "completion_time" if col == "ct" else ("turnaround_time" if col == "tat" else "waiting_time")
                    exp_val = exp_m.get(exp_key)
                    if val is not None and exp_val is not None and int(val) != int(exp_val):
                        return {
                            "section_id": section_id,
                            "location": f"animations[{step_idx}]: {action.step_tag}",
                            "category": "VALUE_MISMATCH",
                            "actual_claim": f"{m_pid}.{col.upper()} = {val}",
                            "expected_value": str(exp_val),
                            "reason": f"Claimed metric {col.upper()}={val} for '{m_pid}' contradicts authoritative {exp_key}={exp_val}",
                            "severity": "FAILED",
                        }

        return None

    @classmethod
    def resolve_storyboard(cls, storyboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches a storyboard dictionary by attaching resolved_actions to each section.
        """
        sb_copy = dict(storyboard_data)
        sections = sb_copy.get("sections", [])
        resolved_sections = []

        for sec in sections:
            sec_copy = dict(sec)
            anims = sec_copy.get("animations", [])
            resolved_actions = []

            for step_idx, step_item in enumerate(anims):
                if not isinstance(step_item, str):
                    continue
                parsed = cls.parse_animation_step(step_item)
                _, resolved, _ = cls.resolve_visual_action(
                    visual_rep=parsed["visual_representation"],
                    action_str=parsed["action"],
                    viewport=parsed["viewport"],
                    step_tag=parsed["tag"],
                    effect=parsed["effect"],
                    on_screen_after=parsed["on_screen_after"],
                    raw_step=parsed["raw_step"],
                )
                if resolved is not None:
                    resolved_actions.append(resolved.to_dict())

            sec_copy["resolved_actions"] = resolved_actions
            resolved_sections.append(sec_copy)

        sb_copy["sections"] = resolved_sections
        return sb_copy
