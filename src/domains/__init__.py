"""
CineLearn Extensible Domain Visual Component Framework
======================================================
Provides the foundational VisualComponent protocol and the VisualRegistry
for domain-independent educational video generation.
"""

from typing import Dict, List, Optional, Tuple, Type, Any, Callable, Sequence
import importlib
import pkgutil
import inspect
import re
from pathlib import Path
from manim import VGroup, Mobject
import numpy as np

from .layout import RectangularViewport, GenericContainerLayout
from .svg_sanitizer import is_safe_svg, validate_svg_content


class VisualComponent(VGroup):
    """
    Universal base protocol for CineLearn visual components.
    Extends Manim's native VGroup so it can be passed to any Manim animation,
    transformation, or renderer seamlessly.
    """
    component_id: str = "generic"
    domain: str = "generic"
    preferred_aspect: float = 1.6
    min_scale_threshold: float = 0.55  # Below this, sub-labels may lose visual legibility

    def __init__(self, *vmobjects, **kwargs):
        super().__init__(*vmobjects)
        self._semantic_handles: Dict[str, Mobject] = {}
        self.is_compressed: bool = False
        self.applied_scale_factor: float = 1.0

    def register_semantic_handle(self, name: str, mobject: Mobject) -> None:
        """Registers a named sub-element (e.g. 'node_A', 'bond_1', 'atom_O') for targeted animations."""
        self._semantic_handles[name] = mobject

    def get_semantic_handle(self, name: str) -> Optional[Mobject]:
        """Retrieves a named sub-element for targeted animation."""
        return self._semantic_handles.get(name)

    def auto_fit_container(self, max_width: float, max_height: float, padding: float = 0.04) -> Tuple[float, bool]:
        """
        Calculates and applies optimal scaling to strictly fit within the target container.
        INVARIANT: scale_factor <= min(target_w / w, target_h / h, 1.0).
        The component is GUARANTEED to NEVER exceed container width or height.
        If scale_factor < min_scale_threshold, is_compressed is flagged for layout auditing,
        but container boundaries are strictly respected.
        """
        w = float(self.width) if hasattr(self, "width") else float(self.get_width())
        h = float(self.height) if hasattr(self, "height") else float(self.get_height())
        if w <= 0.0 or h <= 0.0:
            return 1.0, False

        target_w = float(max_width) * (1.0 - padding)
        target_h = float(max_height) * (1.0 - padding)

        # Scale strictly bounded by container dimensions: NEVER overflow container
        scale_factor = min(target_w / w, target_h / h, 1.0)
        self.scale(scale_factor)
        self.applied_scale_factor = scale_factor

        self.is_compressed = scale_factor < self.min_scale_threshold
        return scale_factor, self.is_compressed

    def get_bounding_box(self) -> Dict[str, float]:
        """Returns standard 2D axis-aligned bounding box."""
        try:
            w = float(self.width) if hasattr(self, "width") else float(self.get_width())
            h = float(self.height) if hasattr(self, "height") else float(self.get_height())
            return {
                "left": float(self.get_left()[0]),
                "right": float(self.get_right()[0]),
                "bottom": float(self.get_bottom()[1]),
                "top": float(self.get_top()[1]),
                "width": w,
                "height": h,
                "center": [float(self.get_center()[0]), float(self.get_center()[1]), 0.0],
            }
        except Exception:
            return {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0, "width": 0.0, "height": 0.0, "center": [0.0, 0.0, 0.0]}

    @classmethod
    def get_manifest(cls) -> Dict[str, Any]:
        """Returns metadata manifest for Stage 1.5 Planner, Stage 2 Storyboard, and Stage 3 Code Generator."""
        # Inspect __init__ signature
        init_params = {}
        try:
            sig = inspect.signature(cls.__init__)
            for name, param in sig.parameters.items():
                if name in ("self", "args", "kwargs", "vmobjects"):
                    continue
                type_name = "Any"
                if param.annotation != inspect.Parameter.empty:
                    if hasattr(param.annotation, "__name__"):
                        type_name = param.annotation.__name__
                    else:
                        type_name = str(param.annotation).replace("typing.", "")
                init_params[name] = {
                    "type": type_name,
                    "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                    "required": param.default == inspect.Parameter.empty,
                }
        except Exception:
            pass

        # Inspect semantic animation methods
        actions = []
        operations = {}
        for m in dir(cls):
            if (m.startswith("animate_") or m.startswith("action_") or m.startswith("highlight_")) and callable(getattr(cls, m)):
                actions.append(m)
                try:
                    fn = getattr(cls, m)
                    sig = inspect.signature(fn)
                    op_params = {}
                    for pname, p in sig.parameters.items():
                        if pname in ("self", "args", "kwargs"):
                            continue
                        ptype = "Any"
                        if p.annotation != inspect.Parameter.empty:
                            ptype = p.annotation.__name__ if hasattr(p.annotation, "__name__") else str(p.annotation).replace("typing.", "")
                        op_params[pname] = {
                            "type": ptype,
                            "default": str(p.default) if p.default != inspect.Parameter.empty else None,
                            "required": p.default == inspect.Parameter.empty,
                        }
                    operations[m] = {
                        "name": m,
                        "description": (inspect.getdoc(fn) or "").strip(),
                        "parameters": op_params,
                    }
                except Exception:
                    operations[m] = {"name": m, "description": "", "parameters": {}}

        supported_handles = list(getattr(cls, "supported_handles", []))
        suggested_viewport = getattr(cls, "suggested_viewport", "dock_main")
        aliases = list(getattr(cls, "aliases", []) or [])

        return {
            "component_id": cls.component_id,
            "domain": cls.domain,
            "description": (cls.__doc__ or "").strip(),
            "aliases": aliases,
            "preferred_aspect": cls.preferred_aspect,
            "min_scale_threshold": cls.min_scale_threshold,
            "suggested_viewport": suggested_viewport,
            "supported_handles": supported_handles,
            "semantic_handles": supported_handles,
            "init_params": init_params,
            "semantic_actions": sorted(actions),
            "operations": operations,
        }


class ProceduralComponent(VisualComponent):
    """
    Tier 3 Fallback & Extensible Escape Hatch.
    Wraps arbitrary user-defined Manim shapes, groups, equations, or SVGs into
    a first-class VisualComponent conforming to the universal protocol.
    """
    component_id = "ProceduralComponent"
    domain = "generic"

    def __init__(self, *submobjects, name: str = "ProceduralVisual", **kwargs):
        self.procedural_name = name

        def _is_empty_mobject(m) -> bool:
            if hasattr(m, "points") and len(m.points) > 0:
                return False
            if hasattr(m, "submobjects") and len(m.submobjects) > 0:
                return all(_is_empty_mobject(sub) for sub in m.submobjects)
            return True

        is_empty = (not submobjects) or all(_is_empty_mobject(m) for m in submobjects)
        if is_empty:
            from manim import RoundedRectangle, Text
            box = RoundedRectangle(
                corner_radius=0.15,
                width=4.5,
                height=2.2,
                stroke_color="#58C4DD",
                stroke_width=2.0,
                fill_color="#101720",
                fill_opacity=0.85,
            )
            box.move_to(np.array([2.225, 0.0, 0.0]))
            title_text = name.replace("_", " ").title()
            label = Text(title_text, font_size=18, color="#FFFFFF").move_to(box.get_center())
            super().__init__(box, label, **kwargs)
            self._placeholder_box = box
            self._placeholder_label = label
            self._is_placeholder = True
            self.register_semantic_handle("box", box)
            self.register_semantic_handle("label", label)
        else:
            super().__init__(*submobjects, **kwargs)
            self._is_placeholder = False

    def add(self, *mobjects):
        if getattr(self, "_is_placeholder", False) and mobjects:
            if hasattr(self, "_placeholder_box") and self._placeholder_box in self.submobjects:
                super().remove(self._placeholder_box)
            if hasattr(self, "_placeholder_label") and self._placeholder_label in self.submobjects:
                super().remove(self._placeholder_label)
            self._is_placeholder = False
        return super().add(*mobjects)

    def get_cell(self, *args, **kwargs):
        """Safe fallback for scripts mistakenly treating ProceduralComponent as a matrix/table."""
        from manim import Mobject
        for sub in self.submobjects:
            has_real = any('get_cell' in cls.__dict__ for cls in sub.__class__.__mro__ if cls is not Mobject)
            if has_real:
                return sub.get_cell(*args, **kwargs)
        return self.submobjects[0] if self.submobjects else self

    def initialize_structure(self, *args, **kwargs):
        """Safe fallback for scripts mistakenly calling initialize_structure on ProceduralComponent."""
        return self

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        d = object.__getattribute__(self, "__dict__")
        semantic_handles = d.get("semantic_handles", {})
        if name in semantic_handles:
            return semantic_handles[name]
        for sub in d.get("submobjects", []):
            if hasattr(sub, name):
                return getattr(sub, name)
        # Return a safe VMobject so indicate/play doesn't crash on synthetic handles like .core
        from manim import VMobject
        dummy = VMobject()
        self.add(dummy)
        return dummy


from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class DomainDescriptor:
    """Metadata descriptor for a domain capability package."""
    name: str
    display_name: str = ""
    keywords: List[str] = field(default_factory=list)
    visual_modes: List[str] = field(default_factory=list)
    description: str = ""


class DomainTraceGuidanceRegistry:
    """Decoupled registry for domain-specific trace formatting in Stage 2 Storyboards."""
    _formatters: Dict[str, Callable[[Dict[str, Any]], str]] = {}

    @classmethod
    def register(cls, algorithm_or_domain: str, formatter: Callable[[Dict[str, Any]], str]) -> Callable:
        key = algorithm_or_domain.lower().strip()
        cls._formatters[key] = formatter
        return formatter

    @classmethod
    def get(cls, algorithm_or_domain: str) -> Optional[Callable[[Dict[str, Any]], str]]:
        return cls._formatters.get(algorithm_or_domain.lower().strip())

    @classmethod
    def format_guidance(cls, algo_state: Dict[str, Any]) -> str:
        if not algo_state or not isinstance(algo_state, dict):
            return ""
        algo = str(algo_state.get("algorithm") or algo_state.get("domain") or "").lower().strip()
        formatter = cls.get(algo)
        if formatter:
            return formatter(algo_state)
        return ""


class DomainFidelityVerifier(ABC):
    """Abstract protocol for domain-specific storyboard fidelity verifiers."""

    @abstractmethod
    def can_verify(self, model: Any, sections: List[Dict[str, Any]]) -> bool:
        pass

    @abstractmethod
    def verify(self, model: Any, sections: List[Dict[str, Any]]) -> Any:
        pass


class DomainFidelityRegistry:
    """Decoupled registry for domain-specific fidelity verifiers used by StoryboardFidelityChecker."""
    _verifiers: Dict[str, DomainFidelityVerifier] = {}

    @classmethod
    def register(cls, algorithm_or_domain: str, verifier: DomainFidelityVerifier) -> DomainFidelityVerifier:
        cls._verifiers[algorithm_or_domain.lower().strip()] = verifier
        return verifier

    @classmethod
    def get(cls, algorithm_or_domain: str) -> Optional[DomainFidelityVerifier]:
        return cls._verifiers.get(algorithm_or_domain.lower().strip())

    @classmethod
    def find_verifier_for_model(cls, model: Any, sections: List[Dict[str, Any]]) -> Optional[DomainFidelityVerifier]:
        algo = getattr(model, "algo_state", {}).get("algorithm") if getattr(model, "algo_state", None) else None
        if not algo:
            if getattr(model, "is_fcfs", False):
                algo = "fcfs"
            elif getattr(model, "is_dijkstra", False):
                algo = "dijkstra"
        if algo:
            v = cls.get(algo)
            if v and v.can_verify(model, sections):
                return v
        for verifier in cls._verifiers.values():
            if verifier.can_verify(model, sections):
                return verifier
        return None


class VisualRegistry:
    """
    Central registry for domain-specific visual components.
    Allows independent domain packages to register components without modifying core engine.
    """
    _registry: Dict[Tuple[str, str], Type[VisualComponent]] = {}
    _domains: Dict[str, DomainDescriptor] = {}
    _action_weights: Dict[str, float] = {}

    @classmethod
    def register_action_weight(cls, action_name: str, weight: float) -> None:
        """Registers a semantic action weight at the domain/registry level."""
        cls._action_weights[action_name.strip()] = float(weight)

    @classmethod
    def get_action_weights(cls) -> Dict[str, float]:
        """
        Collects semantic action weights defined across registered components and domains.
        Returns a dictionary mapping action method names to pedagogical importance weights.
        """
        weights: Dict[str, float] = dict(cls._action_weights)
        for (dom, cid), comp_cls in cls._registry.items():
            comp_weights = getattr(comp_cls, "action_weights", None)
            if isinstance(comp_weights, dict):
                weights.update(comp_weights)
        return weights

    @classmethod
    def register_domain(cls, descriptor: DomainDescriptor) -> None:
        """Registers a domain descriptor with metadata, keywords, and visual modes."""
        cls._domains[descriptor.name.lower().strip()] = descriptor

    @classmethod
    def register(cls, domain: str, component_id: Optional[str] = None, aliases: Optional[Sequence[str]] = None) -> Callable:
        """Decorator to register a VisualComponent class with optional natural aliases."""
        def decorator(subclass: Type[VisualComponent]) -> Type[VisualComponent]:
            cid = component_id or getattr(subclass, "component_id", subclass.__name__)
            dom = domain.lower().strip()
            subclass.domain = dom
            subclass.component_id = cid
            if aliases:
                existing_aliases = list(getattr(subclass, "aliases", []) or [])
                for a in aliases:
                    if a not in existing_aliases:
                        existing_aliases.append(a)
                subclass.aliases = existing_aliases
            cls._registry[(dom, cid.lower())] = subclass
            return subclass
        return decorator

    @classmethod
    def get(cls, domain: str, component_id: str) -> Optional[Type[VisualComponent]]:
        """Retrieves a registered component class by domain and ID."""
        key = (domain.lower().strip(), component_id.lower().strip())
        return cls._registry.get(key)

    @classmethod
    def get_component_by_name(cls, component_name: str) -> Optional[Tuple[str, Type[VisualComponent]]]:
        """
        Retrieves a registered component across any registered domain by component ID, class name,
        or natural language alias (e.g. 'ready queue' -> ReadyQueueVisual, 'PCB' -> ProcessCardVisual).
        Returns tuple of (domain, component_class) or None if not found.
        """
        if not component_name:
            return None
        c_raw = component_name.lower().strip()
        c_alphanumeric = re.sub(r"[^a-z0-9]", "", c_raw)
        if not c_alphanumeric:
            return None

        # 1. Exact match on cid or class name
        for (dom, cid), comp_cls in cls._registry.items():
            if cid == c_raw or comp_cls.__name__.lower() == c_raw:
                return dom, comp_cls

        # 2. Check declared aliases on comp_cls
        for (dom, cid), comp_cls in cls._registry.items():
            aliases = getattr(comp_cls, "aliases", []) or []
            for alias in aliases:
                a_raw = str(alias).lower().strip()
                if a_raw == c_raw or re.sub(r"[^a-z0-9]", "", a_raw) == c_alphanumeric:
                    return dom, comp_cls

        # 3. Normalized alphanumeric match or suffix-stripped match
        for (dom, cid), comp_cls in cls._registry.items():
            cid_alpha = re.sub(r"[^a-z0-9]", "", cid)
            cls_alpha = re.sub(r"[^a-z0-9]", "", comp_cls.__name__.lower())
            if c_alphanumeric in (cid_alpha, cls_alpha):
                return dom, comp_cls
            # If component has "visual" suffix and query omitted it
            if cls_alpha.endswith("visual") and c_alphanumeric == cls_alpha[:-6]:
                return dom, comp_cls
            if cid_alpha.endswith("visual") and c_alphanumeric == cid_alpha[:-6]:
                return dom, comp_cls

        # 4. Fallback substring containment for common compound names (e.g. "ready queue" in "readyqueuevisual")
        for (dom, cid), comp_cls in cls._registry.items():
            cls_alpha = re.sub(r"[^a-z0-9]", "", comp_cls.__name__.lower())
            if len(c_alphanumeric) >= 4 and (c_alphanumeric in cls_alpha or cls_alpha in c_alphanumeric):
                return dom, comp_cls

        return None

    @classmethod
    def resolve_operation(cls, component_name: str, action_name: str) -> Optional[Dict[str, Any]]:
        """
        Resolves an operation string to the canonical method name and metadata on a component.
        Matches exact names ('animate_enqueue') or short names ('enqueue' -> 'animate_enqueue').
        Returns dict with canonical_name, method, parameters, description, or None if unsupported.
        """
        found = cls.get_component_by_name(component_name)
        if not found:
            return None
        _, comp_cls = found
        manifest = comp_cls.get_manifest()
        ops = manifest.get("operations", {})

        action_norm = action_name.strip()
        # 1. Exact match in operations
        if action_norm in ops:
            return {
                "canonical_name": action_norm,
                "method": getattr(comp_cls, action_norm, None),
                "parameters": ops[action_norm].get("parameters", {}),
                "description": ops[action_norm].get("description", ""),
            }

        # 2. Try prefixing 'animate_'
        anim_candidate = f"animate_{action_norm}"
        if anim_candidate in ops:
            return {
                "canonical_name": anim_candidate,
                "method": getattr(comp_cls, anim_candidate, None),
                "parameters": ops[anim_candidate].get("parameters", {}),
                "description": ops[anim_candidate].get("description", ""),
            }

        # 3. Check direct public methods on comp_cls starting with animate_ or set_
        for m in dir(comp_cls):
            if m.startswith("_"):
                continue
            if m == action_norm or m == anim_candidate:
                fn = getattr(comp_cls, m)
                return {
                    "canonical_name": m,
                    "method": fn,
                    "parameters": ops.get(m, {}).get("parameters", {}),
                    "description": inspect.getdoc(fn) or "",
                }

        return None

    @classmethod
    def get_domain_for_topic(cls, topic: str, visual_mode: Optional[str] = None) -> Optional[str]:
        """
        Infers the primary domain string using registered DomainDescriptor metadata and visual modes.
        """
        t = topic.lower().strip()
        v_mode = str(visual_mode).lower().strip() if visual_mode else None

        # 1. Match visual modes if provided
        if v_mode:
            for d_name, desc in cls._domains.items():
                if any(v_mode == vm.lower() for vm in desc.visual_modes):
                    return d_name

        # 2. Match registered domain descriptor keywords
        for d_name, desc in cls._domains.items():
            if any(kw.lower() in t for kw in desc.keywords):
                return d_name

        # 3. Fallback to basic domain name match
        for d_name in cls._domains:
            if d_name in t:
                return d_name

        return None

    @classmethod
    def list_domains(cls) -> List[str]:
        """Returns sorted list of all registered domains."""
        domains = set(d for d, _ in cls._registry.keys()) | set(cls._domains.keys())
        return sorted(list(domains))

    @classmethod
    def list_components(cls, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns metadata manifests for all registered components (optionally filtered by domain)."""
        components = []
        for (dom, cid), comp_cls in cls._registry.items():
            if domain is None or dom == domain.lower().strip():
                components.append(comp_cls.get_manifest())
        return components

    @classmethod
    def get_schema_manifest(cls, domain: Optional[str] = None) -> Dict[str, Any]:
        """Generates JSON-serializable manifest for dynamic prompt injection into LLM stages."""
        manifests = {}
        for (dom, cid), comp_cls in cls._registry.items():
            if domain is None or dom == domain.lower().strip():
                if dom not in manifests:
                    manifests[dom] = []
                manifests[dom].append(comp_cls.get_manifest())
        return manifests

    @classmethod
    def clear(cls) -> None:
        """Clears registry (used primarily in test fixtures)."""
        cls._registry.clear()

    @classmethod
    def discover_domains(cls, base_package: Optional[str] = None) -> List[str]:
        """
        Recursively discovers and imports all domain modules in the package directory
        and its subpackages (e.g. cs, chemistry, math).
        Any module decorated with @VisualRegistry.register will be registered automatically.
        """
        if not base_package:
            base_package = __name__
        discovered = []
        try:
            pkg = importlib.import_module(base_package)
            if not hasattr(pkg, "__path__"):
                return discovered

            def _walk(package_path, package_name):
                for _, mod_name, is_pkg in pkgutil.iter_modules(package_path):
                    if mod_name in ("layout", "__init__", "svg_sanitizer"):
                        continue
                    full_name = f"{package_name}.{mod_name}"
                    try:
                        mod = importlib.import_module(full_name)
                        discovered.append(full_name)
                        # Ensure any VisualComponent subclasses in the module are registered
                        for attr_name in dir(mod):
                            try:
                                obj = getattr(mod, attr_name)
                                if (
                                    isinstance(obj, type)
                                    and issubclass(obj, VisualComponent)
                                    and obj not in (VisualComponent, ProceduralComponent)
                                    and hasattr(obj, "domain")
                                    and hasattr(obj, "component_id")
                                ):
                                    dom = str(obj.domain).lower().strip()
                                    cid = str(obj.component_id).lower().strip()
                                    cls._registry[(dom, cid)] = obj
                            except Exception:
                                pass
                        if is_pkg and hasattr(mod, "__path__"):
                            _walk(mod.__path__, full_name)
                    except Exception:
                        pass

            _walk(pkg.__path__, base_package)
        except Exception:
            pass
        return discovered


# Import Action Resolver after VisualRegistry is fully initialized
from .action_resolver import ResolvedVisualAction, StoryboardActionResolver


__all__ = [
    "RectangularViewport",
    "GenericContainerLayout",
    "VisualComponent",
    "ProceduralComponent",
    "VisualRegistry",
    "DomainDescriptor",
    "DomainTraceGuidanceRegistry",
    "DomainFidelityVerifier",
    "DomainFidelityRegistry",
    "ResolvedVisualAction",
    "StoryboardActionResolver",
    "is_safe_svg",
    "validate_svg_content",
]

# Register core domain descriptors with capability metadata
VisualRegistry.register_domain(
    DomainDescriptor(
        name="cs",
        display_name="Computer Science & Systems",
        keywords=[
            "cpu", "scheduling", "fcfs", "process", "ready queue", "queue", "gantt",
            "operating system", "os", "graph", "dijkstra", "shortest path", "algorithm",
            "knapsack", "dynamic programming", "mutex", "concurrency", "semaphore",
        ],
        visual_modes=["process_flowchart", "state_table_matrix", "network_graph"],
        description="Computer Science, Operating Systems, Algorithms, and Data Structures",
    )
)
VisualRegistry.register_domain(
    DomainDescriptor(
        name="chemistry",
        display_name="Chemistry & Molecular Science",
        keywords=[
            "chemistry", "molecule", "reaction", "kinetics", "activation energy",
            "arrhenius", "enthalpy", "catalyst", "covalent", "bond",
        ],
        visual_modes=["molecular_spatial"],
        description="Chemistry, Molecular Visualizations, Reaction Kinetics",
    )
)
VisualRegistry.register_domain(
    DomainDescriptor(
        name="math",
        display_name="Mathematics & Geometry",
        keywords=[
            "math", "calculus", "derivative", "function", "curve", "number line",
            "pythagorean", "theorem", "geometry", "integral",
        ],
        visual_modes=["coordinate_geometry"],
        description="Mathematics, Geometry, Analysis and Functions",
    )
)

# Automatically discover and register all domain components upon import
VisualRegistry.discover_domains("src.domains")

