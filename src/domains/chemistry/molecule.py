"""
Molecule Visual Component
=========================
Configurable, reusable visual component for chemical structures,
Lewis diagrams, ball-and-stick models, and bond breaking/formation animations.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import numpy as np
from manim import (
    VGroup,
    Circle,
    Text,
    Line,
    Indicate,
    FadeIn,
    FadeOut,
    Transform,
    Animation,
    AnimationGroup,
    SVGMobject,
    Wiggle,
    UP,
    DOWN,
    RIGHT,
    LEFT,
)

from .. import VisualComponent, VisualRegistry
from ..svg_sanitizer import is_safe_svg


# Standard CPK Element Color Palette
CPK_COLORS: Dict[str, str] = {
    "H": "#FFFFFF",   # Hydrogen: White
    "C": "#909090",   # Carbon: Dark Gray
    "N": "#4A90E2",   # Nitrogen: Blue
    "O": "#FF4136",   # Oxygen: Red
    "F": "#83C167",   # Fluorine: Light Green
    "CL": "#2ECC40",  # Chlorine: Green
    "BR": "#85144B",  # Bromine: Maroon
    "I": "#B10DC9",   # Iodine: Violet
    "P": "#FF851B",   # Phosphorus: Orange
    "S": "#FFDC00",   # Sulfur: Yellow
    "NA": "#7FDBFF",  # Sodium: Sky Blue
    "K": "#B10DC9",   # Potassium: Purple
    "CA": "#A0A0A0",  # Calcium: Gray
    "FE": "#E67E22",  # Iron: Rust Orange
}

# Pre-configured common molecular structures for educational rapid composition
MOLECULE_PRESETS: Dict[str, Dict[str, Any]] = {
    "H2O": {
        "atoms": [
            {"id": "O", "element": "O", "pos": [0.0, 0.25, 0.0]},
            {"id": "H1", "element": "H", "pos": [-0.95, -0.42, 0.0]},
            {"id": "H2", "element": "H", "pos": [0.95, -0.42, 0.0]},
        ],
        "bonds": [("O", "H1", 1), ("O", "H2", 1)],
    },
    "CO2": {
        "atoms": [
            {"id": "C", "element": "C", "pos": [0.0, 0.0, 0.0]},
            {"id": "O1", "element": "O", "pos": [-1.45, 0.0, 0.0]},
            {"id": "O2", "element": "O", "pos": [1.45, 0.0, 0.0]},
        ],
        "bonds": [("C", "O1", 2), ("C", "O2", 2)],
    },
    "CH4": {
        "atoms": [
            {"id": "C", "element": "C", "pos": [0.0, 0.0, 0.0]},
            {"id": "H1", "element": "H", "pos": [0.0, 1.15, 0.0]},
            {"id": "H2", "element": "H", "pos": [-1.08, -0.45, 0.0]},
            {"id": "H3", "element": "H", "pos": [1.08, -0.45, 0.0]},
            {"id": "H4", "element": "H", "pos": [0.0, -0.95, 0.0]},
        ],
        "bonds": [("C", "H1", 1), ("C", "H2", 1), ("C", "H3", 1), ("C", "H4", 1)],
    },
    "NH3": {
        "atoms": [
            {"id": "N", "element": "N", "pos": [0.0, 0.35, 0.0]},
            {"id": "H1", "element": "H", "pos": [-0.98, -0.45, 0.0]},
            {"id": "H2", "element": "H", "pos": [0.0, -0.65, 0.0]},
            {"id": "H3", "element": "H", "pos": [0.98, -0.45, 0.0]},
        ],
        "bonds": [("N", "H1", 1), ("N", "H2", 1), ("N", "H3", 1)],
    },
    "O2": {
        "atoms": [
            {"id": "O1", "element": "O", "pos": [-0.85, 0.0, 0.0]},
            {"id": "O2", "element": "O", "pos": [0.85, 0.0, 0.0]},
        ],
        "bonds": [("O1", "O2", 2)],
    },
    "N2": {
        "atoms": [
            {"id": "N1", "element": "N", "pos": [-0.85, 0.0, 0.0]},
            {"id": "N2", "element": "N", "pos": [0.85, 0.0, 0.0]},
        ],
        "bonds": [("N1", "N2", 3)],
    },
}


@VisualRegistry.register("chemistry", "MoleculeVisual")
class MoleculeVisual(VisualComponent):
    """
    Configurable visual component for 2D molecular structures and chemical reactions.
    Supports Lewis structures, ball-and-stick representations, multi-order bonds,
    bond cleavage, bond formation, and dipole polarization.
    """
    component_id = "MoleculeVisual"
    domain = "chemistry"
    aliases = ["molecule", "molecule diagram"]
    preferred_aspect = 1.4

    def __init__(
        self,
        atoms: Union[str, Sequence[Union[Dict[str, Any], Tuple[Any, ...], str]]] = "H2O",
        bonds: Optional[Sequence[Union[Tuple[Any, Any], Tuple[Any, Any, int], Dict[str, Any]]]] = None,
        atom_radius: float = 0.34,
        bond_width: float = 4.0,
        bond_color: str = "#DDDDDD",
        font_size: int = 18,
        svg_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.atom_radius = float(atom_radius)
        self.bond_width = float(bond_width)
        self.bond_color = bond_color
        self.font_size = font_size

        self.atoms_data: Dict[str, Dict[str, Any]] = {}
        self.bonds_data: List[Dict[str, Any]] = []

        # 1. Parse preset or user-specified structure
        self._parse_structure(atoms, bonds)

        # 2. Check for optional safe SVG
        if svg_path and self._try_load_svg(svg_path):
            return

        # 3. Native Manim vector rendering
        self.bonds_group = VGroup()
        self.atoms_group = VGroup()

        self._build_bonds()
        self._build_atoms()

        self.add(self.bonds_group, self.atoms_group)
        self.center()

    @property
    def atoms(self) -> List[Dict[str, Any]]:
        return list(self.atoms_data.values())

    @property
    def bonds(self) -> List[Dict[str, Any]]:
        return list(self.bonds_data)

    def _parse_structure(self, atoms: Any, bonds: Any) -> None:
        """Parses atomic coordinates and bond connectivity."""
        if isinstance(atoms, str) and atoms.upper() in MOLECULE_PRESETS:
            preset = MOLECULE_PRESETS[atoms.upper()]
            for item in preset["atoms"]:
                aid = str(item["id"])
                self.atoms_data[aid] = {
                    "id": aid,
                    "element": str(item["element"]).upper(),
                    "pos": np.array(item["pos"], dtype=float),
                }
            raw_bonds = preset["bonds"] if bonds is None else bonds
        else:
            raw_atoms = list(atoms) if not isinstance(atoms, str) else [atoms]
            for idx, a in enumerate(raw_atoms):
                if isinstance(a, dict):
                    aid = str(a.get("id", f"A{idx}"))
                    elem = str(a.get("element", aid)).upper()
                    pos = np.array(a.get("pos", [idx * 1.2, 0.0, 0.0]), dtype=float)
                    if len(pos) == 2:
                        pos = np.array([pos[0], pos[1], 0.0])
                elif isinstance(a, (tuple, list)):
                    if len(a) >= 3:
                        aid, elem, pos = str(a[0]), str(a[1]).upper(), np.array(a[2], dtype=float)
                    elif len(a) == 2:
                        aid, elem, pos = str(a[0]), str(a[0]).upper(), np.array(a[1], dtype=float)
                    else:
                        aid, elem, pos = f"A{idx}", str(a[0]).upper(), np.array([idx * 1.2, 0.0, 0.0])
                    if len(pos) == 2:
                        pos = np.array([pos[0], pos[1], 0.0])
                else:
                    elem = str(a).upper()
                    aid = f"{elem}{idx + 1}"
                    pos = np.array([idx * 1.2, 0.0, 0.0])
                self.atoms_data[aid] = {"id": aid, "element": elem, "pos": pos}
            raw_bonds = bonds or []

        # Parse bonds
        for b in (raw_bonds or []):
            if isinstance(b, (tuple, list)):
                u, v = str(b[0]), str(b[1])
                order = int(b[2]) if len(b) >= 3 else 1
            elif isinstance(b, dict):
                u, v = str(b.get("u")), str(b.get("v"))
                order = int(b.get("order", 1))
            else:
                continue
            self.bonds_data.append({"u": u, "v": v, "order": order})

    def _try_load_svg(self, svg_path: str) -> bool:
        """Attempts to safely load an external SVG asset with native fallback."""
        try:
            path_obj = Path(svg_path)
            if path_obj.is_file() and is_safe_svg(str(path_obj)):
                svg_mob = SVGMobject(str(path_obj))
                self.add(svg_mob)
                self.register_semantic_handle("molecule_svg", svg_mob)
                self.center()
                return True
        except Exception:
            pass
        return False

    def _get_element_color(self, element: str) -> str:
        """Returns standard CPK color for element symbol."""
        elem_clean = element.strip().upper()
        return CPK_COLORS.get(elem_clean, "#58C4DD")

    def _build_atoms(self) -> None:
        """Constructs Manim circles and element labels for each atom."""
        for aid, data in self.atoms_data.items():
            pos = data["pos"]
            elem = data["element"]
            elem_color = self._get_element_color(elem)

            # High-contrast label color
            txt_color = "#000000" if elem_color in ("#FFFFFF", "#FFDC00", "#7FDBFF") else "#FFFFFF"

            circle = Circle(
                radius=self.atom_radius,
                color=elem_color,
                fill_color=elem_color,
                fill_opacity=0.85,
                stroke_width=2.5,
            )
            circle.move_to(pos)

            label = Text(elem, font_size=self.font_size, color=txt_color, weight="BOLD")
            if label.width > self.atom_radius * 1.55:
                label.scale_to_fit_width(self.atom_radius * 1.55)
            label.move_to(circle.get_center())

            atom_group = VGroup(circle, label)
            self.atoms_group.add(atom_group)

            self.register_semantic_handle(f"atom_{aid}", atom_group)
            self.register_semantic_handle(f"circle_{aid}", circle)
            self.register_semantic_handle(f"symbol_{aid}", label)

    def _create_bond_geometry(
        self,
        pos_u: np.ndarray,
        pos_v: np.ndarray,
        order: int = 1,
        color: Optional[str] = None,
        width: Optional[float] = None,
    ) -> VGroup:
        """Generates single, double, or triple bond lines clipped to atomic perimeters."""
        diff = pos_v - pos_u
        dist = float(np.linalg.norm(diff))
        if dist < 1e-4:
            return VGroup()

        b_color = color or self.bond_color
        b_width = width or self.bond_width
        unit_vec = diff / dist
        normal = np.array([-unit_vec[1], unit_vec[0], 0.0])

        start_pt = pos_u + unit_vec * self.atom_radius
        end_pt = pos_v - unit_vec * self.atom_radius

        bond_group = VGroup()
        if order == 1:
            line = Line(start=start_pt, end=end_pt, color=b_color, stroke_width=b_width)
            bond_group.add(line)
        elif order == 2:
            offset = normal * 0.09
            l1 = Line(start=start_pt + offset, end=end_pt + offset, color=b_color, stroke_width=b_width * 0.85)
            l2 = Line(start=start_pt - offset, end=end_pt - offset, color=b_color, stroke_width=b_width * 0.85)
            bond_group.add(l1, l2)
        elif order >= 3:
            offset = normal * 0.13
            l1 = Line(start=start_pt + offset, end=end_pt + offset, color=b_color, stroke_width=b_width * 0.75)
            l2 = Line(start=start_pt, end=end_pt, color=b_color, stroke_width=b_width * 0.75)
            l3 = Line(start=start_pt - offset, end=end_pt - offset, color=b_color, stroke_width=b_width * 0.75)
            bond_group.add(l1, l2, l3)

        return bond_group

    def _build_bonds(self) -> None:
        """Constructs Manim lines for all bonds."""
        for b in self.bonds_data:
            u, v, order = b["u"], b["v"], b["order"]
            if u not in self.atoms_data or v not in self.atoms_data:
                continue

            pos_u = self.atoms_data[u]["pos"]
            pos_v = self.atoms_data[v]["pos"]

            bond_mob = self._create_bond_geometry(pos_u, pos_v, order=order)
            self.bonds_group.add(bond_mob)

            self.register_semantic_handle(f"bond_{u}_{v}", bond_mob)
            self.register_semantic_handle(f"bond_{v}_{u}", bond_mob)

    # ------------------------------------------------------------------------
    # Semantic Animation Methods
    # ------------------------------------------------------------------------
    def animate_highlight_atom(
        self,
        atom_id: str,
        color: str = "#FFD700",
        scale: float = 1.2,
        run_time: float = 0.6,
    ) -> Animation:
        """Highlights target atom with visual pulse."""
        atom = self.get_semantic_handle(f"atom_{atom_id}")
        if atom is None:
            return Indicate(self, color=color, run_time=run_time)
        return Indicate(atom, color=color, scale_factor=scale, run_time=run_time)

    def animate_highlight_bond(
        self,
        u: str,
        v: str,
        color: str = "#FFD700",
        run_time: float = 0.6,
    ) -> Animation:
        """Highlights target bond connecting atoms u and v."""
        bond = self.get_semantic_handle(f"bond_{u}_{v}")
        if bond is None:
            bond = self.get_semantic_handle(f"bond_{v}_{u}")
        if bond is None:
            return Indicate(self, color=color, run_time=run_time)
        return Indicate(bond, color=color, run_time=run_time)

    def animate_break_bond(
        self,
        u: str,
        v: str,
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Animates chemical bond cleavage (dissolving/fading out)."""
        bond = self.get_semantic_handle(f"bond_{u}_{v}")
        if bond is None:
            bond = self.get_semantic_handle(f"bond_{v}_{u}")
        if bond is None:
            return AnimationGroup(Indicate(self, run_time=run_time))

        atom_u = self.get_semantic_handle(f"atom_{u}")
        atom_v = self.get_semantic_handle(f"atom_{v}")

        anims = [
            FadeOut(bond, run_time=run_time),
            Indicate(bond, color="#FF6B6B", run_time=run_time * 0.5),
        ]
        if atom_u:
            anims.append(Wiggle(atom_u, run_time=run_time * 0.6))
        if atom_v:
            anims.append(Wiggle(atom_v, run_time=run_time * 0.6))

        return AnimationGroup(*anims)

    def animate_form_bond(
        self,
        u: str,
        v: str,
        order: int = 1,
        color: str = "#83C167",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Dynamically generates and animates formation of a new chemical bond."""
        if u not in self.atoms_data or v not in self.atoms_data:
            return AnimationGroup(Indicate(self, run_time=run_time))

        pos_u = self.atoms_data[u]["pos"]
        pos_v = self.atoms_data[v]["pos"]
        new_bond = self._create_bond_geometry(pos_u, pos_v, order=order, color=color)
        self.bonds_group.add(new_bond)

        self.register_semantic_handle(f"bond_{u}_{v}", new_bond)
        self.register_semantic_handle(f"bond_{v}_{u}", new_bond)

        return AnimationGroup(
            FadeIn(new_bond, run_time=run_time),
            Indicate(new_bond, color=color, run_time=run_time),
            lag_ratio=0.3,
        )

    def animate_vibrate_bond(
        self,
        u: str,
        v: str,
        run_time: float = 0.6,
    ) -> AnimationGroup:
        """Simulates infrared thermal stretching vibration along a bond."""
        bond = self.get_semantic_handle(f"bond_{u}_{v}")
        if bond is None:
            bond = self.get_semantic_handle(f"bond_{v}_{u}")
        if bond is None:
            return AnimationGroup(Indicate(self, run_time=run_time))

        return AnimationGroup(
            Wiggle(bond, scale_value=1.15, rotation_angle=0.04, run_time=run_time)
        )

    def animate_polarize(
        self,
        u: str,
        v: str,
        partial_charges: Tuple[str, str] = ("δ+", "δ-"),
        color: str = "#FFA07A",
        run_time: float = 0.8,
    ) -> AnimationGroup:
        """Visualizes bond dipole and partial electronegativity charges (e.g. δ+ on H, δ- on O)."""
        atom_u = self.get_semantic_handle(f"atom_{u}")
        atom_v = self.get_semantic_handle(f"atom_{v}")
        if atom_u is None or atom_v is None:
            return AnimationGroup(Indicate(self, run_time=run_time))

        q_u = Text(partial_charges[0], font_size=14, color=color).next_to(atom_u, UP, buff=0.1)
        q_v = Text(partial_charges[1], font_size=14, color=color).next_to(atom_v, UP, buff=0.1)

        self.add(q_u, q_v)
        self.register_semantic_handle(f"charge_{u}", q_u)
        self.register_semantic_handle(f"charge_{v}", q_v)

        return AnimationGroup(
            FadeIn(q_u, run_time=run_time),
            FadeIn(q_v, run_time=run_time),
            Indicate(atom_u, color=color, run_time=run_time),
            Indicate(atom_v, color=color, run_time=run_time),
        )
