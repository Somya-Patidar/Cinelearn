"""
Tests for Chemistry Domain Visual Components
============================================
Validates MoleculeVisual and ReactionEnergyDiagram functionality, semantic handles,
animation methods, container docking, zero overlap, and subtitle clearance.
"""

import sys
from pathlib import Path

# Ensure CineLearn root is in sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
import numpy as np
from manim import Scene, tempconfig, FadeIn

from src.domains import VisualRegistry
from src.domains.chemistry import MoleculeVisual, ReactionEnergyDiagram
from prompts import base_class

# Dynamically extract TeachingScene from base_class template
_scope = {}
exec(base_class.base_class if hasattr(base_class, "base_class") else base_class, _scope)
TeachingScene = _scope["TeachingScene"]


@pytest.fixture(autouse=True)
def ensure_chemistry_discovered():
    VisualRegistry.discover_domains("src.domains")


class TestMoleculeVisual:
    def test_registration(self):
        cls = VisualRegistry.get("chemistry", "MoleculeVisual")
        assert cls is MoleculeVisual
        manifest = MoleculeVisual.get_manifest()
        assert manifest["component_id"] == "MoleculeVisual"
        assert manifest["domain"] == "chemistry"
        assert "animate_highlight_atom" in manifest["semantic_actions"]
        assert "animate_break_bond" in manifest["semantic_actions"]
        assert "animate_form_bond" in manifest["semantic_actions"]

    def test_preset_molecules(self):
        # 1. H2O (Water)
        water = MoleculeVisual("H2O")
        assert water.get_semantic_handle("atom_O") is not None
        assert water.get_semantic_handle("atom_H1") is not None
        assert water.get_semantic_handle("atom_H2") is not None
        assert water.get_semantic_handle("bond_O_H1") is not None
        assert water.get_semantic_handle("bond_O_H2") is not None

        # 2. CO2 (Carbon Dioxide - double bonds)
        co2 = MoleculeVisual("CO2")
        assert co2.get_semantic_handle("atom_C") is not None
        assert co2.get_semantic_handle("atom_O1") is not None
        assert co2.get_semantic_handle("atom_O2") is not None
        assert co2.get_semantic_handle("bond_C_O1") is not None
        # Double bond has 2 line submobjects
        assert len(co2.get_semantic_handle("bond_C_O1").submobjects) == 2

        # 3. N2 (Nitrogen gas - triple bond)
        n2 = MoleculeVisual("N2")
        assert n2.get_semantic_handle("atom_N1") is not None
        assert n2.get_semantic_handle("atom_N2") is not None
        assert len(n2.get_semantic_handle("bond_N1_N2").submobjects) == 3

    def test_custom_molecular_structure(self):
        # Custom Ethanol fragment (C-C-O)
        custom_atoms = [
            {"id": "C1", "element": "C", "pos": [-1.2, 0.0, 0.0]},
            {"id": "C2", "element": "C", "pos": [0.0, 0.0, 0.0]},
            {"id": "O1", "element": "O", "pos": [1.2, 0.4, 0.0]},
            {"id": "H1", "element": "H", "pos": [1.9, -0.2, 0.0]},
        ]
        custom_bonds = [
            ("C1", "C2", 1),
            ("C2", "O1", 1),
            ("O1", "H1", 1),
        ]
        mol = MoleculeVisual(atoms=custom_atoms, bonds=custom_bonds)

        assert mol.get_semantic_handle("atom_C1") is not None
        assert mol.get_semantic_handle("atom_C2") is not None
        assert mol.get_semantic_handle("atom_O1") is not None
        assert mol.get_semantic_handle("atom_H1") is not None
        assert mol.get_semantic_handle("bond_C1_C2") is not None
        assert mol.get_semantic_handle("bond_C2_O1") is not None
        assert mol.get_semantic_handle("bond_O1_H1") is not None

    def test_semantic_animations(self):
        mol = MoleculeVisual("H2O")

        anim_atom = mol.animate_highlight_atom("O")
        assert anim_atom is not None

        anim_bond = mol.animate_highlight_bond("O", "H1")
        assert anim_bond is not None

        anim_break = mol.animate_break_bond("O", "H1")
        assert anim_break is not None

        anim_form = mol.animate_form_bond("O", "H1", order=1)
        assert anim_form is not None

        anim_vib = mol.animate_vibrate_bond("O", "H2")
        assert anim_vib is not None

        anim_pol = mol.animate_polarize("H1", "O", partial_charges=("δ+", "δ-"))
        assert anim_pol is not None

    def test_svg_fallback_on_unsafe_content(self, tmp_path):
        unsafe_svg = tmp_path / "xss.svg"
        unsafe_svg.write_text('<svg><script>alert(1)</script><circle r="10"/></svg>', encoding="utf-8")
        # Unsafe SVG must be rejected by is_safe_svg and fall back to native Manim rendering
        mol = MoleculeVisual(atoms="H2O", svg_path=str(unsafe_svg))
        assert mol.get_semantic_handle("molecule_svg") is None
        assert mol.get_semantic_handle("atom_O") is not None  # Fallback native atom exists


class TestReactionEnergyDiagram:
    def test_registration(self):
        cls = VisualRegistry.get("chemistry", "ReactionEnergyDiagram")
        assert cls is ReactionEnergyDiagram
        manifest = ReactionEnergyDiagram.get_manifest()
        assert manifest["component_id"] == "ReactionEnergyDiagram"
        assert manifest["domain"] == "chemistry"
        assert "animate_progress_reaction" in manifest["semantic_actions"]
        assert "animate_add_catalyst" in manifest["semantic_actions"]

    def test_exothermic_profile(self):
        # Exothermic: reactants (1.5) > products (0.7), TS at 3.5
        diag = ReactionEnergyDiagram(
            e_reactants=1.5,
            e_transition_state=3.5,
            e_products=0.7,
            catalyzed_e_ts=2.3,
        )

        assert diag.get_semantic_handle("axes") is not None
        assert diag.get_semantic_handle("uncatalyzed_curve") is not None
        assert diag.get_semantic_handle("catalyzed_curve") is not None
        assert diag.get_semantic_handle("particle") is not None
        assert diag.get_semantic_handle("ea_arrow") is not None
        assert diag.get_semantic_handle("delta_h_arrow") is not None
        assert diag.get_semantic_handle("reactants_label") is not None
        assert diag.get_semantic_handle("products_label") is not None
        assert diag.get_semantic_handle("ts_label") is not None

    def test_endothermic_profile(self):
        # Endothermic: reactants (1.0) < products (2.4), TS at 3.8
        diag = ReactionEnergyDiagram(
            e_reactants=1.0,
            e_transition_state=3.8,
            e_products=2.4,
            catalyzed_e_ts=None,
        )

        assert diag.get_semantic_handle("catalyzed_curve") is None
        assert diag.get_semantic_handle("delta_h_label") is not None

    def test_semantic_animations(self):
        diag = ReactionEnergyDiagram(
            e_reactants=1.5,
            e_transition_state=3.5,
            e_products=0.8,
            catalyzed_e_ts=2.4,
        )

        anim_prog = diag.animate_progress_reaction()
        assert anim_prog is not None

        anim_ts = diag.animate_highlight_transition_state()
        assert anim_ts is not None

        anim_ea = diag.animate_show_activation_barrier()
        assert anim_ea is not None

        anim_dh = diag.animate_show_enthalpy()
        assert anim_dh is not None

        anim_cat = diag.animate_add_catalyst()
        assert anim_cat is not None


class TestChemistryDockingAndRendering:
    def test_diagram_and_molecule_composition_docking(self):
        """
        Tests composing ReactionEnergyDiagram in dock_main and MoleculeVisual in dock_inspector,
        verifying zero overlap and subtitle clearance.
        """
        scene = TeachingScene()
        scene.setup_layout(
            "Chemical Kinetics: Activation Energy & Catalysis",
            [
                "Reactants must overcome the activation barrier Ea",
                "Catalysts lower the barrier without altering reaction enthalpy",
                "Product molecules form as the system reaches lower potential energy",
            ],
        )

        # 1. Main visual: ReactionEnergyDiagram
        energy_diag = ReactionEnergyDiagram(
            e_reactants=1.5,
            e_transition_state=3.6,
            e_products=0.8,
            catalyzed_e_ts=2.4,
        )
        scene.dock_main(energy_diag)

        # 2. Inspector visual: MoleculeVisual (showing the formed Water molecule)
        mol = MoleculeVisual("H2O")
        scene.dock_inspector(mol)

        # Verify zero overlap between diagram and molecule
        bb_diag = energy_diag.get_bounding_box()
        bb_mol = mol.get_bounding_box()

        has_overlap, ratio = scene.layout.check_overlap(energy_diag, mol)
        assert not has_overlap, f"Overlap detected between diagram and molecule! Ratio: {ratio}"
        assert ratio == 0.0

        # Verify vertical separation: diagram bottom is above inspector molecule top
        assert bb_diag["bottom"] > bb_mol["top"], (
            f"Diagram bottom ({bb_diag['bottom']}) must be strictly above molecule top ({bb_mol['top']})"
        )

        # Verify subtitle clearance: bottom of molecule must be >= -2.65
        assert bb_mol["bottom"] >= -2.65, (
            f"Molecule bottom ({bb_mol['bottom']}) encroaches on subtitle area (below -2.65)"
        )

        # Verify safe boundaries
        assert bb_diag["left"] >= -2.15
        assert bb_mol["left"] >= -2.15
        assert bb_diag["right"] <= 6.65
        assert bb_mol["right"] <= 6.65

    def test_render_scene_with_chemistry_components(self, tmp_path):
        """
        Renders a Manim scene exercising ReactionEnergyDiagram and MoleculeVisual animations.
        """
        class SampleChemistryScene(TeachingScene):
            def construct(self):
                self.setup_layout(
                    "Exothermic Reaction Mechanism",
                    [
                        "Reactants absorb energy to reach transition state",
                        "Overcoming the barrier yields products with released enthalpy",
                    ],
                )

                diag = ReactionEnergyDiagram(
                    e_reactants=1.4,
                    e_transition_state=3.5,
                    e_products=0.7,
                    catalyzed_e_ts=2.3,
                )
                self.dock_main(diag)

                mol = MoleculeVisual("H2O")
                self.dock_inspector(mol)

                self.play(FadeIn(diag), FadeIn(mol))
                self.play(self.activate_caption(0))
                self.play(diag.animate_show_activation_barrier())
                self.play(diag.animate_progress_reaction(run_time=1.0))
                self.play(self.activate_caption(1))
                self.play(
                    diag.animate_show_enthalpy(),
                    mol.animate_highlight_atom("O"),
                )

        with tempconfig({
            "quality": "low_quality",
            "preview": False,
            "format": "mp4",
            "media_dir": str(tmp_path),
            "disable_caching": True,
        }):
            scene = SampleChemistryScene()
            scene.render()

        rendered_files = list(tmp_path.glob("**/*.mp4"))
        assert len(rendered_files) > 0, "Expected at least one rendered MP4 file"
        assert rendered_files[0].stat().st_size > 1000, "Rendered video file is too small or corrupt"
