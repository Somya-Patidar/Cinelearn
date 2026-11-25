"""
CineLearn — Canonical Domain Scene IR Fixtures.
===============================================

Provides the three verified domain-independent reference fixtures from
docs/architecture/scene_ir_and_spatial_composition_specification.md Section 3.3:
1. Computer Science / Algorithms: Binary Search Convergence
2. Natural Sciences / Chemistry: Arrhenius Activation Energy
3. Mathematics: Geometric Proof of the Pythagorean Theorem
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from .models import SceneIR


# ============================================================================
# Example 1: Binary Search Convergence
# ============================================================================

BINARY_SEARCH_RAW_FIXTURE: Dict[str, Any] = {
    "$schema": "https://cinelearn.org/schemas/scene_ir_v1.json",
    "version": "1.0",
    "scene_id": "Section3Scene",
    "section_index": 3,
    "title": "Narrowing the Search Interval",
    "duration_budget": 14.5,
    "audio_reference": {
        "audio_path": "sec_3_audio.mp3",
        "timestamps_path": "sec_3_timestamps.json",
        "total_duration": 14.2,
    },
    "layout_configuration": {
        "aspect_ratio": "16:9",
        "has_title_banner": True,
        "has_caption_column": True,
        "has_subtitles": True,
        "safety_padding": 0.20,
    },
    "viewports": {
        "array_viewport": {
            "type": "viewport_main",
            "flow": "column",
            "alignment": "center",
            "padding": 0.20,
            "sub_slots": [],
        },
        "inspector_viewport": {
            "type": "viewport_inspector",
            "flow": "row",
            "alignment": "space_between",
            "padding": 0.15,
            "sub_slots": [],
        },
    },
    "inherited_entities": [],
    "entities": [
        {
            "id": "search_array",
            "component_type": "CompositeNodeGraph",
            "visual_tier": "tier3_primitive_tree",
            "viewport": "array_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 1,
            "initial_state": {
                "elements": [
                    {"index": 0, "val": "2", "state": "inactive"},
                    {"index": 1, "val": "5", "state": "inactive"},
                    {"index": 2, "val": "8", "state": "active"},
                    {"index": 3, "val": "12", "state": "active"},
                    {"index": 4, "val": "16", "state": "midpoint"},
                    {"index": 5, "val": "23", "state": "active"},
                    {"index": 6, "val": "38", "state": "inactive"},
                ]
            },
        },
        {
            "id": "pointer_low",
            "component_type": "PointerIndicator",
            "visual_tier": "tier3_primitive_tree",
            "viewport": "array_viewport",
            "slot": None,
            "parent_id": "search_array",
            "z_index": 15,
            "visual_priority": 1,
            "initial_state": {"target_index": 2, "label": "low"},
        },
        {
            "id": "pointer_high",
            "component_type": "PointerIndicator",
            "visual_tier": "tier3_primitive_tree",
            "viewport": "array_viewport",
            "slot": None,
            "parent_id": "search_array",
            "z_index": 15,
            "visual_priority": 1,
            "initial_state": {"target_index": 5, "label": "high"},
        },
        {
            "id": "formula_card",
            "component_type": "ConceptualCard",
            "visual_tier": "tier4_conceptual_card",
            "viewport": "inspector_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 2,
            "initial_state": {
                "title": "Midpoint Calculation",
                "formula": "mid = \\lfloor \\frac{low + high}{2} \\rfloor = \\lfloor \\frac{2 + 5}{2} \\rfloor = 3",
                "badge": "INVARIANT",
            },
        },
    ],
    "timeline": [
        {
            "step_id": "step_1_eval_mid",
            "narration_anchor": {
                "lecture_line_index": 0,
                "text_cue": "We compute the midpoint index between low and high.",
                "audio_timestamp_range": [0.0, 5.2],
            },
            "actions": [
                {"type": "enter", "target_entity": "search_array", "duration": 0.8},
                {"type": "enter", "target_entity": "pointer_low", "duration": 0.6},
                {"type": "enter", "target_entity": "pointer_high", "duration": 0.6},
                {"type": "enter", "target_entity": "formula_card", "duration": 0.8},
            ],
        },
        {
            "step_id": "step_2_compare_target",
            "narration_anchor": {
                "lecture_line_index": 1,
                "text_cue": "Since 23 is greater than the target 12, we discard the right half.",
                "audio_timestamp_range": [5.2, 14.2],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "search_array",
                    "operation": "dim_subrange",
                    "parameters": {"start_idx": 4, "end_idx": 6},
                    "duration": 0.8,
                },
                {
                    "type": "move",
                    "target_entity": "pointer_high",
                    "operation": None,
                    "parameters": {"new_target_index": 3},
                    "duration": 1.0,
                },
                {
                    "type": "transform",
                    "target_entity": "formula_card",
                    "operation": None,
                    "parameters": {
                        "formula": "high = mid - 1 = 4 - 1 = 3",
                        "badge": "INTERVAL_HALVED",
                    },
                    "duration": 0.8,
                },
            ],
        },
    ],
}


# ============================================================================
# Example 2: Chemical Kinetics — Arrhenius Activation Energy
# ============================================================================

CHEMICAL_KINETICS_RAW_FIXTURE: Dict[str, Any] = {
    "$schema": "https://cinelearn.org/schemas/scene_ir_v1.json",
    "version": "1.0",
    "scene_id": "Section2Scene",
    "section_index": 2,
    "title": "Energy Barrier and Activation Energy",
    "duration_budget": 16.0,
    "audio_reference": {
        "audio_path": "sec_2_audio.mp3",
        "timestamps_path": "sec_2_timestamps.json",
        "total_duration": 15.6,
    },
    "layout_configuration": {
        "aspect_ratio": "16:9",
        "has_title_banner": True,
        "has_caption_column": True,
        "has_subtitles": True,
        "safety_padding": 0.20,
    },
    "viewports": {
        "curve_viewport": {
            "type": "viewport_main",
            "flow": "single",
            "alignment": "center",
            "padding": 0.20,
            "sub_slots": [],
        },
        "equation_viewport": {
            "type": "viewport_inspector",
            "flow": "single",
            "alignment": "center",
            "padding": 0.10,
            "sub_slots": [],
        },
    },
    "inherited_entities": [],
    "entities": [
        {
            "id": "energy_diagram",
            "component_type": "ReactionEnergyDiagram",
            "visual_tier": "tier1_registered",
            "viewport": "curve_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 1,
            "initial_state": {
                "reactant_energy": 2.0,
                "product_energy": 0.8,
                "barrier_peak": 5.5,
                "reaction_label": "Exothermic Pathway",
            },
        },
        {
            "id": "arrhenius_equation",
            "component_type": "ConceptualCard",
            "visual_tier": "tier4_conceptual_card",
            "viewport": "equation_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 2,
            "initial_state": {
                "title": "Arrhenius Governing Equation",
                "formula": "k = A \\, e^{-\\frac{E_a}{R T}}",
                "badge": "GOVERNING_LAW",
            },
        },
    ],
    "timeline": [
        {
            "step_id": "step_1_show_profile",
            "narration_anchor": {
                "lecture_line_index": 0,
                "text_cue": "Molecules must overcome a distinct potential energy barrier to react.",
                "audio_timestamp_range": [0.0, 6.5],
            },
            "actions": [
                {"type": "enter", "target_entity": "energy_diagram", "duration": 1.2},
                {"type": "enter", "target_entity": "arrhenius_equation", "duration": 0.8},
            ],
        },
        {
            "step_id": "step_2_highlight_ea",
            "narration_anchor": {
                "lecture_line_index": 1,
                "text_cue": "The difference between reactant energy and the transition state peak is the activation energy, Ea.",
                "audio_timestamp_range": [6.5, 15.6],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "energy_diagram",
                    "operation": "animate_highlight_activation_barrier",
                    "parameters": {"color": "#FFD700", "label": "E_a"},
                    "duration": 1.4,
                },
                {
                    "type": "highlight",
                    "target_entity": "arrhenius_equation",
                    "operation": None,
                    "parameters": {"subterm": "E_a", "color": "#FFD700"},
                    "duration": 0.8,
                },
            ],
        },
    ],
}


# ============================================================================
# Example 3: Mathematics — Geometric Proof of the Pythagorean Theorem
# ============================================================================

PYTHAGOREAN_THEOREM_RAW_FIXTURE: Dict[str, Any] = {
    "$schema": "https://cinelearn.org/schemas/scene_ir_v1.json",
    "version": "1.0",
    "scene_id": "Section4Scene",
    "section_index": 4,
    "title": "Area Equivalence of the Dissected Squares",
    "duration_budget": 17.5,
    "audio_reference": {
        "audio_path": "sec_4_audio.mp3",
        "timestamps_path": "sec_4_timestamps.json",
        "total_duration": 17.0,
    },
    "layout_configuration": {
        "aspect_ratio": "16:9",
        "has_title_banner": True,
        "has_caption_column": True,
        "has_subtitles": True,
        "safety_padding": 0.20,
    },
    "viewports": {
        "left_square_vp": {
            "type": "viewport_split_left",
            "flow": "single",
            "alignment": "center",
            "padding": 0.15,
            "sub_slots": [],
        },
        "right_square_vp": {
            "type": "viewport_split_right",
            "flow": "single",
            "alignment": "center",
            "padding": 0.15,
            "sub_slots": [],
        },
        "proof_summary_vp": {
            "type": "viewport_inspector",
            "flow": "single",
            "alignment": "center",
            "padding": 0.10,
            "sub_slots": [],
        },
    },
    "inherited_entities": [],
    "entities": [
        {
            "id": "square_a_plus_b_left",
            "component_type": "CompositeDissectionSquare",
            "visual_tier": "tier3_primitive_tree",
            "viewport": "left_square_vp",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 1,
            "initial_state": {
                "side_length": 4.0,
                "configuration": "four_triangles_plus_c_square",
                "labels": {"side_a": "a", "side_b": "b", "inner_square": "c^2"},
            },
        },
        {
            "id": "square_a_plus_b_right",
            "component_type": "CompositeDissectionSquare",
            "visual_tier": "tier3_primitive_tree",
            "viewport": "right_square_vp",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 1,
            "initial_state": {
                "side_length": 4.0,
                "configuration": "four_triangles_plus_a2_and_b2",
                "labels": {"square_a": "a^2", "square_b": "b^2"},
            },
        },
        {
            "id": "algebraic_derivation",
            "component_type": "ConceptualCard",
            "visual_tier": "tier4_conceptual_card",
            "viewport": "proof_summary_vp",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 2,
            "initial_state": {
                "title": "Algebraic Equivalence",
                "formula": "c^2 + 4\\left(\\frac{1}{2}ab\\right) = a^2 + b^2 + 4\\left(\\frac{1}{2}ab\\right) \\implies a^2 + b^2 = c^2",
                "badge": "Q.E.D.",
            },
        },
    ],
    "timeline": [
        {
            "step_id": "step_1_present_dissections",
            "narration_anchor": {
                "lecture_line_index": 0,
                "text_cue": "Both large squares have identical total area (a + b) squared.",
                "audio_timestamp_range": [0.0, 7.5],
            },
            "actions": [
                {"type": "enter", "target_entity": "square_a_plus_b_left", "duration": 1.0},
                {"type": "enter", "target_entity": "square_a_plus_b_right", "duration": 1.0},
            ],
        },
        {
            "step_id": "step_2_subtract_triangles",
            "narration_anchor": {
                "lecture_line_index": 1,
                "text_cue": "Subtracting the four identical triangles from both sides leaves c squared equal to a squared plus b squared.",
                "audio_timestamp_range": [7.5, 17.0],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "square_a_plus_b_left",
                    "operation": "dim_triangles",
                    "parameters": {"opacity": 0.2},
                    "duration": 1.0,
                },
                {
                    "type": "component_op",
                    "target_entity": "square_a_plus_b_right",
                    "operation": "dim_triangles",
                    "parameters": {"opacity": 0.2},
                    "duration": 1.0,
                },
                {
                    "type": "enter",
                    "target_entity": "algebraic_derivation",
                    "duration": 1.2,
                },
            ],
        },
    ],
}


def get_binary_search_fixture_dict() -> Dict[str, Any]:
    """Returns a deep-copied raw dictionary of the canonical Binary Search fixture."""
    return copy.deepcopy(BINARY_SEARCH_RAW_FIXTURE)


def get_binary_search_fixture() -> SceneIR:
    """Returns a typed SceneIR model of the canonical Binary Search fixture."""
    return SceneIR.from_dict(get_binary_search_fixture_dict())


def get_chemical_kinetics_fixture_dict() -> Dict[str, Any]:
    """Returns a deep-copied raw dictionary of the canonical Chemical Kinetics fixture."""
    return copy.deepcopy(CHEMICAL_KINETICS_RAW_FIXTURE)


def get_chemical_kinetics_fixture() -> SceneIR:
    """Returns a typed SceneIR model of the canonical Chemical Kinetics fixture."""
    return SceneIR.from_dict(get_chemical_kinetics_fixture_dict())


def get_pythagorean_theorem_fixture_dict() -> Dict[str, Any]:
    """Returns a deep-copied raw dictionary of the canonical Pythagorean Theorem fixture."""
    return copy.deepcopy(PYTHAGOREAN_THEOREM_RAW_FIXTURE)


def get_pythagorean_theorem_fixture() -> SceneIR:
    """Returns a typed SceneIR model of the canonical Pythagorean Theorem fixture."""
    return SceneIR.from_dict(get_pythagorean_theorem_fixture_dict())


# ============================================================================
# Example 4: Operating Systems Scheduling (Shortest Job First)
# ============================================================================

SJF_RAW_FIXTURE: Dict[str, Any] = {
    "$schema": "https://cinelearn.org/schemas/scene_ir_v1.json",
    "version": "1.0",
    "scene_id": "SJFScene",
    "section_index": 1,
    "title": "Shortest Job First (SJF) CPU Scheduling",
    # Duration budget calibrated to 42.0s (FIND-5.3.4.2-01 remediation):
    # - 5 sequential narration cues containing ~84 words (~33-35s spoken duration).
    # - 4 inter-cue pauses (4 x 0.5s = 2.0s) + 1 scene-end pause (1.0s) add 3.0s pacing.
    # - Expected speech + pacing timeline is ~38.0s; 42.0s provides ~10% modest headroom
    #   without being excessively loose.
    "duration_budget": 42.0,
    "audio_reference": {
        "audio_path": "sec_1_sjf_audio.mp3",
        "timestamps_path": "sec_1_sjf_timestamps.json",
        "total_duration": 25.0,
    },
    "layout_configuration": {
        "aspect_ratio": "16:9",
        "has_title_banner": True,
        "has_caption_column": True,
        "has_subtitles": True,
        "safety_padding": 0.20,
    },
    "viewports": {
        "table_viewport": {
            "type": "viewport_split_left",
            "flow": "column",
            "alignment": "center",
            "padding": 0.15,
            "sub_slots": [],
        },
        "gantt_viewport": {
            "type": "viewport_split_right",
            "flow": "column",
            "alignment": "center",
            "padding": 0.15,
            "sub_slots": [],
        },
        "decision_viewport": {
            "type": "viewport_inspector",
            "flow": "row",
            "alignment": "space_between",
            "padding": 0.15,
            "sub_slots": [],
        },
    },
    "inherited_entities": [],
    "entities": [
        {
            "id": "metric_table",
            "component_type": "MetricTableVisual",
            "visual_tier": "tier1_registered",
            "viewport": "table_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 1,
            "initial_state": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 7},
                    {"pid": "P2", "arrival_time": 2, "burst_time": 4},
                    {"pid": "P3", "arrival_time": 4, "burst_time": 1},
                    {"pid": "P4", "arrival_time": 5, "burst_time": 4},
                ],
                "width": 3.8,
                "height": 2.0,
            },
        },
        {
            "id": "gantt",
            "component_type": "GanttChartVisual",
            "visual_tier": "tier1_registered",
            "viewport": "gantt_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 1,
            "initial_state": {
                "total_time": 16,
                "width": 3.7,
                "height": 1.0,
            },
        },
        {
            "id": "decision_card",
            "component_type": "ConceptualCard",
            "visual_tier": "tier4_conceptual_card",
            "viewport": "decision_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 2,
            "initial_state": {
                "title": "Shortest Job First (Non-Preemptive)",
                "formula": r"\min_{P_i \in Ready} \{ BT(P_i) \}",
                "badge": "SELECTION RULE",
            },
        },
    ],
    "timeline": [
        {
            "step_id": "step_1_init_and_p1_arrival",
            "narration_anchor": {
                "lecture_line_index": 0,
                "text_cue": "In non-preemptive SJF, CPU is allocated to shortest job. At time 0, only P1 is available.",
                "audio_timestamp_range": [0.0, 5.0],
            },
            "actions": [
                {"type": "enter", "target_entity": "metric_table", "duration": 0.8},
                {"type": "enter", "target_entity": "gantt", "duration": 0.8},
                {"type": "enter", "target_entity": "decision_card", "duration": 0.8},
                {
                    "type": "component_op",
                    "target_entity": "gantt",
                    "operation": "animate_add_block",
                    "parameters": {"pid": "P1", "start_time": 0, "end_time": 7},
                    "duration": 0.8,
                },
            ],
        },
        {
            "step_id": "step_2_p1_completion_and_p3_selection",
            "narration_anchor": {
                "lecture_line_index": 1,
                "text_cue": "P1 runs until time 7. P2, P3, and P4 have arrived. P3 has shortest burst time of 1.",
                "audio_timestamp_range": [5.0, 10.0],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "metric_table",
                    "operation": "animate_update_process_metrics",
                    "parameters": {"pid": "P1", "ct": 7, "tat": 7, "wt": 0},
                    "duration": 0.8,
                },
                {
                    "type": "transform",
                    "target_entity": "decision_card",
                    "parameters": {
                        "formula": r"Ready at t=7: \{P_2(4), P_3(1), P_4(4)\} \implies P_3",
                        "badge": "P3 SELECTED (BT=1)",
                    },
                    "duration": 0.8,
                },
                {
                    "type": "component_op",
                    "target_entity": "gantt",
                    "operation": "animate_add_block",
                    "parameters": {"pid": "P3", "start_time": 7, "end_time": 8},
                    "duration": 0.8,
                },
            ],
        },
        {
            "step_id": "step_3_p3_completion_and_tie_breaking",
            "narration_anchor": {
                "lecture_line_index": 2,
                "text_cue": "P3 completes at time 8. P2 and P4 tie with burst time 4. Tie is broken by arrival time: P2 arrived earlier.",
                "audio_timestamp_range": [10.0, 15.0],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "metric_table",
                    "operation": "animate_update_process_metrics",
                    "parameters": {"pid": "P3", "ct": 8, "tat": 4, "wt": 3},
                    "duration": 0.8,
                },
                {
                    "type": "transform",
                    "target_entity": "decision_card",
                    "parameters": {
                        "formula": r"Tie: P_2(BT=4, AT=2) \text{ vs } P_4(BT=4, AT=5) \implies P_2",
                        "badge": "TIE-BREAK BY ARRIVAL",
                    },
                    "duration": 0.8,
                },
                {
                    "type": "component_op",
                    "target_entity": "gantt",
                    "operation": "animate_add_block",
                    "parameters": {"pid": "P2", "start_time": 8, "end_time": 12},
                    "duration": 0.8,
                },
            ],
        },
        {
            "step_id": "step_4_p2_completion_and_p4_dispatch",
            "narration_anchor": {
                "lecture_line_index": 3,
                "text_cue": "P2 finishes at time 12. P4 is dispatched and runs until time 16.",
                "audio_timestamp_range": [15.0, 20.0],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "metric_table",
                    "operation": "animate_update_process_metrics",
                    "parameters": {"pid": "P2", "ct": 12, "tat": 10, "wt": 6},
                    "duration": 0.8,
                },
                {
                    "type": "transform",
                    "target_entity": "decision_card",
                    "parameters": {
                        "formula": r"P_4 \text{ executes: } [12, 16], \quad WT = 7, \; TAT = 11",
                        "badge": "FINAL JOB DISPATCH",
                    },
                    "duration": 0.8,
                },
                {
                    "type": "component_op",
                    "target_entity": "gantt",
                    "operation": "animate_add_block",
                    "parameters": {"pid": "P4", "start_time": 12, "end_time": 16},
                    "duration": 0.8,
                },
                {
                    "type": "component_op",
                    "target_entity": "metric_table",
                    "operation": "animate_update_process_metrics",
                    "parameters": {"pid": "P4", "ct": 16, "tat": 11, "wt": 7},
                    "duration": 0.8,
                },
            ],
        },
        {
            "step_id": "step_5_average_metrics_and_summary",
            "narration_anchor": {
                "lecture_line_index": 4,
                "text_cue": "All jobs complete. Average waiting time is 4.0, and average turnaround time is 8.0.",
                "audio_timestamp_range": [20.0, 25.0],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "metric_table",
                    "operation": "animate_show_averages",
                    "parameters": {"avg_tat": 8.0, "avg_wt": 4.0},
                    "duration": 0.8,
                },
                {
                    "type": "transform",
                    "target_entity": "decision_card",
                    "parameters": {
                        "formula": r"Avg WT = \frac{0+6+3+7}{4} = 4.0, \quad Avg TAT = \frac{7+10+4+11}{4} = 8.0",
                        "badge": "SJF COMPLETE",
                    },
                    "duration": 0.8,
                },
            ],
        },
    ],
}


def get_sjf_scheduling_fixture_dict() -> Dict[str, Any]:
    """Returns a deep-copied raw dictionary of the canonical Non-Preemptive SJF Scheduling fixture."""
    return copy.deepcopy(SJF_RAW_FIXTURE)


def get_sjf_scheduling_fixture() -> SceneIR:
    """Returns a typed SceneIR model of the canonical Non-Preemptive SJF Scheduling fixture."""
    return SceneIR.from_dict(get_sjf_scheduling_fixture_dict())


# ============================================================================
# Example 5: First-Come, First-Served (FCFS) CPU Scheduling (Canonical Fixture)
# ============================================================================

FCFS_RAW_FIXTURE: Dict[str, Any] = {
    "$schema": "https://cinelearn.org/schemas/scene_ir_v1.json",
    "version": "1.0",
    "scene_id": "FCFSScene",
    "section_index": 1,
    "title": "First-Come, First-Served (FCFS) CPU Scheduling",
    "duration_budget": 35.0,
    "audio_reference": {
        "audio_path": "sec_1_fcfs_audio.mp3",
        "timestamps_path": "sec_1_fcfs_timestamps.json",
        "total_duration": 25.0,
    },
    "layout_configuration": {
        "aspect_ratio": "16:9",
        "has_title_banner": True,
        "has_caption_column": True,
        "has_subtitles": True,
        "safety_padding": 0.20,
    },
    "viewports": {
        "table_viewport": {
            "type": "viewport_split_left",
            "flow": "column",
            "alignment": "center",
            "padding": 0.15,
            "sub_slots": [],
        },
        "gantt_viewport": {
            "type": "viewport_split_right",
            "flow": "column",
            "alignment": "center",
            "padding": 0.15,
            "sub_slots": [],
        },
        "decision_viewport": {
            "type": "viewport_inspector",
            "flow": "row",
            "alignment": "space_between",
            "padding": 0.15,
            "sub_slots": [],
        },
    },
    "inherited_entities": [],
    "entities": [
        {
            "id": "metric_table",
            "component_type": "MetricTableVisual",
            "visual_tier": "tier1_registered",
            "viewport": "table_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 1,
            "initial_state": {
                "processes": [
                    {"pid": "P1", "arrival_time": 0, "burst_time": 4},
                    {"pid": "P2", "arrival_time": 1, "burst_time": 3},
                    {"pid": "P3", "arrival_time": 2, "burst_time": 1},
                    {"pid": "P4", "arrival_time": 3, "burst_time": 2},
                ],
                "width": 3.8,
                "height": 2.0,
            },
        },
        {
            "id": "gantt",
            "component_type": "GanttChartVisual",
            "visual_tier": "tier1_registered",
            "viewport": "gantt_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 1,
            "initial_state": {
                "total_time": 10,
                "width": 3.7,
                "height": 1.0,
            },
        },
        {
            "id": "decision_card",
            "component_type": "ConceptualCard",
            "visual_tier": "tier4_conceptual_card",
            "viewport": "decision_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 2,
            "initial_state": {
                "title": "First-Come, First-Served (FCFS)",
                "formula": r"\text{Order by Arrival Time: } P_1 \to P_2 \to P_3 \to P_4",
                "badge": "DISPATCH RULE",
            },
        },
    ],
    "timeline": [
        {
            "step_id": "step_1_init_and_p1_dispatch",
            "narration_anchor": {
                "lecture_line_index": 0,
                "text_cue": "In FCFS scheduling, CPU executes processes in order of arrival time. P1 arrives first at time 0.",
                "audio_timestamp_range": [0.0, 5.0],
            },
            "actions": [
                {"type": "enter", "target_entity": "metric_table", "duration": 0.8},
                {"type": "enter", "target_entity": "gantt", "duration": 0.8},
                {"type": "enter", "target_entity": "decision_card", "duration": 0.8},
                {
                    "type": "component_op",
                    "target_entity": "gantt",
                    "operation": "animate_add_block",
                    "parameters": {"pid": "P1", "start_time": 0, "end_time": 4},
                    "duration": 0.8,
                },
            ],
        },
        {
            "step_id": "step_2_p1_completion_and_p2_dispatch",
            "narration_anchor": {
                "lecture_line_index": 1,
                "text_cue": "P1 completes at time 4 with CT 4, TAT 4, and WT 0. P2 is dispatched next.",
                "audio_timestamp_range": [5.0, 10.0],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "metric_table",
                    "operation": "animate_update_process_metrics",
                    "parameters": {"pid": "P1", "ct": 4, "tat": 4, "wt": 0},
                    "duration": 0.8,
                },
                {
                    "type": "transform",
                    "target_entity": "decision_card",
                    "parameters": {
                        "formula": r"P_1 \text{ done: } CT=4, TAT=4, WT=0 \implies P_2 \text{ dispatched}",
                        "badge": "P2 DISPATCHED",
                    },
                    "duration": 0.8,
                },
                {
                    "type": "component_op",
                    "target_entity": "gantt",
                    "operation": "animate_add_block",
                    "parameters": {"pid": "P2", "start_time": 4, "end_time": 7},
                    "duration": 0.8,
                },
            ],
        },
        {
            "step_id": "step_3_p2_completion_and_p3_dispatch",
            "narration_anchor": {
                "lecture_line_index": 2,
                "text_cue": "P2 finishes at time 7 with CT 7, TAT 6, WT 3. P3 is dispatched next.",
                "audio_timestamp_range": [10.0, 15.0],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "metric_table",
                    "operation": "animate_update_process_metrics",
                    "parameters": {"pid": "P2", "ct": 7, "tat": 6, "wt": 3},
                    "duration": 0.8,
                },
                {
                    "type": "transform",
                    "target_entity": "decision_card",
                    "parameters": {
                        "formula": r"P_2 \text{ done: } CT=7, TAT=6, WT=3 \implies P_3 \text{ dispatched}",
                        "badge": "P3 DISPATCHED",
                    },
                    "duration": 0.8,
                },
                {
                    "type": "component_op",
                    "target_entity": "gantt",
                    "operation": "animate_add_block",
                    "parameters": {"pid": "P3", "start_time": 7, "end_time": 8},
                    "duration": 0.8,
                },
            ],
        },
        {
            "step_id": "step_4_p3_completion_and_p4_dispatch",
            "narration_anchor": {
                "lecture_line_index": 3,
                "text_cue": "P3 completes at time 8. P4 is dispatched and runs until time 10.",
                "audio_timestamp_range": [15.0, 20.0],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "metric_table",
                    "operation": "animate_update_process_metrics",
                    "parameters": {"pid": "P3", "ct": 8, "tat": 6, "wt": 5},
                    "duration": 0.8,
                },
                {
                    "type": "transform",
                    "target_entity": "decision_card",
                    "parameters": {
                        "formula": r"P_4 \text{ executes: } [8, 10], \quad CT=10, TAT=7, WT=5",
                        "badge": "P4 DISPATCHED",
                    },
                    "duration": 0.8,
                },
                {
                    "type": "component_op",
                    "target_entity": "gantt",
                    "operation": "animate_add_block",
                    "parameters": {"pid": "P4", "start_time": 8, "end_time": 10},
                    "duration": 0.8,
                },
                {
                    "type": "component_op",
                    "target_entity": "metric_table",
                    "operation": "animate_update_process_metrics",
                    "parameters": {"pid": "P4", "ct": 10, "tat": 7, "wt": 5},
                    "duration": 0.8,
                },
            ],
        },
        {
            "step_id": "step_5_average_metrics_and_summary",
            "narration_anchor": {
                "lecture_line_index": 4,
                "text_cue": "All processes complete. Average waiting time is 3.25, and average turnaround time is 5.75.",
                "audio_timestamp_range": [20.0, 25.0],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "metric_table",
                    "operation": "animate_show_averages",
                    "parameters": {"avg_tat": 5.75, "avg_wt": 3.25},
                    "duration": 0.8,
                },
                {
                    "type": "transform",
                    "target_entity": "decision_card",
                    "parameters": {
                        "formula": r"\text{Avg WT} = \frac{0+3+5+5}{4} = 3.25, \quad \text{Avg TAT} = \frac{4+6+6+7}{4} = 5.75",
                        "badge": "FCFS COMPLETE",
                    },
                    "duration": 0.8,
                },
            ],
        },
    ],
}


def get_fcfs_scheduling_fixture_dict() -> Dict[str, Any]:
    """Returns a deep-copied raw dictionary of the canonical FCFS Scheduling fixture."""
    return copy.deepcopy(FCFS_RAW_FIXTURE)


def get_fcfs_scheduling_fixture() -> SceneIR:
    """Returns a typed SceneIR model of the canonical FCFS Scheduling fixture."""
    return SceneIR.from_dict(get_fcfs_scheduling_fixture_dict())


# ============================================================================
# Example 6: Binary Search Convergence with Target 23 (Protocol v1.1 Fixture)
# ============================================================================

BINARY_SEARCH_TARGET23_RAW_FIXTURE: Dict[str, Any] = {
    "$schema": "https://cinelearn.org/schemas/scene_ir_v1.json",
    "version": "1.0",
    "scene_id": "BinarySearchTarget23Scene",
    "section_index": 1,
    "title": "Binary Search Convergence (Target 23)",
    "duration_budget": 25.0,
    "audio_reference": {
        "audio_path": "sec_1_bs23_audio.mp3",
        "timestamps_path": "sec_1_bs23_timestamps.json",
        "total_duration": 20.0,
    },
    "layout_configuration": {
        "aspect_ratio": "16:9",
        "has_title_banner": True,
        "has_caption_column": True,
        "has_subtitles": True,
        "safety_padding": 0.20,
    },
    "viewports": {
        "array_viewport": {
            "type": "viewport_main",
            "flow": "column",
            "alignment": "center",
            "padding": 0.20,
            "sub_slots": [],
        },
        "inspector_viewport": {
            "type": "viewport_inspector",
            "flow": "row",
            "alignment": "space_between",
            "padding": 0.15,
            "sub_slots": [],
        },
    },
    "inherited_entities": [],
    "entities": [
        {
            "id": "search_array",
            "component_type": "CompositeNodeGraph",
            "visual_tier": "tier3_primitive_tree",
            "viewport": "array_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 1,
            "initial_state": {
                "elements": [
                    {"index": 0, "val": "2", "state": "active"},
                    {"index": 1, "val": "5", "state": "active"},
                    {"index": 2, "val": "8", "state": "active"},
                    {"index": 3, "val": "12", "state": "midpoint"},
                    {"index": 4, "val": "16", "state": "active"},
                    {"index": 5, "val": "23", "state": "active"},
                    {"index": 6, "val": "38", "state": "active"},
                ]
            },
        },
        {
            "id": "pointer_low",
            "component_type": "PointerIndicator",
            "visual_tier": "tier3_primitive_tree",
            "viewport": "array_viewport",
            "slot": None,
            "parent_id": "search_array",
            "z_index": 15,
            "visual_priority": 1,
            "initial_state": {"target_index": 0, "label": "low"},
        },
        {
            "id": "pointer_high",
            "component_type": "PointerIndicator",
            "visual_tier": "tier3_primitive_tree",
            "viewport": "array_viewport",
            "slot": None,
            "parent_id": "search_array",
            "z_index": 15,
            "visual_priority": 1,
            "initial_state": {"target_index": 6, "label": "high"},
        },
        {
            "id": "formula_card",
            "component_type": "ConceptualCard",
            "visual_tier": "tier4_conceptual_card",
            "viewport": "inspector_viewport",
            "slot": None,
            "parent_id": None,
            "z_index": 10,
            "visual_priority": 2,
            "initial_state": {
                "title": "Binary Search (Target = 23)",
                "formula": r"low = 0, \; high = 6 \implies mid = \lfloor \frac{0 + 6}{2} \rfloor = 3",
                "badge": "INITIAL INTERVAL",
            },
        },
    ],
    "timeline": [
        {
            "step_id": "step_1_eval_mid",
            "narration_anchor": {
                "lecture_line_index": 0,
                "text_cue": "Search for target 23 in sorted array. Low is 0 and high is 6. Midpoint index is 3 with value 12.",
                "audio_timestamp_range": [0.0, 5.0],
            },
            "actions": [
                {"type": "enter", "target_entity": "search_array", "duration": 0.8},
                {"type": "enter", "target_entity": "pointer_low", "duration": 0.6},
                {"type": "enter", "target_entity": "pointer_high", "duration": 0.6},
                {"type": "enter", "target_entity": "formula_card", "duration": 0.8},
            ],
        },
        {
            "step_id": "step_2_compare_target_and_discard_left",
            "narration_anchor": {
                "lecture_line_index": 1,
                "text_cue": "Since array[3] = 12 is less than 23, the target lies in the right half. Discard indices 0 to 3.",
                "audio_timestamp_range": [5.0, 12.0],
            },
            "actions": [
                {
                    "type": "component_op",
                    "target_entity": "search_array",
                    "operation": "dim_subrange",
                    "parameters": {"start_idx": 0, "end_idx": 3},
                    "duration": 0.8,
                },
                {
                    "type": "move",
                    "target_entity": "pointer_low",
                    "operation": None,
                    "parameters": {"new_target_index": 4},
                    "duration": 1.0,
                },
                {
                    "type": "transform",
                    "target_entity": "formula_card",
                    "operation": None,
                    "parameters": {
                        "formula": r"A[3]=12 < 23 \implies low = mid + 1 = 4",
                        "badge": "DISCARD LEFT HALF",
                    },
                    "duration": 0.8,
                },
            ],
        },
        {
            "step_id": "step_3_second_mid_and_target_found",
            "narration_anchor": {
                "lecture_line_index": 2,
                "text_cue": "With low=4 and high=6, midpoint is 5. Array[5] is 23, matching our target!",
                "audio_timestamp_range": [12.0, 18.0],
            },
            "actions": [
                {
                    "type": "transform",
                    "target_entity": "formula_card",
                    "operation": None,
                    "parameters": {
                        "formula": r"mid = \lfloor \frac{4+6}{2} \rfloor = 5; \; A[5] = 23 \equiv 23 \implies \text{Found at index 5}",
                        "badge": "TARGET FOUND",
                    },
                    "duration": 0.8,
                },
            ],
        },
    ],
}


def get_binary_search_target23_fixture_dict() -> Dict[str, Any]:
    """Returns a deep-copied raw dictionary of the canonical Binary Search Target 23 fixture."""
    return copy.deepcopy(BINARY_SEARCH_TARGET23_RAW_FIXTURE)


def get_binary_search_target23_fixture() -> SceneIR:
    """Returns a typed SceneIR model of the canonical Binary Search Target 23 fixture."""
    return SceneIR.from_dict(get_binary_search_target23_fixture_dict())

