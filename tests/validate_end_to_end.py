"""
CineLearn End-to-End Pipeline Validation Driver
================================================
Validates Stage 1 -> Stage 2 -> Stage 3 -> Manim Rendering -> Audio & Subtitles -> Visual QA
across representative topics in CS, Chemistry, and Mathematics.
"""

import sys
import os
import json
import time
import inspect
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.agent import TeachingVideoAgent, RunConfig, get_api_and_output
from src.domains import VisualRegistry
from src.local_checker import LocalVisualChecker


def inspect_generated_code(code: str, domain: str) -> Dict[str, Any]:
    """Analyzes generated Manim code for component usage, semantic actions, and API validity."""
    analysis = {
        "components_detected": [],
        "semantic_actions_called": [],
        "invalid_calls": [],
        "docking_calls": [],
        "uses_dock_main": "dock_main" in code,
        "uses_dock_inspector": "dock_inspector" in code,
        "uses_dock_split": ("dock_split_left" in code or "dock_split_right" in code),
    }

    # Discover registered components
    manifest = VisualRegistry.get_schema_manifest()
    known_components = {}
    for dom, comps in manifest.items():
        for comp in comps:
            cid = comp["component_id"]
            known_components[cid] = {
                "domain": dom,
                "actions": comp["semantic_actions"],
                "params": comp["init_params"],
            }

    for cid, info in known_components.items():
        if cid in code:
            analysis["components_detected"].append({"component_id": cid, "domain": info["domain"]})
            # Check which semantic actions were invoked on this component
            for action in info["actions"]:
                if f".{action}(" in code:
                    analysis["semantic_actions_called"].append(f"{cid}.{action}")

    # Check for potential nonexistent method calls on registered components
    # by scanning for .animate_ patterns
    import re
    called_anims = re.findall(r"\.(animate_[a-zA-Z0-9_]+)\(", code)
    for anim in called_anims:
        is_known = False
        for cid, info in known_components.items():
            if anim in info["actions"]:
                is_known = True
                break
        if not is_known and anim not in ("animate_fade", "animate_slide", "animate_zoom", "animate_highlight"):
            analysis["invalid_calls"].append(anim)

    return analysis


def validate_topic_pipeline(topic: str, domain: str, output_folder: Path, cfg: RunConfig) -> Dict[str, Any]:
    """Runs the complete 5-stage CineLearn pipeline and performs deep verification."""
    print(f"\n{'='*70}\n[START] Validating Topic: {topic} (Domain: {domain})\n{'='*70}")
    start_time = time.time()
    results = {
        "topic": topic,
        "domain": domain,
        "output_folder": str(output_folder),
        "stages": {},
        "overall_success": False,
    }

    output_folder.mkdir(parents=True, exist_ok=True)

    agent = TeachingVideoAgent(
        idx=1,
        topic=topic,
        level="intermediate",
        folder=output_folder,
        cfg=cfg,
    )

    # ------------------------------------------------------------------------
    # Stage 1: Outline
    # ------------------------------------------------------------------------
    print(f"\n--- [Stage 1] Outline Generation ---")
    s1_start = time.time()
    try:
        outline = agent.generate_outline()
        s1_time = round(time.time() - s1_start, 2)
        if not outline or not getattr(outline, "sections", None):
            results["stages"]["stage1"] = {"success": False, "error": "No sections in outline", "duration_sec": s1_time}
            return results

        results["stages"]["stage1"] = {
            "success": True,
            "duration_sec": s1_time,
            "sections_count": len(outline.sections),
            "section_titles": [
                s.get("title", "") if isinstance(s, dict) else getattr(s, "title", str(s))
                for s in outline.sections
            ],
            "worked_example_spec": getattr(outline, "worked_example_spec", None) or (outline.get("worked_example_spec") if isinstance(outline, dict) else None),
        }
        print(f"✅ Stage 1 passed ({s1_time}s): {len(outline.sections)} sections created.")
    except Exception as e:
        results["stages"]["stage1"] = {"success": False, "error": str(e)}
        return results

    # ------------------------------------------------------------------------
    # Stage 2: Storyboard
    # ------------------------------------------------------------------------
    print(f"\n--- [Stage 2] Storyboard Planning ---")
    s2_start = time.time()
    try:
        storyboard = agent.generate_storyboard()
        s2_time = round(time.time() - s2_start, 2)
        if not storyboard or len(agent.sections) == 0:
            results["stages"]["stage2"] = {"success": False, "error": "Empty storyboard", "duration_sec": s2_time}
            return results

        planned_components = []
        for sec in agent.sections:
            sec_dict = sec.to_dict() if hasattr(sec, "to_dict") else getattr(sec, "__dict__", {})
            planned_components.append({
                "section_id": sec.id,
                "title": sec.title,
                "visual_design": sec_dict.get("visual_design", ""),
                "layout_plan": sec_dict.get("layout_plan", ""),
                "planned_components": sec_dict.get("planned_components", []),
            })

        results["stages"]["stage2"] = {
            "success": True,
            "duration_sec": s2_time,
            "sections_count": len(agent.sections),
            "storyboard_plans": planned_components,
        }
        print(f"✅ Stage 2 passed ({s2_time}s): Storyboard with visual component plans created.")
    except Exception as e:
        results["stages"]["stage2"] = {"success": False, "error": str(e)}
        return results

    # ------------------------------------------------------------------------
    # Narration Audio Pre-generation
    # ------------------------------------------------------------------------
    print(f"\n--- Narration Audio Generation & Synchronization ---")
    if agent.use_audio:
        try:
            agent.pregenerate_all_section_audios()
            print(f"✅ Narration audio files pre-generated.")
        except Exception as e:
            print(f"⚠️ Audio pre-generation error: {e}")

    # ------------------------------------------------------------------------
    # Stage 3: Code Generation
    # ------------------------------------------------------------------------
    print(f"\n--- [Stage 3] Code Generation & API Analysis ---")
    s3_start = time.time()
    try:
        codes = agent.generate_codes()
        s3_time = round(time.time() - s3_start, 2)
        if not codes:
            results["stages"]["stage3"] = {"success": False, "error": "No codes generated", "duration_sec": s3_time}
            return results

        code_analyses = {}
        for sec_id, code in codes.items():
            code_analyses[sec_id] = inspect_generated_code(code, domain)

        results["stages"]["stage3"] = {
            "success": True,
            "duration_sec": s3_time,
            "sections_coded": len(codes),
            "code_analyses": code_analyses,
        }
        print(f"✅ Stage 3 passed ({s3_time}s): Manim code generated and analyzed.")
    except Exception as e:
        results["stages"]["stage3"] = {"success": False, "error": str(e)}
        return results

    # ------------------------------------------------------------------------
    # Stage 4: Rendering & Visual QA
    # ------------------------------------------------------------------------
    print(f"\n--- [Stage 4] Manim Rendering & Local Visual QA ---")
    s4_start = time.time()
    try:
        rendered = agent.render_all_sections()
        s4_time = round(time.time() - s4_start, 2)

        # Run LocalVisualChecker on rendered layout files
        checker = LocalVisualChecker()
        qa_reports = {}
        for sec in agent.sections:
            sec_id = sec.id
            code = codes.get(sec_id, "")
            res = checker.check_section(sec_id, code, agent.output_dir)

            qa_reports[sec_id] = {
                "rendered_video": rendered.get(sec_id),
                "rendered_success": sec_id in rendered and rendered[sec_id] is not None and Path(rendered[sec_id]).exists(),
                "total_issues": len(res.issues),
                "has_errors": res.has_errors(),
                "issues": [
                    {"type": i.issue_type, "severity": i.severity, "desc": i.description}
                    for i in res.issues
                ],
            }

        results["stages"]["stage4"] = {
            "success": len(rendered) == len(agent.sections),
            "duration_sec": s4_time,
            "rendered_sections": len(rendered),
            "qa_reports": qa_reports,
        }
        print(f"✅ Stage 4 passed ({s4_time}s): Sections rendered and QA inspected.")
    except Exception as e:
        results["stages"]["stage4"] = {"success": False, "error": str(e)}
        return results

    # ------------------------------------------------------------------------
    # Stage 5: Final Merge, Audio Synchronization & Subtitles
    # ------------------------------------------------------------------------
    print(f"\n--- [Stage 5] Final Merge, Master Audio & Subtitles ---")
    s5_start = time.time()
    try:
        final_video = agent.merge_videos()
        s5_time = round(time.time() - s5_start, 2)
        has_video = final_video is not None and Path(final_video).exists() and Path(final_video).stat().st_size > 1000

        srt_file = agent.output_dir / f"{Path(final_video).stem}.srt" if has_video else None
        has_subtitles = srt_file is not None and srt_file.exists() and srt_file.stat().st_size > 50

        results["stages"]["stage5"] = {
            "success": has_video,
            "duration_sec": s5_time,
            "final_video_path": str(final_video) if has_video else None,
            "final_video_size_bytes": Path(final_video).stat().st_size if has_video else 0,
            "subtitles_generated": has_subtitles,
            "subtitles_path": str(srt_file) if has_subtitles else None,
        }
        results["overall_success"] = has_video
        print(f"✅ Stage 5 passed ({s5_time}s): Final video created at {final_video}")
    except Exception as e:
        results["stages"]["stage5"] = {"success": False, "error": str(e)}

    total_time = round(time.time() - start_time, 2)
    results["total_pipeline_time_sec"] = total_time
    print(f"\n{'='*70}\n[COMPLETE] Topic: {topic} (Success: {results['overall_success']}, Time: {total_time}s)\n{'='*70}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None, help="Index of topic (0: CS, 1: Chemistry, 2: Math)")
    cli_args = parser.parse_args()

    topics = [
        {"domain": "cs", "topic": "Dijkstra's shortest path", "folder_name": "cs_dijkstra"},
        {"domain": "chemistry", "topic": "Reaction kinetics and activation energy", "folder_name": "chem_kinetics"},
        {"domain": "math", "topic": "Understanding derivatives and rates of change", "folder_name": "math_derivatives"},
    ]

    selected_topics = [topics[cli_args.index]] if cli_args.index is not None else topics

    base_output = _PROJECT_ROOT / "TEST_CASES" / "phase4_validation"
    base_output.mkdir(parents=True, exist_ok=True)

    api, _ = get_api_and_output(provider="ollama", model="qwen2.5-coder:32b")
    cfg = RunConfig(
        api=api,
        provider="ollama",
        model="qwen2.5-coder:32b",
        audio=True,
        burn_subtitles=True,
        quality="low",  # 480p15 for fast, lightweight rendering
        parallel=False,
    )

    all_reports = {}
    for item in selected_topics:
        target_folder = base_output / item["folder_name"]
        report = validate_topic_pipeline(
            topic=item["topic"],
            domain=item["domain"],
            output_folder=target_folder,
            cfg=cfg,
        )
        all_reports[item["domain"]] = report
        # Save individual report
        rep_file = target_folder / "validation_report.json"
        rep_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Save summary report
    summary_file = base_output / "all_validation_reports.json"
    summary_file.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(f"\n🎉 All validations finished! Summary saved to {summary_file}")
