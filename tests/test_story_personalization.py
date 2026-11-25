import unittest
import json
from prompts.stage1 import get_prompt1_outline
from prompts.stage2 import get_prompt2_storyboard
from prompts.stage3 import get_prompt3_code
from prompts.base_class import base_class
from src.agent import Section, TeachingOutline


class TestStoryPersonalizationPrompts(unittest.TestCase):
    def test_stage1_prompt_contains_story_framework_and_phases(self):
        prompt = get_prompt1_outline(
            knowledge_point="Dijkstra's Algorithm",
            level="intermediate",
            analogy="Pokémon",
        )
        # Check story guidance
        self.assertIn("Story-Driven Personalization & Analogy Guidance", prompt)
        self.assertIn("Build a Coherent Story Around the Interest", prompt)
        self.assertIn("entity mappings", prompt.lower())
        self.assertIn("Phase 1 (Introduction)", prompt)
        self.assertIn("Phase 2 (Concept Introduction)", prompt)
        self.assertIn("Phase 3 (Step-by-Step Explanation)", prompt)
        self.assertIn("Phase 4 (Worked Example)", prompt)
        self.assertIn("Phase 5 (Deeper Explanation & Boundaries)", prompt)
        self.assertIn("Phase 6 (Conclusion / Resolution)", prompt)

        # Check JSON schema elements
        self.assertIn('"story_framework"', prompt)
        self.assertIn('"premise"', prompt)
        self.assertIn('"protagonist_goal"', prompt)
        self.assertIn('"entity_mappings"', prompt)
        self.assertIn('"story_progression"', prompt)
        self.assertIn('"story_step"', prompt)

    def test_stage2_prompt_requires_continuous_story_progression(self):
        mock_outline = json.dumps({
            "topic": "Dijkstra's Algorithm",
            "story_framework": {
                "premise": "A Pokémon trainer navigating towns connected by routes.",
                "protagonist_goal": "Find the fastest route from Pallet Town to Indigo Plateau.",
                "entity_mappings": {
                    "towns": "graph vertices",
                    "routes": "edges",
                    "travel minutes": "edge weights",
                    "tentative journey times": "distance array d[v]",
                },
                "story_progression": "Trainer starts at Town A, explores candidate towns, relaxes route times, and reaches Indigo Plateau.",
            },
            "sections": [
                {
                    "id": "section_1",
                    "title": "Towns and Routes",
                    "story_step": "Trainer maps Kanto towns and route travel times.",
                    "content": "Graph vertices and non-negative edge weights.",
                    "example": "Towns A, B, C with route times 4, 2, 1.",
                }
            ]
        })

        prompt = get_prompt2_storyboard(
            outline=mock_outline,
            level="intermediate",
            analogy="Pokémon",
        )
        self.assertIn("Story-Driven Storyboard & Narration Requirements", prompt)
        self.assertIn("Continuous Narrative Progression Across ALL Sections", prompt)
        self.assertIn("DO NOT introduce 'Pokémon' in Section 1 and then revert", prompt)
        self.assertIn("Worked Example Section", prompt)
        self.assertIn("Deeper / Boundary Section", prompt)
        self.assertIn("Conclusion Section", prompt)
        self.assertIn('"story_beat"', prompt)

    def test_stage3_prompt_contains_story_visual_entities_and_continuity(self):
        sec = Section(
            id="section_2",
            title="Sec 2: Tentative Travel Times",
            lecture_lines=["The trainer records current best travel estimates in a Pokédex notebook."],
            animations=["Display table tracking tentative travel times for towns A, B, C."],
            story_beat="Trainer initializes tentative journey durations from Pallet Town.",
        )
        prompt = get_prompt3_code(
            regenerate_note="",
            section=sec,
            base_class=base_class,
            audio_meta={"total_duration": 30.0, "lines": [{"line_index": 0, "duration": 4.0}]},
            level="intermediate",
            analogy="Pokémon",
        )
        self.assertIn("STORY-DRIVEN VISUAL CONTINUITY & ENTITY REPRESENTATION", prompt)
        self.assertIn("Locations / States / Nodes", prompt)
        self.assertIn("Pathways / Edges / Connections", prompt)
        self.assertIn("Traveler / Active State / Focus", prompt)
        self.assertIn("Visual Continuity Across Scenes", prompt)
        self.assertIn("Visualizing Analogy Boundaries & Breakdown", prompt)
        self.assertIn("Story Beat / Narrative Step: Trainer initializes tentative journey durations", prompt)

    def test_pipeline_dataclass_story_propagation(self):
        # 1. Simulate Outline with story_framework
        raw_outline_json = {
            "topic": "Dijkstra's Algorithm",
            "target_audience": "Undergraduate CS Students",
            "learning_level": "intermediate",
            "story_framework": {
                "premise": "Pokémon trainer navigating Kanto routes",
                "protagonist_goal": "Find least-time route to Indigo Plateau",
                "entity_mappings": {
                    "towns": "vertices",
                    "routes": "edges",
                    "travel times": "weights",
                    "best times": "d[v]"
                },
                "story_progression": "Progressively expand explored territory from source town"
            },
            "sections": [
                {
                    "id": "section_1",
                    "title": "Mapping the Region",
                    "story_step": "Trainer maps Kanto towns and routes",
                    "content": "Graph definition G=(V,E)",
                    "example": "Routes with travel times"
                }
            ]
        }
        outline_obj = TeachingOutline(
            topic=raw_outline_json["topic"],
            target_audience=raw_outline_json["target_audience"],
            sections=raw_outline_json["sections"],
            learning_level=raw_outline_json["learning_level"],
            story_framework=raw_outline_json["story_framework"],
        )
        self.assertIsNotNone(outline_obj.story_framework)
        self.assertEqual(outline_obj.story_framework["protagonist_goal"], "Find least-time route to Indigo Plateau")

        # 2. Simulate Storyboard Section with story_beat
        raw_sec = {
            "id": "section_1",
            "title": "Sec 1: Mapping the Region",
            "story_beat": "Trainer examines region map and notes route travel times",
            "lecture_lines": ["Welcome to Kanto. We must reach the Pokémon League in minimum travel time."],
            "animations": ["Display pentagonal town map with route minutes."]
        }
        sec_obj = Section(
            id=raw_sec["id"],
            title=raw_sec["title"],
            lecture_lines=raw_sec["lecture_lines"],
            animations=raw_sec["animations"],
            story_beat=raw_sec["story_beat"]
        )
        self.assertEqual(sec_obj.story_beat, "Trainer examines region map and notes route travel times")

        # 3. Simulate Stage 3 prompt receiving the section
        code_prompt = get_prompt3_code(
            regenerate_note="",
            section=sec_obj,
            base_class=base_class,
            level="intermediate",
            analogy="Pokémon"
        )
        self.assertIn("Trainer examines region map and notes route travel times", code_prompt)

    def test_quality_configuration(self):
        import tempfile
        import shutil
        from src.agent import RunConfig, TeachingVideoAgent
        tmp_dir = tempfile.mkdtemp()
        try:
            # Test default quality
            cfg_def = RunConfig()
            agent_def = TeachingVideoAgent(idx=0, knowledge_point="Prim", folder=tmp_dir, cfg=cfg_def)
            self.assertEqual(agent_def.quality, "medium")
            self.assertEqual(agent_def.manim_quality_flag, "-qm")
            self.assertEqual(agent_def.quality_folder, "720p30")

            # Test high quality
            cfg_high = RunConfig(quality="high")
            agent_high = TeachingVideoAgent(idx=0, knowledge_point="Prim", folder=tmp_dir, cfg=cfg_high)
            self.assertEqual(agent_high.quality, "high")
            self.assertEqual(agent_high.manim_quality_flag, "-qh")
            self.assertEqual(agent_high.quality_folder, "1080p60")

            # Test low quality
            cfg_low = RunConfig(quality="low")
            agent_low = TeachingVideoAgent(idx=0, knowledge_point="Prim", folder=tmp_dir, cfg=cfg_low)
            self.assertEqual(agent_low.quality, "low")
            self.assertEqual(agent_low.manim_quality_flag, "-ql")
            self.assertEqual(agent_low.quality_folder, "480p15")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_safe_json_loads_latex_escapes(self):
        from src.utils import safe_json_loads
        # Test case with unescaped LaTeX backslashes like \quad, \le, \in, and escaped single quote \'
        problematic_json = r'{"formula": "W(T) - w(e\') \le W(T)", "symbol": "\quad \in \sum", "valid": "\n \t \\ \" "}'
        parsed = safe_json_loads(problematic_json)
        self.assertIn("formula", parsed)
        self.assertEqual(parsed["valid"], "\n \t \\ \" ")
        self.assertIn("W(T)", parsed["formula"])


if __name__ == "__main__":
    unittest.main()
