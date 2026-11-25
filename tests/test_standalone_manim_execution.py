"""
Integration Test: Standalone Manim Execution with Shared Implementations
========================================================================
Validates that generated standalone scene scripts can safely import shared
classes from src.domains (GenericContainerLayout, VisualComponent, VisualRegistry)
and render to a valid MP4 video without inlined duplicated code.
"""

import sys
from pathlib import Path

# Ensure CineLearn root is in sys.path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
from prompts import base_class
from src.utils import replace_base_class, run_manim_script


class TestStandaloneManimExecution:
    """Verifies standalone Manim CLI rendering of scene using shared src.domains architecture."""

    def test_standalone_scene_render_via_manim_cli(self, tmp_path):
        scene_code = '''
class StandaloneIntegrationTestScene(TeachingScene):
    def construct(self):
        self.setup_layout(
            title_text="Standalone Single Source of Truth Test",
            lecture_lines=[
                "Line 1: Shared layout imported cleanly",
                "Line 2: VisualRegistry accessible in scene"
            ]
        )

        # 1. Test shared ProceduralComponent
        proc = ProceduralComponent(
            Square(side_length=2.0, color="#58C4DD", fill_opacity=0.6),
            name="TestSharedBox"
        )
        self.dock_main(proc)

        # 2. Test shared VisualRegistry lookup at runtime
        @VisualRegistry.register("standalone_test", "MockInspectorTable")
        class MockInspectorTable(VisualComponent):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                rect = Rectangle(width=5.0, height=0.7, color="#83C167", fill_opacity=0.8)
                txt = Text("Registry Table", font_size=16, color=WHITE)
                self.add(rect, txt)

        table_cls = VisualRegistry.get("standalone_test", "MockInspectorTable")
        assert table_cls is not None, "VisualRegistry lookup failed at scene runtime"
        table_instance = table_cls()
        self.dock_inspector(table_instance)

        # Play short animation
        self.play(FadeIn(proc), FadeIn(table_instance), run_time=0.4)
        self.wait(0.2)
'''

        # 1. Assemble code using CineLearn's replace_base_class pipeline function
        full_code = replace_base_class(scene_code, base_class)

        # Check that full_code does NOT contain duplicated inlined class definitions
        assert "class RectangularViewport:" not in full_code, "Duplicate RectangularViewport still inlined!"

        # 2. Write to standalone script file
        script_file = tmp_path / "standalone_scene.py"
        script_file.write_text(full_code, encoding="utf-8")

        # 3. Render via Manim CLI subprocess
        out_media = tmp_path / "media"
        output_mp4 = run_manim_script(
            filename=str(script_file),
            scene_name="StandaloneIntegrationTestScene",
            output_dir=str(out_media)
        )

        # 4. Verify resulting MP4 video
        assert output_mp4 is not None
        mp4_path = Path(output_mp4)
        assert mp4_path.exists(), f"Rendered video does not exist: {output_mp4}"
        assert mp4_path.stat().st_size > 1000, f"Rendered video is too small ({mp4_path.stat().st_size} bytes)"
