"""
Unit and Integration Tests for CineLearn Pedagogical Pacing and Dynamic Video Retiming.
======================================================================================

Covers all required testing scenarios from Milestone: Multi-Pass Pedagogical Pacing & Dynamic Video Retiming:
1. Natural Narration Pauses & Acoustic Room Tone Engine:
   - Acoustic room tone presence (eliminates digital zero-amplitude vacuum cutoffs).
   - Contextual pause duration calculation based on punctuation and pedagogical transitions.
   - Crossfading speech segments with boundary envelope smoothing.
   - Provider prosody controls (TTS rate configuration).
   - Cache invalidation and hash consistency.
2. Visual Animation Pacing & Dynamic Animation Retimer:
   - Safe dynamic retiming of meaningful animations (e.g. animate_enqueue, Create, Write).
   - Preservation of transient emphasis and caption activations (e.g. activate_caption, Indicate).
   - Duration clamping to [min_duration, max_duration] bounds.
   - Stable visual hold balancing (adjusting self.wait to accommodate learner inspection).
   - Distinguishing meaningful animation vs. stable visual hold vs. inspection pause.
   - Multi-action segment duration distribution.
   - AST syntax validation ensuring zero corrupted syntax or runtime regressions.
   - Graceful fallback when syntax is malformed.
3. Multi-Pass Timing & Pedagogical Pacing Audit:
   - Audit detection of rushed animations (anim < 1.0s while speech >= 3.0s).
   - Audit detection of excessive static holds (> 5.0s static freeze).
   - Inspection window validation.
   - Animation-to-speech ratio tracking.
4. Domain Decoupling & Subject Agnosticism:
   - OS scheduling (ProcessCardVisual, ReadyQueueVisual, CPUExecutionVisual, GanttChartVisual).
   - Computer Science Algorithms (GraphVisual Dijkstra edge relaxation).
   - Chemistry (ReactionEnergyDiagram transition state and enthalpy).
   - Mathematics (FunctionCurve plot and tangent line).
"""

import ast
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, patch
import pytest
from pydub import AudioSegment

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from timing_contract import (
    VisualActionTiming,
    NarrationSegmentTiming,
    SceneTimingContract,
    AnimationRetimer,
    PacingAuditResult,
)
from audio_manager import AudioManager, AcousticPauseEngine, compute_narration_hash


class TestPedagogicalPacingAndRetiming:

    @pytest.fixture(autouse=True)
    def setup_fixture(self, tmp_path):
        self.tmp_dir = tmp_path
        self.pause_engine = AcousticPauseEngine()
        self.audio_mgr = AudioManager()

    # =========================================================================
    # 1. Natural Narration Pauses & Acoustic Room Tone Engine
    # =========================================================================

    def test_acoustic_room_tone_presence(self):
        """Validates that pauses contain subtle ambient room presence rather than 0-amplitude digital silence."""
        dur_sec = 0.8
        room_tone = self.pause_engine.generate_room_tone(dur_sec, sample_rate=44100, dbfs=-58.0)
        assert len(room_tone) == int(dur_sec * 1000)
        assert room_tone.channels == 2
        assert room_tone.frame_rate == 44100
        # Check that it is NOT dead 0 silence
        raw_samples = room_tone.get_array_of_samples()
        has_non_zero = any(s != 0 for s in raw_samples)
        assert has_non_zero, "Ambient room tone must not be mathematical zero silence"
        # Confirm dBFS is in safe ambient comfort noise range (-65 to -52 dBFS)
        assert -65.0 <= room_tone.dBFS <= -52.0

    def test_contextual_pause_durations(self):
        """Verifies that pause lengths vary appropriately by punctuation and structural transition."""
        pe = self.pause_engine
        # Clause / comma pause
        dur_comma = pe.calculate_pause_duration(prev_text="First-come, first-served,")
        assert 0.30 <= dur_comma <= 0.40

        # Sentence end period
        dur_period = pe.calculate_pause_duration(prev_text="Each process runs to completion.")
        assert 0.50 <= dur_period <= 0.65

        # Question mark
        dur_question = pe.calculate_pause_duration(prev_text="Which process executes next?")
        assert 0.50 <= dur_question <= 0.65

        # Major story beat transition
        dur_beat = pe.calculate_pause_duration(prev_text="We have completed CPU setup.", story_beat_change=True)
        assert 0.75 <= dur_beat <= 0.90

        # Section end
        dur_sec_end = pe.calculate_pause_duration(is_section_end=True)
        assert 1.0 <= dur_sec_end <= 1.2

        # Explicit override preserved
        dur_override = pe.calculate_pause_duration(explicit_pause=1.75)
        assert dur_override == 1.75

    def test_crossfade_speech_boundary_smoothing(self):
        """Verifies speech segments are joined with crossfaded boundary envelopes."""
        # Create two 1-second dummy speech audio segments
        seg1 = AudioSegment.silent(duration=1000)
        seg2 = AudioSegment.silent(duration=1000)
        joined = self.pause_engine.crossfade_speech(seg1, seg2, pause_sec=0.5, fade_ms=40)
        # Expected duration: 1000 + 500 + 1000 = 2500 ms
        assert len(joined) == 2500

    def test_tts_prosody_rate_configuration(self):
        """Validates that speaking rate is configurable and passed to TTS communicator."""
        mgr = AudioManager(tts_rate="-6%")
        assert mgr.tts_rate == "-6%"

        # Mock edge_tts.Communicate to verify rate is propagated
        with patch("edge_tts.Communicate") as mock_comm:
            mock_inst = MagicMock()
            async def dummy_stream():
                yield {"type": "audio", "data": b"\x00" * 2000}
            mock_inst.stream = dummy_stream
            mock_comm.return_value = mock_inst

            out_mp3 = self.tmp_dir / "test_prosody.mp3"
            with patch.object(AudioSegment, "from_file", return_value=AudioSegment.silent(duration=1500)):
                dur = mgr.generate_speech("Testing custom prosody rate", str(out_mp3), rate="-6%")
                assert dur == 1.5
                mock_comm.assert_called_with("Testing custom prosody rate", mgr.tts_voice, rate="-6%")

    def test_cache_consistency_with_natural_pauses(self):
        """Validates that narration changes invalidate audio while identical text reuses cache with natural pauses."""
        lines = [
            "Operating systems manage hardware resources.",
            "CPU scheduling decides which process executes next."
        ]
        def mock_speech(text, path, voice=None, rate=None):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            AudioSegment.silent(duration=2000).export(path, format="mp3")
            return 2.0

        with patch.object(self.audio_mgr, "generate_speech", side_effect=mock_speech):
            meta1 = self.audio_mgr.generate_section_audio("sec_pacing_cache", lines, output_dir=self.tmp_dir)
            assert meta1["narration_hash"] == compute_narration_hash(lines)
            ts_path = self.tmp_dir / "sec_pacing_cache_timestamps.json"
            assert ts_path.exists()

            # Second call with identical text reuses cache without re-calling generate_speech
            with patch.object(self.audio_mgr, "generate_speech") as mock_gen:
                meta2 = self.audio_mgr.generate_section_audio("sec_pacing_cache", lines, output_dir=self.tmp_dir)
                mock_gen.assert_not_called()
                assert meta2["narration_hash"] == meta1["narration_hash"]

    # =========================================================================
    # 2. Visual Animation Pacing & Dynamic Animation Retimer
    # =========================================================================

    def test_animation_retimer_scales_meaningful_animation(self):
        """Verifies that meaningful animations (e.g. animate_enqueue) are retimed to fill an appropriate fraction of speech."""
        code = '''
        # === Animation for Lecture Line 1 ===
        t0 = self.renderer.time
        self.play(*self.activate_caption(0, "#58C4DD"), run_time=0.6)
        self.play(queue.animate_enqueue("P1"), run_time=0.8)
        self.wait(max(0.1, 5.50 - (self.renderer.time - t0)))
        '''
        contract = SceneTimingContract.from_storyboard_section({
            "id": "section_test",
            "lecture_lines": ["Process P1 enters the FIFO ready queue and waits for CPU dispatch."],
            "animations": ["L1-1 | animate_enqueue | ReadyQueueVisual | action:animate_enqueue"],
        })
        contract.update_actual_audio_durations([5.50])

        retimed_code, adjustments = AnimationRetimer.retime_scene_code(code, contract)
        assert len(adjustments) == 1
        adj = adjustments[0]
        # Animation should be scaled up from 0.8s toward ~2.75s (50% of 5.5s)
        assert adj["retimed_anim_sec"] >= 2.0
        assert adj["retimed_anim_sec"] <= 3.5
        assert f"run_time={adj['retimed_anim_sec']:.2f}" in retimed_code
        # activate_caption should remain fast (0.6s)
        assert "run_time=0.6" in retimed_code
        # Code must be syntactically valid Python
        ast.parse(textwrap.dedent(retimed_code))

    def test_animation_retimer_duration_clamping(self):
        """Verifies minimum and maximum duration constraints are respected."""
        # Case A: Very long speech (12 seconds) -> animation should clamp to max_anim_dur (3.5s)
        code_long = '''
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(card), run_time=1.0)
        self.wait(10.0)
        '''
        contract_long = SceneTimingContract.from_storyboard_section({
            "id": "sec_long",
            "lecture_lines": ["This is an extremely long explanation covering many nuances."],
            "animations": ["L1-1 | FadeIn | ProcessCardVisual"],
        })
        contract_long.update_actual_audio_durations([12.0])
        retimed_long, adj_long = AnimationRetimer.retime_scene_code(code_long, contract_long, max_anim_dur=3.5)
        assert adj_long[0]["retimed_anim_sec"] == 3.5

        # Case B: Very short speech (1.2 seconds) -> animation should clamp to min_anim_dur (0.8s)
        code_short = '''
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(card), run_time=0.3)
        self.wait(0.5)
        '''
        contract_short = SceneTimingContract.from_storyboard_section({
            "id": "sec_short",
            "lecture_lines": ["Quick step."],
            "animations": ["L1-1 | FadeIn | ProcessCardVisual"],
        })
        contract_short.update_actual_audio_durations([1.2])
        retimed_short, adj_short = AnimationRetimer.retime_scene_code(code_short, contract_short, min_anim_dur=0.8)
        assert adj_short[0]["retimed_anim_sec"] == 0.8

    def test_animation_retimer_preserves_transient_highlights(self):
        """Verifies transient calls (activate_caption, Indicate) are not mistakenly stretched."""
        code = '''
        # === Animation for Lecture Line 1 ===
        self.play(*self.activate_caption(0, "#FFD700"), run_time=0.5)
        self.play(Indicate(header), run_time=0.6)
        self.play(Create(graph), run_time=1.0)
        self.wait(3.5)
        '''
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_highlight",
            "lecture_lines": ["Observe the graph network shown on screen."],
            "animations": ["L1-1 | Create | GraphVisual"],
        })
        contract.update_actual_audio_durations([5.0])
        retimed_code, adj = AnimationRetimer.retime_scene_code(code, contract)
        # activate_caption and Indicate should remain fast
        assert "run_time=0.5" in retimed_code
        assert "run_time=0.6" in retimed_code
        # Create(graph) should be retimed to ~2.5s
        assert f"run_time={adj[0]['retimed_anim_sec']:.2f}" in retimed_code
        ast.parse(textwrap.dedent(retimed_code))

    def test_animation_retimer_distributes_across_multiple_actions(self):
        """Verifies multiple meaningful actions within a single block share the animation budget proportionally."""
        code = '''
        # === Animation for Lecture Line 1 ===
        self.play(Create(axes), run_time=0.8)
        self.play(Create(curve), run_time=0.8)
        self.wait(4.0)
        '''
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_math",
            "lecture_lines": ["We plot the function axes and construct the parabolic trajectory."],
            "animations": [
                "L1-1 | Create | Axes",
                "L1-2 | Create | FunctionCurve",
            ],
        })
        contract.update_actual_audio_durations([6.0])
        retimed_code, adj = AnimationRetimer.retime_scene_code(code, contract)
        assert adj[0]["meaningful_actions_count"] == 2
        # Total budget ~3.0s shared between 2 actions -> ~1.50s each
        per_action = adj[0]["retimed_anim_sec"] / 2.0
        assert 1.2 <= per_action <= 1.8
        ast.parse(textwrap.dedent(retimed_code))

    def test_animation_retimer_graceful_syntax_fallback(self):
        """Verifies that if retiming somehow causes a SyntaxError, it falls back safely to original code."""
        malformed_code = "def construct(self):\n   invalid python syntax !!! :::"
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_err",
            "lecture_lines": ["Test line."],
            "animations": ["L1-1 | FadeIn | Card"],
        })
        code_out, adj = AnimationRetimer.retime_scene_code(malformed_code, contract)
        assert code_out == malformed_code

    # =========================================================================
    # 3. Multi-Pass Timing & Pedagogical Pacing Audit
    # =========================================================================

    def test_pacing_audit_detects_rushed_animation(self):
        """Verifies audit flags a rushed animation (< 1.0s) paired with long speech (>= 3.0s)."""
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_rushed",
            "lecture_lines": ["A detailed explanation of process dispatch and memory addressing."],
            "animations": ["L1-1 | animate_dispatch | CPUExecutionVisual | action:animate_dispatch"],
        })
        contract.update_actual_audio_durations([5.0])
        # Rendered video animation only lasted 0.6 seconds
        contract.reconcile_with_rendered_video(
            actual_video_duration=5.0,
            rendered_line_timings=[{"line_index": 0, "start_time": 0.0, "duration": 0.6}]
        )
        audit = contract.audit_pacing()
        assert not audit.is_well_paced
        assert len(audit.rushed_animations) == 1
        assert audit.rushed_animations[0]["animation_sec"] == 0.6
        assert audit.rushed_animations[0]["speech_sec"] == 5.0

    def test_pacing_audit_detects_excessive_hold(self):
        """Verifies audit flags an excessive static screen freeze (> 5.0s)."""
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_hold",
            "lecture_lines": ["An excessively long explanation with zero visual progression."],
            "animations": ["L1-1 | FadeIn | Card"],
        })
        contract.update_actual_audio_durations([8.5])
        contract.reconcile_with_rendered_video(
            actual_video_duration=8.5,
            rendered_line_timings=[{"line_index": 0, "start_time": 0.0, "duration": 1.5}]
        )
        # Visual hold is 8.5 - 1.5 = 7.0 seconds > 5.0 seconds
        audit = contract.audit_pacing(max_hold_sec=5.0)
        assert not audit.is_well_paced
        assert len(audit.excessive_holds) == 1
        assert audit.excessive_holds[0]["hold_sec"] == 7.0

    def test_pacing_audit_passes_on_well_paced_scene(self):
        """Verifies audit passes cleanly on a properly retimed, balanced scene."""
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_optimal",
            "lecture_lines": ["Process P1 is dispatched to the CPU and begins execution."],
            "animations": ["L1-1 | animate_dispatch | CPUExecutionVisual"],
        })
        contract.update_actual_audio_durations([4.5])
        # Animation runs 2.5s, inspection hold 2.0s
        contract.reconcile_with_rendered_video(
            actual_video_duration=5.0,
            rendered_line_timings=[{"line_index": 0, "start_time": 0.0, "duration": 2.5}]
        )
        audit = contract.audit_pacing()
        assert audit.is_well_paced
        assert len(audit.rushed_animations) == 0
        assert len(audit.excessive_holds) == 0
        assert audit.animation_to_speech_ratios[0] >= 0.40

    # =========================================================================
    # 4. Domain Decoupling & Subject Agnosticism
    # =========================================================================

    def test_domain_agnostic_retiming_across_all_disciplines(self):
        """
        Validates that dynamic retiming works identically across all educational domains
        (Chemistry, Mathematics, Graph Algorithms, and OS Scheduling) without any domain-specific branches.
        """
        cases = [
            # 1. Chemistry: Reaction Kinetics
            {
                "topic": "Chemical Kinetics",
                "code": '''
                # === Animation for Lecture Line 1 ===
                self.play(diagram.animate_progress_reaction(), run_time=0.8)
                self.wait(4.0)
                ''',
                "line": "Reactants collide with sufficient kinetic energy to overcome the activation barrier.",
                "anim": "L1-1 | animate_progress_reaction | ReactionEnergyDiagram",
                "speech_dur": 5.2,
            },
            # 2. Mathematics: Taylor Series / Derivatives
            {
                "topic": "Calculus Derivatives",
                "code": '''
                # === Animation for Lecture Line 1 ===
                self.play(curve.animate_tangent_line(x=2.0), run_time=0.7)
                self.wait(3.5)
                ''',
                "line": "The tangent line represents the instantaneous rate of change at that coordinate.",
                "anim": "L1-1 | animate_tangent_line | FunctionCurve",
                "speech_dur": 4.8,
            },
            # 3. Computer Science: Dijkstra Shortest Path
            {
                "topic": "Graph Algorithms",
                "code": '''
                # === Animation for Lecture Line 1 ===
                self.play(graph.animate_relax_edge("A", "B"), run_time=0.6)
                self.wait(3.8)
                ''',
                "line": "We relax the edge from node A to node B because a shorter path was discovered.",
                "anim": "L1-1 | animate_relax_edge | GraphVisual",
                "speech_dur": 4.9,
            },
            # 4. Operating Systems: CPU Scheduling
            {
                "topic": "OS Scheduling",
                "code": '''
                # === Animation for Lecture Line 1 ===
                self.play(core.animate_dispatch(process), run_time=0.8)
                self.wait(4.2)
                ''',
                "line": "The scheduler dispatches process P1 from the ready queue to the active CPU core.",
                "anim": "L1-1 | animate_dispatch | CPUExecutionVisual",
                "speech_dur": 5.4,
            },
        ]

        for case in cases:
            contract = SceneTimingContract.from_storyboard_section({
                "id": f"sec_{case['topic'].lower().replace(' ', '_')}",
                "lecture_lines": [case["line"]],
                "animations": [case["anim"]],
            })
            contract.update_actual_audio_durations([case["speech_dur"]])
            retimed_code, adj = AnimationRetimer.retime_scene_code(case["code"], contract)
            assert len(adj) == 1, f"Failed retiming for domain topic: {case['topic']}"
            retimed_anim = adj[0]["retimed_anim_sec"]
            # Target animation should be ~50% of speech duration
            assert 2.0 <= retimed_anim <= 3.2, f"Animation duration out of bounds for {case['topic']}: {retimed_anim}"
            ast.parse(textwrap.dedent(retimed_code))

    # =========================================================================
    # 5. Hardened Retimer Edge Cases & Pedagogical Pacing Policies
    # =========================================================================

    def test_animation_retimer_handles_multiline_and_trailing_commas(self):
        """Verifies multiline self.play calls with trailing commas and nested parens retime safely without double commas."""
        code = '''
        # === Animation for Lecture Line 1 ===
        self.play(
            FadeIn(card),
        )
        self.wait(2.0)
        '''
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_multiline",
            "lecture_lines": ["The process card appears on the screen."],
            "animations": ["L1-1 | FadeIn | ProcessCardVisual"],
        })
        contract.update_actual_audio_durations([4.0])
        retimed_code, adj = AnimationRetimer.retime_scene_code(code, contract)
        assert len(adj) == 1
        # Must not produce double commas (e.g., ",, run_time=...")
        assert ",," not in retimed_code
        assert "run_time=" in retimed_code
        ast.parse(textwrap.dedent(retimed_code))

    def test_animation_retimer_preserves_mid_block_inspection_waits(self):
        """Verifies that intentional mid-block inspection waits are preserved, and only final wait absorbs remaining budget."""
        code = '''
        # === Animation for Lecture Line 1 ===
        self.play(Create(node), run_time=0.8)
        self.wait(0.50)  # Mid-block intentional learner absorption pause
        self.play(Transform(node, target), run_time=0.8)
        self.wait(1.00)  # Final line completion wait
        '''
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_multiwait",
            "lecture_lines": ["First create the node, examine it briefly, then transform it."],
            "animations": [
                "L1-1 | Create | NodeVisual",
                "L1-2 | Transform | NodeVisual",
            ],
        })
        contract.update_actual_audio_durations([6.0])
        retimed_code, adj = AnimationRetimer.retime_scene_code(code, contract)
        assert len(adj) == 1
        # The mid-block wait of 0.50s must be preserved
        assert "self.wait(0.50)" in retimed_code or "self.wait(0.5)" in retimed_code
        # The final wait should be updated to absorb the remaining budget
        assert adj[0]["meaningful_actions_count"] == 2
        ast.parse(textwrap.dedent(retimed_code))

    def test_animation_retimer_pedagogical_action_weighting(self):
        """Verifies major state transitions receive higher duration allocation than minor auxiliary updates."""
        code = '''
        # === Animation for Lecture Line 1 ===
        self.play(core.animate_dispatch(p1), run_time=0.5)
        self.play(timeline.animate_update_time_marker(5), run_time=0.5)
        self.wait(3.0)
        '''
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_weighted",
            "lecture_lines": ["Dispatch process P1 to the core and advance the timeline clock."],
            "animations": [
                "L1-1 | animate_dispatch | CPUExecutionVisual",
                "L1-2 | animate_update_time_marker | TimelineVisual",
            ],
        })
        contract.update_actual_audio_durations([6.0])
        retimed_code, adj = AnimationRetimer.retime_scene_code(code, contract)
        # Parse out individual run_times from retimed code
        m_dispatch = re.search(r"animate_dispatch\(p1\)[^)]*run_time=([0-9.]+)", retimed_code)
        m_timeline = re.search(r"animate_update_time_marker\(5\)[^)]*run_time=([0-9.]+)", retimed_code)
        assert m_dispatch and m_timeline
        dispatch_dur = float(m_dispatch.group(1))
        timeline_dur = float(m_timeline.group(1))
        # Major transition (animate_dispatch, weight 1.5) must get more duration than minor update (weight 0.6)
        assert dispatch_dur > timeline_dur
        assert round(dispatch_dur / timeline_dur, 1) >= 1.8
        ast.parse(textwrap.dedent(retimed_code))

    def test_animation_retimer_strict_idempotence(self):
        """Verifies applying AnimationRetimer repeatedly produces identical code without duration drift."""
        code = '''
        # === Animation for Lecture Line 1 ===
        self.play(core.animate_dispatch(p1), run_time=0.8)
        self.wait(3.5)
        '''
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_idempotent",
            "lecture_lines": ["Dispatching process P1 to the active core."],
            "animations": ["L1-1 | animate_dispatch | CPUExecutionVisual"],
        })
        contract.update_actual_audio_durations([5.0])
        retimed_pass1, adj1 = AnimationRetimer.retime_scene_code(code, contract)
        retimed_pass2, adj2 = AnimationRetimer.retime_scene_code(retimed_pass1, contract)
        assert retimed_pass1 == retimed_pass2
        assert adj1[0]["retimed_anim_sec"] == adj2[0]["retimed_anim_sec"]
        assert adj1[0]["retimed_wait_sec"] == adj2[0]["retimed_wait_sec"]

    def test_long_narration_handling_unintended_vs_intentional(self):
        """
        Verifies distinction between:
        - Case A: Unintended static hold from an overgrown narration line (flags pacing warning).
        - Case B: Legitimate intentional pedagogical inspection hold (passes audit cleanly).
        """
        # Case A: Unintended hold (pacing_preference="balanced")
        contract_unintended = SceneTimingContract.from_storyboard_section({
            "id": "sec_unintended",
            "lecture_lines": ["An excessively long explanation with zero intentional hold specified."],
            "animations": ["L1-1 | FadeIn | Card"],
        })
        contract_unintended.update_actual_audio_durations([9.0])
        contract_unintended.reconcile_with_rendered_video(
            actual_video_duration=9.0,
            rendered_line_timings=[{"line_index": 0, "start_time": 0.0, "duration": 1.5}]
        )
        audit_a = contract_unintended.audit_pacing(max_hold_sec=5.0)
        assert not audit_a.is_well_paced
        assert len(audit_a.excessive_holds) == 1

        # Case B: Intentional hold (pacing_preference="inspection_emphasis")
        contract_intentional = SceneTimingContract.from_storyboard_section({
            "id": "sec_intentional",
            "lecture_lines": ["Observe the final completed Gantt chart showing turnaround times."],
            "animations": ["L1-1 | FadeIn | GanttChartVisual | action:inspect"],
        })
        contract_intentional.segments[0].pacing_preference = "inspection_emphasis"
        contract_intentional.update_actual_audio_durations([9.0])
        contract_intentional.reconcile_with_rendered_video(
            actual_video_duration=9.0,
            rendered_line_timings=[{"line_index": 0, "start_time": 0.0, "duration": 1.5}]
        )
        audit_b = contract_intentional.audit_pacing(max_hold_sec=5.0)
        # Should pass audit because hold is recognized as intentional pedagogical absorption
        assert audit_b.is_well_paced
        assert len(audit_b.excessive_holds) == 0
        assert any(w.get("intentional") for w in audit_b.inspection_windows)

    def test_animation_retimer_preserves_comments_and_strings(self):
        """Verifies comments and string literals with method names or quotes are preserved uncorrupted."""
        code = '''
        # === Animation for Lecture Line 1 ===
        # Important comment before play
        title = Text("Demonstrating self.play(with_inner_text)")
        self.play(
            FadeIn(title),  # Trailing comment on argument
        )
        self.wait(2.0)  # Trailing comment on wait
        '''
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_comments",
            "lecture_lines": ["Here is the title card with a special description."],
            "animations": ["L1-1 | FadeIn | Text"],
        })
        contract.update_actual_audio_durations([5.0])
        retimed_code, adj = AnimationRetimer.retime_scene_code(code, contract)
        assert len(adj) == 1
        assert '# Important comment before play' in retimed_code
        assert 'Text("Demonstrating self.play(with_inner_text)")' in retimed_code
        assert '# Trailing comment on argument' in retimed_code
        ast.parse(textwrap.dedent(retimed_code))

    def test_animation_retimer_missing_runtime_injection(self):
        """Verifies calls without explicit run_time receive injected run_time cleanly."""
        code = '''
        # === Animation for Lecture Line 1 ===
        self.play(Create(box))
        self.wait(1.0)
        '''
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_no_rt",
            "lecture_lines": ["Creating the initial bounding box for the visual element."],
            "animations": ["L1-1 | Create | Rectangle"],
        })
        contract.update_actual_audio_durations([4.0])
        retimed_code, adj = AnimationRetimer.retime_scene_code(code, contract)
        assert len(adj) == 1
        assert "run_time=" in retimed_code
        assert ",," not in retimed_code
        ast.parse(textwrap.dedent(retimed_code))

    def test_audio_cache_invalidation_on_pause_config_change(self, tmp_path):
        """Verifies changing pause duration or voice invalidates cached audio artifacts."""
        mgr = AudioManager()
        lines = ["Testing cache invalidation when pause configuration changes."]

        # 1. Initial generation with default 0.5s pause
        meta1 = mgr.generate_section_audio(
            section_id="sec_cache_test",
            lecture_lines=lines,
            output_dir=tmp_path,
            inter_line_pause=0.5,
            section_end_pause=1.0,
        )
        assert Path(meta1["audio_path"]).exists()
        dur1 = meta1["total_duration"]

        # 2. Re-running with identical parameters returns cached metadata
        meta_cached = mgr.generate_section_audio(
            section_id="sec_cache_test",
            lecture_lines=lines,
            output_dir=tmp_path,
            inter_line_pause=0.5,
            section_end_pause=1.0,
        )
        assert meta_cached["total_duration"] == dur1

        # 3. Changing inter_line_pause or section_end_pause invalidates cache and regenerates
        meta2 = mgr.generate_section_audio(
            section_id="sec_cache_test",
            lecture_lines=lines,
            output_dir=tmp_path,
            inter_line_pause=1.5,
            section_end_pause=2.5,
        )
        # Duration should be longer due to increased end pause
        assert meta2["total_duration"] > dur1 + 1.0

    def test_acoustic_pause_engine_validation(self):
        """Verifies AcousticPauseEngine room-tone presence passes automated acoustic checks."""
        engine = AcousticPauseEngine(ambient_dbfs=-58.0)
        pause_segment = engine.create_natural_pause(1.0)
        val = engine.validate_audio_presence(pause_segment)
        assert val["is_valid"]
        assert val["measured_dbfs"] is not None
        assert -65.0 <= val["measured_dbfs"] <= -50.0
        assert val["duration_ms"] == 1000

    def test_timing_contract_warning_includes_segment_and_action(self):
        """Verifies that pacing warnings explicitly identify segment index, caption, and action IDs."""
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_diag",
            "lecture_lines": ["Detailed explanation without adequate visual animation."],
            "animations": ["L1-1 | FadeIn | Card | action:animate_fade"],
            "captions": ["Detailed explanation"],
        })
        contract.update_actual_audio_durations([6.0])
        contract.reconcile_with_rendered_video(
            actual_video_duration=6.0,
            rendered_line_timings=[{"line_index": 0, "start_time": 0.0, "duration": 0.5}]
        )
        audit = contract.audit_pacing()
        assert len(audit.rushed_animations) == 1
        issue = audit.rushed_animations[0]
        assert issue["segment_id"] == "sec_diag_seg_1"
        assert issue["caption"] == "Detailed explanation"
        assert "L1-1" in issue["associated_actions"]
        assert "L1-1" in issue["reason"]
        assert "Detailed explanation" in issue["reason"]

    def test_storyboard_pacing_tag_parsing(self):
        """Verifies parsing pacing tags (action: inspect or pacing: inspection_emphasis) sets inspection preference."""
        contract = SceneTimingContract.from_storyboard_section({
            "id": "sec_tags",
            "lecture_lines": ["Now observe the complete architecture diagram carefully."],
            "animations": ["L1-1 | FadeIn | ArchDiagram | action: inspect | pacing: inspection_emphasis"],
        })
        assert contract.segments[0].pacing_preference == "inspection_emphasis"
