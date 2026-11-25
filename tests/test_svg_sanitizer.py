"""
Unit Tests for CineLearn XML-based SVG Sanitizer and Security Validator
=======================================================================
Validates that SVG files with script injections, event handlers,
unsafe elements, external entities (XXE), and malformed XML are rejected,
and that SafeSVGMobject falls back to native vector placeholders.
"""

import tempfile
from pathlib import Path
import pytest
from manim import VGroup

from src.domains.svg_sanitizer import validate_svg_content, is_safe_svg
from prompts import base_class

_scope = {}
exec(base_class, _scope)
SafeSVGMobject = _scope["SafeSVGMobject"]


class TestSVGSanitizer:
    """Security and compliance test suite for SVG sanitizer."""

    def test_valid_safe_svg_passes(self):
        valid_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '  <circle cx="50" cy="50" r="40" fill="#58C4DD" stroke="#FFFFFF" stroke-width="2"/>'
            '  <rect x="20" y="20" width="60" height="20" fill="#83C167"/>'
            '</svg>'
        )
        is_valid, reason = validate_svg_content(valid_svg)
        assert is_valid, f"Expected valid SVG to pass, failed with: {reason}"

    def test_script_tag_in_header_rejected(self):
        malicious_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '  <script>alert("xss")</script>'
            '  <circle cx="50" cy="50" r="40"/>'
            '</svg>'
        )
        is_valid, reason = validate_svg_content(malicious_svg)
        assert not is_valid
        assert "script" in reason.lower()

    def test_script_tag_after_2kb_buffer_rejected(self):
        """CRITICAL: Ensures scripts placed past 2048 bytes of padding are strictly rejected."""
        padding = " " * 3000
        malicious_svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            f'{padding}'
            '  <script type="text/javascript">steal_tokens()</script>'
            '</svg>'
        )
        is_valid, reason = validate_svg_content(malicious_svg)
        assert not is_valid
        assert "script" in reason.lower()

    def test_event_handlers_rejected(self):
        test_cases = [
            '<svg xmlns="http://www.w3.org/2000/svg" onload="malicious()"><circle r="10"/></svg>',
            '<svg xmlns="http://www.w3.org/2000/svg"><rect width="10" height="10" onclick="alert(1)"/></svg>',
            '<svg xmlns="http://www.w3.org/2000/svg"><image onerror="logError()"/></svg>',
            '<svg xmlns="http://www.w3.org/2000/svg"><path onmouseover="evil()" d="M0 0 L10 10"/></svg>',
        ]
        for case in test_cases:
            is_valid, reason = validate_svg_content(case)
            assert not is_valid, f"Expected event handler in '{case[:50]}' to be rejected"
            assert "event handler" in reason.lower()

    def test_unsafe_elements_rejected(self):
        unsafe_tags = ["foreignObject", "iframe", "embed", "object", "meta", "link"]
        for tag in unsafe_tags:
            svg = f'<svg xmlns="http://www.w3.org/2000/svg"><{tag} src="http://evil.com"/></svg>'
            is_valid, reason = validate_svg_content(svg)
            assert not is_valid, f"Expected <{tag}> to be rejected"
            assert "unsafe element" in reason.lower()

    def test_entity_declaration_xxe_rejected(self):
        xxe_svg = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE svg ['
            '  <!ENTITY xxe SYSTEM "file:///etc/passwd">'
            ']>'
            '<svg xmlns="http://www.w3.org/2000/svg"><text>&xxe;</text></svg>'
        )
        is_valid, reason = validate_svg_content(xxe_svg)
        assert not is_valid
        assert "entity" in reason.lower()

    def test_external_network_uris_rejected(self):
        cases = [
            '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">'
            '  <image xlink:href="http://attacker.com/leak.png" width="10" height="10"/>'
            '</svg>',
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '  <a href="javascript:alert(1)"><circle r="5"/></a>'
            '</svg>',
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '  <image href="https://evil.org/track" width="5" height="5"/>'
            '</svg>',
        ]
        for case in cases:
            is_valid, reason = validate_svg_content(case)
            assert not is_valid, f"Expected external URI to be rejected: {case[:60]}"
            assert "disallowed uri" in reason.lower() or "remote" in reason.lower()

    def test_malformed_xml_rejected(self):
        malformed = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="10">'  # Unclosed tag
        is_valid, reason = validate_svg_content(malformed)
        assert not is_valid
        assert "malformed" in reason.lower() or "xml" in reason.lower()


class TestSafeSVGMobjectFallback:
    """Verifies that SafeSVGMobject safely falls back to native vector cards on unsafe/malformed SVGs."""

    def test_safe_svg_mobject_renders_clean_svg(self, tmp_path):
        clean_svg = tmp_path / "valid_icon.svg"
        clean_svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '  <circle cx="50" cy="50" r="30" fill="#58C4DD"/>'
            '</svg>',
            encoding="utf-8"
        )
        mob = SafeSVGMobject(file_name=str(clean_svg))
        assert isinstance(mob, VGroup)
        assert len(mob.submobjects) > 0

    def test_safe_svg_mobject_falls_back_on_malicious_svg(self, tmp_path):
        bad_svg = tmp_path / "malicious.svg"
        bad_svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)">'
            '  <circle cx="50" cy="50" r="30"/>'
            '</svg>',
            encoding="utf-8"
        )
        # SafeSVGMobject must not crash and must create placeholder
        mob = SafeSVGMobject(file_name=str(bad_svg))
        assert isinstance(mob, VGroup)
        # Native placeholder has rounded rectangle, text, and 3 accent dots (5 sub-elements)
        assert len(mob.submobjects) >= 5

    def test_safe_svg_mobject_falls_back_on_missing_file(self):
        mob = SafeSVGMobject(file_name="/non/existent/icon.svg")
        assert isinstance(mob, VGroup)
        assert len(mob.submobjects) >= 5
