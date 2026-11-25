"""
CineLearn XML-based SVG Sanitizer and Security Validator
========================================================
Validates SVG assets before loading into Manim's SVG parser.
Strictly rejects unsafe elements, event handlers, script injections,
external network references, entity declarations (XXE), and malformed XML.
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union, Tuple

# Maximum allowed SVG file size: 5 MB
MAX_SVG_SIZE_BYTES = 5 * 1024 * 1024

# Disallowed element tags (local name in lowercase)
UNSAFE_TAGS = {
    "script",
    "foreignobject",
    "iframe",
    "embed",
    "object",
    "applet",
    "meta",
    "link",
    "form",
    "input",
    "button",
    "textarea",
    "select",
}

# Dangerous URI schemes and remote prefixes
DANGEROUS_URI_PATTERNS = [
    re.compile(r"^\s*javascript\s*:", re.IGNORECASE),
    re.compile(r"^\s*vbscript\s*:", re.IGNORECASE),
    re.compile(r"^\s*data\s*:\s*text/html", re.IGNORECASE),
    re.compile(r"^\s*https?\s*://", re.IGNORECASE),
    re.compile(r"^\s*ftp\s*://", re.IGNORECASE),
    re.compile(r"^\s*//", re.IGNORECASE),
]


def _strip_namespace(tag: str) -> str:
    """Removes XML namespace prefix, e.g. '{http://www.w3.org/2000/svg}circle' -> 'circle'."""
    if "}" in tag:
        return tag.split("}", 1)[1].lower()
    return tag.lower()


def validate_svg_content(content: str) -> Tuple[bool, str]:
    """
    Validates SVG text content against security and structure requirements.
    Returns (is_valid, failure_reason).
    """
    if not content or not content.strip():
        return False, "Empty SVG content"

    # 1. Text-level checks for entity declarations / XXE attacks
    lower_content = content.lower()
    if "<!entity" in lower_content:
        return False, "Disallowed <!ENTITY declaration (XXE risk)"
    if "<!doctype" in lower_content and ("system" in lower_content or "public" in lower_content):
        return False, "Disallowed external <!DOCTYPE DTD declaration"
    if "<script" in lower_content:
        return False, "Disallowed <script> tag detected in content"

    # 2. Parse XML
    try:
        root = ET.fromstring(content)
    except ET.ParseError as pe:
        return False, f"Malformed XML: {pe}"
    except Exception as e:
        return False, f"XML parsing exception: {e}"

    # 3. Verify root tag is <svg>
    root_tag = _strip_namespace(root.tag)
    if root_tag != "svg":
        return False, f"Invalid root tag: '{root_tag}' (expected 'svg')"

    # 4. Recursively inspect all elements and attributes
    for elem in root.iter():
        elem_tag = _strip_namespace(elem.tag)

        # Check element tag against blacklist
        if elem_tag in UNSAFE_TAGS:
            return False, f"Disallowed unsafe element: <{elem_tag}>"

        # Check attributes
        for attr_name, attr_val in elem.attrib.items():
            attr_local = _strip_namespace(attr_name)
            val_str = str(attr_val or "").strip()

            # Reject event handlers: any attribute starting with 'on' (onload, onclick, onerror, etc.)
            if attr_local.startswith("on"):
                return False, f"Disallowed event handler attribute: '{attr_name}'"

            # Check for dangerous URIs in link/reference attributes
            for pattern in DANGEROUS_URI_PATTERNS:
                if pattern.search(val_str):
                    return False, f"Disallowed URI scheme or remote reference in attribute '{attr_name}': '{val_str[:50]}'"

    return True, "Valid and safe SVG"


def is_safe_svg(file_path: Union[str, Path]) -> bool:
    """
    Convenience function that checks whether an SVG file is safe and well-formed.
    Returns True if valid and safe, False otherwise.
    """
    if not file_path:
        return False

    path = Path(file_path)
    if not path.is_file():
        return False

    try:
        size = path.stat().st_size
        if size <= 0 or size > MAX_SVG_SIZE_BYTES:
            return False

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        is_valid, _ = validate_svg_content(content)
        return is_valid
    except Exception:
        return False


__all__ = ["validate_svg_content", "is_safe_svg", "MAX_SVG_SIZE_BYTES"]
