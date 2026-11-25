import sys
import json
import requests
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class SmartSVGDownloader:
    """
    Downloads and manages external icon/image assets for Manim animations.
    Extracts [Asset: ...] placeholders generated in storyboards, downloads icons directly
    via Iconfinder or the free Iconify API, and resolves placeholder paths without calling any LLM.
    """

    def __init__(self, assets_dir: str = "./assets/icon", api_function=None, iconfinder_api_key: str = None):
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        # api_function is kept for interface compatibility but is NEVER called
        self.api_function = None
        self.iconfinder_api_key = iconfinder_api_key

    def process_storyboard(self, storyboard: Dict) -> Dict:
        """
        Extract [Asset: ...] placeholders directly from storyboard animations,
        download corresponding icons without calling Gemini, and replace placeholders
        with absolute file paths.
        """
        storyboard_data = json.loads(json.dumps(storyboard))
        sections = storyboard_data.get("sections", [])
        if not sections:
            return storyboard_data

        asset_pattern = re.compile(r"\[Asset:\s*([^\]]+)\]", re.IGNORECASE)

    def _asset_to_manim_visual(self, asset_desc: str) -> str:
        """Convert an asset placeholder into a native Manim visual description."""
        lower = asset_desc.lower()
        if "magnif" in lower:
            return "magnifying glass visual using native Manim shapes (Circle with Line/Arrow handle in #ADD8E6)"
        elif "book" in lower or "librar" in lower:
            return "book visual using native Manim shapes (Rectangle with Text/Lines in #ADD8E6)"
        elif "laptop" in lower or "computer" in lower:
            return "computer visual using native Manim shapes (Rectangle screen and Square base)"
        elif "dog" in lower or "cat" in lower or "character" in lower:
            return "character visual using native Manim shapes (Circle face with Polygon ears and Text label)"
        elif "database" in lower or "server" in lower:
            return "database visual using native Manim shapes (stacked Rectangles/Squares)"
        elif "chest" in lower or "box" in lower:
            return "chest visual using native Manim shapes (Square with stroke_color=GOLD)"
        else:
            clean_name = Path(asset_desc).stem if "/" in asset_desc else asset_desc
            clean_name = re.sub(r"[^\w\s\-]", "", clean_name).strip()
            return f"visual representation of {clean_name} using native Manim shapes (Text, Circle, Square, Arrow)"

    def process_storyboard(self, storyboard: Dict) -> Dict:
        """
        Extract [Asset: ...] placeholders and replace them with Manim-only visual descriptions
        (Text, MathTex, Circle, Square, Polygon, Arrow, etc.) without downloading any external assets.
        All network/IconFinder download logic is completely bypassed.
        """
        storyboard_data = json.loads(json.dumps(storyboard))
        sections = storyboard_data.get("sections", [])
        if not sections:
            return storyboard_data

        asset_pattern = re.compile(r"\[Asset:\s*([^\]]+)\]", re.IGNORECASE)

        total_converted = 0
        for sec in sections:
            new_animations = []
            for anim in sec.get("animations", []):
                anim_str = str(anim)
                matches = asset_pattern.findall(anim_str)
                if matches:
                    for match in matches:
                        replacement = self._asset_to_manim_visual(match.strip())
                        anim_str = anim_str.replace(f"[Asset: {match}]", replacement)
                        # Also replace without space if any
                        anim_str = anim_str.replace(f"[Asset:{match}]", replacement)
                        total_converted += 1
                new_animations.append(anim_str)
            sec["animations"] = new_animations

        print(f"ℹ️ Asset download pipeline bypassed for Gemini Free Tier. Converted {total_converted} asset(s) to native Manim visuals.")
        return storyboard_data

    # --- The following IconFinder and Iconify download methods are commented out for future use ---
    # def _check_cache(self, element: str) -> Optional[str]:
    #     for suffix in [".png", ".svg"]:
    #         filepath = self.assets_dir / f"{element}{suffix}"
    #         if filepath.exists() and filepath.stat().st_size > 0:
    #             return str(filepath.absolute())
    #     return None
    #
    # def _download_element(self, element: str, safe_name: Optional[str] = None) -> Optional[str]:
    #     target_name = safe_name or element
    #     if self.iconfinder_api_key and self.iconfinder_api_key not in ("YOUR_ICONFINDER_KEY", "None", ""):
    #         path = self._download_iconfinder(element, target_name)
    #         if path:
    #             return path
    #     return self._download_iconify(element, target_name)
    #
    # def _download_iconfinder(self, element: str, target_name: str) -> Optional[str]:
    #     try:
    #         url = f"https://api.iconfinder.com/v4/icons/search?query={element}&count=1&premium=0"
    #         headers = {"Authorization": f"Bearer {self.iconfinder_api_key}"}
    #         resp = requests.get(url, headers=headers, timeout=10)
    #         if resp.status_code != 200:
    #             return None
    #         data = resp.json()
    #         if not data.get("icons"):
    #             return None
    #         raster_sizes = data["icons"][0].get("raster_sizes", [])
    #         size_url = None
    #         for size in [256, 128, 512]:
    #             for s in raster_sizes:
    #                 if s["size"] == size:
    #                     size_url = s["formats"][0]["preview_url"]
    #                     break
    #             if size_url:
    #                 break
    #         if not size_url and raster_sizes:
    #             size_url = raster_sizes[-1]["formats"][0]["preview_url"]
    #         if size_url:
    #             img_resp = requests.get(size_url, timeout=10)
    #             if img_resp.status_code == 200:
    #                 filepath = self.assets_dir / f"{target_name}.png"
    #                 filepath.write_bytes(img_resp.content)
    #                 return str(filepath.absolute())
    #     except Exception:
    #         return None
    #
    # def _download_iconify(self, element: str, target_name: str) -> Optional[str]:
    #     try:
    #         search_url = f"https://api.iconify.design/search?query={element}&limit=1"
    #         r = requests.get(search_url, timeout=8)
    #         if r.status_code == 200 and r.json().get("icons"):
    #             icon_id = r.json()["icons"][0]
    #             collection, name = icon_id.split(":", 1)
    #             svg_url = f"https://api.iconify.design/{collection}/{name}.svg"
    #             svg_resp = requests.get(svg_url, timeout=8)
    #             if svg_resp.status_code == 200:
    #                 filepath = self.assets_dir / f"{target_name}.svg"
    #                 filepath.write_text(svg_resp.text, encoding="utf-8")
    #                 return str(filepath.absolute())
    #     except Exception:
    #         return None


def process_storyboard_with_assets(
    storyboard: Dict, api_function=None, assets_dir: str = "./assets/icon", iconfinder_api_key: str = None
) -> Dict:
    downloader = SmartSVGDownloader(assets_dir, api_function=None, iconfinder_api_key=iconfinder_api_key)
    return downloader.process_storyboard(storyboard)


if __name__ == "__main__":
    sb = {
        "sections": [
            {
                "lecture_lines": ["A user searches on a laptop"],
                "animations": ["Fade in a [Asset: laptop] on the left side"],
            },
            {
                "lecture_lines": ["Binary search halves the array"],
                "animations": ["Show array elements"],
            },
        ]
    }

    downloader = SmartSVGDownloader("./assets/icon")
    result = downloader.process_storyboard(sb)
    print(json.dumps(result, indent=2, ensure_ascii=False))
