import os
import sys
import shutil
import subprocess
import json
from typing import List, Any
from manim import *
import multiprocessing
import re
import psutil
from pathlib import Path

try:
    import orjson
    _ORJSON_AVAILABLE = True
except ImportError:
    _ORJSON_AVAILABLE = False


def sanitize_json_string(s: str) -> str:
    """
    Sanitize JSON string by properly escaping unescaped backslashes in string literals.
    Commonly repairs invalid LaTeX escape sequences (like \\quad, \\le, \\', \\text, etc.).
    """
    result = []
    in_string = False
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if not in_string:
            if c == '"':
                in_string = True
            result.append(c)
            i += 1
        else:
            if c == '"':
                in_string = False
                result.append(c)
                i += 1
            elif c == '\\':
                if i + 1 < n:
                    nxt = s[i + 1]
                    if nxt in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't'):
                        result.append(c)
                        result.append(nxt)
                        i += 2
                    elif nxt == 'u' and i + 5 < n and all(ch in "0123456789abcdefABCDEF" for ch in s[i+2:i+6]):
                        result.append(s[i:i+6])
                        i += 6
                    else:
                        result.append('\\\\')
                        i += 1
                else:
                    result.append('\\\\')
                    i += 1
            else:
                result.append(c)
                i += 1
    return "".join(result)


def safe_json_loads(json_str: str) -> Any:
    """Safely loads JSON string using ultra-fast orjson when available, with automatic sanitization fallback."""
    if _ORJSON_AVAILABLE:
        try:
            return orjson.loads(json_str)
        except Exception:
            pass
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        sanitized = sanitize_json_string(json_str)
        return json.loads(sanitized)


def extract_json_from_markdown(text):
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback: if text starts with an unclosed markdown fence, extract the JSON body
    fence_match = re.search(r"```(?:json)?\s*(\{.*)", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).rstrip()
    return text


def extract_answer_from_response(response):
    try:
        content = response.candidates[0].content.parts[0].text
    except Exception:
        try:
            content = response.choices[0].message.content
        except Exception:
            try:
                content = response.text
            except Exception:
                content = str(response)
    content = extract_json_from_markdown(content)
    return content


def fix_png_path(code_str: str, assets_dir: Path) -> str:
    assets_dir = Path(assets_dir).resolve()

    def replacer(match):
        original_path = match.group(1)  # matched XXX.png
        path_obj = Path(original_path)
        # not an absolute path and is not under assets_dir
        if not path_obj.is_absolute():
            # concat to absolute path
            return f'"{assets_dir / path_obj.name}"'
        # absolute path but not under assets_dir
        try:
            if assets_dir not in path_obj.parents:
                return f'"{assets_dir / path_obj.name}"'
        except RuntimeError:
            return f'"{assets_dir / path_obj.name}"'
        return match.group(0)  # keep original

    pattern = r'["\']([^"\']+\.(png|svg))["\']'
    return re.sub(pattern, replacer, code_str)


def get_optimal_workers():
    """Calculate the optimal number of parallel processes adaptively based on # CPU cores and load"""
    try:
        cpu_count = multiprocessing.cpu_count()
    except NotImplementedError:
        cpu_count = 6  # default

    # Manim rendering is CPU-intensive; usually set workers to CPU cores or cores minus one
    # reserve 1 core for system/other processes
    optimal = max(1, cpu_count - 1)

    # If the machine is high-performance multicore (>16 cores),
    # it's appropriate to limit the number of workers to avoid memory overflow
    if optimal > 16:
        optimal = 16

    print(f"⚙️ Detected {cpu_count} cores, using {optimal} parallel processes")
    return optimal


def monitor_system_resources():
    """Monitor system resource usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        print(f"📊 Resource usage: CPU {cpu_percent:.1f}% | Memory {memory.percent:.1f}%")

        if cpu_percent > 95:
            print("⚠️ CPU usage is high")
        if memory.percent > 90:
            print("⚠️ Memory usage is high")

        return True
    except Exception:
        return False


def replace_base_class(code: str, new_class_def: str) -> str:
    """
    Cleanly ensures the resulting code has 'from manim import *',
    the new_class_def (base class and helpers), followed by the specific Scene class.
    Strips duplicate base class definitions, markdown markers, and unescaped quotes.
    """
    # 1. Strip markdown fences if present
    if "```python" in code:
        code = code.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in code:
        code = code.split("```", 1)[1].split("```", 1)[0].strip()

    # 2. Extract scene class: look for class <Name>Scene(TeachingScene): (excluding TeachingScene itself)
    match = re.search(r"(class\s+(?!TeachingScene\b)\w*Scene\s*\((?:TeachingScene|Scene)\)\s*:.*)", code, re.DOTALL)
    if match:
        scene_body = match.group(1).strip()
    else:
        # Fallback: look for any class inheriting from Scene that is not TeachingScene
        match = re.search(r"(class\s+(?!TeachingScene\b)\w+\s*\((?:TeachingScene|Scene)\)\s*:.*)", code, re.DOTALL)
        if match:
            scene_body = match.group(1).strip()
        else:
            scene_body = code.strip()

    # 3. Assemble: from manim import * + math + numpy + color aliases + base_class + scene_body
    header = (
        "from manim import *\n"
        "import math\n"
        "import numpy as np\n\n"
        "LIGHT_SALMON = '#FFA07A'\n"
        "LIGHT_BLUE = '#58C4DD'\n"
        "GREEN_SCREEN = '#83C167'\n"
        "GOLD_ACCENT = '#FFD700'\n"
        "LIGHT_GREEN = '#83C167'\n"
        "LIGHT_RED = '#FF6B6B'\n\n"
    )
    base = new_class_def.strip() + "\n\n"
    return header + base + scene_body + "\n"


# Save the program to the.py file
def save_code_to_file(code: str, filename: str = "scene.py"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Saved code to {filename}")


# Run the manim code to generate a video
def run_manim_script(filename: str, scene_name: str, output_dir: str = "videos") -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scene_name}.mp4")

    manim_cmd = shutil.which("manim") or str(Path(sys.executable).parent / "manim")
    cmd = [
        manim_cmd,
        "-pql",  # play + low quality（can changed to -pqm or -pqh）
        str(filename),  # script path
        scene_name,  # class name
        "--output_file",
        f"{scene_name}.mp4",
        "--media_dir",
        str(output_dir),  # media output directory
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Manim error:", result.stderr.decode())
        raise RuntimeError(f"Failed to render scene {scene_name}.")

    if os.path.exists(output_path):
        print(f"Video saved to {output_path}")
        return output_path

    # Search media_dir for rendered mp4
    file_stem = Path(filename).stem
    candidates = [
        Path(output_dir) / "videos" / file_stem / "480p15" / f"{scene_name}.mp4",
        Path(output_dir) / "videos" / "480p15" / f"{scene_name}.mp4",
    ]
    for c in candidates:
        if c.exists():
            print(f"Video saved to {c}")
            return str(c)

    found = list(Path(output_dir).rglob(f"{scene_name}.mp4"))
    if found:
        print(f"Video saved to {found[0]}")
        return str(found[0])

    print(f"Video saved to {output_path}")
    return output_path


# Use ffmpeg to concatenate multiple mp4 files
def stitch_videos(video_files: List[str], output_path: str = "final_output.mp4"):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    list_file = output_dir / "video_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for vf in video_files:
            f.write(f"file '{os.path.abspath(vf)}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output_path)]
    print("Stitching videos:", cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg stitching error: {result.stderr}")
        raise RuntimeError(f"Failed to stitch videos: {result.stderr}")
    if list_file.exists():
        list_file.unlink()
    print(f"Final stitched video saved to {output_path}")


def topic_to_safe_name(knowledge_point):
    # Allowed: alphanumeric Spaces _ - { } [ ] . , + & ' =
    SAFE_PATTERN = r"[^A-Za-z0-9 _\-\{\}\[\]\+&=\u03C0]"
    safe_name = re.sub(SAFE_PATTERN, "", knowledge_point)
    # Replace consecutive spaces with a single underscore
    safe_name = re.sub(r"\s+", "_", safe_name.strip())
    return safe_name


def get_output_dir(idx, knowledge_point, base_dir, level="intermediate", analogy="", get_safe_name=False, **kwargs):
    # Handle if 4th positional arg was a bool for get_safe_name:
    if isinstance(level, bool):
        get_safe_name = level
        level = kwargs.get("level", "intermediate")
        analogy = kwargs.get("analogy", "")

    safe_name = topic_to_safe_name(knowledge_point)
    norm_level = (level or "intermediate").lower()
    suffix_parts = []

    # Non-default level or analogy gets reflected in folder name to avoid cache collision
    if norm_level in ("beginner", "advanced"):
        suffix_parts.append(norm_level)
    elif norm_level != "intermediate":
        suffix_parts.append(norm_level)

    if analogy and str(analogy).strip():
        safe_analogy = topic_to_safe_name(str(analogy).strip())[:24].strip("_")
        if safe_analogy:
            suffix_parts.append(safe_analogy)

    if suffix_parts:
        folder_name = f"{idx}-{safe_name}_{'_'.join(suffix_parts)}"
    else:
        folder_name = f"{idx}-{safe_name}"

    if get_safe_name:
        return Path(base_dir) / folder_name, safe_name

    return Path(base_dir) / folder_name



def eva_video_list(knowledge_points, base_dir):

    video_list = []
    for idx, kp in enumerate(knowledge_points):
        folder, safe_name = get_output_dir(idx, kp, base_dir, get_safe_name=True)

        # mp4 filename must be safe, the same
        mp4_name = f"{safe_name}.mp4"
        mp4_path = folder / mp4_name
        video_list.append({"path": str(mp4_path), "knowledge_point": kp})
    return video_list


if __name__ == "__main__":
    print(get_optimal_workers())
