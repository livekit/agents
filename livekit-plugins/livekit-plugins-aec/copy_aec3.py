import re
import sys
import shutil
from pathlib import Path

INCLUDE_PATTERN = re.compile(r'#include\s*[<"](.+?)[">]')
ALLOWED_FOLDERS = {
    "modules",
    "api",
    "rtc_base",
    "system_wrappers",
    "rtc_tools",
    "common_audio",
}
SKIP_SUBSTRINGS = ("unittest", "avx2", "mock")


def should_skip_file(file_path: Path) -> bool:
    name_lower = file_path.name.lower()
    return any(sub in name_lower for sub in SKIP_SUBSTRINGS)


MANDATORY_PATHS = [
    "api/environment",
]


def copy_file_with_deps(
    root_input: Path, root_output: Path, rel_path: Path, visited: set
):
    if rel_path in visited:
        return
    visited.add(rel_path)

    in_file = root_input / rel_path
    if not in_file.is_file():
        print(f"Warning: File not found (skipped): {in_file}")
        return

    if should_skip_file(in_file):
        print(f"Skipping file (due to skip rules): {in_file}")
        return

    out_file = root_output / rel_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(in_file, out_file)

    with in_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = INCLUDE_PATTERN.search(line)
            if not match:
                continue

            include_str = match.group(1).strip()
            include_path = Path(include_str)

            if include_path.is_absolute():
                continue

            first_part = include_path.parts[0] if include_path.parts else ""

            if first_part == "third_party":
                print(f"Warning: Include found from third_party: {include_path}")

            if first_part in ALLOWED_FOLDERS:
                dep_file = root_input / include_path
                if not dep_file.is_file():
                    print(f"Warning: Could not find included file: {dep_file}")
                else:
                    copy_file_with_deps(root_input, root_output, include_path, visited)


def copy_directory(dir_path: Path, root_input: Path, root_output: Path, visited: set):
    if not dir_path.is_dir():
        print(f"Warning: Directory not found or not valid: {dir_path}")
        return

    for file_path in dir_path.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() in [".h", ".hpp", ".c", ".cc", ".cpp"]:
            rel_path = file_path.relative_to(root_input)
            copy_file_with_deps(root_input, root_output, rel_path, visited)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_webrtc_root> <output_folder>")
        sys.exit(1)

    input_webrtc_root = Path(sys.argv[1]).resolve()
    output_folder = Path(sys.argv[2]).resolve()

    aec3_path = input_webrtc_root / "modules" / "audio_processing" / "aec3"
    if not aec3_path.is_dir():
        print(f"Error: The path {aec3_path} does not exist or is not a directory.")
        sys.exit(1)

    visited_files = set()

    copy_directory(aec3_path, input_webrtc_root, output_folder, visited_files)

    for mandatory_rel in MANDATORY_PATHS:
        mandatory_path = input_webrtc_root / mandatory_rel
        copy_directory(mandatory_path, input_webrtc_root, output_folder, visited_files)

    print(f"Done. Copied {len(visited_files)} files into {output_folder}")


if __name__ == "__main__":
    main()
