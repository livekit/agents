from __future__ import annotations

import json
import pathlib
import re


def update_py_version(
    project_root: pathlib.Path, py_version_path: pathlib.Path
) -> str | None:
    pkg_file = project_root / "package.json"
    if not pkg_file.exists():
        return

    with open(pkg_file) as f:
        package = json.load(f)
        version = package["version"]

    with open(py_version_path) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("__version__"):
                lines[i] = f'__version__ = "{version}"\n'
                break

    with open(py_version_path, "w") as f:
        f.writelines(lines)

    return version


def update_requirements_txt(example_dir: pathlib.Path, last_versions: dict[str, str]):
    # recursively find all requirements.txt files
    requirements_files = example_dir.rglob("requirements.txt")

    for req_file in requirements_files:
        with open(req_file) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = re.split(r"(>=|==|<=|~=|!=|>)", line.strip())
            if len(parts) <= 1:
                continue

            pkg_name = parts[0].strip()
            if pkg_name in last_versions:
                lines[i] = f"{pkg_name}>={last_versions[pkg_name]}\n"

        with open(req_file, "w") as f:
            f.writelines(lines)


if __name__ == "__main__":
    package_versions = {}

    agents_root = pathlib.Path.cwd() / "livekit-agents"
    plugins_root = pathlib.Path.cwd() / "livekit-plugins"
    examples_root = pathlib.Path.cwd() / "examples"

    agent_version = update_py_version(
        agents_root, agents_root / "livekit" / "agents" / "version.py"
    )
    package_versions["livekit-agents"] = agent_version

    for plugin in plugins_root.iterdir():
        if not plugin.is_dir():
            continue

        plugin_name = plugin.name.removeprefix("livekit-plugins-")
        plugin_name = plugin_name.replace("-", "_")  # module name can't have dashes

        version = update_py_version(
            plugin, plugin / "livekit" / "plugins" / plugin_name / "version.py"
        )
        package_versions[plugin.name] = version

    # update requirements.txt of our examples
    update_requirements_txt(examples_root, package_versions)
