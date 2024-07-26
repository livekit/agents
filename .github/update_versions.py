import json
import pathlib


def update_version(project_root: pathlib.Path, py_version_path: pathlib.Path):
    with open(project_root / "package.json") as f:
        package = json.load(f)
        version = package["version"]

    with open(py_version_path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("__version__"):
                lines[i] = f'__version__ = "{version}"\n'
                break

    with open(py_version_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    livekit_agents = pathlib.Path.cwd() / "livekit-agents"
    update_version(livekit_agents, livekit_agents / "livekit" / "agents" / "version.py")

    plugins_root = pathlib.Path.cwd() / "livekit-plugins"
    plugins = plugins_root.iterdir()
    for plugin in plugins:
        if not plugin.is_dir():
            continue

        plugin_name = plugin.name.split("-")[-1]
        py_version_path = plugin / "livekit" / "plugins" / plugin_name / "version.py"
        update_version(plugin, py_version_path)
