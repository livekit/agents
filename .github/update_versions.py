import pathlib
import re
import yaml
import click
from packaging.version import Version
import colorama
from typing import Dict, Tuple, List

colorama.init()

BUMP_ORDER: Dict[str, int] = {"patch": 0, "minor": 1, "major": 2}

def _esc(*codes: int) -> str:
    return "\033[" + ";".join(str(c) for c in codes) + "m"

def parse_changeset_file(path: pathlib.Path) -> Tuple[Dict[str, str], str]:
    """Parse a changeset file of the form:
    
        ---
        some_yaml_bumps
        ---
        changelog text
    """
    text = path.read_text()
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid changeset file format in {path}")
    
    yaml_part = parts[1].strip()
    changelog_part = parts[2].strip()
    data = yaml.safe_load(yaml_part)
    return data, changelog_part

def load_changesets(dir: pathlib.Path) -> Dict[str, Tuple[str, List[str]]]:
    agg: Dict[str, Tuple[str, List[str]]] = {}
    if not dir.is_dir():
        raise ValueError(f"{dir} is not a directory or does not exist.")
    
    for file in dir.glob("*"):
        if not file.is_file():
            continue
        data, changelog = parse_changeset_file(file)
        for pkg, bump_type in data.items():
            if pkg not in agg:
                agg[pkg] = (bump_type, [changelog])
            else:
                cur_bump, changelogs = agg[pkg]
                if BUMP_ORDER[bump_type] > BUMP_ORDER[cur_bump]:
                    cur_bump = bump_type
                changelogs.append(changelog)
                agg[pkg] = (cur_bump, changelogs)
    return agg

def read_version(f: pathlib.Path) -> str:
    """Read __version__ = \"X.Y.Z\" from a Python file."""
    text = f.read_text()
    m = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    if not m:
        raise ValueError(f"could not find __version__ in {f}")
    return m.group(1)

def write_new_version(f: pathlib.Path, new_version: str) -> None:
    """Substitute a new __version__ = \"X.Y.Z\" in place of the existing one."""
    text = f.read_text()
    new_text = re.sub(
        r'__version__\s*=\s*[\'"][^\'"]*[\'"]',
        f'__version__ = "{new_version}"',
        text,
        count=1
    )
    f.write_text(new_text)

def bump_version(cur: str, bump_type: str) -> str:
    v = Version(cur)
    if bump_type == "patch":
        return f"{v.major}.{v.minor}.{v.micro + 1}"
    if bump_type == "minor":
        return f"{v.major}.{v.minor + 1}.0"
    if bump_type == "major":
        return f"{v.major + 1}.0.0"
    raise ValueError(f"unknown bump type: {bump_type}")

def bump_prerelease(cur: str, bump_type: str) -> str:
    v = Version(cur)
    base = v.base_version

    if bump_type == "dev":
        next_dev = (v.dev + 1) if v.dev is not None else 0
        return f"{base}.dev{next_dev}"
    elif bump_type == "rc":
        if v.pre and v.pre[0] == "rc":
            next_rc = v.pre[1] + 1
        else:
            next_rc = 1
        return f"{base}.rc{next_rc}"
    else:
        raise ValueError(f"unknown prerelease bump type: {bump_type}")

def update_plugins_pyproject_agents_version(new_agents_version: str) -> None:
    plugins_root = pathlib.Path("livekit-plugins")
    for pdir in plugins_root.glob("livekit-plugins-*"):
        pyproject = pdir / "pyproject.toml"
        if pyproject.exists():
            old_text = pyproject.read_text()
            pattern = r'("livekit-agents(?:\[.*?\])?[=><!~]+)([\w\.\-]+)(?=")'

            def replacer(m: re.Match) -> str:
                return f"{m.group(1)}{new_agents_version}"

            new_text = re.sub(pattern, replacer, old_text)
            if new_text != old_text:
                pyproject.write_text(new_text)
                print(f"Updated pyproject.toml in {pdir.name} to use livekit-agents {new_agents_version}")

def update_agents_pyproject_optional_dependencies(plugin_versions: Dict[str, str]) -> None:
    agents_pyproject = pathlib.Path("livekit-agents/pyproject.toml")
    if not agents_pyproject.exists():
        print("Warning: livekit-agents/pyproject.toml not found")
        return
        
    old_text = agents_pyproject.read_text()
    new_text = old_text
    
    for plugin_name, new_version in plugin_versions.items():
        if plugin_name.startswith("livekit-plugins-"):
            dep_key = plugin_name[len("livekit-plugins-"):]
            pattern = rf'({dep_key}\s*=\s*\["livekit-plugins-{re.escape(dep_key)}>=)([\w\.\-]+)(\"])'
            
            def replacer(m: re.Match) -> str:
                return f"{m.group(1)}{new_version}{m.group(3)}"
            
            updated_text = re.sub(pattern, replacer, new_text)
            if updated_text != new_text:
                new_text = updated_text
                print(f"Updated optional dependency {dep_key} to version {new_version}")
    
    if new_text != old_text:
        agents_pyproject.write_text(new_text)
        print("Updated livekit-agents/pyproject.toml optional-dependencies")

def update_versions(changesets: Dict[str, Tuple[str, List[str]]]) -> None:
    agents_ver = pathlib.Path("livekit-agents/livekit/agents/version.py")
    plugins_root = pathlib.Path("livekit-plugins")

    new_agents_version = None
    plugin_versions = {}

    if agents_ver.exists() and "livekit-agents" in changesets:
        bump_type, _ = changesets["livekit-agents"]
        cur = read_version(agents_ver)
        new = bump_version(cur, bump_type)
        print(f"livekit-agents: {_esc(31)}{cur}{_esc(0)} -> {_esc(32)}{new}{_esc(0)}")
        write_new_version(agents_ver, new)
        new_agents_version = new
    else:
        print("Warning: No version.py or no bump info for livekit-agents.")

    for pdir in plugins_root.glob("livekit-plugins-*"):
        vf = pdir / "livekit" / "plugins" / pdir.name.split("livekit-plugins-")[1].replace("-", "_") / "version.py"
        if vf.exists():
            if pdir.name in changesets:
                bump_type, _ = changesets[pdir.name]
                cur = read_version(vf)
                new = bump_version(cur, bump_type)
                print(f"{pdir.name}: {_esc(31)}{cur}{_esc(0)} -> {_esc(32)}{new}{_esc(0)}")
                write_new_version(vf, new)
                plugin_versions[pdir.name] = new
            else:
                print(f"Warning: Found version.py for {pdir.name}, but no bump info in next-release.")
        else:
            print(f"Warning: version.py not found for {pdir.name} at {vf}")

    if new_agents_version:
        update_plugins_pyproject_agents_version(new_agents_version)
    
    if plugin_versions:
        update_agents_pyproject_optional_dependencies(plugin_versions)

def update_versions_ignore_changesets(bump_type: str) -> None:
    agents_ver = pathlib.Path("livekit-agents/livekit/agents/version.py")
    plugins_root = pathlib.Path("livekit-plugins")
    
    new_agents_version = None
    plugin_versions = {}

    if agents_ver.exists():
        cur = read_version(agents_ver)
        new = bump_version(cur, bump_type)
        print(f"livekit-agents: {_esc(31)}{cur}{_esc(0)} -> {_esc(32)}{new}{_esc(0)}")
        write_new_version(agents_ver, new)
        new_agents_version = new
    else:
        print("Warning: No version.py found for livekit-agents.")

    for pdir in plugins_root.glob("livekit-plugins-*"):
        vf = pdir / "livekit" / "plugins" / pdir.name.split("livekit-plugins-")[1].replace("-", "_") / "version.py"
        if vf.exists():
            cur = read_version(vf)
            new = bump_version(cur, bump_type)
            print(f"{pdir.name}: {_esc(31)}{cur}{_esc(0)} -> {_esc(32)}{new}{_esc(0)}")
            write_new_version(vf, new)
            plugin_versions[pdir.name] = new
        else:
            print(f"Warning: version.py not found for {pdir.name} at {vf}")

    if new_agents_version:
        update_plugins_pyproject_agents_version(new_agents_version)
    
    if plugin_versions:
        update_agents_pyproject_optional_dependencies(plugin_versions)

def update_prerelease(prerelease_type: str) -> None:
    agents_ver = pathlib.Path("livekit-agents/livekit/agents/version.py")
    plugins_root = pathlib.Path("livekit-plugins")

    new_agents_version = None
    plugin_versions = {}

    if agents_ver.exists():
        cur = read_version(agents_ver)
        new = bump_prerelease(cur, prerelease_type)
        print(f"livekit-agents: {_esc(31)}{cur}{_esc(0)} -> {_esc(32)}{new}{_esc(0)}")
        write_new_version(agents_ver, new)
        new_agents_version = new
    else:
        print("Warning: No version.py for livekit-agents.")

    for pdir in plugins_root.glob("livekit-plugins-*"):
        vf = pdir / "livekit" / "plugins" / pdir.name.split("livekit-plugins-")[1].replace("-", "_") / "version.py"
        if vf.exists():
            cur = read_version(vf)
            new_v = bump_prerelease(cur, prerelease_type)
            print(f"{pdir.name}: {_esc(31)}{cur}{_esc(0)} -> {_esc(32)}{new_v}{_esc(0)}")
            write_new_version(vf, new_v)
            plugin_versions[pdir.name] = new_v
        else:
            print(f"Warning: version.py not found for {pdir.name} at {vf}")

    if new_agents_version:
        update_plugins_pyproject_agents_version(new_agents_version)
    
    if plugin_versions:
        update_agents_pyproject_optional_dependencies(plugin_versions)

@click.command("bump")
@click.option(
    "--pre", 
    type=click.Choice(["rc", "dev", "none"]), 
    default="none", 
    help="Pre-release type. Use 'none' for normal bump, or 'rc'/'dev' for pre-release."
)
@click.option(
    "--ignore-changesets",
    is_flag=True,
    default=False,
    help="Ignore changeset files and bump all packages using a uniform bump type."
)
@click.option(
    "--bump-type",
    type=click.Choice(["patch", "minor", "major"]),
    default="patch",
    help="Type of version bump to apply when ignoring changesets. Defaults to patch."
)
def bump(pre: str, ignore_changesets: bool, bump_type: str) -> None:
    """
    Single command to do either normal or pre-release bumps.

    For a normal release (with --pre=none), by default the script uses changesets from
    .github/next-release to determine per-package bump types. Use --ignore-changesets to ignore
    the changesets and bump every package with the specified --bump-type.
    
    For pre-release bumps (--pre=rc or --pre=dev), it updates the current versions to a new RC or DEV version.
    In both cases, plugin pyproject.toml references for 'livekit-agents' will be updated if that version changes.
    """
    if pre == "none":
        if ignore_changesets:
            update_versions_ignore_changesets(bump_type)
        else:
            changesets = load_changesets(pathlib.Path(".github/next-release"))
            update_versions(changesets)
    else:
        update_prerelease(pre)

if __name__ == "__main__":
    bump()
