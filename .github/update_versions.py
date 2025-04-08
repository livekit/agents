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
    """Return a dict of package -> (bump_type, [changelogs]) aggregated."""
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
                # If the new bump is higher in BUMP_ORDER, override
                if BUMP_ORDER[bump_type] > BUMP_ORDER[cur_bump]:
                    cur_bump = bump_type
                changelogs.append(changelog)
                agg[pkg] = (cur_bump, changelogs)
    return agg

def read_version(f: pathlib.Path) -> str:
    """Read __version__ = 'X.Y.Z' from a Python file."""
    text = f.read_text()
    m = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    if not m:
        raise ValueError(f"could not find __version__ in {f}")
    return m.group(1)

def write_new_version(f: pathlib.Path, new_version: str) -> None:
    """Substitute a new __version__ = 'X.Y.Z' in place of the existing one."""
    text = f.read_text()
    new_text = re.sub(
        r'__version__\s*=\s*[\'"][^\'"]*[\'"]',
        f"__version__ = '{new_version}'",
        text,
        count=1
    )
    f.write_text(new_text)

def bump_version(cur: str, bump_type: str) -> str:
    """Bump the version (major/minor/patch) based on the bump_type."""
    v = Version(cur)
    if bump_type == "patch":
        return f"{v.major}.{v.minor}.{v.micro + 1}"
    if bump_type == "minor":
        return f"{v.major}.{v.minor + 1}.0"
    if bump_type == "major":
        return f"{v.major + 1}.0.0"
    raise ValueError(f"unknown bump type: {bump_type}")

def bump_prerelease(cur: str, bump_type: str) -> str:
    """Bump the version in a pre-release style (rc/dev)."""
    v = Version(cur)
    base = v.base_version

    if bump_type == "dev":
        # increment dev if present, else start dev0
        next_dev = (v.dev + 1) if v.dev is not None else 0
        return f"{base}.dev{next_dev}"

    elif bump_type == "rc":
        # if already on an rc, increment; otherwise start at rc1
        if v.pre and v.pre[0] == "rc":
            next_rc = v.pre[1] + 1
        else:
            next_rc = 1
        return f"{base}.rc{next_rc}"

    else:
        raise ValueError(f"unknown prerelease bump type: {bump_type}")

def update_plugins_pyproject_agents_version(new_agents_version: str) -> None:
    """
    For each plugin directory that has a pyproject.toml at the top level,
    update any references to livekit-agents>=X.Y.Z (or ==, ~=, etc.)
    to the new_agents_version.
    """
    plugins_root = pathlib.Path("livekit-plugins")
    for pdir in plugins_root.glob("livekit-plugins-*"):
        pyproject = pdir / "pyproject.toml"
        if pyproject.exists():
            old_text = pyproject.read_text()
            # Pattern:
            # Group 1 => "livekit-agents" possibly with extras e.g. [images],
            # then one or more version operators like >=, ==, ~=, etc.
            # Group 2 => the old version (digits, dots, dev, rc, etc.)
            pattern = r'("livekit-agents(?:\[.*?\])?[=><!~]+)([\w\.\-]+)(?=")'

            # Use a replacement function so we don't run into '\1' + '1...' -> '\11...' confusion
            def replacer(m: re.Match) -> str:
                group1 = m.group(1)  # e.g. 'livekit-agents[images]>='
                # We discard the old version (group 2) and insert the new.
                return f"{group1}{new_agents_version}"

            new_text = re.sub(pattern, replacer, old_text)
            if new_text != old_text:
                pyproject.write_text(new_text)
                print(f"Updated pyproject.toml in {pdir.name} to use livekit-agents {new_agents_version}")

def update_versions(changesets: Dict[str, Tuple[str, List[str]]]) -> None:
    """
    Given changesets {package: (bump_type, [changelogs])}, 
    bump versions accordingly and also update references in pyproject.toml if 
    'livekit-agents' was updated.
    """
    agents_ver = pathlib.Path("livekit-agents/livekit/agents/version.py")
    plugins_root = pathlib.Path("livekit-plugins")

    new_agents_version = None

    # handle livekit-agents
    if agents_ver.exists() and "livekit-agents" in changesets:
        bump_type, _ = changesets["livekit-agents"]
        cur = read_version(agents_ver)
        new = bump_version(cur, bump_type)
        print(f"livekit-agents: {_esc(31)}{cur}{_esc(0)} -> {_esc(32)}{new}{_esc(0)}")
        write_new_version(agents_ver, new)
        new_agents_version = new
    else:
        print("Warning: No version.py or no bump info for livekit-agents.")

    # handle each plugin's version.py
    for pdir in plugins_root.glob("livekit-plugins-*"):
        vf = pdir / "livekit" / "plugins" / pdir.name.split("livekit-plugins-")[1].replace("-", "_") / "version.py"
        if vf.exists():
            if pdir.name in changesets:
                bump_type, _ = changesets[pdir.name]
                cur = read_version(vf)
                new = bump_version(cur, bump_type)
                print(f"{pdir.name}: {_esc(31)}{cur}{_esc(0)} -> {_esc(32)}{new}{_esc(0)}")
                write_new_version(vf, new)
            else:
                print(f"Warning: Found version.py for {pdir.name}, but no bump info in next-release.")
        else:
            print(f"Warning: version.py not found for {pdir.name} at {vf}")

    # If we updated livekit-agents version, also update references in each plugin's pyproject.toml
    if new_agents_version:
        update_plugins_pyproject_agents_version(new_agents_version)

def update_prerelease(prerelease_type: str) -> None:
    """
    Perform prerelease (rc or dev) bumps everywhere and also update references in 
    plugin pyproject.toml to the new livekit-agents version, if changed.
    """
    agents_ver = pathlib.Path("livekit-agents/livekit/agents/version.py")
    plugins_root = pathlib.Path("livekit-plugins")

    new_agents_version = None

    # handle livekit-agents
    if agents_ver.exists():
        cur = read_version(agents_ver)
        new = bump_prerelease(cur, prerelease_type)
        print(f"livekit-agents: {_esc(31)}{cur}{_esc(0)} -> {_esc(32)}{new}{_esc(0)}")
        write_new_version(agents_ver, new)
        new_agents_version = new
    else:
        print("Warning: No version.py for livekit-agents.")

    # handle each plugin
    for pdir in plugins_root.glob("livekit-plugins-*"):
        vf = pdir / "livekit" / "plugins" / pdir.name.split("livekit-plugins-")[1].replace("-", "_") / "version.py"
        if vf.exists():
            cur = read_version(vf)
            new_v = bump_prerelease(cur, prerelease_type)
            print(f"{pdir.name}: {_esc(31)}{cur}{_esc(0)} -> {_esc(32)}{new_v}{_esc(0)}")
            write_new_version(vf, new_v)
        else:
            print(f"Warning: version.py not found for {pdir.name} at {vf}")

    # If we updated livekit-agents version, update references in each plugin's pyproject.toml
    if new_agents_version:
        update_plugins_pyproject_agents_version(new_agents_version)

@click.command("bump")
@click.option(
    "--pre", 
    type=click.Choice(["rc", "dev", "none"]), 
    default="none", 
    help="Pre-release type. Use 'none' for normal bump, or 'rc'/'dev' for pre-release."
)
def bump(pre: str) -> None:
    """
    Single command to do either normal or pre-release bumps.

    By default (with --pre=none), this uses changesets in .github/next-release
    to determine major/minor/patch increments per package.

    If --pre=rc or --pre=dev, it updates the current versions to a new RC or DEV version.
    In both cases, we also update pyproject.toml references that point to 
    'livekit-agents>=XYZ' so they match the newly bumped version.
    """
    if pre == "none":
        # Normal release bump: read bumps from changesets
        changesets = load_changesets(pathlib.Path(".github/next-release"))
        # changesets => { "some-package": ("minor", ["changelog1", "changelog2"]), ... }
        update_versions(changesets)
    else:
        # Pre-release bump
        update_prerelease(pre)

if __name__ == "__main__":
    bump()
