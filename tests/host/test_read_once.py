"""A value the core reads once from a provider or an audio output must not change afterwards.

The core reads the ``read_once`` members of a host object when it takes it (a session takes its
providers at registration) and keeps that value for the object's life. The codegen allows
``read_once`` only for values that cannot change after construction
(``rust/ai_slop/lk-agents-codegen/src/decl/resolve/read_once.rs``); this check holds the Python
side to it: no method but ``__init__`` assigns an attribute such a value is read from.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "python"
AGENTS = ROOT / "livekit-agents" / "livekit" / "agents"

# The read-once members of each base, as declared in the ABI (`#[host(read_once)]`).
BASES: dict[str, tuple[Path, frozenset[str]]] = {
    "TTS": (
        AGENTS / "_generated" / "tts.py",
        frozenset(
            {
                "_core_instance_id",
                "_core_label",
                "_core_capabilities",
                "_core_num_channels",
                "_core_retries_itself",
                "_core_conn_options",
                "_core_options",
            }
        ),
    ),
    "STT": (
        AGENTS / "_generated" / "stt.py",
        frozenset(
            {
                "_core_instance_id",
                "_core_label",
                "_core_capabilities",
                "_core_retries_itself",
                "_core_conn_options",
            }
        ),
    ),
    "LLM": (
        AGENTS / "_generated" / "llm.py",
        frozenset({"_core_instance_id", "_core_label", "_core_retries_itself"}),
    ),
    "RealtimeModel": (
        AGENTS / "_generated" / "realtime.py",
        frozenset({"_core_instance_id", "_core_label", "_core_capabilities"}),
    ),
    "VAD": (AGENTS / "_generated" / "vad.py", frozenset({"_core_label", "_core_capabilities"})),
    "AudioOutput": (AGENTS / "voice" / "io.py", frozenset({"label", "can_pause", "sample_rate"})),
}


def _self_path(node: ast.expr) -> tuple[str, ...] | None:
    """``self.a.b`` as ``("a", "b")``; ``None`` for anything not rooted at ``self``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == "self" and parts:
        return tuple(reversed(parts))
    return None


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {f.name: f for f in cls.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _class(path: Path, name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


@dataclass
class _Provider:
    kind: str
    # own methods first, then each base's, the framework base last
    lineage: list[ast.ClassDef] = field(default_factory=list)

    def method(self, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        for cls in self.lineage:
            if (m := _methods(cls).get(name)) is not None:
                return m
        return None

    def backing(self) -> set[tuple[str, ...]]:
        """The attribute paths the read-once members read, through the properties they call."""
        found: set[tuple[str, ...]] = set()
        seen: set[str] = set()
        todo = list(BASES[self.kind][1])
        while todo:
            name = todo.pop()
            if name in seen or (method := self.method(name)) is None:
                continue
            seen.add(name)
            for node in ast.walk(method):
                if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
                    continue
                path = _self_path(node)
                if path is None:
                    continue
                if self.method(path[0]) is not None:
                    todo.append(path[0])
                else:
                    found.add(path)
        # keep the longest reads: `self._opts.auto_mode`, not also `self._opts`
        return {p for p in found if not any(q != p and q[: len(p)] == p for q in found)}

    def assignments_after_init(self) -> list[str]:
        backing = self.backing()
        out = []
        # the framework base's own methods are checked as a provider of their own
        for cls in self.lineage[:-1] if len(self.lineage) > 1 else self.lineage:
            for name, method in _methods(cls).items():
                if name == "__init__":
                    continue
                for node in ast.walk(method):
                    targets: list[ast.expr] = []
                    if isinstance(node, ast.Assign):
                        targets = node.targets
                    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                        targets = [node.target]
                    for target in targets:
                        path = _self_path(target)
                        if path and any(path[: len(b)] == b for b in backing):
                            out.append(f"{cls.name}.{name} assigns self.{'.'.join(path)}")
        return out


def _bases(cls: ast.ClassDef) -> list[tuple[str, bool]]:
    """Each base's name, and whether it is a bare name (a class of the same file) or qualified."""
    out = []
    for base in cls.bases:
        node = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(node, ast.Name):
            out.append((node.id, True))
        elif isinstance(node, ast.Attribute):
            out.append((node.attr, False))
    return out


def _providers(path: Path) -> list[_Provider]:
    """The classes of ``path`` deriving from a base above, through classes of the same file."""
    tree = ast.parse(path.read_text())
    local = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    found = []
    for cls in local.values():
        lineage, kind = [cls], None
        while kind is None:
            bases = _bases(lineage[-1])
            here = next(
                (local[n] for n, bare in bases if bare and n in local and local[n] not in lineage),
                None,
            )
            kind = next((n for n, bare in bases if n in BASES and not (bare and n in local)), None)
            if kind is None and here is not None:
                lineage.append(here)
                continue
            break
        if kind is None:
            continue
        base_path, _ = BASES[kind]
        if path == base_path and cls.name == kind:
            continue
        lineage.append(_class(base_path, kind))
        found.append(_Provider(kind, lineage))
    return found


def _sources() -> list[Path]:
    plugins = ROOT / "livekit-plugins"
    return sorted(AGENTS.rglob("*.py")) + sorted(plugins.glob("*/livekit/**/*.py"))


def _violations(paths: list[Path]) -> list[str]:
    out = []
    for path in paths:
        for provider in _providers(path):
            out += [
                f"{path.as_posix().rpartition('/python/')[2]}: {v}"
                for v in provider.assignments_after_init()
            ]
    for kind, (path, _) in BASES.items():
        out += [
            f"{path.as_posix().rpartition('/python/')[2]}: {v}"
            for v in _Provider(kind, [_class(path, kind)]).assignments_after_init()
        ]
    return out


# Catches: a provider whose `update_options` changes a value the session read once when it took
# the provider (a capability, its label, its channels), which the session would never see.
def test_no_read_once_value_is_assigned_after_construction() -> None:
    assert _violations(_sources()) == []


# Catches: the check passing because it finds nothing, not because nothing is wrong.
def test_the_check_finds_a_capability_changed_by_update_options(tmp_path: Path) -> None:
    plugin = tmp_path / "plugin.py"
    plugin.write_text(
        "from livekit.agents import tts\n"
        "class _Base(tts.TTS):\n"
        "    def _core_num_channels(self):\n"
        "        return 2 if self._opts.stereo else 1\n"
        "class TTS(_Base):\n"
        "    def __init__(self):\n"
        "        self._capabilities = None\n"
        "    def update_options(self, *, streaming, stereo):\n"
        "        self._capabilities.streaming = streaming\n"
        "        self._opts.stereo = stereo\n"
        "        self._sample_rate = 16000\n"
    )
    found = "\n".join(_violations([plugin]))
    assert "TTS.update_options assigns self._capabilities.streaming" in found
    assert "TTS.update_options assigns self._opts.stereo" in found
    # the sample rate is read when each request is made, so changing it is fine
    assert "_sample_rate" not in found
