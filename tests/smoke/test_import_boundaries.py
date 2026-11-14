"""The layers' import boundaries: plugins reach only public ``livekit.agents`` names, the
transports know no kind, and no kind builds on another kind's WebSocket base."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "python" / "livekit-agents" / "livekit" / "agents"
PLUGINS = ROOT / "python" / "livekit-plugins"
KINDS = {"stt", "tts", "llm"}


def _sources(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        if not {"__pycache__", ".venv", "build"} & set(path.parts):
            yield path


def _imports(path: Path) -> Iterator[tuple[int, str, list[str]]]:
    """Each import of ``path``: its line, the absolute module, the names it takes."""
    package = path.relative_to(ROOT / "python" / "livekit-agents").parts[:-1]
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, []
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                base = package[: len(package) - node.level + 1]
                module = ".".join([*base, module] if module else base)
            yield node.lineno, module, [alias.name for alias in node.names]


def _plugin_imports(path: Path) -> Iterator[tuple[int, str, list[str]]]:
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, []
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            yield node.lineno, node.module, [alias.name for alias in node.names]


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


# Catches: a plugin reaching into the framework's private modules or names (the WebSocket
# plumbing, the bindings, a private base), which then breaks on any internal change.
def test_plugins_import_only_public_livekit_agents_names() -> None:
    found = []
    for path in _sources(PLUGINS):
        for line, module, names in _plugin_imports(path):
            if module != "livekit.agents" and not module.startswith("livekit.agents."):
                continue
            taken = [n for n in [*module.split("."), *names] if _private(n)]
            if taken:
                found.append(f"{path.relative_to(ROOT)}:{line}: {module} {taken}")
    assert found == []


def _kinds_imported(path: Path) -> list[str]:
    found = []
    for line, module, names in _imports(path):
        parts = module.split(".")
        if parts[:2] != ["livekit", "agents"]:
            continue
        rest = parts[2:]
        taken = {rest[0]} if rest else set(names)
        if taken & KINDS:
            found.append(f"{path.relative_to(ROOT)}:{line}: {module} {names}")
    return found


# Catches: the HTTP client or the WebSocket plumbing growing kind vocabulary (audio, segments,
# transcripts) by importing a kind, so a transport change drags every kind along.
def test_the_transports_import_no_kind() -> None:
    transports = [*_sources(AGENTS / "http"), AGENTS / "_websocket.py"]
    assert [found for path in transports for found in _kinds_imported(path)] == []


# Catches: one kind building on another's WebSocket base or protocol terms (a TTS stream using
# the STT stream's flush rules), which couples kinds that work differently.
def test_no_kind_imports_another_kind() -> None:
    kinds = {
        "stt": AGENTS / "stt.py",
        "tts": AGENTS / "tts.py",
        "llm": AGENTS / "llm" / "realtime.py",
    }
    found = []
    for kind, path in kinds.items():
        for line, module, names in _imports(path):
            parts = module.split(".")
            if parts[:2] != ["livekit", "agents"] or len(parts) < 3 and not names:
                continue
            other = ({parts[2]} if len(parts) > 2 else set(names)) & (KINDS - {kind})
            if other:
                found.append(f"{path.relative_to(ROOT)}:{line}: {module} {names}")
    assert found == []


WEBSOCKET_LAYER = [
    AGENTS / "_websocket.py",
    AGENTS / "stt.py",
    AGENTS / "tts.py",
    AGENTS / "llm" / "realtime.py",
    AGENTS / "llm" / "openai_protocol.py",
    *_sources(AGENTS / "http"),
]
KIND_WORDS = re.compile(
    r"audio|speech|transcri|segment|flush|synthes|recogni|realtime|media|utterance", re.I
)


# Catches: kind vocabulary (audio, segments, flushes, transcripts) creeping into the shared
# WebSocket plumbing, in a name, a docstring or a comment, so its rules bend toward one kind.
def test_the_websocket_plumbing_speaks_no_kinds_vocabulary() -> None:
    path = AGENTS / "_websocket.py"
    found = [
        f"{path.relative_to(ROOT)}:{line}: {text.strip()}"
        for line, text in enumerate(path.read_text().splitlines(), 1)
        if KIND_WORDS.search(text)
    ]
    assert found == []


# Catches: the shared WebSocket layer importing a kind's generated records or bindings, which
# drags that kind's vocabulary into every kind.
def test_the_websocket_plumbing_imports_no_kinds_generated_module() -> None:
    path = AGENTS / "_websocket.py"
    found = [
        f"{line}: {module} {names}"
        for line, module, names in _imports(path)
        if module.startswith("livekit.agents._generated")
        and (set(module.split(".")[-1].split("_")) | set(names)) & {*KINDS, "realtime"}
    ]
    assert found == []


def _is_module(package: Path, name: str) -> bool:
    return any((package / candidate).exists() for candidate in (f"{name}.py", f"{name}.pyi", name))


# Catches: a WebSocket layer reaching into another module's private names (a helper, a hook),
# so a change there breaks it silently; it takes only public names, and private modules whole.
def test_the_websocket_layers_import_no_private_name_from_another_module() -> None:
    found = []
    for path in WEBSOCKET_LAYER:
        for line, module, names in _imports(path):
            if not module.startswith("livekit.agents"):
                continue
            package = ROOT / "python" / "livekit-agents" / Path(*module.split("."))
            for name in names:
                if _private(name) and not _is_module(package, name):
                    found.append(f"{path.relative_to(ROOT)}:{line}: {module} {name}")
    assert found == []


SHARED_NAMES = {
    "Keepalive",
    "WebSocketMultiplexer",
    "WebSocketPacing",
    "WebSocketTarget",
    "protocol_error",
}


# Catches: a kind re-exporting a marker of the shared layer (one ``Done`` or ``Ready`` meaning
# something else in each kind) instead of declaring what its plugins report in its own terms.
def test_kinds_reexport_only_the_shared_plumbing_names() -> None:
    from livekit.agents import _websocket, stt, tts
    from livekit.agents.llm import realtime

    found = [
        f"{kind.__name__}.{name}"
        for kind in (stt, tts, realtime)
        for name in kind.__all__
        if getattr(kind, name, None) is getattr(_websocket, name, object())
        and name not in SHARED_NAMES
    ]
    assert found == []
    assert set(_websocket.__all__) == SHARED_NAMES
