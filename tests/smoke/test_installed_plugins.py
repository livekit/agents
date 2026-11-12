"""Every installed plugin imports, declares what it imports, and builds its providers offline."""

from __future__ import annotations

import ast
import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import os
import pkgutil
import socket
import sys
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import livekit.plugins
from livekit.agents import llm, stt, tts
from livekit.agents.llm.realtime import RealtimeModel
from livekit.agents.vad import VAD

PLUGINS = Path(__file__).parents[2] / "python" / "livekit-plugins"
PACKAGES = sorted(m.name for m in pkgutil.iter_modules(livekit.plugins.__path__))
PROVIDER_BASES = (stt.STT, tts.TTS, llm.LLM, VAD, RealtimeModel)
Provider = stt.STT | tts.TTS | llm.LLM | VAD | RealtimeModel
CREDENTIAL_ARGS = ("api_key", "api_secret")
BROKEN_EXPORTS = {
    "ai_coustics": "livekit-plugins-ai-coustics 0.3.5 lists LiveKitCloud in __all__ but its "
    "__init__ never imports it (it lives in .auth)",
}


def _extra_args() -> dict[str, dict[str, object]]:
    """What a provider needs besides an API key to be built: endpoints, voices, a graph."""
    from google.auth.credentials import AnonymousCredentials
    from langgraph.graph import END, START, MessagesState, StateGraph

    from livekit.plugins import silero

    graph = StateGraph(MessagesState)
    graph.add_node("reply", lambda state: {"messages": []})
    graph.add_edge(START, "reply")
    graph.add_edge("reply", END)
    azure: dict[str, object] = {"speech_key": "test-key", "speech_region": "eastus"}
    return {
        "azure.STT": azure,
        "azure.TTS": azure,
        "baseten.STT": {"model_endpoint": "wss://baseten.invalid/stt"},
        "baseten.TTS": {"model_endpoint": "https://baseten.invalid/tts"},
        "google.STT": {"credentials": AnonymousCredentials()},  # type: ignore[no-untyped-call]  # google-auth is untyped
        "google.AIPlatformLLM": {
            "endpoint_url": "https://vertex.invalid",
            "project": "project",
            "endpoint_id": "endpoint",
            "access_token": "test-token",
        },
        "langchain.LLMAdapter": {"graph": graph.compile()},
        "clova.STT": {"secret": "test-key", "invoke_url": "https://clova.invalid"},
        "microsoft_ai.STT": {
            "vad": silero.VAD.load(),
            "url": "wss://microsoft.invalid/stt",
            "model": "model",
        },
        "microsoft_ai.TTS": {
            "url": "https://microsoft.invalid",
            "model": "model",
            "voice": "voice",
            "sample_rate": 24000,
        },
        "slng.STT": {"model": "deepgram/nova:3"},
        "slng.TTS": {"model": "cartesia/sonic:3", "voice": "voice"},
    }


def _providers() -> Iterator[tuple[str, Callable[..., Provider]]]:
    for package in PACKAGES:
        try:
            module = importlib.import_module(f"livekit.plugins.{package}")
        except Exception:
            continue  # test_every_plugin_module_imports reports it
        for name in getattr(module, "__all__", ()):
            cls = getattr(module, name, None)  # a missing export is test_every_export_exists's
            if (
                inspect.isclass(cls)
                and issubclass(cls, PROVIDER_BASES)
                and cls.__module__.startswith("livekit.plugins.")
                and not inspect.isabstract(cls)
            ):
                yield f"{package}.{name}", cls


def _declared(distribution: str) -> set[str]:
    """The distributions installing ``distribution`` with all its extras brings in."""
    extras = metadata.metadata(distribution).get_all("Provides-Extra") or []
    seen: set[tuple[str, frozenset[str]]] = set()
    todo = [(canonicalize_name(distribution), frozenset(extras))]
    while todo:
        name, wanted = todo.pop()
        if (name, wanted) in seen:
            continue
        seen.add((name, wanted))
        try:
            requirements = metadata.requires(name) or []
        except metadata.PackageNotFoundError:
            continue
        for line in requirements:
            requirement = Requirement(line)
            marker = requirement.marker
            if marker and not any(marker.evaluate({"extra": e}) for e in {"", *wanted}):
                continue
            todo.append((canonicalize_name(requirement.name), frozenset(requirement.extras)))
    return {name for name, _ in seen}


def _imports(source: Path) -> Iterator[str]:
    """Absolute imports a module runs, leaving out TYPE_CHECKING and guarded optional ones."""

    def walk(node: ast.AST) -> Iterator[str]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.unparse(child.test):
                continue
            if isinstance(child, ast.Try) and any(
                h.type is not None and "ImportError" in ast.unparse(h.type) for h in child.handlers
            ):
                continue
            if isinstance(child, ast.Import):
                yield from (alias.name for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                yield child.module
            yield from walk(child)

    yield from walk(ast.parse(source.read_text()))


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No provider credentials in the environment, and no Python socket may connect."""
    for key in list(os.environ):
        if key.endswith(("_KEY", "_SECRET", "_TOKEN", "_CREDENTIALS", "_ENDPOINT")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("RTZR_CLIENT_ID", "test-id")
    monkeypatch.setenv("RTZR_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("SPITCH_API_KEY", "test-key")

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a provider constructor opened a connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.mark.parametrize("package", PACKAGES)
def test_every_plugin_module_imports(package: str) -> None:
    """Each plugin module imports against the installed SDKs and livekit-agents API.
    https://github.com/livekit/agents/issues/1718"""
    root = importlib.import_module(f"livekit.plugins.{package}")
    failures = []
    for module in pkgutil.walk_packages(root.__path__, prefix=f"{root.__name__}."):
        try:
            importlib.import_module(module.name)
        except Exception as error:
            failures.append(f"{module.name}: {error!r}")
    assert not failures


@pytest.mark.parametrize(
    "package",
    [
        pytest.param(
            p,
            marks=[pytest.mark.xfail(strict=True, reason=BROKEN_EXPORTS[p])]
            if p in BROKEN_EXPORTS
            else [],
        )
        for p in PACKAGES
    ],
)
def test_every_export_exists(package: str) -> None:
    """Every name a plugin lists in ``__all__`` is importable from it."""
    module = importlib.import_module(f"livekit.plugins.{package}")

    missing = [name for name in getattr(module, "__all__", ()) if not hasattr(module, name)]

    assert not missing


@pytest.mark.parametrize(
    "project", sorted(PLUGINS.glob("*/pyproject.toml")), ids=lambda p: p.parent.name
)
def test_every_import_is_a_declared_dependency(project: Path) -> None:
    """A plugin only imports distributions its own requirements (and their extras) install.
    https://github.com/livekit/agents/pull/2090"""
    distribution = tomllib.loads(project.read_text())["project"]["name"]
    declared = _declared(distribution)
    owners = metadata.packages_distributions()
    undeclared = set()
    for source in (project.parent / "livekit").rglob("*.py"):
        for module in _imports(source):
            top = module.partition(".")[0]
            if top in sys.stdlib_module_names or top == "livekit":
                continue
            installers = {canonicalize_name(d) for d in owners.get(top, ())}
            if not installers & declared:
                undeclared.add(f"{source.relative_to(project.parent)}: {module}")
    assert not undeclared


@pytest.mark.usefixtures("offline")
@pytest.mark.parametrize(("name", "cls"), [pytest.param(n, c, id=n) for n, c in _providers()])
async def test_every_provider_builds_offline(name: str, cls: Callable[..., Provider]) -> None:
    """Each exported STT, TTS, LLM, VAD and realtime model builds from an API key alone (or its
    documented endpoint options), without the network, and closes.
    https://github.com/livekit/agents/issues/743"""
    parameters = inspect.signature(cls).parameters
    kwargs: dict[str, object] = {arg: "test-key" for arg in CREDENTIAL_ARGS if arg in parameters}
    kwargs.update(_extra_args().get(name, {}))
    provider = cls(**kwargs)
    assert isinstance(provider, PROVIDER_BASES)
    if hasattr(provider, "aclose"):
        await provider.aclose()


@pytest.mark.parametrize("package", PACKAGES)
def test_every_plugin_is_marked_typed(package: str) -> None:
    """Each plugin package carries its PEP 561 ``py.typed`` marker, so type checkers read it.
    https://github.com/livekit/agents/issues/531"""
    spec = importlib.util.find_spec(f"livekit.plugins.{package}")

    assert spec is not None and spec.submodule_search_locations is not None
    assert any((Path(path) / "py.typed").is_file() for path in spec.submodule_search_locations)
