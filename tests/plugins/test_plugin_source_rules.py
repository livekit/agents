"""Source patterns that hid provider failures in plugins, each a bug class fixed in 9e79bc520b.

Every rule scans every plugin package; none has an exemption marker. A plugin raises a provider
failure to its owner, which logs it once and decides the retry; a finding is fixed in the plugin,
never allowed.
"""

from __future__ import annotations

import ast
import functools
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

PLUGINS = Path(__file__).parents[2] / "python" / "livekit-plugins"
PACKAGES = sorted(p.name for p in PLUGINS.iterdir() if (p / "livekit").is_dir())
LOGGING = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
SOCKET_IO = {"receive", "recv", "receive_str", "receive_json", "receive_bytes"}
SOCKET_IO |= {"send", "send_str", "send_bytes", "send_json"}
COPIED_HELPERS = {"PeriodicCollector", "_resample_audio"}


@functools.cache
def _modules(package: str) -> list[tuple[str, ast.Module]]:
    return [
        (str(path.relative_to(PLUGINS)), ast.parse(path.read_text(), filename=str(path)))
        for path in sorted((PLUGINS / package / "livekit").rglob("*.py"))
    ]


def _is_logging(stmt: ast.stmt, levels: set[str] = LOGGING) -> bool:
    """``logger.<level>(...)`` as a statement."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    func = stmt.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in levels
        and ("logger" in (receiver := ast.unparse(func.value)) or receiver == "logging")
    )


def _log_exceptions(tree: ast.Module) -> Iterator[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for deco in node.decorator_list:
                target = deco.func if isinstance(deco, ast.Call) else deco
                if ast.unparse(target).endswith("log_exceptions"):
                    yield deco.lineno


def _log_then_raise(tree: ast.Module) -> Iterator[int]:
    for node in ast.walk(tree):
        blocks = [getattr(node, name, None) for name in ("body", "orelse", "finalbody")]
        for block in blocks:
            if not (isinstance(block, list) and block and isinstance(block[0], ast.stmt)):
                continue
            for before, after in zip(block, block[1:], strict=False):
                if isinstance(after, ast.Raise) and _is_logging(before, {"error", "exception"}):
                    yield before.lineno


def _swallowed_in_io_loop(tree: ast.Module) -> Iterator[int]:
    """``except Exception`` that only logs, inside a loop that receives from or sends to a
    socket."""
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.While | ast.For | ast.AsyncFor):
            continue
        calls = {
            n.func.attr
            for n in ast.walk(loop)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        over_socket = isinstance(loop, ast.AsyncFor) and any(
            word in ast.unparse(loop.iter) for word in ("ws", "socket")
        )
        if not (calls & SOCKET_IO or over_socket):
            continue
        for node in ast.walk(loop):
            if (
                isinstance(node, ast.ExceptHandler)
                and node.type is not None
                and ast.unparse(node.type) in ("Exception", "BaseException")
                and any(_is_logging(s) for s in node.body)
                and all(_is_logging(s) or isinstance(s, ast.Pass | ast.Continue) for s in node.body)
            ):
                yield node.lineno


def _hand_built_close_error(tree: ast.Module) -> Iterator[int]:
    """An error built from a close code instead of ``ws.close_error``: ``status_code=`` from a
    close code, or a raise under a test of ``close_code`` that is not ``close_error``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "status_code":
            if "close_code" in ast.unparse(node.value):
                yield node.value.lineno
        if isinstance(node, ast.If) and "close_code" in ast.unparse(node.test):
            for stmt in node.body:
                for raise_ in ast.walk(stmt):
                    if isinstance(raise_, ast.Raise) and raise_.exc is not None:
                        if ".close_error(" not in ast.unparse(raise_.exc):
                            yield raise_.lineno


def _untyped_error(tree: ast.Module) -> Iterator[int]:
    """A retry decided in the plugin: a ``retryable=`` argument or assignment, or an
    ``APIError`` that names no ``kind``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "retryable":
            yield node.value.lineno
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr == "retryable":
                    yield target.lineno
        if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == "APIError":
            if not any(k.arg in ("kind", None) for k in node.keywords):
                yield node.lineno


def _copied_helpers(tree: ast.Module) -> Iterator[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            if node.name in COPIED_HELPERS:
                yield node.lineno


RULES: dict[str, Callable[[ast.Module], Iterator[int]]] = {
    # a decorated task logs its failure and returns: its owner never learns the provider failed
    "log_exceptions": _log_exceptions,
    # logged here and again by the error's owner, the provider's message in plain text
    "log_then_raise": _log_then_raise,
    # a provider error frame or a broken send is logged and the loop carries on: the turn hangs
    "swallowed_in_io_loop": _swallowed_in_io_loop,
    # a close code is no HTTP status: 1008/4401 read as retryable, 1011 as a client error
    "hand_built_close_error": _hand_built_close_error,
    # a plugin decides the retry instead of naming what failed: auth retried, refusals retried
    "untyped_error": _untyped_error,
    # a private copy of a framework helper (usage collector, resampler) drifts from the original
    "copied_helpers": _copied_helpers,
}


def _known(rule: str, package: str, reason: str) -> object:
    return pytest.param(
        rule,
        package,
        marks=pytest.mark.xfail(
            strict=True,
            reason=f"https://github.com/livekit/agents/commit/9e79bc520b80553e3ca665884a2fac2beeb63042 reproduces: {reason}",
        ),
        id=f"{rule}-{package}",
    )


_REALTIME_LOGGED = "the realtime session's main task or push_audio is @utils.log_exceptions"
KNOWN = {
    ("log_exceptions", "livekit-plugins-nvidia"): _REALTIME_LOGGED,
    ("log_exceptions", "livekit-plugins-openai"): _REALTIME_LOGGED + " (gpt_live_model)",
    ("log_exceptions", "livekit-plugins-phonic"): _REALTIME_LOGGED,
    ("log_exceptions", "livekit-plugins-ultravox"): _REALTIME_LOGGED,
    ("swallowed_in_io_loop", "livekit-plugins-aws"): (
        "Nova Sonic's audio loop logs any send failure (logger.exception) and carries on"
    ),
    ("swallowed_in_io_loop", "livekit-plugins-nabrah"): (
        "the receive loop logs a message it cannot parse and carries on"
    ),
    ("swallowed_in_io_loop", "livekit-plugins-ultravox"): (
        "the receive loop logs an event it fails to handle and carries on"
    ),
}


def test_the_scan_reads_every_plugin() -> None:
    """Catches the scan silently reading nothing after a layout change."""
    assert len(PACKAGES) > 60
    assert sum(len(_modules(p)) for p in PACKAGES) > 400


# Catches: a bug class of 9e79bc520b coming back in a plugin: a failure logged and dropped,
# logged twice, retried when final, a retry decided in the plugin, or a private copy of a
# framework helper.
@pytest.mark.parametrize(
    ("rule", "package"),
    [
        _known(rule, package, KNOWN[rule, package])
        if (rule, package) in KNOWN
        else pytest.param(rule, package, id=f"{rule}-{package}")
        for rule in RULES
        for package in PACKAGES
    ],
)
def test_no_plugin_hides_or_misreports_provider_failures(rule: str, package: str) -> None:
    found = [f"{path}:{line}" for path, tree in _modules(package) for line in RULES[rule](tree)]
    assert found == []
