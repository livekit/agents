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
VALIDATION = {"ConfigDict", "Field", "TypeAdapter", "model_validate", "model_validate_json"}
VALIDATION |= {"validate_python", "validate_json", "validate_strings"}
SOCKET_BASES = (
    "WebSocketStream",
    "WebSocketMultiplexer",
    "RealtimeWebSocketSession",
    "DuplexWebSocketSession",
)


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


def _closed_parsing(tree: ast.Module) -> Iterator[int]:
    """A pydantic model or validation that refuses what a provider may add: ``extra="forbid"``
    (a new field) or ``strict=True`` (a value in another JSON form), as a config, a validate
    argument, a ``Field`` or a class keyword."""
    for node in ast.walk(tree):
        keywords: list[ast.keyword] = []
        if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] in VALIDATION:
            keywords = node.keywords
        elif isinstance(node, ast.ClassDef):
            keywords = node.keywords
        for keyword in keywords:
            value = ast.unparse(keyword.value)
            if (keyword.arg, value) in (("extra", "'forbid'"), ("strict", "True")):
                yield keyword.value.lineno
            if keyword.arg == "extra" and value.endswith("Extra.forbid"):
                yield keyword.value.lineno
        if isinstance(node, ast.Dict):
            for key, item in zip(node.keys, node.values, strict=True):
                if key is None:
                    continue
                pair = (ast.unparse(key), ast.unparse(item))
                if pair in (("'extra'", "'forbid'"), ("'strict'", "True")):
                    yield item.lineno


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
    # a field or a value form the provider added fails a response the plugin reads
    "closed_parsing": _closed_parsing,
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


KNOWN = {
    ("swallowed_in_io_loop", "livekit-plugins-aws"): (
        "Nova Sonic's audio loop logs any send failure (logger.exception) and carries on"
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


# Provider messages: the TypedDicts a plugin reads a provider's frames as (its socket's
# ``Inbound`` and every discriminated union), and the types their fields nest. A provider adds
# message types, fields and values; only a union's tag, which names the message type, is a
# closed ``Literal``.

_Definition = ast.ClassDef | ast.expr


@functools.cache
def _scopes(package: str) -> dict[str, dict[str, _Definition]]:
    """Each module's top-level classes and aliases, by name."""
    scopes: dict[str, dict[str, _Definition]] = {}
    for path, tree in _modules(package):
        names = scopes[path] = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                names[node.name] = node
            elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(target := node.targets[0], ast.Name):
                    names[target.id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.value is not None:
                    names[node.target.id] = node.value
    return scopes


def _lookup(package: str, path: str, expr: ast.expr) -> tuple[str, _Definition] | None:
    """Where a name is defined: its own module first, then any of the package's."""
    name = ast.unparse(expr).split(".")[-1]
    scopes = _scopes(package)
    if name in scopes[path]:
        return path, scopes[path][name]
    return next(((p, names[name]) for p, names in scopes.items() if name in names), None)


def _head(expr: ast.expr) -> str:
    return ast.unparse(expr.value if isinstance(expr, ast.Subscript) else expr).split(".")[-1]


def _args(expr: ast.Subscript) -> list[ast.expr]:
    return list(expr.slice.elts) if isinstance(expr.slice, ast.Tuple) else [expr.slice]


def _is_typeddict(package: str, path: str, node: ast.ClassDef) -> bool:
    for base in node.bases:
        if _head(base) == "TypedDict":
            return True
        found = _lookup(package, path, base)
        if found is not None and isinstance(found[1], ast.ClassDef) and found[1] is not node:
            if _is_typeddict(package, found[0], found[1]):
                return True
    return False


def _fields(package: str, path: str, node: ast.ClassDef) -> dict[str, tuple[str, ast.AnnAssign]]:
    """A TypedDict's fields, its bases' included, each with the module that declares it."""
    fields: dict[str, tuple[str, ast.AnnAssign]] = {}
    for base in node.bases:
        found = _lookup(package, path, base)
        if found is not None and isinstance(found[1], ast.ClassDef) and found[1] is not node:
            fields.update(_fields(package, found[0], found[1]))
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields[stmt.target.id] = (path, stmt)
    return fields


def _discriminator(metadata: list[ast.expr]) -> str | None:
    for node in (n for m in metadata for n in ast.walk(m)):
        if isinstance(node, ast.keyword) and node.arg == "discriminator":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return None


def _members(
    package: str, path: str, expr: ast.expr, tags: list[str], depth: int = 0
) -> Iterator[tuple[str, ast.ClassDef]]:
    """The TypedDicts a type names, through aliases and unions; ``tags`` gets the
    discriminator an ``Annotated`` union names."""
    if depth > 20:
        return
    if isinstance(expr, ast.Name | ast.Attribute):
        if (found := _lookup(package, path, expr)) is None:
            return
        where, definition = found
        if isinstance(definition, ast.ClassDef):
            if _is_typeddict(package, where, definition):
                yield where, definition
        else:
            yield from _members(package, where, definition, tags, depth + 1)
    elif isinstance(expr, ast.BinOp):
        yield from _members(package, path, expr.left, tags, depth + 1)
        yield from _members(package, path, expr.right, tags, depth + 1)
    elif isinstance(expr, ast.Subscript) and _head(expr) in ("Annotated", "Union"):
        args = _args(expr)
        if _head(expr) == "Annotated":
            if (tag := _discriminator(args[1:])) is not None:
                tags.append(tag)
            args = args[:1]
        for arg in args:
            yield from _members(package, path, arg, tags, depth + 1)


def _is_closed(package: str, path: str, annotation: ast.expr, depth: int = 0) -> bool:
    """Whether an annotation is (or holds) a ``Literal``, directly or through an alias."""
    for node in ast.walk(annotation):
        if isinstance(node, ast.Subscript) and _head(node) == "Literal":
            return True
        if isinstance(node, ast.Name) and depth < 5:
            found = _lookup(package, path, node)
            if found is not None and not isinstance(found[1], ast.ClassDef):
                if _is_closed(package, found[0], found[1], depth + 1):
                    return True
    return False


def _message_unions(package: str) -> Iterator[tuple[str, ast.expr]]:
    """Each type a provider's frames are read as: a socket's ``Inbound`` and every union
    ``Annotated`` with its discriminator."""
    for path, tree in _modules(package):
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Subscript) and _head(base) in SOCKET_BASES:
                        yield path, _args(base)[0]
            elif isinstance(node, ast.Subscript) and _head(node) == "Annotated":
                if _discriminator(_args(node)[1:]) is not None:
                    yield path, node


def _literal_message_fields(package: str) -> Iterator[str]:
    """A ``Literal`` on a provider message's field other than its tag, or on any field of a
    type a message's field nests (which has no tag)."""
    seen: set[int] = set()
    for path, union in _message_unions(package):
        tags: list[str] = []
        members = list(_members(package, path, union, tags))
        if not members:
            continue
        if not tags:
            # the one key every member declares as a Literal, as the socket base reads it
            closed = [
                {
                    key
                    for key, (at, f) in _fields(package, where, member).items()
                    if _is_closed(package, at, f.annotation)
                }
                for where, member in members
            ]
            common = set.intersection(*closed)
            tags = list(common) if len(common) == 1 else []
        pending = [(where, member, tags[0] if tags else None) for where, member in members]
        while pending:
            where, member, tag = pending.pop()
            if id(member) in seen:
                continue
            seen.add(id(member))
            for key, (at, field) in _fields(package, where, member).items():
                if key != tag and _is_closed(package, at, field.annotation):
                    yield f"{at}:{field.lineno}"
                for nested in ast.walk(field.annotation):
                    if isinstance(nested, ast.Name | ast.Attribute):
                        pending += [(w, n, None) for w, n in _members(package, at, nested, [])]


def _never_over_provider_values(package: str) -> Iterator[str]:
    """``assert_never`` over anything but a provider message narrowed by its tag (an ``if``
    ladder testing the tag, a key or the frame's type) or an object of the plugin's own (a
    ``match`` of class patterns: an input item, an option)."""
    tags = {
        key
        for path, names in _scopes(package).items()
        for node in names.values()
        if isinstance(node, ast.ClassDef) and _is_typeddict(package, path, node)
        for key, (at, field) in _fields(package, path, node).items()
        if _is_closed(package, at, field.annotation)
    }
    for path, tree in _modules(package):
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for call in ast.walk(tree):
            if not (isinstance(call, ast.Call) and _head(call.func) == "assert_never"):
                continue
            if not (call.args and _exhausts_own_type(call, call.args[0], parents, tags)):
                yield f"{path}:{call.lineno}"


def _exhausts_own_type(
    call: ast.Call, subject: ast.expr, parents: dict[ast.AST, ast.AST], tags: set[str]
) -> bool:
    statement = parents[call]
    container = parents.get(statement)
    if isinstance(container, ast.match_case):
        match = parents[container]
        assert isinstance(match, ast.Match)
        return all(
            isinstance(case.pattern, ast.MatchClass)
            or (
                isinstance(case.pattern, ast.MatchOr)
                and all(isinstance(p, ast.MatchClass) for p in case.pattern.patterns)
            )
            or case is container
            for case in match.cases
        )
    if not (isinstance(container, ast.If) and statement in container.orelse):
        return False
    tests: list[ast.expr] = []
    node: ast.If = container
    while True:
        tests.append(node.test)
        parent = parents.get(node)
        if not (isinstance(parent, ast.If) and parent.orelse == [node]):
            break
        node = parent
    return all(_tests_tag(test, ast.unparse(subject), tags) for test in tests)


def _tests_tag(test: ast.expr, subject: str, tags: set[str]) -> bool:
    if isinstance(test, ast.BoolOp):
        return all(_tests_tag(value, subject, tags) for value in test.values)
    if isinstance(test, ast.Call) and _head(test.func) == "isinstance":
        return ast.unparse(test.args[0]) == subject
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return False
    left, right = test.left, test.comparators[0]
    if isinstance(test.ops[0], ast.In) and isinstance(left, ast.Constant):
        # a key's presence narrows a union whose members differ in their keys
        return ast.unparse(right) == subject
    return (
        isinstance(test.ops[0], ast.Eq)
        and isinstance(left, ast.Subscript)
        and ast.unparse(left.value) == subject
        and isinstance(left.slice, ast.Constant)
        and left.slice.value in tags
        and isinstance(right, ast.Constant)
    )


PACKAGE_RULES: dict[str, Callable[[str], Iterator[str]]] = {
    # a value the provider added since (a status, a reason) fails a message the plugin reads
    "literal_message_fields": _literal_message_fields,
    # an assert_never over a provider's value: a value it added since raises at runtime
    "never_over_provider_values": _never_over_provider_values,
}


# Catches: a provider adding a message type, a field or a value in a field the plugin reads
# failing the plugin: a closed Literal on a field other than the tag, or an assert_never that a
# value the provider added since reaches at runtime.
@pytest.mark.parametrize(
    ("rule", "package"),
    [
        pytest.param(rule, package, id=f"{rule}-{package}")
        for rule in PACKAGE_RULES
        for package in PACKAGES
    ],
)
def test_no_plugin_closes_what_a_provider_may_extend(rule: str, package: str) -> None:
    assert sorted(set(PACKAGE_RULES[rule](package))) == []


def _text_options_applied_directly(tree: ast.Module) -> Iterator[int]:
    """A TTS ``update_options`` that sets the text options itself
    (``super().update_options(sentence_tokenizer=, pacing=)``) instead of through
    ``with self._updating_options(...)``. A chain (a ``FallbackAdapter``) hands them to its
    members through its own ``update_options``."""
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        if any("FallbackAdapter" in ast.unparse(base) for base in cls.bases):
            continue
        for method in cls.body:
            if not (isinstance(method, ast.FunctionDef) and method.name == "update_options"):
                continue
            for node in ast.walk(method):
                if (
                    isinstance(node, ast.Call)
                    and ast.unparse(node.func) == "super().update_options"
                    and {k.arg for k in node.keywords} & {"sentence_tokenizer", "pacing"}
                ):
                    yield node.lineno


# Catches: a plugin applying the text options (tokenizer, pacing) before a check of its own that
# then raises, leaving the update half applied; twelve plugins did. The base's
# `_updating_options` applies them only once the plugin's checks and new options are done.
@pytest.mark.parametrize("package", PACKAGES)
def test_every_tts_update_is_all_or_nothing(package: str) -> None:
    found = [
        f"{path}:{line}"
        for path, tree in _modules(package)
        for line in _text_options_applied_directly(tree)
    ]
    assert found == []
