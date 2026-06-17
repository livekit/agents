import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


DISPATCHER_PATH = (
    Path(__file__).parents[1] / "examples" / "avatar_agents" / "audio_wave" / "dispatcher.py"
)


def _dispatcher_tree() -> ast.Module:
    return ast.parse(DISPATCHER_PATH.read_text(encoding="utf-8"))


def test_dispatcher_default_host_is_localhost() -> None:
    tree = _dispatcher_tree()

    default_host = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id == "DEFAULT_HOST"
        and isinstance(node.value, ast.Constant)
    )

    assert default_host == "127.0.0.1"


def test_run_server_uses_safe_default_host_constant() -> None:
    tree = _dispatcher_tree()
    run_server = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_server"
    )

    assert isinstance(run_server.args.defaults[0], ast.Name)
    assert run_server.args.defaults[0].id == "DEFAULT_HOST"
