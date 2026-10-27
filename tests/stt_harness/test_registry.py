"""A new streaming provider must supply a local protocol contract."""

import ast
from pathlib import Path

import pytest

from .discovery import modules

pytestmark = pytest.mark.unit


def test_every_streaming_implementation_has_a_contract_and_uses_the_coordinator():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "livekit-plugins").glob("*/livekit/plugins/**/*.py"))
    paths.append(root / "livekit-agents/livekit/agents/inference/stt.py")
    streaming = set()
    missing_coordinator = []
    for path in paths:
        if "stt" not in path.stem and "recognize_stream" not in path.stem:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [
                base.attr if isinstance(base, ast.Attribute) else base.id
                for base in node.bases
                if isinstance(base, (ast.Attribute, ast.Name))
            ]
            if not any(base.endswith(("RecognizeStream", "SpeechStream")) for base in bases):
                continue
            package = (
                path.parts[path.parts.index("livekit-plugins") + 1]
                if "livekit-plugins" in path.parts
                else "inference"
            )
            streaming.add(package)
            run = next(
                (
                    method
                    for method in node.body
                    if isinstance(method, ast.AsyncFunctionDef) and method.name == "_run"
                ),
                None,
            )
            if run is None or any(
                isinstance(d, ast.Name) and d.id == "abstractmethod" for d in run.decorator_list
            ):
                continue
            if not any(
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "_run_connection_tasks"
                for call in ast.walk(node)
            ):
                missing_coordinator.append(f"{path.relative_to(root)}:{node.name}")
    registered = set()
    for path, module in modules():
        assert module.PROVIDERS, f"empty STT protocol file: {path}"
        assert all(case.sources for case in module.PROVIDERS), f"missing protocol sources: {path}"
        registered.add(path.parents[1].name if "livekit-plugins" in path.parts else "inference")
    assert registered == streaming
    assert not missing_coordinator, "STT streams bypass the coordinator: " + ", ".join(
        missing_coordinator
    )
