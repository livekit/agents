import ast
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from livekit.agents.core import endpointing as core

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parent / "core_fixtures" / "endpointing.json"
_CORE_DIR = Path(core.__file__).parent
_CASES: list[dict[str, Any]] = json.loads(_FIXTURES.read_text())

# fixture `type` tags are the cross-language contract
_EVENTS: dict[str, type[Any]] = {
    "user_speech_started": core.UserSpeechStarted,
    "user_speech_ended": core.UserSpeechEnded,
    "agent_speech_started": core.AgentSpeechStarted,
    "agent_speech_ended": core.AgentSpeechEnded,
    "options_updated": core.OptionsUpdated,
}
_OUTPUT_TAGS: dict[type[Any], str] = {
    core.MinDelayUpdated: "min_delay_updated",
    core.UtteranceEndAdjusted: "utterance_end_adjusted",
    core.NonInterruptionOverridden: "non_interruption_overridden",
}


def _matches(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_matches(actual[k], expected[k]) for k in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_matches(a, e) for a, e in zip(actual, expected, strict=True))
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and abs(actual - expected) <= 1e-9
        )
    return actual == expected


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_fixture(case: dict[str, Any]) -> None:
    ep: core.FixedEndpointing = (
        core.DynamicEndpointing(**case["config"])
        if case["mode"] == "dynamic"
        else core.FixedEndpointing(**case["config"])
    )
    for i, step in enumerate(case["steps"]):
        fields = dict(step["event"])
        event = _EVENTS[fields.pop("type")](**fields)
        outputs = [
            {"type": _OUTPUT_TAGS[type(o)], **dataclasses.asdict(o)} for o in ep.handle(event)
        ]
        state = {
            "min_delay": ep.min_delay,
            "max_delay": ep.max_delay,
            "overlapping": ep.overlapping,
        }
        assert _matches(outputs, step["outputs"]), (i, outputs, step["outputs"])
        assert _matches(state, step["state"]), (i, state, step["state"])


def test_core_imports_only_stdlib() -> None:
    for path in _CORE_DIR.rglob("*.py"):
        depth = len(path.relative_to(_CORE_DIR).parts)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            elif isinstance(node, ast.ImportFrom):
                assert node.level <= depth, f"{path}: relative import escapes core"
                continue
            else:
                continue
            for module in modules:
                assert module.split(".")[0] in sys.stdlib_module_names or module.startswith(
                    "livekit.agents.core"
                ), f"{path}: imports {module}"
