"""``EventTypes`` and ``AgentEvent`` name every event a ``VoiceSession`` emits."""

import ast
import inspect
import typing

import pytest

from livekit.agents.voice import _session_callbacks
from livekit.agents.voice.events import AgentEvent, EventTypes

pytestmark = pytest.mark.unit


def _models_by_name() -> dict[str, type]:
    union = typing.get_args(AgentEvent)[0]
    models = {}
    for model in typing.get_args(union):
        fields = getattr(model, "model_fields", None)
        default = fields["type"].default if fields else model.__dataclass_fields__["type"].default
        models[default] = model
    return models


def _emitted_names() -> set[str]:
    """Each ``<session>.emit("name", ..)`` in the session's event routing."""
    tree = ast.parse(inspect.getsource(_session_callbacks))
    names = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "emit"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            continue
        receiver = ast.unparse(node.func.value)
        if receiver in ("self", "self._session", "session"):
            names.add(node.args[0].value)
    return names


def test_every_emitted_session_event_is_named_and_modelled():
    from livekit.agents._generated.session_dispatch import PUBLIC_EVENTS

    names = set(typing.get_args(EventTypes))
    emitted = _emitted_names() | set(PUBLIC_EVENTS.values())
    assert emitted - names == set(), "an emitted event is missing from EventTypes"
    assert names - emitted == set(), "EventTypes names an event the session never emits"
    assert set(_models_by_name()) == names, "AgentEvent and EventTypes disagree"
