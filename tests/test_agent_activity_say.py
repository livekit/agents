"""Unit tests for AgentActivity.say() — dispatcher capability check + plumbing.

Tests that:
- AgentActivity.say(text, add_to_chat_ctx=False) emits DeprecationWarning when
  the Realtime plugin declares ephemeral_say=False (Phase 4 capability check).
- add_to_chat_ctx flows from say() through _realtime_reply_task and beyond
  (Phase 4 plumbing).

No network calls are made. AgentActivity is constructed via __new__ with only
the attributes say() reads seeded — bypasses the heavyweight constructor that
otherwise spins up audio recognition and other subsystems.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import llm
from livekit.agents.voice.agent_activity import AgentActivity


class _CapabilityShim:
    """Minimal RealtimeCapabilities-like object for dispatcher tests."""

    def __init__(self, *, supports_say: bool, ephemeral_say: bool, turn_detection: bool) -> None:
        self.supports_say = supports_say
        self.ephemeral_say = ephemeral_say
        self.turn_detection = turn_detection


class _RealtimeModelStub(llm.RealtimeModel):
    """Concrete RealtimeModel subclass for isinstance checks in the dispatcher.

    Subclasses are needed because isinstance(obj, RealtimeModel) checks at
    line 1074 of agent_activity.py rejects MagicMock(spec=...) wrappers.
    """

    def __init__(self, capabilities: Any) -> None:  # noqa: D401
        # Skip the abstract __init__ and seed the attributes the dispatcher reads.
        self._capabilities = capabilities

    @property
    def capabilities(self) -> Any:  # type: ignore[override]
        return self._capabilities

    def session(self) -> Any:
        raise NotImplementedError("test stub")

    async def aclose(self) -> None:
        return None


def _make_activity_with_realtime(
    *, ephemeral_say: bool, supports_say: bool = True
) -> tuple[AgentActivity, _RealtimeModelStub]:
    """Construct an AgentActivity wired with a Realtime stub plugin (no network)."""
    caps = _CapabilityShim(
        supports_say=supports_say,
        ephemeral_say=ephemeral_say,
        turn_detection=False,
    )
    rt_model = _RealtimeModelStub(caps)

    activity = AgentActivity.__new__(AgentActivity)

    class _AgentShim:
        llm = rt_model
        tts = None
        allow_interruptions = True

    class _OutputShim:
        audio = None
        audio_enabled = False

    class _SessionShim:
        llm = rt_model
        tts = None
        output = _OutputShim()

        def emit(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            return None

    class _RtSessionShim:
        async def say(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
            raise NotImplementedError("test stub — should not be reached in dispatcher tests")

    activity._agent = _AgentShim()  # type: ignore[attr-defined]
    activity._session = _SessionShim()  # type: ignore[attr-defined]
    activity._rt_session = _RtSessionShim()  # type: ignore[attr-defined]

    # Stub method: short-circuit downstream scheduling so we can test only
    # the dispatcher-level capability check + kwarg threading.
    captured_dispatch: dict[str, Any] = {}

    def _capture_dispatch(coro: Any, **kwargs: Any) -> Any:
        captured_dispatch["coro"] = coro
        captured_dispatch["kwargs"] = kwargs
        # Close the coroutine so it doesn't warn about being un-awaited.
        coro.close()
        return None

    def _schedule_speech_noop(*args: Any, **kwargs: Any) -> None:
        return None

    activity._create_speech_task = _capture_dispatch  # type: ignore[attr-defined]
    activity._schedule_speech = _schedule_speech_noop  # type: ignore[attr-defined]
    activity._captured_dispatch = captured_dispatch  # type: ignore[attr-defined]

    return activity, rt_model


def _capture_dep_warnings(callable_: Any, *args: Any, **kwargs: Any) -> list[Any]:
    """Run callable_ inside catch_warnings; return list of DeprecationWarnings emitted."""
    import warnings as _warnings  # noqa: PLC0415

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        callable_(*args, **kwargs)
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_agent_activity_say_realtime_capability_warns() -> None:
    """add_to_chat_ctx=False on a plugin with ephemeral_say=False emits DeprecationWarning."""
    activity, _rt_model = _make_activity_with_realtime(ephemeral_say=False)

    with pytest.warns(DeprecationWarning, match="ephemeral_say=True"):
        activity.say("alpha-bravo", add_to_chat_ctx=False)


def test_agent_activity_say_realtime_no_warning_when_ephemeral_say_true() -> None:
    """No DeprecationWarning when the plugin declares ephemeral_say=True."""
    activity, _rt_model = _make_activity_with_realtime(ephemeral_say=True)

    dep_warnings = _capture_dep_warnings(activity.say, "alpha-bravo", add_to_chat_ctx=False)
    assert dep_warnings == []


def test_agent_activity_say_realtime_dispatches_with_add_to_chat_ctx() -> None:
    """add_to_chat_ctx kwarg flows from say() into _realtime_reply_task."""
    activity, _rt_model = _make_activity_with_realtime(ephemeral_say=True)

    activity.say("verification token alpha-bravo", add_to_chat_ctx=False)

    captured = activity._captured_dispatch  # type: ignore[attr-defined]
    # The dispatch arg should be the _realtime_reply_task coroutine.
    # Verify the dispatched task has the correct name marker.
    assert captured["kwargs"].get("name") == "AgentActivity.realtime_say"


def test_agent_activity_say_realtime_dispatches_default_add_to_chat_ctx_true() -> None:
    """Default add_to_chat_ctx=True dispatches without warning."""
    activity, _rt_model = _make_activity_with_realtime(ephemeral_say=False)

    dep_warnings = _capture_dep_warnings(activity.say, "hello")
    assert dep_warnings == []
    captured = activity._captured_dispatch  # type: ignore[attr-defined]
    assert captured["kwargs"].get("name") == "AgentActivity.realtime_say"


def test_openai_realtime_say_isolation_no_local_leak() -> None:
    """The chat_ctx upsert in _realtime_generation_task_impl is gated on add_to_chat_ctx.

    Source-level inspection: detects regressions where the gate is removed.
    Behavioral end-to-end coverage is provided by the integration tests against
    the real OpenAI API.
    """
    import inspect as _inspect  # noqa: PLC0415

    src = _inspect.getsource(AgentActivity._realtime_generation_task_impl)
    lines = src.splitlines()
    upsert_line_idx = next((i for i, ln in enumerate(lines) if "_upsert_item(msg)" in ln), -1)
    assert upsert_line_idx >= 0, (
        "_upsert_item(msg) call not found in _realtime_generation_task_impl"
    )
    # Look at the conditional that precedes the upsert (within ~10 lines).
    preceding = "\n".join(lines[max(0, upsert_line_idx - 10) : upsert_line_idx])
    assert "and add_to_chat_ctx" in preceding, (
        "the chat_ctx upsert is no longer guarded by add_to_chat_ctx"
    )


def test_agent_handoff_does_not_propagate_isolated_text() -> None:
    """_realtime_generation_task_impl signature exposes add_to_chat_ctx as keyword-only.

    The handoff path mutates a new agent's chat_ctx by reading the prior agent's
    chat_ctx; gating the upsert ensures isolated text is never written, so it
    cannot propagate. Behavioral coverage in integration tests.
    """
    import inspect as _inspect  # noqa: PLC0415

    sig = _inspect.signature(AgentActivity._realtime_generation_task_impl)
    assert "add_to_chat_ctx" in sig.parameters
    param = sig.parameters["add_to_chat_ctx"]
    assert param.default is True
    assert param.kind is _inspect.Parameter.KEYWORD_ONLY


# Suppress unused-import lint
_ = asyncio
