import asyncio
from types import SimpleNamespace

import pytest

from livekit.plugins.aws.experimental.realtime import realtime_model
from livekit.plugins.aws.experimental.realtime.realtime_model import (
    _is_recoverable_validation_error,
)

pytestmark = pytest.mark.unit


def test_system_instability_validation_error_is_recoverable() -> None:
    exc = SimpleNamespace(message="System instability detected. Please retry your request.")

    assert _is_recoverable_validation_error(exc) is True


def test_unrecognized_validation_error_is_not_recoverable() -> None:
    exc = SimpleNamespace(message="The provided request is invalid.")

    assert _is_recoverable_validation_error(exc) is False


async def test_old_response_task_does_not_deactivate_restarted_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetryableStreamError(Exception):
        message = "stream dropped"

    class OtherStreamError(Exception):
        pass

    for exception_name in (
        "ValidationException",
        "ThrottlingException",
        "ModelNotReadyException",
        "ModelErrorException",
        "InvalidEventBytes",
        "ModelTimeoutException",
    ):
        monkeypatch.setattr(realtime_model, exception_name, OtherStreamError)
    monkeypatch.setattr(realtime_model, "ModelStreamErrorException", RetryableStreamError)

    class OutputStream:
        async def receive(self):
            raise RetryableStreamError("stream dropped")

    class StreamResponse:
        async def await_output(self):
            return None, OutputStream()

    session = object.__new__(realtime_model.RealtimeSession)
    session._is_sess_active = asyncio.Event()
    session._is_sess_active.set()
    session._stream_ready = asyncio.Event()
    session._stream_response = StreamResponse()
    session._realtime_model = SimpleNamespace(_label="test")
    session._events = {}
    current_task = asyncio.current_task()
    replacement_task = object()
    session._response_task = current_task

    async def restart_session(_exception):
        session._response_task = replacement_task

    session._restart_session = restart_session

    await session._process_responses()

    assert session._is_sess_active.is_set()


async def test_current_response_task_deactivates_session() -> None:
    class OutputStream:
        async def receive(self):
            return None

    class StreamResponse:
        async def await_output(self):
            return None, OutputStream()

    session = object.__new__(realtime_model.RealtimeSession)
    session._is_sess_active = asyncio.Event()
    session._is_sess_active.set()
    session._stream_ready = asyncio.Event()
    session._stream_response = StreamResponse()
    session._response_task = asyncio.current_task()

    await session._process_responses()

    assert not session._is_sess_active.is_set()
