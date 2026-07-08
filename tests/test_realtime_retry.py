"""Unit tests for realtime generate_reply retry/recovery building blocks.

Covers the hermetic pieces added for realtime response-level retry
(https://github.com/livekit/agents/issues/6205):

- ``RealtimeError.recoverable`` and the shared ``is_fatal_error`` classifier
- the ``RealtimeSession`` reconnect state machine (``reconnecting`` /
  ``wait_reconnected`` / ``_set_reconnecting`` / ``_set_reconnected``) that the
  retry loop waits on
- the base ``cancel_and_wait`` fallback

The retry loop itself (``AgentActivity._realtime_reply_task``) and the OpenAI
plugin internals (timeout tagging, num_retries reset, cancel_and_wait) are
exercised by the realtime integration suites, not here.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from livekit.agents.llm import (
    FATAL_REALTIME_ERROR_CODES,
    RealtimeError,
    RealtimeModelError,
    is_fatal_error,
)

from .fake_realtime import FakeRealtimeModel

pytestmark = pytest.mark.unit


class _FakeAPIError(Exception):
    """Stand-in for a provider error carrying ``code``/``type``/nested ``body``/``error``."""

    def __init__(
        self,
        *,
        code: str | None = None,
        type: str | None = None,
        body: object | None = None,
        error: object | None = None,
    ) -> None:
        super().__init__("fake api error")
        self.code = code
        self.type = type
        self.body = body
        self.error = error


# --------------------------------------------------------------------------- #
# RealtimeError.recoverable
# --------------------------------------------------------------------------- #


def test_realtime_error_recoverable_defaults_true() -> None:
    assert RealtimeError("generate_reply timed out.").recoverable is True


def test_realtime_error_recoverable_explicit_false() -> None:
    assert RealtimeError("fatal", recoverable=False).recoverable is False


# --------------------------------------------------------------------------- #
# is_fatal_error
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", sorted(FATAL_REALTIME_ERROR_CODES))
def test_is_fatal_error_matches_fatal_code(code: str) -> None:
    assert is_fatal_error(_FakeAPIError(code=code)) is True


@pytest.mark.parametrize("type_", sorted(FATAL_REALTIME_ERROR_CODES))
def test_is_fatal_error_matches_fatal_type(type_: str) -> None:
    assert is_fatal_error(_FakeAPIError(type=type_)) is True


def test_is_fatal_error_walks_body_chain() -> None:
    err = _FakeAPIError(body=_FakeAPIError(code="insufficient_quota"))
    assert is_fatal_error(err) is True


def test_is_fatal_error_walks_error_chain() -> None:
    err = _FakeAPIError(error=_FakeAPIError(type="invalid_api_key"))
    assert is_fatal_error(err) is True


def test_is_fatal_error_walks_realtime_model_error() -> None:
    rme = RealtimeModelError(
        timestamp=time.time(),
        label="test",
        error=_FakeAPIError(code="account_deactivated"),
        recoverable=True,  # optimistically tagged; classifier should still catch it
    )
    assert is_fatal_error(rme) is True


def test_is_fatal_error_false_for_transient_code() -> None:
    assert is_fatal_error(_FakeAPIError(code="server_error")) is False


def test_is_fatal_error_false_for_plain_realtime_error() -> None:
    assert is_fatal_error(RealtimeError("generate_reply timed out.")) is False


def test_is_fatal_error_false_for_none() -> None:
    assert is_fatal_error(None) is False


def test_is_fatal_error_handles_cycles() -> None:
    a = _FakeAPIError(code="server_error")
    b = _FakeAPIError(error=a)
    a.error = b  # cycle: must not loop forever
    assert is_fatal_error(a) is False


# --------------------------------------------------------------------------- #
# RealtimeSession reconnect state machine
# --------------------------------------------------------------------------- #


async def test_reconnect_state_initially_connected() -> None:
    sess = FakeRealtimeModel().session()
    assert sess.reconnecting is False
    # not reconnecting -> returns immediately as healthy
    assert await sess.wait_reconnected() is True


async def test_reconnect_state_machine_transitions_and_events() -> None:
    sess = FakeRealtimeModel().session()

    reconnecting_evs: list[object] = []
    reconnected_evs: list[object] = []
    sess.on("session_reconnecting", reconnecting_evs.append)
    sess.on("session_reconnected", reconnected_evs.append)

    sess._set_reconnecting()
    assert sess.reconnecting is True
    assert len(reconnecting_evs) == 1

    # a waiter parked while reconnecting wakes as soon as we reconnect
    waiter = asyncio.ensure_future(sess.wait_reconnected(timeout=1.0))
    await asyncio.sleep(0)  # let the waiter start awaiting
    assert not waiter.done()

    sess._set_reconnected()
    assert sess.reconnecting is False
    assert len(reconnected_evs) == 1
    assert await waiter is True


async def test_wait_reconnected_times_out_while_reconnecting() -> None:
    sess = FakeRealtimeModel().session()
    sess._set_reconnecting()
    assert await sess.wait_reconnected(timeout=0.05) is False
    assert sess.reconnecting is True


async def test_set_reconnecting_is_idempotent() -> None:
    sess = FakeRealtimeModel().session()
    evs: list[object] = []
    sess.on("session_reconnecting", evs.append)

    sess._set_reconnecting()
    sess._set_reconnecting()

    assert sess.reconnecting is True
    assert len(evs) == 1  # only the first transition emits


# --------------------------------------------------------------------------- #
# base cancel_and_wait fallback
# --------------------------------------------------------------------------- #


async def test_base_cancel_and_wait_falls_back_to_interrupt() -> None:
    sess = FakeRealtimeModel().session()
    assert sess.interrupted is False
    await sess.cancel_and_wait()
    assert sess.interrupted is True
