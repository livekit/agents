"""The Bedrock client must outlive a single turn.

`LLMStream._run` used to enter `session.create_client("bedrock-runtime")` per
turn, so every reply paid for a fresh TCP + TLS handshake and the client was
closed again once the stream ended. The client is now created once per `LLM`
and reused, which is also what lets the plugin take part in the `LLM.prewarm()`
hook (upstream #6484) that the anthropic, openai, google and mistralai plugins
already implement.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from livekit.plugins.aws import llm as aws_llm
from livekit.plugins.aws.llm import LLM, LLMStream

pytestmark = pytest.mark.unit


class _FakeClient:
    """Stands in for an aiobotocore client returned by `create_client`."""

    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.converse_stream_calls = 0

    async def __aenter__(self) -> _FakeClient:
        self.entered = True
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.exited = True

    async def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.converse_stream_calls += 1
        return {
            "ResponseMetadata": {"RequestId": "req-1", "HTTPStatusCode": 200},
            "stream": _empty_stream(),
        }


async def _empty_stream() -> AsyncIterator[dict[str, Any]]:
    for chunk in ():
        yield chunk


class _FakeSession:
    """Minimal stand-in for aiobotocore's AioSession."""

    def __init__(self) -> None:
        self.create_client_calls = 0
        self.services: list[str] = []
        self.client = _FakeClient()

    def set_credentials(self, access_key: str, secret_key: str) -> None:
        pass

    def set_config_variable(self, name: str, value: Any) -> None:
        pass

    def create_client(self, service: str, **kwargs: Any) -> _FakeClient:
        self.create_client_calls += 1
        self.services.append(service)
        return self.client


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> _FakeSession:
    fake = _FakeSession()
    monkeypatch.setattr(aws_llm, "_resolve_session", lambda _session: fake)
    return fake


def _make_llm() -> LLM:
    return LLM(model="amazon.nova-2-lite-v1:0")


def _make_stream(llm: LLM) -> LLMStream:
    """Build a stream without the base class's chat plumbing.

    `_run` only reaches `_event_ch` when the response carries chunks, so an
    empty stream exercises the client handling on its own.
    """
    stream = object.__new__(LLMStream)
    stream._llm = llm
    stream._opts = {"modelId": "amazon.nova-2-lite-v1:0"}
    stream._tool_call_id = None
    stream._fnc_name = None
    stream._fnc_raw_arguments = None
    stream._text = ""
    stream._event_ch = None
    return stream


async def test_client_is_created_once_and_reused(session: _FakeSession) -> None:
    llm = _make_llm()

    first = await llm._get_client()
    second = await llm._get_client()

    assert first is second
    assert session.create_client_calls == 1
    assert session.services == ["bedrock-runtime"]


async def test_consecutive_turns_share_one_client(session: _FakeSession) -> None:
    llm = _make_llm()
    stream = _make_stream(llm)

    await stream._run()
    await stream._run()

    assert session.create_client_calls == 1
    assert session.client.converse_stream_calls == 2
    assert session.client.exited is False


async def test_aclose_closes_the_cached_client(session: _FakeSession) -> None:
    llm = _make_llm()
    await llm._get_client()

    await llm.aclose()

    assert session.client.exited is True
    assert llm._client is None


async def test_prewarm_establishes_the_client(session: _FakeSession) -> None:
    llm = _make_llm()

    await llm._prewarm_impl()

    assert session.create_client_calls == 1
    assert session.client.entered is True
