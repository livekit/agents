"""Credential-free Bedrock client and response lifecycle regression tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiobotocore.session import AioSession

from livekit.agents import APIConnectOptions
from livekit.agents.llm import ChatContext
from livekit.plugins.aws import LLM

pytestmark = pytest.mark.unit


class _Response:
    def __init__(self, *, blocked: bool = False) -> None:
        self.closed = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        self.started.set()
        await self.release.wait()
        yield {"contentBlockDelta": {"delta": {"text": "hello"}}}

    def close(self) -> None:
        self.closed = True


class _Bedrock:
    def __init__(self) -> None:
        self.attempts = 0
        self.opened = 0
        self.closed = 0
        self.fail_initialization = False
        self.block_initialization = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.block_response = False
        self.response_created = asyncio.Event()
        self.request_failures = 0
        self.block_request = False
        self.request_started = asyncio.Event()
        self.request_release = asyncio.Event()
        self.request_cancelled = False
        self.responses: list[_Response] = []

    @asynccontextmanager
    async def create_client(self, *args: Any, **kwargs: Any) -> AsyncIterator[_Bedrock]:
        assert args == ("bedrock-runtime",)
        assert kwargs["config"].user_agent_extra == "x-client-framework:livekit-plugins-aws"
        self.attempts += 1
        self.entered.set()
        if self.block_initialization:
            await self.release.wait()
        if self.fail_initialization:
            self.fail_initialization = False
            raise OSError("initialization failed")
        self.opened += 1
        try:
            yield self
        finally:
            self.closed += 1

    async def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.request_started.set()
        if self.block_request:
            try:
                await self.request_release.wait()
            except asyncio.CancelledError:
                self.request_cancelled = True
                raise
        if self.request_failures:
            self.request_failures -= 1
            raise OSError("temporary request failure")
        response = _Response(blocked=self.block_response)
        self.responses.append(response)
        self.response_created.set()
        return {
            "ResponseMetadata": {"RequestId": "test-request", "HTTPStatusCode": 200},
            "stream": response,
        }


@pytest.fixture
async def bedrock(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[LLM, _Bedrock]]:
    backend = _Bedrock()
    session = AioSession()
    monkeypatch.setattr(session, "create_client", backend.create_client)
    model = LLM(session=session)
    assert model._session is session
    try:
        yield model, backend
    finally:
        backend.release.set()
        backend.request_release.set()
        for response in backend.responses:
            response.release.set()
        await model.aclose()


async def _turn(model: LLM, *, retries: int = 0) -> str:
    context = ChatContext()
    context.add_message(role="user", content="hello")
    response = await model.chat(
        chat_ctx=context,
        conn_options=APIConnectOptions(max_retry=retries, retry_interval=0),
    ).collect()
    return response.text


async def test_client_is_lazy_and_unused_close_is_idempotent(bedrock: tuple[LLM, _Bedrock]) -> None:
    model, backend = bedrock
    assert backend.attempts == 0
    await model.aclose()
    await model.aclose()
    assert backend.attempts == backend.closed == 0


async def test_completed_turns_reuse_client_until_model_close(
    bedrock: tuple[LLM, _Bedrock],
) -> None:
    model, backend = bedrock
    assert await _turn(model) == "hello"
    assert await _turn(model) == "hello"
    assert backend.opened == 1
    assert backend.closed == 0
    assert all(response.closed for response in backend.responses)
    await model.aclose()
    await model.aclose()
    assert backend.closed == 1


async def test_concurrent_first_turns_share_initialization(bedrock: tuple[LLM, _Bedrock]) -> None:
    model, backend = bedrock
    backend.block_initialization = True
    first = asyncio.create_task(_turn(model))
    await asyncio.wait_for(backend.entered.wait(), 2)
    second = asyncio.create_task(_turn(model))
    await asyncio.sleep(0)
    backend.release.set()
    assert await asyncio.gather(first, second) == ["hello", "hello"]
    assert backend.attempts == backend.opened == 1
    assert backend.closed == 0


async def test_failed_initialization_can_retry(bedrock: tuple[LLM, _Bedrock]) -> None:
    model, backend = bedrock
    backend.fail_initialization = True
    assert await _turn(model, retries=1) == "hello"
    assert backend.attempts == 2
    assert backend.opened == 1
    assert backend.closed == 0


async def test_cancelled_initialization_does_not_poison_next_turn(
    bedrock: tuple[LLM, _Bedrock],
) -> None:
    model, backend = bedrock
    backend.block_initialization = True
    task = asyncio.create_task(_turn(model))
    await asyncio.wait_for(backend.entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    backend.release.set()
    assert await _turn(model) == "hello"
    assert backend.attempts == 2
    assert backend.opened == 1
    assert backend.closed == 0


async def test_cancelled_response_closes_body_but_keeps_client(
    bedrock: tuple[LLM, _Bedrock],
) -> None:
    model, backend = bedrock
    backend.block_response = True
    task = asyncio.create_task(_turn(model))
    await asyncio.wait_for(backend.entered.wait(), 2)
    await asyncio.wait_for(backend.response_created.wait(), 2)
    response = backend.responses[0]
    await asyncio.wait_for(response.started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert response.closed
    assert backend.closed == 0
    backend.block_response = False
    assert await _turn(model) == "hello"
    assert backend.opened == 1


async def test_closed_model_does_not_open_another_client(bedrock: tuple[LLM, _Bedrock]) -> None:
    model, backend = bedrock
    assert await _turn(model) == "hello"
    await model.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await _turn(model)
    assert backend.opened == backend.closed == 1


async def test_request_retry_reuses_initialized_client(bedrock: tuple[LLM, _Bedrock]) -> None:
    model, backend = bedrock
    backend.request_failures = 1
    assert await _turn(model, retries=1) == "hello"
    assert backend.attempts == backend.opened == 1
    assert backend.closed == 0
    assert backend.responses[0].closed


async def test_cancelling_one_turn_does_not_close_another_response(
    bedrock: tuple[LLM, _Bedrock],
) -> None:
    model, backend = bedrock
    backend.block_response = True
    first = asyncio.create_task(_turn(model))
    await asyncio.wait_for(backend.response_created.wait(), 2)
    first_response = backend.responses[0]
    await asyncio.wait_for(first_response.started.wait(), 2)
    backend.response_created.clear()
    second = asyncio.create_task(_turn(model))
    await asyncio.wait_for(backend.response_created.wait(), 2)
    second_response = backend.responses[1]
    await asyncio.wait_for(second_response.started.wait(), 2)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert first_response.closed
    assert not second_response.closed
    assert backend.closed == 0
    second_response.release.set()
    assert await second == "hello"
    assert second_response.closed
    assert backend.opened == 1


async def test_close_cancels_pending_initialization(
    bedrock: tuple[LLM, _Bedrock],
) -> None:
    model, backend = bedrock
    backend.block_initialization = True
    turn = asyncio.create_task(_turn(model))
    await asyncio.wait_for(backend.entered.wait(), 2)
    await asyncio.wait_for(model.aclose(), 2)
    assert await turn == ""
    assert backend.attempts == 1
    assert backend.opened == backend.closed == 0
    with pytest.raises(RuntimeError, match="closed"):
        await _turn(model)


async def test_close_cancels_request_before_closing_shared_client(
    bedrock: tuple[LLM, _Bedrock],
) -> None:
    model, backend = bedrock
    backend.block_request = True
    turn = asyncio.create_task(_turn(model))
    await asyncio.wait_for(backend.request_started.wait(), 2)
    await asyncio.wait_for(model.aclose(), 2)
    assert backend.request_cancelled
    assert backend.opened == backend.closed == 1
    assert await turn == ""


async def test_close_cancels_response_and_closes_its_body(
    bedrock: tuple[LLM, _Bedrock],
) -> None:
    model, backend = bedrock
    backend.block_response = True
    turn = asyncio.create_task(_turn(model))
    await asyncio.wait_for(backend.response_created.wait(), 2)
    response = backend.responses[0]
    await asyncio.wait_for(response.started.wait(), 2)
    await asyncio.wait_for(model.aclose(), 2)
    assert response.closed
    assert backend.opened == backend.closed == 1
    assert await turn == ""


async def test_initialized_model_rejects_cross_loop_use(bedrock: tuple[LLM, _Bedrock]) -> None:
    model, backend = bedrock
    assert await _turn(model) == "hello"
    with pytest.raises(RuntimeError, match="same event loop"):
        await asyncio.to_thread(lambda: asyncio.run(_turn(model)))
    assert backend.opened == 1
    assert backend.closed == 0
    assert await _turn(model) == "hello"


async def test_pending_initialization_rejects_cross_loop_use(
    bedrock: tuple[LLM, _Bedrock],
) -> None:
    model, backend = bedrock
    backend.block_initialization = True
    turn = asyncio.create_task(_turn(model))
    await asyncio.wait_for(backend.entered.wait(), 2)
    with pytest.raises(RuntimeError, match="same event loop"):
        await asyncio.to_thread(lambda: asyncio.run(_turn(model)))
    backend.release.set()
    assert await turn == "hello"
    assert backend.attempts == 1


@pytest.mark.parametrize("initializing", [False, True])
async def test_cross_loop_close_leaves_model_usable(
    bedrock: tuple[LLM, _Bedrock], initializing: bool
) -> None:
    model, backend = bedrock
    turn = None
    if initializing:
        backend.block_initialization = True
        turn = asyncio.create_task(_turn(model))
        await asyncio.wait_for(backend.entered.wait(), 2)
    else:
        assert await _turn(model) == "hello"
    with pytest.raises(RuntimeError, match="same event loop"):
        await asyncio.to_thread(lambda: asyncio.run(model.aclose()))
    if turn is not None:
        backend.release.set()
        assert await turn == "hello"
    assert await _turn(model) == "hello"
    assert backend.opened == 1
    assert backend.closed == 0
