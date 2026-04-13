from __future__ import annotations

import asyncio

import pytest

from livekit.agents import llm
from livekit.agents.llm import ChatContext
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.plugins.turn_detector import LLMTurnDetector


class _QueuedLLMStream(llm.LLMStream):
    def __init__(self, fake_llm: _QueuedLLM, *, chat_ctx: ChatContext) -> None:
        super().__init__(
            fake_llm,
            chat_ctx=chat_ctx,
            tools=[],
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )
        self._fake_llm = fake_llm

    async def _run(self) -> None:
        behavior = self._fake_llm.next_behavior()
        if behavior.sleep:
            await asyncio.sleep(behavior.sleep)
        if behavior.raise_exc is not None:
            raise behavior.raise_exc
        if behavior.content is not None:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id="1",
                    delta=llm.ChoiceDelta(role="assistant", content=behavior.content),
                )
            )


class _Behavior:
    def __init__(
        self,
        content: str | None = None,
        sleep: float = 0.0,
        raise_exc: Exception | None = None,
    ) -> None:
        self.content = content
        self.sleep = sleep
        self.raise_exc = raise_exc


class _QueuedLLM(llm.LLM):
    """Fake LLM that returns queued behaviors in order."""

    def __init__(self, behaviors: list[_Behavior]) -> None:
        super().__init__()
        self._behaviors = list(behaviors)
        self.calls = 0

    def next_behavior(self) -> _Behavior:
        self.calls += 1
        if not self._behaviors:
            return _Behavior(content="")
        return self._behaviors.pop(0)

    @property
    def model(self) -> str:
        return "fake-llm"

    def chat(
        self,
        *,
        chat_ctx,
        tools=None,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=None,
        tool_choice=None,
        extra_kwargs=None,
    ):
        return _QueuedLLMStream(self, chat_ctx=chat_ctx)


def _ctx_with_user(text: str) -> ChatContext:
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content=text)
    return ctx


def test_provider_and_model_properties():
    det = LLMTurnDetector(llm=_QueuedLLM([]))
    assert det.provider == "llm"
    assert det.model == "fake-llm"


@pytest.mark.asyncio
async def test_supports_language_always_true():
    det = LLMTurnDetector(llm=_QueuedLLM([]))
    assert await det.supports_language(None) is True


@pytest.mark.asyncio
async def test_unlikely_threshold_returns_ctor_value():
    det = LLMTurnDetector(llm=_QueuedLLM([]), unlikely_threshold=0.7)
    assert await det.unlikely_threshold(None) == 0.7


@pytest.mark.asyncio
async def test_predict_returns_high_probability_when_llm_says_one():
    fake = _QueuedLLM([_Behavior(content="1")])
    det = LLMTurnDetector(llm=fake)
    prob = await det.predict_end_of_turn(_ctx_with_user("the meeting is at three"))
    assert prob == pytest.approx(0.95)
    assert fake.calls == 1


@pytest.mark.asyncio
async def test_predict_returns_low_probability_when_llm_says_zero():
    fake = _QueuedLLM([_Behavior(content="0")])
    det = LLMTurnDetector(llm=fake)
    prob = await det.predict_end_of_turn(_ctx_with_user("so i was thinking that"))
    assert prob == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_predict_tolerates_whitespace_around_token():
    fake = _QueuedLLM([_Behavior(content=" 1 \n")])
    det = LLMTurnDetector(llm=fake)
    prob = await det.predict_end_of_turn(_ctx_with_user("done speaking"))
    assert prob == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_predict_returns_neutral_on_garbage():
    fake = _QueuedLLM([_Behavior(content="yes definitely")])
    det = LLMTurnDetector(llm=fake)
    prob = await det.predict_end_of_turn(_ctx_with_user("finished"))
    assert prob == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_predict_returns_neutral_on_empty_response():
    fake = _QueuedLLM([_Behavior(content="")])
    det = LLMTurnDetector(llm=fake)
    prob = await det.predict_end_of_turn(_ctx_with_user("something"))
    assert prob == pytest.approx(0.5)
