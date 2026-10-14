from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import NOT_GIVEN, NotGivenOr, function_tool, utils
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    GenerationCreatedEvent,
    MessageGeneration,
    RealtimeModelFallbackAdapter,
)
from livekit.agents.llm.tool_context import Tool

from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession

pytestmark = pytest.mark.unit


class _AtomicSession(FakeRealtimeSession):
    def __init__(self, model: FakeRealtimeModel) -> None:
        super().__init__(model)
        self.initialized = False
        self.typed_inputs: list[tuple[str, NotGivenOr[str]]] = []
        self.playback_reports: list[dict[str, Any]] = []
        self.playback_reported = asyncio.Event()

    async def initialize(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        tools: NotGivenOr[list[Tool]] = NOT_GIVEN,
    ) -> None:
        self.initialized = True
        await super().initialize(instructions=instructions, chat_ctx=chat_ctx, tools=tools)

    async def update_chat_ctx(self, chat_ctx: ChatContext) -> None:
        assert self.initialized, "initialize must open the provider session first"
        await super().update_chat_ctx(chat_ctx)

    async def set_input_text(
        self, text: str, *, message_id: NotGivenOr[str] = NOT_GIVEN
    ) -> ChatMessage:
        self.typed_inputs.append((text, message_id))
        return await super().set_input_text(text, message_id=message_id)

    def playback_finished(self, **kwargs: Any) -> None:
        self.playback_reports.append(kwargs)
        self.playback_reported.set()


class _AtomicModel(FakeRealtimeModel):
    def session(self, *, turn_detection_disabled: bool = False) -> _AtomicSession:
        session = _AtomicSession(self)
        self.created_sessions.append(session)
        return session


def _generation(response_id: str | None, message_id: str) -> GenerationCreatedEvent:
    messages = utils.aio.Chan[MessageGeneration]()
    text = utils.aio.Chan[str]()
    audio = utils.aio.Chan()
    functions = utils.aio.Chan()
    modalities = asyncio.get_running_loop().create_future()
    modalities.set_result(["audio", "text"])
    messages.send_nowait(
        MessageGeneration(
            message_id=message_id, text_stream=text, audio_stream=audio, modalities=modalities
        )
    )
    for channel in (messages, text, audio, functions):
        channel.close()
    return GenerationCreatedEvent(
        message_stream=messages,
        function_stream=functions,
        user_initiated=True,
        response_id=response_id,
    )


async def test_atomic_initialization_and_typed_turn_survive_restart() -> None:
    model = _AtomicModel()
    adapter = RealtimeModelFallbackAdapter([model])
    session = adapter.session()
    initial = ChatContext.empty()
    initial.add_message(role="user", content="history")
    try:
        await session.initialize(instructions="Be concise.", chat_ctx=initial, tools=[])
        first = model.active_session
        assert first.initialized
        typed = await session.set_input_text("new turn", message_id="typed-turn")
        assert first.typed_inputs == [("new turn", "typed-turn")]
        assert typed.id == "typed-turn"
        await adapter.restart_session()
        second = model.active_session
        assert second is not first
        assert second.initialized
        assert second.updated_instructions == "Be concise."
        assert second.chat_ctx.get_by_id("typed-turn") == typed
        await session.set_input_text("next turn", message_id="next-turn")
        assert second.typed_inputs == [("next turn", "next-turn")]
    finally:
        await session.aclose()
        await adapter.aclose()


@pytest.mark.parametrize("clear", [False, True])
async def test_initialization_preserves_omitted_settings_across_restart(clear: bool) -> None:
    @function_tool
    async def lookup() -> str:
        """Look up a value."""
        return "value"

    model = _AtomicModel()
    adapter = RealtimeModelFallbackAdapter([model])
    session = adapter.session()
    try:
        await session.initialize(instructions="Saved instructions", tools=[lookup])
        if clear:
            await session.initialize(instructions="", tools=[])
        await session.initialize(chat_ctx=ChatContext.empty())
        await adapter.restart_session()
        replacement = model.active_session
        assert replacement.updated_instructions == ("" if clear else "Saved instructions")
        assert list(replacement.tools.function_tools.values()) == ([] if clear else [lookup])
    finally:
        await session.aclose()
        await adapter.aclose()


async def test_cancelled_accepted_reply_interrupts_provider_with_default_playback_hook() -> None:
    model = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter([model])
    session = adapter.session()
    try:
        original = model.active_session
        future = session.generate_reply()
        original._reply_futs[0].set_result(_generation("accepted", "unread"))
        future.cancel()
        await adapter.restart_session()
        await asyncio.sleep(0)
        assert original.interrupted
        assert not model.active_session.interrupted
    finally:
        await session.aclose()
        await adapter.aclose()


@pytest.mark.parametrize("source", ["reply", "say", "event"])
@pytest.mark.parametrize("response_id", ["response-1", None])
async def test_playback_reaches_owning_child_after_swap(
    source: str, response_id: str | None
) -> None:
    model = _AtomicModel()
    adapter = RealtimeModelFallbackAdapter([model])
    session = adapter.session()
    try:
        await session.initialize()
        original = model.active_session
        generation = _generation(response_id, "original-message")
        if source == "event":
            received = []
            session.on("generation_created", received.append)
            original.emit("generation_created", generation)
            exposed = received[0]
        else:
            future = session.say("hello") if source == "say" else session.generate_reply()
            pending = original.say_futs if source == "say" else original._reply_futs
            pending[0].set_result(generation)
            exposed = await future
        messages = [message async for message in exposed.message_stream]
        assert len(messages) == 1
        await adapter.restart_session()
        replacement = model.active_session
        assert exposed.playback_session is original
        exposed.playback_session.playback_finished(
            response_id=response_id,
            message_id="original-message",
            audio_end_ms=125,
            interrupted=True,
            interrupted_by_user=True,
            modalities=["audio", "text"],
        )
        assert len(original.playback_reports) == 1
        assert original.playback_reports[0]["audio_end_ms"] == 125
        assert replacement.playback_reports == []
    finally:
        await session.aclose()
        await adapter.aclose()


async def test_skipped_playback_without_message_uses_original_response_owner() -> None:
    model = _AtomicModel()
    adapter = RealtimeModelFallbackAdapter([model])
    session = adapter.session()
    try:
        await session.initialize()
        original = model.active_session
        future = session.generate_reply()
        original._reply_futs[0].set_result(_generation("skipped-response", "unread-message"))
        generation = await future
        await adapter.restart_session()
        assert generation.playback_session is original
        generation.playback_session.playback_finished(
            response_id="skipped-response",
            message_id=None,
            audio_end_ms=0,
            interrupted=True,
            interrupted_by_user=False,
        )
        assert len(original.playback_reports) == 1
        assert model.active_session.playback_reports == []
    finally:
        await session.aclose()
        await adapter.aclose()


async def test_cancelled_reply_cancels_provider_request() -> None:
    model = _AtomicModel()
    session = RealtimeModelFallbackAdapter([model]).session()
    try:
        future = session.generate_reply()
        future.cancel()
        await asyncio.sleep(0)
        assert model.active_session._reply_futs[0].cancelled()
    finally:
        await session.aclose()


@pytest.mark.parametrize("nested", [False, True])
async def test_cancelled_wrapper_reports_already_accepted_generation_to_original_child(
    nested: bool,
) -> None:
    model = _AtomicModel()
    adapter = RealtimeModelFallbackAdapter([model])
    if nested:
        adapter = RealtimeModelFallbackAdapter([adapter])
    session = adapter.session()
    try:
        await session.initialize()
        original = model.active_session
        future = session.generate_reply()
        original._reply_futs[0].set_result(_generation("accepted-response", "unread-message"))
        # The provider accepted the request, but its completion callback has not run.
        future.cancel()
        await adapter.restart_session()
        await asyncio.wait_for(original.playback_reported.wait(), timeout=1)
        assert len(original.playback_reports) == 1
        assert original.playback_reports[0] == {
            "response_id": "accepted-response",
            "message_id": None,
            "audio_end_ms": 0,
            "interrupted": True,
            "interrupted_by_user": False,
        }
        assert original.interrupted
        assert not model.active_session.interrupted
        assert model.active_session.playback_reports == []
    finally:
        await session.aclose()
        await adapter.aclose()


@pytest.mark.parametrize("consume_messages", [False, True])
async def test_reused_session_ids_keep_distinct_playback_owners(consume_messages: bool) -> None:
    model = _AtomicModel()
    adapter = RealtimeModelFallbackAdapter([model])
    session = adapter.session()
    try:
        await session.initialize()
        original = model.active_session
        generations = []
        for index in range(2):
            if index:
                await adapter.restart_session()
            future = session.generate_reply()
            model.active_session._reply_futs[0].set_result(_generation("reused", "reused-message"))
            generation = await future
            generations.append(generation)
            assert generation.response_id == "reused"
            if consume_messages:
                assert len([message async for message in generation.message_stream]) == 1
        replacement = model.active_session
        assert generations[0].playback_session is original
        assert generations[1].playback_session is replacement
        for index, generation in enumerate(generations):
            generation.playback_session.playback_finished(
                response_id=generation.response_id,
                message_id="reused-message" if consume_messages else None,
                audio_end_ms=index * 125,
                interrupted=True,
                interrupted_by_user=False,
            )
        assert [report["audio_end_ms"] for report in original.playback_reports] == [0]
        assert [report["audio_end_ms"] for report in replacement.playback_reports] == [125]
    finally:
        await session.aclose()
        await adapter.aclose()


async def test_nested_fallback_preserves_the_provider_playback_owner() -> None:
    model = _AtomicModel()
    inner = RealtimeModelFallbackAdapter([model])
    outer = RealtimeModelFallbackAdapter([inner])
    session = outer.session()
    try:
        await session.initialize()
        original = model.active_session
        future = session.generate_reply()
        original._reply_futs[0].set_result(_generation("nested-response", "nested-message"))
        generation = await future
        await outer.restart_session()
        assert generation.playback_session is original
        generation.playback_session.playback_finished(
            response_id=generation.response_id,
            message_id=None,
            audio_end_ms=0,
            interrupted=True,
            interrupted_by_user=False,
        )
        assert len(original.playback_reports) == 1
        assert model.active_session.playback_reports == []
    finally:
        await session.aclose()
        await outer.aclose()


async def test_direct_playback_hook_forwards_to_active_child() -> None:
    model = _AtomicModel()
    session = RealtimeModelFallbackAdapter([model]).session()
    try:
        session.playback_finished(
            response_id="direct-response",
            message_id=None,
            audio_end_ms=0,
            interrupted=True,
            interrupted_by_user=False,
        )
        assert len(model.active_session.playback_reports) == 1
    finally:
        await session.aclose()


async def test_zero_cooldown_attempts_each_failing_replacement_once() -> None:
    class RepeatedStartup(BaseException):
        """Stop the synchronous retry loop before it can hang the test runner."""

    class BoundedFailingModel(FakeRealtimeModel):
        def session(self, *, turn_detection_disabled: bool = False) -> FakeRealtimeSession:
            if len(self.created_sessions) >= 3:
                raise RepeatedStartup
            return super().session(turn_detection_disabled=turn_detection_disabled)

    primary = BoundedFailingModel()
    backup = BoundedFailingModel()
    adapter = RealtimeModelFallbackAdapter([primary, backup], cooldown=0)
    session = adapter.session()
    errors = []
    session.on("error", errors.append)
    primary.bring_up_error = RuntimeError("primary startup failed")
    backup.bring_up_error = RuntimeError("backup startup failed")
    try:
        try:
            await adapter.restart_session()
        except RepeatedStartup:
            pytest.fail("swap retried a failed candidate instead of exhausting the model list")
        assert len(primary.created_sessions) == 2  # initial child plus one restart attempt
        assert len(backup.created_sessions) == 1
        assert len(errors) == 1
        assert errors[0].recoverable is False
    finally:
        await session.aclose()
        await adapter.aclose()


@pytest.mark.parametrize("exhausted", [False, True])
@pytest.mark.parametrize("cooldown", [0, 10])
async def test_initialization_tries_each_candidate_and_preserves_history(exhausted, cooldown):
    failures = [FakeRealtimeModel(), FakeRealtimeModel()]
    for index, model in enumerate(failures):
        model.bring_up_error = RuntimeError(f"startup failure {index}")
    backup = FakeRealtimeModel()
    adapter = RealtimeModelFallbackAdapter(
        failures if exhausted else [*failures, backup], cooldown=cooldown
    )
    session = adapter.session()
    errors = []
    session.on("error", errors.append)
    history = ChatContext.empty()
    history.add_message(role="user", content="Saved history")
    try:
        if exhausted:
            with pytest.raises(RuntimeError, match="startup failure 1"):
                await session.initialize(
                    instructions="Saved instructions", chat_ctx=history, tools=[]
                )
        else:
            await session.initialize(instructions="Saved instructions", chat_ctx=history, tools=[])
            assert backup.active_session.chat_ctx.items == history.items
            assert backup.active_session.updated_instructions == "Saved instructions"
            assert not backup.active_session.closed
        assert all(len(model.created_sessions) == 1 for model in failures)
        assert all(model.active_session.closed for model in failures)
        assert len(errors) == 2
        assert errors[0].recoverable
        assert errors[1].recoverable is (not exhausted)
    finally:
        await session.aclose()
        await adapter.aclose()


async def test_closing_during_initialization_cancels_and_closes_child():
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowSession(FakeRealtimeSession):
        async def initialize(self, **kwargs):
            entered.set()
            await release.wait()
            await super().initialize(**kwargs)

    class SlowModel(FakeRealtimeModel):
        def session(self, *, turn_detection_disabled=False):
            child = SlowSession(self)
            self.created_sessions.append(child)
            return child

    model = SlowModel()
    adapter = RealtimeModelFallbackAdapter([model])
    session = adapter.session()
    initializing = asyncio.create_task(session.initialize())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(session.aclose(), 1)
        with pytest.raises(asyncio.CancelledError):
            await initializing
        assert model.active_session.closed
    finally:
        release.set()
        await utils.aio.cancel_and_wait(initializing)
        await session.aclose()
        await adapter.aclose()
