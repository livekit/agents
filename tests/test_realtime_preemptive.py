"""
Tests for preemptive (speculative) generation with RealtimeModel pipelines.

A text-mode RealtimeModel (no server-side turn detection) driven by a client EOT signal starts
generating on the eager transcript, before the user turn is confirmed:

* the reply task issues ``rt_session.generate_reply`` *before* the authorization gate (the head
  start), while the local history commit and playout stay gated on authorization
* if the confirmed turn still matches, the in-flight generation is adopted (scheduled)
* otherwise it is rolled back: the response is cancelled, the server-side chat context restored,
  and the fallback generation waits for the rollback so it can't collide with the cancelled
  response (single-active-response APIs)
"""

from __future__ import annotations

import asyncio
from types import MethodType, SimpleNamespace
from typing import Any, cast

import pytest

from livekit.agents import Agent, AgentSession, llm, utils
from livekit.agents.llm import ChatMessage, FunctionCall, GenerationCreatedEvent, MessageGeneration
from livekit.agents.voice import ModelSettings
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import _PreemptiveGenerationInfo
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, fake_capabilities

pytestmark = pytest.mark.unit


def _text_realtime_model() -> FakeRealtimeModel:
    return FakeRealtimeModel(
        capabilities=fake_capabilities(turn_detection=False, audio_output=False)
    )


class _FakeActivity(SimpleNamespace):
    """The attribute surface the preemptive realtime paths touch, around a FakeRealtimeSession."""

    def __init__(self, rt_session: FakeRealtimeSession) -> None:
        authorization_allowed = asyncio.Event()
        authorization_allowed.set()
        user_silence = asyncio.Event()
        user_silence.set()
        generation_calls: list[dict[str, Any]] = []
        conversation_items: list[llm.ChatMessage] = []

        async def _realtime_generation_task(**kwargs: Any) -> None:
            generation_calls.append(kwargs)

        super().__init__(
            _rt_session=rt_session,
            _authorization_allowed=authorization_allowed,
            _user_silence_event=user_silence,
            tools=[],
            _on_enter_ignored_tools=lambda tool_ctx: [],
            _tool_choice=None,
            _agent=SimpleNamespace(_chat_ctx=llm.ChatContext.empty()),
            _session=SimpleNamespace(
                _conversation_item_added=conversation_items.append,
                _update_agent_state=lambda state: None,
            ),
            _realtime_generation_task=_realtime_generation_task,
            generation_calls=generation_calls,
            conversation_items=conversation_items,
            _realtime_preemptive_cleanup=None,
            _realtime_preemptive_remote_items=[],
            _realtime_preemptive_user_item_ids=set(),
        )
        # bind the real helpers so the reply task exercises the actual gating logic
        for name in (
            "_wait_for_speech_authorization",
            "_wait_for_realtime_session_idle",
            "_wait_for_active_response_cleared",
            "_rollback_preemptive_realtime_generation",
            "_on_remote_item_added",
        ):
            setattr(self, name, MethodType(getattr(AgentActivity, name), self))
        # route the fake session's server acks through the real mirroring handler
        rt_session.on("remote_item_added", self._on_remote_item_added)
        self._create_speech_task = lambda coro, speech_handle=None, name=None: asyncio.create_task(
            coro, name=name
        )


def _run_preemptive_reply(
    activity: _FakeActivity, speech_handle: SpeechHandle, *, content: str = "hello"
) -> asyncio.Task[None]:
    coro = AgentActivity._realtime_reply_task(
        cast(AgentActivity, activity),
        speech_handle=speech_handle,
        model_settings=ModelSettings(),
        user_message=ChatMessage(role="user", content=[content]),
        preemptive=True,
    )
    return asyncio.create_task(coro)


async def _wait_for_reply_fut(rt_session: FakeRealtimeSession) -> asyncio.Future:
    while not rt_session._reply_futs:
        await asyncio.sleep(0)
    return rt_session._reply_futs[-1]


def _fake_generation_ev() -> GenerationCreatedEvent:
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    message_ch.close()
    function_ch.close()
    return GenerationCreatedEvent(
        message_stream=message_ch, function_stream=function_ch, user_initiated=True
    )


# ---------------------------------------------------------------------------
# _realtime_reply_task, preemptive=True
# ---------------------------------------------------------------------------


async def test_preemptive_generates_before_authorization() -> None:
    # the whole point of the preemptive path: the response is requested while the speech is
    # NOT authorized, and nothing is committed to the local history yet
    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()

    task = _run_preemptive_reply(activity, handle)
    await _wait_for_reply_fut(rt_session)

    assert rt_session.generate_reply_calls == 1
    assert not handle._authorize_event.is_set()  # generation started without authorization
    # the user message was pushed to the server context...
    assert any(item.type == "message" and item.role == "user" for item in rt_session.chat_ctx.items)
    # ...but not committed locally, and no conversation event was emitted
    assert not activity._agent._chat_ctx.items
    assert activity.conversation_items == []
    assert activity.generation_calls == []

    handle._cancel()
    await task


async def test_preemptive_adoption_commits_and_drives_generation() -> None:
    # once authorized (the confirmed turn matched), the user message is committed and the
    # buffered generation is handed to the native realtime generation task
    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()

    task = _run_preemptive_reply(activity, handle)
    fut = await _wait_for_reply_fut(rt_session)
    fut.set_result(_fake_generation_ev())

    handle._authorize_generation()
    await task

    assert rt_session.generate_reply_calls == 1
    assert len(activity.generation_calls) == 1
    # local history commit happened exactly once, at adoption time
    user_items = [
        item
        for item in activity._agent._chat_ctx.items
        if item.type == "message" and item.role == "user"
    ]
    assert len(user_items) == 1
    assert activity.conversation_items == user_items
    assert not handle.done()


async def test_preemptive_rollback_cancels_and_restores() -> None:
    # interrupted before authorization (superseded or invalidated): the pending generation is
    # cancelled, the server context restored, and nothing leaks into the local history
    import time

    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()

    task = _run_preemptive_reply(activity, handle)
    fut = await _wait_for_reply_fut(rt_session)

    t0 = time.monotonic()
    handle._cancel()
    await task

    # regression: the rollback must never wait on its own cleanup future (it used to burn a
    # full 3s timeout on every rollback)
    assert time.monotonic() - t0 < 1.0

    assert fut.cancelled()  # the pending generation was cancelled
    # the speculative user message was removed from the server context
    assert not [
        item for item in rt_session.chat_ctx.items if item.type == "message" and item.role == "user"
    ]
    assert not activity._agent._chat_ctx.items
    assert activity.conversation_items == []
    assert activity.generation_calls == []
    # the cleanup future resolved so follow-up reply tasks can proceed
    assert activity._realtime_preemptive_cleanup is not None
    assert activity._realtime_preemptive_cleanup.done()


async def test_preemptive_rollback_interrupts_created_response() -> None:
    # if the speculative response was already created server-side, rollback must cancel it via
    # interrupt() and wait for the server to release it before completing
    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()

    task = _run_preemptive_reply(activity, handle)
    fut = await _wait_for_reply_fut(rt_session)
    fut.set_result(_fake_generation_ev())
    rt_session.active_generation = True  # the response is streaming server-side

    handle._cancel()
    await asyncio.sleep(0.05)  # rollback is waiting for the response to clear
    assert rt_session.interrupted
    assert activity._realtime_preemptive_cleanup is not None
    assert not activity._realtime_preemptive_cleanup.done()

    rt_session.active_generation = False  # server acks the cancellation
    await task
    assert activity._realtime_preemptive_cleanup.done()


async def test_preemptive_supersession_serializes_cleanup() -> None:
    # a superseding speculation (the user paused, then kept talking) must wait for the previous
    # rollback to fully restore the server context before snapshotting and pushing its own
    # message -- otherwise the late restore deletes the new turn's message from the server
    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)

    handle1 = SpeechHandle.create()
    task1 = _run_preemptive_reply(activity, handle1, content="one")
    await _wait_for_reply_fut(rt_session)

    # supersede: cancel #1 and immediately start #2 (mirrors on_preemptive_generation)
    handle1._cancel()
    handle2 = SpeechHandle.create()
    task2 = _run_preemptive_reply(activity, handle2, content="two")

    # #2's generation must appear (after #1's rollback), then adopt it
    while len(rt_session._reply_futs) < 2:
        await asyncio.sleep(0)
    rt_session._reply_futs[-1].set_result(_fake_generation_ev())
    handle2._authorize_generation()
    await task1
    await task2

    # the server context contains exactly the adopted turn's message: #1's rollback neither
    # left its own message behind nor deleted #2's
    user_texts = [
        item.text_content
        for item in rt_session.chat_ctx.items
        if item.type == "message" and item.role == "user"
    ]
    assert user_texts == ["two"]
    assert rt_session.generate_reply_calls == 2


async def test_preemptive_generation_failure_falls_back() -> None:
    # a speculative generate_reply rejected before adoption must mark the handle done (with the
    # error) and wake the parked reply task so it rolls back -- the adoption check then falls
    # back to a normal generation instead of scheduling a dead speech
    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()

    task = _run_preemptive_reply(activity, handle)
    fut = await _wait_for_reply_fut(rt_session)

    fut.set_exception(llm.RealtimeError("conversation_already_has_active_response"))
    await task  # the failure wakes the parked task, which rolls back

    assert handle.done()
    assert isinstance(handle.exception(), llm.RealtimeError)
    # rollback restored the server context (the failed response was never created, so no
    # interrupt was needed)
    assert not [i for i in rt_session.chat_ctx.items if i.type == "message" and i.role == "user"]
    assert not rt_session.interrupted
    assert activity._realtime_preemptive_cleanup is not None
    assert activity._realtime_preemptive_cleanup.done()


async def test_reply_task_waits_for_speculation_cleanup() -> None:
    # a non-preemptive reply (e.g. the fallback for the confirmed turn) must not generate while
    # a speculation rollback is still cleaning up the session
    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)
    cleanup_fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    activity._realtime_preemptive_cleanup = cleanup_fut

    handle = SpeechHandle.create()
    handle._authorize_generation()
    task = asyncio.create_task(
        AgentActivity._realtime_reply_task(
            cast(AgentActivity, activity),
            speech_handle=handle,
            model_settings=ModelSettings(),
            user_message=ChatMessage(role="user", content=["hello"]),
        )
    )

    await asyncio.sleep(0.05)
    assert rt_session.generate_reply_calls == 0  # still waiting on the cleanup

    cleanup_fut.set_result(None)
    fut = await _wait_for_reply_fut(rt_session)
    fut.set_result(_fake_generation_ev())
    await task
    assert rt_session.generate_reply_calls == 1


async def test_reply_task_waits_for_active_response_to_clear() -> None:
    # single-active-response serialization: generation is deferred until the server releases
    # the previous (e.g. just-interrupted) response
    rt_session = _text_realtime_model().session()
    rt_session.active_generation = True
    activity = _FakeActivity(rt_session)

    handle = SpeechHandle.create()
    handle._authorize_generation()
    task = asyncio.create_task(
        AgentActivity._realtime_reply_task(
            cast(AgentActivity, activity),
            speech_handle=handle,
            model_settings=ModelSettings(),
            user_message=ChatMessage(role="user", content=["hello"]),
        )
    )

    await asyncio.sleep(0.05)
    assert rt_session.generate_reply_calls == 0

    rt_session.active_generation = False
    fut = await _wait_for_reply_fut(rt_session)
    fut.set_result(_fake_generation_ev())
    await task
    assert rt_session.generate_reply_calls == 1


async def test_mirror_hold_buffers_and_replays_on_adoption() -> None:
    # server acks for the speculation's items (the pushed user message, the response output)
    # must not reach the local history mid-speculation -- they'd mutate the chat context and
    # invalidate the preemptive match; on adoption they're replayed with id-dedup
    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()

    task = _run_preemptive_reply(activity, handle)
    fut = await _wait_for_reply_fut(rt_session)  # the push has been acked by now
    rt_session.emit_remote_item(ChatMessage(role="assistant", content=["spec reply"]))

    # nothing reached the local history while the speculation is unresolved
    assert activity._agent._chat_ctx.items == []
    assert len(activity._realtime_preemptive_remote_items) == 2  # user ack + response item

    fut.set_result(_fake_generation_ev())
    handle._authorize_generation()
    await task

    roles = [i.role for i in activity._agent._chat_ctx.items if i.type == "message"]
    assert roles == ["user", "assistant"]  # exactly one of each: replay dedupes by id
    assert activity._realtime_preemptive_remote_items == []
    assert activity._realtime_preemptive_user_item_ids == set()


async def test_mirror_hold_rollback_removes_buffered_items() -> None:
    # on rollback, the held items are removed server-side and never reach the local history
    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()

    task = _run_preemptive_reply(activity, handle)
    await _wait_for_reply_fut(rt_session)
    rt_session.emit_remote_item(ChatMessage(role="assistant", content=["spec reply"]))

    handle._cancel()
    await task

    assert activity._agent._chat_ctx.items == []
    assert rt_session.chat_ctx.items == []  # user push and response item both removed
    assert activity._realtime_preemptive_remote_items == []
    assert activity._realtime_preemptive_user_item_ids == set()


async def test_mirror_hold_passes_unrelated_user_items() -> None:
    # user items not owned by the speculation (e.g. the committed turn audio) apply normally
    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()

    task = _run_preemptive_reply(activity, handle)
    await _wait_for_reply_fut(rt_session)
    rt_session.emit_remote_item(ChatMessage(role="user", content=["committed audio turn"]))

    local_users = [
        i.text_content
        for i in activity._agent._chat_ctx.items
        if i.type == "message" and i.role == "user"
    ]
    assert local_users == ["committed audio turn"]

    handle._cancel()
    await task
    # the rollback must not have removed the unrelated user item from the server context
    assert [
        i.text_content
        for i in rt_session.chat_ctx.items
        if i.type == "message" and i.role == "user"
    ] == ["committed audio turn"]


async def test_mirror_applies_normally_without_speculation() -> None:
    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)

    rt_session.emit_remote_item(ChatMessage(role="assistant", content=["hi"]))

    assert [i.role for i in activity._agent._chat_ctx.items] == ["assistant"]
    assert activity._realtime_preemptive_remote_items == []


async def test_speculation_ttl_rolls_back_stranded_speculation(monkeypatch) -> None:
    # a speculation is only resolved by a subsequent event; if none ever arrives (the user
    # vanished mid-utterance), the TTL cancels it so the cleanup future can't tax every later
    # reply with its wait timeout
    from livekit.agents.voice import agent_activity as activity_mod

    monkeypatch.setattr(activity_mod, "PREEMPTIVE_GENERATION_TTL", 0.2)

    rt_session = _text_realtime_model().session()
    activity = _FakeActivity(rt_session)
    handle = SpeechHandle.create()

    task = _run_preemptive_reply(activity, handle)
    fut = await _wait_for_reply_fut(rt_session)

    # no adoption, no cancellation: the TTL must resolve it
    await asyncio.wait_for(task, timeout=2.0)

    assert fut.cancelled()
    assert activity._realtime_preemptive_cleanup is not None
    assert activity._realtime_preemptive_cleanup.done()
    assert not [i for i in rt_session.chat_ctx.items if i.type == "message" and i.role == "user"]


# ---------------------------------------------------------------------------
# on_preemptive_generation eligibility
# ---------------------------------------------------------------------------


class _FakePreemptiveActivity(SimpleNamespace):
    """The attribute surface on_preemptive_generation touches."""

    def __init__(self, llm_obj: Any, rt_session: FakeRealtimeSession | None) -> None:
        generate_reply_calls: list[dict[str, Any]] = []

        def _generate_reply(**kwargs: Any) -> SpeechHandle:
            generate_reply_calls.append(kwargs)
            return SpeechHandle.create()

        super().__init__(
            llm=llm_obj,
            _rt_session=rt_session,
            preemptive_generation_opts={
                "enabled": True,
                "max_speech_duration": 10.0,
                "max_retries": 3,
            },
            _scheduling_paused=False,
            _new_turns_blocked=False,
            _current_speech=None,
            _preemptive_generation=None,
            _preemptive_generation_count=0,
            _agent=SimpleNamespace(chat_ctx=llm.ChatContext.empty()),
            tools=[],
            _tool_choice=None,
            _generate_reply=_generate_reply,
            generate_reply_calls=generate_reply_calls,
            _cancel_preemptive_generation=lambda: None,
        )
        self._can_preemptively_generate = MethodType(AgentActivity._can_preemptively_generate, self)
        self.on_preemptive_generation = MethodType(AgentActivity.on_preemptive_generation, self)


def _preemptive_info(transcript: str = "hello") -> _PreemptiveGenerationInfo:
    return _PreemptiveGenerationInfo(
        new_transcript=transcript, transcript_confidence=1.0, started_speaking_at=None
    )


async def test_on_preemptive_generation_starts_for_text_realtime() -> None:
    model = _text_realtime_model()
    rt_session = model.session()
    activity = _FakePreemptiveActivity(model, rt_session)

    activity.on_preemptive_generation(_preemptive_info())

    assert len(activity.generate_reply_calls) == 1
    call = activity.generate_reply_calls[0]
    assert call["schedule_speech"] is False
    assert call["preemptive"] is True
    assert activity._preemptive_generation is not None


async def test_on_preemptive_generation_skips_server_turn_detection() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(turn_detection=True))
    rt_session = model.session()
    activity = _FakePreemptiveActivity(model, rt_session)

    activity.on_preemptive_generation(_preemptive_info())

    assert activity.generate_reply_calls == []
    assert activity._preemptive_generation is None


async def test_on_preemptive_generation_skips_active_response() -> None:
    # the realtime API allows a single active response: never speculate while one is in flight
    # (e.g. the opening greeting the user talked over)
    model = _text_realtime_model()
    rt_session = model.session()
    rt_session.active_generation = True
    activity = _FakePreemptiveActivity(model, rt_session)

    activity.on_preemptive_generation(_preemptive_info())

    assert activity.generate_reply_calls == []
    assert activity._preemptive_generation is None


async def test_on_preemptive_generation_requires_rt_session() -> None:
    model = _text_realtime_model()
    activity = _FakePreemptiveActivity(model, None)

    activity.on_preemptive_generation(_preemptive_info())

    assert activity.generate_reply_calls == []
    assert activity._preemptive_generation is None


# ---------------------------------------------------------------------------
# full AgentSession integration
# ---------------------------------------------------------------------------


def _make_realtime_agent_session() -> tuple[AgentSession, FakeRealtimeModel]:
    from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
    from .fake_stt import FakeSTT, FakeUserSpeech
    from .fake_tts import FakeTTS, FakeTTSResponse
    from .fake_vad import FakeVAD

    speeches = [
        FakeUserSpeech(start_time=0.5, end_time=1.5, transcript="hello there", stt_delay=0.1)
    ]
    model = _text_realtime_model()
    session = AgentSession(
        llm=model,
        stt=FakeSTT(fake_user_speeches=speeches),
        vad=FakeVAD(
            fake_user_speeches=speeches, min_silence_duration=0.5, min_speech_duration=0.05
        ),
        tts=FakeTTS(
            fake_responses=[
                FakeTTSResponse(audio_duration=1.0, input="Hi!", ttfb=0.1, duration=0.2),
                FakeTTSResponse(audio_duration=1.0, input="Hi!", ttfb=0.1, duration=0.2),
            ]
        ),
        turn_handling={
            "turn_detection": "vad",
            "endpointing": {"min_delay": 0.5, "max_delay": 6.0},
            "interruption": {"min_duration": 0.5, "false_interruption_timeout": 2.0},
        },
        aec_warmup_duration=None,
    )
    session.input.audio = FakeAudioInput()
    session.output.audio = FakeAudioOutput()
    session.output.transcription = FakeTextOutput()
    return session, model


def _streamed_generation(text: str) -> GenerationCreatedEvent:
    message_ch = utils.aio.Chan[MessageGeneration]()
    function_ch = utils.aio.Chan[FunctionCall]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan["Any"]()
    modalities_fut: asyncio.Future = asyncio.get_event_loop().create_future()
    modalities_fut.set_result(["text"])
    message_ch.send_nowait(
        MessageGeneration(
            message_id=utils.shortuuid("item_"),
            text_stream=text_ch,
            audio_stream=audio_ch,
            modalities=modalities_fut,
        )
    )
    text_ch.send_nowait(text)
    text_ch.close()
    audio_ch.close()
    message_ch.close()
    function_ch.close()
    return GenerationCreatedEvent(
        message_stream=message_ch, function_stream=function_ch, user_initiated=True
    )


async def _run_realtime_session(
    session: AgentSession,
    model: FakeRealtimeModel,
    agent: Agent,
    reply_times: list[float],
) -> None:
    """Run the session while resolving every generate_reply with a scripted generation."""
    import time as time_mod

    from .fake_io import FakeAudioInput
    from .fake_stt import FakeSTT

    await session.start(agent)

    rt_session = model.active_session
    resolved: set[asyncio.Future] = set()

    async def _reply_watcher() -> None:
        while True:
            for fut in rt_session._reply_futs:
                if fut not in resolved:
                    resolved.add(fut)
                    reply_times.append(time_mod.time())

                    def _resolve(f: asyncio.Future = fut) -> None:
                        if not f.done():
                            f.set_result(_streamed_generation("Hi!"))

                    # small delay to mimic the server's response.created latency
                    asyncio.get_event_loop().call_later(0.15, _resolve)
            await asyncio.sleep(0.01)

    watcher = asyncio.create_task(_reply_watcher())
    try:
        stt = session.stt
        audio_input = session.input.audio
        assert isinstance(stt, FakeSTT)
        assert isinstance(audio_input, FakeAudioInput)
        audio_input.push(0.1)
        await stt.fake_user_speeches_done
        await asyncio.sleep(4.0)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        import contextlib

        with contextlib.suppress(RuntimeError):
            await session.drain()
        await session.aclose()


async def test_realtime_preemptive_adoption_e2e() -> None:
    # full turn: the speculative generation starts on the transcript (before end of turn) and,
    # since nothing changes, it is adopted -- exactly one generate_reply for the whole turn

    session, model = _make_realtime_agent_session()

    reply_times: list[float] = []
    agent = Agent(instructions="You are a helpful assistant.")
    await _run_realtime_session(session, model, agent, reply_times)

    rt_session = model.active_session
    assert rt_session.generate_reply_calls == 1  # speculation adopted; no second generation

    # the speculative generation was requested before the turn was committed: the speculation
    # starts at the final transcript (~speech end + stt delay) while the turn commits at
    # end-of-turn (~speech end + VAD silence + endpointing delay), marked by clear_audio()
    assert len(reply_times) == 1
    assert rt_session.audio_cleared_at is not None
    assert reply_times[0] < rt_session.audio_cleared_at

    # the user turn was committed exactly once, and the assistant reply made it to the history
    history = agent.chat_ctx.items
    user_msgs = [i for i in history if i.type == "message" and i.role == "user"]
    assistant_msgs = [i for i in history if i.type == "message" and i.role == "assistant"]
    assert len(user_msgs) == 1
    assert user_msgs[0].text_content == "hello there"
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].text_content == "Hi!"

    # adopted: the buffered turn audio was dropped, not committed (the reply came from the
    # pushed transcript)
    assert rt_session.audio_cleared
    assert not rt_session.committed


async def test_realtime_preemptive_invalidation_falls_back_e2e() -> None:
    # on_user_turn_completed edits the transcript, invalidating the speculation: the speculative
    # response is rolled back and a normal generation follows (two generate_reply calls total)

    session, model = _make_realtime_agent_session()

    class EditingAgent(Agent):
        async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message) -> None:
            new_message.content = ["hello there, edited"]

    reply_times: list[float] = []
    agent = EditingAgent(instructions="You are a helpful assistant.")
    await _run_realtime_session(session, model, agent, reply_times)

    rt_session = model.active_session
    # speculation + fallback
    assert rt_session.generate_reply_calls == 2
    # invalidated: the real turn audio was committed for the fallback generation
    assert rt_session.committed
    # the rolled-back speculative user message was removed from the server context (the realtime
    # fallback generates from the committed audio, so no user text message remains)
    stale_user_msgs = [
        i for i in rt_session.chat_ctx.items if i.type == "message" and i.role == "user"
    ]
    assert stale_user_msgs == []
    # the fallback reply still made it to the history
    assistant_msgs = [
        i for i in agent.chat_ctx.items if i.type == "message" and i.role == "assistant"
    ]
    assert len(assistant_msgs) == 1


async def test_realtime_preemptive_skip_reply_cancels_speculation_e2e() -> None:
    # commit_user_turn() forces skip_reply for realtime sessions: a pending speculation must be
    # cancelled and rolled back. Previously the realtime skip_reply branch returned early and
    # left the speculative reply task parked forever -- the cleanup future never resolved
    # (blocking every later reply on its timeout) and the speculative response kept the
    # single-active-response slot occupied.
    import contextlib

    from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics

    from .fake_io import FakeAudioInput

    session, model = _make_realtime_agent_session()
    agent = Agent(instructions="Answer in one short sentence.")
    await session.start(agent)

    audio_input = session.input.audio
    assert isinstance(audio_input, FakeAudioInput)
    audio_input.push(0.1)  # kick off the scripted fake STT/VAD timeline

    rt_session = model.active_session
    loop = asyncio.get_event_loop()

    # wait for the speculation to start (preemptive fires on the scripted final transcript)
    deadline = loop.time() + 5.0
    while not rt_session._reply_futs:
        assert loop.time() < deadline, "speculation never started"
        await asyncio.sleep(0.01)
    spec_fut = rt_session._reply_futs[0]

    activity = session._activity
    assert activity is not None
    assert activity._preemptive_generation is not None

    # the skip_reply end-of-turn (what commit_user_turn produces for realtime)
    info = _EndOfTurnInfo(
        skip_reply=True,
        new_transcript="hello there",
        transcript_confidence=1.0,
        metrics=_EndOfTurnMetrics(None, None, None, None),
    )
    assert activity.on_end_of_turn(info) is True

    cleanup = activity._realtime_preemptive_cleanup
    assert cleanup is not None
    # the cancelled speculation's rollback must resolve the cleanup future promptly
    await asyncio.wait_for(asyncio.shield(cleanup), timeout=2.0)

    assert activity._preemptive_generation is None
    assert spec_fut.cancelled()
    # the rollback removed the speculative user message from the server context
    assert not [i for i in rt_session.chat_ctx.items if i.type == "message" and i.role == "user"]

    # resolve any follow-up generation (the natural VAD end-of-turn) so teardown is clean
    await asyncio.sleep(1.0)
    for fut in rt_session._reply_futs:
        if not fut.done():
            fut.set_result(_fake_generation_ev())
    with contextlib.suppress(Exception):
        await asyncio.wait_for(session.aclose(), timeout=10.0)


async def test_realtime_preemptive_pause_cancels_parked_speculation_e2e() -> None:
    # pause() (the AgentTask handoff path) must cancel a parked speculation like drain() does;
    # previously it waited forever on the never-authorized speculative speech task, hanging the
    # handoff (and any session close queued behind it)
    import contextlib

    from .fake_io import FakeAudioInput

    session, model = _make_realtime_agent_session()
    agent = Agent(instructions="Answer in one short sentence.")
    await session.start(agent)

    audio_input = session.input.audio
    assert isinstance(audio_input, FakeAudioInput)
    audio_input.push(0.1)

    rt_session = model.active_session
    loop = asyncio.get_event_loop()
    deadline = loop.time() + 5.0
    while not rt_session._reply_futs:
        assert loop.time() < deadline, "speculation never started"
        await asyncio.sleep(0.01)

    activity = session._activity
    assert activity is not None
    assert activity._preemptive_generation is not None

    reusable = await asyncio.wait_for(activity.pause(blocked_tasks=[]), timeout=5.0)

    assert activity._preemptive_generation is None
    cleanup = activity._realtime_preemptive_cleanup
    assert cleanup is not None and cleanup.done()

    # restore the activity so the session can tear down normally, resolving any follow-up
    # generation (the natural end-of-turn) so drain doesn't wait on it
    await activity.resume(reuse_resources=reusable)
    await asyncio.sleep(1.0)
    for fut in rt_session._reply_futs:
        if not fut.done():
            fut.set_result(_fake_generation_ev())
    with contextlib.suppress(Exception):
        await asyncio.wait_for(session.aclose(), timeout=10.0)


async def test_realtime_preemptive_stop_response_commits_audio_e2e() -> None:
    # StopResponse from on_user_turn_completed ignores the turn while a speculation is pending:
    # the deferred audio commit must still resolve (commit, matching the non-preemptive
    # behavior) so the buffered audio can't leak into a later turn's commit, and the stale
    # speculation is dropped and rolled back
    from livekit.agents.llm import StopResponse

    session, model = _make_realtime_agent_session()

    class StoppingAgent(Agent):
        async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message) -> None:
            raise StopResponse()

    reply_times: list[float] = []
    agent = StoppingAgent(instructions="You are a helpful assistant.")
    await _run_realtime_session(session, model, agent, reply_times)

    rt_session = model.active_session
    # only the speculative generation ran; the ignored turn produced no fallback
    assert rt_session.generate_reply_calls == 1
    # the deferred audio commit resolved
    assert rt_session.committed
    # the speculation was rolled back: no user text message left in the server context
    assert not [i for i in rt_session.chat_ctx.items if i.type == "message" and i.role == "user"]
    # and no assistant reply was produced
    assert not [i for i in agent.chat_ctx.items if i.type == "message" and i.role == "assistant"]
