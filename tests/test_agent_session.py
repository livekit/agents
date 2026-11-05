import asyncio
from collections.abc import AsyncIterable, Callable

import pytest

from livekit import rtc
from livekit.agents import (
    Agent,
    APIConnectOptions,
    ModelSettings,
    RunContext,
    StopResponse,
    VoiceSession,
    utils,
)
from livekit.agents.llm import ChatContext, ChatMessage

from .fake_io import FakeAudioInput, FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import FakeSTT, FakeUserSpeech
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


def test_pre_start_provider_updates_and_views_need_no_event_loop() -> None:
    first, replacement = FakeTTS(), FakeTTS()
    session = VoiceSession(vad=None, tts=first, turn_detection="manual")
    session.update_options(tts=replacement, endpointing_opts={"min_delay": 0.4})
    assert session.tts is replacement
    assert session.options["endpointing"]["min_delay"] == 0.4
    assert session.keyterms == []
    assert session.usage == VoiceSession(vad=None).usage


async def test_agent_switch_preserves_the_conversation_history() -> None:
    history = ChatContext.empty()
    history.add_message(role="user", content="My booking reference is ABC123.")
    first = Agent(instructions="Welcome the caller.", name="reception")
    second = Agent(instructions="Help with bookings.", name="bookings")
    async with VoiceSession(vad=None, turn_handling={"turn_detection": None}) as session:
        await session.start(first, chat_ctx=history)
        session.update_agent(second)
        await asyncio.wait_for(session.wait_for_idle(), timeout=5)
        messages = [item.text_content for item in session.history.items if item.type == "message"]
        assert "My booking reference is ABC123." in messages
        assert session.current_agent is second


async def test_agent_switch_preserves_session_listening_options() -> None:
    async with VoiceSession(
        vad=None,
        turn_detection="manual",
        endpointing_opts={"min_delay": 0.3, "max_delay": 2.0},
    ) as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        session.update_agent(Agent(instructions="Help with bookings.", name="bookings"))
        await asyncio.wait_for(session.wait_for_idle(), timeout=5)
        assert session.options["turn_detection_mode"] == "manual"
        assert session.options["endpointing"]["min_delay"] == 0.3


async def test_temporary_options_restore_the_base_after_an_exception() -> None:
    session = VoiceSession(vad=None, turn_handling={"turn_detection": None})
    try:
        with pytest.raises(RuntimeError, match="interaction failed"):
            async with session.override_options(turn_detection="manual"):
                session.update_options(turn_detection="vad")
                raise RuntimeError("interaction failed")
        assert session.options["turn_detection_mode"] == "vad"
    finally:
        await session.aclose()


async def test_native_defaults_preserve_an_explicitly_disabled_user_away_timer() -> None:
    async with VoiceSession(vad=None, user_away_timeout=None) as session:
        assert session.options["user_away_timeout"] is None
        assert session.options["session_close_transcript_timeout"] == 2.0
        assert session.options["max_tool_steps"] == 3
    with pytest.raises(TypeError):
        VoiceSession(vad=None, max_tool_steps=None)


async def test_live_session_changes_interruption_and_delivery_settings() -> None:
    async def interruption(**expected: object) -> None:
        # the running core applies a change on its own loop
        for _ in range(100):
            if expected.items() <= session.options["interruption"].items():
                return
            await asyncio.sleep(0.01)
        assert expected.items() <= session.options["interruption"].items()

    async with VoiceSession(vad=None, turn_detection="manual") as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        session.update_options(
            interruption={"min_duration": 1.5},
            min_consecutive_speech_delay=0.4,
            tts_text_transforms=None,
        )
        async with session.override_options(interruption={"enabled": False}):
            async with session.override_options(interruption={"min_words": 2}):
                await interruption(enabled=False, min_words=2, min_duration=1.5)
            await interruption(enabled=False, min_words=0)
        await interruption(enabled=True, min_words=0, min_duration=1.5)
        assert session.options["min_consecutive_speech_delay"] == 0.4
        assert session.options["tts_text_transforms"] is None


async def _eventually(check: Callable[[], bool]) -> None:
    # the running core applies a change on its own loop
    for _ in range(100):
        if check():
            return
        await asyncio.sleep(0.01)
    assert check()


def _endpointing(session: VoiceSession) -> tuple[float, float]:
    endpointing = session.options["endpointing"]
    return endpointing["min_delay"], endpointing["max_delay"]


async def test_an_override_opened_before_start_is_released_after_it() -> None:
    async with VoiceSession(
        vad=None, turn_detection="manual", endpointing_opts={"min_delay": 0.3, "max_delay": 2.0}
    ) as session:
        async with session.override_options(endpointing_opts={"min_delay": 1.2}):
            await session.start(Agent(instructions="Welcome the caller.", name="reception"))
            await _eventually(lambda: _endpointing(session) == (1.2, 2.0))
        await _eventually(lambda: _endpointing(session) == (0.3, 2.0))


async def test_a_base_change_shows_through_the_settings_an_override_leaves_unset() -> None:
    async with VoiceSession(
        vad=None, turn_detection="manual", endpointing_opts={"min_delay": 0.3, "max_delay": 2.0}
    ) as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        async with session.override_options(endpointing_opts={"min_delay": 1.2}):
            session.update_options(endpointing_opts={"max_delay": 4.0})
            await _eventually(lambda: _endpointing(session) == (1.2, 4.0))
        await _eventually(lambda: _endpointing(session) == (0.3, 4.0))


async def test_overrides_released_out_of_order_leave_the_one_still_held() -> None:
    async with VoiceSession(
        vad=None, turn_detection="manual", endpointing_opts={"min_delay": 0.3, "max_delay": 2.0}
    ) as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        outer = session.override_options(endpointing_opts={"min_delay": 0.9})
        inner = session.override_options(endpointing_opts={"max_delay": 3.0})
        await outer.__aenter__()
        await inner.__aenter__()
        await outer.__aexit__(None, None, None)
        await _eventually(lambda: _endpointing(session) == (0.3, 3.0))
        await inner.__aexit__(None, None, None)
        await _eventually(lambda: _endpointing(session) == (0.3, 2.0))


async def test_an_override_on_a_session_closed_inside_it_raises_the_block_error() -> None:
    base, specialist = FakeTTS(fake_audio_duration=0.1), FakeTTS(fake_audio_duration=0.1)
    async with _voiced_session(base) as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        with pytest.raises(RuntimeError, match="interaction failed") as raised:
            async with session.override_options(tts=specialist, turn_detection="vad"):
                await session.aclose()
                raise RuntimeError("interaction failed")
        assert any("restoring the overridden providers" in n for n in raised.value.__notes__)


async def test_an_override_ending_on_a_session_closed_inside_it_ends_quietly() -> None:
    base, specialist = FakeTTS(fake_audio_duration=0.1), FakeTTS(fake_audio_duration=0.1)
    async with _voiced_session(base) as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        async with session.override_options(tts=specialist):
            await session.aclose()
        assert session.tts is base


async def test_provider_overrides_keep_base_updates_across_start_and_release_out_of_order() -> None:
    base, outer_tts, inner_tts, replacement = [FakeTTS(fake_audio_duration=0.1) for _ in range(4)]
    async with _voiced_session(base) as session:
        outer = session.override_options(tts=outer_tts, endpointing_opts={"min_delay": 0.9})
        await outer.__aenter__()
        try:
            await session.start(Agent(instructions="Welcome the caller.", name="reception"))
            assert session.tts is outer_tts
            async with session.override_options(tts=inner_tts):
                session.update_options(tts=replacement)
                assert session.tts is inner_tts
                await outer.__aexit__(None, None, None)
                assert session.tts is inner_tts
            assert session.tts is replacement
            await session.say("The updated base provider is restored.")
            assert _synthesized(replacement) == 1
            assert _synthesized(base) == 0
            assert _synthesized(outer_tts) == 0
            assert _synthesized(inner_tts) == 0
        finally:
            await outer.__aexit__(None, None, None)


async def test_provider_override_can_remove_a_provider_and_restore_an_updated_base() -> None:
    base, replacement = FakeTTS(fake_audio_duration=0.1), FakeTTS(fake_audio_duration=0.1)
    async with _voiced_session(base) as session:
        async with session.override_options(tts=None):
            assert session.tts is None
            session.update_options(tts=replacement)
            assert session.tts is None
        assert session.tts is replacement


def _voiced_session(tts: FakeTTS) -> VoiceSession:
    session = VoiceSession(vad=None, tts=tts, turn_handling={"turn_detection": None})
    session.output.audio = FakeAudioOutput()
    return session


def _synthesized(tts: FakeTTS) -> int:
    streams = 0
    while True:
        try:
            tts.stream_ch.recv_nowait()
        except utils.aio.ChanEmpty:
            return streams
        streams += 1


async def test_a_session_without_conn_options_keeps_the_provider_policy() -> None:
    class PolicyTTS(FakeTTS):
        def _core_conn_options(self) -> APIConnectOptions | None:
            return APIConnectOptions(read_timeout=7.0, max_retry=1)

    tts = PolicyTTS(fake_audio_duration=0.1)
    async with _voiced_session(tts) as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        await asyncio.wait_for(session.say("One moment."), timeout=5)
    stream = tts.stream_ch.recv_nowait()
    assert stream._conn_options.read_timeout == 7.0


async def test_update_options_replaces_the_session_voice_in_place() -> None:
    first, second = FakeTTS(fake_audio_duration=0.1), FakeTTS(fake_audio_duration=0.1)
    async with _voiced_session(first) as session:
        agent = Agent(instructions="Welcome the caller.", name="reception")
        await session.start(agent)
        session.update_options(tts=second)
        await asyncio.wait_for(session.say("One moment."), timeout=5)
        assert session.tts is second
        assert session.current_agent is agent
        assert _synthesized(first) == 0, "the replaced voice must not speak"
        assert _synthesized(second) == 1


async def test_override_options_speaks_with_the_layer_then_the_base() -> None:
    base, specialist = FakeTTS(fake_audio_duration=0.1), FakeTTS(fake_audio_duration=0.1)
    async with _voiced_session(base) as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        async with session.override_options(tts=specialist):
            await asyncio.wait_for(session.say("Let me read the terms back."), timeout=5)
        await asyncio.wait_for(session.say("Anything else?"), timeout=5)
        assert session.tts is base
        assert _synthesized(specialist) == 1
        assert _synthesized(base) == 1


async def test_close_before_start_prevents_opening_a_session() -> None:
    session = VoiceSession(vad=None, turn_detection="manual")
    await session.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
    assert session._owner.session is None


async def test_selecting_an_agent_before_start_keeps_its_callbacks_configurable() -> None:
    agent = Agent(instructions="Welcome the caller.", name="reception")
    entered = asyncio.Event()
    async with VoiceSession(vad=None, turn_detection="manual") as session:
        session.update_agent(agent)
        assert session.current_agent is agent

        @agent.on_enter
        async def greet(session: VoiceSession) -> None:
            entered.set()

        await session.start(agent)
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert session.current_agent is agent


async def test_driver_close_cleans_up_even_when_a_close_listener_raises() -> None:
    session = VoiceSession(vad=None, turn_detection="manual")

    @session.on("close")
    def raising_listener(event: object) -> None:
        raise ValueError("application close listener failed")

    await session.start(Agent(instructions="Welcome the caller.", name="reception"))
    try:
        # Simulate the driver closing itself, rather than the public close method.
        session._running.shutdown(drain=False)

        async def cleaned_up() -> None:
            while not session.store._closed:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(cleaned_up(), timeout=5)
    finally:
        await session.aclose()


async def test_failed_start_releases_acquired_toolsets_and_preserves_the_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.llm import Toolset
    from livekit.agents.voice._session_callbacks import _AgentHooks

    released: list[str] = []

    class Database(Toolset):
        async def aclose(self) -> None:
            released.append(self.id)

    failure = ValueError("invalid session capability")

    def fail_after_acquiring(hooks: _AgentHooks, tools):
        raise failure

    monkeypatch.setattr(_AgentHooks, "_core_tool_infos", fail_after_acquiring)
    session = VoiceSession(vad=None, tools=[Database(id="database")], turn_detection="manual")
    with pytest.raises(ValueError) as caught:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
    assert caught.value is failure
    assert released == ["database"]
    assert session.store._closed
    await session.aclose()
    assert released == ["database"]


async def test_cancelled_start_waiter_can_rejoin_the_native_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.voice._session_bindings import SessionHost

    preparing, release = asyncio.Event(), asyncio.Event()
    prepare = SessionHost.prepare
    calls = 0

    async def gated(host: SessionHost):
        nonlocal calls
        calls += 1
        preparing.set()
        await release.wait()
        return await prepare(host)

    monkeypatch.setattr(SessionHost, "prepare", gated)
    session = VoiceSession(vad=None, turn_detection="manual")
    agent = Agent(instructions="Welcome the caller.", name="reception")
    first = asyncio.create_task(session.start(agent))
    try:
        await asyncio.wait_for(preparing.wait(), timeout=5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(session.start(agent))
        release.set()
        await asyncio.wait_for(second, timeout=5)
        assert session.current_agent is agent
        assert calls == 1
    finally:
        release.set()
        await session.aclose()


async def test_a_session_tts_node_runs_for_each_agent_holding_the_call() -> None:
    tts = FakeTTS(fake_audio_duration=0.1)
    spoken_for: list[str] = []
    session = _voiced_session(tts)

    @session.tts_node
    def voice(
        ctx: RunContext, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        spoken_for.append(ctx.agent.name)
        return session.default.tts_node(ctx, text, model_settings)

    async with session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        await asyncio.wait_for(session.say("One moment."), timeout=5)
        handed_off = asyncio.Event()
        session.on(
            "conversation_item_added",
            lambda ev: ev.item.type == "agent_handoff" and handed_off.set(),
        )
        session.update_agent(Agent(instructions="Help with bookings.", name="bookings"))
        await asyncio.wait_for(handed_off.wait(), timeout=5)
        await asyncio.wait_for(session.say("Anything else?"), timeout=5)
        with pytest.raises(RuntimeError, match="before the session starts"):
            session.tts_node(voice)
    assert spoken_for == ["reception", "bookings"]
    assert _synthesized(tts) == 2


async def test_stop_response_from_the_turn_hook_skips_the_reply() -> None:
    speech = [FakeUserSpeech(start_time=0.1, end_time=0.4, transcript="hold on", stt_delay=0.05)]
    model = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hold on", content="Sure.", ttft=0, duration=0)]
    )
    session = VoiceSession(
        vad=FakeVAD(fake_user_speeches=speech, min_silence_duration=0.1),
        stt=FakeSTT(fake_user_speeches=speech),
        llm=model,
        tts=FakeTTS(fake_audio_duration=0.1),
        turn_handling={"turn_detection": None, "endpointing": {"min_delay": 0.1}},
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    session.output.audio = FakeAudioOutput()
    heard = asyncio.Event()

    @session.on_user_turn_completed
    async def hold_on(ctx: RunContext, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        assert new_message.text_content == "hold on"
        heard.set()
        raise StopResponse()

    async with session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        audio_input.push(0.1)
        await asyncio.wait_for(heard.wait(), timeout=5)
        await asyncio.sleep(0.5)
        replies = [i for i in session.history.items if i.type == "message" and i.role != "user"]
    assert replies == [], replies
    assert _synthesized(session.tts) == 0  # type: ignore[arg-type]
