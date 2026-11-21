"""Who holds the floor in a realtime session: turn detection, interruptions and overlap."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    TurnHandlingOptions,
    function_tool,
    inference,
    llm,
    utils,
)
from livekit.agents.llm import RealtimeModelFallbackAdapter
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.agents.voice.turn import TurnDetectionMode

from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel, FakeRealtimeSession, _audio_frame, fake_capabilities
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.fixture(autouse=True)
def _livekit_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # inference.TurnDetector() and the adaptive detector read these; keep construction hermetic
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")


def _activity(session: AgentSession) -> AgentActivity:
    return AgentActivity(Agent(instructions="test"), session)


# --- server-side turn detection: client config x model capabilities -------------


_CAN_DISABLE = {"can_disable_turn_detection": True}
_PINNED = {"turn_detection": True, "can_disable_turn_detection": False}


@pytest.mark.parametrize(
    ("caps", "has_vad", "handling", "server_td"),
    [
        pytest.param(
            _CAN_DISABLE, True, {"turn_detection": "turn_detector"}, False, id="turn-detector"
        ),
        pytest.param(_CAN_DISABLE, True, {"turn_detection": "vad"}, False, id="vad"),
        pytest.param(
            _CAN_DISABLE,
            True,
            {"interruption": {"mode": "adaptive"}},
            False,
            id="adaptive-interruption",
        ),
        pytest.param(
            _CAN_DISABLE, False, {"turn_detection": "manual"}, False, id="manual-without-vad"
        ),
        pytest.param(
            _CAN_DISABLE, True, {"interruption": {"mode": "vad"}}, False, id="vad-interruption"
        ),
        # "stt" turn detection is rejected for realtime models
        pytest.param(_CAN_DISABLE, True, {"turn_detection": "stt"}, True, id="stt"),
        pytest.param(
            _CAN_DISABLE, True, {"turn_detection": "realtime_llm"}, True, id="realtime-llm"
        ),
        pytest.param(_CAN_DISABLE, True, {}, True, id="no-client-trigger"),
        # without a VAD, disabling would leave no turn detection at all
        pytest.param(
            _CAN_DISABLE, False, {"turn_detection": "vad"}, True, id="vad-mode-without-vad"
        ),
        # turn detection pinned on the model wins over the client
        pytest.param(_PINNED, True, {"turn_detection": "vad"}, True, id="pinned-on-model"),
        pytest.param(
            {"turn_detection": False, "can_disable_turn_detection": False},
            True,
            {"turn_detection": "vad"},
            False,
            id="off-on-model",
        ),
        # server VAD with create_response=False lets a turn-taking agent disable interruptions
        pytest.param(
            {"turn_detection": False, "can_disable_turn_detection": False},
            True,
            {"interruption": {"enabled": False}},
            False,
            id="client-turn-taking-uninterruptible",
        ),
    ],
)
def test_server_side_turn_detection_resolution(
    caps: dict[str, bool], has_vad: bool, handling: dict[str, Any], server_td: bool
) -> None:
    if handling.get("turn_detection") == "turn_detector":
        handling = {**handling, "turn_detection": inference.TurnDetector()}
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(**caps)),
        vad=FakeVAD(fake_user_speeches=[]) if has_vad else None,
        turn_handling=TurnHandlingOptions(**handling),  # type: ignore[typeddict-item]
    )

    assert _activity(session)._rt_turn_detection_enabled is server_td


def test_non_realtime_llm_reports_no_server_side_td() -> None:
    session = AgentSession(
        llm=FakeLLM(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
    )

    assert _activity(session)._rt_turn_detection_enabled is False


@pytest.mark.parametrize(
    ("initial", "changed_to", "warns"),
    [
        (None, "manual", True),
        ("vad", "realtime_llm", True),
        # reverting to automatic re-selects rather than pinning a client mode
        ("vad", None, False),
        ("vad", "manual", False),
    ],
)
def test_update_options_warns_when_server_td_would_flip(
    caplog: pytest.LogCaptureFixture,
    initial: TurnDetectionMode | None,
    changed_to: TurnDetectionMode | None,
    warns: bool,
) -> None:
    # the realtime session resolves server-side turn detection at start and never re-syncs
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection=initial)
        if initial
        else TurnHandlingOptions(),
    )
    session._activity = _activity(session)

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        session.update_options(turn_detection=changed_to)

    assert warns is any(
        r.levelno == logging.WARNING and "resolved at session start" in r.getMessage()
        for r in caplog.records
    )


def _td_ignored_warned(caplog: pytest.LogCaptureFixture) -> bool:
    return any(
        r.levelno == logging.WARNING and "ignoring the turn_detection setting" in r.getMessage()
        for r in caplog.records
    )


def test_default_turn_detector_is_superseded_silently(caplog: pytest.LogCaptureFixture) -> None:
    # nothing configured client-side: the eager TurnDetector() default is not user intent
    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
    )
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        activity = _activity(session)

    assert activity._rt_turn_detection_enabled is True
    assert activity._turn_detection is None
    assert not _td_ignored_warned(caplog)


def test_session_turn_detector_warns_when_model_keeps_server_td(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # can_disable=False: the user's TurnDetector really is dropped, so say so
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(turn_detection=True, can_disable_turn_detection=False)
        ),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(turn_detection=inference.TurnDetector()),
    )
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        activity = _activity(session)

    assert activity._turn_detection is None
    assert _td_ignored_warned(caplog)


def test_agent_turn_detector_warns_when_model_keeps_server_td(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # an agent-level override is user intent as well, even on a default session
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(turn_detection=True, can_disable_turn_detection=False)
        ),
        vad=FakeVAD(fake_user_speeches=[]),
    )
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        AgentActivity(Agent(instructions="test", turn_detection=inference.TurnDetector()), session)

    assert _td_ignored_warned(caplog)


def test_fallback_adapter_propagates_disable_to_underlying_session() -> None:
    m1 = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True))
    m2 = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True))
    adapter = RealtimeModelFallbackAdapter([m1, m2])

    assert adapter.capabilities.can_disable_turn_detection is True

    adapter.session(turn_detection_disabled=True)
    assert m1.created_sessions[-1].turn_detection_disabled is True


def test_fallback_adapter_support_is_conservative_and() -> None:
    m1 = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True))
    m2 = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=False))
    adapter = RealtimeModelFallbackAdapter([m1, m2])

    assert adapter.capabilities.can_disable_turn_detection is False


async def test_rt_session_reuse_respects_turn_detection_resolution() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(can_disable_turn_detection=True))

    def _act(**handling: object) -> AgentActivity:
        session = AgentSession(
            llm=model,
            vad=FakeVAD(fake_user_speeches=[]),
            turn_handling=TurnHandlingOptions(**handling),  # type: ignore[typeddict-item]
        )
        return _activity(session)

    server_on = _act()  # no client trigger -> server-side TD on
    client_a = _act(turn_detection="vad")  # client vad -> server-side TD off
    client_b = _act(turn_detection="vad")
    assert server_on._rt_turn_detection_enabled is True
    assert client_a._rt_turn_detection_enabled is False

    # differing resolution -> refuse reuse, fall back to a fresh session
    server_on._rt_session = model.session()
    assert (await server_on._detach_reusable_resources(client_a)).rt_session is None

    # same resolution -> reuse
    client_a._rt_session = model.session()
    assert (await client_a._detach_reusable_resources(client_b)).rt_session is not None


# --- user speech during a reply ------------------------------------------------


def _speech_started(*, server_turn_detection: bool, allow_interruptions: bool) -> AgentActivity:
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=server_turn_detection, can_disable_turn_detection=False
            )
        ),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(interruption={"enabled": allow_interruptions}),
    )
    activity = _activity(session)
    activity._current_speech = SpeechHandle.create(allow_interruptions=allow_interruptions)
    activity._rt_session = MagicMock()
    # a bare MagicMock reports every capability as true; report the model's own instead
    activity._rt_session.capabilities = activity.llm.capabilities
    activity._on_input_speech_started(llm.InputSpeechStartedEvent())
    return activity


def test_input_speech_started_keeps_uninterruptible_speech(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # server VAD still commits and transcribes audio, but neither side interrupts, and this
    # valid configuration logs no error
    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        activity = _speech_started(server_turn_detection=False, allow_interruptions=False)

    assert activity._current_speech is not None
    assert activity._current_speech.interrupted is False
    activity._rt_session.interrupt.assert_not_called()
    assert not caplog.records


def test_server_speech_start_is_an_audio_activity_interruption() -> None:
    activity = _speech_started(server_turn_detection=True, allow_interruptions=True)

    handle = activity._current_speech
    assert handle is not None
    assert handle.interrupted is True
    assert handle._interrupt_source == "audio_activity"
    activity._rt_session.interrupt.assert_called_once()


# --- overlapping speech -------------------------------------------------------


class _TracingAudioOutput(FakeAudioOutput):
    """Counts how often the playout was cleared or paused."""

    def __init__(self, *, can_pause: bool = False) -> None:
        super().__init__(can_pause=can_pause)
        self.clears = 0
        self.pauses = 0

    def clear_buffer(self) -> None:
        self.clears += 1
        super().clear_buffer()

    def pause(self) -> None:
        self.pauses += 1
        super().pause()


async def _start_speech(
    session: AgentSession, model: FakeRealtimeModel
) -> tuple[utils.aio.Chan[rtc.AudioFrame], utils.aio.Chan[str]]:
    """Drive one assistant generation and leave it mid-playout."""
    session.generate_reply()
    for _ in range(50):
        if model.active_session._reply_futs:
            break
        await asyncio.sleep(0)
    assert model.active_session._reply_futs, "the session never asked the model for a reply"

    message_ch = utils.aio.Chan[llm.MessageGeneration]()
    function_ch = utils.aio.Chan[llm.FunctionCall]()
    text_ch = utils.aio.Chan[str]()
    audio_ch = utils.aio.Chan[rtc.AudioFrame]()
    modalities = asyncio.Future[list[str]]()
    modalities.set_result(["audio", "text"])

    message_ch.send_nowait(
        llm.MessageGeneration(
            message_id="msg-1",
            text_stream=text_ch,
            audio_stream=audio_ch,
            modalities=modalities,
        )
    )
    message_ch.close()
    function_ch.close()
    text_ch.send_nowait("the weather today is")
    audio_ch.send_nowait(
        rtc.AudioFrame(
            data=b"\x00\x00" * 2400, sample_rate=24000, num_channels=1, samples_per_channel=2400
        )
    )

    model.active_session._reply_futs[0].set_result(
        llm.GenerationCreatedEvent(
            message_stream=message_ch,
            function_stream=function_ch,
            user_initiated=True,
        )
    )
    # let the speech reach playout, but leave the streams open so it stays in flight
    for _ in range(50):
        await asyncio.sleep(0)
    return audio_ch, text_ch


@pytest.mark.parametrize("overlapping", [True, False])
async def test_speech_started_interrupts_only_a_model_that_cannot_overlap(
    overlapping: bool,
) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(supports_overlapping_speech=overlapping)
    )
    audio_out = _TracingAudioOutput()

    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        session.output.audio = audio_out
        await session.start(Agent(instructions="be concise"))
        audio_ch, text_ch = await _start_speech(session, model)
        speech = session.current_speech
        assert speech is not None

        model.active_session.emit("input_speech_started", llm.InputSpeechStartedEvent())
        for _ in range(20):
            await asyncio.sleep(0)

        assert speech.interrupted is not overlapping
        assert (audio_out.clears > 0) is not overlapping
        assert session.user_state == "speaking"

        audio_ch.close()
        text_ch.close()


@pytest.mark.parametrize("overlapping", [False, True])
async def test_audio_activity_interruption_stays_off_under_server_turn_detection(
    overlapping: bool,
) -> None:
    # the model detects the user itself, so the VAD path defers to it either way
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(supports_overlapping_speech=overlapping)
    )
    audio_out = _TracingAudioOutput()

    async with AgentSession(llm=model, vad=FakeVAD(), aec_warmup_duration=None) as session:
        session.output.audio = audio_out
        await session.start(Agent(instructions="be concise"))
        audio_ch, text_ch = await _start_speech(session, model)
        speech = session.current_speech
        assert speech is not None
        assert session._activity is not None

        session._activity._interrupt_by_audio_activity()
        for _ in range(20):
            await asyncio.sleep(0)

        assert speech.interrupted is False
        assert audio_out.clears == 0

        audio_ch.close()
        text_ch.close()


@pytest.mark.parametrize("overlapping", [False, True])
async def test_allow_interruptions_false_is_rejected_under_server_turn_detection(
    overlapping: bool,
) -> None:
    # the model decides when to yield either way, so the setting cannot be honored
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(supports_overlapping_speech=overlapping)
    )
    async with AgentSession(
        llm=model,
        turn_handling=TurnHandlingOptions(interruption={"enabled": False}),
        aec_warmup_duration=None,
    ) as session:
        with pytest.raises(ValueError, match="allow_interruptions cannot be False"):
            await session.start(Agent(instructions="be concise"))


async def test_the_model_speaks_over_a_caller_who_is_still_talking() -> None:
    # the caller's turn must not hold back a model that decides for itself when to speak, and
    # a pausable output must not park the burst either
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_overlapping_speech=True))
    audio_out = _TracingAudioOutput(can_pause=True)

    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        session.output.audio = audio_out
        await session.start(Agent(instructions="be concise"))
        activity = session._activity
        assert activity is not None

        # what a configured VAD reports when the caller starts talking, and keeps talking
        activity.on_start_of_speech(None, speech_start_time=time.time())
        assert not activity._user_silence_event.is_set()

        audio_ch, text_ch = await _start_speech(session, model)

        assert audio_out.pauses == 0
        assert audio_out.clears == 0
        assert session.agent_state == "speaking"

        audio_ch.close()
        text_ch.close()


async def test_a_model_that_cannot_overlap_still_waits_the_caller_out() -> None:
    # the caller's turn keeps gating every other realtime model
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_overlapping_speech=False))

    async with AgentSession(llm=model, aec_warmup_duration=None) as session:
        session.output.audio = _TracingAudioOutput(can_pause=True)
        await session.start(Agent(instructions="be concise"))
        activity = session._activity
        assert activity is not None

        activity.on_start_of_speech(None, speech_start_time=time.time())
        session.generate_reply()
        for _ in range(50):
            await asyncio.sleep(0)

        # the reply is never even asked for while the caller holds the floor
        assert not model.active_session._reply_futs
        assert session.agent_state != "speaking"


async def test_client_side_turn_taking_takes_the_floor_back() -> None:
    # overlap is the model's only while it owns turn-taking; once the client drives turns the
    # framework does, and its VAD would otherwise cut a reply it never gated
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            supports_overlapping_speech=True, can_disable_turn_detection=True
        )
    )

    async with AgentSession(
        llm=model,
        vad=FakeVAD(),
        turn_handling=TurnHandlingOptions(turn_detection="vad"),
        aec_warmup_duration=None,
    ) as session:
        session.output.audio = _TracingAudioOutput(can_pause=True)
        await session.start(Agent(instructions="be concise"))
        activity = session._activity
        assert activity is not None
        assert activity._rt_turn_detection_enabled is False
        assert activity._rt_overlapping_speech_enabled is False

        activity.on_start_of_speech(None, speech_start_time=time.time())
        session.generate_reply()
        for _ in range(50):
            await asyncio.sleep(0)

        assert not model.active_session._reply_futs
        assert session.agent_state != "speaking"


# --- interrupting a reply cancels its response ---------------------------------
# the cancel is session-wide, so a speech whose generation already finished must never send one


@asynccontextmanager
async def _speaking_reply() -> AsyncIterator[
    tuple[
        AgentSession,
        FakeRealtimeSession,
        SpeechHandle,
        FakeAudioOutput,
        utils.aio.Chan[llm.FunctionCall],
    ]
]:
    """Drive a realtime reply that is still playing, with its function stream left open."""
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    audio_output = FakeAudioOutput(can_pause=True)

    async with AgentSession(llm=model) as session:
        session.output.audio = audio_output
        await session.start(Agent(instructions="test"))

        handle = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])

        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("a message the app cuts short")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(1.0))
        audio_ch.close()

        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-id",
            )
        )

        while audio_output._started_at is None:
            await asyncio.sleep(0)

        yield session, model.active_session, handle, audio_output, function_ch


async def test_forced_speech_interrupt_cancels_the_response() -> None:
    async with _speaking_reply() as (_, rt_session, handle, _, function_ch):
        handle.interrupt(force=True)
        await asyncio.sleep(0.1)

        assert rt_session.interrupted

        function_ch.close()
        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)


async def test_interrupting_a_played_out_response_spares_the_newer_reply() -> None:
    # the model finishes the response long before the buffered audio does, so from here on a
    # cancel would land on whichever response the model started next
    async with _speaking_reply() as (_, rt_session, handle, _, function_ch):
        function_ch.close()
        await asyncio.sleep(0.1)

        handle.interrupt(force=True)
        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)

        assert handle.interrupted
        assert not rt_session.interrupted


async def test_interrupting_a_queued_finished_response_spares_the_newer_reply() -> None:
    # a server-initiated turn waits for the speech ahead of it to play out, and its own
    # response can end, and be replaced, long before it is ever authorized to speak
    async with _speaking_reply() as (session, rt_session, _, _, function_ch):
        queued: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: queued.append(ev.speech_handle))

        b_message_ch = utils.aio.Chan[llm.MessageGeneration]()
        b_function_ch = utils.aio.Chan[llm.FunctionCall]()
        b_message_ch.close()
        b_function_ch.close()
        rt_session.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=b_message_ch,
                function_stream=b_function_ch,
                user_initiated=False,
                response_id="response-b",
            ),
        )
        await asyncio.sleep(0.1)

        assert len(queued) == 1
        queued[0].interrupt(force=True)
        await asyncio.sleep(0.1)

        assert queued[0].interrupted
        assert not rt_session.interrupted

        function_ch.close()


async def test_paused_playout_keeps_the_response() -> None:
    async with _speaking_reply() as (_, rt_session, handle, audio_output, function_ch):
        function_ch.close()
        audio_output.pause()
        await asyncio.sleep(1.0)

        assert not handle.interrupted
        assert not rt_session.interrupted

        audio_output.resume()
        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)

        assert not rt_session.interrupted


async def test_interrupt_racing_the_generation_event_cancels_the_response() -> None:
    # the generation event and the interrupt land in the same tick, so the reply task returns
    # before it ever forwards the response: nothing else is left to cancel it
    model = FakeRealtimeModel(capabilities=fake_capabilities())

    async with AgentSession(llm=model) as session:
        await session.start(Agent(instructions="test"))

        handle = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        message_ch.close()
        function_ch.close()
        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-id",
            )
        )
        handle.interrupt(force=True)
        await asyncio.wait_for(handle.wait_for_playout(), timeout=5)

        assert model.active_session.interrupted


async def test_queued_reply_interrupted_before_playout_leaves_server_history() -> None:
    # the server already holds a reply that waits behind the current speech; interrupted
    # before it is authorized, the user never hears it, so the server must not keep it
    async with _speaking_reply() as (session, rt_session, _, _, function_ch):
        queued: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: queued.append(ev.speech_handle))

        rt_session._chat_ctx = rt_session.chat_ctx.copy()
        rt_session._chat_ctx.add_message(role="assistant", content="never heard", id="unplayed")

        b_message_ch = utils.aio.Chan[llm.MessageGeneration]()
        b_function_ch = utils.aio.Chan[llm.FunctionCall]()
        b_message_ch.close()
        b_function_ch.close()
        rt_session.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=b_message_ch,
                function_stream=b_function_ch,
                user_initiated=False,
                response_id="response-b",
            ),
        )
        await asyncio.sleep(0.1)

        assert len(queued) == 1
        queued[0].interrupt(force=True)
        await asyncio.sleep(0.1)
        function_ch.close()

        assert rt_session.chat_ctx.get_by_id("unplayed") is None


async def test_queued_reply_interrupted_before_playout_drops_its_placeholder() -> None:
    # remote_item_added copies the reply into the local context, and the sync only deletes what
    # the local context lacks, so a surviving placeholder keeps the reply on the server
    async with _speaking_reply() as (session, rt_session, _, _, function_ch):
        queued: list[SpeechHandle] = []
        session.on("speech_created", lambda ev: queued.append(ev.speech_handle))

        reply = llm.ChatMessage(role="assistant", content=[], id="unplayed")
        rt_session._chat_ctx = rt_session.chat_ctx.copy()
        rt_session._chat_ctx.items.append(reply)

        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        text_ch.close()
        audio_ch.close()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])

        b_message_ch = utils.aio.Chan[llm.MessageGeneration]()
        b_function_ch = utils.aio.Chan[llm.FunctionCall]()
        b_message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="unplayed",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        b_message_ch.close()
        b_function_ch.close()
        rt_session.emit(
            "generation_created",
            llm.GenerationCreatedEvent(
                message_stream=b_message_ch,
                function_stream=b_function_ch,
                user_initiated=False,
                response_id="response-b",
            ),
        )
        rt_session.emit(
            "remote_item_added", llm.RemoteItemAddedEvent(previous_item_id=None, item=reply)
        )
        await asyncio.sleep(0.1)

        assert session.current_agent._chat_ctx.get_by_id("unplayed") is not None
        assert len(queued) == 1
        queued[0].interrupt(force=True)
        await asyncio.sleep(0.1)
        function_ch.close()

        assert session.current_agent._chat_ctx.get_by_id("unplayed") is None
        assert rt_session.chat_ctx.get_by_id("unplayed") is None


async def test_interrupting_a_finished_speech_spares_the_newer_reply() -> None:
    # the older handle stays alive for its tool while the next reply generates, and the cancel
    # it would send is the one that reply is running on
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()

    class ToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="test")

        @function_tool
        async def lookup(self) -> str:
            """Look something up."""
            tool_started.set()
            await release_tool.wait()
            return "ok"

    model = FakeRealtimeModel(capabilities=fake_capabilities())

    async with AgentSession(llm=model) as session:
        session.output.audio = FakeAudioOutput(can_pause=True)
        await session.start(ToolAgent())
        rt_session = model.active_session

        handle_a = session.generate_reply()
        while not rt_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["audio", "text"])

        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-a",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        text_ch.send_nowait("let me check")
        text_ch.close()
        audio_ch.send_nowait(_audio_frame(0.2))
        audio_ch.close()
        function_ch.send_nowait(llm.FunctionCall(call_id="1", name="lookup", arguments="{}"))
        function_ch.close()

        rt_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="response-a",
            )
        )

        await asyncio.wait_for(tool_started.wait(), timeout=5)
        handle_b = session.generate_reply()

        # A's generation ends with its playout, and only then is B authorized to ask for a response
        async def _wait_for_second_reply() -> None:
            while rt_session.generate_reply_calls < 2:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_wait_for_second_reply(), timeout=5)
        assert not handle_a.done()
        assert session._activity is not None
        assert handle_a in session._activity._background_speeches

        rt_session.interrupted = False
        handle_a.interrupt(force=True)
        await asyncio.sleep(0.1)

        assert handle_a.interrupted
        assert not rt_session.interrupted

        release_tool.set()
        b_message_ch = utils.aio.Chan[llm.MessageGeneration]()
        b_function_ch = utils.aio.Chan[llm.FunctionCall]()
        b_message_ch.close()
        b_function_ch.close()
        rt_session._reply_futs[1].set_result(
            llm.GenerationCreatedEvent(
                message_stream=b_message_ch,
                function_stream=b_function_ch,
                user_initiated=True,
                response_id="response-b",
            )
        )

        await asyncio.wait_for(handle_b.wait_for_playout(), timeout=5)
        await asyncio.wait_for(handle_a.wait_for_playout(), timeout=5)

        assert not rt_session.interrupted
