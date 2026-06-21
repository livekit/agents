from types import SimpleNamespace

from livekit import rtc
from livekit.agents import llm, utils
from livekit.plugins.google.realtime.realtime_api import (
    RealtimeSession,
    _ResponseGeneration,  # pyright: ignore[reportPrivateUsage]
)


def _make_session() -> RealtimeSession:
    session = RealtimeSession.__new__(RealtimeSession)
    session._current_generation = None
    session._turn_complete_fallback_task = None
    session._opts = SimpleNamespace(output_audio_transcription=object())
    session._chat_ctx = llm.ChatContext.empty()
    session._pending_generation_fut = None
    session.emit = lambda *args, **kwargs: None
    return session


def _make_generation(response_id: str = "GR_test") -> _ResponseGeneration:
    return _ResponseGeneration(
        message_ch=utils.aio.Chan[llm.MessageGeneration](),
        function_ch=utils.aio.Chan[llm.FunctionCall](),
        input_id="GI_test",
        response_id=response_id,
        text_ch=utils.aio.Chan[str](),
        audio_ch=utils.aio.Chan[rtc.AudioFrame](),
    )


def _server_content(
    *, generation_complete: bool = False, turn_complete: bool = False
) -> SimpleNamespace:
    return SimpleNamespace(
        model_turn=None,
        input_transcription=None,
        output_transcription=None,
        generation_complete=generation_complete,
        turn_complete=turn_complete,
        interrupted=False,
    )


async def test_generation_complete_schedules_turn_complete_fallback() -> None:
    sess = _make_session()
    gen = _make_generation("GR_schedule")
    sess._current_generation = gen

    scheduled: list[str] = []
    sess._schedule_turn_complete_fallback = scheduled.append

    sess._handle_server_content(_server_content(generation_complete=True))

    assert scheduled == [gen.response_id]
    assert not gen._done


async def test_turn_complete_still_finalizes_generation() -> None:
    sess = _make_session()
    gen = _make_generation("GR_turn")
    sess._current_generation = gen

    sess._handle_server_content(_server_content(turn_complete=True))

    assert gen._done


async def test_turn_complete_fallback_finalizes_generation() -> None:
    sess = _make_session()
    gen = _make_generation("GR_fallback")
    sess._current_generation = gen

    await sess._wait_for_turn_complete_fallback(response_id=gen.response_id, timeout=0.0)

    assert gen._done
    assert gen.audio_ch.closed
    assert gen.message_ch.closed
