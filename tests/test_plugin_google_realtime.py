from __future__ import annotations

import logging

import pytest
from google.genai import types

from livekit.agents import llm, utils
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit

# 10ms of silence at the output sample rate (24kHz mono, 16-bit)
_PCM_FRAME = b"\x00\x01" * 240


async def _make_session(
    monkeypatch: pytest.MonkeyPatch, *, model: str | None = None
) -> RealtimeSession:
    """A session whose background connect loop is stopped before it hits the network."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    rt_model = RealtimeModel(model=model) if model else RealtimeModel()
    session = rt_model.session()
    # cancel the connect loop before the event loop ever schedules it, so no
    # websocket connection is attempted
    session._msg_ch.close()
    await utils.aio.cancel_and_wait(session._main_atask)
    return session


def _tool_result_ctx() -> llm.ChatContext:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.items.append(
        llm.FunctionCall(id="fc_1", call_id="call_1", name="get_weather", arguments="{}")
    )
    chat_ctx.items.append(
        llm.FunctionCallOutput(
            id="fco_1", call_id="call_1", name="get_weather", output="sunny", is_error=False
        )
    )
    return chat_ctx


def _audio_content(**kwargs: object) -> types.LiveServerContent:
    return types.LiveServerContent(
        model_turn=types.Content(
            parts=[types.Part(inline_data=types.Blob(data=_PCM_FRAME, mime_type="audio/pcm"))]
        ),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_output_streams_close_on_generation_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generation_complete ends the audio/text segment; finalization waits for turn_complete.

    Gemini delays turn_complete until it estimates client-side playback has finished, so
    keying the stream close off turn_complete makes AudioSegmentEnd (and the finalized
    transcript) arrive seconds late (issue #6421). Both streams must close on
    generation_complete, while the generation stays open until turn_complete for input
    transcription and metrics.
    """
    session = await _make_session(monkeypatch)
    session._start_new_generation()
    gen = session._current_generation
    assert gen is not None

    session._handle_server_content(
        _audio_content(
            output_transcription=types.Transcription(text="hello"),
            generation_complete=True,
        )
    )

    # audio and text were consumed and both segments ended immediately
    assert gen._first_token_timestamp is not None
    assert gen.output_text == "hello"
    assert gen.audio_ch.closed
    assert gen.text_ch.closed
    # but the generation is still open for trailing input transcription until turn_complete
    assert not gen._done
    assert not gen.message_ch.closed

    session._handle_server_content(types.LiveServerContent(turn_complete=True))

    assert gen._done
    assert gen.message_ch.closed


async def test_late_content_after_generation_complete_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stray audio/text after generation_complete is dropped (not pushed to a closed stream)."""
    session = await _make_session(monkeypatch)
    session._start_new_generation()
    gen = session._current_generation
    assert gen is not None

    session._handle_server_content(_audio_content(generation_complete=True))
    assert gen.audio_ch.closed and gen.text_ch.closed

    with caplog.at_level(logging.WARNING):
        # must not raise ChanClosed, must not append to the transcript, and must warn
        session._handle_server_content(
            _audio_content(output_transcription=types.Transcription(text="late"))
        )

    assert gen.audio_ch.closed and gen.text_ch.closed
    assert gen.output_text == ""
    assert not gen._done
    assert any("after generation completed" in r.message for r in caplog.records)


async def test_tool_result_buffered_while_session_restarting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool result produced while the socket is restarting is stashed, not sent (issue #6479).

    update_tools() sets _session_should_close; a tool result arriving in that window would be
    delivered to the dying session and never reach the model, hanging the turn. It must be
    buffered for replay after the reconnect instead.
    """
    session = await _make_session(monkeypatch)
    # pretend a session is live and that it's being torn down (e.g. by update_tools())
    session._active_session = object()  # type: ignore[assignment]
    session._session_should_close.set()
    session._msg_ch = utils.aio.Chan()

    await session.update_chat_ctx(_tool_result_ctx())

    # buffered for replay, and nothing was sent to the dying session's send channel
    assert session._pending_tool_result is not None
    assert session._pending_tool_result.function_responses
    assert session._msg_ch.empty()


async def test_tool_result_sent_when_session_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the session is not restarting, the tool result is sent normally (no buffering)."""
    session = await _make_session(monkeypatch)
    session._active_session = object()  # type: ignore[assignment]
    # _make_session closes the send channel to stop the connect loop; reopen it so we can
    # observe what update_chat_ctx sends (sends to a closed channel are silently suppressed)
    session._msg_ch = utils.aio.Chan()

    await session.update_chat_ctx(_tool_result_ctx())

    assert session._pending_tool_result is None
    sent = [session._msg_ch.recv_nowait() for _ in range(session._msg_ch.qsize())]
    assert any(isinstance(m, types.LiveClientToolResponse) and m.function_responses for m in sent)


async def test_generate_reply_allowed_without_mutable_chat_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_reply() must work for live-preview models that lack mutable_chat_context.

    The old hard block raised RealtimeError for these models; they support generation, they
    just need a realtime text input to nudge it rather than an appended client-content turn.
    """
    session = await _make_session(monkeypatch, model="gemini-3.1-flash-live-preview")
    assert not session._realtime_model.capabilities.mutable_chat_context
    session._msg_ch = utils.aio.Chan()

    fut = session.generate_reply()

    # no longer rejected up-front
    assert not (fut.done() and fut.exception() is not None)

    sent = [session._msg_ch.recv_nowait() for _ in range(session._msg_ch.qsize())]
    # nudged via a realtime text input, not an appended client-content turn
    assert any(isinstance(m, types.LiveClientRealtimeInput) and m.text for m in sent)
    assert not any(isinstance(m, types.LiveClientContent) for m in sent)

    fut.cancel()


async def test_generation_completed_flag_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """_generation_completed tracks the turn lifecycle so trailing model_turns can be guarded."""
    session = await _make_session(monkeypatch)
    # idle before any generation
    assert session._generation_completed is True

    session._start_new_generation()
    # a generation is now in flight
    assert session._generation_completed is False

    session._handle_server_content(types.LiveServerContent(turn_complete=True))
    # completion signal flips it back
    assert session._generation_completed is True


async def test_tool_call_marks_generation_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool call completes the current generation so the post-tool reply starts a fresh one."""
    session = await _make_session(monkeypatch)
    session._start_new_generation()
    assert session._generation_completed is False

    session._handle_tool_calls(
        types.LiveServerToolCall(
            function_calls=[types.FunctionCall(id="call_1", name="get_weather", args={})]
        )
    )

    assert session._generation_completed is True
    assert session._current_generation is not None and session._current_generation._done
