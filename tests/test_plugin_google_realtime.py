from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from google.genai import types

from livekit.agents import llm, utils
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit

# 10ms of silence at the output sample rate (24kHz mono, 16-bit)
_PCM_FRAME = b"\x00\x01" * 240


@asynccontextmanager
async def _make_session(
    monkeypatch: pytest.MonkeyPatch, *, model: str | None = None
) -> AsyncIterator[RealtimeSession]:
    """A session whose background connect loop is stopped before it hits the network.

    Closed on exit so the genai http clients are released here instead of by
    ``AsyncClient.__del__``, which schedules ``aclose()`` on whatever event loop
    is running when the collector happens to reach them.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    rt_model = RealtimeModel(model=model) if model else RealtimeModel()
    session = rt_model.session()
    # cancel the connect loop before the event loop ever schedules it, so no
    # websocket connection is attempted
    session._msg_ch.close()
    await utils.aio.cancel_and_wait(session._main_atask)
    try:
        yield session
    finally:
        await session.aclose()


def _tool_result_ctx(*call_ids: str) -> llm.ChatContext:
    """A chat context with one function_call + function_call_output pair per call id."""
    call_ids = call_ids or ("call_1",)
    chat_ctx = llm.ChatContext.empty()
    for call_id in call_ids:
        chat_ctx.items.append(
            llm.FunctionCall(
                id=f"fc_{call_id}", call_id=call_id, name="get_weather", arguments="{}"
            )
        )
        chat_ctx.items.append(
            llm.FunctionCallOutput(
                id=f"fco_{call_id}",
                call_id=call_id,
                name="get_weather",
                output="sunny",
                is_error=False,
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
    async with _make_session(monkeypatch) as session:
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
    async with _make_session(monkeypatch) as session:
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


async def test_session_close_releases_the_genai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aclose() must release the genai http clients.

    Otherwise they live until the collector runs ``AsyncClient.__del__``, which
    does ``asyncio.get_running_loop().create_task(self.aclose())`` - creating
    pending tasks on whatever event loop is running at that moment.
    """
    closed = False

    async with _make_session(monkeypatch) as session:
        real_aclose = session._client.aio.aclose

        async def _spy() -> None:
            nonlocal closed
            closed = True
            await real_aclose()

        monkeypatch.setattr(session._client.aio, "aclose", _spy)

    assert closed


async def test_tool_result_buffered_while_session_restarting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool result produced while the socket is restarting is stashed, not sent (issue #6479).

    update_tools() sets _session_should_close; a tool result arriving in that window would be
    delivered to the dying session and never reach the model, hanging the turn. It must be
    buffered for replay after the reconnect instead.
    """
    async with _make_session(monkeypatch) as session:
        # pretend a session is live and that it's being torn down (e.g. by update_tools())
        session._active_session = object()  # type: ignore[assignment]
        session._session_should_close.set()
        session._msg_ch = utils.aio.Chan()

        await session.update_chat_ctx(_tool_result_ctx())

        # buffered for replay, and nothing was sent to the dying session's send channel
        assert session._pending_tool_result is not None
        assert session._pending_tool_result.function_responses
        assert session._msg_ch.empty()


async def test_tool_result_buffered_when_no_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool result finishing during the reconnect window (no active session) is buffered.

    During a restart _main_task nulls _active_session while it reconnects. A tool result landing
    in that window must not be dropped: connect-time chat-context replay excludes
    function_call_output items, so it has to be buffered for replay (issue #6479 follow-up).
    """
    async with _make_session(monkeypatch) as session:
        # mid-reconnect: a session was established before, the socket is momentarily gone
        session._connected_once = True
        assert session._active_session is None
        assert not session._session_should_close.is_set()
        session._msg_ch = utils.aio.Chan()

        await session.update_chat_ctx(_tool_result_ctx())

        # buffered for replay rather than silently dropped
        assert session._pending_tool_result is not None
        assert session._pending_tool_result.function_responses
        assert session._msg_ch.empty()


async def test_initial_context_sync_does_not_buffer_historical_tool_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first context push (before any connect) must not resend historical tool outputs.

    At activity start / agent handoff the framework pushes the full chat context while no
    session exists yet. That context routinely holds function_call_output items from earlier
    turns; buffering+replaying them would make the model reply to stale results and speak
    unprompted, so they must be dropped, not buffered.
    """
    async with _make_session(monkeypatch) as session:
        # never connected: this is the initial context sync, not a restart
        assert session._connected_once is False
        assert session._active_session is None
        session._msg_ch = utils.aio.Chan()

        await session.update_chat_ctx(_tool_result_ctx())

        assert session._pending_tool_result is None
        assert session._msg_ch.empty()


async def test_multiple_tool_results_buffered_while_restarting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several tool results landing during one reconnect window accumulate (none dropped)."""
    async with _make_session(monkeypatch) as session:
        session._active_session = object()  # type: ignore[assignment]
        session._session_should_close.set()
        session._msg_ch = utils.aio.Chan()

        # first result lands...
        await session.update_chat_ctx(_tool_result_ctx("call_1"))
        # ...then a second one before the reconnect completes (superset diff yields call_2)
        await session.update_chat_ctx(_tool_result_ctx("call_1", "call_2"))

        assert session._pending_tool_result is not None
        responses = session._pending_tool_result.function_responses or []
        assert len(responses) == 2


async def test_tool_result_sent_when_session_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the session is not restarting, the tool result is sent normally (no buffering)."""
    async with _make_session(monkeypatch) as session:
        session._active_session = object()  # type: ignore[assignment]
        # _make_session closes the send channel to stop the connect loop; reopen it so we can
        # observe what update_chat_ctx sends (sends to a closed channel are silently suppressed)
        session._msg_ch = utils.aio.Chan()

        await session.update_chat_ctx(_tool_result_ctx())

        assert session._pending_tool_result is None
        sent = [session._msg_ch.recv_nowait() for _ in range(session._msg_ch.qsize())]
        assert any(
            isinstance(m, types.LiveClientToolResponse) and m.function_responses for m in sent
        )


async def test_generate_reply_allowed_without_mutable_chat_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_reply() must work for live-preview models that lack mutable_chat_context.

    The old hard block raised RealtimeError for these models; they support generation, they
    just need a realtime text input to nudge it rather than an appended client-content turn.
    """
    async with _make_session(monkeypatch, model="gemini-3.1-flash-live-preview") as session:
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
    async with _make_session(monkeypatch) as session:
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
    async with _make_session(monkeypatch) as session:
        session._start_new_generation()
        assert session._generation_completed is False

        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[types.FunctionCall(id="call_1", name="get_weather", args={})]
            )
        )

        assert session._generation_completed is True
        assert session._current_generation is not None and session._current_generation._done
