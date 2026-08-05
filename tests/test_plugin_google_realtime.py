from __future__ import annotations

import asyncio
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
async def _make_session(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[RealtimeSession]:
    """A session whose background connect loop is stopped before it hits the network.

    Closed on exit so the genai http clients are released here instead of by
    ``AsyncClient.__del__``, which schedules ``aclose()`` on whatever event loop
    is running when the collector happens to reach them.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    session = RealtimeModel().session()
    # cancel the connect loop before the event loop ever schedules it, so no
    # websocket connection is attempted
    session._msg_ch.close()
    await utils.aio.cancel_and_wait(session._main_atask)
    try:
        yield session
    finally:
        await session.aclose()


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


async def test_aborted_turn_fails_generate_reply_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turn the server rejects never creates a generation (#6708).

    ``generate_reply``'s future is only ever resolved by ``generation_created``, so an
    abort left the caller waiting out the full 5s timeout for an outcome the server had
    already reported in ~250ms.
    """
    async with _make_session(monkeypatch) as session:
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        session._pending_generation_fut = fut

        session._handle_server_content(
            types.LiveServerContent(
                turn_complete=True,
                turn_complete_reason=types.TurnCompleteReason.MALFORMED_FUNCTION_CALL,
            )
        )

        assert fut.done(), "the caller is still waiting on a turn the server already ended"
        with pytest.raises(llm.RealtimeError, match="MALFORMED_FUNCTION_CALL"):
            fut.result()


async def test_turn_without_a_reason_still_fails_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        session._pending_generation_fut = fut

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert fut.done()
        with pytest.raises(llm.RealtimeError):
            fut.result()


async def test_content_without_turn_complete_leaves_the_caller_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # only the end of a turn settles it; a stray frame must not fail a live request
    async with _make_session(monkeypatch) as session:
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        session._pending_generation_fut = fut

        session._handle_server_content(types.LiveServerContent(interrupted=True))

        assert not fut.done()
        fut.cancel()
