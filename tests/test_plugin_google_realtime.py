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


async def test_tool_choice_never_reaches_the_connect_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No tool_choice value may leak into the LiveConnectConfig sent to the API.

    The Google Realtime API rejects a tool_choice parameter (issue #4770), so whatever the
    user asks for, the connect config must stay free of it.
    """
    choices: list[llm.ToolChoice] = [
        "auto",
        "none",
        "required",
        {"type": "function", "function": {"name": "get_weather"}},
    ]
    async with _make_session(monkeypatch) as session:
        for choice in choices:
            session.update_options(tool_choice=choice)
            payload = session._build_connect_config().model_dump(exclude_none=True)
            assert "tool_choice" not in payload
            assert "tool_config" not in payload


async def test_update_options_tool_choice_none_is_emulated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """tool_choice='none' is kept and emulated by rejecting tool calls, with a warning."""
    async with _make_session(monkeypatch) as session:
        with caplog.at_level(logging.WARNING):
            session.update_options(tool_choice="none")

        assert session._opts.tool_choice == "none"
        assert any("tool_choice='none'" in r.message for r in caplog.records)


async def test_update_options_unsupported_tool_choice_falls_back_to_auto(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Values the API cannot express warn and are normalized to 'auto'.

    The session must store what it actually does - keeping 'required' around while warning
    about an 'auto' fallback leaves the state lying about the behavior (issue #4770).
    """
    async with _make_session(monkeypatch) as session:
        with caplog.at_level(logging.WARNING):
            session.update_options(tool_choice="required")
            session.update_options(
                tool_choice={"type": "function", "function": {"name": "get_weather"}}
            )

        assert session._opts.tool_choice == "auto"
        not_supported = [
            r for r in caplog.records if "not supported by the Google Realtime API" in r.message
        ]
        assert len(not_supported) == 2


async def test_generate_reply_warns_on_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """generate_reply has no per-response tool_choice; it must warn instead of dropping it
    silently, the same way per-response tools is handled (issue #4770)."""
    async with _make_session(monkeypatch) as session:
        with caplog.at_level(logging.WARNING):
            fut = session.generate_reply(tool_choice="none")

        assert any("per-response tool_choice" in r.message for r in caplog.records)

        # don't leave the generation pending until its 5s timeout fires; cancelling fires
        # interrupt(), which is a no-op here because _make_session closed the send channel
        fut.cancel()
