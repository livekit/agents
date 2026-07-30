from __future__ import annotations

import logging

import pytest
from google.genai import types

from livekit.agents import utils
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit

# 10ms of silence at the output sample rate (24kHz mono, 16-bit)
_PCM_FRAME = b"\x00\x01" * 240


async def _make_session(monkeypatch: pytest.MonkeyPatch) -> RealtimeSession:
    """A session whose background connect loop is stopped before it hits the network."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    session = RealtimeModel().session()
    # cancel the connect loop before the event loop ever schedules it, so no
    # websocket connection is attempted
    session._msg_ch.close()
    await utils.aio.cancel_and_wait(session._main_atask)
    return session


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
