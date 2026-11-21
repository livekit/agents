"""TTS retry logging must keep the synthesized text out of the log body.

A log record's body cannot be redacted — the telemetry pipeline only strips
attributes carrying a ``pii`` segment (see ``telemetry/traces.py``). The TTS retry
path interpolates the exception into the message, and that exception quotes the text
being synthesized (``APIError("no audio frames were pushed for text: ...")``), so the
text used to reach every log sink — including the Cloud log export when the project
enables redaction.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from livekit.agents import APIConnectionError, APIConnectOptions, APIError
from livekit.agents.telemetry import pii

from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

SYNTHESIZED_TEXT = "your card number is 4111 1111 1111 1111"


async def _synthesize_until_error(tts_v: FakeTTS, *, max_retry: int) -> None:
    stream = tts_v.synthesize(
        SYNTHESIZED_TEXT,
        conn_options=APIConnectOptions(max_retry=max_retry, timeout=0.5, retry_interval=0.0),
    )
    with pytest.raises(APIError):
        async for _ in stream:
            pass


async def _stream_until_error(tts_v: FakeTTS, *, max_retry: int) -> None:
    stream = tts_v.stream(
        conn_options=APIConnectOptions(max_retry=max_retry, timeout=0.5, retry_interval=0.0)
    )
    stream.push_text(SYNTHESIZED_TEXT)
    stream.end_input()
    with pytest.raises(APIError):
        async for _ in stream:
            pass


def _assert_body_is_clean(
    caplog: pytest.LogCaptureFixture,
    message: str,
    *,
    streamed: bool,
    has_retry_interval: bool = False,
) -> None:
    retries = [r for r in caplog.records if message in r.getMessage()]
    assert retries, f"expected {message!r} to be logged"

    for record in retries:
        assert SYNTHESIZED_TEXT not in record.getMessage(), (
            "the synthesized text must not be interpolated into the log body, "
            "which no redaction can reach"
        )
        # it is still available to the operator, under a key the PII filter can strip
        assert SYNTHESIZED_TEXT in record.__dict__["lk.pii.error"]
        # so with redaction on, what reaches an exporter carries none of it
        assert SYNTHESIZED_TEXT not in str(pii.filter_attributes(record.__dict__))
        # the class and the retry delay survive redaction, so the failure stays diagnosable
        assert record.__dict__["error_type"] == "APIError"
        assert record.__dict__["streamed"] is streamed
        # the retry delay replaces what the message used to carry; the fallback
        # recovery sites never logged one
        assert ("retry_interval" in record.__dict__) is has_retry_interval
        if has_retry_interval:
            assert record.__dict__["retry_interval"] >= 0.0


async def test_retry_warning_keeps_the_text_out_of_the_log_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A retried synthesis failure must not put the text in the message body."""
    fake_tts = FakeTTS(fake_audio_duration=0.0)  # pushes no audio -> retryable APIError
    try:
        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            await _synthesize_until_error(fake_tts, max_retry=1)
    finally:
        await fake_tts.aclose()

    _assert_body_is_clean(
        caplog, "failed to synthesize speech", streamed=False, has_retry_interval=True
    )


async def test_streaming_retry_warning_keeps_the_text_out_of_the_log_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The streaming retry path has the same message and needs the same treatment."""
    fake_tts = FakeTTS(fake_audio_duration=0.0)
    try:
        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            await _stream_until_error(fake_tts, max_retry=1)
    finally:
        await fake_tts.aclose()

    _assert_body_is_clean(
        caplog, "failed to synthesize speech", streamed=True, has_retry_interval=True
    )


async def test_fallback_recovery_warning_keeps_the_text_out_of_the_log_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fallback adapter re-synthesizes the same text when it probes recovery."""
    from livekit.agents.tts import FallbackAdapter

    fake_tts = FakeTTS(fake_audio_duration=0.0)
    adapter = FallbackAdapter([fake_tts], max_retry_per_tts=1)
    conn_options = APIConnectOptions(max_retry=0, timeout=0.5, retry_interval=0.0)
    try:
        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            with pytest.raises(APIConnectionError):
                async with adapter.synthesize(
                    SYNTHESIZED_TEXT, conn_options=conn_options
                ) as stream:
                    async for _ in stream:
                        pass

            for _ in range(500):
                if any("recovery failed" in r.getMessage() for r in caplog.records):
                    break
                await asyncio.sleep(0.02)
    finally:
        await adapter.aclose()

    _assert_body_is_clean(caplog, "recovery failed", streamed=False)


async def test_streamed_fallback_recovery_warning_keeps_the_text_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The streamed fallback recovery probe is a separate site with its own message."""
    from livekit.agents.tts import FallbackAdapter

    fake_tts = FakeTTS(fake_audio_duration=0.0)
    adapter = FallbackAdapter([fake_tts], max_retry_per_tts=1)
    conn_options = APIConnectOptions(max_retry=0, timeout=0.5, retry_interval=0.0)
    try:
        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            with pytest.raises(APIConnectionError):
                async with adapter.stream(conn_options=conn_options) as stream:
                    stream.push_text(SYNTHESIZED_TEXT)
                    stream.end_input()
                    async for _ in stream:
                        pass

            for _ in range(500):
                if any("recovery failed" in r.getMessage() for r in caplog.records):
                    break
                await asyncio.sleep(0.02)
    finally:
        await adapter.aclose()

    _assert_body_is_clean(caplog, "recovery failed", streamed=True)
