"""Baseten unified STT (``model="unified"``): session, turns, and errors.

The server speaks first with ``session.created``; the stream configures itself with
``session.update``, sends base64 audio, flushes the last turn when the input ends,
then sends ``close`` and reads until the server's own close. Finals the server still
owed arrive in between and must reach the caller.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Literal

import aiohttp
import pytest
from aiohttp import web

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, stt
from livekit.plugins.baseten import STT

pytestmark = pytest.mark.unit

_SAMPLE_RATE = 16000

# reply: a partial and a final per flush. failed: transcription.failed instead.
# split: the turn is cut at the length limit, then finished by the flush.
# empty: a final with no text. at_capacity / invalid_event: an error, then a close.
# drop: go away (1001, a replica shutting down) on the flush, before its final.
# drop_once: drop the first connection that way, then reply on the next.
# A flushed turn with no audio behind it gets an empty final, as from the real server.
_Mode = Literal[
    "reply", "failed", "split", "empty", "at_capacity", "invalid_event", "drop", "drop_once"
]

_CLOSE_CODES = {"at_capacity": 1013, "invalid_event": 1008}


def _done(turn_id: int, texts: list[str], **extra: Any) -> dict[str, Any]:
    segments = [{"text": t, "start": 0.5 + i, "end": 1.5 + i} for i, t in enumerate(texts)]
    return {
        "type": "transcription.done",
        "transcript": {"turn_id": turn_id, "segments": segments, "language": "en"},
        **extra,
    }


class _FakeUnifiedServer:
    """Speaks the Baseten unified STT protocol: session events, base64 audio, flush, close."""

    def __init__(self, mode: _Mode = "reply", refusals: list[dict[str, Any]] | None = None):
        self.mode = mode
        self.refusals = refusals or []
        self.session_updates: list[dict[str, Any]] = []
        self.audio_bytes = 0
        self.audio_per_connection: list[int] = []
        self.control: list[str] = []
        self.connections = 0

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "session_id": "sess_test"})
        if self.mode in _CLOSE_CODES:
            await ws.send_json({"type": "error", "code": self.mode, "message": "refused"})
            await ws.close(code=_CLOSE_CODES[self.mode])
            return ws
        self.audio_per_connection.append(0)
        turn_id = 0
        turn_audio = 0
        async for msg in ws:
            event = json.loads(msg.data)
            kind = event["type"]
            if kind == "session.update":
                self.session_updates.append(event["session"])
                await ws.send_json(
                    {
                        "type": "session.updated",
                        "session_id": "sess_test",
                        "refusals": self.refusals,
                    }
                )
                continue
            if kind == "input_audio_buffer.append":
                n = len(base64.b64decode(event["audio"]))
                self.audio_bytes += n
                self.audio_per_connection[-1] += n
                turn_audio += n
                continue
            self.control.append(kind)
            if kind == "close":
                await ws.close()
                break
            if self.mode == "drop" or (self.mode == "drop_once" and self.connections == 1):
                await ws.close(code=1001)
                break
            # input_audio_buffer.flush
            await ws.send_json(
                {
                    "type": "transcription.delta",
                    "transcript": {
                        "turn_id": turn_id,
                        "segments": [{"text": "hel", "start": 0.5, "end": 1.0}],
                    },
                }
            )
            if self.mode == "failed":
                await ws.send_json(
                    {
                        "type": "transcription.failed",
                        "turn_id": turn_id,
                        "code": "decode_failed",
                        "message": "boom",
                    }
                )
            elif self.mode == "empty" or not turn_audio:
                await ws.send_json(_done(turn_id, [""]))
            elif self.mode == "split":
                await ws.send_json(_done(turn_id, ["hello"], ended_by="max_turn_length"))
                turn_id += 1
                await ws.send_json(_done(turn_id, ["again"]))
            else:
                await ws.send_json(_done(turn_id, ["hello", " world"]))
            turn_id += 1
            turn_audio = 0
        return ws


async def _recognize(
    server: _FakeUnifiedServer, samples: int = 512, max_retry: int = 0, **stt_kwargs: Any
) -> stt.SpeechEvent:
    app = web.Application()
    app.router.add_get("/ws", server.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    try:
        async with aiohttp.ClientSession() as session:
            baseten_stt = STT(
                model="unified",
                api_key="test-key",
                model_endpoint=f"ws://127.0.0.1:{port}/ws",
                http_session=session,
                **stt_kwargs,
            )
            frame = rtc.AudioFrame(
                data=b"\x00\x00" * samples,
                sample_rate=_SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=samples,
            )
            return await asyncio.wait_for(
                baseten_stt.recognize(
                    [frame],
                    conn_options=APIConnectOptions(
                        max_retry=max_retry, retry_interval=0.01, timeout=5
                    ),
                ),
                timeout=15,
            )
    finally:
        await runner.cleanup()


async def _run_stream(
    server: _FakeUnifiedServer, samples: int = 512, max_retry: int = 0, **stt_kwargs: Any
) -> list[stt.SpeechEvent]:
    app = web.Application()
    app.router.add_get("/ws", server.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    try:
        async with aiohttp.ClientSession() as session:
            baseten_stt = STT(
                model="unified",
                api_key="test-key",
                model_endpoint=f"ws://127.0.0.1:{port}/ws",
                http_session=session,
                **stt_kwargs,
            )
            stream = baseten_stt.stream(
                conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.01, timeout=5)
            )
            stream.push_frame(
                rtc.AudioFrame(
                    data=b"\x00\x00" * samples,
                    sample_rate=_SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=samples,
                )
            )
            stream.end_input()

            async def collect() -> list[stt.SpeechEvent]:
                return [ev async for ev in stream]

            try:
                return await asyncio.wait_for(collect(), timeout=15)
            finally:
                await stream.aclose()
    finally:
        await runner.cleanup()


def _types(events: list[stt.SpeechEvent]) -> list[stt.SpeechEventType]:
    return [ev.type for ev in events]


def _final_texts(events: list[stt.SpeechEvent]) -> list[str]:
    return [
        ev.alternatives[0].text for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]


async def test_default_session_update_leaves_model_defaults_to_the_deployment() -> None:
    server = _FakeUnifiedServer()
    await _run_stream(server)

    # no language and no vad: the deployment's per-model defaults apply
    assert server.session_updates == [
        {
            "input_format": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
            "segmentation_mode": "server_vad",
            "partial_interval_s": 0.5,
        }
    ]


async def test_options_reach_the_session_update() -> None:
    server = _FakeUnifiedServer()
    await _run_stream(
        server,
        language="en-US",
        vad_threshold=0.6,
        vad_min_silence_duration_ms=400,
        partial_transcript_interval_s=1.0,
    )

    (update,) = server.session_updates
    assert update["language"] == "en-US"
    assert update["partial_interval_s"] == 1.0
    assert update["vad"] == {"threshold": 0.6, "silence_s": 0.4}


async def test_client_flush_sends_no_vad_and_asks_livekit_to_flush() -> None:
    server = _FakeUnifiedServer()
    await _run_stream(
        server,
        segmentation_mode="client_flush",
        vad_threshold=0.6,
        enable_partial_transcripts=False,
    )

    # the server refuses vad under client_flush, where no silence ends a turn
    assert server.session_updates == [
        {
            "input_format": {"encoding": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
            "segmentation_mode": "client_flush",
        }
    ]
    assert STT(
        model="unified", api_key="k", model_id="m", segmentation_mode="client_flush"
    ).capabilities.manual_flush
    assert not STT(model="unified", api_key="k", model_id="m").capabilities.manual_flush


async def test_end_input_flushes_once_then_closes_and_delivers_the_final() -> None:
    server = _FakeUnifiedServer()
    # 1.5 chunks of 100 ms: the trailing 800 samples only go out when the input ends
    events = await _run_stream(server, samples=2400)

    assert server.audio_bytes == 2400 * 2
    assert server.control == ["input_audio_buffer.flush", "close"]
    assert _types(events) == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    final = events[2].alternatives[0]
    # segments carry their own separators, so they join with ""
    assert final.text == "hello world"
    assert final.language == "en"
    assert final.start_time == pytest.approx(0.5, abs=0.2)
    assert final.end_time == pytest.approx(2.5, abs=0.2)


async def test_failed_turn_closes_the_turn_and_keeps_the_session() -> None:
    server = _FakeUnifiedServer("failed")
    events = await _run_stream(server)

    assert server.control == ["input_audio_buffer.flush", "close"]
    assert _final_texts(events) == []
    assert _types(events) == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_turn_cut_at_the_length_limit_does_not_end_speech() -> None:
    server = _FakeUnifiedServer("split")
    events = await _run_stream(server)

    assert _final_texts(events) == ["hello", "again"]
    assert _types(events).count(stt.SpeechEventType.START_OF_SPEECH) == 1
    assert _types(events)[-1] == stt.SpeechEventType.END_OF_SPEECH
    assert _types(events).count(stt.SpeechEventType.END_OF_SPEECH) == 1


async def test_empty_final_does_not_end_a_turn_it_never_opened() -> None:
    server = _FakeUnifiedServer("empty")
    events = await _run_stream(server, enable_partial_transcripts=False)

    # the fake still sends its partial; only the final is empty
    assert _final_texts(events) == []
    assert stt.SpeechEventType.FINAL_TRANSCRIPT not in _types(events)


async def test_partials_off_drops_partials_the_server_sends_anyway() -> None:
    server = _FakeUnifiedServer()
    events = await _run_stream(server, enable_partial_transcripts=False)

    # the fake sends a partial whatever was asked for
    assert "partial_interval_s" not in server.session_updates[0]
    assert _types(events) == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_refusals_are_logged_and_the_session_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    refusal = {
        "parameter": "partial_interval_s",
        "requested": 0.1,
        "granted": 0.3,
        "reason": "below the backend's floor",
    }
    server = _FakeUnifiedServer(refusals=[refusal])
    events = await _run_stream(server, partial_transcript_interval_s=0.1)

    assert _final_texts(events) == ["hello world"]
    assert any("partial_interval_s" in r.getMessage() for r in caplog.records)


async def test_at_capacity_is_retried_on_a_new_connection() -> None:
    server = _FakeUnifiedServer("at_capacity")
    with pytest.raises(APIConnectionError):
        await _run_stream(server, max_retry=1)
    assert server.connections == 2


async def test_invalid_event_is_not_retried() -> None:
    # a request the server rejects once is rejected every time
    server = _FakeUnifiedServer("invalid_event")
    with pytest.raises(APIStatusError) as exc_info:
        await _run_stream(server, max_retry=3)
    assert exc_info.value.status_code == 400
    assert server.connections == 1


async def test_close_that_skips_an_owed_final_is_a_connection_error() -> None:
    # the input has ended by then, but only the server's 1000 answer to `close`
    # says every final was delivered
    server = _FakeUnifiedServer("drop")
    with pytest.raises(APIConnectionError):
        await _run_stream(server)
    assert server.connections == 1


async def test_recognize_resends_the_whole_buffer_after_a_drop() -> None:
    # the first socket takes all the audio and then drops; a stream reconnecting
    # on its own would send the second one nothing and return an empty success
    server = _FakeUnifiedServer("drop_once")
    event = await _recognize(server, samples=2400, max_retry=1)

    assert server.connections == 2
    assert server.audio_per_connection == [2400 * 2, 2400 * 2]
    assert event.alternatives[0].text == "hello world"


def test_mulaw_is_rejected() -> None:
    with pytest.raises(ValueError):
        STT(model="unified", api_key="k", model_id="m", encoding="pcm_mulaw")
