"""Baseten Real-time transcription API v2 (``model="streaming-v2"``): session, turns, and errors.

The server speaks first with ``session.created``; the stream configures itself with
``session.update``, sends base64 audio, flushes the last turn when the input ends,
then sends ``close`` and reads until the server's own close. Finals the server still
owed arrive in between and must reach the caller.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
# drop_second_turn: the first connection answers its first flush, then drops on the next.
# fail_then_drop: the first connection fails the turn on its first flush, then drops.
# partial_then_drop: the first connection sends a partial on its first flush, then drops.
# drop_on_audio: the first connection drops on its first audio, before any flush.
# A flushed turn with no audio behind it gets an empty final, as from the real server.
_Mode = Literal[
    "reply",
    "failed",
    "split",
    "empty",
    "at_capacity",
    "invalid_event",
    "drop",
    "drop_once",
    "drop_second_turn",
    "fail_then_drop",
    "partial_then_drop",
    "drop_on_audio",
]

_CLOSE_CODES = {"at_capacity": 1013, "invalid_event": 1008}


def _done(turn_id: int, texts: list[str], **extra: Any) -> dict[str, Any]:
    segments = [{"text": t, "start": 0.5 + i, "end": 1.5 + i} for i, t in enumerate(texts)]
    return {
        "type": "transcription.done",
        "transcript": {"turn_id": turn_id, "segments": segments, "language": "en"},
        **extra,
    }


class _FakeStreamingV2Server:
    """Speaks the Baseten v2 transcription protocol: session events, base64 audio, flush, close."""

    def __init__(
        self,
        mode: _Mode = "reply",
        refusals: list[dict[str, Any]] | None = None,
        max_turn_s: float | None = None,
    ):
        self.mode = mode
        self.refusals = refusals or []
        self.max_turn_s = max_turn_s
        self.session_updates: list[dict[str, Any]] = []
        self.audio_bytes = 0
        self.audio_per_connection: list[int] = []
        self.control: list[str] = []
        self.connections = 0

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json(
            {
                "type": "session.created",
                "session_id": "sess_test",
                "capabilities": {"max_turn_s": self.max_turn_s},
            }
        )
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
                if self.mode == "drop_on_audio" and self.connections == 1:
                    await ws.close(code=1001)
                    break
                continue
            self.control.append(kind)
            if kind == "close":
                await ws.close()
                break
            if (
                self.mode == "drop"
                or (self.mode == "drop_once" and self.connections == 1)
                or (self.mode == "drop_second_turn" and self.connections == 1 and turn_id == 1)
            ):
                await ws.close(code=1001)
                break
            if self.mode == "fail_then_drop" and self.connections == 1:
                # the client cannot tell whether the cap or the flush ended this turn
                await ws.send_json(
                    {
                        "type": "transcription.failed",
                        "turn_id": turn_id,
                        "code": "decode_failed",
                        "message": "boom",
                    }
                )
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
            if self.mode == "partial_then_drop" and self.connections == 1:
                await ws.close(code=1001)
                break
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


@asynccontextmanager
async def _serve(server: _FakeStreamingV2Server, **stt_kwargs: Any) -> AsyncIterator[STT]:
    """Run `server` locally and yield a v2 `STT` pointed at it."""
    app = web.Application()
    app.router.add_get("/ws", server.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]

    try:
        async with aiohttp.ClientSession() as session:
            yield STT(
                model="streaming-v2",
                api_key="test-key",
                model_endpoint=f"ws://127.0.0.1:{port}/ws",
                http_session=session,
                **stt_kwargs,
            )
    finally:
        await runner.cleanup()


def _silence(samples: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * samples,
        sample_rate=_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=samples,
    )


async def _recognize(
    server: _FakeStreamingV2Server, samples: int = 512, max_retry: int = 0, **stt_kwargs: Any
) -> stt.SpeechEvent:
    async with _serve(server, **stt_kwargs) as baseten_stt:
        return await asyncio.wait_for(
            baseten_stt.recognize(
                [_silence(samples)],
                conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.01, timeout=5),
            ),
            timeout=15,
        )


async def _run_stream(
    server: _FakeStreamingV2Server,
    samples: int | list[int] = 512,
    max_retry: int = 0,
    start_time_offset: float = 0.0,
    **stt_kwargs: Any,
) -> list[stt.SpeechEvent]:
    """Stream `samples` of silence, flushing between pieces when given a list."""
    async with _serve(server, **stt_kwargs) as baseten_stt:
        stream = baseten_stt.stream(
            conn_options=APIConnectOptions(max_retry=max_retry, retry_interval=0.01, timeout=5)
        )
        # as `Agent.stt_node` does, before the stream first connects
        stream.start_time_offset = start_time_offset
        pieces = samples if isinstance(samples, list) else [samples]
        for i, n in enumerate(pieces):
            if i:
                stream.flush()
            stream.push_frame(_silence(n))
        stream.end_input()

        async def collect() -> list[stt.SpeechEvent]:
            return [ev async for ev in stream]

        try:
            return await asyncio.wait_for(collect(), timeout=15)
        finally:
            await stream.aclose()


def _types(events: list[stt.SpeechEvent]) -> list[stt.SpeechEventType]:
    return [ev.type for ev in events]


def _final_texts(events: list[stt.SpeechEvent]) -> list[str]:
    """The finals that carry words; empty ones only answer a flush."""
    return [
        ev.alternatives[0].text
        for ev in events
        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT and ev.alternatives[0].text
    ]


async def test_default_session_update_leaves_model_defaults_to_the_deployment() -> None:
    server = _FakeStreamingV2Server()
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
    server = _FakeStreamingV2Server()
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


async def test_client_flush_sends_no_vad_and_asks_livekit_to_flush(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = _FakeStreamingV2Server()
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
    # dropped, but not quietly
    assert any("vad_threshold" in r.getMessage() for r in caplog.records)
    assert STT(
        model="streaming-v2", api_key="k", model_id="m", segmentation_mode="client_flush"
    ).capabilities.manual_flush
    assert not STT(model="streaming-v2", api_key="k", model_id="m").capabilities.manual_flush


async def test_end_input_flushes_once_then_closes_and_delivers_the_final() -> None:
    server = _FakeStreamingV2Server()
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
    # the server sends no confidence, so none is made up
    assert final.confidence == 0.0
    assert final.start_time == pytest.approx(0.5, abs=0.2)
    assert final.end_time == pytest.approx(2.5, abs=0.2)


async def test_failed_turn_closes_the_turn_and_keeps_the_session() -> None:
    server = _FakeStreamingV2Server("failed")
    events = await _run_stream(server)

    assert server.control == ["input_audio_buffer.flush", "close"]
    assert _final_texts(events) == []
    # an empty final, so a flush waiting on the turn is answered
    assert _types(events) == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_turn_cut_at_the_length_limit_does_not_end_speech() -> None:
    server = _FakeStreamingV2Server("split")
    events = await _run_stream(server)

    assert _final_texts(events) == ["hello", "again"]
    assert _types(events).count(stt.SpeechEventType.START_OF_SPEECH) == 1
    assert _types(events)[-1] == stt.SpeechEventType.END_OF_SPEECH
    assert _types(events).count(stt.SpeechEventType.END_OF_SPEECH) == 1


async def test_empty_final_answers_the_flush_without_opening_a_turn() -> None:
    server = _FakeStreamingV2Server("empty")
    events = await _run_stream(server, enable_partial_transcripts=False)

    # the framework waits on a final after its flush, but a turn with no words
    # must not start or end speech (under turn_detection="stt" that would
    # commit a turn for the agent to answer)
    assert _types(events) == [stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert events[0].alternatives[0].text == ""


async def test_partials_off_drops_partials_the_server_sends_anyway() -> None:
    server = _FakeStreamingV2Server()
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
    server = _FakeStreamingV2Server(refusals=[refusal])
    events = await _run_stream(server, partial_transcript_interval_s=0.1)

    assert _final_texts(events) == ["hello world"]
    assert any("partial_interval_s" in r.getMessage() for r in caplog.records)


async def test_at_capacity_is_retried_on_a_new_connection() -> None:
    server = _FakeStreamingV2Server("at_capacity")
    with pytest.raises(APIConnectionError):
        await _run_stream(server, max_retry=1)
    assert server.connections == 2


async def test_invalid_event_is_not_retried() -> None:
    # a request the server rejects once is rejected every time
    server = _FakeStreamingV2Server("invalid_event")
    with pytest.raises(APIStatusError) as exc_info:
        await _run_stream(server, max_retry=3)
    assert exc_info.value.status_code == 400
    assert server.connections == 1


async def test_close_that_skips_an_owed_final_is_a_connection_error() -> None:
    # the input has ended by then, but only the server's 1000 answer to `close`
    # says every final was delivered
    server = _FakeStreamingV2Server("drop")
    with pytest.raises(APIConnectionError):
        await _run_stream(server)
    assert server.connections == 1


async def test_recognize_resends_the_whole_buffer_after_a_drop() -> None:
    # the first socket takes all the audio and then drops; a stream reconnecting
    # on its own would send the second one nothing and return an empty success
    server = _FakeStreamingV2Server("drop_once")
    event = await _recognize(server, samples=2400, max_retry=1)

    assert server.connections == 2
    assert server.audio_per_connection == [2400 * 2, 2400 * 2]
    assert event.alternatives[0].text == "hello world"


async def test_recognize_reports_a_failed_turn_as_an_error() -> None:
    # not as a successful, empty transcript; and retried like any server fault
    server = _FakeStreamingV2Server("failed")
    with pytest.raises(APIStatusError) as exc_info:
        await _recognize(server)
    assert exc_info.value.status_code == 500

    server = _FakeStreamingV2Server("failed")
    with pytest.raises(APIConnectionError):
        await _recognize(server, max_retry=1)
    assert server.connections == 2


async def test_stream_reconnect_replays_the_unfinished_turn() -> None:
    # the first socket takes the whole turn and drops before its final
    server = _FakeStreamingV2Server("drop_once")
    events = await _run_stream(server, samples=2400, max_retry=1)

    assert server.audio_per_connection == [2400 * 2, 2400 * 2]
    assert server.control == [
        "input_audio_buffer.flush",
        "input_audio_buffer.flush",
        "close",
    ]
    assert _final_texts(events) == ["hello world"]


async def test_stream_reconnect_under_client_flush_skips_finished_turns() -> None:
    # turn 1 (3 s) is final before the drop, so only turn 2 (1 s) is sent again
    server = _FakeStreamingV2Server("drop_second_turn")
    events = await _run_stream(
        server, samples=[48000, 16000], max_retry=1, segmentation_mode="client_flush"
    )

    assert server.audio_per_connection == [64000 * 2, 16000 * 2]
    assert _final_texts(events) == ["hello world", "hello world"]
    # the replayed turn is placed after turn 1 on the stream's timeline
    finals = [ev for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals[1].alternatives[0].start_time == pytest.approx(3.5, abs=0.2)


async def test_stream_reconnect_under_server_vad_replays_from_the_last_word() -> None:
    # turn 1's last word ends at 2.5 s of its 3 s: the 0.5 s after it, and all
    # of turn 2 (1 s), are sent again
    server = _FakeStreamingV2Server("drop_second_turn")
    await _run_stream(server, samples=[48000, 16000], max_retry=1)

    assert server.audio_per_connection == [64000 * 2, (8000 + 16000) * 2]


async def test_failed_turn_that_may_be_capped_keeps_later_audio_for_replay() -> None:
    # 3 s before the flush under a 2 s cap: the failure may be the capped first
    # turn, with the rest still unanswered when the socket drops, so all of it
    # is sent again
    server = _FakeStreamingV2Server("fail_then_drop", max_turn_s=2.0)
    events = await _run_stream(server, samples=48000, max_retry=1, segmentation_mode="client_flush")

    assert server.audio_per_connection == [48000 * 2, 48000 * 2]
    assert _final_texts(events) == ["hello world"]


async def test_failed_turn_shorter_than_the_cap_is_not_replayed() -> None:
    # 1 s before the flush under a 2 s cap: the flush ended the failed turn, so
    # nothing of it is sent again
    server = _FakeStreamingV2Server("fail_then_drop", max_turn_s=2.0)
    events = await _run_stream(server, samples=16000, max_retry=1, segmentation_mode="client_flush")

    assert server.audio_per_connection == [16000 * 2, 0]
    assert _final_texts(events) == []


async def test_transcript_times_are_placed_in_the_session() -> None:
    # Agent.stt_node starts this stream 40 s into the session. The framework
    # adds the time to reconnect (at least 0.1 s) to the offset, but the
    # replayed audio's own position already places the second turn, exactly
    server = _FakeStreamingV2Server("drop_second_turn")
    events = await _run_stream(
        server,
        samples=[48000, 16000],
        max_retry=1,
        start_time_offset=40.0,
        segmentation_mode="client_flush",
    )

    finals = [
        ev.alternatives[0] for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert [f.start_time for f in finals[:2]] == [
        pytest.approx(40.5, abs=0.01),
        pytest.approx(43.5, abs=0.01),
    ]


async def test_reconnect_mid_turn_does_not_start_speech_again() -> None:
    # the turn opened on the first socket is still open on the event channel;
    # its replay continues it
    server = _FakeStreamingV2Server("partial_then_drop")
    events = await _run_stream(server, max_retry=1)

    assert _types(events) == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.INTERIM_TRANSCRIPT,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_audio_held_by_the_chunker_survives_a_reconnect() -> None:
    # 2400 samples go in: one 1600-sample chunk is sent, and the socket drops on
    # it with the other 800 still waiting in the chunker for the next frame
    server = _FakeStreamingV2Server("drop_on_audio")
    async with _serve(server) as baseten_stt:
        stream = baseten_stt.stream(
            conn_options=APIConnectOptions(max_retry=1, retry_interval=0.01, timeout=5)
        )
        stream.push_frame(_silence(2400))

        async def reconnected() -> None:
            while server.connections < 2:
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(reconnected(), timeout=5)
            stream.push_frame(_silence(800))
            stream.end_input()
            await asyncio.wait_for(_drain(stream), timeout=15)
        finally:
            await stream.aclose()

    # the replayed chunk, then the held 800 with the new 800
    assert server.audio_per_connection == [1600 * 2, (1600 + 1600) * 2]


async def _drain(stream: stt.SpeechStream) -> None:
    async for _ in stream:
        pass


def test_mulaw_is_rejected() -> None:
    with pytest.raises(ValueError):
        STT(model="streaming-v2", api_key="k", model_id="m", encoding="pcm_mulaw")
