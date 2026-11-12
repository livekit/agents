"""Speechmatics: the Agent STT session over a local socket, and TTS PCM framing over HTTP."""

from __future__ import annotations

import contextlib
import json
import struct
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from livekit.agents import APIConnectOptions, APIError, http, stt
from livekit.plugins.speechmatics import STT, TTS, AdditionalVocabEntry, SpeakerIdentifier
from tests_new.fakes import audio_frame, until

NO_RETRY = APIConnectOptions(max_retry=0)

# `transcription_config` is `additionalProperties: false` in the agent input spec; a literal
# so an SDK that starts emitting a new field is caught here rather than on the wire.
WIRE_WHITELIST = frozenset(
    {
        "additional_vocab",
        "diarization",
        "domain",
        "emit_sentences",
        "enable_partials",
        "language",
        "model",
        "operating_point",
        "output_locale",
        "speaker_diarization_config",
        "transcript_filtering_config",
    }
)


def segment(
    transcript: str, *, final: bool = True, start: float = 1.5, end: float = 2.5
) -> dict[str, object]:
    return {
        "message": "AddSegment" if final else "AddPartialSegment",
        "segment": {"transcript": transcript, "speaker": "S1"},
        "metadata": {"start_time": start, "end_time": end},
    }


class SpeechmaticsPeer:
    """``/v2/agent``: answers StartRecognition, replays ``replies``, and ends on EndOfStream."""

    def __init__(self) -> None:
        # Any: the StartRecognition JSON as received; the SDK has no type for the wire form
        self.start: dict[str, Any] = {}
        self.replies: list[dict[str, object]] = []
        self.at_end_of_stream: list[dict[str, object]] = []
        self.speakers: list[dict[str, object]] = []
        self.socket: web.WebSocketResponse | None = None

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        ws = self.socket = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            if message.type != web.WSMsgType.TEXT:
                continue
            event = json.loads(message.data)
            if event["message"] == "StartRecognition":
                self.start = event
                await ws.send_json({"message": "RecognitionStarted", "id": "session-1"})
                for reply in self.replies:
                    await ws.send_json(reply)
            elif event["message"] == "GetSpeakers":
                await ws.send_json({"message": "SpeakersResult", "speakers": self.speakers})
            elif event["message"] == "EndOfStream":
                for reply in self.at_end_of_stream:
                    await ws.send_json(reply)
                await ws.send_json({"message": "EndOfTranscript"})
                await ws.close()
        return ws


@pytest.fixture
async def peer() -> AsyncIterator[tuple[SpeechmaticsPeer, str]]:
    peer = SpeechmaticsPeer()
    app = web.Application()
    app.router.add_get("/v2/agent", peer.handle)
    async with TestServer(app) as server, http.open():
        yield peer, str(server.make_url("/v2/agent").with_scheme("ws"))


async def transcribe(stream: stt.RecognizeStream, seconds: float = 0) -> list[stt.SpeechEvent]:
    for _ in range(round(seconds * 10)):
        stream.push_frame(audio_frame(0.1))
    stream.end_input()
    events = [event async for event in stream]
    await stream.aclose()
    return events


async def test_session_config_sends_only_whitelisted_fields(
    peer: tuple[SpeechmaticsPeer, str],
) -> None:
    """Options reach StartRecognition, and nothing the agent spec rejects is sent.

    https://github.com/livekit/agents/pull/7028 https://github.com/livekit/agents/issues/3792
    """
    server, url = peer
    model = STT(
        api_key="k",
        base_url=url,
        vad=None,
        language="ja",
        output_locale="ja-JP",
        domain="finance",
        include_partials=True,
        enable_diarization=True,
        max_speakers=3,
        additional_vocab=[AdditionalVocabEntry(content="LiveKit")],
    )

    await transcribe(model.stream(conn_options=NO_RETRY))

    sent = server.start["transcription_config"]
    assert set(sent) <= WIRE_WHITELIST, f"not on the wire whitelist: {set(sent) - WIRE_WHITELIST}"
    assert (sent["language"], sent["output_locale"], sent["domain"]) == ("ja", "ja-JP", "finance")
    assert sent["enable_partials"] is True
    assert sent["diarization"] == "speaker"
    assert sent["speaker_diarization_config"]["max_speakers"] == 3
    assert [entry["content"] for entry in sent["additional_vocab"]] == ["LiveKit"]


async def test_segments_become_transcripts_with_provider_timing(
    peer: tuple[SpeechmaticsPeer, str],
) -> None:
    """Segments carry the stream's language and provider times, and no invented confidence.

    https://github.com/livekit/agents/pull/3176 https://github.com/livekit/agents/issues/5880
    """
    server, url = peer
    server.replies = [
        segment("hello", final=False, start=0.75, end=1.0),
        segment("hello world", start=0.75, end=1.25),
    ]
    stream = STT(api_key="k", base_url=url, vad=None, language="en").stream(
        language="fr", conn_options=NO_RETRY
    )
    stream.start_time_offset = 2.0

    events = await transcribe(stream)

    interim, final = events
    assert final.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert interim.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
    data = final.alternatives[0]
    assert (data.text, data.speaker_id, data.language) == ("hello world", "S1", "fr")
    assert server.start["transcription_config"]["language"] == "fr"
    assert data.start_time >= 2.75  # after the offset, plus the time to the first audio
    assert data.end_time - data.start_time == pytest.approx(0.5)
    assert interim.alternatives[0].start_time == data.start_time
    assert data.confidence == interim.alternatives[0].confidence == 0.0


async def test_turns_become_speech_events_and_usage_bills_the_audio_sent(
    peer: tuple[SpeechmaticsPeer, str],
) -> None:
    """StartOfTurn / EndOfTurn frame speech; usage is the audio sent, reported once.

    https://github.com/livekit/agents/pull/7028
    """
    server, url = peer
    server.at_end_of_stream = [
        {"message": "StartOfTurn"},
        {"message": "EndOfTurn"},
        {"message": "EndOfTurn"},
    ]

    events = await transcribe(STT(api_key="k", base_url=url, vad=None).stream(), seconds=1.0)

    assert [event.type for event in events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.RECOGNITION_USAGE,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    assert events[2].recognition_usage is not None
    assert events[2].recognition_usage.audio_duration == pytest.approx(1.0)


async def test_server_error_fails_the_stream_without_retry(
    peer: tuple[SpeechmaticsPeer, str],
) -> None:
    """An ``Error`` ends the session as a non-retryable error carrying the message.

    https://github.com/livekit/agents/pull/7028
    """
    server, url = peer
    error: dict[str, object] = {"message": "Error", "type": "invalid_config", "reason": "nope"}
    server.replies = [{"message": "Warning", "reason": "slow"}, segment("still here"), error]
    stream = STT(api_key="k", base_url=url, vad=None).stream(conn_options=NO_RETRY)
    received: list[stt.SpeechEvent] = []

    with pytest.raises(APIError, match="invalid_config") as raised:
        try:
            async for event in stream:
                received.append(event)
        finally:
            await stream.aclose()

    assert [event.alternatives[0].text for event in received] == ["still here"]
    assert raised.value.retryable is False
    assert raised.value.body == error


async def test_invalid_speaker_reply_fails_the_request_not_recognition(
    peer: tuple[SpeechmaticsPeer, str],
) -> None:
    """An invalid SpeakersResult fails only ``get_speaker_ids()``; recognition keeps going."""
    server, url = peer
    server.speakers = [{"label": "Agent", "speaker_identifiers": 7}]
    model = STT(api_key="k", base_url=url, vad=None, enable_diarization=True)
    stream = model.stream(conn_options=NO_RETRY)
    stream.push_frame(audio_frame(0.1))
    await until(lambda: server.start, "StartRecognition")

    with pytest.raises(APIError, match="invalid speaker identifiers") as raised:
        await model.get_speaker_ids()
    server.speakers = [{"label": "Agent", "speaker_identifiers": ["voice-1"]}]
    speakers = await model.get_speaker_ids()
    assert server.socket is not None
    await server.socket.send_json(segment("Still listening"))
    event = await anext(stream)
    await stream.aclose()

    assert isinstance(raised.value.__cause__, ValidationError)
    [speaker] = speakers
    assert isinstance(speaker, SpeakerIdentifier)
    assert (speaker.label, speaker.speaker_identifiers) == ("Agent", ["voice-1"])
    assert event.alternatives[0].text == "Still listening"


async def test_tts_incomplete_pcm_sample_is_a_provider_failure() -> None:
    """A response ending mid-sample fails the synthesis instead of playing a torn sample."""

    async def generate(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(struct.pack("<1600h", *range(1600)))
        await response.write(b"\xff")
        return response

    app = web.Application()
    app.router.add_post("/generate/{voice}", generate)
    async with TestServer(app) as server, http.open():
        model = TTS(api_key="k", base_url=str(server.make_url("")).rstrip("/"))

        with pytest.raises(APIError, match="incomplete PCM sample"):
            async with model.synthesize("hello", conn_options=NO_RETRY) as stream:
                async for _ in stream:
                    pass


@pytest.mark.xfail(
    strict=True,
    reason="a server Error after end_input ends `async for` cleanly; only aclose() raises it",
)
async def test_server_error_after_end_of_input_fails_the_iteration(
    peer: tuple[SpeechmaticsPeer, str],
) -> None:
    """An ``Error`` answering the end of input fails ``async for``, not only ``aclose()``."""
    server, url = peer
    server.at_end_of_stream = [{"message": "Error", "type": "invalid_config", "reason": "nope"}]
    stream = STT(api_key="k", base_url=url, vad=None).stream(conn_options=NO_RETRY)
    stream.push_frame(audio_frame(0.1))
    stream.end_input()

    try:
        with pytest.raises(APIError, match="invalid_config"):
            async for _ in stream:
                pass
    finally:
        with contextlib.suppress(APIError):
            await stream.aclose()
