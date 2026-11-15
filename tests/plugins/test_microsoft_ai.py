"""Microsoft AI plugin: realtime STT and SSML TTS wire contracts, against a local peer.

https://github.com/livekit/agents/pull/7378
"""

from __future__ import annotations

import base64
import io
import os
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import CIMultiDict

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, http, stt
from livekit.plugins import microsoft_ai
from tests.fakes import FakeVAD, until

OPTIONS = APIConnectOptions(max_retry=0, connect_timeout=0.5, read_timeout=0.5)
PCM = b"\x81\xff" * 3717
VOICE = "en-US-Dummy:test-synthesizer"
SSML = "{http://www.w3.org/2001/10/synthesis}"


# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


@dataclass
class MicrosoftAI:
    """The local Microsoft AI endpoints: their switches and what they received."""

    stt_url: str = ""
    tts_url: str = ""
    auto_update: bool = True
    auto_commit: bool = True
    drop_after_audio: bool = False
    connections: list[CIMultiDict[str]] = field(default_factory=list)
    socket: web.WebSocketResponse | None = None
    sent: list[Wire] = field(default_factory=list)
    commits: list[bytes] = field(default_factory=list)
    tts_status: int = 200
    tts_body: bytes = b""
    tts_requests: list[tuple[str, CIMultiDict[str], bytes]] = field(default_factory=list)


@pytest.fixture
async def microsoft(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[MicrosoftAI]:
    """A local Microsoft AI: a realtime transcription socket and an SSML synthesis endpoint.

    The socket acknowledges ``session.update`` when ``auto_update`` and answers each commit with
    a ``turn N`` transcript when ``auto_commit``; ``drop_after_audio`` closes it with a server error at the first
    audio. ``sent`` lists the client's events and ``commits`` the audio of each commit.
    Synthesis answers with ``tts_status`` and ``tts_body``; ``tts_requests`` keeps each request.
    """
    peer = MicrosoftAI()

    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        peer.connections.append(request.headers.copy())
        socket = peer.socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_json({"type": "session.created", "session": {"id": "test-session"}})
        pending = bytearray()
        async for message in socket:
            event = message.json()
            peer.sent.append(event)
            if event["type"] == "session.update" and peer.auto_update:
                await socket.send_json({"type": "session.updated"})
            elif event["type"] == "input_audio_buffer.append":
                pending.extend(base64.b64decode(event["audio"], validate=True))
                if peer.drop_after_audio:
                    await socket.close(code=1011)
            elif event["type"] == "input_audio_buffer.commit":
                peer.commits.append(bytes(pending))
                pending.clear()
                if peer.auto_commit:
                    await _complete(
                        socket, f"item-{len(peer.commits)}", f"turn {len(peer.commits)}"
                    )
        return socket

    async def synthesize(request: web.Request) -> web.Response:
        peer.tts_requests.append(
            (request.query_string, request.headers.copy(), await request.read())
        )
        return web.Response(status=peer.tts_status, body=peer.tts_body, content_type="audio/wav")

    for name in list(os.environ):
        if name.startswith("MICROSOFT_AI_"):
            monkeypatch.delenv(name)
    app = web.Application()
    app.router.add_get("/v1/realtime", transcribe)
    app.router.add_post("/cognitiveservices/v1", synthesize)
    async with TestServer(app) as server, http.open():
        base = server.make_url("")
        peer.stt_url = str(
            base.with_scheme("ws")
            .with_path("/v1/realtime")
            .with_query(intent="transcription", deployment="dummy")
        )
        peer.tts_url = str(base.with_path("/cognitiveservices/v1").with_query(deployment="dummy"))
        yield peer


async def _transcript(socket: web.WebSocketResponse, kind: str, **fields: object) -> None:
    event_type = f"conversation.item.input_audio_transcription.{kind}"
    await socket.send_json({"type": event_type, "item_id": "item-1", **fields})


async def _complete(socket: web.WebSocketResponse, item: str, text: str) -> None:
    await socket.send_json({"type": "input_audio_buffer.committed", "item_id": item})
    for kind, fields in (("delta", {"delta": text}), ("completed", {"transcript": text})):
        event_type = f"conversation.item.input_audio_transcription.{kind}"
        await socket.send_json({"type": event_type, "item_id": item, **fields})


def _audio(samples: int = 800) -> rtc.AudioFrame:
    return rtc.AudioFrame(b"\x81\xff" * samples, 16000, 1, samples)


def _wav(pcm: bytes) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(pcm)
    return output.getvalue()


def _stt(peer: MicrosoftAI) -> microsoft_ai.STT:
    return microsoft_ai.STT(
        vad=None, url=peer.stt_url, model="test-transcriber", api_key="dummy-stt-key"
    )


def _tts(peer: MicrosoftAI) -> microsoft_ai.TTS:
    return microsoft_ai.TTS(
        url=peer.tts_url,
        model="test-synthesizer",
        voice=VOICE,
        sample_rate=24000,
        api_key="dummy-tts-key",
    )


def _finals(events: list[stt.SpeechEvent]) -> list[str]:
    final = stt.SpeechEventType.FINAL_TRANSCRIPT
    return [event.alternatives[0].text for event in events if event.type == final]


def _turn(events: list[stt.SpeechEvent]) -> list[stt.SpeechEvent]:
    """The recognition's events, without the usage the framework reports."""
    return [event for event in events if event.type != stt.SpeechEventType.RECOGNITION_USAGE]


def _sent(peer: MicrosoftAI, kind: str) -> int:
    return sum(event["type"] == kind for event in peer.sent)


async def test_handshake_gates_audio_and_sends_exact_config(microsoft: MicrosoftAI) -> None:
    """Audio waits for session.updated; the session.update carries the exact config."""
    microsoft.auto_update = False
    instance = _stt(microsoft)
    async with instance, instance.stream(language="en", conn_options=OPTIONS) as stream:
        stream.push_frame(_audio())
        stream.end_input()
        await until(lambda: microsoft.sent, "the session.update arrives")
        before_ack = list(microsoft.sent)

        assert microsoft.socket is not None
        await microsoft.socket.send_json({"type": "session.updated"})
        events = [event async for event in stream]

    assert before_ack == [
        {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 16000},
                        "transcription": {"model": "test-transcriber", "language": "en"},
                        "turn_detection": None,
                        "noise_reduction": None,
                    }
                },
            },
        }
    ]
    assert _finals(events) == ["turn 1"]
    [headers] = microsoft.connections
    assert headers["Authorization"] == "Bearer dummy-stt-key"


async def test_revisions_deltas_and_final_are_not_duplicated(microsoft: MicrosoftAI) -> None:
    """Revisions replace the pending tail, repeated deltas and finals count once."""
    microsoft.auto_commit = False
    instance = _stt(microsoft)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(_audio())
        await until(lambda: _sent(microsoft, "input_audio_buffer.append"), "audio is appended")
        socket = microsoft.socket
        assert socket is not None
        texts: list[str] = []

        async def say(kind: str, **fields: object) -> None:
            await _transcript(socket, kind, **fields)
            event = await anext(stream)
            if event.type == stt.SpeechEventType.START_OF_SPEECH:
                event = await anext(stream)
            texts.append(event.alternatives[0].text)

        await say("intermediate", intermediate="helo", event_id="revision-1")
        await say("intermediate", intermediate="hello")
        await say("delta", delta="hello ", event_id="delta-1")
        await _transcript(socket, "delta", delta="hello ", event_id="delta-1")
        await say("intermediate", intermediate="wurld")
        await say("intermediate", intermediate="world")
        await _transcript(socket, "delta", delta="world")
        stream.end_input()
        await until(lambda: microsoft.commits, "the turn is committed")
        await socket.send_json({"type": "input_audio_buffer.committed", "item_id": "item-1"})
        await _transcript(socket, "completed", transcript="hello world")
        await _transcript(socket, "completed", transcript="hello world")
        events = [event async for event in stream]

    assert texts == ["helo", "hello", "hello ", "hello wurld", "hello world"]
    assert _finals(events) == ["hello world"]
    assert not any(event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT for event in events)
    assert _turn(events)[-1].type == stt.SpeechEventType.END_OF_SPEECH


async def test_empty_completion_retracts_the_interim(microsoft: MicrosoftAI) -> None:
    """A completion with no text after an interim withdraws the interim and ends the turn."""
    microsoft.auto_commit = False
    instance = _stt(microsoft)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(_audio())
        await until(lambda: _sent(microsoft, "input_audio_buffer.append"), "audio is appended")
        socket = microsoft.socket
        assert socket is not None
        await _transcript(socket, "intermediate", intermediate="uh")
        stream.end_input()
        await until(lambda: microsoft.commits, "the turn is committed")
        await socket.send_json({"type": "input_audio_buffer.committed", "item_id": "item-1"})
        await _transcript(socket, "completed", transcript="")
        events = [event async for event in stream]

    turn = _turn(events)
    assert _finals(events) == []
    assert turn[-2].type == stt.SpeechEventType.INTERIM_TRANSCRIPT
    assert turn[-2].alternatives[0].text == ""
    assert turn[-1].type == stt.SpeechEventType.END_OF_SPEECH


async def test_a_transcript_sent_before_a_close_after_the_audio_ended_is_delivered(
    microsoft: MicrosoftAI,
) -> None:
    """The provider transcribes after the commit: a close that follows the end of the audio
    still lets the transcript it sent first through."""
    microsoft.auto_commit = False
    instance = _stt(microsoft)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(_audio())
        stream.end_input()
        await until(lambda: microsoft.commits, "the turn is committed")
        socket = microsoft.socket
        assert socket is not None
        await _complete(socket, "item-1", "goodbye")
        await socket.close()
        events = [event async for event in stream]

    assert _finals(events) == ["goodbye"]
    assert _turn(events)[-1].type == stt.SpeechEventType.END_OF_SPEECH


async def test_a_close_before_the_transcript_of_a_commit_is_not_a_clean_end(
    microsoft: MicrosoftAI,
) -> None:
    """A commit owes its transcript: a close after the end of the audio that leaves it unsent
    fails the stream instead of ending it as if nothing was said."""
    microsoft.auto_commit = False
    instance = _stt(microsoft)
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        stream.push_frame(_audio())
        stream.end_input()
        await until(lambda: microsoft.commits, "the turn is committed")
        socket = microsoft.socket
        assert socket is not None
        await socket.close()
        with pytest.raises(APIConnectionError):
            async for _ in stream:
                pass


async def test_disconnect_after_audio_reconnects(microsoft: MicrosoftAI) -> None:
    """A socket lost after audio reconnects, as every streaming STT does."""
    microsoft.drop_after_audio = True
    instance = _stt(microsoft)
    options = APIConnectOptions(max_retry=3, connect_timeout=0.2, read_timeout=0.2)

    async with instance, instance.stream(conn_options=options) as stream:
        stream.push_frame(_audio(817))
        stream.end_input()
        async for _ in stream:
            pass

    assert len(microsoft.connections) == 2


SILENT, SPOKEN = b"\0\0" * 800, b"\x00\x40" * 800  # 50 ms frames, quiet and loud


# Catches: idle silence uploaded (and billed) between turns, a turn's onset clipped by a gate
# that waits for the VAD's start, speech frames dropped on their way to the provider, or the
# usage measuring the audio pushed instead of the audio sent.
async def test_with_a_vad_only_speech_and_its_pre_roll_are_uploaded(
    microsoft: MicrosoftAI,
) -> None:
    vad = FakeVAD()
    instance = microsoft_ai.STT(
        vad=vad, url=microsoft.stt_url, model="test-transcriber", api_key="dummy-stt-key"
    )
    # each turn: 1 s of silence, 0.5 s of speech, 0.4 s of silence (its last 0.2 s ends it)
    turn = [SILENT] * 20 + [SPOKEN] * 10 + [SILENT] * 8
    async with instance, instance.stream(conn_options=OPTIONS) as stream:
        pushed = 0.0
        for pcm in turn * 2:
            stream.push_frame(rtc.AudioFrame(pcm, 16000, 1, 800))
            pushed += 0.05
            await until(lambda at=pushed: vad.heard >= at - 1e-6, "the VAD judges the frame")
        stream.end_input()
        events = [event async for event in stream]

    # 300 ms before the speech, all of it, and the 200 ms of quiet the VAD took to end it
    heard = SILENT * 6 + SPOKEN * 10 + SILENT * 4
    assert microsoft.commits == [heard, heard]
    appended = b"".join(
        base64.b64decode(event["audio"])
        for event in microsoft.sent
        if event["type"] == "input_audio_buffer.append"
    )
    assert appended == heard * 2
    assert _finals(events) == ["turn 1", "turn 2"]
    # 1 s sent per turn (20 frames of 50 ms), of the 3.8 s pushed
    usage = sum(e.recognition_usage.audio_duration or 0.0 for e in events if e.recognition_usage)
    assert usage == pytest.approx(2.0)


async def test_complete_wav_request_and_exact_audio_framing(microsoft: MicrosoftAI) -> None:
    """One SSML request with the key header; the WAV's PCM plays exactly, in one final segment."""
    microsoft.tts_body = _wav(PCM)
    instance = _tts(microsoft)

    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        events = [event async for event in stream]

    assert b"".join(event.frame.data.tobytes() for event in events) == PCM
    assert all((e.frame.sample_rate, e.frame.num_channels) == (24000, 1) for e in events)
    assert [event.is_final for event in events].count(True) == 1 and events[-1].is_final
    assert len({event.request_id for event in events}) == 1
    [(query, headers, body)] = microsoft.tts_requests
    assert query == "deployment=dummy"
    root = ElementTree.fromstring(body)
    assert root.attrib == {"version": "1.0", "{http://www.w3.org/XML/1998/namespace}lang": "en-US"}
    voice = root.find(f"{SSML}voice")
    assert voice is not None and (voice.text, voice.attrib["name"]) == ("Hello.", VOICE)
    assert headers["Ocp-Apim-Subscription-Key"] == "dummy-tts-key"
    assert "Authorization" not in headers
    assert headers["Accept"] == "audio/wav"
    assert headers["Content-Type"] == "application/ssml+xml"
    assert headers["X-Microsoft-OutputFormat"] == "riff-24khz-16bit-mono-pcm"


@pytest.mark.parametrize("status", [302, 400, 401, 403, 422, 429, 500, 503])
async def test_http_errors_are_explicit_and_do_not_echo_response_bodies(
    microsoft: MicrosoftAI, status: int
) -> None:
    """A failed synthesis keeps its status, retries only transient ones, and hides the body."""
    microsoft.tts_status, microsoft.tts_body = status, b"do-not-log-this-dummy-secret"
    instance = _tts(microsoft)

    async with instance, instance.synthesize("Hello.", conn_options=OPTIONS) as stream:
        with pytest.raises(APIStatusError) as caught:
            async for _ in stream:
                pass

    assert caught.value.status_code == status
    assert caught.value.retryable == (status in (429, 500, 503))
    assert "do-not-log" not in str(caught.value)
    assert len(microsoft.tts_requests) == 1


async def test_text_is_escaped_not_interpreted_as_ssml(microsoft: MicrosoftAI) -> None:
    """Text that looks like SSML is sent as literal text."""
    text = '</voice><audio src="https://example.invalid/audio"/> & <voice>literal'
    microsoft.tts_body = _wav(PCM)
    instance = _tts(microsoft)

    async with instance, instance.synthesize(text, conn_options=OPTIONS) as stream:
        async for _ in stream:
            pass

    [(_, _, body)] = microsoft.tts_requests
    voice = ElementTree.fromstring(body).find(f"{SSML}voice")
    assert voice is not None and voice.text == text
    assert list(voice) == []
    assert b"&lt;audio" in body and b"&amp;" in body
