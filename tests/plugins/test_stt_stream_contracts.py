"""Contracts every streaming STT plugin keeps, each run against its provider's wire on a local
WsPeer: a refused request (an error frame, an auth or policy close) fails the stream for good,
a server error close or a dropped connection reconnects, a reconnect closes the open speech
first, the audio taken is reported as usage once, and closing the stream ends every wait.

A case says how to build the plugin on the peer and what its provider sends to open speech and
to report an error in band. A plugin whose wire the peer cannot speak is skipped with what is
missing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from mistralai.client import Mistral
from palabra_ai.client import REGIONS, Region

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, http, stt
from livekit.plugins import (
    assemblyai,
    aws,
    azure,
    baseten,
    cartesia,
    deepgram,
    elevenlabs,
    fireworksai,
    gladia,
    gnani,
    google,
    gradium,
    inworld,
    meta,
    microsoft_ai,
    mistralai,
    nabrah,
    nvidia,
    openai,
    palabra,
    rtzr,
    sarvam,
    simplismart,
    slng,
    smallestai,
    soniox,
    speechmatics,
    telnyx,
    xai,
)
from livekit.plugins.google.beta import GeminiSTT
from tests.fakes import FakeVAD, until

if TYPE_CHECKING:
    from tests.plugins.conftest import Frame, WsPeer

from tests.plugins.conftest import Close

RATE = 16000
SOS, EOS = stt.SpeechEventType.START_OF_SPEECH, stt.SpeechEventType.END_OF_SPEECH


@dataclass
class SttCase:
    """How a streaming STT plugin meets its provider on a WsPeer."""

    make: Callable[[WsPeer], stt.STT]
    speech: list[Frame]
    """What the provider sends while the person speaks: the stream starts speech on it."""
    error: list[Frame] | None
    """An error the provider reports in band; None when it only closes the socket."""
    hello: list[Frame] = field(default_factory=list)
    """What the provider sends on accept."""
    skip: str | None = None
    """What the peer is missing to speak this provider's wire; None when it can."""
    single_connection: str | None = None
    """Why the plugin never opens a second connection; None when it reconnects."""
    racy: dict[str, str] = field(default_factory=dict)
    """Contracts whose outcome a race in the plugin decides, with what races."""


def _deepgram_interim(text: str) -> Frame:
    return {
        "type": "Results",
        "is_final": False,
        "channel": {"alternatives": [{"transcript": text, "confidence": 0.9}]},
    }


def _gladia(peer: WsPeer) -> stt.STT:
    async def init_session(_: web.Request) -> web.StreamResponse:
        return web.json_response({"id": "session", "url": peer.ws_url + "/session"})

    peer.http = init_session
    return gladia.STT(api_key="key", base_url=peer.url + "/v2/live")


def _meta(peer: WsPeer) -> stt.STT:
    """Muse takes only wss:// URLs: the session connects them to the peer's plain socket."""
    session = http.http_session()
    connect = session.ws_connect

    # Any: ws_connect's keyword options, forwarded untouched
    async def local_tls_terminator(url: str, **kwargs: Any) -> http.WebSocket:
        return await connect(url.replace("wss://", "ws://", 1), **kwargs)

    session.ws_connect = local_tls_terminator  # type: ignore[method-assign]
    url = peer.url.replace("http://", "wss://", 1) + "/v1/asr/realtime"
    return meta.STT(api_key="key", url=url, http_session=session)


def _microsoft_ai_interim(text: str) -> Frame:
    return {
        "type": "conversation.item.input_audio_transcription.intermediate",
        "item_id": "item-1",
        "intermediate": text,
    }


def _mistral_session_created() -> Frame:
    audio_format = {"encoding": "pcm_s16le", "sample_rate": 16000}
    session = {"request_id": "request", "model": "voxtral", "audio_format": audio_format}
    return {"type": "session.created", "session": session}


def _palabra(peer: WsPeer) -> stt.STT:
    """The SDK resolves its endpoint from a region name: the peer is registered as one."""
    REGIONS["peer"] = Region(stt=peer.ws_url + "/asr/v1/speech-to-text/stream")
    return palabra.STT(api_key="key", region="peer")


def _speechmatics(peer: WsPeer) -> stt.STT:
    """Speechmatics answers EndOfStream with EndOfTranscript, which the SDK's close awaits."""
    peer.reply = lambda _, message: (
        [{"message": "EndOfTranscript"}] if '"EndOfStream"' in str(message) else []
    )
    return speechmatics.STT(api_key="key", base_url=peer.ws_url + "/v2/agent", vad=None)


CASES: dict[str, SttCase] = {
    "deepgram": SttCase(
        make=lambda peer: deepgram.STT(api_key="key", base_url=peer.ws_url + "/v1/listen"),
        speech=[_deepgram_interim("hello")],
        error=None,
    ),
    "assemblyai": SttCase(
        make=lambda peer: assemblyai.STT(api_key="key", base_url=peer.url),
        speech=[{"type": "SpeechStarted", "timestamp": 0}],
        error=[{"error": "invalid audio"}, Close(3005, "invalid audio")],
    ),
    # -- aws .. microsoft_ai
    "aws": SttCase(
        make=lambda _: aws.STT(region="us-east-1"),
        speech=[],
        error=None,
        skip="Transcribe streams over the AWS SDK's HTTP/2 event stream, not a websocket",
    ),
    "azure": SttCase(
        make=lambda _: azure.STT(speech_key="key", speech_region="region"),
        speech=[],
        error=None,
        skip="the Azure Speech SDK owns its native transport and takes no websocket URL",
    ),
    "baseten": SttCase(
        make=lambda peer: baseten.STT(api_key="key", model_endpoint=peer.ws_url + "/websocket"),
        speech=[],
        error=[{"type": "error", "message": "invalid audio"}],
    ),
    "cartesia": SttCase(
        make=lambda peer: cartesia.STT(
            api_key="key", base_url=peer.url, audio_chunk_duration_ms=20
        ),
        speech=[{"type": "turn.start", "request_id": "request"}],
        error=[{"type": "error", "status_code": 400, "message": "invalid audio"}],
    ),
    "deepgram_v2": SttCase(
        make=lambda peer: deepgram.STTv2(api_key="key", base_url=peer.ws_url + "/v2/listen"),
        speech=[{"type": "TurnInfo", "event": "StartOfTurn", "transcript": "", "words": []}],
        error=[{"type": "Error", "code": "INVALID_AUDIO", "description": "invalid audio"}],
    ),
    "elevenlabs": SttCase(
        make=lambda peer: elevenlabs.STT(
            api_key="key", base_url=peer.url, model="scribe_v2_realtime"
        ),
        speech=[{"message_type": "partial_transcript", "text": "hello"}],
        error=[{"message_type": "input_error", "message": "invalid audio"}],
        hello=[{"message_type": "session_started", "session_id": "session"}],
    ),
    "fireworksai": SttCase(
        make=lambda peer: fireworksai.STT(api_key="key", base_url=peer.ws_url),
        speech=[{"segments": [{"id": 0, "text": "hello", "words": [{"word": "hello"}]}]}],
        error=None,
    ),
    "gladia": SttCase(
        make=_gladia,
        speech=[{"type": "transcript", "data": {"is_final": False, "utterance": {"text": "hi"}}}],
        error=[{"type": "error", "data": {"message": "invalid audio"}}],
        racy={
            "usage": "a close during a send reconnects, a close between sends ends the "
            "recognition (gladia/stt.py:900)"
        },
    ),
    "gnani": SttCase(
        make=lambda peer: gnani.STT(api_key="key", base_url=peer.url),
        speech=[{"type": "speech_start"}],
        error=[{"type": "error", "message": "invalid audio"}],
        hello=[{"type": "connected"}],
    ),
    "google": SttCase(
        make=lambda _: google.STT(credentials_info={}),
        speech=[],
        error=None,
        skip="Cloud Speech streams over gRPC, not a websocket",
    ),
    "google_gemini": SttCase(
        make=lambda peer: GeminiSTT(api_key="key", http_options={"base_url": peer.url}),
        speech=[],
        error=None,
        skip="the genai SDK connects only over wss:// and the peer serves plain ws://",
    ),
    "gradium": SttCase(
        make=lambda peer: gradium.STT(api_key="key", model_endpoint=peer.ws_url + "/api/speech"),
        speech=[{"type": "text", "text": "hello", "start_s": 0.0}],
        error=[{"type": "error", "code": 400, "message": "invalid audio"}],
        hello=[{"type": "ready", "frame_size": 1920, "delay_in_tokens": 4}],
    ),
    "inworld": SttCase(
        make=lambda peer: inworld.STT(api_key="key", base_url=peer.url),
        speech=[{"result": {"speechStarted": {}}}],
        error=None,
    ),
    "meta": SttCase(
        make=_meta,
        speech=[{"type": "speechStart", "turnId": "turn-1"}],
        error=[{"type": "error", "errorCode": "invalid_request", "message": "invalid audio"}],
        hello=[{"sessionId": "session"}],
    ),
    "microsoft_ai": SttCase(
        make=lambda peer: microsoft_ai.STT(
            vad=None, url=peer.ws_url + "/v1/realtime", model="transcriber", api_key="key"
        ),
        speech=[_microsoft_ai_interim("hello")],
        error=[{"type": "error", "error": {"code": "invalid_api_key", "message": "refused"}}],
        hello=[{"type": "session.created"}, {"type": "session.updated"}],
    ),
    # -- mistralai .. xai
    "mistralai": SttCase(
        make=lambda peer: mistralai.STT(
            client=Mistral(api_key="key", server_url=peer.url),
            model="voxtral-mini-transcribe-realtime-2602",
            vad=FakeVAD(),
        ),
        # speech starts on the local VAD, never on a provider frame
        speech=[],
        error=[{"type": "error", "error": {"code": 400, "message": "invalid audio"}}],
        hello=[_mistral_session_created()],
    ),
    "nabrah": SttCase(
        make=lambda peer: nabrah.STT(api_key="key", base_url=peer.ws_url),
        speech=[{"type": "transcript", "text": "hello"}],
        error=[{"type": "error", "message": "invalid audio"}],
        hello=[{"type": "ready"}],
    ),
    "nvidia": SttCase(
        make=lambda _: nvidia.STT(api_key="key"),
        speech=[],
        error=None,
        skip="Riva streams over gRPC, not a websocket",
    ),
    "openai": SttCase(
        make=lambda peer: openai.STT(
            api_key="key", base_url=peer.url + "/v1", use_realtime=True, vad=None
        ),
        speech=[{"type": "input_audio_buffer.speech_started", "item_id": "item-1"}],
        error=[{"type": "error", "error": {"message": "invalid audio"}}],
    ),
    "palabra": SttCase(
        make=_palabra,
        speech=[
            {
                "message_type": "transcription",
                "transcription_id": "tr-1",
                "is_eos": False,
                "segment": {"text": "hello"},
            }
        ],
        error=[{"message_type": "error", "data": {"code": "VALIDATION_ERROR", "desc": "bad"}}],
    ),
    "rtzr": SttCase(
        make=lambda _: rtzr.STT(),
        speech=[],
        error=None,
        skip="the client dials openapi.vito.ai and takes no base URL",
    ),
    "sarvam": SttCase(
        make=lambda _: sarvam.STT(api_key="key"),
        speech=[],
        error=None,
        skip="stream() dials Sarvam's module URL and never base_url",
    ),
    "sarvam_streaming": SttCase(
        make=lambda peer: sarvam.STTRealtime(
            api_key="key", language="hi-IN", base_url=peer.ws_url + "/ws"
        ),
        speech=[{"event": "vad.speech_start", "utterance_idx": 0}],
        error=[
            {
                "event": "error",
                "code": "invalid_request",
                "message": "invalid audio",
                "is_fatal": True,
                "status_code": 400,
            }
        ],
        hello=[{"event": "session.begin", "session_id": "session", "request_id": "request"}],
        single_connection="the endpoint bills per connection: stream() forces max_retry=0",
    ),
    "simplismart": SttCase(
        make=lambda peer: simplismart.STT(api_key="key", base_url=peer.url, streaming=True),
        speech=[],
        error=None,
        skip="streaming always dials wss://<host>/ws/audio and the peer serves plain ws://",
    ),
    "slng": SttCase(
        make=lambda peer: slng.STT(api_key="key", model="deepgram/nova:3", slng_base_url=peer.url),
        speech=[{"type": "partial_transcript", "transcript": "hello"}],
        error=[{"type": "error", "message": "invalid audio", "code": 400}],
        hello=[{"type": "ready"}],
    ),
    "smallestai": SttCase(
        make=lambda peer: smallestai.STT(api_key="key", base_url=peer.url, vad_events=True),
        speech=[{"type": "speech_started"}],
        error=[{"type": "error", "status": "400", "message": "invalid audio"}],
    ),
    "soniox": SttCase(
        make=lambda peer: soniox.STT(api_key="key", base_url=peer.ws_url + "/transcribe"),
        speech=[{"tokens": [{"text": "hello", "is_final": False}]}],
        error=[{"error_code": 400, "error_message": "invalid audio"}],
    ),
    "speechmatics": SttCase(
        make=_speechmatics,
        speech=[{"message": "StartOfTurn"}],
        error=[{"message": "Error", "type": "invalid_config", "reason": "invalid audio"}],
        hello=[{"message": "RecognitionStarted", "id": "session"}],
        racy={
            "reconnect": "when the SDK records a close as a session error before the next audio "
            "send, _process_audio breaks silently and the stream hangs (speechmatics/stt.py:667)",
            "close": "the SDK's socket close races its session task: the socket sometimes stays "
            "open after aclose()",
            "auth_close": "a close is noticed only when the next audio is sent: with none left, "
            "the stream hangs before it reads the close code (speechmatics/stt.py:667)",
        },
    ),
    "telnyx": SttCase(
        make=lambda peer: telnyx.STT(api_key="key", base_url=peer.ws_url),
        speech=[{"transcript": "hello", "is_final": False}],
        error=None,
    ),
    "xai": SttCase(
        make=lambda _: xai.STT(api_key="key"),
        speech=[],
        error=None,
        skip="the stream dials xAI's module URL and takes no base URL",
    ),
}

# (contract, plugin): what reproduces a known product bug, as a strict xfail
KNOWN: dict[tuple[str, str], str] = {
    ("speech_reconnect", "assemblyai"): (
        "a reconnect after SpeechStarted sends no END_OF_SPEECH: the plugin keeps no speaking "
        "state, so the new socket's SpeechStarted opens a second speech inside the first"
    ),
    # -- aws .. microsoft_ai
    ("reconnect", "gladia"): (
        "the receive loop iterates the socket (gladia/stt.py:900), which ends silently on a close: "
        "the recognition ends instead of reconnecting"
    ),
    ("speech_reconnect", "gladia"): (
        "a 1011 close mid-speech ends the recognition instead of reconnecting (gladia/stt.py:900)"
    ),
    ("reconnect", "gnani"): (
        "stream() forces max_retry=0 (gnani/stt.py:235): a 1011 close or a dropped connection "
        "fails the stream"
    ),
    ("speech_reconnect", "gnani"): (
        "stream() forces max_retry=0 (gnani/stt.py:235), and the plugin keeps no speaking state "
        "to end on a reconnect"
    ),
    ("usage", "gnani"): (
        "stream() forces max_retry=0 (gnani/stt.py:235): a 1011 close fails the stream before "
        "the second attempt's audio is taken"
    ),
    ("speech_reconnect", "gradium"): (
        "the speaking flag is local to recv_task (gradium/stt.py:281): a reconnect drops it "
        "without END_OF_SPEECH, so the new socket's text opens a second speech"
    ),
    ("speech_reconnect", "meta"): (
        "a reconnect keeps the open turn in _turns (meta/stt.py:384) without ending it: no "
        "END_OF_SPEECH, and the new socket's speechStart never opens a speech"
    ),
    ("speech_reconnect", "microsoft_ai"): (
        "a reconnect keeps _speaking set (microsoft_ai/stt.py:342) without END_OF_SPEECH, so "
        "the new socket's transcript never starts a speech"
    ),
    # -- mistralai .. xai
    ("speech_reconnect", "nabrah"): (
        "a close after the first transcript fails the stream instead of reconnecting "
        "(nabrah/stt.py:308)"
    ),
    ("speech_reconnect", "palabra"): (
        "an utterance started on the old socket is dropped with its registry: no END_OF_SPEECH "
        "before the new socket's START_OF_SPEECH (palabra/stt.py:230)"
    ),
    ("speech_reconnect", "slng"): (
        "the internal same-endpoint replay keeps speech_started across sockets: no END_OF_SPEECH, "
        "and the new socket's speech never starts (slng/stt.py:1089)"
    ),
    ("speech_reconnect", "soniox"): (
        "is_speaking is local to one connection's receive task: a reconnect drops it without "
        "END_OF_SPEECH (soniox/stt.py:410)"
    ),
    ("usage", "speechmatics"): (
        "a close is noticed only when the next audio is sent: once input stops, the closed "
        "last attempt never fails the stream (speechmatics/stt.py:663)"
    ),
    ("speech_reconnect", "speechmatics"): (
        "StartOfTurn opens speech with no speaking state: a reconnect sends no END_OF_SPEECH "
        "(speechmatics/stt.py:785)"
    ),
    ("speech_reconnect", "telnyx"): (
        "_run resets _speaking on a new connection without sending END_OF_SPEECH "
        "(telnyx/stt.py:192)"
    ),
}


def _params(
    contract: str,
    *,
    needs_error: bool = False,
    needs_speech: bool = False,
    needs_reconnect: bool = False,
) -> list[object]:
    params: list[object] = []
    for name, case in CASES.items():
        marks: list[pytest.MarkDecorator] = []
        if case.skip is not None:
            marks.append(pytest.mark.skip(reason=f"{name}: {case.skip}"))
        if needs_speech and not case.speech:
            marks.append(pytest.mark.skip(reason=f"{name} emits no START_OF_SPEECH"))
        if needs_error and case.error is None:
            marks.append(pytest.mark.skip(reason=f"{name} reports errors only as a close"))
        if needs_reconnect and case.single_connection is not None:
            marks.append(pytest.mark.skip(reason=f"{name}: {case.single_connection}"))
        if race := case.racy.get(contract):
            marks.append(pytest.mark.skip(reason=f"{name} races: {race}"))
        if reason := KNOWN.get((contract, name)):
            marks.append(
                pytest.mark.xfail(
                    strict=True,
                    reason=f"https://github.com/livekit/agents/commit/9e79bc520b80553e3ca665884a2fac2beeb63042 reproduces: {reason}",
                )
            )
        params.append(pytest.param(name, marks=marks))
    return params


def _audio(seconds: float = 0.1) -> rtc.AudioFrame:
    samples = round(RATE * seconds)
    return rtc.AudioFrame(b"\0\0" * samples, RATE, 1, samples)


async def _events(stream: stt.RecognizeStream, into: list[stt.SpeechEvent]) -> None:
    async for event in stream:
        into.append(event)


async def _open(
    name: str, peer: WsPeer, options: APIConnectOptions
) -> tuple[stt.STT, stt.RecognizeStream]:
    case = CASES[name]
    peer.hello = case.hello
    model = case.make(peer)
    return model, model.stream(conn_options=options)


async def _fails(
    name: str, peer: WsPeer, frames: list[Frame]
) -> tuple[APIError, list[stt.SpeechEvent]]:
    """Every connection gets ``frames`` once audio flows; returns how the stream failed."""
    model, stream = await _open(name, peer, APIConnectOptions(max_retry=2, retry_interval=0))
    peer.reply = lambda connection, _: frames if len(connection.received) == 1 else []
    events: list[stt.SpeechEvent] = []
    try:
        stream.push_frame(_audio())
        with pytest.raises(APIError) as failure:
            async with asyncio.timeout(2):
                await _events(stream, events)
    finally:
        await stream.aclose()
        await model.aclose()
    return failure.value, events


# Catches: a provider error frame logged and dropped, so the turn waits forever, or retried on
# a fresh socket as if the provider had not refused the request.
@pytest.mark.parametrize("name", _params("error_frame", needs_error=True))
async def test_a_provider_error_frame_fails_the_stream_for_good(name: str, ws_peer: WsPeer) -> None:
    error = CASES[name].error
    assert error is not None

    failure, _ = await _fails(name, ws_peer, error)

    assert failure.retryable is False
    assert len(ws_peer.connections) == 1


# Catches: an auth or policy close read as a status code (1008 and 4401 retried as transient)
# or logged as an unexpected close and retried until the budget runs out.
@pytest.mark.parametrize("code", [1008, 4401])
@pytest.mark.parametrize("name", _params("auth_close"))
async def test_an_auth_or_policy_close_fails_the_stream_for_good(
    name: str, code: int, ws_peer: WsPeer
) -> None:
    failure, _ = await _fails(name, ws_peer, [Close(code, "refused")])

    assert failure.retryable is False
    assert len(ws_peer.connections) == 1


# Catches: a server error close or a dropped connection failing the stream instead of
# reconnecting.
@pytest.mark.parametrize("close", [Close(1011), Close(None)], ids=["1011", "dropped"])
@pytest.mark.parametrize("name", _params("reconnect", needs_reconnect=True))
async def test_a_server_error_close_or_a_dropped_connection_reconnects(
    name: str, close: Close, ws_peer: WsPeer
) -> None:
    model, stream = await _open(name, ws_peer, APIConnectOptions(max_retry=1, retry_interval=0))
    events: list[stt.SpeechEvent] = []
    reader = asyncio.create_task(_events(stream, events))
    try:
        stream.push_frame(_audio())
        await ws_peer.connected()
        await ws_peer.send(close)
        stream.push_frame(_audio())
        await ws_peer.connected(2)
        assert not reader.done(), reader.exception()
    finally:
        await stream.aclose()
        await model.aclose()
        await reader


# Catches: a reconnect in the middle of speech leaving the speech open, so the new connection's
# START_OF_SPEECH lands inside the old one and the turn never ends.
@pytest.mark.parametrize(
    "name", _params("speech_reconnect", needs_speech=True, needs_reconnect=True)
)
async def test_a_reconnect_mid_speech_ends_the_open_speech_first(
    name: str, ws_peer: WsPeer
) -> None:
    case = CASES[name]
    model, stream = await _open(name, ws_peer, APIConnectOptions(max_retry=1, retry_interval=0))
    events: list[stt.SpeechEvent] = []
    reader = asyncio.create_task(_events(stream, events))
    try:
        stream.push_frame(_audio())
        await ws_peer.connected()
        await ws_peer.send(*case.speech)
        await until(lambda: SOS in [e.type for e in events], "the first socket's speech starts")
        await ws_peer.send(Close(1011))
        stream.push_frame(_audio())
        await ws_peer.connected(2)
        await ws_peer.send(*case.speech)
        await until(
            lambda: [e.type for e in events].count(SOS) == 2, "the second socket's speech starts"
        )
    finally:
        await stream.aclose()
        await model.aclose()
        await reader

    kinds = [e.type for e in events if e.type in (SOS, EOS)]
    assert kinds[: kinds.index(SOS, 1) + 1] == [SOS, EOS, SOS]


# Catches: audio reported as RECOGNITION_USAGE twice (a plugin's own collector beside the
# framework's) or never for an attempt that failed.
@pytest.mark.parametrize("name", _params("usage", needs_reconnect=True))
async def test_the_audio_pushed_is_reported_once_across_retries_and_failure(
    name: str, ws_peer: WsPeer
) -> None:
    model, stream = await _open(name, ws_peer, APIConnectOptions(max_retry=1, retry_interval=0))
    if model.capabilities.reports_usage:
        await stream.aclose()
        await model.aclose()
        pytest.skip(f"{name} reports its own usage")
    events: list[stt.SpeechEvent] = []
    reader = asyncio.create_task(_events(stream, events))
    try:
        for close, connection in ((Close(1011), 1), (Close(1008), 2)):
            for _ in range(5):
                stream.push_frame(_audio())
            await ws_peer.connected(connection)
            await until(lambda: stream.input.empty(), "the plugin takes the audio")
            await ws_peer.send(close)
        with pytest.raises(APIError):
            async with asyncio.timeout(2):
                await reader
    finally:
        await stream.aclose()
        await model.aclose()

    reported = sum(e.recognition_usage.audio_duration or 0.0 for e in events if e.recognition_usage)
    # a plugin that takes another rate leaves the resampler's tail (~15 ms) untaken
    assert reported == pytest.approx(1.0, abs=0.02)


# Catches: a send, receive or keepalive wait that outlives the stream's close, leaking its task
# (the suite's leaked-task check fails the test) or holding aclose() open.
@pytest.mark.parametrize("name", _params("close"))
async def test_closing_the_stream_ends_every_pending_wait(name: str, ws_peer: WsPeer) -> None:
    model, stream = await _open(name, ws_peer, APIConnectOptions(max_retry=0))
    events: list[stt.SpeechEvent] = []
    reader = asyncio.create_task(_events(stream, events))
    stream.push_frame(_audio())
    await ws_peer.connected()

    await stream.aclose()
    await model.aclose()

    await until(reader.done, "the reader ends with the stream")
    await until(lambda: ws_peer.connections[0].socket.closed, "the plugin closes its socket")
