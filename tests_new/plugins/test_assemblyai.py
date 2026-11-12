"""AssemblyAI STT: connect config, turn events and wire contracts, against a local peer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, http, stt
from livekit.agents.stt import SpeechEventType
from livekit.plugins.assemblyai import STT
from tests_new.fakes import until

NO_RETRY = APIConnectOptions(max_retry=0)


@dataclass
class AssemblyAI:
    """The local AssemblyAI endpoint: its events and what it received."""

    base_url: str = ""
    events: list[dict[str, object]] = field(default_factory=list)
    queries: list[dict[str, str]] = field(default_factory=list)
    received: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)


@pytest.fixture
async def assemblyai() -> AsyncIterator[AssemblyAI]:
    """A local AssemblyAI streaming endpoint.

    After the first audio it sends ``events``; it answers ``Terminate`` with ``Termination``.
    ``queries`` holds each connection's query and ``received`` every JSON message.
    """
    peer = AssemblyAI()

    async def ws(request: web.Request) -> web.WebSocketResponse:
        peer.queries.append(dict(request.query))
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        heard = False
        async for incoming in socket:
            if incoming.type == web.WSMsgType.BINARY and not heard:
                heard = True
                for event in peer.events:
                    await socket.send_json(event)
            elif incoming.type == web.WSMsgType.TEXT:
                message = incoming.json()
                peer.received.put_nowait(message)
                if message == {"type": "Terminate"}:
                    await socket.send_json({"type": "Termination", "audio_duration_seconds": 0.0})
                    await socket.close()
        return socket

    app = web.Application()
    app.router.add_get("/v3/ws", ws)
    async with TestServer(app) as server, http.open():
        peer.base_url = str(server.make_url(""))
        yield peer


def _audio() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        b"\x00\x00" * 1600, sample_rate=16000, num_channels=1, samples_per_channel=1600
    )


async def _hear(model: STT) -> tuple[list[stt.SpeechEvent], stt.RecognizeStream]:
    """Pushes 100 ms of audio, ends the input and returns every event the stream produced."""
    async with model.stream(conn_options=NO_RETRY) as stream:
        stream.push_frame(_audio())
        stream.end_input()
        events = [event async for event in stream]
    await model.aclose()
    return events, stream


def _turn(*words: str, **fields: object) -> dict[str, object]:
    timed = [
        {"text": w, "start": i * 100, "end": i * 100 + 80, "confidence": 0.9}
        for i, w in enumerate(words)
    ]
    return {"type": "Turn", "words": timed, **fields}


async def test_speech_started_uses_start_time_anchor(assemblyai: AssemblyAI) -> None:
    """SpeechStarted's stream-relative ms become wall-clock time from the first audio sent.

    https://github.com/livekit/agents/pull/5479
    """
    assemblyai.events = [{"type": "SpeechStarted", "timestamp": 500}]

    events, stream = await _hear(STT(api_key="key", base_url=assemblyai.base_url))

    [started] = [e for e in events if e.type == SpeechEventType.START_OF_SPEECH]
    assert started.speech_start_time == pytest.approx(stream.start_time + 0.5)


@pytest.mark.parametrize("mode", ["min_latency", "balanced", "max_accuracy"])
async def test_mode_omits_silence_defaults_when_unset(
    assemblyai: AssemblyAI, mode: Literal["min_latency", "balanced", "max_accuracy"]
) -> None:
    """A mode without explicit turn silence sends no silence windows that would override it.

    https://github.com/livekit/agents/pull/6220
    """
    model = STT(api_key="key", model="universal-3-5-pro", mode=mode, base_url=assemblyai.base_url)

    await _hear(model)

    [query] = assemblyai.queries
    assert query["mode"] == mode
    assert "min_turn_silence" not in query
    assert "max_turn_silence" not in query


async def test_language_confidence_coexists_with_end_of_turn_confidence(
    assemblyai: AssemblyAI,
) -> None:
    """Both scores ride the final transcript's metadata. https://github.com/livekit/agents/pull/7339"""
    assemblyai.events = [
        _turn(
            "hola",
            end_of_turn=True,
            transcript="hola",
            end_of_turn_confidence=1.0,
            language_confidence=0.92,
        )
    ]

    events, _ = await _hear(STT(api_key="key", base_url=assemblyai.base_url))

    [final] = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
    assert final.alternatives[0].metadata == {
        "end_of_turn_confidence": 1.0,
        "language_confidence": 0.92,
    }


@pytest.mark.parametrize("formatted", [False, True])
async def test_short_turn_emits_one_final_and_one_end(
    assemblyai: AssemblyAI, formatted: bool
) -> None:
    """A short turn is final once, after formatting when asked, then ends speech.

    https://github.com/livekit/agents/issues/1560
    """
    turn = _turn("yes", end_of_turn=True, transcript="yes")
    assemblyai.events = [turn]
    if formatted:
        assemblyai.events.append({**turn, "turn_is_formatted": True, "transcript": "Yes."})

    events, _ = await _hear(
        STT(api_key="key", format_turns=formatted, base_url=assemblyai.base_url)
    )

    ends = (SpeechEventType.FINAL_TRANSCRIPT, SpeechEventType.END_OF_SPEECH)
    assert [(e.type, [a.text for a in e.alternatives]) for e in events if e.type in ends] == [
        (SpeechEventType.FINAL_TRANSCRIPT, ["Yes." if formatted else "yes"]),
        (SpeechEventType.END_OF_SPEECH, []),
    ]


async def test_interim_text_is_cumulative_within_a_turn(assemblyai: AssemblyAI) -> None:
    """Interim transcripts carry the whole turn so far, never only the latest chunk.

    https://github.com/livekit/agents/issues/4779
    """
    said = ["I", "need", "you", "to", "be", "honest"]
    assemblyai.events = [
        _turn(*said[:n], utterance=" ".join(said[n - 2 : n])) for n in (2, 4, 6)
    ] + [_turn(*said, end_of_turn=True, transcript=" ".join(said))]

    events, _ = await _hear(STT(api_key="key", base_url=assemblyai.base_url))

    interims = [
        e.alternatives[0].text for e in events if e.type == SpeechEventType.INTERIM_TRANSCRIPT
    ]
    assert interims[:3] == ["I need", "I need you to", "I need you to be honest"]


async def test_provider_close_reason_reaches_the_caller() -> None:
    """A provider error close surfaces as a typed error with its code and reason intact.

    https://github.com/livekit/agents/pull/1499
    """

    async def refuse(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_json({"error": "Invalid audio encoding"})
        await socket.close(code=3005, message=b"Invalid audio encoding")
        return socket

    app = web.Application()
    app.router.add_get("/v3/ws", refuse)
    async with TestServer(app) as server, http.open():
        model = STT(api_key="key", base_url=str(server.make_url("")))

        with pytest.raises(APIStatusError) as caught:
            async with model.stream(conn_options=NO_RETRY) as stream:
                async for _ in stream:
                    pass
        await model.aclose()

    assert caught.value.status_code == 3005
    assert "Invalid audio encoding" in str(caught.value.body)


async def test_keyterms_survive_query_escaping_and_termination_is_acknowledged(
    assemblyai: AssemblyAI,
) -> None:
    """Vocabulary survives query escaping and ending the input terminates the session.

    https://github.com/livekit/agents/pull/3001
    """
    terms = ["two words", "München", "C++ & a/b?x=1#end"]

    events, _ = await _hear(STT(api_key="key", keyterms_prompt=terms, base_url=assemblyai.base_url))

    assert events == []
    assert [json.loads(query["keyterms_prompt"]) for query in assemblyai.queries] == [terms]
    assert assemblyai.received.get_nowait() == {"type": "Terminate"}


async def test_language_codes_in_connect_config(assemblyai: AssemblyAI) -> None:
    """language_codes is sent as a JSON array. https://github.com/livekit/agents/pull/6443"""
    model = STT(
        api_key="key",
        model="universal-3-5-pro",
        language_codes=["en", "es"],
        base_url=assemblyai.base_url,
    )

    await _hear(model)

    assert json.loads(assemblyai.queries[0]["language_codes"]) == ["en", "es"]


@pytest.mark.parametrize(
    "languages",
    [
        ["en", "fr"],
        pytest.param(
            ["en", "fr", "de", "es", "it", "pt", "nl", "ja", "ko", "zh", "hi"],
            marks=pytest.mark.xfail(
                strict=True,
                reason="https://github.com/livekit/agents/pull/8178 reproduces: "
                "universal-3-6-pro rejects more than ten languages",
            ),
        ),
    ],
)
async def test_u3_context_and_explicit_updates_reach_provider(
    assemblyai: AssemblyAI, languages: list[str]
) -> None:
    """U3 connects with its context and languages; a live update reaches the open socket.

    https://github.com/livekit/agents/pull/8178
    """
    model = STT(
        api_key="key",
        model="universal-3-6-pro",
        language_codes=languages,
        agent_context="The agent asked about Munich.",
        base_url=assemblyai.base_url,
    )
    async with model.stream(conn_options=NO_RETRY) as stream:
        stream.push_frame(_audio())
        await until(lambda: assemblyai.queries, "the stream connects")

        model.update_options(continuous_partials=False, agent_context="The next question.")
        update = await assemblyai.received.get()
    await model.aclose()

    assert update["continuous_partials"] is False
    assert update["agent_context"] == "The next question."
    [query] = assemblyai.queries
    assert "continuous_partials" not in query
    assert json.loads(query["language_codes"]) == languages
    assert query["agent_context"] == "The agent asked about Munich."
