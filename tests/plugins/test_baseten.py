"""Baseten plugin: STT transcript timing and input end, LLM mid-conversation instructions."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIError, _core, http, stt, tts
from livekit.agents.llm import ChatContext
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins.baseten import LLM, STT, TTS, list_voices, register_voice
from livekit.plugins.baseten.models import STTModels
from tests.fakes import until
from tests.plugins.conftest import Connection, Frame, WsPeer

# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


async def test_qwen_eof_waits_for_every_commit_not_the_first_flush_response() -> None:
    commits = 0

    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        nonlocal commits
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for raw in socket:
            if json.loads(raw.data).get("type") != "input_audio_buffer.commit":
                continue
            commits += 1
            if commits == 2:
                for text in ("first", "second"):
                    await socket.send_json(
                        {
                            "is_final": True,
                            "is_end_of_audio_flush": True,
                            "segments": [{"text": text}],
                        }
                    )
        return socket

    app = web.Application()
    app.router.add_get("/stt", transcribe)
    async with TestServer(app) as server, http.open():
        model = STT(
            model="qwen3-asr",
            api_key="key",
            model_endpoint=str(server.make_url("/stt").with_scheme("ws")),
        )
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_frame(rtc.AudioFrame(bytes(3200), 16000, 1, 1600))
            stream.flush()
            stream.push_frame(rtc.AudioFrame(bytes(3200), 16000, 1, 1600))
            stream.end_input()
            finals = [
                event.alternatives[0].text
                async for event in stream
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
            ]
        await model.aclose()
    assert commits == 2
    assert finals == ["first", "second"]


@pytest.mark.parametrize("model", ["whisper", "qwen3-asr"])
@pytest.mark.parametrize("invalid_timing", [False, True])
async def test_transcript_timing_is_validated_before_delivery(
    model: STTModels, invalid_timing: bool
) -> None:
    """Valid timing is delivered shifted by the stream offset; invalid timing fails the stream."""

    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.send_json(
            {
                "type": "transcription",
                "is_final": True,
                "is_end_of_audio_flush": True,
                "language_code": "en",
                "segments": [
                    {
                        "text": "Hello",
                        "start_time": 1.0,
                        "end_time": 1.5,
                        "word_timestamps": [
                            {
                                "word": "Hello",
                                "start_time": "invalid" if invalid_timing else 1.0,
                                "end_time": 1.5,
                                "prob": 0.9,
                            }
                        ],
                    }
                ],
            }
        )
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/stt", transcribe)
    async with TestServer(app) as server, http.open() as session:
        provider = STT(
            model=model,
            api_key="test",
            model_endpoint=str(server.make_url("/stt").with_scheme("ws")),
            show_word_timestamps=True,
            http_session=session,
        )
        async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.start_time_offset = 0.25
            # audio the provider answers: an input with none owes no reply
            stream.push_frame(rtc.AudioFrame(bytes(3200), 16000, 1, 1600))
            stream.end_input()
            if invalid_timing:
                with pytest.raises(APIError) as failure:
                    async for _ in stream:
                        pass
                assert failure.value.kind == "protocol"
            else:
                async for event in stream:
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        speech = event.alternatives[0]
                        break
                else:
                    pytest.fail("No final transcript was delivered")
                assert speech.text == "Hello"
                assert speech.start_time == pytest.approx(1.0 + stream.start_time_offset)
                assert speech.end_time == pytest.approx(1.5 + stream.start_time_offset)
                assert speech.words is not None
                assert speech.words[0].start_time == speech.start_time
                assert speech.words[0].end_time == speech.end_time


async def test_ending_input_flushes_buffered_audio_and_delivers_its_transcript() -> None:
    """Every sample is sent and its final transcript delivered; https://github.com/livekit/agents/pull/8144."""
    audio = bytearray()
    controls: list[Wire] = []

    async def transcribe(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()  # the session metadata
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                audio.extend(msg.data)
                continue
            controls.append(json.loads(msg.data))
            if controls[-1].get("type") == "end_audio":
                segment = {"text": "check in", "start_time": 0.0, "end_time": 0.048}
                await ws.send_json(
                    {"type": "transcription", "is_final": True, "segments": [segment]}
                )
                await ws.send_json({"type": "end_audio", "body": {"status": "finished"}})
            break
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/stt", transcribe)
    async with TestServer(app) as server, http.open() as session:
        provider = STT(
            api_key="test",
            model_endpoint=str(server.make_url("/stt").with_scheme("ws")),
            http_session=session,
        )
        finals: list[str] = []
        async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_frame(
                rtc.AudioFrame(
                    b"\x01\x00" * 768, sample_rate=16000, num_channels=1, samples_per_channel=768
                )
            )
            stream.end_input()
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    finals.append(event.alternatives[0].text)

    assert len(audio) == 768 * 2
    assert controls == [{"type": "end_audio"}]
    assert finals == ["check in"]


@contextlib.asynccontextmanager
async def _serve(stream: bytes) -> AsyncIterator[tuple[str, list[bytes]]]:
    """A local chat-completions endpoint replaying `stream`; yields its base_url and bodies."""
    received: list[bytes] = []

    async def complete(request: web.Request) -> web.Response:
        received.append(await request.read())
        return web.Response(body=stream, content_type="text/event-stream")

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    async with TestServer(app) as server:
        yield str(server.make_url("/v1")), received


PREAMBLE = "You are a helpful assistant."
INSTRUCTIONS = "Ask the caller for the year they were born."
INLINED = f"<instructions>\n{INSTRUCTIONS}\n</instructions>"

INLINE_MODELS = ["Qwen/Qwen3.8-27B"]

_ONE_TOKEN = (
    b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":0,"model":"test",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}'
    b"\n\ndata: [DONE]\n\n"
)


async def _sent_messages(
    model: str, chat_ctx: ChatContext, *, inline: NotGivenOr[bool] = NOT_GIVEN
) -> list[Wire]:
    """The messages a Baseten LLM sends for `chat_ctx`, recorded by a local endpoint."""
    async with _serve(_ONE_TOKEN) as (base_url, received):
        llm_model = LLM(
            model=model,
            api_key="test",
            base_url=base_url,
            inline_mid_conversation_instructions=inline,
        )
        try:
            async with llm_model.chat(chat_ctx=chat_ctx) as stream:
                async for _ in stream:
                    pass
        finally:
            await llm_model.aclose()

    assert len(received) == 1
    return cast(list[Wire], json.loads(received[0])["messages"])


def _per_turn_ctx() -> ChatContext:
    # the shape generate_reply(instructions=...) produces: a trailing system message
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=[PREAMBLE])
    chat_ctx.add_message(role="assistant", content=["Hello! How can I help you?"])
    chat_ctx.add_message(role="user", content=["I'd like to refill my prescription."])
    chat_ctx.add_message(role="system", content=[INSTRUCTIONS])
    return chat_ctx


@pytest.mark.parametrize("model", INLINE_MODELS)
async def test_per_turn_instructions_are_inlined_for_single_system_models(model: str) -> None:
    """https://github.com/livekit/agents/pull/7388"""
    messages = await _sent_messages(model, _per_turn_ctx())

    assert [m["role"] for m in messages] == ["system", "assistant", "user", "user"]
    assert messages[0] == {"role": "system", "content": PREAMBLE}
    assert messages[-1] == {"role": "user", "content": INLINED}


async def _qwen3_say(model: tts.TTS, text: str) -> bytes:
    async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
        stream.push_text(text)
        stream.end_input()
        return b"".join([event.frame.data.tobytes() async for event in stream])


# Catches: a parked qwen3 socket left to the server's 30 s idle drop (the next turn then pays a
# connect and a config round trip), or its warm-keep flush sent while a turn holds the socket.
async def test_qwen3_keeps_a_parked_socket_warm_with_an_empty_flush(ws_peer: WsPeer) -> None:
    def answer(_c: Connection, message: str | bytes) -> list[Frame]:
        if json.loads(message).get("type") != "input.done":
            return []
        return [{"type": "audio.start"}, b"\x01\x02" * 2400, {"type": "session.done"}]

    ws_peer.reply = answer
    model = TTS(model="qwen3-tts", api_key="key", model_endpoint=ws_peer.ws_url, voice="voice")
    try:
        assert await _qwen3_say(model, "Hello there.")
        connection = await ws_peer.connected()

        def flushes() -> list[str]:
            return [m for m in connection.texts if json.loads(m)["type"] == "input.done"]

        assert len(flushes()) == 1, "the turn's own flush only"
        _core.skip_host_time(15)
        await until(lambda: len(flushes()) == 2, "the parked socket's warm-keep")
        assert await _qwen3_say(model, "Again.")
        assert len(ws_peer.connections) == 1, "the next turn takes the warm socket"
    finally:
        await model.aclose()


# Catches: a voice-control reply that is not a JSON object, or a refused voice.add/voice.list,
# surfacing as an untyped RuntimeError or pydantic error a caller cannot branch on.
@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        ('["not", "an", "object"]', "protocol"),
        (b"\x00", "protocol"),
        ({"type": "error", "message": "consent missing"}, "server"),
    ],
)
async def test_qwen3_voice_control_failures_are_typed(
    ws_peer: WsPeer, reply: Frame, kind: str
) -> None:
    ws_peer.reply = lambda _c, _m: [reply]
    with pytest.raises(APIError) as listed:
        await list_voices(model_endpoint=ws_peer.ws_url, api_key="key")
    assert listed.value.kind == kind
    with pytest.raises(APIError) as added:
        await register_voice(
            model_endpoint=ws_peer.ws_url, name="v", ref_audio_path=__file__, api_key="key"
        )
    assert added.value.kind == kind
