"""Smallest AI STT plugin: speech events at the public streaming boundary."""

import asyncio
import base64
import json
import struct

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, _core, http, stt
from livekit.plugins.smallestai import STT, TTS
from tests.fakes import until
from tests.plugins.conftest import Connection, Frame, WsPeer

PCM = b"\x01\x02" * 2400


async def test_a_final_between_vad_boundaries_reports_its_words_and_utterances() -> None:
    """The server VAD bounds the speech; the final carries its words and shifted utterances."""

    async def recognize(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_json({"type": "speech_started"})
        await socket.send_json(
            {
                "session_id": "session",
                "transcript": "Hello",
                "is_final": True,
                "words": [{"word": "Hello", "start": 0.1, "end": 0.5, "confidence": 0.9}],
                "utterances": [{"start": 0.1, "end": 0.5, "text": "Hello"}],
            }
        )
        await socket.send_json({"type": "speech_ended"})
        await socket.send_json({"is_last": True})
        async for _ in socket:
            pass
        return socket

    app = web.Application()
    app.router.add_get("/stt/live", recognize)
    async with TestServer(app) as server, http.open() as session:
        provider = STT(api_key="key", base_url=str(server.make_url("")), http_session=session)
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                # audio the provider answers: an input with none owes no reply
                stream.push_frame(rtc.AudioFrame(bytes(3200), 16000, 1, 1600))
                stream.end_input()
                async with asyncio.timeout(1):
                    events = [event async for event in stream]
        finally:
            await provider.aclose()
    speech_events = [
        event for event in events if event.type != stt.SpeechEventType.RECOGNITION_USAGE
    ]
    assert [event.type for event in speech_events] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]
    transcript = speech_events[1].alternatives[0]
    assert transcript.text == "Hello"
    assert transcript.words is not None and [word.text for word in transcript.words] == ["Hello"]
    assert transcript.metadata == {
        "utterances": [
            {
                "start": transcript.words[0].start_time,
                "end": transcript.words[0].end_time,
                "text": "Hello",
            }
        ]
    }


async def test_finalize_and_close_stream_deliver_last_transcript() -> None:
    """Finalization preserves speech boundaries and closes with the final transcript. https://github.com/livekit/agents/pull/5562 https://github.com/livekit/agents/pull/7379 https://github.com/livekit/agents/pull/5953"""
    import json

    from livekit import rtc

    commands = []
    query = []

    async def recognize(request: web.Request) -> web.StreamResponse:
        query.append(dict(request.query))
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for msg in socket:
            if msg.type != web.WSMsgType.TEXT:
                continue
            command = json.loads(msg.data)["type"]
            commands.append(command)
            if command == "finalize" and commands.count("finalize") == 1:
                await socket.send_json({"type": "speech_started"})
                await socket.send_json({"transcript": "Hello", "is_final": True})
                await socket.send_json({"transcript": "...", "is_final": True})
                await socket.send_json({"transcript": "", "is_final": True, "from_finalize": True})
            elif command == "close_stream":
                await socket.send_json({"transcript": "world", "is_final": True, "is_last": True})
        return socket

    app = web.Application()
    app.router.add_get("/stt/live", recognize)
    async with TestServer(app) as server, http.open() as session:
        provider = STT(
            api_key="key",
            language="es",
            sample_rate=24000,
            base_url=str(server.make_url("")),
            http_session=session,
            vad_events=True,
            diarize=True,
            keywords=[("LiveKit", 2)],
        )
        try:
            async with provider.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
                stream.push_frame(
                    rtc.AudioFrame(
                        data=bytes(2400),
                        sample_rate=24000,
                        num_channels=1,
                        samples_per_channel=1200,
                    )
                )
                stream.flush()
                stream.end_input()
                async with asyncio.timeout(2):
                    events = [ev async for ev in stream]
        finally:
            await provider.aclose()
    assert commands[-1] == "close_stream"
    assert set(commands[:-1]) == {"finalize"}
    finals = [
        ev.alternatives[0].text for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]
    assert finals == ["Hello", "world"]
    assert sum(ev.type == stt.SpeechEventType.START_OF_SPEECH for ev in events) == 2
    assert sum(ev.type == stt.SpeechEventType.END_OF_SPEECH for ev in events) == 2
    assert query[0]["language"] == "es" and query[0]["sample_rate"] == "24000"
    assert query[0]["diarize"] == "true" and query[0]["keywords"] == "LiveKit:2"


@pytest.mark.parametrize(
    ("output_format", "encoded", "sample"), [("ulaw", 0x00, -32124), ("alaw", 0x55, -8)]
)
async def test_g711_synthesis_decodes_to_pcm(output_format: str, encoded: int, sample: int) -> None:
    """The ulaw/alaw output formats play as PCM instead of failing the decoder."""

    async def synthesize(request: web.Request) -> web.Response:
        return web.Response(body=bytes([encoded]) * 800, content_type="audio/basic")

    app = web.Application()
    app.router.add_post("/tts", synthesize)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            sample_rate=8000,
            output_format=output_format,
            base_url=str(server.make_url("")).rstrip("/"),
            http_session=session,
        )
        async with provider.synthesize("Hello.", conn_options=APIConnectOptions(max_retry=0)) as s:
            pcm = b"".join([event.frame.data.tobytes() async for event in s])
        await provider.aclose()

    assert pcm[: 800 * 2] == struct.pack("<h", sample) * 800


# Catches: every turn paying Smallest's connect on its critical path (no pool, no prewarm), or a
# socket that carried a continuation reused by the next turn, which may still get its audio.
async def test_streams_take_a_socket_opened_ahead_and_never_a_used_one(ws_peer: WsPeer) -> None:
    def answer(_c: Connection, message: str | bytes) -> list[Frame]:
        packet = json.loads(message)
        if packet["continue"]:
            return [{"status": "chunk", "data": {"audio": base64.b64encode(PCM).decode()}}]
        return []

    ws_peer.reply = answer
    model = TTS(api_key="key", ws_url=ws_peer.ws_url)
    try:
        model.prewarm()
        first = await ws_peer.connected()
        assert not first.received, "opened ahead of any turn"
        for text in ("Hello there.", "Again."):
            stream = model.stream(conn_options=APIConnectOptions(max_retry=0))
            stream.push_text(text)
            stream.flush()
            async for _ in stream:
                break  # its first audio
            _core.skip_host_time(1)  # the text idles: the segment is done
            stream.end_input()
            await stream.aclose()
        await until(lambda: len(ws_peer.connections) == 3, "the next turn's socket opens ahead")
        turns = [[json.loads(m)["text"] for m in c.texts if m] for c in ws_peer.connections]
        assert turns == [["Hello there. ", ""], ["Again. ", ""], []], "one turn a socket"
    finally:
        await model.aclose()


async def test_continuations_wait_for_late_audio_and_restart_silence_after_each_chunk(
    ws_peer: WsPeer,
) -> None:
    model = TTS(api_key="key", ws_url=ws_peer.ws_url)
    heard: list[bytes] = []
    try:
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("Hello there.")
            stream.end_input()

            async def read() -> None:
                async for event in stream:
                    heard.append(event.frame.data.tobytes())

            reader = asyncio.create_task(read())
            try:
                connection = await ws_peer.connected()
                await until(
                    lambda: any(not json.loads(m)["continue"] for m in connection.texts),
                    "the continuation's closing fragment is sent",
                )
                # Synthesis starts later than the idle interval; no audio has arrived yet.
                _core.skip_host_time(1)
                chunk = {"status": "chunk", "data": {"audio": base64.b64encode(PCM).decode()}}
                await ws_peer.send(chunk, {"status": "complete"}, connection=0)
                await until(lambda: sum(map(len, heard)) > 0, "the first release is heard")
                _core.skip_host_time(0.4)
                await ws_peer.send(chunk, {"status": "complete"}, connection=0)
                await until(lambda: sum(map(len, heard)) >= len(PCM), "the later release is heard")
                _core.skip_host_time(0.4)
                await ws_peer.send(chunk, {"status": "complete"}, connection=0)
                await until(
                    lambda: sum(map(len, heard)) >= len(PCM) * 2,
                    "audio after the original silence deadline is still heard",
                )
                _core.skip_host_time(0.7)
                await until(reader.done, "the continuation ends after received audio goes silent")
                await reader
                assert b"".join(heard) == PCM * 3
            finally:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
    finally:
        await model.aclose()


async def test_continuation_segments_drain_on_separate_sockets(ws_peer: WsPeer) -> None:
    def answer(_connection: Connection, message: str | bytes) -> list[Frame]:
        if json.loads(message)["continue"]:
            return [{"status": "chunk", "data": {"audio": base64.b64encode(PCM).decode()}}]
        return []

    ws_peer.reply = answer
    model = TTS(api_key="key", ws_url=ws_peer.ws_url)
    heard: list[str] = []
    try:
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("First sentence.")
            stream.flush()
            stream.push_text("Second sentence.")
            stream.end_input()

            async def read() -> None:
                async for event in stream:
                    heard.append(event.segment_id)

            reader = asyncio.create_task(read())
            try:
                for count in (1, 2):
                    await until(
                        lambda count=count: len(set(heard)) >= count, "the segment's audio arrives"
                    )
                    _core.skip_host_time(0.7)
                await until(reader.done, "both segments finish")
                await reader
                used = [c for c in ws_peer.connections if c.texts]
                assert len(used) == 2
                contexts = [{json.loads(m)["context_id"] for m in c.texts} for c in used]
                assert all(len(ids) == 1 for ids in contexts)
                assert contexts[0].isdisjoint(contexts[1])
                assert len(set(heard)) == 2
            finally:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
    finally:
        await model.aclose()
