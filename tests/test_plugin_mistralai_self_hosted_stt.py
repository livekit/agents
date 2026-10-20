"""SelfHostedSTT against a fake vLLM `/v1/realtime` server.

The fake follows vllm/entrypoints/speech_to_text/realtime/connection.py: a non-final
commit starts a run that reads appended audio until a final commit, the run ends with
one `transcription.done`, and the server then drops whatever audio it still holds.
Each 80 ms chunk's samples carry a word number, and the run emits " w<n>" for it.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field

import aiohttp
import numpy as np
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, APIStatusError, stt
from livekit.plugins.mistralai import SelfHostedSTT
from livekit.plugins.mistralai.self_hosted_stt import _realtime_url

pytestmark = pytest.mark.plugin("mistralai")

MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
CHUNK_SAMPLES = 1280  # 80 ms at 16 kHz


@dataclass
class FakeVLLM:
    done_delay: float = 0.0
    # end a run on its own after this many chunks, as max_model_len does
    max_chunks_per_run: int | None = None
    connections: int = 0
    headers: list[dict[str, str]] = field(default_factory=list)
    client_events: list[str] = field(default_factory=list)

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        self.headers.append(dict(request.headers))
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "id": f"sess-{self.connections}"})

        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        validated = False
        generation: asyncio.Task | None = None

        async def run() -> None:
            text = ""
            chunks = 0
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                chunks += 1
                if chunk[0] > 0:
                    delta = f" w{int(chunk[0])}"
                    text += delta
                    await ws.send_json({"type": "transcription.delta", "delta": delta})
                if self.max_chunks_per_run and chunks >= self.max_chunks_per_run:
                    break
            await asyncio.sleep(self.done_delay)
            await ws.send_json(
                {
                    "type": "transcription.done",
                    "text": text,
                    "usage": {"prompt_tokens": chunks, "completion_tokens": len(text.split())},
                }
            )
            while not queue.empty():
                queue.get_nowait()

        async for msg in ws:
            event = json.loads(msg.data)
            kind = event["type"]
            self.client_events.append(kind + (":final" if event.get("final") else ""))
            if kind == "session.update":
                if event.get("model") != MODEL:
                    await ws.send_json(
                        {
                            "type": "error",
                            "error": f"The model `{event.get('model')}` does not exist.",
                            "code": "model_not_found",
                        }
                    )
                else:
                    validated = True
            elif kind == "input_audio_buffer.append":
                pcm = np.frombuffer(base64.b64decode(event["audio"]), dtype=np.int16)
                for i in range(0, len(pcm), CHUNK_SAMPLES):
                    queue.put_nowait(pcm[i : i + CHUNK_SAMPLES])
            elif kind == "input_audio_buffer.commit":
                if not validated:
                    await ws.send_json(
                        {"type": "error", "error": "not validated", "code": "model_not_validated"}
                    )
                elif event.get("final"):
                    queue.put_nowait(None)
                elif generation is None or generation.done():
                    generation = asyncio.create_task(run())

        if generation is not None:
            generation.cancel()
        return ws


def _word(n: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=np.full(CHUNK_SAMPLES, n, dtype=np.int16).tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=CHUNK_SAMPLES,
    )


async def _serve(fake: FakeVLLM) -> TestServer:
    app = web.Application()
    app.router.add_get("/v1/realtime", fake.handler)
    server = TestServer(app)
    await server.start_server()
    return server


async def _collect(stream: stt.RecognizeStream) -> list[stt.SpeechEvent]:
    return [ev async for ev in stream]


def _finals(events: list[stt.SpeechEvent]) -> list[str]:
    return [
        ev.alternatives[0].text for ev in events if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]


def _interims(events: list[stt.SpeechEvent]) -> list[str]:
    return [
        ev.alternatives[0].text
        for ev in events
        if ev.type == stt.SpeechEventType.INTERIM_TRANSCRIPT
    ]


async def _transcribe(fake: FakeVLLM, script, *, model: str = MODEL, **kwargs):
    """Run `script(stream)` against the fake and return every event the stream emits."""
    server = await _serve(fake)
    try:
        async with aiohttp.ClientSession() as session:
            engine = SelfHostedSTT(
                base_url=str(server.make_url("")), model=model, http_session=session, **kwargs
            )
            stream = engine.stream(conn_options=APIConnectOptions(max_retry=0, timeout=5))
            collector = asyncio.create_task(_collect(stream))
            await script(stream)
            stream.end_input()
            return await asyncio.wait_for(collector, 10)
    finally:
        await server.close()


async def test_a_flushed_turn_becomes_one_final_transcript() -> None:
    async def script(stream: stt.RecognizeStream) -> None:
        stream.push_frame(_word(1))
        stream.push_frame(_word(2))
        stream.push_frame(_word(0))
        stream.flush()

    events = await _transcribe(FakeVLLM(), script)

    assert _interims(events) == ["w1", "w1 w2"]
    assert _finals(events) == ["w1 w2"]
    usage = [ev for ev in events if ev.type == stt.SpeechEventType.RECOGNITION_USAGE]
    assert usage[0].recognition_usage is not None
    assert usage[0].recognition_usage.audio_duration == pytest.approx(0.24)


async def test_every_turn_gets_its_own_run() -> None:
    fake = FakeVLLM()

    async def script(stream: stt.RecognizeStream) -> None:
        stream.push_frame(_word(1))
        stream.flush()
        await asyncio.sleep(0.2)
        stream.push_frame(_word(2))
        stream.flush()

    events = await _transcribe(fake, script)

    assert _finals(events) == ["w1", "w2"]
    assert fake.connections == 1
    # each run starts with a non-final commit and ends with a final one
    commits = [e for e in fake.client_events if e.startswith("input_audio_buffer.commit")]
    assert commits[:4] == [
        "input_audio_buffer.commit",
        "input_audio_buffer.commit:final",
        "input_audio_buffer.commit",
        "input_audio_buffer.commit:final",
    ]


async def test_speech_right_after_a_turn_ends_is_not_dropped() -> None:
    # vLLM clears its queue after `done`: audio sent between the final commit and the
    # `done` would vanish unless the stream holds it for the next run
    fake = FakeVLLM(done_delay=0.3)

    async def script(stream: stt.RecognizeStream) -> None:
        stream.push_frame(_word(1))
        stream.flush()
        await asyncio.sleep(0.05)
        stream.push_frame(_word(2))
        stream.push_frame(_word(3))
        stream.flush()

    events = await _transcribe(fake, script)

    assert _finals(events) == ["w1", "w2 w3"]


async def test_input_ending_mid_turn_still_transcribes_the_turn() -> None:
    async def script(stream: stt.RecognizeStream) -> None:
        stream.push_frame(_word(4))
        stream.push_frame(_word(5))

    events = await _transcribe(FakeVLLM(), script)

    assert _finals(events) == ["w4 w5"]


async def test_input_ending_while_a_turn_finishes_keeps_the_held_audio() -> None:
    fake = FakeVLLM(done_delay=0.3)

    async def script(stream: stt.RecognizeStream) -> None:
        stream.push_frame(_word(1))
        stream.flush()
        await asyncio.sleep(0.05)
        stream.push_frame(_word(2))

    events = await _transcribe(fake, script)

    assert _finals(events) == ["w1", "w2"]


async def test_a_flush_with_no_audio_sends_nothing() -> None:
    fake = FakeVLLM()

    async def script(stream: stt.RecognizeStream) -> None:
        stream.flush()
        stream.flush()

    events = await _transcribe(fake, script)

    assert _finals(events) == []
    assert "input_audio_buffer.commit:final" not in fake.client_events


async def test_a_run_the_server_ends_early_is_restarted() -> None:
    fake = FakeVLLM(max_chunks_per_run=2)

    async def script(stream: stt.RecognizeStream) -> None:
        stream.push_frame(_word(1))
        stream.push_frame(_word(2))
        await asyncio.sleep(0.2)
        stream.push_frame(_word(3))
        stream.flush()

    events = await _transcribe(fake, script)

    assert _finals(events) == ["w1 w2", "w3"]


async def test_an_unknown_model_fails_without_retrying() -> None:
    fake = FakeVLLM()

    async def script(stream: stt.RecognizeStream) -> None:
        stream.push_frame(_word(1))
        stream.flush()

    with pytest.raises(APIStatusError, match="does not exist"):
        await _transcribe(fake, script, model="not-served")
    assert fake.connections == 1


async def test_api_key_is_sent_as_a_bearer_token() -> None:
    fake = FakeVLLM()

    async def script(stream: stt.RecognizeStream) -> None:
        stream.push_frame(_word(1))

    await _transcribe(fake, script, api_key="secret")

    assert fake.headers[0]["Authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://gpu:8000", "ws://gpu:8000/v1/realtime"),
        ("http://gpu:8000/", "ws://gpu:8000/v1/realtime"),
        ("https://asr.example.com/v1", "wss://asr.example.com/v1/realtime"),
        ("ws://gpu:8000/v1/realtime", "ws://gpu:8000/v1/realtime"),
    ],
)
def test_realtime_url(base_url: str, expected: str) -> None:
    assert _realtime_url(base_url) == expected


def test_capabilities_ask_the_framework_to_flush_turns() -> None:
    engine = SelfHostedSTT()
    assert engine.capabilities.streaming
    assert engine.capabilities.manual_flush
    assert not engine.capabilities.offline_recognize
    assert engine.provider == "vLLM"
    assert engine.model == MODEL


async def test_the_session_flush_at_end_of_turn_yields_the_user_transcript() -> None:
    from livekit.agents import Agent, AgentSession, UserInputTranscribedEvent

    from .fake_vad import FakeVAD

    fake = FakeVLLM()
    server = await _serve(fake)
    async with aiohttp.ClientSession() as http:
        session = AgentSession(
            stt=SelfHostedSTT(base_url=str(server.make_url("")), http_session=http),
            vad=FakeVAD(),
            turn_handling={"turn_detection": "manual"},
            session_close_transcript_timeout=0.0,
        )
        finals: asyncio.Queue[str] = asyncio.Queue()

        @session.on("user_input_transcribed")
        def _on_transcript(ev: UserInputTranscribedEvent) -> None:
            if ev.is_final:
                finals.put_nowait(ev.transcript)

        await session.start(Agent(instructions="test"))
        try:
            pipeline = session._activity._audio_recognition._stt_pipeline
            for n in (6, 7, 0):
                pipeline.audio_ch.send_nowait(_word(n))
            # what the session does at a VAD end of speech or a manual turn commit
            assert pipeline.flush()
            assert await asyncio.wait_for(finals.get(), 5) == "w6 w7"
        finally:
            await session.aclose()
            await server.close()

    assert "input_audio_buffer.commit:final" in fake.client_events
