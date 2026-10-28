"""Gemini Live uses WebSockets; Google Cloud STT uses a separate gRPC SDK contract."""

import asyncio
import base64
from types import SimpleNamespace as NS

from tests.stt_harness.protocol import Protocol, Provider, SDKProvider
from tests.stt_harness.server import json_message


class Gemini(Protocol):
    async def receive(self, message):
        event = json_message(message)
        peer = self.connection
        if not self.configured:
            assert "setup" in event
            self.configured = True
            peer.send({"setupComplete": {}})
            return
        realtime = event.get("realtimeInput", event.get("realtime_input"))
        assert realtime is not None
        if realtime.get("audioStreamEnd", realtime.get("audio_stream_end")):
            peer.end_input()
            await peer.finalize()
        else:
            audio = realtime["audio"]
            assert audio.get("mimeType", audio.get("mime_type")) == "audio/pcm;rate=16000"
            await peer.accept_audio(base64.b64decode(audio["data"], validate=True))

    async def send_transcript(self, result):
        content = {
            "inputTranscription" if result.final else "interimInputTranscription": {
                "text": result.text
            }
        }
        if result.final:
            content.update({"generationComplete": True, "turnComplete": True})
        self.connection.send({"serverContent": content}, final=result.final)


def create_stt(server, monkeypatch):
    from livekit.plugins.google.beta.gemini_stt import STT

    return STT(
        api_key="test-key",
        vertexai=False,
        http_options={"base_url": server.url, "async_client_args": {"ssl": server.ssl_context}},
    )


PROVIDERS = (
    Provider(
        "gemini",
        Gemini,
        create_stt,
        ("https://ai.google.dev/gemini-api/docs/live-api/live-transcribe",),
        uses_websocket_monitor=False,
        tls=True,
        require_completion=False,
        completion_on_close=False,
        live_constructor="livekit.plugins.google.beta.gemini_stt:STT",
        required_env=("GOOGLE_API_KEY",),
        live_options={"vertexai": False},
        live_options_from_env={"api_key": "GOOGLE_API_KEY"},
    ),
)


def create_sdk_fixture(monkeypatch):
    from tests.fake_stt_sdk import Pool, Server

    server = Server()
    from google.cloud.speech_v1.types import cloud_speech

    from livekit.plugins.google import stt as module

    class Call:
        def __aiter__(self):
            return server.events()

        cancel = staticmethod(server.cancel)

    async def streaming_recognize(*, requests):

        async def consume():
            async for request in requests:
                audio = getattr(request, "audio", getattr(request, "audio_content", b""))
                if audio:
                    server.audio.append(audio)
            server.stop()

        server.consumer = asyncio.create_task(consume())
        server.connected.set()
        return Call()

    monkeypatch.setattr(module, "gauth_default", lambda: (None, "test-project"))
    provider = module.STT()
    provider._pool = Pool(server, NS(streaming_recognize=streaming_recognize))
    server.final = cloud_speech.StreamingRecognizeResponse(
        results=[
            {
                "is_final": True,
                "alternatives": [{"transcript": "words", "confidence": 1.0}],
                "language_code": "en-US",
            }
        ]
    )
    return provider, server


PROVIDERS += (
    SDKProvider(
        "google",
        create_sdk_fixture,
        ("https://docs.cloud.google.com/speech-to-text/docs/v1/speech-to-text-requests",),
        pools_connections=True,
    ),
)
