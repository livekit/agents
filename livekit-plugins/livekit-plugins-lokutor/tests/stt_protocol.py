"""Lokutor ``WS /ws/stt``: a language message, binary PCM, then "end" and "done"."""

import aiohttp

from tests.stt_harness.protocol import Protocol, Provider, RequestError
from tests.stt_harness.server import json_message

SOURCE = "https://docs.lokutor.com/sdk/stt.md"


class Lokutor(Protocol):
    async def receive(self, message):
        peer = self.connection
        if not self.configured:
            config = json_message(message)
            assert set(config) == {"language"}
            assert isinstance(config["language"], str)
            self.configured = True
            peer.send({"type": "status", "data": "ready", "lang": config["language"]})
        elif message.type == aiohttp.WSMsgType.BINARY:
            await peer.accept_audio(message.data)
        else:
            assert json_message(message) == {"type": "end"}
            # Any final transcript, then "done", then the server closes.
            peer.end_input()
            await peer.finalize()
            peer.send({"type": "status", "data": "done"}, final=True)
            peer.close()

    async def send_transcript(self, result):
        self.connection.send(
            {"type": "transcript", "data": result.text, "isFinal": result.final, "lang": "en"},
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.lokutor import STT

    return STT(api_key="test-key", base_url=server.ws_url, language="en")


PROVIDERS = (
    Provider(
        "lokutor",
        Lokutor,
        create_stt,
        (SOURCE,),
        # Only "done" completes the stream; a close before it may have lost the last final.
        completion_on_close=False,
        request_errors=(
            # The concurrent-session limit refuses the handshake.
            RequestError("rate_limited", http_status=429, retryable=True),
            RequestError("error", body={"type": "error", "data": "session rejected"}),
        ),
        live_constructor="livekit.plugins.lokutor:STT",
        required_env=("LOKUTOR_API_KEY",),
        live_options={"language": "en"},
    ),
)
