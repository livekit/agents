"""Agent STT: acknowledge each audio frame and wait for EndOfTranscript at EOF."""

import aiohttp

from tests.stt_harness.protocol import Protocol, Provider
from tests.stt_harness.server import json_message


class AgentSTT(Protocol):
    async def receive(self, message):
        peer = self.connection
        if message.type == aiohttp.WSMsgType.BINARY:
            assert self.configured
            self.seq += 1
            await peer.acknowledge({"message": "AudioAdded", "seq_no": self.seq})
            await peer.accept_audio(message.data)
            return
        event = json_message(message)
        kind = event["message"]
        if not self.configured:
            assert kind == "StartRecognition"
            assert event["audio_format"]["sample_rate"] == 16000
            self.configured = True
            self.seq = 0
            peer.send({"message": "RecognitionStarted", "id": "test-session"})
        elif kind == "EndOfStream":
            if not peer.server.allow_partial_input:
                assert event["last_seq_no"] == self.seq
            peer.end_input()
            await peer.finalize()
            peer.send({"message": "EndOfTranscript"}, final=True)
        elif kind == "ForceEndOfUtterance":
            await peer.finalize()
        else:
            assert kind in ("SetRecognitionConfig", "GetSpeakers")

    async def send_transcript(self, result):
        self.connection.send(
            {
                "message": "AddSegment" if result.final else "AddPartialSegment",
                "segment": {"transcript": result.text, "speaker": "S1"},
                "metadata": {"start_time": result.start, "end_time": result.end},
            },
            final=result.final,
        )


def create_stt(server, monkeypatch):
    from livekit.plugins.speechmatics import STT, TurnDetectionMode

    return STT(
        api_key="test-key",
        base_url=server.ws_url + "/v2/agent",
        vad=None,
        turn_detection_mode=TurnDetectionMode.VAD,
    )


PROVIDERS = (
    Provider(
        "speechmatics",
        AgentSTT,
        create_stt,
        ("https://docs.speechmatics.com/api-ref/agent-stt-websocket",),
        uses_websocket_monitor=False,
        completion_on_close=False,
    ),
)
