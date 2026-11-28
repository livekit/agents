"""Rime TranscribeStreaming gRPC: config, accepted, audio, half-close, then done and OK.

https://docs.rime.ai/api-reference/stt/grpc
https://docs.rime.ai/api-reference/sdk/stt
Ending the audio source ends the utterance; a successful stream returns one final
transcript, and a stream that ends without `done` and an OK status has failed.
"""

import asyncio

from tests.stt_harness.protocol import SDKProvider


def create_sdk_fixture(monkeypatch):
    import grpc
    from rime_api import speech_to_text_pb2 as proto
    from rimelabs_sdk.stt import _transport

    from livekit import rtc
    from livekit.agents import vad
    from livekit.plugins import rime
    from tests.fake_stt_sdk import Server

    class RimeServer(Server):
        """An RPC exists only while VAD reports an utterance, so track every one."""

        def __init__(self):
            super().__init__()
            self.calls = []
            self.drop_requested = False

        @property
        def stream_closed(self):
            return all(call.cancelled for call in self.calls)

        @stream_closed.setter
        def stream_closed(self, value):
            pass

    server = RimeServer()

    class Call:
        """grpc.aio stream-stream call; messages cross the SDK's real serializers."""

        def __init__(self, serialize, deserialize):
            self.serialize = serialize
            self.deserialize = deserialize
            self.responses = asyncio.Queue()
            self.cancelled = False
            self.contract = None
            self.config = None

        def respond(self, **payload):
            message = proto.StreamingTranscriptionResponse(**payload)
            self.responses.put_nowait(self.deserialize(message.SerializeToString()))

        def end(self):
            self.responses.put_nowait(grpc.aio.EOF)

        def language(self):
            return proto.ResolvedLanguage(tag="en", source=proto.LANGUAGE_SOURCE_SELECTED)

        async def write(self, request):
            request = proto.StreamingTranscriptionRequest.FromString(self.serialize(request))
            if request.WhichOneof("payload") == "config":
                assert self.config is None, "duplicate streaming config"
                self.config = request.config
                assert self.config.language == "en"
                self.respond(
                    accepted=proto.StreamingAccepted(
                        output_contract=self.config.output_contract, language=self.language()
                    )
                )
                if server.drop_requested:
                    # the service completes while the utterance input is still open
                    self.end()
                return
            assert self.config is not None, "audio arrived before the streaming config"
            assert request.audio, "an empty audio message does not end the utterance"
            server.audio.append(request.audio)

        async def done_writing(self):
            server.stop()

        async def read(self):
            return await self.responses.get()

        def finish(self):
            assert self.config.output_contract == proto.STREAMING_OUTPUT_CONTRACT_REVISED_HYPOTHESES
            self.respond(hypothesis=proto.TranscriptionHypothesis(text="words", revision=1))
            self.respond(
                done=proto.TranscriptionDone(text="words", revision=1, language=self.language())
            )
            self.end()

        async def code(self):
            return grpc.StatusCode.OK

        async def details(self):
            return ""

        async def initial_metadata(self):
            return (("x-request-id", "test-request"),)

        async def trailing_metadata(self):
            return ()

        def cancel(self):
            self.cancelled = True

        def done(self):
            return self.cancelled

    class Channel:
        async def channel_ready(self):
            pass

        def stream_stream(self, path, request_serializer, response_deserializer):
            assert path == "/rime.SpeechToText/TranscribeStreaming"

            def start(*, metadata):
                assert dict(metadata)["authorization"]
                call = Call(request_serializer, response_deserializer)
                server.calls.append(call)
                return call

            return start

        async def close(self):
            pass

    class UtteranceVAD(vad.VAD):
        """One utterance per segment: speech starts on its first frame."""

        def __init__(self):
            super().__init__(capabilities=vad.VADCapabilities(update_interval=0.1))

        def stream(self):
            return UtteranceStream(self)

    class UtteranceStream(vad.VADStream):
        async def _main_task(self):
            started = False
            async for frame in self._input_ch:
                if not isinstance(frame, rtc.AudioFrame):
                    continue
                kind = (
                    vad.VADEventType.INFERENCE_DONE if started else vad.VADEventType.START_OF_SPEECH
                )
                started = True
                self._event_ch.send_nowait(
                    vad.VADEvent(
                        type=kind,
                        samples_index=0,
                        timestamp=0,
                        speech_duration=frame.duration,
                        silence_duration=0,
                        frames=[frame],
                    )
                )
            if not started:
                # input ended with nothing to commit, so no request is needed
                server.stopped.set()

    monkeypatch.setattr(_transport, "make_channel", lambda policy: Channel())
    provider = rime.STT(api_key="test-key", vad=UtteranceVAD())

    # The client owns one shared channel until STT.aclose closes it.
    connection = server.track_connection()
    close_client = provider._client.close

    async def close():
        await close_client()
        connection.closed = True

    provider._client.close = close
    streams = []
    open_stream = provider.stream

    def stream(**kwargs):
        streams.append(open_stream(**kwargs))
        return streams[-1]

    provider.stream = stream

    def finish():
        server.calls[-1].finish()

    def drop():
        # a remote completion needs an open utterance request
        server.drop_requested = True
        streams[-1].push_frame(rtc.AudioFrame.create(16000, 1, 320))

    server.finish = finish
    server.drop = drop
    server.connected.set()
    return provider, server


PROVIDERS = (
    SDKProvider(
        "rime",
        create_sdk_fixture,
        tuple(__doc__.splitlines()[2:4]),
        minimum_python=(3, 11),
        pools_connections=True,
        live_constructor="livekit.plugins.rime:STT",
        required_env=("RIME_API_KEY",),
    ),
)
