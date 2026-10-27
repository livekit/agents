"""HTTP/2 event-stream audio; an empty AudioEvent ends input before response EOF."""

from types import SimpleNamespace as NS

from tests.stt_harness.protocol import SDKProvider


def create_sdk_fixture(monkeypatch):
    from tests.fake_stt_sdk import Server

    server = Server()
    from livekit.plugins.aws import stt as module

    class Audio:
        async def send(self, event):
            await server.send_audio(event.value.audio_chunk)

        async def close(self):
            pass

    class Output:
        def __aiter__(self):
            return server.events()

    class Client:
        def __init__(self, **kwargs):
            pass

        async def start_stream_transcription(self, **kwargs):
            server.connected.set()

            async def await_output():
                return (None, Output())

            return NS(input_stream=Audio(), await_output=await_output)

    monkeypatch.setattr(module, "TranscribeStreamingClient", Client)
    provider = module.STT(
        region="us-east-1",
        sample_rate=16000,
        credentials=module.Credentials(access_key_id="test", secret_access_key="test"),
    )
    server.final = NS(
        value=module.TranscriptEvent(
            transcript=NS(
                results=[
                    NS(
                        is_partial=False,
                        alternatives=[NS(transcript="words", items=[])],
                        result_id="test",
                        start_time=0.0,
                        end_time=0.1,
                        language_code="en-US",
                    )
                ]
            )
        )
    )
    return provider, server


PROVIDERS = (
    SDKProvider(
        "aws",
        create_sdk_fixture,
        ("https://docs.aws.amazon.com/transcribe/latest/dg/streaming-setting-up.html",),
        minimum_python=(3, 12),
    ),
)
