"""Riva gRPC: configuration, audio requests, request EOF, final results and response EOF."""

import asyncio
from types import SimpleNamespace as NS

from tests.stt_harness.protocol import SDKProvider


def create_sdk_fixture(monkeypatch):
    from tests.fake_stt_sdk import Server

    server = Server()
    import queue
    import threading

    from livekit.plugins.nvidia import stt as module

    loop = asyncio.get_running_loop()
    responses = queue.Queue()

    class Service:
        def streaming_response_generator(self, audio, config):

            def consume():
                server.audio.extend(audio)
                if server.fail_stop:
                    responses.put(ConnectionResetError("test end-of-input failed"))
                else:
                    loop.call_soon_threadsafe(server.stopped.set)

            threading.Thread(target=consume, daemon=True).start()
            while True:
                response = responses.get()
                if response is None:
                    return
                if isinstance(response, Exception):
                    raise response
                yield response

    monkeypatch.setattr(
        module.auth,
        "create_riva_auth",
        lambda **kw: NS(channel=NS(close=lambda: responses.put(None))),
    )
    monkeypatch.setattr(module.riva.client, "ASRService", lambda auth: Service())
    provider = module.STT(api_key="test")
    server.connected.set()

    def finish():
        responses.put(
            NS(
                results=[
                    NS(
                        is_final=True,
                        alternatives=[NS(transcript="words", confidence=1.0, words=[])],
                    )
                ]
            )
        )
        responses.put(None)

    server.finish = finish
    server.drop = lambda: responses.put(None)
    return provider, server


PROVIDERS = (
    SDKProvider(
        "nvidia",
        create_sdk_fixture,
        (
            "https://docs.nvidia.com/deeplearning/riva/user-guide/docs/reference/protos/riva_asr.proto.html",
        ),
    ),
)
