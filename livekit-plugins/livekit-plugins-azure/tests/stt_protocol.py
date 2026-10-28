"""Native Speech SDK: close PushAudioInputStream, receive final recognition and session_stopped."""

from types import SimpleNamespace as NS

from tests.stt_harness.protocol import SDKProvider


def create_sdk_fixture(monkeypatch):
    from tests.fake_stt_sdk import Server

    server = Server()
    from livekit.plugins.azure import stt as module

    class Signal:
        def connect(self, callback):
            self.callback = callback

        def emit(self, value=None):
            self.callback(value)

    class Recognizer:
        def __init__(self):
            server.recognizers.append(self)
            for event in (
                "recognizing",
                "recognized",
                "speech_start_detected",
                "speech_end_detected",
                "session_started",
                "session_stopped",
                "canceled",
            ):
                setattr(self, event, Signal())

        def start_continuous_recognition(self):
            self.session_started.emit()
            server.connected.set()

        def stop_continuous_recognition(self):
            server.closed = True

    server.recognizers = []
    monkeypatch.setattr(module, "_create_speech_recognizer", lambda **kw: Recognizer())
    monkeypatch.setattr(
        module.speechsdk.audio,
        "PushAudioInputStream",
        lambda **kw: NS(write=server.audio.append, close=server.stop),
    )
    monkeypatch.setattr(
        module.speechsdk, "AutoDetectSourceLanguageResult", lambda r: NS(language="en")
    )
    provider = module.STT(speech_key="test", speech_region="test")

    def finish():
        recognizer = server.recognizers[-1]
        recognizer.recognized.emit(NS(result=NS(text="words", offset=0, duration=1000000)))
        recognizer.session_stopped.emit()

    server.finish = finish
    server.drop = lambda: server.recognizers[-1].session_stopped.emit()
    return provider, server


PROVIDERS = (
    SDKProvider(
        "azure",
        create_sdk_fixture,
        (
            "https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-use-audio-input-streams",
        ),
        live_constructor="livekit.plugins.azure:STT",
        required_env=("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION"),
    ),
)
