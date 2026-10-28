from collections.abc import Generator

from .auth import Auth
from .proto.riva_audio_pb2 import AudioEncoding
from .proto.riva_tts_pb2 import SynthesizeSpeechResponse
from .proto.riva_tts_pb2_grpc import RivaSpeechSynthesisStub

class SpeechSynthesisService:
    auth: Auth
    stub: RivaSpeechSynthesisStub
    def __init__(self, auth: Auth) -> None: ...
    def synthesize_online(
        self,
        text: str,
        voice_name: str | None = None,
        language_code: str = "en-US",
        encoding: AudioEncoding.ValueType = ...,
        sample_rate_hz: int = 44100,
        audio_prompt_file: str | None = None,
        audio_prompt_encoding: AudioEncoding.ValueType = ...,
        quality: int = 20,
    ) -> Generator[SynthesizeSpeechResponse]: ...
