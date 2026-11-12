from collections.abc import Generator, Iterable

from .auth import Auth
from .proto.riva_asr_pb2 import StreamingRecognitionConfig, StreamingRecognizeResponse
from .proto.riva_asr_pb2_grpc import RivaSpeechRecognitionStub

class ASRService:
    auth: Auth
    stub: RivaSpeechRecognitionStub
    def __init__(self, auth: Auth) -> None: ...
    def streaming_response_generator(
        self, audio_chunks: Iterable[bytes], streaming_config: StreamingRecognitionConfig
    ) -> Generator[StreamingRecognizeResponse]: ...
