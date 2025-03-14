import ssl
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


@dataclass
class TranscriptionConfig:
    """Real-time: Defines transcription parameters."""

    language: str = "en"
    """ISO 639-1 language code. eg. `en`"""

    operating_point: str = "enhanced"
    """Specifies which acoustic model to use."""

    output_locale: Optional[str] = None
    """RFC-5646 language code for transcript output. eg. `en-AU`"""

    diarization: Optional[str] = None
    """Indicates type of diarization to use, if any."""

    additional_vocab: Optional[Dict] = None
    """Additional vocabulary that is not part of the standard language."""

    punctuation_overrides: Optional[Dict] = None
    """Permitted punctuation marks for advanced punctuation."""

    enable_entities: Optional[bool] = None
    """Indicates if inverse text normalization entity output is enabled."""

    max_delay: float = 0.7
    """Maximum acceptable delay."""

    max_delay_mode: Optional[str] = None
    """Determines whether the threshold specified in max_delay can be exceeded
    if a potential entity is detected. Flexible means if a potential entity
    is detected, then the max_delay can be overriden until the end of that
    entity. Fixed means that max_delay specified ignores any potential
    entity that would not be completed within that threshold."""

    streaming_mode: Optional[bool] = None
    """Indicates if we run the engine in streaming mode, or regular RT mode."""

    enable_partials: bool = True
    """Indicates if partials for transcription, where words are produced
    immediately, is enabled."""

    def asdict(self) -> Dict[Any, Any]:
        """Returns model as a dict while excluding None values recursively."""
        return asdict(
            self, dict_factory=lambda x: {k: v for (k, v) in x if v is not None}
        )


@dataclass
class AudioSettings:
    """Real-time: Defines audio parameters."""

    encoding: str = "pcm_s16le"
    """Encoding format when raw audio is used. Allowed values are
    `pcm_f32le`, `pcm_s16le` and `mulaw`."""

    sample_rate: int = 16000
    """Sampling rate in hertz."""

    def asdict(self):
        return {
            "type": "raw",
            "encoding": self.encoding,
            "sample_rate": self.sample_rate,
        }


@dataclass
class ConnectionSettings:
    """Defines connection parameters."""

    url: str = "wss://eu2.rt.speechmatics.com/v2"
    """Websocket server endpoint."""

    ssl_context: ssl.SSLContext = field(default_factory=ssl.create_default_context)
    """SSL context."""

    api_key: Optional[str] = None
    """api key to authenticate a customer."""

    get_access_token: Optional[bool] = True
    """Automatically generate a temporary token for authentication."""


class ClientMessageType(str, Enum):
    # pylint: disable=invalid-name
    """Real-time: Defines various messages sent from client to server."""

    StartRecognition = "StartRecognition"
    """Initiates a recognition job based on configuration set previously."""

    AddAudio = "AddAudio"
    """Adds more audio data to the recognition job. The server confirms
    receipt by sending an :py:attr:`ServerMessageType.AudioAdded` message."""

    EndOfStream = "EndOfStream"
    """Indicates that the client has no more audio to send."""

    SetRecognitionConfig = "SetRecognitionConfig"
    """Allows the client to re-configure the recognition session."""


class ServerMessageType(str, Enum):
    """Real-time: Defines various message types sent from server to client."""

    RecognitionStarted = "RecognitionStarted"
    """Server response to :py:attr:`ClientMessageType.StartRecognition`,
    acknowledging that a recognition session has started."""

    AudioAdded = "AudioAdded"
    """Server response to :py:attr:`ClientMessageType.AddAudio`, indicating
    that audio has been added successfully."""

    AddPartialTranscript = "AddPartialTranscript"
    """Indicates a partial transcript, which is an incomplete transcript that
    is immediately produced and may change as more context becomes available.
    """

    AddTranscript = "AddTranscript"
    """Indicates the final transcript of a part of the audio."""

    EndOfTranscript = "EndOfTranscript"
    """Server response to :py:attr:`ClientMessageType.EndOfStream`,
    after the server has finished sending all :py:attr:`AddTranscript`
    messages."""

    Info = "Info"
    """Indicates a generic info message."""

    Warning = "Warning"
    """Indicates a generic warning message."""

    Error = "Error"
    """Indicates n generic error message."""
