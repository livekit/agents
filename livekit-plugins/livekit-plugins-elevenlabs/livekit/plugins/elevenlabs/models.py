from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

TTSModels = Literal[
    "eleven_monolingual_v1",
    "eleven_multilingual_v1",
    "eleven_multilingual_v2",
    "eleven_turbo_v2",
    "eleven_turbo_v2_5",
    "eleven_flash_v2_5",
    "eleven_flash_v2",
    "eleven_v3",
    "eleven_v3_conversational",
    "eleven_v4",
    "eleven_v4_turbo",
]

# Synthesized via text-to-dialogue instead of text-to-speech (see tts.py).
DIALOGUE_TTS_MODEL_PREFIXES = ("eleven_v3", "eleven_v4")


def is_dialogue_model(model: str) -> bool:
    """Whether `model` must be synthesized via the text-to-dialogue API."""
    return model.startswith(DIALOGUE_TTS_MODEL_PREFIXES)


# https://elevenlabs.io/docs/api-reference/text-to-speech/convert#request.query.output_format
TTSEncoding = Literal[
    "mp3_22050_32",
    "mp3_24000_48",
    "mp3_44100",
    "mp3_44100_32",
    "mp3_44100_64",
    "mp3_44100_96",
    "mp3_44100_128",
    "mp3_44100_192",
    "opus_48000_32",
    "opus_48000_64",
    "opus_48000_96",
    "opus_48000_128",
    "opus_48000_192",
    "pcm_8000",
    "pcm_16000",
    "pcm_22050",
    "pcm_24000",
    "pcm_32000",
    "pcm_44100",
    "pcm_48000",
]

STTRealtimeSampleRates = Literal[
    8000,
    16000,
    22050,
    24000,
    44100,
    48000,
]


class STTWord(BaseModel):
    model_config = ConfigDict(strict=True)

    text: str = ""
    start: float = 0.0
    end: float = 0.0
    type: str = ""
    logprob: float | None = None
    speaker_id: str | None = None


class STTResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    message_type: str = ""
    text: str = ""
    language_code: str | None = None
    words: list[STTWord] | None = None
    session_id: str = "unknown"
    message: str = "Unknown error"
    details: str = ""


class STTBatchResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    text: str
    language_code: str | None = None
    words: list[STTWord] | None = None


class TTSAlignment(BaseModel):
    model_config = ConfigDict(strict=True)

    chars: list[str] | None = None
    starts_ms: list[float] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "charStartTimesMs", "charsStartTimesMs", "char_start_times_ms"
        ),
    )
    durations_ms: list[float] | None = Field(
        default=None,
        validation_alias=AliasChoices("charDurationsMs", "charsDurationsMs", "char_durations_ms"),
    )


class TTSFrame(BaseModel):
    model_config = ConfigDict(strict=True)

    context_id: str | None = None
    camel_context_id: str | None = Field(default=None, alias="contextId")
    audio: str | None = None
    error: str | None = None
    is_final: bool = False
    is_final_legacy: bool = Field(default=False, alias="isFinal")
    alignment: TTSAlignment | None = None
    normalized_alignment: TTSAlignment | None = Field(default=None, alias="normalizedAlignment")
