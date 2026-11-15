from typing import Literal

from pydantic import BaseModel

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

# The websocket a model streams on: text-to-speech multi-stream-input for the speech models,
# text-to-dialogue multi-stream-input for eleven_v3, and text-to-dialogue stream-input (one
# context a socket) for eleven_v4, which text-to-dialogue multi-stream-input does not accept.
TTSEndpoint = Literal["speech", "dialogue", "dialogue_single"]


def tts_endpoint(model: str) -> TTSEndpoint:
    if model.startswith("eleven_v4"):
        return "dialogue_single"
    if model.startswith("eleven_v3"):
        return "dialogue"
    return "speech"


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
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    type: str = ""
    logprob: float | None = None
    speaker_id: str | None = None


class STTBatchResponse(BaseModel):
    text: str
    language_code: str | None = None
    words: list[STTWord] | None = None
