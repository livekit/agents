from dataclasses import dataclass
from typing import Callable

from .. import tokenize
from .stt_forwarder import STTSegmentsForwarder
from .tts_forwarder import TTSSegmentsForwarder


@dataclass(frozen=True)
class AssistantTranscriptionOptions:
    user_transcription: bool = True
    """Whether to forward the user transcription to the client"""
    agent_transcription: bool = True
    """Whether to forward the agent transcription to the client"""
    agent_transcription_speed: float = 1.0
    """The speed at which the agent's speech transcription is forwarded to the client.
    We try to mimic the agent's speech speed by adjusting the transcription speed."""
    sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer()
    """The tokenizer used to split the speech into sentences.
    This is used to decide when to mark a transcript as final for the agent transcription."""
    word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
        ignore_punctuation=False
    )
    """The tokenizer used to split the speech into words.
    This is used to simulate the "interim results" of the agent transcription."""
    hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word
    """A function that takes a string (word) as input and returns a list of strings,
    representing the hyphenated parts of the word."""
    use_tts_alignment: bool = False
    """Whether to use the TTS alignment to align the agent transcription with the TTS audio."""


__all__ = [
    "AssistantTranscriptionOptions",
    "TTSSegmentsForwarder",
    "STTSegmentsForwarder",
]
