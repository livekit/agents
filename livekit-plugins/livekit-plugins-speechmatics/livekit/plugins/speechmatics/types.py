from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from livekit.agents.stt import SpeechData
from speechmatics.rt import TranscriptionConfig

__all__ = ["TranscriptionConfig"]


class EndOfUtteranceMode(str, Enum):
    """End of turn delay options for transcription."""

    NONE = "none"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


class DiarizationFocusMode(str, Enum):
    """Speaker focus mode for diarization."""

    RETAIN = "retain"
    IGNORE = "ignore"


@dataclass
class AdditionalVocabEntry:
    """Additional vocabulary entry.

    Attributes:
        content: The word to add to the dictionary.
        sounds_like: Similar words to the word.
    """

    content: str
    sounds_like: list[str] | None = None


@dataclass
class DiarizationKnownSpeaker:
    """Known speakers for speaker diarization.

    Attributes:
        label: The label of the speaker.
        speaker_identifiers: One or more data strings for the speaker.
    """

    label: str
    speaker_identifiers: list[str]


@dataclass
class SpeechFragment:
    """Fragment of an utterance.

    Parameters:
        start_time: Start time of the fragment in seconds (from session start).
        end_time: End time of the fragment in seconds (from session start).
        language: Language of the fragment. Defaults to `Language.EN`.
        is_eos: Whether the fragment is the end of a sentence. Defaults to `False`.
        is_final: Whether the fragment is the final fragment. Defaults to `False`.
        is_disfluency: Whether the fragment is a disfluency. Defaults to `False`.
        is_punctuation: Whether the fragment is a punctuation. Defaults to `False`.
        attaches_to: Whether the fragment attaches to the previous or next fragment (punctuation). Defaults to empty string.
        content: Content of the fragment. Defaults to empty string.
        speaker: Speaker of the fragment (if diarization is enabled). Defaults to `None`.
        confidence: Confidence of the fragment (0.0 to 1.0). Defaults to `1.0`.
        result: Raw result of the fragment from the TTS.
    """

    start_time: float
    end_time: float
    language: str = "en"
    is_eos: bool = False
    is_final: bool = False
    is_disfluency: bool = False
    is_punctuation: bool = False
    attaches_to: str = ""
    content: str = ""
    speaker: str | None = None
    confidence: float = 1.0
    result: Any | None = None


@dataclass
class SpeakerFragments:
    """SpeechFragment items grouped by speaker_id.

    Parameters:
        start_time: The start time of the fragments.
        end_time: The end time of the fragments.
        language: The language of the fragments. Defaults to `en`.
        speaker_id: The ID of the speaker.
        is_active: Whether the speaker is active (emits frame). Defaults to `False`.
        fragments: The list of SpeechFragment items.
    """

    start_time: float
    end_time: float
    language: str
    speaker_id: str | None = None
    is_active: bool = False
    fragments: list[SpeechFragment] = field(default_factory=list)

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return f"SpeakerFragments(speaker_id: {self.speaker_id}, start_time: {self.start_time}, end_time: {self.end_time}, language: {self.language}, text: {self._format_text()})"

    def _format_text(self, format: str = "{text}") -> str:
        """Wrap text with speaker ID in an optional f-string format.

        Args:
            format: Format to wrap the text with.

        Returns:
            str: The wrapped text.
        """
        # Cumulative contents
        content = ""

        # Assemble the text
        for frag in self.fragments:
            if content == "" or frag.attaches_to == "previous":
                content += frag.content
            else:
                content += " " + frag.content

        # Format the text, if format is provided
        if self.speaker_id:
            return format.format(**{"speaker_id": self.speaker_id, "text": content})

        # Return the text
        return content

    def _as_speech_data(self, active_format: str, passive_format: str) -> SpeechData:
        """Return a SpeechData from the fragments.

        Args:
            active_format: Format to wrap the text with.
            passive_format: Format to wrap the text with. Defaults to `active_format`.

        Returns:
            SpeechData: The SpeechData object.
        """
        return SpeechData(
            language=self.language,
            text=self._format_text(active_format if self.is_active else passive_format),
            speaker_id=self.speaker_id,
            start_time=self.start_time,
            end_time=self.end_time,
            confidence=1.0,
        )


@dataclass
class AudioSettings:
    """Real-time: Defines audio parameters."""

    encoding: str = "pcm_s16le"
    """Encoding format when raw audio is used. Allowed values are
    `pcm_f32le`, `pcm_s16le` and `mulaw`."""

    sample_rate: int = 16000
    """Sampling rate in hertz."""
