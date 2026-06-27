from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TranslatorOptions:
    """Configuration options for a Pinch real-time translation session.

    Attributes:
        source_language: BCP-47 language code for the input audio
            (e.g. ``"en-US"``).  Pinch uses this as a hint to improve
            accuracy even when auto-detection is on.
        target_language: BCP-47 language code for the translated output
            (e.g. ``"es-ES"``).
        voice_type: Voice style for the synthesised translation.
            One of ``"clone"`` (default), ``"female"``, or ``"male"``.
    """

    source_language: str
    target_language: str
    voice_type: str = field(default="clone")

    def __post_init__(self) -> None:
        allowed_voice_types = {"clone", "female", "male"}
        if self.voice_type not in allowed_voice_types:
            raise ValueError(
                f"voice_type must be one of {allowed_voice_types!r}, got {self.voice_type!r}."
            )


@dataclass
class TranscriptEvent:
    """A transcript message emitted from the Pinch data channel.

    Both interim and final transcripts for the source language as well as
    the translated language are delivered through this object.

    Attributes:
        type: Message kind — either ``"original_transcript"`` (source
            language recognition) or ``"translated_transcript"`` (target
            language output).
        text: The recognised or translated text for this segment.
        is_final: ``True`` once Pinch has committed the segment and will
            not update it further.
        language_detected: BCP-47 code of the language Pinch detected in
            the audio (e.g. ``"en-US"``).
        timestamp: Unix timestamp (seconds, fractional) reported by Pinch
            for when the utterance ended.
        confidence: Raw confidence score returned by Pinch (0–1 range,
            ``0`` when not provided).
    """

    type: str
    text: str
    is_final: bool
    language_detected: str
    timestamp: float
    confidence: float = field(default=0.0)

    @property
    def is_original(self) -> bool:
        """``True`` if this is a source-language transcript."""
        return self.type == "original_transcript"

    @property
    def is_translated(self) -> bool:
        """``True`` if this is a translated-language transcript."""
        return self.type == "translated_transcript"
