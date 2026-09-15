from typing import Literal

TTSModels = Literal["Maya Calyx"]
"""Currently documented model. TTS also accepts future server-supported model strings."""

TTSLanguages = Literal["hi", "te", "en", "ta", "bn", "gu", "kn", "ml", "mr", "or", "pa"]
"""Documented language codes; en denotes Indian English. Omit for mixed-language input."""
