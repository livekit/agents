"""Speech steering presets.

A preset is a plain :class:`SpeechSteeringOptions` value — pass it as
``speech_steering`` directly, or spread it and override individual fields
(``{**presets.CASUAL_VERBAL_PRESET, "pace": "slow"}``). The framework renders
it into per-provider delivery guidelines at resolution time, so the same
preset works on any markup-capable TTS provider.
"""

from .agent_session import NonverbalOptions, SpeechSteeringOptions

PROFESSIONAL_VERBAL_PRESET: SpeechSteeringOptions = {
    "disfluencies": True,
    "nonverbal_sounds": NonverbalOptions(breathing=True),
}
"""Composed, natural delivery: audible breathing and light fillers, no laughter."""

CASUAL_VERBAL_PRESET: SpeechSteeringOptions = {
    "disfluencies": True,
    "nonverbal_sounds": NonverbalOptions(laughing=True, breathing=True),
}
"""Relaxed, friendly delivery: PROFESSIONAL_VERBAL_PRESET plus laughter."""

__all__ = ["PROFESSIONAL_VERBAL_PRESET", "CASUAL_VERBAL_PRESET"]
