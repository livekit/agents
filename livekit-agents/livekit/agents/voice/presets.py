"""Speech steering presets.

A preset is a plain :class:`SpeechSteeringOptions` value — pass it as
``speech_steering`` directly, or spread it and override individual fields
(``{**presets.CASUAL, "pace": "slow"}``). The framework renders it into
per-provider delivery guidelines at resolution time, so the same preset
works on any markup-capable TTS provider.

Careful with ``nonverbal_sounds``: it is atomic, so overriding it replaces the
preset's sounds wholesale — ``{**presets.CASUAL, "nonverbal_sounds": {"laughing": True}}``
turns everything else OFF. To adjust a single sound, spread one level deeper::

    {
        **presets.CASUAL,
        "nonverbal_sounds": {**presets.CASUAL["nonverbal_sounds"], "laughing": True},
    }
"""

from .agent_session import NonverbalOptions, SpeechSteeringOptions

FORMAL: SpeechSteeringOptions = {
    "disfluencies": True,
    "nonverbal_sounds": NonverbalOptions(breathing=True),
}
"""Composed, natural delivery: audible breathing and light fillers; no laughter,
sighing, or any other non-verbal sound."""

CASUAL: SpeechSteeringOptions = {
    "disfluencies": True,
    "nonverbal_sounds": NonverbalOptions(breathing=True, sighing=True),
}
"""Relaxed, friendly delivery: FORMAL plus sighs.

Laughter is deliberately excluded — current TTS laugh renditions sound
unnatural; re-enable with ``laughing=True`` once they improve."""

__all__ = ["FORMAL", "CASUAL"]
