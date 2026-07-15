"""Speech steering presets.

A preset is a plain :class:`SpeechSteeringOptions` value — pass it as
``speech_steering`` directly, or reference it by name and override individual
fields on top (``{"preset": "casual", "pace": "slow"}``). With a ``preset``,
the other keys are sparse overrides — ``nonverbal_sounds`` merges field-by-field,
so ``{"preset": "casual", "nonverbal_sounds": {"laughing": True}}`` keeps the
preset's other sounds on. The framework renders the result into per-provider
delivery guidelines at resolution time, so the same preset works on any
markup-capable TTS provider.
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

# name -> preset, for the ``preset`` key of SpeechSteeringOptions; keep in sync
# with the SpeechSteeringPreset literal
_BY_NAME: dict[str, SpeechSteeringOptions] = {
    "formal": FORMAL,
    "casual": CASUAL,
}

__all__ = ["FORMAL", "CASUAL"]
