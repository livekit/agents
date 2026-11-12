from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

Emotion = Literal[
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "concerned",
    "confused",
    "disgusted",
    "excited",
    "scared",
    "skeptical",
]
"""An expression the avatar can hold."""

EMOTIONS: tuple[Emotion, ...] = get_args(Emotion)
"""Every emotion the avatar can hold, in the order the model defines them."""

USERDATA_EMOTION_CUES = "keyframe.emotion_cues"
"""
The key for emotion cues in an audio frame's ``userdata``: a list of :class:`EmotionCue`.

A custom ``tts_node`` can attach cues to the frames it yields, and ``KeyframeAudioOutput``
writes them into the framed audio stream timed against the frame's audio.
"""


@dataclass(frozen=True)
class EmotionCue:
    """An emotion change timed against the audio frame that carries it.

    Attributes:
        emotion: The expression to switch to.
        offset: Seconds from the frame's first sample. May be negative (the change
            belongs to audio already sent) or run past the frame's end.
    """

    emotion: Emotion
    offset: float = 0.0
