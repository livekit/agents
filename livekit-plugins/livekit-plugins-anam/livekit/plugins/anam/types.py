from dataclasses import dataclass
from typing import Literal

# Anam's built-in director-note styles — single source of truth for both the persona-config
# ``presetStyle`` and the inline cue tags (see :data:`director_notes.DIRECTOR_NOTE_TAGS`, derived
# from this). Keep in sync with Anam's preset styles.
PresetStyle = Literal[
    "happy",
    "warm",
    "playful",
    "laughter",
    "curious",
    "supportive",
    "concerned",
    "sad",
    "surprised",
    "angry",
    "distressed",
    "neutral",
]


@dataclass
class DirectorNotes:
    """Per-session director-notes overrides forwarded to Anam.

    Mirrors the ``directorNotes`` field of the Anam persona config. Every field
    is optional; unset (``None``) fields are omitted from the request so Anam
    falls back to the avatar model / cue defaults.

    Attributes:
        expressivity: Normalized expressivity in [0, 1] controlling how strongly the
            avatar responds to ``presetStyle``/``customStylePrompt`` and any inline
            cues: 1 responds more strongly, 0 less strongly. ``None`` falls back to
            the default value of 0.35. Out-of-range values are rejected by Anam with
            an HTTP 400.
        presetStyle: A built-in expressive style (one of :data:`PresetStyle`, e.g. ``"happy"``,
            ``"warm"``, ``"playful"``). Mutually exclusive with ``customStylePrompt`` — Anam
            rejects the pair with an HTTP 400.
        customStylePrompt: A free-form expressive style prompt. Mutually exclusive
            with ``presetStyle``.
    """

    expressivity: float | None = None
    presetStyle: PresetStyle | None = None
    customStylePrompt: str | None = None


@dataclass
class PersonaConfig:
    """Configuration for Anam avatar persona.

    See https://anam.ai/docs/integrations/livekit/configuration for the persona
    config reference.
    """

    name: str
    avatarId: str
    avatarModel: str | None = None
    directorNotes: DirectorNotes | None = None


@dataclass
class SessionOptions:
    """Per-session output options forwarded to Anam's session-token API.

    Mirrors the ``sessionOptions`` field of the Anam session-token request.

    Attributes:
        video_width: Output video frame width in pixels. Provide together with
            ``video_height`` (both or neither). Omit to use the avatar model's
            default output size. Supported pairs are model-dependent and are
            validated by Anam; an unsupported pair is rejected with an HTTP 400
            rather than silently downgraded.
        video_height: Output video frame height in pixels. See ``video_width``.
    """

    video_width: int | None = None
    video_height: int | None = None
