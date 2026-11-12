# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/tts/markup.rs
#     rust/crates/lk-agents-abi/src/voice/session_options.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Literal, TypeAlias

from typing_extensions import TypedDict


class NonverbalOptions(TypedDict, total=False):
    """Non-verbal vocalizations the TTS may produce (sounds that aren't words).

    A sparse opt-out: omitted keys default to ON, and a category set to
    ``False`` is never advertised to the LLM — ``{"laughing": False}``
    removes laughter and nothing else. Together the keys cover every sound
    the providers offer. Each key may later widen to ``bool | <Sound>Options``
    (e.g. ``laughing: bool | LaughingOptions``) for per-sound control without
    breaking existing callers.
    """

    laughing: bool
    """laugh, chuckle, giggle — and laugh-speak delivery"""
    breathing: bool
    """audible breath, inhale, exhale"""
    sighing: bool
    crying: bool
    vocalizing: bool
    """non-lexical voiced sounds — humming a tune, sing-song or sung delivery"""
    mouth_sounds: bool
    """tsk, tongue-click, lip-smack"""
    reflex_sounds: bool
    """cough, clearing the throat, yawn"""


Pace: TypeAlias = Literal["slow", "normal", "fast"]


class SpeechSteeringOptions(TypedDict, total=False):
    """Steers verbal delivery and non-verbal sounds in generated speech.

    Every key is a sparse override on the default (full sound vocabulary,
    light fillers): the expressive instructions already tell the LLM to match
    its delivery to the register of the moment, so most agents need no
    steering at all — set a key only to take an option away regardless of
    context.
    """

    disfluencies: bool
    """Filler words such as "um" / "uh". On by default
    (``DEFAULT_SPEECH_STEERING_OPTIONS``); set ``False`` to opt out.
    """
    nonverbal_sounds: NonverbalSounds
    """Which non-verbal sounds the TTS may make. ``True`` (and omitting the
    key) keeps the provider's full vocabulary, ``False`` disables every sound,
    and a ``NonverbalOptions`` dict toggles per category (omitted categories
    stay enabled).
    """
    pace: Pace


class ExpressiveOptions(TypedDict, total=False):
    """Configuration for the expressive pipeline, passed as ``AgentSession(expressive=...)``
    or ``Agent(expressive=...)`` (the agent value overrides the session's).

    Controls how TTS markup instructions are injected into the LLM when expressive is
    enabled. All keys are optional; common shapes:

    - ``{"speech_steering": {...}}`` — steer delivery and non-verbal sounds on top of
      the provider-agnostic default instructions.
    - ``{"tts_instructions_template": "..."}`` — a fully custom prompt.
    - ``{"tts_instructions_append": "..."}`` — your own rules appended to the template.

    Any explicit template overrides the default; unset parts fall back to the
    provider-agnostic default.
    """

    speech_steering: SpeechSteeringOptions
    tts_instructions_template: str
    tts_instructions_append: str


NonverbalSounds: TypeAlias = bool | NonverbalOptions
"""`nonverbal_sounds: bool | NonverbalOptions`. `Enabled(true)` and `Per(default)` stay apart so a
round-tripped config is not rewritten.
"""


NonverbalOptions.__module__ = "livekit.agents.voice.agent_session"
SpeechSteeringOptions.__module__ = "livekit.agents.voice.agent_session"
ExpressiveOptions.__module__ = "livekit.agents.voice.agent_session"
