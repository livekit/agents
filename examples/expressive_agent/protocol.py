"""The contract between the frontend and this agent.

The frontend never talks to the agent directly: it asks a token endpoint for a
token, and that endpoint puts a JSON blob on the agent dispatch. This module is
the one place that blob's shape and the voices it can name are written down, so
the token endpoint, the agent, and the UI can't drift apart silently.

Frontend -> agent, as ``RoomAgentDispatch.metadata``::

    {"expressive": true, "tts": "fishaudio"}

Agent -> frontend, as participant attributes once the session is up::

    {"expressive": "true", "tts_provider": "fishaudio", "tts_label": "..."}

Attributes are strings by protocol, hence "true" rather than true.

The agent is the only validator: the token endpoint passes the blob through
untouched, so adding a voice here needs no frontend deploy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("expressive-agent")

DEFAULT_VOICE = "fishaudio"


@dataclass(frozen=True, slots=True)
class Voice:
    provider: str
    model: str
    voice: str
    label: str


VOICES: dict[str, Voice] = {
    "fishaudio": Voice(
        provider="fishaudio",
        model="fishaudio/s2.1-pro",
        voice="51b44863613e405a896f7f4294c6e6d0",
        label="Fish Audio S2.1 Pro (Marley)",
    ),
    "inworld": Voice(
        provider="inworld",
        model="inworld/inworld-tts-2",
        voice="Ashley",
        label="Inworld TTS 2 (Ashley)",
    ),
    "cartesia": Voice(
        provider="cartesia",
        model="cartesia/sonic-3",
        voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        label="Cartesia Sonic 3 (Jacqueline)",
    ),
    "xai": Voice(
        provider="xai",
        model="xai/tts-1",
        voice="eve",
        label="xAI TTS 1 (Eve)",
    ),
}


class SessionRequest(BaseModel, extra="ignore"):
    """The dispatch metadata, as sent. Unknown fields are ignored so an older
    agent still starts against a newer frontend."""

    expressive: bool = True
    tts: str | None = None

    @classmethod
    def parse(cls, metadata: str | None) -> SessionRequest:
        """Read a dispatch metadata blob. Anything malformed falls back to
        defaults, because a demo that starts with the wrong voice beats one that
        fails to start."""
        if not metadata:
            return cls()
        try:
            return cls.model_validate_json(metadata)
        except ValidationError:
            logger.warning("ignoring malformed dispatch metadata", extra={"metadata": metadata})
            return cls()

    def resolve(self) -> SessionConfig:
        voice = VOICES.get(self.tts or "", VOICES[DEFAULT_VOICE])
        return SessionConfig(expressive=self.expressive, voice=voice)


class SessionConfig(BaseModel, arbitrary_types_allowed=True):
    """A request with the voice resolved to one this agent can actually run."""

    expressive: bool
    voice: Voice

    def attributes(self) -> dict[str, str]:
        """The participant attributes echoing this config back to the frontend."""
        return {
            "expressive": "true" if self.expressive else "false",
            "tts_provider": self.voice.provider,
            "tts_label": self.voice.label,
        }
