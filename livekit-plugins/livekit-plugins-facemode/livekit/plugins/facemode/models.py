"""Typed FaceMode API request and response models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

SUPPORTED_INPUT_PROVIDERS = frozenset(
    {
        "deepgram",
        "gemini",
        "gnani",
        "elevenlabs",
        "openai",
        "cartesia",
        "sarvam",
        "custom",
    }
)


@dataclass(frozen=True, slots=True)
class LiveKitRoom:
    """Credentials for a customer-owned LiveKit room."""

    url: str
    token: str = field(repr=False)
    name: str = ""
    type: Literal["livekit"] = field(default="livekit", kw_only=True)

    def __post_init__(self) -> None:
        if self.type != "livekit":
            raise ValueError("room.type must be 'livekit'")
        if not self.url:
            raise ValueError("room.url is required")
        if not self.token:
            raise ValueError("room.token is required")

    def to_payload(self) -> dict[str, str]:
        """Return the API representation without exposing credentials in reprs."""
        payload = {"type": self.type, "url": self.url, "token": self.token}
        if self.name:
            payload["name"] = self.name
        return payload

    @classmethod
    def from_value(cls, value: LiveKitRoom | Mapping[str, Any]) -> LiveKitRoom:
        """Normalize a typed room or mapping supplied by an integration caller."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("room must be an object")
        room_type = value.get("type")
        if room_type != "livekit":
            raise ValueError("room.type must be 'livekit'")
        url = value.get("url")
        token = value.get("token")
        name = value.get("name", "")
        if not isinstance(url, str) or not url:
            raise ValueError("room.url is required")
        if not isinstance(token, str) or not token:
            raise ValueError("room.token is required")
        if not isinstance(name, str):
            raise ValueError("room.name must be a string")
        return cls(url=url, token=token, name=name)


@dataclass(frozen=True, slots=True)
class SessionRequest:
    """Request body used to create a FaceMode session."""

    avatar_id: str
    room: LiveKitRoom
    room_name: str = ""
    input_provider: str | None = None

    def __post_init__(self) -> None:
        if self.input_provider is not None and self.input_provider not in SUPPORTED_INPUT_PROVIDERS:
            raise ValueError(
                "input_provider must be one of: " + ", ".join(sorted(SUPPORTED_INPUT_PROVIDERS))
            )

    def to_payload(self) -> dict[str, Any]:
        """Return the room-object API wire representation."""
        payload: dict[str, Any] = {
            "avatarId": self.avatar_id,
            "room": self.room.to_payload(),
        }
        if self.room_name:
            payload["livekit_room_id"] = self.room_name
        if self.input_provider:
            payload["inputProvider"] = self.input_provider
        return payload


@dataclass(frozen=True, slots=True)
class IngestionDetails:
    ready: bool
    url: str = ""
    ws_token: str = field(default="", repr=False)
    authority: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class SessionDetails:
    session_id: str
    room_name: str
    room: LiveKitRoom
    ingestion: IngestionDetails
    worker_status: str | None = None
    avatar_participant_identity: str | None = None
    input_provider: str | None = None

    @classmethod
    def from_api(
        cls,
        payload: Mapping[str, Any],
        *,
        fallback: SessionDetails | None = None,
    ) -> SessionDetails:
        """Parse current and legacy response envelopes, including pending ingestion."""
        root: Mapping[str, Any] = payload
        nested = payload.get("data")
        if isinstance(nested, Mapping):
            root = nested

        session_value = root.get("session")
        session = session_value if isinstance(session_value, Mapping) else {}
        ingestion_value = root.get("ingestion") or session.get("ingestion")
        ingestion = ingestion_value if isinstance(ingestion_value, Mapping) else {}

        session_id = _first_text(
            session.get("id"),
            session.get("sessionId"),
            root.get("sessionId"),
            root.get("id"),
            root.get("jobId"),
        )
        if not session_id:
            raise ValueError("FaceMode session response is missing a session ID")

        ingestion_url = _first_text(
            ingestion.get("url"),
            ingestion.get("websocketUrl"),
            root.get("websocketUrl"),
            root.get("websocket_url"),
        )
        ws_token = _first_text(
            ingestion.get("wsToken"),
            ingestion.get("ws_token"),
            ingestion.get("token"),
            root.get("ingestionToken"),
            root.get("ingestion_token"),
        )
        ready = ingestion.get("ready") is True or (
            "ready" not in ingestion and bool(ingestion_url and ws_token)
        )
        if ready and (not ingestion_url or not ws_token):
            raise ValueError("FaceMode reported ready ingestion without WebSocket credentials")

        return cls(
            session_id=session_id,
            room_name=_first_text(
                root.get("roomName"),
                session.get("roomName"),
                fallback.room_name if fallback else "",
            ),
            room=_parse_room(root, session, fallback.room if fallback else None),
            ingestion=IngestionDetails(
                ready=ready,
                url=ingestion_url,
                ws_token=ws_token,
                authority=_first_optional_text(ingestion.get("authority")),
                headers=_parse_headers(ingestion.get("headers")),
            ),
            worker_status=_first_optional_text(
                root.get("workerStatus"),
                root.get("worker_status"),
                session.get("workerStatus"),
                session.get("worker_status"),
            ),
            avatar_participant_identity=_first_optional_text(
                root.get("avatarParticipantIdentity"),
                root.get("avatar_participant_identity"),
                session.get("avatarParticipantIdentity"),
                session.get("avatar_participant_identity"),
            ),
            input_provider=_first_optional_text(
                root.get("inputProvider"),
                root.get("input_provider"),
                session.get("inputProvider"),
                session.get("input_provider"),
            ),
        )


def _parse_room(
    root: Mapping[str, Any],
    session: Mapping[str, Any],
    fallback: LiveKitRoom | None,
) -> LiveKitRoom:
    room_value = root.get("room") or session.get("room")
    if isinstance(room_value, Mapping):
        try:
            return LiveKitRoom.from_value(room_value)
        except ValueError:
            if fallback is None:
                raise
    if fallback is not None:
        return fallback
    raise ValueError("FaceMode session response is missing a LiveKit room")


def _parse_headers(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(name): header_value
        for name, header_value in value.items()
        if str(name) and isinstance(header_value, str)
    }


def _first_text(*values: Any) -> str:
    """Return the first non-empty value as text."""
    for value in values:
        if value is not None and str(value):
            return str(value)
    return ""


def _first_optional_text(*values: Any) -> str | None:
    value = _first_text(*values)
    return value or None
