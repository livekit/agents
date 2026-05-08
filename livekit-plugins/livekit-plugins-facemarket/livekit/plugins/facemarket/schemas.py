from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SessionInfo:
    session_id: str
    livekit_url: str | None = None
    room_token: str | None = None
    room_id: str | None = None
    green_screen: dict[str, Any] | None = None


@dataclass(slots=True)
class StartSessionRequest:
    avatar_id: str
    agent_identity: str
    livekit_url: str
    room_name: str
    renderer_token: str
    coordinator_token: str

    def to_payload(self) -> dict[str, str]:
        return {
            "avatarId": self.avatar_id,
            "agentIdentity": self.agent_identity,
            "livekitUrl": self.livekit_url,
            "roomName": self.room_name,
            "rendererToken": self.renderer_token,
            "coordinatorToken": self.coordinator_token,
        }


@dataclass(slots=True)
class PlatformResponse:
    code: int
    message: str
    data: dict[str, Any]
