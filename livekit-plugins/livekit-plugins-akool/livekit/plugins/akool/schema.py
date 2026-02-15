import os
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ModeType(int, Enum):
    RETELLING = 1
    DIALOGUE = 2


class Credentials(BaseModel):
    livekit_url: str = Field(default=os.getenv("LIVEKIT_URL"), description="Livekit URL")
    livekit_token: str = Field(default=os.getenv("LIVEKIT_TOKEN"), description="Livekit token")
    audio_only_from_data_stream: bool = Field(
        default=True, description="Whether to only publish audio from the data stream"
    )


class VoiceSettings(BaseModel):
    speed: Optional[float] = Field(default=None, description="Speed of the voice")
    pron_map: Optional[dict[str, str]] = Field(
        default=None, description="Pronunciation map for the voice"
    )


class AvatarConfig(BaseModel):
    avatar_id: str = Field(default="dvp_Tristan_cloth2_1080P", description="Avatar ID")
    duration: Optional[int] = Field(default=None, gt=0, le=3600, description="Duration in seconds")
    knowledge_id: Optional[str] = Field(default=None, description="Knowledge ID")
    voice_id: Optional[str] = Field(default=None, description="Voice ID to change avatar’s voice")
    voice_url: Optional[str] = Field(default=None, description="Custom voice model URL")
    language: Optional[str] = Field(default=None, description="the Language which llm response")
    mode_type: ModeType = Field(default=ModeType.DIALOGUE, description="1: Retelling, 2: Dialogue")
    background_url: Optional[str] = Field(default=None, description="Background image URL")
    voice_params: Optional[VoiceSettings] = Field(default=None, description="Voice parameters")
    scene_mode: Literal["meeting"] = Field(
        default="meeting",
        description="Scene mode, receive audio and only do lipsync, then send audio and video",
    )


class CreateSessionRequest(AvatarConfig):
    """
    https://docs.akool.com/ai-tools-suite/live-avatar#create-session
    """

    stream_type: Literal["livekit"] = "livekit"
    credentials: Credentials = Field(default=Credentials(), description="Livekit credentials")


class CreateSessionResponse(BaseModel):
    _id: str
    status: int
    stream_type: str
