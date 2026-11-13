from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, JsonValue

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    http,
)
from livekit.agents.voice.avatar import (
    AvatarSession as BaseAvatarSession,
    AvatarStart,
    DataStreamAudioOutput,
)


class _Token(BaseModel):
    session_token: str


@dataclass
class SimliConfig:
    """
    Args:
        api_key (str): Simli API Key
        face_id (str): Simli Face ID
        emotion_id (str):
            Emotion ID for Trinity Faces, defaults to happy_0.
            See https://docs.simli.com/emotions
        max_session_length (int):
            Absolute maximum session duration, avatar will disconnect after this time
            even if it's speaking.
        max_idle_time (int):
            Maximum duration the avatar is not speaking for before the avatar disconnects.
    """

    api_key: str
    face_id: str
    emotion_id: str = "92f24a0c-f046-45df-8df0-af7449c04571"
    max_session_length: int = 600
    max_idle_time: int = 30

    def create_json(self) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        result["faceId"] = f"{self.face_id}/{self.emotion_id}"
        result["handleSilence"] = True
        result["maxSessionLength"] = self.max_session_length
        result["maxIdleTime"] = self.max_idle_time
        return result


class AvatarSession(BaseAvatarSession):
    """A Simli avatar session"""

    sample_rate = 16000

    def __init__(
        self,
        *,
        simli_config: SimliConfig,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(
            avatar_participant_identity=avatar_participant_identity,
            avatar_participant_name=avatar_participant_name,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
        self._simli_config = simli_config
        self._api_url = api_url or "https://api.simli.ai"

    @property
    def provider(self) -> str:
        return "simli"

    async def create_session(self, ctx: AvatarStart) -> None:
        async with http.http_session().post(
            f"{self._api_url}/compose/token",
            json=self._simli_config.create_json(),
            headers={"x-simli-api-key": self._simli_config.api_key},
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()
            token = _Token.model_validate_json(await response.text()).session_token
        async with http.http_session().post(
            f"{self._api_url}/integrations/livekit/agents",
            json={
                "session_token": token,
                "livekit_token": ctx.livekit_token(),
                "livekit_url": ctx.livekit_url,
            },
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()

    def audio_output(self, ctx: AvatarStart) -> DataStreamAudioOutput:
        return DataStreamAudioOutput(
            room=ctx.room, destination_identity=self.avatar_identity, sample_rate=self.sample_rate
        )
