from __future__ import annotations

import os

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectOptions,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from ._output import (
    CONTROL_TOPIC as CONTROL_TOPIC,  # re-exported: these used to live in this module
    SAMPLE_RATE as SAMPLE_RATE,
    KeyframeAudioOutput,
)
from .api import DEFAULT_API_URL, KeyframeAPI
from .errors import KeyframeException
from .log import logger
from .types import EMOTIONS, Emotion

_AVATAR_AGENT_IDENTITY = "keyframe-avatar-agent"
_AVATAR_AGENT_NAME = "keyframe-avatar-agent"

# persona-2.0-live models (persona-2.0-live-preview, ...) render the eleven EMOTIONS on
# cues timed against the audio stream; persona-1.5-live takes raw audio and untimed emotions
_TIMED_EMOTION_MODEL_PREFIX = "persona-2.0-live"
# the expressions persona-1.5-live renders; it shows any other name as neutral
_LEGACY_EMOTIONS: tuple[Emotion, ...] = ("neutral", "happy", "sad", "angry")


class AvatarSession(BaseAvatarSession):
    """A Keyframe avatar session for the LiveKit Agents framework."""

    def __init__(
        self,
        *,
        persona_id: NotGivenOr[str] = NOT_GIVEN,
        persona_slug: NotGivenOr[str] = NOT_GIVEN,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__()
        self._conn_options = conn_options
        self._avatar_participant_identity = (
            avatar_participant_identity
            if utils.is_given(avatar_participant_identity)
            else _AVATAR_AGENT_IDENTITY
        )
        self._avatar_participant_name = (
            avatar_participant_name
            if utils.is_given(avatar_participant_name)
            else _AVATAR_AGENT_NAME
        )

        # Resolve persona
        has_id = utils.is_given(persona_id)
        has_slug = utils.is_given(persona_slug)
        if has_id == has_slug:
            raise KeyframeException("Provide exactly one of persona_id or persona_slug")

        self._persona_id: str | None = None
        self._persona_slug: str | None = None
        if utils.is_given(persona_id):
            self._persona_id = persona_id
        if utils.is_given(persona_slug):
            self._persona_slug = persona_slug

        # Resolve API config
        api_url_val = (
            api_url if utils.is_given(api_url) else os.getenv("KEYFRAME_API_URL", DEFAULT_API_URL)
        )
        api_key_val = api_key if utils.is_given(api_key) else os.getenv("KEYFRAME_API_KEY")

        if not api_key_val:
            raise KeyframeException(
                "KEYFRAME_API_KEY must be set by arguments or environment variables"
            )

        self._api_url = api_url_val
        self._api_key = api_key_val
        self._audio_output: KeyframeAudioOutput | None = None
        self._emotions: tuple[Emotion, ...] = EMOTIONS

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return "keyframe"

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        await super().start(agent_session, room)

        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise KeyframeException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        job_ctx = get_job_context()
        local_participant_identity = job_ctx.local_participant_identity

        # Mint a LiveKit token for the avatar worker with publish_on_behalf
        livekit_token = (
            api.AccessToken(
                api_key=livekit_api_key,
                api_secret=livekit_api_secret,
            )
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room.name,
                    can_publish=True,
                    can_subscribe=True,
                    # after joining, the avatar sets a `kfl.protocol` attribute on its own
                    # participant to advertise the audio framing it accepts; LiveKit
                    # silently drops a participant's attribute updates without this grant
                    can_update_own_metadata=True,
                )
            )
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity})
            .to_jwt()
        )

        # Call API to create a reservation and dispatch
        async with KeyframeAPI(
            api_key=self._api_key,
            api_url=self._api_url,
            conn_options=self._conn_options,
        ) as keyframe_api:
            result = await keyframe_api.create_plugin_session(
                persona_id=self._persona_id,
                persona_slug=self._persona_slug,
                room_name=room.name,
                livekit_url=livekit_url,
                livekit_token=livekit_token,
                source_participant_identity=local_participant_identity,
            )
            model_id = result.get("model_id")
            logger.debug(
                "Keyframe plugin session created",
                extra={"lk.pii.session_id": result.get("reservation_id"), "model_id": model_id},
            )

        if model_id is None:
            logger.warning(
                "the Keyframe API did not report the persona's model; emotions will be untimed"
            )

        self._room = room

        timed = model_id is not None and model_id.startswith(_TIMED_EMOTION_MODEL_PREFIX)
        # with no model reported, allow every name: the avatar shows the ones it lacks as neutral
        self._emotions = EMOTIONS if timed or model_id is None else _LEGACY_EMOTIONS
        self._audio_output = KeyframeAudioOutput(
            room,
            destination_identity=self._avatar_participant_identity,
            framed=timed,
        )
        agent_session.output.replace_audio_tail(self._audio_output)

    async def set_emotion(self, emotion: Emotion) -> None:
        """Set the avatar's emotional expression.

        The avatar switches from the next video frame it renders. While it is idle, or
        when the call comes before a reply's audio (as a tool call the LLM makes before
        answering does), the change shows as soon as it can. In the middle of a reply the
        avatar has usually rendered ahead of playback, so the change shows when playback
        reaches that point, which with a fast TTS and renderer can be the end of the reply.
        An avatar that predates protocol 2 applies a mid-reply change after the audio it
        has already received instead, which is usually the end of the reply too.

        For a change on a particular word, use ``emotion_tts_node``: it times the LLM's
        expression markers against the synthesized audio in-band, so they are exact
        however fast the TTS and the renderer run.

        Args:
            emotion: One of :data:`EMOTIONS`. A ``persona-1.5-live`` model renders only
                neutral, happy, sad and angry. A name the persona's model lacks is ignored
                with a warning.
        """
        if emotion not in self._emotions:
            logger.warning(
                "ignoring emotion %r: this persona renders %s", emotion, ", ".join(self._emotions)
            )
            return

        if self._audio_output is None:
            logger.warning("set_emotion() called before start()")
            return

        await self._audio_output.set_emotion(emotion)
