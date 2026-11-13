from __future__ import annotations

from typing import TYPE_CHECKING

from livekit import rtc

from .. import _core
from .._generated.public_inference_avatar import *  # noqa: F403
from ..job import _server_setting, get_job_context
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import is_given
from ..voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from . import _resolve_credentials

if TYPE_CHECKING:
    from ..voice import VoiceSession


class AvatarSession(BaseAvatarSession):
    """An avatar session provisioned through LiveKit Inference with LiveKit credentials,
    instead of a provider API key. Media and RPC flow in-room over DataStream.

    The gateway mints the avatar worker's room token, so the project that serves inference
    must be the project that owns the room.

    Example::

        avatar = inference.AvatarSession(
            "lemonslice",
            extra_kwargs=LemonSliceOptions(image_url="https://...", prompt="..."),
        )
        await avatar.start(session, room=ctx.room)
        await avatar.wait_for_join()
    """

    def __init__(
        self,
        model: str,
        *,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[LemonSliceOptions | dict[str, object]] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            model: ``"<provider>"`` or ``"<provider>/<avatar_id>"`` (e.g. ``"lemonslice"`` or
                ``"lemonslice/<agent_id>"``).
            avatar_participant_identity: Room identity for the avatar worker; defaults to
                ``"<provider>-inference-avatar"``.
            avatar_participant_name: Display name for the avatar worker.
            extra_kwargs: Provider-specific options, such as :class:`LemonSliceOptions`; unknown
                keys are forwarded, subject to the gateway's allowlist. ``image_url`` cannot be
                combined with an avatar id in ``model``.
            base_url: Inference gateway base URL; defaults to ``get_default_inference_url()``.
            api_key: ``LIVEKIT_INFERENCE_API_KEY``, else ``LIVEKIT_API_KEY`` by default.
            api_secret: ``LIVEKIT_INFERENCE_API_SECRET``, else ``LIVEKIT_API_SECRET`` by default.
            conn_options: Retry and phase limits of the create and terminate requests.
        """
        super().__init__()
        creds = _resolve_credentials(base_url, api_key, api_secret)
        self._native = _core.InferenceAvatar(
            model,
            avatar_identity=avatar_participant_identity or None,
            avatar_name=avatar_participant_name or None,
            extra_kwargs=dict(extra_kwargs) if is_given(extra_kwargs) else None,
            base_url=creds.base_url,
            api_key=creds.api_key,
            api_secret=creds.api_secret,
            conn_options=conn_options,
        )

    @property
    def avatar_identity(self) -> str:
        return self._native.avatar_identity

    @property
    def provider(self) -> str:
        return self._native.provider

    @property
    def session_id(self) -> str | None:
        """The gateway-generated avatar session id (available after start())."""
        return self._native.session_id

    @property
    def provider_session_id(self) -> str | None:
        """The provider's own session id (available after start())."""
        return self._native.provider_session_id

    async def start(
        self,
        agent_session: VoiceSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Create the avatar through the gateway and stream the agent's audio to it.

        ``livekit_url`` (else ``LIVEKIT_URL``) is the URL the provider dials with the token the
        gateway mints. Usable inside an agent job or standalone with a connected room. A second
        call while one runs, or after one succeeded, raises ``RuntimeError``.
        """
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            agent_identity = job_ctx.local_participant_identity
            room_sid = job_ctx.job.room.sid or await room.sid
        elif room.isconnected():
            agent_identity = room.local_participant.identity
            room_sid = await room.sid
        else:
            raise RuntimeError(
                "AvatarSession.start() needs a connected room or an agent job context; "
                "connect the room before calling start()"
            )

        sample_rate = await self._native.create(
            room_name=room.name,
            room_sid=room_sid,
            livekit_url=livekit_url or _server_setting("LIVEKIT_URL"),
            agent_identity=agent_identity,
        )
        await super().start(agent_session, room)
        # at the gateway's rate; avatar.start() normally runs before session.start(), so no audio
        # has flowed yet
        self._attach_audio_output(
            agent_session,
            DataStreamAudioOutput(
                room=room,
                destination_identity=self.avatar_identity,
                sample_rate=sample_rate,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                clear_buffer_timeout=None,
                wait_playback_start=True,
            ),
        )

    async def aclose(self) -> None:
        """Terminate the provider session so it stops billing now, then leave. A failed
        terminate is logged and retried by the next ``aclose()``."""
        try:
            await self._native.aclose()
        finally:
            await super().aclose()
