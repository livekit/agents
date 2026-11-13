from __future__ import annotations

import dataclasses

from livekit import rtc

from .. import _core
from .._generated.public_inference_avatar import *  # noqa: F403
from ..job import get_job_context
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import is_given
from ..voice.avatar import AvatarSession as BaseAvatarSession, AvatarStart, DataStreamAudioOutput
from . import _resolve_credentials


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
        await avatar.start(session, ctx.room)
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
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
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
            livekit_url: The URL the provider dials with the token the gateway mints;
                ``LIVEKIT_URL`` by default.
            conn_options: Retry and phase limits of the create and terminate requests.
        """
        creds = _resolve_credentials(base_url, api_key, api_secret)
        # the gateway call retries itself under conn_options
        super().__init__(
            livekit_url=livekit_url,
            livekit_api_key=creds.api_key,
            livekit_api_secret=creds.api_secret,
            conn_options=dataclasses.replace(conn_options, max_retry=0),
        )
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
        self._sample_rate = 0

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

    async def create_session(self, ctx: AvatarStart) -> str | None:
        job_ctx = get_job_context(required=False)
        self._sample_rate = await self._native.create(
            room_name=ctx.room.name,
            room_sid=(job_ctx.job.room.sid if job_ctx else "") or await ctx.room.sid,
            livekit_url=ctx.livekit_url,
            agent_identity=ctx.agent_identity,
        )
        return self._native.session_id

    def audio_output(self, ctx: AvatarStart) -> DataStreamAudioOutput:
        # at the gateway's rate; avatar.start() normally runs before session.start(), so no audio
        # has flowed yet
        return DataStreamAudioOutput(
            room=ctx.room,
            destination_identity=self.avatar_identity,
            sample_rate=self._sample_rate,
            wait_remote_track=rtc.TrackKind.KIND_VIDEO,
            clear_buffer_timeout=None,
            wait_playback_start=True,
        )

    async def aclose(self) -> None:
        """Terminate the provider session so it stops billing now, then leave. A failed
        terminate is logged and retried by the next ``aclose()``."""
        try:
            await self._native.aclose()
        finally:
            await super().aclose()
