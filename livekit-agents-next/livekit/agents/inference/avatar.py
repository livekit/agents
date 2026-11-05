from __future__ import annotations

from typing import TYPE_CHECKING

from livekit import rtc
from livekit.agents._generated.avatar_options import (
    LemonSliceOptions as LemonSliceOptions,
)

from .. import _core
from ..job import _server_setting, get_job_context
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import is_given
from ..voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from . import Credentials, get_inference_headers

if TYPE_CHECKING:
    import aiohttp

    from ..voice import VoiceSession


class AvatarSession(BaseAvatarSession):
    """An avatar session provisioned through LiveKit Inference.

    Unlike the BYOK avatar plugins (which call the avatar provider directly with
    a customer API key), this calls the LiveKit Inference gateway with LiveKit
    credentials; the gateway creates the provider session with LiveKit's
    wholesale key. Media and RPC still flow in-room over DataStream, exactly as
    the BYOK plugins do.

    Example::

        avatar = inference.AvatarSession(
            "lemonslice",
            extra_kwargs=LemonSliceOptions(image_url="https://...", prompt="..."),
        )
        await avatar.start(session, room=ctx.room)
        await avatar.wait_for_join()

    Unlike the BYOK plugins, this does **not** mint the avatar worker's room
    token. The gateway mints it, scoped to this room and identity and stamped
    with ``lk.publish_on_behalf`` and ``lk.avatar_provider``; a token the client
    signed could not be trusted to carry those. The room fields this class sends
    (``room_name`` / ``avatar_identity`` / ``agent_identity``) are the inputs to
    that mint, so all three are required.

    Credentials are gateway auth only (``api_key`` / ``api_secret``): resolved
    from the arguments, then ``LIVEKIT_INFERENCE_API_KEY`` /
    ``LIVEKIT_INFERENCE_API_SECRET``, then ``LIVEKIT_API_KEY`` /
    ``LIVEKIT_API_SECRET``. Because the gateway signs the worker token with the
    key it authenticated, the project that serves inference must be the project
    that owns the room — pointing ``LIVEKIT_INFERENCE_*`` at a different project
    than the room is no longer supported.
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
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            model: ``"<provider>"`` or ``"<provider>/<avatar_id>"``
                (e.g. ``"lemonslice"`` or ``"lemonslice/agent_abc"``). For
                LemonSlice the ``<avatar_id>`` is the agent id (the gateway maps
                it onto LemonSlice's ``agent_id``); pass a pre-built agent as
                ``"lemonslice/<agent_id>"``.
            avatar_participant_identity: Room identity for the avatar worker.
                Defaults to ``"<provider>-inference-avatar"`` (the ``inference``
                marker distinguishes it from the BYOK plugins' worker identity).
            avatar_participant_name: Display name for the avatar worker.
            extra_kwargs: Provider-specific options. For LemonSlice use
                :class:`LemonSliceOptions` (``image_url`` / ``prompt`` /
                ``idle_prompt`` / ``idle_timeout``); known keys map onto the
                gateway's first-class request fields and anything else is
                forwarded verbatim, subject to the gateway's per-provider
                allowlist. ``image_url`` is mutually exclusive with a catalog id
                in the model string — passing both raises here (and the gateway
                rejects the combination too).
            base_url: Inference gateway base URL. Defaults to the environment's
                gateway (see ``get_default_inference_url``).
            api_key: Gateway API key (see the class docstring for resolution).
            api_secret: Gateway API secret (see the class docstring).
            http_session: Unused; the core owns the connection.
            conn_options: Retry and phase limits of the create and terminate requests.
        """
        super().__init__()
        creds = Credentials.resolve(base_url, api_key, api_secret)
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
            headers=list(get_inference_headers().items()),
        )
        await super().start(agent_session, room)
        # The audio tail goes to the avatar worker at the gateway's rate: in the canonical flow
        # avatar.start() runs before session.start(), so no audio has flowed yet.
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
            await self._native.aclose(list(get_inference_headers().items()))
        finally:
            await super().aclose()
