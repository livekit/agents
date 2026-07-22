from __future__ import annotations

import asyncio
import os
import uuid
from typing import TYPE_CHECKING, Any

import aiohttp

from livekit import api, rtc

from .._exceptions import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    create_api_error_from_http,
)
from ..job import get_job_context
from ..log import logger
from ..types import (
    ATTRIBUTE_PUBLISH_ON_BEHALF,
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import is_given
from ..voice.avatar import AvatarSession as BaseAvatarSession, DataStreamAudioOutput
from ._utils import (
    HEADER_INFERENCE_PROVIDER,
    create_access_token,
    get_default_inference_url,
    get_inference_headers,
)

if TYPE_CHECKING:
    from ..voice import AgentSession

# Fallback audio sample rate when the gateway response omits one. The gateway
# is authoritative (it returns the per-provider rate) so this is only a
# fallback for older gateways.
_DEFAULT_SAMPLE_RATE = 16000
# Total request timeout for gateway calls. sock_connect (conn_options.timeout)
# bounds the TCP connect; this bounds the whole request so a gateway that
# accepts the connection but never responds can't hang start()/aclose()
# indefinitely. Matches the BYOK lemonslice plugin's request timeout.
_REQUEST_TIMEOUT = 60.0
# lk.avatar_provider tags the avatar worker participant with its provider so
# server-side (participant-lifetime) avatar-minutes metering can attribute the
# worker's room time to the right SKU. See the inference-avatar plan.
_ATTRIBUTE_AVATAR_PROVIDER = "lk.avatar_provider"


def _parse_avatar_model(model: str) -> tuple[str, str | None]:
    """Parse an avatar model string into (provider, avatar_id).

    ``"lemonslice"`` -> ``("lemonslice", None)``; ``"lemonslice/agent_abc"`` ->
    ``("lemonslice", "agent_abc")``. The id (when present) is a provider catalog
    id forwarded to the gateway as ``avatar_id``.
    """
    provider, _, avatar_id = model.partition("/")
    provider = provider.strip()
    if not provider:
        raise ValueError(
            f"invalid avatar model string: {model!r} (expected 'provider' or 'provider/<id>')"
        )
    avatar_id = avatar_id.strip()
    return provider, (avatar_id or None)


class AvatarSession(BaseAvatarSession):
    """An avatar session provisioned through LiveKit Inference.

    Unlike the BYOK avatar plugins (which call the avatar provider directly with
    a customer API key), this calls the LiveKit Inference gateway with LiveKit
    credentials; the gateway creates the provider session with LiveKit's
    wholesale key. Media and RPC still flow in-room over DataStream, exactly as
    the BYOK plugins do.

    Example::

        avatar = inference.AvatarSession("lemonslice", image_url="https://...", prompt="...")
        await avatar.start(session, room=ctx.room)
        await avatar.wait_for_join()

    Credentials come from two independent sources:

    - Gateway auth (``api_key`` / ``api_secret``): resolved from the arguments,
      then ``LIVEKIT_INFERENCE_API_KEY`` / ``LIVEKIT_INFERENCE_API_SECRET``,
      then ``LIVEKIT_API_KEY`` / ``LIVEKIT_API_SECRET``.
    - The avatar worker's room token: minted locally from the room project's
      credentials, resolved from ``start()`` arguments then ``LIVEKIT_URL`` /
      ``LIVEKIT_API_KEY`` / ``LIVEKIT_API_SECRET``. These may differ from the
      gateway credentials when ``LIVEKIT_INFERENCE_*`` points at a different
      project.
    """

    def __init__(
        self,
        model: str,
        *,
        image_url: NotGivenOr[str] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        idle_prompt: NotGivenOr[str] = NOT_GIVEN,
        idle_timeout: NotGivenOr[int] = NOT_GIVEN,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Args:
            model: ``"<provider>"`` or ``"<provider>/<avatar_id>"``
                (e.g. ``"lemonslice"`` or ``"lemonslice/agent_abc"``).
            image_url: Appearance source for image-sourced avatars (LemonSlice).
                Mutually exclusive with a catalog id in the model string.
            prompt: Speaking prompt (mapped to the provider's agent_prompt).
            idle_prompt: Idle prompt (mapped to the provider's agent_idle_prompt).
            idle_timeout: Provider idle timeout in seconds; the gateway clamps it
                to the provider's configured bounds.
            avatar_participant_identity: Room identity for the avatar worker.
                Defaults to ``"<provider>-avatar-agent"``.
            avatar_participant_name: Display name for the avatar worker.
            extra_kwargs: Provider-specific extras forwarded verbatim (subject to
                the gateway's per-provider allowlist).
            base_url: Inference gateway base URL. Defaults to the environment's
                gateway (see ``get_default_inference_url``).
            api_key: Gateway API key (see the class docstring for resolution).
            api_secret: Gateway API secret (see the class docstring).
            http_session: Optional aiohttp session; one is created per call if
                omitted.
            conn_options: Retry/timeout options for the create call.
        """
        super().__init__()

        self._provider, self._avatar_id = _parse_avatar_model(model)
        self._image_url = image_url
        self._prompt = prompt
        self._idle_prompt = idle_prompt
        self._idle_timeout = idle_timeout
        self._extra_kwargs = extra_kwargs
        self._conn_options = conn_options
        self._http_session = http_session

        self._base_url = base_url if is_given(base_url) else get_default_inference_url()
        self._api_key = (
            api_key
            if is_given(api_key)
            else os.getenv("LIVEKIT_INFERENCE_API_KEY", os.getenv("LIVEKIT_API_KEY", ""))
        )
        if not self._api_key:
            raise ValueError(
                "api_key is required, either as argument or set LIVEKIT_API_KEY environment variable"
            )
        self._api_secret = (
            api_secret
            if is_given(api_secret)
            else os.getenv("LIVEKIT_INFERENCE_API_SECRET", os.getenv("LIVEKIT_API_SECRET", ""))
        )
        if not self._api_secret:
            raise ValueError(
                "api_secret is required, either as argument or set LIVEKIT_API_SECRET environment variable"
            )

        self._avatar_participant_identity = (
            avatar_participant_identity
            if is_given(avatar_participant_identity)
            else f"{self._provider}-avatar-agent"
        )
        self._avatar_participant_name = (
            avatar_participant_name
            if is_given(avatar_participant_name)
            else self._avatar_participant_identity
        )

        self._session_id: str | None = None
        self._provider_session_id: str | None = None
        # HMAC issued by the gateway alongside provider_session_id; required by
        # /avatar/sessions/terminate to prove this project owns the session.
        self._terminate_token: str | None = None

    @property
    def avatar_identity(self) -> str:
        return self._avatar_participant_identity

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def session_id(self) -> str | None:
        """The gateway-generated avatar session id (available after start())."""
        return self._session_id

    @property
    def provider_session_id(self) -> str | None:
        """The provider's own session id (available after start())."""
        return self._provider_session_id

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if self._session_id is not None or self._provider_session_id is not None:
            raise RuntimeError(
                "AvatarSession.start() may only be called once per instance; "
                "create a new AvatarSession to start another avatar"
            )

        await super().start(agent_session, room)

        lk_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        lk_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        lk_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not lk_url or not lk_key or not lk_secret:
            raise ValueError(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or the LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET "
                "environment variables"
            )

        # Usable inside an agent job or standalone (scripts/tests). The base
        # class already tolerates a missing job context; derive the same values
        # from the connected room when there is none.
        job_ctx = get_job_context(required=False)
        if job_ctx is not None:
            local_participant_identity = job_ctx.local_participant_identity
            room_sid = job_ctx.job.room.sid or await room.sid
        elif room.isconnected():
            local_participant_identity = room.local_participant.identity
            room_sid = await room.sid
        else:
            raise RuntimeError(
                "AvatarSession.start() needs a connected room or an agent job context; "
                "connect the room before calling start()"
            )

        # Mint the avatar worker's room token locally, identical to the BYOK
        # plugins, plus an lk.avatar_provider attribute for server-side metering.
        worker_token = (
            api.AccessToken(api_key=lk_key, api_secret=lk_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            .with_attributes(
                {
                    ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity,
                    _ATTRIBUTE_AVATAR_PROVIDER: self._provider,
                }
            )
            .to_jwt()
        )

        create_resp = await self._create_session(
            room_name=room.name,
            room_sid=room_sid,
            livekit_url=lk_url,
            worker_token=worker_token,
            agent_identity=local_participant_identity,
        )

        self._session_id = create_resp.get("session_id")
        self._provider_session_id = create_resp.get("provider_session_id")
        self._terminate_token = create_resp.get("terminate_token")
        sample_rate = create_resp.get("sample_rate") or _DEFAULT_SAMPLE_RATE

        if self._provider_session_id and not self._terminate_token:
            # aclose() cannot terminate without this; the provider session will
            # run to its own idle timeout instead of being stopped explicitly.
            logger.warning(
                "avatar gateway create response had no terminate_token; this "
                "session cannot be explicitly terminated and will bill until "
                "its provider idle timeout",
                extra={
                    "provider": self._provider,
                    "session_id": self._session_id,
                    "provider_session_id": self._provider_session_id,
                },
            )

        # Rebind the audio tail to the avatar worker using the gateway-reported
        # sample rate. Done after the create response (unlike BYOK, which
        # hardcodes the rate and rebinds first): in the canonical flow
        # avatar.start() runs before session.start(), so no audio has flowed yet
        # and nothing is lost. wait_remote_track buffers until the video track
        # appears; replace_audio_tail keeps the TranscriptSynchronizer /
        # RecorderAudioOutput chain intact.
        agent_session.output.replace_audio_tail(
            DataStreamAudioOutput(
                room=room,
                destination_identity=self._avatar_participant_identity,
                sample_rate=sample_rate,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                clear_buffer_timeout=None,
                wait_playback_start=True,
            ),
        )

        logger.debug(
            "inference avatar session created",
            extra={
                "provider": self._provider,
                "session_id": self._session_id,
                "provider_session_id": self._provider_session_id,
            },
        )

    async def aclose(self) -> None:
        # Terminate the provider session through the gateway so it stops billing
        # immediately instead of lingering until its idle timeout. Best-effort:
        # a failure must not block participant cleanup in the base class, so
        # base cleanup runs in `finally` even if terminate raises or is
        # cancelled (e.g. job-shutdown deadline). The ids are cleared only on
        # a confirmed terminate so a second aclose() call can still retry.
        provider_session_id = self._provider_session_id
        terminate_token = self._terminate_token
        try:
            if provider_session_id and terminate_token:
                try:
                    await self._terminate_session(provider_session_id, terminate_token)
                except Exception:
                    logger.warning(
                        "failed to terminate inference avatar session; it will "
                        "keep billing until its provider idle timeout unless "
                        "aclose() is called again",
                        extra={
                            "provider": self._provider,
                            "provider_session_id": provider_session_id,
                        },
                        exc_info=True,
                    )
                else:
                    self._provider_session_id = None
                    self._terminate_token = None
            elif provider_session_id:
                logger.debug(
                    "no terminate_token for this avatar session; skipping explicit terminate",
                    extra={
                        "provider": self._provider,
                        "provider_session_id": provider_session_id,
                    },
                )
        finally:
            await super().aclose()

    async def _create_session(
        self,
        *,
        room_name: str,
        room_sid: str,
        livekit_url: str,
        worker_token: str,
        agent_identity: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "provider": self._provider,
            "livekit_url": livekit_url,
            "livekit_token": worker_token,
            "room_name": room_name,
            "room_sid": room_sid,
            "avatar_identity": self._avatar_participant_identity,
            "agent_identity": agent_identity,
        }
        if self._avatar_id:
            payload["avatar_id"] = self._avatar_id
        if is_given(self._image_url):
            payload["image_url"] = self._image_url
        if is_given(self._prompt):
            payload["prompt"] = self._prompt
        if is_given(self._idle_prompt):
            payload["idle_prompt"] = self._idle_prompt
        if is_given(self._idle_timeout):
            payload["idle_timeout_s"] = self._idle_timeout
        if is_given(self._extra_kwargs):
            payload["extra_kwargs"] = self._extra_kwargs

        # One idempotency key per start(), stable across retries, so a retried
        # create replays the first result on the gateway instead of paying for a
        # second provider session.
        idempotency_key = uuid.uuid4().hex
        url = f"{self._base_url.rstrip('/')}/avatar/sessions"

        session = self._http_session or aiohttp.ClientSession()
        try:
            last_exc: Exception | None = None
            for i in range(self._conn_options.max_retry + 1):
                headers = {
                    **get_inference_headers(),
                    "Authorization": f"Bearer {create_access_token(self._api_key, self._api_secret)}",
                    HEADER_INFERENCE_PROVIDER: self._provider,
                    "Idempotency-Key": idempotency_key,
                    "Content-Type": "application/json",
                }
                try:
                    async with session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(
                            total=_REQUEST_TIMEOUT,
                            sock_connect=self._conn_options.timeout,
                        ),
                    ) as response:
                        if not response.ok:
                            text = await response.text()
                            raise create_api_error_from_http(
                                f"avatar gateway returned an error: {text}",
                                status=response.status,
                                body=text,
                            )
                        return await response.json()  # type: ignore[no-any-return]
                except asyncio.TimeoutError:
                    last_exc = APITimeoutError(
                        f"avatar gateway create timed out after attempt {i + 1}"
                    )
                    logger.warning(
                        "avatar gateway request timed out",
                        extra={"provider": self._provider, "attempt": i},
                    )
                except aiohttp.ClientError as e:
                    last_exc = APIConnectionError(str(e))
                    logger.warning(
                        "failed to call avatar gateway",
                        extra={"provider": self._provider, "error": str(e)},
                    )
                except APIError as e:
                    last_exc = e
                    if not e.retryable:
                        raise
                    logger.warning(
                        "avatar gateway returned a retryable error",
                        extra={"error": str(e)},
                    )

                if i < self._conn_options.max_retry:
                    await asyncio.sleep(self._conn_options._interval_for_retry(i))

            if last_exc is not None:
                raise last_exc
            raise APIConnectionError("failed to create avatar session after all retries")
        finally:
            if self._http_session is None:
                await session.close()

    async def _terminate_session(self, provider_session_id: str, terminate_token: str) -> None:
        url = f"{self._base_url.rstrip('/')}/avatar/sessions/terminate"
        headers = {
            **get_inference_headers(),
            "Authorization": f"Bearer {create_access_token(self._api_key, self._api_secret)}",
            HEADER_INFERENCE_PROVIDER: self._provider,
            "Content-Type": "application/json",
        }
        payload = {
            "provider": self._provider,
            "provider_session_id": provider_session_id,
            "terminate_token": terminate_token,
        }

        session = self._http_session or aiohttp.ClientSession()
        try:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=_REQUEST_TIMEOUT, sock_connect=self._conn_options.timeout
                ),
            ) as response:
                if not response.ok:
                    text = await response.text()
                    raise create_api_error_from_http(
                        f"avatar gateway returned an error: {text}",
                        status=response.status,
                        body=text,
                    )
        finally:
            if self._http_session is None:
                await session.close()
