from __future__ import annotations

import asyncio
import io
import os
import sys
from typing import TYPE_CHECKING, Literal

import aiohttp
from loguru import logger as _logger
from PIL import Image

from livekit import api, rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    NotGivenOr,
    get_job_context,
    utils,
)
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import (
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioOutput,
    QueueAudioOutput,
)

from .log import logger
from .video_generator import BithumanGenerator

if TYPE_CHECKING:
    from bithuman import AsyncBithuman  # type: ignore

_logger.remove()
_logger.add(sys.stdout, level="INFO")


_AVATAR_AGENT_IDENTITY = "bithuman-avatar-agent"
_AVATAR_AGENT_NAME = "bithuman-avatar-agent"


class BitHumanException(Exception):
    """Exception for BitHuman errors"""


class AvatarSession:
    """A Beyond Presence avatar session"""

    def __init__(
        self,
        *,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        api_token: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[Literal["expression", "essence"]] = "essence",
        model_path: NotGivenOr[str | None] = NOT_GIVEN,
        runtime: NotGivenOr[AsyncBithuman | None] = NOT_GIVEN,
        avatar_image: NotGivenOr[Image.Image | str] = NOT_GIVEN,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Initialize a BitHuman avatar session.

        Args:
            api_url: The BitHuman API URL.
            api_secret: The BitHuman API secret.
            api_token: The BitHuman API token.
            model: The BitHuman model to use.
            model_path: The path to the BitHuman model.
            runtime: The BitHuman runtime to use.
            avatar_image: The avatar image to use.
            avatar_id: The avatar ID to use.
            conn_options: The connection options to use.
            avatar_participant_identity: The avatar participant identity to use.
            avatar_participant_name: The avatar participant name to use.

        Model Types:
            BitHuman supports two model types with different capabilities:

            - **expression**: Provides dynamic real-time facial expressions and emotional responses.
              This model can generate live emotional expressions based on the content and context,
              offering more natural and interactive avatar behavior.

            - **essence**: Uses predefined actions and expressions. This model provides consistent
              and predictable avatar behavior with pre-configured gestures and expressions.

        Parameter Combinations:
            The following parameter combinations determine the avatar mode and behavior:

            1. **Local Mode (model_path provided)**:
               - `model_path`: Loads the BitHuman SDK locally for processing
               - Works with both expression and essence models
               - Requires BITHUMAN_API_SECRET or BITHUMAN_API_TOKEN

            2. **Cloud Mode with avatar_image**:
               - `avatar_image`: Custom avatar image for personalization
               - `model`: Defaults to "expression" for dynamic emotional expressions
               - Provides real-time expression generation based on the custom image

            3. **Cloud Mode with avatar_id**:
               - `avatar_id`: Pre-configured avatar identifier
               - `model`: Defaults to "essence" if not specified, but can be set to either:
                 * "expression" for dynamic emotional responses
                 * "essence" for predefined actions and expressions
               - Allows flexibility in choosing the interaction style
        """
        self._api_url = (
            api_url
            or os.getenv("BITHUMAN_API_URL")
            or "https://auth.api.bithuman.ai/v1/runtime-tokens/request"
        )
        self._api_secret = api_secret or os.getenv("BITHUMAN_API_SECRET")
        self._api_token = api_token or os.getenv("BITHUMAN_API_TOKEN")
        self._model_path = model_path or os.getenv("BITHUMAN_MODEL_PATH")
        self._avatar_id = avatar_id
        self._avatar_participant_identity = avatar_participant_identity or _AVATAR_AGENT_IDENTITY
        self._avatar_participant_name = avatar_participant_name or _AVATAR_AGENT_NAME

        # set default mode based on model_path, avatar_image or avatar_id presence
        self._mode = (
            "cloud" if utils.is_given(avatar_image) or utils.is_given(avatar_id) else "local"
        )
        self._model = model

        # validate mode-specific requirements
        if self._mode == "local":
            if self._model_path is None:
                raise BitHumanException(
                    "`model_path` or BITHUMAN_MODEL_PATH env must be set for local mode"
                )
            if self._api_secret is None and self._api_token is None:
                raise BitHumanException(
                    "BITHUMAN_API_SECRET or BITHUMAN_API_TOKEN are required for local mode"
                )
        elif self._mode == "cloud":
            if not utils.is_given(avatar_image) and not utils.is_given(avatar_id):
                raise BitHumanException("`avatar_image` or `avatar_id` must be set for cloud mode")
            if self._api_secret is None:
                raise BitHumanException("BITHUMAN_API_SECRET are required for cloud mode")
            if self._api_url is None:
                raise BitHumanException("BITHUMAN_API_URL are required for cloud mode")

        self._avatar_image: Image.Image | str | None = None
        if isinstance(avatar_image, Image.Image):
            self._avatar_image = avatar_image
        elif isinstance(avatar_image, str):
            if os.path.exists(avatar_image):
                self._avatar_image = Image.open(avatar_image)
            elif avatar_image.startswith("http"):
                self._avatar_image = avatar_image
            else:
                raise BitHumanException(f"Invalid avatar image: {avatar_image}")

        self._conn_options = conn_options
        self._http_session: aiohttp.ClientSession | None = None
        self._avatar_runner: AvatarRunner | None = None
        self._runtime = runtime

    async def start(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if self._mode == "local":
            await self._start_local(agent_session, room)
        elif self._mode == "cloud":
            await self._start_cloud(
                agent_session,
                room,
                livekit_url=livekit_url,
                livekit_api_key=livekit_api_key,
                livekit_api_secret=livekit_api_secret,
            )
        else:
            raise BitHumanException(f"Invalid mode: {self._mode}")

    async def _start_local(self, agent_session: AgentSession, room: rtc.Room) -> None:
        from bithuman import AsyncBithuman

        if self._runtime:
            runtime = self._runtime
            await runtime._initialize_token()  # refresh the token
        else:
            kwargs = {
                "model_path": self._model_path,
            }
            if self._api_secret:
                kwargs["api_secret"] = self._api_secret
            if self._api_token:
                kwargs["token"] = self._api_token
            if self._api_url:
                kwargs["api_url"] = self._api_url

            runtime = await AsyncBithuman.create(**kwargs)
            self._runtime = runtime
            await runtime.start()

        video_generator = BithumanGenerator(runtime)

        try:
            job_ctx = get_job_context()

            async def _on_shutdown() -> None:
                runtime.cleanup()

            job_ctx.add_shutdown_callback(_on_shutdown)
        except RuntimeError:
            pass

        output_width, output_height = video_generator.video_resolution
        avatar_options = AvatarOptions(
            video_width=output_width,
            video_height=output_height,
            video_fps=video_generator.video_fps,
            audio_sample_rate=video_generator.audio_sample_rate,
            audio_channels=1,
        )

        audio_buffer = QueueAudioOutput(
            sample_rate=runtime.settings.INPUT_SAMPLE_RATE,
            can_pause=True,
        )
        # create avatar runner
        self._avatar_runner = AvatarRunner(
            room=room,
            video_gen=video_generator,
            audio_recv=audio_buffer,
            options=avatar_options,
        )
        await self._avatar_runner.start()

        agent_session.output.audio = audio_buffer

    async def _start_cloud(
        self,
        agent_session: AgentSession,
        room: rtc.Room,
        *,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        livekit_url = livekit_url or (os.getenv("LIVEKIT_URL") or NOT_GIVEN)
        livekit_api_key = livekit_api_key or (os.getenv("LIVEKIT_API_KEY") or NOT_GIVEN)
        livekit_api_secret = livekit_api_secret or (os.getenv("LIVEKIT_API_SECRET") or NOT_GIVEN)
        if not livekit_url or not livekit_api_key or not livekit_api_secret:
            raise BitHumanException(
                "livekit_url, livekit_api_key, and livekit_api_secret must be set "
                "by arguments or environment variables"
            )

        # Prepare attributes for JWT token
        attributes: dict[str, str] = {
            ATTRIBUTE_PUBLISH_ON_BEHALF: room.local_participant.identity,
        }

        # Only add api_secret if it's not None
        if self._api_secret is not None:
            attributes["api_secret"] = self._api_secret

        # Only add agent_id if it's actually provided (not NotGiven)
        if utils.is_given(self._avatar_id):
            attributes["agent_id"] = self._avatar_id

        # Only add image if it's actually provided (not NotGiven)
        # if utils.is_given(self._avatar_image) and self._avatar_image is not None:
        #     attributes["image"] = self._avatar_image

        livekit_token = (
            api.AccessToken(api_key=livekit_api_key, api_secret=livekit_api_secret)
            .with_kind("agent")
            .with_identity(self._avatar_participant_identity)
            .with_name(self._avatar_participant_name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            # allow the avatar agent to publish audio and video on behalf of your local agent
            .with_attributes(attributes)
            .to_jwt()
        )

        logger.debug("starting avatar session")
        await self._start_cloud_agent(livekit_url, livekit_token, room.name)

        agent_session.output.audio = DataStreamAudioOutput(
            room=room,
            destination_identity=self._avatar_participant_identity,
        )

    async def _start_cloud_agent(
        self, livekit_url: str, livekit_token: str, room_name: str
    ) -> None:
        assert self._api_url is not None, "api_url is not set"
        assert self._api_secret is not None, "api_secret is not set"

        # Prepare JSON data
        json_data = {
            "livekit_url": livekit_url,
            "livekit_token": livekit_token,
            "room_name": room_name,
            "mode": "gpu"
            if (utils.is_given(self._avatar_image) and self._avatar_image is not None)
            or self._model == "expression"
            else "cpu",
        }

        # Handle avatar image
        if isinstance(self._avatar_image, Image.Image):
            img_byte_arr = io.BytesIO()
            self._avatar_image.save(img_byte_arr, format="JPEG", quality=95)
            img_byte_arr.seek(0)
            # Convert to base64 for JSON serialization
            import base64

            json_data["image"] = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        elif isinstance(self._avatar_image, bytes):
            # Convert bytes to base64 for JSON serialization
            import base64

            json_data["image"] = base64.b64encode(self._avatar_image).decode("utf-8")
        elif isinstance(self._avatar_image, str):
            json_data["image"] = self._avatar_image

        if utils.is_given(self._avatar_id):
            json_data["agent_id"] = self._avatar_id

        for i in range(self._conn_options.max_retry):
            try:
                async with self._ensure_http_session().post(
                    self._api_url,
                    headers={
                        "Content-Type": "application/json",
                        "api-secret": self._api_secret,
                    },
                    json=json_data,
                    timeout=aiohttp.ClientTimeout(sock_connect=self._conn_options.timeout),
                ) as response:
                    if not response.ok:
                        text = await response.text()
                        raise APIStatusError(
                            "Server returned an error", status_code=response.status, body=text
                        )
                    return

            except Exception as e:
                if isinstance(e, APIConnectionError):
                    logger.warning("failed to call bithuman avatar api", extra={"error": str(e)})
                else:
                    logger.exception("failed to call bithuman avatar api")

                if i < self._conn_options.max_retry - 1:
                    await asyncio.sleep(self._conn_options.retry_interval)

        raise APIConnectionError("Failed to start Bithuman Avatar Session after all retries")

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()

        return self._http_session

    @property
    def runtime(self) -> AsyncBithuman:
        if self._runtime is None:
            raise BitHumanException("Runtime not initialized")
        return self._runtime
