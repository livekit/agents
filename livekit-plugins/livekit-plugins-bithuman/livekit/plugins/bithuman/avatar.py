from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Literal
from urllib.parse import parse_qs, urlparse

import aiohttp
import cv2
import numpy as np
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
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioOutput,
    QueueAudioOutput,
    VideoGenerator,
)

from .log import logger

if TYPE_CHECKING:
    from bithuman import AsyncBithuman  # type: ignore

_logger.remove()
_logger.add(sys.stdout, level="INFO")


_AVATAR_AGENT_IDENTITY = "bithuman-avatar-agent"
_AVATAR_AGENT_NAME = "bithuman-avatar-agent"


def _is_valid_base64(s: str) -> bool:
    """
    Strictly validate if a string is valid base64 encoded data.

    Args:
        s: String to validate

    Returns:
        True if the string is valid base64, False otherwise
    """
    import re

    # Base64 strings should only contain A-Z, a-z, 0-9, +, /, and = for padding
    # Remove whitespace for validation
    s_clean = s.strip().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

    # Check if string is empty after cleaning
    if not s_clean:
        return False

    # Base64 strings must have length that is a multiple of 4 (after padding)
    # Padding can be 0, 1, or 2 '=' characters
    if len(s_clean) % 4 != 0:
        return False

    # Check if string contains only valid base64 characters
    base64_pattern = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
    if not base64_pattern.match(s_clean):
        return False

    # Try to decode and verify it doesn't raise an exception
    try:
        decoded = base64.b64decode(s_clean)
        # Additional check: decoded data should not be empty
        return len(decoded) > 0
    except Exception:
        return False


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
            elif avatar_image.startswith(("http://", "https://")):
                self._avatar_image = avatar_image
            elif _is_valid_base64(avatar_image):
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
            logger.debug("previous transaction id: %s", runtime.transaction_id)
            runtime._regenerate_transaction_id()
            logger.debug("new transaction id: %s", runtime.transaction_id)
            await runtime._initialize_token()
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

        audio_buffer = QueueAudioOutput(sample_rate=runtime.settings.INPUT_SAMPLE_RATE)
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

        job_ctx = get_job_context()
        local_participant_identity = job_ctx.local_participant_identity

        # Prepare attributes for JWT token
        attributes: dict[str, str] = {
            ATTRIBUTE_PUBLISH_ON_BEHALF: local_participant_identity,
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

        # Determine if using custom API endpoint (not the default BitHuman auth API)
        # Custom endpoints use multipart/form-data format for direct avatar worker requests
        is_custom_endpoint = not self._is_default_api_url()

        if is_custom_endpoint and self._model == "expression":
            # Use FormData format for custom endpoints
            # Parse async parameter from URL if present
            async_mode = self._parse_async_parameter_from_url()
            await self._send_formdata_request(
                livekit_url, livekit_token, room_name, async_mode=async_mode
            )
        else:
            # Default BitHuman API requires api_secret
            assert self._api_secret is not None, "api_secret is not set"
            # Use JSON format for default BitHuman API
            await self._send_json_request(livekit_url, livekit_token, room_name)

    def _is_default_api_url(self) -> bool:
        """
        Check if using the default BitHuman API URL.

        Returns:
            True if using default auth.api.bithuman.ai endpoint, False otherwise.
        """
        if self._api_url is None:
            return True
        try:
            parsed = urlparse(self._api_url)
            hostname = parsed.hostname
            if hostname is None:
                return False
            default_domains = ["auth.api.bithuman.ai", "api.bithuman.ai"]
            return hostname in default_domains
        except Exception:
            # If parsing fails, fallback to substring matching
            default_domains = ["auth.api.bithuman.ai", "api.bithuman.ai"]
            return any(domain in self._api_url for domain in default_domains)

    async def _send_json_request(
        self, livekit_url: str, livekit_token: str, room_name: str
    ) -> None:
        """
        Send request using JSON format (for default BitHuman API).

        Args:
            livekit_url: LiveKit server URL
            livekit_token: JWT token for room access
            room_name: Name of the LiveKit room
        """
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

        # Handle avatar image - convert to base64 for JSON serialization
        if isinstance(self._avatar_image, Image.Image):
            img_byte_arr = io.BytesIO()
            self._avatar_image.save(img_byte_arr, format="JPEG", quality=95)
            img_byte_arr.seek(0)
            json_data["image"] = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        elif isinstance(self._avatar_image, str):
            json_data["image"] = self._avatar_image

        if utils.is_given(self._avatar_id):
            json_data["agent_id"] = self._avatar_id

        assert self._api_secret is not None, "api_secret is required for default API"

        headers = {
            "Content-Type": "application/json",
            "api-secret": self._api_secret,
        }

        await self._send_request_with_retry(
            headers=headers,
            json_data=json_data,
            form_data=None,
        )

    def _parse_async_parameter_from_url(self) -> bool | None:
        """
        Parse async parameter from api_url if present.

        Returns:
            True if async=true, False if async=false, None if not present
        """
        if self._api_url is None:
            return None

        try:
            parsed = urlparse(self._api_url)
            query_params = parse_qs(parsed.query)

            if "async" in query_params:
                async_value = query_params["async"][0].lower()
                if async_value == "true":
                    return True
                elif async_value == "false":
                    return False
        except Exception:
            # If parsing fails, return None (don't add async_mode parameter)
            pass

        return None

    async def _send_formdata_request(
        self, livekit_url: str, livekit_token: str, room_name: str, async_mode: bool | None = None
    ) -> None:
        """
        Send request using multipart/form-data format (for custom avatar worker endpoints).

        This format is used for direct communication with avatar workers like:
        - gpu-avatar-worker (FLOAT model)
        - cpu-avatar-worker
        - Cerebrium deployments

        Args:
            livekit_url: LiveKit server URL
            livekit_token: JWT token for room access
            room_name: Name of the LiveKit room
            async_mode: Optional async_mode parameter (parsed from URL if async parameter present)
        """
        # Build form data with required fields
        form_data = aiohttp.FormData()
        form_data.add_field("livekit_url", livekit_url)
        form_data.add_field("livekit_token", livekit_token)
        form_data.add_field("room_name", room_name)

        # Add async_mode parameter if parsed from URL
        # FastAPI Form bool accepts "true"/"false" strings and converts them to boolean
        if async_mode is not None:
            form_data.add_field("async_mode", "true" if async_mode else "false")

        # Handle avatar image - send as file upload or URL
        if isinstance(self._avatar_image, Image.Image):
            # Convert PIL Image to bytes and upload as file
            img_byte_arr = io.BytesIO()
            self._avatar_image.save(img_byte_arr, format="JPEG", quality=95)
            img_byte_arr.seek(0)
            form_data.add_field(
                "avatar_image",
                img_byte_arr,
                filename="avatar.jpg",
                content_type="image/jpeg",
            )
        elif isinstance(self._avatar_image, str):
            # String can be URL or base64 - check if it's a URL
            if self._avatar_image.startswith(("http://", "https://")):
                form_data.add_field("avatar_image_url", self._avatar_image)
            elif _is_valid_base64(self._avatar_image):
                # Valid base64 string, decode and upload as file
                try:
                    decoded_image = base64.b64decode(self._avatar_image)
                    img_byte_arr = io.BytesIO(decoded_image)
                    form_data.add_field(
                        "avatar_image",
                        img_byte_arr,
                        filename="avatar.jpg",
                        content_type="image/jpeg",
                    )
                except Exception as err:
                    # If decode fails despite validation, raise error
                    raise BitHumanException(
                        f"Failed to decode base64 avatar image: {self._avatar_image[:50]}..."
                    ) from err
            else:
                # Not a URL and not valid base64, raise error
                raise BitHumanException(
                    f"Invalid avatar image string: must be a URL (starting with http:// or https://) "
                    f"or valid base64 encoded data. Got: {self._avatar_image[:50]}..."
                )

        # Add avatar_id if provided
        if utils.is_given(self._avatar_id):
            form_data.add_field("avatar_id", self._avatar_id)

        # Authorization header for custom endpoints uses api_token (Bearer token format)
        # Note: api_token is different from api_secret - token is for direct API access,
        # while secret is for BitHuman's authentication service
        auth_token = self._api_token or self._api_secret
        if auth_token is None:
            raise BitHumanException(
                "api_token or api_secret is required for custom endpoint requests. "
                "Set BITHUMAN_API_TOKEN or BITHUMAN_API_SECRET environment variable."
            )

        headers = {
            "Authorization": f"Bearer {auth_token}",
        }

        await self._send_request_with_retry(
            headers=headers,
            json_data=None,
            form_data=form_data,
        )

    async def _send_request_with_retry(
        self,
        headers: dict[str, str],
        json_data: dict | None = None,
        form_data: aiohttp.FormData | None = None,
    ) -> None:
        """
        Send HTTP request with retry logic.

        Handles both JSON and FormData request formats with configurable retry behavior.

        Args:
            headers: HTTP headers to include in the request
            json_data: JSON payload (mutually exclusive with form_data)
            form_data: FormData payload (mutually exclusive with json_data)

        Raises:
            APIConnectionError: If all retry attempts fail
        """
        for i in range(self._conn_options.max_retry):
            try:
                async with self._ensure_http_session().post(
                    self._api_url,
                    headers=headers,
                    json=json_data,
                    data=form_data,
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


class BithumanGenerator(VideoGenerator):
    def __init__(self, runtime: AsyncBithuman):
        self._runtime = runtime

    @property
    def video_resolution(self) -> tuple[int, int]:
        frame = self._runtime.get_first_frame()
        if frame is None:
            raise ValueError("Failed to read frame")
        return frame.shape[1], frame.shape[0]

    @property
    def video_fps(self) -> int:
        return self._runtime.settings.FPS  # type: ignore

    @property
    def audio_sample_rate(self) -> int:
        return self._runtime.settings.INPUT_SAMPLE_RATE  # type: ignore

    @utils.log_exceptions(logger=logger)
    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            await self._runtime.flush()
            return
        await self._runtime.push_audio(bytes(frame.data), frame.sample_rate, last_chunk=False)

    def clear_buffer(self) -> None:
        self._runtime.interrupt()

    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._stream_impl()

    async def _stream_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        def create_video_frame(image: np.ndarray) -> rtc.VideoFrame:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return rtc.VideoFrame(
                width=image.shape[1],
                height=image.shape[0],
                type=rtc.VideoBufferType.RGB24,
                data=image.tobytes(),
            )

        async for frame in self._runtime.run():
            if frame.bgr_image is not None:
                video_frame = create_video_frame(frame.bgr_image)
                yield video_frame

            audio_chunk = frame.audio_chunk
            if audio_chunk is not None:
                audio_frame = rtc.AudioFrame(
                    data=audio_chunk.bytes,
                    sample_rate=audio_chunk.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(audio_chunk.array),
                )
                yield audio_frame

            if frame.end_of_speech:
                yield AudioSegmentEnd()

    async def stop(self) -> None:
        await self._runtime.stop()
