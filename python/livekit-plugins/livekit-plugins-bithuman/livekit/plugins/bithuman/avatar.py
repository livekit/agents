from __future__ import annotations

import base64
import io
import os
import sys
from collections.abc import AsyncGenerator, AsyncIterator
from importlib import import_module
from typing import Literal, Protocol, cast
from urllib.parse import parse_qs, urlparse

import numpy as np
from loguru import logger as _logger
from numpy.typing import NDArray
from PIL import Image

import livekit.rtc as rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    http,
    utils,
)
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    AvatarSession as BaseAvatarSession,
    AvatarStart,
    DataStreamAudioOutput,
    QueueAudioOutput,
    VideoGenerator,
)

_logger.remove()
_logger.add(sys.stdout, level="INFO")


# The avatar models a cloud session may ask for, in BitHuman's own spelling.
# "expression"/"essence" are the first-generation names and keep resolving to
# the first-generation engines; "expression-2"/"essence-2" name the current
# ones. The server validates this string and refuses an unknown one, so a typo
# is an error at session start rather than a different engine rendering.
BitHumanModel = Literal["expression", "essence", "expression-2", "essence-2"]


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


class RuntimeSettings(Protocol):
    FPS: int
    INPUT_SAMPLE_RATE: int


class AudioChunk(Protocol):
    @property
    def bytes(self) -> bytes: ...

    @property
    def array(self) -> NDArray[np.int16]: ...

    sample_rate: int


class RenderedFrame(Protocol):
    bgr_image: NDArray[np.uint8] | None
    audio_chunk: AudioChunk | None
    end_of_speech: bool


class BitHumanRuntime(Protocol):
    """SDK capabilities consumed by a local avatar session and its video generator."""

    settings: RuntimeSettings

    def get_first_frame(self) -> NDArray[np.uint8] | None: ...
    async def push_audio(self, data: bytes, sample_rate: int, *, last_chunk: bool) -> None: ...
    async def flush(self) -> None: ...
    def interrupt(self) -> None: ...
    def run(self) -> AsyncIterator[RenderedFrame]: ...
    async def stop(self) -> None: ...
    def cleanup(self) -> None: ...


class _RuntimeFactory(Protocol):
    async def create(
        self,
        *,
        model_path: str | None,
        api_secret: str | None,
        token: str | None,
        api_url: str,
    ) -> BitHumanRuntime: ...


class AvatarSession(BaseAvatarSession):
    """A bitHuman avatar session"""

    def __init__(
        self,
        *,
        api_url: NotGivenOr[str] = NOT_GIVEN,
        api_secret: NotGivenOr[str] = NOT_GIVEN,
        api_token: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[BitHumanModel] = NOT_GIVEN,
        model_path: NotGivenOr[str | None] = NOT_GIVEN,
        runtime: NotGivenOr[BitHumanRuntime | None] = NOT_GIVEN,
        avatar_image: NotGivenOr[Image.Image | str] = NOT_GIVEN,
        avatar_id: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        avatar_participant_identity: NotGivenOr[str] = NOT_GIVEN,
        avatar_participant_name: NotGivenOr[str] = NOT_GIVEN,
        livekit_url: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_key: NotGivenOr[str] = NOT_GIVEN,
        livekit_api_secret: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Initialize a bitHuman avatar session.

        Args:
            api_url: The bitHuman API URL. Defaults to ``BITHUMAN_API_URL``, then to
                ``https://auth.api.bithuman.ai/v1/runtime-tokens/request``.
            api_secret: The bitHuman API secret. Defaults to ``BITHUMAN_API_SECRET``.
                In cloud mode this value is sent to the avatar in the avatar
                participant's attributes, which every participant in the room can
                read: pass a one-hour token minted with
                ``POST https://api.bithuman.ai/v1/runtime-tokens/mint``
                (``"scope": "livekit-cloud"``) instead of your account secret.
            api_token: A bitHuman runtime token. Defaults to ``BITHUMAN_API_TOKEN``.
                Local mode accepts it in place of ``api_secret``; the default cloud
                endpoint does not use it.
            model: The model to pin a cloud session to (see Model Types). Not used in
                local mode, where the ``.imx`` file determines the model.
            model_path: Path to a bitHuman ``.imx`` model file, for local mode.
                Defaults to ``BITHUMAN_MODEL_PATH``.
            runtime: A caller-owned local runtime prepared for this session. The plugin
                stops playback on close; the caller manages authentication, preparation
                before any later session, and final cleanup. Restart support depends on
                the supplied SDK runtime.
            avatar_image: A portrait for a cloud photo session (Expression 1 only): a
                PIL image, a local file path, an ``http(s)`` URL or base64-encoded
                image data.
            avatar_id: The code of a bitHuman agent to render in cloud mode.
            conn_options: Retry and timeout options for the cloud session request.
            avatar_participant_identity: The avatar participant identity to use.
            avatar_participant_name: The avatar participant name to use.
            livekit_url: The URL the avatar joins through; ``LIVEKIT_URL`` by default.
            livekit_api_key: Signs the avatar's token; ``LIVEKIT_API_KEY`` by default.
            livekit_api_secret: Signs the avatar's token; ``LIVEKIT_API_SECRET`` by default.

        Model Types:
            bitHuman has two model families, Essence and Expression, each in two
            generations. In cloud mode the name is sent to bitHuman with the session, so
            the model you name is the model that renders:

            - **essence-2** (Essence 2): a photoreal person, created from one portrait.

            - **expression-2** (Expression 2): any character, including people, animals
              and cartoons, created from one portrait.

            - **essence** / **expression**: the first generation, Essence 1 and
              Expression 1. They keep working unchanged; Expression 1 runs in the
              bitHuman cloud only.

            Naming a model is OPTIONAL and it is a pin, not a hint: an avatar has to have
            been prepared for the model you ask for, and bitHuman refuses the session
            (naming the models that avatar can be served as) if it has not. Leave ``model``
            unset and nothing is sent, so bitHuman resolves the avatar exactly as it did
            before this parameter existed — which is why adding it changes nothing for
            sessions that do not use it.

        Parameter Combinations:
            The following parameter combinations determine the avatar mode and behavior:

            1. **Local Mode (neither avatar_image nor avatar_id provided)**:
               - `model_path`: an Essence 2, Expression 2 or Essence 1 ``.imx`` file,
                 rendered inside this process by the bitHuman SDK, with no GPU
                 required. Expression 2 files need ``pip install "bithuman[expression-2]"``.
               - `model` is not used: the file determines the model
               - Requires BITHUMAN_API_SECRET or BITHUMAN_API_TOKEN

            2. **Cloud Mode with avatar_image**:
               - `avatar_image`: a portrait animated with no agent, on Expression 1 only;
                 bitHuman refuses a photo session on any other model. To use Essence 2
                 or Expression 2, create an agent from the photo and pass `avatar_id`.

            3. **Cloud Mode with avatar_id**:
               - `avatar_id`: the code of an existing bitHuman agent
               - `model`: optional. Name any of the four models above to pin the
                 session to it; leave it unset and bitHuman picks the model the
                 avatar is prepared for, exactly as before this parameter existed.

            Cloud mode also needs LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET (or
            the matching arguments) to create the avatar's room token.
        """
        super().__init__(
            avatar_participant_identity=avatar_participant_identity,
            avatar_participant_name=avatar_participant_name,
            livekit_url=livekit_url,
            livekit_api_key=livekit_api_key,
            livekit_api_secret=livekit_api_secret,
            conn_options=conn_options,
        )
        self._provided_runtime = runtime if utils.is_given(runtime) else None
        self._api_url = (
            api_url
            or os.getenv("BITHUMAN_API_URL")
            or "https://auth.api.bithuman.ai/v1/runtime-tokens/request"
        )
        self._api_secret = api_secret or os.getenv("BITHUMAN_API_SECRET")
        self._api_token = api_token or os.getenv("BITHUMAN_API_TOKEN")
        self._model_path = model_path or os.getenv("BITHUMAN_MODEL_PATH")
        self._avatar_id = avatar_id

        # set default mode based on model_path, avatar_image or avatar_id presence
        self._mode = (
            "cloud" if utils.is_given(avatar_image) or utils.is_given(avatar_id) else "local"
        )
        # ★KEPT RAW, on purpose. An earlier draft normalised NOT_GIVEN to "essence"
        # here. That is wrong in a way that only shows up for existing users: the
        # signature used to default to "essence", so normalising made the plugin
        # send `model: "essence"` on EVERY session, including ones whose caller
        # never named a model. Before this PR nothing was sent and the server chose
        # the generation; after it, an avatar prepared only as essence-2 would be
        # told "essence" and REFUSED (the API answers 400 naming what the avatar
        # can be served as). So the sentinel survives to the request, where it
        # means "do not name a model" — and `str()` is never called on it.
        self._model = model

        # validate mode-specific requirements
        if self._mode == "local" and self._provided_runtime is None:
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

        self._avatar_runner: AvatarRunner | None = None
        self._local_audio: QueueAudioOutput | None = None
        self._runtime = self._provided_runtime

    @property
    def _is_expression_model(self) -> bool:
        """True for every generation of the expression family.

        Read in two places, the custom-endpoint request format and the legacy
        ``mode`` field, and it must give the same answer in both, or a session
        is dispatched one way and billed the other.
        """
        return utils.is_given(self._model) and str(self._model).startswith("expression")

    @property
    def avatar_identity(self) -> str:
        # In local mode the avatar video is published by the local agent participant,
        # so the avatar identity is the local participant's identity.
        if self._mode == "local" and self._room is not None:
            return self._room.local_participant.identity
        return super().avatar_identity

    @property
    def provider(self) -> str:
        return "bithuman"

    async def create_session(self, ctx: AvatarStart) -> None:
        if self._mode == "local":
            await self._start_local(ctx.room)
        elif self._is_expression_model and not self._is_default_api_url():
            # a self-hosted avatar worker takes the session as a form
            await self._post(ctx, self._session_form(ctx))
        else:
            await self._post(ctx, self._session_json(ctx))

    def audio_output(self, ctx: AvatarStart) -> DataStreamAudioOutput | QueueAudioOutput:
        if self._local_audio is not None:
            return self._local_audio
        return DataStreamAudioOutput(
            room=ctx.room, destination_identity=self.avatar_identity, wait_playback_start=False
        )

    async def _start_local(self, room: rtc.Room) -> None:
        runtime = self._provided_runtime
        if runtime is None:
            # BitHuman's native AsyncBithuman export has no complete type stub.
            factory = cast(_RuntimeFactory, import_module("bithuman").AsyncBithuman)
            runtime = await factory.create(
                model_path=self._model_path,
                api_secret=self._api_secret,
                token=self._api_token,
                api_url=self._api_url,
            )
        self._runtime = runtime

        video_generator = BithumanGenerator(runtime)
        output_width, output_height = video_generator.video_resolution
        avatar_options = AvatarOptions(
            video_width=output_width,
            video_height=output_height,
            video_fps=video_generator.video_fps,
            audio_sample_rate=video_generator.audio_sample_rate,
            audio_channels=1,
        )
        self._local_audio = QueueAudioOutput(
            sample_rate=runtime.settings.INPUT_SAMPLE_RATE, wait_playback_start=True
        )
        self._avatar_runner = AvatarRunner(
            room=room,
            video_gen=video_generator,
            audio_recv=self._local_audio,
            options=avatar_options,
        )
        await self._avatar_runner.start()

    def _session_token(self, ctx: AvatarStart) -> str:
        attributes: dict[str, str] = {}
        if self._api_secret is not None:
            attributes["api_secret"] = self._api_secret
        if utils.is_given(self._avatar_id):
            attributes["agent_id"] = self._avatar_id
        return ctx.livekit_token(attributes=attributes)

    def _is_default_api_url(self) -> bool:
        return urlparse(self._api_url).hostname in ("auth.api.bithuman.ai", "api.bithuman.ai")

    def _session_json(self, ctx: AvatarStart) -> dict[str, str]:
        # `model` names the avatar model for this session explicitly. Without it the
        # server falls back to `mode`, which can only distinguish the two
        # first-generation engines. `mode` is still sent, unchanged, for servers that
        # predate `model`.
        json_data = {
            "livekit_url": ctx.livekit_url,
            "livekit_token": self._session_token(ctx),
            "room_name": ctx.room.name,
            "mode": "gpu" if self._avatar_image is not None or self._is_expression_model else "cpu",
        }
        if utils.is_given(self._model):
            json_data["model"] = str(self._model)
        if isinstance(self._avatar_image, Image.Image):
            json_data["image"] = base64.b64encode(_jpeg(self._avatar_image)).decode("utf-8")
        elif isinstance(self._avatar_image, str):
            json_data["image"] = self._avatar_image
        if utils.is_given(self._avatar_id):
            json_data["agent_id"] = self._avatar_id
        return json_data

    def _session_form(self, ctx: AvatarStart) -> http.FormData:
        """The session for a self-hosted avatar worker (gpu-avatar-worker, cpu-avatar-worker,
        Cerebrium deployments)."""
        form_data = http.FormData()
        form_data.add_field("livekit_url", ctx.livekit_url)
        form_data.add_field("livekit_token", self._session_token(ctx))
        form_data.add_field("room_name", ctx.room.name)
        # without it a worker cannot tell `expression` from `expression-2`; workers that
        # predate the field ignore it
        if utils.is_given(self._model):
            form_data.add_field("model", str(self._model))
        async_mode = parse_qs(urlparse(self._api_url).query).get("async", [""])[0].lower()
        if async_mode in ("true", "false"):
            form_data.add_field("async_mode", async_mode)

        image = self._avatar_image
        if isinstance(image, str) and image.startswith(("http://", "https://")):
            form_data.add_field("avatar_image_url", image)
        elif image is not None:
            data = _jpeg(image) if isinstance(image, Image.Image) else base64.b64decode(image)
            form_data.add_field(
                "avatar_image", io.BytesIO(data), filename="avatar.jpg", content_type="image/jpeg"
            )
        if utils.is_given(self._avatar_id):
            form_data.add_field("avatar_id", self._avatar_id)
        return form_data

    async def _post(self, ctx: AvatarStart, body: dict[str, str] | http.FormData) -> None:
        if isinstance(body, http.FormData):
            # a self-hosted worker takes a bearer token: the runtime token, else the secret
            headers = {"Authorization": f"Bearer {self._api_token or self._api_secret}"}
        else:
            headers = {"api-secret": self._api_secret or ""}
        async with http.http_session().post(
            self._api_url,
            headers=headers,
            json=None if isinstance(body, http.FormData) else body,
            data=body if isinstance(body, http.FormData) else None,
            conn_options=ctx.conn_options,
        ) as response:
            await response.raise_for_status()

    @property
    def runtime(self) -> BitHumanRuntime:
        if self._runtime is None:
            raise BitHumanException("Runtime not initialized")
        return self._runtime

    async def aclose(self) -> None:
        runner, self._avatar_runner = self._avatar_runner, None
        runtime, self._runtime = self._runtime, None
        self._local_audio = None
        try:
            try:
                if runner is not None:
                    await runner.aclose()
            finally:
                if self._mode == "local" and runtime is not None:
                    try:
                        await runtime.stop()
                    finally:
                        if self._provided_runtime is None:
                            runtime.cleanup()
        finally:
            await super().aclose()


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


class BithumanGenerator(VideoGenerator):
    def __init__(self, runtime: BitHumanRuntime) -> None:
        self._runtime = runtime

    @property
    def video_resolution(self) -> tuple[int, int]:
        frame = self._runtime.get_first_frame()
        if frame is None:
            raise ValueError("Failed to read frame")
        return frame.shape[1], frame.shape[0]

    @property
    def video_fps(self) -> int:
        return self._runtime.settings.FPS

    @property
    def audio_sample_rate(self) -> int:
        return self._runtime.settings.INPUT_SAMPLE_RATE

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
        def create_video_frame(image: NDArray[np.uint8]) -> rtc.VideoFrame:
            image = np.ascontiguousarray(image[:, :, ::-1])  # BGR -> RGB
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
