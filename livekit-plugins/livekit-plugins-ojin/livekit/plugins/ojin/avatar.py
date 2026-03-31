from __future__ import annotations

# mypy: disable-error-code=import-untyped
import asyncio
import os
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from livekit import rtc
from livekit.agents import NOT_GIVEN, AgentSession, NotGivenOr, utils
from livekit.agents.voice.avatar import (
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    QueueAudioOutput,
    VideoGenerator,
)
from ojin.ojin_client_messages import (
    FrameType,
    OjinAudioInputMessage,
    OjinCancelInteractionMessage,
    OjinEndInteractionMessage,
    OjinInteractionResponseMessage,
    OjinSessionReadyMessage,
    OjinSessionReadyPing,
)

from .errors import OjinException
from .frames import decode_jpeg_to_rgb24_sync, pcm16_bytes_to_audio_frame, rgb24_to_video_frame
from .log import logger

_DEFAULT_WS_URL = "wss://models.ojin.ai/realtime"
_DEFAULT_VIDEO_WIDTH = 512
_DEFAULT_VIDEO_HEIGHT = 512
_DEFAULT_VIDEO_FPS = 25
_DEFAULT_AUDIO_SAMPLE_RATE = 16000
_DEFAULT_AUDIO_CHANNELS = 1

_HIGH_WATERMARK_FRAMES = 70
_LOW_WATERMARK_FRAMES = 40


class OjinVideoGenerator(VideoGenerator):
    """VideoGenerator implementation for Ojin avatar rendering.

    Sends audio to Ojin via OjinClient and yields decoded video/audio frames.
    Implements hysteresis-based buffer management for backpressure.
    """

    def __init__(
        self,
        client: Any,
        *,
        video_width: int = _DEFAULT_VIDEO_WIDTH,
        video_height: int = _DEFAULT_VIDEO_HEIGHT,
    ) -> None:
        self._client = client
        self._video_width = video_width
        self._video_height = video_height
        self._drop_mode = False
        self._pending_frames = 0
        self._output_ch = utils.aio.Chan[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]()
        self._closed = False
        self._interaction_started = False

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """Push an audio frame to Ojin or signal end of segment."""
        if isinstance(frame, AudioSegmentEnd):
            self._interaction_started = False
            await self._client.send_message(OjinEndInteractionMessage())
            return

        if not self._interaction_started:
            self._interaction_started = True
            self._client.start_interaction()

        await self._client.send_message(OjinAudioInputMessage(audio_int16_bytes=bytes(frame.data)))

    async def clear_buffer(self) -> None:
        """Cancel current interaction and immediately signal segment end."""
        self._interaction_started = False
        await self._client.send_message(OjinCancelInteractionMessage())

        # Clear our own output channel (client already clears its internal queue)
        while True:
            try:
                self._output_ch.recv_nowait()
                self._pending_frames -= 1
            except utils.aio.channel.ChanEmpty:
                break

        # Reset drop mode
        self._drop_mode = False
        self._pending_frames = 0

        # Immediately enqueue AudioSegmentEnd (don't wait for is_final_response)
        self._output_ch.send_nowait(AudioSegmentEnd())

    def __aiter__(
        self,
    ) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._stream_impl()

    async def _stream_impl(
        self,
    ) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        """Receive messages from Ojin client and yield decoded frames."""
        while not self._closed:
            try:
                msg = await self._client.receive_message()
            except Exception as e:
                if self._closed:
                    return
                raise OjinException(
                    f"Failed to receive message from Ojin: {e}",
                    retryable=True,
                    code="RECEIVE_ERROR",
                    origin="stream",
                ) from e

            if msg is None:
                # Only returned when client._cancelled is True
                continue

            if isinstance(msg, OjinSessionReadyMessage):
                logger.debug("received sessionReady (in stream)")
                continue

            if isinstance(msg, OjinSessionReadyPing):
                # Discard ping messages
                continue

            if isinstance(msg, OjinInteractionResponseMessage):
                # Hysteresis buffer management
                self._pending_frames += 1

                if self._pending_frames >= _HIGH_WATERMARK_FRAMES:
                    self._drop_mode = True
                elif self._pending_frames <= _LOW_WATERMARK_FRAMES:
                    self._drop_mode = False

                if self._drop_mode and msg.frame_type == FrameType.IDLE:
                    # Drop idle frames atomically (both audio and video)
                    self._pending_frames -= 1
                    if msg.is_final_response:
                        yield AudioSegmentEnd()
                    continue

                # Decode JPEG to VideoFrame via thread pool
                if msg.video_frame_bytes:
                    width, height, rgb_data = await asyncio.to_thread(
                        decode_jpeg_to_rgb24_sync, msg.video_frame_bytes
                    )
                    video_frame = rgb24_to_video_frame(width, height, rgb_data)
                    yield video_frame

                # Convert PCM audio
                if msg.audio_frame_bytes:
                    audio_frame = pcm16_bytes_to_audio_frame(
                        msg.audio_frame_bytes,
                        sample_rate=_DEFAULT_AUDIO_SAMPLE_RATE,
                        channels=_DEFAULT_AUDIO_CHANNELS,
                    )
                    yield audio_frame

                self._pending_frames -= 1

                if msg.is_final_response:
                    yield AudioSegmentEnd()

    async def aclose(self) -> None:
        """Close the generator."""
        self._closed = True


class AvatarSession:
    """An Ojin avatar session using the local runner pattern.

    Connects to Ojin's realtime WebSocket API, sends TTS audio,
    and publishes synchronized avatar video/audio tracks to the room.
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        config_id: NotGivenOr[str] = NOT_GIVEN,
        ws_url: NotGivenOr[str] = NOT_GIVEN,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ) -> None:
        self._api_key = api_key if utils.is_given(api_key) else os.environ.get("OJIN_API_KEY")
        self._config_id = (
            config_id if utils.is_given(config_id) else os.environ.get("OJIN_CONFIG_ID")
        )
        self._ws_url: str = (
            ws_url if utils.is_given(ws_url) else os.environ.get("OJIN_WS_URL", _DEFAULT_WS_URL)
        )
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay

        if not self._api_key:
            raise OjinException(
                "api_key must be provided or set via OJIN_API_KEY environment variable",
                retryable=False,
                code="MISSING_API_KEY",
                origin="constructor",
            )
        if not self._config_id:
            raise OjinException(
                "config_id must be provided or set via OJIN_CONFIG_ID environment variable",
                retryable=False,
                code="MISSING_CONFIG_ID",
                origin="constructor",
            )

        self._client: Any = None
        self._generator: OjinVideoGenerator | None = None
        self._avatar_runner: AvatarRunner | None = None

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        """Start the Ojin avatar session.

        Creates OjinClient, waits for sessionReady, creates the video generator,
        and wires up QueueAudioOutput + AvatarRunner.
        """
        from ojin.ojin_client import OjinClient

        # 1. Create OjinClient
        self._client = OjinClient(
            self._ws_url,
            self._api_key,
            self._config_id,
            reconnect_attempts=self._reconnect_attempts,
            reconnect_delay=self._reconnect_delay,
        )

        # 2. Connect to Ojin
        await self._client.connect()

        # 3. Wait for sessionReady
        video_width = _DEFAULT_VIDEO_WIDTH
        video_height = _DEFAULT_VIDEO_HEIGHT
        while True:
            msg = await self._client.receive_message()
            if isinstance(msg, OjinSessionReadyMessage):
                logger.info("Ojin session ready")
                # Extract video params from sessionReady if available
                if msg.parameters:
                    video_width = msg.parameters.get("video_width", _DEFAULT_VIDEO_WIDTH)
                    video_height = msg.parameters.get("video_height", _DEFAULT_VIDEO_HEIGHT)
                break
            if isinstance(msg, OjinSessionReadyPing):
                continue
            logger.debug("received non-ready message while waiting for sessionReady: %s", type(msg))

        # 4. Create video generator
        self._generator = OjinVideoGenerator(
            self._client,
            video_width=video_width,
            video_height=video_height,
        )

        # 5. Create QueueAudioOutput
        audio_buffer = QueueAudioOutput(sample_rate=_DEFAULT_AUDIO_SAMPLE_RATE)

        # 6. Create AvatarRunner
        avatar_options = AvatarOptions(
            video_width=video_width,
            video_height=video_height,
            video_fps=_DEFAULT_VIDEO_FPS,
            audio_sample_rate=_DEFAULT_AUDIO_SAMPLE_RATE,
            audio_channels=_DEFAULT_AUDIO_CHANNELS,
        )

        self._avatar_runner = AvatarRunner(
            room=room,
            audio_recv=audio_buffer,
            video_gen=self._generator,
            options=avatar_options,
        )

        # 7. Start runner
        await self._avatar_runner.start()

        # 8. Replace audio output
        agent_session.output.audio = audio_buffer

    async def aclose(self) -> None:
        """Idempotent shutdown: close runner, generator, client."""
        if self._avatar_runner is not None:
            await self._avatar_runner.aclose()
            self._avatar_runner = None

        if self._generator is not None:
            await self._generator.aclose()
            self._generator = None

        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                logger.debug("error closing Ojin client", exc_info=True)
            self._client = None
