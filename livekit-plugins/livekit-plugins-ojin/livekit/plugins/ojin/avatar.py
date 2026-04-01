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

_SESSION_READY_TIMEOUT = 30.0


class OjinVideoGenerator(VideoGenerator):
    """VideoGenerator implementation for Ojin avatar rendering.

    Sends audio to Ojin via OjinClient and yields decoded video/audio frames.
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
        self._closed = False
        self._interaction_started = False
        self._interrupted = False

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        """Push an audio frame to Ojin or signal end of segment."""
        if isinstance(frame, AudioSegmentEnd):
            await self._client.send_message(OjinEndInteractionMessage())
            self._interaction_started = False
            return

        if not self._interaction_started:
            self._interaction_started = True
            self._client.start_interaction()

        await self._client.send_message(OjinAudioInputMessage(audio_int16_bytes=bytes(frame.data)))

    async def clear_buffer(self) -> None:
        """Cancel current interaction and signal interruption to the stream."""
        self._interrupted = True
        await self._client.send_message(OjinCancelInteractionMessage())
        self._interaction_started = False

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
                # Returned when client is cancelled (e.g. after clear_buffer)
                if self._interrupted:
                    self._interrupted = False
                    yield AudioSegmentEnd()
                continue

            if isinstance(msg, OjinSessionReadyMessage):
                logger.debug("received sessionReady (in stream)")
                continue

            if isinstance(msg, OjinSessionReadyPing):
                continue

            if isinstance(msg, OjinInteractionResponseMessage):
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
        session_ready_timeout: float = _SESSION_READY_TIMEOUT,
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
        self._session_ready_timeout = session_ready_timeout

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

    async def _wait_session_ready(self) -> OjinSessionReadyMessage:
        """Wait for sessionReady message from Ojin, raising on connection loss."""
        while True:
            msg = await self._client.receive_message()
            if isinstance(msg, OjinSessionReadyMessage):
                return msg
            if isinstance(msg, OjinSessionReadyPing):
                continue
            if msg is None:
                raise OjinException(
                    "Connection lost while waiting for Ojin sessionReady",
                    retryable=True,
                    code="CONNECTION_LOST",
                    origin="start",
                )
            logger.debug("received non-ready message while waiting for sessionReady: %s", type(msg))

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        """Start the Ojin avatar session.

        Creates OjinClient, waits for sessionReady, creates the video generator,
        and wires up QueueAudioOutput + AvatarRunner.
        """
        from ojin.ojin_client import OjinClient

        self._client = OjinClient(
            self._ws_url,
            self._api_key,
            self._config_id,
            reconnect_attempts=self._reconnect_attempts,
            reconnect_delay=self._reconnect_delay,
        )

        try:
            await self._client.connect()

            session_ready = await asyncio.wait_for(
                self._wait_session_ready(), timeout=self._session_ready_timeout
            )
            logger.info("Ojin session ready")

            video_width = _DEFAULT_VIDEO_WIDTH
            video_height = _DEFAULT_VIDEO_HEIGHT
            if session_ready.parameters:
                video_width = session_ready.parameters.get("video_width", _DEFAULT_VIDEO_WIDTH)
                video_height = session_ready.parameters.get("video_height", _DEFAULT_VIDEO_HEIGHT)

            self._generator = OjinVideoGenerator(
                self._client,
                video_width=video_width,
                video_height=video_height,
            )

            audio_buffer = QueueAudioOutput(sample_rate=_DEFAULT_AUDIO_SAMPLE_RATE)

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

            await self._avatar_runner.start()
            agent_session.output.audio = audio_buffer
        except asyncio.TimeoutError as e:
            await self.aclose()
            raise OjinException(
                f"Timed out waiting for Ojin sessionReady after {self._session_ready_timeout}s",
                retryable=True,
                code="SESSION_READY_TIMEOUT",
                origin="start",
            ) from e
        except Exception:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """Idempotent best-effort shutdown: close runner, generator, client."""
        if self._avatar_runner is not None:
            try:
                await self._avatar_runner.aclose()
            except Exception:
                logger.debug("error closing avatar runner", exc_info=True)
            finally:
                self._avatar_runner = None

        if self._generator is not None:
            try:
                await self._generator.aclose()
            except Exception:
                logger.debug("error closing video generator", exc_info=True)
            finally:
                self._generator = None

        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                logger.debug("error closing Ojin client", exc_info=True)
            finally:
                self._client = None
