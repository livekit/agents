import asyncio
import time
from typing import Any

from livekit import rtc
from livekit.agents.log import logger
from livekit.agents.utils import aio


class RobustMicrophone:
    """
    A robust microphone capture utility that wraps rtc.MediaDevices().open_input()
    and automatically restarts the audio stream if it stalls (e.g. if the microphone
    cable becomes loose).

    This class exposes an `rtc.AudioSource` as `.source` which you can use to
    create your LocalAudioTrack.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        num_channels: int = 1,
        stall_timeout: float = 2.0,
        **open_input_kwargs: Any,
    ) -> None:
        self._sample_rate = sample_rate
        self._num_channels = num_channels
        self._stall_timeout = stall_timeout
        self._kwargs = open_input_kwargs

        self._devices = rtc.MediaDevices()
        self._source = rtc.AudioSource(self._sample_rate, self._num_channels)

        self._mic_obj: Any = None
        self._mic_track: rtc.LocalAudioTrack | None = None
        self._mic_stream: rtc.AudioStream | None = None

        self._running = False
        self._monitor_task: asyncio.Task[None] | None = None
        self._last_frame_time: float = time.monotonic()

    @property
    def source(self) -> rtc.AudioSource:
        return self._source

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def aclose(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._monitor_task:
            await aio.cancel_and_wait(self._monitor_task)
        await self._close_internal()

    async def _close_internal(self) -> None:
        if self._mic_stream:
            await self._mic_stream.aclose()
            self._mic_stream = None
        self._mic_track = None
        self._mic_obj = None

    def _start_internal(self) -> None:
        # Provide defaults for sample_rate and num_channels if not provided by user
        kwargs = dict(self._kwargs)
        if "sample_rate" not in kwargs:
            kwargs["sample_rate"] = self._sample_rate
        if "num_channels" not in kwargs:
            kwargs["num_channels"] = self._num_channels

        self._mic_obj = self._devices.open_input(**kwargs)
        self._mic_track = rtc.LocalAudioTrack.create_audio_track("robust-mic-internal", self._mic_obj.source)
        self._mic_stream = rtc.AudioStream.from_track(self._mic_track)
        self._last_frame_time = time.monotonic()

    async def _monitor_loop(self) -> None:
        self._start_internal()

        while self._running:
            try:
                assert self._mic_stream is not None

                # Wait for the next audio event with a timeout
                event = await asyncio.wait_for(
                    self._mic_stream.__anext__(), timeout=self._stall_timeout
                )
                self._last_frame_time = time.monotonic()

                # Forward the captured frame to our own source
                await self._source.capture_frame(event.frame)

            except asyncio.TimeoutError:
                # Stall detected!
                logger.warning(
                    f"RobustMicrophone: No audio frames received for {self._stall_timeout}s. Restarting microphone..."
                )
                await self._close_internal()
                await asyncio.sleep(0.5)  # Brief pause before reconnecting
                self._start_internal()
            except Exception as e:
                logger.error(f"RobustMicrophone error in monitor loop: {e}", exc_info=e)
                await asyncio.sleep(1.0)
                await self._close_internal()
                self._start_internal()
